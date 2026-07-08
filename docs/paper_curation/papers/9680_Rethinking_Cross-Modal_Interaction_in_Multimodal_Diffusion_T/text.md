# arXiv:2506.07986v3[cs.CV]23Jul2025

## Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers

Zhengyao Lv1∗ Tianlin Pan2,3∗ Chenyang Si2∗ Zhaoxi Chen4 Wangmeng Zuo5 Ziwei Liu4† Kwan-Yee K. Wong1†

1The University of Hong Kong 2Nanjing University 3University of Chinese Academy of Sciences 4Nanyang Technological University 5Harbin Institute of Technology

cszy98@gmail.com pantianlin23@mails.ucas.ac.cn chenyang.si@nju.edu.cn zhaoxi001@ntu.edu.sg cswmzuo@gmail.com ziwei.liu@ntu.edu.sg kykwong@cs.hku.hk

FLUX.1-Dev

###### SD3.5 SD3.5 + TACA FLUX.1-Dev + TACA

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

a clock on the top of a sheep

[Figure 5]

[Figure 6]

a car on the right of a bee

The wooden picture frame and glass photo display the metallic trophy on the bookshelf

Figure 1. We propose TACA, a parameter-efficient method that dynamically rebalances cross-modal attention in multimodal diffusion transformers to improve text-image alignment.

### Abstract

demonstrating its ability to improve text-image alignment in terms of object appearance, attribute binding, and spatial relationships. Our findings highlight the importance of balancing cross-modal attention in improving semantic fidelity in text-to-image diffusion models. Our codes are publicly available at https://github.com/Vchitect/TACA.

Multimodal Diffusion Transformers (MM-DiTs) have achieved remarkable progress in text-driven visual generation. However, even state-of-the-art MM-DiT models like FLUX struggle with achieving precise alignment between text prompts and generated content. We identify two key issues in the attention mechanism of MMDiT, namely 1) the suppression of cross-modal attention due to token imbalance between visual and textual modalities and 2) the lack of timestep-aware attention weighting, which hinder the alignment. To address these issues, we propose Temperature-Adjusted Cross-modal Attention (TACA), a parameter-efficient method that dynamically rebalances multimodal interactions through temperature scaling and timestep-dependent adjustment. When combined with LoRA fine-tuning, TACA significantly enhances text-image alignment on the T2I-CompBench benchmark with minimal computational overhead. We tested TACA on state-of-the-art models like FLUX and SD3.5,

### 1. Introduction

Diffusion models [13, 40], driven by iterative denoising processes, have emerged as a powerful paradigm in generative modeling and various visual generation tasks [26, 46, 47, 52]. The field has witnessed significant architectural evolution, beginning with U-Net-based designs [38] that dominated early diffusion models [9, 28, 33, 37]. Recent advances introduced transformer-based architectures through Diffusion Transformers (DiT) [5, 30], demonstrating superior scalability and training stability. This progression culminated in Multimodal Diffusion Transformers (MMDiT) [10], which unify text and visual tokens through a concatenated self-attention mechanism, resulting in state-

*Equal Contribution. †Corresponding Author.

of-the-art text-to-image/video models like Stable Diffusion 3/3.5 [10, 41], FLUX [20], CogVideo [14, 51], and HunyuanVideo [19].

[Figure 7]

- Figure 2. Object missing in text-to-image models. Even in SOTA models like FLUX.1 Dev, we can still observe cases with missing objects. Prompts: “The square painting was next to the round mirror”, “a blue bench and a green car”.

Although the MM-DiT architecture has undergone significant advancements, state-of-the-art text-to-image models like FLUX still exhibit critical limitations, particularly in generating images with precise semantic alignment (see Fig. 2). Analysis of the sampling process reveals that early denoising steps require strong text-visual interaction to create a proper semantic layout, while later steps focus on refining the details. Semantic discrepancies between the text and synthesized images often stem from flawed initial layouts (see Fig. 3).

In typical U-Net/DiT-based text-to-image diffusion models, the cross-attention block enables modal interaction between textual and visual tokens to synthesize text-aligned images. Our analysis of the attention maps of MM-DiT layers suggests that semantic discrepancies may arise from the suppression of cross-modal attention, specifically due to the numerical asymmetry between the number of visual and text tokens. The overwhelming number of visual tokens can dilute the textual guidance in the unified softmax function of the MM-DiT architecture, resulting in the visual tokens paying significantly less attention to the textual tokens compared to the typical cross-attention paradigm (see Fig. 4). Furthermore, we noticed that current MM-DiT architectures employ static attention mechanisms with the same weighting for all timesteps, which is ill-suited to the time-varying demands of semantic composition and detail synthesis during the denoising process (see Fig. 3). This temporal dynamic remains unaddressed in existing approaches, leading to suboptimal modality balancing.

Based on the above observations, we propose Temperature-Adjusted Cross-modal Attention (TACA), a straightforward yet effective enhancement to the MM-DiT attention mechanism. Our approach introduces two key innovations, namely (1) modality-specific temperature scaling to mitigate cross-attention suppression, and (2) timestep-dependent adjustments to cross-modal inter-

actions. TACA only requires a temperature coefficient γ(t) that adapts to the denoising timesteps, allowing for easy implementation with minimal code modifications. To mitigate potential artifacts introduced by amplified cross-attention, we complement TACA with Low-Rank Adaptation (LoRA) [15] fine-tuning for distributional alignment, helping the model generate images that better match real-world distributions.

Experiments on T2I-CompBench [16] validate the effectiveness of our method across various model architectures. For FLUX.1-Dev, incorporating TACA results in substantial improvements, yielding relative gains of 16.4% in spatial relationship understanding and 5.9% in shape accuracy. Similarly, when applied to SD3.5-Medium, TACA boosts spatial relationship accuracy by 28.3% and shape accuracy by 2.9%. These benchmark results, combined with the visual improvements shown in Fig. 1, highlight a significant enhancement in text-image alignment achieved by our approach.

In summary, our principal contributions are:

- • We systematically analyze MM-DiT’s unified attention mechanism, and reveal cross-attention suppression and timestep insensitivity being two key factors limiting textimage alignment in text-to-image generation.
- • We propose TACA, the first approach to dynamically balance multimodal interactions through temperature scaling and temporal adaptation in diffusion transformers.
- • Extensive benchmark results demonstrate that TACA can effectively improve semantic alignment with minimal computational overhead.

### 2. Related Work

#### 2.1. Diffusion Transformers

A central challenge in developing transformer-based textto-image/video (T2I, T2V) diffusion models lies in the effective integration of multimodal data, primarily text and visual information. Several approaches, including Diffusion Transformers (DiT [30]), CrossDiT (PixArt-α [5]), and MM-DiT (Stable Diffusion 3 [10]), tackle this challenge with distinct methods for cross-modal interaction and textimage alignment.

The original DiT [30] introduced transformers [1, 43] as replacements for U-Net backbones [38] in diffusion models [13, 40]. While not inherently multimodal, DiT established critical conditioning mechanisms via adaptive layer normalization (adaLN) [31]. This technique modulates transformer activations using timestep embeddings and class labels, enabling controlled generation based on singlemodality inputs. While effective for class-conditional generation, DiT lacks explicit mechanisms for text-image alignment, limiting its applicability in multimodal tasks.

CrossDiT (PixArt-α) [5] introduced cross-modal fusion by

[Figure 8]

γTemperatureScalingFactor

Denoising Steps

- Figure 3. The denoising process. This figure shows the predicted x0 in each step of the denoising process for the prompt “The black chair is on the right of the wooden table” with FLUX.1 Dev. This observation leads to our hypothesis that visual-text cross-attention plays a more significant role than visual self-attention specifically during these initial steps where the image’s overall composition is determined. Additionally, as the temperature scaling factor γ increases in the cross-modal section of MM-DiT’s unified softmax function, the initial image composition progressively aligns more closely with the corresponding text.

Layer 1

Layer 11

1.2

1.2

| |Typical Cross-Attention<br><br>Full Attention<br><br>Ours ( = 1.1)<br><br>Ours ( = 1.2)<br><br>Ours ( = 1.3)<br><br><br>| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| |Typical Cross-Attention<br><br>Full Attention<br><br>Ours ( = 1.1)<br><br>Ours ( = 1.2)<br><br>Ours ( = 1.3)<br><br><br>| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

1.0

1.0

RelativeMagnitude

RelativeMagnitude

0.8

0.8

0.6

0.6

0.4

0.4

0.2

0.2

0.0

0.0

0 5 10 15 20 25 30 Step

0 5 10 15 20 25 30 Step

Layer 21

Layer 31

1.2

1.2

| || |Typical Cross-Attention<br><br>Full Attention|
|---|---|
| |Ours ( = 1.1)<br><br>Ours ( = 1.2)<br><br>Ours ( = 1.3)<br><br><br>|
| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| |Typical Cross-Attention<br><br>Full Attention<br><br>Ours ( = 1.1)<br><br>Ours ( = 1.2)<br><br>Ours ( = 1.3)<br><br><br>| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

1.0

1.0

RelativeMagnitude

RelativeMagnitude

0.8

0.8

0.6

0.6

0.4

0.4

0.2

0.2

0.0

0.0

0 5 10 15 20 25 30 Step

0 5 10 15 20 25 30 Step

Figure 4. Relative magnitude of visual-text attention between the typical cross attention and MM-DiT full attention (averaged over 50 samples). The numerical asymmetry between the number of visual and text tokens suppresses the magnitude of cross attention, leading to weak alignment between the generated image and the given text prompt. We can amplify the cross-attention by boosting the coefficient γ, thereby strengthening the alignment between the image and text.

integrating text-guided cross-attention into the DiT backbone. In this framework, cross-attention replaces adaLN for text conditioning, which enables dynamic per-token modulation based on linguistic semantics. However, CrossDiT uses a unidirectional update approach that prevents the image from influencing the textual representation. This hinders its ability to model feedback loops and nuanced interdependencies between the text and generated image.

MM-DiT (Stable Diffusion 3) [10] represents a paradigm shift by introducing bidirectional cross-modal attention and a unified token space for text and visual modalities. By concatenating text and image tokens into a single sequence and employing a decomposed attention matrix, MM-DiT enables full self-attention across modalities, capturing complex inter-modal relationships. Besides, the integration of multiple text encoders (e.g., CLIP [34] and T5 [35]) further improves the model’s ability to understand and generate text with greater accuracy.

#### 2.2. Text-to-Image Alignment

Prior research has explored generating images from text prompts using pre-trained models without requiring further training. Some employ techniques such as CLIP-guided optimization [12, 25] to align images with text by optimizing CLIP scores within the model’s latent space. Additionally, cross-attention-based approaches [7] are used to enhance spatial layout and details in generated images, thereby improve adherence to the textual description’s structure.

Additionally, more recent research has explored augmenting guidance-based models to enhance semantic control, primarily through layout planning modules [6, 8, 18, 22, 32, 48, 49] and feedback-driven optimization [3, 11, 42].

Another direction involves attention-based methods [2, 4, 21, 27, 36, 45] that modify or constrain the attention maps within U-Net models to improve textual alignment. However, these attention-based techniques generally do not readily translate to contemporary DiT-based architectures.

### 3. Methodology

#### 3.1. Preliminaries

Diffusion-based generative models operate through a forward diffusion process and a reverse denoising process [13]. The forward process systematically degrades data samples through gradual noise injection, while the reverse process learns to recover the original data structure through iterative refinement.

The diffusion mechanism progressively corrupts training samples x0 ∼ q(x0) over T discrete timesteps according to a predetermined variance schedule {βt}Tt=1. This corruption follows a Markov chain where each transition adds isotropic Gaussian noise:

q(xt|xt−1) = N xt; 1 − βtxt−1,βtI . (1)

The denoising phase constitutes a learned reversal of this progressive corruption. This reverse process estimates the ancestral distribution q(xt−1|xt) by learning:

pθ(xt−1|xt) = N xt−1;µθ(xt,t),σt2I , (2) where σt2 is typically fixed as βt or β˜t = 1−α¯

1−α¯t βt with α¯t = ts=1(1 − βs). The mean µθ is derived through a noise prediction network ϵθ. This network, conventionally implemented as a time-conditional U-Net [38] or vision

t−1

transformers [5, 10, 30] in more recent works, is optimized to predict the noise component presents in xt, enabling precise incremental denoising.

Multimodal Diffusion Transformer (MM-DiT) [10] is a novel approach to adopt transformers as the noise prediction network in diffusion models. The MM-DiT architecture concatenates text and visual tokens into a single input sequence after projecting both modalities to a shared dimensional space. The concatenated sequence undergoes multi-head self-attention where every token attends to all others, regardless of modality. Mathematically, if we use H to denote the number of attention heads, Nx and Nc to denote the sequence length of visual and text tokens respectively, and D to denote the dimension of the token embeddings, then for visual tokens x ∈ RH×N

x×D and text tokens c ∈ RH×N

c×D, we have:

WQc c WQx x

WKc c WKx x

WVc c WVx x

, (3)

Q =

, K =

, V =

and

Attention(Q,K,V ) = softmax

##### QKT

√

D

V , (4)

where the QKT term can be expanded to

QKT =

=

WQc c(WKc c)T WQc c(WKx x)T WQx x(WKc c)T WQx x(WKx x)T

(5)

QtxtKTtxt QtxtKTvis QvisKTtxt QvisKTvis

. (6)

As we can see in Eq 6, this MM-DiT formulation allows four interaction types: text-text, text-visual, visual-text, and visual-visual attentions, all within a single operation.

#### 3.2. Suppression of Cross-Attention and TimestepInsensitive Weighting in MM-DiT

While the unified attention mechanism of MM-DiT provides computational efficiency through joint modality processing, it introduces inherent issues when balancing different modalities.

Suppression of Cross-Attention This issue stems from the numerical asymmetry between the number of visual and text tokens (Nx ≫ Nc), which creates a systematic bias in attention weight distribution. Consider the attention computation for visual tokens in Eq. 6. Each visual token’s attention weights over text tokens (QvisKTtxt) must compete against visual-visual interactions (QvisKTvis) in the softmax denominator. Formally, the probability of the i-th visual token attending to the j-th text token guidance becomes:

vt ij/τ

es

Pvis(i, j−)txt =

Ntxt k=1 esvtik/τ + Nk=1vis esvvik/τ , (7)

√

√

where svtik = Q(visi)KTtxt(k)/

D and svvik = Qvis(i)KTvis(k)/

D. When Nvis ≫ Ntxt, the sum over visual-visual interactions dominates the denominator, even if individual svvik values are modest. For example, when using FLUX.1 Dev [20] to generate a 1024 × 1024 image, we have Nvis/Ntxt = 4096/512 = 8. In this case, the visual-text cross-attention probabilities would be much lower than in the typical paradigm:

vt ij/τ

es

Pvis(i, j−)txt ≈

Nvis k=1 esvvik/τ (Full Attention) (8)

vt ij/τ

es

Ntxt k=1 esvtik/τ (Typical Cross-Attention) (9)

≪

This suppression of Pvis−txt, which can be observed in Fig. 4, weakens the alignment between visual and textual features. The model struggles to effectively leverage textual guidance to refine visual representations because the influence of the text tokens is diluted by the overwhelming presence of visual tokens. Crucial semantic relationships encoded in the text may be overlooked, leading to a visual representation that is less informed by the corresponding textual description, like the bad cases shown in Fig. 2.

Timestep-Insensitive QK Weighting MM-DiT’s current architecture employs timestep-agnostic projection of latent states into query and key vectors. This approach fails to account for the evolving influence of textual guidance throughout the denoising process. As illustrated in Fig. 3, the initial denoising steps are crucial for establishing the image’s global layout, heavily influenced by the text prompt. Consequently, the cross-attention mechanism, responsible for integrating textual information, should be weighted more heavily than visual self-attention during these early stages. MM-DiT’s static weighting strategy, therefore, limits its ability in optimally leveraging textual guidance and adapting to the changing demands of the denoising process.

Formally, when t is large (i.e., early in the denoising pro-

cess) and cross-modal guidance should dominate, svtik values fail to receive proportionally larger magnitudes compared to

svvik . Since WQ and WK are optimized for global performance across all timesteps, they cannot focus on amplifying

visual-text interactions in the early stages. This potentially leads to the overall layout of the generated image not aligning with the text prompt.

#### 3.3. Temperature-Adjusted Cross-modal Attention

To address the issues mentioned in Section 3.2, we propose Temperature-Adjusted Cross-modal Attention (TACA), a simple yet effective modification to the attention mechanism of MM-DiT. Our approach introduces two key innovations, namely modality-specific temperature scaling and timestep-dependent adjustment of cross-modal interactions.

Temperature Scaling

[Figure 9]

[Figure 10]

[Figure 11]

###### a brown backpack and a blue horse

[Figure 12]

[Figure 13]

[Figure 14]

a fabric hat and a glass mirror

[Figure 15]

[Figure 16]

[Figure 17]

A white cat with black stomach takes a pose

Figure 5. Temperature scaling helps text-image alignment. From this figure, we can see that as the temperature scaling factor γ increases, the characteristics of “brown backpack”, “glass mirror” and “black stomach” become more obvious.

Modality-Specific Temperature Scaling To mitigate the suppression of cross-attention caused by the dominance of visual tokens (Nvis ≫ Ntxt), we amplify the logits of visual-text interactions through a temperature coefficient γ > 1. The modified attention probability for visual-text interaction becomes:

vt ij/τ

eγs

Pvis(i, j−)txt =

Ntxt k=1 eγsvtik/τ + Nk=1vis esvvik/τ , (10)

This scaling effectively rebalances the competition in softmax by increasing the relative weights of cross-modal interactions. The γ coefficient acts as a signal booster for text-guided attention. As shown in Fig. 5, the generated image and text prompt become more consistent as γ increases. Timestep-Dependent Adjustment To compensate for the insensitivity of QK weights with respect to the timestep, we make γ timestep-dependent to account for the varying importance of cross-attention during denoising based on the observations in Fig. 3. Specifically, we employ a piecewise function:

γ0 t ≥ tthresh 1 t < tthresh

(11)

γ(t) =

where tthresh is a threshold for the timestep that separates the layout formation and detail refinement phases. This design aligns with the denoising dynamics where early steps

Baseline Ours

[Figure 18]

[Figure 19]

| |
|---|

| |
|---|

Temperature Scaling

Prompt:

The square painting was next to the round mirror

Step 1 Step 2 Step 3 Step 4 Step 5

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

| |
|---|

| |
|---|

| |
|---|

Visualization of attention map difference (Ours – Baseline, First 5/30 steps)

Figure 6. Attention map differences. We conducted a visualization of the alterations in the visual-text attention map during the initial stages of the denoising process, as influenced by our proposed method. In contrast to the baseline, our approach substantially amplifies the attention directed toward the text in the early steps.

(i.e., large t) require strong text guidance to establish image composition and later steps (i.e., small t) focus on visual details when self-attention dominates. By effecting the attention map, image tokens can better attend to the relevant text tokens, as shown in Fig. 6.

Notably, TACA introduces no new learnable parameters, with the temperature scaling implemented via a simple element-wise operation during attention computation. The γ0 and tthresh parameters can be tuned through minimal ablation studies, making our approach both efficient and practical for deployment in existing MM-DiT architectures.

LoRA Training for Artifact Suppression While temperature scaling in TACA significantly improves text-image alignment, the amplified cross-modal attention logits can alter the output distribution of the denoising process, occasionally introducing artifacts such as distorted object boundaries or inconsistent textures. To mitigate this, we employ Low-Rank Adaptation (LoRA) [15] to fine-tune the model, encouraging it to recover the real image distribution while preserving the benefits of temperature scaling.

We apply LoRA to the attention layers of MM-DiT, where the temperature scaling exerts the most direct influence. For a weight matrix W ∈ Rd×k, LoRA adaptation is formulated as

W′ = W + α · BA, B ∈ Rd×r, A ∈ Rr×k (12)

where r ≪ min(d,k) is the rank of the adaptation, and α scales the low-rank update. Only B and A are trainable during fine-tuning, keeping the original W frozen.

### 4. Experiments

#### 4.1. Experiment Settings

Evaluation Metrics and Datasets We evaluate our method on the T2I-CompBench benchmark [16], a comprehensive evaluation suite for text-to-image alignment. All experiments use the LAION dataset [39] with captions refined by the LLaVA model [24] to enhance semantic precision. We randomly sampled 10K image-text pairs as the

FLUX.1 Dev Ours FLUX.1 Dev Ours FLUX.1 Dev Ours

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

The sweet banana split topped with the crunchy nuts

The red vase was on top of the wooden shelf The juicy watermelon quenched the thirsty mouth and the dry throat

and the soft whipped cream

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

The crisp green lettuce lay next to the juicy red tomato

The crisp apple lay beside the rough stone and the silky fabric

The fluffy white clouds floated above the tall

skyscrapers

SD3.5-Medium Ours SD3.5-Medium Ours SD3.5-Medium Ours

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

The bright green leaves contrasted with the dull grey

The green chair is on the right of the wooden table The glossy apple sat next to the fuzzy peach and the prickly pear

pavement

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

The green plant was on the left of the white lamp The blue pencil was next to the white paper

The flickering fireflies danced in the dark forest and

the moonlit meadow

Figure 7. Comparison of samples generated by FLUX.1 Dev and Stable Diffusion 3.5 Medium with and without TACA.

Table 1. Comparison of alignment evaluation on T2I-CompBench [16] for FLUX.1-Dev-based and SD3.5-Medium-based models. The best results for each model group are highlighted in bold.

Attribute Binding Object Relationship Color ↑ Shape ↑ Texture ↑ Spatial ↑ Non-Spatial ↑ Complex ↑

Model

FLUX.1-Dev 0.7678 0.5064 0.6756 0.2066 0.3035 0.4359 FLUX.1-Dev + TACA (r = 64) 0.7843 0.5362 0.6872 0.2405 0.3041 0.4494 FLUX.1-Dev + TACA (r = 16) 0.7842 0.5347 0.6814 0.2321 0.3046 0.4479

SD3.5-Medium 0.7890 0.5770 0.7328 0.2087 0.3104 0.4441 SD3.5-Medium + TACA (r = 64) 0.8074 0.5938 0.7522 0.2678 0.3106 0.4470 SD3.5-Medium + TACA (r = 16) 0.7984 0.5834 0.7467 0.2374 0.3111 0.4505

training dataset for our LoRA model. To ensure reproducibility throughout all evaluation phases, the random seed is fixed to 42, while all other parameters remain at their default values as provided by the Diffusers library [44].

Implementation Details We conduct experiments on a single NVIDIA A100 80GB GPU using the ai-toolkit codebase [29], with LoRA adapters implemented for FLUX.1 Dev [20] and SD3.5 Medium [41] models. We adopt the AdamW optimizer with a learning rate of 1 × 10−4 and a

batch size of 4 for training. We evaluate two LoRA configurations: (r,α) = (16,16) and (64,64).

To emphasize semantic alignment, we sample timesteps t ≥ tthresh = 970 within the range t ∈ (0,1000). In the flow matching scheduler, a 30-step denoising process allocates the first three steps to t ∈ (970,1000] (i.e., the initial 10% of the diffusion process), while the remaining 27 steps cover t ∈ [0,970). Setting tthresh = 970 focuses training on these early steps where semantic information is most

prominent.

Under the flow-matching paradigm [23], the model predicts velocity v instead of noise ϵ. We fine-tune it with the following velocity prediction loss:

L = Ex0, t≥tthresh ∥v(xt, t) − vθ(xt, t, Ptxt, γ(t))∥22 , (13)

where Ptxt represents text prompts and γ(t) induces the modified temperature coefficient. For benchmark results, we set the base temperature scaling factor as γ0 = 1.2, which is selected in Section 4.3. This formulation ensures the model learns the correct velocity field while adapting to temperature-scaled attention.

#### 4.2. Main Results

Quantitative Comparison To quantitatively evaluate the effectiveness of our proposed TACA, we conduct a comprehensive comparison against baseline models. Table 1 presents the alignment performance of FLUX.1-Dev and SD3.5-Medium models, respectively, with and without the integration of TACA. For FLUX.1-Dev, the incorporation of TACA, particularly with a rank r = 64, consistently improves performance across all Attribute Binding metrics and Spatial Relationship. Similarly, for SD3.5-Medium, TACA with r = 64 yields significant gains in Attribute Binding and Spatial Relationship, and TACA with r = 16 achieves the best performance on Non-Spatial Relationship and Complex prompt evaluation. These results demonstrate that TACA effectively enhances the alignment capabilities of different MM-DiT models across various dimensions of text-to-image generation quality.

Image Quality Evaluation We use widely adopted image quality assessment models MUSIQ [17] and MANIQA [50] to evaluate visual quality. As shown in the Table 2, TACA improves text-image alignment without sacrificing image quality on both SD3.5 and FLUX. Additionally, Fig. 7 presents further visual comparison results.

Table 2. Results of image quality assessment.

Metric SD3.5 +TACA FLUX +TACA MUSIQ ↑ 0.7182 0.7210 0.7186 0.7212 MANIQA ↑ 0.4883 0.4921 0.5149 0.5292

User Study We invited 50 participants for our user study. From the T2I-Compbench [16] dataset, we sampled 25 prompts and generated images using FLUX.1 Dev model with and without our TACA method. These images, along with their corresponding text prompts, were presented to the participants. Participants were asked to indicate their preferred image based on three criteria, namely overall visual appeal, attribute (color/shape/texture) quality, and text-image alignment. The results, as summarized in Table 3, demonstrate that a majority of participants favored the images generated by the model incorporating the TACA method. This suggests that our method yields improvements in text alignment and does not ruin image quality.

Table 3. Results of user study.

Evaluation Criteria FLUX FLUX + TACA Overall 23.58% 76.42% Attribute Quality 29.25% 70.75% Text-Image Alignment 17.75% 82.25%

#### 4.3. Ablation Study

Effect of Temperature Scaling Factor (γ0) The temperature scaling factor γ0 plays a crucial role in modulating the influence of textual guidance during the denoising process. We explore the impact of different γ0 values on compositional generation performance. Table 4 presents the results.

Table 4. Ablation study on the effect of the temperature scaling factor (γ0). We randomly sampled 100 prompts for each attribute from the T2I-CompBench dataset to conduct the evaluation. Here “LoRA Only” refers to training a LoRA model solely on the identical dataset using the same hyperparameters without our proposed method. Bold indicates the best score and underline indicates the second best score for each attribute.

Model Color ↑ Shape ↑ Texture ↑ Spatial ↑ FLUX.1-Dev 0.798 0.591 0.755 0.193 LoRA Only 0.805 0.592 0.759 0.187

- Ours (γ0 = 1.1) 0.803 0.603 0.780 0.199

- Ours (γ0 = 1.2) 0.839 0.634 0.790 0.207

- Ours (γ0 = 1.3) 0.787 0.650 0.766 0.225

The results in Table 4 demonstrate that our proposed method consistently outperforms both the baseline FLUX.1-Dev and the LoRA-only approach across all attributes for reasonable values of γ0. We observe improvements in Color, Shape, Texture, and Spatial compositional accuracy. Notably, γ0 = 1.2 yields the best overall balance, achieving the highest scores in Color and Texture, and the second-best in Shape and Spatial. Increasing γ0 further to 1.3 leads to slight improvements in Shape and Spatial, but a decline in Color and Texture. This suggests that a moderate increase in textual influence is beneficial, but excessive amplification can negatively impact certain aspects of compositional generation.

To understand the mechanism behind this improvement, we analyze the CLIP similarity between the predicted intermediate latent representations and the text prompt at each denoising step for varying γ0. As shown in Figure 8 (b), increasing γ0 leads to a higher CLIP similarity, particularly in the initial denoising steps. This indicates that TACA effectively enhances the text-image alignment early in the generation process, guiding the model towards generating images that are more consistent with the textual description.

Sensitivity of γ0 and tthresh We further investigate the sensitivity of our method to the choice of γ0 and the threshold timestep tthresh beyond which TACA is applied. Table 5 presents the average Attribute (Color, Shape, Texture) and Spatial scores for different values of γ0 and tthresh on both

Training Free With LoRA Training Free With LoRA

- 13

- 14

- 15

- 16

- 17

- 18

- 19

Baseline

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Training Free

With LoRA

| |
|---|

CLIPSimilarityScore

| |
|---|

| |
|---|

| |
|---|

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0 5 10 15 20 25 30 Step

Figure 8. The effect of LoRA training. (a) On the left, we show qualitative results comparing training-free image generation to generation with LoRA. The ‘training-free’ examples exhibit artifacts, such as the floating bowl, which are significantly reduced by LoRA training. (b) On the right, we present a quantitative evaluation of CLIP Scores for training-free and LoRA-trained models across the denoising steps, demonstrating that LoRA maintains strong text-image alignment and does not detract from the semantic benefits of our approach.

FLUX and SD3.5 models.

Table 5. Sensitivity analysis of γ0 and tthresh. The baseline corresponds to the respective models without TACA.

FLUX SD3.5

FLUX SD3.5 attr spa attr spa attr spa attr spa

γ0

tthresh

Baseline 0.715 0.193 0.797 0.159 Baseline 0.715 0.193 0.797 0.159 1.15 0.740 0.197 0.810 0.181 970 0.754 0.207 0.815 0.183 1.20 0.754 0.207 0.815 0.183 950 0.757 0.218 0.811 0.172 1.25 0.737 0.216 0.816 0.176 930 0.763 0.208 0.811 0.171

As shown in Table 5, the performance exhibits minimal variation across a reasonable range of both γ0 (e.g., 1.15 to 1.25) and tthresh (e.g., 930 to 970). This indicates that our method is not overly sensitive to the precise selection of these parameters, suggesting practical robustness. Furthermore, this robustness is observed across different base models, highlighting the general applicability of the tested parameter ranges.

The Effect of LoRA Training In the original TACA method, the introduction of factor γ(t) induces a shift in the output distribution of each attention layer. These modified outputs are subsequently processed by the feed-forward networks within the transformer blocks. Consequently, the overall output distribution of the diffusion transformer deviates from the distribution inherent in real images, which manifests as visual artifacts like unsupported floating bowls and distorted bridge connections in Fig 8 (a). To address this issue, we hypothesize that training a LoRA [15] module can effectively mitigate these artifacts. The rationale is that by fine-tuning the attention layer weights with a limited number of training samples, the LoRA module enables the modified model to readjust its output distribution to better align with the real image distribution.

Empirical findings from our experiments demonstrate that the incorporation of LoRA significantly enhances image quality and effectively mitigates these unrealistic artifacts, as evidenced in Fig. 8 (a). Concurrently, we evaluated whether the introduction of LoRA compromises the seman-

tic enhancement facilitated by the temperature coefficient γ(t). The comparative analysis of CLIP Scores for trainingfree and LoRA configurations for 50 samples, presented in Fig. 8 (b), reveals that LoRA exerts a negligible impact on text-image alignment.

### 5. Conclusion and Discussion

In this paper, we addressed two issues in MM-DiTs that limit text-image alignment in text-to-image generation: suppressed cross-attention due to token imbalance and timestep-insensitive attention weighting. We introduced Temperature-Adjusted Cross-modal Attention (TACA), a simple modification that dynamically balances multimodal interactions using temperature scaling and timestep-dependent adjustment. Combined with LoRA fine-tuning to reduce artifacts, TACA significantly improves text-image alignment on the T2I-CompBench benchmark. Our work demonstrates that strategically reweighting crossmodal interactions leads to more semantically accurate and visually coherent image generation, offering a promising approach for diffusion model research and applications.

Our work has mainly two limitations: 1) While improvements in text alignment were observed in training-free text-to-video experiments, we encountered a dilution effect when training a LoRA, wherein gains from increasing the temperature factor were diminished. 2) Our method lacks the ability to adaptively select an appropriate scaling factor based on the actual degree of text alignment.

### 6. Acknowledgement

This study is partially supported by the Ministry of Education, Singapore, under its MOE AcRF Tier 2 (MOET2EP20221-0012, MOE-T2EP20223-0002), and under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).

### References

- [1] An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 2
- [2] Aishwarya Agarwal, Srikrishna Karanam, K. J. Joseph, Apoorv Saxena, Koustava Goswami, and Balaji Vasan Srinivasan. A-star: Test-time attention segregation and retention for text-to-image synthesis. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 2283–2293,

2023. 3

- [3] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023. 3
- [4] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG), 42:1 – 10, 2023. 3
- [5] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James T. Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. The Twelfth International Conference on Learning Representations, 2023. 1, 2, 4
- [6] Minghao Chen, Iro Laina, and Andrea Vedaldi. Trainingfree layout control with cross-attention guidance. 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 5331–5341, 2023. 3
- [7] Minghao Chen et al. Training-free layout control with crossattention guidance. arXiv preprint arXiv:2304.03373, 2023. 3
- [8] Omer Dahary, Or Patashnik, Kfir Aberman, and Daniel Cohen-Or. Be yourself: Bounded attention for multi-subject text-to-image generation. arXiv preprint arXiv:2403.16990,

2024. 3

- [9] Prafulla Dhariwal and Alex Nichol. Diffusion models beat gans on image synthesis. arXiv preprint arXiv:2105.05233,

2021. 1

- [10] Patrick Esser, Sumith Kulal, A. Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024. 1, 2, 3, 4
- [11] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, P. Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. arXiv preprint arXiv:2305.16381, 2023. 3
- [12] Federico A. Galatolo, Mario G.C.A. Cimino, and Gigliola Vaglini. Generating images from caption and vice versa via clip-guided generative latent space search. arXiv preprint arXiv:2102.01645, 2021. 3
- [13] Jonathan Ho, Ajay Jain, and P. Abbeel. Denoising diffusion probabilistic models. arXiv preprint arXiv:2006.11239,

2020. 1, 2, 3

- [14] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 2
- [15] J. Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 2, 5, 8
- [16] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. arXiv preprint arXiv:2307.06350, 2023. 2, 5, 6, 7
- [17] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. arXiv preprint arXiv:2108.05997, 2021. 7
- [18] Yunji Kim, Jiyoung Lee, Jin-Hwa Kim, Jung-Woo Ha, and Jun-Yan Zhu. Dense text-to-image generation with attention modulation. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 7667–7677, 2023. 3
- [19] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2
- [20] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 2, 4, 6
- [21] Yumeng Li, Margret Keuper, Dan Zhang, and Anna Khoreva. Divide & bind your attention for improved generative semantic nursing. arXiv preprint arXiv:2307.10864, 2023. 3
- [22] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22511–22521, 2023. 3
- [23] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 7
- [24] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485. 5
- [25] Xingchao Liu, Chengyue Gong, Lemeng Wu, Shujian Zhang, Hao Su, and Qiang Liu. Fusedream: Training-free text-to-image generation with improved clip+gan space optimization. arXiv preprint arXiv:2112.01573, 2021. 3
- [26] Xiaoyu Liu, Yuxiang Wei, Ming Liu, Xianhui Lin, Peiran Ren, Xuansong Xie, and Wangmeng Zuo. Smartcontrol: Enhancing controlnet for handling rough visual conditions. In European Conference on Computer Vision, pages 1–17. Springer, 2024. 1
- [27] Tuna Han Salih Meral, Enis Simsar, Federico Tombari, and Pinar Yanardag. Conform: Contrast is all you need for highfidelity text-to-image diffusion models. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9005–9014, 2023. 3
- [28] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation

- and editing with text-guided diffusion models. In International Conference on Machine Learning, 2021. 1
- [29] Ostris. Ai toolkit, 2025. 6
- [30] William S. Peebles and Saining Xie. Scalable diffusion models with transformers. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2022. 1, 2, 4
- [31] Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron C. Courville. Film: Visual reasoning with a general conditioning layer. In AAAI Conference on Artificial Intelligence, 2017. 2
- [32] Quynh Phung, Songwei Ge, and Jia-Bin Huang. Grounded text-to-image synthesis with attention refocusing. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7932–7942, 2023. 3
- [33] Dustin Podell, Zion English, Kyle Lacey, A. Blattmann, Tim Dockhorn, Jonas Muller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 1
- [34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021. 3
- [35] Colin Raffel, Noam M. Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21:140:1–140:67, 2019. 3
- [36] Royi Rassin, Eran Hirsch, Daniel Glickman, Shauli Ravfogel, Yoav Goldberg, and Gal Chechik. Linguistic binding in diffusion models: Enhancing attribute correspondence through attention map alignment. arXiv preprint arXiv:2306.08877, 2023. 3
- [37] Robin Rombach, A. Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10674–10685, 2021. 1
- [38] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. arXiv preprint arXiv:1505.04597, 2015. 1, 2, 3
- [39] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 5
- [40] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 1, 2
- [41] Stability-AI. Stable diffusion 3.5, 2024. 2, 6
- [42] Jiao Sun, Deqing Fu, Yushi Hu, Su Wang, Royi Rassin, Da-Cheng Juan, Dana Alon, Charles Herrmann, Sjoerd van Steenkiste, Ranjay Krishna, and Cyrus Rashtchian. Dreamsync: Aligning text-to-image generation with image understanding feedback. arXiv preprint arXiv:2311.17946, 2023. 3

- [43] Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Neural Information Processing Systems, 2017. 2
- [44] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, Dhruv Nair, Sayak Paul, William Berman, Yiyi Xu, Steven Liu, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/ diffusers, 2022. 6
- [45] Zirui Wang, Zhizhou Sha, Zheng Ding, Yilin Wang, and Zhuowen Tu. Tokencompose: Text-to-image diffusion with token-level supervision. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8553–8564, 2023. 3
- [46] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15943–15953, 2023. 1
- [47] Yuxiang Wei, Yiheng Zheng, Yabo Zhang, Ming Liu, Zhilong Ji, Lei Zhang, and Wangmeng Zuo. Personalized image generation with deep generative models: A decade survey. arXiv preprint arXiv:2502.13081, 2025. 1
- [48] Jinheng Xie, Yuexiang Li, Yawen Huang, Haozhe Liu, Wentian Zhang, Yefeng Zheng, and Mike Zheng Shou. Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 7418–7427, 2023. 3
- [49] Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and Bin Cui. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. arXiv preprint arXiv:2401.11708, 2024. 3
- [50] Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. arXiv preprint arXiv:2204.08958,

2022. 7

- [51] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2
- [52] Yabo Zhang, Xinpeng Zhou, Yihan Zeng, Hang Xu, Hui Li, and Wangmeng Zuo. Framepainter: Endowing interactive image editing with video diffusion priors. arXiv preprint arXiv:2501.08225, 2025. 1

## Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers Supplementary Material

Overview. In the supplementary material, we provide further details to support our work. Section A elaborates on the implementation of TACA, including code snippets and a speed comparison of different approaches. Section B presents additional ablation studies focusing on text alignment, examining the effect of CFG guidance scale and the content/length of prompts. Section C explains why we choose LoRA rather than full-parameter finetune. Finally, Section D showcases more qualitative results with visual comparisons on both short and long prompts using FLUX.1 Dev and SD3.5 Medium.

### A. Code Implementation Details

Given that TACA necessitates modifications to the attention mechanism, and that the functions for computing attention are typically encapsulated within pre-compiled C/C++ binary libraries, directly reimplementing these attention computation functions using PyTorch would result in a significant performance degradation. To minimize the performance impact of modifying the attention mechanism while retaining the convenience of PyTorch, the following two implementation approaches for TACA can be adopted:

##### Flex Attention

- 1 from torch.nn.attention.flex_attention import flex_attention

- 2 gamma = 1.2

- 3 encoder_size = 512 # T5 encoder seq_len for FLUX

- 4

- 5 def score_mod(score, batch, head, token_q, token_kv):

- 6

- 7 condition = (token_q >= encoder_size) & ( token_kv < encoder_size)

- 8 score = torch.where(condition, score * gamma, score)

- 9 return score

- 10

- 11 hidden_states = flex_attention(query, key, value, score_mod=score_mod)

Listing 1. PyTorch Flex Attention

##### Selective Attention Recomposition

- 1 gamma = 1.2

- 2 encoder_size = 512 # T5 encoder seq_len for FLUX

- 3 key_scaled = key.clone()

- 4

- 5 # Shape of Q, K, V (B, H, N, D)

- 6 key_scaled[:, :, :encoder_size, :] *= gamma

- 7

- 8 # You can also change this into flash attention

- 9 hidden_states = F.scaled_dot_product_attention(

- 10 query, key_scaled, value, attn_mask= attention_mask, dropout_p=0.0, is_causal= False

- 11 )

- 12

- 13 hidden_states_orig = F. scaled_dot_product_attention(

- 14 query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False

- 15 )

- 16

- 17 hidden_states[:, :, :encoder_size, :] = hidden_states_orig[:, :, :encoder_size, :]

Listing 2. Selective Attention Recomposition

We conducted empirical evaluations of the computational speed of both proposed methods, comparing them against PyTorch’s native scaled dot-product attention implementation. All experiments employ a 30-step denoising process to generate 1024 × 1024 images via FLUX.1 Dev on a single Nvidia A100 80G GPU. We recorded the performance differential for both a single denoising step and for the complete 30-step denoising process (assuming temperature factor γ modification applied only to the initial 10% of steps). The results of this speed evaluation are presented in Table 6.

|Method|Single Step All 30 Steps Speedup<br><br>|
|---|---|
|Baseline Flex Selective|0.47 sec 14 sec 1.0x 2.13 sec 19 sec 0.74x 0.95 sec 16 sec 0.88x|

Table 6. Speed Comparison of Different Approaches

### B. Further Ablation Study on Text Alignment

#### B.1. The scale of CFG guidance

To investigate the text alignment improvements offered by our TACA method in comparison to increasing the CFG guidance scale (commonly employed in text-to-image models to enhance alignment, often at the cost of image quality), we conducted a series of ablation studies. These experiments aimed to determine whether TACA maintains its efficacy across varying CFG guidance scales and across different models. The results, presented in Table 7, reveal the effects of different CFG scales and the impact of TACA on both FLUX.1 Dev and SD3.5-Medium.

For FLUX.1 Dev, the default guidance scale of 3.5 appears to be a “sweet spot”: further increases in CFG intensity beyond this point yield minimal gains in text alignment, and, notably, performance across several metrics degrades significantly. Concurrently, our TACA method demonstrated effectiveness across diverse guidance scales, suggesting its general applicability.

Model Settings Color ↑ Shape ↑ Texture ↑ Spatial ↑

CFG = 3.5 (Default) 0.798 0.591 0.755 0.193 CFG = 3.5 + TACA 0.839 0.634 0.790 0.207 CFG = 5 0.787 0.553 0.756 0.175 CFG = 5 + TACA 0.835 0.635 0.757 0.224 CFG = 10 0.667 0.571 0.740 0.137 CFG = 10 + TACA 0.751 0.633 0.699 0.191

FLUX.1 Dev

CFG = 7 (Default) 0.812 0.730 0.850 0.159 CFG = 7 + TACA 0.843 0.737 0.864 0.183 CFG = 10 0.804 0.727 0.853 0.191 CFG = 10 + TACA 0.820 0.765 0.863 0.206

SD3.5-Medium

Table 7. Ablation study on the effect of CFG scale with and without TACA (with γ0 = 1.2) on FLUX.1 Dev and SD3.5-Medium. We randomly sampled 100 prompts for each attribute from the T2I-CompBench dataset to conduct the evaluation. For both models, bold indicates the best score and underline indicates the second-best score for each attribute.

For SD3.5-Medium, increasing the CFG scale also enhances text-image alignment but tends to degrade visual fidelity, resulting in reduced metrics (e.g., Color score drops at CFG=10 compared to CFG=7). Our TACA method, however, directly reinforces the dependence of image tokens on textual tokens, improving alignment without such adverse effects. TACA consistently improves results across different CFG scales on SD3.5-Medium, showing both generalization and complementarity.

Overall, the combined results across both models indicate that while increasing CFG can improve alignment to some extent, it often comes at the cost of overall performance. TACA, on the other hand, offers a more targeted and effective approach to enhancing text-image alignment, being beneficial and complementary across different CFG scales and diffusion models.

#### B.2. The content of the prompt

We have identified several prevalent issues regarding text alignment in state-of-the-art text-to-image models. Our TACA can mitigate these issues to a certain extent.

- • Difficulty in handling unrealistic scenarios, such as “a blue sun and a yellow sea”.
- • Difficulty in handling spatial relationships, such as with the prompt “a squirrel to the left of the man”. Models frequently interpret the left side of the image as the left side specified in the text, rather than the left side relative to the man’s frame of reference within the image.
- • Difficulty in handling specific numerical quantities. For instance, when prompted for four vases, the model may generate images containing five or three vases.

#### B.3. The length of the prompt

We also observe that models are more prone to omitting details from longer prompts, particularly when the prompt’s token count exceeds the maximum token limit supported by

the CLIP text encoder.

Our proposed TACA method demonstrates comparatively more widespread effectiveness for mitigating the attribute missing issues often found in longer prompt, rather than the shorter ones. Currently, a mature benchmark for evaluating the text-image alignment capabilities of text-toimage models with long prompts is lacking, despite the practical prevalence of longer prompts in real-world applications. Therefore, we have manually curated a set of authentic, long prompts from the internet to assess our method’s performance, and the corresponding results are presented in Fig. ??.

### C. Full parameter fine-tuning vs LoRA

In addition to LoRA training, we also experimented with full parameter fine-tuning as an alternative approach. However, we found that this method required significantly more computational resources and storage, especially for large models like FLUX.1 Dev. Moreover, our experiments revealed that full parameter fine-tuning is highly sensitive to learning rate settings. If the learning rate is set too high, the generated images tend to appear blurry or overly stylized, resembling oil paintings. On the other hand, if the learning rate is too low, the model struggles to learn the original data distribution effectively. These challenges, combined with the lack of superior artifact reduction compared to LoRA, led us to conclude that LoRA training is a more robust, efficient, and practical solution.

### D. More Qualitative Results

a black dress and a pink purse

a black pen and a white notebook

FLUX.1 Dev Ours FLUX.1 Dev Ours

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

a blue shirt and a yellow tie

a blue spoon and a silver plate

FLUX.1 Dev Ours FLUX.1 Dev Ours

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

a brown backpack and a blue horse

a brown bathroom a sink and a white mirror

FLUX.1 Dev Ours FLUX.1 Dev Ours

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

a fabric hat and a glass mirror

a fabric shirt and a glass window

a green bench and a blue boat

###### a metallic car and a wooden door

FLUX.1 Dev Ours FLUX.1 Dev Ours

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

a metallic fork and a wooden spoon

a plastic chair and a fabric pants

FLUX.1 Dev Ours FLUX.1 Dev Ours

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

a rubber band and a plastic cutlery

A white cat with black stomach takes a pose

FLUX.1 Dev Ours FLUX.1 Dev Ours

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

a white phone and a black charger

a wooden fork and a glass bowl

Many stoplights flash yellow on a snow covered street.

The blue pencil was next to the white paper

FLUX.1 Dev Ours FLUX.1 Dev Ours

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

The blue plate was on top of the white tablecloth.

The brown shoes were next to the black boots.

FLUX.1 Dev Ours FLUX.1 Dev Ours

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

The circular disk was wedged between the rectangular prism and the triangular pyramid.

The cold snowflake landed on the warm hand and the icy ground.

FLUX.1 Dev Ours FLUX.1 Dev Ours

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

The crisp apple lay beside the rough stone and the silky fabric.

The crisp green lettuce lay next to the juicy red tomato.

###### The fluffy white clouds floated above the tall skyscrapers.

The juicy watermelon quenched the thirsty mouth and the dry throat.

FLUX.1 Dev Ours FLUX.1 Dev Ours

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

The plastic sunglasses and fluffy beach towel lay on the wooden deck.

The prickly cactus stood tall on the dry sand and the rocky terrain.

FLUX.1 Dev Ours FLUX.1 Dev Ours

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

The red vase was on top of the wooden shelf.

The sweet banana split topped with the crunchy nuts and the soft whipped cream

FLUX.1 Dev Ours FLUX.1 Dev Ours

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

The wooden picture frame and glass photo display the metallic trophy on the bookshelf.

Two toilet stall, one blue and the other orange.

a backpack on the right of a person

###### a bag on the left of a dog

SD3.5 Medium SD3.5 Medium

Ours Ours

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

a bee on the top of a boy

a blue banana and a green vase

SD3.5 Medium SD3.5 Medium

Ours Ours

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

a blue bicycle and a red helmet

a blue cat and a brown chair

SD3.5 Medium SD3.5 Medium

Ours Ours

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

a book on the bottom of a cat

A bright white wall in a bathroom adds appeal to a yellow tiled floor

###### a car on the right of a bee

a clock on the top of a sheep

SD3.5 Medium SD3.5 Medium

Ours Ours

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

a fish on side of a airplane

a green backpack and a blue banana

SD3.5 Medium SD3.5 Medium

Ours Ours

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

A green purse is sitting on a brown bench

a red tomato and a yellow pepper

SD3.5 Medium SD3.5 Medium

Ours Ours

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

The bitter coffee sat next to the sweet donut and the savory bagel

The gentle, soothing touch of the massage therapist melted away the stress and tension, a therapeutic release

FLUX.1 Dev Ours FLUX.1 Dev Ours

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

in the style of Jed-clrfl, highly detailed 8K image, This is a super high-quality 8K photo. This is a high-quality photo of the cars from the Pixar Cars series, just like real cars. It shows Lightning McQueen racing against other cars from

A captivating, cinematic shot of a sun-drenched coastal city, nestled between towering, dramatic cliffs. The crystal-clear ocean waters lap gently against the pristine sandy shores, where colorful sailboats dot the horizon. The bustling marketplace is filled with the sounds of spirited haggling and laughter, while the tantalizing aroma of fresh seafood permeates the air. A mysterious, darkcloaked figure stands on a cliff overlooking the city, their gaze lost in the vast expanse of the sea, as if contemplating the mysteries of the world beyond. The painting captures the essence of an enchanting, picturesque landscape, with the dark fantasy element adding a touch of intrigue and depth.

the Pixar Cars series in a grand racing stadium. There are a lot of spectator cars in the stadium. Lightning McQueen is

in the lead, Jackson Storm is in second place, and Cheek Hicks is in third place. Several cars collide, creating huge flames and smoke, but Lightning McQueen, Jackson Storm and Chick Hicks luckily escape the crash and remain in the lead.

FLUX.1 Dev Ours FLUX.1 Dev Ours

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

High-angle close-up view of a collection of fallen leaves. The leaves are various shades of dark brown, deep purple, and muted gray tones, indicating the leaf fall's late autumn or early winter state. A single, distinctly lightercolored leaf, almost white, stands out amid the darker leaves, appearing to be a different type of plant. The light leaf has a maple-like shape. The leaves are densely packed together, creating a textured surface. The overall impression is one of a forest floor covered in decaying

A cozy oil painting of a pebble path going up towards a beautiful, blood-orange sunset with blue hue. On the left of the path are olive trees, on the right of the path is an old and worn out wall. The path is full of fireflies. The colors are intense, conveying a sense of isolation and vastness. The brushstrokes are bold and sweeping, while the fireflies are the focus of detail.

leaves. The lighting is diffused, not overly bright, and creates an overall subdued and muted atmosphere.

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

A very smug-looking chicken stands on a stage in a farmyard, wearing a glittery, oversized tuxedo and holding a tiny microphone. Behind him is a homemade banner that says, “Chicken Idol: Sponsored by Buzz.” To his left, a panel of farm animals—an unimpressed cow, a sheep with headphones, and a pig with sunglasses—sits at a long judging table with little buzz coins as score paddles. The chicken strikes an overly dramatic pose, one wing outstretched, the other clutching the microphone, as if he’s about to belt out a power ballad. His beak is wide open, mid-squawk, with musical notes floating comically out of it

A Cinematic Photography. Black-and-white photograph captures a young woman seated at a small round table in a cozy café setting. She is positioned slightly off-center to the right, with her left elbow resting on the table and her

head propped up on her hand, giving her a contemplative, pensive expression. Her hair is straight and falls just below her shoulders, framing her face. She is wearing a formfitting, long-sleeved black dress that accentuates her slender physique, paired with opaque black tights and glossy black high-heeled shoes. The table in front of her is simple, with a small white cup or saucer, likely for a beverage, placed on it. The café's interior features a rustic charm, with metal folding chairs and a wooden table with a worn finish. The background shows the glass front of the café, revealing a blurred street scene with parked cars and other indistinct objects, suggesting a bustling urban environment. The lighting is soft, likely natural, creating gentle shadows and highlights that enhance the textures of her clothing and the café's surfaces. The overall atmosphere is intimate and reflective, with a touch of melancholy due to the monochrome palette.

FLUX.1 Dev

Ours

[Figure 165]

[Figure 166]

FLUX.1 Dev Ours

A striking eco-brutalist living room with a tall ceiling, defined by its clean lines, raw textures, and a harmonious connection to nature. A floor-to-ceiling window offers an expansive view of a lush garden filled with swaying palm trees, allowing natural light to flood the space and create a serene, open atmosphere. The centerpiece of the room is a pair of sleek, white, modular sofas arranged to foster conversation. Their minimalist design contrasts beautifully with the raw concrete walls and polished concrete floor, which are signature elements of brutalist architecture. A soft, woven area rug in neutral tones grounds the seating area, adding warmth and texture. Above the sofas, contemporary art pieces in bold, abstract shapes and earthy hues are mounted on the wall, providing a dynamic focal point. The artwork juxtaposes the rugged materials of the room with creative, modern energy. A low, natural wood coffee table with an organic, irregular shape sits in the center, its surface adorned with a few carefully placed itemsâ ceramic vases, books, and a touch of greenery in a glass terrarium. Nearby, a statement plant, such as a tall monstera or bird-of-paradise, adds to the room's ecofriendly vibe. The lighting includes a sculptural pendant light hanging from the tall ceiling, its design inspired by nature with a modern twist. Subtle recessed lighting emphasizes the raw textures of the walls without overpowering the natural light streaming through the window. The lush garden outside, with its vibrant greenery and tall palm trees, feels like an extension of the room itself, framed perfectly by the expansive window. The interplay of natural elements, rugged architecture, and minimalist design creates a tranquil yet visually compelling space.

[Figure 167]

[Figure 168]

A breathtaking view from a magical fairy space station in outer space, gazing towards Earth. Iridescent ivory walkways lead between hybrid fantasy/sci-fi housing pods, moored together by sparkling gossamer ropes, stretch between each colossal pod. Along the path, elegant white lattice archways rise at regular intervals, each intricately detailed with swirling patterns. These arches are adorned with an explosion of vivid flowers in every imaginable color, from crimson roses to golden sunflowers and delicate lavender blooms, their petals seemingly untouched by gravity. The Earth below is illuminated, showing vibrant blues and greens, while the vast blackness of space is dotted with twinkling stars, adding a magical, dreamlike quality.

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

In the heart of a verdant Swiss village, nestled between towering, alpine mountains, a subtle yet profound moment of connection unfolds. The colorful timberframed houses, their facades painted in bright yellow and

a woman with long blonde hair, wearing a black top and a ring on her finger, standing in a city square with a fountain in the background, looking up and to the left, thoughtful pose, realistic city life scene, bright daylight

red, stand as a tapestry of harmony against the serene greens and blues of nature. The air is crisp, carrying the scent of pine and the faint, distant sound of a rushing river. A charming pedestrian bridge, its wooden planks worn smooth by countless footsteps, arcs gracefully over the water, connecting two sides of the village. Under the soft, golden light of the setting sun, a middle-aged couple, both with the weathered yet gentle faces of those who have lived and loved deeply, stand at the edge of the bridge. The man, a robust figure with a thick beard and a knitted cap, leans against the railing, his eyes locked on the woman standing a few steps away. She, with long, auburn hair that cascades down her back and a dress of deep blue, turns to meet his gaze. Her cheeks are flushed, and a gentle smile plays on her lips. The bridge, a symbol of connection and passage, amplifies the sense of mutual understanding and shared history between them. The wooden beams and slate roofs of the houses around them echo the steadfast nature of their bond, a bond that has weathered the seasons and the trials of time. The river, with its constant, soothing flow, mirrors the deep, enduring current of their love. In this moment, the world around them seems to pause, as if acknowledging the profound and timeless connection that exists between two hearts.

FLUX.1 Dev Ours

[Figure 173]

[Figure 174]

a man with fair skin, wearing a white vest and a purple shirt, standing in front of a futuristic blue and black machine with various buttons and screens, looking to his left with a surprised expression, dark and moody background with a hint of a spaceship in the distance, science fiction style

FLUX.1 Dev

Ours

FLUX.1 Dev Ours

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

a man with dark hair, wearing a blue apron, and a woman with shoulder-length hair, wearing a black shirt, engaging in conversation in a rustic kitchen, wooden wall background with various objects including a deer head, a clock, and a shelf with items, warm and cozy atmosphere

a young man with short dark hair, wearing a black cap and a blue life jacket, standing on a boat moving through water, looking to the side with a slight smile, rocky shore and water in the background, casual and candid capture

