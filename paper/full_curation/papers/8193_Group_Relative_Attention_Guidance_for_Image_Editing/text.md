# arXiv:2510.24657v2[cs.CV]28Nov2025

### Group Relative Attention Guidance for Image Editing

Xuanpu Zhang1,2*†, Xuesong Niu2*, Ruidong Chen1, Dan Song1, Jianhao Zeng1, Penghui Du2, Haoxiang Cao2, Kai Wu2‡, An-an Liu1‡

1 Tianjin University 2 Kolors Team, Kuaishou Technology

https://little-misfit.github.io/GRAG-Image-Editing/

[Figure 1]

Figure 1. Variation of editing strength with respect to the relative attention guidance scale. Our approach enables continuous and finegrained control of editing strength, striking a user-aligned balance between instruction following and consistency of original image.

#### Abstract

simple yet effective method that reweights the delta values of different tokens to modulate the focus of the model on the input image relative to the editing instruction, enabling continuous and fine-grained control over editing intensity without any tuning. Extensive experiments conducted on existing image editing frameworks demonstrate that GRAG can be integrated with as few as four lines of code, consistently enhancing editing quality. Moreover, compared to the commonly used Classifier-Free Guidance, GRAG achieves smoother and more precise control over the degree of editing. Our code will be released at https://github.com/little-misfit/GRAG-Image-Editing.

Recently, image editing based on Diffusion-inTransformer (DiT) models has undergone rapid development. However, existing editing methods often lack effective control over the degree of editing, limiting their ability to achieve more customized results. To address this limitation, we investigate the MM-Attention mechanism within the DiT model and observe that the Query (Q) and Key (K) tokens share a bias vector that is only layer-dependent. We interpret this bias as representing the model’s inherent editing behavior, while the delta between each token and its corresponding bias encodes the content-specific editing signals. Based on this insight, we propose Group Relative Attention Guidance (GRAG), a

#### 1. Introduction

Recently, Diffusion Transformer [26] models have once again advanced the field of text-to-image generation [4, 7, 10] and image editing [1, 2, 17–20, 44]. DIT employs a multi-modal attention mechanism [10] (MM-Attention) as

*Equal Contribution. †This work was conducted during the internship at Kolors Team. ‡Corresponding author.

its core to progressively inject semantic information from text into noisy latents, ultimately generating high-quality visual outputs through iterative denoising. Unlike UNetbased models [13, 27, 30, 36, 47] that separate crossattention and self-attention, the unified attention mechanism of DiTs provides a more holistic contextual understanding. This inherent advantage enables it to perform complex image editing even without task-specific fine-tuning [1, 40]. More recently, models such as Kontext [4, 18] and QwenEdit [44] further enhance text-driven editing capabilities by continuing training on specialized instruction-editing datasets, demonstrating powerful controllability and generalization.

[Figure 2]

Figure 2. The visualization of the embedding features input to the attention layer, where a significant bias can be observed across different tokens.

However, a persistent challenge for these instructionbased models is balancing the trade-off between maintaining fidelity to the source image and responsiveness to the editing instruction. As a result, this forces users to rely on external prompt-engineering tools or perform multiple inferences to achieve satisfactory outputs. To address this challenge, we conduct an in-depth investigation into the model’s internal feature propagation, specifically how textual and visual features are integrated during the editing process. Our analysis reveals that in the MM-Attention, the token distributions of the query and key embeddings tend to cluster around a dominant bias vector, as shown in Figure 2. Based on this finding, we demonstrate that by modulating the deviation of each token from this bias, it is able to achieve continuous control over the editing strength, ultimately producing controllable editing outputs.

Our investigation begins with an analysis of the embedding features in each attention layer [15]. We identify a consistent phenomenon: within each layer, feature values concentrate around a shared bias vector. Based on the formulation of MM-Attention, this bias phenomenon can be interpreted as an intrinsic inductive pattern introduced by the architecture itself. We hypothesize that the variation of individual tokens from this bias encodes crucial contextual understanding (the theoretical analysis is presented in Section 4).

This insight directly motivates our method, Group

Relative Attention Guidance (GRAG), a guidance mechanism also inspired by the Group Relative Policy Optimization (GRPO) [34] strategy.

As illustrated in Figure 3, GRAG first computes the average Key embedding within each token group to determine a collective editing direction (the common bias vector). Then, a weighting coefficient λ is used to modulate each token’s ∆ vector relative to the bias, enhancing those aligned with the editing intent while suppressing conflicting ones. This process leads to more precise and controllable editing outputs. We validate our method on state-of-theart DIT-based editing models [18, 21, 44]. With a fixed guidance scale, our approach achieves a better trade-off between the editing responsiveness and image consistency, while continuous coefficient adjustment on fixed samples yields smooth and progressive editing outputs (as shown in Figure 1).

[Figure 3]

Figure 3. Group Relative Attention Guidance.

Finally, our contributions can be summarized in three aspects: (a) Through extensive experiments, we identify the presence of a bias distribution in the Query and Key embeddings of MM-DIT, and we provide a mathematical analysis of its role in image editing tasks. (b) We introduce Group Relative Attention Guidance (GRAG), a novel approach that leverages the relative relationships among tokens to modulate the image editing process, enabling precise and controllable editing by modulating their deviations from the group bias. (c) We conduct extensive experiments on multiple baselines, and evaluate performance across diverse editing tasks, demonstrating the effectiveness of our method.

#### 2. Related Work

##### 2.1. Text-driven Image Editing

Early works such as InstructPix2Pix [5] demonstrated that synthetic instruction–response pairs can effectively finetune diffusion models for image editing, while trainingfree methods like Textual Inversion and DreamBooth [11, 32] enabled editing with off-the-shelf generative models [11, 30]. Building on this foundation, subsequent editors—including Emu Edit [35], OmniGen [45], HiDreamE1 [6], and ICEdit [48]—enhanced instruction-driven editing through refined datasets and architectures, while LoRAbased methods [14] introduced task-specific parameter tuning for diffusion transformers. Proprietary multimodal systems such as GPT-4V [25] and Gemini [38], along with platforms like Midjourney [24] and RunwayML [33], have

further integrated these advances into end-to-end creative workflows. Kontext[18] extends the FLUX.1[4] model for editing tasks, leveraging its strong contextual modeling capability to achieve high consistency with reference images. In contrast, Qwen-Edit [44] and Step1X-Edit [21] enhance instruction comprehension through vision language models, enabling more complex and flexible editing operations. Despite progress, instruction-driven image editing still faces two challenges: (i) striking a balance between editing effectiveness and consistency with the source image, (ii) achieving precise and continuous control over editing effects.

##### 2.2. Editing Strength Control for Image Editing

Recent progress in controlling editing strength mainly focused on text-to-image [4, 10] (T2I) generation. Methods [3, 8, 12] such as ConceptSlier [12] adjust the influence of specific textual tokens to modulate the corresponding visual concepts, while TACA [23] improves text–image alignment by reweighting the attention scores of text tokens in multi-modal attention. Although effective in generative settings, these approaches are not directly applicable to real image editing (TI2I), where both the source image and the editing instruction must be jointly considered. To enable controllable editing on real images, Freeflux [43] employs a gated attention mechanism to regulate the spatial extent of edited regions, and SaaS [49] modulates instruction strength through attention reweighting, but direct manipulation of attention maps often causes artifacts and limits compatibility with fast attention mechanisms such as Flash Attention [9]. We address these challenges by introducing Group Relative Attention Guidance (GRAG), which analyzes internal attention representations of DiT models and modulates embedding features for precise and continuous control of editing strength with minimal overhead.

#### 3. Preliminaries

Multi-Modal Diffusion Transformers. The multi-modal diffusion transformer framework, known as multi-modal diffusion transformers (MM-DiT) [10, 26, 39], merges both textual and visual modalities to generate images that align with the semantics of the textual inputs. FLUX incorporates a unified text-image self-attention mechanism, which aligns the multimodal information within each MM-DiT layer. Moreover, FLUX enhances the CLIP [28] text encoder by integrating the T5 [29] encoder, significantly improving its text understanding capabilities.

The MM-DiT layer uses a combined attention mechanism to fuse textual and visual data. Initially, the text tokens T and image tokens I are mapped into a shared space:

Qt = TWQt , Kt = TWKt , Vt = TWVt , Qi = IWQi , Ki = IWKi , Vi = IWVi ,

(1)

where WQt ,WKt ,WVt ∈ Rd

t×d and WQi ,WKi ,WVi ∈ Rd

i×d represent the projection matrices, and d denotes the shared dimension. Subsequently, the joint attention Ajoint is calculated by combining the queries and keys from both the text and image modalities:

[Qt ⊕ Qi][Kt ⊕ Ki]⊤ √

[Vt ⊕ Vi] (2)

Ajoint = Softmax

d

where ⊕ denotes the token-wise concatenation of the text and image tokens. During the image editing process, the visual information consists of both the editing target and the original image: Qi = [Qe ⊕ Qs], Ki = [Ke ⊕ Ks] and Vi = [Ve⊕Vs]. The computation process of the corresponding attention map during editing image token update is as follows:

Sedit(i, j) = Softmax(Qe[Kt ⊕ Ki])(editi, j)

i e,ktj⟩

e⟨q

,

=

Nimg

Nimg

Ntxt

i e,ktp⟩

i e,kep⟩

i e,ksp⟩

e⟨q

e⟨q

e⟨q

+

+

p=1

p=1

p=1

Text

Source

Editing

Note: For simplicity, the √

d is omitted.

(3)

#### 4. Bias Vector In The Embedding Vectors

The attention layer of MM-DiT serves as the key location where editing instructions and conditional image information are fused, with the query and key embeddings directly influencing the proportion of content sampled from each token. Our experiments reveal a significant bias in the distribution of embedding features along the sequence dimension, concentrated at fixed positions within each token. We hypothesize that this bias serves as a key factor in contextual understanding during the image editing process of DiT.

Concentrated distribution of embedding vectors. For each attention layer of the transformer, we extract the query and key embeddings with shape Q,K ∈ RB×S×H×D. For analysis purposes, we fix the batch size to B = 1 and partition the sequence dimension S into six semantically meaningful components: Qtext, Qedit, Qsource, and similarly Ktext, Kedit, Ksource. Here, Qtext,Ktext ∈ RN

text×H×D and the remaining components belong to RN

img×H×D. We apply L2 normalization along the Ntext or Nimg dimension, reducing each component to a representation in E ∈ RH×D, where each element Eh,d represents the norm of the corresponding component in head h and dimension d. Taking Qedit as an example, Eh,d is computed as:

Eh,d = ∥Q:,h,d∥2 =

Nimg

Q2s,h,d (4)

s=1

[Figure 4]

Figure 4. (Kontext-Layer 2) Aggregating different tokens along the sequence dimension, we visualize the embedding features across the dimension and head axes. The visual features are concentrated at positions corresponding to high RoPE frequencies, while textual features are associated with low frequencies.

The visualization results of E are shown in Figure 4. In the embedding vector space, each dimension index corresponds to a component, where the dark red regions in Figure 4 indicate positions with larger magnitudes that contribute more to the inner product between different token embeddings. By examining the relationship between RoPE (Rotary Position Embedding [37]) and dimension indices, we observe that text embeddings concentrate in low-frequency components associated with semantics, while image embeddings concentrate in high-frequency components capturing spatial relations. This finding suggests that the two modalities are not fully aligned in the shared embedding space. Furthermore, we investigate the distribution of token embeddings in the vector space. Figure 5 presents the mean vector magnitudes and standard deviations across different attention heads, further revealing the presence of a significant bias vector among tokens in the embedding space.

Analysis of the bias vector. The above findings suggest that the query and key embeddings in the attention layer exhibit a decomposable structure, where each can be represented as the sum of a dominant bias component and an independent variation:

###### qi = qbias + ∆qi, ki = kb + ∆ki (5)

We also observe that the feature distributions of the same layer remain highly similar across different time steps and input samples. Based on this phenomenon, we hypothesize that the bias vector qbias,kbias is related to the model weights and represents a fixed “editing action” during the image editing process, while the variations of individual tokens relative to this bias vector correspond to the “content” being edited. Based on Equation 3, we can derive:

###### i e,∆ktj⟩

e,ktbias⟩e⟨q

i

e⟨q

Sedit(i, j) =

,

e⟨qei,ktbias⟩Σt + e⟨qei,kebias⟩Σe + e⟨qei,ksbias⟩Σs

(6)

[Figure 5]

Figure 5. (Kontext-Layer 2) Mean vector magnitudes and standard deviations across different attention heads. A significant bias vector exists in the embedding space.

e,∆ktp⟩,Σe =

i

Note: For simplicity, the Σt = Np=1txt e⟨q

e,∆kep⟩,Σs = Np=1img e⟨q

Nimg p=1 e⟨q

e,∆ksp⟩.

i

i

A strong shared bias component in both query and key embedding can dilute the influence of ∆k, thereby reducing the sensitivity of attention scores to specific semantic differences. This insight naturally suggests that by modulating the magnitude of ∆k, one can effectively control the extent to which the conditioning signals (e.g., edit instructions) influence the final output.

[Figure 6]

Figure 6. An illustration of applying Group Relative Attention Guidance in the MM-DiT image editing model. (a) The MM-Attention map corresponding to the query Qe, where GRAG is applied. (b) The processing of relative modulation to the source image’s key embeddings. Red denotes enhanced tokens, while blue denotes suppressed tokens.

#### 5. Group Relative Attention Guidance

The variations between individual token embeddings and the bias vector reflect how the editing content relates to the current layer’s editing action. By modulating their relative relationship, it becomes possible to achieve accurate and continuous control over the editing instructions. Based on this insight, we propose Group Relative Attention Guidance (GRAG). As illustrated in Figure 6, we modify the crossattention component of the MM-Attention corresponding to the query Qe. In Figure 6, Ks is selected as a group of tokens, to which group-relative modulation is applied.

Algorithm 1 Group-Relative Attention Guidance Input: Embedding Q,K,V ∈ RB×S×H×D, token index

istart,iend, guidance scale λ,δ. Output: Updated attention Aˆ.

- 1: Q,K,V = RoPE(Q),RoPE(K),V
- 2: Ks ← K[:,istart : iend,:,:]
- 3: Kbias ← mean(Ks,dim = 1)
- 4: K∆ ← Ks − Kbias
- 5: K[:,istart : iend,:,:] ← λ ∗ Kbias + δ ∗ K∆
- 6: Aˆ ← Attention(Q,K,V )

Formally, let ksi denote the conditional key embedding corresponding to token i, where i = 1,...,Nimg. We first compute a group-level bias component as the mean of all conditional keys:

Nimg

1 Nimg

ksj (7)

kbias =

j=1

The deviation of each token from this bias is then defined as:

∆ki = ksi − kbias (8)

To control the influence of token-level variations, we introduce a tunable parameter λ that scales these deviations:

kˆsi = λ · kbias + δ · ksi − kbias (9)

where kˆsi denotes the updated key embedding under group relative attention guidance.

The scaling factor, λ and δ, are introduced to modulate the balance between the shared bias and token-specific variations. Both λ and δ are positive real numbers. Specifically, λ > 1 enhances the influence of the selected tokens on the final image content, while λ < 1 reduces their impact. On the other hand, δ adjusts the focus intensity towards the selected tokens: δ > 1 results in a more concentrated and precise editing impact, whereas δ < 1 leads to a more diffused editing effect. The pseudo-code of Group Relative Attention Guidance is presented in Algorithm 1, which consists of only four lines and can be seamlessly integrated into existing methods.

#### 6. Experiment

##### 6.1. Experiment Setting

Baselines. We apply the fixed GRAG parameters across six image editing models to validate GRAG’s modulation effectiveness. Kontext [18], Step1X-Edit [21] and QwenEdit [44] are Text-Image to Image (TI2I) editing method. And three training-free Text to Image (T2I) editing methods based on Flux.1-Dev[4] models (Flowedit [17], Stableflow [1], Stableflow+). In these methods, GRAG is applied to the attention layers where source image features are injected. For training-based methods, GRAG is applied to all timesteps and layers, while for training-free methods, it is applied only during attention injection. The random seed is fixed to 42. All experiments are conducted with a batch size of 1 and 24 inference steps.

[Figure 7]

Figure 7. Visualization results on training-based image editing method.

[Figure 8]

- Figure 8. Visualization results on training-free image editing method. We update the original first-order inversion in StableFlow with a second-order ODE inversion method[31, 41], referred to as StableFlow+.

We compare our approach with the mainstream guidance method: Classifier-Free Guidance (CFG) and two attention control methods. Gated Attention [43] (denoted as Attngate) controls the editing strength by setting a threshold on the attention correlation values computed between tokens, thereby limiting the number of tokens that respond to the editing instruction. In contrast, Attention Reweighting [49] (Attn-weight) adjusts the editing strength by reweighting the attention scores produced by the SoftMax layer. All controllable editing strength analysis experiments are conducted based on the Qwen-Edit model.

Evaluation. We evaluate our method on PIE [16]. This benchmark covers a diverse range of editing tasks, including object addition/removal, style transfer, and pose modification. For quantitative evaluation, we adopt two complementary perspectives. Following previous works, we adopt LPIPS[46] and SSIM[42] as quantitative metrics to evaluate the content preservation ability in non-edited regions.

To assess the alignment between editing results and human preference, we employ the image editing reward model EditScore[22]. EditScore measures three aspects: consistency with the original image (Cons), prompt following (PF), and overall edit score (EditScore).

##### 6.2. Qualitative Analysis

We apply GRAG to three mainstream MM-DiT-based image editing models, with qualitative results shown in Figure 7. On Step1X-Edit and Qwen-Edit, our method improves consistency between the edited images and the original references while preserving the intended editing effects, yielding more realistic and natural outcomes. Since Step1X-Edit and Qwen-Edit leverage vision–language models to encode editing instructions, the additional instruction information often enhances responsiveness but reduces consistency. We select the source image tokens as group and apply GRAG to enhance the response of edit-related tokens to the editing instructions

[Figure 9]

- Figure 9. Visualization results of different editing strength control methods. Attn-gate controls edits by limiting token activation and cannot enhance existing results (replaced with blanks). Red arrows mark visual artifacts. Overall, adjusting δ achieves the most continuous and precise editing control.

while suppressing the response of irrelevant tokens. For instance, in the first column of Figure 7, GRAG successfully changes the texture of the bird while retaining the details of the tree trunk; in the fifth column, it alters the color of the apple while preserving fine-grained surface details. These examples demonstrate the ability of GRAG to achieve precise and continuous control over edits while maintaining fidelity to the source image. For the original Kontext model, we select the text tokens as the group and apply GRAG to enhance the model’s response to the editing instructions. As shown on the right side of Figure 7, the baseline fails to respond to the editing instruction, with no change in content, whereas applying GRAG enables successful editing.

As shown in Figure 8, our approach achieves adjustment of the editing results, indicating that GRAG remains effective in training-free editing method.

##### 6.3. Quantitative Analysis

As shown in Tab 1, we perform quantitative evaluations on the PIE dataset. Step1X-Edit and Qwen-Edit exhibit enhanced consistency between the edited outputs and the original images after integrating GRAG, as indicated by improvements in LPIPS, SSIM, and Cons. Although a slight decline is observed in PF, the overall EditScore, which reflects overall editing quality, increases. In contrast, Kontext demonstrates a noticeable improvement in PF and achieves a higher EditScore after applying GRAG. These trends align well with the visual results.

Model LPIPS↓ SSIM↑ Cons↑ PF↑ EditScore↑ Training-Based

Kontext-Dev 0.3061 0.9213 8.9051 6.9051 6.0887 +GRAG 0.3873 0.8156 8.6788 7.4177 6.4081 Step1X-Edit 0.3228 0.9042 8.4714 7.8406 6.8292 +GRAG 0.3174 0.9137 8.6240 8.0406 7.0045 Qwen-Edit 0.3428 0.8506 8.5211 8.4806 7.2576 +GRAG 0.3042 0.9263 8.9440 8.3303 7.3245 Training-Free

Flowedit 0.3758 0.8237 6.8794 5.0531 4.6635 +GRAG 0.3670 0.8312 7.2223 4.8954 4.6697 StableFlow 0.3219 0.9185 8.9309 2.2177 2.4573 +GRAG 0.3292 0.9098 8.8731 2.7429 3.0303 StableFlow+ 0.3691 0.8229 7.3599 5.3926 5.0970 +GRAG 0.3595 0.8316 7.7997 4.8395 4.7251

Table 1. Quantitative results on different image editing methods.

As shown in Table 1-Bottom, GRAG also provides performance gains on training-free editing methods. However, its adaptability is relatively lower compared to trainingbased methods. We attribute this to the fact that GRAG primarily modulates the cross-attention component in MMAttention (Figure 6), whereas in untrained T2I models, source image features are introduced through the edit–edit self-attention branch (Figure 6-b). We provide further discussion in the Supplementary Material.

##### 6.4. Ablation Study

Comparison with other guidance method. Although Classifier-Free Guidance (CFG) can influence the generation results by adjusting the denoising process, it provides

[Figure 10]

- Figure 10. Comparison of different guidance strategies under varying guidance strengths. The data in the line chart correspond to Table 2. The δ parameter yields the most continuous and effective editing guidance.

Method LPIPS ↓ SSIM↑ Cons ↑ PF ↑ EditScore ↑

CFG = 5.00 0.3381 0.8548 8.3989 8.4640 7.1857 CFG = 4.00 0.3428 0.8506 8.5211 8.4806 7.2576 CFG = 3.00 0.3312 0.8659 8.6251 8.3954 7.2761 CFG = 2.00 0.3297 0.8709 8.6669 8.2566 7.2247 CFG = 1.00 0.3306 0.8734 8.7686 7.6760 6.8294

- Attn-weight = 0.8 0.3223 0.8940 8.5846 8.4503 7.2808

- Attn-weight = 1.0 0.3428 0.8506 8.5211 8.4806 7.2576 Attn-weight = 7.0 0.3055 0.9261 8.8301 8.2159 7.2214

Attn-weight = 10.0 0.2916 0.9484 9.1130 6.7797 5.9555 Attn-weight = 16.0 0.2825 0.9560 9.2361 4.5230 3.9953

Attn-gate = 1.0 0.3428 0.8506 8.5211 8.4806 7.2576 Attn-gate = 0.9 0.3094 0.9031 8.6460 8.1206 7.4004 Attn-gate = 0.7 0.3103 0.9033 8.6655 7.9611 7.2531 Attn-gate = 0.3 0.3296 0.8673 8.1051 7.7184 4.6505

- λ = 0.95,δ = 1.00 0.3316 0.8660 8.5286 8.4886 7.2725

- λ = 1.00,δ = 1.00 0.3428 0.8506 8.5211 8.4806 7.2576 λ = 1.05,δ = 1.00 0.3123 0.9156 8.5914 8.3977 7.2990 λ = 1.10,δ = 1.00 0.3194 0.9034 8.3291 7.9251 7.1992 λ = 1.15,δ = 1.00 0.3307 0.8865 8.3720 8.3269 7.1863

- λ = 1.00,δ = 0.95 0.3508 0.8344 8.2543 8.4394 7.1991

- λ = 1.00,δ = 1.00 0.3428 0.8506 8.5211 8.4806 7.2576 λ = 1.00,δ = 1.05 0.3188 0.8855 8.7549 8.3651 7.2679 λ = 1.00,δ = 1.10 0.3034 0.9206 8.9537 7.9611 6.9872 λ = 1.00,δ = 1.15 0.2907 0.9408 9.1731 6.9291 6.1730

- λ = 0.95,δ = 0.95 0.3454 0.8434 8.3560 8.3394 7.2162

- λ = 1.00,δ = 1.00 0.3428 0.8506 8.5211 8.4806 7.2576 λ = 1.05,δ = 1.05 0.3042 0.9263 8.9440 8.3303 7.3245 λ = 1.10,δ = 1.10 0.2971 0.9391 9.0234 7.9674 7.0243 λ = 1.15,δ = 1.15 0.2885 0.9448 9.1051 6.6091 5.9955

Table 2. Continuity and effectiveness analysis of different editing strength control methods. Appropriate parameters are selected for each method within its effective working range.

only limited control over editing strength and often leads to a noticeable degradation in image quality when the guidance scale is small (e.g., CFG = 1). Both attention-based control methods produce visible effects on the edited results; however, Attn-gate exhibits discontinuous control and tends to generate distorted outputs, as shown in the second row on the left of Figure 9. Attn-weight achieves a modulation effect most similar to ours, but it may cause incorrect edits—for example, the cat is mistakenly replaced by a horse in Figure 9—and it still fails to ensure smooth and consistent editing, where the flowers first wither and then become unnaturally vibrant as the editing strength in-

creases. As shown in Table 2 and Figure 10, GRAG enables precise and continuous control over the editing, producing smooth and consistent adjustments as the editing strength increases, the visual comparsion shown in Figure 9. Such controllability is crucial for customized image editing applications.

Effectiveness Analysis of Group Relative. We analyze the influence of the λ and δ parameters in Eq. 9 on the editing results. Three groups of experiments are conducted: only λ, only δ, and both λ and δ simultaneously. The qualitative results are shown in Figure 9, while the quantitative results on the PIE benchmark are presented in Table 2 and Figure 10. Adjusting λ alone shows no significant impact on the editing results, corresponding to the fluctuating curves in Figure 10, which indicates that tuning λ cannot effectively control the editing strength. In contrast, jointly adjusting λ and δ enables a certain degree of controllable editing but fails to achieve continuous precision. Moreover, this simultaneous adjustment often degrades visual fidelity, leading to undesirable artifacts such as the distorted flowers and the visible artifacts of the horse sample in Figure 9. Adjusting δ alone yields the best results, corresponding to the smoothest metric variation in Figure 10 and the most continuous editing transitions in Figure 9.

#### 7. Conclusion

In this work, we revisited the internal attention mechanism of Diffusion-in-Transformer (DiT) models and revealed the presence of a shared bias vector that governs editing behavior. Building on this insight, we introduced Group Relative Attention Guidance (GRAG), a lightweight yet effective strategy that modulates token deviations from the group bias to achieve fine-grained and continuous control over editing strength. GRAG can be seamlessly integrated into existing DiT-based editors, consistently improving both controllability and fidelity. Our findings provide new insights into the internal dynamics of multi-modal attention and offer a practical direction for enhancing controllable image editing in future DiT architectures.

#### References

- [1] Omri Avrahami, Or Patashnik, Ohad Fried, Egor Nemchinov, Kfir Aberman, Dani Lischinski, and Daniel CohenOr. Stable flow: Vital layers for training-free image editing. arXiv preprint arXiv:2411.14430, 2024. 1, 2, 5
- [2] Yuxiang Bao, Huijie Liu, Xun Gao, Huan Fu, and Guoliang Kang. Freeinv: Free lunch for improving ddim inversion. arXiv preprint arXiv:2503.23035, 2025. 1
- [3] Stefan Andreas Baumann, Felix Krause, Michael Neumayr, Nick Stracke, Melvin Sevi, Vincent Tao Hu, and Bj¨orn Ommer. Continuous, subject-specific attribute control in T2I models by identifying semantic directions. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 13231–13241. Computer Vision Foundation / IEEE, 2025. 3
- [4] Black Forest Labs. Flux.1-dev. https : / / huggingface . co / black - forest - labs / FLUX.1-dev, 2024. 1, 2, 3, 5
- [5] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023. 2
- [6] Qi Cai, Jingwen Chen, Yang Chen, Yehao Li, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Yiheng Zhang, Fengbin Gao, Peihan Xu, et al. Hidream-i1: A high-efficient image generative foundation model with sparse diffusion transformer. arXiv preprint arXiv:2505.22705, 2025. 2
- [7] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 1
- [8] Yusuf Dalva, Kavana Venkatesh, and Pinar Yanardag. Fluxspace: Disentangled semantic editing in rectified flow models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 13083–13092. Computer Vision Foundation / IEEE, 2025. 3
- [9] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memory-efficient exact attention with io-awareness. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. 3
- [10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 1, 3

- [11] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 2

- [12] Rohit Gandikota, Joanna Materzynska, Tingrui Zhou, Antonio Torralba, and David Bau. Concept sliders: Lora adaptors for precise control in diffusion models. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XL, pages 172–188. Springer, 2024. 3
- [13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [14] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 2
- [15] Mingyu Jin, Kai Mei, Wujiang Xu, Mingjie Sun, Ruixiang Tang, Mengnan Du, Zirui Liu, and Yongfeng Zhang. Massive values in self-attention modules are the key to contextual knowledge understanding. arXiv preprint arXiv:2502.01563,

2025. 2

- [16] Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. Pnp inversion: Boosting diffusion-based editing with 3 lines of code. International Conference on Learning Representations (ICLR), 2024. 6
- [17] Vladimir Kulikov, Matan Kleiner, Inbar HubermanSpiegelglas, and Tomer Michaeli. Flowedit: Inversion-free text-based editing using pre-trained flow models. CoRR, abs/2412.08629, 2024. 1, 5
- [18] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742,

2025. 2, 3, 5

- [19] Huijie Liu, Bingcan Wang, Jie Hu, Xiaoming Wei, and Guoliang Kang. Omni-dish: Photorealistic and faithful image generation and editing for arbitrary chinese dishes. arXiv preprint arXiv:2504.09948, 2025.
- [20] Huijie Liu, Jingyun Wang, Shuai Ma, Jie Hu, Xiaoming Wei, and Guoliang Kang. Separate motion from appearance: Customizing motion via customizing text-to-video diffusion models. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 9227–9236, 2025. 1
- [21] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 2, 3, 5
- [22] Xin Luo, Jiahao Wang, Chenyuan Wu, Shitao Xiao, Xiyan Jiang, Defu Lian, Jiajun Zhang, Dong Li, and Zheng Liu. Editscore: Unlocking online RL for image editing via highfidelity reward modeling. CoRR, abs/2509.23909, 2025. 6
- [23] Zhengyao Lv, Tianlin Pan, Chenyang Si, Zhaoxi Chen, Wangmeng Zuo, Ziwei Liu, and Kwan-Yee K Wong. Rethinking cross-modal interaction in multimodal diffusion transformers. arXiv preprint arXiv:2506.07986, 2025. 3
- [24] Midjourney. Midjourney. https://www.midjourney. com, 2022. 2
- [25] OpenAI. Gpt-4v(ision) system card. https://openai. com/research/gpt-4v-system-card, 2023. 2

- [26] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 1, 3

- [27] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2
- [28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 3
- [29] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020. 3
- [30] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [31] Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Semantic image inversion and editing using rectified stochastic differential equations. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. 6
- [32] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 2
- [33] Runway. Runway. https://runwayml.com, 2023. 2
- [34] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 2
- [35] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8871– 8879, 2024. 2
- [36] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. 2
- [37] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 4

- [38] Gemini Team et al. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 2
- [39] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 3
- [40] Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing. arXiv preprint arXiv:2411.04746, 2024. 2
- [41] Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing. CoRR, abs/2411.04746, 2024. 6
- [42] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 6
- [43] Tianyi Wei, Yifan Zhou, Dongdong Chen, and Xingang Pan. Freeflux: Understanding and exploiting layer-specific roles in rope-based mmdit for versatile image editing. CoRR, abs/2503.16153, 2025. 3, 6
- [44] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 1, 2, 3, 5
- [45] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13294–13304, 2025. 2
- [46] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 6
- [47] Xuanpu Zhang, Dan Song, Pengxin Zhan, Tianyu Chang, Jianhao Zeng, Qingguo Chen, Weihua Luo, and An-An Liu. Boow-vton: Boosting in-the-wild virtual try-on via mask-free pseudo data training. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 26399–26408. Computer Vision Foundation / IEEE, 2025. 2
- [48] Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. In-context edit: Enabling instructional image editing with incontext generation in large scale diffusion transformer. arXiv preprint arXiv:2504.20690, 2025. 2
- [49] Chao Zhou, Tianyi Wei, and Nenghai Yu. Scale your instructions: Enhance the instruction-following fidelity of unified image generation model by self-adaptive attention scaling. CoRR, abs/2507.16240, 2025. 3, 6

## Group Relative Attention Guidance for Image Editing Supplementary Material

#### Contents

- A. Pytorch Implementation of GRAG 2
- B. Theoretical Analysis of Group Relative Attention Guidance 3

- B.1. Toy Experiment of the GRAG Theoretical Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- B.2. Relationship Between Attention Entropy and GRAG . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- B.3. GRAG of Source Image Tokens . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- B.4. GRAG of Editing Text Tokens . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- C. Limitation & Discussion 8
- D. Additional Qualitative Results 9
- E. Additional Feature Visualization 11

- E.1. Kontext Embedding Visualization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- E.2. Qwen-Edit Embedding Visualization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

#### A. Pytorch Implementation of GRAG

The proposed Group Relative Attention Guidance (GRAG) can be seamlessly integrated into existing DiT-based image editing models with only a few lines of code modification. Below, we provide an example implementation of GRAG based on a typical MM-Attention block from the Diffusers library in PyTorch.

Listing 1. Implementation Code of GRAG

- 1 # Apply RoPE

- 2 if image_rotary_emb is not None:

- 3 img_freqs, txt_freqs = image_rotary_emb

- 4 img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)

- 5 img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)

- 6 txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)

- 7 txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

- 8

- 9 # Apply GRAG scaling

- 10 s idx, e idx, bias scale, delta scale = 4096, 8192, 1.0, 1.05

- 11 group bias = img key[:, s idx:e idx, :, :].mean(dim=1)

- 12 img key[:, s idx:e idx, :, :] = bias scale * group bias +

- 13 delta scale * (img key[:, s idx:e idx, :, :] - group bias)

- 14

- 15 # Joint attention computation

- 16 joint_query = torch.cat([txt_query, img_query], dim=1)

- 17 joint_key = torch.cat([txt_key, img_key], dim=1)

- 18 joint_value = torch.cat([txt_value, img_value], dim=1)

- 19

- 20 joint_hidden_states = dispatch_attention_fn(

- 21 joint_query,

- 22 joint_key,

- 23 joint_value,

- 24 attn_mask=attention_mask,

- 25 dropout_p=0.0,

- 26 is_causal=False,

- 27 backend=self._attention_backend,

- 28 )

#### B. Theoretical Analysis of Group Relative Attention Guidance

##### B.1. Toy Experiment of the GRAG Theoretical Analysis

To further validate the theoretical analysis presented in the main paper and Section 5, we design a controlled toy experiment that isolates the effect of Group Relative Attention Guidance (GRAG) on attention score modulation. The aim is to empirically verify how adjusting the deviation scaling factor δ and the bias scaling factor λ influences the distribution of attention across token groups.

Experimental Setup. We simplify the MM-Attention mechanism in a minimal setting consisting of a single query q = 1 and three key tokens.

i⟩

e⟨q,k

A(q,ki) =

e⟨q,ks1⟩ + e⟨q,ks2⟩ + e⟨q,kt⟩ , (S1)

where ks1 = 1.9 and ks2 = 1.2 denote two source-image tokens and kt = 3.4 denotes the editing-text token. Following the setup in Section 5 of the main paper, two tokens form the source image token group, while the remaining token represents the editing text token. The raw inner-product responses ⟨q,ki⟩ of these tokens serve as the unmodulated attention logits. This abstraction preserves the structural essence of text–image cross-attention while allowing us to precisely examine how GRAG affects token-wise attention allocation. Ideally, when increasing editing strength toward higher image–content consistency, the model should maintain responsiveness to the editing instruction while smoothly increasing the contribution of selected source-image tokens.

We compare four representative modulation strategies:

- • Attention Weight - γ: Directly scaling attention scores after the Softmax layer.
- • GRAG - λ: Modulating only the bias component while holding the deviation term fixed.
- • GRAG - λ,δ: Jointly scaling both the bias component and token-level deviations.
- • GRAG - δ: Modulating only the deviation component while keeping the bias fixed.

Experimental Results. Figure S1 illustrates this desired behavior using a toy example. The horizontal axis denotes the respective modulation parameter, and the vertical axis shows the post-Softmax attention scores of three token groups: suppressed source-image tokens, enhanced source-image tokens, and editing-text tokens. As seen in subfigures (a)–(c), increasing γ or λ, or jointly tuning λ and δ, rapidly suppresses attention to the editing-text tokens, causing the editing instruction to collapse and resulting in artifacts or editing failure despite increased attention to the image tokens.

In contrast, deviation-only modulation via GRAG-δ preserves stable attention on editing-text tokens while smoothly adjusting the relative emphasis among source-image tokens. This yields the most continuous and controllable editing-strength transition, free of undesirable discontinuities or instruction loss. Overall, GRAG-δ achieves the desired balance between instruction following and image consistency, providing the most reliable and precise editing-strength control among all tested strategies.

[Figure 11]

- Figure S1. Toy Experiment: Effects of different attention modulation strategies on the attention scores within MM-Attention. GRAG-δ enhances or suppresses image tokens while preserving responsiveness to the editing instruction, enabling continuous and precise control over editing strength. The left vertical line marks the lower modification boundary, and the y-axis denotes the baseline without any modulation.

##### B.2. Relationship Between Attention Entropy and GRAG

To quantify how GRAG influences the distribution of attention inside MM-Attention, we measure the attention entropy of each attention map. Given an attention distribution A(l) ∈ RN from layer l, its entropy is defined as:

N

A(il) log A(il), (S2)

H(A(l)) = −

i=1

where lower entropy indicates a more concentrated attention pattern and stronger focus on specific tokens.

We compute attention entropy for all layers of the Qwen-Image-Edit model under different GRAG values. Specifically, for each GRAG scale, we extract the attention maps associated with the source-image tokens and compute their entropies using Equation (S2). The aggregated results are shown in Figure S2(a). As the GRAG value increases, the attention entropy in every layer decreases monotonically, indicating that GRAG strengthens the model’s focus on conditioning information and thereby increases the effective editing strength.

To provide a more intuitive interpretation, we visualize the attention maps corresponding to the same settings. As shown in Figure S2(b), increasing the GRAG value causes the query’s attention to progressively concentrate on the original cat contours. This focused attention corresponds to stronger preservation of source-image structure, leading to improved consistency between the edited output and the reference image.

These results collectively demonstrate that GRAG modulates editing strength by adjusting the concentration of crossattention, aligning the empirical observations with the theoretical analysis presented in the main paper.

[Figure 12]

- Figure S2. Relationship between attention entropy and GRAG. (a) Increasing the GRAG value on source image tokens leads to a monotonic decrease in attention entropy across layers. (b) The query’s attention over source image tokens becomes progressively more concentrated as the GRAG value increases.

##### B.3. GRAG of Source Image Tokens

To further understand the distinct roles of the bias and variation components in GRAG, we separately perform continuous adjustments on the text-embedding scaling factors λ (bias scaling) and δ (variation scaling).

kˆsi = λ · kbias + δ · ksi − kbias (S3)

The results are shown in Fig. S3. When only the bias component is modulated, the overall editing effect remains relatively stable, yet noticeable changes in image quality occur. In contrast, when only the variation component is adjusted, the semantic content of the edited image changes rapidly with increasing editing strength, while the overall image quality remains largely consistent. These observations support our interpretation that the bias component corresponds to the model’s intrinsic editing behavior, whereas the variation component encodes the actual content-specific editing signals.

[Figure 13]

- Figure S3. Effects of the bias and variation components in image-key embeddings.

##### B.4. GRAG of Editing Text Tokens

We conduct the same decomposition-based analysis on the text tokens to examine the roles of the bias and variation components, with results shown in Fig. S4. Similar to the observations for image tokens, the bias component governs the model’s inherent editing behavior, while the variation component drives the content-specific semantic changes. However, applying GRAG to text tokens reveals a noticeably stronger degree of semantic-level editing control, indicating that modulation of text embeddings provides a more direct and expressive pathway for manipulating editing intensity.

[Figure 14]

- Figure S4. Effects of the bias and variation components in text-key embeddings.

#### C. Limitation & Discussion

We observe that the effectiveness of GRAG is inherently limited when applied to training-free image editing methods. As illustrated in Fig. S5, training-based TI2I models possess a unified architecture in which the editing instruction and sourceimage information interact within the same MM-Attention layers, yielding better compatibility with GRAG. In contrast, training-free approaches are built upon T2I models that are not originally designed for editing; they rely on additional inversion procedures and attention injection to approximate the interaction between edited and source-image features. This structural mismatch constrains the impact of GRAG and lead to degraded image quality or ineffective edits (Fig. S6).

[Figure 15]

Figure S5. Structural differences between training-based and training-free editing methods.

[Figure 16]

- Figure S6. Failure cases of applying GRAG to training-free editing methods.

#### D. Additional Qualitative Results

[Figure 17]

- Figure S7. Additional qualitative results on existing image editing models.

[Figure 18]

###### Figure S8. Additional comparison results with existing editing-strength control methods.

#### E. Additional Feature Visualization

We provide additional Kontext and Qwen-Edit model feature distribution statistics corresponding to Figures 2, 4, and 5 in the main paper. Consistent with the experiments presented in the main text, we analyze different image editing samples (IDs) across various denoising steps and model layers to examine the correlation between feature distributions and these three factors. Figure S9, Figure S15 presents direct visualizations of the feature distributions, where the TokenNumber dimension is downsampled by a factor of 4 and the Dim dimension by a factor of 2. Figure S10, Figure S16 shows aggregating different tokens along the sequence dimension. Figures S11–S14, Figures S17–S20 illustrate the mean and variance of token-wise feature distributions across different attention heads, corresponding to the different embedding features.

##### E.1. Kontext Embedding Visualization

[Figure 19]

- Figure S9. Additional visualizations of text and image embedding features. Features within the same layer share similar distributions, indicating limited correlation with model inputs or denoising steps. Please zoom in to view finer details.

[Figure 20]

###### Figure S10. Additional visualizations of aggregating different tokens along the sequence dimension. Please zoom in to view finer details.

[Figure 21]

###### Figure S11. Additional visualizations of Query-edit embedding mean vector magnitudes and standard deviations across different attention

[Figure 22]

###### Figure S12. Additional visualizations of Key-text embedding mean vector magnitudes and standard deviations across different attention

[Figure 23]

###### Figure S13. Additional visualizations of Key-edit embedding mean vector magnitudes and standard deviations across different attention

[Figure 24]

###### Figure S14. Additional visualizations of Key-src embedding mean vector magnitudes and standard deviations across different attention

##### E.2. Qwen-Edit Embedding Visualization

[Figure 25]

- Figure S15. Additional visualizations of text and image embedding features. Features within the same layer share similar distributions, indicating limited correlation with model inputs or denoising steps. Please zoom in to view finer details.

[Figure 26]

###### Figure S16. Additional visualizations of aggregating different tokens along the sequence dimension. Please zoom in to view finer details.

[Figure 27]

###### Figure S17. Additional visualizations of Query-edit embedding mean vector magnitudes and standard deviations across different attention

[Figure 28]

###### Figure S18. Additional visualizations of Key-text embedding mean vector magnitudes and standard deviations across different attention

[Figure 29]

###### Figure S19. Additional visualizations of Key-edit embedding mean vector magnitudes and standard deviations across different attention

[Figure 30]

###### Figure S20. Additional visualizations of Key-src embedding mean vector magnitudes and standard deviations across different attention

