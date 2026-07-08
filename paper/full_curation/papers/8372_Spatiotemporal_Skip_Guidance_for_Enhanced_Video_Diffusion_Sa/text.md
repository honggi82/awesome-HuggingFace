# arXiv:2411.18664v1[cs.CV]27Nov2024

## Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling

Junha Hyung∗1 Kinam Kim∗1 Susung Hong2 Min-Jung Kim1 Jaegul Choo1 1KAIST AI 2University of Washington

{sharpeeee, kinamplify, emjay73, jchoo}@kaist.ac.kr, susung@cs.washington.edu

”A close-up shot of a butterfly landing on the nose of a woman, highlighting her smile and the details of the butterfly’s wings.”

”A close-up of a woman’s face with colored powder exploding around her, creating an abstract splash of vibrant hues.”

Figure 1. Visual comparison of video quality between CFG (top row) and our STG method (bottom row). Best viewed in Acrobat Reader; click on the images to watch the videos.

#### Abstract

Diffusion models have emerged as a powerful tool for generating high-quality images, videos, and 3D content. While sampling guidance techniques like CFG improve quality, they reduce diversity and motion. Autoguidance mitigates these issues but demands extra weak model training, limiting its practicality for large-scale models. In this

* indicates equal contribution.

work, we introduce Spatiotemporal Skip Guidance (STG), a simple training-free sampling guidance method for enhancing transformer-based video diffusion models. STG employs an implicit weak model via self-perturbation, avoiding the need for external models or additional training. By selectively skipping spatiotemporal layers, STG produces an aligned, degraded version of the original model to boost sample quality without compromising diversity or dynamic degree. Our contributions include: (1) introducing STG as

an efficient, high-performing guidance technique for video diffusion models, (2) eliminating the need for auxiliary models by simulating a weak model through layer skipping, and (3) ensuring quality-enhanced guidance without compromising sample diversity or dynamics unlike CFG. For additional results, visit https://junhahyung. github.io/STGuidance/.

#### 1. Introduction

Diffusion models [11, 28–30] are a successful class of generative models known for their flexibility in modeling complex data distributions, achieving impressive results in image, video, and 3D generation. By progressively denoising random noise, they enable robust generalization, making them a leading choice for realistic content generation and often surpassing GAN-based methods [5, 9, 18, 27]. Building on this success, video diffusion models [4, 12, 26, 31, 36] generate high-quality videos by using temporal or 3D attention layers to handle sequential frames.

Meanwhile, to enhance sample quality, sampling guidance techniques such as Classifier-Free Guidance (CFG) [10] and Autoguidance [19] have been introduced to guide the denoising process. These techniques employ weak models to predict poor trajectories, steering the main model away from them and pushing samples toward highquality regions on the data manifold. However, CFG often reduces diversity, leading to saturated or overly simplified results [6, 19].

Autoguidance [19] addresses this issue by using a weak model trained on the same task, conditioning, and data distribution as the main model. The main drawback of this approach, however, is the need to train an additional weak model, which is impractical for large-scale models. Alternative methods such as PAG [1] and SEG [13] use selfperturbation to implicitly mimic a weak model, avoiding the need for extra training. Yet, these methods are designed specifically for image generation diffusion models, applying self-perturbation to 2D spatial attention maps.

In this work, we propose Spatiotemporal Skip Guidance (STG), a simple and effective sampling guidance method for video diffusion models that significantly enhances the performance of any transformer-based video diffusion model without additional training. Specifically, we use an implicit weak model for guidance through selfperturbation, eliminating the need for explicit weak models and their associated training costs. This is especially crucial for video diffusion models, where training costs are high.

Our implicit weak model, a key component of our framework, is deliberately designed as a degraded but “aligned” version of the original video generation model. As demonstrated in Autoguidance [19], having the weak model share the same task, conditioning, and data distribution as the

High Quality

 1

 1

### 1

### 1



g

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Low Quality

###### CFG Ours

Figure 2. Comparison between CFG and STG, with the band conceptually representing the noisy data manifold. In STG, the weak model and the main model are aligned along the direction of increasing quality. In contrast, the two models in CFG differ not only in quality but also in aspects such as diversity and prompt alignment capabilities.

main model is essential for quality improvement, as we expect both models to exhibit similar, aligned errors in the same parts of samples. We conceptually illustrate this alignment in Fig. 2, where our weak model and main model are arranged in a direction of increasing quality. To achieve this, we apply spatiotemporal perturbation to both spatial and temporal attention layers—or, in the case of 3D attention, to the entire layer—by selectively skipping certain layers. This straightforward approach effectively nullifies specific residual or attention layers, generating a lower-quality version of the model that simulates an aligned weak model.

To ensure samples remain on the data manifold, even when using large guidance scales, we employ optional techniques such as the rescaling [21] and restart [34] methods. Rescaling the latent code constrains its variance, addressing the issue of larger variance causing saturation in the results [21]. Meanwhile, the restart sampling technique leverages the error contraction property of forward SDEs [34] to keep the sampling trajectory on the manifold. These techniques help prevent overshooting in the sampling trajectory, which could otherwise push samples off the manifold and result in saturated or distorted outputs.

Our key contributions are as follows:

- • We propose STG—a surprisingly simple sampling guidance framework that significantly boosts the performance of video diffusion models.
- • Our method introduces an implicit weak model by skipping spatiotemporal layers in video diffusion models, eliminating the need for additional training or external models.
- • Our method enhances sample quality during guidance without reducing diversity or limiting the dynamics of generated videos.

#### 2. Related Work

Guidance with trained weak model Classifier-Free Guidance (CFG) [10] improves conditional generation in Diffusion Models by using an implicit unconditional model as a weak model. However, differences in tasks between the unconditional and conditional models can reduce sample diversity [6, 19] and increase sampling trajectory curvature [6], leading to overshooting the data manifold and producing skewed or oversaturated images.

Autoguidance [19] mitigates these issues by employing a bad version of the main model as a weak model, trained with reduced capacity and compute to ensure alignment. This alignment allows the guidance algorithm to correct errors by analyzing prediction differences. While effective, it requires additional training, which is challenging for largescale video diffusion models.

Guidance with training-free weak model Another line of work avoids additional training by using selfperturbation of the main model to mimic a weak model. Self-Attention Guidance (SAG) [14] blurs high-attention regions, Perturbed Attention Guidance (PAG) [1] replaces attention maps with identity matrices, and Smoothed Energy Guidance (SEG) [13] applies Gaussian blur to the attention weights to smooth the energy landscape. These methods guide sampling toward high-quality outputs by leveraging differences in predictions from the weakened model. While effective, they are primarily designed for image diffusion models with 2D self-attention. We aim to extend this approach to video diffusion models, which require handling temporal dynamics with additional temporal or 3D spatiotemporal attention layers.

#### 3. Preliminaries

##### 3.1. Diffusion models

Diffusion models generate samples by progressively removing noise from noisy data, restoring the original data distribution. Song et al. [30] defined the process of adding noise to the data using a stochastic differential equation (SDE):

β(t) 2

dx = −

xdt + β(t)dw, (1)

where β(t) is a time-dependent noise schedule, and w represents the standard Wiener process. Corresponding reversetime SDE is:

β(t) 2

dx = −

x − β(t)∇x log pt(x) dt + β(t)dw¯ (2)

where dw¯ is the Wiener process running backward in time. Here, the score function ∇x log pt(x) is approximated by

a neural network sθ(x(t)) trained using denoising score matching [33]:

θ∗ = arg min

Et λ(t)Ex

Ex

t|x0 [ ∥sθ(xt) − ∇xt

0

θ

log pt(xt|x0)∥22 . (3)

##### 3.2. Classifier Guidance

Classifier Guidance (CG) [24] reformulates the reverse process of a diffusion model by incorporating an external classifier pϕ as

p˜θ(xt|y) ∝ pθ(xt)pϕ(y|xt)λ, (4)

where y is the desired class label and λ controls the guidance strength. The score function is then derived as:

∇xt

log p˜θ(xt|y) = ∇xt

log pθ(xt)

log pϕ(y|xt). (5)

+ λ∇xt

By substituting the score function in Eq. 2 with Eq. 5, sampling from the desired class condition y becomes possible.

##### 3.3. Classifier-Free Guidance

Classifier-Free Guidance (CFG) [10] uses Bayes’ rule to replace a classifier-guided score with a linear combination of conditional and unconditional score estimates:

ϵ˜λθ(xt) = ϵθ(xt) + λ(ϵθ(xt) − ϵθ(xt|ϕ)). (6)

CFG jointly trains the unconditional model ϵθ(xt|ϕ) and the conditional model ϵθ(xt|c) (= ϵθ(xt)) within a single model by setting the condition c to a null token ϕ. Using this guided denoising process, the model better captures the conditions and often generates high-fidelity outputs.

#### 4. Method 4.1. Optimal Weak Model Design

Our goal is to construct an aligned weak model that captures a distribution similar to the original model while generating slightly lower-quality samples. This ensures that the guidance gradient points toward improved quality, as shown in Fig. 2. Misaligned models, as seen in CFG [10], may lead to unintended outcomes like reduced diversity, while Autoguidance [19] requires additional weak model training, making it impractical for large-scale models.

To address these limitations, we design an implicit weak model by leveraging the original model itself, eliminating the need for external training. This training-free approach ensures computational efficiency and better alignment, as the weak model is derived directly from the original, preserving most network weights.

To achieve this, we apply perturbation methods directly to the main model’s forward pass, creating an implicit weak

Models Imaging Quality Aesthetic Quality Motion Smoothness Dynamic Degree Temporal Flickering

Mochi (CFG) 0.524 0.507 0.985 0.87 0.976 Mochi (STG) 0.628 0.554 0.988 0.86 0.978

Open-Sora (CFG) 0.561 0.493 0.982 0.902 0.975 Open-Sora (STG) 0.606 0.509 0.987 0.895 0.976

Table 1. Quantitative results for Mochi [31] and Open-Sora [36] on VBench [15] T2V benchmarks.

Models FVD (↓) IS Imaging Quality Aesthetic Quality Motion Smoothness Dynamic Degree

SVD (CFG) 151.3 38.0 0.687 0.637 0.966 0.562 SVD (STG) 128.7 38.5 0.694 0.639 0.968 0.694

Table 2. Quantitative results for SVD [4] on FVD, IS, and VBench [15] I2V benchmarks.

model. Specifically, we use spatiotemporal perturbation, a key component for aligning the weak model with video diffusion models, which will be elaborated upon in detail.

##### 4.2. Sampling from High Quality Samples

Similar to CG [24], we define our goal as conditioning the model on an imaginary label yg that represents high-quality samples, leading to the sampling distribution

p˜θ(xt|yg) ∝ pθ(xt)pϕ(yg|xt)w. (7)

Using w > 0 sharpens the distribution, promoting the generation of high-quality samples. From Eq. 7, the score is derived as

∇xt

log p˜θ(xt|yg)

log pϕ(yg|xt). (8)

= ∇xt

log pθ(xt) + w∇xt

Rather than using an external classifier pϕ, we propose using our model pθ as an implicit classifier. We design this classifier to be inversely proportional to the probability of an imaginary “bad” label yb, expressed as

1 pθ(yb|xt)

. (9)

pϕ(yg|xt) ∝

Using Bayes’ rule, the function can be expressed in terms of the marginal posterior as follows:

pθ(xt) pθ(yb)pθ(xt|yb)

1 pθ(yb|xt)

= ∇xt

∇xt

log

log

(log pθ(xt) − log pθ(xt|yb)), (10) leading to the score of the target distribution as:

= ∇xt

∇xt

log p˜θ(xt|yg) = ∇xt

log pθ(xt)

(log pθ(xt) − log pθ(xt|yb)). (11)

+ w∇xt

We can sample from p˜θ(xt|yg) by substituting the score function in Eq. 2 with Eq. 11, resulting in:

dx = −

β(t) 2

log pt(xt) (12)

x − β(t)(∇xt

+ w∇xt

(log pθ(xt) − log pθ(xt|yb)) dt + β(t)dw,¯

and solving the reverse SDE. Since the score function is approximated using the neural network, we can generate samples using

ϵ˜wθ (xt) = ϵθ(xt) + w(ϵθ(xt) − ϵbθ(xt)). (13)

An interesting approach here is to model ∇xt

log pθ(xt|yb) by perturbing the forward pass of

ϵθ(xt), denoted as ϵbθ(xt). Now the main focus is designing a perturbation that effectively yields the weak model

capable of estimating ϵbθ(xt), aligning closely with ϵθ(xt), as discussed in Sec. 4.1. Ideally, we want a distribution that

deviates minimally from ϵθ(xt) while producing slightly lower-quality samples.

##### 4.3. Spatiotemporal Skip Guidance (STG)

We introduce Spatiotemporal Skip Guidance (STG), a simple guidance method designed for video diffusion models that generates diverse, high-fidelity samples while addressing the limitations of existing methods. STG implicitly simulates an aligned weak model through spatiotemporal perturbation, capturing the spatiotemporal dynamics of video data.

An interesting discovery of this paper is that skipping layers within the network is an effective approach for constructing an aligned weak model. Modern neural network architectures for video diffusion models, such as ADM [24], DiT [25], and SiT [23], are partially or fully transformerbased and contain attention layers and residual blocks. These architectures are well-suited for layer skipping, as

A macro cinematography animation showing a butterfly emerging from its chrysalis, filmed with side-lit lighting ...

[Figure 7]

w=0 w=0.5 w=1.5 w=2.0

Figure 3. Selected frames from videos generated by Mochi [31] with increasing STG scales.

[Figure 8]

Figure 4. Comparison of CFG and STG across varying scales in terms of Imaging Quality and FVD.

they can still produce plausible outputs without significant degradation, even when a few layers are removed.

Layer skipping is advantageous for generating an aligned weak model because it retains most of the neural network’s weights and forward pass, resulting in similar predictions and distributions. This approach offers a clear advantage over methods that rely on external models, such as Autoguidance [19], or alternative objectives like CFG [10], where forward passes differ significantly.

Moreover, our method applies perturbations to both spatial and temporal layers (or spatiotemporal layers), unlike existing image-based perturbation methods [13, 14] limited to 2D spatial attention layers. This dual-layer perturbation is essential for aligning the weak model with the main video diffusion model. We now discuss various STG configurations that can be applied across different video diffusion model architectures.

Residual skip To skip an entire residual block, we modify Res(zl) to Res

′

(zl), Res(zl) = zl+1 = fl(zl) + zl, (14) Res

′

###### (zl) = zl+1 = zl, (15)

where zl and zl+1 represent the features at the lth and (l + 1)th layers, respectively, and fl denotes the lth neural net layer. The residual layers, which add small residuals fl(zl) to the original feature zl, ensure that the perturbed zl+1 does not deviate significantly from the original zl+1.

This reduces out-of-distribution issues in consecutive layers, generating perturbed but aligned samples.

Attention skip Self-attention computes a linear combination of value tensors as

SA(Q,K,V ) = Softmax

QKT √

d

V = AV, (16)

where Q ∈ R(h×w×f)×d, K ∈ R(h×w×f)×d, V ∈ R(h×w×f)×d are the query, key, and value matrices, respectively. Here, h, w, f, and d represent the height, width, frame number, and channel dimensions.

We can skip this layer partially by passing the value matrix directly to the next layer without computing its linear combination. This is equivalent to replacing the attention matrix A with an identity matrix I ∈ Rhwf×hwf, resulting in

′

(Q,K,V ) = IV. (17) This represents a 3D extension of PAG [1].

SA

Factorized attention While recent models like Movie Gen [26] and Mochi [31] utilize full 3D spatiotemporal attention layers, many models, such as SVD [4] and OpenSora [36], still use factorized attention layers for efficiency. These models employ sequential 2D spatial attention and 1D temporal attention to approximate 3D spatiotemporal attention. For factorized models, we apply skip perturbation of Eq. 17 to spatial and temporal layers separately and use their linear combination for the final guidance.

We denote the spatial and temporal perturbation labels as ysb and ytb, respectively. In practice, ysb and ytb are rarely independent, as they can influence each other. For instance, random spatial perturbations that create varied color adjustments across frames may disrupt temporal continuity. Similarly, random temporal perturbations can affect spatial consistency, leading to distortions in specific frames. However, for our skip perturbations, we can loosely assume independence, as skipping layers primarily reduces details in residual networks, which may not inherently affect temporal consistency.

Therefore, to simplify the derivation, we initially assume independence between spatial and temporal perturbations.

- (1) A romantic scene of a couple dancing under string lights in a backyard, with warm, golden tones highlighting their laughter.
- (2) An animation showing a floating castle drifting above the clouds, with birds flying around it and sunlight casting golden rays ...
- (3) A realistic documentary-style video of artisans crafting pottery, with the scene unfolding and transforming as hands shape clay ...
- (4) A ghost in a white bedsheet faces a mirror. The ghost’s reflection can be seen in the mirror. The ghost is in a dusty attic ... CFG

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

STG

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

CFG

STG

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

CFG

STG

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

CFG

STG

Figure 5. Qualitative comparison between CFG and STG on videos generated by Mochi [31].

We will revisit this assumption and propose an adjustment afterward. Under this independence assumption, the joint distribution can be expressed as

pθ(xt|yb) = pθ(xt|ysb)pθ(xt|ytb). (18)

Following the same approach as in Eq. 9 and using Eq. 11, we modify the score in Eq. 8 as follows:

∇xt

log˜pθ(xt|yg) = ∇xt

log pθ(xt)

+ w∇xt

- (log pθ(xt) − log pθ(xt|ys))

+ w∇xt

- (log pθ(xt) − log pθ(xt|yt)). (19)

By replacing the score with the estimated denoiser and utilizing separate scales for spatial and temporal terms, we obtain the following equation:

ϵ˜wθ (xt) = ϵθ(xt) + w1(ϵθ(xt) − ϵsθ(xt))

+ w2(ϵθ(xt) − ϵtθ(xt)) (20)

where ϵsθ(xt) and ϵtθ(xt) represent spatially and temporally perturbed models, respectively.

Next, we revisit our assumption of independence between spatial and temporal perturbations. While Eq. 20 works well in practice, we can derive an alternative STG formulation that uses orthogonalization to isolate the independent components of the spatial and temporal guidance. Inspired by a negative prompting technique [3], we can modify the Eq. 20 as follows:

ϵ˜wθ (xt) = ϵθ(xt) + w1∆s

⟨∆s,∆t⟩ ∥∆s∥2

∆s , (21)

+ w2 ∆t −

###### where ∆s = ϵθ(xt) − ϵsθ(xt) and ∆t = ϵθ(xt) − ϵtθ(xt).

Manifold Constrained Guidance Even with a wellaligned weak model, larger guidance scales w can drive samples off the data manifold, resulting in poor quality and oversaturation. To address error accumulation at high guidance scales, we explore optional techniques to keep samples constrained to the manifold. Rescaling the latent code [21] helps mitigate this issue by constraining its variance, as larger variance is known to cause saturation in the results. Additionally, Restart sampling [34] demonstrates that introducing stochasticity can correct off-manifold deviations. Building on this idea, we incorporate stochastic forward processes into our sampling guidance framework as an optional method. While this approach modestly improves final sample quality and reduces saturation, it introduces additional computational overhead. Further details are provided in Appendix A3.1.

#### 5. Experiments

- 5.1. Overview We employ three models for our experiments:

- • Mochi [31] is a text-to-video model built on AsymmDiT blocks, containing a total of 10 billion parameters and utilizing 3D self-attention in its spatiotemporal layers.
- • Open-Sora [36] is a text-to-video model built on STDiT blocks with 1.1 billion parameters, employing factorized spatial and temporal attention layers.
- • SVD [4] is an image-to-video model with 1.5 billion parameters that leverages factorized spatial and temporal attention within a UNet architecture.

We evaluate the proposed method using widely adopted datasets and metrics to ensure a comprehensive performance analysis.

- • UCF-101 The UCF-101 dataset comprises 13,320 videos organized into 101 action classes. Using this dataset, we assess the Fr´echet Video Distance (FVD) and Inception Score (IS) of the image-to-video model SVD [4]. Following DIGAN [35], we calculate FVD and IS on 2,048 and 10,000 samples, respectively. For conditioning, initial frames from UCF-101 videos are used as inputs to the SVD model, with each input frame serving as the starting frame of the generated videos.
- • VBench We use VBench [15] for automatic evaluation across various video metrics. SVD is evaluated using the image-to-video (I2V) framework on 355 samples from the VBench I2V dataset with 5 random seeds. Mochi and Open-Sora are evaluated with the text-to-video (T2V) framework: Open-Sora uses the standard VBench prompt list, while Mochi uses 100 randomly selected prompts due to computational limits.
- • EvalCrafter We perform human evaluations using 700 prompts from the EvalCrafter dataset [22].
- • LLM-Generated Prompts We use a set of 100 selected prompts generated by Claude 3.5 Sonnet [2] for our demo and qualitative comparisons.

For evaluation, CFG is used together with STG for our evaluation. We did not use Restart sampling or guidance orthogonalization; additional results with these techniques are in the Appendix sections A3.1 and A3.2.

- 5.2. Results

Qualitative Comparison Fig. 5 compares CFG and our method. Videos from the naive CFG model often show blurry objects with indistinct shapes, while STG produces clearer, more vivid frames with sharper image quality. STG also reduces temporal inconsistency and flickering, especially in dynamic videos with large motion where CFG frequently fails. Spatial guidance enhances object structure, and temporal guidance improves consistency. Please refer to the Appendix sections A3 and A4 for additional results.

Models FVD (↓) IS Imaging Quality Aesthetic Quality Motion Smoothness Dynamic Degree Temporal Flickering

Mochi (STG-R) - - 0.628 0.554 0.988 0.86 0.978 Mochi (STG-A) - - 0.555 0.541 0.987 0.86 0.976

Open-Sora (STG-R) - - 0.550 0.474 0.981 0.894 0.977 Open-Sora (STG-A) - - 0.606 0.509 0.987 0.895 0.976

SVD (STG-R) 155.9 39.3 0.687 0.637 0.965 0.641 SVD (STG-A) 128.7 38.5 0.694 0.639 0.968 0.694 -

Table 3. Comparison of STG-R (residual skip) and STG-A (attention skip) across Mochi [31], Open-Sora [36], and SVD [4]. STG-R shows stronger performance on Mochi, while STG-A yields better results on Open-Sora and SVD.

Models FVD (↓) IS Imaging Quality Aesthetic Quality Motion Smoothness Dynamic Degree

CFG 151.3 38.0 0.687 0.637 0.966 0.562 + Spatial 133.8 38.3 0.691 0.639 0.967 0.659 + Temporal 128.7 38.5 0.694 0.638 0.968 0.694

Table 4. Ablation study results on SVD [4] factorized attention, showing the impact of adding spatial and temporal guidance.

Quantitative Comparison We compare CFG and STG using the FVD-Imaging Quality (VBench) plot across different scales in Fig. 4. FVD measures video distribution, while Imaging Quality assesses frame clarity. Higher CFG scales improve Imaging Quality but reduce diversity, as reflected in higher FVD. STG avoids this trade-off, maintaining diversity at increased scales.

T2V and I2V VBench metrics in Tab.1 and Tab.2 show notable improvements in frame-level quality (Imaging and Aesthetic Quality). Temporal quality improves qualitatively, though metrics like Motion Smoothness and Temporal Flickering show marginal gains, as these scores are near saturation (∼ 0.9x).

For Dynamic Degree, we aim to keep it unchanged, as it may not correlate with video quality. This is achieved in the T2V metric, but in the I2V model, CFG increases the influence of the conditioned image, reducing motion. STG, while not directly affecting Dynamic Degree, mitigates CFG’s impact, leading to increased Dynamic Degree in I2V when used together.

Human Evaluation We provide human evaluation results on 700 prompts from the EvalCrafter dataset [22] in the Appendix A2.

##### 5.3. Ablation Study

Fig. 3 displays selected frames from videos generated by Mochi using various STG scales, where w = 0 corresponds to the CFG-only model without STG. Increasing the STG scale results in more vivid colors and finer details compared to the monotonous colors at w = 0. Notably, unlike with CFG, sampling diversity is preserved as the STG scale increases, as shown in Fig. 4.

We performed an ablation study on SVD to evaluate spatial and temporal guidance (Tab. 4). Spatial guidance notably reduced FVD and improved metrics, while tempo-

ral guidance further boosted performance, confirming their combined effectiveness for spatiotemporal generation.

We also test two STG variants—residual skip (STGR) and attention skip (STG-A)-on SVD, Open-Sora, and Mochi. As shown in Tab. 3, STG-R performs well with Mochi, while STG-A is more effective for the other models. This is likely due to Mochi’s higher parameter count and layer depth, enabling more extensive skipping without triggering out-of-distribution (OOD) issues in consecutive layers. Furthermore, Mochi’s spatiotemporal layers consist of a single 3D attention layer, so STG-R skips a residual layer with just one attention layer. In contrast, SVD and Open-Sora use factorized attention, meaning STG-R skips a residual layer with two attention layers, which may cause excessive perturbation and lead to OOD issues in subsequent layers. More results are available in the Appendix A3.

#### 6. Conclusion

We proposed Spatiotemporal Skip Guidance (STG), a simple, training-free method for video diffusion models. By simulating an aligned weak model via spatiotemporal skipping, STG offers strong guidance for high-fidelity video generation, achieving notable qualitative and quantitative improvements. We hope this work advances video diffusion models and inspires further research in the field.

Limitation and Ethical Considerations STG’s performance depends on scale and layer selection, with the optimal configuration varying across models, requiring users to set these through heuristic tuning. While video quality improvements are notable, they also raise ethical concerns about misuse, underscoring the importance of using this technology responsibly and constructively.

#### References

- [1] Donghoon Ahn, Hyoungwon Cho, Jaewon Min, Wooseok Jang, Jungwoo Kim, SeonHwa Kim, Hyun Hee Park, Kyong Hwan Jin, and Seungryong Kim. Self-rectifying diffusion sampling with perturbed-attention guidance. arXiv preprint arXiv:2403.17377, 2024. 2, 3, 5, 6
- [2] Anthropic. The claude 3 model family: Opus, sonnet, haiku. In -, 2024. 7
- [3] Mohammadreza Armandpour, Ali Sadeghian, Huangjie Zheng, Amir Sadeghian, and Mingyuan Zhou. Re-imagine the negative prompt algorithm: Transform 2d diffusion into 3d, alleviate janus problem and beyond. arXiv preprint arXiv:2304.04968, 2023. 7
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 4, 5, 7, 8, 1, 6, 10, 12, 13, 16, 17, 22
- [5] Andrew Brock. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096,

2018. 2

- [6] Hyungjin Chung, Jeongsol Kim, Geon Yeong Park, Hyelin Nam, and Jong Chul Ye. Cfg++: Manifold-constrained classifier free guidance for diffusion models. arXiv preprint arXiv:2406.08070, 2024. 2, 3
- [7] Zihan Ding, Xiao-Yang Liu, Miao Yin, and Linghe Kong. Tgan: Deep tensor generative adversarial nets for large image generation. arXiv preprint arXiv:1901.09953, 2019. 1
- [8] Songwei Ge, Aniruddha Mahapatra, Gaurav Parmar, JunYan Zhu, and Jia-Bin Huang. On the content bias in fr´echet video distance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7277– 7288, 2024. 1
- [9] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020. 2
- [10] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 2, 3, 5
- [11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [12] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022. 2
- [13] Susung Hong. Smoothed energy guidance: Guiding diffusion models with reduced energy curvature of attention. arXiv preprint arXiv:2408.00760, 2024. 2, 3, 5, 7, 17
- [14] Susung Hong, Gyuseong Lee, Wooseok Jang, and Seungryong Kim. Improving sample quality of diffusion models using self-attention guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7462– 7471, 2023. 3, 5
- [15] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin,

- Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 4, 7
- [16] Yangqing Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor Darrell. Caffe: Convolutional architecture for fast feature embedding. In Proceedings of the 22nd ACM international conference on Multimedia, pages 675–678, 2014. 1
- [17] Andrej Karpathy, George Toderici, Sanketh Shetty, Thomas Leung, Rahul Sukthankar, and Li Fei-Fei. Large-scale video classification with convolutional neural networks. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 1725–1732, 2014. 1
- [18] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8110–8119, 2020. 2
- [19] Tero Karras, Miika Aittala, Tuomas Kynk¨a¨anniemi, Jaakko Lehtinen, Timo Aila, and Samuli Laine. Guiding a diffusion model with a bad version of itself. arXiv preprint arXiv:2406.02507, 2024. 2, 3, 5
- [20] Black Forest Labs. Flux: Github repository. https: //github.com/black-forest-labs/flux, 2024. Accessed: 2024-11-22. 4
- [21] Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. Common diffusion noise schedules and sample steps are flawed. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 5404–5411, 2024. 2, 7, 5, 9
- [22] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22139–22149, 2024. 7, 8, 4
- [23] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740,

2024. 4

- [24] Soumik Mukhopadhyay, Matthew Gwilliam, Vatsal Agarwal, Namitha Padmanabhan, Archana Swaminathan, Srinidhi Hegde, Tianyi Zhou, and Abhinav Shrivastava. Diffusion models beat gans on image classification. arXiv preprint arXiv:2307.08702, 2023. 3, 4
- [25] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

- 2023. 4

[26] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720,

- 2024. 2, 5

- [27] Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems, 32, 2019. 2
- [28] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [29] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015.
- [30] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 2, 3
- [31] Genmo Team. Mochi 1. https://github.com/ genmoai/models, 2024. 2, 4, 5, 6, 7, 8, 1, 9, 14, 18, 19
- [32] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. Learning spatiotemporal features with 3d convolutional networks. In Proceedings of the IEEE international conference on computer vision, pages 4489–4497,

2015. 1

- [33] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661– 1674, 2011. 3
- [34] Yilun Xu, Mingyang Deng, Xiang Cheng, Yonglong Tian, Ziming Liu, and Tommi Jaakkola. Restart sampling for improving generative processes. Advances in Neural Information Processing Systems, 36:76806–76838, 2023. 2, 7, 5
- [35] Sihyun Yu, Jihoon Tack, Sangwoo Mo, Hyunsu Kim, Junho Kim, Jung-Woo Ha, and Jinwoo Shin. Generating videos with dynamics-aware implicit generative adversarial networks. arXiv preprint arXiv:2202.10571, 2022. 7
- [36] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. https://github.com/hpcaitech/Open-Sora, 2024. 2, 4, 5, 7, 8, 1, 11, 15, 20, 21

#### A1. Experimental Details

- A1.1. Sampling Algorithm

- Algorithm 1: Spatiotemporal Skip Guidance (STG)

Input: ϵθ, ϵs,tθ : Main model and spatiotemporally perturbed model respectively. w: Spatiotemporal guidance scale.

Σt: Variance at step t. Output: Generated video Vout.

- 1 xT ∼ N(0,I)
- 2 for t ← T,T − 1,...,1 do

- 3 ϵt ← ϵθ(xt) ϵs,tt ← ϵs,tθ (xt) ϵ˜t ← ϵt + w(ϵt − ϵs,tt ) xt−1 ∼ N √ 1α

t

xt − 1−α

√1−α¯ttϵ˜t ,Σt

- 4 return Vout

- Algorithm 2: Spatiotemporal Skip Guidance (STG) for factorized attention

Input: ϵθ, ϵsθ, ϵtθ: Main model, spatially perturbed, and temporally perturbed models respectively. w1, w2: Guidance scales. Σt: Variance at step t. Output: Generated video Vout.

- 1 xT ∼ N(0,I)
- 2 for t ← T,T − 1,...,1 do

- 3 ϵt ← ϵθ(xt) ϵst ← ϵsθ(xt) ϵtt ← ϵtθ(xt) ϵ˜t ← ϵt + w1(ϵt − ϵst) + w2(ϵt − ϵtt) xt−1 ∼ N √ 1α

t

xt − 1−α

√1−α¯ttϵ˜t ,Σt

- 4 return Vout

- A1.2. Computational Resources

For evaluation, we utilized an NVIDIA A100 40GB GPU for SVD [4], while Open-Sora [36] and Mochi [31] were evaluated using NVIDIA H100 or A100 80GB GPUs.

- A1.3. Implementation Details The default model scales for CFG are as follows: SVD [4] uses a scale of 3.0, Open-Sora [36] uses 7.0, and Mochi [31] uses
- 4.5. For STG, the configurations vary, with STG-A using a scale of 2.0 and STG-R using 1.0. STG is applied to the 8th layer of SVD, which has a total of 16 layers, and the 12th layer of Open-Sora, which has a total of 28 layers. For Mochi, which has 48 layers in total, STG is applied at the 35th layer.

- A1.4. Metrics

To evaluate model performance across different datasets, several methodologies were employed. FVD was assessed using the VideoMAE [8] model. IS was evaluated with the C3D model [16, 17, 32], following the setup of TGAN [7]. For VBench - Imaging Quality, the MUSIQ image quality predictor, trained on the SPAQ dataset, was used. VBench - Aesthetic Quality was measured using the LAION aesthetic predictor, applied to individual video frames. VBench - Dynamic Degree was evaluated with the RAFT flow estimator to quantify the degree of dynamics. VBench - Motion Smoothness was calculated as the mean absolute error (MAE) between dropped and reconstructed frames using a video frame interpolation model. Finally, VBench - Temporal Flickering was assessed by generating static frames and computing the mean absolute difference between consecutive frames.

##### A1.5. Prompts Used EvalCrafter prompt

- 1. 2 Dog and a whale, ocean adventure
- 2. Teddy bear and 3 real bear
- 3. Goldfish in glass
- 4. A small bird sits atop a blooming flower stem.
- 5. A fluffy teddy bear sits on a bed of soft pillows surrounded by children’s toys.
- 6. A peaceful cow grazing in a green field under the clear blue sky.
- 7. Unicorn sliding on a rainbow
- 8. Four godzillas
- 9. A fluffy grey and white cat is lazily stretched out on a sunny window sill, enjoying a nap after a long day of lounging.
- 10. A curious cat peers from the window, watching the world outside.
- 11. A horse
- 12. A pig
- 13. A squirrel
- 14. A bird
- 15. A zebra
- 16. Two elephants are playing on the beach and enjoying a delicious beef stroganoff meal.
- 17. Two fish eating spaghetti on a subway
- 18. A pod of dolphins gracefully swim and jump in the ocean.
- 19. A peaceful cow grazing in a green field under the clear blue sky.
- 20. A cute and chubby giant panda is enjoying a bamboo meal in a lush forest. The panda is relaxed and content as it eats, and occasionally stops to scratch its ear with its paw.
- 21. Dragon flying over the city at night
- 22. Pikachu snowboarding
- 23. A cat drinking beer
- 24. A dog wearing VR goggles on a boat
- 25. A giraffe eating an apple
- 26. Five camels walking in the desert
- 27. Mickey Mouse is dancing on white background
- 28. A happy pig rolling in the mud on a sunny day.
- 29. In an African savanna, a majestic lion is prancing behind a small timid rabbit. The rabbit tried to run away, but the lion catches up easily.
- 30. 3 sheep enjoying spaghetti together
- 31. A photo of a Corgi dog riding a bike in Times Square. It is wearing sunglasses and a beach hat.
- 32. A pod of dolphins gracefully swim and jump in the ocean.
- 33. In the lush forest, a tiger is wandering around with a vigilant gaze while the birds chirp and monkeys play.
- 34. The teddy bear and rabbit were snuggled up together. The teddy bear was hugging the rabbit, and the rabbit was nuzzled up to the teddy bear’s soft fur.
- 35. A slithering snake moves through the lush green grass.
- 36. A pair of bright green tree frogs cling to a branch in a vibrant tropical rainforest.
- 37. Four fluffy white Persian kittens snuggle together in a cozy basket by the fireplace.
- 38. Eight fluffy yellow ducklings waddle behind their mother, exploring the edge of a pond.
- 39. A family of four fluffy, blue penguins waddled along the icy shore.
- 40. Two white swans gracefully swam in the serene lake.
- 41. In a small forest, a colorful bird was flying around gracefully. Its shiny feathers reflected the sun rays, creating a beautiful sight.
- 42. A spider spins a web, weaving intricate patterns with its silk.
- 43. ...

###### Curated prompt for demos

- 1. Sloth with pink sunglasses lays on a donut float in a pool. The sloth is holding a tropical drink. The world is tropical. The sunlight casts a shadow.
- 2. A vibrant, top-down video of a kayak gliding through multicolored waters, showcasing shifting hues from blue to red to illustrate varying flow speeds or temperatures. The paddle interacts with the water, creating dynamic ripples and currents.
- 3. A slow-motion capture of a beautiful woman in a flowing dress spinning in a field of sunflowers, with petals swirling around her.
- 4. An exotic video of rabbits on the moon making rice cakes under a star-filled sky, with Earth visible in the background.
- 5. A handsome man walking confidently through a bustling city street at night, illuminated by neon lights and reflections in puddles.
- 6. A close-up shot of a butterfly landing on the nose of a woman, highlighting her smile and the intricate details of the butterfly’s wings.
- 7. A top-down video of a table filled with colorful dishes from different cuisines, with hands reaching in to serve food and clinking glasses.
- 8. A majestic bird’s-eye view of a couple holding hands while walking along the shore of a beach with sparkling turquoise waves.
- 9. A surreal scene of a forest where the leaves glow in neon colors, with a person walking down a path as fireflies dance around them.
- 10. A drone shot of a desert at sunset, where shadows stretch and shift, capturing a lone traveler moving gracefully through the sand dunes.
- 11. A close-up of a beautiful woman’s face with colored powder exploding around her, creating an abstract splash of vibrant hues.
- 12. A panoramic view of a tropical waterfall surrounded by lush greenery, with a rainbow forming in the mist.
- 13. A whimsical video of floating lanterns being released into the sky over a calm lake, with reflections on the water creating a mirror effect.
- 14. A time-lapse video of an artist painting a mural on a city wall, where each frame shows a burst of color and detail.
- 15. An overhead video of koi fish swimming in a pond with rippling water, with their scales reflecting shades of gold, orange, and white.
- 16. A slow-motion clip of a handsome person diving into a crystal-clear ocean, with water splashing and bubbles forming intricate patterns.
- 17. A fantastical scene of a meadow where flowers bloom and change colors in sync with the music, with a person dancing among them.
- 18. A top-down video of a hot air balloon festival, showing multicolored balloons lifting off and dotting the sky.
- 19. A beautiful woman sitting by a window as rain drizzles down, creating streaks and patterns on the glass.
- 20. A cinematic shot of a person walking through a field of lavender during golden hour, with the wind gently swaying the purple blossoms.
- 21. An exotic video of floating jellyfish in the ocean, their translucent bodies glowing with bioluminescence in shades of blue and purple.
- 22. A playful video of puppies running across a vibrant, flower-filled meadow, filmed in slow motion to capture their joyful expressions.
- 23. A captivating aerial view of a cityscape at sunrise, with skyscrapers casting long shadows and golden light reflecting on windows.
- 24. A stunning slow-motion shot of a bird taking flight over a reflective lake, with water droplets glistening as they scatter.
- 25. A romantic scene of a couple dancing under string lights in a backyard, with warm, golden tones highlighting their laughter.
- 26. ...

Prompt for figures in the main paper Fig2:

• A macro cinematography animation showing a butterfly emerging from its chrysalis, filmed with side-lit lighting to

accentuate the texture of its wings. Fig4:

- • A 50mm lens shot of a couple embracing under string lights as the camera slowly tracks them, capturing their shared laughter in a soft, cinematic glow.
- • An animation showing a floating castle drifting above the clouds, with birds flying around it and sunlight casting golden rays, evoking the feeling of wonder seen in classic animations.
- • A realistic documentary-style video of artisans crafting pottery, with the scene unfolding and transforming as hands shape clay under diffused lighting.
- • A ghost in a white bedsheet faces a mirror. The ghost’s reflection can be seen in the mirror. The ghost is in a dusty attic, filled with old beams, cloth-covered furniture. The attic is reflected in the mirror. The light is cool and natural. The ghost dances in front of the mirror.

#### A2. Human Evaluation

We conducted user studies following EvalCrafter [22] to evaluate subjective opinions across five key aspects: (1) Video Quality, reflecting the clarity of the generated video, with higher scores indicating reduced blur, noise, or visual artifacts; (2) Text-Video Alignment, assessing the correspondence between the input text prompt and the generated video, particularly focusing on the accuracy of generated motions; (3) Motion Quality, evaluating the correctness and realism of the motions depicted in the video; (4) Temporal Consistency, measuring the frame-to-frame coherence, distinct from Motion Quality as it requires users to assess the smoothness of movement; and (5) Subjective Likeness, akin to an aesthetic score, where higher values signify better alignment with human preferences.

For each metric, feedback was collected from seven users, who rated videos on a scale from 1 to 5, with higher scores representing better alignment. To ensure fairness, the video sequences were randomly shuffled before being presented to users.

We used 700 prompts from EvalCrafter for text-to-video (T2V) generation with Mochi [31]. Additionally, we employed FLUX.1 [dev] [20] to generate images from these prompts, which served as input to the image-to-video (I2V) model (SVD [4]). The results, shown in Fig. 6, demonstrate that incorporating STG leads to improved quality across all evaluated aspects.

[Figure 25]

- Figure 6. User study results for STG on SVD and Mochi, using 700 prompts from EvalCrafter [22]. For I2V generation of SVD, we employed FLUX.1 [dev] [20] to generate images from these prompts, which served as input to the model. The results demonstrate that incorporating STG leads to improved quality across all evaluated aspects.

#### A3. Ablation Study

##### A3.1. Manifold Constrained Guidance

As discussed in the main paper, sampling guidance techniques, including STG, utilize scale guidance, which can sometimes cause the sampling trajectory to deviate from the data manifold. This deviation is particularly noticeable when STG is applied with large scales or to videos that are already bright, often resulting in broken videos or over-saturation due to manifold overshooting. To mitigate these issues, we propose a set of optional techniques that can serve as effective remedies.

First, we leverage the error contraction property of stochastic processes [34] by incorporating stochastic forward processes into the sampling guidance framework. This technique, referred to as STG with Restart, is detailed in Algorithm 10. While this method moderately enhances the quality of the final samples and resolves issues such as broken videos (as illustrated in Fig. 7), it introduces additional computational overhead.

Additionally, increased variance in the latent code [21] has been observed in over-saturated results. Consequently, oversaturation can be effectively mitigated using a rescaling technique [21], which constrains the variance of the latent code. This method, referred to as STG with Rescaling, is detailed in Algorithm 6. As shown in Fig. 8, videos generated with larger variance (second row) often display saturated colors, which are successfully resolved by applying variance rescaling (third row). Unlike the Restart method, Rescaling introduces negligible computational overhead, making it the preferred approach for addressing over-saturation.

Algorithm 3: Spatiotemporal Skip Guidance with Restart

Input: ϵθ, ϵs,tθ : Main model and spatiotemporal perturbed model respectively. w: Spatiotemporal guidance scale.

Σt: Variance at step t. tmin,tmax: Restart interval. K: Number of Restart iterations. Output: Generated video Vout.

- 1 xT ∼ N(0,I)
- 2 for t ← T,T − 1,...,1 do

- 3 ϵt ← ϵθ(xt) ϵs,tt ← ϵs,tθ (xt) ϵ˜t ← ϵt + w(ϵt − ϵs,tt ) xt−1 ∼ N √ 1α

t

xt − 1−α

√1−α¯ttϵ˜t ,Σt

- 4 if t = tmin then

- 5 x0t

min ← xt−1

- 6 for k ← 0,...,K − 1 do

- 7 ϵrestart ∼ N(0,Σrestart) xkt+1

max ← xkt

min

+ ϵrestart

- 8 for t′ ← tmax,tmax − 1,...,tmin do

- 9 ϵt′ ← ϵθ(xkt′+1) ϵs,tt′ ← ϵs,tθ (xkt′+1) ϵ˜t′ ← ϵt′ + w(ϵt′ − ϵs,tt′ ) xkt′+1−1 ∼ N √α 1t′

xkt′+1 − 1−α

t′

√

1−α¯t′

ϵ˜t′ ,Σt′

- 10 return Vout

##### A3.2. STG with Orthogonalization

As discussed in the main paper, for SVD and Open-Sora, which utilize factorized spatial and temporal attention, it is possible to orthogonalize spatial and temporal guidance. The detailed algorithm for this approach is provided in Algorithm 8. However, we do not implement orthogonalization in practice, as it does not demonstrate any performance improvement, as shown in Table 5.

##### A3.3. Layer Ablation

STG can be applied to different layers, and we conduct an ablation study to evaluate the impact of skipping various layers for STG on Mochi [31]. The results are presented in Fig. 9. Mochi consists of 48 layers in total, and we experimented with layer skipping at layers 30, 32, and 35. Our findings show that skipping later layers has a more significant effect on quality

Algorithm 4: Spatiotemporal Skip Guidance (STG) with Rescaling

Input: ϵθ, ϵs,tθ : Main model and spatiotemporal perturbed model respectively. w: Spatiotemporal guidance scale. rescale: Rescaling factor. Σt: Variance at step t. Output: Generated video Vout.

- 1 xT ∼ N(0,I)
- 2 for t ← T,T − 1,...,1 do

- 3 ϵt ← ϵθ(xt) ϵs,tt ← ϵs,tθ (xt) ϵ˜t ← ϵt + w(ϵt − ϵs,tt )
- 4 stdϵ ← std(ϵt) stdϵ˜ ← std(˜ϵt) factor ← std

ϵ

stdϵ˜ factor ← rescale · factor + (1 − rescale) ϵ˜t ← ϵ˜t · factor

- 5 xt−1 ∼ N √ 1α

t

xt − 1−α

√1−α¯tt ϵ˜t ,Σt

- 6 return Vout

Models FVD (↓) IS Imaging Quality Aesthetic Quality Motion Smoothness Dynamic Degree

SVD (STG) 128.7 38.5 0.694 0.639 0.968 0.694 SVD (STG-ORTH) 130.4 38.4 0.691 0.637 0.967 0.692

Table 5. Ablation results of STG on SVD [4], evaluating the impact of orthogonalizing spatial and temporal guidance (STG-ORTH). Our findings show no performance gain from applying orthogonalization; therefore, we do not adopt it.

Algorithm 5: Spatiotemporal Skip Guidance (STG) with Orthogonalization

Input: ϵθ, ϵsθ, ϵtθ: Main model, spatially perturbed, and temporally perturbed models respectively. w1, w2: Guidance scales. Σt: Variance at step t. Output: Generated video Vout.

- 1 xT ∼ N(0,I)
- 2 for t ← T,T − 1,...,1 do

- 3 ϵt ← ϵθ(xt) ϵst ← ϵsθ(xt) ϵtt ← ϵtθ(xt)
- 4 ∆s ← ϵt − ϵst ∆t ← ϵt − ϵtt
- 5 ∆⊥t ← ∆t − ⟨∆

s,∆t⟩

∥∆s∥2 · ∆s

- 6 ϵ˜t ← ϵt + w1∆s + w2∆⊥t
- 7 xt−1 ∼ N √ 1α

t

xt − 1−α

√1−α¯tt ϵ˜t ,Σt

- 8 return Vout

improvements, as these layers are primarily responsible for refining texture details. Throughout all experiments in this paper, we consistently skip layer 35.

##### A3.4. Effect of Spatial and Temporal Guidance

For models with factorized attention layers, guidance can be applied separately to spatial and temporal layers. When using STG-A, it functions similarly to applying PAG [1] to the spatial attention layers, and we refer to this method as Spatial PAG (SPAG). When spatial guidance is applied alone, as shown for SVD in Fig. 10 and for Open-Sora in Fig. 11, the results struggle to maintain clarity during motion and exhibit poor temporal consistency. For instance, significant artifacts appear near the wings in the second row of Fig. 10, and around the legs of the chicken in Fig. 11.

We further investigate the individual contributions of spatial and temporal guidance. In Fig. 12, we compare results with and without Spatial Guidance (SPAG). The results show that while CFG fails to maintain clear object structures, resulting in blurry videos, SPAG significantly enhances object structure and improves clarity.

Similarly, in Fig. 13, we present results with and without Temporal Guidance (TPAG). The results reveal that CFG struggles to ensure frame-to-frame consistency, with the shape and color of the jelly varying noticeably across frames, leading to

a disjointed video. In contrast, TPAG effectively preserves the jelly’s appearance throughout the sequence, creating a more cohesive video and significantly improving Temporal Consistency.

##### A3.5. Attention Skip and Residual Skip

We compare the performance of attention skip (STG-A) and residual skip (STG-R) in Mochi [31] and Open-Sora [36]. The results for Mochi in Fig. 14 indicate that STG-R delivers greater qualitative improvements for Mochi. On the other hand, the results for Open-Sora in Fig. 15 and SVD in Fig. 16 demonstrate that STG-A delivers greater qualitative improvements for these models. Based on these findings, we use STG-A for Open-Sora and SVD, and use STG-R for Mochi in all experiments presented in the paper.

##### A3.6. Weak Model Visualization

We visualize the results of one-step prediction using different denoising methods (weak models, CFG, and STG) at timestep 30 in Fig. 18 and timestep 24 in Fig. 19, rows (a) to (e). Row (c) shows results denoised using the spatiotemporally perturbed model, ϵs,tθ (xt), which generally produces blurrier outcomes compared to row (b), where the unconditional weak model ϵθ(xt|ϕ) of CFG is applied. By moving away from the blurry weak model, STG achieves clear and well-defined structures with natural color tones. In contrast, CFG often produces unnatural color artifacts and broken structures. For example, the video predicted by CFG renders the girl’s arm on the left unnaturally red, the man’s arm on the right unnaturally dark, and the trees and leaves in the background blurry. By comparison, STG consistently generates videos with enhanced structure and natural, well-balanced color tones.

##### A3.7. Other Perturbation Methods

In addition to SPAG (spatial perturbation using PAG), we explore other perturbation techniques. One such approach is SEG [13], which applies Gaussian blurring to the attention map. A comparison of CFG, SEG, and STG is presented in Fig. 17. The results frequently show broken outputs in both CFG and SEG. In contrast, incorporating layer skipping alongside temporal perturbation, as in STG, consistently produces improved results.

#### A4. Qualitative Comparison

We provide additional qualitative comparisons using STG for SVD, Open-Sora, and Mochi. The results demonstrate that applying STG enhances the aesthetic appeal and fidelity of the videos, as shown in Fig. 20. In Open-Sora, we observe flickering artifacts frequently in the videos. By applying STG, these flickering artifacts are noticeably reduced, as illustrated in Fig. 21.

For I2V models such as SVD, as discussed in the main paper, STG not only enhances the structural quality of the generated videos but also increases their dynamic degree. This is because STG mitigates the effect of CFG, which tends to force generated videos to rigidly adhere to the conditioning image. This effect is visualized in Fig. 22.

We provide more video results in the zip file.

(Image condition is given for SVD.)

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

RestartSTGSTG

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Prompt: A group of people sitting on a green bench under an orange tree.

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

STGRestartSTG

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

- Figure 7. Quality Improvement with Restart STG. Top: Results for SVD [4]. Bottom: Results for Mochi [31]. The results demonstrate that while STG occasionally fails to generate videos correctly in certain cases, applying Restart resolves these issues, producing high-quality and accurate outputs.

Prompt: A young woman with glasses is jogging in the park wearing a pink headband.

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

CFGSTGRescaleSTG

- Figure 8. Comparison of CFG, STG, and Rescaled STG on Mochi [31]. When STG is applied using large scales or to bright videos, it often suffers from over-saturation caused by manifold deviation. One potential cause of this issue is the increased variance in the latent code, which is effectively mitigated by the rescaling technique proposed in [21].

Prompt: A close-up shot of a butterfly landing on the nose of a woman, highlighting her smile and the details of the butterfly’s wings.

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

STG-l:32STG-l:35CFGSTG-l:30

- Figure 9. Ablation study on the effect of skipping different layers for STG on Mochi [31]. Our results indicate that skipping later layers has a greater impact on quality improvements, as these layers primarily contribute to texture details. For all experiments, we consistently skip layer 35 (denoted as STG-l:35).

(Image condition is given for SVD.)

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

STGCFGSPAG

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

- Figure 10. Qualitative Comparison of CFG, SPAG, and STG on SVD [4]. PAG applied only to spatial layers is referred to as SPAG. The results show that while CFG and SPAG fail to preserve object clarity under motion, STG successfully achieves this.

Prompt: Brown chicken hunting for its food.

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

STGCFGSPAG

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

- Figure 11. Qualitative Comparison of CFG, SPAG, and STG on Open-Sora [36]. The results show CFG fails to generate the object’s head accurately, and SPAG struggles with the legs, whereas STG successfully generates all components correctly.

(Image condition is given for SVD.)

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

CFGSPAG

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

- Figure 12. Qualitative Comparison of Object Structure in SVD [4] with and without Spatial Guidance. Spatial Guidance is represented by SPAG, which applies PAG only to the spatial layer. The results indicate that while CFG struggles to maintain clear object structures, leading to blurry videos, SPAG effectively enhances object structure and improves clarity.

(Image condition is given for SVD.)

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

CFGTPAG

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

- Figure 13. Qualitative Comparison of Temporal Consistency in SVD [4] with and without Temporal Guidance (TPAG). The results reveal that CFG struggles to ensure frame-to-frame consistency, with the shape and color of the jelly varying noticeably across frames, leading to a disjointed video. In contrast, TPAG effectively preserves the jelly’s appearance throughout the sequence, creating a more cohesive video and significantly improving Temporal Consistency.

Prompt: A close-up shot of a butterfly landing on the nose of a woman, highlighting her smile and the details of the butterfly’s wings.

[Figure 96]

[Figure 97]

[Figure 98]

STG-ASTG-R

Prompt: Cinematic 8k scene of a couple dancing under warmly glowing string lights in an intimate backyard setting, ...

[Figure 99]

[Figure 100]

[Figure 101]

STG-ASTG-R

- Figure 14. Comparison of attention skip (STG-A) and residual skip (STG-R) in Mochi [31]. The results indicate that STG-R delivers greater qualitative improvements for Mochi.

STG-RSTG-A

Prompt: A close-up portrait of a woman set against a snowy backdrop. The woman is wearing a golden crown...

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Prompt: A moment of a woman in a white wedding dress, adorned with a pearl necklace and veil, standing...

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

STG-RSTG-A

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

- Figure 15. Comparison of attention skip (STG-A) and residual skip (STG-R) in Open-Sora [36]. The results indicate that STG-A delivers greater qualitative improvements for Open-Sora.

STG-ASTG-R

(Image condition is given for SVD.)

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

STG-ASTG-R

(Image condition is given for SVD.)

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

- Figure 16. Comparison of attention skip (STG-A) and residual skip (STG-R) in SVD [4]. The results indicate that STG-A delivers greater qualitative improvements for SVD.

[Figure 129]

(Image condition is given for SVD.)

[Figure 130]

[Figure 131]

STGSEGCFG

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

- Figure 17. Comparison of CFG, SEG [13], and STG in SVD [4]. The results show that CFG and SEG generate an unnatural nose for the person, whereas STG successfully generates all components naturally.

(a)

- (b)
- (c)
- (d)
- (e)

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

- (f)
- (g)

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

- Figure 18. Weak model visualization for Mochi [31]. Generated video using one-step prediction from timestep 30 (t = 30). (a) ϵθ(xt),

(b) ϵθ(xt|ϕ), (c) ϵs,tθ (xt), (d) CFG, (e) STG (f) Final video (CFG) (g) Final video (STG). The video predicted by CFG exhibits unnatural colors in certain areas and broken structures. In contrast, the video generated with STG demonstrates improved structural integrity and

more natural color tones.

- (a)

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

- (b)

[Figure 164]

- (c)

[Figure 165]

[Figure 166]

[Figure 167]

- (d)
- (e)

[Figure 168]

[Figure 169]

[Figure 170]

(g)

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

- (f)

[Figure 179]

- Figure 19. Weak model visualization for Mochi [31]. Video generated using one-step prediction from timestep 24 (t = 24). (a) ϵθ(xt), (b)

ϵθ(xt|ϕ), (c) ϵs,tθ (xt), (d) CFG, (e) STG (f) Final video (CFG) (g) Final video (STG). The result demonstrates that STG effectively guides the model to maintain structural integrity and realistic color distribution while avoiding the unintended artifacts present in CFG predictions.

Prompt: a neon-lit cityscape at night, featuring towering skyscrapers and crowded streets. The streets are bustling...

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

STGCFG

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

Prompt: A fluffy grey and white cat is lazily stretched out on a sunny window sill, enjoying a nap after a long day of lounging.

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

STGCFG

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

Prompt: Iron Man is walking towards the camera in the rain at night, with a lot of fog behind him. Science fiction movie, close-up.

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

STGCFG

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

- Figure 20. Qualitative comparison of video quality with and without STG applied on Open-Sora [36]. The results demonstrate that applying STG enhances the video’s aesthetic appeal and fidelity.

Prompt: A dog wearing vr goggles on a boat.

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

STGCFG

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

Prompt: A cyborg standing on top of a skyscraper, overseeing the city, back view, cyberpunk vibe, 2077, NYC, intricate details, 4K.

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

STGCFG

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

- Figure 21. Qualitative comparison of Temporal Flickering in Open-Sora [36] with and without STG. Without STG, temporal flickering is observed around the object, causing sudden bright flashes that disrupt the video experience. STG significantly reduces these artifacts, resulting in smoother and more cohesive motion.

(Image condition is given for SVD.)

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

STGCFG

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

(Image condition is given for SVD.)

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

STGCFG

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

- Figure 22. Qualitative Comparison of Dynamic Degree in SVD [4] with and without STG. The results show CFG results in limited object motion, whereas STG mitigates the restrictive effects of CFG, effectively enhancing the motion.

