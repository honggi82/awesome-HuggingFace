## HarmonyView: Harmonizing Consistency and Diversity in One-Image-to-3D

# arXiv:2312.15980v1[cs.CV]26Dec2023

Sangmin Woo1* Byeongjun Park1* Hyojun Go2 Jin-Young Kim2 Changick Kim1 1KAIST 2Twelve Labs

1{smwoo95, pbj3810, changick}@kaist.ac.kr 2{william, jeremy}@twelvelabs.io

https://byeongjun-park.github.io/HarmonyView/

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

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

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Input Generated diverse and multi-view coherent images Mesh

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

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

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Input HarmonyView (Ours) SyncDreamer [33]

Figure 1. HarmonyView for one-image-to-3D. HarmonyView generates realistic 3D content using just a single image. It excels at maintaining visual and geometric consistency across generated views while enhancing the diversity of novel views, even in complex scenes.

*Equal contribution

1

### Abstract

Recent progress in single-image 3D generation highlights the importance of multi-view coherency, leveraging 3D priors from large-scale diffusion models pretrained on Internetscale images. However, the aspect of novel-view diversity remains underexplored within the research landscape due to the ambiguity in converting a 2D image into 3D content, where numerous potential shapes can emerge. Here, we aim to address this research gap by simultaneously addressing both consistency and diversity. Yet, striking a balance between these two aspects poses a considerable challenge due to their inherent trade-offs. This work introduces HarmonyView, a simple yet effective diffusion sampling technique adept at decomposing two intricate aspects in single-image 3D generation: consistency and diversity. This approach paves the way for a more nuanced exploration of the two critical dimensions within the sampling process. Moreover, we propose a new evaluation metric based on CLIP image and text encoders to comprehensively assess the diversity of the generated views, which closely aligns with human evaluators’ judgments. In experiments, HarmonyView achieves a harmonious balance, demonstrating a win-win scenario in both consistency and diversity.

### 1. Introduction

Humans can effortlessly imagine the 3D form of an object from just a single camera view, drawing upon their prior knowledge of the 3D world. Yet, emulating this human capability in machines remains a longstanding challenge in the field of computer vision [2, 43, 57, 65, 68, 86]. The fundamental hurdle lies in the inherent ambiguity of deducing 3D structure from a single 2D image since a single image essentially collapses the three dimensions of the real world into a 2D representation. Consequently, countless 3D configurations of an object can be projected onto the same 2D image. This ambiguity has ignited the quest for innovative solutions for single-image 3D generation [1, 25, 27, 30, 31, 33, 35, 46, 51, 53–55, 61– 63, 73, 74, 81, 82, 87, 88].

One prevalent strategy is to generate multi-view images from a single 2D image [31, 32, 61, 72], and process them using techniques such as Neural Radiance Fields (NeRFs) [39] to create 3D representations. Regarding this, recent studies [32, 33, 61, 72, 81, 82] highlight the importance of maintaining multi-view coherency. This ensures that the generated 3D objects to be coherent across diverse viewpoints, empowering NeRF to produce accurate and realistic 3D reconstructions. To achieve this, researchers harness the capabilities of large-scale diffusion models [50], particularly those trained on a vast collection of 2D images. The abundance of 2D images provides a rich variety of views for the same ob-

ject, allowing the model to learn view-to-view relationships and acquire geometric priors about the 3D world. On top of this, some works [33, 61] introduce a refinement stage that fine-tunes the view alignment to accommodate variations in camera angles. This adjustment is a key factor in achieving the desired multi-view coherency, which directly impacts the realism of the resulting 3D representation. This progress has notably enhanced the utility of the generated 3D contents, making them more suitable for various applications [45, 75].

An equally significant but often overlooked aspect in single-image 3D generation is the novel-view diversity. The ill-posed nature of this task necessitates dealing with numerous potential 3D interpretations of a given 2D image. Recent works [32, 33, 61, 71] showcase the potential of creating diverse 3D contents by leveraging the capability of diffusion models in generating diverse 2D samples. However, balancing the pursuit of consistency and diversity remains a challenge due to their inherent trade-off: maintaining visual consistency between generated multi-view images and the input view image directly contributes to sample quality but comes at the cost of limiting diversity. Although current multi-view diffusion models [33, 61] attempt to optimize both aspects simultaneously, they fall short of fully unraveling their intricacies. This poses a crucial question: Can we navigate towards a harmonious balance between these two fundamental aspects in single-image 3D generation, thereby unlocking their full potential?

This work aims to address this question by introducing a simple yet effective diffusion sampling technique, termed HarmonyView. This technique effectively decomposes the intricacies in balancing consistency and diversity, enabling a more nuanced exploration of these two fundamental facets in single-image 3D generation. Notably, HarmonyView provides a means to exert explicit control over the sampling process, facilitating a more refined and controlled generation of 3D contents. This versatility of HarmonyView is illustrated in Fig. 1. Our method achieves a harmonious balance, demonstrating mutual benefits in both consistency and diversity. HarmonyView generates geometrically coherent 3D contents that faithfully represent the input image for visible parts while also capturing diverse yet plausible modes for occluded parts. Another challenge we face is the absence of standardized metrics for assessing the diversity of generated multi-views. To address this gap and provide a more comprehensive assessment of the consistency and diversity of 3D contents, we introduce a novel evaluation metric based on both the CLIP image and text encoders [20, 47].

In experiments, we quantitatively compare HarmonyView against state-of-the-art techniques, spanning two tasks: novelview synthesis and 3D reconstruction. In both tasks, HarmonyView consistently outperforms baseline methods across all metrics. Our qualitative results further highlight the efficacy of HarmonyView, showcasing faithful reconstructions

with remarkable visual quality, even in complex scenes. Moreover, we show that our proposed metric closely aligns with the assessments made by human evaluators. Lastly, HarmonyView can be seamlessly integrated with off-the-shelf text-to-image diffusion models (e.g., Stable Diffusion [50]), enabling it to perform text-to-image-to-3D generation.

### 2. Related Work

Lifting 2D pretrained models for 3D generation. Recent research endeavors [3, 29, 36, 55, 63, 67, 71, 74, 88] are centered on the idea of lifting 2D pre-trained models [47, 50] to create 3D models from textual prompts, without the need for explicit 3D data. The key insight lies in leveraging 3D priors acquired by diffusion models during pre-training on Internet-scale data. This enables them to dream up novel 3D shapes guided by text descriptions. DreamFusion [44] distills pre-trained Stable Diffusion [50] using Score Distillation Sampling (SDS) to extract a Neural Radiance Field (NeRF) [39] from a given text prompt. DreamFields [23] generates 3D models based on text prompts by optimizing the CLIP [47] distance between the CLIP text embedding and NeRF [39] renderings. However, accurately representing 3D details with word embeddings remains a challenge.

Similarly, some works [37, 46, 62, 80] extend the distillation process to train NeRF for the 2D-to-3D task. NeuralLift360 [80] utilizes a depth-aware NeRF to generate scenes guided by diffusion models and incorporates a distillation loss for CLIP-guided diffusion prior [47]. Magic123 [46] uses SDS loss to train a NeRF and then fine-tunes a mesh representation. Due to the reliance on SDS loss, these methods necessitate textual inversion [15] to find a suitable text description for the input image. Such a process needs perscene optimization, making it time-consuming and requiring tedious parameter tuning for satisfactory quality.

Another line of work [31, 32, 61, 72] uses 2D diffusion models to generate multi-view images then use them for 3D reconstruction with NeRF [39, 69]. 3DiM [72] views novel-view synthesis as an image-to-image translation problem and uses a pose-conditional diffusion model to predict novel views from an input view. Zero-1-to-3 [32] enables zero-shot 3D creation from arbitrary images by fine-tuning Stable Diffusion [50] with relative camera pose. Our work, falling into this category, is able to convert arbitrary 2D images to 3D without SDS loss [44]. It seamlessly integrates with other frameworks, such as text-to-2D [41, 48, 50] and neural reconstruction methods [39, 69], streamlining the text-to-image-to-3D process. Unlike prior distillation-based methods [37, 80] confined to a singular mode, our approach offers greater flexibility for generating diverse 3D contents.

Consistency and diversity in 3D generation. The primary challenge in single-image 3D content creation lies in maintaining multi-view coherency. Various approaches [32,

33, 72, 81, 82] attempt to tackle this challenge: Viewset Diffusion [61] utilizes a diffusion model trained on multi-view

- 2D data to output 2D viewsets and corresponding 3D models. SyncDreamer [33] introduces a 3D-aware feature attention that synchronizes intermediate states of noisy multi-views. Despite these efforts, achieving complete geometric coherence in generated views remains a challenge.

On the other hand, diversity across generated 3D samples is another critical aspect in single-image 3D generation. However, only a few works in the related literature specifically address this issue, often limited to domains such as face generation [11] or starting from text for 3D generation [71]. Recent studies [32, 33, 61, 82] showcase the potential of pre-trained diffusion models [50] in generating diverse multi-view images. However, there is still significant room for exploration in balancing consistency and diversity. In our work, we aim to unlock the potential of diffusion models, allowing for reasoning about diverse modes for novel views while being faithful to the input view for observable parts. We achieve this by breaking down the formulation of multi-view diffusion model into two fundamental aspects: visual consistency with input view and diversity of novel views. Additionally, we propose the CD score to address the absence of a standardized diversity measure in existing literature.

3. Method

Our goal is to create a high-quality 3D object from a single input image, denoted as y. To achieve this, we use the diffusion model [59] to generate a cohesive set of N views at pre-

defined viewpoints, denoted as x0(1:N) = {x(1)0 ,...,x(0N)}. These mutli-view images are then utilized in NeRF-like techniques [39, 69] for 3D reconstruction. The key to a realistic

- 3D object lies in the consistency across the generated views. If they exhibit coherent appearance and geometry, the resulting 3D object will appear more natural. Therefore, ensuring consistency is crucial for achieving our goal. Recent works [33, 53, 61] address multi-view generation by jointly optimizing the distribution of multiple views. Building upon them, we aim to enhance both consistency and diversity by decomposing their formulation during diffusion sampling.

#### 3.1. Diffusion Models

We address the challenge of generating a 3D representation from a single, partially observed image using diffusion models [58, 59]. These models inherently possess the capability to capture diverse modes [79], making them well-suited for the task. We adopt the setup of DDPM [22], which defines a forward diffusion process transforming an initial data sample x0 into a sequence of noisy samples x1,...,xT over T steps, approximating a Gaussian noise distribution. In practice, we perform the forward process by directly transitioning to a

noised version of a sample using the equation:

xt = √α¯tx0 + √1 − α¯tϵ, (1)

where ϵ ∼ N(0,I) is a Gaussian noise, α¯t is a noise schedule monotonically decreasing with timestep t (with α¯0 = 1), and xt is a noisy version of the input x0 at timestep t.

The reverse denoising process “undo” the forward steps to recover the original data from noisy observations. Typically, this process is learned by optimizing a noise prediction model ϵθ(xt,t) on a data distribution q(x0). DDPM [22] defines the following simple loss:

0∼q(x0),ϵ∼N(0,1),t∼U[1,T]∥ϵ − ϵθ(xt;t)∥22.

Lsimple = Ex

(2)

- 3.2. Multi-view Diffusion Models SyncDreamer [33] introduces a multi-view diffusion model

that captures the joint distribution of N novel views x0(1:N) given an input view y. This model extends the DDPM forward process (Eq. (1)) by adding random noises independently to each view at every time step:

x(tn) = √α¯tx(0n) + √1 − α¯tϵ(n). (3)

Here, n denotes the view index. A noise prediction model ϵθ predicts the noise of the n-th view ϵ(n), given the condition of an input view y, the view difference between the input view and the n-th target view ∆v(n), and noisy multi views

xt(1:N). Hereafter, we define the pair (y,∆v(n)) as the reference view condition r(n) to simplify notation. Similar to Eq. (2), the loss for the noise prediction model is defined as:

0 ,ϵ(1:N),t∥ϵ(n) − ϵθ(x(n);t,c(n))∥22, (4)

L = Ex(1:N)

where c(n) = (r(n),x(1:t N)) and ϵ(1:N) represents Gaussian noise of size N × H × W added to all N views.

#### 3.3. HarmonyView

Diffusion sampling guidance. Classifier-guided diffusion [12] uses a noise-robust classifier p(l|xt), which estimates the class label l given a noisy sample xt, to guide the diffusion process with gradients ∇xt

log p(l|xt). This classifier requires bespoke training to cope with high noise levels (where timestep t is large) and to provide meaningful signals all the way through the sampling process. Classifierfree guidance [21] uses a single conditional diffusion model pθ(x|l) with conditioning dropout, which intermittently replaces l (typically 10%) with a null token ϕ (representing the absence of conditioning information) for unconditional predictions. This models an implicit classifier directly from a diffusion model without the need for an extra classifier trained on noisy input. These conditional diffusion models [12, 21] dramatically improve sample quality by enhancing the conditioning signal but with a trade-off in diversity.

What’s wrong with multi-view diffusion sampling? From Eq. (4), we derive an unconditional diffusion model

p(x(n)) parameterized by a score estimator ϵθ(x(tn);t) and conditional diffusion model p(x(n)|c(n)) parameterized by

ϵθ(x(tn);t,c(tn)). These two models are learned via a single neural network following the classifier-free guidance [21]. During sampling, the multi-view diffusion model adjusts its prediction as follows (t is omitted for clarity):

ϵˆθ(x(tn);c(n)) = ϵθ(x(tn);c(n)) + s · (ϵθ(x(tn);c(n)) − ϵθ(x(tn))),

(5) where s represents a guidance scale.

The model output is extrapolated further in the direction of ϵθ(x(tn);ct(n)) and away from ϵθ(x(tn)). Remind that c(n) = (r(n),xt(1:N)). Thus, the scaling of s affects both the input view condition r(n) and the multi-view condition xt(1:N) simultaneously. As evidenced by Table 5, increasing s encourages multi-view coherency and diversity in the generated views. Yet, this comes with a trade-off: it simultaneously diminishes the visual consistency with the input view. While the inherent trade-off between these two dimensions is obvious in this context, managing competing objectives under a single guidance poses a considerable challenge. In essence, the model tends to generate diverse and geometrically coherent multi-view images, but differ in visual aspects (e.g., color, texture) from the input view, resulting in sub-optimal quality. Empirical observations, shown in Fig. 2 and Table 1, substantiate that this formulation manifests a conflict between the objectives of consistency and diversity.

Harmonizing consistency and diversity. To address the aforementioned challenge, we introduce a method termed “HarmonyView”. Our approach leverages two implicit clas-

sifiers. One classifier pi(r(n)|x(tn),xt(1:N)) guides the target view x(tn) and multi-views xt(1:N) to be more visually consistent with the input view r(n). Another classifier pi(x(1:t N)|xt(n),r(n)) contains uncertainty in both the target (xt(1:N)) and conditional (x(tn)) elements. This contributes to capturing diverse modes. Together, they synergistically guide the synchronization of noisy multi-views x(1:t N), facilitating geometric coherency among clean multi-views. Based on these, we redefine the score estimation as follows:

ϵ˜θ(x(tn);c(n)) = ϵθ(x(tn);c(n))

(6)

- − s1σt∇x(n)

t

log pi(r(n)|x(tn),x(1:t N))

- − s2σt∇x(n)

log pi(xt(1:N)|x(tn),r(n)),

t

where s1 and s2 are guidance scales and σt is a noise scheduling parameter. By properly balancing these terms, we can obtain multi-view coherent images that align well with the semantic content of the input image while being diverse across different samples.

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

Input No Guidance Baseline (Eq. (5)) Only s1 Only s2 Ours (Eq. (9))

Figure 2. Qualitative comparison of several instantiations for multi-view diffusion guidance on novel-view synthesis. Our decomposition of Eq. (5) yields two guidance parameters: s1 for input-target visual consistency and s2 for diversity in the novel views. With these parameters, our final formulation Eq. (9) enables the generation of a diverse set of multi-view coherent images that well reflect the input view.

According to Bayes’ rule, pi(r(n)|x(tn),x(1:t N)) ∝ p(x(tn)|c(n))/p(x(tn)|x(1:t N)) and pi(x(1:t N)|x(tn),r(n)) ∝ p(x(tn)|c(n))/p(x(tn)|r(n)). Hence, the diffusion scores of these two implicit classifiers can be derived as follows:

log pi(r(n)|x(tn),x(1:t N))

∇x(n)

t

1 σt

(ϵθ(x(tn);c(n)) − ϵθ(x(tn);x(1:t N))).

= −

(7)

log pi(x(1:t N)|x(tn),r(n))

∇x(n)

t

(8)

1 σt

(ϵθ(x(tn);c(n)) − ϵθ(x(tn);r(n)).

= −

Finally, these terms are plugged into Eq. (6) and yields:

ϵ˜θ(x(tn);c(n)) = ϵθ(x(tn);c(n))

- + s1 · (ϵθ(x(tn);c(n)) − ϵθ(x(tn);x(1:t N))
- + s2 · (ϵθ(x(tn);c(n)) − ϵθ(x(tn);r(n)).

(9)

This formulation effectively decomposes consistency and diversity, offering a nuanced approach that grants control over both dimensions. While simple, our decomposition achieves a win-win scenario, striking a harmonious balance in generating samples that are both consistent and diverse (see Fig. 2 and Table 1).

#### 3.4. Consistency-Diversity (CD) Score

We propose the CD score with two key principles: (1) Diversity of novel views: It is preferable that the generated images exhibit diverse and occasionally creative appearances that are not easily imaginable from the input image. (2) Semantic consistency: While pursuing diversity, it is crucial to maintain semantic consistency, i.e., the generated images should retain their semantic content consistently, regardless of variations in the camera viewpoint. To operationalize this evaluation, CD score utilizes CLIP [47] image (ΨI) and text encoders (ΨT), akin to CLIP score [20]. Diversity

Method s s1 s2 PSNR↑ SSIM↑ LPIPS↓ Eflow↓ CD↑ No Guidance 20.51 0.818 0.144 2.270 0.640 Baseline (Eq. (5)) ✓ 20.19 0.819 0.140 2.071 0.717

✓ 20.32 0.822 0.141 2.136 0.764 ✓ 21.03 0.828 0.128 2.146 0.668 ✓ ✓ 20.69 0.825 0.133 1.945 0.792

Ours (Eq. (9))

Table 1. Ablative study of multi-view diffusion guidance on novel-view synthesis. Metrics measure sample quality with PSNR, SSIM, LPIPS; multi-view coherency with Eflow; and diversity with CD score. Our final design strikes the best balance across the metrics. Here, we set s = 1, s1 = 2, s2 = 1.

(D) measures the average dissimilarity of generated views {x(1),...,x(N)} from a reference view y, reflecting how distinct the generated images are from the reference view, emphasizing creative variations. The diversity is computed by averaging the cosine similarity of each generated view with the reference view using CLIP image encoders.

N

1 N

D =

n=1

1 − cos(ΨI(y),ΨI(x(n))) . (10)

Semantic variance (SV ar) quantifies the variance in semantic changes across views. This measures how similar the generated images are to a given text prompt, “An image of {OBJECT}.” The semantic variance is calculated by averaging the cosine similarity between the CLIP text embedding of the prompt and the CLIP image embedding of each generated view, followed by measuring the variance of these values across views.

N

1 N

S¯ =

n=1

1 N

SV ar =

cos(ΨT(text),ΨI(x(n))),

N

(cos(ΨT(text),ΨI(x(n))) − S¯)2.

n=1

(11)

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

[Figure 154]

[Figure 155]

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

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

Input HarmonyView SyncDreamer [33] Zero123 [32]

- Figure 3. Novel-view synthesis comparison. HarmonyView generates plausible novel views while preserving coherence across views.

Method PSNR↑ SSIM↑ LPIPS↓ Eflow↓ CD↑ Realfusion [37] 15.26 0.722 0.283 - Zero123 [32] 18.98 0.795 0.166 3.820 0.628 SyncDreamer [33] 20.19 0.819 0.140 2.071 0.717 HarmonyView 20.69 0.825 0.133 1.945 0.792

Table 2. Novel-view synthesis on GSO [13] dataset. We report PSNR, SSIM, LPIPS, Eflow, and CD score.

The CD score is then computed as the ratio of diversity to semantic variances across views:

CD Score = D/SV ar. (12)

We note that the CD score is reference-free, i.e., it does not require any ground truth images to measure the score.

### 4. Experiments

Due to space constraints, we provide detailed information regarding implementation details and baselines in Appendix. Dataset. Following [31–33], we used the Google Scanned Object (GSO) [13] dataset, adopting the same data split as in [33], for our evaluation. In addition, we utilized Internetcollected images, including those curated by [33], to assess the generation ability for complex objects or scenes.

Tasks and metrics. For the novel-view synthesis task, we used three standard metrics – PSNR, SSIM [70], LPIPS [85] – to measure sample quality compared to GT images. We measured diversity using the CD score. As a multi-view coherency metric, we propose Eflow, which measures the ℓ1 distance between optical flow estimates from RAFT [64] for both GT and generated images. For the single-view 3D reconstruction task, we used Chamfer distance to evaluate point-by-point shape similarity and volumetric IoU to quantify the overlap between reconstructed and GT shapes.

User Likert Score (1-5)↑ Quality Consistency Diversity

Methods CD↑

Zero123 [32] 0.752 3.208 3.167 2.854 SyncDreamer [33] 0.722 3.417 3.208 2.708 HarmonyView 0.804 3.958 3.479 3.813

Table 3. Novel-view synthesis on in-the-wild images. We report the CD score and 5-scale user Likert score, assessing quality, consistency, and diversity. Notably, the CD score shows strong alignment with human judgments. The test images are collected by [33].

#### 4.1. Comparative Results

Novel-view synthesis. Table 2 shows the quantitative results for novel-view synthesis on the GSO [13] dataset. Here, HarmonyView outperforms state-of-the-art methods across all metrics. We confirm that HarmonyView generates images of superior quality, as indicated by PSNR, SSIM and LPIPS. It particularly excels in achieving multi-view coherency (indicated by Eflow) and generating diverse views that are faithful to the semantics of the input view (indicated by CD score). In Fig. 3, we present the qualitative results. Zero123 [32] produces multi-view incoherent images or implausible images, e.g., eyes on the back. SyncDreamer [33] generates images that lack visual similarity to the input view or contain deficiencies, e.g., flatness or hole on the back. In contrast, HarmonyView generates diverse yet plausible multi-view images while maintaining geometric coherence across views. In Table 3, we examine novel-view synthesis methods on in-the-wild images curated by [33]. For evaluation, we use CD score and user Likert ratings (1 to 5) along three criteria: quality, consistency, and diversity. While SyncDreamer [33] excels in quality and consistency scores when compared to Zero123 [32], Zero123 performs better in diversity and CD score. Notably, HarmonyView stands out with

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

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

Input HarmonyView SyncDreamer [33] Zero123 [32] One-2-3-45 [31] Point-E [42] Shap-E [26]

- Figure 4. 3D reconstruction comparison. HarmonyView stands out in creating high-quality 3D meshes where other often fails. HarmonyView, SyncDreamer [33], and Zero123 [32] use the vanilla NeuS [69] for 3D reconstruction.

Method Chamfer Dist.↓ Volume IoU↑

Realfusion [37] 0.0819 0.2741 Magic123 [46] 0.0516 0.4528 One-2-3-45 [31] 0.0629 0.4086 Point-E [42] 0.0426 0.2875 Shap-E [26] 0.0436 0.3584 Zero123 [32] 0.0339 0.5035 SyncDreamer [33] 0.0261 0.5421 HarmonyView 0.0187 0.6401

Table 4. 3D reconstruction on GSO [13] dataset. HarmonyView demonstrates substantial improvements over competitive baselines.

the highest CD score and superior user ratings. This suggests that HarmonyView effectively produces visually pleasing, realistic, and diverse images while being coherent across multiple views. The correlation between the CD score and the diversity score underscores the efficacy of the CD score in capturing the diversity of generated images.

3D reconstruction. In Table 4, we quantitatively compare our approach against various other 3D generation methods [26, 31–33, 37, 42, 46]. Both our method and SDS-free methods [32, 33] utilize NeuS [69], a neural reconstruction method for converting multi-view images into 3D shapes. To achieve faithful reconstruction of 3D mesh that aligns well with ground truth, the generated multi-view images should be geometrically coherent. Notably, HarmonyView achieves the best results by a significant margin in both Chamfer distance and volumetric IoU metrics, demonstrating the proficiency of HarmonyView in producing multi-view coherent images. We also present a qualitative comparison in Fig. 4. The results showcase the remarkable quality of HarmonyView. While competing methods often struggle with incomplete reconstructions (e.g., Point-E, Shap-E), fall short in capturing small details (e.g., Zero123), and show discontinuities (e.g.,

Method s s1 s2 PSNR↑ SSIM↑ LPIPS↓ Eflow↓ CD↑

- 0.5 - - 20.55 0.822 0.137 2.074 0.685
- 1.0 - - 20.19 0.819 0.140 2.071 0.717 1.5 - - 19.76 0.814 0.146 2.011 0.711

Baseline (Eq. (5))

- - 0.0 1.0 20.32 0.822 0.141 2.136 0.764
- - 1.0 1.0 20.55 0.824 0.135 2.009 0.772
- - 3.0 1.0 20.73 0.825 0.132 1.950 0.737
- - 2.0 0.0 21.03 0.828 0.128 2.146 0.668
- - 2.0 0.6 20.90 0.827 0.130 1.996 0.770
- - 2.0 0.8 20.80 0.826 0.131 2.009 0.774
- - 2.0 1.2 20.56 0.824 0.135 1.996 0.760
- - 2.0 1.0 20.69 0.825 0.133 1.945 0.792

Ours (Eq. (9))

Table 5. Guidance scale study on novel-view synthesis. We compare two instantiations of multi-view diffusion guidance: Eq. (5) and Eq. (9). Our approach consistently outperforms the baseline. Increasing s1 tends to enhance PSNR, SSIM, and LPIPS, while higher s2 tends to improve CD score. Notably, the combined effect of s1 and s2 synergistically improves Eflow.

SyncDreamer) or artifacts (e.g., One-2-3-45), our method produces high-quality 3D meshes characterized by accurate geometry and a realistic appearance.

#### 4.2. Analysis

Scale study. In Table 5, we investigate two instantiations of multi-view diffusion guidance with different scale configurations: baseline (Eq. (5)) and our approach (Eq. (9)). As s increases from 0.5 to 1.5 in the baseline method, Eflow (indicating multi-view coherency) and CD score (indicating diversity) show an increasing trend. Simultaneously, PSNR, SSIM, and LPIPS (indicating visual consistency) show a declining trend. This implies a trade-off between visual consistency and diversity. In contrast, our method involves parameters s1 and s2. We observe that increasing s1 provides stronger guidance in aligning multi-view images with the input view, leading to direct improvements in PSNR, SSIM,

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

Input HarmonyView SyncDreamer [33]

[Figure 219]

- Figure 5. 3D reconstruction for complex object or scene. HarmonyView successfully reconstructs the details, while SyncDreamer fails.

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

An astronaut riding a horse

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

A cute panda riding a car

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

A boxer toy made of wood

Input text Text to image Generated images Mesh

Figure 6. Text-to-Image-to-3D. HarmonyView, when combined with text-to-image frameworks [41, 48, 50], enables text-to-3D.

and LPIPS. Keeping s1 fixed at 2.0, elevating s2 tends to yield improved CD score, indicating an enhanced diversity in the generated images. However, given the inherent conflict between consistency and diversity, an increase in s2 introduces a trade-off. We note that our approach consistently outperforms the baseline across various configurations, striking a nuanced balance between consistency and diversity. Essentially, our decomposition provides more explicit control over those two dimensions, enabling a better balance. Additionally, the synergy between s1 and s2 notably enhances Eflow, leading to improved 3D alignment across multiple views.

Generalization to complex objects or scenes. Even in challenging scenarios, either with a highly detailed single object or multiple objects within a single scene, HarmonyView excels at capturing intricate details that SyncDreamer [33] might miss. The results are shown in Fig. 5. Our model well generates multi-view coherent images even in such scenarios, enabling the smooth reconstruction of natural-looking meshes without any discontinuities.

Compatibility with text-to-image models. HarmonyView seamlessly integrates with off-the-shelf text-to-image models [48, 50]. These models convert textual descriptions into

2D images, which our model further transforms into highquality multi-view images and 3D meshes. Visual examples are shown in Fig. 6. Notably, our model excels in capturing the essence or mood of the given 2D image, even managing to create plausible details for occluded parts. This demonstrates strong generalization capability, allowing it to perform well even with unstructured real-world images.

Runtime. HarmonyView generates 64 images (i.e., 4 instances × 16 views) in only one minute, with 50 DDIM [59] sampling steps on an 80GB A100 GPU. Despite the additional forward pass through the diffusion model, HarmonyView takes less runtime than SyncDreamer [33], which requires about 2.7 minutes with 200 DDIM sampling steps.

Additional results & analysis. Please see Appendix for more qualitative examples and analysis on the CD score, etc.

### 5. Conclusion

In this study, we have introduced HarmonyView, a simple yet effective technique that adeptly balances two fundamental aspects in a single-image 3D generation: consistency and diversity. By providing explicit control over the diffusion

sampling process, HarmonyView achieves a harmonious equilibrium, facilitating the generation of diverse yet plausible novel views while enhancing consistency. Our proposed evaluation metric CD score effectively measures the diversity of generated multi-views, closely aligning with human evaluators’ judgments. Experiments show the superiority of HarmonyView over state-of-the-art methods in both novel-view synthesis and 3D reconstruction tasks. The visual fidelity and faithful reconstructions achieved by HarmonyView highlight its efficacy and potential for various applications.

### References

- [1] Eric R Chan, Koki Nagano, Matthew A Chan, Alexander W Bergman, Jeong Joon Park, Axel Levy, Miika Aittala, Shalini De Mello, Tero Karras, and Gordon Wetzstein. Generative novel view synthesis with 3d-aware diffusion models. arXiv preprint arXiv:2304.02602, 2023. 2
- [2] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An informationrich 3d model repository. arXiv preprint arXiv:1512.03012,

2015. 2, 13

- [3] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. arXiv preprint arXiv:2303.13873, 2023. 3
- [4] Zhiqin Chen. A review of deep learning-powered mesh reconstruction methods. arXiv preprint arXiv:2303.02879, 2023. 13
- [5] Zhiqin Chen and Hao Zhang. Learning implicit fields for generative shape modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5939–5948, 2019. 13
- [6] Zhiqin Chen and Hao Zhang. Neural marching cubes. ACM Transactions on Graphics (TOG), 40(6):1–15, 2021. 13
- [7] Zhiqin Chen, Andrea Tagliasacchi, and Hao Zhang. Bsp-net: Generating compact meshes via binary space partitioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 45–54, 2020. 13
- [8] Yen-Chi Cheng, Hsin-Ying Lee, Sergey Tulyakov, Alexander G Schwing, and Liang-Yan Gui. Sdfusion: Multimodal 3d shape completion, reconstruction, and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4456–4465, 2023. 13
- [9] Christopher B Choy, Danfei Xu, JunYoung Gwak, Kevin Chen, and Silvio Savarese. 3d-r2n2: A unified approach for single and multi-view 3d object reconstruction. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part VIII 14, pages 628–644. Springer, 2016. 13
- [10] Boyang Deng, Kyle Genova, Soroosh Yazdani, Sofien Bouaziz, Geoffrey Hinton, and Andrea Tagliasacchi. Cvxnet: Learnable convex decomposition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 31–44, 2020. 13

- [11] Rahul Dey and Vishnu Naresh Boddeti. Generating diverse 3d reconstructions from a single occluded face image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1547–1557, 2022. 3
- [12] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 4
- [13] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A highquality dataset of 3d scanned household items. In 2022 International Conference on Robotics and Automation (ICRA), pages 2553–2560. IEEE, 2022. 6, 7, 16
- [14] Haoqiang Fan, Hao Su, and Leonidas J Guibas. A point set generation network for 3d object reconstruction from a single image. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 605–613, 2017. 13
- [15] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618,

2022. 3

- [16] Kyle Genova, Forrester Cole, Avneesh Sud, Aaron Sarna, and Thomas Funkhouser. Local deep implicit functions for 3d shape. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4857–4866,

2020. 13

- [17] Georgia Gkioxari, Jitendra Malik, and Justin Johnson. Mesh r-cnn. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9785–9795, 2019. 13
- [18] Thibault Groueix, Matthew Fisher, Vladimir G Kim, Bryan C Russell, and Mathieu Aubry. A papier-mˆach´e approach to learning 3d surface generation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 216–224, 2018. 13
- [19] Anchit Gupta, Wenhan Xiong, Yixin Nie, Ian Jones, and Barlas O˘guz. 3dgen: Triplane latent diffusion for textured mesh generation. arXiv preprint arXiv:2303.05371, 2023. 13
- [20] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021. 2, 5
- [21] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 4
- [22] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3, 4
- [23] Ajay Jain, Ben Mildenhall, Jonathan T Barron, Pieter Abbeel, and Ben Poole. Zero-shot text-guided object generation with dream fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 867–876,

2022. 3

- [24] Varun Jampani, Huiwen Chang, Kyle Sargent, Abhishek Kar, Richard Tucker, Michael Krainin, Dominik Kaeser, William T Freeman, David Salesin, Brian Curless, et al. Slide: Single image 3d photography with soft layering and depth-aware inpainting. In Proceedings of the IEEE/CVF International

- Conference on Computer Vision, pages 12518–12527, 2021. 13
- [25] Yifan Jiang, Hao Tang, Jen-Hao Rick Chang, Liangchen Song, Zhangyang Wang, and Liangliang Cao. Efficient-3dim: Learning a generalizable single-image novel-view synthesizer in one day. arXiv preprint arXiv:2310.03015, 2023. 2
- [26] Heewoo Jun and Alex Nichol. Shap-e: Generating conditional 3d implicit functions. arXiv preprint arXiv:2305.02463, 2023. 7, 13, 17
- [27] Yash Kant, Aliaksandr Siarohin, Michael Vasilkovsky, Riza Alp Guler, Jian Ren, Sergey Tulyakov, and Igor Gilitschenski. invs: Repurposing diffusion inpainters for novel view synthesis. arXiv preprint arXiv:2310.16167, 2023. 2
- [28] Yiyi Liao, Simon Donne, and Andreas Geiger. Deep marching cubes: Learning explicit surface representations. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2916–2925, 2018. 13
- [29] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, MingYu Liu, and Tsung-Yi Lin. Magic3d: High-resolution textto-3d content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 300–309, 2023. 3
- [30] Yukang Lin, Haonan Han, Chaoqun Gong, Zunnan Xu, Yachao Zhang, and Xiu Li. Consistent123: One image to highly consistent 3d asset using case-aware diffusion priors. arXiv preprint arXiv:2309.17261, 2023. 2
- [31] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Zexiang Xu, Hao Su, et al. One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. arXiv preprint arXiv:2306.16928, 2023. 2, 3, 6, 7, 13, 17
- [32] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9298– 9309, 2023. 2, 3, 6, 7, 13, 14, 15, 16, 17
- [33] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453, 2023. 1, 2, 3, 4, 6, 7, 8, 13, 14, 15, 16, 17
- [34] Zhen Liu, Yao Feng, Michael J Black, Derek Nowrouzezahrai, Liam Paull, and Weiyang Liu. Meshdiffusion: Scorebased generative 3d mesh modeling. arXiv preprint arXiv:2303.08133, 2023. 13
- [35] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. arXiv preprint arXiv:2310.15008, 2023. 2
- [36] Jonathan Lorraine, Kevin Xie, Xiaohui Zeng, Chen-Hsuan Lin, Towaki Takikawa, Nicholas Sharp, Tsung-Yi Lin, MingYu Liu, Sanja Fidler, and James Lucas. Att3d: Amortized text-to-3d object synthesis. arXiv preprint arXiv:2306.07349,

2023. 3

- [37] Luke Melas-Kyriazi, Iro Laina, Christian Rupprecht, and Andrea Vedaldi. Realfusion: 360deg reconstruction of any object

- from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8446–8455, 2023. 3, 6, 7, 13
- [38] Luke Melas-Kyriazi, Christian Rupprecht, and Andrea Vedaldi. Pc2: Projection-conditioned point cloud diffusion for single-image 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12923–12932, 2023. 13
- [39] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 2, 3
- [40] Fangzhou Mu, Jian Wang, Yicheng Wu, and Yin Li. 3d photo stylization: Learning to generate stylized novel views from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16273– 16282, 2022. 13
- [41] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 3, 8
- [42] Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751, 2022. 7, 13, 17
- [43] Byeongjun Park, Hyojun Go, and Changick Kim. Bridging implicit and explicit geometric transformations for singleimage view synthesis. arXiv preprint arXiv:2209.07105, 2022. 2
- [44] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 3, 13
- [45] Simon Prince, Adrian David Cheok, Farzam Farbiz, Todd Williamson, Nikolas Johnson, Mark Billinghurst, and Hirokazu Kato. 3d live: Real time captured content for mixed reality. In Proceedings. International Symposium on Mixed and Augmented Reality, pages 7–317. IEEE, 2002. 2
- [46] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, et al. Magic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors. arXiv preprint arXiv:2306.17843, 2023. 2, 3, 7, 13
- [47] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2, 3, 5
- [48] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. PMLR, 2021. 3, 8
- [49] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular

- depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 2020. 13
- [50] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 2, 3, 8, 13
- [51] Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann, Hong-Xing Yu, Yunzhi Zhang, Eric Ryan Chan, Dmitry Lagun, Li Fei-Fei, Deqing Sun, et al. Zeronvs: Zero-shot 360degree view synthesis from a single real image. arXiv preprint arXiv:2310.17994, 2023. 2
- [52] Gopal Sharma, Difan Liu, Subhransu Maji, Evangelos Kalogerakis, Siddhartha Chaudhuri, and Radom´ır Mˇech. Parsenet: A parametric surface fitting network for 3d point clouds. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VII 16, pages 261–276. Springer, 2020. 13
- [53] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110, 2023. 2, 3
- [54] Yukai Shi, Jianan Wang, He Cao, Boshi Tang, Xianbiao Qi, Tianyu Yang, Yukun Huang, Shilong Liu, Lei Zhang, and Heung-Yeung Shum. Toss: High-quality text-guided novel view synthesis from a single image. arXiv preprint arXiv:2310.10644, 2023.
- [55] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023. 2, 3
- [56] Meng-Li Shih, Shih-Yang Su, Johannes Kopf, and Jia-Bin Huang. 3d photography using context-aware layered depth inpainting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8028–8038,

2020. 13

- [57] Ayan Sinha, Asim Unmesh, Qixing Huang, and Karthik Ramani. Surfnet: Generating 3d shape surfaces using deep residual networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6040–6049,

2017. 2, 13

- [58] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 3
- [59] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502,

2020. 3, 8

- [60] Xingyuan Sun, Jiajun Wu, Xiuming Zhang, Zhoutong Zhang, Chengkai Zhang, Tianfan Xue, Joshua B Tenenbaum, and William T Freeman. Pix3d: Dataset and methods for singleimage 3d shape modeling. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2974–2983, 2018. 13
- [61] Stanislaw Szymanowicz, Christian Rupprecht, and Andrea Vedaldi. Viewset diffusion:(0-) image-conditioned 3d genera-

tive models from 2d data. arXiv preprint arXiv:2306.07881,

2023. 2, 3

- [62] Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi, Lizhuang Ma, and Dong Chen. Make-it-3d: High-fidelity 3d creation from a single image with diffusion prior. arXiv preprint arXiv:2303.14184, 2023. 3
- [63] Shitao Tang, Fuyang Zhang, Jiacheng Chen, Peng Wang, and Yasutaka Furukawa. Mvdiffusion: Enabling holistic multiview image generation with correspondence-aware diffusion. arXiv preprint arXiv:2307.01097, 2023. 2, 3
- [64] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In European conference on computer vision, pages 402–419. Springer, 2020. 6
- [65] Shubham Tulsiani, Abhishek Kar, Joao Carreira, and Jitendra Malik. Learning category-specific deformable 3d models for object reconstruction. IEEE transactions on pattern analysis and machine intelligence, 39(4):719–731, 2016. 2, 13
- [66] Shubham Tulsiani, Tinghui Zhou, Alexei A Efros, and Jitendra Malik. Multi-view supervision for single-view reconstruction via differentiable ray consistency. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2626–2634, 2017. 13
- [67] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12619–12629, 2023. 3
- [68] Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, and Yu-Gang Jiang. Pixel2mesh: Generating 3d mesh models from single rgb images. In Proceedings of the European conference on computer vision (ECCV), pages 52–67,

2018. 2, 13

- [69] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. arXiv preprint arXiv:2106.10689, 2021. 3, 7, 13
- [70] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 6
- [71] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. arXiv preprint arXiv:2305.16213, 2023. 2, 3
- [72] Daniel Watson, William Chan, Ricardo Martin-Brualla, Jonathan Ho, Andrea Tagliasacchi, and Mohammad Norouzi. Novel view synthesis with diffusion models. arXiv preprint arXiv:2210.04628, 2022. 2, 3
- [73] Haohan Weng, Tianyu Yang, Jianan Wang, Yu Li, Tong Zhang, CL Chen, and Lei Zhang. Consistent123: Improve consistency for one image to 3d object synthesis. arXiv preprint arXiv:2310.08092, 2023. 2
- [74] Zhenzhen Weng, Zeyu Wang, and Serena Yeung. Zeroavatar: Zero-shot 3d avatar generation from a single image. arXiv preprint arXiv:2305.16411, 2023. 2, 3
- [75] Sunu Wibirama, Paulus Insap Santosa, Putu Widyarani, Nanda Brilianto, and Wina Hafidh. Physical discomfort and

- eye movements during arbitrary and optical flow-like motions in stereo 3d contents. Virtual Reality, 24(1):39–51, 2020. 2
- [76] Jiajun Wu, Chengkai Zhang, Xiuming Zhang, Zhoutong Zhang, William T Freeman, and Joshua B Tenenbaum. Learning shape priors for single-view 3d completion and reconstruction. In Proceedings of the European Conference on Computer Vision (ECCV), pages 646–662, 2018. 13
- [77] Rundi Wu, Yixin Zhuang, Kai Xu, Hao Zhang, and Baoquan Chen. Pq-net: A generative part seq2seq network for 3d shapes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 829–838,

2020. 13

- [78] Shangzhe Wu, Christian Rupprecht, and Andrea Vedaldi. Unsupervised learning of probably symmetric deformable 3d objects from images in the wild. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1–10, 2020. 13
- [79] Zhisheng Xiao, Karsten Kreis, and Arash Vahdat. Tackling the generative learning trilemma with denoising diffusion gans. arXiv preprint arXiv:2112.07804, 2021. 3
- [80] Dejia Xu, Yifan Jiang, Peihao Wang, Zhiwen Fan, Yi Wang, and Zhangyang Wang. Neurallift-360: Lifting an in-the-wild 2d photo to a 3d object with 360deg views. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4479–4489, 2023. 3
- [81] Jiayu Yang, Ziang Cheng, Yunfei Duan, Pan Ji, and Hongdong Li. Consistnet: Enforcing 3d consistency for multi-view images diffusion. arXiv preprint arXiv:2310.10343, 2023. 2, 3
- [82] Jianglong Ye, Peng Wang, Kejie Li, Yichun Shi, and Heng Wang. Consistent-1-to-3: Consistent image to 3d view synthesis via geometry-aware diffusion models. arXiv preprint arXiv:2310.03020, 2023. 2, 3
- [83] Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S Huang. Free-form image inpainting with gated convolution. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4471–4480, 2019. 13
- [84] Xiaohui Zeng, Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, and Karsten Kreis. Lion: Latent point diffusion models for 3d shape generation. arXiv preprint arXiv:2210.06978, 2022. 13
- [85] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 6
- [86] Xiuming Zhang, Zhoutong Zhang, Chengkai Zhang, Josh Tenenbaum, Bill Freeman, and Jiajun Wu. Learning to reconstruct shapes from unseen classes. Advances in neural information processing systems, 31, 2018. 2, 13
- [87] Zhizhuo Zhou and Shubham Tulsiani. Sparsefusion: Distilling view-conditioned diffusion for 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12588–12597, 2023. 2
- [88] Zi-Xin Zou, Weihao Cheng, Yan-Pei Cao, Shi-Sheng Huang, Ying Shan, and Song-Hai Zhang. Sparse3d: Distilling multiview-consistent diffusion for object reconstruction from sparse views. arXiv preprint arXiv:2308.14078, 2023. 2, 3

### A. More Related Work

The challenge of one-image 3D generation has recently attracted significant attention, with various approaches and methods proposed to address this complex problem [4]. In this section, we provide a brief review of the literature.

Classical 3D generative methods. Early works can be broadly categorized into two main groups: primitive-based approaches and depth estimation approaches. Primitivebased approaches [18, 65, 68], focus on the fitting of primitive 3D shapes to 2D images, seeking to align synthetic models with observed image features. They often employ iterative optimization to refine the pose and shape of the model until a satisfactory fit is achieved. On the other hand, depth estimation approaches [78, 86] typically follow a two-step process: They first use a monocular depth estimator (e.g., MiDaS [49]) to predict the 3D geometry, which is then used to render artistic effects through multi-plane images [24, 56] or point clouds [40]. To address imperfections, a pre-trained inpainting model [83] is often applied to fill in missing holes. However, these early approaches may struggle with generalization to real-world data or new object categories.

3D native models. A line of research [7, 9, 10, 16, 18, 57, 76] follows an encoder-decoder framework for modeling the image-to-3D data distribution, which involves the use of global shape latent codes to directly encode the shape information from 3D assets (e.g., ShapeNet [2], Pix3D [60]). In contrast, other works utilize local features and representation-specific 3D generative models that leverage priors constructed from 3D primitives in various formats: point clouds [14, 18, 38, 77, 84], voxels [5, 8, 9, 66], meshes [6, 17, 28, 34, 68], or parametric surfaces [19, 52]. While these 3D native models show impressive performance, they often require extensive 3D data and are constrained to specific object classes within that data. They also suffer from quality degradation when handling real-world images due to domain disparities. Recently, Point-E [42] and Shap-E [26] propose learning text-to-3D diffusion models on large-scale 3D assets to mitigate some of these limitations.

### B. Additional Experimental Setup

Diversity evaluation. Due to the inherent stochastic nature of diffusion models, the outputs they generate can be different w.r.t. the random seed used for their generation. Therefore, the computed metrics can differ depending on the seed we use. To evaluate the diversity of generated samples from each model, we randomly sample 4 instances using different random seeds from the same input image. We then use the CD score to quantify the diversity. By calculating the CD score across these sampled instances (each derived from a different random seed but originating from the sample input images), we obtain an average CD score. This average

CD score represents the overall dissimilarity or diversity observed among the generated samples. The reported values in the main paper are the average CD score calculated across these sampled instances.

Technical details. HarmonyView is built upon the pretrained models of SyncDreamer [33], which generates a set of N = 16 multi-view images, each with an elevation of 30◦ and azimuths evenly distributed in the range of [0◦,360◦]. We assume that the azimuth of both the input view and the first target view is set to 0◦. The viewpoint differences ∆v(n) are calculated based on the differences in elevation and azimuth between the input view and target view. At test time, similar to [32, 33, 37, 46], we estimate an elevation angle and use it as an input. To reconstruct the 3D mesh, we use foreground masks for generated images using CarveKit1, and train the NeuS [69] for 2k steps. For text-to-image-to-3D, 2D images from the input text are created with the assistance of DALL-E-32.

Baselines. In our work, we employ several state-of-the-art methods as baseline models: Zero123 [32], RealFusion [37], Magic123 [46], One-2-3-45 [31], Point-E [42], Shap-E [26], and SyncDreamer [33]. Zero123 [32] is able to generate novel-view images of an object from various viewpoints given a single-view image. Moreover, its integration with the SDS loss [44] bolsters its capability for 3D reconstruction from single-view images. RealFusion [37] leverages Stable Diffusion [50] and the SDS loss for achieving highquality single-view reconstruction. Magic123 [46] builds upon the strengths of Zero123 [32] and RealFusion [37], resulting in a method that further improves the overall quality of 3D reconstruction. One-2-3-45 [31] takes a direct approach by regressing Signed Distance Functions (SDFs) from the output images of Zero123 [32]. Point-E [42] and Shap-E [26] represent 3D generative models trained on an extensive 3D dataset. Both models exhibit the capability to convert a single-view image into either a point cloud or a shape encoded in an MLP. SyncDreamer [33] produces multi-view coherent images from a single-view image by synchronizing intermediate states of generated images using a 3D-aware feature attention mechanism.

### C. Correlation between CD score and Human Evaluation

To assess the efficacy of HarmonyView against SyncDreamer [33] and Zero123 [32], we conducted a user study where participants rated the three approaches using a 5-point Likert-scale (1-5), evaluating (a) Quality, (b) Consistency, and (c) Diversity. Our user study, showcased in Fig. 7, reveals a consistent alignment between the CD Score (CD Score =

- 1https://github.com/OPHoperHPO/image-background-remove-tool
- 2https://cdn.openai.com/papers/dall-e-3.pdf

[Figure 240]

[Figure 241]

(a) Quality (b) Consistency

[Figure 242]

(c) Diversity

- Figure 7. User evaluation examples. We perform a user study to evaluate the effectiveness of our approach, HarmonyView, in comparison to SyncDreamer [33] and Zero123 [32]. Participants were asked to rate the three approaches using a 5-point Likert-scale (1-5), assessing (a) Quality, (b) Consistency, and (c) Diversity.

D/SV ar) and human evaluation metrics. Throughout the study, we observed that the CD Score reliably reflects the correlation between two key factors: SV ar, measuring the diversity in generated images’ alignment with a given text prompt, and D, evaluating creative variation against a reference view using CLIP image encoders.

- 1. Semantic Variance (SV ar) and Consistency: Lower Semantic Variance consistently corresponds to higher consistency in human evaluation. In simpler terms, when the generated images are more aligned in their interpretation of the text prompt, human evaluators tend to agree more on the perceived consistency. This correlation implies that there’s a negative relationship between Semantic Variance and Consistency — lower variance often leads to higher agreement among evaluators.
- 2. Diversity Score (D) and Quality Perception: Higher Diversity Scores tend to lead to lower quality perceptions in

human evaluation. This suggests a somewhat negative correlation between Diversity Score and Quality Perception. Put differently, when the diversity among the generated images is higher — meaning they deviate more from the reference image — human evaluators tend to perceive lower quality. Conversely, higher similarity between the generated images and the reference image correlates with higher perceived quality. In essence, when the visual similarity between the input and target views is higher, the quality tends to be perceived as better by human evaluators.

These findings collectively underscore the critical balance needed between semantic diversity and adherence to the reference image in the pursuit of generating high-quality images aligned with text prompts. Achieving this delicate equilibrium is pivotal to ensure that generated images are diverse enough to capture different interpretations while also being faithful enough to the reference to maintain perceived

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

[Figure 258]

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

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

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

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

Input only s1 only s2 Both

- Figure 8. Qualitative ablation study on novel-view synthesis. Our HarmonyView guides the multi-view diffusion process with two parameters, s1 and s2 (see Eq. (9)). The nuanced interplay between s1 and s2 impacts consistency and diversity throughout the generation process. By skillfully balancing these guiding principles, we can achieve a win-win scenario: generate diverse images that maintain coherence across multiple views and stay faithful to the input view.

quality. HarmonyView demonstrated the highest CD score compared to SyncDreamer [33] and Zero123 [32], indicating that our generated images strike a winning balance between consistency and diversity, excelling in both aspects of fidelity to the reference image and semantic variation.

### D. Additional Results

#### D.1. Novel-view Synthesis

Qualitative ablation study. Our HarmonyView decomposes multi-view diffusion guidance into two distinct guidance components (see Eq. (9)): s1 primarily serves to ensure visual consistency between the input and target views, while s2 focuses on amplifying diversity across novel viewpoints. The significance of this approach is showcased in Fig. 8, where we visually demonstrate how each guidance factor

influences the synthesized images. When prioritizing s1, the quality of synthesis improves significantly as it focuses on aligning the visual consistency between the input and target views. However, in specific cases, like the deer sample, it generates multiple faces of the deer, leading to what’s known as the “Janus problem” — creating facial features on the rear side akin to the front, causing visual anomalies. On the other hand, emphasizing s2 results in increased diversity across the generated samples. However, a fundamental trade-off exists between these two aspects — quality and diversity making it challenging to optimize for both simultaneously. Yet, by employing both s1 and s2 in tandem, we can achieve a win-win scenario. This division allows us to precisely discern the impact of each guidance factor on the generation process. By skillfully balancing these guiding principles, our method becomes empowered to generate a rich and varied

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

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

Input HarmonyView SyncDreamer [33] Zero123 [32]

- Figure 9. Additional novel-view synthesis comparison. HarmonyView creates diverse, coherent multi-view images for complex scenes, effortlessly generating realistic front views from rear-view input images.

PSNR↑ SSIM↑ LPIPS↓ Eflow↓ Best Avg. Var. Best Avg. Var.∗ Best Avg. Var.∗ Best Avg. Var.

Method

Zero123 [32] 18.98 18.79 0.048 0.795 0.792 1.003 0.166 0.170 2.025 3.820 4.185 0.197 SyncDreamer [33] 20.19 19.74 0.242 0.819 0.813 4.465 0.140 0.148 7.922 2.071 2.446 0.458 HarmonyView 20.69 20.24 0.260 0.825 0.819 5.295 0.133 0.140 8.038 1.945 2.350 0.510

Table 6. Statistical analysis of novel-view synthesis on GSO [13] dataset. We report PSNR, SSIM, LPIPS, and Eflow for the best-matched instance with GT, as well as the average and variance across four instances. The variances marked as ∗ are reported with scaling by 10−5.

array of images, exhibiting both multi-view coherence and fidelity to the input view.

geometric coherence across these views. Unlike its counterparts, HarmonyView maintains a harmonious relationship between different views, ensuring the consistent appearance, shapes, and elements of objects. In addition, HarmonyView can extrapolate realistic frontal views from the rear-view input image (see third sample). This further underscores the versatility and robustness of HarmonyView. Overall, HarmonyView is able to generate a diverse set of images while maintaining a sense of realism and coherence across the multiple views.

Qualitative comparison. Figure 9 provides a glimpse into the capabilities and limitations of different novel-view synthesis methods. Zero123 [32] frequently generates images that lack coherence across multiple viewpoints. These synthesized images often contain implausible variations, such as alterations in the number of cymbals or trees based on the view, or even changes in the shape of eyes. These inconsistencies underscore the struggle of Zero123 to maintain coherence and realism across different perspectives, leading to discrepancies that compromise the overall quality of multi-view synthesis. SyncDreamer [33] faces challenges in preserving the expected visual similarity across different viewpoints. The generated images often display deviations in overall size, empty or missing regions, or distorted forms, leading to an overall loss of visual completeness and integrity. Instances where facial features are erased or distorted represent the difficulties SyncDreamer encounters in maintaining the visual fidelity expected across diverse views. In stark contrast, HarmonyView stands out for its ability to generate diverse yet plausible multi-view images while preserving

Statistical analysis. In Table 6, we conduct a comprehensive statistical analysis on the GSO [13] dataset, evaluating the performance of three methods: HarmonyView, Zeor123 [32], and SyncDreamer [33]. We report PSNR, SSIM, LPIPS, and Eflow for the best-matched instance with ground truth, as well as the average and variance across four instances. Upon comparison, HarmonyView demonstrates superior performance across all metrics when compared to Zeor123 and SyncDreamer. It attains the highest scores in PSNR and SSIM, indicating better image quality in terms of both fidelity and structural similarity when compared to the ground truth. Moreover, HarmonyView also exhibits the

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

Input HarmonyView SyncDreamer [33] Zero123 [32] One-2-3-45 [31] Point-E [42] Shap-E [26]

- Figure 10. Additional 3D reconstruction comparison. HarmonyView excels at generating high-fidelity 3D meshes that achieve precise geometry with a realistic appearance while sidestepping common pitfalls for comprehensive and captivating reconstructions.

lowest LPIPS and Eflow scores, signifying reduced perceptual differences and flow errors when matched against the ground truth. Interestingly, HarmonyView shows higher variability (indicated by larger variance values) across instances compared to other methods. This variability might imply that while HarmonyView generally performs well, its performance might fluctuate more across different instances or scenarios compared to the Zero123 and SyncDreamer. Nevertheless, it is essential to note that this variability in performance also reflects its diversity in samples. This could imply that while HarmonyView showcases a broader range of outputs, it still maintains a high level of image quality.

#### D.2. 3D Reconstruction

In Fig. 10, the results exemplify HarmonyView’s exceptional quality compared to other methods evaluated for 3D reconstruction. While contrasting with competing methods, it is evident that these approaches encounter various challenges in handling the reconstruction process. For instance, both Point-E [42] and Shap-E [26] struggle significantly with incomplete reconstructions, failing to capture the entirety of the intended 3D shapes. This deficiency results in reconstructions that lack certain crucial elements, undermining the fidelity of the output. In the case of One-2-3-45 [31], the method exhibits a tendency to produce ambiguous shapes, failing to accurately represent the intended shape contours. Furthermore, Zero123 [32] faces difficulties in capturing fine elements within the reconstructed shapes, which diminishes the overall fidelity and detail level of the output.

SyncDremaer [33] also shows discontinuities or holes within the generated 3D meshes. These imperfections detract from the coherence and completeness of the reconstructed shape. In contrast, HarmonyView produces high-quality 3D meshes that achieve accurate geometry while maintaining a realistic appearance. Its ability to circumvent the pitfalls experienced by other methods speaks volumes about its capability to generate comprehensive, detailed, and visually compelling reconstructions.

### E. Discussion

#### E.1. Limitations & Future Work

While HarmonyView demonstrates promising results in enhancing both visual consistency and novel-view diversity in single-image 3D content generation, several limitations warrant further investigation. Firstly, our multi-view diffusion formulation somewhat mitigates inherent trade-offs between consistency and diversity to achieve a certain level of Pareto optimality. However, the complete separation of these aspects to eliminate the trade-off entirely remains a challenging pursuit. Secondly, HarmonyView’s current focus primarily revolves around object-centric scenes. This poses limitations when dealing with complex scenarios involving multiple interacting objects, varying scales, and intricate geometries. Expanding the technique to encompass such diverse and intricate scenes demands innovative approaches that account for object interactions, spatial relationships, and contextual understanding within the scene. Moreover,

our current setting typically involves single objects without backgrounds, simplifying the requirements for realism and diversity. The ignorance of background significantly reduces the expectations of synthesizing diverse images. To accommodate in-the-wild multi-object scenes with complex backgrounds, HarmonyView requires the use of an external background removal tool (e.g., CarveKit). Addressing these limitations effectively presents ample opportunities for innovation and refinement within the field. Exploring these avenues promises to advance the field towards more comprehensive and realistic 3D content generation from single images.

#### E.2. Ethical Considerations

The advancements in one-image-to-3D bring forth several ethical considerations that demand careful attention. One key concern is the potential misuse of generated 3D content. These advancements could be exploited to create deceptive or misleading visual information, leading to misinformation or even malicious activities like deepfakes, where fabricated content is passed off as genuine, potentially causing harm, misinformation, or manipulation. It is essential to establish responsible usage guidelines and ethical standards to prevent the abuse of this technology. Another critical concern is the inherent bias within the training data, which might lead to biased representations or unfair outcomes. Ensuring diverse and representative training datasets and continuously monitoring and addressing biases are essential to mitigate such risks. Moreover, the technology poses privacy implications, as it could be used to reconstruct 3D models of objects and scenes from any images. Images taken without consent or from public spaces could be used to reconstruct detailed 3D models, potentially violating personal privacy boundaries. As such, it is crucial to implement appropriate safeguards and obtain informed consent when working with images containing personal information.

