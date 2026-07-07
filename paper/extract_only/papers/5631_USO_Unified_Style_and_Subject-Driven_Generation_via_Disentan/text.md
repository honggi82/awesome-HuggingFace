# arXiv:2508.18966v1[cs.CV]26Aug2025

[Figure 1]

[Figure 2]

### USO: Unified Style and Subject-Driven Generation via Disentangled and Reward Learning

#### Shaojin Wu Mengqi Huang ∗ Yufeng Cheng Wenxu Wu Jiahe Tian Yiming Luo Fei Ding † Qian He

UXO Team, Intelligent Creation Lab, ByteDance

#### Abstract

Existing literature typically treats style-driven and subject-driven generation as two disjoint tasks: the former prioritizes stylistic similarity, whereas the latter insists on subject consistency, resulting in an apparent antagonism. We argue that both objectives can be unified under a single framework because they ultimately concern the disentanglement and re-composition of content and style, a long-standing theme in style-driven research. To this end, we present USO, a Unified Style-Subject Optimized customization model. First, we construct a large-scale triplet dataset consisting of content images, style images, and their corresponding stylized content images. Second, we introduce a disentangled learning scheme that simultaneously aligns style features and disentangles content from style through two complementary objectives, style-alignment training and content–style disentanglement training. Third, we incorporate a style reward-learning paradigm denoted as SRL to further enhance the model’s performance. Finally, we release USO-Bench, the first benchmark that jointly evaluates style similarity and subject fidelity across multiple metrics. Extensive experiments demonstrate that USO achieves state-of-the-art performance among open-source models along both dimensions of subject consistency and style similarity. Code and model: https://github.com/bytedance/USO

Date: August 27, 2025 Project Page: https://bytedance.github.io/USO/ Correspondence: Shaojin Wu at wushaojin@bytedance.com

#### 1 Introduction

The significant advancements in image generation over the past years have greatly improved generative controllability, fundamentally changing how humans create images, i.e., whether through abstract textual descriptions, specific visual reference images, or both. Research on leveraging both textual and visual conditions has attracted increasing interest, giving rise to numerous real-world tasks such as style-driven generation and subject-driven generation. While textual conditions are typically explicit, visual conditions are inherently noisy, as images intrinsically embody a rich spectrum of features (e.g., style, appearance), of which only a specific one is relevant to a specific task.

* Corresponding author † Project lead.

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Subject/Identity Driven Generation

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Subject/Identity Driven Stylization

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Style Driven Generation

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Multi-style Driven Generation

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Style-subject Driven Generation (Layout-preserved)

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Style-subject Driven Generation (Layout-shifted)

Figure 1 Showcase of the versatile abilities of the USO model. Prompts are in Table 6.

[Figure 54]

Subject-Driven

[Figure 55]

Content Disentanglement

onsistency

[Figure 56]

C

Subject

[Figure 57]

[Figure 58]

[Figure 59]

Reference Images

Style-Subject-Driven Unified Generation

[Figure 60]

[Figure 61]

Triplet Data

Disentangled and Reward Learning

Style S

imilarity

Style Disentanglement

[Figure 62]

Style-Driven

[Figure 63]

- Figure 2 Illustration of our motivation. By jointly disentangling content and style across tasks, we unify style-driven and subject-driven generation within a single framework.

For instance, style-driven generation requires only the style feature from the reference images, whereas other features constitute noise. Therefore, a fundamental and long-standing challenge in these tasks is to accurately include all required features from the reference image while simultaneously excluding other noisy ones, e.g., including only the style in style-driven generation or only the subject’s appearance in subject-driven generation.

Extensive efforts in the literature have been dedicated to disentangling different features in visual conditions (i.e., reference images). On the one hand, in the realm of style-driven generation, DEADiff [23] employs QFormer to selectively query only the style features from reference images. CSGO [37] constructs contentstyle-stylized triplets to facilitate style-content decoupling during training. StyleStudio [16] introduces style-based classifier-free guidance (SCFG) to enable selective control over stylistic elements and to mitigate the influence of irrelevant features. On the other hand, subject-driven generation methods primarily focus on disentangling subject appearance features or constructing more effectively disentangled paired data. For example, RealCustom [10, 19] proposes a dual-inference framework that selectively incorporates subjectrelevant features into subject-specific regions. UNO [34] leverages the in-context capabilities of DiT to progressively improve both the quality of paired data and the model itself. To conclude, existing methods primarily focus on task-specific disentanglement by designing tailored datasets or model architectures for each individual task, thereby performing disentanglement in an isolated, single-task context.

In this study, we argue that a more comprehensive and precise disentanglement approach should fully account for the coupling and complementarity between different generation tasks. Each task should not only learn which features to include, but, more importantly, also learn which features to exclude, i.e., features that are often required by other tasks. Therefore, learning to include certain features in one task inherently informs and enhances the process of learning to exclude those same features in a complementary task, and vice versa. For example, style-driven generation aims to incorporate stylistic features while excluding subject appearance features, whereas subject-driven generation does the exact opposite. The ability to learn and include subject appearance features in subject-driven generation can, in turn, help style-driven generation more effectively learn to exclude these features, thereby improving disentanglement for both tasks. In conclusion, we believe

that jointly modeling complementary tasks enables a mutually reinforcing disentanglement process, leading to a more precise separation of relevant and irrelevant features for each task.

Based on the above motivation, we propose a novel cross-task co-disentanglement paradigm to unify subject-driven and style-driven generation, and, more importantly, to mutually enhance the performance of both tasks, as illustrated in Figure 2. Specifically, this co-disentanglement paradigm is implemented through a subject-for-style data curation framework and a style-for-subject model training framework. The subject-for-style framework first utilizes a state-of-the-art subject model to generate high-quality style data, while the style-for-subject framework subsequently trains a more effective subject model under the guidance of style rewards and disentangled training. Technically, on the one hand, for the subject-for-style data curation framework, we build upon a state-of-the-art subject-driven model [34] and further develop both a stylization expert and a de-stylization expert to curate stylized and non-stylized images. This process ultimately constructs triplet data pairs in the form of <style reference, de-stylized subject reference, stylized subject result> for subsequent model training. On the other hand, for the style-for-subject model training framework, we propose a Unified Style-Subject Optimized (USO) customization model, which introduces progressive style alignment and style-subject disentanglement training, both supervised by a style reward.

Our contributions are summarized as follows:

Concepts: We point out that existing style-driven and subject-driven methods focus solely on isolated disentanglement within each task, neglecting their potential complementarity and thus leading to suboptimal disentanglement. For the first time, we propose a novel cross-task co-disentanglement paradigm that unifies style-driven and subject-driven tasks, enabling mutual enhancement and achieving significant performance improvements for both.

Technique: We present a novel cross-task triplet curation framework that bridges style-driven and subjectdriven generation. Building on this, we introduce USO, a unified customization architecture that incorporates progressive style-alignment training, content–style disentanglement training, and a style reward learning paradigm to further promote cross-task disentanglement. We further release USO-Bench, to the best of our knowledge, the first benchmark tailored for evaluating cross-task customization.

Performance: Extensive evaluations on USO-Bench and DreamBench [25] show that USO achieves stateof-the-art results on subject-driven, style-driven, and joint style-subject-driven tasks, attaining the highest CLIP-T, DINO, and CSD scores. USO can handle individual tasks and their free-form combinations while exhibiting superior subject consistency, style fidelity, and text controllability as shown in Figure 1.

#### 2 Related Work

##### 2.1 Style Transfer

Style Transfer aims to apply the style in the reference image to the given content image or fully generated image. Early work like adaptive instance normalization [11] achieved impressive style transfer results with layout-preserved results by simply using a pre-trained network as the style encoder and well-designed injection modules.

The recent powerful text-to-image generation base models, like Stable Diffusion [4, 21] and FLUX [14], along with style transfer plugins built upon them, have significantly improved the convenience and effectiveness of performing this task. Several are even training-free, like StyleAlign [35] and StylePrompt [12] which transfer the style via simple query-key swapping in the specific self-attention layers. Other training-based methods can theoretically achieve better fitting and style transfer performance, but they also raise concerns of content leakage. IP-adapter [39] and DEADiff [23] demonstrate the style transfer ability with a new decoupled cross-attention layer trained with coupled data, and overcome the content leakage by decreasing the injection weights in inference-time. InstanceStyle [28], StyleShot [7] and B-lora [5] provide more detailed time-aware and layer-aware injection strategies to disentangle the style and content feature injections. However, those disentanglement analyses are tied to the specific model architecture and hard to migrate.

[Figure 64]

[Figure 65]

[Figure 66]

|[Figure 67]<br><br>Curated Stylized Data| |finetune|
|---|---|---|
| | | |
| | | |

Stylized/De-stylized Expert Models

UNO Model

[Figure 68]

USO Training Dataset

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

style transfer

Stylized Expert Models

[Figure 75]

style similarity 𝐼 𝐼 𝐼

[Figure 76]

Style Image 𝐼

Layout-Preserved Triplets VLM-based Filter

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

de-stylized

De-stylized Expert Models

subject consistency

𝐼 𝐼 𝐼

Content Image 𝐼

Target Image 𝐼

Layout-Shifted Triplets

- Figure 3 Illustration of our proposed cross-task triplet curation framework, which systematically generates layoutpreserved and layout-shifted triplets.

- 2.2 Subject-Driven generation

Subject-driven generation refers to generating images of the same subject conditioned on a text instruction and reference images of given subjects. Dreambooth [25] and IP-Adapter [39] turn a UNet-based text-toimage model into a subject-driven model by parameter-efficient tuning or a newly introduced attention plug-in. Recently, popular image-generation foundation models have shifted from UNet-based architectures to transformer-based ones. The inherent in-context learning capabilities of transformers have greatly enriched research on subject-driven generation. ICLoRA [9], OmniControl [27], UNO [34], and FLUX.1 Kontext [15] use shared attention between the generated image and reference image to train a text-to-image DiT into a subject-driven variant. It is worth noting that some of them have extended the reference subject to other types. OmniControl [27] supports layout control image as a reference, UNO [34] supports multiple reference images input, and DreamO [20] can work for simple style transfer. They have indicated that various types of reference-guided generation can be unified within the DiT in-context framework. This further prompts the question of whether jointly addressing different tasks in this setting could lead to mutual benefits across them.

- 3 Methodology

- 3.1 Preliminary

Latent diffusion models [21, 24] have evolved from UNet-based architectures to DiT-based designs, with steadily improving foundational capabilities. MM-DiT [4, 14] further elevates image-generation quality, spawning numerous downstream applications and unlocking greater controllability in text-to-image generation [34]. It incorporates a multi-modal attention mechanism that can be seamlessly extended to an in-context generation framework: the conditioned tokens are directly concatenated with the text prompt and the noisy latent, yielding the formulation:

Attention([c,zt,zc]) = softmax

QK⊤

√

d

V, (1)

where Z = [c,zt,zc] denotes the concatenation of text tokens, noisy latent, and condition tokens. This allows both representations to function within their own respective spaces while still taking the other into account.

- 3.2 Cross-Task Triplet Curation Framework

This section details the construction of cross-task triplets for USO training. Although prior works [29, 37] have explored triplet generation, they retain the original layout, preventing any pose or spatial re-arrangement

Stage 1: Style Alignment Training Stage 2: Content-Style Disentanglement Training

[Figure 82]

[Figure 83]

𝓛

𝓛

𝐼 𝐼

style similarity style similarity

[Figure 84]

[Figure 85]

MM-DiT Blocks

MM-DiT Blocks

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Hierarchical Projector

Hierarchical Projector

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

Noisy Latent

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

SigLIP

###### VAE

VAE

SigLIP

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

An old man sits and eats bread

An old man sits and eats bread

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Style Image 𝐼

Style Image 𝐼

Content Image 𝐼

Target Image 𝐼

Target Image 𝐼

- Figure 4 Illustration of the training framework of USO. USO unifies subject-driven and style-driven generation in two stages: Stage 1 aligns SigLIP embeddings via style-alignment training to yield a style-capable model; Stage 2 disentangles the conditional encoders and trains on triplets to enable the joint conditional generation. Finally, a style-reward learning paradigm supervises both stages to yield a stronger unified model.

of the subject. To jointly enable subject-driven and style-driven generation beyond simple instruction-based edits, we curate a new USO dataset expressly designed for this unified objective.

Figure 3 provides an overview of USO dataset. Our co-disentanglement paradigm starts from a subjectfor-style data curation framework. Among many possible tasks, subject-driven (i.e., UNO-1M [34]) and instruction-based editing (i.e., X2I2 [32]) datasets are comparatively easy to collect at scale, enabling targeted task-specific corpora. In particular, subject-driven data emphasizes learning from content cues while preserving subject identity and consistency; instruction-based editing bridges styles by preserving spatial layout and transferring appearance between realistic and stylized domains in both directions. These resources naturally support training domain-specialist models and, through deliberate dataset design, induce the capabilities we care about (e.g., extracting task-relevant features conditioned on image type). Guided by these insights, we curate 200k stylized image pairs sourced from publicly licensed datasets and augmented with samples synthesized by state-of-the-art text-to-image models. Using these data, we trained two complementary experts on top of the leading customization framework UNO [34]: (1) a stylized expert model that performs style-driven generation conditioned on a style-reference image, producing a new subject rendered in the target style (Irefs from Itgt), and (2) a de-stylization expert model that inverts a stylized image to a photorealistic counterpart, allowing either flexible layout shifts or preservation (Irefc from Itgt).

Each curated stylized image serves as the target Itgt. We synthesize its style reference Irefs via the stylization expert and its content reference Irefc via the de-stylization expert. Following [34], a VLM-based filter enforces style similarity between Itgt and Irefs and subject consistency between Itgt and Irefc . This yields two kinds of triplets, shown in Figure 3: layout-preserved and layout-shifted. Unlike prior work [29, 37], which focuses solely on style-driven generation and confines itself to layout-preserved triplets, our cross-task triplets achieve deeper content–style disentanglement across tasks and are used to train USO.

##### 3.3 Unified Customization Framework (USO)

In this section, we describe how we unify two tasks that have traditionally been treated separately, style-driven and subject-driven generation, into a single model. Each task demands the model to master distinct knowledge: the former emphasizes style similarity, while the latter insists on subject consistency. By excelling at both simultaneously, the model naturally disentangles content from style, a long-standing focus of style-driven

generation, which in turn boosts the quality of stylization and customization. Beyond merely preserving layout during style-driven generation, the model can now freely recombine any subject with any style.

###### 3.3.1 Style Alignment Training.

As illustrated in Figure 4, We start from a pre-trained text-to-image (T2I) model and fine-tune it into a stylized variant via our proposed style-alignment training. Unlike prior in-context generation approaches that rely solely on a VAE E(·) to encode the conditioned image Iref, we argue that style is a more abstract cue demanding richer semantic information. Therefore, we employ the semantic encoder SigLIP instead of the VAE to process the reference style image Irefs . While subject-driven or identity-preserving tasks typically emphasize high-level semantics, style-driven tasks must simultaneously handle two extremes: high-level semantics to accommodate large geometric deformations (e.g., 3-D cartoon styles) and low-level details to reproduce subtle brushstrokes (e.g., pencil sketches). Following recent works like [41], we introduce a lightweight Hierarchical Projector MProj(·) to project multi-scale, fine-grained visual features zs from the extracted SigLIP embeddings {ci}Ni=1, where N represents the layer indices of SigLIP. This process can be formulated as:

zs = Concatenate(MProj({ci}Ni=1)), (2)

Specifically, we assign the style tokens zs the same positional indices as the text tokens c, then feed the fused multimodal sequence z1 into the DiT model as its input:

- z1 = Concatenate(zs,c,zt), (3)

In this stage, we freeze all parameters except the Hierarchical Projector, enabling the extracted style features to be rapidly aligned with the native textual distribution. Consequently, the pretrained T2I model is converted into a stylized variant capable of accepting style-reference images as conditional input.

3.3.2 Content-Style Disentanglement Training.

Building on Stage 1, we introduce subject conditioning in Stage 2 as shown in Figure 4. Following recent paradigms [27, 34], the content image Irefc is encoded into pure conditional tokens zc by a frozen VAE encoder E(·). We formulate USO as a multi-image conditioned model, yet explicitly disentangle content and style features via separate encoders. This design alleviates content leakage, where extraneous style-image details undesirably appear in the output, and also helps the model learn to exclude undesired features for the specific task, as introduced in Section 1.

During training, the Hierarchical Projector remains frozen while the DiT parameters are unfrozen. Content tokens receive positional indices via UnoPE [34] in its diagonal layout. The final multimodal input sequence z2 is thus expressed as:

- z2 = Concatenate(zs,c,zt,zc), (4)

Consequently, USO can directly handle both subject-driven and style-driven tasks on the proposed triplet dataset.

Compared with prior open-source style-driven methods, most of which either (i) rigidly preserve the content layout while altering its style [29] or (ii) retain layout via an external ControlNet at the cost of subject consistency with the content image [23, 28]-USO removes these constraints. Trained on our triplet data, it freely re-positions the subject from the content image into any scene while re-rendering it in the style of the reference image.

###### 3.3.3 Style Reward Learning

Beyond the standard flow-matching objective, we introduce Style Reward Learning (SRL) to explicitly disentangle style from content during optimization. Flow-matching pre-trains the model by minimizing the L2 distance between the predicted velocity vθ(xt,t) and the true velocity vt = dα

dt ϵ. Building on this, we denote the training objective as LPre, which can be computed as:

dt x0 + dσ

t

t

0,t,ϵ[w(t)∥vθ − vt∥2] (5)

LPre = Ex

Algorithm 1 Style Reward Learning (SRL) with Flow Matching

Require: Customization model net with pretrained parameters θ; pretrain loss LPre; reward loss LSRL; reward model MRM; balancing coefficient λ; noise-schedule steps T; SRL fine-tuning interval [ts,te]; dataset D = {(y,I0,Irefc ,Irefs )}, y is prompt, I0 is target image and Irefc ,Irefs are reference content and style images (Section 3.2)

- 1: for (y,I0,Irefc ,Irefs ) ∈ D do
- 2: LPre ← netθ(y,I0,Irefc ,Irefs ) // calculate pretrain loss with Equation (5)
- 3: t ∼ U(ts,te) // pick a random time step in [ts,te]
- 4: xT ∼ N(0,I)
- 5: for τ = T,...,t + 1 do
- 6: vˆτ ← no-grad(netθ(y,xτ,Irefc ,Irefs ))
- 7: xτ−1 ← xτ − vˆτ∆t // reverse-step update
- 8: end for
- 9: vˆt ← netθ(y,xt,Irefc ,Irefs )
- 10: Iˆ0 ← decode(xt − vˆt∆t) // predict original image
- 11: LSRL ← −MRM(Iˆ0,Irefs ) // calculate SRL loss with negative reward with Equation (6)
- 12: L ← LPre + λLSRL
- 13: θ ← θ − η ∇θL // update model parameters via gradient descent (η is learning rate)
- 14: end for

where w(t) is a weighting function, vθ is a neural network parameterized by θ, and αt = 1 − t,σt = t are continuous-time coefficients with t ∈ [0,T = 1]. The sampling process is from t = T with xT ∼ N(0,I) and stops at t = 0, solving the PF-ODE via dxt = vθ(xt,t)dt.

Extending ReFL [38] from T2I generation to the reference-to-image setting, where generation is conditioned on both an image reference and and its corresponding text prompt, SRL alternates between computing a reward score and back-propagating the reward signal. As shown in Figure 4, we define the reward score as the style similarity between the reference style image Irefs and the generated stylized image Iθ, measured by either a VLM-based filter or the CSD model MRM(·) [26, 37]. The reward loss is defined as:

i∼Y[ϕ(MRM(yi,Iθ(yi)))] (6)

LSRL = Ey

where Y = {yi}ni=1 is the prompt set, ϕ maps reward scores to per-sample loss values, and Iθ denotes the image generated by the diffusion model with parameters θ corresponding to prompt y.

The final objective combines both losses: L = LPre + λLSRL, λ = 0 before step S, λ = 1 thereafter. (7)

As shown in Algorithm 1, we present the detailed SRL algorithm. The entire process comprises gradient-free inference followed by a reward-backward step.

Qwen-Image Edit

FLUX.1 Kontext dev

###### Reference Image Prompt USO (Ours) OmniGen2 BAGEL UNO

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Wool felt style, a clock on top of fabric.

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Transform the style of image into Lego building block wind.

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

The man is reading a book in a cafe.

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

The woman crouched down in the garden and carefully trimmed the flower branches.

Figure 5 Qualitative comparison with different methods on subject-driven generation.

Reference Image Prompt USO (Ours) StyleStudio DreamO CSGO InstantStyle DEADiff

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

A dog lying on its stomach.

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

A cat sleeping on a chair.

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

A child standing beside a huge cat.

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

Small boat in the lake.

Figure 6 Qualitative comparison with different methods on style-driven generation.

The woman/man in flower shops carefully match bouquets, conveying beautiful emotions and blessings with flowers.

Text Prompts

The woman/man gave an impassioned speech on the podium.

The woman/man with a mountain in Reference the background.

The woman/man is reading a book in a cafe.

Images

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

USO (Ours)

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

InfiniteYou

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

USO (Ours)

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

InfiniteYou

Figure 7 Qualitative comparison with different methods on identity-driven generation.

Content&Style Reference

Content&Style Reference

USO (Ours) OmniStyle StyleID

USO (Ours) OmniStyle StyleID

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

“” “”

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

A can with a city in the background.

The woman is reading a book in a cafe.

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

… gave an impassioned speech on the podium.

The man is skateboarding on the street.

###### Figure 8 Qualitative comparison with different methods on style-subject-driven generation.

#### 4 Experiments

##### 4.1 Experiments Setting

USO Unified Benchmark. To enable a comprehensive evaluation, we introduce USO-Bench, a unified benchmark built from 50 content images (20 human-centric, 30 object-centric) paired with 50 style references. We further craft 30 subject-driven prompts that span pose variation, descriptive stylization, and instructive stylization, along with 30 style-driven prompts. We generate four images per prompt for both subject-driven and style-driven tasks, and a single image for the combined style-subject-driven task. This yields 6000 samples for subject-driven generation, 7040 for style-driven generation, and 29500 for the combined task; full construction details are provided in the supplementary material.

Model Subject-driven generation Style-driven generation

CLIP-I↑ DINO↑ CLIP-T↑ CSD↑ CLIP-T↑

RealCustom++ [10] 0.314 0.615 0.303 - RealGeneral [18] 0.485 0.732 0.275 - UNO [34] 0.605 0.789 0.264 - BAGEL [3] 0.516 0.741 0.298 - OmniGen2 [32] 0.475 0.723 0.302 - FLUX.1 Kontext dev [15] 0.579 0.775 0.287 - Qwen-Image Edit [31] 0.544 0.756 0.302 - -

DEADiff [23] - - - 0.462 0.274 InstantStyle-XL [28] - - - 0.540 0.276 CSGO [37] - - - 0.452 0.272 StyleStudio [16] - - - 0.348 0.282

DreamO [20] 0.588 0.787 0.280 0.454 0.278 USO (Ours) 0.623 0.793 0.288 0.557 0.282

- Table 1 Quantitative results for subject-driven and style-driven generation on USO-Bench.

Model CSD↑ CLIP-T↑

StyleID [2] 0.407 0.230 OmniStyle [29] 0.365 0.229

USO (Ours) 0.495 0.283

- Table 2 Quantitative results for style-subject-driven driven generation on USO-Bench.

Evaluation Metrics. For quantitative evaluation, we assess each task along three dimensions: (1) subject consistency, measured by the cosine similarity of CLIP-I and DINO embeddings following [34]; (2) style similarity, reported via the CSD score [26] for both style-driven and style-subject-driven generation, following [37]; and (3) text–image alignment, evaluated with CLIP-T across all three tasks.

Comparative Methods. As a unified customization framework, USO is evaluated against both task-specific and unified baselines. For subject-driven generation, we benchmark RealCustom++ [19], RealGeneral [18], UNO [34], OmniGen2 [32], BAGEL [3], FLUX.1 Kontext dev [15], and Qwen-Image Edit [31]. For style-driven generation, we compare StyleStudio [16], DreamO [20], CSGO [37], InstantStyle [28], and DEADiff [23]. For the joint style-subject-driven setting with dual conditioning, we compare OmniStyle [29] and StyleID [2]. We also compared with InfiniteYou [13] to further demonstrate the positive effect of our proposed method on identity tasks.

##### 4.2 Experimental Results

Subject-Driven Generation. As shown in Figure 5, the first two columns demonstrate that USO simultaneously satisfies both descriptive and instructive style edits while maintaining high subject consistency. In contrast, competing methods either fail to apply the style or lose the subject. The last two columns further illustrate USO’s strength in preserving human appearance and identity; it adheres strictly to the textual prompt and almost perfectly retains facial and bodily features, whereas other approaches fall short. When the prompt is “The man is reading a book in a cafe”, FLUX.1 Kontext dev [15] achieves decent facial similarity but carries copy-paste risks. In Figure 7 we compare with task-specific identity-preserving methods; USO produces more realistic, non-plastic results with higher identity consistency. As reported in Table 1, USO significantly outperforms prior work, achieving the highest DINO and CLIP-I scores and a leading CLIP-T score.

Style-Driven Generation. Figure 6 shows that USO outperforms task-specific baselines in preserving the original style, including global color palettes and painterly brushwork. In the last two columns, given highly abstract references such as material textures or Pixar-style renderings, USO handles them almost flawlessly while prior methods struggle, demonstrating the generalization power of our cross-task co-disentanglement. Quantitatively, Table 1 confirms that USO achieves the highest CSD and CLIP-T scores among all style-driven approaches.

Style-Subject-Driven Generation. As illustrated in Figure 8, we evaluate USO on both layout-preserved and layout-shifted scenarios. When the input prompt is empty, USO not only preserves the original layout of the content reference but also delivers the strongest style adherence. In the last two columns, under a more complex prompt, USO simultaneously preserves the subject and identity consistency, matches the reference style, and aligns with the text, while other methods lag markedly and merely adhere to the text. Table 2 corroborates these observations, showing USO achieves the highest CSD and CLIP-T scores and substantially outperforms all baselines.

User Study. We further conduct an online user-study questionnaire to compare state-of-the-art subject-driven and style-driven methods. Questionnaires were distributed to both domain experts and non-experts, who ranked the best results for each task. (1) Subject-driven tasks were evaluated on text fidelity, visual appeal, subject consistency, and overall quality. (2) Style-driven tasks were judged on text fidelity, visual appeal, style similarity, and overall quality. As shown in Figure 9, our USO achieves top performance on both tasks, validating the effectiveness of our cross-task co-disentanglement and showcasing its capability to deliver state-of-the-art results.

Subject-driven Generation

Style-driven Generation

USO (ours)

USO (ours)

visual appeal

subject consistency

Kontext dev

StyleStudio

|fidelity<br><br>|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>|
|---|---|
| |subject<br><br>|

| |style<br><br>1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>|
|---|---|
|appeal<br><br>| |

OmniGen2

DreamO

BAGEL

CSGO

UNO

InstantStyle

RealGeneral

DEADiff

RealCustom++

text fid

similarity

ct consistency

visual ap

overall performance

overall performance

###### Figure 9 Radar charts of user evaluation of methods for subject-driven and style-driven generation on different dimensions.

#### 5 Ablation Study

##### 5.1 Effect of Style Reward Learning (SRL).

For style-driven task: As shown in Figure 10, the last three columns reveal a clear boost in style similarity for both style-driven and style-subject-driven tasks; the stroke textures and painting style closely match the reference images, confirming the effectiveness of our style reward learning.

For subject-driven task: In the first three and final columns of Figure 10, we observe a notable improvement in subject and identity consistency, with more uniform details and higher facial similarity.

As shown in Table 3, removing SRL leads to a sharp drop in the CSD score and simultaneous declines in CLIP-I and CLIP-T. Notably, we rely solely on style reward and introduce no identity-specific data; nevertheless, the unified model benefits in content consistency. By sharpening the model’s ability to extract and retain desired features, SRL yields an overall improvement across all tasks, strongly validating our motivation. Beyond gains in subject and identity fidelity, we observe a noticeable enhancement in aesthetic quality (e.g., texture as in VMix [33]) and a marked reduction in the plastic artifact, which is an issue that has long challenged text-to-image generation [33]. Through SRL training, the model exhibits emerging properties even in tasks not explicitly targeted during training.

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

Reference Images

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

w/ SRL

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

w/o SRL

Figure 10 Ablation study of SRL. The blue boxes denote content reference and the purple boxes denote style reference. Prompts are "A toy with a mountain in the background.", "The man on the beach.", "The woman is skateboarding on the street.", "A beautiful woman.", "A beautiful woman.", "The woman gave an impassioned speech on the podium." from left to right.

- Table 3 Ablation study of different components proposed in USO.

Model Subject-driven Style-subject-driven

CLIP-I↑ CLIP-T↑ CSD↑ CLIP-T↑

USO (Ours) 0.623 0.288 0.495 0.283

w/o SRL 0.620 0.284 0.413 0.280 w/o SAT 0.621 0.275 0.409 0.280 w/o DE 0.594 0.269 0.382 0.277

Table 4 Ablation study of different projector in USO.

Model CSD↑ CLIP-T↑

resampler (depth=1) 0.336 0.279 resampler, unfreeze siglip 0.155 0.288 mlp (depth=1) 0.277 0.284 mlp, unfreeze siglip 0.179 0.288 hierarchical projector 0.402 0.284

##### 5.2 Effect of Style Alignment Training (SAT).

Removing SAT and instead jointly fine-tuning both SigLIP and DiT from scratch degrades CLIP-T on subject-driven tasks and lowers CSD on style-subject-driven tasks ( Table 3). Qualitatively, Figure 11 shows

Content&Style Reference

USO (Ours) w/o SRL w/o SAT w/o DE

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

Style-driven: A cheetah.

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

Subject-driven: The man is reading a book in a cafe.

Figure 11 Ablation study of USO. Zoom in for details.

that the oil-painting style of the “cheetah” example becomes noticeably weaker.

##### 5.3 Effect of Disentangled Encoder (DE).

Replacing the disentangled encoders with a single VAE for both style and content images harms nearly every metric ( Table 3). Visually, the “cheetah” reverts to a more photorealistic appearance, while the man’s identity suffers a marked loss ( Figure 11).

##### 5.4 Effect of Hierarchical Projector.

- Table 4 shows that the hierarchical projector yields the highest CSD and a leading CLIP-T score, substantially benefiting style-alignment training.

#### 6 Conclusion

In this paper, we present USO, a unified framework capable of subject-driven, style-driven, and joint stylesubject-driven generation. We introduce a cross-task co-disentanglement paradigm that first constructs a systematic triplet-curation pipeline, then applies progressive style-alignment and content–style disentanglement training on the curated triplets. Additionally, we propose a style-reward learning paradigm to further boost performance. To comprehensively evaluate our method, we construct USO-Bench, a unified benchmark that provides both task-specific and joint evaluation for existing approaches. Finally, extensive experiments demonstrate that USO sets new state-of-the-art results on subject-driven, style-driven, and their joint style-subject-driven tasks, exhibiting superior subject consistency, style fidelity, and text controllability.

#### References

- [1] Wenhu Chen, Hexiang Hu, Chitwan Saharia, and William W Cohen. Re-imagen: Retrieval-augmented text-toimage generator. arXiv preprint arXiv:2209.14491, 2022.

- [2] Jiwoo Chung, Sangeek Hyun, and Jae-Pil Heo. Style injection in diffusion: A training-free approach for adapting large-scale diffusion models for style transfer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8795–8805, 2024.

- [3] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

- [4] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.

- [5] Yarden Frenkel, Yael Vinker, Ariel Shamir, and Daniel Cohen-Or. Implicit style-content separation using b-lora. In European Conference on Computer Vision, pages 181–198. Springer, 2024.

- [6] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.

- [7] Junyao Gao, Yanchen Liu, Yanan Sun, Yinhao Tang, Yanhong Zeng, Kai Chen, and Cairong Zhao. Styleshot: A snapshot on any style. arXiv preprint arXiv:2407.01414, 2024.

- [8] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

- [9] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775, 2024.

- [10] Mengqi Huang, Zhendong Mao, Mingcong Liu, Qian He, and Yongdong Zhang. Realcustom: narrowing real text word for real-time open-domain text-to-image customization. In CVPR, pages 7476–7485, 2024.

- [11] Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In ICCV, 2017.

- [12] Jaeseok Jeong, Junho Kim, Yunjey Choi, Gayoung Lee, and Youngjung Uh. Visual style prompting with swapping self-attention. arXiv preprint arXiv:2402.12974, 2024.

- [13] Liming Jiang, Qing Yan, Yumin Jia, Zichuan Liu, Hao Kang, and Xin Lu. Infiniteyou: Flexible photo recrafting while preserving your identity. arXiv preprint arXiv:2503.16418, 2025.

- [14] Black Forest Labs. Flux: Official inference repository for flux.1 models, 2024. URL https://github.com/ black-forest-labs/flux. Accessed: 2025-02-07.
- [15] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

- [16] Mingkun Lei, Xue Song, Beier Zhu, Hao Wang, and Chi Zhang. Stylestudio: Text-driven style transfer with selective control of style elements. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 23443–23452, 2025.

- [17] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pre-trained subject representation for controllable text-to-image generation and editing. Advances in Neural Information Processing Systems, 36:30146–30166, 2023.

- [18] Yijing Lin, Mengqi Huang, Shuhan Zhuang, and Zhendong Mao. Realgeneral: Unifying visual generation via temporal in-context learning with video models. arXiv preprint arXiv:2503.10406, 2025.

- [19] Zhendong Mao, Mengqi Huang, Fei Ding, Mingcong Liu, Qian He, and Yongdong Zhang. Realcustom++: Representing images as real-word for real-time customization. arXiv preprint arXiv:2408.09744, 2024.

- [20] Chong Mou, Yanze Wu, Wenxu Wu, Zinan Guo, Pengze Zhang, Yufeng Cheng, Yiming Luo, Fei Ding, Shiwen Zhang, Xinghui Li, et al. Dreamo: A unified framework for image customization. arXiv preprint arXiv:2504.16915, 2025.

- [21] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In ICLR, 2024. URL https://openreview.net/forum?id=di52zR8xgf.

- [22] Senthil Purushwalkam, Akash Gokul, Shafiq Joty, and Nikhil Naik. Bootpig: Bootstrapping zero-shot personalized image generation capabilities in pretrained diffusion models. arXiv preprint arXiv:2401.13974, 2024.

- [23] Tianhao Qi, Shancheng Fang, Yanze Wu, Hongtao Xie, Jiawei Liu, Lang Chen, Qian He, and Yongdong Zhang. Deadiff: An efficient stylization diffusion model with disentangled representations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8693–8702, 2024.

- [24] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684–10695, 2022.

- [25] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, pages 22500–22510, 2023.

- [26] Gowthami Somepalli, Anubhav Gupta, Kamal Gupta, Shramay Palta, Micah Goldblum, Jonas Geiping, Abhinav Shrivastava, and Tom Goldstein. Measuring style similarity in diffusion models. arXiv preprint arXiv:2404.01292, 2024.

- [27] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 3, 2024.

- [28] Haofan Wang, Matteo Spinelli, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards style-preserving in text-to-image generation. arXiv preprint arXiv:2404.02733, 2024.

- [29] Ye Wang, Ruiqi Liu, Jiang Lin, Fei Liu, Zili Yi, Yilin Wang, and Rui Ma. Omnistyle: Filtering high quality style transfer data at scale. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7847–7856, 2025.

- [30] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. In CVPR, pages 15943–15953, 2023.

- [31] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

- [32] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.

- [33] Shaojin Wu, Fei Ding, Mengqi Huang, Wei Liu, and Qian He. Vmix: Improving text-to-image diffusion model with cross-attention mixing control. arXiv preprint arXiv:2412.20800, 2024.

- [34] Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-to-more generalization: Unlocking more controllability by in-context generation. arXiv preprint arXiv:2504.02160, 2025.

- [35] Zongze Wu, Yotam Nitzan, Eli Shechtman, and Dani Lischinski. Stylealign: Analysis and applications of aligned stylegan models. arXiv preprint arXiv:2110.11323, 2021.

- [36] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024.

- [37] Peng Xing, Haofan Wang, Yanpeng Sun, Qixun Wang, Xu Bai, Hao Ai, Renyuan Huang, and Zechao Li. Csgo: Content-style composition in text-to-image generation. arXiv preprint arXiv:2408.16766, 2024.

- [38] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.

- [39] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

- [40] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.

- [41] Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan, et al. Ssr-encoder: Encoding selective subject representation for subject-driven generation. In CVPR, pages 8069–8078, 2024.

## USO: Unified Style and Subject-Driven Generation via Disentangled and Reward Learning

#### Appendix

##### F.1 Experiments Setting

- F.1.1 Implementation Details.

We begin with FLUX.1 dev [14] and the SigLIP [40] pretrained model. For style alignment stage, we train on pairs {Irefs ,Itgt} for 23,000 steps at batch size 16, learning rate 8e − 5, resolution 768 and reward steps S = 16,000. For content-style disentanglement stage, we train on triplets {Irefc ,Irefs ,Itgt} for 21,000 steps at batch size 64, learning rate 8e − 5, resolution 1024 and reward steps S = 18,000. LoRA [8] rank 128 is used throughout.

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

HumanObject

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

…

…

…

…

+

+

[Figure 273]

Descriptive prompts

The xx is reading a book in a cafe.

Instructive stylization

Transform the style of image into yy style

Descriptive stylization

yy style, the xx is reading a book in a cafe.

[Figure 274]

Subject-driven (Content reference)

[Figure 275]

Style-driven (Style reference)

[Figure 276]

Style-subject-driven (Combination reference)

Stylization prompts

A villa on the coast. A cat sleeping on a chair. A beautiful woman. Handsome boy. A canvas bag. A duck. A child standing beside a huge cat Eiffel tower. …

Layout-preserved prompts

Layout-shifted prompts

“” (empty prompt)

The xx is reading a book in a cafe.

Generation Prompts

Reference Images

USO-Bench

Figure 12 Examples of USO-Bench.

- F.1.2 Details of USO-Bench.

USO-Bench is built to evaluate subject-driven, style-driven, and joint style-subject-driven generation. As shown in Figure 12, each subject-driven sample uses three prompt types: descriptive, instructive-stylization, and descriptive-stylization. By pairing these prompts with style-reference images from style-driven tasks, we obtain style-subject-driven samples via their Cartesian product. The resulting prompts are further split into layout-shifted and layout-preserved variants.

##### F.2 More Results

###### F.2.1 Quantitative Evaluation on DreamBench [25].

To further assess USO, we evaluate it on DreamBench [25] in addition to USO-Bench. Following UNO [34], we generate six images per prompt, yielding 4,500 image groups across all subjects. As shown in Table 5, USO achieves the highest CLIP-I and DINO scores, and with a CLIP-T score of 0.317, it trails the top result (0.318) by only a narrow margin. These results demonstrate USO’s superior subject consistency among state-of-the-art methods.

Method DINO ↑ CLIP-I ↑ CLIP-T ↑ Oracle(reference images) 0.774 0.885 Textual Inversion [6] 0.569 0.780 0.255 DreamBooth [25] 0.668 0.803 0.305 BLIP-Diffusion [17] 0.670 0.805 0.302 ELITE [30] 0.647 0.772 0.296 Re-Imagen [1] 0.600 0.740 0.270 BootPIG[22] 0.674 0.797 0.311 SSR-Encoder[41] 0.612 0.821 0.308 RealCustom++ [10, 19] 0.702 0.794 0.318 OmniGen [36] 0.693 0.801 0.315 OminiControl [27] 0.684 0.799 0.312 FLUX.1 IP-Adapter 0.582 0.820 0.288 UNO [34] 0.760 0.835 0.304 USO (Ours) 0.777 0.838 0.317

Table 5 Quantitative results for single-subject driven generation on Dreambench [25].

Scenarios Prompt

Subject/Identity Driven Generation (1) "The girl is riding a bike in the street."

(2) "The man is driving a car in the street." (3) "A sophisticated gentleman exuding confidence. He is dressed in a 1990s brown plaid jacket with a high collar, paired with a dark grey turtleneck. His trousers are tailored and charcoal in color, complemented by a sleek leather belt. The background showcases an elegant library with bookshelves, a marble fireplace, and warm lighting, creating a refined and cozy atmosphere. His relaxed posture and casual hand-in-pocket stance add to his composed and stylish demeanor" (4) "The woman is reading a book in a cafe."

Subject/Identity Driven Stylization (1) "Sketch style, a bowl with a mountain in the background."

(2) "Illustration style, a dog on the beach."

- (3) "Transform to Picasso’s style of work, Cubism."
- (4) "Ghibli style, The woman rides a deer in the forest."

Style Driven Generation (1)"A shark."

- (2) "Small boat in the lake."
- (3) "A beautiful woman."

(4) "The top chef is stir-frying in the kitchen." Multi-style Driven Generation (1) "A beautiful woman."

(2) "A duck, with words read "USO", "inspires creativity"." (3) "A man."

Style-subject Driven Generation (1) "" (Layout-preserved) (2) ""

(3) ""

Style-subject Driven Generation (1) "A toy in the jungle." (Layout-shifted) (2) "A cat on the beach."

(3) "The woman gave an impassioned speech on the podium."

Table 6 Text prompts used in Figure 1.

- F.2.2 Additional Results. We present additional qualitative results from USO:

- • From Figures 13 to 16, USO demonstrates the ability to extract task-relevant content features while maintaining subject consistency across diverse textual prompts—capabilities that prior work typically treats as isolated tasks (e.g., subject-driven generation, instruction-based stylized editing, and identity preservation).
- • In Figures 17 and 18, USO exhibits high stylistic fidelity, capturing both fine-grained characteristics (e.g., brushwork and material textures) and abstract artistic styles—far beyond simple color transfer.
- • In Figures 19 and 20, USO freely combines arbitrary subjects with arbitrary styles, supporting both layout-preserving and layout-shifting generations.

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

(1) A dog in the jungle. (2) A dog on a cobblestone street. (3) A dog on the beach. (4) A dog on top of a wooden floor. (5) A dog on top of fabric. (6) Transform the style of image into Studio Ghibli anime style. (7) Oil painting style, a dog with a house in the background. (8) Pixel style, a dog on a cobblestone street. (9) Retro comic style, a dog in the jungle.

###### Figure 13 More results on subject-driven generation.

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

(1) This man is surfing, with the waves behind him chasing after him. (2) Handsome man is playing Piano. (3) This man in suit was playing the violin on the stage when a beam of light shone upon him. (4) This man is holding a cat in the garden. (5) This man stood on the moon and made a "yeah" sign, with a miniature of the Earth behind him. (6) The boy is reading a book in the coffe. (7) This man is playing football on the playground under the setting sun. (8) Handsome man in the city. (9) This man is in the water, with fish circling around him.

[Figure 296]

###### Figure 14 More results on subject-driven generation.

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

(1) The man is reading a book in a cafe. (2) The man carried a backpack with a kitten inside. (3) A man in a silver sequin jacket dances in a club, strobe lights bouncing off his coat like. (4) A man fixes a bike at dusk, wrench shining in orange twilight. (5) This man was walking on the street at night, with the blurry neon lights behind him reading "USO”.

[Figure 306]

- (6) Sketch style, the man is walking with a dog, on the path in the park.
- (7) Pixel style, the man in flower shops carefully match bouquets, conveying beautiful emotions and blessings with flowers. (8) Lego building block wind, the man is reading a book in a cafe. (9) Studio Ghibli anime style, The man gave an impassioned speech on the podium.

###### Figure 15 More results on identity-driven generation.

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

(1) The woman crouched down in the garden and carefully trimmed the flower branches. (2) The woman is reading a book in a cafe. (3) A woman in a black leather jacket at night, streetlights streaking past like gold lines, her jacket collar flipping to catch cool blue neon. (4) A woman is mixing paint in a sunny art studio. (5) This woman writes on the blackboard, side view, the blackboard blurs "USO inspires creativity”. (6) Retro comic style, the woman is walking in a retro alley, with the sky drizzling and the raindrops clearly visible. (7) Pixel style, the woman crouched down in the garden and carefully trimmed the flower branches. (8) 3D Cartoon Style, the woman rides a deer in the forest. (9) Studio Ghibli anime style, the woman gave an impassioned speech on the podium.

[Figure 316]

###### Figure 16 More results on identity-driven generation.

[Figure 317]

|Reference Images<br><br>Text Prompts|
|---|
|A child standing beside a huge cat<br><br>A cat sleeping on a chair.<br><br>A villa on the coast.<br><br>A beautiful woman.<br><br>Handsome boy.<br><br>A canvas bag.<br><br>A duck.|

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

|Reference Images<br><br>Text Prompts|
|---|
|The cat chased the butterfly in the snow.<br><br>Farmhouse.<br><br>Llighthouse.<br><br>A tulip.<br><br>The boy stands under a banyan tree, with an endless field behind him.<br><br>The top chef is stirfrying in the kitchen.<br><br>The bag was placed on the mall shelf.|

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

|Content Reference<br><br>Style Reference|
|---|

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

###### Figure 19 More results on style-subject-driven generation. We set prompt to empty for layout-preserved generation.

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

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

- (1) “”
- (2) The woman in flower shops carefully match bouquets, conveying beautiful emotions and blessings with flowers.
- (3) The woman is reading a book in a cafe.
- (4) The woman gave an impassioned speech on the podium.
- (5) The woman with a mountain in the background.
- (6) The woman on the beach.
- (7) Night fell and the woman stood under the street lamp.
- (8) This woman is holding a cat.

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

- Figure 20 More results on style-subject-driven generation. USO supports any subject combined with any style in any scenario.

