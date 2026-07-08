# arXiv:2503.07334v4[cs.CV]14Nov2025

## Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment

### Xing Xie1,2*, Jiawei Liu1*, Ziyue Lin3, Huijie Fan1†, Zhi Han1, Yandong Tang1, Liangqiong Qu3†

1State Key Laboratory of Robotics and Intelligent Systems, Shenyang Institute of Automation, Chinese Academy of Sciences. 2University of Chinese Academy of Sciences. 3School of Computing and Data Science, The University of Hong Kong. {xiexing, liujiawei, fanhuijie, hanzhi, ytang}@sia.cn, ziyue lin@connect.hku.hk, liangqqu@hku.hk

###### Abstract

We present Autoregressive Representation Alignment (ARRA), a new training framework that unlocks globalcoherent text-to-image generation in autoregressive LLMs without architectural modifications. Different from prior works that require complex architectural redesigns, ARRA aligns LLM’s hidden states with visual representations from external visual foundational models via a global visual alignment loss and a hybrid token, <HYBNEXT>. This token enforces dual constraints: local next-token prediction and global semantic distillation, enabling LLMs to implicitly learn spatial and contextual coherence while retaining their original autoregressive paradigm. Extensive experiments validate ARRA’s plug-and-play versatility. When training T2I LLMs from scratch, ARRA reduces FID by 16.6% (ImageNet), 12.0% (LAION-COCO) for autoregressive LLMs like LlamaGen, without modifying original architecture and inference mechanism. For training from text-generation-only LLMs, ARRA reduces FID by 25.5% (MIMIC-CXR), 8.8% (DeepEyeNet) for advanced LLMs like Chameleon. For domain adaptation, ARRA aligns general-purpose LLMs with specialized models (e.g., BioMedCLIP), achieving an 18.6% FID reduction over direct fine-tuning on medical imaging (MIMIC-CXR). These results demonstrate that training objective redesign, rather than architectural modifications, can resolve cross-modal global coherence challenges. ARRA offers a complementary paradigm for advancing autoregressive models. The code is available at https://github.com/HKU-HealthAI/ARRA.

### 1 Introduction

Large language models (LLMs) (Achiam et al. 2023; Yang et al. 2025; Guo et al. 2025) and subsequent multimodal LLMs (MLLMs) (Team et al. 2023; Wang et al. 2024; Liu et al. 2024b; Zhu et al. 2025) have revolutionized the field of generative AI. These models, built on the autoregressive (AR) paradigm, show remarkable scalability and generalization capabilities in complex understanding tasks through a simple yet powerful next-token prediction framework.

Inspired by the success of LLMs, some researchers seek to replicate autoregressive “next-token prediction” paradigm

*These authors contributed equally. †Corresponding author

Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

for text-to-image generation. This paradigm is directly adopted in DALL·E (Ramesh et al. 2021), Parti (Yu et al. 2022), and LlamaGen (Sun et al. 2024a) for image generation, where images are treated as sequences of discrete tokens. However, although “next-token prediction” paradigm excels in language tasks, where local dependencies naturally align with sequential structure, it struggles to bridge the significant cross-modal gap between language and images. As shown in Fig. 1(c), optimizing only for local nexttoken prediction forces the model to focus on isolated tokenlevel features, neglecting the global coherence required for spatially structured visual content. This sometimes leads to fragmented parts on generated images, such as misaligned ribs in X-rays, where fine-grained details fail to harmonize into a unified whole. It can also cause semantic mismatches, as shown in Fig. 1(e), where global information is not maintained, leading to inconsistencies in the generated images.

Recognizing this limitation, recent efforts aim to inject global constraints into autoregressive frameworks to fully unlock the potential of LLMs in image generation (Zhou et al. 2024; Xie et al. 2024; Tian et al. 2024). Bidirectional attention mechanisms are introduced in Transfusion (Zhou et al. 2024) and Show-O (Xie et al. 2024) to model global image structure through patch diffusion and mask token modeling, respectively. These methods achieve promising results in generating high-quality images and demonstrate the potential of LLMs for multimodal generation. However, they rely on architectural modifications, such as cross-modal attention layers or grafted diffusion modules. While effective, such adaptations often deviate from standard LLM frameworks, limiting their compatibility with pretrained LLMs that excel under pure autoregressive paradigms. For instance, repurposing an off-the-shelf LLM for text-to-image generation would require retraining these modified components, losing benefits of existing scaling laws and generalization capabilities. This practical constraint raises a critical question: Can we unlock the full potential of LLMs for text-to-image generation without altering the original architecture or inference mechanism?

We address this by proposing Autoregressive Representation Alignment (ARRA), a novel training framework that redefines how LLMs learn text-to-image generation. Different from prior works that modify architectures (e.g., adding attention layers or diffusion modules), ARRA preserves the

[Figure 1]

[Figure 2]

- (a)
- (b)

- (c) semantic visual consistency

[Figure 3]

[Figure 4]

- (d)
- (e)
- (f)

NTP-based LLM

[Figure 5]

[Figure 6]

Language

| | |
|---|---|
|MixedAuto-Regr|Modal essive LM|
| | |

| |
|---|

Vision

Small/trace right pleural effusion. Mild enlargement of the cardiac silhouette.

Very large new left-sided pleural effusion.

ARRA

[Figure 7]

[Figure 8]

Language

representation

External

Mixed-Modal Auto-Regressive LM

Vision

[Figure 9]

semantic visual consistency

- Figure 1: ARRA enables high-quality text-to-image generation through a redefined training objective that promotes global coherence. (a)(c) Traditional next-token prediction (NTP)-based LLMs rely solely on the autoregressive loss (AR loss) of the next token <NEXT> for local constraints. (b)(d) ARRA constructs a hybrid token <HYBNEXT>, which is aligned by introducing external global visual representation, ensuring that <HYBNEXT> is constrained both locally by the AR loss and globally by the global visual alignment loss (GVA loss). (e)(f) ARRA demonstrates advantages in semantic consistency and visual continuity.

original LLM framework while injecting global constraints directly into the training objective. Our key insight is simple: global coherence does not require architectural complexity; it can instead be achieved through a redefined training paradigm. Specifically, ARRA augments the standard autoregressive loss with a global visual alignment loss that aligns the LLM’s latent representations with semantic guidance from pretrained foundational models (Fig. 2). To bridge local and global learning, we introduce a hybrid token, <HYBNEXT>, which serves as a bidirectional anchor. Locally, it predicts the next token via standard codebook indices. Globally, its latent embedding aligns with compressed visual features extracted from external models (e.g., BioMedCLIP (Zhang et al. 2023) or MedSAM (Ma et al. 2024)) via our novel global visual alignment loss. By distilling rich semantic features (e.g., spatial relationships, object coherence) from external models into the <HYBNEXT> token during training, ARRA enables autoregressive sequences to implicitly learn global structure. More crucially, this alignment occurs only during training, leaving the LLM’s inference process untouched and preserving its inference-time efficiency.

Our experiments demonstrate the versatility of ARRA framework for both natural and medical image generation tasks, achieving improvements without architectural modifications. ARRA supports three key capabilities on AR LLMs: (1) ARRA enhances training T2I LLMs from scratch. Applied to LlamaGen (Sun et al. 2024a) with varying parameter scales, it consistently improves generation performance and exhibits strong scalability. (2) ARRA effectively transforms pretrained text-generation-only LLMs into T2I generators. When integrated into LLMs without image generation capabilities, such as Chameleon (Team 2024), ARRA yields steady improvements. (3) ARRA facilitates adaptation of general-purpose generative models to special domains. By

integrating domain-specific priors (e.g., BioMedCLIP, MedSAM) into LLMs with image generation capabilities, such as Lumina-mGPT (Liu et al. 2024a), ARRA substantially outperforms direct fine-tuning. These capabilities confirm the plug-and-play flexibility of ARRA framework.

The main contributions are summarized below:

- (i) We propose Autoregressive Representation Alignment,

a novel training framework that redefines how LLMs learn text-to-image generation by decoupling global structure learning from model design. By aligning training objective with external representations, ARRA resolves local dependency limitations in LLMs while retaining original architectures and inference efficiency.

- (ii) We introduce the <HYBNEXT> token, a novel mecha-

nism that bridges local next-token prediction with global semantic alignment via distillation from external models (e.g., BioMedCLIP or MedSAM), enabling implicit learning of spatial and contextual relationships.

- (iii) We provide a detailed experimental analysis, offer-

ing comprehensive insights into the selection of alignment tokens, aggregation strategy, and external representations. These findings serve as a practical guide for the effective utilization of representation alignment.

- (iv) ARRA offers plug-and-play flexibility, facilitating

training T2I LLMs from scratch, transforming pretrained text-generation-only LLMs into T2I generators, and adapting general generative models to specific domains, all without architectural modifications. These capabilities are validated on both natural and medical image generation tasks using advanced AR models.

### 2 Related Work

###### 2.1 Visual Generation Models

Diffusions. The success of diffusion models has revolutionized the image generation paradigm (Qu et al. 2015, 2019;

###### Text Prompt Representation Extraction

###### VQ-Tokenizer

|66|19|5|43|
|---|---|---|---|
|79|85|34|28|
|27|8|10|25|

[Figure 10]

[Figure 11]

A bouquet of artificial peonies in shades of pink and red, arranged in a

Quantize

VQ Encoder

Text Tokenizer

I

decorative vase.

[Figure 12]

ℰ

I q

Visual Encoder

ℰ𝐹

###### HYB NEXT

###### Hybrid Next token

19 5 23 54 37 66 19 5 43

48 BOI

fF

[Figure 13]

Transformer Block

Autoregressive

Rep.Align.

| | | | | | | |
|---|---|---|---|---|---|---|

[Figure 14]

Projection

Mixed-Modal

#### ……

Hidden states

Layer

Feature

Auto-Regressive LM

i fL

fA

Aggregation

Transformer Block

fGF

[Figure 15]

[Figure 16]

19 5 23 54 37 48 BOI

66 19 5 43 79

AR

GVA

- Figure 2: Proposed ARRA Framework. We define the next token predicted in the autoregressive sequence as the “HYBRID next token”, denoted as <HYBNEXT>. During training, <HYBNEXT> is constrained not only locally by the autoregressive loss

LAR from the next token prediction and LLM codebook matching, but also globally through visual alignment loss LGVA, which modulates its hidden states using externally well-trained representations. We extract visual representations from a pretrained foundational model and further aggregate these features to obtain semantically enriched representations for alignment.

###### 2.2 LLMs for Text-to-Image Generation

Rombach et al. 2022a; Liu et al. 2024c; Xie et al. 2025). DiTs (Peebles and Xie 2023) show strong scalability by replacing or integrating U-Net with Transformers. This inspires subsequent models such as SD3 (Esser et al. 2024), Imagen3 (Baldridge et al. 2024), which achieve new stateof-the-art performance in image generation. Recent work REPA (Yu et al. 2025) explores enhancing diffusion models with external representations. It utilizes patch-wise representations alignment, matching each patch-level hidden state of the diffusion transformer with corresponding patch tokens from an external encoder. This alignment improves generation performance without modifying architecture. However, REPA’s strategy is not directly compatible with AR models. AR models generate image tokens sequentially and do not produce all patch tokens simultaneously during training, making patch-wise alignment infeasible. Different from this, ARRA introduces a novel hybrid token to bridge local nexttoken prediction and global alignment, enabling effective integration of the alignment mechanism into AR architectures. For further discussion, please refer to Appendix 2.

Currently, researchers pay attention to LLM-based text-toimage generation models, aiming to replicate the success of LLMs in language tasks. Early works leverage diffusion models as tools to extend LLMs (Dong et al. 2024; Ge et al. 2024; Sun et al. 2024b). These works utilize LLMs as feature extractors to guide diffusion generators for visual generation. Such models are complex to design and do not fully unleash the generative potential of the LLM for visual generation. Recent works (Sun et al. 2024a; Team 2024; Lu et al. 2024; Liu et al. 2024a) attempt to unify text and image modeling within a single LLM, enhancing generation performance through tokenizer optimization (Sun et al. 2024a), early fusion modeling (Team 2024), and flexible resolution modeling (Liu et al. 2024a). These works discretize both text and images into tokens, which are then fed into the LLM for sequence modeling by next token prediction. However, the local constraints provided by next-token prediction struggle to bridge the cross-modal gap between language and images.

### 3 Proposed Method

Autoregressive models. Early pioneering works, VQ-VAE (Van Den Oord, Vinyals et al. 2017), VQ-GAN (Esser, Rombach, and Ommer 2021) and DALL-E (Ramesh et al. 2021), demonstrated the potential of AR models for image generation. Subsequent works such as RQ-Transformer (Lee et al. 2022) also follows a raster-scan manner and enhances image generation performance through extra scales or stacked codes. Recent works achieve performance comparable to diffusion models by employing scale modeling (Tian et al. 2024) and eliminating vector quantization (Li et al. 2024a; Fan et al. 2025). Additionally, masked prediction autoregressive models (Chang et al. 2022; Li et al. 2024b) employ BERT-like (Devlin et al. 2019) masked prediction modeling, improving generation efficiency and quality.

We aim to achieve high-quality image generation without altering the next-token prediction paradigm of LLMs. We argue that the inherent cross-domain gaps in LLMs pose significant challenges for image generation tasks when the model lacks the ability to learn global features. To address this, we propose an Autoregressive Representation Alignment (ARRA) framework (see Fig. 2), which leverages the inherent representation capabilities of well-pretrained foundation models to facilitate the training of complex text-toimage autoregressive generation. Our framework is applied only during training, without affecting inference. It can enable the generation of high-quality images with exceptional semantic consistency in a cost-effective manner.

###### 3.1 Overview

Our goal is to train an autoregressive model Mθ leveraging representations derived from an external foundation visual

encoder EF. Mθ takes a text prompt T as input and generates a target image I. During training, T and I are first tok-

enized into token sequences sT and sI, respectively. These token sequences are then used to train the transformer-based

autoregressive model Mθ. Meanwhile, foundation visual encoder EF encodes I into the global visual representation fGF, which is used to align with the feature fA extracted from xt by Mθ. During image generation, the alignment module is removed, and image tokens are generated by Mθ through next-token prediction. Finally, the output image tokens are decoded into pixel space by an image decoder to produce the target image I. We describe the autoregressive modeling process in Section 3.2 and detail our autoregressive representation alignment framework in Section 3.3.

###### 3.2 Modeling via Next-Token Prediction The autoregressive architecture comprises two core compo-

nents: (1) A transformer-based autoregressive model Mθ for probabilistic modeling of token sequences. (2) A VQ-based model (Esser, Rombach, and Ommer 2021) with encoder E, quantizer Q and decoder D for transformation between image pixels and discrete token sequences. Formulation involves the following two parts:

Tokenization. In order to apply the next-token prediction modeling in the image domain, it is first required to convert the continuous 2D image pixels into discrete sequences. This process consists of two steps: (1) 2D image pixels to 2D image tokens, and (2) 2D image tokens to a 1D token sequences. Specifically, given a image I ∈ RH×W×3, we first obtain the image feature map f = E(I) ∈ Rh×w×d, where h = H/c, w = W/c, d is the dimension of the codes, c denotes a compression factor. Subsequently, we convert f into discrete tokens by q = Q(f) ∈ Zh×w, where the quantizer Q(·) maps each vector f(i,j) in the image feature map f to the code index q(i,j) of its nearest vector z(i,j) in the codebook Z. The image tokens q are then reshaped into a 1D token sequence sI = {xI1,xI2,xI3,...,xIn} with a length of h · w , arranged according to the raster scan order.

For text prompt T, we obtain the discrete sequence

through sT = T (T) = {xT1 ,xT2 ,xT3 ,...,xTn}, where T (·) denotes a text tokenizer.

Next token prediction modeling. We combine text sequences sT and image sequences sI to obtain discrete tokens x = {x1,x2,x3,...,xn}, where xn is an integer from a tokenizer’s vocabulary V . The next-token prediction paradigm posits the probability of current token xt depends only on its prefix (x1,x2,x3,...,xt−1). The likelihood of sequence modeling can be expressed as:

n

p(xt|x1,x2,...,xt−1). (1)

p(x) =

t=1

The autoregressive model Mθ formulates the generative task as predicting the distribution of the next token and opti-

mizes the likelihood pθ(x) through cross-entropy (CE) loss:

[−log pθ(xt|x<t)]. (2)

LAR(θ) = Ex

t

During training, the autoregressive model Mθ relies on the previous tokens x<t to predict the next token xt, where the hidden state of xt in the i-th layer of Mθ is denoted by fLi . In the last layer, the hidden state fL−1 is then passed through the LLM head to compute the probability distribution pθ for xt. In the original autoregressive model, pθ is constrained only by the local context of a single token (i.e., xt), lacking the ability to capture global information. This limitation restricts model’s capacity to capture complex cross-modal relationships. To address this problem, we propose Autoregressive Representation Alignment.

###### 3.3 Autoregressive Representation Alignment

We align the visual representations extracted from the pretrained foundational model with LLM’s hidden states and investigate impact of different alignment strategies. The goal of alignment is to enable the hidden state of autoregressive transformer to acquire external global representations, providing meaningful guidance for reconstructing image.

Pre-trained Visual Representation Extraction. Let EF be a pretrained foundation model’s visual encoder and I be a target image. We encode I as a visual representation by fF = EF(I) ∈ RN×D, where N,D denotes the embedding length and dimension of fF. fF is aggregated to the global visual representation fGF, i.e., fGF = agg(fF) ∈ R1×D, where agg(·) denotes a feature aggregation operation. This aggregation operation fully extracts global information in the features and facilitates alignment with the hidden states of autoregressive model. For CLIP series, inspired by (Raghu et al. 2021), we use the <CLS> token representation from the Transformer-based visual encoder as global visual representation. For SAM series, which lack a <CLS> token, we instead apply average pooling over all patch features for feature aggregation operation.

Hybrid Next Token. We define the next token predicted by LLM sequence in our framework as “HYBRID next token”, denoted as <HYBNEXT>. Unlike a “locally constrained token” in previous autoregressive models that are solely constrained by the LLM codebook, our <HYBNEXT> can fully incorporate external, well-trained global visual representations, making it a “globally and locally constrained token”.

Global Visual Representation Alignment. We obtain the hidden state fLi of <HYBNEXT> from the autoregressive model Mθ. The hidden state fLi is converted to fA ∈ R1×D by a projection layer Aϕ to align with the global visual representation fGF ∈ R1×D, i.e., fA = Aϕ(fLi ), where Aϕ is a two-layer MLP. Representation alignment is achieved through a Global Visual Alignment loss LGVA, which maximizes the similarity between the projected feature fA and the global visual representation fGF:

LGVA(θ,ϕ) = sim(fA,fGF). (3) where sim(·,·) denotes cosine similarity loss. This alignment enables the <HYBNEXT> token to learn global visual representation, bridging the cross-modal gap and making the token prediction process more reliable.

Therefore, the autoregressive model can be jointly optimized through the following composite loss function:

LARRA(θ,ϕ) = LAR(θ) + λLGVA(θ,ϕ). (4)

λ serves as a balancing hyperparameter that controls the relative importance of the alignment objective. Experimentally, we set λ = 1.

###### 3.4 Versatile ARRA for Diverse Scenarios

Our ARRA framework is flexible and plug-and-play, supporting different training scenarios and LLM frameworks. Therefore, we provide three representative model variants:

- (1) ARRA-Base. It trains an LLM from scratch with ran-

dom initialization, supporting settings where no pretrained models are available and showcasing ARRA’s ability to learn multimodal alignment from the ground up.

- (2) ARRA. It initializes with a pretrained LLM that has

strong text generation capabilities, enabling efficient extension to text-to-image tasks with text-generation-only LLMs.

- (3) ARRA-Adapt. It builds on pretrained LLMs with

both text and image generation capabilities, allowing adaptation to specialized domains such as medical imaging by leveraging special-domain priors.

- 4 Experimental Analysis and Results

We first perform a comprehensive component analysis of the ARRA framework to investigate how different design choices, such as alignment mechanism, feature aggregation strategy, and external encoder selection, impact alignment performance. Based on analysis, we finalize the ARRA framework and compare the three model variants, ARRABase, ARRA, and ARRA-Adapt, with advanced generative models to evaluate their performance and adaptability.

###### 4.1 Experimental Setup

Datasets. We evaluate our model on both natural and medical image datasets. For natural images, we conduct evaluations on the ImageNet (Deng et al. 2009) dataset and a 2.4M high-quality subset of LAION-COCO (Schuhmann et al. 2022). For medical imaging, we use the MIMIC-CXR (Johnson et al. 2019) and DeepEyeNet (Huang et al. 2021) datasets. Detailed preprocessing is provided in Appendix 3. Implementation Details. We adopt LlamaGen (Sun et al. 2024a), Chameleon 7B (Team 2024) and Lumina-mGPT 7B (Liu et al. 2024a) as respective baseline models for the ARRA-Base, ARRA, and ARRA-Adapt model variants, respectively, to verify the effectiveness of our framework.

###### 4.2 Component Analysis of Alignment

To better understand the core design principles of ARRA and build an effective model, we analyze how different alignment component designs affect the effectiveness of representation alignment in autoregressive generation. We focus on the following key questions:

- • Alignment mechanism: Does token-level alignment (our proposed <HYBNEXT>) outperform fixed-position alignment (<REP>) in preserving global visual constraints during autoregressive generation? (Table 1)
- • Feature aggregation strategy: How do different types of features extracted from the same visual encoder affect the generation performance? (Table 2)

• Visual encoder selection on alignment: How does external encoder selection (general vs. specialized and crossmodal vs. vision-only) impact generation? (Table 3)

Target token FID ↓ MS-SSIM ↑ CLIP-Score ↑ <REP> 4.85 0.410 0.4527

- <HYBNEXT> 4.15 0.422 0.4576 w/o align. 5.10 0.401 0.4518

<REP> 6.26 0.401 0.4499

- <HYBNEXT> 5.30 0.405 0.4532 w/o align. 7.11 0.383 0.4460

- Table 1: Impact of different alignment mechanism selection on MIMIC-CXR. Gray represents ARRA-adapt model, and white represents ARRA model.

Alignment mechanism. We compare two strategies for integrating visual representations: (1) aligning features to a fixed <REP> token at the start of the generated sequence, and (2) aligning to the hidden state of <HYBNEXT>, a hybrid token interleaved at every generation step. As shown in Table 1, <HYBNEXT> yields superior performance. We argue that <HYBNEXT> allows for comprehensive traversal of every token during training sampling, ensuring that each token is effectively constrained by external global representations. In contrast, <REP> suffers from the “attention sink” (Liu et al. 2024a; Xiao et al. 2023) effect, where attention to the fixed token decays over long sequences, leading to degraded outputs. This leads to a key insight: Takeaway 1. Aligning visual representations to a hybrid token interleaved at each generation step <HYBNEXT> is more effective than using a fixed token <REP>, as it prevents attention decay and ensures consistent constraint by external representations.

CLS Avgpool FID ↓ MS-SSIM ↑ CLIP-Score ↑

✓ 5.30 0.405 0.4532 ✓ 6.56 0.387 0.4434 ✓ ✓ 5.93 0.385 0.4465

- Table 2: Impact of using different feature aggregation strategies on MIMIC-CXR with ARRA.

Feature aggregation strategy. To investigate how different types of features extracted from the visual encoder affect the generation performance, we understand three strategies for aggregating features from a single visual encoder: (1) the [CLS] token representation, (2) average pooling of all image patch representations, and (3) a concatenation of both representations. As shown in Table 2, the [CLS] token representation yields optimal performance. We attribute this superiority to the [CLS] token’s ability to aggregate global visual information through self-attention mechanisms. This ability provides a compact yet comprehensive representation for cross-modal alignment. This finding consistent with (Raghu et al. 2021), which establishes that [CLS] tokens capture global representations while patch tokens focus on local features. This global-local distinction suggests that autoregressive image generation models particularly benefit

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

###### ARRA-Base-343MARRA-Base-111MLlamaGen-111M

###### ARRA-Base-3.1BLlamaGen-3.1BARRA-Base-3.1BLlamaGen-3.1B

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

A heart-shaped bowl filled with heart-healthy foods like broccoli, nuts, and blueberries, viewed from above.

Pan-fried veal chop with a lemon slice and thyme garnish, served with a side salad and creamy polenta.

A basket filled with blue flowers, including daisies and other blooms, sits on a tablecloth.

A portrait painting of an elderly man with a long beard, likely self-portrait by Claude Monet.

Golden Retriever Goldfish Tobacco shop White Fox

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

LlamaGen-343M

A two-tiered cake adorned with fresh flowers, including red roses and purple irises, sits on a wooden stand.

A blue wedding arch adorned with white and pink flowers, set against a backdrop of greenery and a serene ocean view.

A charming country cottage with a thatched roof is nestled amidst a vibrant garden filled with colorful flowers.

A cup of latte with a heart-shaped latte art design on a wooden table.

Golden Retriever

Goldfish Tobacco shop White Fox

- Figure 3: ARRA-Base improves the generation of LlamaGen. Left are text-conditional image generation results on the LAIONCOCO dataset, and right are category-conditional image generation results on the Imagenet dataset.

from external global semantic representations as guidance. These results lead to a key insight: Takeaway 2. The [CLS] token representation in foundation models effectively aggregates global visual information, providing comprehensive guidance for cross-modal alignment.

demonstrate superior performance. Their cross-modal training helps bridge the gap between text and image modalities, enabling LLMs to learn what to generate (semantics) before how to generate (pixel details). Conversely, when fine-tuning pretrained LLMs with image-generation capabilities (ARRA-Adapt), domain-specific encoders like BioMedCLIP and MedSAM dominate. BioMedCLIP injects medical-specific semantics, while MedSAM provides structural priors (e.g., organ shapes) through its segmentationfocused features. These results lead to a key insight: Takeaway 3. When the LLM lacks image generation capabilities, cross-modal encoders are crucial for semantic grounding. However, for LLMs with image generation capabilities, domain-specific encoders are more effective, as they provide fine-grained features needed for domain-specific adaptation.

Target Rep. FID ↓ MS-SSIM ↑ CLIP-Score ↑

- BioMedCLIP (Zhang et al. 2023) 4.15 0.422 0.4576 Med-SAM (Ma et al. 2024) 4.08 0.398 0.4542

CLIP-L (Radford et al. 2021a) 4.63 0.407 0.4519 w/o align. 5.10 0.401 0.4518

- BioMedCLIP (Zhang et al. 2023) 5.30 0.405 0.4532 Med-SAM (Ma et al. 2024) 6.54 0.384 0.4465

CLIP-L (Radford et al. 2021a) 5.16 0.394 0.4450 w/o align. 7.11 0.383 0.4460

- Table 3: Impact of alignment with representation extracted from different encoders on MIMIC-CXR.

###### 4.3 Main Comparison Results

Based on analysis, we finalize ARRA framework with CLIP and BioMedCLIP for alignment on natural and medical datasets, respectively. We use [HYBNEXT] and [CLS] as our alignment strategy. We compare ARRA framework under the three proposed model variants (ARRA-Base, ARRA, ARRA-Adapt) to demonstrate its superiority and versatility. Facilitating training T2I LLMs from scratch: performance comparison of ARRA-Base. We evaluate the generation performance and model scalability of ARRA-Base variant, which is trained from scratch, on large-scale datasets. We first conduct experiments on ImageNet, using LlamaGen (Sun et al. 2024a) with 111M and 343M parameters as baselines. We then train models on a 2.4M

Visual encoder selection on alignment. To evaluate how visual encoder selection impacts cross-modal alignment and generation quality, we conduct experiments with three encoders: BioMedCLIP (Zhang et al. 2023) (domainspecific cross-modal encoders), CLIP (Radford et al. 2021b) (general-purpose cross-modal encoders), and Med-SAM (Ma et al. 2024) (domain-specific pure visual encoders). As shown in Table 3, all pretrained encoders improve generation performance compared to without alignment. This improvement arises from global constraints imposed by external representations, which regularize the autoregressive generation process. Notably, for training from a textgeneration-only LLM (ARRA), BioMedCLIP and CLIP

ImageNet LAION-COCO Model

343M 775M 1.4B 3.1B FID ↓ IS ↑ FID ↓ IS ↑ (FID, CLIP-Score) (FID, CLIP-Score) (FID, CLIP-Score) (FID, CLIP-Score)

111M 343M

Model

LlamaGen 5.47 193.6 4.33 286.6 LlamaGen (12.78, 0.2470) (11.79, 0.2512) (11.57, 0.2510) (11.88, 0.2605) ARRA-Base 5.08 201.2 3.61 262.6 ARRA-Base (11.67, 0.2500) (10.81, 0.2571) (10.50, 0.2589) (10.45, 0.2615)

- Table 4: Quantitative performance comparison of ARRA-Base and LlamaGen models with different scales on ImageNet datasets and 2.4M subset of LAION-COCO datasets, respectively.

Chest-Xray: MIMIC-CXR Fundus: DeepEyeNet

|Method FID ↓ IS ↑ MS-SSIM ↑ CLIP-Score ↑|FID ↓ MS-SSIM ↑ CLIP-Score ↑|
|---|---|
|Original SD V2-1(Rombach et al. 2022b) 71.68 2.365 0.128 0.2333<br><br>DreamBooth SD (Ruiz et al. 2023) 60.40 2.269 0.270 0.3693 MINIM (Wang et al. 2025) 15.62 2.468 0.317 0.4423 UniXGen (Lee et al. 2024a) 30.75 2.437 0.361 0.4128<br><br>LLM-CXR (Lee et al. 2024b) 5.88 2.134 0.395 0.4374 Chameleon (Team 2024) 7.11 2.498 0.383 0.4460|166.45 0.141 0.2164 80.70 0.352 0.3061 59.71 0.333 0.2862<br><br>- - -<br><br>- - -<br><br><br>38.37 0.341 0.3234|
|ARRA 5.30 2.587 0.405 0.4532 ARRA-Adapt 4.15 2.746 0.422 0.4576|35.01 0.376 0.3354 34.70 0.392 0.3389<br><br>|

- Table 5: Compare different methods with ARRA and ARRA-Adapt on MIMIC-CXR and DeepEyeNet datasets, respectively.

[Figure 35]

- Figure 4: Visual comparison of different methods with ARRA and ARRA-Adapt on the MIMIC-CXR dataset.

high-quality subset of LAION-COCO and employ LlamaGen variants of increasing sizes (343M to 3.1B parameters) as baselines. As shown in Table 4 and Fig. 3, ARRABase outperforms the baseline model across all scales. The FID decreased by 16.6% and 12.0% on the ImageNet and LAION-COCO datasets, respectively. Meanwhile, as the model size increased, the performance remained steadily improved (FID decreased from 11.67 to 10.45), demonstrating ARRA maintains strong scalability of autoregressive model. These results lead to a key insight: Takeaway 4. ARRABase enables efficient training of T2I LLMs from scratch, while preserving strong model scalability.

Boosting pretrained LLMs for T2I generation and domain adaptation: performance of ARRA and ARRAAdapt. We evaluate ARRA and ARRA-Adapt, initialized with text-generation-only and general image generation pretrained LLMs, respectively, on the MIMIC-CXR and Deep-

EyeNet datasets. Both variants are benchmarked against state-of-the-art diffusion models (Stable Diffusion (Rombach et al. 2022b), DreamBooth (Ruiz et al. 2023), MINIM (Wang et al. 2025)) and autoregressive models (LLM-CXR (Lee et al. 2024b), UniXGen (Lee et al. 2024a), Chameleon (Team 2024)). As shown in Table 5, our model achieves superior performance in both visual quality and semantic alignment. Compared to the chameleon, ARRA achieves a 25.5% and 8.8% reduction in FID on the MIMIC-CXR and DeepEyeNet datasets, respectively. On the MIMICCXR dataset, ARRA surpasses the best-performing baseline, LLM-CXR, reducing the FID from 5.88 to 5.30 and increasing the CLIP-Score from 0.4374 to 0.4532. The ARRA-Adapt variant delivers even stronger results, achieving an FID of 4.15 and a CLIP-Score of 0.4576. In addition, as shown in Fig. 4, our models demonstrate superior alignment with fine-grained clinical details, such as lesion location and severity. These results lead to a key insight: Takeaway 5. ARRA enables a more effective transformation of text-generated-only LLMs into T2I generators, while ARRAAdapt substantially improves domain adaptation and aligns general image-generation LLMs with specialized fields more effectively, both outperforming baseline approaches.

### 5 Conclusion

We propose Autoregressive Representation Alignment (ARRA), a framework that enhances autoregressive image generation by injecting external visual representations during training. This approach enriches global semantic understanding while maintaining the model’s original autoregressive paradigm during generation. Experiments on natural and medical image generation tasks demonstrate ARRA’s versatility, offering a cost-effective framework for training autoregressive text-to-image generation models. Our work bridges the gap between multimodal domains and provides novel insights into modeling unified multimodal generation.

### 6 Acknowledgments

This work was supported by the National Natural Science Foundation of China (62306253, 61873259), the Early Career Fund (27207025), the National Key Research and Development Program of China under Grant 2024YFB4707700, the National Natural Science Foundation of China under Grant U23A20343 and the Guangdong Natural Science Fund-General Program (2024A1515010233).

### References

Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Baldridge, J.; Bauer, J.; Bhutani, M.; Brichtova, N.; Bunner, A.; Chan, K.; Chen, Y.; Dieleman, S.; Du, Y.; Eaton-Rosen, Z.; et al. 2024. Imagen 3. arXiv preprint arXiv:2408.07009. Chang, H.; Zhang, H.; Jiang, L.; Liu, C.; and Freeman, W. T.

- 2022. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11315–11325. Cohen, J. P.; Viviano, J. D.; Bertin, P.; Morrison, P.; Torabian, P.; Guarrera, M.; Lungren, M. P.; Chaudhari, A.; Brooks, R.; Hashir, M.; et al. 2022. TorchXRayVision: A library of chest X-ray datasets and models. In International Conference on Medical Imaging with Deep Learning, 231–

249. PMLR.

Conover, M.; Hayes, M.; Mathur, A.; Xie, J.; Wan, J.; Shah, S.; Ghodsi, A.; Wendell, P.; Zaharia, M.; and Xin, R.

- 2023. Free Dolly: Introducing the World’s First Truly Open Instruction-Tuned LLM. Deng, J.; Dong, W.; Socher, R.; Li, L.-J.; Li, K.; and FeiFei, L. 2009. ImageNet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, 248–255. Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), 4171–4186. Dong, R.; Han, C.; Peng, Y.; Qi, Z.; Ge, Z.; Yang, J.; Zhao, L.; Sun, J.; Zhou, H.; Wei, H.; Kong, X.; Zhang, X.; Ma, K.; and Yi, L. 2024. DreamLLM: Synergistic Multimodal Comprehension and Creation. In The Twelfth International Conference on Learning Representations. Esser, P.; Kulal, S.; Blattmann, A.; Entezari, R.; M¨uller, J.; Saini, H.; Levi, Y.; Lorenz, D.; Sauer, A.; Boesel, F.; Podell, D.; Dockhorn, T.; English, Z.; Lacey, K.; Goodwin, A.; Marek, Y.; and Rombach, R. 2024. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. arXiv:2403.03206. Esser, P.; Rombach, R.; and Ommer, B. 2021. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 12873–12883.

Fan, L.; Li, T.; Qin, S.; Li, Y.; Sun, C.; Rubinstein, M.; Sun, D.; He, K.; and Tian, Y. 2025. Fluid: Scaling autoregressive text-to-image generative models with continuous tokens.

Fang, R.; Duan, C.; Wang, K.; Huang, L.; Li, H.; Yan, S.; Tian, H.; Zeng, X.; Zhao, R.; Dai, J.; et al. 2025. Got: Unleashing reasoning capability of multimodal large language model for visual generation and editing. arXiv preprint arXiv:2503.10639.

Ge, Y.; Zhao, S.; Zhu, J.; Ge, Y.; Yi, K.; Song, L.; Li, C.; Ding, X.; and Shan, Y. 2024. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396.

Guo, D.; Yang, D.; Zhang, H.; Song, J.; Zhang, R.; Xu, R.; Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Heusel, M.; Ramsauer, H.; Unterthiner, T.; Nessler, B.; and Hochreiter, S. 2017. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30.

Huang, J.-H.; Yang, C.-H. H.; Liu, F.; Tian, M.; Liu, Y.-C.; Wu, T.-W.; Lin, I.-H.; Wang, K.; Morikawa, H.; Chang, H.; Tegner, J.; and Worring, M. 2021. DeepOpht: Medical Report Generation for Retinal Images via Deep Models and Visual Explanation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2442–2452.

Johnson, A. E.; Pollard, T. J.; Greenbaum, N. R.; Lungren, M. P.; Deng, C.-y.; Peng, Y.; Lu, Z.; Mark, R. G.; Berkowitz, S. J.; and Horng, S. 2019. MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs. arXiv preprint arXiv:1901.07042.

Lee, D.; Kim, C.; Kim, S.; Cho, M.; and Han, W.-S. 2022. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11523–11532.

Lee, H.; Kim, W.; Kim, J.-H.; Kim, T.; Kim, J.; Sunwoo, L.; Choi, E.; et al. 2024a. Vision-Language Generative Model for View-Specific Chest X-ray Generation. In Conference on Health, Inference, and Learning, 280–296. PMLR.

Lee, S.; Kim, W. J.; Chang, J.; and Ye, J. C. 2024b. LLMCXR: instruction-finetuned LLM for CXR image understanding and generation. In 12th International Conference on Learning Representations, ICLR 2024.

- Li, T.; Tian, Y.; Li, H.; Deng, M.; and He, K. 2024a. Autoregressive Image Generation without Vector Quantization. arXiv preprint arXiv:2406.11838.
- Li, T.; Tian, Y.; Li, H.; Deng, M.; and He, K. 2024b. Autoregressive Image Generation without Vector Quantization. arXiv:2406.11838.

Liu, D.; Zhao, S.; Zhuo, L.; Lin, W.; Qiao, Y.; Li, H.; and Gao, P. 2024a. Lumina-mgpt: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining. arXiv preprint arXiv:2408.02657.

Liu, H.; Li, C.; Li, Y.; and Lee, Y. J. 2024b. Improved baselines with visual instruction tuning. In Proceedings of

the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 26296–26306. Liu, J.; Wang, Q.; Fan, H.; Wang, Y.; Tang, Y.; and Qu, L.

- 2024c. Residual denoising diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2773–2783. Lu, J.; Clark, C.; Lee, S.; Zhang, Z.; Khosla, S.; Marten, R.; Hoiem, D.; and Kembhavi, A. 2024. Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision Language Audio and Action. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 26439– 26455. Ma, J.; He, Y.; Li, F.; Han, L.; You, C.; and Wang, B. 2024. Segment Anything in Medical Images. Nature Communications, 15: 654. Peebles, W.; and Xie, S. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, 4195–4205. Qu, L.; Tian, J.; Han, Z.; and Tang, Y. 2015. Pixel-wise orthogonal decomposition for color illumination invariant and shadow-free image. Optics express, 23(3): 2220–2239. Qu, L.; Wang, S.; Yap, P.-T.; and Shen, D. 2019. Waveletbased semi-supervised adversarial learning for synthesizing realistic 7T from 3T MRI. In International conference on medical image computing and computer-assisted intervention, 786–794. Springer. Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; Krueger, G.; and Sutskever, I. 2021a. Learning Transferable Visual Models From Natural Language Supervision. arXiv:2103.00020. Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021b. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748–8763. PmLR. Raghu, M.; Unterthiner, T.; Kornblith, S.; Zhang, C.; and Dosovitskiy, A. 2021. Do vision transformers see like convolutional neural networks? Advances in neural information processing systems, 34: 12116–12128. Ramesh, A.; Pavlov, M.; Goh, G.; Gray, S.; Voss, C.; Radford, A.; Chen, M.; and Sutskever, I. 2021. Zero-shot text-toimage generation. In International conference on machine learning, 8821–8831. Pmlr. Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Om-

- mer, B. 2022a. High-Resolution Image Synthesis with Latent Diffusion Models. arXiv:2112.10752. Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Om-
- mer, B. 2022b. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 10684– 10695. Ruiz, N.; Li, Y.; Jampani, V.; Pritch, Y.; Rubinstein, M.; and Aberman, K. 2023. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 22500–22510.

Schuhmann, C.; Beaumont, R.; Vencu, R.; Gordon, C.; Wightman, R.; Cherti, M.; Coombes, T.; Katta, A.; Mullis, C.; Wortsman, M.; et al. 2022. Laion-5b: An open largescale dataset for training next generation image-text models. Advances in neural information processing systems, 35: 25278–25294.

- Sun, P.; Jiang, Y.; Chen, S.; Zhang, S.; Peng, B.; Luo, P.; and Yuan, Z. 2024a. Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation. arXiv:2406.06525.
- Sun, Q.; Cui, Y.; Zhang, X.; Zhang, F.; Yu, Q.; Wang, Y.; Rao, Y.; Liu, J.; Huang, T.; and Wang, X. 2024b. Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14398–14409.

Team, C. 2024. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818.

Team, G.; Anil, R.; Borgeaud, S.; Alayrac, J.-B.; Yu, J.; Soricut, R.; Schalkwyk, J.; Dai, A. M.; Hauth, A.; Millican, K.; et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Tian, K.; Jiang, Y.; Yuan, Z.; Peng, B.; and Wang, L. 2024. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv preprint arXiv:2404.02905. Van Den Oord, A.; Vinyals, O.; et al. 2017. Neural discrete representation learning. Advances in neural information processing systems, 30.

Wang, J.; Wang, K.; Yu, Y.; Lu, Y.; Xiao, W.; Sun, Z.; Liu, F.; Zou, Z.; Gao, Y.; Yang, L.; et al. 2025. Self-improving generative foundation model for synthetic medical image generation and clinical applications. Nature Medicine, 31(2): 609–617.

Wang, X.; Zhang, X.; Luo, Z.; Sun, Q.; Cui, Y.; Wang, J.; Zhang, F.; Wang, Y.; Li, Z.; Yu, Q.; et al. 2024. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869.

Xiao, G.; Tian, Y.; Chen, B.; Han, S.; and Lewis, M. 2023. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453.

Xie, J.; Mao, W.; Bai, Z.; Zhang, D. J.; Wang, W.; Lin, K. Q.; Gu, Y.; Chen, Z.; Yang, Z.; and Shou, M. Z. 2024. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528.

Xie, X.; Liu, J.; Fan, H.; Han, Z.; Tang, Y.; and Qu, L. 2025. DVG-Diffusion: Dual-View Guided Diffusion Model for CT Reconstruction from X-Rays. arXiv preprint arXiv:2503.17804.

Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Yu, J.; Xu, Y.; Koh, J. Y.; Luong, T.; Baid, G.; Wang, Z.; Vasudevan, V.; Ku, A.; Yang, Y.; Ayan, B. K.; Hutchinson, B.; Han, W.; Parekh, Z.; Li, X.; Zhang, H.; Baldridge, J.; and Wu, Y. 2022. Scaling Autoregressive Models for ContentRich Text-to-Image Generation. arXiv:2206.10789.

Yu, S.; Kwak, S.; Jang, H.; Jeong, J.; Huang, J.; Shin, J.; and Xie, S. 2025. Representation Alignment for Generation:

Training Diffusion Transformers Is Easier Than You Think. In International Conference on Learning Representations.

Zhang, S.; Xu, Y.; Usuyama, N.; Xu, H.; Bagga, J.; Tinn, R.; Preston, S.; Rao, R.; Wei, M.; Valluri, N.; et al. 2023. BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. arXiv preprint arXiv:2303.00915.

Zhou, C.; Yu, L.; Babu, A.; Tirumala, K.; Yasunaga, M.; Shamis, L.; Kahn, J.; Ma, X.; Zettlemoyer, L.; and Levy, O. 2024. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039.

Zhu, J.; Wang, W.; Chen, Z.; Liu, Z.; Ye, S.; Gu, L.; Tian, H.; Duan, Y.; Su, W.; Shao, J.; et al. 2025. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479.

### 1 Overview

Our supplementary materials include the following sections:

- • Discussion with related work (Section 2).
- • Additional implementation details about ARRA and other comparative methods in MIMIC-CXR and DeepEyeNet (Section 3);
- • Implementation details of component analysis of alignment. (Section 4);
- • Ablation study, including generalizability at different resolutions, choices of projection layers, regularization coefficients λ, training objectives and alignment depth (Section 5);
- • More visual results on the MIMIC-CXR dataset (Section 6);
- • Implementation details and more results on the LAIONCOCO and ImageNet datasets(Section 7);

### 2 Discussion with related work

Discussion with REPA (Yu et al. 2025): While both works leverage feature alignment concepts, ARRA presents fundamentally distinct contributions from REPA:

Different motivation: REPA resolves challenges in learning high-quality internal representations for diffusion models by injecting representations from external clean images. In sharp contrast, ARRA addresses a fundamental limitation of autoregressive LLMs in text-to-image generation: the inherent inability to capture global coherence. Unlike prior works requiring complex architectural redesigns to impose global constraints, ARRA demonstrates that such global coherence naturally emerges through our alignment-driven training paradigm, achieved by aligning LLM hidden states with visual representations from an external visual encoder. REPA not compatible with autoregressive models (AR): REPA utilizes patch-wise alignment, matching each patchlevel hidden states of the diffusion transformer with the patch token from an external encoder, i.e., all patches of DiTs ⇔ all patches of external encoder. But AR does not output all tokens of image patches during training, i.e., wo/ all patches of AR ⇎ all patches of external encoder. Thus, REPA is incompatible with AR. To make the feature alignment idea compatible with the design of AR, we thus design a novel hybrid token <HYBNEXT>, which serves as a bidirectional anchor: local next token prediction via standard codebook and global information injection via a feature alignment loss. Results in Sec. 4.2.1 validate the superiority of our <HYBNEXT> design.

### 3 Additional implementation details in MIMIC-CXR and DeepEyeNet

Datasets. For the MIMIC-CXR, we employ three distinct radiographic views: posteroanterior (PA), anteroposterior (AP), and lateral (LATENT). Input prompts are generated from the “impression” section of diagnostic reports, formatted as: “{view} view chest X-ray image, {impression}”. We select 221,238 pairs for training and 1,000 pairs for testing. DeepEyeNet (Huang et al. 2021) is a comprehensive fundus

dataset, containing more than 15k high-quality “text-image” pairs. In our experiments, we derive prompts from the “clinical-description” section and selectively choose 7,190 training pairs and 1,089 testing pairs. All images are preprocessed through center cropping to 512×512 pixels.

Evaluation Metrics. We employ three quantitative metrics to evaluate the proposed method: Fr´echet Inception Distance (FID), Multi-Scale Structural Similarity (MS-SSIM), and CLIP-Score. For X-ray image generation, FID is computed using features extracted from a CXR-pre-trained DenseNet121 (XRV) (Cohen et al. 2022), while for fundus images, we utilize features from an ImageNet-pre-trained Inception V3 (Heusel et al. 2017). The CLIP-Score metric is derived from the cosine similarity between image-text feature pairs encoded through BioMedCLIP (Zhang et al. 2023), a largescale medical vision-language model.

ARRA and ARRA-Adapt. All experiments on medical images are implemented using PyTorch on 4 NVIDIA A6000 Ada GPUs. We employ a batch size of 6 with a learning rate of 2e-5, optimized by AdamW (weight decay=0.05, β1=0.9, β2=0.95). To stabilize training, we apply z-loss regularization (Team 2024) with a weight of 1e-5. All data are pretokenized before training to increase throughput. The VQ tokenizer operates with a downsampling rate of 16, resulting in a 1024-token representation for each image.

Original SD V2-1. (Rombach et al. 2022b) Stablediffusion-2-1 is a diffusion model trained from natural images, and we obtain its official pre-training weights, which are used directly to generate images without training, to demonstrate its zero-shot generation capability.

DreamBooth SD. (Ruiz et al. 2023) We adapt Dreambooth to fine-tune the stable-diffusion-xl-base model, choosing “X-rays” and “fundus” as class identifiers [class noun] on the MIMIC-CXR and DeepEyesNet datasets, respectively, and supervised the fine-tuning using the full data.

MINIM. (Wang et al. 2025) MINIM is a medical image generation model based on stable-diffusion-v1-4 for full fine-tuning. We use its official implementation for training and generation.

UniXGen. (Lee et al. 2024a) The UniXGen model is a unified model for X-ray image understanding and generation, fully trained on the MIMIC-CXR dataset. We use its official implementation and obtain its pre-training weights for generation.

LLM-CXR. (Lee et al. 2024b) The LLM-CXR model is a unified model for X-ray image understanding and generation using dolly-v2-3b (Conover et al. 2023) as a baseline for full parameter fine-tuning training. We implement training and inference on the MIMIC-CXR dataset using its official implementation.

Chameleon. (Team 2024) Chameleon is a multimodal large language model trained on the natural image datasets. Its open-source weights are only for visual language understanding. We follow its official implementation and use 7B pre-training weights to fine-tune full parameters for training on the x-ray and fundus image dataset.

[Figure 36]

Figure 1: More visualizations on the chest X-ray generation task across the MIMIC-CXR datasets.

### 4 Implementation of component analysis

For the Alignment mechanism, we align the representations extracted from BioMedCLIP. For the Feature aggregation strategy, we align the representations extracted from the BioMedCLIP model to the hidden state of <HYBNEXT>. For the Visual encoder selection on alignment, we align the representations to the hidden state of <HYBNEXT>.

### 5 More Ablation study

Generalizability at different resolutions. We validate the generalization capability of the ARRA framework at resolutions of 256×256 and 512×512, as shown in Table 1. Experimental results demonstrate that ARRA can stably improve the generation performance of baseline models at different resolutions, demonstrating its resolution generalization ability.

Table 1: Generalizability at different resolutions.

Resolution Model FID ↓ MS-SSIM ↑ CLIP-Score ↑

256×256 Chameleon 9.42 0.369 0.4405 256×256 ARRA 6.11 0.388 0.4436 512×512 Chameleon 7.11 0.383 0.4460 512×512 ARRA 5.30 0.405 0.4532

Table 2: Compare performance of different alignment heads. MLP Maxpool FID ↓ MS-SSIM ↑ CLIP-Score ↑

###### ✓ 5.30 0.405 0.4532 ✓ 5.37 0.386 0.4375

Projection layer. We evaluate the performance of different projection layers, including a two-layer MLP and a direct pooling layer. As shown in Table 2, our experiments reveal that using a two-layer MLP outperforms the pooling layer, even when the projection layer is discarded during testing. We hypothesize that, compared to the non-trainable pooling layer, employing a trainable MLP as a ”soft link” during training allows the model to better learn information-rich and precise representations while adaptively filtering out irrelevant information.

Effect of λ. We also examine the effect of the regularization coefficient λ by training models with different coefficients from 0.5 to 2 and comparing the performance. As shown in Table 3, the optimal value is reached when the regularization coefficient λ = 1.

Training objective. We compare two alignment objectives: MSE loss and cosine similarity loss. As shown in Table 4, both achieve similar FID and MS-SSIM scores, but cosine similarity yields better semantic accuracy (higher CLIP-

Table 3: Ablation study for alignment loss coefficient λ.

λ FID ↓ MS-SSIM ↑ CLIP-Score ↑ 0.5 6.22 0.389 0.4468

- 0.8 5.38 0.399 0.4460

- 1 5.30 0.405 0.4532

1.5 5.87 0.394 0.4477

- 2 6.12 0.390 0.4479

Score). This advantage stems from its focus on directional similarity rather than absolute distances, making it more robust for cross-modal alignment tasks.

Table 4: Impact of different alignment objectives.

Objective FID ↓ MS-SSIM ↑ CLIP-Score ↑ Cos. sim. 5.30 0.405 0.4532

MSE 5.32 0.405 0.4454

Alignment depth. As shown in Table 5, we investigate the effects of representation alignment at different layers of the Transformer architecture. Our results demonstrate that alignment at various layers consistently enhances model performance, with early-layer alignment (e.g., layer 1) yielding optimal results. We argue that early-layer alignment provides sufficient global semantic guidance while allowing subsequent layers to focus on capturing high-frequency details, thereby establishing robust representations for image generation.

These results lead to a key insight: Takeaway 4. Performing representation alignment at earlier LLM layers provides sufficient global semantic guidance while allowing deeper layers to focus on capturing high-frequency details.

Table 5: Impact of aligning representation to different layers on MIMIC-CXR. Gray part represents the ARRA-adapt, and the white part represents the ARRA.

Layer Index FID ↓ MS-SSIM ↑ CLIP-Score ↑

- layer1 4.15 0.422 0.4576 layer5 4.58 0.421 0.4587

- layer20 4.46 0.413 0.4563

- layer30 4.51 0.412 0.4537

- layer31 4.53 0.418 0.4606 w/o align. 5.10 0.401 0.4518

layer1 5.30 0.405 0.4532 layer5 6.21 0.385 0.4467

- layer20 5.70 0.404 0.4500

- layer30 5.04 0.398 0.4478
- layer31 5.71 0.373 0.4500 w/o align. 7.11 0.383 0.4460

### 6 More visualizations on MIMIC-CXR

We present additional visual results from experiments on MIMIC-CXR, including:

• Visual comparisons of ARRA with other methods on Xray image generation tasks (Fig. 1);

[Figure 37]

- Figure 2: Visualization of alignment with features extracted from different encoders.

[Figure 38]

- Figure 3: Visualization of different lesion localizations and lesion levels.

- • Visualizations of baseline models using different pretrained encoders for alignment (Fig. 2);
- • Visualization of different lesion localizations and lesion levels. (Fig. 3);
- • Convergence performance visualizations of baseline models and those using the ARRA framework (Fig. 4).

### 7 More implementation and results in LAION-COCO and ImageNet

###### 7.1 Datasets

We collect 2.4M high-quality image-text pairs from the LAION-COCO (Schuhmann et al. 2022) dataset as our training data. We additionally collect 10K samples as a test set. All images are center-cropped to 256×256. We use subset provied by GoT (Fang et al. 2025), filtering with similarity > 0.3. For ImageNet (Deng et al. 2009), we conduct

[Figure 39]

Figure 4: More visual comparison of images generated by the baseline and the ARRA model (with alignment) during the iteration process.

experiments on the ImageNet 256×256 conditional generation benchmark.

###### 7.2 Implementation details

ARRA-Base. We apply the ARRA framework on LlamaGen (Sun et al. 2024a) to test the generalization ability of our model. In our experiments, we use a batch size of 32 with a learning rate of 1e-4 and a vocabulary size of 16,384. For the visual foundation model, we use CLIP-L (Radford et al. 2021a), where the extracted feature, our target representation, has the shape (batch size, 512). In the 111M model, the dimension of the hidden states is 768, while in the 343M model, the dimension of the hidden states is 1024. We use an MLP layer as the feature summarization tool to reduce the dimension of hidden states, making it possible to be aligned with our target. We set λ = 1 and use cosine similarity to calculate the loss between the target representation and the hidden states.

###### 7.3 More experiments results

This is a more comprehensive experiment results of ARRABase on the Imagenet datasets (Table 6), where we prove that ARRA enhances the generation ability of LlamaGen. ARRA works for both 111M and 343M models, indicating that this method is scalable.

###### 7.4 Visualizations

In this section, we present additional visualizations (Fig. 5, 6, 7, 8, 9) on the LAION-COCO and ImageNet datasets, comparing ARRA-Base and LlamaGen.

Table 6: ARRA-Base in image generation task across the ImageNet Dataset (classifier-free guidance=2.00)

Method Para epochs data FID ↓ IS ↑

LlamaGen 111M 100 0.26M 9.328 126.4 LlamaGen 111M 150 0.26M 8.421 142.7 LlamaGen 111M 200 0.26M 8.010 145.4 LlamaGen 111M 250 0.26M 7.580 161.8

- LlamaGen 111M 300 0.26M 7.283 156.7

- LlamaGen 111M 300 1.28M 5.464 193.6 LlamaGen 343M 100 0.26M 5.192 187.8 LlamaGen 343M 150 0.26M 4.666 206.7 LlamaGen 343M 200 0.26M 4.419 202.0

- LlamaGen 343M 300 0.26M 4.294 218.6

- LlamaGen 343M 300 1.28M 4.327 286.6

ARRA-Base 111M 100 0.26M 8.710 133.0 ARRA-Base 111M 150 0.26M 7.873 142.2 ARRA-Base 111M 200 0.26M 7.483 145.8 ARRA-Base 111M 250 0.26M 6.737 162.7 ARRA-Base 111M 300 0.26M 6.653 159.0 ARRA-Base 343M 100 0.26M 4.982 192.1 ARRA-Base 343M 150 0.26M 4.544 200.8 ARRA-Base 343M 200 0.26M 4.182 206.8 ARRA-Base 343M 300 0.26M 3.971 219.1

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

##### ARRA-Base-3.1BLlamaGen-3.1BARRA-Base-3.1BLlamaGen-3.1B

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Two men in suits whisper to each other.

A serene autumn landscape featuring a path by a lake, with trees displaying vibrant fall foliage

A person adorned in elaborate, gold and yellow jewelry and headdress, with intricate details and a dramatic,

Create a festive wreath using a grapevine base and a variety of greenery and berries, following this easy tutorial.

otherworldly appearance.

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

A symmetrical pattern resembling the Flower of Life, with concentric circles and interconnected lines in shades of green and yellow.

Grilled pork chops with rosemary and chives on a white plate with a fork

A large, brown, plush teddy bear with a soft, curly texture and brown paws.

A beautifully decorated Christmas tree with white and silver ornaments and lights, set against a white door and window

Figure 5: More visualizations of ARRA-Base-3.1B on the LAION-COCO dataset.

[Figure 56]

###### Figure 6: More visualizations of ARRA-Base-343M on ImageNet dataset.

[Figure 57]

###### Figure 7: More visualizations of LlamaGen-343M on ImageNet dataset.

[Figure 58]

###### Figure 8: More visualizations of ARRA-Base-111M on ImageNet dataset.

[Figure 59]

###### Figure 9: More visualizations of LlamaGen-111M on ImageNet dataset.

