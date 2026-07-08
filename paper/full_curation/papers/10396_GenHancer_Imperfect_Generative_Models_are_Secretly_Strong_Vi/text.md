## GenHancer: Imperfect Generative Models are Secretly Strong Vision-Centric Enhancers

Shijie Ma1,2, Yuying Ge1, , Teng Wang1, Yuxin Guo1,2, Yixiao Ge1, Ying Shan1 1ARC Lab, Tencent PCG 2Institute of Automation, CAS https://mashijie1028.github.io/GenHancer

# arXiv:2503.19480v3[cs.CV]31Jul2025

###### (a)Pipeline of Generative Visual Enhancement

###### (b) Trends of Generation & Visual Representations

(i) #Iterations (ii) #Blocks (iii) Condition: [CLS] + n% Local Tokens (iv) Pre-trained or From Scratch

[Figure 1]

Generative Model

0.94

0.96

0.97

Generation (CLIP score)

[Figure 2]

0.71

0.96

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

0.93

[Figure 7]

[Figure 8]

0.91

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

0.66

0.82

[Figure 13]

0.92

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

0.69

0.61

[Figure 25]

0.55

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

Reconstruction

[Figure 40]

original

original

original

original

| | |
|---|---|
| | |

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

###### Condition

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

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

100 1000

10000 3 blocks 6 blocks 9 blocks 0% 10% 50% 80% 100% From Scratch Pretrain

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

Projector

17.8

[Figure 83]

[Figure 84]

🔥

18.5

[Figure 85]

Visual Encoder

20.4

19.6

Visual Rep. (MMVP-VLM)

22.2

30.4

28.9

25.2

31.1

31.9

31.1

31.9

31.9

Figure 1. Perfect generation (reconstruction) does not always yield desirable visual representations. (a) Pipeline of fine-grained visual enhancements, where generative models take visual tokens as conditions and perform reconstruction. (b) Experiments across four dimensions, i.e., training iterations, denoiser size, ratio of local tokens as conditions, and whether to use pre-trained denoisers. We measure generation (CLIP score ↑) and visual representations (MMVP-VLM ↑) performance. As the results demonstrate, although increasing the number of training iterations, adding more denoiser blocks, using a larger ratio of local tokens as conditions, and employing pre-trained denoisers lead to better generation results, the performance of visual representations does not always improve. Best viewed zoomed in.

#### Abstract

Additionally, we demonstrate that lightweight denoisers can yield remarkable improvements. (3) Generation paradigms: We explore both continuous and discrete denoisers with desirable outcomes, validating the versatility of our method. Through our in-depth explorations, we have finally arrived at an effective method, namely GenHancer, which consistently outperforms prior arts on the MMVP-VLM benchmark, e.g., 6.0% on OpenAICLIP. The enhanced CLIP can be further plugged into multimodal large language models for better vision-centric performance. All the models and codes are made publicly available.

The synergy between generative and discriminative models receives growing attention. While discriminative Contrastive Language-Image Pre-Training (CLIP) excels in high-level semantics, it struggles with perceiving finegrained visual details. Generally, to enhance representations, generative models take CLIP’s visual features as conditions for reconstruction. However, the underlying principle remains underexplored. In this work, we empirically found that visually perfect generations are not always optimal for representation enhancement. The essence lies in effectively extracting fine-grained knowledge from generative models while mitigating irrelevant information. To explore critical factors, we delve into three aspects: (1) Conditioning mechanisms: We found that even a small number of local tokens can drastically reduce the difficulty of reconstruction, leading to collapsed training. We thus conclude that utilizing only global visual tokens as conditions is the most effective strategy. (2) Denoising configurations: We observed that end-to-end training introduces extraneous information. To address this, we propose a two-stage training strategy to prioritize learning useful visual knowledge.

#### 1. Introduction

Generative and discriminative models have evolved rapidly in recent years [3, 30, 50, 62]. Both of them exhibit complementary strengths, where generative models like diffusion models [16, 42, 63] and rectified flow [9, 29] capture lowlevel visual details, while discriminative models like Contrastive Language-Image Pre-Training (CLIP) [41, 66] and DINO [39] excel in high-level semantics. This complementary nature enables a synergistic relationship between them.

MetaCLIP-L

OpenAI@336

OpenAI@224

MetaCLIP-H

SigLIP@384

SigLIP@224

(a) More Efficient Generative Models (b) Stronger Visual Representations

- Figure 2. Comparison with prior method [58]. (a) We only need a lightweight denoiser, but (b) achieve stronger performance than DIVA [58], which relies on pre-trained heavy generative models.

Pioneering work [65] has shown that discriminative models can facilitate the training of generative models through feature alignment. Conversely, generative models can also enhance discriminative models by improving their ability to understand fine-grained visual patterns, e.g., orientation, color and quantity. This enhancement is particularly pertinent for models like CLIP, which have inherent visual shortcomings [53] that could also limit Multimodal Large Language Models (MLLMs) [31, 52] in vision-centric tasks. Recent works [18, 57, 58] have attempted to enhance CLIP ViT by using the visual features of ViT [7] as conditional inputs for generative models. These models perform self-supervised reconstruction to compel the discriminative model to capture fine-grained visual details, as illustrated in Fig. 1 (a). While these approaches demonstrate the potential of enhancing representations through generative models, they often rely on pre-trained heavy denoisers and do not explore the underlying principle.

To enable generative models to enhance visual representations, a natural question arises: Do we need a perfect generative model to achieve this enhancement? To address this question, we conducted preliminary experiments from several dimensions, including #training iterations, the ratio of local tokens as conditions, the size of denoisers, and whether to use a pre-trained generative model (denoiser), as in Fig. 1 (b). The answer is that perfect generation (reconstruction) does not always yield desirable visual representations. For example, in Fig. 1 (iii), introducing more local tokens as conditions can significantly improve reconstruction, while the visual enhancement will be drastically degraded. In Fig. 1 (iv), although the pre-trained denoiser exhibits better reconstruction, its representations are weaker.

This leads us to further investigate the key points for generative models to effectively enhance visual representations. We argue that generative models simultaneously contain useful knowledge, like visual patterns and details, as well as irrelevant information, like the gap between CLIP ViT’s feature space and generative models’ condition space. To effectively enhance representations, our general philosophy is that discriminative models should prioritize learning useful knowledge from generative models while circumvent-

ing irrelevant information. Furthermore, generative models can be divided into continuous [29, 32] and discrete [8] ones, with different denoising objectives, which should also be considered. Consequently, we conduct in-depth explorations from three key aspects: conditioning mechanisms, denoising configurations, and generation paradigms.

###### Key Point #1: Which part of the visual information

should generative models focus on? As in Fig. 1 (a), generative models take visual tokens of discriminative models as conditions. The choice of different tokens significantly impacts the outcomes. In this regard, we find only the global token (i.e., class token) could yield desirable visual enhancements. We attribute this to the fact that class token alone helps maximize mutual information between visual representations and generative models, while local tokens bring about information leakage and drastically reduce the task’s difficulty, resulting in collapsed learning.

###### Key Point #2: How to design denoising configura-

tions to transfer useful information for visual representations? The structure of the denoiser could determine the enhancement effects. Additionally, before training CLIP, it is essential to mitigate irrelevant information. Therefore, we investigate the influence of different sizes of the denoiser and training stages. In this paper, we propose GenHancer, a two-stage post-training method for visual enhancements. In the first stage, we pre-train the projector and denoiser while freezing the ViT, learning basic reconstruction abilities and mitigating irrelevant information. In the second stage, we fine-tune CLIP ViT to enhance its fine-grained visual representations. Meanwhile, we empirically found that a lightweight denoiser is sufficient to achieve remarkable results, which is more efficient yet stronger, as in Fig. 2.

###### Key Point #3: Do two types of denoisers share a com-

mon enhancing principle for visual representations? For both continuous and discrete denoisers, we present tailormade designs, including denoiser and conditioning structure. Moreover, we reveal that previous Key Points #1, #2 apply to both types, indicating the versatility of our method.

Our contributions are summarized as follows:

- • We conduct an in-depth study on visual representation enhancements with generative models and make the innovative discovery that perfect reconstruction and pre-trained models are not necessary. This leads us to explore three key aspects: conditioning mechanisms, denoising configurations, and the generation paradigms.
- • We propose GenHancer, a two-stage post-training method with only lightweight denoisers for visual enhancements, which uses only the class token as the conditional input to perform self-supervised reconstruction. Our method is applicable to both continuous and discrete denoisers.
- • Comprehensive vision-centric evaluations show that our enhanced CLIP significantly outperforms prior methods that rely on pre-trained heavy denoisers, as in Fig. 2.

#### 2. Related Works

MLLMs and Vision Encoders. Currently, MLLMs [15, 28] predominantly employ CLIP [41, 47] for visual encoding. Tong et al. [53] identified several failure patterns in CLIP, which hinder the fine-grained visual understanding. To overcome this issue, early efforts [20, 52, 53] employed an ensemble of visual experts to combat the visual shortcomings. More recently, ROSS [57] leverages intrinsic visual activations and incorporates a self-supervised visual reconstruction loss during training MLLMs. Complementarily, DIVA [58] proposes to enhance CLIP’s fine-grained abilities through diffusion feedback. Similar to [58], we independently enhance CLIP’s internal representations, which not only strengthen CLIP as a vision-language retriever but also enable the enhanced CLIP to be seamlessly integrated into MLLMs in a plug-and-play manner for better fine-grained vision-centric performance.

Enhancing Visual Representations with Diffusion Models. Early works [34, 44, 51] utilize generative models as data augmenters [36, 37, 49]. Another line of works [5, 18, 59] leverages self-supervised reconstruction tasks with diffusion models, which helps models grasp visual details and learn fine-grained representations. Similarly, DIVA [58] takes CLIP’s features as conditional inputs to the diffusion model [42], addressing its visual shortcomings through reconstruction. In summary, prior arts predominantly rely on diffusion models [11, 42], whereas we apply our method to both continuous and discrete generative models.

Vision-Centric Benchmarks. Canonical evaluations of MLLMs focus on fundamental multimodal Q&A capabilities across various domains, e.g., general perception and cognition [10], text and characters [45], scientific fields [33], and potential hallucinations [13, 27] in MLLMs. However, these benchmarks could not effectively assess a model’s fine-grained [14, 15] visual perception abilities, such as object color, quantity, orientation, and viewpoint. To solve this issue, Tong et al. [53] systematically explore the failure modes of CLIP and propose a challenging MMVP benchmark with 9 visual patterns. CV-Bench [52] further expands with 2,600 vision-centric VQA questions, covering dimensions like spatial relationships, count, depth, and distance of both 2D and 3D domains. Besides, NaturalBench [26] curates natural adversarial samples that are easy for humans but MLLMs struggle with. In this paper, we employ these vision-centric benchmarks to comprehensively evaluate models’ fine-grained visual abilities.

#### 3. Preliminaries of Generative Models

In principle, generative models can be divided into continuous and discrete ones. For continuous generative models, we focus on the recently popular rectified flow [29, 32],

while discrete generative models are conventionally built upon pre-trained codebooks [8, 55] for discrete modeling.

Rectified Flow (RF). Most generative models explicitly or implicitly learn a mapping from a basic distribution, e.g., Gaussian distribution N(0,I), to a target distribution, typically the real data distribution pdata. The core idea of RF is to learn an Ordinary Differential Equation (ODE) dZt = u(Zt,t)dt that follows a straight path from π0 to π1. Here u(Zt,t) is a time-conditional velocity field. This could be achieved by solving a least squares regression problem: minu 0 1 E ∥(X1 − X0) − u(Xt,t)∥2 dt, where Xt = tX1+(1−t)X0. In practice, we use ϕ to parameterize u, and t is basically sampled from the uniform distribution U(0,1). The learning objective of RF is:

2 2

LRF = Et,x

0,x1 (x1 − x0) − uϕ tx1 + (1 − t)x0,t

, where t ∼ U(0,1), x0 ∼ N(0,I), x1 ∼ pdata.

(1) Discrete Generative Models. For discrete modeling, one should first learn a discrete codebook, where images are represented by their corresponding indices. For example, VQ-GAN [8] employs some schemes [12, 67] to learn a discrete codebook of perceptually rich representations. Subsequently, given indices s<i of image x, the discrete generative model pϕ learns to predict the categorical distribution of the next index si via the cross-entropy objective:

L

pϕ(si|s<i), (2)

LCE = Ex∼p

data − log

i=1

where L denotes the sequence length of a sample. pϕ could be any form of model capable of modeling discrete distributions, e.g., PixelCNNs [54] and Transformers [8, 56].

Conditional Generation. To achieve conditional generation, one could incorporate the condition c, e.g., class labels or text prompts, into the parameterized model in Eq. (1) and Eq. (2) as uϕ(xt,t,c) and pϕ(si|s<i,c), respectively.

#### 4. Method

##### 4.1. Overview and Formulation

Overview. We propose a two-stage post-training method, namely GenHancer, to enhance CLIP ViT’s fine-grained representations, as in Fig. 3 (a). To capture key information from generative models, we delve into three aspects: First, the choice of visual tokens for condition determines the difficulty of the reconstruction task, which is crucial for enhancement (Sec. 4.2). Second, we introduce denoising configurations, which enable ViT to capture useful knowledge while mitigating irrelevant information (Sec. 4.3). Third, we present tailored design for both continuous and discrete generative models (Sec. 4.4), also shown in Fig. 3 (b), (c).

(c) Discrete Generative Model

###### (a) Overall Pipeline

###### (b) Continuous Generative Model

[Figure 86]

Generation

[Figure 87]

[Figure 88]

[Figure 89]

Generation

Generation

[Figure 90]

[Figure 91]

[Figure 92]

VQ-GAN

[Figure 93]

Single-DiT

[Figure 94]

[Figure 95]

MM-DiT

Perceiver

[Figure 96]

VAE

regression

add noise

reconstruction

Denoiser

cross-entropy

cond.

[Figure 97]

[Figure 98]

Denoiser

[Figure 99]

Denoiser

[Figure 100]

[Figure 101]

cond.

[Figure 102]

Projector

cond.

Projector

[Figure 103]

Projector

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

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

[CLS]

[CLS]

[Figure 124]

[Figure 125]

[Figure 126]

Representation

Representation

Representation

Visual Encoder

Visual Encoder

Visual Encoder

Training Recipes

[Figure 127]

###### Training Stages Visual Encoder

[Figure 128]

Projector

[Figure 129]

Denoiser

Generation Paradigms

###### Conditional Tokens

###### Denoising Conﬁgurations

[Figure 130]

[Figure 131]

[Figure 132]

❄ 🔥 🔥 🔥 🔥/❄ 🔥/❄

- Stage-1
- Stage-2

- 1. Continuous Model
- 2. Discrete Model

- 1. NO Local Tokens
- 2. NO Info. Leakage

- 1. Two-Stage Training
- 2. T-Sampling

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

- Figure 3. The two-stage post-training framework for visual enhancements. (a) Overall training pipeline. (b) Continuous generative model as the denoiser. We employ a lightweight FLUX-like DiT [24] (but with fewer blocks) and employ a regression loss of flow matching. (c) Discrete generative model as the denoiser. We choose a lightweight Perceiver [19] and employ cross-entropy loss to predict masked tokens.

Notations. Here, two types of generative models are uniformly represented as gϕ parameterized by ϕ. Let vθ denote CLIP’s visual encoder with parameters θ, whose features are connected to gϕ as conditions through projector hω, i.e., hω ◦vθ(x). The input sample is x, which becomes x in the denoising space, e.g., VAE [21] and VQ-GAN [8] for continuous and discrete denoisers, respectively.

Proof. The mutual information could be written as: I(V ;G) = H(G) − H(G|V ). Through reconstruction in Eq. (3) or Eq. (4), by conditioning G on V , V is trained to approximate the distribution of G. Consequently, H(G|V ) decreases during training. While H(G) is fixed, the decrease in H(G|V ) leads to the increase in I(V ;G).

| |
|---|

From the results in Fig. 1 (b)(i), the reconstruction improves as training progresses, which corresponds to an increasing I(V ;G). However, visual representations might decrease. In light of this, for the enhancement of visual representations, the knowledge in G can be decomposed into useful knowledge G1 (e.g., basic semantics, visual patterns) and irrelevant information G2 like the gap between feature space of vθ and condition space of gϕ. In this regard, to effectively enhance visual representations, our underlying philosophy is: The visual encoder should learn to capture useful knowledge from generative models as much as possible, i.e., maxI(V ;G1), while avoiding irrelevant information, i.e., minI(V ;G2). This equals to applying regularization on V to prevent overfitting to G2:

Repurposing Conditional Generation to Self-supervised Reconstruction. Generative models can capture lowlevel details. To transfer this capability to vθ, we replace the original condition c with the visual feature vθ(x). By reconstructing the visual inputs, vθ learns to grasp low-level visual details and is enhanced with fine-grained representations. In this sense, we transform the original conditional generation into a self-supervised reconstruction task. The learning objectives for continuous Lc and discrete generative models Ld can be re-written as Eq. (3) and Eq. (4):

0, x1 ( x1 − x0) − gϕ xt,t,hω ◦ vθ(x) 22, where t ∼ U(0,1), xt = t x1 + (1 − t) x0,

Lc = Et,x, x

(3) Ld = Ex − log

L

I(V ;G1)−λI(V ;G2) ⇒ max

max

I(V ;G1)+λd(V ;V0),

gϕ si|s<i,hω ◦ vθ(x) . (4)

V

V

(5) V0 is the initial visual model and d(·) is a distance metric.

i=1

Here, hω ◦ vθ(x) serves as the conditional input of gϕ.

##### 4.2. Conditional Visual Tokens

Formulation. Let G and V denote random variables of features of gϕ and vθ. I(·) and H(·) denote mutual information and entropy. Then we have the following theorem:

The choice of conditional visual tokens is crucial for visual enhancement. If too many tokens are fed to the generative model, the reconstruction becomes excessively easy. The reason is that local tokens directly correspond to image areas with information leakage. In this case, I(V ;G1) in Eq. (5) becomes small and vθ fails to grasp useful information from gϕ. To ensure a remarkable I(V ;G1), we

Theorem 1. When gϕ is fixed, self-supervised reconstruction is equivalent to maximizing the mutual information

I(V ;G) between V and G. The knowledge learned by vθ from gϕ can be interpreted as the increase in I(V ;G).

argue that the number of local tokens should be carefully controlled. Our experiments show that even a small number of local tokens, though achieving good reconstruction quality, can still cause marginal visual enhancement, as in Fig. 1 (iii). As a result, we propose that the visual condition features should exclusively comprise only the class token [CLS]. This strategy applies to both continuous and discrete models, as validated in Fig. 5 of Sec. 5.3.

##### 4.3. Denoising Configurations

To effectively enhance visual representations, we aim to maximize I(V ;G1) while suppressing I(V ;G2) in Eq. (5). In this regard, our explorations are three-fold: training stages, timestamp sampling of the continuous denoiser, and the update strategy for vθ.

Two Stage Training. An important source of G2 is the gap between the feature space of vθ and the conditions of gϕ, which is irrelevant to representation learning and could degrade the performance. Furthermore, since gϕ is lightweight and randomly initialized, it could introduce potential noise to vθ at the beginning. Consequently, we propose a two-stage training pipeline. At Stage-1, we train the denoiser gϕ and the projector hω while freezing vθ, in which gϕ acquires basic generative capabilities for visual enhancements and hω learns to bridge the space gap, thereby reducing I(V ;G2). In Stage-2, we focus on enlarging I(V ;G1) and train vθ to improve fine-grained representations. Moreover, we empirically found that as long as Stage-1 is performed sufficiently, the impact of whether the denoiser and projector are trained in Stage-2 is negligible.

Low Rank Adaption (LoRA) of vθ. The pre-trained visual encoder vθ possesses strong global semantics, i.e., V0, which should be maintained when incorporating finegrained perception. To prevent vθ from overfitting during reconstruction, we update vθ using LoRA [17], which implicitly constrains d(V,V0) in Eq. (5).

Timestamp Sampling. For continuous models like RF, timestamp sampling is of vital importance. Conventionally, RF [32] is trained to predict velocity across timestamps uniformly in [0,1]. Considering xt = tx1 + (1 − t)x0, prior works [9] uncover that the velocity target at intermediate timestamps, i.e., t ≈ 0.5, is more challenging. In our case, sampling intermediate timestamps more frequently could increase the difficulty of the reconstruction task, thus effectively amplifying I(V ;G1) and allowing the visual encoder vθ to effectively acquire useful fine-grained knowledge from G1. In this regard, we propose scaled LogitNormal sampling for timestamps, as shown below:

t = sigmoid(s · ε), where ε ∼ N(0,1). (6) Here, ε is sampled from the normal distribution, sigmoid(x) = 1+exp(1 −x), and s > 0 is the scale hyperparameter that controls the extent to which sampling is

focused on the intermediate timestamps. Smaller s results in more frequent sampling around 0.5. The diagrams of distributions in various s are illustrated in the Appendix.

##### 4.4. Generation Paradigms

For both types of generative models, we need to design architectures for denoisers and implementation of the conditioning mechanism. Notably, our denoiser is lightweight and randomly initialized, without pre-trained weights of heavy denoisers like Stable Diffusion [42] in [58].

Continuous Generative Models. We choose RF as the continuous denoiser, which is modeled in the latent space of pre-trained VAE [21]. The structure is inherited from FLUX.1-dev [24], consisting of n× Multimodal Diffusion Transformer (MM-DiT) [9, 40] blocks and 2n× singlestream DiT (Single-DiT) blocks, as shown in Fig. 3 (b). By default, we set n = 2, which is very efficient with ∼ 1/10 parameters of the original FLUX.1-dev denoiser. Similar to DiT [9, 40], the condition of visual tokens ([CLS] of vθ) is introduced through the modulation mechanism via adaptive layernorm [1, 40]. The learning objective is the regression of flow matching in Eq. (3).

Discrete Generative Models. Here, we choose Perceiver [19] as the discrete denoiser, building upon off-theshelf VQ-GAN’s codebook [8]. We first mask a certain proportion of input tokens. The condition of visual features is introduced via a cross-attention module, as depicted in Fig. 3 (c). Specifically, we set the query as the unmasked tokens s<i, while the key and value are the concatenation of the unmasked tokens and [CLS] of vθ. They are collectively fed to the Perceiver with cross-entropy loss to predict the masked token indices si, as in Eq. (4).

#### 5. Experiments 5.1. Experimental Setup

Implementation Details. For continuous generative models, we choose RF, whose structure is similar to FLUX.1-dev [24], but with only 2 MM-DiT and 4 Single-DiT blocks (∼ 10% of the parameters). The discrete denoiser is parameterized by a 6-layer Perceiver to predict the masked tokens indexed VQ-GAN’s codebook [8]. Similar to [61], the mask ratio is randomly sampled from 50% to 90%. For both generative models, we only take the [CLS] token of CLIP ViT as the conditional input while dropping other local tokens to prevent information leakage. We choose the scale factor in Eq. (6) as 1 by default.

Training Details. Our training process consists of two stages, each involving one epoch on the CC3M [43] dataset. We choose AdamW as the optimizer, with a learning rate of 1e-4 and 1e-5 for Stage-1 and Stage-2, respectively. At Stage-2, we optimize the visual encoder using LoRA with a rank of 16. We employ a global batch size of 256.

- Table 1. Performance of various CLIP backbones in MMVP-VLM benchmark. Here, we report our results using the continuous denoiser. The enhanced CLIP consistently outperforms prior methods across various visual patterns. The visual patterns are symbolized as: ☼: Orientation and Direction, : Presence of Specific Features, : State and Condition, : Quantity and Count, : Positional and Relational Context, : Color and Appearance, : Structural and Physical Characteristics, : Texts, : Viewpoint and Perspective.

CLIP Backbone #Params (M) Resolution Method ☼ Average

Original 13.3 13.3 20.0 20.0 13.3 53.3 20.0 6.7 13.3 19.3 + DIVA 13.3 20.0 40.0 6.7 20.0 53.3 46.7 20.0 13.3 25.9 + Ours 13.3 33.3 33.3 20.0 6.7 73.3 46.7 20.0 40.0 31.9 (+6.0)

OpenAI ViT-L-14 427.6 2242

Original 0.0 20.0 40.0 20.0 6.7 20.0 33.3 6.7 33.3 20.0 + DIVA 26.7 20.0 33.3 13.3 13.3 46.7 26.7 6.7 40.0 25.2 + Ours 6.7 20.0 33.3 20.0 6.7 73.3 53.3 26.7 26.7 29.6 (+4.4)

OpenAI ViT-L-14 427.9 3362

Original 13.3 6.7 66.7 6.7 33.3 46.7 20.0 6.7 13.3 23.7 + DIVA 6.7 6.7 60.0 0.0 26.7 66.7 20.0 20.0 40.0 27.4 + Ours 13.3 20.0 53.3 13.3 26.7 80.0 33.3 13.3 33.3 31.9 (+4.5)

MetaCLIP ViT-L-14 427.6 2242

Original 6.7 13.3 60.0 13.3 6.7 53.3 26.7 13.3 33.3 25.2 + DIVA 13.3 20.0 53.3 33.3 13.3 66.7 33.3 13.3 40.0 31.9 + Ours 20.0 20.0 66.7 26.7 26.7 66.7 33.3 20.0 53.3 37.0 (+5.1)

MetaCLIP ViT-H-14 986.1 2242

- SigLIP ViT-SO-14 877.4 2242

Original 26.7 20.0 53.3 40.0 20.0 66.7 40.0 20.0 53.3 37.8 + DIVA 13.3 26.7 60.0 46.7 13.3 73.3 53.3 26.7 53.3 40.7 + Ours 20.0 20.0 66.7 60.0 20.0 86.7 40.0 13.0 53.3 42.2 (+1.5)

- SigLIP ViT-SO-14 878.0 3842

Original 20.0 26.7 60.0 33.3 13.3 66.7 33.3 26.7 53.3 37.0 + DIVA 26.7 33.3 53.3 26.7 13.3 80.0 40.0 26.7 46.7 38.5 + Ours 26.7 20.0 66.7 33.3 13.3 86.7 40.0 26.7 46.7 40.0 (+1.5)

Comparative Baseline. Similar to [58], our method GenHancer independently enhances CLIP via post-tuning. When equipped with our enhanced CLIP and trained with original recipes, MLLMs could perform better on visioncentric benchmarks. In this regard, GenHancer could be viewed as a plug-and-play vision-enhancement method for MLLMs. We primarily compare with DIVA [58].

|a minion smiling with tongue out<br><br>a minion smiling without tongue out<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>Text Original Image Recon. DIVA Recon. Ours<br><br>[Figure 150]<br><br>DIVA ❌ Ours ✅<br><br>[Figure 151]|
|---|

Evaluation Protocol. Following [58], we perform visual enhancements on six CLIP backbones, including OpenAICLIP ViT-L @224/@336 [41], MetaCLIP@224 ViTL/H [62] and SigLIP-SO-14 @224/@384 [66]. We use MMVP-VLM [53] to evaluate fine-grained perception abilities. Subsequently, we follow the official training recipes of LLaVA-1.5 [31] to train MLLMs with our enhanced CLIP ViT. The resulting MLLMs are comprehensively evaluated on vision-centric benchmarks like MMVP-MLLM [53], CV-Bench [52] and NaturalBench [26], as well as multimodal understanding benchmarks, including POPE [27] ScienceQA [33] and HallusionBench [13].

Figure 4. Qualitative results. Although DIVA achieves better reconstructions of input images, it fails to perceive fine-grained visual details between ‘tongue out’ and ‘without tongue out’.

DIVA by 6.0% and 4.5% on OpenAICLIP and MetaCLIP, respectively. Besides, CLIP’s visual shortcomings are effectively addressed after post-training, e.g., we improved MetaCLIP’s color perception ( ) from 46.7% to 80.0%, and enhanced its viewpoint understanding ( ) by 20%.

Qualitative Evaluations. We present two cases in Fig. 4. Although DIVA achieves better reconstructions, our method correctly retrieves images for given texts, while DIVA fails. This further emphasizes that better reconstruction does not necessarily lead to better representations.

##### 5.2. Comparative Results

Our method significantly enhances CLIP’s fine-grained visual perception abilities. We evaluate CLIP models on the challenging MMVP-VLM benchmark [53], which contains 9 fine-grained visual patterns for a comprehensive vision-centric evaluation. As in Table 1, our method with only a lightweight denoiser, surpasses the previous method [58] that employed a heavy pre-trained denoiser across multiple CLIP backbones, with variations in resolution and parameters. For example, our method outperforms

Plug-and-play vision-centric enhancements for MLLMs. Our method independently enhances CLIP ViT with fine-grained representations. Considering that existing MLLMs [2, 30, 31] predominantly use CLIP ViT as the visual encoder, we replace the original CLIP with the enhanced CLIP as a plug-and-play module and integrate it into MLLMs to explore the impact of the enhanced visual representations on MLLMs’ final performance. For fair

- Table 2. Comprehensive evaluation of MLLMs (LLaVA-1.5 [31]), including vision-centric and conventional MLLM benchmarks. † We use official DIVA CLIP checkpoints [58] to reproduce the results. ‡ Similar to [25], we select the choice with the highest likelihood as MLLM’s prediction. Hallusion: HallusionBench [13]. SciQA: ScienceQA [33]. Bold and underline indicate the best and the second best.

LLM CLIP

Vision-Centric Benchmarks Conventional MLLM Benchmarks MMVPMLLM [53]

NaturalBench [26]‡ CV-Bench 2D [52] CV-Bench 3D [52]

POPE [27] SciQAIMG [33]

Hallusion Acc Q-Acc I-Acc G-Acc ADE20K COCO rand pop adv Avg. [13]

Vicuna-7B

Original 24.7 76.4 53.6 56.4 17.6 49.6 60.9 58.7 87.3 86.1 84.2 66.8 27.6 DIVA† 31.3 75.3 51.7 56.1 22.3 51.3 63.4 60.2 87.9 87.0 84.6 66.3 28.6 Ours 30.7 77.3 55.6 59.1 24.4 52.9 63.6 63.2 88.1 86.7 84.6 66.5 28.4

Vicuna-13B

Original 30.7 76.3 52.9 55.1 13.8 52.6 63.3 65.0 87.1 86.2 84.5 71.6 24.5 DIVA† 35.3 76.0 52.7 56.0 16.8 53.2 64.3 65.8 88.1 87.4 84.8 71.8 25.2 Ours 36.7 77.2 55.3 58.7 22.9 55.3 64.3 66.4 87.8 87.0 84.9 72.3 26.4

- Table 3. Performance of zero-shot classification and retrieval that require global semantics. We report the results of original and post-tuned OpenAICLIP@224.

Local Tokens Ratio

0% 10% 20% 50% 80% 100%

30

| |
|---|

MMVP-VLM

| |
|---|

| |
|---|

25

Classification Retrieval-Image@5 Retrieval-Text@5 IN-1K C100 SUN397 Cars Flickr30k COCO Flickr30k COCO

| |
|---|

Method

| |
|---|

20

Original 75.5 76.1 67.5 77.7 87.2 61.1 97.4 79.2 Ours 75.6 76.1 67.5 77.6 87.3 61.2 97.2 79.4

15

Continuous Discrete

- Figure 5. Performance of CLIP across various conditional visual tokens on MMVP-VLM, i.e., [CLS] + n% [LOCAL].

Cont. O@224 Cont. S@224 Disc. O@224 Disc. S@224

15

20

25

30

35

40

MMVP-VLM

Training Setup

| |
|---|

End-to-End Two-Stage

| |
|---|

- Figure 6. Comparison of CLIP with end-to-end and the proposed two-stage training on MMVP-VLM. Here, Cont. and Disc. denote continuous and discrete denoisers. O: OpenAICLIP. S: SigLIP.

comparisons, we adopt the same training setup as LLaVA1.5 [31], i.e., training data and stages, to train MLLMs. For DIVA [58], we adopt the official CLIP checkpoints. We conduct a comprehensive evaluation of the MLLMs on multiple vision-centric benchmarks, including MMVPMLLM [53], CV-Bench [52] and NaturalBench [26], as well as some general multimodal understanding benchmarks. Results in Table 2 show that visual enhancement of CLIP is effectively transferred to MLLMs, resulting in significant improvements across vision-centric benchmarks. For instance, compared to the original CLIP in Vicuna7B MLLM, we achieved 6.0% and 4.5% improvements on MMVP-MLLM and CV-Bench 3D, respectively.

cant performance degradation, which suggests that local tokens carry substantial signals for reconstruction, making the task too easy with information leakage. Consequently, this prevents the visual encoder from effectively learning finegrained details and brings about a limited I(V ;G1). The conclusion applies to both types of gϕ. Therefore, we propose to choose only the class token as the condition.

Visual enhancements do not hurt CLIP’s original global semantics. CLIP has inherently strong global semantics in classification-based tasks [35, 38]. To explore how finegrained enhancements affect this ability, we evaluate zeroshot classification on datasets like ImageNet-1K [6], CIFAR100 [23], Stanford Cars [22], and SUN397 [60] and zero-shot cross-modal retrieval tasks on Flick30k [64] and COCO [4]. Table 3 reveals that the performance difference is minimal (< 0.3%) across various settings, which means that our method could enhance CLIP’s fine-grained understanding without forgetting its global semantics [46, 48].

- Key Point #2.1: Two-Stage Training. As elaborated in Sec. 4.3, in Stage-1 of the two-stage training scheme, the projector learns to bridge the gap between the feature space of the visual encoder and the condition space of the de-

noiser, which serves as irrelevant information G2. Ablations comparing end-to-end with the proposed two-stage training are illustrated in Fig. 6. End-to-end training consistently exhibits a performance drop of over 5% across various settings. This indicates that our two-stage training is crucial in preventing interference from G2.

- Key Point #2.2: Timestamp Sampling for Continuous Denoisers. Timestamp sampling of continuous denoisers

##### 5.3. Key Explorations and Ablations

Key Point #1: Selecting Conditional Visual Tokens. As in Sec. 4.2, selecting conditional visual tokens is critical for enhancing representations. We conduct experiments by choosing the class token and different proportions of local tokens, i.e., [CLS] + n% [LOCAL]. As displayed in Fig. 5, even a very small ratio (10%) leads to signifi-

- Table 4. Comparison of timestamp sampling in continuous denoisers on MMVP-VLM. O: OpenAICLIP. M: MetaCLIP.

Distribution Scale O@224 O@336 M@224 Uniform N/A 21.5 22.2 23.7

Logit-Normal

0.1 27.4 25.9 26.7

- 0.5 28.2 28.9 29.6
- 1.0 31.9 29.6 31.9 5.0 24.5 25.9 25.9

10.0 20.7 20.0 21.5

- Table 5. Performance on SigLIP@224 across different sizes of lightweight continuous and discrete denoisers.

#DiT Blocks (MM+Single) 1+2 2+4 3+6 4+8 MMVP-VLM 41.5 42.2 42.2 41.5

Continuous

#Perceiver Layers 2 4 6 8 MMVP-VLM 41.5 43.7 45.2 43.7

Discrete

is also pivotal for vθ to learn the fine-grained knowledge from gϕ, i.e., I(V ;G1). We compare our proposed scaled Logit-Normal sampling with standard uniform sampling, as shown in Table 4. Compared to uniform sampling, ours favors sampling closer to the middle (t = 0.5), i.e., in the middle of two distributions xt = tx1 + (1 − t)x0, making denoising more challenging and more beneficial for enhancing I(V ;G1). For example, our proposed distribution outperforms uniform sampling by 10.4%, 7.4% and 8.2% on three CLIP backbones in Table 4. Additionally, when the scale s is too small (e.g., s = 0.1, sampling too around 0.5) or too large (e.g., s = 10, sampling close to 0 or 1), the lack of diversity in t can lead to suboptimal results due to the lack of diversity. In this work, we set s = 1 by default.

- Key Point #2.3: Sizes of lightweight denoisers. We further explore the impact of the size of lightweight denoisers. For the continuous RF, we consider the number of blocks in MM-DiT and Single DiT. We consider the number of layers for Perceiver. Table 5 demonstrates that the denoiser could perform remarkably well with a relatively small size, indicating the efficiency of our lightweight denoisers.
- Key Point #3: Continuous and Discrete Denoisers. Table 6 demonstrates the performance with continuous and discrete denoisers. Both of them surpass previous work [58] on various backbones. For example, the discrete denoiser obtains a 4.5% performance gain on SigLIP@224 [66]. In summary, our method is general and applies to both continuous and discrete models. It is efficient with lightweight denoisers but strong enough to outperform prior arts [58]. Notably, previous Key Points #1∼#2 are consistently applicable to both continuous and discrete denoisers, further highlighting the versatility of our method.

##### 5.4. Further Analysis

Why are improvements on SigLIP relatively small? In Table 1, we observe that the improvement on SigLIP is relatively smaller compared to OpenAICLIP and MetaCLIP.

- Table 6. Performance of our method with our continuous and

discrete denoisers on MMVP-VLM (average of all visual patterns). Bold and underline indicate the best and the second best.

Method OpenAI@224 SigLIP@224 SigLIP@384

DIVA 25.9 40.7 38.5 Continuous 31.9 42.2 40.0 Discrete 28.9 45.2 40.7

- Table 7. Efficiency comparison of our lightweight RF denoiser with pre-trained FLUX.1-dev.

Efficiency MMVP-VLM #Params Memory Time/100 iters OpenAI Meta-H

Denoiser

Pre-trained 11.90B 37.33G 198.57s 32.6 37.1 Lightweight 1.31B 13.07G 20.55s 31.9 37.1

Specifically, the performance gain over the original SigLIP is ∼ 3.7%, less than that for others, i.e., > 10%. Unlike the other two backbones [41, 62], SigLIP [66] does not explicitly train a distinct class token. In practice, we extract the pooler output of SigLIP as the condition for the denoiser, which is obtained by aggregating all local tokens through attention and linear layers. We attribute the relatively small improvement on SigLIP to the indirect leakage of local information through the pooler output, which hinders the enhancement of I(V ;G1). This is consistent with the discussion in Sec. 4.2 and the results in Fig. 5.

Efficiency analysis compared with pre-trained FLUX. We provide a comparison between our lightweight RF (n = 2) and original FLUX.1-dev [24] across the following dimensions: #params of denoisers, per-device GPU memory and training time of 100 iterations. To ensure fair comparisons, we fix a per-device batch size of 2. As Table 7 shows, our lightweight denoiser is much more efficient than the pre-trained heavy one. Specifically, our lightweight denoiser has approximately 1/10 of the parameters, occupies about 1/3 of the memory, and is 10 times faster in training, while the final performance remains comparable.

#### 6. Conclusive Remarks

In this paper, we delve into the underlying principles of how generative models enhance visual representations. We innovatively uncover that the perfect generation does not always yield optimal representations. The pivot is to learn useful knowledge from the generative model while mitigating irrelevant information. Our key findings lie in three aspects. (1) Conditioning mechanism. We found that local tokens could make the reconstruction task too easy, while class token alone as the condition makes the reconstruction task meaningful and significantly enhances visual representations. (2) Denoising configurations. We propose a novel two-stage post-training method to enable vision encoders committed to learning fine-grained knowledge while alleviating irrelevant content. (3) Our model design en-

ables both continuous and discrete denoisers to effectively enhance visual representations. Vision-centric evaluations demonstrate that our method with lightweight denoisers can significantly outperform previous methods relying on heavy pre-trained generative models. We hope this work will inspire further in-depth explorations into the synergy between generative and discriminative models, as well as the relationship between generation and understanding tasks.

#### References

- [1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450,

2016. 5

- [2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 6
- [3] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1:8, 2024. 1
- [4] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015. 7
- [5] Xinlei Chen, Zhuang Liu, Saining Xie, and Kaiming He. Deconstructing denoising diffusion models for self-supervised learning. In The Thirteenth International Conference on Learning Representations, 2025. 3
- [6] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 7
- [7] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. 2
- [8] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 2, 3, 4, 5
- [9] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 1, 5

- [10] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 3

- [11] Michael Fuest, Pingchuan Ma, Ming Gui, Johannes Schusterbauer, Vincent Tao Hu, and Bjorn Ommer. Diffusion models and representation learning: A survey. arXiv preprint arXiv:2407.00783, 2024. 3
- [12] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 3
- [13] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14375–14385, 2024. 3, 6, 7
- [14] Yuxin Guo, Siyang Sun, Shuailei Ma, Kecheng Zheng, Xiaoyi Bao, Shijie Ma, Wei Zou, and Yun Zheng. Crossmae: Cross-modality masked autoencoders for region-aware audio-visual pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26721–26731, 2024. 3
- [15] Yuxin Guo, Shuailei Ma, Shijie Ma, Xiaoyi Bao, Chen-Wei Xie, Kecheng Zheng, Tingyu Weng, Siyang Sun, Yun Zheng, and Wei Zou. Aligned better, listen better for audio-visual large language models. In The Thirteenth International Conference on Learning Representations, 2025. 3
- [16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1
- [17] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. 5
- [18] Drew A Hudson, Daniel Zoran, Mateusz Malinowski, Andrew K Lampinen, Andrew Jaegle, James L McClelland, Loic Matthey, Felix Hill, and Alexander Lerchner. Soda: Bottleneck diffusion models for representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23115–23127, 2024. 2,

- 3

[19] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International conference on machine learning, pages 4651–4664. PMLR, 2021.

- 4, 5

- [20] O˘guzhan Fatih Kar, Alessio Tonioni, Petra Poklukar, Achin Kulshrestha, Amir Zamir, and Federico Tombari. Brave: Broadening the visual encoding of vision-language models. In European Conference on Computer Vision, pages 113–

132. Springer, 2024. 3

- [21] Diederik P Kingma, Max Welling, et al. Auto-encoding variational bayes, 2013. 4, 5
- [22] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In Proceedings of the IEEE international conference on computer vision workshops, pages 554–561, 2013. 7

- [23] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. 7
- [24] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 4, 5, 8
- [25] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13299–13308, 2024. 7
- [26] Baiqi Li, Zhiqiu Lin, Wenxuan Peng, Jean de Dieu Nyandwi, Daniel Jiang, Zixian Ma, Simran Khanuja, Ranjay Krishna, Graham Neubig, and Deva Ramanan. Naturalbench: Evaluating vision-language models on natural adversarial samples. Advances in Neural Information Processing Systems, 37:17044–17068, 2024. 3, 6, 7
- [27] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 292–305, 2023. 3, 6, 7
- [28] Haokun Lin, Teng Wang, Yixiao Ge, Yuying Ge, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun, and Ying Shan. Toklip: Marry visual tokens to clip for multimodal comprehension and generation. arXiv preprint arXiv:2505.05422,

2025. 3

- [29] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. 1, 2, 3
- [30] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 1, 6
- [31] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 2, 6, 7
- [32] Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023. 2, 3, 5
- [33] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521,

2022. 3, 6, 7

- [34] Run Luo, Yunshui Li, Longze Chen, Wanwei He, Ting-En Lin, Ziqiang Liu, Lei Zhang, Zikai Song, Hamid AlinejadRokny, Xiaobo Xia, Tongliang Liu, Binyuan Hui, and Min Yang. DEEM: Diffusion models serve as the eyes of large language models for image perception. In The Thirteenth International Conference on Learning Representations, 2025. 3
- [35] Shijie Ma, Fei Zhu, Zhun Zhong, Wenzhuo Liu, Xu-Yao Zhang, and Cheng-Lin Liu. Happy: A debiased learning framework for continual generalized category discovery. Advances in Neural Information Processing Systems, 37:50850–50875, 2024. 7

- [36] Shijie Ma, Fei Zhu, Zhun Zhong, Xu-Yao Zhang, and ChengLin Liu. Active generalized category discovery. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16890–16900, 2024. 3
- [37] Shijie Ma, Fei Zhu, Zhen Cheng, and Xu-Yao Zhang. Towards trustworthy dataset distillation. Pattern Recognition, 157:110875, 2025. 3
- [38] Shijie Ma, Fei Zhu, Xu-Yao Zhang, and Cheng-Lin Liu. Protogcd: Unified and unbiased prototype learning for generalized category discovery. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 7
- [39] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. 1
- [40] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 5

- [41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 1, 3, 6, 8, 12
- [42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 3, 5, 12
- [43] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of ACL, 2018. 5, 12
- [44] Jordan Shipard, Arnold Wiliem, Kien Nguyen Thanh, Wei Xiang, and Clinton Fookes. Diversity is definitely needed: Improving model-agnostic zero-shot classification via stable diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 769–778,

2023. 3

- [45] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 3
- [46] Haoru Tan, Sitong Wu, Fei Du, Yukang Chen, Zhibin Wang, Fan Wang, and Xiaojuan Qi. Data pruning via movingone-sample-out. In Neural Information Processing Systems (NeurIPS), 2023. 7
- [47] Haoru Tan, Sitong Wu, Zhuotao Tian, Yukang Chen, Xiaojuan Qi, and Jiaya Jia. Saco loss: Sample-wise affinity

- consistency for vision-language pre-training. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 3
- [48] Haoru Tan, Sitong Wu, Wei Huang, Shizhen Zhao, and Xiaojuan Qi. Data pruning by information maximization. In International Conference on Learning Representations (ICLR),

2025. 7

- [49] Haoru Tan, Sitong Wu, Bo Zhao, Zeke Xie, and XIAOJUAN QI. Diff-in: Data influence estimation with differential approximation, 2025. 3
- [50] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2025. 1
- [51] Yonglong Tian, Lijie Fan, Phillip Isola, Huiwen Chang, and Dilip Krishnan. Stablerep: Synthetic images from text-toimage models make strong visual representation learners. Advances in Neural Information Processing Systems, 36: 48382–48402, 2023. 3
- [52] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2024. 2, 3, 6, 7
- [53] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9568–9578, 2024. 2, 3, 6, 7
- [54] Aaron Van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelcnn decoders. Advances in neural information processing systems, 29, 2016. 3
- [55] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 3
- [56] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 3
- [57] Haochen Wang, Anlin Zheng, Yucheng Zhao, Tiancai Wang, Zheng Ge, Xiangyu Zhang, and Zhaoxiang Zhang. Reconstructive visual instruction tuning. In The Thirteenth International Conference on Learning Representations, 2025. 2, 3, 12
- [58] Wenxuan Wang, Quan Sun, Fan Zhang, Yepeng Tang, Jing Liu, and Xinlong Wang. Diffusion feedback helps CLIP see better. In The Thirteenth International Conference on Learning Representations, 2025. 2, 3, 5, 6, 7, 8, 12
- [59] Chen Wei, Karttikeya Mangalam, Po-Yao Huang, Yanghao Li, Haoqi Fan, Hu Xu, Huiyu Wang, Cihang Xie, Alan Yuille, and Christoph Feichtenhofer. Diffusion models as masked autoencoders. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16284– 16294, 2023. 3
- [60] Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene

- recognition from abbey to zoo. In 2010 IEEE computer society conference on computer vision and pattern recognition, pages 3485–3492. IEEE, 2010. 7
- [61] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. In The Thirteenth International Conference on Learning Representations, 2025. 5
- [62] Hu Xu, Saining Xie, Xiaoqing Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying CLIP data. In The Twelfth International Conference on Learning Representations, 2024. 1, 6, 8
- [63] Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and MingHsuan Yang. Diffusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 56(4): 1–39, 2023. 1
- [64] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the association for computational linguistics, 2:67–78, 2014. 7
- [65] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In The Thirteenth International Conference on Learning Representations, 2025. 2
- [66] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023. 1, 6, 8
- [67] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 3

## GenHancer: Imperfect Generative Models are Secretly Strong Vision-Centric Enhancers

### Supplementary Material

#### Overview

In this appendix, we provide additional descriptions of the following contents:

- • Relationship with prior works in Appendix A, including some discussions about the differences.
- • More training details of hyperparameters in Appendix B.
- • Diagrams of various timestamp sampling distributions in Appendix C.
- • Additional experimental results in Appendix D.
- • Additional qualitative results and cases of the enhanced CLIP (Appendix E) and MLLMs with our enhanced CLIP (Appendix F).
- • We also attach algorithms of our two-stage training with continuous and dicrete denoisers in Appendix G.

#### A. Relationship with Prior Works

In this paper, we propose a two-stage post-training method to enhance discriminative models’ fine-grained visual representations. For discriminative models, we primarily choose CLIP [41], considering its wide range of applications. Specifically, CLIP is inherently a vision-language model, capable of image-text retrieval and matching. Additionally, CLIP ViT is widely employed as a visual encoder in Multimodal Large Language Models (MLLMs). Note that our approach follows a post-training paradigm, where we enhance the fine-grained capabilities of a pre-trained CLIP ViT, while preserving its original global semantics.

Comparison with DIVA [58]. DIVA is a pioneering work and proposes to enhance visual representations of CLIP ViT through diffusion feedback. It independently enhances CLIP ViT’s visual representations with the guidance of pretrained stable diffusion [42]. Similar to DIVA, our work focuses on enhancing CLIP ViT’s internal visual representations. The enhanced CLIP itself could be a more competent vision-language model with better image-text retrieval performance. Furthermore, the enhanced CLIP ViT serves as a plug-and-play module and could be seamlessly plugged into MLLMs. When using the same training recipes but with the enhanced vision encoder, MLLMs could be more capable on several vision-centric benchmarks, with better fine-grained perception of visual details and overcoming visual shortcomings brought about by the original CLIP.

Different from DIVA, we delve into the underlying principles of how generative models enhance vision models from various orthogonal dimensions. Notably, we only employ lightweight denoisers without pre-trained weights

of heavy generative models. Our method is efficient yet stronger than DIVA. We also provide several key insights about how to enhance visual representations, i.e., conditioning mechanisms and training configurations. We further explore the implementation of both continuous and discrete generative models. When equipped with corresponding tailor-made designs, both continuous and discrete denoisers outperform DIVA.

Comparison with ROSS [57]. Ross is a pioneering work that explores the intrinsic signals in vision modality and proposes to append vision-centric self-supervision into the training of MLLMs. The core difference between ROSS and our method is that, ROSS is directly oriented to training better MLLMs. In most cases, ROSS freezes CLIP ViT and enhances the vision-centric performance of MLLMs through the parameters of LLMs. In contrast, our method is directly oriented to enhance CLIP ViT’s visual representations. Our method is more general, and the resulting enhanced CLIP could be plugged into various MLLMs. In summary, we independently enhance CLIP ViT, which could be merged into MLLMs for further enhancements, while ROSS directly enhances MLLMs with the ViT frozen.

#### B. More Training Details

Default training settings. Our training process consists of two stages, each involving one epoch on the CC3M [43] dataset. We choose AdamW as the optimizer, with a learning rate of 1e-4 and 1e-5 for Stage-1 and Stage-2, respectively. At Stage-2, we optimize the visual encoder using LoRA with a rank of 16. We train the model on 8 GPUs with a per-device batch size of 16, and the gradient accumulation steps are set as 2, resulting in a global batch size of 256. We plug LoRA to CLIP ViT, with a rank of 16, and an α of 16. Additionally, we employ dropout with a ratio of 0.1 within LoRA.

Detailed settings in Fig. 1 of main manuscript. The default settings are: a lightweight denoiser with 2 MM-DiT and 4 Single-DiT blocks, using only the [CLS] as the condition, and two-stage training with 100,000 steps in stage1 and 5,000 steps in stage-2. Each of the four aspects in Fig. 1(b) modifies only one parameter or dimension at a time. Specifically, (i) changes #iters in stage-2 from 100 to 10,000. (ii) varies the number of denoiser-blocks (n×MMDiT+2n×Single-DiT) from n = 1 to n = 3. (iii) conditions denoisers with [CLS] along with n% of local tokens. (iv) compares the lightweight denoiser with n = 4 and the pretrained heavy FLUX with n = 19.

- 16

Density

Logit-Normal (s = 0.1)

0.0 0.2 0.4 0.6 0.8 1.0

t

0.0

0.5

1.0

1.5

2.0

2.5

3.0

Density

Logit-Normal (s = 0.5)

0.0 0.2 0.4 0.6 0.8 1.0

t

0.0

0.2

0.4

0.6

0.8

1.0

1.2

1.4

1.6

Density

Logit-Normal (s = 1)

0.0 0.2 0.4 0.6 0.8 1.0

t

0.0

2.5

5.0

7.5

10.0

12.5

15.0

- 17.5

14

12

10

8

6

4

2

0

0.0 0.2 0.4 0.6 0.8 1.0

t

Logit-Normal (s = 5)

Logit-Normal (s = 10)

Uniform Distribution

1.0

30

25

0.8

20

Density

Density

Density

0.6

15

0.4

10

0.2

5

0

0.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

t

t

Figure 7. Probability density function of different distributions.

Detailed settings in Table 7 of main manuscript. The pretrained FLUX is also trained under the same setting, i.e., two-stage training and only the [CLS] serves as the condition. Without these proposed keypoints, pretrained denoiser also fails to gain desirable results, e.g., 32.9→22.2 on MMVP, further indicating the generality of our method.

#### C. Diagrams of Timestamp Sampling

The scaled Logit-Normal timestamp sampling is as follows: t = sigmoid(s · ε), where ε ∼ N(0,1). (7)

We provide some illustrative diagrams to show the distribution of several candidate distributions, as shown in Fig. 7. In our scaled Logit-Normal sampling, as s decreases, the distribution becomes more focused on sampling around the middle (t = 0.5). Conversely, as s increases, the distribution becomes more biased towards sampling at the extremes, i.e., t = 0 or 1.

#### D. More Experimental Results

The effect of LoRA. In Stage-2, we apply LoRA to the visual model. The reason is that direct training on the visual model causes rapid updates, which can easily damage the model’s high-level semantics and lead to overfitting. By using LoRA, the model can be trained on a larger variety of samples, allowing it to learn more generalizable and finegrained representations. We conduct experiments on several CLIP backbones, and compare the performance with direct training and LoRA training, as shown in Fig. 8. The performance with LoRA for visual encoder consistently outperforms the cases of direct training.

w/o LoRA w/ LoRA

37.5

| |
|---|

35.0

32.5

MMVP-VLM

30.0

27.5

25.0

22.5

20.0

OpenAI@224 OpenAI@336 MetaCLIP-L MetaCLIP-H

Figure 8. The effect of LoRA on several CLIP backbones.

Table 8. Performance of various mask ratios on OpenAICLIP@224.

Mask Ratio (%) 50 60 70 75 80 85 90 random (50-90) MMVP-VLM 28.1 27.4 28.9 27.4 26.7 25.9 25.9 28.9

Whether to update the denoiser and projector in Stage2. In the main text, we argue that in Stage-1, the visual encoder should be fixed, and we train the denoiser and projector. In this way, the projector could learn to bridge the gap between the feature spaces, which serves as the irrelevant information G2 for visual enhancements. In Stage-2, we begin to train CLIP ViT to enhance its visual representations. We empirically found that whether the denoiser and projector are updated in Stage-2 has marginal impacts on the final results, as long as stage-1 training is sufficient. The results are shown in Fig. 10.

Performance with various mask ratios. In the discrete denoiser, we apply masking mechanisms. Here, we provide experimental results across various mask ratios of OpenAICLIP@224, as shown in Table 8.

Text Image

Text Image

Text Image

[Figure 152]

[Figure 153]

[Figure 154]

a duck facing right

a car facing front

a minion smiling with tongue out

[Figure 155]

[Figure 156]

[Figure 157]

a minion smiling without tongue out

a duck facing left

a car facing back

[Figure 158]

[Figure 159]

Ours ✅ Original CLIP ❌ Ours ✅ Original CLIP ❌

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

###### Ours ✅ Original CLIP ❌

Text Image

Text Image

Text Image

[Figure 164]

[Figure 165]

[Figure 166]

black keyboard

- 3 drinks
- 4 drinks

uneven road surface

[Figure 167]

[Figure 168]

[Figure 169]

smooth road surface

white keyboard

[Figure 170]

[Figure 171]

Ours ✅ Original CLIP ❌ Ours ✅ Original CLIP ❌

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

###### Ours ✅ Original CLIP ❌

###### Figure 9. Qualitative results of CLIP. The enhanced CLIP overcomes the original visual shortcomings in fine-grained details.

Train only ViT Train all

37.5

| |
|---|

35.0

32.5

MMVP-VLM

30.0

27.5

25.0

22.5

20.0

OpenAI@224 OpenAI@336 MetaCLIP-L MetaCLIP-H

- Figure 10. The performance of whether to update the denoiser and the projector in Stage-2.

#### E. Qualitative Results of CLIPs

We provide further qualitative results of the original CLIP and our enhanced CLIP, as shown in Fig. 9. The enhanced CLIP overcomes original visual shortcomings in fine-grained details, including color, quantity, structural characteristics and state.

#### F. Qualitative Results of MLLMs

We provide qualitative results of LLaVA-1.5 with original CLIP ViT and our enhanced CLIP ViT, as shown in Fig 11. Our enhanced visual model could further boost MLLMs’ fine-grained visual perception abilities.

#### G. Algorithms

For a clearer and more thorough understanding of our method, we attach the algorithm details of two-stage posttraining with continuous and discrete denoisers in Algorithm 1 and Algorithm 2, respectively.

[Figure 176]

[Figure 177]

###### Question: From which angle is this image taken? (a) Front (b) Side

###### Question: From which angle is this image taken? (a) Front (b) Side

LLaVA 1.5

LLaVA 1.5

The image is taken from a front angle, as the green snake is seen from the front, with its head and body visible.

The image is taken from a front angle, as the snake is seen resting on a branch.

[Figure 178]

[Figure 179]

✅ ❌

- Answer: (a)

The image is taken from a side view, with the green snake resting on a tree branch.

- Answer: (b)

Answer: (a)

LLaVA 1.5 + Our ViT

LLaVA 1.5 + Our ViT

The image is taken from a front angle, allowing us to see the green snake on the tree branch.

[Figure 180]

[Figure 181]

✅

✅

Answer: (a)

Question:

Is the edge of the heart in the picture white or dark-colored? (a) White (b) Dark-colored

The edge of the heart in the picture is white. Answer: (a)

The edge of the heart in the picture is white.

- Answer: (a)

LLaVA 1.5

LLaVA 1.5 + Our ViT

Question:

Is the edge of the heart in the picture white or dark-colored? (a) White (b) Dark-colored

The edge of the heart in the picture is white.

- Answer: (a)

The edge of the heart in the picture is dark-colored.

- Answer: (b)

LLaVA 1.5

LLaVA 1.5 + Our ViT

[Figure 182]

✅

[Figure 183]

✅ ❌

[Figure 184]

[Figure 185]

✅

[Figure 186]

[Figure 187]

Question:

In this image, how many eyes can you see on the animal? (a) 1 (b) 2

In the image, there are two eyes visible on the bird.

- Answer: (b)

Question:

[Figure 188]

[Figure 189]

In this image, how many eyes can you see on the animal? (a) 1 (b) 2

LLaVA 1.5

LLaVA 1.5

In the image, there are two eyes visible on the animal. Answer: (b)

[Figure 190]

❌ ✅

[Figure 191]

LLaVA 1.5 + Our ViT

LLaVA 1.5 + Our ViT

In the image, there is only one eye visible on the bird. Answer: (a)

In the image, there are two eyes visible on the animal. Answer: (b)

[Figure 192]

[Figure 193]

✅

✅

- Figure 11. Qualitative results of MLLMs on MMVP-MLLM benchmark. When equipped with our enhanced CLIP, MLLMs produce better vision-centric performance.

- Algorithm 1 Two-stage Visual Enhancements with Continuous Lightweight Denoiser

Input: Lightweight and random-initialized denoiser gϕ(·), with lightweight FLUX-like architecture (MM-DiT + Single-DiT). Input: Pre-trained CLIP ViT vθ(·) for fine-grained visual representation enhancements. Input: Random initialized projector hω(·) to bridge the feature space of vθ and condition space of gϕ. Input: The scale hyperparameter s in the proposed scaled Logit-Normal sampling. Input: Pre-trained VAE vae(·) to provide latent space for generative modeling. Input: Image-only training dataset D without annotations.

- 1: # =================================== Stage-1 ==================================
- 2: for x in D do
- 3: ▷ Prepare input data for generative modeling in latent space: x1 = vae(x) and x0 ∼ N(0, I).
- 4: ▷ Interpolating in the feature space: xt = t x1 + (1 − t) x0.
- 5: ▷ Visual encoding as conditions for denoisers: hω ◦ vθ(x).
- 6: ▷ Timestamp sampling via scaled Logit-Normal distributions: ε ∼ N(0, 1) then t = sigmoid(s · ε).
- 7: ▷ Denoising regression objective (flow matching): # only update gϕ and hω.

arg min

ϕ,ω

Et,x, x0, x1 ( x1 − x0) − gϕ xt, t, hω ◦ vθ(x) 22.

- 8: end for
- 9: # =================================== Stage-2 ==================================
- 10: Plug LoRA upon vθ.
- 11: for x in D do
- 12: ▷ Prepare input data for generative modeling in latent space: x1 = vae(x) and x0 ∼ N(0, I).
- 13: ▷ Interpolating in the feature space: xt = t x1 + (1 − t) x0.
- 14: ▷ Visual encoding as conditions for denoisers: hω ◦ vθ(x).
- 15: ▷ Timestamp sampling via scaled Logit-Normal distributions: ε ∼ N(0, 1) then t = sigmoid(s · ε).
- 16: ▷ Denoising regression objective (flow matching): # update vθ. Optional: gϕ and hω.

arg min

θ

Et,x, x0, x1 ( x1 − x0) − gϕ xt, t, hω ◦ vθ(x) 22.

- 17: end for Output: The enhanced visual model v⋆θ with stronger fine-grained representations.

- Algorithm 2 Two-stage Visual Enhancements with Discrete Lightweight Denoiser

Input: Lightweight and random-initialized denoiser gϕ(·), instantiated with a lightweight Perceiver. Input: Pre-trained CLIP ViT vθ(·) for fine-grained visual representation enhancements. Input: Random initialized projector hω(·) to bridge the feature space of vθ and condition space of gϕ. Input: Mask ratio r for discrete modeling. Input: Pre-trained VQ-GAN vq-gan(·) to discrete indices for generative modeling. Input: Image-only training dataset D without annotations.

- 1: # =================================== Stage-1 ==================================
- 2: for x in D do
- 3: ▷ Obtain latent embeddings and corresponding discrete indices of input data in VQ-GAN’s codebook: x, s = vq-gan(x).
- 4: ▷ Masking x’s tokens with ratio r to obtain masked part xmask, smask and unmasked part xunmask, sunmask.
- 5: ▷ Visual encoding and obtain conditions via cross-attention for denoisers:

Q = xunmask, K, V = concat xunmask; hω ◦ vθ(x) ,

cω,θ = cross-attn(Q, K, V ).

- 6: ▷ Denoising cross-entropy objective (masked index prediction): # only update gϕ and hω.

arg min

ϕ,ω

Ex − log

L

i=1

gϕ smask|sunmask, cω,θ .

- 7: end for
- 8: # =================================== Stage-2 ==================================
- 9: Plug LoRA upon vθ.
- 10: for x in D do
- 11: ▷ Obtain latent embeddings and corresponding discrete indices of input data in VQ-GAN’s codebook: x, s = vq-gan(x).
- 12: ▷ Masking x’s tokens with ratio r to obtain masked part xmask, smask and unmasked part xunmask, sunmask.
- 13: ▷ Visual encoding and obtain conditions via cross-attention for denoisers:

Q = xunmask, K, V = concat xunmask; hω ◦ vθ(x) ,

cω,θ = cross-attn(Q, K, V ).

- 14: ▷ Denoising cross-entropy objective (masked index prediction): # update vθ. Optional: gϕ and hω.

arg min

θ

Ex − log

L

i=1

gϕ smask|sunmask, cω,θ .

- 15: end for Output: The enhanced visual model v⋆θ with stronger fine-grained representations.

