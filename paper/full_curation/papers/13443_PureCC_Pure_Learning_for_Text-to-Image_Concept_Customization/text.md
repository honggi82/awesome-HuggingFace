## PureCC: Pure Learning for Text-to-Image Concept Customization

KL Divergence (to Original):

- • Ours: 0.0097
- • LoRA: 0.0099
- • Dreambooth:0.0367

Zhichao Liao1*† Xiaole Xian2*† Qingyu Li4 Wenyu Qin4 Meng Wang4 Weicheng Xie2,3 Siyang Song5 Pingfa Feng1 Long Zeng1 Liang Pan6

1Tsinghua University 2School of Computer Science & Software Engineering, Shenzhen University 3Guangdong Provincial Key Laboratory of Intelligent Information Processing, Shenzhen University

4Kling Team, Kuaishou Technology 5University of Exeter 6S-Lab, Nanyang Technological University

# arXiv:2603.07561v2[cs.CV]19May2026

###### (a) Disruption of the Original Model’s Behavior

(c) Decline in Quality of the Generated Images

A scene of a dog in a large steaming pot.

[V] dog A scene of a [V] dog in a large steaming pot.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

HPSv2.1

[Figure 7]

[Figure 8]

Step=200

[Figure 9]

[Figure 10]

[Figure 11]

(b) Degradation of Prompt Adherence Ability

CLIP-T

Two minimalist berry bowls, placed on a bright window.

[V] bowl Two minimalist [V] bowls, placed on a bright window.

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Step = 400

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Original (SD 3.5-M ) Custom Set DreamBooth LoRA PureCC (Ours)

Figure 1. We introduce PureCC, a novel concept customization approach. (a) PureCC effectively maintains target-unrelated image elements with original model’s behavior after the personalized concept insertion. (b) Existing methods such as DreamBooth [36] and LoRA [18] fail to follow the prompt ‘placed on a bright window’ during custom generation. (c) The declined curve indicates that existing methods compromise the original model’s ability of prompt adherence (CLIP-T [33]) and generating high-quality images (HPSv2.1 [47]).

tations as implicit guidance and a trainable flow model producing the original conditional prediction, jointly achieving pure learning for personalized concepts. Furthermore, PureCC introduces a novel adaptive guidance scale λ⋆ to dynamically adjust the guidance strength of the target concept, balancing customization fidelity and model preservation. Extensive experiments show that PureCC achieves state-of-the-art performance in preserving the original behavior and capabilities while enabling high-fidelity concept customization. The code is available at https: //github.com/lzc-sg/PureCC.

### Abstract

Existing concept customization methods have achieved remarkable outcomes in high-fidelity and multi-concept customization. However, they often neglect the influence on the original model’s behavior and capabilities when learning new personalized concepts. To address this issue, we propose PureCC. PureCC introduces a novel decoupled learning objective for concept customization, which combines the implicit guidance of the target concept with the original conditional prediction. This separated form enables PureCC to substantially focus on the original model during training. Moreover, based on this objective, PureCC designs a dual-branch training pipeline that includes a frozen extractor providing purified target concept represen-

(d) Distribution Shift

[Figure 22]

[Figure 23]

### 1. Introduction

Concept customization [22, 36], an important task in custom text-to-image (T2I) generation, allows users to synthesize personalized concepts (e.g., new subjects or styles) contextualized in different scenes using only a few reference images (3-5). Benefiting from the development of generative models like diffusion [32, 35] and flow-based mod-

Step=200

[Figure 24]

[Figure 25]

[Figure 26]

* Equal Contribution. † This work was conducted during the author’s internship at Kling

Team, Kuaishou Technology. Corresponding author.

[Figure 27]

Step = 400

[Figure 28]

[Figure 29]

[Figure 30]

els [10, 28], it has attained impressive results in various application fields, including continuous content creation [4], artistic production [11], and advertising design [23, 26].

Most methods [13, 36, 40] learn personalized concepts by adapting the distribution of pre-trained model to match the user-specific concept distribution through full finetuning or parameter-efficient techniques like LoRA [18]. They generally associate the target concept with an identifier [V] and enable personalized generation via prompt injection during inference. However, existing research mainly emphasizes high-fidelity and multi-concept customization while overlooking two significant issues:

- • Disruption of the Original Model’s Behavior: An ideal personalized concept insertion should focus solely on adjusting the concept-related aspects while keeping the image elements unrelated to the target concept consistent with the original model’s behavior. However, as the case illustrated in Fig. 1 (a), existing methods fail to alter only the original dog to the target [V] dog, with changing unrelated original image elements such as background, style and lighting. This is because they treat all language-vision knowledge in the custom set as the learning source, but with limited reference images for learning, the generative model struggles to differentiate the target concept from other redundant information in the custom set, and to establish a unique association with the identifier [V]. Therefore, during custom generation, it leads to undesirable predictions of the target concept, disrupting the original model’s behavior. To our knowledge, such disruptions in concept customization have not been addressed or studied in previous works.
- • Degradation of the Original Model’s Capability: T2I generative models are pre-trained on large-scale multimodal databases, enabling them to effectively follow text prompts and generate high-quality images. However, after transforming the pre-trained model into a custom model, existing methods tend to diminish these generative capabilities as shown in Fig. 1 (b) and (c). This issue arises because existing methods lack specific consideration for the original model in their learning objectives. Thus, when learning the personalized concept on scarce data, there is a risk of original data distribution drift as shown in Fig. 2, resulting in the degradation of the model’s capability to adhere to prompt inputs and generate high-quality images.

To address these issues, in this paper, we propose PureCC, a novel concept customization fine-tuning method that aims to purely learn personalized concepts while minimizing the influence on the original model’s behavior and capability. Specifically, PureCC designs an innovative learning objective to guide fine-tuning, which can be formed as a distinct combination of implicit guidance of the target personalized concept and original conditional prediction. This separated form allows PureCC to practically consider the original model while learning the per-

KL Divergence to Original Distribution (×"#!"): Ours: 0.09; LoRA: 0.21; DreamBooth: 1.79

KL Divergence to Original Distribution (×"#!#): Ours: 0.21; LoRA: 0.86; DreamBooth: 3.35

[Figure 31]

[Figure 32]

###### (a) Distribution in ‘[V] dog’ case (b) Distribution in ‘[V] bowl’ case

Figure 2. Original Distribution Drift. Visualization and KL Divergence results demonstrated that existing methods, which adjust pre-trained models to align with the target distribution for learning personalized concepts, lead to distribution drift.

sonalized concept. To decouple the target concept from the custom set, we first introduce a representation extractor that uses a pre-trained flow-based model as the backbone, and we employ layer-wise tunable concept embeddings to finetune this model on the custom set, enhancing the understanding and representation of target concepts. Then, we introduce our dual-branch training pipeline to purely learn the target concept while preserving original model’s behavior and capability, which comprises the frozen representation extractor and a trainable flow model. During training, the frozen representation extractor provides a relatively pure representation of the target concept, while the trainable model offers a basic conditional prediction, serving as the implicit guidance and the original prediction for the proposed learning objective, respectively. Moreover, we propose a novel adaptive guidance scale λ⋆ based on the representation alignment between dual branches to dynamically adjust the strength of the target concept guidance, effectively balancing the trade-off between personalized concept fidelity and original model preservation. Extensive experiments show that our method achieves state-of-the-art performance in preserving the original behavior and capabilities while enabling high-fidelity concept customization. In summary, our contributions are as follows:

- • We introduce PureCC, a novel concept customization method, which reformulates a learning objective to purely learn the personalized concepts while minimizing the impact on the original model’s behavior and capability.
- • We design a dual-branch training pipeline based on our learning objective with a frozen representation extractor and a trainable model, providing specific implicit concept guidance and original conditional prediction.
- • We introduce an adaptive scale λ⋆ based on cross-branch representation alignment, dynamically balancing concept fidelity and preservation of original model.

### 2. Related Work

Diffusion and Flow-based Models are recent mainstream generative models. Diffusion models [17, 35, 42] aim to learn Stochastic Differential Equations (SDEs) that control the diffusion process. Flow-based models [28] offer an al-

ternative approach by directly modeling sample trajectories using Ordinary Differential Equations (ODEs) instead of SDEs. Recent research [10, 29, 54] has shown that ODEbased approaches attain faster convergence and improved controllability in T2I generation. In this study, we select the flow model SD3.5-M [10] as the basis for our research. Concept Customization focuses on extending the pretrained T2I model to generate personalized concepts. Existing methods can be categorized into Tuning-free methods [7, 30, 49–51] and Tuning-based methods [9, 12– 14, 18, 19, 22, 36, 48]. Tuning-free methods typically encode the reference image as feature embeddings and integrate them into the base models in a specific way (e.g., text embeddings [49] or the cross-attention layer [50, 51]). Conversely, Tuning-based methods optimize specific parameters on a limited set of images to embed the personalized concept into the generative model. DreamBooth [36] proposes to address subject-driven generation by fine-tuning all pre-trained model parameters, while several works employ textual inversion [12, 44] to learn word embeddings of personalized concepts. LoRA [18] and its variants [9, 13, 27, 40, 53] introduce additional low-rank subspaces to learn target concepts, reducing computational overhead. Although existing methods have made significant progress in enhancing concept fidelity and multi-concept customization, they often overlook the disruption of the original model’s behav-

- ior and capabilities caused by concept insertion. Guidance in Generative Models aims to achieve better controllability in the generation process [24]. Classifier Guidance [8] uses an additional pre-trained classifier to provide class-specific guidance for controllable generation and [1, 3, 15, 45, 52] incorporate additional regularization guidance within the diffusion model to improve semantic perception and text-image alignment. Although the above explicit guidelines are intuitive, they typically rely on external control components beyond the base model, making them less flexible and computationally demanding. To address these issues, another line proposes implicit guidance. Classifier-Free Guidance (CFG) [16] treats the generative model itself as a conditional guidance branch, thereby eliminating the need for an external classifier. Subsequently, various studies [5, 6, 37, 38] have explored more advanced forms of implicit guidance to improve sample diversity and fidelity. In this work, we extend the idea of implicit guidance into the training pipeline, formulating concept customization guided by implicit personalized concept representation, enabling pure and controllable concept insertion.

### 3. Preliminary

Conditional Flow Matching. Suppose that x0 ∼ q(x|y) is a data sample of the true distribution and x1 ∼ p(x|y) represents a sample of source distribution. Recent conditional flow-based models adopt the Rectified Flow [29]

framework, which defines the source data sample xt as

xt = (1 − t)x0 + tx1, (1)

for t ∈ [0,1]. Then a transformer model is trained to directly regress the velocity field dtd xt = vtθ(xt|y) by minimizing the Conditional Flow Matching (CFM) loss

vt xt − vtθ xt|y 22 , (2) where the target velocity field is vt xt = x1 − x0.

LCFM = Et,x

t

Concept Customization. Given a limited set of reference images with the same personalized concept, most methods optimize specific text tokens (e.g., [V]) to learn the target concept. In the custom set, the identifier [V] is typically combined with basic textual descriptions of the reference image (termed as Base text in this paper) to form a complete corpus. In a case, the Complete text is “ A [V] dog standing on a surfboard, riding a wave”, where “A dog standing on a surfboard, riding a wave” is the Base text and “[V] dog” is the Target text. This Complete text is then encoded into textual embedding ycomplete by the pre-trained text encoder E(·) (e.g., CLIP [33], T5 [34]). Finally, the pre-trained flow model achieves personalized concept learning by finetuning on a custom set using the loss:

vt xt − vtθ xt|ycomplete 22 . (3)

LCC = Et,x

t

Implicit Guidance. In Classifier-Free Guidance (CFG) [16], a flow model vtθ xt|y is trained to predict both conditional and unconditional velocity fields. This is achieved by introducing y = ∅, which denotes the null condition. During inference, the guided velocity field is formed by

vˆtθ x|y = (1 − w) · vtθ x|y = ∅ + w · vtθ x|y , (4)

where w is the guidance scale. CFG treats the generative model itself as an implicit classifier to guide the generation process. To intuitively understand implicit conditional guidance, we can rewrite Eq. 4 as follows:

vˆtθ(x|y) = vtθ(x|y = ∅) + w · vtθ x|y) − vtθ x|y = ∅)

Implicit Conditional Guidance

(5)

### 4. Methodology

#### 4.1. Learning Objective in PureCC

To address the limitations of existing methods, PureCC introduces a novel learning objective for PCC task. Specifically, inspired by the form within CFG in Eq. 5, the guided velocity field of conditional generation can be viewed as adding an implicit conditional guidance to the unconditional prediction. Similarly, we consider the goal velocity field of concept customization as adding implicit guidance

[Figure 33]

[Figure 34]

###### Representation Extractor Pure Learning Pipeline

Velocity Flow Space (Frozen Model !!"" )

[Figure 35]

[Figure 36]

Prompt: “A Photo of [V] Dog”

Reference Images

'%&- (% ∅)

[Figure 37]

[Figure 38]

[Figure 39]

Frozen Representation Extractor !#$$

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

###### Implicit Guidance of Target Concept

[Figure 44]

Text Encoder

Implicit Guidance of Target Concept

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

…

…

DiT-Block

DiT-Block

DiT-Block

[Figure 45]

[Figure 46]

[Figure 47]

R()%-.)

[Figure 48]

###### “ ∅ ”

|Y#%&+| |
|---|---|
| | |
| | |
|Y#%&"| |
| | |
| | |
|Y#%&,<br><br>Y#%&-| |

[Figure 49]

…

Adaptive Scale

%∗ %∗ %∗

Layer-WiseTunableConceptEmbeddings

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

%∗

[Figure 54]

[Figure 55]

DiT-Block

[Figure 56]

"#%&:

R()%-.)

“ [V] dog”

[Figure 57]

[Figure 58]

…

%*,+ &* #%&'()$*$) iteration = 400

'%&- (% )%-.)

Adaptive Scale ,∗

[Figure 59]

|| |
|---|
<br><br>|[Figure 60]|
|---|
<br><br>Y#%&+ Y#%&" … Y#%&-| | | |
|---|---|---|---|
| | | | |

x#

DiT-Block

'%&+ (% )'()*+,%,) iteration = 100

'%&+ (% )'()*+,%,)

ℒ01&2//

…

[Figure 61]

[Figure 62]

[Figure 63]

…

DiT-Block

DiT-Block

DiT-Block

Original Conditional Prediction

DiT-Block

…

ℒ//

…

#%&'()$*$:

“A[V] dog standing on a surfboard, riding a wave.”

…

Velocity Flow Space(Trainable Flow Model !!"! )

…

[Figure 64]

'%&+ (% )/-0,)

[Figure 65]

…

DiT-Block

Original Conditional Prediction

#!"#$:

Trainable Flow Model !#$#

[Figure 66]

Training Process Guidance Process Flow Moving

“A dog standing on a surfboard, riding a wave.”

'%&+ (% )/-0,)

ℒ(()*+

(a) (b) (c)

- Figure 3. Overview of our PureCC. (a). We first fine-tune a flow model on the custom set as representation extractor. (b). During the pure learning stage, the representation extractor remains frozen and provides the target concept representation, which is then controlled by our adaptive scale λ⋆ to implicitly guide the trainable model. The trainable model is initialized from another pre-trained flow model and provides original conditional prediction using the Base Text as input. The entire pipeline is trained on the custom set using LPureCC and LCC. (c). demonstrates the process of using our designed LPureCC to purely learn the target concept in the velocity flow space.

of the target concept to the original conditional prediction. Therefore, we define the learning objective as a combination of the original model component and the target concept component:

where ycomplete = {Ycompletel }Ll=1. By introducing tunable concept embeddings at different layers, the representa-

tion extractor can capture more detailed textures of the target concept, leading to a more comprehensive understanding. Notably, these embeddings will be preserved and used to replace the corresponding concept embeddings during the subsequent learning stage.

vtPureCC = vtoriginal + λ · vttarget, (6)

where λ is a guidance scale like w in Eq. 5. This decoupled form enables the model to substantially focus on the original model while learning the target concept.

#### 4.3. Pure Learning Pipeline in PureCC

We present our pure learning pipeline in Fig. 3 (b). This pipeline utilizes a dual-branch architecture comprising: (1) a frozen representation extractor vθ

#### 4.2. Representation Extractor

t (·) sourced from Sec. 4.2; (2) a trainable model initialized from another pretrained flow model vθ

Existing methods use all custom language-vision data as a training source. However, due to the scarcity of reference images, they fail to decouple the target concept representation from the custom set as guidance for fine-tuning. To alleviate this issue, PureCC first designs a representation extractor as shown in Fig. 3 (a), which treats a generative model as the backbone because of its powerful text-image understanding ability. Specifically, we employ LoRA [18] to fine-tune a pre-trained flow model vθ

1

t (·) to purely learn the target concept. Implicit Guidance of the Target Concept. In the frozen branch, to alleviate the disruption of the original behavior, the extractor endeavors to provide a relatively pure representation of the target concept, serving as implicit guidance for the trainable branch. Specifically, based on our extractor’s deep understanding of the target concept, we separately input the Target Text and the null condition “∅” into the extractor. By subtracting their prediction outputs, we obtain the representation bias R(ytar), which contains abundant information related to the target concept, as our vttarget in the learning objective:

2

t (·) on custom set.

1

Layer-Wise Tunable Concept Embeddings. To further enhance the model’s understanding of personalized concept, we introduce layer-wise tunable concept embeddings {Ytarl }Ll=1 for each transformer layer, where L denotes the total number of layers. These tunable embeddings replace the original embedding of [V] in the input prompt embeddings. For example, the prompt “A photo of a [V] dog” is transformed into “A photo of a [Vl] dog” in each layer. Thus, the complete textural embedding for the l-th layer is:

vttarget = R(ytar) = vθ

t (xt|ytar) − vθ

t (xt|∅), (8)

1

1

where ytar = {Ytarl }Ll=1 denotes the textural condition with the layer-wise target concept embeddings.

Original Conditional Prediction. In the trainable branch, the flow model takes an additional input ybase to predict the corresponding velocity field vθ

Ycompletel = [ybase;Ytarl ]. (7) Subsequently, the model is optimized using the loss in Eq. 3, i.e., LRepCC = Et,x

t (xt|ybase). Due to vθ

2

2 2

0,x1 (x1 − x0) − vθ

,

t (xt|ycomplete)

t (xt|ybase) sufficiently representing the performance of

1

2

[Figure 67]

A [V] cat lounging on a floating bookshelf in a magical library, with a soft golden glow from enchanted lanterns.

consistent with the guidance, λ⋆ will decrease to reduce the focus on the target concept and avoid contaminating the original model. Conversely, if the trainable model has learned the direction of the target concept relatively well, λ⋆ will increase to reinforce the learning of the target concept. This adaptive mechanism balances target concept fidelity and original model preservation.

KL Divergence to Original Distribution: ! = 1.0: 1.42×10-2; ! = 3.0: 1.46×10-2 ! = 5.0: 3.44×10-2; !*(Adaptive): 1.92×10-2

[Figure 68]

[Figure 69]

[Figure 70]

HPSv2.1: 29.6

HPSv2.1: 27.4

HPSv2.1: 29.8

[Figure 71]

CLIP-T: 31.6

CLIP-T: 31.8

CLIP-T: 32.5

# = 1.0 # = 3.0

Original

[Figure 72]

[Figure 73]

[Figure 74]

HPSv2.1: 29.9

HPSv2.1: 24.17

Overall Loss. Our learning objective in Eq. 6 is refined as:

CLIP-T: 28.44

CLIP-T: 30.9

# = 5.0 #∗(Adaptive)

Custom Set

###### +λ⋆ · vtθ1(xt|ytar) − vtθ1(xt|∅)

vtPureCC = vtθ2(xt|ybase)

###### .

- Figure 4. Motivation of Adaptive Scale λ⋆. A small λ can preserve the original model’s behavior and capabilities but leads to a decrease in the fidelity of the target concept. Conversely, when λ is excessively large, the personalized concept dominates the learning objective, causing the final distribution to drift away from the original distribution. This results in a degradation of the model’s generative ability: the underlying prompt cannot be followed and lower CLIP-T and HPSv2.1 scores.

original

target

(13)

And the training loss based on this objective is:

2 2

LPureCC = Et,xt vtPureCC − vtθ2 xt|ycomplete

, (14)

while Fig. 3 (c) illustrates the optimization process of this loss in the velocity flow space. As flow matching loss is responsible for predicting the velocity field and preserving the generative prior, we combine the loss in Eq. 3, i.e.,

the original model, we employ it as vtoriginal to substantially consider the original model:

2 2

0,x1 (x1 − x0) − vθ

, with our proposed LPureCC. Therefore, the overall loss is:

LCC = Et,x

t (xt|ycomplete)

2

vtoriginal = vθ

t (xt|ybase). (9)

2

#### 4.4. Adaptive Guidance Scale λ⋆

LPCC = LCC + η · LPureCC, (15)

Although the objective in Eq. 6 is effective for purely learning personalized concepts, its performance relies on an unbounded empirical parameter λ, which controls the guidance strength of the target concept. Improper tuning of this scale leads to undesirable artifacts, as illustrated in Fig. 4. To balance this trade-off, we propose an adaptive mechanism to dynamically find the optimal λ. Specifically, the target concept representation used as the guidance can be expressed as the representation bias R(ytar) of the frozen model vθ

where η is the hyperparameter for regularization strength. Finally, as shown in Alg. 1, the training process enables PureCC to achieve pure learning for personalized concepts while minimizing the impact on the original model’s behavior and capability.

Algorithm 1 PureCC Training Pipeline

Require: Initialize flow model vtθ1(·); custom Set D

- 1: Training for Representation Extractor
- 2: for training iteration k = 1 to K do
- 3: Sampling (x, ycomplete) in D
- 4: Encode text prompts with layer-wise tunable embeddings: {Ycompletel }Ll=1 in Eq. 7
- 5: Adapt the flow matching loss LRepCC in Eq. 3 to optimize
- 6: Update θ1 via LoRA fine-tuning
- 7: end for

Require: Initialize learnable model vtθ2(·); Freeze the representation ex-

tractor vtθ1(·); custom Set D

- 8: Pure Learning for Personalized Concept
- 9: for training iteration k = 1 to K do
- 10: Sampling (x, ytar, ybase, ycomplete) in D
- 11: Compute implicit guidance of target concept vttarget = R(ytar) in Eq. 8
- 12: Compute original conditional predictions and complete conditional predictions: vtoriginal = vtθ2(xt|ybase), and vtθ2(xt|ycomplete)
- 13: Compute PureCC learning objective: vtPureCC in Eq. 13
- 14: Compute the adaptive scaling factor: λ⋆ in Eq. 12
- 15: Compute the PureCC loss: LPureCC in Eq. 14
- 16: Optimize overall loss: LPCC in Eq. 15
- 17: Update θ2
- 18: end for

t (·). Meanwhile, in the trainable model vθ

t (·), we can acquire a similar form which represents the learned target concept representation:

2

1

R(ycomplete, ybase) = vtθ2(xt|ycomplete) − vtθ2(xt|ybase).

(10)

Then, we obtain the adaptive scale λ⋆ by minimizing the projection error between the learned representation R(ycomplete,ybase) in the trainable model and the guidance representation R(ytar) in the frozen model:

∥R(ycomplete,ybase) − λ · R(ytar)∥22 (11) By differentiating, it can yield a closed-form solution:

λ⋆ = arg min

λ

λ⋆ = ⟨R(ycomplete,ybase),R(ytar)⟩ ∥R(ytar)∥2

, (12)

where <,> denotes the inner product of the corresponding representation. λ⋆ serves as the projection coefficient of R(ycomplete,ybase) on R(ytar) Intuitively, during training, if the trainable model has not yet learned the direction

[Figure 75]

[Figure 76]

CIFC

###### Base Text

A [V1] robot floating on the water, surrounded by bubbles and rubber ducks

Instance Concept: [V1] robot

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

A robot floating on the water, surrounded by bubbles and rubber ducks.

[Figure 87]

- Instance Concept: [V2] dog

- Instance Concept: [V3] man

A [V2] dog standing on a surfboard riding a wave, bright sunny day.

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

“A dog standing on a surfboard riding a wave, bright sunny day.”

[Figure 98]

A [V3] man sitting under a big oak tree reading a book, sunlight filtering through the leaves

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

“A man sitting under a big oak tree reading a book, sunlight filtering through the leaves”

[Figure 109]

A magical forest with glowing mushrooms and fairies, in [V4] style

- Style Concept: [V4] style

- Style Concept: [V5] style

- Style Concept: [V6] style

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

“A magical forest with glowing mushrooms and fairies.”

[Figure 120]

A knight in silver armor riding a horse across a bridge made of light in [V5] style

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

“A knight in silver armor riding a horse across a bridge made of light.”

[Figure 131]

An elderly man sitting on a bench under a tree with golden leaves that shimmer like stars, in [V6] style

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

An elderly man sitting on a bench under a tree with golden leaves that shimmer like stars.”

[Figure 142]

Original DreamBooth

Custom Set

DreamBooth+EWC Mix-of-Show DreamO UNO

CIFC

PureCC (Ours)

- Figure 5. Qualitative Comparison with SOTAs including Tuning-based methods: DreamBooth [36], DreamBooth + EWC [39], Mix-ofShow [13], CIFC [9], and Tuning-free methods: DreamO [31] UNO [46].

### 5. Experiments

Evaluation Metrics. For target concept fidelity, we employ CLIP-I (target) [33] and DINO [2] similarity to measure the consistency between generated images and the custom set for instance-level concepts, and CSD [41] (a CLIP-based style encoder) for style consistency. We introduce additional preservation metrics to quantify the custom model about the preservation of the original model’s capability, including ∆CLIP-T (base) for text alignment, ∆HPSv2.1 and ∆PickScore for quality and aesthetic [25] preservation. Where we report differential metrics ∆M = Mcustom(I(ycomplete)) − Moriginal(I(ybase)), M denotes a base metric (e.g. CLIP-T [33], HPSv2.1 [47], PickScore [21]) and I(ybase) denotes generated image conditional on base text. The smaller ∆M indicates better preservation. Seg-Cons [20] measures segmentation consistency between outputs of the custom model and the original model under the Complete text and Base text respectively, reflecting behavior preservation.

#### 5.1. Experimental Setup

Dataset. To ensure a fair qualitative evaluation with previous methods, we select 14 personalized concepts from the dataset proposed by DreamBooth [36]. Furthermore, to evaluate the adaptability of our method in broader scenar-

- ios, we additionally construct a batch of images containing 16 personalized concepts, covering both commonly used instance concepts (e.g., Pikachu, Yann LeCun) and style concepts (e.g., cartoon, sketch). For a comprehensive quantitative evaluation, we create DreamBenchPCC, which extends the DreamBench [36] benchmark by adding an image set with 12 additional style concepts.

Implementation Details. We adopt SD 3.5-M [10] as our base model. For a fair comparison, Tuning-based baselines such as DreamBooth [36], B-LoRA [11], LoRA-S [53], and Mix-of-Show [13] use the same pretrained backbone. Following previous works [9, 53], we set the LoRA rank to 4 and use a learning rate of 1.0×10−4 to update both the flow model and the layer-wise tunable embeddings.

#### 5.2. Qualitative Evaluation

Single-Concept Customization. We compare our method with representative baselines across both instance and style

- Table 1. Quantitative Comparison Results on DreamBenchCC. Since UNO and DreamO are Tuning-free methods that do not require fine-tuning the base model, our comparison for them focuses mainly on their concept responsiveness.

Instance Style

Method

Preservation Concept Responsiveness Preservation Concept Responsiveness ∆ CLIP-T (base) (↑) ∆ HPSv2.1 (↑) ∆ PickScore (↑) Seg-Cons (↑) CLIP-I (target) (↑) DINO (↑) ∆ CLIP-T (base) (↑) ∆ HPSv2.1 (↑) ∆ PickScore (↑) CSD (↑)

Dreambooth [36] -4.81 -2.17 -3.90 18.38 0.63 0.62 -6.23 -2.08 -1.83 0.57 Dreambooth + EWC [39] -4.17 -2.20 -3.16 26.37 0.62 0.61 -7.90 -2.04 -1.57 0.60 Mix-of-Show [13] -2.71 -1.08 -2.63 15.72 0.72 0.61 -4.93 -1.63 -1.73 0.62 CIFC [9] -1.93 -1.62 -2.08 13.23 0.78 0.65 -4.70 -1.21 -1.25 0.64

DreamO [31] - - - - 0.71 0.67 - - - 0.65 UNO [46] - - - - 0.69 0.62 - - - 0.34

###### Ours (PureCC) -0.31 +0.10 -0.67 69.37 0.81 0.73 -0.26 -0.92 -0.59 0.63

[V1] man [V2] sunglasses

A [V1] man wearing a red scarf with [V2] sunglasses.

ment, where the adaptation of one concept unintentionally alters the appearance or context of another (e.g., color contamination between [V1] man and [V2] sunglasses, or structural distortion between [V3] pikachu and [V4] lighthouse. In contrast, our method preserves the independence of each learned concept while integrating them coherently into a single composition. This demonstrates that our PureCC enables pure multi-concept customization and effectively mitigates cross-concept interference.

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

A huge [V3] toy in the distance by the [V4] lighthouse.

[V3] pikachu [V4] lighthouse

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

Original Mix-of-show LoRA-S PureCC (Ours)

Custom Set

- Figure 6. Qualitative comparison in Multi-Concept Customization with Mix-of-show [13], LoRA-S [53].

B-LoRA CIFC DreamO PureCC (Ours)

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

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

Instance Style A [V1] dog playing in a blooming garden, vibrant flowers, in [V2] style.

|A dog playing in a blooming garden, vibrant flowers.|
|---|

[V2] style A [V1] dog playing in a blooming garden, vibrant flowers, in [V3] style.

[V1] dog [V3] style

[V1] dog

Original

- Figure 7. Qualitative comparison of style–instance customization across different methods, including CIFC [9], B-LoRA [11], DreamO [31]. B-LoRA is a tuning-based approach specifically designed for balancing style and content adaptation. Each case combines an instance concept with a specific style.

Style-Instance Customization. To further evaluate our capability in composing heterogeneous concepts such as instance and style, we conduct experiments on cross-domain customization scenarios. As shown in Fig. 7, our method achieves a more balanced style transfer, faithfully preserving the object structure while accurately rendering the custom artistic style. In contrast, existing tuning-based approaches tend to overfit the style or distort object identity.

Analysis of Predictions during Pure Learning. We intuitively visualize the Pure Learning process and its learning guidance. As shown in Fig. 8, the prediction x0complete, evolves progressively during training: initially it is similar to xoriginal0 and gradually moves toward the the xPureCC0 which both preserves the original model’s behavior and successfully expresses the target concept. This process demonstrates that the objective LPureCC purely incorporates the target concept while preserving the original content. Overall, this visualization reveals that PureCC enables an additive and pure integration of new concepts, rather than disrupting the original model’s generative behavior.

customization tasks. As illustrated in Fig. 5, the baseline methods often fail to preserve the original model’s behavior and capabilities after learning new concepts. For instance, DreamBooth and Mix-of-Show exhibit severe inconsistency with the original behavior and alter global composition and background textures. In contrast, our method achieves pure concept learning, which accurately adapts to new concepts while preserving non-target attributes such as background, lighting, and pose. These results show that our PureCC effectively produce high-fidelity customization without sacrificing the original model’s behavior and capabilities.

#### 5.3. Quantitative Evaluation

Tab. 1 presents the quantitative comparison of our method and existing approaches on the DreamBenchPCC. Across both instance and style concept customization, our method consistently achieves superior performance in Preservation and Concept Responsiveness metrics. In the Preservation aspect, our method attains the smallest gaps in ∆CLIPT(base), ∆HPSv2.1, and ∆PickScore, indicating that it best preserves the original model’s capability, including semantic alignment, aesthetic quality, and human preference. Furthermore, the high Seg-Cons score (69.37) demonstrates that our approach maintains the spatial and structural consistency of the original model’s outputs, effectively miti-

Multi-Concept Customization. PureCC encourages a disentangled and purified representation of each concept, allowing different customized concepts to remain relatively independent without semantic interference. To validate this, we compare our method with several tuning-based approaches under multi-concept personalization settings. As shown in Fig. 6, tuning-based methods such as Mix-ofShow and LoRA-S often suffer from semantic entangle-

Custom Set Original + Target Original Ours (iter = 100) Ours (iter = 200) Ours (iter = 300) Ours (iter = 400)

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

"!!"#$%&& "!!'$()(*+, "!!-'./,%0%

Pure Learning Process

- Figure 8. Visualization of Pure Learning Process. xPureCC0 denotes the images obtained by integrating the velocity field {vtPureCC}Tt=1. Similarly, xoriginal0 and xcomplete0 are based on {vtoriginal}Tt=1 and {vtcomplete}Tt=1, respectively. “iter” denotes the training iteration.

- Table 2. Ablation Study on the Pure Learning. “Merged Learning Stage” refers to the training setting where the first-stage Representation Extractor vtθ1 and the second-stage Pure Learning of vtθ2 are conducted jointly.

Preservation Concept Responsiveness

Strategy

∆ CLIP-T (base) (↑) ∆ HPSv2.1 (↑) ∆ PickScore (↑) Seg-Cons (↑) CLIP-I (target) (↑) DINO (↑)

LCC -4.52 -2.01 -2.95 23.74 0.65 0.66 Merged Training Stage -1.17 -0.34 -1.08 54.37 0.50 0.41

###### LCC +LPureCC -0.31 +0.10 -0.67 69.37 0.81 0.73

[V1] backpack A [V1] backpack on a wooden desk, in a classroom.

solely LCC, the integration of the PureCC loss (LCC + LPureCC) leads to substantial enhancements in all preservation metrics including ∆CLIP-T(base), ∆HPSv2.1, and ∆PickScore, while maintaining strong concept responsiveness in CLIP-I(target) and DINO. These results validate that the PureCC objective effectively prevents the degradation of prior knowledge during fine-tuning, thereby preserving both the behavior and capability. In the ”Merged Learning Stage” setting, although joint optimization preserves the original model, the representation extractor does not adequately learn the target concept representation. As a result, the guidance becomes under-expressive, leading to a significant decline in the fidelity of the target concept.

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[V2] toy

A red [V2] toy on a windowsill in sunlight.

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

Original Custom Set w/o !!"#$% % Merged Learning Stage PureCC

Figure 9. Visualization of the Ablation Study.

Table 3. Ablation Study of the λ⋆.

Adaptive λ⋆. The quantitative results further validate the issues shown in Fig. 4, demonstrating the necessity of our adaptive scale. As shown in Tab. 6, using a fixed λ leads to clear limitations. When λ = 1 is too small, the concept guidance becomes insufficient, resulting in weak adaptation to the target concept with noticeably lower CLIP-I (0.43) and CSD (0.26) scores. Increasing λ strengthens concept responsiveness, but simultaneously harms preservation, as reflected by larger degradation in ∆CLIP-T(base).

Instance Style

Strategy

∆CLIP-T (base) (↑) CLIP-I (target) (↑) ∆CLIP-T (base) (↑) CSD (↑)

λ = 1.0 -0.18 0.43 -0.67 0.26 λ = 3.0 -0.51 0.58 -0.93 0.61 λ = 5.0 -2.67 0.73 -4.21 0.42

λ = λ⋆ (adaptive) -0.31 0.81 -0.26 0.63

gating behavioral disruption commonly observed in tuningbased methods such as DreamBooth and CIFC. In terms of Concept Responsiveness, our approach achieves competitive or superior scores on CLIP-I(target), DINO, and CSD, suggesting that we can accurately express both instance and style personalized concepts without compromising generative fidelity. These results validate that PureCC effectively integrates new concepts in a stable and semantically aligned manner, achieving state-of-the-art overall performance.

### 6. Conclusion

PureCC effectively addresses the challenge of preserving the original model’s behavior and capabilities while achieving high-fidelity concept customization. By introducing a decoupled learning objective and a dual-branch training pipeline, PureCC ensures pure learning for personalized concepts. The adaptive guidance scale λ⋆ further enhances the balance between customization fidelity and model preservation. Extensive experiments demonstrate that PureCC outperforms existing methods in maintaining the original model while enabling concept customization.

#### 5.4. Ablation Study

Pure Learning. To confirm the importance of both the loss design and the two-stage training strategy in our PureCC, we perform quantitative and qualitative ablation studies in Tab. 2 and Fig. 9 Compared to the baseline, which optimizes

### 7. Acknowledgement

This work was supported by the National Key Research and Development Program of China (Grant No. 2022YFB3303101), the National Natural Science Foundation of China (Grant No. 62276170), and the Guangdong Provincial Key Laboratory (Grant No. 2023B1212060076).

### References

- [1] Minghong Cai, Xiaodong Cun, Xiaoyu Li, Wenze Liu, Zhaoyang Zhang, Yong Zhang, Ying Shan, and Xiangyu Yue. Ditctrl: Exploring attention control in multi-modal diffusion transformer for tuning-free multi-prompt longer video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7763–7772, 2025. 3
- [2] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 6, 12
- [3] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG), 42(4):1–10, 2023. 3
- [4] Jinshu Chen, Xinghui Li, Xu Bai, Tianxiang Ma, Pengze Zhang, Zhuowei Chen, Gen Li, Lijie Liu, Songtao Zhao, Bingchuan Li, et al. Omniinsert: Mask-free video insertion of any reference via diffusion transformer models. arXiv preprint arXiv:2509.17627, 2025. 2
- [5] Hyungjin Chung, Jeongsol Kim, Geon Yeong Park, Hyelin Nam, and Jong Chul Ye. Cfg++: Manifold-constrained classifier free guidance for diffusion models. arXiv preprint arXiv:2406.08070, 2024. 3
- [6] Geoffrey Cideron, Andrea Agostinelli, Johan Ferret, Sertan Girgin, Romuald Elie, Olivier Bachem, Sarah Perrin, and Alexandre Ram´e. Diversity-rewarded cfg distillation. arXiv preprint arXiv:2410.06084, 2024. 3
- [7] Yusuf Dalva, Hidir Yesiltepe, and Pinar Yanardag. Lorashop: Training-free multi-concept image generation and editing with rectified flow transformers. arXiv preprint arXiv:2505.23758, 2025. 3
- [8] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 3
- [9] Jiahua Dong, Wenqi Liang, Hongliu Li, Duzhen Zhang, Meng Cao, Henghui Ding, Salman H Khan, and Fahad Shahbaz Khan. How to continually adapt text-to-image diffusion models for flexible customization? Advances in Neural Information Processing Systems, 37:130057–130083, 2024. 3, 6, 7
- [10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis.

- In Forty-first International Conference on Machine Learning, 2024. 2, 3, 6, 12
- [11] Yarden Frenkel, Yael Vinker, Ariel Shamir, and Daniel Cohen-Or. Implicit style-content separation using b-lora. In European Conference on Computer Vision, pages 181–198. Springer, 2024. 2, 6, 7, 12
- [12] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 3
- [13] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized lowrank adaptation for multi-concept customization of diffusion models. Advances in Neural Information Processing Systems, 36:15890–15902, 2023. 2, 3, 6, 7, 12
- [14] Zinan Guo, Yanze Wu, Zhuowei Chen, Lang Chen, Peng Zhang, and Qian He. Pulid: Pure and lightning id customization via contrastive alignment. arXiv preprint arXiv:2404.16022, 2024. 3
- [15] Qiyuan He, Jinghao Wang, Ziwei Liu, and Angela Yao. Aid: Attention interpolation of text-to-image diffusion. arXiv preprint arXiv:2403.17924, 2024. 3
- [16] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 3
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [18] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 1, 2, 3, 4
- [19] Jiannan Huang, Jun Hao Liew, Hanshu Yan, Yuyang Yin, Yao Zhao, Humphrey Shi, and Yunchao Wei. Classdiffusion: More aligned personalization tuning with explicit class guidance. arXiv preprint arXiv:2405.17532, 2024. 3
- [20] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 6, 12
- [21] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation.

2023. 6, 12

- [22] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1931–1941, 2023. 1, 3
- [23] Xinghui Li, Qichao Sun, Pengze Zhang, Fulong Ye, Zhichao Liao, Wanquan Feng, Songtao Zhao, and Qian He. Anydressing: Customizable multi-garment virtual dressing via latent diffusion models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 23723–23733. IEEE, 2025. 2

- [24] Zhichao Liao, Fengyuan Piao, Di Huang, Xinghui Li, Yue Ma, Pingfa Feng, Heming Fang, and Long Zeng. Freehand sketch generation from mechanical components. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 6755–6764, 2024. 3
- [25] Zhichao Liao, Xiaokun Liu, Wenyu Qin, Qingyu Li, Qiulin Wang, Pengfei Wan, Di Zhang, Long Zeng, and Pingfa Feng. Humanaesexpert: Advancing a multi-modality foundation model for human image aesthetic assessment. arXiv preprint arXiv:2503.23907, 2025. 6, 16
- [26] Ente Lin, Xujie Zhang, Fuwei Zhao, Yuxuan Luo, Xin Dong, Long Zeng, and Xiaodan Liang. Dreamfit: Garment-centric human generation via a lightweight anything-dressing encoder. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 5218–5226, 2025. 2
- [27] Yang Lin, Xinyu Ma, Xu Chu, Yujie Jin, Zhibang Yang, Yasha Wang, and Hong Mei. Lora dropout as a sparsity regularizer for overfitting control. arXiv preprint arXiv:2404.09610, 2024. 3
- [28] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 2
- [29] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 3
- [30] Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subjectdiffusion: Open domain personalized text-to-image generation without test-time fine-tuning. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 3
- [31] Chong Mou, Yanze Wu, Wenxu Wu, Zinan Guo, Pengze Zhang, Yufeng Cheng, Yiming Luo, Fei Ding, Shiwen Zhang, Xinghui Li, et al. Dreamo: A unified framework for image customization. arXiv preprint arXiv:2504.16915,

2025. 6, 7, 12

- [32] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 1

- [33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 1, 3, 6, 12
- [34] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020. 3
- [35] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2
- [36] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven

- generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 1, 2, 3, 6, 7, 12, 13
- [37] Seyedmorteza Sadat, Otmar Hilliges, and Romann M Weber. Eliminating oversaturation and artifacts of high guidance scales in diffusion models. In The Thirteenth International Conference on Learning Representations, 2024. 3
- [38] Seyedmorteza Sadat, Manuel Kansy, Otmar Hilliges, and Romann M Weber. No training, no problem: Rethinking classifier-free guidance for diffusion models. arXiv preprint arXiv:2407.02687, 2024. 3
- [39] Joan Serra, Didac Suris, Marius Miron, and Alexandros Karatzoglou. Overcoming catastrophic forgetting with hard attention to the task. In International conference on machine learning, pages 4548–4557. PMLR, 2018. 6, 7
- [40] Enis Simsar, Thomas Hofmann, Federico Tombari, and Pinar Yanardag. Loraclr: Contrastive adaptation for customization of diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13189–13198,

2025. 2, 3

- [41] Gowthami Somepalli, Anubhav Gupta, Kamal Gupta, Shramay Palta, Micah Goldblum, Jonas Geiping, Abhinav Shrivastava, and Tom Goldstein. Measuring style similarity in diffusion models. arXiv preprint arXiv:2404.01292, 2024. 6, 12
- [42] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2
- [43] Claude 3.5 Sonnet. Claude 3.5 sonnet. https://www. anthropic . com / news / claude - 3 - 5 - sonnet,

2024. 12

- [44] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. p+: Extended textual conditioning in text-toimage generation. arXiv preprint arXiv:2303.09522, 2023. 3
- [45] Zirui Wang, Zhizhou Sha, Zheng Ding, Yilin Wang, and Zhuowen Tu. Tokencompose: Text-to-image diffusion with token-level supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8553–8564, 2024. 3
- [46] Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-to-more generalization: Unlocking more controllability by in-context generation. arXiv preprint arXiv:2504.02160, 2025. 6, 7, 12
- [47] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341,

2023. 1, 6, 12

- [48] Xiaole Xian, Zhichao Liao, Qingyu Li, Wenyu Qin, Pengfei Wan, Weicheng Xie, Long Zeng, Linlin Shen, and Pingfa Feng. Spf-portrait: Towards pure text-to-portrait customization with semantic pollution-free fine-tuning. arXiv preprint arXiv:2504.00396, 2025. 3
- [49] Guangxuan Xiao, Tianwei Yin, William T Freeman, Fr´edo Durand, and Song Han. Fastcomposer: Tuning-free multisubject image generation with localized attention. Interna-

- tional Journal of Computer Vision, 133(3):1175–1194, 2025. 3
- [50] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 3

- [51] Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan, et al. Ssr-encoder: Encoding selective subject representation for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8069–8078, 2024. 3
- [52] Yang Zhang, Teoh Tze Tzun, Lim Wei Hern, and Kenji Kawaguchi. Enhancing semantic fidelity in text-to-image synthesis: Attention regulation in diffusion models. In European Conference on Computer Vision, pages 70–86. Springer, 2024. 3
- [53] Ming Zhong, Yelong Shen, Shuohang Wang, Yadong Lu, Yizhu Jiao, Siru Ouyang, Donghan Yu, Jiawei Han, and Weizhu Chen. Multi-lora composition for image generation. arXiv preprint arXiv:2402.16843, 2024. 3, 6, 7, 12
- [54] Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Xiangyang Zhu, Fu-Yun Wang, Zhanyu Ma, et al. Lumina-next: Making lumina-t2x stronger and faster with next-dit. Advances in Neural Information Processing Systems, 37:131278–131315, 2024. 3

## PureCC: Pure Learning for Text-to-Image Concept Customization Supplementary Material

### 8. Dataset Details

To ensure a fair Qualitative Evaluation with previous methods, we selected 14 personalized concepts from the dataset proposed by DreamBooth [36]. Some samples can be seen in Fig. 10. Furthermore, to assess the adaptability of our method across a wider range of scenarios, we additionally collected a batch of novel personalized concepts, which includes 11 commonly used instance concepts, such as Pikachu and Yann LeCun, as well as 5 style concepts, such as cartoon and sketch. Some samples can be seen in Fig. 11. Thus, we constructed a Qualitative Evaluation dataset comprising a total of 30 personalized concepts. For comprehensive Quantitative Evaluation, we created DreamBenchPCC, which extends DreamBench [36] with 12 additional style concepts to balance the proportion of instance and style concepts. Some style samples can be seen in Fig. 12. We used the state-of-the-art large multi-modal model Claude 3.5 Sonnet [43] to caption all newly collected images.

### 9. More Implementation Details

We perform training on an NVIDIA A100 GPU with a batch size of 2. For each personalized concept, both the representation extractor vθ

t and the trainable model vθ

t are trained in 400 steps. All images are generated using the default inference setting of 28 timesteps.

1

2

### 10. Evaluation Metrics Details

Since we are working with a new task setting—Pure Concept Customization—we specifically applied representative metrics to suit our task setting for quantitative evaluation.

Fidelity of the personalized concept. For instance-level concepts, we employ CLIP-I (target) [33] and DINO [2] to evaluate the similarity between the target concept in the generated images and the target concept in the reference images from the custom set. For style-level concepts, we use CSD [41] (a CLIP-based style encoder) to evaluate style consistency.

Original model preservation. To evaluate the custom model’s preservation of the original model’s capabilities, we report differential metrics ∆M = Mcustom(I(ycomplete)) − Moriginal(I(ybase)), where M denotes a base metric (e.g. CLIP-T [33], HPSv2.1 [47], PickScore [21]). Mcustom(I(ycomplete)) represents the metric score of the image generated using the custom model and the Complete text, and Moriginal(I(ybase)) represents the metric score of the image generated using the original

model and the Base text. Thus, the smaller ∆M indicates better preservation. We specifically use ∆CLIP-T(base) to assess the ability to follow Base text, as well as ∆HPSv2.1 and ∆PickScore to evaluate the retention of the ability to generate high-quality and aesthetically pleasing images. Moreover, we use Seg-Cons [20] to measures segmentation consistency between outputs of the custom model and the original model under the Complete text and Base text respectively, reflecting original behavior preservation.

### 11. Qualitative and Quantitative Evaluation Details.

We performed qualitative evaluations on our personalized dataset, which has been expanded to include new instance and style concepts. This dataset comprises a total of 30 personalized concepts, as shown in Fig. 10 and Fig. 11. For a comprehensive quantitative evaluation, we utilized DreamBenchPCC, as shown in Fig. 12 For a fair comparison, tuning-based baselines such as DreamBooth [36], B-LoRA [11], LoRA-S [53], and Mix-of-Show [13] are all trained using the same pretrained backbone, SD 3.5M [10]. For tuning-free baselines such as DreamO [31] and UNO [46], we follow their standard usage and provide one reference image from the customization set with prompts to enable personalized generation. For stylization cases, we prepend the prompts with the instruction “Generate a same-style image” to ensure consistent style conditioning. Since tuning-free methods do not require fine-tuning the pre-trained model, they fundamentally differ from our task setting, which addresses the damage to the original model caused by adapting the pre-trained model to learn personalized concepts during fine-tuning. Therefore, when comparing with them, we primarily focus on evaluating their outstanding performance in target concept fidelity.

We provided more qualitative evaluation results in Fig. 13 and Fig. 14.

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

Figure 10. Some samples selected from the dataset proposed by DreamBooth [36].

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

- Figure 11. Some samples additionally collected by us.

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

[Figure 240]

[Figure 241]

- Figure 12. Some style samples in our DreamBenchPCC.

[Figure 242]

[Figure 243]

[Figure 244]

Bamboo forest shrouded in morning fog, a single red lantern hanging from a gatepost, soft light glowing within, in [V] style

A [V] man sitting on a wooden floor with a blanket around their shoulders, book open on their lap, a cat curled against his leg.

A [V] woman playing a violin, eyes closed, bow gliding slowly, dust motes swirling in the sunbeam that cuts through the empty concert hall

A [V] woman sitting at a grand piano, fingers hovering above the keys.

A [V] toy abandoned on a rainy porch step, raindrops pooling around its feet

A [V] man is cutting the meat in the kitchen.

A portrait of a woman, in [V] style

A kid sleeping with a [V] toy in his arms

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

Original

Original

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 257]

[Figure 259]

Custom Set

Custom Set

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

DreamBooth

DreamBooth

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 275]

[Figure 276]

DreamBooth +EWC

DreamBooth +EWC

[Figure 278]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

Mix-of-Show

Mix-of-Show

[Figure 285]

[Figure 286]

[Figure 288]

[Figure 289]

[Figure 292]

CIFC

CIFC

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 298]

DreamO

DreamO

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

UNO

UNO

[Figure 309]

[Figure 310]

[Figure 312]

[Figure 315]

[Figure 316]

PureCC (Ours)

PureCC (Ours)

###### Figure 13. More qualitative evaluation results

Bamboo forest shrouded in morning fog, a single red lantern hanging from a gatepost, soft light glowing within, in [V] style

A [V] man sitting on a wooden floor with a blanket around their shoulders, book open on their lap, a cat curled against his leg.

A [V] woman playing a violin, eyes closed, bow gliding slowly, dust motes swirling in the sunbeam that cuts through the empty concert hall

A [V] woman sitting at a grand piano, fingers hovering above the keys.

A [V] toy abandoned on a rainy porch step, raindrops pooling around its feet

A [V] man is cutting the meat in the kitchen.

A portrait of a woman, in [V] style

A kid sleeping with a [V] toy in his arms

[Figure 317]

[Figure 318]

[Figure 322]

[Figure 323]

[Figure 324]

Original

Original

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

Custom Set

Custom Set

[Figure 334]

[Figure 335]

[Figure 338]

[Figure 339]

[Figure 340]

DreamBooth

DreamBooth

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

DreamBooth +EWC

DreamBooth +EWC

[Figure 349]

[Figure 351]

[Figure 352]

[Figure 355]

[Figure 356]

Mix-of-Show

Mix-of-Show

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 362]

[Figure 363]

CIFC

CIFC

[Figure 366]

[Figure 367]

[Figure 369]

[Figure 371]

[Figure 372]

DreamO

DreamO

[Figure 373]

[Figure 374]

[Figure 376]

[Figure 379]

[Figure 380]

UNO

UNO

[Figure 383]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

PureCC (Ours)

PureCC (Ours)

###### Figure 14. More qualitative evaluation results

### 12. Computational Cost

Compared with previous approaches, our method introduces an additional training stage and employs an additional model branch in the Pure Learning (Stage-2) phase, which inevitably increases training time and GPU memory usage. To clarify, as shown in Tab. 4, we emphasize that: 1) Although an extra training stage is required, completely training a single personalized concept using PureCC only takes 0.33 A100 hours, which remains highly efficient. 2) While the dual-branch design in Pure Learning indeed requires additional memory, in practice we only need to load one main network (DiT) along with the LoRA modules corresponding to vθ

t (·) and vθ

t (·). Therefore, the overall GPU memory consumption does not increase significantly.

1

2

Table 4. Computation Time and Memory Usage of Training under BFloat16 datatype.

Representation Extrator (stage-1) Pure Learning (stage-2) Memory Cost 28GB 30GB Training Time 0.13 (A100 Hour) 0.2 (A100 Hour)

Tab. 5 compares the computational cost of training and inference with the baselines, showing that our training remains efficient. During inference, we only need to use the single model vθ

t (·), thus incurring no additional overhead over the baselines.

2

Table 5. Comparisons on a single NVIDIA A100 GPU

| |DreamBooth LoRA Mix-of-Show CIFC|PureCC (Ours)|
|---|---|---|
|Training<br><br>|58.0G / 0.25h 28.0G / 0.13h 32.0G / 0.23h 31.0G / 0.28h<br><br>|30.0G / 0.33h|
|Inference<br><br>|17.4G / 4.46s 17.8G / 4.72s 18.4G / 5.80s 18.0G / 5.48s|17.8G / 4.72s|

### 13. Analysis of Hyperparameter η in LPCC

Since our Pure Concept Customization loss LPCC introduces a weighting parameter η to modulate the pure learning loss LPureCC, we further analyze the sensitivity of η. As shown in the qualitative results in Fig. 15 and the quantitative comparison in Tab. 6, an excessively large η leads to over-injection of the target concept, harming visual fidelity, whereas an overly small η causes the model to be dominated by LCC, resulting in the degradation of the original model’s behavior and capabilities.

### 14. Analysis of the Original Conditional Prediction vtoriginal

In the main paper, we employ vtoriginal = vθ

t (xt|ybase), treating the output of the trainable model as the original conditional prediction. A more effective and intuitive strategy would be to use vtoriginal = vθ

2

t (xt|ybase), i.e., obtain the original conditional prediction directly from another frozen pre-trained model. However, this approach would

3

introduce an additional model into the stage of Pure Learning, resulting in a non-negligible increase in computational cost. Therefore, we evaluate whether vθ

t (xt|ybase) can reliably approximate the performance of the original model. As shown in Fig. 16, the effect of pure learning achieved when the Original Conditional Prediction uses vθ

2

t (xt|ybase) from the trainable model closely matches the effect of introducing vθ

2

t (xt|ybase) from an additional frozen pre-trained model. Thus, with virtually no loss in generation quality, we adopt vθ

3

t (xt|ybase) as the Original Conditional Prediction throughout our Pure Learning pipeline because it is more computationally efficient.

2

### 15. Ablation Study Details

We provide a detailed explanation of our ablation study here. To demonstrate the effectiveness of our proposed novel learning objective, we designed an ablation experiment for LPureCC. This experiment involves fine-tuning using only the traditional LCC for concept customization and comparing it with fine-tuning using our complete learning objective LPCC (LCC + LPureCC). Furthermore, to demonstrate the effectiveness of our training strategy—specifically, first obtaining a target concept representation extractor through fine-tuning on the custom set and then keeping it frozen during the pure learning stage to stably provide purified target concept representations. We designed a comparative experiment: the ‘Merged Learning Stage’ experiment. In the ‘Merged Learning Stage’ experimental setup, we do not pre-fine-tune the representation extractor on the custom set. Instead, we directly use a pre-trained flow model as the representation extractor vθ

t . During the pure learning stage, it remains trainable, allowing it to simultaneously learn the target concept and provide target concept representations for the pure learning branch vθ

1

t . By comparing this jointly conducted approach with our proposed method, we can demonstrate the effectiveness of dividing our training into two distinct stages.

2

### 16. User Study

Besides qualitative and quantitative comparisons, to thoroughly evaluate our method, we carried out a user study to determine whether our method is preferred by humans for pure concept customization. We engaged 42 participants from diverse social backgrounds, with each test session lasting approximately 30 minutes. During the investigation, as illustrated in Fig. 17, participants conducted pairwise comparisons between our method and competing approaches across four dimensions: (1) Original Behavior Consistency, (2) Base Text Alignment, (3) Aesthetic Preference [25], and (4) Target Concept Fidelity. For “Original Behavior Consistency,” users were asked to select which of the two images better maintained consistency with the

Original Custom Set != !"# != ! != !"# != !"#

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

A [V] clock hanging from a tree branch by a chain, twilight sky behind

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

A [V] toy floating in zero gravity inside a giant glass snow globe, snowflakes frozen mid-fall around it

[Figure 403]

[Figure 404]

Figure 15. Qualitative Analysis of the η.

Table 6. Quantitative of the η.

Instance Style

Strategy

∆CLIP-T (base) (↑) CLIP-I (target) (↑) ∆CLIP-T (base) (↑) CSD (↑)

- η = 0.5 -1.32 0.55 -2.02 0.49

- η = 1.0 -0.31 0.81 -0.26 0.63

- η = 1.5 -0.43 0.80 -0.26 0.60
- η = 2.0 -0.76 0.67 -0.41 0.39

Trainable Model Pre-trained Model

Original Custom Set

A [V] car on a rocky mountain road, with the sun glinting off its stainless steel surface and distant cliffs in the background.

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

A [V] bridge carved from stone, ancient and cracked, leading into a fog-covered forest

[Figure 415]

[Figure 416]

Figure 16. Results compared to which Original Conditional Prediction is provided by the Trainable model vtθ2(xt|ybase) or another Frozen pre-trained model vtθ3(xt|ybase) .

original model’s outputs, disregarding the insertion of personalized concepts. For “Base Text Alignment,” users were tasked with choosing which of the two images more accurately align with the base text’s description. For “Aesthetic Preference,” users determined which image better matched

their aesthetic preferences, taking into account factors such as visual quality and the absence of artifacts or distortions. For “Target Attribute Fidelity,” users assessed which image more accurately generated visual content that resembled the target concepts in the custom set. The results, as shown in

Original CompleteText Method-1 Method-2

CustomSet

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

A [V1] robot floating on the water, surrounded by bubbles and rubber ducks

###### The 13th of 138 question

[Figure 422]

Original Behavior Consistency ( the better method to keep consistent with original model’s behavior)

Base Text Alignment (the better method to adhere to the base text prompt)

Aesthetic Preference ( the better method to align your aesthetic standard)

Target Concept Fidelity ( the better method to match the target concept)

Original CustomSet CompleteText Method-1 Method-2

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

A knight in silver armor riding a horse across a bridge made of light in [V5] style

###### The 14th of 138 question

[Figure 428]

Original Behavior Consistency ( the better method to keep consistent with original model’s behavior)

Base Text Alignment (the better method to adhere to the base text prompt)

Aesthetic Preference ( the better method to align your aesthetic standard)

Target Concept Fidelity ( the better method to match the target concept)

###### Original CustomSet CompleteText Method-1 Method-2

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

A [V2] dog standing on a surfboard riding a

###### The 15th of 138 question

[Figure 434]

wave, bright sunny day

Original Behavior Consistency ( the better method to keep consistent with original model’s behavior)

Base Text Alignment (the better method to adhere to the base text prompt)

Aesthetic Preference ( the better method to align your aesthetic standard)

Target Concept Fidelity ( the better method to match the target concept)

Pre 4 5 6 … 35 Next TurntoPage

###### Figure 17. The investigation page in user study.

Table 7. The results of User Study. (PureCC vs SOTAs)

PureCC (Ours) DreamBooth PureCC (Ours) DreamBooth+EWC PureCC (Ours) Mix-of-Show PureCC (Ours) CIFC Original Behavior Consistency 98.5% 1.5% 96.9% 3.1% 91.5% 8.5% 94.6% 5.4% Base Text Alignment 66.2% 33.8% 62.3% 37.7% 55.4% 44.6% 58.5% 41.5% Aesthetic Preference 71.9% 28.1% 73.5% 26.5% 64.3% 35.7% 56.4% 43.6% Target Attribute Fidelity 67.7% 32.3% 75.4% 24.6% 52.3% 47.7% 54.6% 45.4%

Tab. 7, demonstrate that our method significantly improves the preservation of the original model’s behavior and capabilities while also achieving customization effects for the target concept that are comparable to those of existing methods focused on fidelity to personalized concepts. This un-

derscores our strong capability to maintain the integrity of the original model while seamlessly adapting to new concepts. This comprehensive evaluation framework guarantees a thorough and objective assessment of our method’s performance compared to existing approaches.

[Figure 436]

[Figure 439]

18

- Instance Concept: [V2] dog

- Instance Concept: [V3] man

A [V3] man sitting under a big oak tree reading a book, sunlight filtering through the leaves

A magical forest with glowing mushrooms and fairies, in [V4] style

- Style Concept: [V4] style

- Style Concept: [V5] style

- Style Concept: [V6] style

An elderly man sitting on a bench under a tree with golden leaves that shimmer like stars, in [V6] style

