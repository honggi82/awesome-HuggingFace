# arXiv:2511.17282v1[cs.CV]21Nov2025

## Where Culture Fades: Revealing the Cultural Gap in Text-to-Image Generation

Chuancheng Shi1* Shangze Li2* Shiming Guo1* Simiao Xie1† Wenhua Wu1† Jingtong Dou1 Chao Wu2 Canran Xiao3 Cong Wang4 Zifeng Cheng4 Fei Shen5‡ Tat-Seng Chua5 1The University of Sydney 2Nanjing University of Science and Technology 3Central South University 4Nanjing University 5National University of Singapore

##### Abstract

Multilingual text-to-image (T2I) models have advanced rapidly in terms of visual realism and semantic alignment, and are now widely utilized. Yet outputs vary across cultural contexts: because language carries cultural connotations, images synthesized from multilingual prompts should preserve cross-lingual cultural consistency. We conduct a comprehensive analysis showing that current T2I models often produce culturally neutral or English-biased results under multilingual prompts. Analyses of two representative models indicate that the issue stems not from missing cultural knowledge but from insufficient activation of culturerelated representations. We propose a probing method that localizes culture-sensitive signals to a small set of neurons in a few fixed layers. Guided by this finding, we introduce two complementary alignment strategies: (1) inferencetime cultural activation that amplifies the identified neurons without backbone fine-tuned; and (2) layer-targeted cultural enhancement that updates only culturally relevant layers. Experiments on our CultureBench demonstrate consistent improvements over strong baselines in cultural consistency while preserving fidelity and diversity.

##### 1. Introduction

Ensuring cultural fairness and representation in generative AI [5, 10, 13, 17, 20, 23, 36, 40, 42] is essential for global accessibility and cultural diversity, aligning with the United Nations’ principles of inclusiveness and universality. Yet, when prompted in different languages, many state-of-theart methods [1, 7, 22, 34, 48] frequently produce culturally neutral or English-biased images, which weakens crosslingual cultural correspondence. Here, we use cultural consistency to denote the extent to which generated images ex-

*Equal first authors. †Equal second authors. ‡Corresponding author.

(a) Cross-lingual Semantic–Cultural Alignment and T2I Decoupling

[Figure 1]

[Figure 2]

(ii) Recommender System

###### (i) Text to Image Model

[Figure 3]

[Figure 4]

Рекомендовані пам'ятки (uk)

[Figure 5]

###### (pt) Um edifício tradicional

(zh) 一顿传统早餐

(ja) おすすめのアトラクション

[Figure 6]

(A traditional building)

(A traditional breakfast)

###### (Recommended attractions)

(Recommended attractions)

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

|[Figure 11]<br><br>[Figure 12]<br><br>TOP SEARCH RESULT|
|---|

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Найкращі фотографії пам’яток культури України (uk)

###### (ja)一度は見る価値がある日本の絶景20選

(The best photos of cultural monuments of Ukraine)

(20 breathtaking views in Japan that are worth seeing at least once)

[Figure 17]

###### (iii) Large Language Model

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

###### (de) Ein Frühstück

###### (tr) geleneksel bir bina

[Figure 22]

[Figure 23]

###### (pt)O que é um pequeno-almoço tradicional?

(zh)传统早餐中都有什么?

(A traditional breakfast)

(A traditional building)

(What is in a traditional breakfast?)

[Figure 24]

[Figure 25]

(What is in a traditional breakfast?)

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

(zh) 中国各地早餐风格丰富多样:北方：豆浆配油条、煎饼

(pt)Um pequeno-almoço tradicional português consiste

果子,南方：肠粉、烧卖、虾饺、米粉,西南地区：小面、

geralmente em pão fresco, café, leite e, por vezes, sumo de laranja natural, bolos caseiros ou croissants.

粉蒸肉、豆花饭等。

(Breakfast styles vary across China: in the north: soy milk with

(A traditional Portuguese breakfast usually consists of fresh

fried dough sticks and pancakes; in the south: rice noodle rolls, siomai, shrimp dumplings, and rice noodles; in the southwest:

bread, coffee, milk, and sometimes freshly squeezed orange juice,

homemade cakes, or croissants.)

small noodles, steamed pork with rice flour, and tofu rice.)

(b) Explicit cultural cues alignment

[Figure 33]

###### (th)เคร่ืองดนตรีวัฒนธรรมไทย

###### (it)architettura italiana

(ja)和食ディナー (Japanese dinner)

###### (de)Deutsches Kunsthandwerk

###### (zh)穿着中式服装的人

(An Italian architecture)

(German handicrafts)

(Thai cultural musical instruments)

(People wearing Chinese clothes)

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Figure 1. Cultural alignment in local languages. (a) LLMs/recommenders keep cultural consistency, but T2I models falter with “noun-only” prompts. (b) Adding a “culture-style modifier + noun” restores consistency.

hibit visual elements that are statistically associated with the target language’s cultural context, beyond mere semantic correctness. To avoid conflating culturally typical elements with stereotypes, we require that cultural cues be both contextually appropriate and roughly consistent with real-world cultural statistics.

Despite strong gains in semantic and visual fidelity, diffusion-based T2I models [35, 37] lag in cultural generalization. Prior work largely targets cross-lingual encoder alignment [27, 48] while overlooking cross-cultural grounding, defined as generating visuals that reflect each language’s socio-cultural context. As shown in Figure 1(a), models often capture only literal meanings, for example, “a traditional building” in Portuguese or Turkish, while missing culture-specific cues. In contrast, language-based systems such as LLMs [1, 30, 41] and recommender engines [16, 32] produce localized responses, revealing a cross-modal gap in cultural grounding.

We argue that the issue is insufficient activation rather than missing knowledge. Large-scale training corpora

[Figure 43]

results for you.Okay, I will generate the

Al Annotated Caption

Task Proposal

This is a picture in [xxxx] style. Please give me a brief description of what is in the picture. It is necessary to clarify the cultural style.

Task

文

A

Search for the linguistic features of fifteen countries and compile the data.

AI Annotation Caption

[Figure 44]

###### Review Completed

Translate

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

This is a Spanish-style building with a solemn and symmetrical facade, featuring towering

beginning

twin towers and exquisite stone carvings.

[Figure 60]

[Figure 61]

###### Spanish

Este es un edificio de estilo español con una fachada solemne y simétrica, quecuenta con imponentes torres

Loading.......

| |
|---|

gemelas y exquisitas tallas en piedra.

###### Organize pictures

Human Annotation Caption

Arab

Manual Search Information

[Figure 62]

Japan China

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Este es un edificio con una fachada solemne y simétrica que presenta imponentes torres gemelas y exquisitas tallas en piedra.

Human add "noun" annotations.

###### Browser, provide us with cultural images from fifteen different countries.

Ukraine

France

[Figure 67]

[Figure 68]

[Figure 69]

A 文

Okay.....this...... is the result......

Italy

Spain

This is a building with a solemn and symmetrical facade featuring towering twin towers and exquisite stone carvings.

[Figure 70]

[Figure 71]

Poland

[Figure 72]

[Figure 73]

[Figure 74]

- Figure 2. Overview of the CultureBench pipeline. First, manually collect and rigorously quality-control datasets from 15 linguistic regions; annotate “culture-style modifier noun” captions using GPT5-Nano [30] and, through human annotation, “noun-only” captions; convert annotated content into local languages via translation tools, supplemented by manual review.

already contain diverse cultural attributes, and explicit prompts can elicit them. As shown in Figure 1(b), adding culture-style modifiers, for example, “people wearing Chinese clothes” or “an Italian architecture,” yields images with clear country-specific characteristics. The inconsistency happens because “noun-only” prompts don’t strongly trigger cultural knowledge, so the model gives literal but not cultural interpretations.

produce culturally neutral or subtly English-biased outputs under multilingual prompts, thereby hindering crosslingual cultural consistency.

- • We develop a probing framework that localizes culturesensitive signals to a few fixed layers and neurons by contrasting attention patterns and Top-K SAE activations, indicating that failures stem from insufficient activation rather than missing knowledge.
- • We propose two alignment strategies: a zero-training inference-time activation scheme and a layer-specific fine-tuning scheme that updates only culturally relevant layers, improving cultural consistency while preserving fidelity and diversity.
- • We curate and release CultureBench, a 15-country benchmark with multilingual prompts and images, enabling evaluation of cross-cultural consistency and culturally adaptive training.

To validate this hypothesis and mitigate cultural inconsistency in T2I models, we first examine two representative systems [27, 48] and show that the failure arises from insufficient activation of culture-related knowledge rather than its absence, and that the effect is observed across two architecturally different diffusion methods. We then introduce a two-stage probing method. We begin by comparing attention distributions between culture-style modifiers and nouns to localize culture-sensitive layers. Next, we use the Top-K SAE [4] to quantify activation differences between prompts with explicit cultural cues and “noun-only” prompts, revealing that culture-relevant representations cluster in a few fixed layers and a small set of neurons. Guided by these findings, we propose two complementary strategies: (1) a zero-training activation scheme that amplifies the responses of the identified neurons at inference time, and (2) a layerspecific fine-tuned scheme that updates only culturally relevant layers to improve consistency. Finally, experiments on our CultureBench show consistent improvements over strong baselines in cultural consistency while preserving fidelity and diversity. We highlight the following contributions:

##### 2. Related Work

Cultural Text-to-Image Generation. Prior work has examined cultural bias, fairness, and stereotyping in T2I systems [9, 18, 26]. For example, SCoFT [25] expands cultural coverage through CCUB to mitigate stereotyping, and ViSAGe [19] quantifies visual stereotypes. However, existing studies remain largely focused on fairness and debiasing, with a strong English-centric orientation and limited coverage of low-resource languages and diverse cultural contexts [6, 39]. Clear definitions and systematic evaluation of cross-lingual cultural consistency are still lacking. Noun-only or short prompts often collapse to culturally neutral or implicitly English-biased generations in

• We empirically show that multilingual T2I models often

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

- Figure 3. Data distribution of the proposed CultureBench dataset across 15 languages. The dataset is divided into train, test, and neuron-detection subsets with a ratio of 7:2:1.

multilingual settings, yet a unified diagnostic framework is absent [24]. At the representational level, prior work provides little insight into where culture-sensitive features reside within the model or how to control them at the layer or neuron level [21, 31]. Methodologically, there is still a lack of lightweight, plug-and-play interventions and corresponding benchmarks that do not require large-scale retraining.

Neuron Interpretability. Neuron-level interpretability links model activations to learned concepts in vision and language [2, 15, 29, 49]. Recent work further distinguishes shared and language-specific semantics at the neuron or direction level, clarifying how abstract meanings distribute across layers [14, 41, 45]. In multimodal settings, causal probing shows that individual neurons can be driven by visual concepts via textual proxies and directly manipulated to control semantics [11, 33, 38]. For concreteness, FEMN [33] locates sparse neuron groups mediating specific concepts (e.g., “smile,” “striped”) in a CLIP-style model and demonstrates that directly patching or amplifying those units causally increases or suppresses the target concept in zero-shot classification and retrieval, establishing neuronlevel semantic control. However, prior studies still focus mainly on object- or attribute-level notions and provide limited diagnostics for cultural representations, especially those that vary under multilingual prompts [8, 28, 43].

##### 3. CultureBench Dataset

Data Collection. As shown in Figure 2, we collect culturally representative images across 15 language and region groups via geo-constrained web search using native and translated keywords. Each image is manually verified for clarity, authenticity, and representativeness to ensure a

[Figure 91]

Figure 4. Verify the hypothesis. Within the CultureBench test subset, performances under “culture-style modifier + noun” and “noun-only” prompt conditions are compared. Quantitative evaluation is conducted using CultureVQA.

faithful depiction of its target culture.

Data Annotations. CultureBench provides multidimensional annotations for assessing cultural awareness, including cultural categories, geographic regions, languages, and image content. For each sample, we provide two textual descriptions: a “culture-style modifier + noun” caption generated by GPT5-Nano [30] and a human-written “noun-only” caption. To reduce subjectivity in defining and labeling cultural attributes, we invited domain experts to review the taxonomy and a subset of samples. Annotators kept only culturally appropriate, statistically grounded cues and treated mismatched ones as stereotypes.

Data Distribution. As shown in Figure 3, the dataset contains 7,932 samples split into train, test, and neurondetection subsets at a 7:2:1 ratio: the train subset is used for model and adapter training, the test subset for quantitative evaluation, and the neuron-detection subset for layerand neuron-level probing. Training is strictly limited to the train subset. Neither the test subset nor the neuron-detection subset is accessed during training, hyperparameter tuning, or model selection.

Cultural Evaluation. We propose CultureVQA, a singlechoice VQA task built on CultureBench. For each image, Qwen3-VL [47] and Gemini-2.5-Flash [3] must choose one label from 15 language/region categories or an “unrecognisable” option. We report accuracy as the proportion of samples whose predicted label exactly matches the ground truth. In a pilot study, we found that the cultural labels predicted by these models are highly consistent with human annotations, supporting their use as automatic evaluators. This setup tests a model’s ability to perform cultural attribution from visual cues alone, without textual prompts.

##### 4. Cultural Probing

###### 4.1. Cultural Gap

To test the hypothesis that the issue lies in insufficient activation rather than knowledge absence, we conducted a uni-

###### (a) Culture Layer Detect

###### Culture-style modifiers + nouns

Text Encoder Block Attention CA (Culture)

[Figure 92]

(zh) 一个穿着中式服装的人

|[Figure 93]|
|---|
|[Figure 94]|
|[Figure 95]|

∆CA

Culture Layer

###### A person wearing Chinese clothing

TextEmbedding

… (de) Deutsche traditionelle Box German traditional box

Layer1

Layer2

Layer3

Layern

#### …

Attention

MLP

Nouns

CA (Nouns)

…

(zh) 一个穿着服装的人 A person wearing clothing … (de) traditionelle Box

|[Figure 96]|
|---|

traditional box

###### (b) Culture Neuron Detect

Culture-style modifiers + nouns

| |
|---|
| |
| |
| |
| |
| |
| |

WFSWFScultnoun

(zh) 一个穿着中式服装的人

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

###### A person wearing Chinese clothing

TextEmbedding

|𝜀|
|---|

FFnouncult

… (de) Deutsche traditionelle Box German traditional box

CultureLayer

[Figure 97]

Attention

cultural neurons

Layer1

Layer2

Layern

Layer3

MLP

| |
|---|
| |
| |

…

…

| |
|---|
| |
| |
| |
| |
| |
| |

|Nouns| |
|---|---|
|(zh) 一个穿着服装的人<br><br>A person wearing clothing<br><br>…<br><br>(de) traditionelle Box traditional box| |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

|𝜀|
|---|

Top-K

Text Encoder

Figure 5. Methods for Neuronal Detection. (a) By comparing attention allocation between cultural-style modifiers and nouns across text-encoder layers, the layer with the largest divergence is designated as the culturally sensitive layer. (b) At this layer, features from the “culture-style modifier + noun” and “noun-only” prompts are fed into an SAE [4] to obtain sparse features, revealing neurons with heightened sensitivity to cultural cues.

fied controlled experiment. Specifically, we used prompts composed of “culture-style modifier + noun” as inputs to AltDiffusion [48] and PEA-Diffusion [27], generated corresponding images, and evaluated them using the CultureVQA accuracy metric. As a control, we repeated the same process with “noun-only” prompts. As shown in Figure 4, prompts combining “culture-style modifier + nouns” achieved the best CultureVQA performance, 44.39 for AltDiffusion and 35.62 for PEA-Diffusion, both substantially higher than the “noun-only” prompts. This consistent trend across two structurally different diffusion models suggests that the phenomenon is not model-specific and provides empirical support for our hypothesis.

###### 4.2. Culture Layer Detection

As illustrated in Figure 5(a), we compare the attention directed towards the target noun under paired “culturestyle modifier + noun” and “noun-only” prompts, averaging across heads to obtain hierarchical cultural relevance scores. “Culture-style modifier + noun” prompts consistently yield significantly higher scores than “noun-only” prompts, indicating that they are encoded with cultural semantics.

We define paired prompts for each target concept: a “culture-style modifier + noun” prompt Pcult and a “nounonly” prompt Pnoun. The former augments the noun with a culture-style modifier, while the latter removes it. We create N such pairs covering diverse cultural elements to ensure generality.

For each prompt, we annotate two token categories, the culture modifiers Tcult and the target nouns Tnoun. And ex-

[Figure 98]

Figure 6. PEA-Diffusion cultural sensitivity. ∆CA peaks layer

16. AltDiffusion results are provided in the appendix.

tract layerwise attention. Let A(l) ∈ RB×H×S×S be the multi-head attention at layer l, where B is the batch size, H is the number of attention heads, and S is the sequence length. We average heads for robustness:

H

1 H

A¯(l) =

Ah(l), (1)

h=1

and retain only the entries from Tcult to Tnoun. Subsequently, we derive the subset of keyword pairs from A¯(l) denoted as A¯key(l) ∈ RB×|T

cult|×|Tnoun|.

If layer l encodes culture semantics, the attention from culture modifiers to target nouns under a cultural prompt should exceed that under its “noun-only” prompt. To quantify this, let tcult ∈ Tcult and tnoun ∈ Tnoun denote a cultural modifier and target noun. For a prompt P, the attention from cultural modifiers to target nouns in layer l is:

[Figure 99]

Figure 7. Neuronal detection result. The weighted frequency scores show only a few salient peaks per culture, indicating culture-specific neurons. We define the Top-K set as the peak neurons, with K adapting to the number of salient peaks.

###### A¯key(l)t

cult→tnoun |Tcult| · |Tnoun|

###### CA(P,l) = t

cult tnoun

. (2)

For all culture/noun pairs (Pcult,Pnoun), we compute

N

1 N

[CA(Pcult,i,l) − CA(Pnoun,i,l)]. (3)

∆CA(l) =

i=1

A Larger ∆CA(l) indicates that layer l more effectively separates “culture-style modifier” from “noun” semantics.

We compute ∆CA(l) across prompt pairs and seeds, and mark a layer as culture-sensitive if its ∆CA notably exceeds the mean of its two neighboring layers. Based on this workflow, Figure 6 shows the detection results from PEADiffusion, revealing clear global peaks at a specific layer.

The results indicate that culturally relevant semantics are not uniformly distributed throughout the network, but rather concentrated in a key layer.

###### 4.3. Culture Neuron Detection

We localize culturally sensitive neurons in the key layer. As shown in Figure. 5(b), we apply a Top-K SAE [4] to obtain a decomposition of attention features and select culturespecific neurons via comparative analysis.

From the key layer, we construct cultural features Fcult ∈ RN

noun×Datt, where Ncult and Nnoun are the numbers of token pairs for the cultural and noun prompts, respectively. The attention feature dimension is Datt = |Tcult| × |Tnoun|. We use a weighted frequency score combining activation frequency and amplitude to quantify neuronal responses to cultural semantics. Specifically, let m ∈ {1,2,...,Datt} indexes a specific neuron in the attention feature space. The activation frequency fcult(m) is defined as follows: for all Ncult lexemes with cultural elements, compute the proportion of samples in which the activation Zcult[i,m] of the mth neuro exceeds ϵ:

cult×Datt and noun features Fnoun ∈ RN

1 Ncult

fcult(m) =

Ncult

I(Zcult[i,m] > ϵ). (4)

i=1

Table 1. Validating Culture-Sensitive Neuron Detection in PEA-Diffusion. Neuronal accuracy on the test subset with “cultural style modifier + noun” prompts.

|Method<br><br>|CultureVQA ↑|
|---|---|
|PEA-Diffusion [27]<br><br>+ Masked Top-K Neurons<br><br>+ Masked Random Neurons|35.62 7.65 (-27.97) 33.04 (-2.58)<br><br>|

Here i denotes the ith sample in Ncult, I(·) is the indicator function.Using the same method, we obtained fnoun(m).

For each neuron, we define the mean activation magnitude µcult(m) by calculating the magnitude of its activation in the cultural attention feature and adding it to the magnitude of its activation in the noun attention feature token:

Ncult i=1 (Zcult[i,m] · I(Zcult[i,m] > ϵ))

. (5)

µcult(m) =

Ncult i=1 I(Zcult[i,m] > ϵ) + β

Here β denotes a small positive constant added for numerical stability, ensuring that the denominator never becomes zero and preventing degenerate cases when the activation frequency is extremely low. Using the same method, we obtained µnoun(m). µnoun(m) denotes the average activation magnitude of the mth neuron on the noun subset Fnoun.

Ultimately, to better capture neurons with both high firing rates and strong responses, we combine the activation frequency of cultural modifiers with their average activation magnitude to obtain a weighted frequency score WFScult:

WFScult(m) = fcult(m) · µcult(m). (6)

Using the same method, we obtained WFSnoun(m). WFSnoun(m) is the weighted frequency score for noun subsets. After computing WFScult and WFSnoun, we rank neurons by WFScult and select the top-K candidates. Neurons with substantial noun-side salience are removed, and the remaining ones are designated as culturally sensitive neurons.

According to the aforementioned algorithm, as illustrated in Figure 7, six distinctly different cultural backgrounds are presented; in each instance, one or more pronounced peaks emerge, each bearing a different index value.

This indicates that the locations of corresponding cultural neurons do not overlap across cultures.

###### 4.4. Cultural Validation

To verify whether our detector accurately localizes culturesensitive neurons, we designed three controlled settings (Table 5): (1) Baseline (no masking), (2) Masked Top-K Neurons (masking the Top-K culture-sensitive neurons identified by our method), and (3) Random Mask (masking the same number of neurons at random). Masking the identified Top-K neurons reduces the mean CultureVQA score

from 35.62 to 7.65, whereas random masking yields a comparable score (33.04) to the baseline. The sharp, targeted degradation absent under random masking indicates that the localized neurons are highly related to cultural semantics, thereby providing empirical support for the accuracy and interpretability of our neuron localization method.

##### 5. Methods

###### 5.1. Zero-Training Neuron Amplifier

Upon identifying neurons sensitive to cultural information, we amplify underutilised cultural representations by intervening on a targeted neuronal subset Mcult within the key layer, thereby modulating attention-related features and enhancing cultural attributes in the generated images.

We define the original attention correlation features that need to be interfered Fraw ∈ RB×S

pair×Datt where B is the batch size, Spair is the number of prompt words in a single batch. The latent vector Zraw ∈ RB×S

pair×Mcult after inputting them into the SAE encoder is given by:

Zraw = SAE.encode(Fraw), (7)

where the SAE, through sparse coding, decomposes the complex internal representations into more independent and semantically coherent neurons. At the same time, Zraw[b,p,m] denotes the initial activation value of the m-th neuron for the p-th prompt word in the b-th batch. We enhance cultural representation by modulating culturally specific neurons with manually defined λ. The formula is:

(1 + λ)Zraw[b,p,m] if m ∈ Mcult Zraw[b,p,m] otherwise

.

Zenh[b,p,m] =

(8) Here Zenh ∈ RB×S

pair×M is the modulated latent vector , M is the collection of all neurons. λ ∈ R is the feature fusion coefficient. We then map Zenh back to the attention space:

Frec enh = SAE.decode(Zenh), (9)

where SAE.decode(·) represents the SAE decoder and Frec enh ∈ RB×S

pair×Datt is the attention association feature reconstructed after modulation. This step preserves the original semantic structure while enhancing culturally specific attention patterns.

###### 5.2. Fine-Tuned Layer Enhancer

To enable adaptive cultural representation without the need for manual adjustment of modulation strength, we propose a layer-specific, fine-tuned scheme that updates only culturally relevant layers to improve consistency. We first analyze the text encoder layer by layer to identify the layer most sensitive to cultural cues, denoted as lc. A small trainable module is inserted only into this layer, while all other parameters remain frozen.

Let h denote the hidden representation of the text encoder at layer lc. The enhancer produces an enhanced representation h˜ via a residual transformation:

###### h˜ = h + g W2 σ(W1h) , (10)

where h˜ is the culture-enhanced hidden state, σ(·) is a nonlinear activation, g(·) denotes a normalization layer used to stabilize the residual transformation, and W1, W2 are small trainable matrices. During training, only the enhancer parameters are updated while keeping the backbone fixed. Given a “noun-only” prompt p, the text encoder with enhancer fθ,ϕ and generator G produce an image:

###### xˆ = G fθ,ϕ(p) . (11)

The generated image is compared with a ground-truth cultural image x∗(p), which is directly taken from the CultureBench dataset as the human-curated cultural reference corresponding to the “noun-only” prompt p, using a pixellevel mean squared error (MSE) loss:

N

1 N

LMSE =

i=1

x ˆi − x∗i (p) 22, (12)

where N is the number of pixels. Only the enhancer parameters are optimized:

ϕ∗ = arg min

LMSE. (13)

ϕ

During inference, the trained layer enhancer modulates the hidden representation h˜ according to the input prompt, thereby enhancing cultural consistency in the generated images while maintaining the original semantic structure.

##### 6. Experiments And Analysis

###### 6.1. Implementation Details

Metrics. Following PEA-Diffusion [27], we employ CLIPScore [12] and ImageReward [46] to evaluate text-image alignment and perceptual preference. Concurrently, we employ LPIPS [50] to measure perceptual similarity and reconstruction fidelity, while introducing CultureVQA as a novel metric for cultural recognition.

Hyperparameters. In Fine-Tuned Layer Enhancer method, we train for 2,000 steps under mixed precision using AdamW (learning rate 5 × 10−5) with a batch size of 1. For the zero-training variant, we set λ = 6. All experiments are conducted on a single NVIDIA A6000 GPU.

###### 6.2. Compare with SOTA Methods

We conduct a quantitative comparison between our zerotraining and fine-tuned built on PEA-Diffusion, and several state-of-the-art (SOTA) methods, including AltDiffusion [48], PEA-Diffusion [27], StableDiffusion XL [34],

Table 2. Quantitative comparisons with SOTA methods. Using the “noun-only” prompts on the test subset. The best performance is marked in bold, and the second-best is underlined.

###### Method CultureVQA ↑ CLIPScore ↑ ImageReward ↑ LPIPS ↓

StableDiffusion XL [34] 9.36 0.211 -1.82 0.756 FLUX.1-dev [22] 14.83 0.224 -0.88 0.692 Show-o2 [44] 16.43 0.234 -0.91 0.691 PEA-Diffusion [27] 21.65 0.253 -0.65 0.673 AltDiffusion [48] 23.05 0.282 -0.11 0.688 StableDiffusion 3.5 [7] 25.13 0.242 -1.01 0.715

Ours (Zero-Training) 33.91 (+12.32) 0.291 (+0.038) 0.33 (+0.98) 0.654 (-0.019) Ours (Fine-Tuned) 36.63 (+14.98) 0.290 (+0.037) 0.31 (+0.42) 0.661 (-0.012)

###### (ko) 흰색 도자기 그릇

###### (th) อาคารแบบดั้งเดิม

(zh) 传统建筑

###### (de) Box aus Metall

###### (ru) Серебряный молочник

###### (tr) Barbekü restoranı

###### (uk) традиційна пляшка

###### (nl) beschilderd porselein

(it) Costumi popolari

(ja) 豪華なディナー

###### (ar) ايودي زرطم شامق

(White porcelain bowl)

(Traditional building)

(Hand embroidered fabric)

(Box made of metal)

(Silver milk jug)

(BBQ restaurant)

(Traditional bottle)

(Painted porcelain)

(Folk costumes)

(Luxurious dinner)

(Traditional building)

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

StableDiffusionXLStableDiffusion3.5FLUX.1-devShow-o2Ours

Fail

Fail

Fail

Fail

Fail

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

Fail Fail

Fail

Fail

Fail

Fail

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Fail Fail Fail Fail

Fail

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

AltDiffusionPEA-Diffusion

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

Fail

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

Fail

Fail

Fail

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

Figure 8. Qualitative comparison of generation results. Our approach generates images that are more culturally appropriate.

StableDiffusion 3.5 [7], FLUX.1-dev [22], and Show-

- o2 [44]. This evaluation measures semantic alignment and visual fidelity, providing a comprehensive benchmark for cross-cultural image generation.

Quantitative Results. To evaluate whether our cultural alignment strategies enhance cultural understanding with-

- out compromising semantic or visual quality, we designed a quantitative comparison experiment (Table 2). The results show that our fine-tuned model achieves a CultureVQA score of 36.63, significantly outperforming AltDiffusion (23.05) and PEA-Diffusion (21.65). The zero-training variant attains the highest CLIPScore (0.291) and ImageReward (0.33), while maintaining a competitive LPIPS. Therefore,

our method achieves simultaneous gains in cultural understanding, text-image consistency, and perceptual quality, verifying that interpretable cultural alignment can be realized without sacrificing visual fidelity.

Qualitative Results. To test whether our method can recover cultural features under “noun-only” prompts without losing semantic consistency, we conducted qualitative comparisons with several mainstream models (Figure 8). The results show that our approach consistently generates images that reflect the target region’s cultural characteristics while preserving semantic accuracy, whereas other models often regress to culture-neutral generic prototypes. For example, under prompts such as “Box made of metal” or

[Figure 168]

- Figure 9. User Study. Evaluated using MCC, SCC, and CSR metrics, where higher scores indicate greater perceived realism and user preference.

Original Image 𝝀 = 𝟎 𝝀 = 𝟏 𝝀 = 𝟐 𝝀 = 𝟑 𝝀 = 𝟒 𝝀 = 𝟓 𝝀 = 𝟔 𝝀 = 𝟕 𝝀 = 𝟖

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

традиционныйчайникru

raditionalkettle)T

(zh)传统建筑

raditionalbuilding)T

(ja)豪華なディナー

(A sumptuousdinner)

[Figure 176]

- Figure 10. Hyperparameter results. The effect of cultural enrichment varies under different λ values.

“Silver milk jug”, our method produces culturally grounded depictions, while PEA-Diffusion and AltDiffusion generate correct objects but lack regional style; StableDiffusion XL/3.5 and FLUX.1-dev tend toward neutral aesthetics, and Show-o2 exhibits both cultural inconsistency and semantic drift. Overall, our method balances cultural and semantic coherence while maintaining high image.

User Study. Building on the quantitative and qualitative results, we conducted a human-centered user study on the CultureBench platform with 50 experts in cultural studies. Participants evaluated cultural perception using three metrics: MCC (Multi-Choice Culture), SCC (SingleChoice Culture), and CSR (Cultural Semantic Relevance, 1–5 scale). Higher scores indicate stronger cultural alignment and preference. As shown in Figure 9, our method outperforms all baselines across all three metrics. Notably, the human-rated CSR score reaches 77.6, significantly higher than the second-best score of 60.4, highlighting a clear advantage in cultural semantic fidelity. This human-evaluated CSR metric further strengthens our evaluation pipeline. Overall, these results confirm that our method can generate culturally aligned content accurately and consistently.

###### 6.3. Hyperparameter Sensitivity Analysis

To investigate the impact of hyperparameter λ on the cultural attributes of generated images in zero-shot training, we compared results under different λ values while keeping model parameters and the “noun-only” prompt constant (see Figure 10 and Figure 11). When λ = 0, the output perfectly matched the original image. As λ increases, the images progressively align with the prototypical features of the target culture, accompanied by a corresponding rise in CultureVQA scores. A peak is reached at λ = 7 (35.92), followed by a slight decline at λ = 8 (32.61). This indicates that λ effectively modulates the intensity of cultural consis-

[Figure 177]

Figure 11. Hyperparameter results. Performance variations of CultureVQA under different λ values.

Table 3. Ablation studies on CultureBench. On the CultureBench test subset, we conducted ablation analyses of two methods under both zero-training and fine-tuned settings. The best performance is marked in bold.

Model Method CultureVQA ↑

w/o Ours 23.05

w/ Random (Zero-Training) 20.38 (-2.67) w/ Ours (Zero-Training) 30.06 (+7.01) w/ Random (Fine-Tuned) 21.04 (-2.01) w/ Ours (Fine-Tuned) 32.66 (+9.61)

AltDiffusion [48]

w/o Ours 21.65 w/ Random (Zero-Training) 21.04 (-0.61) w/ Ours (Zero-Training) 33.91 (+12.26) w/ Random (Fine-Tuned) 22.34 (+0.69) w/ Ours (Fine-Tuned) 36.63 (+14.98)

PEA-Diffusion [27]

tency, though excessively high values may induce overfitting and marginally degrade metric performance. Therefore, we select λ = 7.

###### 6.4. Ablation Study

To evaluate the effectiveness of our proposed components and ensure that the performance gains are not caused by random enhancement, we conducted ablation studies on the CultureBench (Table 3). Both the zero-training and finetuned variants exhibit clear improvements: under the zerotraining setting, CultureVQA increases by 7.01 on AltDiffusion and 12.26 on PEA-Diffusion; after fine-tuning, the gains further rise to 9.61 and 14.98, respectively. Random activation or random fine-tuning yields only minimal or even negative improvements. These results confirm that the performance gains stem from the targeted activation of culturally sensitive neurons. Furthermore, both AltDiffusion and PEA-Diffusion exhibit consistent performance improvements, further validating the universality and crossarchitecture effectiveness of our neuron detection and enhancement framework.

##### 7. Conclusion

In this work, we reveal that multilingual T2I models often yield culturally neutral or English-biased images not due to missing knowledge, but due to weak activation of culture-sensitive representations. By localizing culture signals to a small neuron set in a few layers, we introduce two lightweight remedies: inference-time cultural activation and

layer-targeted cultural enhancement. On CultureBench, both strategies consistently improve cross-lingual cultural consistency while preserving fidelity and diversity. This points toward practical, controllable cultural alignment for inclusive generative models.

##### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 1

- [2] Lihu Chen, Adam Dejl, and Francesca Toni. Identifying query-relevant neurons in large language models for longform texts. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 23595–23604, 2025. 3
- [3] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 3
- [4] Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. arXiv preprint arXiv:2309.08600, 2023. 2, 4, 5
- [5] Junyuan Deng, Xinyi Wu, Yongxing Yang, Congchao Zhu, Song Wang, and Zhenyao Wu. Acquire and then adapt: Squeezing out text-to-image model for image restoration. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 23195–23206, 2025. 1
- [6] Moreno D’Inc`a, Elia Peruzzo, Massimiliano Mancini, Dejia Xu, Vidit Goel, Xingqian Xu, Zhangyang Wang, Humphrey Shi, and Nicu Sebe. Openbias: Open-set bias detection in text-to-image generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12225–12235, 2024. 2
- [7] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 1, 7, 14

- [8] Karen Fort, Laura Alonso Alemany, Luciana Benotti, Julien Bezan¸con, Claudia Borg, Marthese Borg, Yongjian Chen, Fanny Ducel, Yoann Dupont, Guido Ivetta, et al. Your stereotypical mileage may vary: Practical challenges of evaluating biases in multiple languages and cultural contexts. In The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LRECCOLING 2024), pages 17764–17769, 2024. 3
- [9] Felix Friedrich, Manuel Brack, Lukas Struppek, Dominik Hintersdorf, Patrick Schramowski, Sasha Luccioni, and Kristian Kersting. Fair diffusion: Instructing text-toimage generation models on fairness. arXiv preprint arXiv:2302.10893, 2023. 2

- [10] Haoyang He, Jiangning Zhang, Hongxu Chen, Xuhai Chen, Zhishan Li, Xu Chen, Yabiao Wang, Chengjie Wang, and Lei Xie. A diffusion-based framework for multi-class anomaly detection. In Proceedings of the AAAI conference on artificial intelligence, pages 8472–8480, 2024. 1
- [11] Qinqin He, Jiaqi Weng, Jialing Tao, and Hui Xue. A single neuron works: Precise concept erasure in text-to-image diffusion models. arXiv preprint arXiv:2509.21008, 2025. 3
- [12] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718,

2021. 6

- [13] Teng Hu, Jiangning Zhang, Ran Yi, Yuzhen Du, Xu Chen, Liang Liu, Yabiao Wang, and Chengjie Wang. Anomalydiffusion: Few-shot anomaly image generation with diffusion model. In Proceedings of the AAAI conference on artificial intelligence, pages 8526–8534, 2024. 1
- [14] Chongxuan Huang, Yongshi Ye, Biao Fu, Qifeng Su, and Xiaodong Shi. From neurons to semantics: Evaluating crosslinguistic alignment capabilities of large language models via neurons alignment. arXiv preprint arXiv:2507.14900,

2025. 3

- [15] Jiahao Huo, Yibo Yan, Boren Hu, Yutao Yue, and Xuming Hu. Mmneuron: Discovering neuron-level domain-specific interpretation in multimodal large language model. arXiv preprint arXiv:2406.11193, 2024. 3
- [16] Andreea Iana, Goran Glavaˇs, and Heiko Paulheim. Mind your language: A multilingual dataset for cross-lingual news recommendation. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 553–563, 2024. 1
- [17] Jinhyeok Jang, Chan-Hyun Youn, Minsu Jeon, and Changha Lee. Rethinking peculiar images by diffusion models: Revealing local minima’s role. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2454–2461, 2024. 1
- [18] Suchae Jeong, Inseong Choi, Youngsik Yun, and Jihie Kim. Culture-trip: Culturally-aware text-to-image generation with iterative prompt refinement. arXiv preprint arXiv:2502.16902, 2025. 2
- [19] Akshita Jha, Vinodkumar Prabhakaran, Remi Denton, Sarah Laszlo, Shachi Dave, Rida Qadri, Chandan K Reddy, and Sunipa Dev. Visage: A global-scale analysis of visual stereotypes in text-to-image generation. arXiv preprint arXiv:2401.06310, 2024. 2
- [20] Chengyou Jia, Minnan Luo, Zhuohang Dang, Guang Dai, Xiaojun Chang, Mengmeng Wang, and Jingdong Wang. Ssmg: Spatial-semantic map guided diffusion model for free-form layout-to-image generation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2480– 2488, 2024. 1
- [21] Nupur Kumari, Bingliang Zhang, Sheng-Yu Wang, Eli Shechtman, Richard Zhang, and Jun-Yan Zhu. Ablating concepts in text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22691–22702, 2023. 3
- [22] Black Forest Labs. black-forest-labs/flux github page, 2024. 1, 7, 14

- [23] Shikai Li, Jianglin Fu, Kaiyuan Liu, Wentao Wang, KwanYee Lin, and Wayne Wu. Cosmicman: A text-to-image foundation model for humans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6955–6965, 2024. 1
- [24] Yaoyiran Li, Ching Yun Chang, Stephen Rawls, Ivan Vuli´c, and Anna Korhonen. Translation-enhanced multilingual text-to-image generation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9174–9193, 2023. 3
- [25] Zhixuan Liu, Peter Schaldenbrand, Beverley-Claire Okogwu, Wenxuan Peng, Youngsik Yun, Andrew Hundt, Jihie Kim, and Jean Oh. Scoft: Self-contrastive fine-tuning for equitable image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10822–10832, 2024. 2
- [26] Sasha Luccioni, Christopher Akiki, Margaret Mitchell, and Yacine Jernite. Stable bias: Evaluating societal representations in diffusion models. Advances in Neural Information Processing Systems, 36:56338–56351, 2023. 2
- [27] Jian Ma, Chen Chen, Qingsong Xie, and Haonan Lu. Peadiffusion: Parameter-efficient adapter with knowledge distillation in non-english text-to-image generation. In European Conference on Computer Vision, pages 89–105. Springer,

2024. 1, 2, 4, 5, 6, 7, 8, 12, 14

- [28] Margaret Mitchell, Giuseppe Attanasio, Ioana Baldini, Miruna Clinciu, Jordan Clive, Pieter Delobelle, Manan Dey, Sil Hamilton, Timm Dill, Jad Doughman, et al. Shades: Towards a multilingual assessment of stereotypes in large language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 11995–12041, 2025. 3
- [29] Jesse Mu and Jacob Andreas. Compositional explanations of neurons. Advances in Neural Information Processing Systems, 33:17153–17163, 2020. 3
- [30] OpenAI. Chatgpt (gpt-5 model). https://chat. openai.com/, 2025. Large language model developed by OpenAI, accessed on 2025-11-08. 1, 2, 3
- [31] Hadas Orgad, Bahjat Kawar, and Yonatan Belinkov. Editing implicit assumptions in text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7053–7061, 2023. 3
- [32] Makbule Gulcin Ozsoy. Multilingual prompts in llmbased recommenders: Performance across languages. arXiv preprint arXiv:2409.07604, 2024. 1
- [33] Haowen Pan, Yixin Cao, Xiaozhi Wang, Xun Yang, and Meng Wang. Finding and editing multi-modal neurons in pre-trained transformers. arXiv preprint arXiv:2311.07470,

2023. 3

- [34] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 1, 6, 7, 14
- [35] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image gener-

ation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 1

- [36] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 1
- [37] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 1
- [38] Sarah Schwettmann, Neil Chowdhury, Samuel Klein, David Bau, and Antonio Torralba. Multimodal neurons in pretrained text-only transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2862–2867, 2023. 3
- [39] Preethi Seshadri, Sameer Singh, and Yanai Elazar. The bias amplification paradox in text-to-image generation. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6367–6384, 2024. 2
- [40] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without test-time finetuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8543–8552, 2024. 1
- [41] Tianyi Tang, Wenyang Luo, Haoyang Huang, Dongdong Zhang, Xiaolei Wang, Xin Zhao, Furu Wei, and Ji-Rong Wen. Language-specific neurons: The key to multilingual capabilities in large language models. arXiv preprint arXiv:2402.16438, 2024. 1, 3
- [42] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1921–1930, 2023. 1
- [43] Bin Wang, Zhengyuan Liu, Xin Huang, Fangkai Jiao, Yang Ding, AiTi Aw, and Nancy Chen. Seaeval for multilingual foundation models: From cross-lingual alignment to cultural reasoning. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 370–390, 2024. 3
- [44] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Showo2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025. 7, 14
- [45] Wanying Xie, Yang Feng, Shuhao Gu, and Dong Yu. Importance-based neuron allocation for multilingual neural machine translation. arXiv preprint arXiv:2107.06569,

2021. 3

- [46] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-

- to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023. 6
- [47] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 3
- [48] Fulong Ye, Guang Liu, Xinya Wu, and Ledell Wu. Altdiffusion: A multilingual text-to-image diffusion model. In Proceedings of the AAAI conference on artificial intelligence, pages 6648–6656, 2024. 1, 2, 4, 6, 7, 8, 12, 14
- [49] Zeping Yu and Sophia Ananiadou. Interpreting arithmetic mechanism in large language models through comparative neuron analysis. arXiv preprint arXiv:2409.14144, 2024. 3
- [50] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 6

## Where Culture Fades: Revealing the Cultural Gap in Text-to-Image Generation

### Supplementary Material

[Figure 178]

|Country: China Language: Chinese<br><br>Culture-style modifier + noun: (zh) 中国风格的古典亭台临水而建，飞檐翘角，绿树环绕，尽显园林雅韵。<br><br>Culture-style modifier + noun: The classical Chinese pavilions are built by the water, with upturned eaves and surrounded by green trees, showcasing the elegance of the garden.<br><br>Noun-only: (zh) 古典亭台临水而建，飞檐翘角，绿树环绕。<br><br>Noun-only: Classical pavilions are built by the water, with upturned eaves and surrounded by green trees.|
|---|
|Country: Japan Language: Japanese Culture-style modifier + noun: (ja)花を抱きしめて優しく微笑む、和風の石造りうさぎの置物です。<br><br>Culture-style modifier + noun: This is a Japanese-style stone rabbit figurine, smiling gently while holding a flower.<br><br>Noun-only: (ja)これは石で作られたウサギの置物です。笑顔で、手に花を持っています。<br><br>Noun-only: This is a stone rabbit figurine, smiling gently while holding a flower.|
|Country: Italy Language: Italian Culture-style modifier + noun: (it) Una fumante ciotola di pasta a spirale in stile italiano.<br><br>Culture-style modifier + noun: A steaming bowl of spiral pasta.<br><br>Noun-only: (it) Una ciotola di noodles fumanti.<br><br>Noun-only: A bowl of steaming hot noodles.|

|Country: Russia<br><br>Language: Russian Culture-style modifier + noun: (ru) Серебряная резная эмалевая банка для хранения в русском стиле.<br><br>Culture-style modifier + noun: Russian-style silver enamel carved storage jar.<br><br>Noun-only: (ru) Серебряная эмалевая резная банка для хранения.<br><br>Noun-only: Silver enamel carved storage jar.|
|---|
|Country: Thailand<br><br>Language: Thai<br><br>Culture-style modifier + noun: (th) เสือผ้าสไตล์ไทยสีฟาอ่อนแสดงให้เห็นถึงเสน่ห์ของวัฒนธรรมไทยได้อย่างสง่างาม.<br><br>Culture-style modifier + noun: The light blue traditional Thai-style clothing elegantly showcases the charm of Thai culture.<br><br>Noun-only: (th) เสือผ้าสีฟาอ่อนแสดงให้เห็นถึงเสน่ห์ทางวัฒนธรรมอย่างสง่างาม.<br><br>Noun-only: The light blue clothing elegantly showcases cultural charm.|
|Country: Ukraine<br><br>Language: Ukrainian<br><br>Culture-style modifier + noun: (uk) Бандуру, традиційний український інструмент, тримає в руках жінка.<br><br>Culture-style modifier + noun: The bandura, a traditional Ukrainian instrument, is held in a woman's arms.<br><br>Noun-only: (uk) Традиційна цитра, яку тримає жінка в руках.<br><br>Noun-only: A traditional zither, held in a woman's arms.|

[Figure 179]

[Figure 180]

[Figure 181]

้ ้ ้ ้

[Figure 182]

[Figure 183]

Figure 12. Examples of the geographical and cultural composition of the CultureBench dataset.

The appendices provide additional details that support and extend the main paper. Appendix A describes the dataset’s geographic composition, evaluation protocol, and explains the selection of specific countries. Appendix B further validates the conjecture presented in the main text. Appendix C presents further experimental results and ablation studies. Appendix D details the user study’s design and conduct. Appendix E provides additional visualization results for PEA-Diffusion [27] and AltDiffusion [48] under both methods. Appendix F addresses common issues. Appendix G covers the limitations of our work. Appendix H reflects on the ethical considerations of this research.

##### A. Details of CultureBench Dataset

This study presents the dataset’s geographical composition within a broad cultural classification framework. It encompasses major cultural spheres such as the Arab world, East Asia, continental Europe, Latin America, and parts of Africa. This approach reflects the macro-sociological context of the data sources. It does not represent an essentialist or homogenized understanding of cultures. Cultural spheres are delineated by principal nations and languages, including Arabic, Chinese, Japanese, Korean, Thai, German, Russian, Italian, Dutch, Polish, Turkish, Ukrainian, Spanish, Portuguese, and French in parts of Africa. Figure 12 illustrates an exemplar composition of this study’s dataset.

Details of Data Collection. During data collection, we used Google Image Search 1 and complementary web resources. These included Wikipedia 2 and other public search platforms. We obtained publicly available images, setting keywords based on various cultural contexts to cover diverse scenarios and object types. After automated ex-

- 1https://images.google.com
- 2https://www.wikipedia.org

traction, images undergo preliminary screening. This eliminates low-quality, semantically irrelevant, or copyrightinfringing samples. A cross-cultural expert team then manually reviews images, prioritizing the exclusion of those conveying stereotypes or biases. In addition, these experts define and validate expert-curated cultural subdivisions within each broad cultural group. These include regional styles, ethnic or indigenous traditions, and locally distinctive visual elements. The team uses these subdivisions to guide manual filtering and annotation. This ensures an accurate representation of the intended cultural context without incorporating visual elements from other cultural backgrounds. As a result, CultureBench captures both macro-level cultural identity and finer-grained intracultural diversity. Through this process, we have enhanced the cultural accuracy and fairness of our cross-cultural visual data while ensuring its diversity.

##### B. Further Evidence on the Culture Gap

To test whether the cultural gap arises from under-activation rather than knowledge absence, we conduct a unified controlled experiment (Figure. 13). Using a culture-neutral English prompt and fixing all stochastic factors (e.g., seed and sampler), we generate images while selectively activating different culture-neuron sets identified in previous analyses. Despite identical prompts, activating China, Japan, Italy, Germany, or Ukraine neuron sets yields clearly distinct cultural styles, whereas the baseline remains culturally neutral. This controlled intervention demonstrates that the model already encodes rich culture-specific representations, and that the failure stems primarily from insufficient activation during generation rather than missing cultural knowledge.

PEA-Diffusion

Original

###### Arab China Korea France Portugal

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

A tallbuilding

AltDiffusion

###### Original China France Germany Italy Ukraine

A traditionalclothing

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

Figure 13. Further evidence on the culture gap. For a fixed English prompt and identical sampling settings, activating different cultureneuron sets steers both PEA-Diffusion (top) and AltDiffusion (bottom) toward distinct cultural styles.

- Table 4. Comparison of accuracy between CultureVQA and human experts on the CultureBench dataset.

|Model / Group<br><br>|Accuracy|
|---|---|
|Human Experts (Avg.) CultureVQA (Ours)|94.18% 91.57%<br><br>|

##### C. More Experiments

###### C.1. Reliability of CultureVQA

To assess the consistency of CultureVQA with human subjective cognition, we invited 30 domain experts to participate in comparative experiments using the CultureBench dataset. Each question contained four real-world images, and only one matched the specified cultural element. CultureVQA selected one image per question, and the experts made their choices under the same conditions. We then calculated and compared the accuracy rates for both CultureVQA and the experts. As shown in Table 4, CultureVQA achieved an accuracy of 91.57%, while human experts reached an average accuracy of 94.18%, resulting in a gap of only 2.61 percentage points. This relatively small difference indicates that CultureVQA’s performance closely approaches that of human experts, demonstrating high consistency with human cultural recognition in this task.

###### C.2. Cultural Probing Universal Type

For clarity in the main text, only the detection results from PEA-Diffusion are presented. This section additionally demonstrates the culture-sensitive layers and neurons associated with AltDiffusion for comparison purposes.

[Figure 196]

- Figure 14. AltDiffusion cultural sensitivity. ∆CA peaks layer

14. Therefore, layer 14 is culturally sensitive.

[Figure 197]

- Figure 15. AltDiffusion neuronal detection result. The weighted frequency scores show only a few salient peaks per culture, indicating culture-specific neurons.

Culture Layer Detection. To test the generality of our detection method beyond PEA-Diffusion, we also analyze AltDiffusion. Following the main paper’s probing procedure, we create paired prompts (“culture-style modifier + noun” and “noun-only”), annotate modifier and noun token groups, and extract cross-attention maps from all layers. We then compare aggregated attention from cultural modifiers to their paired nouns, yielding layerwise cultural-attention

- Table 5. Validating Culture-Sensitive Neuron Detection in AltDiffusion. Neuronal accuracy on the test subset with “cultural style modifier + noun” prompts.

|Method<br><br>|CultureVQA ↑|
|---|---|
|AltDiffusion [48]<br><br>+ Masked Top-K Neurons<br><br>+ Masked Random Neurons<br><br>|44.54 12.04 (-32.50) 42.45 (-2.09)|

[Figure 198]

Figure 16. Selection of Threshold K. The red dashed vertical line indicates the chosen threshold K. Neurons to the left of the line correspond to the selected Top-K culture-sensitive neurons, while the scores to the right fall below the cutoff and are discarded.

contrast curves for AltDiffusion (Figure 14).

Culture Neuron Detection. We employ the same methodology as in the main text to localize culturally sensitive neurons within the AltDiffusion culture layer. As illustrated in Figure 15, we present culturally sensitive neurons across six distinct cultural contexts. This demonstrates our approach’s robust capability to detect culturally sensitive neurons across varied architectural frameworks.

###### C.3. Cultural Validation in AltDiffusion

To analyze the accuracy of culture neurons in AltDiffusion, we validate these neurons for each cultural group. For all 15 cultural groups, we report CultureVQA scores in three controlled settings: (1) Baseline (no masking), (2) Masked TopK Neurons (masking the Top-K identified culture-sensitive neurons), and (3) Random Mask (masking the same number of neurons at random). Table 5 shows that masking the TopK neurons causes a substantial drop in CultureVQA performance. The mean score falls by 32.50 points compared to the AltDiffusion baseline. In contrast, randomly masking the same number of neurons reduces the mean score by just 2.09 points. This negligible drop stays close to the baseline. These results indicate that the culture-sensitive neurons we identify in AltDiffusion concentrate cultural information and capture cultural representations.

###### C.4. Selection of Threshold K

To clarify how we choose the threshold K, we visualize the weighted frequency scores (WFS) of candidate neurons and inspect their response patterns. The main factor guiding the choice of K is the presence of a small subset of neurons that exhibit prominently higher responses than the rest. As shown in Figure 16, we plot the WFS curves for

Table 6. Quantitative analysis across domains. Generate using 100 captions from outside the domain and calculate the performance of CultureVQA. Bold rows mark the highest performance within each baseline group.

###### Method CultureVQA ↑

StableDiffusion XL [34] 10.00 FLUX.1-dev [22] 17.00 Show-o2 [44] 15.00 PEA-Diffusion [27] 15.00 AltDiffusion [48] 15.00 StableDiffusion 3.5 [7] 17.00

Baseline: PEA-Diffusion

###### Ours (Zero-Training) 22.00 (+7.00) Ours (Fine-Tuned) 35.00 (+20.00)

Baseline: AltDiffusion

###### Ours (Zero-Training) 20.00 (+5.00) Ours (Fine-Tuned) 32.00 (+17.00)

three different cultural groups; in each case, a few leading neurons form clear peaks, followed by a rapid decay. We set K at the elbow before this sharp drop, so that neurons with salient high responses are retained, while the long tail of weakly responsive neurons is discarded.

- C.5. Details of SAE we set the sparsity coefficient to α = 321 . We adopt a TopK SAE with a hidden layer dimension of 4096. The model is optimized with AdamW using a learning rate of 0.0004 and a constant learning-rate schedule without warmup. We use an MSE reconstruction loss to encourage faithful feature reconstruction.

- C.6. Cross-Domain Generalization Results

We evaluated our method’s generalization using quantitative and qualitative tests on cue words not present in the CultureBench dataset. We used 100 out-of-domain captions to ensure the evaluation reflects true and reliable out-ofdistribution performance.

Quantitative Results. Table 6 shows that our method consistently improves cross-domain cultural accuracy for both PEA-Diffusion and AltDiffusion. Under the zero-training setting, our approach yields gains of +7.00 and +5.00 CultureVQA points over PEA-Diffusion and AltDiffusion, respectively, outperforming all existing off-the-shelf models. When fine-tuned, the improvements increase to +20.00 and +17.00 points for PEA-Diffusion and AltDiffusion, respectively, with our method achieving the highest accuracy within each baseline group. These results show that the culture-sensitive neurons identified by our method provide a substantial, transferable cultural prior, enabling robust generalization to captions outside the training distribution.

Qualitative Results. As illustrated in Figure 17, when prompted with the Chinese instruction “A wood carving

Ours (AltDiffusion) Zero-Training

Ours (AltDiffusion) Fine-Tuned

Ours (PEA-Diffusion) Zero-Training

Ours (PEA-Diffusion) Fine-Tuned

Stable Diffusion XL FLUX.1-dev Show-o2 AltDiffusion PEA-Diffusion Stable Diffusion 3.5

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

(A wood carving exhibit)

一个木雕展品(zh)

Fail Fail Fail

###### Fail

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

(A beautiful silk headscarf) Une robe élégante(fr)

ليمجيريرحباجح(ar)

Fail Fail

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

(An elegant dress)

Figure 17. Cross-domain qualitative experiments. Our approach generates images that are more culturally appropriate.

exhibit”, most models exhibited semantic comprehension errors. Their outputs were inconsistent with the prompt’s meaning. Our approach, however, maintained both semantic and cultural coherence. In contrast to Arabic instructions like “A beautiful silk headscarf” or French prompts such as “An elegant dress”, our approach offers greater precision. It captures key textual details regarding fabric texture, style, and aesthetic quality. It also adeptly integrates regional cultural symbols and aesthetic preferences in color coordination, pattern design, and figure representation. This enables the generation of images better suited to local cultural contexts. Taken together, these qualitative findings indicate that our approach is not only more robust and expressive but also more culturally sensitive across languages and regions. It substantially improves cross-linguistic and cross-cultural text-image alignment and image generation, thereby further validating its effectiveness and broad applicability in realworld, culturally diverse scenarios.

##### D. Detail of User Study

As shown in Figure 18, we conducted a controlled user study to evaluate the perceptual effectiveness of this method. Fifty participants, aged 25-47 (23 female), completed three sequential perceptual tasks. Each task assessed the subjective performance of generated images, focusing on cultural alignment and preference.

Multi-Choice Culture. During the Multi-Choice Culture (MCC) mission, we first used the noun-only prompt within CultureBench and fed it to multiple image generation models, obtaining numerous candidate images. Next, participants were asked to select all images they deemed representative of a given target culture. This selection process then measured each model’s capacity to generate culturally coherent images without textual cultural cues.

Single-Choice Culture. Single-Choice Culture (SCC) defines cultural perception as a forced-choice task of cultural

identification. For each noun-only prompt, we generate an image using a specific model. Participants view the modelgenerated image and are asked to infer its most probable cultural category. We provide a fixed set of 16 options: 15 predefined cultural categories plus an additional ’Uncertain’ option. Participants must select a single option based precisely on the image’s visual content.

Cultural Semantic Relevance. Cultural Semantic Relevance (CSR) is employed to evaluate the subjective cultural semantic alignment of images within a given target culture. For each CSR item, we present an image generated by a specific model, explicitly informing participants of its stylistic origin. Participants are required to rate the image’s alignment with the target culture within the context of that model’s style, on a scale of 1 to 5.

##### E. More Results

To demonstrate the effectiveness of our approach more intuitively, we generated both an unenhanced baseline image and an enhanced image using the same prompt, while keeping the model’s seed and other generative parameters fixed. The comparison between the two is shown in Figure 19. The results show that the enhanced images more faithfully reflect the cultural context, including scene elements, character depictions, and fine textures. This leads to more accurate visual expressions of the target culture and further highlights the effectiveness of our approach in achieving cultural consistency.

##### F. More Discussion

▷ Q1. What do we mean by “cultural consistency” in this work? We define cultural consistency as the degree to which an image from a multilingual prompt shows statistically grounded, contextually appropriate cultural cues tied to the target language’s socio-cultural context, rather than

|Task Type 1. Which of the following pictures are close to a “Russian-style silver milk jug” ? (for MCC)<br><br>[Figure 229]<br><br>[Figure 230]<br><br>[Figure 231]<br><br>[Figure 232]<br><br>[Figure 233]<br><br>[Figure 234]<br><br>[Figure 235]<br><br>□ A □ B □ C □ D □ E □ F □ G<br><br>Task Type 2. Which country's image is this? (for SCC)<br><br>[Figure 236]<br><br>□ Unknown □ Arab □ China □ France<br>□ Germany □ Italy □ Japan □ Korea<br>□ Northlands □ Poland □ Portugal □ Russia<br>□ Spain □ Thailand □ Turkey □ Ukraine<br><br><br>Task Type 3. How well the image align with the Japanese style.<br><br>(for CSR)<br><br>[Figure 237]<br><br>□ 1 □ 2 □ 3<br>□ 4 □ 5<br>|
|---|

###### Figure 18. Example interface of the user study comprising three task types. MCC (top), SCC (bottom-left), and CSR (bottom-right).

AltDiffusion (Zero-Training) AltDiffusion (Fine-Tuned)

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

France Japan

Arab China

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

PEA-Diffusion (Zero-Training)

PEA-Diffusion (Fine-Tuned)

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

Spain

Russia

Italy

Portugal

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

Figure 19. More Results. Further examples of results generated by AltDiffusion and PEA-Diffusion under different methods.

merely literal semantic correctness. Our approach uses (i) a fixed language–region mapping, (ii) observable, moderate visual grounding (such as architecture, clothing, artifacts), and (iii) expert screening to remove stereotypical or inappropriate cues. These elements turn a vague idea into an operational objective that can be measured and optimized.

- ▷ Q2. Why is CultureBench designed as a medium-scale diagnostic benchmark rather than a larger dataset? CultureBench is deliberately scaled to a moderate size, as it serves as a controlled diagnostic benchmark for assessing the cultural behaviour of multilingual text-to-image models rather than functioning as a pre-training corpus: Approximately 7.9k images, meticulously annotated across 15 language-culture regions, suffice to reveal systemic cultural omissions under “noun-only” prompts and support neural-level analysis, whilst ensuring the feasibility of highquality, de-stereotyped annotation. We thus regard it as an extensible starting point rather than a comprehensive catalogue covering all cultures.
- ▷ Q3. Why do we rely on CultureVQA, and how reliable is

it as an evaluation metric? We adopt CultureVQA because it directly addresses the challenge of evaluating whether images accurately reflect the nuanced cultural attributes encoded in multilingual prompts. Unlike existing automatic metrics (e.g., FID, CLIPScore), which focus on pixel-level similarity or general correspondence, CultureVQA frames evaluation as a semantic recognition problem, allowing assessment of cultural correctness. By using a VQA-style cultural identification task, CultureVQA enables consistent, scalable, and multi-class evaluation across 15 cultural regions. Its reliability is supported by a human–model consistency study (Appendix C.1).

▷ Q4. What evidence supports the hypothesis that cultural knowledge exists but is under-activated? Our claim is posed as a hypothesis and supported by converging empirical evidence rather than a formal proof. First, explicit culture-style modifiers (e.g., “Italian architecture”, “a person in traditional Chinese clothing”) consistently trigger culturally grounded generations. In contrast, noun-only prompts collapse to neutral, culture-agnostic prototypes, in-

dicating that the cultural capability is present but insufficiently activated. Second, masking the Top-K neurons identified by our probing method results in a sharp drop in CultureVQA performance, whereas masking the same number of random neurons has minimal effect, suggesting a causal relationship between these neurons and cultural semantics. Third, attention-based probing reveals a stable culture-sensitive layer in which “culture-style modifier + noun” and “noun-only” prompts yield distinct activation patterns. Together, these findings show that the model has cultural representations, but they are under-activated without explicit cues.

- ▷ Q5. Why are our neuron- and layer-level interventions reasonable, and do they over-claim causality? Our neuron- and layer-level interventions are deliberately minimal and localized: both the zero-training neuron amplifier and the fine-tuned layer enhancer operate only within the culture-sensitive layer and within a small subspace identified by our probing method, without modifying the diffusion backbone. Adjusting these neurons reliably improves CultureVQA scores and human-perceived cultural fidelity, while leaving CLIPScore, ImageReward, LPIPS, and visual diversity essentially unchanged. This indicates that the intervention is targeted rather than disruptive. Importantly, we do not claim to establish definitive causal attributions for individual neurons. Instead, we frame our approach as a lightweight and interpretable control mechanism that leverages empirically responsive subspaces. It offers practical gains in cultural consistency while serving as a useful starting point for future, more formal causal analyses.
- ▷ Q6. Is testing only on CultureBench lacking external validation? We agree that relying solely on CultureBench would limit external validation, which is why we include a cross-domain experiment using 100 out-of-distribution captions not appearing in CultureBench (Appendix C.6). Our method still yields consistent gains in CultureVQA under both zero-training and fine-tuning in this out-of-domain setting, showing that the improvement is not tied to CultureBench’s specific prompts.
- ▷ Q7. This paper uses ChatGPT data to refine cultural products, but could bias and errors be introduced? The use of ChatGPT-generated text may introduce biases; hence, we have implemented multiple dedicated measures in our data construction process to minimise these risks. Firstly, the ChatGPT content we utilise is strictly confined to “culture-style modifier + noun”. All corresponding images are sourced exclusively from publicly available real-world reference materials and undergo multi-stage human expert review to eliminate stereotypes, semantic errors, and culturally inaccurate cues. Furthermore, the ChatGPTgenerated modifiers themselves undergo expert filtering to ensure they do not contain stereotypical or inappropriate cultural descriptions.

- ▷ Q8. When switching generative models, is it necessary to re-execute the “Cultural Sensitivity Layer and Cultural Neuron Detection” step within our methodology? Whether re-detection is required depends primarily on whether the text encoder of the new model has changed. If the new generative model continues to use the same text encoder as in our experiments, re-detection is generally unnecessary; if the text encoder has been altered, the detection steps must be re-executed.
- ▷ Q9. Mean square error (MSE) cannot assess cultural consistency, so is it appropriate to use it? MSE is not in itself an effective metric for measuring cultural consistency, and we therefore did not employ it as an evaluation criterion. Within our framework, MSE serves as the training signal for specific layer enhancers, rather than the objective we use to gauge cultural consistency.
- ▷ Q10. What are the roles of ϵ and β? ϵ is set to 0 by default, since activations greater than zero are treated as neuron firing. β is a small stability constant added to the denominator to prevent it from becoming zero when a neuron has zero or very few activations.

##### G. Limitation

CultureBench currently covers 15 cultural regions, but this still reflects only a limited portion of global cultural diversity. Because the benchmark is built from publicly available image resources, some cultures, especially those from lowresource or marginalized communities, remain underrepresented. We emphasize that the currently included cultural regions are carefully curated and remain valid, but they do not yet exhaust the diversity within each region or across countries. In future iterations, we intend to identify underrepresented cultural groups by partnering with relevant organizations, actively seeking additional image sources, and introducing finer intra-cultural subdivisions, thereby expanding the benchmark to more countries and territories for a more comprehensive and inclusive evaluation suite.

##### H. Ethics Statement

In this research, we acknowledge the potential misuse of image synthesis techniques, such as ours, for generating deceptive content and spreading disinformation, a serious concern we address explicitly. However, we also note the substantial progress made in detection and prevention mechanisms in this domain. Our framework supports critical research initiatives and encourages third-party oversight, aiming to strike a balance between technological advancement and security considerations. This balanced approach promotes responsible deployment while preserving the innovation potential.

