# arXiv:2506.12450v3[cs.CL]11Oct2025

## Language Surgery in Multilingual Large Language Models

Joanito Agili Lopo∗1,2, Muhammad Ravi Shulthan Habibi∗1,3, Tack Hwa Wong∗1, Muhammad Ilham Ghozali1,3, Fajri Koto1,4, Genta Indra Winata1,5, Peerat Limkonchotiwat1,6, Alham Fikri Aji1,4, Samuel Cahyawijaya*1,7 1SEACrowd 2Kreasof AI 3Universitas Indonesia 4MBZUAI 5Capital One 6AI Singapore 7Cohere {amalopo99,muhammadravi251001,tackhwawong00}@gmail.com muhammad.ilham.gozali@gmail.com,samuelcahyawijaya@cohere.com Code: https://github.com/SEACrowd/itlc

### Abstract

Large Language Models (LLMs) have demonstrated remarkable generalization capabilities across tasks and languages, revolutionizing natural language processing. This paper investigates the naturally emerging representation alignment in LLMs, particularly in the middle layers, and its implications for disentangling language-specific and language-agnostic information. We empirically confirm the existence of this alignment, analyze its behavior in comparison to explicitly designed alignment models, and demonstrate its potential for languagespecific manipulation without semantic degradation. Building on these findings, we propose Inference-Time Language Control (ITLC), a novel method that leverages latent injection to enable precise cross-lingual language control and mitigate language confusion in LLMs. Our experiments highlight ITLC’s strong crosslingual control capabilities while preserving semantic integrity in target languages. Furthermore, we demonstrate its effectiveness in alleviating the cross-lingual language confusion problem, which persists even in current largescale LLMs, leading to inconsistent language generation. This work advances our understanding of representation alignment in LLMs and introduces a practical solution for enhancing their monolingual and cross-lingual performance.

### 1 Introduction

Large Language Models (LLMs) have revolutionized natural language processing, demonstrating remarkable generalization capabilities across diverse tasks and languages (Brown et al., 2020; Le Scao et al., 2023; Anil et al., 2023; Team et al., 2025; Cohere et al., 2025; Singh et al., 2025). Their ability to adapt to new tasks in few-shot and even zeroshot settings highlights their efficiency and versatility (Bang et al., 2023; Susanto et al., 2025).

*Equal contributions. See Appendix K for further details.

小 孩 子

Language-specific space

[Figure 1]

Late layers

|TEN→ZH = RZH - REN|
|---|

Lang Manipulation: = + TEN→ZH

Language-agnostic space they

[Figure 2]

Early layers

small

小 孩

They are all

kids

|Input|
|---|

Figure 1: We inspect the alignment in the middle layer representation of LLMs, allowing us to disentangle the language-specific and language-agnostic information. By exploiting this behavior, we are able to achieve Inference-Time Language Control (ITLC), alleviating the language confusion problem in LLMs.

Prior works have identified a naturally emerging representation alignment across layers in LLMs, particularly in the middle layers of LLMs (Chang et al., 2022; Zhao et al., 2024a). This emerging alignment in LLMs is the key factor in their ability to handle multiple languages (Cahyawijaya, 2024; Tang et al., 2024; Wilie et al., 2025), which is pivotal for their cross-lingual capabilities. However, several questions remain open, such as whether this emerging alignment behaves similarly to alignment in models trained with enforced alignment objectives (Reimers and Gurevych, 2020; Yang et al., 2019a; Feng et al., 2022; Limkonchotiwat et al., 2022, 2024), how this alignment can be utilized to further enhance LLMs, etc.

In this work, we investigate the phenomenon of representation alignment in LLMs, focusing on its occurrence, distinction, and potential applications.

We aim to confirm the presence of representation alignment and contrast it with alignment in LLMs with strictly designed alignment, such as multilingual SentenceBERT (Reimers and Gurevych, 2019) or LaBSE (Feng et al., 2022). Our findings highlight that, unlike LLMs with strictly designed alignment, the naturally emerging alignment in recent LLMs demonstrates a much stronger retention of language-specific information with much smaller performance drop in the aligned representation compared to the unaligned layers which we conjecture to be the minimum required languagespecific information required to perform do decoding in the correct language.

To this end, we exploit the bottleneck of language-specific information in the aligned representation and develop a simple test-time intervention method to control the decoding language, namely inference-time language control (ITLC). Specifically, we extracted a low-rank language vector from the aligned representations using linear discriminant analysis (Balakrishnama and Ganapathiraju, 1998; Tharwat et al., 2017), aggregated them per language to create language vectors, and perform a simple vector translation to control the decoding language as shown in Figure 1 1. We show the effectiveness of ITLC in mitigating the language confusion problem (Marchisio et al., 2024). Furthermore, we conduct an extensive evaluation to test that, unlike other approaches, ITLC can control the language with minimal loss of semantic.

Our contribution in this work is fourfold:

- • We confirm the presence of representation alignment in LLMs, providing empirical evidence of this phenomenon (§3.2).
- • We contrast natural alignment with strictly designed alignment, highlighting their comparable impact on cross-lingual generalization while emphasizing their differences in alignment locations and the extent of languagespecific information retention (§3.2).
- • We investigate a method to extract languagespecific information from aligned representations, showcasing the potential for languagespecific manipulation while preserving the semantic integrity of the generation (§4.1).
- • We introduce ITLC, a novel method that enables cross-lingual language control and miti-

1Note that, during the inference step, we only need to perform a single vector addition operation to control the language

- as everything else can be precomputed.

gates language confusion problems that retain semantic integrity in target languages (§5).

### 2 Related Work

#### 2.1 Representation Alignment in LLMs

Representation alignment refers to the process by which semantically identical inputs expressed in different languages are mapped to similar internal embeddings within LLMs (Park et al., 2024b; Wu and Dredze, 2020; Chang et al., 2022). Originally, representation alignment is strictly embedded into the modeling objective to ensure output consistency across languages and to enable a better cross-lingual transfer (Pires et al., 2019; Wu and Dredze, 2019; Reimers and Gurevych, 2020; Feng et al., 2022; Choenni et al., 2024). Wendler et al. (2024); Zhao et al. (2024a); Mousi et al. (2024) have observed a tendency for LLMs to align representations across different languages by measuring the similarity between embeddings of parallel sentences across different languages (Ham and Kim, 2021; Gaschi et al., 2023; Cahyawijaya, 2024). Inspired from previous studies, our work measures the degree of alignment across various layers between strictly and naturally aligned models to contrast the two and understand its relation to languagespecific and language-agnostic capabilities (Kulshreshtha et al., 2020; Libovický et al., 2020; Hua et al., 2024; Wilie et al., 2025) of LLMs.

#### 2.2 Latent Controllability in LLMs

LLMs controllability is crucial for ensuring that the systems adhere with human intentions. Through mechanisms such as adapter (Pfeiffer et al., 2020; Hu et al., 2022), prompting (Lin et al., 2021; Bai et al., 2022), latent manipulation (Madotto et al.,

- 2020; Ansell et al., 2021), etc, we aim to gain control over the behavior of LLMs. Various aspects have been explored in LLM controllability, including internal knowledge (Madotto et al., 2020; Xu et al., 2022), styles & personas (Lin et al.,
- 2021; Wagner and Ultes, 2024; Cao, 2024), languages (Üstün et al., 2020; Ansell et al., 2021), human values (Bai et al., 2022; Cahyawijaya et al., 2025a), etc. Li et al. (2023b); Duan et al. (2024); Ji et al. (2024); Chen et al. (2024) show that latent states in LLMs exhibit discernible patterns for distinguishing truthful outputs from hallucinated ones, suggesting an intrinsic awareness of fabrication. Similar methods are also introduced for stylistic and safety control (Subramani et al., 2022; Kwak

[Figure 3]

- Figure 2: Cross-lingual similarity across different layers in LaBSE and Qwen2.5-0.5B. LaBSE exhibits high cross-lingual similarity in its final layer, whereas Qwen2.5-0.5B shows this similarity in the middle layer. This difference suggests that the alignment of representations occurs at distinct positions within the two models.

et al., 2023). These underscore the potential of latent interventions for precise control over LLM behavior. ITLC extends the latent manipulation methods for controlling the generated language in inference time, demonstrating how languagespecific information can be extracted and manipulated without losing semantic meaning. This opens new avenues for controlling language generation and mitigating confusion problems.

### 3 Understanding Representation Alignment in LLMs

Prior works (Chang et al., 2022; Zhao et al.,

- 2024a; Cahyawijaya, 2024; Wilie et al., 2025; Payoungkhamdee et al., 2025) demonstrate the existence of emerging representation alignment in LLMs. We take a step further to provide a deeper understanding to this behavior by contrasting it with alignment in strictly-aligned LLMs. Specifically, we observe the correlation between the degree of alignment with the cross-lingual generalization and language identification (LID) capability, which are the proxies to their language-agnostic and language-specific capabilities, respectively.

#### 3.1 Experiment Settings

Model Settings As a measure of alignment, we compute the average cosine similarity of the latent representation of a sentence in one language with the representation of parallel sentences in the other languages. For the LLM with strictly designed alignment, we employ LaBSE (Feng et al., 2022).

For the LLM with emerging representation alignment, we employ multilingual decoder-only LLM, i.e., Qwen2.5 (Qwen et al., 2025). Specifically, we employ Qwen2.5-0.5B with 500M parameters to have a comparable scale with the LaBSE model with 471M parameters. To measure the LID capability, we take the latent representation of both models in the first, middle, and last layers. In this case, we are interested in comparing the behavior between the strictly aligned representation in LaBSE and the emerging aligned representation in Qwen2.5-0.5B. Following Cahyawijaya et al. (2025b), we measure LID performance by linear probing and kNN to measure linear separability and cluster closeness within each language class. More details about the experiment are presented in Appendix B and Appendix C.

Datasets We employ a set of multilingual evaluation datasets. To measure the degree of alignment, we employ 7 datasets: FLORES-200 (Team et al., 2022), NTREX-128 (Federmann et al., 2022), NusaX (Winata et al., 2023), NusaWrites (Cahyawijaya et al., 2023), BUCC (Zweigenbaum et al., 2017), Tatoeba (Tiedemann, 2020), and Bible Corpus (McCarthy et al., 2020). For cross-lingual evaluation, we incorporate 4 datasets: SIB200 (Adelani et al., 2024), INCLUDE-BASE (Sridhar et al., 2020), XCOPA (Ponti et al., 2020), and PAWSX (Yang et al., 2019b). For LID evaluation, we incorporate 3 datasets, i.e., FLORES-200, NTREX128, and NusaX. The detailed description of each

LaBSE Qwen2.5-0.5B Method Layer FLORES-200 NTREX-128 NusaX FLORES-200 NTREX-128 NusaX Linear Probing

First 95.13 93.29 97.30 94.21 91.42 95.55 Middle 94.18 92.68 94.51 91.76 90.04 87.09 Last 70.89 74.36 65.44 92.46 90.27 88.77

First 88.35 90.43 81.78 83.69 86.06 65.79 Middle 78.85 81.30 45.37 55.32 54.73 25.05 Last 3.92 1.63 0.00 71.73 81.86 29.39

KNN

- Table 1: LID performance by layer and classification method for LaBSE and QWEN2.5-0.5B. Red bold text highlights the LID scores on the layer where alignment occurs in each corresponding model. LID performance is consistently lower in a layer where the representation is aligned across all models and classification methods.

dataset is shown in Appendix A.

#### 3.2 Experiment Result

Strictly and Naturally Aligned LLMs LaBSE and Qwen2.5-0.5B demonstrate distinct patterns in cross-lingual representation alignment. As shown in Figure 2, LaBSE demonstrates a distributed alignment strength across deeper layers, with the middle and last layers achieving high average similarity scores (0.758 and 0.754, respectively). This aligns with the training objective of LaBSE, which aligns the representation on the last layer. In contrast, Qwen2.5-0.5B exhibits a more localized alignment pattern, with the middle layer showing a strikingly higher average similarity (0.922) than both the first (0.591) and last (0.375) layers. This suggests that Qwen2.5-0.5B concentrates representation alignment sharply in the middle layer, achieving both higher and more stable cross-lingual representation. See detailed analysis in Appendix B.1.

This result displays distinct layer-wise behaviors in retaining the language-specific and languageagnostic information within the two types of LLMs. Specifically, for model with strict alignment, aligned representation is located in the layer where the objective is applied to – the last layer in the case of LaBSE –, while in LLMs with natural alignment, the aligned representation is formed in the middle layers and breaks as the representation goes closer into the last layer. This aligns with prior works (Chang et al., 2022; Tang et al., 2024; Wilie et al., 2025) that show the representation alignment naturally emerges in the middle layer of LLMs.

Representation Alignment and LanguageSpecific Information As shown in Table 1, the LID performance of LaBSE and Qwen2.5-0.5B models evaluated using both KNN and linear probing reveals that the first layer consistently achieves

the highest LID F1 scores across all datasets. For LaBSE, the aligned representation in the last layer exhibits notably weaker performance, particularly for the FLORES-200 and NusaX datasets. Similarly, in Qwen2.5-0.5B, the aligned representation in the middle layer shows weaker LID performance compared to the first and last layers. These empirical findings highlight three key insights: (1) language-specific information, such as surfacelevel features and general linguistic patterns, is more dominant in the early layers; (2) the degree of alignment is negatively correlated with the amount of language-specific information retained; and (3) unlike strictly aligned LLMs, the aligned representation in LLMs with emerging alignment retains more language-specific information, which potentially serves as the basis for determining the language of the generated sequence.

### 4 Inference-Time Language Control

Building on the insights presented in §3, we explore a method to control the language of the generated sequence with minimal semantic loss. Specifically, we develop a method to extract languagespecific information at the layer where representation alignment occurs in LLMs. Using this information, we gather language-specific vectors from each language and use them to manipulate the language-specific information during the inference time. With this language-specific intervention, we aim to steer the model toward utilizing languagespecific features, allowing us to perform InferenceTime Language Control (ITLC).

ITLC offers two key advantages over existing intervention methods: Unlike existing intervention methods that are limited to either crosslingual (Wang et al., 2024) or monolingual (Nie et al., 2025) scenarios, and unlike approaches that

Method Qwen2.5-0.5B Qwen2.5-0.5B-Instruct Qwen2.5-7B Qwen2.5-7B-Instruct Llama-3.1-8B Llama-3.1-8B-Instruct Monolingual

Baseline 59.91 83.66 55.24 78.89 56.98 94.63 ICL (5-shot) 53.62 80.30 62.78 74.13 69.86 88.57

+ ITLC (ours) 74.38 86.28 69.55 81.01 82.18 93.21 PEFT 82.91 89.85 83.80 88.28 93.01 96.66 + ITLC(ours) 86.17 90.51 85.60 90.12 96.03 97.19 ITLC (ours) 81.21 82.20 63.40 84.89 75.77 96.41

###### Cross-lingual

Baseline 35.36 57.69 60.61 78.81 26.13 83.25 ICL (5-shot) 50.63 69.70 69.37 78.51 62.38 86.68

+ ITLC (ours) 87.58 88.07 84.90 84.04 88.15 90.34 PEFT 77.55 84.34 82.66 83.56 89.73 91.13 + ITLC (ours) 90.51 89.85 83.92 84.10 88.98 93.60 ITLC (ours) 85.61 86.79 74.40 84.73 81.68 89.06

- Table 2: Main results for LPR metrics on LCB across different LLMs in monolingual and cross-lingual settings. Blue rows denote methods combined with ITLC. Bold values represent the best result for each model. All results have been applied with the QA/Chat template during inference.

Method

Qwen2.5 0.5B

Qwen2.5 0.5B Instruct

Llama-3.1 8B

Llama-3.1 8B Instruct

Baseline 34.97 52.28 25.05 80.68 INCLINE 43.82 56.54 34.69 80.63 ReCoVeR 88.43 84.21 88.79 90.29 ITLC (ours) 81.22 81.97 76.38 85.65

- Table 3: Comparison of cross-lingual LPR metrics on LCB across baseline and state-of-the-art methods for 6 languages (AR, ES, HI, ID, RU, ZH). Bold values represent the best result for each model. All results have been applied with the QA/Chat template during inference.

the Singular Value Decomposition (SVD) solver in order to handle high-dimensional embeddings efficiently and select the top k eigenvectors corresponding to the largest eigenvalues to form W ∈ Rd×k. Let D = {(hi,li)}Ni=1 denote a dataset of hidden states hi ∈ Rd labeled with language classes li ∈ {1,...,K}, this projects hidden states to a lower-dimensional space z = hTW ∈ Rk.

To validate the quality of the projection and select the optimal number of components k, we train a neural network classifier with a single linear layer on the projected training data z. We experiment with several k values and evaluate classification accuracy on a test set. Finally, we take

require interventions across all layers (Sterz et al.,

- 2025; Yunfan et al., 2025), ITLC is effective in both settings while intervening at only a single middle layer.

- k = 100 because LID performance significantly drops on higher components, indicating a major loss of language-specific information. More details on the LDA settings are shown in Appendix D

Language Vector Using the LDA-projected space, we construct language vectors by leveraging the neural network’s weights to identify active dimensions for each language. For each language

- l we extract the weight matrix U ∈ RK×k from

#### 4.1 Methods

Latent Extraction Latent extraction techniques are employed to isolate language-specific information from the model’s representations. Specifically, we extract hidden states from various large language models to capture language-specific features

- at their middle representation layers. Given an input sequence from the FLORES-200 dataset (Team et al., 2022), we compute the hidden states h ∈ Rd at a specified layer, where d represents the hidden state dimension of the respective model. Finally, we apply mean pooling to ensure that only meaningful token embeddings contribute to the final representation.

the neural network’s linear layer, where ul,j represents the contribution of dimension j ∈ {1,...,k} to language l. We define a threshold τ = 0.01 and select active dimensions for language l as Al = {j | |ul,j| > τ}. The language vector vl ∈ Rk for language l is computed as the mean of projected hidden states zi over samples of language l, restricted to active dimensions:

Linear Discriminant Analysis To disentangle language-specific information, we apply Linear Discriminant Analysis (LDA) to maximize class separability and reduce dimensionality. We use

1 Nl hi∈l zi[j], if j ∈ Al, 0, otherwise,

vl[j] =

[Figure 4]

- Figure 3: Comparison of LPR metrics on LCB between Qwen2.5-7B-Instruct with ITLC and Qwen2.5-7b-EAX across 14 languages in monolingual and cross-lingual settings.

where Nl is the number of samples for language l, and zi[j] is the j-th component of the projected hidden state.

Vector Injection To enable injection, we project the language vector back to the original embedding space using the pseudo-inverse: vlorig = vlW† ∈ Rd. By applying this, we retain the original embedding of the input and modify it with the language vector inverse projection. For cross-lingual settings with a source language x (e.g., English) and target language y (e.g., Indonesian), we compute a shift vector 2:

δ = −vxorig + vyorig.

For monolingual settings where source and target languages are identical (x = y), we treat the shift vector as the language vector itself:

##### δ = vxorig.

The shift vector is injected into the hidden states at the middle layer during inference into both the prompt and the generated tokens. Formally, we apply:

h′t = ht + αδ, ∀t ∈ [1,Ttotal]

where ht is the middle-layer hidden state at position t, α is a scaling factor, h′t is the corresponding modified hidden state, and Ttotal is the total number of tokens during inference covering both input and generated tokens. We provide an ablation of different language shift strategies in Appendix E.

2We demonstrate the importance of subtracting the source language vector in Appendix G.4.

[Figure 5]

Figure 4: Cross-lingual LPR performance on LCB, comparing base and instruct shift vector applications.

### 5 Impact of ITLC

We demonstrate the effectiveness of ITLC on mitigating the language confusion problem (Marchisio et al., 2024). We also compare our method with another test-time intervention methods specifically designed for language confusion (Sterz et al., 2025) 3. Furthermore, we showcase that ITLC can also perform language control while being highly efficient with minimal semantic loss compared to other existing test-time intervention methods (Wang et al., 2024).

#### 5.1 Experiment Setting

Dataset For language confusion evaluation, we utilize the Language Confusion Benchmark (LCB) (Marchisio et al., 2024), which contains both monolingual and cross-lingual settings across 14 languages. For semantic retention assessment, We utilize the Dolly multilingual dataset from Aya

3We also find another related test-time intervention (Yunfan et al., 2025), nonetheless the code is not published so we could not empirically compare ITLC with their approach.

Evaluation Suite (Singh et al., 2024) 4 by taking 200 QA sentences in nine various languages from diverse regions and language families: Indonesian (ID), Thai (TH), Turkish (TR), Japanese (JA), French (FR), Spanish (ES), Arabic (AR), Chinese (ZH), and Korean (KO). The description of datasets is shown in Appendix A.

Model Settings We experiment on two families of multilingual LLMs: Qwen2.5 (0.5B and 7B), and Llama-3.1-8B, and their instruct variants. Specifically, for cross-lingual control with the base model, the model will start to generate by having several target contexts, while in the instruct model, we add a language-identified prompt (i.e., Please answer in XX language) at the beginning of the sentence. See Appendix F for more details on language confusion and Appendix H for more details on semantic retention.

Evaluation Our evaluation on language confusion problem based on official metrics defined in Marchisio et al. (2024): Line-level Pass Rate (LPR). Meanwhile, we evaluate the cross-lingual generation performance based on chrF++ and multilingual BERT F1 5 metrics. Additionally, we conduct a human evaluation with native annotators in both EN→XX and XX→EN directions, focusing on 30 samples covering 3 aspects: naturalness, promptcompletion relevance, and answer correctness using likert score ranging from [1...5]. The human annotation guideline is presented in Appendix J.

- 5.2 Results 5.2.1 ITLC in Mitigating Language Confusion

As shown in Table 2, our proposed method, ITLC, surpasses both baseline and in-context learning (ICL) configurations across models of varying parameter scales in cross-lingual settings. This superior performance is consistent in monolingual settings with only one exception, where the Qwen2.5-0.5B-Instruct model performs slightly worse than the baseline, demonstrating that ITLC effectively shifts the model’s language output in cross-lingual settings. For the base model, crosslingual performance improves progressively with few-shot examples, as they utilize English inputs

- 4https://huggingface.co/datasets/CohereLabs/

aya_evaluation_suite/viewer/dolly_machine_ translated.

- 5https://huggingface.co/google-bert/

bert-base-multilingual-cased

TR JA

TR JA

| |
|---|

TH

TH

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

FR

FR

| |
|---|

12

68

10

| |
|---|

66

| |
|---|

| |
|---|

| |
|---|

8

64

| | |
|---|---|
| | |

| |
|---|

| |
|---|

| |
|---|

| |
|---|

6

62

| |
|---|

| |
|---|

4

60

| | |
|---|---|
| | |

ID

ID

2

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

ES

ES

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

| | |
|---|---|
| | |

| |
|---|

| |
|---|

| |
|---|

| |
|---|

ZH

ZH

| |
|---|

| |
|---|

| |
|---|

AR

AR

KO

KO

Qwen2.5-0.5B Baseline

Qwen2.5-7B Baseline

Llama-3.1-8B Baseline

Qwen2.5-0.5B ITLC

Qwen2.5-7B ITLC

Llama-3.1-8B ITLC

Figure 5: Generation performance for different target languages on Qwen2.5 and Llama-3.1 Instruct models based on chrF++ (Left) and BERT F1 (Right). Baseline denotes the same model prompted in the same language as the desired target language.

with explicit target-language instructions, reinforcing input-output alignment. In contrast, the instruct model exhibits minimal variation in few-shot settings compared to ITLC, as its instruction-tuning inherently supports multilingual prompting without dependency on few-shot quantity. These results demonstrate that our approach enhances crosslingual language consistency while accommodating training objective differences between base and instruct models. Moreover, ITLC achieves competitive performance on instruct model compared to parameter-efficient fine-tuning (PEFT): LoRA finetuning method (Hu et al., 2022), without requiring any changes to the LLM weights. Notably, our method can further mitigate language confusion when combined with ICL and PEFT. The combination of PEFT + ITLC consistently achieves the best results in monolingual settings across all models, while in cross-lingual settings, different combinations prove optimal depending on the model, with ICL + ITLC and PEFT + ITLC both achieving top performance on various models. A detailed perlanguage breakdown of the results is presented in Table 26 and Table 27.

Comparison of ITLC with other test-time intervention methods While INCLINE (Wang et al., 2024) was originally designed to project representations from various languages into English to enhance LLM performance on low-resource languages, we adapt and reverse this mechanism to project from English into various target languages. Due to computational constraints, we compare our method, ITLC, against INCLINE and ReCoVeR (Sterz et al., 2025) using two model families, Qwen2.5-0.5B and Llama-3.1-8B, and their instruct variants across six target lan-

guages. As shown in Table 3, ITLC outperforms INCLINE across all model configurations. Notably, INCLINE shows limited improvement on instruction-tuned models, with almost no performance gain on Llama-3.1-8B-Instruct, suggesting that methods relying solely on the last token may be ineffective at mitigating language confusion in instruction-following models. Although ReCoVeR achieves the highest performance overall, ITLC demonstrates competitive results on instructiontuned models while being considerably more efficient. This indicates that intervention at a single middle layer is sufficient for mitigating language confusion, compared to ReCoVeR’s approach of intervening across all layers.

Comparison of ITLC with Cross-lingual Optimized Model Due to computational constraints, we were unable to perform full parameter finetuning. Instead, we use another model, Qwen2.57b-EAX (Yang et al., 2025), which was fine-tuned on Qwen2.5-7B and optimized for cross-lingual translation ability. As shown in Figure 3, our ITLC achieves similar results to the upperbound on average monolingual LPR (84.89% vs 85.28%). However, for cross-lingual settings, our method achieves 84.73% on average LPR compared to 92.54% for the upperbound. Notably, there is a substantial performance gap for Indonesian (ID), Japanese (JA), and Chinese (ZH). We observe that our ITLC exhibits code-switching to English when handling these languages, indicating that our method may not fully eliminate the source language vector for these languages and might require further language-specific tuning of the scaling factor α, or that our ITLC cannot adequately disentangle the language vector and capture the language-specific information well for these languages. A detailed per-language breakdown of the results is presented in Table 24 and Table 25

Transferability of Language Vector to PostTrained Models Interestingly, as shown in Figure 4, applying language vectors gathered from the base model to the instruct model achieves comparable performance to its native instruct vectors which suggests the effectiveness of language shift from the base model for cross-lingual control even in the instruct model. This transferability indicates that the relative distance between languagespecific and that the resulting language-specific features from the pre-training phase is robust to downstream adaptation, including tasks generaliza-

Model Lang Shift Nat. Rel. Cor. Qwen2.5-7B-Instruct

ID→ID 3.66 4.43 3.46 TH→TH 3.13 2.63 2.23 ZH→ZH 4.30 4.20 4.13

Baseline

EN→ID 4.00 4.90 3.96 EN→TH 2.46 3.93 3.40 EN→ZH 4.63 4.80 4.53

ITLC

Llama-3.1-8B-Instruct

ID→ID 4.50 4.46 3.86 TH→TH 3.20 2.36 2.36 ZH→ZH 3.96 4.33 3.76

Baseline

EN→ID 3.83 3.83 3.53 EN→TH 3.40 2.93 2.60 EN→ZH 4.76 4.66 4.53

ITLC

Table 4: Human evaluation of ITLC response quality in Qwen2.5 and Llama-3.1. Nat., Rel., and Cor. respectively denote naturalness, relevance, and answer correctness ranging from [1...5]. Baseline denotes the same model prompted in the monolingual setting.

tion from instruction-tuning and value alignment in RLHF and preference-tuning. This evidence implies that the cross-lingual symmetry – i.e., the geometric alignment between language representations – constructed during the fine-tuning is preserved even after various downstream refinement of the model. The preservation of these relationships implies that language-specific cues are retained as invariant properties across model versions, enabling consistent cross-lingual language control through ITLC despite parameter updates during downstream fine-tuning, instruction-tuning, preference-tuning, and RLHF.

#### 5.2.2 Semantic Retention in ITLC

Cross-lingual Semantic Retention We demonstrated that the proposed ITLC method not only improve the target language fidelity but is also able to effectively control cross-lingual generation and retain the semantic information, implying contextually accurate generation. As shown in Figure 5, statistically, our proposed ITLC method improved 2% across the chrF++ and 3-5% in BERT F1 metrics in the instruct model (refers to Table 14); the same investigation also occurred in the base models (refers to Table 13). The highest performance shows on Spain (ES), French (FR), and Indonesia (ID), it outperforms its baseline by 1-2%, which is prompted in the same language as the desired target language.

However, we found that in some languages, such

Lang Shift Prompt (EN) Generated Output (XX) Literal Translation (EN)

물론입니다! 다음은각항 目的是中文，与问题无关。

EN → KO Tell me whether each of these are animals or not: Alligator, Grass, Tree, Boat, Dolphin, Shark, Fish, Fence, Dog, Car

Of course! Next, each item purpose is in Chinese, unrelated to the question. Continuing to answer your question: each item are all in Korean as ” 각項目들”, which is incorrectly expressed, please correct it. - Alligator (Alligator . . .

继续回答您的 질문: 각항목들은 都是韩语로” 各項 目들” 이라고表述有误，请 수정해주세요. - 악어 (Alligator . . .

土豆 chip 袋在打开后会变软，主要是因为密封包 装中的环境发生了变化。未开封时，在真空或惰性 气体环境下保存的薯片可以保持脆度和新鲜口感较 长时间不被氧化影响风味与品质。. . .

EN → ZH Why do potato chip bags become stale after opening?

Potato chip bags soften after opening, mainly because the sealed packaging environment changes. When unopened, chips stored in vacuum or inert gas can maintain crispness and freshness longer without oxidation affecting flavor and quality. . . .

EN → JA Suggest 5 horror films to me もちろんです。お这儿应该是「は」です。「も」「し」

Of course. Oh, here it should be “は.”When used with “も”or “し,”if the previous part is negative and the following starts with a positive expression, then it connects with conjunctions like

と连用して时，前が否定形の词で后ろ的始まり是 肯定表現時、接続詞” でも”” しか”” さえ ￿ 等と 并 . . .

“でも,”“しか,”“さえ,”etc. . . .

Figure 6: Examples of the lowest generated outputs score from Qwen2.5-7B-Instruct on Korean, Chinese, and Japanese in EN→XX, evaluated with the BERT F1 score. The literal translation column is translated from the generated output, and it is done by using ChatGPT.

as Korean (KO), it retains less cross-lingual semantics due to the unique challenges of distinct syntax and semantics (Park et al., 2020, 2024a), which happens across models. Further investigation revealed that many overlaps or code switching occur between these languages. For example, in Figure 6, EN→KO direction, the generated output contains Japanese tokens (denoted in blue), while the literal output being disconnected from the context. Additionally, in Japanese output generation, it seems like answering out of context, while in Chinese produced coherent and well-structured sentences. See Appendix I for more detailed examples.

Human Evaluation We further conduct a human evaluation to validate our findings regarding the semantic retention in ITLC. We recruit native speakers to annotate 30 generation samples in Indonesia (ID), Thai (TH), and Chinese (ZH). Based on results presented in Table 4, we found that our ITLC proposed method tends to have a similar level of semantics compared to the monolingual baseline (prompted in the same target language), with Qwen2.5-7B-Instruct performing quite better in terms of Relevance and Correctness metrics compared to the Llama-3.1-8B-Instruct. Meanwhile, our ITLC method performs much better than baseline in Indonesia and Thai in Qwen2.5 models, showed that our injection vector could improved the semantic transferability across languages, enabling the model to retain both relevance and correctness. Overall, our results validate the capability of ITLC to maintain relevance and correctness in cross-lingual generation, highlighting its potential for enhancing cross-lingual performance of LLMs.

### 6 Conclusion

Our work explores the phenomenon of representation alignment in LLMs, confirming its occurrence and elucidating its behavior compared to strictly designed alignment models. We have demonstrated the potential for disentangling language-specific and language-agnostic information, enabling effective language-specific manipulation without semantic loss. Furthermore, we have shown the practical applications of language control manipulation in enhancing language control and mitigating confusion problems. Our ITLC method demonstrates significant gains on the language confusion benchmark, achieving an average improvement of 9% in monolingual and 26.7% in cross-lingual settings. It also achieves comparable performance to existing test-time intervention approaches, while being much more efficient (requiring only a single middle layer intervention). Ultimately, our work not only advances the theoretical understanding of representation alignment in LLMs but also introduces a practical and effective solution for enhancing crosslingual capabilities, paving the way for more robust and versatile LLLMs in multilingual contexts.

### Limitations

The study has several limitations that should be considered when interpreting the results. First, the coverage of LLMs is limited to a specific set of models for representation alignment, particularly Qwen and LaBSE and only one model size (0.5B parameters), which may not be representative of all LLMs. The findings may not generalize to other models with different architectures or training data, as the behavior of representation alignment can vary significantly across different LLMs. Future

research should aim to include a more diverse range of models to validate the generalizability of the results.

Second, the evaluation is conducted on a limited number of languages, which may not capture the full spectrum of linguistic diversity. The study focuses on a subset of languages, and the results may not extend to languages with different typological features or those that are underrepresented in the training data. Expanding the evaluation to include a broader range of languages, especially lowresource languages, would provide a more comprehensive understanding of the model’s capabilities and limitations.

Moreover, The scaling factor α affects different models differently, requiring careful adjustment for optimal performance. Due to the nature of Linear Discriminant Analysis (LDA), the number of components (n_components) is constrained by the number of target language classes. This constraint introduces a trade-off, the number of target language hidden states that need to be extracted depends on the chosen n_components, potentially causing computational overhead, and vice versa.

Additionally, the human evaluation is based on only 30 samples per language, which may not provide a comprehensive assessment of the model’s performance. While the sample size is sufficient for preliminary analysis, a larger dataset would be necessary to draw more robust conclusions. Increasing the number of samples and involving a more diverse group of evaluators could enhance the reliability and validity of the findings.

### Ethical Considerations

The research involves the use of LLMs, which might raise ethical considerations regarding bias, fairness, and transparency on the generated results. To ensure ethical conduct, the study adheres to the following principles: (1) Bias Mitigation: The models used are evaluated for potential biases, and efforts are made to mitigate any identified biases.

- (2) Fairness: The evaluation is conducted across multiple languages from diverse regions and language families to ensure fairness and inclusivity.
- (3) Transparency: The methodology and results are presented transparently to allow for replication and verification. (4) Privacy: No personal data is used in the evaluation, and all data is anonymized to protect privacy. (5) Accountability: The researchers take responsibility for the ethical implications of

the study and are committed to addressing any concerns that may arise.

We also acknowledge that our research utilized AI tools for writing, rewriting, and generating code. Although these tools offer significant advantages in terms of efficiency and productivity, their use raises important ethical considerations. We recognize the potential for bias and errors inherent in AI-generated content and have taken steps to mitigate these risks through rigorous human review and validation. Furthermore, we are mindful of the potential impact on the broader software development community, particularly regarding job displacement and the need for upskilling. We believe that responsible AI integration should prioritize transparency, accountability, and the empowerment of human developers, ensuring that these tools augment rather than replace human expertise. This research aims to contribute to the ongoing dialogue on ethical AI development and usage, advocating for a future where AI tools are harnessed responsibly to enhance human creativity and innovation in the field of software engineering.

### References

David Ifeoluwa Adelani, Hannah Liu, Xiaoyu Shen, Nikita Vassilyev, Jesujoba O. Alabi, Yanke Mao, Haonan Gao, and En-Shiun Annie Lee. 2024. SIB-200: A simple, inclusive, and big evaluation dataset for topic classification in 200+ languages and dialects. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 226–245, St. Julian’s, Malta. Association for Computational Linguistics.

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, and 1 others. 2023. Palm 2 technical report. arXiv preprint arXiv:2305.10403.

Alan Ansell, Edoardo Maria Ponti, Jonas Pfeiffer, Sebastian Ruder, Goran Glavaš, Ivan Vuli´c, and Anna Korhonen. 2021. MAD-G: Multilingual adapter generation for efficient cross-lingual transfer. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 4762–4781, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez,

and 32 others. 2022. Constitutional ai: Harmlessness from ai feedback. Preprint, arXiv:2212.08073.

Suresh Balakrishnama and Aravind Ganapathiraju. 1998. Linear discriminant analysis-a brief tutorial. Institute for Signal and information Processing, 18(1998):1–8.

Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, Quyet V. Do, Yan Xu, and Pascale Fung. 2023. A multitask, multilingual, multimodal evaluation of ChatGPT on reasoning, hallucination, and interactivity. In Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the AsiaPacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 675–718, Nusa Dua, Bali. Association for Computational Linguistics.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, and 1 others. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Samuel Cahyawijaya. 2024. Llm for everyone: Representing the underrepresented in large language models. Preprint, arXiv:2409.13897.

Samuel Cahyawijaya, Delong Chen, Yejin Bang, Leila Khalatbari, Bryan Wilie, Ziwei Ji, Etsuko Ishii, and Pascale Fung. 2025a. High-dimension human value representation in large language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5303–5330, Albuquerque, New Mexico. Association for Computational Linguistics.

Samuel Cahyawijaya, Delong Chen, Yejin Bang, Leila Khalatbari, Bryan Wilie, Ziwei Ji, Etsuko Ishii, and Pascale Fung. 2025b. High-dimension human value representation in large language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5303–5330, Albuquerque, New Mexico. Association for Computational Linguistics.

Samuel Cahyawijaya, Holy Lovenia, Fajri Koto, Dea Adhista, Emmanuel Dave, Sarah Oktavianti, Salsabil Akbar, Jhonson Lee, Nuur Shadieq, Tjeng Wawan Cenggoro, Hanung Linuwih, Bryan Wilie, Galih Muridan, Genta Winata, David Moeljadi, Alham Fikri Aji, Ayu Purwarianti, and Pascale Fung. 2023. NusaWrites: Constructing high-quality corpora for underrepresented and extremely lowresource languages. In Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific

Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 921–945, Nusa Dua, Bali. Association for Computational Linguistics.

Lang Cao. 2024. Learn to refuse: Making large language models more controllable and reliable through knowledge scope limitation and refusal mechanism. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 3628–3646, Miami, Florida, USA. Association for Computational Linguistics.

Tyler Chang, Zhuowen Tu, and Benjamin Bergen. 2022. The geometry of multilingual language model representations. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 119–136, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, and Jieping Ye. 2024. INSIDE: LLMs’ internal states retain the power of hallucination detection. In The Twelfth International Conference on Learning Representations.

Rochelle Choenni, Dan Garrette, and Ekaterina Shutova. 2024. How do languages influence each other? studying cross-lingual data sharing during lm fine-tuning. Preprint, arXiv:2305.13286.

Team Cohere, :, Aakanksha, Arash Ahmadian, Marwan Ahmed, Jay Alammar, Milad Alizadeh, Yazeed Alnumay, Sophia Althammer, Arkady Arkhangorodsky, Viraat Aryabumi, Dennis Aumiller, Raphaël Avalos, Zahara Aviv, Sammie Bae, Saurabh Baji, Alexandre Barbet, Max Bartolo, Björn Bebensee, and 211 others. 2025. Command a: An enterprise-ready large language model. Preprint, arXiv:2504.00698.

Hanyu Duan, Yi Yang, and Kar Yan Tam. 2024. Do llms know about hallucination? an empirical investigation of llm’s hidden states. Preprint, arXiv:2402.09733.

Christian Federmann, Tom Kocmi, and Ying Xin. 2022. NTREX-128 – news test references for MT evaluation of 128 languages. In Proceedings of the First Workshop on Scaling Up Multilingual Evaluation, pages 21–24, Online. Association for Computational Linguistics.

Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, and Wei Wang. 2022. Languageagnostic bert sentence embedding. Preprint, arXiv:2007.01852.

Felix Gaschi, Patricio Cerda, Parisa Rastin, and Yannick Toussaint. 2023. Exploring the relationship between alignment and cross-lingual transfer in multilingual transformers. In Findings of the Association for Computational Linguistics: ACL 2023, pages 3020–3042, Toronto, Canada. Association for Computational Linguistics.

Jiyeon Ham and Eun-Sol Kim. 2021. Semantic alignment with calibrated similarity for multilingual sentence embedding. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 1781–1791, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Tianze Hua, Tian Yun, and Ellie Pavlick. 2024. mOthello: When do cross-lingual representation alignment and cross-lingual transfer emerge in multilingual models? In Findings of the Association for Computational Linguistics: NAACL 2024, pages 1585–1598, Mexico City, Mexico. Association for Computational Linguistics.

Ziwei Ji, Delong Chen, Etsuko Ishii, Samuel Cahyawijaya, Yejin Bang, Bryan Wilie, and Pascale Fung. 2024. LLM internal states reveal hallucination risk faced with a query. In Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pages 88–104, Miami, Florida, US. Association for Computational Linguistics.

Saurabh Kulshreshtha, Jose Luis Redondo Garcia, and Ching-Yun Chang. 2020. Cross-lingual alignment methods for multilingual BERT: A comparative study. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 933–942, Online. Association for Computational Linguistics.

Jin Myung Kwak, Minseon Kim, and Sung Ju Hwang. 2023. Language detoxification with attributediscriminative latent space. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10149–10171, Toronto, Canada. Association for Computational Linguistics.

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili´c, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, and 1 others. 2023. Bloom: A 176bparameter open-access multilingual language model.

Haonan Li, Fajri Koto, Minghao Wu, Alham Fikri Aji, and Timothy Baldwin. 2023a. Bactrian-x : A multilingual replicable instruction-following model with low-rank adaptation. Preprint, arXiv:2305.15011.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023b. Inferencetime intervention: Eliciting truthful answers from a language model. In Thirty-seventh Conference on Neural Information Processing Systems.

Jindˇrich Libovický, Rudolf Rosa, and Alexander Fraser. 2020. On the language neutrality of pre-trained multilingual representations. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1663–1674, Online. Association for Computational Linguistics.

Peerat Limkonchotiwat, Wuttikorn Ponwitayarat, Lalita Lowphansirikul, Potsawee Manakul, Can Udomcharoenchaikit, Ekapol Chuangsuwanich, and Sarana Nutanong. 2024. McCrolin: Multi-consistency crosslingual training for retrieval question answering. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 2780–2793, Miami, Florida, USA. Association for Computational Linguistics.

Peerat Limkonchotiwat, Wuttikorn Ponwitayarat, Can Udomcharoenchaikit, Ekapol Chuangsuwanich, and Sarana Nutanong. 2022. CL-ReLKT: Cross-lingual language knowledge transfer for multilingual retrieval question answering. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 2141–2155, Seattle, United States. Association for Computational Linguistics.

Zhaojiang Lin, Zihan Liu, Genta Indra Winata, Samuel Cahyawijaya, Andrea Madotto, Yejin Bang, Etsuko Ishii, and Pascale Fung. 2021. XPersona: Evaluating multilingual personalized chatbot. In Proceedings of the 3rd Workshop on Natural Language Processing for Conversational AI, pages 102–112, Online. Association for Computational Linguistics.

Andrea Madotto, Samuel Cahyawijaya, Genta Indra Winata, Yan Xu, Zihan Liu, Zhaojiang Lin, and Pascale Fung. 2020. Learning knowledge bases with parameters for task-oriented dialogue systems. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 2372–2394, Online. Association for Computational Linguistics.

Kelly Marchisio, Wei-Yin Ko, Alexandre Berard, Théo Dehaze, and Sebastian Ruder. 2024. Understanding and mitigating language confusion in LLMs. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 6653– 6677, Miami, Florida, USA. Association for Computational Linguistics.

Arya D. McCarthy, Rachel Wicks, Dylan Lewis, Aaron Mueller, Winston Wu, Oliver Adams, Garrett Nicolai, Matt Post, and David Yarowsky. 2020. The Johns Hopkins University Bible corpus: 1600+ tongues for typological exploration. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 2884–2892, Marseille, France. European Language Resources Association.

Basel Mousi, Nadir Durrani, Fahim Dalvi, Majd Hawasly, and Ahmed Abdelali. 2024. Exploring alignment in shared cross-lingual spaces. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6326–6348, Bangkok, Thailand. Association for Computational Linguistics.

Ercong Nie, Helmut Schmid, and Hinrich Schütze. 2025. Mechanistic understanding and mitigation of language confusion in english-centric large language models. Preprint, arXiv:2505.16538.

Chanjun Park, Hyeonwoo Kim, Dahyun Kim, SeongHwan Cho, Sanghoon Kim, Sukyung Lee, Yungi Kim, and Hwalsuk Lee. 2024a. Open Ko-LLM leaderboard: Evaluating large language models in Korean with Ko-h5 benchmark. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3220– 3234, Bangkok, Thailand. Association for Computational Linguistics.

Kiho Park, Yo Joong Choe, and Victor Veitch. 2024b. The linear representation hypothesis and the geometry of large language models. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org.

Kyubyong Park, Joohong Lee, Seongbo Jang, and Dawoon Jung. 2020. An empirical study of tokenization strategies for various Korean NLP tasks. In Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing, pages 133–142, Suzhou, China. Association for Computational Linguistics.

Patomporn Payoungkhamdee, Pume Tuchinda, Jinheon Baek, Samuel Cahyawijaya, Can Udomcharoenchaikit, Potsawee Manakul, Peerat Limkonchotiwat, Ekapol Chuangsuwanich, and Sarana Nutanong. 2025. Towards better understanding of program-ofthought reasoning in cross-lingual and multilingual environments. Preprint, arXiv:2502.17956.

Jonas Pfeiffer, Andreas Rücklé, Clifton Poth, Aishwarya Kamath, Ivan Vuli´c, Sebastian Ruder, Kyunghyun Cho, and Iryna Gurevych. 2020. AdapterHub: A framework for adapting transformers. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 46–54, Online. Association for Computational Linguistics.

Wannaphong Phatthiyaphaibun, Korakot Chaovavanich, Charin Polpanumas, Arthit Suriyawongkul, Lalita Lowphansirikul, Pattarawat Chormai, Peerat Limkonchotiwat, Thanathip Suntorntip, and Can Udomcharoenchaikit. 2023. PyThaiNLP: Thai natural language processing in python. In Proceedings of the 3rd Workshop for Natural Language Processing Open Source Software (NLP-OSS 2023), pages 25–36, Singapore. Association for Computational Linguistics.

Telmo Pires, Eva Schlinger, and Dan Garrette. 2019. How multilingual is multilingual BERT? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4996–5001, Florence, Italy. Association for Computational Linguistics.

Edoardo Maria Ponti, Goran Glavaš, Olga Majewska, Qianchu Liu, Ivan Vuli´c, and Anna Korhonen. 2020. XCOPA: A multilingual dataset for causal commonsense reasoning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2362–2376, Online. Association for Computational Linguistics.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, and 25 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

- Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.
- Nils Reimers and Iryna Gurevych. 2020. Making monolingual sentence embeddings multilingual using knowledge distillation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4512–4525, Online. Association for Computational Linguistics.

Shivalika Singh, Angelika Romanou, Clémentine Fourrier, David I. Adelani, Jian Gang Ngui, Daniel Vila-Suero, Peerat Limkonchotiwat, Kelly Marchisio, Wei Qi Leong, Yosephine Susanto, Raymond Ng, Shayne Longpre, Wei-Yin Ko, Sebastian Ruder, Madeline Smith, Antoine Bosselut, Alice Oh, Andre F. T. Martins, Leshem Choshen, and 5 others. 2025. Global mmlu: Understanding and addressing cultural and linguistic biases in multilingual evaluation. Preprint, arXiv:2412.03304.

Shivalika Singh, Freddie Vargus, Daniel D’souza, Börje F. Karlsson, Abinaya Mahendiran, Wei-Yin Ko, Herumb Shandilya, Jay Patel, Deividas Mataciunas, Laura O’Mahony, Mike Zhang, Ramith Hettiarachchi, Joseph Wilson, Marina Machado, Luisa Moura, Dominik Krzemi´nski, Hakimeh Fadaei, Irem Ergun, Ifeoma Okoh, and 14 others. 2024. Aya dataset: An open-access collection for multilingual instruction tuning. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11521– 11567, Bangkok, Thailand. Association for Computational Linguistics.

Advaith Sridhar, Rohith Gandhi Ganesan, Pratyush Kumar, and Mitesh Khapra. 2020. Include: A large scale dataset for indian sign language recognition. MM ’20. Association for Computing Machinery.

Hannah Sterz, Fabian David Schmidt, Goran Glavaš, and Ivan Vuli´c. 2025. Recover the target language: Language steering without sacrificing task performance. Preprint, arXiv:2509.14814.

Nishant Subramani, Nivedita Suresh, and Matthew Peters. 2022. Extracting latent steering vectors from pretrained language models. In Findings of the Association for Computational Linguistics: ACL 2022, pages 566–581, Dublin, Ireland. Association for Computational Linguistics.

Yosephine Susanto, Adithya Venkatadri Hulagadri, Jann Railey Montalan, Jian Gang Ngui, Xian Bin

Yong, Weiqi Leong, Hamsawardhini Rengarajan, Peerat Limkonchotiwat, Yifan Mai, and William Chandra Tjhi. 2025. Sea-helm: Southeast asian holistic evaluation of language models. Preprint, arXiv:2502.14301.

Tianyi Tang, Wenyang Luo, Haoyang Huang, Dongdong Zhang, Xiaolei Wang, Xin Zhao, Furu Wei, and Ji-Rong Wen. 2024. Language-specific neurons: The key to multilingual capabilities in large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5701–5715, Bangkok, Thailand. Association for Computational Linguistics.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, and 197 others. 2025. Gemma 3 technical report. Preprint, arXiv:2503.19786.

NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, and 20 others. 2022. No language left behind: Scaling human-centered machine translation. Preprint, arXiv:2207.04672.

Alaa Tharwat, Tarek Gaber, Abdelhameed Ibrahim, and Aboul Ella Hassanien. 2017. Linear discriminant analysis: A detailed tutorial. AI Commun., 30(2):169–190.

Jörg Tiedemann. 2020. The tatoeba translation challenge – realistic data sets for low resource and multilingual MT. In Proceedings of the Fifth Conference on Machine Translation, pages 1174–1182, Online. Association for Computational Linguistics.

Ahmet Üstün, Arianna Bisazza, Gosse Bouma, and Gertjan van Noord. 2020. UDapter: Language adaptation for truly Universal Dependency parsing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2302–2315, Online. Association for Computational Linguistics.

Nicolas Wagner and Stefan Ultes. 2024. On the controllability of large language models for dialogue interaction. In Proceedings of the 25th Annual Meeting of the Special Interest Group on Discourse and Dialogue, pages 216–221, Kyoto, Japan. Association for Computational Linguistics.

Weixuan Wang, Minghao Wu, Barry Haddow, and Alexandra Birch. 2024. Bridging the language gaps in large language models with inference-time crosslingual intervention. Preprint, arXiv:2410.12462.

Chris Wendler, Veniamin Veselovsky, Giovanni Monea, and Robert West. 2024. Do llamas work in English? on the latent language of multilingual transformers. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15366–15394, Bangkok, Thailand. Association for Computational Linguistics.

Bryan Wilie, Samuel Cahyawijaya, Junxian He, and Pascale Fung. 2025. High-dimensional interlingual representations of large language models. Preprint, arXiv:2503.11280.

Genta Indra Winata, Alham Fikri Aji, Samuel Cahyawijaya, Rahmad Mahendra, Fajri Koto, Ade Romadhony, Kemal Kurniawan, David Moeljadi, Radityo Eko Prasojo, Pascale Fung, Timothy Baldwin, Jey Han Lau, Rico Sennrich, and Sebastian Ruder. 2023. NusaX: Multilingual parallel sentiment dataset for 10 Indonesian local languages. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 815–834, Dubrovnik, Croatia. Association for Computational Linguistics.

- Shijie Wu and Mark Dredze. 2019. Beto, bentz, becas: The surprising cross-lingual effectiveness of BERT. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 833–844, Hong Kong, China. Association for Computational Linguistics.
- Shijie Wu and Mark Dredze. 2020. Do explicit alignments robustly improve multilingual encoders? In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4471–4482, Online. Association for Computational Linguistics.

Yan Xu, Etsuko Ishii, Samuel Cahyawijaya, Zihan Liu, Genta Indra Winata, Andrea Madotto, Dan Su, and Pascale Fung. 2022. Retrieval-free knowledgegrounded dialogue response generation with adapters. In Proceedings of the Second DialDoc Workshop on Document-grounded Dialogue and Conversational Question Answering, pages 93–107, Dublin, Ireland. Association for Computational Linguistics.

Sen Yang, Yu Bao, Yu Lu, Jiajun Chen, Shujian Huang, and Shanbo Cheng. 2025. Enanchored-x2x: Englishanchored optimization for many-to-many translation. Preprint, arXiv:2509.19770.

Yinfei Yang, Gustavo Hernandez Abrego, Steve Yuan, Mandy Guo, Qinlan Shen, Daniel Cer, Yun-Hsuan Sung, Brian Strope, and Ray Kurzweil. 2019a. Improving multilingual sentence embedding using bidirectional dual encoder with additive margin softmax. arXiv preprint arXiv:1902.08564.

Yinfei Yang, Yuan Zhang, Chris Tar, and Jason Baldridge. 2019b. PAWS-X: A cross-lingual adversarial dataset for paraphrase identification. In

Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3687– 3692, Hong Kong, China. Association for Computational Linguistics.

Xie Yunfan, Lixin Zou, Dan Luo, Min Tang, Chenliang Li, Xiangyang Luo, and Liming Dong. 2025. Mitigating language confusion through inferencetime intervention. In Proceedings of the 31st International Conference on Computational Linguistics, pages 8418–8431, Abu Dhabi, UAE. Association for Computational Linguistics.

Yiran Zhao, Wenxuan Zhang, Guizhen Chen, Kenji Kawaguchi, and Lidong Bing. 2024a. How do large language models handle multilingualism? In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang, Yunlin Mao, Daoze Zhang, Zeyinzi Jiang, Zhikai Wu, Baole Ai, Ang Wang, Wenmeng Zhou, and Yingda Chen. 2024b. Swift:a scalable lightweight infrastructure for fine-tuning. Preprint, arXiv:2408.05517.

Pierre Zweigenbaum, Serge Sharoff, and Reinhard Rapp. 2017. Overview of the second BUCC shared task: Spotting parallel sentences in comparable corpora. In Proceedings of the 10th Workshop on Building and Using Comparable Corpora, pages 60–67, Vancouver, Canada. Association for Computational Linguistics.

### A Details of All Evaluation Datasets

The following tables present the full details of dataset sizes used in this study. Refer to Table 5, Table 6, Table 7, Table 8 and Table 9.

### B Detail Experiment for Understanding Representation Alignment in LLMs

#### B.1 Cosine Similarity Distributions Across Datasets

To better understand the representational behavior of the models, we analyzed the distribution of cosine similarity scores across layers. For LaBSE, the average cosine similarity increases from the first layer (mean = 0.6335, std = 0.0920) to the middle layer (mean = 0.7580, std = 0.1182), and remains comparably high in the last layer (mean = 0.7544, std = 0.1150). This trend suggests that semantic alignment becomes stronger toward the middle and final layers, with relatively low variability, indicating consistent behavior across input samples. These observations align with prior findings that intermediate layers in multilingual encoders often capture the most transferable features.

In contrast, Qwen2.5-0.5B exhibits a markedly different pattern. While the middle layer achieves the highest average similarity (mean = 0.9218, std = 0.0871), the first layer has a lower mean and higher variance (mean = 0.5913, std = 0.1650), indicating less stable representations early in the network. Notably, the last layer shows a substantial drop in similarity (mean = 0.3745) and a sharp increase in variability (std = 0.3988), suggesting a divergence in representational behavior, potentially due to task-specific tuning or greater representational fragmentation. This may help explain the weaker correlations between cosine similarity and task performance observed in Qwen’s final layers.

These findings reinforce the role of middle layers in capturing semantically meaningful and transferable representations, particularly in instructiontuned or general-purpose multilingual models. See Figure 2 for the histogram plot and Figure 7 for the bar chart per alignment dataset.

#### B.2 Additional Analysis For Alignment and Downstream Correlation

As shown in Table 10, the correlation between cosine similarity and downstream performance varies by dataset, layer, and model architecture. The following sections provide detailed interpretations.

Dataset Train Test Total # Languages SIB200 143,705 41,820 185,525 205 INCLUDE-BASE 890 22,638 23,528 44 XCOPA 1,100 5,500 6,600 11 PAWS-X 345,807 14,000 359,807 7

Table 5: Dataset sizes and number of languages for downstream tasks.

[Figure 6]

[Figure 7]

(a) Mean Cosine Similarity Score on LaBSE Model (b) Mean Cosine Similarity Score on Qwen2.5-0.5B Model

- Figure 7: Layer-wise cosine similarity distributions of LaBSE and Qwen2.5-0.5B models across different datasets.

Dataset Total # Languages FLORES-200 1,012 204 NTREX-128 1,997 128 NusaX 400 12 NusaWrites 14,800 9 (language pairs) BUCC 35,000 4 (language pairs) Tatoeba 88,877 112 (language pairs) BibleCorpus 85,533 828 (language pairs)

- Table 6: Total example counts and number of languages for alignment tasks. We only use test set for this alignment task.

Dataset Train Test Total # Languages

FLORES-200 997 1012 2,009 204 NTREX-128 - 1,997 1,997 128 NusaX 500 400 400 12

- Table 7: Total example counts per language and number of languages for for LID tasks.

SIB200 For LaBSE, correlation values are consistently strong and statistically significant across all layers. The first (Pearson r = 0.323), middle (Pearson r = 0.309), and last (Pearson r = 0.210) layers all demonstrate meaningful positive correlations with performance (p ≈ 0), indicating that

Dataset Train Test Total # Languages

FLORES-200 997 1012 2,009 204 Dolly - 1,800 - 9

- Table 8: Total example counts per language and number of languages for Language Control.

Dataset Total # Languages Monolingual

Aya 100 5 Dolly 100 5 Okapi 100 10 Native prompts 100 4

Cross-lingual

Okapi 100 14 shareGPT 100 14 Complex prompts 99 14

Table 9: Total example counts per language and number of languages for Language Confusion tasks, taken from Language Confusion Benchmark. Only test set is available.

cosine similarity is well-aligned with task accuracy throughout the network. This suggests that SIB200 benefits from LaBSE’s cross-lingual representations, especially in the earlier and middle layers. In contrast, Qwen2.5-0.5B shows very weak but statistically significant correlations (r ≤ 0.12 across all layers). While the trends are consistent, the effect sizes are negligible, suggesting that cosine similarity has limited practical influence on performance for Qwen2.5-0.5B on this dataset.

INCLUDE-BASE For LaBSE, correlations between cosine similarity and performance are negligible and statistically non-significant across all layers, with Pearson r values close to zero (−0.041, 0.005, −0.021). This suggests no meaningful alignment between representational similarity and task accuracy. In contrast, Qwen2.5-0.5B exhibits

###### Dataset Model Layer Pearson r R2 p-value

First 0.323 0.104 <10−300 Middle 0.309 0.096 <10−300

LaBSE

Last 0.210 0.044 <10−205

SIB200

First 0.060 0.004 <10−17 Middle 0.123 0.015 <10−69

Qwen2.5-0.5B

Last 0.043 0.002 <10−9

First -0.041 0.002 0.233 Middle 0.005 0.000 0.884 Last -0.021 0.000 0.545

LaBSE

INCLUDE-BASE

First 0.183 0.034 <10−7 Middle 0.142 0.020 <10−4

Qwen2.5-0.5B

Last 0.168 0.028 <10−6

First -0.115 0.013 0.458 Middle -0.026 0.001 0.867 Last 0.144 0.021 0.352

LaBSE

XCOPA

First 0.292 0.085 0.055 Middle -0.139 0.019 0.368

Qwen2.5-0.5B

Last 0.538 0.289 <0.001

First 0.141 0.020 0.484 Middle 0.270 0.073 0.173 Last 0.146 0.021 0.467

LaBSE

PAWS-X

First 0.228 0.052 0.252 Middle 0.532 0.283 0.004 Last 0.369 0.136 0.059

Qwen2.5-0.5B

- Table 10: Pearson correlation coefficients (r), R2, and p-values for the relationship between cosine similarity and task performance across different transformer layers on LaBSE and Qwen2.5-0.5B.

weak but statistically significant positive correlations (Pearson r range: 0.14–0.18), indicating that higher cosine similarity is marginally associated with improved performance. Despite the small effect sizes, these results highlight a slight but consistent behavioural alignment in Qwen2.5-0.5B on this dataset.

XCOPA For LaBSE, correlation values across layers are weak and statistically insignificant, suggesting minimal alignment between representational similarity and model performance. In contrast, Qwen2.5-0.5B exhibits a strong and statistically significant positive correlation in the last layer (Pearson r = 0.538, p < 0.001), implying that deeper representations may be more predictive for XCOPA.

PAWS-X LaBSE shows weak, non-significant positive correlations across layers. However, Qwen2.5-0.5B demonstrates a strong positive correlation in the middle layer (Pearson r = 0.532, p ≈ 0.004), suggesting that intermediate representations capture more alignment-relevant features for paraphrase detection.

Downstream Performance Relative to Random Baselines To provide a clearer picture of cross-lingual generalization and behavior alignment, we present a set of bar charts

comparing the performance of LaBSE and Qwen2.5-0.5B across four downstream evaluation datasets—SIB200, INCLUDE-BASE, XCOPA, and PAWS-X—relative to their respective random baselines.

On XCOPA and PAWS-X, LaBSE yields nearrandom or below-random performance, indicating that its fixed representations struggle with crosslingual commonsense reasoning and paraphrase detection. For SIB200, LaBSE performs slightly above the random baseline, suggesting limited task sensitivity in multilingual sentence similarity settings. However, its performance on INCLUDEBASE remains weak, staying near or below the random baseline and highlighting deficiencies in broader multilingual alignment.

In contrast, Qwen2.5-0.5B demonstrates stronger generalization on both SIB200 and INCLUDE-BASE, significantly outperforming its baseline and showing evidence of better cross-lingual task adaptation. However, it faces challenges on XCOPA and PAWS-X, where its performance hovers around or falls below baseline, pointing to possible limitations in zero-shot commonsense reasoning and paraphrase understanding across languages.

These comparisons highlight the differing strengths and weaknesses of encoder-only and decoder-only multilingual models across select zero-shot evaluation tasks. See Figure 8.

B.3 Additional Analysis For Alignment and LID Correlation

As shown in Table 11, the correlation between alignment (as measured by cosine similarity) and downstream LID performance varies notably across datasets, model architectures, and transformer layers. The following sections provide detailed interpretations for each dataset to contextualize these trends.

FLORES-200 On the FLORES-200 dataset, we observe a moderate negative correlation between cosine similarity and LID performance for both LaBSE and Qwen2.5-0.5B. The strength of the correlation increases in deeper layers, with the last layer showing the strongest correlation (r = −0.707, p < 10−31) for LaBSE. Qwen2.5-0.5B, however, exhibits its strongest negative correlation in the middle layer (r = −0.432, p < 10−9), indicating that as the embeddings become more aligned (i.e., higher cosine similarity), the language identity

[Figure 8]

(a) Performance of LaBSE across downstream tasks compared to random baselines.

[Figure 9]

(b) Performance of Qwen2.5-0.5B across downstream tasks compared to random baselines.

- Figure 8: Comparison of LaBSE and Qwen2.5-0.5B performance across various downstream tasks and their corresponding random baselines.

Dataset Model Layer Pearson r R2 p-value

First 0.024 0.001 0.732 Middle -0.122 0.015 0.084

LaBSE

Last -0.707 0.500 <10−31

FLORES-200

First -0.142 0.020 0.043 Middle -0.432 0.186 <10−9

Qwen2.5-0.5B

Last -0.278 0.077 <10−4

First 0.254 0.065 0.012 Middle -0.173 0.030 0.089

LaBSE

Last -0.621 0.385 <10−11

NTREX-128

First -0.232 0.054 0.021 Middle -0.476 0.226 <10−6 Last -0.340 0.115 0.001

Qwen2.5-0.5B

First -0.566 0.320 0.112 Middle -0.872 0.760 0.002 Last – – –

LaBSE

NusaX

First -0.455 0.207 0.218 Middle -0.873 0.763 0.002

Qwen2.5-0.5B

Last -0.045 0.002 0.910

- Table 11: Pearson correlation coefficients (r), R2, and pvalues for the relationship between KNN LID F1 score using mean-pooled embedding and alignment cosine similarity across different transformer layers on LaBSE and Qwen2.5-0.5B.

signal tends to weaken, potentially due to semantic abstraction. The statistically significant p-values across all layers confirm the robustness of this relationship. These findings reinforce the idea that high alignment may come at the cost of LID separability, especially in final layers for LaBSE and middle layer for Qwen2.5-0.5B, where representations are more semantically homogenized.

NTREX-128 For NTREX-128, the correlation trends diverge between the two models. LaBSE exhibits its strongest negative correlation in the the last layer (Pearson r = −0.621, p < 10−11), with a positive correlation in the first layer (Pearson r = 0.254, p = 0.012) and weak negative correlation in the middle (Pearson r = −0.173, p = 0.089). This suggests that early representations in LaBSE may still retain relatively distinct language features that diminish with depth. In con-

trast, Qwen2.5-0.5B shows more consistent negative correlations across all layers, particularly in the middle layer (Pearson r = −0.476, p < 10−6). These results highlight a more uniform degradation of LID-relevant information in Qwen’s architecture compared to LaBSE.

NusaX For NusaX, alignment-LID correlations exhibit distinct patterns. LaBSE shows a weak correlation in the first layer (Pearson r = −0.566, p = 0.112), a highly negative correlation in the middle layer (Pearson r = −0.872, p = 0.002), and no measurable correlation in the last layer (–), which we assume reflects a perfect inverse relationship (Pearson r ≈ −1) due to complete LID failure. Qwen2.5-0.5B follows a similar pattern, with its most negative correlation in the middle layer (Pearson r = −0.873, p = 0.002) and negligible correlations in the first (Pearson r = −0.455, p = 0.218) and last layers (Pearson r = −0.045, p = 0.910). The correlations for both models are the most negative observed across all datasets, suggesting alignment disproportionately degrades language signals in low-resource settings. This extreme inverse relationship likely stems from the models’ lack of prior exposure to NusaX languages during training, limiting their ability to retain language identity in aligned embeddings.

### C LID Methods and Results

#### C.1 Methods

To investigate language-specific information in multilingual representations, we analyze two distinct paradigms: (1) frozen embeddings from pretrained decoder-only LLMs (Qwen-2.5) and (2) specialized multilingual sentence encoders (LaBSE). We evaluate whether linguistic identity is recoverable from their hidden states and how

FLORES-200 NTREX-128 NusaX

###### Model Method Layer CLS Mean CLS Mean CLS Mean

First 80.65 88.35 87.02 90.43 64.12 81.78 Middle 65.11 78.85 71.37 81.30 33.89 45.37 Last 7.65 3.92 3.45 1.63 0.54 0.00

KNN

LaBSE

First 93.47 95.13 92.21 93.29 89.16 97.30 Middle 92.99 94.18 92.33 92.68 88.00 94.51 Last 30.03 70.89 22.91 74.36 56.00 65.44

Linear Probing

First – 83.69 – 86.06 – 65.79 Middle – 55.32 – 54.73 – 25.05 Last – 71.73 – 81.86 – 29.39

KNN

Qwen2.5-0.5B

First – 94.21 – 91.42 – 95.55 Middle – 91.76 – 90.04 – 87.09 Last – 92.46 – 90.27 – 88.77

Linear Probing

- Table 12: F1 score for KNN and linear classifiers by layer and pooling on FLORES-200, NTREX-128, and NusaX.

pooling strategies affect clusterability (via nonparametric KNN retrieval) and linear separability (via supervised classification heads).

KNN-based Language Identification We hypothesize that language identity manifests as separable clusters in the hidden space, which can be detected via non-parametric nearest-neighbor retrieval.

For both Qwen-2.5 and LaBSE, hidden states are extracted from the first (ℓ = 1), middle (ℓ = m), and final (ℓ = L) layers. Let Hℓ ∈ RT×d denote the hidden states at layer ℓ for a sequence of length T. Sentence-level embeddings are derived as follows:

- • Qwen-2.5: Only mean pooling is applied:

eℓmean =

1 T

T

t=1

Hℓt ∈ Rd.

- • LaBSE: Both CLS and mean pooling are compared:

1 T

eℓCLS = Hℓ[CLS], eℓmean =

T

Hℓt ∈ Rd.

t=1

For each layer ℓ ∈ {1,m,L} and pooling strategy pool ∈ {mean,CLS}, we construct reference sets:

200,204 i=1,j=1

Rℓpool = eℓ,pool(i,j),y(j)

,

where y(j) is the language label for the j-th language in FLORES-200, and i indexes the examples within each language. This results in a total of 200 × 204 = 40,800 reference embeddings. For Qwen-2.5, only Rℓmean is used, while LaBSE employs both RℓCLS and Rℓmean.

We evaluate on three test sets: Flores-200, NTREX-128, and NusaX. To ensure fair comparison, we retain only languages overlapping with the FLORES-200 train set:

Loverlap = Ltest ∩ LFLORES-train,

where Ltest is the language set of the test dataset, and LFLORES-train contains the 204 languages in the FLORES-200 train set. For a test embedding eℓtest,pool, we compute its L2 distance to all reference embeddings in Rℓpool:

2 2

d eℓtest,pool,eℓ,ref,pool(i,j) = eℓtest,pool − eℓ,ref,pool(i,j)

,

- ∀i ∈ {1,...,200},
- ∀j ∈ {1,...,204}.

The predicted language yˆtest is obtained via majority vote over the k = 256 nearest neighbors:

1(y(j) = l),

yˆtest = arg max

l∈Loverlap (i,j)∈Nk

where Nk denotes the set of indices for the top-k neighbors, and 1 is the indicator function.

Linear Classification Head To complement our non-parametric analysis, we probe the linear separability of language identity in Qwen-2.5 and LaBSE embeddings. This evaluates whether linguistic boundaries are geometrically aligned with hyperplanes in the hidden space, which would suggest that language control can be achieved through simple affine transformations.

Similar to the KNN-based approach, embeddings are extracted identically. For each dataset D ∈ {FLORES-200, NTREX-128, NusaX} and

each layer ℓ ∈ {1,m,L} representing early, middle, and last layers respectively, we train a separate linear layer to map embeddings eℓ ∈ Rd to language logits zℓ ∈ RC, where C is the number of languages. The classifier for each layer is defined as:

zℓ = Wℓeℓ + bℓ, Wℓ ∈ RC×d,bℓ ∈ RC, with cross-entropy loss minimized during training. C.2 Results

Our analysis reveals distinct layer-wise behaviors in language identification (LID) performance across LaBSE and Qwen2.5-0.5B models, focus on mean-pooled embedding.

KNN-based Language Identification The KNN method highlights significant performance variations across layers. As shown in Table 1, for LaBSE, the first layer achieves robust results, with mean F1 scores of 88.35% on FLORES-200, 90.43% on NTREX-128, and 81.78% on NusaX. Performance declines moderately in the middle layer, yielding 78.85% for FLORES-200, 81.30% for NTREX-128, and 45.37% for NusaX. The last layer exhibits catastrophic degradation, collapsing to 3.92%, 1.63%, and 0.00% on the respective datasets. This suggests that deeper LaBSE layers lose language-discriminative features critical for KNN classification.

For Qwen2.5-0.5B, the first layer similarly outperforms middle layers, with mean F1 scores of 83.69% on FLORES-200, 86.06% on NTREX-128, and 65.79% on NusaX. The middle layer shows the weakest results across all datasets: 55.32%, 54.73%, and 25.05%, respectively, while the last layer partially recovers to 71.73%, 81.86%, and 29.39%. This non-monotonic trend suggests limited retention of language-specific signals in the middle layer of Qwen2.5-0.5B.

LaBSE, trained for semantic alignment, shows severe degradation in its final layer, with near-zero

- F1 scores across datasets, as deeper layers erase language-specific signals required for KNN classification. In contrast, Qwen2.5-0.5B, a standard pretrained LLM, experiences a performance dip in its middle layer but recovers partially in the final layer, retaining sufficient linguistic discriminability. This divergence underscores a key architectural tradeoff: contrastive models like LaBSE discard lexical or syntactic patterns in deeper layers to prioritize semantic invariance, while standard LLMs preserve

partial language-identifying features across layers despite progressive abstraction.

Linear-probing-based Language Identification For LaBSE, the First Layer consistently achieves the highest LID F1 scores across all datasets, with a significant drop in performance observed in the Last Layer. The NusaX dataset delivers the best overall results, particularly in the First Layer, where it reaches 97.30% F1 score. However, the Last Layer shows notably weaker performance, especially for the FLORES-200 and NusaX datasets. These findings suggest that earlier layers of LaBSE retain more language-identification-relevant features, such as surface-level linguistic cues, compared to deeper layers (see Table 1).

Similarly, in the Qwen2.5-0.5B model, the First Layer consistently outperforms the Middle Layer in LID F1 scores across all datasets. The NusaX dataset again produces the best results, with 95.55% F1 score, while NTREX-128 exhibits the lowest performance across all layers. These results indicate that the shallow First Layer of Qwen2.5-0.5B is more effective for language identification tasks than deeper layers, such as the Middle Layer, which shows weaker performance (refer to Table 1).

Overall, both models show that their highest LID performance occurs in the First Layer, with F1 scores declining as the layers get deeper. The NusaX dataset consistently yields the best performance, while the Last Layer in LaBSE and the Middle Layer in Qwen2.5-0.5B exhibit the weakest results. These trends suggest that shallow layers retain more language-specific information, which is crucial for language identification, likely due to their greater focus on surface-level features and general linguistic patterns. Table 12 further illustrate the comparative performance across layers and pooling techniques for both LaBSE and Qwen2.5-0.5B models.

Classifier Comparison: KNN vs. Linear Head As shown in Table 12, linear classifiers achieve superior F1 scores compared to KNN across layers, suggesting their ability to identify languagediscriminative features within linearly separable subspaces. However, linear methods exhibit attenuated performance gaps between layers, for instance, the difference between first and middle layers in Qwen2.5-0.5B is less than 5% with linear classifiers, while KNN reveals differences exceeding 30%. Similarly, LaBSE’s linear classifier reduces the last-layer performance gap to under 25%,

whereas KNN shows near-complete degradation. This contrast implies that parametric linear methods, while more accurate overall, may obscure layer-specific language information degradation due to their reliance on learned projections. In contrast, KNN’s non-parametric nature might more directly reflect the geometric structure of embeddings, amplifying sensitivity to layer-wise shifts in language information quality.

Pooling Method Comparison: CLS Token vs. Mean As shown in Table 12, the effectiveness of pooling strategies varies across layers. In first and middle layers, mean pooling achieves superior performance, with F1 margins exceeding 10% over CLS token pooling under KNN. However, in last layers, CLS token pooling shows limited resilience under KNN, marginally outperforming mean pooling in isolated cases despite near-random overall performance. Linear classifiers amplify mean pooling’s advantage across all layers, suggesting its robustness to layer-specific degradation.

This suggests that mean pooling better preserves language-discriminative signals across layers, likely due to its aggregation of token-level features. In contrast, the CLS token, optimized for semantic tasks, exhibits sharper performance declines in deeper layers, particularly under nonparametric methods like KNN. These observations highlight the interplay between pooling strategy, layer depth, and classification method in language identification tasks.

### D Language Vector Setting

Linear Discriminant Analysis (LDA) (Balakrishnama and Ganapathiraju, 1998; Tharwat et al., 2017) is utilized to construct language vectors by extracting language-specific features from the Qwen2.5-0.5B model’s scaled hidden states, optimizing cross-lingual control through class separability. We evaluate various component sizes (20, 40, 50, 100, 150, 203) to balance LID accuracy and unused variance, fitting an LDA model and training a linear neural network (with 10 epochs, Adam optimizer, and CrossEntropyLoss) to achieve a peak accuracy of approximately 90.63% at 100 components. The unused variance is minimized, ensuring retained discriminative information for injection (δ) with pruning, which enhances language targeting while the Figure 10 visually confirms this optimal trade-off.

### E Ablation on Language Shift Strategy

Language Shift Strategy We assess various strategies for injecting the language vector in ITLC. Specifically, we explore three strategies based on the temporal scope of the latent intervention: (1) prompt only, (2) generated tokens only, and (3)

both phases. Let h(tm) ∈ Rd denote the hidden state at position t in the middle layer m, and h(m)

′

t

denotes its language-shifted counterpart:

- • Prompt-Only (prompt-only): Applies injection exclusively to input prompt processing:

h(m)

′

t =

h(tm) + αδ, ∀t ∈ [1,Tinput] h(tm), ∀t > Tinput

- • Generated-Only (gen-only): Restricts injection to autoregressive generation:

h(m)

′

t =

h(tm), ∀t ∈ [1,Tinput] h(tm) + αδ, ∀t ∈ [Tinput + 1,Ttotal]

- • Prompt and Generated (prompt-and-gen): Applies injection throughout both phases:

′

h(m)

t = h(tm) + αδ, ∀t ∈ [1,Ttotal]

where Tinput is the input prompt length and Ttotal = Tinput + N the total sequence length after generating N tokens.

Ablation Result All three language shift strategies are compared in cross-lingual setting using the Qwen2.5-0.5B and Qwen2.5-0.5B-Instruct, as shown in Figure 9. The prompt-and-gen strategy consistently achieves the strongest performance, followed by gen-only and then prompt-only. This indicates that while the prompt-only approach may aid the model in understanding the input context in the target language, and the gen-only strategy directly shifts the generation process into target language, while the prompt-and-gen method effectively combines both advantages via injecting the shift language vector into all timesteps.

### F Experiement Settings for Language Confusion

#### F.1 Baseline

The results discussed is focus on Line-level Pass Rate (LPR). Word-level Pass Rate (WPR) is mostly excluded in discussion because WPR for Latinscript languages is compromised by its fundamental reliance on Unicode character ranges, a

Qwen2.5-0.5B Qwen2.5-7B Llama-3.1-8B

Lang

Baseline ITLC Baseline ITLC Baseline ITLC chrF++ BERT F1 chrF++ BERT F1 chrF++ BERT F1 chrF++ BERT F1 chrF++ BERT F1 chrF++ BERT F1

ID 7.71 61.38 8.46 63.74 8.21 62.98 9.26 65.19 8.63 60.58 8.91 64.94 TH 3.39 62.12 3.42 63.78 3.62 62.55 3.88 63.90 2.96 59.02 4.28 64.37 TR 6.42 59.36 6.78 60.67 6.94 61.31 7.59 62.96 8.37 58.20 8.62 63.76 JA 1.90 59.98 2.11 61.53 2.08 60.14 1.84 61.15 1.52 53.26 2.60 62.94 FR 7.53 61.63 8.89 64.03 8.11 63.03 9.51 65.24 7.97 59.86 8.90 64.51 ES 8.51 62.66 9.43 64.90 9.30 64.24 10.01 65.65 8.69 61.14 9.84 65.65 AR 5.11 61.89 5.68 64.31 5.35 62.39 6.78 65.98 4.28 59.70 6.45 65.59 KO 1.86 60.93 2.08 61.90 2.09 61.67 2.14 62.35 2.14 54.61 3.31 65.19 ZH 2.61 62.26 2.97 64.85 2.73 62.93 3.33 65.18 2.00 55.14 2.67 63.98

AVG 5.01 61.36 5.53 63.30 5.38 62.36 6.04 64.18 5.39 58.39 6.76 64.29

- Table 13: Generation performance for different target languages on Qwen2.5 and Llama-3.1 base version. Baseline denotes the same model prompted in the same language as the desired target language. Bold values indicate the best score for each metric across all models and settings.

Lang

Qwen2.5-0.5B-Instruct Qwen2.5-7B-Instruct Llama-3.1-8B-Instruct

Baseline ITLC Baseline ITLC Baseline ITLC chrF++ BERT F1 chrF++ BERT F1 chrF++ BERT F1 chrF++ BERT F1 chrF++ BERT F1 chrF++ BERT F1

ID 7.71 61.38 8.46 63.74 8.21 62.98 9.26 65.19 9.67 64.58 11.55 66.97 TH 3.39 62.12 3.42 63.78 3.62 62.55 3.88 63.90 5.42 64.68 6.67 67.82 TR 6.42 59.36 6.78 60.67 6.94 61.31 7.59 62.96 9.37 63.37 10.49 65.15 JA 1.90 59.98 2.11 61.53 2.08 60.14 1.84 61.15 3.33 63.29 4.11 66.23 FR 7.53 61.63 8.89 64.03 8.11 63.03 9.51 65.24 9.44 64.28 11.40 67.52 ES 8.51 62.66 9.43 64.90 9.30 64.24 10.01 65.65 10.32 64.78 12.24 67.68 AR 5.11 61.89 5.68 64.31 5.35 62.39 6.78 65.98 6.88 64.82 8.66 67.55 KO 1.86 60.93 2.08 61.90 2.09 61.67 2.14 62.35 3.74 64.52 4.59 66.99 ZH 2.61 62.26 2.97 64.85 2.73 62.93 3.33 65.18 2.58 64.26 3.70 66.82

AVG 5.41 61.79 6.11 63.74 5.82 63.24 6.48 64.96 6.97 64.80 8.49 67.19

- Table 14: Generation performance for different target languages on Qwen2.5 and Llama-3.1 Instruction version. Baseline denotes the same model prompted in the same language as the desired target language. Bold values indicate the best score within each model, and the overall best across models.

[Figure 10]

- Figure 9: Cross-lingual LPR performance across different vector injection strategies.

limitation highlighted in (Marchisio et al., 2024). For Latin-script WPR evaluation, we use the following Unicode ranges: Basic Latin, Latin1 Supplement, Latin Extended-A through Latin

Extended-G, and Latin Extended Additional 6. We use the following generation hyperparameters: max_new_tokens=256 and top_k=50. We apply nucleus sampling with top_p=0.9 and use a moderate temperature of 0.7.

#### F.2 In-context learning (ICL)

We follow all the original settings for ICL in the LCB benchmark. For the Q/A template, we use the Q: A: format, while the chat template adopts the model-specific instruction-tuning structure. Crosslingual few-shot prompts follow the benchmark’s original setup, where English inputs include instructions such as Respond in <TARGET_LANG>. For monolingual few-shot prompts, both inputs and outputs are in the same language.It is important to note that the few-shot demonstrations are provided in languages distinct from the target language of the current task. This design aims to guide the

6Taken from Wikipedia: https://en.m.wikipedia. org/wiki/Unicode_block.

0.925

0.900

0.875

0.850

Accuracy(%)

0.825

0.800

0.775

0.750

25 50 75 100 125 150 175 200 Number of Components

0.200

0.175

0.150

UnusedVariance

0.125

0.100

0.075

0.050

0.025

0.000

25 50 75 100 125 150 175 200 Number of Components

- Figure 10: Controlling the number of language feature representations by using LDA performance accuracy (Left) and unused variance (Right) across number of components.

Lang Shift Prompt (XX) Generated Output (EN) Translation (XX)

KO → EN 사과먹는 5가지방법은무엇일까요? Here are five common ways to enjoy apples: 1. Raw and fresh : Eating an apple straight from the fruit bowl is a simple yet delicious way of enjoying it. 2. Juiced or blended into smoothies: Apples can be ju . . .

What are 5 ways to eat apples?

ES → EN ¿ Cómo escapar de un helicóptero atrapado en el agua?

Si usted encuentra sí mismos stuck inside a helicopter that has crashed into the water: 1. Remain calm: Panic can cloud your judgment and increase risk of injury or drowning. 2. Verify everyone’s safety first before attempting to escape from any. . .

How to escape from a helicopter trapped in water?

JA → EN 音楽家が音楽から収入を得る 方法のリストを用意してくださ い.

Here are some ways that a musician can earn income from their music: 1￿”0307Live Performances: Playing concerts, gigs and shows at various venues such as clubs, festivals or events. 2Ṙecording Music Sales : Selling albums through digital platforms like . . .

Provide a list of ways musicians can earn income from their music craft. . . .

Figure 11: Examples of generated outputs from Llama-3.1-8B-Instruct with injection in XX→EN.

model in understanding the task semantics rather than identifying the appropriate output language.

#### F.3 Inference-Time Language Control (ITLC)

We use the following scaling factor α values: for Qwen2.5-0.5B and Qwen2.5-0.5B-Instruct, α = 0.5; for Qwen2.5-7B and Qwen2.5-7B-Instruct, α = 1.3. For Llama-3.1-8B, α = 0.15; and for Llama-3.1-8B-Instruct, α = 0.10. See Appendix G.1 for details on the selection of scaling factor.

#### F.4 Parameter-Efficient Fine-Tuning (PEFT)

For training data preparation, we use the Alpaca subset from the Bactrian-X dataset (Li et al., 2023a). Specifically, we extract the first 1,000 rows per language across the 14 languages included in the LCB benchmark (500 for monolingual and 500 for cross-lingual), resulting in a total of 14,000 samples. For monolingual data, we directly use the same samples as in Bactrian-X. For cross-lingual data, we replace the inputs with their corresponding

English variants in Bactrian-X and append the instruction "Please respond in <TARGET_LANG>" to the prompt. For training data of the base model, we concatenate the inputs and outputs and treat the result as the output sequence without applying the Q/A template.

The model is trained for one epoch using the LoRA fine-tuning technique (Hu et al., 2022) with the following settings: warmup_ratio = 0.05, batch_size = 1, gradient_accumulation_steps = 16, learning_rate = 1e-4, lora_rank = 8, and lora_alpha = 32. We employ the MS-SWIFT framework (Zhao et al., 2024b) for model training, with all other hyperparameters kept at their default values.

#### F.5 Combination of ICL and ITLC

We apply ITLC only to the current input prompt using the scaling factor specified in Appendix F.3, but not to the few-shot examples, since the languages used in the few-shot examples differ from that of

Lang Shift Prompt (EN) Generated Output (XX) Literal Translation (EN) EN → TR How can you open a can without using your hands?

Bir kâğıt parçasını, delikten geçir ve tutacak şekilde yerleştir. Kapağın üzerine basınçlı bir cisimle vurun (örneğin bıçak). Bu sayede metalin çarka girmesiyle . . .

Pass a piece of paper through the hole and place it so that it holds. Strike the cover with a pressurized object (for example, a knife). In this way, with the metal entering the wheel... . . .

EN → ES Can you help me avoid credit card debt?

¡Claro que sí, puedo ayudarte a evitar deudas con tarjetas crédito. Aquí te presento algunas consejos: 1) Crea un presupuesto y controla tus gastos para saber cuántos ingresos tienes. . .

Of course! I can help you avoid credit card debt. Here are some tips: 1) Create a budget and track your expenses. This will help you understand how much income you have and where your money is going.

EN → TR What shapes are made up of straight lines?

Evinizin değerini artırmak için bazı yollar şunlardır: 1. Dekorasyon: Evi modern ve şık bir şekilde dekore etmek, evin görünümünü iyileştirir. 2. Ren . . .

Here are some ways to increase the value of your home: 1. Decoration: Decorating the house in a modern and stylish way improves its appearance. 2. Paint ⋯. . .

Figure 12: Examples of generated outputs from Llama-3.1-8B-Instruct with injection in EN→XX.

the current task (see Appendix F.2).

#### F.6 Combination of PEFT and ITLC

The LDA is trained based on hidden states extracted from the pre-trained model. We do not retrain the LDA using new hidden states after PEFT. For other details, please refer to Appendix F.4 and Appendix F.3

#### F.7 INCLINE

We extract sentence representations from the FLORES-200 dataset (Team et al., 2022) and use the same scaling factor α as defined in Appendix F.3 during inference.

#### F.8 ReCoVeR

We extract sentence representations from the FLORES-200 dataset (Team et al., 2022) and apply a scaling factor of α = 0.2 for Llama-3.1-8B and its instruct variant, and α = 0.3 for Qwen2.5-0.5B and its instruct variant.

### G Language Confusion Result

- G.1 Ablation Study of Scaling for Different Language Vector Injection Strategies

As shown in Table 15, Table 16 and Table 17 Our analysis reveals distinct optimal scaling factors for cross-lingual LCPR across injection strategies: prompt-only achieves peak performance at scaling 0.8 (65.71), gen-only at 0.6 (71.35), and promptand-gen at 0.5 (78.93). Notably, prompt-and-gen outperforms other strategies, suggesting combined injection better preserves cross-lingual alignment. The scaling factor for the Qwen2.5-0.5B model family is adopted from our ablation study. However, due to computational constraints, a similar study was not feasible for the Qwen2.5-7B and

Scaling Monolingual Cross-lingual LCPR LPR WPR LCPR LPR WPR

- prompt-0.1 64.86 81.01 65.67 33.97 23.75 74.74
- prompt-0.2 66.39 82.14 66.75 38.88 28.91 75.37
- prompt-0.3 65.59 82.86 65.78 46.03 37.86 72.56
- prompt-0.4 65.45 82.79 65.53 57.20 51.97 72.27
- prompt-0.5 65.87 82.73 62.50 62.93 61.63 73.43
- prompt-0.6 64.92 82.64 65.24 63.91 63.83 73.20
- prompt-0.7 64.78 81.03 65.52 64.63 66.09 71.74
- prompt-0.8 63.69 80.40 65.28 65.71 66.41 74.24
- prompt-0.9 61.25 75.81 64.15 64.59 64.79 73.30 prompt-1.0 60.39 74.98 63.87 62.97 63.35 72.79

- Table 15: Performance (LCPR / LPR / WPR) of Qwen2.5-0.5B on LCB under the prompt-only setting with base shift vector, evaluated across different language vector scaling factors, α.

Scaling Monolingual Cross-lingual LCPR LPR WPR LCPR LPR WPR

- gen-0.1 64.75 83.99 63.85 35.07 24.79 74.92
- gen-0.2 65.35 85.09 65.01 39.93 28.96 75.92
- gen-0.3 62.61 86.55 59.29 48.08 38.97 71.16
- gen-0.4 59.61 86.23 54.95 57.49 57.82 64.37
- gen-0.5 59.61 86.85 54.76 67.00 74.04 66.07
- gen-0.6 60.05 87.49 58.14 71.35 80.46 67.67
- gen-0.7 58.01 87.41 55.72 69.39 80.73 66.57
- gen-0.8 52.45 82.78 52.35 65.84 75.74 65.93
- gen-0.9 47.07 75.83 50.58 58.61 68.51 63.73 gen-1.0 40.44 71.15 54.91 51.25 61.85 61.83

- Table 16: Performance (LCPR / LPR / WPR) of Qwen2.5-0.5B on LCB under the generated-only setting with base shift vector, evaluated across different language vector scaling factors, α.

Llama3.1-8B families. For these models, we instead conducted a limited manual evaluation, we randomly generated outputs for a range of scaling factors across different target languages and selected the best-performing value based on human assessment.

Scaling Monolingual Cross-lingual LCPR LPR WPR LCPR LPR WPR

- prompt-and-gen-0.1 64.21 84.27 63.77 39.48 28.69 75.74
- prompt-and-gen-0.2 63.25 86.34 61.76 50.04 41.18 75.07
- prompt-and-gen-0.3 62.94 88.24 60.85 64.22 64.18 72.53
- prompt-and-gen-0.4 60.79 88.06 59.09 75.88 80.58 75.78
- prompt-and-gen-0.5 59.98 87.11 59.41 78.93 85.08 77.15
- prompt-and-gen-0.6 57.01 86.37 55.90 77.21 84.13 74.90
- prompt-and-gen-0.7 53.56 82.91 53.63 72.57 81.98 71.51
- prompt-and-gen-0.8 49.00 77.27 51.33 68.22 76.80 70.08
- prompt-and-gen-0.9 40.41 70.51 48.16 60.97 69.07 66.44 prompt-and-gen-1.0 36.60 70.01 51.30 52.51 61.07 63.82

Table 17: Performance (LCPR / LPR / WPR) of Qwen2.5-0.5B on LCB under the prompt-and-generated setting with base shift vector, evaluated across different language vector scaling factors, α.

- G.2 Impact of In-context learning (ICL) on Monolingual and Cross-lingual Performance

As shown in Table 18, Table 19, Table 20, Table 21, Table 22 and Table 23, in the monolingual setting, the impact of few-shot prompting varies inconsistently across models. Qwen2.5-0.5B and Qwen2.5-0.5B-Instruct exhibit decreased LPR, while Qwen2.5-7B and Llama-3.1-8B show increased LPR. For instruction-tuned models, both Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct demonstrate reduced LPR. This unstable and unpredictable behavior may stem from the design of monolingual few-shot prompts, which introduce conflicting linguistic signals that models with limited capacity struggle to resolve effectively 7.

In the cross-lingual setting, few-shot prompting consistently improves performance across all base models (Qwen2.5-0.5B, Qwen2.5-7B, and Llama-3.1-8B). This improvement can be attributed to the few-shot examples, which utilize English inputs paired with explicit target-language directives, thereby reinforcing the desired input-output alignment. These results indicate that Englishcentric prompting effectively stimulates crosslingual adaptation in base models. However, the effect differs for instruction-tuned models: while smaller models like Qwen2.5-0.5B-Instruct benefit from few-shot examples, larger models (Qwen2.57B-Instruct and Llama-3.1-8B-Instruct) show minimal gains. This stability suggests that instructiontuning pre-aligns their multilingual capabilities, rendering additional in-context examples largely redundant.

The divergent impact of ICL across models in-

7Please refer to Appendix F.2.

Method Monolingual Cross-lingual LCPR LPR WPR LCPR LPR WPR

Qwen2.5-0.5B 65.27 81.58 65.15 29.41 19.75 73.45 + Q/A template (0-shot) 59.26 59.91 73.35 44.68 35.36 75.94

+ PEFT 75.96 82.91 78.30 76.15 77.55 80.56

- + 1-shot 56.12 55.38 73.70 47.42 37.95 75.42
- + 2-shot 51.59 49.70 70.98 49.36 41.64 75.03
- + 3-shot 52.52 51.51 72.07 53.16 46.65 77.07
- + 4-shot 54.16 52.95 74.15 55.03 48.23 77.60
- + 5-shot 54.47 53.62 70.40 56.78 50.63 76.16

+ ITLC (apply base shift vector)

+ prompt-only (α = 0.8) 63.69 80.40 65.28 65.71 66.41 74.24 + gen-only (α = 0.6) 60.05 87.49 58.14 71.35 80.46 67.67 + prompt-and-gen (α = 0.5) 59.98 87.11 59.41 78.93 85.08 77.15

+ Q/A template 62.50 81.21 64.60 81.30 85.61 80.84 + PEFT 73.68 86.17 73.26 87.66 90.51 86.15 + 5-shot 57.65 74.38 61.13 81.51 87.58 79.01

+ ITLC (apply instruct shift vector)

+ prompt-only (α = 0.8) 63.11 79.95 64.18 63.08 63.77 73.04 + gen-only (α = 0.6) 55.89 86.38 55.32 68.70 78.99 65.36 + prompt-and-gen (α = 0.5) 58.48 87.24 57.21 76.06 82.31 75.74

Table 18: Performance (LCPR / LPR / WPR) of Qwen2.5-0.5B on LCB under monolingual and crosslingual settings.

dicates that the effectiveness of few-shot prompting might contingent upon the model’s instructionfollowing aptitude, contextual understanding, preexisting upper-bound capability, and the depth of alignment achieved during its instruction-tuning process 8.

- G.3 Chat/QA Template Efficacy Across Settings

The findings are consistent with those observed in the in-context learning (ICL) setting for LPR performance, with one key exception: applying the chat template to instruction-tuned models consistently yields better performance, as shown in Table 19.

- G.4 Effect of Source Language Shift Vector

As shown in Figure 13, subtracting the source language shift vector reduces the model’s bias toward the source language (English) and guides the model to generate content in the target language more effectively, compared to directly adding the target language shift vector.

### H Experiment setting for semantic retention and human evaluation

#### H.1 Generation Hyperparameter

The generation process for the language control and language confusion results uses specific hyperparameter to balance creativity and control. We set max_new_tokens=50, and set top_k to 50. We

8All discussed results are based on experiments that apply the official chat/QA templates during inference.

Method Monolingual Cross-lingual LCPR LPR WPR LCPR LPR WPR

Qwen2.5-0.5B-Instruct 74.79 82.61 77.94 38.75 27.22 78.40 + Chat template (0-shot) 74.52 83.66 77.12 63.00 57.69 79.50

+ PEFT 80.13 89.85 77.77 79.46 84.34 80.01

- + 1-shot 72.94 78.83 77.79 66.82 61.42 82.12
- + 2-shot 73.95 78.41 79.43 68.19 64.21 80.99
- + 3-shot 74.61 78.88 76.99 69.43 65.94 81.42
- + 4-shot 75.82 80.89 80.07 69.56 67.28 79.62
- + 5-shot 75.44 80.30 79.36 71.43 69.70 79.74

+ ITLC (apply base shift vector)

+ prompt-only (α = 0.8) 67.33 74.82 76.35 76.05 77.68 81.11 + gen-only (α = 0.6) 67.00 84.07 65.83 75.56 82.42 74.51

- + prompt-and-gen (α = 0.5) 67.73 81.70 68.96 81.51 85.32 80.55

+ ITLC (apply instruct shift vector)

+ prompt-only (α = 0.8) 66.78 74.96 73.08 73.26 76.37 79.20 + gen-only (α = 0.6) 67.42 83.64 65.46 73.95 84.06 71.40

- + prompt-and-gen (α = 0.5) 68.20 82.20 68.05 80.96 86.79 78.84

+ 5-shot 68.93 86.28 66.47 83.98 88.07 82.00 + PEFT 68.16 90.51 62.58 85.38 89.85 82.83

- Table 19: Performance (LCPR / LPR / WPR) of Qwen2.5-0.5B-Instruct on LCB under monolingual and cross-lingual settings.

[Figure 11]

Figure 13: Cross-lingual LPR performance on LCB with and without subtracting the source language shift vector across Qwen2.5-0.5B and Qwen2.5-0.5B-Instruct, using prompt-and-gen injection strategy with α = 0.5.

Method Monolingual Cross-lingual LCPR LPR WPR LCPR LPR WPR

Qwen2.5-7B 68.15 77.71 71.40 41.03 29.72 75.33 + Q/A template (0-shot) 53.97 55.24 73.84 65.68 60.61 76.88

+ PEFT 73.46 83.80 72.80 78.93 82.66 79.51 + 5-shot 63.23 62.78 75.77 72.15 69.37 79.45

+ ITLC (apply base shift vector)

+ prompt-and-gen (α = 1.3) 67.05 80.07 67.33 61.70 59.84 70.84 + Q/A template 58.10 63.40 72.36 70.71 74.40 72.72 + PEFT 73.12 85.60 72.40 78.25 83.92 78.39 + 5-shot 65.24 69.55 73.42 79.60 84.90 77.13

- Table 20: Performance (LCPR / LPR / WPR) of Qwen2.5-7B on LCB under monolingual and crosslingual settings.

apply nucleus sampling with top_p=0.9, and use a moderate temperature of 0.7 to encourage focused yet varied outputs. To reduce repetitive phrases, we apply a repetition_penalty of 1.5. We keep all other hyperparameters at their model-specific default values and use each instruct model’s native

Method Monolingual Cross-lingual LCPR LPR WPR LCPR LPR WPR

Qwen2.5-7B-Instruct (with chat template) 60.83 78.89 58.78 66.16 78.81 62.37 + 5-shot 54.46 74.13 53.93 65.79 78.51 61.44 + PEFT 75.03 88.28 73.19 78.32 83.56 77.93

+ ITLC (apply base shift vector)

+ prompt-and-gen (α = 1.3) 62.44 85.89 56.76 66.91 83.45 60.34

+ ITLC (apply instruct shift vector)

+ prompt-and-gen (α = 1.3) 61.35 84.89 56.97 66.89 84.73 60.02 + 5-shot 57.75 81.01 53.73 66.26 84.04 58.97 + PEFT 75.62 90.12 72.33 77.50 84.10 76.70

- Table 21: Performance (LCPR / LPR / WPR) of Qwen2.5-7B-Instruct on LCB under monolingual and cross-lingual settings.

Method Monolingual Cross-lingual LCPR LPR WPR LCPR LPR WPR

Llama-3.1-8B 43.52 44.07 59.66 1.46 0.74 88.10 + Q/A template (0-shot) 63.68 56.98 82.26 39.01 26.13 87.27

+ PEFT 79.16 93.01 72.80 82.04 89.73 77.83 + 5-shot 72.24 69.86 79.13 70.67 62.38 83.91

+ ITLC (apply base shift vector)

+ prompt-and-gen (α = 0.15) 50.97 60.77 57.07 60.69 69.69 57.74 + Q/A template 73.13 75.77 77.28 81.29 81.68 82.78 + PEFT 78.50 96.03 72.08 83.74 88.98 81.21 + 5-shot 76.43 82.18 76.47 83.14 88.15 80.47

- Table 22: Performance (LCPR / LPR / WPR) of Llama3.1-8B on LCB under monolingual and cross-lingual settings.

Method Monolingual Cross-lingual LCPR LPR WPR LCPR LPR WPR

Llama-3.1-8B-Instruct (with chat template) 83.05 94.63 76.11 79.34 83.25 77.01 + 5-shot 82.27 88.57 79.88 84.32 86.68 82.77 + PEFT 79.00 96.66 71.00 81.26 91.13 75.29

+ ITLC (apply base shift vector)

+ prompt-and-gen (α = 0.10) 82.50 95.68 75.68 83.48 88.52 80.37

+ ITLC (apply instruct shift vector)

+ prompt-and-gen (α = 0.10) 81.76 96.41 74.51 82.91 89.06 78.99 + 5-shot 85.25 93.21 79.82 86.60 90.34 83.95 + PEFT 79.04 97.19 71.36 83.44 93.60 77.05

- Table 23: Performance (LCPR / LPR / WPR) of Llama3.1-8B-Instruct on LCB under monolingual and crosslingual settings.

chat template. H.2 Monolingual & Crosslingual Prompting

Our experiments on the baseline (monolingual) and ITLC (cross-lingual) settings use slightly different prompt strategies. Specifically, for the baseline, we aim to measure the upper bound of performance within a particular language, whereas ITLC involves different input and target languages.

To ensure fairness and consistency in model output generation, we designed distinct input prompts for the base model, Qwen2.5, and Llama-3.1. In the base version, to control the contextual generation in cross-lingual settings, we prepend an early portion of the target language output—approximately 30% of the sentence length—as a guidance signal for the model to continue generating coherent text. This approach helps ensure that the language vector

receives sufficient signal to produce linguistically and semantically coherent outputs.

Additionally, for non-Latin scripts such as Japanese and Chinese, we adopt a different segmentation strategy. Instead of splitting based on newlines, as in Latin-script languages, we apply language-specific tokenizers such as PyThaiNLP (Phatthiyaphaibun et al., 2023), Nagisa9, and Jieba10. The proportional segment length is then determined based on the number of tokens or phrases produced by these tokenizers.

### I Additional Examples of Cross-lingual Generation

Figure 11 and Figure 12 present several examples of generated outputs across multiple source languages targeting English. Overall, our ITLC method successfully shifts to the desired target language and demonstrates effective cross-lingual generation.

### J Annotation Guidelines

#### J.1 Context of the Annotation Task

The annotation task involves evaluating the quality of cross-lingual language generation, where a model generates responses in a target language based on input prompts in a source language. The goal is to assess how well the model performs in terms of naturalness, relevance, and answer correctness. This evaluation is crucial for understanding the model’s capabilities and identifying areas for improvement.

#### J.2 Detailed Scoring Guidelines J.2.1 Naturalness (1-5):

- • 1: The response sounds very unnatural, robotic, or translated. It lacks fluency and typical language patterns of the target language, making it sound artificial and unnatural.
- • 2: The response is somewhat unnatural, with noticeable awkwardness or unnatural word choices. It may sound stilted or forced.
- • 3: The response is moderately natural, with some minor awkwardness but generally understandable. It flows reasonably well but has room for improvement.
- • 4: The response is mostly natural, with only slight deviations from typical language use. It

- 9https://github.com/taishi-i/nagisa
- 10https://github.com/fxsjy/jieba

sounds almost native-like but may have minor imperfections.

• 5: The response is completely natural, indistinguishable from text written by a native speaker. It flows smoothly and uses language patterns typical of the target language.

#### J.2.2 Relevance (1-5):

- • 1: The response is completely irrelevant to the input prompt. It fails to address the topic or question posed.
- • 2: The response is somewhat relevant but misses key points or goes off-topic. It may touch on related ideas but does not fully address the prompt.
- • 3: The response is moderately relevant, addressing some aspects of the prompt but lacking completeness. It covers some key points but omits important details.
- • 4: The response is highly relevant, addressing most key points of the prompt. It provides a comprehensive answer but may miss minor details.
- • 5: The response is completely relevant, fully addressing all aspects of the prompt. It covers all key points and provides a thorough answer.

#### J.2.3 Correctness (1-5):

- • 1: The response contains major factual errors or inaccuracies. It provides incorrect information or contradicts known facts.
- • 2: The response contains some factual errors or inaccuracies. It may be partially correct but includes misleading or incorrect details.
- • 3: The response is mostly correct but may have minor inaccuracies or omissions. It is generally accurate but requires minor corrections.
- • 4: The response is highly accurate, with only minor details potentially incorrect. It is reliable and trustworthy but may have small errors.
- • 5: The response is completely accurate and factually correct. It provides precise and reliable information without any errors.

#### J.3 Additional Notes

• Contextual Understanding: Annotators should consider the context of the input prompt and the intended audience when evaluating naturalness and relevance. A response

that is natural and relevant in one context may not be in another.

- • Consistency: Annotators should strive for consistency in their annotations across different examples. This helps ensure that the evaluation is fair and reliable.
- • Examples: Providing clear examples of each rating level for each category can help annotators understand the expected standards and make consistent judgments.
- • Feedback: Encourage annotators to provide feedback on ambiguous cases or areas where the guidelines could be improved. This can help refine the annotation process and improve the quality of the evaluations.

### K Authors’ Contributions

Joanito Agili Lopo is primarily responsible for developing the methodology as well as conducting experiments related to semantic retention and human evaluation. Muhammad Ravi Shulthan Habibi is responsible for the representation alignment and language identification (LID) experiments using linear probing. Tack Hwa Wong co-developed the methodology with Joanito Agili Lopo and is responsible for the LID experiments using k-nearest neighbours (KNN), as well as all experiments related to language confusion and language confusion benchmark (LCB). Samuel Cahyawijaya provided the research topic, overall direction, ideas, and guidance throughout the entire work. All co–first authors contributed to the writing of the first draft, and all authors participated in the review and editing process.

Cross-lingual Model AVG AR ES HI ID RU ZH

Qwen2.5-0.5B 34.97 31.72 48.12 3.03 42.44 48.77 35.74 + INCLINE 43.82 34.94 74.17 6.58 56.38 59.22 31.63 + ReCoVeR 88.43 99.66 97.02 64.67 84.88 98.99 85.38 + ITLC (ours) 81.22 98.32 94.61 32.32 83.17 97.65 81.25

Llama-3.1-8B 25.05 10.60 37.63 25.71 38.13 17.61 20.59 + INCLINE 34.69 19.61 39.25 38.92 40.46 32.36 37.56 + ReCoVeR 88.79 100.00 84.30 93.44 70.97 98.69 85.37 + ITLC (ours) 76.38 90.41 83.57 76.43 62.37 97.29 48.24

Table 24: LPR metrics for the base model on LCB across baseline and state-of-the-art methods, with a detailed language-wise breakdown for cross-lingual settings. All results have been applied with the QA/Chat template during inference.

Cross-lingual Model AVG AR ES HI ID RU ZH

Qwen2.5-0.5B-Instruct 52.28 65.41 72.65 3.02 54.35 77.12 41.14 + INCLINE 56.54 68.35 80.35 1.13 52.19 68.08 69.16 + ReCoVeR 84.21 100.00 97.66 60.36 58.86 99.31 89.04 + ITLC (ours) 81.97 98.97 95.31 49.03 64.39 98.98 85.13

Llama-3.1-8B-Instruct 80.68 87.12 89.27 82.76 73.89 87.93 63.14 + INCLINE 80.63 86.80 89.60 81.10 70.21 86.58 69.51 + ReCoVeR 90.29 100.00 93.30 95.24 67.96 99.32 85.92 + ITLC (ours) 85.65 95.60 92.96 93.97 72.55 95.98 62.84

Table 25: LPR metrics for the instruct model on LCB across baseline and state-of-the-art methods, with a detailed language-wise breakdown for cross-lingual settings. All results have been applied with the QA/Chat template during inference.

Monolingual Model AVG AR DE EN ES FR HI ID IT JA KO PT RU TR VI ZH

Qwen2.5-0.5B 59.91 45.84 78.79 97.00 75.20 64.67 0.00 57.00 76.00 32.00 54.55 64.00 64.29 30.00 81.82 77.50 + ICL (5-shot) 53.62 56.12 84.00 96.44 64.86 54.53 4.17 64.00 65.66 19.19 40.40 45.00 73.74 25.00 68.37 42.81

- + ITLC (ours) 74.38 77.94 94.00 99.49 94.33 89.67 0.00 78.00 77.00 55.56 74.75 79.50 87.00 55.00 74.00 79.50

- + PEFT 82.91 94.00 99.00 70.50 93.00 94.67 0.00 90.00 98.00 64.00 86.00 91.00 95.00 92.00 94.00 82.50

+ ITLC (ours) 86.17 99.33 100.00 77.00 99.33 99.33 8.25 94.00 100.00 57.00 81.82 98.00 98.00 99.00 90.00 91.50 + ITLC (ours) 81.21 91.00 96.00 97.98 98.67 98.00 0.00 84.00 100.00 58.00 81.00 95.00 81.00 75.00 70.00 92.50 Qwen2.5-7B 55.24 29.43 73.00 98.48 70.04 66.17 1.01 63.00 78.00 39.00 22.68 65.00 36.08 26.80 83.00 76.85 + ICL (5-shot) 62.78 43.26 79.00 96.39 71.84 74.02 15.96 73.74 82.00 59.00 44.79 50.36 65.66 56.25 82.00 47.50

+ ITLC (ours) 69.55 51.22 86.87 97.94 77.44 82.25 8.70 86.00 91.00 64.00 53.54 64.50 77.55 56.25 89.00 57.00

- + PEFT 83.80 95.00 99.00 49.58 94.00 94.00 6.06 91.00 97.98 75.00 85.00 91.94 96.00 94.00 100.00 88.50

- + ITLC (ours) 85.60 98.67 99.00 52.97 97.67 95.67 8.00 95.00 94.00 76.00 89.00 95.00 97.00 97.00 100.00 89.00

+ ITLC (ours) 63.40 52.79 76.00 98.99 84.71 77.53 0.00 75.00 78.00 45.92 31.00 77.44 60.61 21.65 85.86 85.50 Llama-3.1-8B 56.98 39.57 55.00 95.38 69.56 59.43 30.21 57.58 55.56 25.51 43.30 71.29 67.37 81.82 61.00 42.19 + ICL (5-shot) 69.86 67.53 75.00 95.47 69.33 63.67 63.64 73.00 73.00 67.00 49.00 67.82 70.00 70.00 76.00 67.50

+ ITLC (ours) 82.18 79.26 90.00 99.50 92.67 84.00 65.00 66.00 90.00 87.00 68.37 86.92 89.00 70.00 87.00 78.00 + PEFT 93.01 98.00 98.00 69.50 94.67 92.00 92.00 84.00 99.00 94.00 93.00 93.50 97.00 95.00 98.00 97.50 + ITLC (ours) 96.03 100.00 97.00 91.50 97.67 97.33 96.00 91.00 99.00 95.00 97.00 92.50 99.00 99.00 98.00 90.50 + ITLC (ours) 75.77 62.08 78.00 99.00 89.29 84.02 50.00 66.67 81.00 58.76 76.84 85.78 93.81 86.73 75.51 49.00

Cross-lingual Model AVG AR DE EN ES FR HI ID IT JA KO PT RU TR VI ZH

Qwen2.5-0.5B 35.36 31.72 43.27 – 48.12 46.45 3.03 42.44 40.33 14.40 10.12 45.11 48.77 34.23 51.28 35.74 + ICL (5-shot) 50.63 54.79 63.97 – 54.62 63.02 12.07 61.97 63.05 24.74 29.57 55.90 67.84 61.61 69.21 26.38

+ ITLC (ours) 87.58 99.66 97.99 – 96.62 97.33 39.07 85.52 95.26 72.91 88.95 90.95 98.99 91.96 92.63 78.24 + PEFT 77.55 89.25 90.26 – 90.94 90.94 11.04 75.41 88.25 68.52 65.32 82.23 90.94 83.53 90.26 68.84 + ITLC (ours) 90.51 100.00 99.67 – 96.65 97.32 63.78 85.61 98.99 69.22 88.97 90.97 99.67 96.99 96.99 82.26

- + ITLC (ours) 85.61 98.32 96.97 – 94.61 95.63 32.32 83.17 99.00 61.20 82.55 88.28 97.65 92.96 94.60 81.25

Qwen2.5-7B 60.61 62.24 67.82 – 71.07 68.68 24.87 60.80 67.31 51.90 50.29 68.40 69.21 59.40 72.07 54.42 + ICL (5-shot) 69.37 70.22 77.42 – 75.04 75.20 36.45 70.53 81.43 59.16 59.26 70.02 84.24 77.05 79.20 55.94

+ ITLC (ours) 84.90 88.57 95.50 – 90.40 92.14 65.67 84.03 90.37 57.86 85.17 88.74 94.48 91.58 90.92 73.18 + PEFT 82.66 93.62 93.23 – 89.27 89.93 24.20 83.16 86.25 76.87 80.56 86.84 95.29 91.57 90.93 75.53

+ ITLC (ours) 83.92 97.65 97.95 – 96.99 95.31 30.78 87.60 93.97 35.09 74.47 92.59 97.65 96.99 96.98 80.89 + ITLC (ours) 74.40 83.00 89.49 – 89.51 87.43 27.12 76.65 87.42 32.80 58.81 87.35 91.49 82.97 85.93 61.61 Llama-3.1-8B 26.13 10.60 28.03 – 37.63 36.09 25.71 38.13 37.14 18.88 16.49 31.77 17.61 20.14 27.05 20.59 + ICL (5-shot) 62.38 65.02 60.66 – 66.88 56.64 65.72 71.81 65.46 46.49 68.77 56.07 69.50 73.40 63.12 43.83

+ ITLC (ours) 88.15 85.24 96.97 – 87.62 84.40 76.23 76.51 87.56 93.79 96.94 89.34 99.66 92.87 92.58 74.44 + PEFT 89.73 93.61 92.27 – 91.28 93.64 93.62 76.16 89.60 85.57 85.50 89.22 94.24 92.28 94.30 84.90 + ITLC (ours) 88.98 98.99 96.96 – 86.21 75.21 98.65 67.22 89.96 88.95 95.61 84.58 99.33 95.31 92.96 75.86 + ITLC (ours) 81.68 90.41 96.13 – 83.57 71.68 76.43 62.37 89.12 75.72 89.11 82.56 97.29 87.75 93.09 48.24

- Table 26: LPR metrics for the base model on LCB, with a detailed language-wise breakdown for both monolingual and cross-lingual settings. All results have been applied with the QA/Chat template during inference.

Monolingual Model AVG AR DE EN ES FR HI ID IT JA KO PT RU TR VI ZH

Qwen2.5-0.5B-Instruct 83.66 96.33 94.00 99.50 89.67 95.33 0.00 70.00 94.00 82.00 83.51 87.00 95.00 89.00 87.63 92.00 + ICL (5-shot) 80.30 93.56 95.00 97.50 87.67 89.67 2.04 69.00 94.00 67.00 78.72 83.50 89.90 86.00 87.88 83.00

+ ITLC (ours) 86.28 98.33 98.00 98.50 97.67 96.67 13.00 82.00 98.00 80.00 77.00 94.00 89.00 96.00 95.00 81.00 + PEFT 89.85 99.00 99.00 96.50 95.67 97.67 14.43 87.00 100.00 83.00 93.94 95.50 100.00 95.00 99.00 92.00

+ ITLC (ours) 90.51 100.00 98.00 100.00 98.67 100.00 29.00 94.00 100.00 80.00 81.00 98.00 88.00 99.00 99.00 93.00 + ITLC (ours) 82.20 100.00 99.00 100.00 98.67 98.33 7.00 74.00 100.00 80.00 72.00 95.50 39.00 95.00 82.00 92.50 Qwen2.5-7B-Instruct 78.89 81.03 96.00 95.49 87.17 87.97 31.58 72.00 91.00 55.00 61.54 84.50 81.32 88.89 87.88 82.00 + ICL (5-shot) 74.13 70.08 91.92 90.91 77.12 83.72 38.46 64.65 84.85 50.00 68.66 79.72 71.28 77.66 86.87 76.00

+ ITLC (ours) 81.01 80.85 92.00 92.88 86.68 86.45 51.61 68.00 87.88 75.00 81.32 83.27 71.58 90.43 84.21 83.00 + PEFT 88.28 97.66 92.00 99.00 93.30 94.56 13.40 88.00 97.00 84.00 84.38 95.00 94.95 99.00 99.00 93.00

+ ITLC (ours) 90.12 99.33 98.00 98.49 96.99 96.00 20.20 89.00 96.00 80.00 91.75 96.50 97.00 97.00 99.00 96.50 + ITLC (ours) 84.89 89.29 96.00 95.50 91.91 94.28 42.11 76.77 92.00 72.00 81.32 87.00 82.47 92.78 89.90 90.00 Llama-3.1-8B-Instruct 94.63 97.00 99.00 98.00 95.67 95.33 90.00 82.00 97.00 95.00 89.00 91.00 98.00 100.00 100.00 92.50 + ICL (5-shot) 88.57 93.33 99.00 16.50 95.67 96.33 92.00 89.00 97.00 86.00 96.00 89.50 94.00 94.79 100.00 89.50

+ ITLC (ours) 93.21 97.00 98.00 46.74 96.00 98.00 99.00 90.00 99.00 95.00 98.00 95.00 99.00 92.86 100.00 94.50 + PEFT 96.66 98.67 97.00 97.50 95.33 98.00 96.00 95.00 99.00 91.00 91.00 97.00 95.96 100.00 100.00 98.50 + ITLC (ours) 97.19 100.00 100.00 98.99 97.67 97.67 95.00 89.00 100.00 93.00 94.00 96.50 98.00 100.00 100.00 98.00 + ITLC (ours) 96.41 99.33 99.00 99.00 96.33 98.00 94.00 88.00 99.00 92.00 97.00 94.50 100.00 98.00 100.00 92.00

Cross-lingual Model AVG AR DE EN ES FR HI ID IT JA KO PT RU TR VI ZH

Qwen2.5-0.5B-Instruct 57.69 65.41 72.12 – 72.65 71.82 3.02 54.35 63.95 45.09 39.18 68.47 77.12 62.79 70.60 41.14 + ICL (5-shot) 69.70 81.82 83.57 – 79.01 80.73 8.38 67.25 80.70 61.51 63.66 73.39 83.97 79.14 75.93 56.71

- + ITLC (ours) 88.07 100.00 97.98 – 95.93 93.27 64.93 65.85 95.29 79.26 87.21 87.81 99.00 96.63 97.99 71.78

+ PEFT 84.34 91.72 92.75 – 93.50 93.16 14.45 85.75 94.12 85.07 77.16 90.56 95.55 90.59 96.50 79.85

- + ITLC (ours) 89.85 100.00 98.65 – 95.95 93.95 61.88 76.10 96.94 77.24 86.95 92.65 98.98 96.29 98.30 84.00

+ ITLC (ours) 86.79 98.97 97.64 – 95.31 92.27 49.03 64.39 97.31 73.24 79.27 91.98 98.98 93.25 98.32 85.13 Qwen2.5-7B-Instruct 78.81 81.96 88.38 – 83.92 84.49 52.64 73.14 82.92 71.04 79.19 85.16 80.32 88.54 77.73 73.85 + ICL (5-shot) 78.51 79.33 92.24 – 84.59 86.68 58.27 66.12 87.17 67.86 78.04 80.94 85.12 84.47 73.78 74.52

+ ITLC (ours) 84.04 86.46 96.95 – 87.60 91.83 62.21 70.94 91.90 69.84 87.71 87.16 90.19 92.39 85.86 75.53 + PEFT 83.56 94.54 91.04 – 91.73 91.22 27.18 82.67 87.85 79.03 82.33 87.08 95.29 88.69 91.48 79.72

+ ITLC (ours) 84.10 98.65 98.26 – 94.55 96.23 26.55 84.20 96.29 37.19 80.42 92.61 96.64 95.22 96.31 84.23 + ITLC (ours) 84.73 86.71 95.21 – 90.56 91.19 57.03 75.07 94.23 63.87 88.74 87.56 92.88 92.92 90.95 79.27 Llama-3.1-8B-Instruct 83.25 87.12 89.92 – 89.27 85.58 82.76 73.89 89.25 71.51 80.10 82.57 87.93 90.52 91.94 63.14 + ICL (5-shot) 86.68 86.24 91.60 – 89.60 91.94 86.17 74.53 90.26 81.89 90.80 80.90 92.18 90.26 94.29 72.83

+ ITLC (ours) 90.34 92.62 97.32 – 93.63 90.88 95.30 71.87 94.27 89.95 95.96 85.53 96.31 94.63 93.63 72.86 + PEFT 91.13 95.22 94.18 – 95.30 94.96 92.22 79.09 94.20 87.18 89.06 86.86 93.82 91.28 93.57 88.84 + ITLC (ours) 93.60 97.54 96.96 – 94.64 94.59 96.93 80.14 93.91 93.96 94.54 91.29 96.94 96.26 96.27 86.46 + ITLC (ours) 89.06 95.60 97.99 – 92.96 93.64 93.97 72.55 92.62 83.60 91.98 83.94 95.98 93.95 95.30 62.84

- Table 27: LPR metrics for the instruct model on LCB, with a detailed language-wise breakdown for both monolingual and cross-lingual settings. All results have been applied with the QA/Chat template during inference.

