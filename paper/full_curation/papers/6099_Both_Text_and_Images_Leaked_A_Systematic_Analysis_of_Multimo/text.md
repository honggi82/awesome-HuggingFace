# arXiv:2411.03823v3[cs.CV]20Sep2025

## Both Text and Images Leaked! A Systematic Analysis of Data Contamination in Multimodal LLM

Dingjie Song†,L, Sicheng Lai†,C, Mingxuan WangC, Shunian ChenC, Lichao SunL∗, Benyou WangC* LLehigh University CThe Chinese University of Hong Kong, Shenzhen https://github.com/MLLM-Data-Contamination/MM-Detect

[Figure 1]

### Abstract

[Figure 2]

[Figure 3]

Contamination Accumulation

[Figure 4]

[Figure 5]

The rapid advancement of multimodal large language models (MLLMs) has significantly enhanced performance across benchmarks. However, data contamination—unintentional memorization of benchmark data during model training—poses critical challenges for fair evaluation. Existing detection methods for unimodal large language models (LLMs) are inadequate for MLLMs due to multimodal data complexity and multi-phase training. We systematically analyze multimodal data contamination using our analytical framework, MM-DETECT, which defines two contamination categories—unimodal and cross-modal—and effectively quantifies contamination severity across multiple-choice and caption-based Visual Question Answering tasks. Evaluations on twelve MLLMs and five benchmarks reveal significant contamination, particularly in proprietary models and older benchmarks. Crucially, contamination sometimes originates during unimodal pre-training rather than solely from multimodal fine-tuning. Our insights refine contamination understanding, guiding evaluation practices and improving multimodal model reliability.

Unimodal Contamination

Cross-modal Contamination

Pure-text Pre-train Data Multimodal Post-train Data

[Figure 6]

[Figure 7]

Question Answer

Question Answer

Contains OR Contains

Image

Question

Image Question Answer

A Sample in Multimodal Benchmark

[Figure 8]

[Figure 9]

[Figure 10]

LLM MLLM Contaminated

[Figure 11]

Figure 1: An analytical breakdown illustrating different forms and origins of multimodal data contamination across distinct training stages of MLLMs.

- 2023; Bai et al., 2023b). The issue of data contamination, occurring when training or test data of benchmarks is exposed during the model training phase (Xu et al., 2024), could potentially instigate inequitable performance comparisons among models. This not only creates a dilemma for users in model selection but also poses a significant hurdle to further advancements in this domain.

Existing contamination detection methods only focus on LLMs (Yeom et al., 2018; Deng et al.,

- 2024; Dong et al., 2024), showing limitations when applied to MLLMs, due to their multimodal data complexity and multi-stage training processes (Liu et al., 2023a; Li et al., 2023; Tie et al., 2025). Thus, systematic analytical frameworks tailored explicitly for multimodal contamination are urgently needed.

### 1 Introduction

The development of MLLMs has exceeded expectations (Liu et al., 2023a; Lin et al., 2023), showcasing extraordinary performance on various multimodal benchmarks (Lu et al., 2022; Liu et al., 2023b; Song et al., 2024), even surpassing human performance. However, due to the partial obscurity associated with MLLMs training (OpenAI, 2023; Reid et al., 2024; Huang et al., 2024), it remains challenging to definitively ascertain the impact of training data on model performance, despite some works showing the employment of the training set of certain datasets (Liu et al., 2023a; Chen et al.,

In this study, we address three key questions:

- • How can we effectively quantify and detect multimodal data contamination?
- • What is the degree of contamination across different MLLMs and benchmark datasets?
- • When is contamination predominantly introduced—during unimodal pre-training or multimodal fine-tuning?

*Lichao and Benyou are the corresponding authors (lis221@lehigh.edu,wangbenyou@cuhk.edu.cn); † means equal contribution.

To comprehensively answer these questions, we first define Multimodal Data Contamination, as it pertains to the modality of data sources exposed to the MLLMs, into two categories: Unimodal Contamination and Cross-modal Contamination, as illustrated in Figure 1. Subsequently, we unveil a detection framework designed explicitly as an analytical tool, MM-DETECT, which incorporates two methods, Option Order Sensitivity Test and Slot Guessing for Perturbed Caption, designed to handle two common types of Visual Question Answering (VQA) tasks: multiple-choice and captionbased questions, respectively.

To corroborate the validity and sensitivity of our approach, we deliberately induce contamination in MLLMs, simulating realistic contamination scenarios. Experimental results demonstrate the effectiveness of MM-DETECT in identifying varying contamination degrees. Our evaluations on twelve widely-used MLLMs across five prevalent VQA datasets reveal significant contamination among both proprietary and open-source models. Critically, contamination is not only prevalent in multimodal training data but also can originates from unimodal pre-training phases, impacting older benchmarks disproportionately.

In summary, this work provides the first systematic analytical examination of multimodal data contamination, making the following explicit analytical contributions:

- • We analytically characterize multimodal contamination into clearly defined unimodal and cross-modal categories, introducing MMDETECT as an essential analytical tool.
- • We systematically quantify how benchmark leakage inflates performance metrics, providing clear insights into dataset and model susceptibility to contamination.
- • We present novel analytical insights indicating that contamination not solely emerges during the multimodal training stage but could also from unimodal pre-training stage, critically refining current understandings of contamination dynamics.

### 2 Preliminaries

We formally define the multimodal data contamination and outline the unique challenges associated with its detection.

#### 2.1 Definition of Multimodal Data Contamination

In contrast to single-modal contamination, multimodal contamination may arise from both unimodal and multimodal data sources, as depicted in Figure 1. The training data for MLLMs gener-

ally consists of pure text pre-training data Dpretrain and multimodal alignment or instruction-following data Dvision. Consider an instance (x,i,y) from a benchmark dataset D, where x represents the text input, i is the image input, and y is the label. Data contamination in MLLMs can be categorized into the following two cases:

- • Unimodal Contamination: The pair (x,y) or the input x appears in Dpretrain.
- • Cross-modal Contamination: The triplet (x,i,y) or the pair (x,i) appears in Dvision.

In both cases, models trained on these data may gain an unfair advantage.

#### 2.2 Challenges in Multimodal Detection

The challenges of multimodal contamination detection mainly arise from two aspects.

Challenge I: Inefficiency of Unimodal Methods. Despite the prevalence of unimodal detection methods, their application in multimodal scenarios often encounters difficulties. For example, retrievalbased methods (Brown et al., 2020; Touvron et al., 2023a) attempt to detect contamination by retrieving large-scale corpora used for model training. Yet, they struggle when retrieving multimodal information. Similarly, logits-based methods (Shi et al., 2024; Yeom et al., 2018) rely on observing the distribution of low-probability tokens in model outputs, but the disparity in token probability distributions is less pronounced in instructiontuned MLLMs. Masking-based methods (Deng et al., 2024), which assess training contamination by evaluating a model’s ability to predict specific missing or masked text, face challenges when images in multimodal samples provide clues, leading to overestimated contamination detection. Finally, comparison-based methods (Dong et al., 2024) that measure contamination by comparing model outputs with benchmark data prove to be ineffective for image caption tasks due to low output similarity. To validate these inefficiencies, we have conducted comprehensive experiments with compelling results, which are detailed in Appendix A.

Option Order Sensitivity Test

Proprietary MLLMs

Shuffle Options

[Figure 12]

Multichoice Dataset

Shuffled Dataset

[Figure 13]

Evaluate using Atomic Metrics

[Figure 14]

Step I. Generation

Step II. Detection

Multi-choice Prediction

CR PCR Δ Φ

Slot Guessing for Perturbed Captions

Open-source MLLMs

[Figure 15]

[Figure 16]

[Figure 17]

Back Translate

Caption Dataset

Back-translated Dataset

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Mask Word Prediction

Figure 2: The overview of proposed MM-DETECT framework.

Challenge II: Multi-stage Training in MLLMs. Another challenge in detecting contamination in MLLMs is the multi-stage nature of their training (Yin et al., 2023). Each stage may be subject to data contamination. 1) Initially, the pretraining corpus could contain the textual components of questions from benchmark samples. Moreover, in certain native multimodal model training (Reid et al., 2024), samples may be entirely exposed. 2) Subsequently, during multimodal fine-tuning, the model may utilize training samples of some benchmarks, leading to skewed performance improvements. 3) Furthermore, some models employ extensive mixed image-text data from the internet for modality alignment training (Lin et al., 2023; Bai et al., 2023b), potentially introducing additional contamination. Given the challenges, the development of an effective detection framework for multimodal contamination becomes an urgent need.

Test (§3.1) and Slot Guessing for Perturbed Captions (§3.2), tailored for multiple-choice and image captioning tasks, respectively.

• The second step involves the application of predefined metrics to detect contamination (§3.3), conducting thorough analyses at both the dataset and instance levels.

#### 3.1 Option Order Sensitivity Test

This method is based on a reasonable and intuitive premise that if the model’s performance is highly sensitive to the order of the options, as shown in Figure 3, it indicates potential contamination, leading the model to memorize a certain canonical order of the options.

[Figure 22]

Based on the discussion above, we have designed a detection method specifically tailored for multimodal contamination, with a particular focus on VQA tasks. Additionally, we have developed a heuristic method to trace the introduction of contamination across different training phases.

Figure 3: An example of Option Order Sensitivity Test applied to a contaminated model.

### 3 Detection Framework: MM-DETECT

Method Formulation. Let D be a dataset consisting of n datapoints. Each datapoint di (i ∈ {1,...,n}) comprises a question Qi, an associated image Ii, and a set of answer choices Ai = {a1i,a2i,...,ami }, where m is the number of choices and the correct answer is denoted by aci.

We introduce the multimodal contamination detection framework, MM-DETECT, designed explicitly to support our systematic analysis of contamination phenomena. The core philosophy of MMDETECT is to detect the unusual discrepancies in model performance before and after semanticirrelevant perturbations. As depicted in Figure 2, this framework operates in two primary steps:

To introduce positional variation, the set Ai is randomly shuffled to obtain a new set A′i, ensur-

ing that the index of the correct answer aci in A′i differs from its original position in Ai. The final prompts, before and after shuffling, are constructed

• The first step is to generate perturbed datasets using two methods: Option Order Sensitivity

by concatenating the image, question and choices:

P = Concat(Ii,Qi,Ai),

P′ = Concat(Ii,Qi,A′i),

where P and P′ are the inputs to the model, and Qi and Ii remain unchanged throughout this process.

#### 3.2 Slot Guessing for Perturbed Caption

This method is based on the intuition that if a model can predict a missing and important part of a sentence but fails with the back-translated version (from English to Chinese, then back to English), it likely indicates that the model has encountered the original sentence during training.

[Figure 23]

Figure 4: A simple example shows the procedure.

As shown in Figure 4, the keywords identified are “woods” and “bike”. Since the image contains “woods”, a correct guess by the model may stem from its multimodal capabilities rather than data contamination. However, if the model fails to predict “bike”, which is also present in the image, this may indicate potential leakage of this instance.

Method Formulation. Let D be a dataset containing n datapoints. Each datapoint di (i ∈ {1,...,n}) consists of an image-caption pair, where the caption Si describes the visual features of the corresponding image Ii. We first apply a back-translation function, where we use the Google Translate API for Python to implement back-translation, to Si:1

##### Si′ = fback-translate(Si).

1A quantitative analysis of the semantic and lexical similarity between the original and back-translated captions is provided in Appendix D.1.

resulting in a paraphrased version Si′. Next, we perform keyword extraction2 on both Si and Si′:

##### Ki = fkeyword(Si), Ki′ = fkeyword(Si′),

where Ki and Ki′ denote the extracted keywords from Si and Si′, respectively. We then apply a masking function fmask to replace the extracted keywords with a placeholder token [MASK]:

Si,mask = fmask(Si,Ki), Si,′ mask = fmask(Si′,Ki′). The final prompt guiding the model to complete the masked-word prediction can be represented as:

Pi = Concat(Ii,Qi,Si,mask), Pi′ = Concat(Ii,Qi,Si,′ mask).

#### 3.3 Detection Metrics

Detection Metrics serve as the core analytical instruments within MM-DETECT. Having introduced two detection methods, we now delineate the atomic metrics for the detection pipeline, which consists of two primary steps.

- Step 1: Correct Rate Calculation. This step assesses the model’s performance on benchmark D before and after perturbation. We denote the correct rate (CR) and perturbed correct rate (PCR) uniformly for both Option Order Sensitivity Test (using Accuracy) and Slot Guessing (using Exact Match). Here, N and N′ are the counts of correct answers before and after perturbation, respectively. They are calculated as:

CR =

N |D|

, PCR =

N′ |D|

.

- Step 2: Contamination Degree Analysis. This step quantifies the model’s contamination degree based on the performance variation pre- and postperturbation. Specifically, we introduce two metrics to evaluate contamination at both dataset and instance levels.

Dataset Level Metric. We evaluate the reduction in atomic metrics, denoted as ∆:

##### ∆ = PCR − CR

This reduction indicates the model’s familiarity or memory of the original benchmark relative to the

2We employ the Stanford POS Tagger (Toutanvoa and Manning, 2000), targeting nouns, adjectives, and verbs, as they encapsulate the core meaning of the sentences.

perturbed set, thereby offering insights into potential contamination at the dataset level. A significant negative ∆ suggests potential extensive leakage in the benchmark dataset, leading to highly perturbation-sensitive model performance.

Recognizing that contamination can occur anywhere in the training data, we inserted contaminated samples into the visual instruction tuning dataset (Dtuning) at three positions, early, mid, and late, creating two groups of contaminated training sets using 1340 ScienceQA test samples or 1000 NoCaps validation samples. Corresponding models, termed Early Cont., Mid Cont., and Late Cont., were then trained for comparison with the baseline.

Instance Level Metric. Despite a nonsignificant or positive ∆, contamination may still occur at the instance level, as some instances may still have been unintentionally included during training. To identify such instances, we compute X, the count of cases where the model provided correct answers before perturbation but incorrect answers after. The instance leakage metric Φ is then obtained by dividing X by the dataset size:

|ScienceQA Test Set<br><br>|NoCaps Val. Set|
|---|---|
|CR PCR ∆<br><br>|CR PCR ∆|

Models

Baseline 61.4 61.5 0.01 33.0 32.1 -0.9 Early Cont. 71.5 68.1 -3.4 37.5 32.0 -5.5 Mid Cont. 69.4 67.3 -2.1 38.5 35.1 -3.4 Late Cont. 70.2 66.9 -3.3 38.7 32.6 -6.1

X |D|

Φ =

,

Table 1: Detection results on contamination using the ScienceQA test set and NoCaps validation set.

where a larger Φ indicates a higher likelihood of instance leakage.

Table 1 shows that incorporating contaminated data during training increases both the model’s performance and its sensitivity to perturbations. Compared with the baseline, ScienceQA-contaminated models exhibit average increases in CR and PCR of 9.0% and 5.9%, while NoCaps-contaminated models show increases of 5.2% and 1.1%. Moreover, all contaminated models demonstrate a marked decrease in ∆, confirming that MM-DETECT effectively identifies data contamination.

Compared to methods relying solely on accuracy or perplexity, MM-DETECT explicitly highlights performance drop after perturbations, preventing exaggeration or underestimation of contamination. Moreover, it offers advantages of lower computational overhead, higher sensitivity, and effective black-box applicability, thus serving as an essential analytical toolkit in our study.

### 4 Evaluating MM-DETECT with Intentional Contamination

4.2 MM-DETECT is Sensitive and Fine-grained

This section tackles our first overarching research question — How can we effectively quantify and detect multimodal data contamination? To operationalise this goal, we break RQ1 into three subquestions:

We evaluated MM-DETECT’s sensitivity by varying leakage levels in the training set. Using the fully contaminated model as our baseline, we trained additional models with moderate and minimal contamination, by inserting reduced amounts (10% and 50%) of contaminated data at the late position of the training set, to assess leakage impact.

- SQ1 (Effectiveness) Is MM-DETECT able to detect contamination regardless of where it is injected?
- SQ2 (Sensitivity) How finely can MM-DETECT measure different leakage levels?
- SQ3 (Bias Diagnostic) When training-set data leak, can MM-DETECT reveal the evaluation bias?

|[Figure 24]<br><br>56.2<br><br>64.9<br><br>70.2<br><br>56.0<br><br>63.3<br><br>66.9<br><br>28.5<br><br>34.8<br><br>38.7<br><br>27.4<br><br>32.4 32.6<br><br>25.0<br><br>30.0<br><br>35.0<br><br>40.0<br><br>45.0<br><br>50.0<br><br>55.0<br><br>60.0<br><br>65.0<br><br>70.0<br><br>75.0<br><br>10% 50% 100%<br><br>Performance(%)<br><br>Leakage Levels<br><br>[Figure 25]<br><br>ScienceQA-CR ScienceQA-PCR NoCaps-CR NoCaps-PCR<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>Figure 5: Performance and atomic metrics evaluated|
|---|

Performance(%)

We answer these sub-questions by adopting the LLaVA framework and training a suite of 7Bparameter models with intentionally contaminated data during the visual-instruction tuning phase. The contamination protocol and data split follow §5.1.

#### 4.1 MM-DETECT is An Effective Detector

under varying leakage levels on the ScienceQA test set and NoCaps validation set.

We reproduced the LLaVA-1.5-7B experiment to obtain a baseline model without contamination.

As illustrated in Figure 5, increasing contamination from 10% to 50% to 100% results in corresponding increases in CR and PCR, alongside progressively larger ∆ values. The findings confirm that our framework can accurately differentiate between varying leakage levels in datasets.

- 4.3 MM-DETECT Diagnoses Evaluation Bias from Training-set Leakage

We investigated whether MM-DETECT can detect training set leakage by comparing models trained with and without benchmark data contamination. For the ScienceQA experiment, we appended 2000 ScienceQA training samples to the training dataset, creating a contaminated model. For the COCO experiment, we removed the COCO-Caption2017 training data from the original training dataset, resulting in a model without leakage.

Model Dataset CR PCR ∆

Clean ScienceQA 61.4 61.5 0.01 Leaked ScienceQA 64.3 63.8 -0.5

Clean COCO-Caption2017 32.5 31.9 -0.6 Leaked COCO-Caption2017 38.1 34.9 -3.2

Table 2: Performance of models trained without (Clean) and with (Leaked) training set contamination.

- Table 2 compares the models’ performance. On

the ScienceQA test set, the contaminated model outperforms the clean model by 2.9% in CR and 2.3% in PCR, with a ∆ of -0.5. On the COCOCaption2017 validation set, the model trained with COCO data shows a ∆ of -3.2. The results indicate that training set leakage inflates performance and that MM-DETECT effectively detects it.

MM-DETECT also shows strong robustness: ∆ is unaffected by prompt changes (Appendix D.3).

Takeaways

Both training and test set leakage can result in unfairness, and the degree of contamination can be detected through MM-DETECT effectively.

5 Assessing the Extent of Contamination in MLLMs

In this section, we systematically quantify the extent of contamination across various MLLMs and benchmarks, addressing our second research question — What is the degree of contamination?

- 5.1 Setup

Models. We conducted evaluations on nine opensource MLLMs, including LLaVA-1.5-7B (Liu

- et al., 2023a), VILA1.5-3B (Lin et al., 2023), Qwen-VL-Chat (Bai et al., 2023b), fuyu-8b3, idefics2-8b (Laurençon et al., 2024), Phi-3-vision128k-instruct (Abdin et al., 2024), Yi-VL-6B (AI
- et al., 2024), InternVL2-8B (Chen et al., 2023, 2024b), DeepSeek-VL2-Tiny (Wu et al., 2024), as well as three proprietary MLLMs: GPT-4o-202408-06 (OpenAI, 2023), Gemini-1.5-Pro-002 (Reid et al., 2024), and Claude-3.5-Sonnet-2024-06-204.

Benchmark Datasets. Our analysis leverages two multi-choice datasets: ScienceQA (Lu et al., 2022) and MMStar (Chen et al., 2024a), along with three caption datasets: COCO-Caption2017 (Lin et al., 2015), NoCaps (Agrawal et al., 2019), and Vintage5. MMStar and Vintage, owing to their recent inception, serve to contrast leakage levels with other datasets. We randomly selected 2000 and 1340 samples from ScienceQA’s training and test sets, respectively, with 1000 samples from the other datasets. Given the unavailability of public test labels for COCO-Caption2017 and NoCaps, we used their validation sets.

#### 5.2 Main Results

Multi-choice Datasets. Table 3 yields several conclusions: (1) Both open-source and proprietary models exhibit contamination. For example, on the ScienceQA training set, both opensource models like LLaVA-1.5-7B and idefics2-8b and proprietary Gemini-1.5-Pro show minor contamination degree. (2) Proprietary models are more contaminated. Claude-3.5-Sonnet, for instance, registers a severe ∆ with higher Φ values on ScienceQA datasets, indicating extensive leakage. (3) Training set leakage is more pronounced than test set leakage. On the ScienceQA dataset, models generally exhibit larger ∆ values in the training set, for instance, Claude-3.5-Sonnet shows ∆ = −5.3 on training versus −2.4 on the test set, while most models have near-zero ∆ on the test set. (4) Older benchmarks are more prone to leak. The older ScienceQA test set shows more leakage compared to the newer MMStar validation set.

- 3https://www.adept.ai/blog/fuyu-8b
- 4https://www.anthropic.com/news/

claude-3-5-sonnet

- 5https://huggingface.co/datasets/

SilentAntagonist/vintage-artworks-60k-captioned 6Following §4.1: multi-choice leakage levels by

- ∆—minor (−1.6 < ∆ ≤ −0.2), partial (−2.9 < ∆ ≤

- −1.6), severe (∆ ≤ −2.9); caption leakage levels by

∆—minor (−2.4 < ∆ ≤ −1.1), partial (−5.0 < ∆ ≤

- −2.4), severe (∆ ≤ −5.0). See Appendix C.

Model ScienceQA Training Set ScienceQA Test Set MMStar Validation Set Metric CR PCR ∆ Φ CR PCR ∆ Φ CR PCR ∆ Φ Open-source MLLMs LLaVA-1.5-7B 59.7 58.6 -1.1 12.7 60.3 61.6 1.3 10.5 38.9 41.7 2.8 11.0 VILA1.5-3B 57.7 58.3 0.6 14.5 60.3 59.8 -0.5 14.8 38.6 37.6 -1.0 13.9 Qwen-VL-Chat 58.4 60.8 2.5 13.3 60.3 60.4 0.1 13.7 40.9 44.2 3.3 13.2 fuyu-8b 36.5 37.5 1.0 13.4 37.4 36.9 -0.5 14.9 28.2 27.0 -1.2 17.7 idefics2-8b 85.1 84.0 -1.2 3.7 84.0 84.3 0.3 2.8 48.2 49.3 1.1 7.9 Phi-3-vision-128k-instruct 90.5 90.4 -0.1 4.6 88.4 89.1 0.7 3.9 48.7 51.9 3.2 7.2 Yi-VL-6B 60.5 61.8 1.3 10.0 59.5 61.3 1.8 9.6 38.8 44.0 5.2 9.3 InternVL2-8B 94.1 93.9 -0.3 2.0 92.3 93.1 0.8 1.7 56.9 60.1 3.2 5.1 DeepSeek-VL2-Tiny 86.4 86.5 0.1 5.3 87.1 86.9 -0.2 5.3 51.1 52.1 1.0 10.7 Proprietary MLLMs GPT-4o 69.9 70.0 0.1 2.7 69.1 69.7 0.6 2.8 48.6 50.5 1.9 9.4 Gemini-1.5-Pro 68.5 67.9 -0.6 6.6 66.5 66.2 -0.3 7.1 45.7 45.5 -0.2 9.9 Claude-3.5-Sonnet 70.3 65.0 -5.3 15.3 67.3 64.9 -2.4 12.4 36.3 36.4 0.1 15.9

- Table 3: Comparison of MLLMs on multi-choice datasets. Bold values represent the most significant ∆ or Φ; color codes denote contamination degree: green for minor leakage, yellow for partial leakage, and red for severe leakage.6

Model COCO Validation Set NoCaps Validation Set Vintage Training Set Metric CR PCR ∆ Φ CR PCR ∆ Φ CR PCR ∆ Φ Open-source MLLMs LLaVA-1.5-7B 34.6 34.0 -0.6 19.0 30.9 28.5 -2.4 17.9 10.8 10.1 -0.7 9.0 VILA1.5-3B 19.1 20.5 1.4 13.0 19.1 20.5 1.4 13.0 1.5 2.2 0.7 1.5 Qwen-VL-Chat 32.2 30.3 -1.9 19.2 28.7 27.3 -1.4 16.7 15.1 15.4 0.3 12.4 fuyu-8b 9.6 10.6 1.0 7.8 10.0 9.8 -0.2 8.3 2.4 3.3 0.9 2.3 idefics2-8b 43.5 42.3 -1.2 21.2 42.6 37.5 -5.1 23.3 18.5 17.0 -1.5 14.5 Phi-3-vision-128k-instruct 38.8 39.3 0.5 19.4 36.9 33.3 -3.6 19.7 17.4 11.7 -5.7 14.3 Yi-VL-6B 43.9 43.3 -0.6 19.4 37.2 36.1 -1.1 17.5 3.3 4.2 0.9 2.8 InternVL2-8B 53.3 51.9 -1.4 20.4 48.0 46.2 -1.8 20.9 28.0 28.7 0.7 18.8 DeepSeek-VL2-Tiny 23.8 21.4 -2.4 13.5 19.3 18.1 -1.2 12.2 7.5 6.9 -0.6 6.3 Proprietary MLLMs GPT-4o 58.1 54.4 -3.7 23.1 54.2 55.1 0.9 19.4 36.3 38.4 2.1 20.1 Gemini-1.5-Pro 57.5 55.3 -2.2 21.6 51.2 52.0 0.8 18.7 46.3 41.0 -5.3 28.3 Claude-3.5-Sonnet 53.7 51.0 -2.7 21.8 50.8 51.5 0.7 20.0 35.2 33.0 -2.2 21.3

- Table 4: Comparison of MLLMs on caption datasets. Bold values represent the most significant ∆ or Φ; color codes denote contamination degree: green for minor leakage, yellow for partial leakage, and red for severe leakage.

Caption Datasets. Table 4 yields several conclusions: (1) Both open-source and proprietary models exhibit contamination on caption datasets. For example, in the COCO Validation Set, open-source models such as DeepSeek-VL2Tiny and proprietary models like GPT-4o record a significant contamination degree. (2) Leakage levels vary significantly by benchmark. For example, on the NoCaps Validation Set, open-source models exhibit more pronounced contamination degree than proprietary models, whereas the trend reverses on the COCO Validation Set. These findings confirm that caption datasets are vulnerable to leakage, with proprietary models generally exhibiting more pronounced contamination effects.

Takeaways

Multimodal data contamination, at both dataset and instance levels, is prevalent in open-source and proprietary MLLMs across multi-choice and image caption datasets.

### 6 Identifying the Origin of Contamination in MLLMs

In this section, we address our third research question — When is contamination predominantly introduced? Although the training data for some MLLMs is openly documented, an important question remains: if contamination does not arise during the multimodal training phase, could it stem from the unimodal (pre-training) phase, as defined in §2.1? To address this possibility, we examined

the underlying LLMs of the evaluated MLLMs and conducted a series of experiments (§6.1). We also explored the origins of cross-modal contamination arising during visual instruction tuning (§6.2).

- 6.1 Heuristic Detection of Unimodal Pre-training Contamination

We employed a heuristic approach based on the intuition that if an LLM can correctly answer an image-required question without the image when random guessing is effectively inhibited, it may indicate the leakage of that instance.

Experiment Setup. We used MMStar as the benchmark, where every question relies on visual input for correct answers. The tested models include LLaMA2-7B (Touvron et al., 2023b) (used by LLaVA-1.5 and VILA), Qwen-7B (Bai et al., 2023a) (used by Qwen-VL), Mistral-7B-v0.1 (Jiang et al., 2023) (used by idefics2), Phi-3-small128k-instruct (Abdin et al., 2024) (used by Phi-3vision), Yi-6B (AI et al., 2024) (used by Yi-VL), and Internlm2-7B (Cai et al., 2024) (used by InternVL2). To inhibit random guessing, we appended the prompt “If you do not know the answer, output I don’t know” to the instructions. A sanity check in Appendix D.2 confirms that this uncertainty clause effectively suppresses lucky guesses, validating its inclusion in our main protocol. Accuracy — the frequency with which models correctly answer questions without image input — is reported as the primary metric. Note that we did not evaluate Fuyu-8B and proprietary models since their unimodal LLM components and training data remain undisclosed.

Model Accuracy ΦM LLaMA2-7b (LLaVA-1.5 & VILA) 25.6 11.0 Qwen-7B (Qwen-VL) 13.2 13.2 Internlm2-7B (InternVL2) 11.0 5.1 Mistral-7B-v0.1 (idefics2) 10.7 7.9 Phi-3-small-128k-instruct (Phi-3-vision) 6.1 7.2 Yi-6B (Yi-VL) 3.4 9.3

- Table 5: Contamination rates of LLMs used by MLLMs. ΦM denotes the Φ of the respective MLLMs.

Main Results. Table 5 yields several conclusions: (1) Contamination occurs in LLM. All models exhibit varied contamination rates, indicating that their pre-training data likely included text from multimodal benchmarks. (2) Elevated LLM contamination correlates with increased MLLM

leakage. For instance, VILA1.5-3B and Qwen-VLChat exhibit significant Φ values that mirror their underlying LLM contamination levels. These findings suggest that contamination in these MLLMs may originate partly from the LLMs’ pre-training phase, rather than solely from multimodal training.

6.2 Analyzing Cross-modal Contamination in Multimodal Fine-tuning

To investigate the origins of cross-modal contamination, we scrutinize the visual instruction tuning data of MLLMs. We delve into the construction process of three benchmark datasets: ScienceQA, COCO Caption, and Nocaps, comparing them with the training data and its sources of various opensource MLLMs to analyze the degree of overlap.

Model ScienceQA COCO Caption Nocaps Phi-3-Vision 0.7 0.5 -3.6

VILA -0.5 1.4 1.4

Idefics2 0.3 -1.2 -5.1 LLaVA-1.5 1.3 -0.6 -2.4

Yi-VL 1.8 -0.6 -1.1 DeepSeek-VL2 -0.2 -2.4 -1.2 Qwen-VL-Chat 0.1 -1.9 -1.4

InternVL2 0.8 -1.4 -1.8

Table 6: Depiction of the overlap between the training data of MLLMs and the benchmarks, as well as the contamination degree ∆ of MLLMs on benchmarks. Green signifies no overlap, yellow suggests potential overlap, and Red indicates partial or entire overlap.

As Table 6 illustrates, MLLMs marked in red and yellow typically exhibit a significant contamination degree. Yet, even MLLMs labeled in green aren’t exempt from the risk of cross-modal contamination. This is because some models have been trained on large-scale interleaved image-text datasets (e.g., OBELICS (Laurenon et al., 2023)), datasets derived from online sources (e.g., Conceptual Caption (Sharma et al., 2018)), or in-house data. Furthermore, some models haven’t fully disclosed their training data, which may lead to overlooked potential leaks in benchmark datasets. A detailed overlap analysis is in Appendix E.

Takeaways

The contamination in MLLMs may not only stem from cross-modal contamination but also from unimodal contamination, both of which can significantly impact the overall performance.

### 7 Conclusion and Future Work

In this study, we systematically analyzed multimodal data contamination in MLLMs through our proposed detection framework, MM-DETECT. We demonstrated that MM-DETECT effectively quantifies and detects varying contamination degrees, revealing significant performance biases induced by benchmark leakage. Importantly, we identified that contamination originates notably from both unimodal pre-training and multimodal fine-tuning phases, impacting the reliability and fairness of multimodal evaluations.

Future work will focus on two key areas:

- • Firstly, standardizing the use of multimodal datasets and reporting potential contamination impacts to minimize contamination, thereby enhancing data consistency and quality.
- • Secondly, creating a continuously updated benchmarking system for the ongoing evaluation of multimodal model performance.

This will support advancements and broader applications in this field.

### Limitations

We acknowledge several limitations in our work. First, this work is limited to discussions around visual modalities, and does not yet cover other modalities such as audio or video. Second, we only selected widely used and representative multimodal datasets for detection, including multiple-choice datasets and caption datasets, without testing additional datasets, such as open-ended generation and cloze questions. However, we speculate that the method Slot Guessing for Perturbed Caption may also apply to other types of image-featureanalyzing benchmarks. Third, the effectiveness of Option Order Sensitivity Test can be undermined by option shuffling, which, while potentially improving model performance, is computationally expensive and may increase the training cost. Fourth, as a perturbation-based black-box detector, MMDETECT might underestimate contamination if a model generalizes sufficiently to answer perturbed questions correctly. Although dataset-level evaluations reduce this risk, completely eliminating such false-negative cases remains an open challenge.

### Acknowledgments

This work was supported by the Shenzhen Science and Technology Program

(JCYJ20220818103001002), Shenzhen Doctoral Startup Funding (RCBS20221008093330065), Tianyuan Fund for Mathematics of National Natural Science Foundation of China (NSFC) (12326608), Shenzhen Science and Technology Program (Shenzhen Key Laboratory Grant No. ZDSYS20230626091302006), and Shenzhen Stability Science Program 2023, Shenzhen Key Lab of Multi-Modal Cognitive Computing.

### References

Marah I Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat S. Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck, Sébastien Bubeck, Martin Cai, Caio César Teodoro Mendes, Weizhu Chen, Vishrav Chaudhary, Parul Chopra, Allie Del Giorno, Gustavo de Rosa, Matthew Dixon, Ronen Eldan, Dan Iter, Amit Garg, Abhishek Goswami, Suriya Gunasekar, Emman Haider, Junheng Hao, Russell J. Hewett, Jamie Huynh, Mojan Javaheripi, Xin Jin, Piero Kauffmann, Nikos Karampatziakis, Dongwoo Kim, Mahoud Khademi, Lev Kurilenko, James R. Lee, Yin Tat Lee, Yuanzhi Li, Chen Liang, Weishung Liu, Eric Lin, Zeqi Lin, Piyush Madan, Arindam Mitra, Hardik Modi, Anh Nguyen, Brandon Norick, Barun Patra, Daniel Perez-Becker, Thomas Portet, Reid Pryzant, Heyang Qin, Marko Radmilac, Corby Rosset, Sambudha Roy, Olatunji Ruwase, Olli Saarikivi, Amin Saied, Adil Salim, Michael Santacroce, Shital Shah, Ning Shang, Hiteshi Sharma, Xia Song, Masahiro Tanaka, Xin Wang, Rachel Ward, Guanhua Wang, Philipp Witte, Michael Wyatt, Can Xu, Jiahang Xu, Sonali Yadav, Fan Yang, Ziyi Yang, Donghan Yu, Chengruidong Zhang, Cyril Zhang, Jianwen Zhang, Li Lyna Zhang, Yi Zhang, Yue Zhang, Yunan Zhang, and Xiren Zhou. 2024. Phi-3 technical report: A highly capable language model locally on your phone. CoRR, abs/2404.14219.

Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. 2019. nocaps: novel object captioning at scale. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE.

01. AI, :, Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, Kaidong Yu, Peng Liu, Qiang Liu, Shawn Yue, Senbin Yang, Shiming Yang, Tao Yu, Wen Xie, Wenhao Huang, Xiaohui Hu, Xiaoyi Ren, Xinyao Niu, Pengcheng Nie, Yuchi Xu, Yudong Liu, Yue Wang, Yuxuan Cai, Zhenyu Gu, Zhiyuan Liu, and Zonghong Dai. 2024. Yi: Open foundation models by 01.ai. Preprint, arXiv:2403.04652.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei

Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. 2023a. Qwen technical report. CoRR, abs/2309.16609.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023b. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. Preprint, arXiv:2308.12966.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. CoRR, abs/2005.14165.

Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, Xiaoyi Dong, Haodong Duan, Qi Fan, Zhaoye Fei, Yang Gao, Jiaye Ge, Chenya Gu, Yuzhe Gu, Tao Gui, Aijia Guo, Qipeng Guo, Conghui He, Yingfan Hu, Ting Huang, Tao Jiang, Penglong Jiao, Zhenjiang Jin, Zhikai Lei, Jiaxing Li, Jingwen Li, Linyang Li, Shuaibin Li, Wei Li, Yining Li, Hongwei Liu, Jiangning Liu, Jiawei Hong, Kaiwen Liu, Kuikun Liu, Xiaoran Liu, Chengqi Lv, Haijun Lv, Kai Lv, Li Ma, Runyuan Ma, Zerun Ma, Wenchang Ning, Linke Ouyang, Jiantao Qiu, Yuan Qu, Fukai Shang, Yunfan Shao, Demin Song, Zifan Song, Zhihao Sui, Peng Sun, Yu Sun, Huanze Tang, Bin Wang, Guoteng Wang, Jiaqi Wang, Jiayu Wang, Rui Wang, Yudong Wang, Ziyi Wang, Xingjian Wei, Qizhen Weng, Fan Wu, Yingtong Xiong, and et al. 2024. Internlm2 technical report. CoRR, abs/2403.17297.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. 2024a. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330.

Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. 2024b. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong

Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. 2023. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238.

Chunyuan Deng, Yilun Zhao, Xiangru Tang, Mark Gerstein, and Arman Cohan. 2024. Investigating data contamination in modern benchmarks for large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), NAACL 2024, Mexico City, Mexico, June 16-21, 2024, pages 8706–8719. Association for Computational Linguistics.

Yihong Dong, Xue Jiang, Huanyu Liu, Zhi Jin, Bin Gu, Mengfei Yang, and Ge Li. 2024. Generalization or memorization: Data contamination and trustworthy evaluation for large language models. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pages 12039–12050. Association for Computational Linguistics.

Yue Huang, Lichao Sun, Haoran Wang, Siyuan Wu, Qihui Zhang, Yuan Li, Chujie Gao, Yixin Huang, Wenhan Lyu, Yixuan Zhang, Xiner Li, Hanchi Sun, Zhengliang Liu, Yixin Liu, Yijue Wang, Zhikun Zhang, Bertie Vidgen, Bhavya Kailkhura, Caiming Xiong, Chaowei Xiao, Chunyuan Li, Eric P. Xing, Furong Huang, Hao Liu, Heng Ji, Hongyi Wang, Huan Zhang, Huaxiu Yao, Manolis Kellis, Marinka Zitnik, Meng Jiang, Mohit Bansal, James Zou, Jian Pei, Jian Liu, Jianfeng Gao, Jiawei Han, Jieyu Zhao, Jiliang Tang, Jindong Wang, Joaquin Vanschoren, John C. Mitchell, Kai Shu, Kaidi Xu, Kai-Wei Chang, Lifang He, Lifu Huang, Michael Backes, Neil Zhenqiang Gong, Philip S. Yu, Pin-Yu Chen, Quanquan Gu, Ran Xu, Rex Ying, Shuiwang Ji, Suman Jana, Tianlong Chen, Tianming Liu, Tianyi Zhou, William Wang, Xiang Li, Xiangliang Zhang, Xiao Wang, Xing Xie, Xun Chen, Xuyu Wang, Yan Liu, Yanfang Ye, Yinzhi Cao, Yong Chen, and Yue Zhao. 2024. Position: Trustllm: Trustworthiness in large language models. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. Preprint, arXiv:2310.06825.

Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. 2024. What matters when building vision-language models? CoRR, abs/2405.02246.

Hugo Laurenon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas

Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, and Victor Sanh. 2023. Obelics: An open web-scale filtered dataset of interleaved image-text documents. Preprint, arXiv:2306.16527.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR.

Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. 2023. VILA: on pre-training for visual language models. CoRR, abs/2312.07533.

Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. 2015. Microsoft coco: Common objects in context. Preprint, arXiv:1405.0312.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023a. Improved baselines with visual instruction tuning. CoRR, abs/2310.03744.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. 2023b. Mmbench: Is your multi-modal model an all-around player? CoRR, abs/2307.06281.

Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, KaiWei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS).

OpenAI. 2023. GPT-4 technical report. CoRR, abs/2303.08774.

Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318.

Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084.

Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. 2018. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of ACL.

Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, and Luke Zettlemoyer. 2024. Detecting pretraining data from large language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Dingjie Song, Shunian Chen, Guiming Hardy Chen, Fei Yu, Xiang Wan, and Benyou Wang. 2024. Milebench: Benchmarking mllms in long context. arXiv preprint arXiv:2404.18532.

Guiyao Tie, Zeli Zhao, Dingjie Song, Fuyang Wei, Rong Zhou, Yurou Dai, Wen Yin, Zhejian Yang, Jiangyue Yan, Yao Su, Zhenhan Dai, Yifeng Xie, Yihan Cao, Lichao Sun, Pan Zhou, Lifang He, Hechang Chen, Yu Zhang, Qingsong Wen, Tianming Liu, Neil Zhenqiang Gong, Jiliang Tang, Caiming Xiong, Heng Ji, Philip S. Yu, and Jianfeng Gao. 2025. A survey on post-training of large language models. CoRR, abs/2503.06072.

Kristina Toutanvoa and Christopher D. Manning. 2000. Enriching the knowledge sources used in a maximum entropy part-of-speech tagger. In Joint SIGDAT Conference on Empirical Methods in Natural Language Processing and Very Large Corpora, EMNLP 2000, Hong Kong, October 7-8, 2000, pages 63–70. Association for Computational Linguistics.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian CantonFerrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023a. Llama 2: Open foundation and finetuned chat models. CoRR, abs/2307.09288.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian CantonFerrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura,

Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288.

Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, Zhenda Xie, Yu Wu, Kai Hu, Jiawei Wang, Yaofeng Sun, Yukun Li, Yishi Piao, Kang Guan, Aixin Liu, Xin Xie, Yuxiang You, Kai Dong, Xingkai Yu, Haowei Zhang, Liang Zhao, Yisong Wang, and Chong Ruan. 2024. Deepseekvl2: Mixture-of-experts vision-language models for advanced multimodal understanding. Preprint, arXiv:2412.10302.

Ruijie Xu, Zengzhi Wang, Run-Ze Fan, and Pengfei Liu. 2024. Benchmarking benchmark leakage in large language models. arXiv preprint arXiv:2404.18824.

Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha. 2018. Privacy risk in machine learning: Analyzing the connection to overfitting. In 31st IEEE Computer Security Foundations Symposium, CSF 2018, Oxford, United Kingdom, July 9-12, 2018, pages 268–282. IEEE Computer Society.

Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. 2023. A survey on multimodal large language models. arXiv preprint arXiv:2306.13549.

### A Inefficiency of Unimodal Methods

We demonstrate the results of traditional unimodal contamination detection methods applied to MLLMs.

#### A.1 Logits-base

These methods determine contamination by observing the distribution of low-probability tokens in model outputs. However, MLLMs typically undergo instruction fine-tuning, which enhances their instruction-following capabilities, leading to less significant differences in token probability distributions. As shown in Table 7, LLaVA-1.5-13b exhibits extremely low perplexity on multimodal benchmark datasets.

Dataset Perplexity Split ScienceQA 1.4498 Training Set

MMStar 1.4359 Validation Set COCO-Caption2017 1.7530 Validation Set

NoCaps 1.8155 Validation Set

- Table 7: Perplexity of LLaVA-1.5-13b on various multimodal benchmarks (100 samples randomly selected from each dataset).

A.2 Masking-base

These methods involve masking phrases or sentences and providing data from the benchmark to guide the model in filling in the missing parts. However, multimodal datasets often contain images that include the masked portions of sentences, effectively providing answers to the model. This results in significantly higher success rates for MLLMs in predicting missing parts compared to unimodal language models, leading to exaggerated contamination detection. As shown in Table 8, LLaVA1.5-13b has a high probability of Exact Match for predicting the masked word.

Dataset Exact Match ROUGE-L F1 Split

COCO-Caption2017 0.24 0.36 Validation Set NoCaps 0.22 0.29 Validation Set

- Table 8: Contamination detection of LLaVA-1.5-13b using TS-Guessing (question-based) on various multimodal benchmarks (100 samples randomly selected from each dataset).

#### A.3 Comparison-base

These methods identify contamination by comparing the similarity between models’ outputs and

benchmark data. However, MLLMs often undergo data augmentation, causing their outputs to diverge significantly from the labels in benchmark data, making effective contamination detection challenging. From Table 9, we can see that CDD (Contamination Detection via Output Distribution) indicates a contamination metric of 0% across all multimodal benchmark datasets.

Dataset Contamination Metric Split

COCO-Caption2017 0.0000% Validation Set NoCaps 0.0000% Validation Set

- Table 9: Contamination detection of LLaVA-1.5-13b using CDD (Contamination Detection via Output Distribution) on various multimodal benchmarks (100 samples randomly selected from each dataset).

### B Detailed Slot Guessing Pipeline

#### B.1 Back-Translation

The back-translation function applies a two-step translation process to generate a paraphrased caption Si′ from the original caption Si. In this method, we use the Google Translate API to translate the caption into Chinese and then back into the original language to generate the paraphrase.

- Algorithm 1 Back-Translation

- 1: Input: Original caption Si
- 2: Translate Si to an intermediate language L
- 3: Translate the resulting caption back from language L to the original language
- 4: Output: Paraphrased caption Si′

B.2 Keyword Extraction

We extract keywords from both the original caption Si and the paraphrased caption Si′ using the Stanford POS Tagger. Keywords are identified as nouns (NN), adjectives (JJ), and verbs (VB), which are considered to encapsulate the core meaning of the sentence. We apply this process to both captions.

- Algorithm 2 Keyword Extraction

- 1: Input: Caption S
- 2: Apply POS tagging to S to obtain tags for each word
- 3: Extract words whose POS tags are in {NN, JJ, VB}
- 4: Output: List of extracted keywords K

#### B.3 Keyword Masking

We apply a masking function to randomly select one keyword from the extracted keywords and replace it with a placeholder token [MASK]. This is

done by identifying the position of the selected keyword in the sentence and substituting it with the placeholder.

- Algorithm 3 Keyword Masking

- 1: Input: Caption S, Keywords K
- 2: If K is empty then return "failed"
- 3: Randomly select a keyword k from K
- 4: Find the first occurrence of k in S
- 5: Replace k with the placeholder [MASK]
- 6: Output: Masked caption Smask

C Contamination Degree Analysis

Based on §4.1, the degrees on multi-choice datasets are defined as: ∆ ∈ (−1.6,−0.2] for minor leakage, ∆ ∈ (−2.9,−1.6] for partial leakage, and ∆ ≤ −2.9 for severe leakage. Based on §4.1, the degrees on caption datasets are defined as: ∆ ∈ (−2.4,−1.1] for minor leakage, ∆ ∈ (−5.0,−2.4] for partial leakage, and ∆ ≤ −5.0 for severe leakage. Details are shown in the algorithm 4.

- Algorithm 4 Contamination Degree Analysis Require: Benchmark dataset D, Model M

- 1: Define contamination degree CMinor, CPartial, CSevere
- 2: if D is multiple-choice then
- 3: Generate perturbed set Dpert via §3.1
- 4: else
- 5: Generate perturbed set Dpert via §3.2
- 6: end if
- 7: Compute CR, PCR, ∆, Φ using §3.3
- 8: if multiple-choice then
- 9: C ←

 



CMinor, ∆ ∈ (−1.6, −0.2] CPartial, ∆ ∈ (−2.9, −1.6] CSevere, ∆ ≤ −2.9

- 10: else
- 11: C ←

 



CMinor, ∆ ∈ (−2.4, −1.1] CPartial, ∆ ∈ (−5.0, −2.4] CSevere, ∆ ≤ −5.0

- 12: end if Ensure: CR, PCR, ∆, Φ, C

### D Other Experiments

D.1 Semantic&Lexical Similarity After Back-Translation

Setup. To quantify how much meaning and wording change during our caption perturbation step (§3.2), we applied an English→Chinese→English back-translation to every caption in three validation splits – COCO-Caption, NoCaps, and our Vintage dataset. For each original (c) and back-translated caption (c˜) we computed

• SBERT cosine similarity (Reimers and Gurevych, 2019) as a sentence-level semantic score, and

• BLEU-4 (Papineni et al., 2002) as a tokenoverlap lexical score.

We additionally report the Pearson correlation between the two metrics across captions within each dataset.

Dataset Avg. SBERT ↑ Avg. BLEU ↑ Correlation r

COCO Caption 0.894 0.236 0.386 NoCaps 0.887 0.264 0.410 Vintage 0.914 0.441 0.423

- Table 10: Average semantic (SBERT) and lexical (BLEU-4) similarity between original and backtranslated captions, together with their Pearson correlation (r).

#### Key Observations.

- • High semantic preservation. All three datasets record SBERT scores close to 0.9, indicating that back-translation keeps the meaning of captions largely intact; the VINTAGE split achieves the strongest preservation (0.914).
- • Substantial lexical variation. BLEU-4 values are comparatively low, showing that wording and surface forms differ considerably—consistent with the presence of synonym substitutions and syntactic reshuffling introduced by back-translation.
- • Weak yet positive coupling. Pearson correlations between the two metrics lie in the 0.380.42 band, suggesting only a mild positive relationship: captions that keep more tokens also tend to retain semantics, but plenty of cases preserve meaning even with low lexical overlap.

These results justify using back-translation as a semantics-preserving yet lexically diversifying perturbation in our contamination-detection pipeline.

D.2 Sanity Check for the “I don’t know” Instruction

Setup. To verify that appending the uncertainty clause “If you do not know the answer, output

“I don’t know”.” effectively suppresses random guessing, we performed a pilot experiment on 1000 randomly sampled questions from MMSTAR. All images were removed, so a truly vision-grounded model should either fail or explicitly abstain. We

evaluated the unimodal LLaMA2-7B language model under two settings:

- • Deter: deterministic decoding with the uncertainty instruction;
- • Non-Deter: deterministic decoding without the instruction.

Results. Table 11 shows that the instruction causes the model to respond “I don’t know” 238 times and reduces apparent accuracy from 44.8% to 25.6% (a drop of 19.2%). This confirms that nearly half of the seemingly correct answers in the uninstructed setting are likely due to lucky guesses rather than genuine reasoning, justifying our decision to include the clause in all main experiments.

Setting Accuracy (%) # “I don’t know” Deter (+instruction) 25.6 238 NonDeter (-instruction) 44.8 0

Table 11: Effect of the uncertainty instruction on LLaMA2-7B.

“I don’t know” will therefore be treated as an explicit abstention in the main study, ensuring reported accuracies reflect genuine visionlanguage capabilities rather than random chance.

#### D.3 Prompt Consistency

Our main experiments used a fixed prompt template across all evaluations. However, since different prompt wordings may influence model performance, we conducted additional experiments to assess the robustness of our methods. In particular, we tested two representative MLLMs (Claude3.5-Sonnet and Gemini-1.5-Pro) on SCIENCEQA and COCOCAPTION under alternative prompt templates. Results show that while absolute accuracy values fluctuate slightly, the contaminationsensitive metric ∆ remains nearly unchanged, confirming that our method is robust to prompt variations.

Option Order Sensitivity Test. The original prompt was formulated as follows:

Prompt 1 (Original) Please answer the following multichoice question. Question: {question} Reply with answer only.

We then compared two variants:

###### MLLMs Prompt Templates CR PCR ∆ Φ

- Claude-3.5-Sonnet Prompt 1 (Orig.) 70.3 65.0 -5.3 15.3
- Claude-3.5-Sonnet Prompt 2 70.4 (+0.1) 65.0 (+0.0) -5.4 (-0.1) 15.5 (+0.2)
- Claude-3.5-Sonnet Prompt 3 70.4 (+0.1) 65.0 (+0.0) -5.4 (-0.1) 15.5 (+0.2)

- Gemini-1.5-Pro Prompt 1 (Orig.) 68.5 67.9 -0.6 6.6
- Gemini-1.5-Pro Prompt 2 68.6 (+0.1) 68.0 (+0.1) -0.6 (+0.0) 6.5 (-0.1)
- Gemini-1.5-Pro Prompt 3 68.6 (+0.1) 68.0 (+0.1) -0.6 (+0.0) 6.5 (-0.1)

- Table 12: Prompt consistency test on ScienceQA. ∆ remains stable despite prompt variations.

MLLMs Prompt Templates CR PCR ∆ Φ

- Claude-3.5-Sonnet Prompt 1 (Orig.) 53.7 51.0 -2.7 21.8
- Claude-3.5-Sonnet Prompt 2 53.9 (+0.2) 51.1 (+0.1) -2.8 (-0.1) 22.0 (+0.2)
- Claude-3.5-Sonnet Prompt 3 53.8 (+0.1) 51.1 (+0.1) -2.7 (+0.0) 21.9 (+0.1)

- Gemini-1.5-Pro Prompt 1 (Orig.) 57.5 55.3 -2.2 21.6
- Gemini-1.5-Pro Prompt 2 57.6 (+0.1) 55.5 (+0.2) -2.1 (+0.1) 21.5 (-0.1)
- Gemini-1.5-Pro Prompt 3 57.6 (+0.1) 55.5 (+0.2) -2.1 (+0.1) 21.6 (+0.0)

- Table 13: Prompt consistency test on COCOCaption. Results show negligible effect on ∆.

- Prompt 2 Please respond to the following multiplechoice question. Question: {question} Provide only the answer.
- Prompt 3 Answer the multiple-choice question below. Question: {question} Reply with your answer only.

Results on ScienceQA are shown in Table 12. Across all prompt templates, ∆ changes by at most 0.1, highlighting stability.

Slot Guessing for Perturbed Caption. The original caption-prompt was:

Prompt 1 (Original) Fill in the [MASK] of the following sentence in one word: {caption} Only reply with the word you fill in the [MASK].

We added two reworded prompts:

- Prompt 2 Complete the [MASK] in the sentence below with a single word: {caption} Respond only with the word you used to replace [MASK].
- Prompt 3 Provide one word to fill in the [MASK] in the following sentence:

{caption} Your reply should only include the word you have selected for [MASK].

Results on COCOCaption (Table 13) again confirm robustness: CR and PCR shift slightly, but ∆ stays virtually unaffected. Overall, these experiments confirm that our detection framework is robust to natural prompt variations, further supporting its generality.

### E Detailed Dataset Overlap Analysis

It is impractical to quantify overlapping samples: 1) Many models do not release their complete training datasets publicly; instead, they only mention the data sources in their technical reports. 2) Even if we had access to complete training datasets, identifying specific overlapping samples using matching algorithms (such as exact match) remains challenging. This is because the original benchmarks might have undergone data augmentation before being used for model training, and multimodal benchmarks include images, both of which complicate the practical utility of matching algorithms. The feasible approach is manually reviewing the technical reports of these models to verify whether their training data overlaps with benchmarks, as shown in the table 14.

|MLLMs<br><br>|Multimodal Alignment/Pretraining Data<br><br>|Supervised Fine-Tuning Data|
|---|---|---|
|Phi-3-Vision<br><br>|Alignment Data includes FLD-5B.<br><br>|Not yet released|
| |Open Images is one source of FLD-5B.| |
| |Open Images is also a source of Nocaps.| |
| |Therefore, there is potential overlap in Nocaps.| |
|VILA|No overlap|Includes RefCOCO, VQAv2, GQA|
| | |MS COCO is a source of RefCOCO, VQAv2.|
| | |GQA’s source is Visual Genome Scene Graph, which also includes MS COCO.|
| | |COCO Caption’s source is MS COCO, and NoCaps’ source includes COCO Caption.|
| | |Therefore, there is potential overlap in COCO Caption and NoCaps.|
|Idefics2<br><br>|Alignment Data includes SBU Captions|SFT Data includes SBU Captions: potential overlap in COCO Caption and NoCaps.|
| |SBU Captions’ source includes Flickr| |
| |COCO Caption’s source includes MS COCO, and MS COCO’s source includes Flickr| |
| |NoCaps’ source includes COCO Caption| |
| |Therefore, there is potential overlap in COCO Caption and NoCaps.| |
|LLaVA-1.5|Alignment Data includes SBU Captions: COCO Caption and NoCaps with potential overlap.<br><br>|SFT Data includes RefCOCO, VQAv2, GQA: COCO Caption and NoCaps with potential overlap.|
|Yi-VL|Alignment Data includes Flickr, VQAv2, RefCOCO:|SFT Data includes GQA: COCO Caption and NoCaps with potential overlap.|
| |COCO Caption and NoCaps with potential overlap.| |
|DeepSeek-VL2<br><br>|No overlap<br><br>|SFT Data includes Flickr, GQA: COCO Caption and NoCaps with potential overlap.|
|Qwen-VL-Chat|Directly uses COCO Caption in the pretraining stage,<br><br>|Not yet released|
| |therefore there is partial or entire overlap in COCO Caption and NoCaps.| |
|InternVL2<br><br>|Alignment Data includes COCO Caption: partial or entire overlap in COCO Caption and NoCaps.|SFT Data includes ScienceQA, therefore there is partial or entire overlap in ScienceQA.|

###### Table 14: Comparison of MLLMs and Their Data Sources

