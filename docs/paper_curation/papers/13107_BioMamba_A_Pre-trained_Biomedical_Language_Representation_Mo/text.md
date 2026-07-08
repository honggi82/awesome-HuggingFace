# arXiv:2408.02600v3[cs.CL]10Jun2026

## BioMamba: Domain-Adaptive Biomedical Language Models

Ling Yue1,*, Mingzhi Zhu1,*, Sixue Xing1, Yunning Cao2,†, Yanbo Wang3,†, Shimin Shan3, Jinfei Liu4, Vijil Chenthamarakshan5, Shaowu Pan1, Payel Das5,†, and Tianfan Fu6

1Rensselaer Polytechnic Institute, Troy, USA. 2Jiaxing New Jies Thermal Power Co. Ltd., Jiaxing, China. 3North University of China, Taiyuan, China. 4Zhejiang University, Hangzhou, China. 5IBM Research, Yorktown Heights, USA. 6Nanjing University, Nanjing, China. †Address correspondence to: caoyn@zjmi.com (Y.C.); yanbowang@nuc.edu.cn (Y.W.); daspa@us.ibm.com (P.D.); *These authors contributed equally to this work.

Abstract

Background. Biomedical language models should improve performance on biomedical text while retaining general-language-modeling fluency. For Mamba-based models, this trade-off has not been systematically studied across biomedical literature and clinical text.

Methods. We developed BioMamba, a family of biomedical Mamba2 models at five scales (130M, 370M, 780M, 1.3B, and 2.7B) obtained by continued pretraining of released public Mamba2 checkpoints on a balanced 80%/10%/10% mixture of PubMed abstracts, the Colossal Clean Crawled Corpus (C4), and Wikipedia. The architecture is unchanged from Mamba2; the contribution is the adaptation recipe and the accompanying open-weight checkpoints. We evaluated internal language modeling on fixed held-out sets, out-of-domain multiple-choice benchmarks, and three downstream tasks across multiple model scales: clinical note completion and discharge summary generation on MIMIC-IV-Note, and biomedical yes/no question answering on BioASQ and PubMedQA.

Results. Across five scales, BioMamba consistently lowered PubMed perplexity, improved Wikipedia-style held-out perplexity by 1.46–4.72 PPL, and left C4 perplexity essentially unchanged (|∆PPL| ≤ 0.17). On six out-of-domain multiple-choice benchmarks, BioMamba stayed within ±3 percentage points of Mamba2 with no systematic regression. After supervised finetuning, BioMamba+SFT matched or exceeded Mamba2+SFT on MIMIC-IV note completion and discharge summary generation at every evaluated scale (paired-bootstrap P(∆ > 0) ≥ 0.997 at 2.7B), and improved PubMedQA at every scale. The strongest model (BioMamba-2.7B)

reached a PubMed perplexity of 5.28 and accuracies of 90.24% and 73.00% on BioASQ and PubMedQA, respectively.

Conclusions. A balanced domain-adaptive continued pretraining recipe strengthens Mamba2 language models on biomedical literature and clinical text while preserving general-languagemodeling fluency.

### 1 Introduction

Biomedical research and health data science increasingly rely on unstructured text, including PubMed abstracts, full-text articles, clinical guidelines, trial reports, and clinical documentation. Domainadapted language models are therefore important for biomedical information extraction, evidence synthesis, summarization, completion, and question answering [1]. Transformer-based biomedical models such as BioBERT [2], SciBERT [3], PubMedBERT [4], BioGPT [5], SciFive [6], and BioBART [7] have shown that adapting general-purpose language models to biomedical corpora can yield strong performance. In clinical natural language processing, ClinicalBERT [8], GatorTron [9], and ClinicalT5 [10] further demonstrate the value of domain adaptation for clinical text. Recent work in health data science and public-health artificial intelligence has also emphasized that models used in biomedical workflows should be accurate, transparent, and practically useful [11–16].

Most biomedical language models still use Transformer self-attention, which has quadratic cost in sequence length [17]. This cost becomes restrictive when biomedical inputs are long, terminologydense, and distributed across multiple sentences or sections. State space models, especially structured state space models [18], offer an alternative. Mamba introduced selective state space layers for linear-time sequence modeling, and later work further strengthened the connection between state space models and attention while preserving favorable scaling for long contexts [19, 20].

Despite this progress, biomedical adaptation of Mamba models remains limited. Biomedical language modeling is still dominated by Transformer families and related open medical language models [2, 4, 5, 21–24], and the few reported Mamba-based biomedical adaptations have focused on narrower settings such as clinical notes [25]. Second, domain adaptation risks catastrophic forgetting: a model may improve on biomedical text while losing general-language-modeling fluency, which is a particular concern since biomedical writing often mixes specialized terminology with broader scientific language. This trade-off has been documented in domain-adaptive pretraining literature [26, 27], and in work explicitly targeting general knowledge preservation [28].

We address this gap with BioMamba, a family of biomedical models built by continued pretraining of released Mamba2 checkpoints. Our contribution is a reproducible biomedical continued pretraining recipe together with a family of open-weight BioMamba checkpoints at 130M, 370M, 780M, 1.3B, and 2.7B parameters; the architecture itself is unchanged from Mamba2. The main focus is therefore the adaptation strategy, its empirical effects, and the accompanying resource release rather than an architectural modification. This study makes three contributions:

- • A public release of the training and evaluation code and all five checkpoints at https://github. com/lingyue404/BioMamba under the Apache-2.0 license, so that the biomedical natural language processing community can directly use or extend the models.

- • A systematic investigation of the continued pretraining pipeline showing that an appropriate 80%/10%/10% PubMed/C4/Wikipedia mixture can enhance biomedical language modeling while preserving general-language-modeling fluency and mitigating catastrophic forgetting.
- • Controlled base-versus-adapted experiments showing that BioMamba achieves consistent improvements across downstream tasks, including biomedical question answering (PubMedQA, BioASQ), clinical note completion, and discharge summary generation on MIMIC-IV.

### 2 Methods

#### 2.1 Overview

This study tests whether biomedical continued pretraining [26, 27] can improve Mamba language models while retaining general-language-modeling fluency. Figure 1 shows the overall workflow. We started from public Mamba2 checkpoints, continued pretraining on a mixed corpus of PubMed, C4, and Wikipedia, evaluated language-model quality on fixed in-domain and out-of-domain validation sets, and then fine-tuned the resulting models on clinical note completion, discharge summary generation, and biomedical question answering.

Before analysis, we specified which comparisons would be treated as primary. The primary comparison is between each base model and its BioMamba counterpart under a shared tokenizer and fixed evaluation sets. These comparisons isolate the effect of biomedical adaptation most directly. Comparisons with public checkpoints are included only for context because tokenizers and prior training histories differ across released models.

#### 2.2 Model family and training data

BioMamba was obtained by continued pretraining of released Hugging Face Mamba2 checkpoints from state-spaces/mamba2-* [19, 20] at five model scales spanning 130M–2.7B parameters. To keep tokenization fixed across internal comparisons, all models use the shared GPT-NeoX tokenizer [29] from state-spaces/mamba-2.8b-hf, which has a vocabulary of 50,280 tokens.

All BioMamba variants were trained autoregressively with the standard next-token objective,

T

LLM = −

t=1

log P(xt | x<t;θ), (1)

where xt is the token at position t, x<t denotes the preceding context, and θ represents model parameters.

PubMed abstracts indexed in Medical Literature Analysis and Retrieval System Online (MEDLINE) provided the biomedical data [30], and we added C4 [31] and English Wikipedia [32] in smaller proportions as the general-domain component of the mixture. After tokenization, filtering, and shuffling, the final corpus contained approximately 508K sequences of length 1,024 tokens (Table 1). Mixture selection was performed in a separate 130M pilot ablation before retraining the final 80%/10%/10% configuration used for the main BioMamba checkpoints.

- A Architecture Comparison RNN

Hidden State ht

▼

Recurrent Update

▼

Output ot

O(n) sequential

➔

Transformer

Positional Encoding

▼

Multi-Head Self-Attention

▼

Add & Layer Norm

▼

Feed-Forward Network

▼

Add & Layer Norm

O(n²) parallel

➔

Mamba (SSM)

Linear Projection

▼

1D Convolution

▼

Selective State Space (S6)

▼

Gated Residual

▼

Output Projection

O(n) parallel

- B BioMamba Training & Evaluation Pipeline

TRAINING DATA PubMed

###### DOWNSTREAM TASKS

MODEL FAMILY

SFT

CPT

###### ? BioASQ & PubMedQA

MEDLINE abstracts 80%

###### BioMamba

Biomedical yes/no QA

Supervised FineTuning

Continued Pretraining

Mamba2-based

###### C4

###### Note Completion

130M 370M 780M 1.3B 2.7B

Web text corpus 10%

3 epochs · AdamW WSD schedule

MIMIC-IV clinical notes

Full parameter Task-specific

###### W Wikipedia

###### Discharge Summary

English Wikipedia 10%

MIMIC-IV summarization

- Figure 1: Overview of BioMamba. (A) Architecture comparison: RNN, Transformer, and Mamba (SSM), with key distinguishing components highlighted. (B) Training and evaluation pipeline: continued pretraining on a mixed PubMed/C4/Wikipedia corpus produces BioMamba models at five scales (130M–2.7B), which are then fine-tuned for biomedical question answering, clinical note completion, and discharge summary generation. WSD: Warmup–Stable–Decay learning-rate schedule.

#### 2.3 Training procedure

Continued pretraining was performed for 3 epochs on the mixed PubMed/C4/Wikipedia corpus using AdamW [33] optimization with weight decay 0.1, BF16 mixed precision, a maximum sequence length of 1,024 tokens, gradient clipping of 1.0, and random seed 42. To keep optimization roughly comparable across scales, we adjusted per-device microbatch size and gradient accumulation so that the effective batch size stayed approximately constant, at 240–256 sequences for the 130M–1.3B models and 192 sequences for the 2.7B model. This yielded 5,967–7,953 optimizer steps over the full 3-epoch training budget. Exponential moving average of model weights was enabled throughout training, and the checkpoint with the lowest validation loss was kept for downstream evaluation.

To mitigate catastrophic forgetting, we employed a conservative warmup–stable–decay (WSD) schedule together with layer-wise learning-rate decay, which updates lower layers more conservatively than higher layers using a layer-wise decay factor of 0.90 at 130M and 0.95 for the larger models. Peak learning rates decreased with model size, from 1.5 × 10−6 at 130M to 2.0 × 10−7 at 2.7B (4.5 × 10−7 for 370M, 3.0 × 10−7 for 780M, and 2.5 × 10−7 for 1.3B).

Shared and model-specific implementation details, including exact scheduler configuration, perscale microbatch settings, hardware allocation, and memory usage, are reported in Supplementary

- Table 1: Continued pretraining data composition. C4 denotes the Colossal Clean Crawled Corpus.

Component Source Proportion Sequences PubMed Abstracts indexed in MEDLINE (1,024-

∼80% ∼424K

token sequences)

C4 General-domain web text ∼10% ∼42K Wikipedia English Wikipedia ∼10% ∼42K

###### Total 100% ∼508K

Section S2. As shown later in the Results, training remained stable across all scales, with smooth decreases in validation loss and the best validation point occurring at or near the final step in every run.

#### 2.4 Evaluation setup

Internal language-model evaluation. The primary language-model evaluation used fixed validation sets of 1,000 held-out sequences for PubMed [30] and C4 [31], drawn from the same data pools as continued pretraining but excluded from the training split, and tokenized with the shared tokenizer. For the Wikipedia held-out set, we used the Wikitext-103 raw split, referred to throughout this paper as the Wikipedia-style held-out set for brevity, and tokenized it with the same shared tokenizer. This is the most controlled comparison in the study because the tokenizer is held constant across all internal evaluations and each benchmark split is fixed in advance.

Contextual comparison with public checkpoints. To place BioMamba alongside released biomedical and medically adapted checkpoints, we also evaluated public models on the same raw text using each model’s native tokenizer. Because perplexity depends on tokenization, these results are contextual rather than strictly comparable and are interpreted only as secondary model-level evidence. Exact checkpoint sources are listed in Supplementary Section S3.

#### 2.5 Downstream tasks

We evaluated downstream transfer to clinical text on two complementary supervised generation tasks constructed from de-identified discharge summaries in MIMIC-IV-Note v2.2 [34] within the broader MIMIC-IV resource [35]. Note completion evaluates continuation ability: the model receives the first half of a discharge note and generates its continuation. Discharge summary generation evaluates summarization ability: the model receives structured admission sections and generates discharge sections. To avoid information leakage, splits were defined at the patient level, and quantitative evaluation used 500 randomly sampled held-out test notes per task. These clinical generation experiments were conducted at all five model scales. The two clinical generation settings are closely related to prior work on hospital-course summarization and discharge-note generation [36–40]. Full data construction, fine-tuning settings, and qualitative examples are provided in Supplementary Sections S5 and S6.

We also evaluated downstream transfer on biomedical yes/no question answering [1] because it provides a simple and interpretable test of whether the model can use biomedical knowledge after continued pretraining. Supervised fine-tuning used a mixture of Biomedical Semantic Indexing and Question Answering (BioASQ) yes/no training examples from the BioASQ challenge series [41–43] aggregated from BioASQ 7B [44] and 13B [45] (1,114 samples after class balancing) and a binary yes/no split derived from the PubMed Question Answering (PubMedQA) dataset [46] (800 training samples). Held-out evaluation was conducted on BioASQ 13B and the corresponding 200-sample binary PubMedQA test split. For both downstream settings, supervised fine-tuning updated all model parameters rather than using parameter-efficient adapters.

- 2.6 Metrics and analysis For language modeling, we report cross-entropy-derived perplexity (PPL),

PPL = exp(L¯), (2)

where L¯ is the mean next-token cross-entropy on the evaluation set. Lower PPL indicates better language modeling. For downstream question answering, we report accuracy and macro-F1 score, where the macro-F1 score is the arithmetic mean of the class-specific yes and no F1 scores. Classspecific yes and no F1 scores are reported when helpful to show which class benefited most from continued pretraining.

For clinical text generation, note completion is treated as a completion task and discharge summary generation as a summarization task. For both, we report ROUGE-1, ROUGE-2, and ROUGE-L [47]. ROUGE-n measures the n-gram recall between the generated text and the reference:

ROUGE-n = s∈R gramn∈s Countmatch(gramn) s∈R gramn∈s Count(gramn)

, (3)

where R is the set of reference sentences and Countmatch counts the number of n-grams that appear in both the generated and reference texts. ROUGE-L uses the longest common subsequence (LCS) instead of fixed-length n-grams:

(1 + β2) · Plcs · Rlcs Rlcs + β2 · Plcs

ROUGE-L =

, (4)

where Plcs = LCS(X,Y )/|X| and Rlcs = LCS(X,Y )/|Y | are LCS-based precision and recall, and β is set to favor recall. Higher ROUGE scores indicate greater overlap with the reference text.

For downstream accuracy, we report 95% Wilson binomial confidence intervals in the text when helpful to reflect the small size of the BioASQ test set (n = 82).

### 3 Results

#### 3.1 Biomedical language modeling

Quantitative results. Under the tokenizer-controlled internal comparison, continued pretraining lowered PubMed perplexity at every model scale (Table 2). Continued pretraining yielded consistent reductions in PubMed perplexity across all scales, dropping from 9.41 to 8.42 in the 130M model and scaling down to 5.28 in the 2.7B model. The relative PubMed perplexity reduction ranged from 7% to 11% across scales, demonstrating that biomedical adaptation yields consistent gains even at the largest model size. Wikipedia-style held-out perplexity also decreased at every scale, consistent with the inclusion of Wikipedia in the continued pretraining mixture. By contrast, C4 perplexity changed by at most about 1%, so PubMed gains did not come at the cost of out-of-domain C4 performance before any task-specific fine-tuning.

- Table 2: Internal perplexity comparison between base Mamba2 models and BioMamba. The base models are the public checkpoints used for initialization.

###### Model PubMed PPL (↓) Wikipedia PPL (↓) C4 PPL (↓)

Mamba2-130M (base) 9.41 23.09 26.58 BioMamba-130M 8.42 18.76 26.58

Mamba2-370M (base) 7.41 16.17 20.06 BioMamba-370M 6.75 13.36 20.01

Mamba2-780M (base) 6.61 13.54 17.32 BioMamba-780M 6.15 11.58 17.32

- Mamba2-1.3B (base) 6.12 11.91 15.59

- BioMamba-1.3B 5.66 10.04 15.43

Mamba2-2.7B (base) 5.67 10.28 13.99

- BioMamba-2.7B 5.28 8.85 13.87

Case study. The smallest model, BioMamba-130M, shows the main pattern clearly. Continued pretraining lowered PubMed perplexity from 9.41 to 8.42 and Wikipedia-style held-out perplexity from 23.09 to 18.76, while C4 perplexity remained 26.58. This example shows that the mixed-corpus strategy improved biomedical language modeling without harming the general-language-modeling fluency measured on C4.

Contextual comparison with public checkpoints. Table 3 reports raw-text perplexity for BioMamba alongside released biomedical and medically adapted checkpoints, each evaluated with its own native tokenizer. Because tokenizers, pretraining data, and adaptation protocols differ, we use these results only to situate BioMamba within the released-model landscape rather than as a controlled architecture-level comparison; within that contextual view, BioMamba-2.7B falls among the strongest raw-text perplexities in this set, while BioMamba-130M remains in a competitive range at much smaller scale.

- 4

- 5

- 6

- 7

- 8

- 9

- 10

Mamba2 (base) BioMamba

| |
|---|

11%

PubMedPerplexity()

9%

7%

8%

7%

130M 370M 780M 1.3B 2.7B

Model Scale

- Figure 2: PubMed perplexity comparison between base Mamba2 models and BioMamba across five model scales. Percentage labels indicate the relative perplexity reduction achieved by continued pretraining at each fixed model scale. The within-scale gap between each base model and its BioMamba counterpart is largest at 130M, indicating that the marginal benefit of continued pretraining is most pronounced for the smallest model.

- Table 3: Raw-text perplexity comparison with released biomedical and medically adapted language models, each evaluated on the same text with its own native tokenizer.

Model # Params PubMed PPL (↓) Wikipedia PPL (↓) C4 PPL (↓)

BioMamba-130M (ours) 130M 11.08 20.49 29.01 BioGPT 347M 15.09 61.31 104.62 Bio-Medical-Llama-3.2-1B 1B 14.46 22.54 36.42 Gemma3-1B (fine-tuned) ∼1B 19.90 35.01 41.67 BioGPT-Large 1.5B 10.00 35.83 50.79 BioGPT-Large-PubMedQA 1.5B 9.26 43.54 71.96 Meditron3-Gemma2-2B 2B 6.67 11.61 15.62 BioMedLM 2.7B 25.27 100.24 95.74 BioMamba-2.7B (ours) 2.7B 6.52 9.71 14.91

#### 3.2 Out-of-domain evaluation and evaluation-budget stability

To evaluate whether the 80%/10%/10% recipe preserves general-language-modeling fluency beyond the in-domain perplexity benchmarks, we added two complementary analyses. First, we evaluated all five BioMamba checkpoints and their Mamba2 base counterparts on six out-of-domain multiple-choice benchmarks (LAMBADA, HellaSwag, ARC-Easy, ARC-Challenge, PIQA, OpenBookQA), scored by continuation log-likelihood. Across tasks and scales, BioMamba stays within ±3 percentage points of Mamba2 on every task with no systematic regression (Table 4); the per-scale mean change ranges only from −0.51 to +0.58 percentage points, which is consistent with the absence of catastrophic forgetting on broad commonsense reasoning, reading comprehension, and elementary-science benchmarks.

- Table 4: Out-of-domain multiple-choice summary across six benchmarks. For each scale we report the mean BioMamba−Mamba2 accuracy difference averaged over the six tasks and the range. All per-task ∆ lie inside ±3pp at every scale. The full accuracy grid is reported in Supplementary Table S14.

Scale Mean ∆ (pp) Per-task ∆ range (pp)

130M −0.22 [−2.10, +0.80] 370M +0.22 [−1.60, +1.67] 780M −0.51 [−1.20, +0.53]

- 1.3B +0.58 [−0.33, +2.36]
- 2.7B +0.10 [−0.67, +0.60]

Second, we checked that the Section 3.1 perplexity numbers remain stable when the evaluation budget is increased from 1024 to 2048 and 4096 tokens. We therefore re-evaluated C4, Wikitext-103, and PubMed-medline at evaluation budgets of 1024, 2048, and 4096 tokens. Across all scales and budgets, values agree within ±0.01 PPL; Table 5 therefore reports a representative single column rather than a three-column grid (the 2048 column is representative of 1024 and 4096 under this rounding). The Mamba2→BioMamba deltas are |∆PPL| ≤ 0.17 on C4, −1.46 to −4.72 PPL on Wikitext-103, and −0.11 to −0.44 PPL on PubMed-medline, with the Wikitext-103 gain largest at the smaller scales and decreasing monotonically with model size—consistent with the Section 3.1 finding that continued pretraining is especially helpful in the lower-cost regime.

- Table 5: Extended-budget perplexity for Mamba2 base and BioMamba on C4, Wikitext-103, and PubMed-medline. Each numeric cell shows Mamba2→BioMamba with the delta in parentheses.

Scale C4 PPL Wikitext-103 PPL PubMed PPL

130M 22.42 → 22.40 (−0.02) 23.42 → 18.70 (−4.72) 11.67 → 11.23 (−0.44) 370M 17.09 → 17.05 (−0.04) 16.41 → 13.61 (−2.80) 9.06 → 8.80 (−0.26) 780M 14.84 → 14.81 (−0.03) 13.66 → 11.62 (−2.04) 7.99 → 7.85 (−0.14)

- 1.3B 13.37 → 13.25 (−0.12) 12.03 → 10.06 (−1.97) 7.40 → 7.26 (−0.14)
- 2.7B 12.10 → 11.97 (−0.13) 10.36 → 8.90 (−1.46) 6.80 → 6.69 (−0.11)

#### 3.3 Clinical note completion and discharge summary generation on MIMIC-IV

We next asked whether biomedical adaptation learned from PubMed would transfer to clinical documentation. We therefore evaluated the models on MIMIC-IV-Note v2.2, using the patient-level split and 500-sample held-out evaluation described in the Methods and in Supplementary Section S5.

Clinical supervised fine-tuning. Starting from both the Mamba2 base checkpoints and the BioMamba checkpoints, we performed clinical supervised fine-tuning (SFT) on two generation tasks constructed from the discharge notes. Training data comprised up to 20,000 note-completion samples and approximately 12,893 discharge-generation samples (32,893 total; 90% train, 10% validation). Instruction masking was applied so that only response tokens contributed to the loss. BioMamba+SFT uses a lower learning rate and more training epochs than Mamba2+SFT because aggressive fine-tuning tends to overwrite the information acquired during continued pretraining; to verify that this design choice does not create an unfair comparison, we additionally ran a matched-budget control in which Mamba2 was fine-tuned with the exact BioMamba SFT recipe, and the BioMamba schedule did not close the gap (Supplementary Section S9). We also found that validation loss was not a reliable proxy for downstream ROUGE in the BioMamba setting, which further motivates the conservative BioMamba SFT schedule. Full SFT data construction, hyperparameter configurations, and tuning details are provided in Supplementary Section S5.

Evaluation tasks. We evaluated two clinical generation tasks that reflect different capabilities. In note completion, the model receives the first half of a discharge note and generates a 128-token continuation (greedy decoding), scored against the ground-truth second half by ROUGE-1, ROUGE-2, and ROUGE-L. In discharge summary generation, the model receives structured admission sections and generates discharge sections, again scored by ROUGE.

Results. Table 6 presents MIMIC-IV ROUGE across all five scales, with the Mamba2+SFT / BioMamba+SFT label convention of Table 12. For the 2.7B row we additionally report, in Supplementary Section S14, (i) per-arm 95% percentile-bootstrap confidence intervals for ROUGE-1 and ROUGE-L

- (Supplementary Table S20) and (ii) paired-bootstrap significance on per-sample ROUGE-1 differences
- (Supplementary Table S21), both with 1000 resamples over the 500 test samples. Three patterns emerge.

First, clinical SFT is the dominant driver of generation quality: at the 130M, 370M, 780M, and 1.3B scales where base Mamba2 rows are reported, both Mamba2+SFT and BioMamba+SFT substantially outperform the corresponding base models on every ROUGE metric.

Second, BioMamba+SFT consistently matches or outperforms Mamba2+SFT on ROUGE-1 and ROUGE-L at every scale on both completion and discharge generation, with the single exception of a 0.01-point ROUGE-2 gap on 1.3B discharge (3.70 vs. 3.69). The advantage is most pronounced on discharge generation at 130M (ROUGE-1 9.74 vs. 8.79, ∆=+0.95). At 2.7B, paired-bootstrap tests on per-sample ROUGE-1 differences give P(∆ > 0) = 1.00 for completion and P(∆ > 0) = 0.997 for discharge, so the advantage is not within test-sample noise.

Third, within each arm MIMIC-IV ROUGE improves with scale up to 1.3B (BioMamba+SFT reaches 8.11 / 10.11 on completion / discharge at 1.3B) and is slightly lower at 2.7B (7.57 / 9.39); the BioMamba+SFT advantage over Mamba2+SFT nonetheless persists at every scale, so the effect of continued pretraining is not absorbed by scaling on this dataset.

- Table 6: MIMIC-IV clinical generation results (ROUGE %, 500 test samples per task). Boldface indicates the best result within each model size. Comp = note completion; Disch = discharge summary generation. R-1/2/L = Rouge-1/2/L. The 2.7B block omits a base-only row because Mamba2+SFT is the appropriate fairness baseline at that scale. Per-arm 95% percentile-bootstrap CIs for R-1 and R-L at 2.7B (Supplementary Table S20) and paired-bootstrap significance on per-sample ROUGE-1 differences at 2.7B (Supplementary Table S21) are both in Supplementary Section S14.

Note completion Discharge generation Model R-1 ↑ R-2 ↑ R-L ↑ R-1 ↑ R-2 ↑ R-L ↑ 130M

- Mamba2 4.63 0.80 3.43 4.83 0.35 3.56

- Mamba2+SFT 6.84 2.36 4.95 8.79 2.76 6.30 BioMamba+SFT 7.07 2.45 5.10 9.74 3.33 6.96

370M Mamba2 5.04 0.97 3.69 4.50 0.33 3.31

- Mamba2+SFT 7.85 3.04 5.62 8.96 2.69 6.19 BioMamba+SFT 7.94 3.20 5.69 9.22 3.00 6.39

780M

Mamba2 5.46 1.17 3.86 5.22 0.37 3.59 Mamba2+SFT 7.90 3.24 5.73 8.86 2.73 6.14

###### BioMamba+SFT 7.90 3.32 5.78 9.10 2.97 6.30

- 1.3B Mamba2 5.61 1.28 3.95 5.33 0.42 3.68 Mamba2+SFT 7.93 3.28 5.76 9.99 3.70 7.04

BioMamba+SFT 8.11 3.33 5.89 10.11 3.69 7.04

- 2.7B Mamba2 5.55 1.19 3.85 7.04 0.80 4.71 Mamba2+SFT 7.16 2.45 4.98 8.87 2.76 6.21 BioMamba+SFT 7.57 2.83 5.30 9.39 3.01 6.46

Representative case studies. The ROUGE improvements corresponded to clear qualitative differences in clinical generation. Table 7 shows a representative note-completion example from the

- 1.3B model family. The base Mamba2 model incorrectly propagated the cardiac abbreviation “RRR” to unrelated organ systems, producing clinically implausible output. Mamba2+SFT corrected the basic clinical register, whereas BioMamba+SFT produced the most complete continuation, including pertinent negatives and a natural transition to the discharge physical-examination section.

Table 8 shows a representative discharge-summary generation example after vascular surgery. Here the base Mamba2 model largely repeated procedural details, Mamba2+SFT generated only a partial discharge framework, and BioMamba+SFT moved directly to a structured discharge-medication

section with a clinically coherent regimen. These examples indicate that the quantitative gains reflect better completion and summarization behavior rather than only small changes in lexical overlap. Additional qualitative cases, including multi-panel laboratory continuation, medicationreconciliation continuation, Hospital Course generation, and concise discharge instructions, are provided in Supplementary Section S6.

- Table 7: Representative note-completion case study from the 1.3B model family. The input ends with “...HEART: RRR, no MRG, nl”. Each non-reference row contains a short excerpt from the model’s generation (before the em-dash) and a one-line author reading of that excerpt. Full raw outputs for this and three additional examples are in Supplementary Section S6. Source Excerpt (before em-dash) + author reading (after em-dash) R-1

Reference S1-S2. LUNGS: CTA bilat, no ronchi/rales/wheezes. . . ABDOMEN: Mid-

line surgical scar. Right sided PD catheter... EXTREMITIES: WWP, no edema, 2+ peripheral pulses. NEURO: Awake, A&Ox3, CNs II-XII grossly intact. (reference text; no reading.)

—

Mamba2 “LUNGS: RRR, no rales; ABDOMEN: RRR, no tenderness; URINARY: RRR... DIAPER: RRR” — incorrectly applies the cardiac rhythm descriptor “RRR” to unrelated organ systems.

2.73

Mamba2+SFT “S1-S2; LUNGS: CTAB, no r/rh/wh. . . ABDOMEN: Soft, NTND.. . EXTREMITIES: WWP” — clinically plausible continuation with standard abbreviations.

3.65

BioMamba+SFT “S1-S2; LUNGS: CTAB, no wheezes, rales, rhonchi... ABDOMEN: Soft, NTND, no rebound or guarding... NEURO: A&Ox3, moving all four extremities... Physical Exam on Discharge:” — adds pertinent negatives and transitions to the discharge physical-examination section.

3.84

- Table 8: Representative discharge-summary generation case study from the 1.3B model family. Input: Service: SURGERY; Chief Complaint: Foot Ulcer; Procedure: Ultrasound-guided access to the left common femoral artery, selective catheterization of the right superficial femoral artery... Cell text is a short excerpt from the model’s generation (verbatim tokens in quotes) followed by an author reading (in italics). Full raw outputs for this and three additional examples are in Supplementary Section S6.

Source Excerpt + author reading (italicized) R-1 R-2 Reference DISCHARGE MEDICATIONS: 1. Clopidogrel 75mg PO

— —

DAILY 2. Sulfameth/Trimethoprim DS 2 TAB PO BID... (reference text; no reading.)

Mamba2 “8 x 80mm Innova stent post dilated with a 7mm balloon” (procedure phrasing repeated) — repeats procedure details and fails to transition to discharge content.

10.85 0.68

Mamba2+SFT Generates a partial discharge framework and follow-up surgical description, but no explicit medication list. (Excerpt omitted here for space; see Supplementary Section S6.)

17.87 2.08

BioMamba+SFT “DISCHARGE MEDICATIONS: Aspirin 81mg, Atorvastatin 40mg, Metoprolol Tartrate 50mg...” — concisely summarizes the procedure and moves directly to a structured medication regimen consistent with post-vascular surgery care.

###### 23.05 3.75

These results show that biomedical knowledge acquired during PubMed continued pretraining transfers to clinical text generation. The consistent advantage of BioMamba+SFT over Mamba2+SFT suggests that literature-based continued pretraining adds value even when clinical SFT data are available, although learning-rate tuning remains important during fine-tuning.

Output diagnostics. To complement the ROUGE comparison with quantitative summaries of generation behavior, we computed standard output diagnostics on 1.3B discharge generation under nucleus-0.9 and nucleus-0.95 decoding (Table 9): average length, distinct-1 and distinct-2 token-type ratios, and a repeated-4-gram rate. Greedy decoding remains the apples-to-apples primary setting for Table 6; we report diversity and repetition diagnostics only for the higher-entropy nucleus settings, where these metrics are informative. Under nucleus-0.95, a standard deployment setting for open-ended clinical text, BioMamba has slightly better diversity and a lower repeated-4-gram rate than Mamba2; under nucleus-0.9 the two arms are close. A lightweight section-header F1 structure metric is reported in Supplementary Table S19.

- Table 9: MIMIC-IV discharge output diagnostics at 1.3B under nucleus-0.9 and nucleus-0.95 decoding: average generated length, distinct-1 and distinct-2 token-type ratios, and the repeated-4-gram rate. Greedy decoding used in Table 6 is omitted from this table.

Model Decoding Avg length Dist-1 Dist-2 Rep-4-gram

Mamba2+SFT nucleus-0.9 74.6 0.730 0.876 0.071 BioMamba+SFT nucleus-0.9 74.8 0.719 0.865 0.077

Mamba2+SFT nucleus-0.95 74.1 0.770 0.914 0.040 BioMamba+SFT nucleus-0.95 74.6 0.777 0.918 0.036

Factuality proxy analysis. To move beyond ROUGE as a surface-lexical metric and provide a grounding-oriented check of clinical generation, we added an NLI-ensemble factuality proxy on

- 2.7B MIMIC discharge generations (Table 10). Using two MNLI checkpoints (roberta-large-mnli and microsoft/deberta-v2-xlarge-mnli), a generation is flagged as contradictory only when both checkpoints independently output a contradiction probability above 0.7. We scored entailment of each generated discharge section against the input admission text only. Across all six tested cells the contradiction rate lies in the 7.0–9.4% range, with Wilson 95% CIs overlapping substantially between arms at every decoding. Macro-averaged across the three decoding strategies, BioMamba (7.4%) is below Mamba2 (8.1%); under nucleus-0.95 sampling Mamba2 rises to 9.4% while BioMamba stays at 7.0%—a 2.4 pp gap in BioMamba’s favour. Continued pretraining therefore does not introduce systematic source-contradictory output relative to the base model, and is more stable than the base model under higher-entropy sampling. We present this analysis as a factuality proxy rather than a clinician audit and treat safety, calibration, and workflow readiness as out-of-scope for this study (see Section 4); the NLI protocol and the schema of the auto-flagged clinician-review candidate list are in Supplementary Section S12.

- Table 10: NLI-ensemble factuality proxy on 2.7B MIMIC discharge generation (n = 500 per cell; two MNLI checkpoints, roberta-large-mnli and microsoft/deberta-v2-xlarge-mnli). M2+SFT = Mamba2+SFT; BM+SFT = BioMamba+SFT. Contradiction rate is computed against the input admission text only, lower is better. 95% confidence intervals are Wilson score intervals.

2.7B arm Decoding Contradictions / n Rate (↓) [Wilson 95% CI]

M2+SFT

greedy 35 / 500 7.0% [5.1, 9.6] nucleus-0.9 39 / 500 7.8% [5.8, 10.5] nucleus-0.95 47 / 500 9.4% [7.1, 12.3]

macro-avg — 8.1%

BM+SFT

greedy 37 / 500 7.4% [5.4, 10.0] nucleus-0.9 39 / 500 7.8% [5.8, 10.5] nucleus-0.95 35 / 500 7.0% [5.1, 9.6]

macro-avg — 7.4%

Failure modes. The remaining errors are consistent with the behaviors already visible in the qualitative examples. For MIMIC, the main residual issues are occasional repetition under greedy decoding, occasional truncation of the target section when the 128-token budget is reached before the discharge medication list completes, and incomplete coverage of medication-level detail in a subset of generations; these are directly reflected in the nucleus-0.95 diagnostic improvements and in the qualitative case studies in Supplementary Section S6. For the question-answering tasks, most BioASQ errors at larger scales are shared between Mamba2+SFT and BioMamba+SFT: at 780M, every item BioMamba+SFT gets wrong is also missed by Mamba2+SFT, and at 1.3B the overlap is 11 of 12 (see Supplementary Section S11). Residual disagreements at larger scales are therefore confined to a handful of genuinely hard items rather than indicating a regression attributable to continued pretraining.

- 3.4 Contextual comparison with external models on MIMIC-IV

- Table 11 compares BioMamba’s clinical generation results with seven public biomedical or medically adapted language models on the same 500 MIMIC-IV test samples under identical greedy decoding. The external baselines were not given MIMIC-specific supervised fine-tuning, so this is a contextual rather than task-matched comparison; caveats are summarized in the caption.

Within this contextual setup, the task-specific BioMamba rows lie above the untuned external baselines on both MIMIC tasks. We interpret this only as evidence that BioMamba is competitive relative to released checkpoints in the current landscape, not as a controlled superiority claim, because the comparison does not isolate architecture, tokenizer, pretraining data, or task-adaptation protocol. The observed gap should therefore be read descriptively rather than causally.

#### 3.5 Biomedical question answering on BioASQ

BioASQ yes/no question answering complements the MIMIC experiments because it tests literatureoriented biomedical reasoning rather than clinical document generation. Downstream performance after supervised fine-tuning, together with the corresponding Mamba2+SFT baseline accuracies and

- Table 11: Comparison of BioMamba with external biomedical and medically adapted language models on MIMIC-IV clinical generation (ROUGE %, 500 test samples). Comp = note completion; Disch = discharge summary generation; R-1/2/L = Rouge-1/2/L.

Note completion Discharge generation R-1 ↑ R-2 ↑ R-L ↑ R-1 ↑ R-2 ↑ R-L ↑

Model Architecture Params

BioGPT GPT-2 347M 1.20 0.16 1.01 1.53 0.09 1.26 BioMedLM GPT-2 2.7B 1.58 0.16 1.17 2.18 0.13 1.59 BioGPT-Large GPT-2 1.5B 3.85 0.54 2.81 3.20 0.20 2.42 Gemma3-1B (fine-tuned) Gemma3 1B 4.18 0.65 2.83 5.55 0.42 3.77 BioGPT-Large-PubMedQA GPT-2 1.5B 4.75 0.75 3.04 5.28 0.41 3.30 Bio-Medical-Llama-3.2-1B Llama 3.2 1B 4.80 0.82 3.19 5.58 0.43 3.70 Meditron3-Gemma2-2B Gemma2 2B 5.13 0.89 3.37 6.04 0.55 3.88

BioMamba-130M+SFT Mamba2 130M 7.07 2.45 5.10 9.74 3.33 6.96 BioMamba-1.3B+SFT Mamba2 1.3B 8.11 3.33 5.89 10.11 3.69 7.04

gains, is summarized in Table 12. BioASQ gains from continued pretraining are scale-dependent and non-monotonic: the 370M row is slightly below the Mamba2+SFT baseline, 130M, 780M, and 1.3B are positive, and 2.7B ties the baseline. Because the BioASQ yes/no test split contains only 82 items, a single-question difference corresponds to 1.22 percentage points, which matches the magnitude of the 370M and 2.7B deltas; a 3-seed reproduction across all five scales (seeds {13,42,97}; Supplementary Table S16) gives σ ≤ 0.021 on BioASQ and σ ≤ 0.032 on PubMedQA, and an item-level error analysis (Supplementary Table S18) shows that BioMamba fixes more BioASQ items than it breaks at every scale while most residual errors are shared between the two arms. In contrast to the non-monotonic BioASQ pattern, PubMedQA gains from continued pretraining are positive at every scale. Within the BioMamba series itself, performance improved overall with model scale: BioASQ accuracy increased from 68.29% at 130M to 90.24% at 2.7B, and the macro-F1 score increased from 0.634 to 0.890. The strongest model, BioMamba-2.7B, achieved 90.24% accuracy (74/82; 95% Wilson confidence interval, 81.9% to 95.0%) and a macro-F1 score of 0.890. Together with the internal perplexity and MIMIC findings, these results indicate that gains from biomedical language modeling transfer across biomedical literature and clinical text rather than remaining confined to perplexity.

#### 3.6 Biomedical question answering on PubMedQA

Binary PubMedQA was the third evaluation task. Relative to the corresponding Mamba2+SFT baselines, BioMamba improved PubMedQA accuracy at every scale, and performance also improved overall with model scale within the BioMamba series (Table 12). PubMedQA accuracy increased from 63.00% at 130M to 73.00% at 2.7B, and the macro-F1 score increased from 0.661 to 0.773. The strongest model, BioMamba-2.7B, achieved 73.00% accuracy (146/200; 95% confidence interval, 66.5% to 78.7%) and a macro-F1 score of 0.773.

#### 3.7 Ablation studies

Data-mixture ablation. The first ablation examined the composition of the continued pretraining corpus. As shown in Table 13, PubMed-only continued pretraining improved in-domain perplexity

- Table 12: Downstream question-answering results across model scales. M2+SFT = Mamba2+SFT; BM+SFT = BioMamba+SFT. Each M2+SFT and BM+SFT cell reports raw correct/total counts together with accuracy. Gain = BM+SFT − M2+SFT in percentage points; F1 is the BM+SFT macro-F1. Test-set sizes: BioASQ n = 82; PubMedQA n = 200. The corresponding 95% Wilson binomial confidence intervals for the BM+SFT cells are in Supplementary Table S17; three-seed reproduction means ± standard deviations are in Supplementary Table S16.

BioASQ PubMedQA Scale M2+SFT BM+SFT Gain F1 M2+SFT BM+SFT Gain F1 130M

44/82 (53.66%)

56/82 (68.29%)

112/200 (56.00%)

126/200 (63.00%)

+7.00 0.661 370M

+14.63 0.634

65/82 (79.27%)

64/82 (78.05%) −1.22 0.721

131/200 (65.50%)

132/200 (66.00%)

+0.50 0.690 780M

64/82 (78.05%)

65/82 (79.27%)

129/200 (64.50%)

135/200 (67.50%)

+1.22 0.778

+3.00 0.711

- 1.3B

68/82 (82.93%)

70/82 (85.37%)

+2.44 0.831

130/200 (65.00%)

140/200 (70.00%)

+5.00 0.740

- 2.7B

74/82 (90.24%)

146/200 (73.00%)

74/82 (90.24%)

141/200 (70.50%)

+0.00 0.890

+2.50 0.773

95

90.2

90

85.4

85

Accuracy(%)

79.3

78.0

80

75

| |
|---|

73.0

68.3

70

70.0

| |
|---|

| |
|---|

65

67.5

BioASQ

66.0

| |
|---|

PubMedQA

63.0

60

130M 370M 780M 1.3B 2.7B

Model Scale

- Figure 3: Downstream question-answering accuracy as a function of model scale. Both BioASQ and PubMedQA accuracy improve consistently from 130M to 2.7B parameters.

but worsened both Wikipedia and C4 perplexity, consistent with forgetting. Most mixed-corpus settings retained the PubMed gain, but many still weakened at least one held-out benchmark. The selected 10% C4 + 10% Wikipedia setting was the most balanced configuration: it achieved the lowest PubMed perplexity, the largest Wikipedia improvement, and essentially unchanged C4 perplexity. These results support the final 80% PubMed, 10% C4, and 10% Wikipedia mixture used for the main BioMamba checkpoints. A complementary 370M PubMed-only control, reported in Supplementary Table S13, confirms that this mixture remains load-bearing at a larger scale: PubMed-only continued pretraining worsens Wikipedia-style held-out perplexity by 69% and C4 perplexity by 15%, and is also slightly worse on held-out PubMed itself (9.32 → 9.91), consistent with the 130M pilot where the 80%/10%/10% mixture already matched PubMed-only on PubMed (8.41 vs. 8.47); the gap is larger at 370M, indicating that domain-only continued pretraining incurs a larger forgetting penalty at scale even on in-domain text.

- Table 13: Full continued pretraining data-mixing ablation for the 130M pilot selection run. Negative ∆ values indicate lower perplexity than the base model.

C4 ∆ No continued pretraining 0 0 9.39 — 23.29 — 26.51 PubMed only 0 0 8.47 ↓9.8% 24.50 ↑5.2% 27.90 ↑5.2% C4 only, 5% 5 0 8.48 ↓9.7% 24.03 ↑3.2% 26.53 ↑0.1% C4 only, 10% 10 0 8.99 ↓4.2% 25.28 ↑8.5% 27.80 ↑4.9% C4 only, 20% 20 0 8.48 ↓9.7% 24.19 ↑3.9% 26.34 ↓0.7% C4 only, 30% 30 0 8.49 ↓9.6% 24.25 ↑4.1% 26.29 ↓0.8% Wikipedia only, 10% 0 10 8.48 ↓9.7% 23.79 ↑2.2% 27.07 ↑2.1% Wikipedia only, 20% 0 20 8.49 ↓9.5% 23.83 ↑2.3% 27.07 ↑2.1% Wikipedia only, 30% 0 30 8.50 ↓9.5% 23.84 ↑2.4% 27.08 ↑2.2% 5% C4 + 5% Wikipedia 5 5 8.48 ↓9.7% 23.75 ↑2.0% 26.50 ↓0.0% 5% C4 + 10% Wikipedia 5 10 8.49 ↓9.6% 23.76 ↑2.0% 26.50 ↓0.0% 10% C4 + 5% Wikipedia 10 5 8.49 ↓9.6% 23.78 ↑2.1% 26.42 ↓0.3% 10% C4 + 10% Wikipedia 10 10 8.41 ↓10.4% 18.95 ↓18.6% 26.51 0.0% 10% C4 + 20% Wikipedia 10 20 8.50 ↓9.5% 23.76 ↑2.0% 26.45 ↓0.2% 15% C4 + 15% Wikipedia 15 15 8.50 ↓9.5% 23.77 ↑2.1% 26.39 ↓0.4% 20% C4 + 10% Wikipedia 20 10 8.49 ↓9.5% 23.75 ↑2.0% 26.34 ↓0.6% 20% C4 + 20% Wikipedia 20 20 8.50 ↓9.5% 23.70 ↑1.8% 26.36 ↓0.6%

Configuration C4 (%)

Wikipedia (%)

PubMed PPL

PubMed ∆

Wikipedia PPL

Wikipedia ∆

C4 PPL

Continued pretraining ablation before supervised fine-tuning. The second ablation examined the role of continued pretraining before supervised fine-tuning. At 130M (Table 12), continued pretraining increased BioASQ accuracy from 53.66% (44/82; 95% confidence interval, 42.9% to 64.0%) to 68.29% (56/82; 95% confidence interval, 57.6% to 77.4%), with macro-F1 rising from 0.536 to 0.634; PubMedQA accuracy increased from 56.00% (112/200; 95% confidence interval, 49.1% to 62.7%) to 63.00% (126/200; 95% confidence interval, 56.1% to 69.4%), with macro-F1 from 0.570 to 0.661. The class-specific pattern differs between benchmarks: on BioASQ the yes F1 rose from 0.513 to

- 0.768 while the no F1 dropped from 0.558 to 0.500, indicating that in the smallest model biomedical pretraining mainly strengthened recognition of affirmative evidence patterns; on PubMedQA both yes and no F1 improved (0.667→0.727 and 0.472→0.595), consistent with a more symmetric benefit (Supplementary Table S6).

###### PubMed PPL %

###### Wikipedia PPL %

###### C4 PPL %

[Figure 1]

[Figure 2]

[Figure 3]

- -9.8 -9.7 -9.7 -9.7

-9.7 -9.6

- -9.7 -9.6

|-10.4|
|---|

-9.5

-9.5

- -9.5 -9.5 -9.5

+5.2 +3.2 +8.5 +3.9

+5.2 +0.1 +4.9 -0.7

0%

+2.0 +2.1

0.0 -0.3

5%

Wikipedia%

- +2.2 +2.0 -18.6 +2.0

+2.1

- +2.3 +2.0 +1.8

+2.1 0.0 0.0 -0.2

10%

-0.4

15%

+2.1 -0.2 -0.6

20%

0% 5% 10% 15% 20%

0% 5% 10% 15% 20%

0% 5% 10% 15% 20%

C4 %

C4 %

C4 %

- Figure 4: Data-mixture ablation heatmap showing the perplexity change (%) relative to the base model for PubMed, Wikipedia, and C4. Each cell corresponds to a C4/Wikipedia mixture ratio; the red box highlights the selected 10%/10% configuration.

Across scales (Table 12), on PubMedQA continued pretraining improved accuracy at every scale, with gains ranging from 0.50 to 7.00 percentage points. On BioASQ, the gain was large at 130M, modest at 780M and 1.3B, neutral at 2.7B, and slightly negative at 370M. These results show that the value of biomedical continued pretraining after task-specific supervision depends on both model scale and downstream task.

#### 3.8 Learning curves and convergence

Optimization remained stable across all scales (Figure 5): validation loss and perplexity decreased smoothly during continued pretraining, and the best validation point occurred at the final step in every run (lowest validation losses: 2.11 / 1.90 / 1.80 / 1.72 / 1.65 at 130M / 370M / 780M /

- 1.3B / 2.7B). These patterns indicate that the three-epoch schedule was well calibrated and that the reported language-model improvements are unlikely to be artifacts of unstable optimization or selective early stopping.

### 4 Discussion

This study examined whether biomedical continued pretraining can improve Mamba language models without substantially weakening general-language-modeling fluency. The results support this goal across both biomedical literature and clinical text. Under tokenizer-controlled internal evaluation, BioMamba lowered PubMed perplexity at every model scale, improved Wikipedia-style held-out perplexity, and left C4 performance largely unchanged; a six-task out-of-domain multiple-choice suite and an evaluation-budget stability check (Section 3.2) further support preservation on the benchmarks tested. After supervised fine-tuning, the models also transferred across three downstream tasks: note completion, discharge summary generation, and biomedical yes/no question answering. Taken

Continual Pre-Training: Validation Loss

2.2

2.123

2.1

BioMamba-130m BioMamba-370m BioMamba-780m

ValidationLoss

2.0

- BioMamba-1.3b

- BioMamba-2.7b

| |
|---|

1.897

| |
|---|

1.9

| |
|---|

1.800

1.8

1.720

1.7

1.649

1.6

0 1000 2000 3000 4000 5000 6000 7000

Training Steps

(a) Validation loss.

Continual Pre-Training: Validation Perplexity

9.0

8.5

8.35

BioMamba-130m BioMamba-370m BioMamba-780m

8.0

7.5

Perplexity

- BioMamba-1.3b

- BioMamba-2.7b

7.0

| |
|---|

| |
|---|

6.66

| |
|---|

| |
|---|

| |
|---|

| |
|---|

6.5

6.05

6.0

5.59

5.5

5.20

0 1000 2000 3000 4000 5000 6000 7000

Training Steps

(b) Validation perplexity.

- Figure 5: Training dynamics during continued pretraining. All model scales converged smoothly, and the best validation point occurred at the final step of every run. The 3-epoch continued pretraining budget was chosen because validation loss had effectively plateaued by the end of training; additional training epochs produced diminishing reductions in validation loss, so the 3-epoch schedule is convergence-driven rather than a fixed-budget compromise.

together, the findings support balanced domain adaptation as a practical strategy for biomedical state space language models.

#### 4.1 Practical biomedical and clinical applications

The most immediate use of BioMamba is workflow support rather than autonomous bedside decision making. The PubMed, BioASQ, and PubMedQA results are most relevant to tasks such as literature screening, evidence extraction, abstract triage, biomedical question answering, and drafting structured evidence summaries [48, 49]. The MIMIC-IV results extend this picture to clinical documentation support, especially note completion and discharge summary generation, which are closely related to hospital-course summarization and discharge-note generation tasks studied previously [36–39, 50]. These tasks require models to handle biomedical terminology together with broader scientific or clinical language, which fits the balanced adaptation strategy used here. Recent studies of medical large language models report strong performance on clinical knowledge and medical question answering benchmarks [51–54], but discussions of clinical deployment similarly suggest that the most credible near-term role of such models is workflow support with retrieval and human review rather than autonomous clinical decision making [55, 56].

The scale analysis also has practical implications. Continued pretraining produced its largest downstream gains in the smaller models, suggesting that useful biomedical adaptation is not limited to the largest checkpoint. This is relevant for institutions that require lower inference cost, local deployment, or more reproducible fine-tuning under limited computational budgets. At the same time, the present results do not justify direct use for patient-specific diagnosis, prognosis, or treatment recommendation. BioMamba was trained mainly on PubMed-centered text and evaluated on benchmark question answering and clinical generation rather than on prospective clinical tasks. Its most suitable role is therefore as an assistive component in literature-grounded or documentationsupport pipelines with retrieval, source attribution, and expert oversight [55, 57].

#### 4.2 Related Work

Biomedical language modeling has been dominated by Transformer-based architectures, including SciBERT [3], BioBERT [2], PubMedBERT [4], BioGPT [5], SciFive [6], BioBART [7], and BioMedLM [21], which established the value of domain-specific pretraining for biomedical natural language processing. Text-to-text pretraining paradigms such as T5 [58] and link-aware literature pretraining approaches such as LinkBERT [59] further expanded the design space for biomedical language modeling. Clinical-domain models such as ClinicalBERT [8], GatorTron [9], ClinicalT5 [10], and broader EHR-oriented language models [60, 61] extended domain adaptation to clinical notes [62]. Work on hospital-course summarization and clinical text generation has also expanded, including dataset construction, model adaptation, and shared-task evaluation [36–40, 50]. Biomedical question answering has a mature benchmark ecosystem centered on BioASQ [41–45] and PubMedQA [46]. More recent open medical language models such as PMC-LLaMA [22], BioMistral [23], MEDITRON-70B [24], and Clinical Camel [63] still rely mainly on Transformer-style backbones. In parallel, structured state space models [18], Mamba [19], and state space duality refinements [20] have provided an alternative framework with favorable scaling for long inputs [64]. However, biomedical adaptation of these models remains limited, and published work has so far focused more narrowly on settings such as clinical text modeling [25].

The present study differs from much of this literature in two ways. First, the main evidence is based on tokenizer-controlled comparisons between each released base checkpoint and the corresponding checkpoint after continued pretraining. This design isolates the effect of biomedical adaptation more directly than comparisons across unrelated public models. Second, the adaptation strategy explicitly addresses the trade-off between biomedical specialization and retention of general-language-modeling fluency. That trade-off matters in biomedical literature workflows, where specialized terminology is often mixed with broader scientific language, and it has been emphasized in continued pretraining studies [26–28]. The question also relates to ongoing debate about whether generalist foundation models can replace specialized biomedical or clinical models [65, 66]. Recent benchmark studies in biomedical NLP and real-world clinical practice text [67, 68], together with cautionary few-shot results in the biomedical domain [69], suggest that conclusions depend strongly on task formulation and evaluation protocol. Recent methodological work has also argued that benchmark gains should be interpreted cautiously and connected to realistic biomedical use settings [57]. In that context, the present study aims to provide controlled and reproducible evidence on biomedical continued pretraining for Mamba-family models rather than to claim clinical readiness.

#### 4.3 New Findings

Several findings merit emphasis.

First, the mixture ratio mattered more than the raw inclusion of non-biomedical data. At 130M, PubMed-only continued pretraining worsened Wikipedia-style held-out perplexity by 5.2% and C4 perplexity by 5.2% relative to the base model, and most alternative mixtures we tested retained the PubMed gain but still weakened at least one held-out benchmark. At 370M, the PubMed-only control was worse than the 80/10/10 recipe on Wikipedia (+69%), C4 (+15%), and even held-out

PubMed itself (9.32 → 9.91; Supplementary Table S13), indicating that the forgetting penalty grows with scale and that adding general-domain data did not trade off against in-domain performance. The 80/10/10 recipe therefore appears effective because it strikes a specific balance rather than because it simply adds more data, and this balance is load-bearing beyond the original 130M pilot.

Second, the downstream effect of continued pretraining after supervised fine-tuning was jointly task- and scale-dependent: PubMedQA gains were positive at every scale, MIMIC-IV gains were positive at every scale on both completion and discharge generation, while BioASQ gains were non-monotonic and the 370M and 2.7B deltas fell within the 1.22 pp single-question resolution and the seed-level variance reported in Supplementary Table S16. A likely explanation is that smaller models benefit more from additional domain exposure, whereas larger models may already contain enough transferable knowledge for some supervised tasks—yet biomedical adaptation and scaling remained complementary, since the BioMamba+SFT advantage over Mamba2+SFT persisted at every evaluated scale even where absolute ROUGE plateaued at 1.3B on MIMIC-IV.

#### 4.4 Limitations

Several limitations should be considered when interpreting these results. The continued pretraining corpus was restricted to PubMed abstracts and did not include full-text biomedical articles, clinical narratives, electronic health records, or multimodal clinical data [70]. The downstream evaluation, although broader than yes/no question answering alone, remained limited to three task families—note completion, discharge summary generation, and yes/no question answering—and therefore does not cover the full range of biomedical language model utility. Additional tasks such as information extraction, retrieval, evidence attribution, broader summarization, and long-context reasoning were not evaluated, despite their prominence in recent biomedical and clinical benchmark suites [67, 68].

The mixture-ratio search was performed primarily in a 130M pilot study, and the selected mixture was then applied across all model scales, with a single 370M cross-scale ablation confirming that the mixture is load-bearing at larger scales. Although the cross-scale results were broadly consistent, this design does not exclude the possibility that different scales would benefit from different data proportions or training schedules.

Our comparisons with publicly released biomedical checkpoints (Tables 3 and 11) are contextual rather than controlled, as discussed in the Methods and Results sections, and should not be read as leaderboard-style performance claims. In addition, the benchmark datasets were modest in size, especially the BioASQ yes/no evaluation set, so small absolute differences should be interpreted cautiously.

A further limitation is that the study did not assess calibration, uncertainty estimation, hallucination behavior, robustness to distribution shift, or safety in real clinical workflows [71]. Our factuality analysis (Section 3) uses NLI contradiction as a proxy for source faithfulness; this does not replace a clinician audit, and the current study does not establish clinical safety, uncertainty calibration, or workflow readiness. A prospective clinician audit with a structured error taxonomy is left to future work. The evaluation scope of this study is in-domain biomedical QA and clinical generation, together with general-language-modeling fluency; broader capabilities such as retrieval-grounded reasoning,

evidence-grounded citation, structured information extraction, long-context clinical reasoning, and prospective deployment were not evaluated and are therefore hypothesized rather than demonstrated. Improvement on benchmark question answering or clinical generation does not by itself establish clinical validity or deployment readiness [57]. The present results should therefore be interpreted as evidence about language-model adaptation and transfer, not as evidence that BioMamba is suitable for autonomous clinical use.

#### 4.5 Future Work

Several directions follow directly from these limitations. An immediate next step is to expand the continued pretraining corpus to include full-text biomedical articles, clinical guidelines, trial reports, and carefully governed clinical narratives when available. It would also be useful to study whether the optimal PubMed/C4/Wikipedia balance changes with model size, training budget, or target task. More systematic ablations on data mixture, training duration, and learning-rate schedules may further refine the adaptation strategy.

Future evaluation should extend beyond the present tasks of note completion, discharge summary generation, and yes/no question answering to tasks that are closer to biomedical and clinical workflows [72], including retrieval-augmented question answering, evidence extraction, literature screening, broader summarization, uncertainty estimation, and source-grounded generation [55, 56]. It will also be important to assess robustness, calibration, and error patterns in settings with expert review. More broadly, the present results suggest that biomedical Mamba models may be most useful as components in retrieval-grounded and clinician-supervised systems rather than as standalone models.

### 5 Conclusion

BioMamba shows that biomedical domain-adaptive continued pretraining can strengthen public Mamba2 checkpoints without changing the underlying architecture. Across five model scales from 130M to 2.7B, BioMamba improves biomedical language modeling, preserves general-languagemodeling fluency through a balanced 80%/10%/10% PubMed/C4/Wikipedia training corpus, and transfers effectively to three downstream tasks: clinical note completion, discharge summary generation, and biomedical question answering. Beyond the empirical findings, our main contribution is a reproducible biomedical continued pretraining recipe together with a family of open-weight BioMamba checkpoints (130M / 370M / 780M / 1.3B / 2.7B), released publicly with the full training and evaluation code to provide the biomedical natural language processing community with a reusable resource.

### Ethical Approval

No new human-subject data were collected. This study used public text corpora and benchmark datasets together with de-identified clinical notes from MIMIC-IV-Note v2.2, accessed under Phys-

ioNet’s credentialed data use terms. All figures in this manuscript were created by the authors, who hold the copyright.

### Data Availability

Training and evaluation code, configurations, and the five BioMamba checkpoints (130M / 370M / 780M / 1.3B / 2.7B) are publicly available at https://github.com/lingyue404/BioMamba; the released checkpoints are also collected at https://huggingface.co/collections/zmzfpc/biomamba. The release is under the Apache-2.0 license, matching the upstream Mamba2 checkpoints. The continued pretraining corpora were loaded from Hugging Face as cyrilzakka/pubmed-medline (default, revision 432681e19469e93e6c42878d5f41fec400974fb8), wikimedia/wikipedia (20231101.en, revision b04c8d1ceb2f5cd4588862100d08de323dccfbaa), and allenai/c4 (en, revision 1588ec454efa1a09f29cd18ddd04fe05fc8653a2); the Wikipedia-style held-out evaluation set uses the Wikitext-103 raw split, as described in Supplementary Section S15. BioASQ and PubMedQA are publicly available from their original sources. MIMIC-IV-Note v2.2 is available to credentialed users through PhysioNet under its data-use requirements. Public checkpoint identifiers used for the contextual comparisons are listed in Supplementary Section S3, and additional reproducibility details (corpus snapshots, held-out construction, preprocessing rules, prompt templates, and model-selection criteria) are provided in Supplementary Section S15.

### Funding

Tianfan Fu is supported by Young Scientists Fund (C Class) of the National Natural Science Foundation of China (Grant No. 62506154), the Fundamental Research Funds for the Central Universities and Nanjing University International Collaboration Initiative (Grant No. 020214380129) and the “111 Center”(No. B26023). Yanbo Wang is supported by the Joint Funds of the National Natural Science Foundation of China (No. U21A20524).

### Supplementary Materials

All supplementary materials are provided as an appendix to this manuscript.

### Acknowledgments

Tianfan Fu is supported by Young Scientists Fund (C Class) of the National Natural Science Foundation of China (Grant No. 62506154), the Fundamental Research Funds for the Central Universities and Nanjing University International Collaboration Initiative (Grant No. 020214380129) and the “111 Center” (No. B26023). Yanbo Wang is supported by the Joint Funds of the National Natural Science Foundation of China (No. U21A20524). We thank the BioASQ organizers, the PubMedQA authors, and the broader biomedical natural language processing community for maintaining the

open benchmarks and open-source tooling that made these experiments possible. We also thank Liantao Ma (Peking University) and Yinghao Zhu (The University of Hong Kong) for assistance with experiments and data processing.

### References

- 1. Jin Q, Yuan Z, Xiong G, et al. Biomedical Question Answering: A Survey of Approaches and Challenges. ACM Computing Surveys (CSUR) 2021;55:1–36.
- 2. Lee J, Yoon W, Kim S, et al. BioBERT: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics 2020;36:1234–40.
- 3. Beltagy I, Lo K, and Cohan A. SciBERT: A pretrained language model for scientific text. arXiv preprint arXiv:1903.10676 2019.
- 4. Gu Y, Tinn R, Cheng H, et al. Domain-specific language model pretraining for biomedical natural language processing. ACM Transactions on Computing for Healthcare (HEALTH) 2021;3:1–23.
- 5. Luo R, Sun L, Xia Y, et al. BioGPT: generative pre-trained transformer for biomedical text generation and mining. Briefings in bioinformatics 2022;23:bbac409.
- 6. Phan L, Anibal J, Tran H, et al. SciFive: a text-to-text transformer model for biomedical literature. ArXiv 2021;abs/2106.03598.
- 7. Yuan H, Yuan Z, Gan R, Zhang J, Xie Y, and Yu S. BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model. ArXiv 2022;abs/2204.03905.
- 8. Alsentzer E, Murphy JR, Boag W, et al. Publicly Available Clinical BERT Embeddings. ArXiv 2019;abs/1904.03323.
- 9. Yang X, Pournejatian NM, Shin HC, et al. GatorTron: A Large Clinical Language Model to Unlock Patient Information from Unstructured Electronic Health Records. ArXiv 2022;abs/2203.03540.
- 10. Lu Q, Dou D, and Nguyen TH. ClinicalT5: A Generative Language Model for Clinical Text. 2022:5436–43.
- 11. Lu Y, Chen T, Hao N, Van Rechem C, Chen J, and Fu T. Uncertainty quantification and interpretability for clinical trial approval prediction. Health Data Science 2024;4:0126.
- 12. Li S, Miao D, Wu Q, et al. Federated learning in healthcare: a benchmark comparison of engineering and statistical approaches for structured data analysis. Health Data Science 2024;4:0196.
- 13. Tao Y, Li X, Yisha Z, Yang S, Zhan S, and Sun F. LitAutoScreener: Development and Validation of an Automated Literature Screening Tool in Evidence-Based Medicine Driven by Large Language Models. Health Data Science 2025;5:0322.
- 14. Zhang Y, Yao Q, Yue L, et al. Emerging drug interaction prediction enabled by a flow-based graph neural network with biomedical network. Nature Computational Science 2023;3:1023–33.

- 15. Rahujo A, Atif D, Inam SA, Khan AA, and Ullah S. A survey on the applications of transfer learning to enhance the performance of large language models in healthcare systems. Discover Artificial Intelligence 2025;5:90.
- 16. Inam SA. A review of artificial intelligence for predicting climate driven infectious disease outbreaks to enhance global health resilience. Discover Public Health 2025;22:738.
- 17. Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. In: Advances in neural information processing systems. 2017:5998–6008.
- 18. Gu A, Goel K, and R’e C. Efficiently Modeling Long Sequences with Structured State Spaces. ArXiv 2021;abs/2111.00396.
- 19. Gu A and Dao T. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752 2023.
- 20. Dao T and Gu A. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060 2024.
- 21. Bolton E, Venigalla A, Yasunaga M, et al. Biomedlm: A 2.7 b parameter language model trained on biomedical text. arXiv preprint arXiv:2403.18421 2024.
- 22. Wu C, Zhang X, Zhang Y, Wang Y, and Xie W. PMC-LLaMA: Towards Building Open-source Language Models for Medicine. In: 2023.
- 23. Labrak Y, Bazoge A, Morin E, Gourraud P, Rouvier M, and Dufour R. BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains. 2024:5848–64.
- 24. Chen Z, Cano AH, Romanou A, et al. MEDITRON-70B: Scaling Medical Pretraining for Large Language Models. ArXiv 2023;abs/2311.16079.
- 25. Yang Z, Mitra A, Kwon S, and Yu H. Clinicalmamba: A generative clinical language model on longitudinal clinical notes. arXiv preprint arXiv:2403.05795 2024.
- 26. Gururangan S, Marasović A, Swayamdipta S, et al. Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks. ArXiv 2020;abs/2004.10964.
- 27. Parmar J, Satheesh S, Patwary M, Shoeybi M, and Catanzaro B. Reuse, Don’t Retrain: A Recipe for Continued Pretraining of Language Models. ArXiv 2024;abs/2407.07263.
- 28. Ke Z, Shao Y, Lin H, Xu H, Shu L, and Liu B. Adapting a Language Model While Preserving its General Knowledge. 2023:10177–88.
- 29. Black S, Biderman S, Hallahan E, et al. Gpt-neox-20b: An open-source autoregressive language model. In: Proceedings of BigScience Episode# 5–Workshop on Challenges & Perspectives in Creating Large Language Models. 2022:95–136.
- 30. Canese K and Weis S. PubMed: the bibliographic database. The NCBI handbook 2013;2.
- 31. Dodge J, Marasovic A, Ilharco G, et al. Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus. 2021:1286–305.
- 32. Merity S, Xiong C, Bradbury J, and Socher R. Pointer Sentinel Mixture Models. ArXiv 2016;abs/1609.07843.

- 33. Loshchilov I and Hutter F. Decoupled Weight Decay Regularization. 2017.
- 34. Johnson A, Pollard T, Horng S, Celi LA, and Mark R. MIMIC-IV-Note: Deidentified free-text clinical notes. PhysioNet 2023. Version 2.2.
- 35. Johnson AE, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible electronic health record dataset. Scientific data 2023;10:1.
- 36. Adams G, Alsentzer E, Ketenci M, Zucker J, and Elhadad N. What’s in a Summary? Laying the Groundwork for Advances in Hospital-Course Summarization. Proceedings of the conference. Association for Computational Linguistics. North American Chapter. Meeting 2021;2021:4794– 811.
- 37. Hartman VC and Campion T. A Day-to-Day Approach for Automating the Hospital Course Section of the Discharge Summary. AMIA ... Annual Symposium proceedings. AMIA Symposium 2022;2022:216–25.
- 38. Searle T, Ibrahim ZM, Teo J, and Dobson R. Discharge summary hospital course summarisation of in patient Electronic Health Record text with clinical concept guided deep pre-trained Transformer models. Journal of biomedical informatics 2022:104358.
- 39. Aali A, Veen DV, Arefeen Y, et al. A dataset and benchmark for hospital course summarization with adapted large language models. Journal of the American Medical Informatics Association : JAMIA 2024.
- 40. Xu J, Chen Z, Johnston A, et al. Overview of the First Shared Task on Clinical Text Generation: RRG24 and “Discharge Me!”. 2024:85–98.
- 41. Tsatsaronis G, Balikas G, Malakasiotis P, et al. An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition. BMC Bioinformatics 2015;16.
- 42. Krithara A, Nentidis A, Bougiatiotis K, and Paliouras G. BioASQ-QA: A manually curated corpus for Biomedical Question Answering. Scientific Data 2022;10.
- 43. Nentidis A, Katsimpras G, Krithara A, et al. Overview of BioASQ 2023: The eleventh BioASQ challenge on Large-Scale Biomedical Semantic Indexing and Question Answering. arXiv:2307.05131 2023.
- 44. Nentidis A, Bougiatiotis K, Krithara A, and Paliouras G. Results of the Seventh Edition of the BioASQ Challenge. In: Machine Learning and Knowledge Discovery in Databases. Ed. by Cellier P and Driessens K. Cham: Springer International Publishing, 2020:553–68.
- 45. Nentidis A, Katsimpras G, Krithara A, et al. Overview of BioASQ 2025: The Thirteenth BioASQ Challenge on Large-Scale Biomedical Semantic Indexing and Question Answering. 2025:173–98.
- 46. Jin Q, Dhingra B, Liu Z, Cohen WW, and Lu X. PubMedQA: A Dataset for Biomedical Research Question Answering. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. 2019:2567–77.
- 47. Lin CY. ROUGE: A Package for Automatic Evaluation of Summaries. In: Text Summarization Branches Out. Barcelona, Spain: Association for Computational Linguistics, 2004:74–81.

- 48. Yue L, Xing S, Chen J, and Fu T. Clinicalagent: Clinical trial multi-agent system with large language model-based reasoning. In: Proceedings of the 15th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics. 2024:1–10.
- 49. Yue L, Xing S, Chen J, and Fu T. Trialenroll: Predicting clinical trial enrollment success with deep & cross network and large language models. In: Proceedings of the 15th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics. 2024:1–9.
- 50. Small W, Austrian JS, O’Donnell L, et al. Evaluating Hospital Course Summarization by an Electronic Health Record–Based Large Language Model. JAMA Network Open 2025;8.
- 51. Peng C, Yang X, Chen A, et al. A study of generative large language model for medical research and healthcare. NPJ Digital Medicine 2023;6.
- 52. Singhal K, Azizi S, Tu T, et al. Large language models encode clinical knowledge. Nature 2022;620:172–80.
- 53. Singhal K, Tu T, Gottweis J, et al. Toward expert-level medical question answering with large language models. Nature Medicine 2025;31:943–50.
- 54. Yang K, Hong M, Zhang J, et al. ECG-LM: understanding electrocardiogram with a large language model. Health Data Science 2025;5:0221.
- 55. Artsi Y, Sorin V, Glicksberg BS, Korfiatis P, Nadkarni GN, and Klang E. Large language models in real-world clinical workflows: a systematic review of applications and implementation. Frontiers in Digital Health 2025;7:1659134.
- 56. Liu S, McCoy AB, and Wright A. Improving Large Language Model Applications in Biomedicine with Retrieval-Augmented Generation: A Systematic Review, Meta-Analysis, and Clinical Development Guidelines. Journal of the American Medical Informatics Association 2025;32:605– 15.
- 57. Agrawal M, Chen IY, Gulamali F, and Joshi S. The Evaluation Illusion of Large Language Models in Medicine. npj Digital Medicine 2025;8:600.
- 58. Raffel C, Shazeer N, Roberts A, et al. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. J. Mach. Learn. Res. 2019;21:140:1–140:67.
- 59. Yasunaga M, Leskovec J, and Liang P. LinkBERT: Pretraining Language Models with Document Links. 2022:8003–16.
- 60. Yang X, Chen A, Pournejatian NM, et al. A large language model for electronic health records. NPJ Digital Medicine 2022;5.
- 61. Kweon S, Kim J, Kim J, et al. Publicly Shareable Clinical Large Language Model Built on Synthetic Clinical Notes. 2023:5148–68.
- 62. Li S, Li X, Yu K, et al. Bridging data gaps in healthcare: a scoping review of transfer learning in structured data analysis. Health Data Science 2025;5:0321.
- 63. Toma A, Lawler P, Ba J, Krishnan RG, Rubin B, and Wang B. Clinical Camel: An Open Expert-Level Medical Language Model with Dialogue-Based Knowledge Encoding. In: 2023.

- 64. Xu B, Lu Y, Li C, et al. Smiles-mamba: Chemical mamba foundation models for drug admet prediction. arXiv preprint arXiv:2408.05696 2024.
- 65. Lehman EP, Hernandez E, Mahajan D, et al. Do We Still Need Clinical Language Models? ArXiv 2023;abs/2302.08091.
- 66. Nori H, Lee YT, Zhang S, et al. Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine. ArXiv 2023;abs/2311.16452.
- 67. Chen Q, Du J, Hu Y, et al. Benchmarking large language models for biomedical natural language processing applications and recommendations. Nature Communications 2023;16.
- 68. Wu J, Gu B, Zhou R, et al. BRIDGE: Benchmarking Large Language Models for Understanding Real-world Clinical Practice Text. ArXiv 2025;abs/2504.19467.
- 69. Moradi M, Blagec K, Haberl F, and Samwald M. GPT-3 Models are Poor Few-Shot Learners in the Biomedical Domain. ArXiv 2021;abs/2109.02555.
- 70. Ren W, Liu Z, Wu Y, et al. Moving beyond medical statistics: a systematic review on missing data handling in electronic health records. Health Data Science 2024;4:0176.
- 71. Wan Z, Guo Y, Bao S, Wang Q, and Malin BA. Evaluating sex and age biases in multimodal large language models for skin disease identification from dermatoscopic images. Health Data Science 2025;5:0256.
- 72. Agnikula KS, Balls-BerryJoyce Joy E, et al. Social and behavioral determinants of health in the era of artificial intelligence with electronic health records: a scoping review. Health Data Science 2021.

### Supplementary Materials

This supplement provides implementation details, evaluation details, exact public checkpoint sources, class-specific downstream results, additional clinical generation case studies, and supplementary tables covering the cross-scale mixture ablation, out-of-domain evaluation and evaluation-budget stability, matched-budget SFT control, 3-seed QA reproduction, BioASQ item-level error analysis, NLI factuality protocol, MIMIC output diagnostics, bootstrap confidence intervals, and reproducibility details.

### S1 Experimental Environment

All continued pretraining and language-model evaluation experiments were conducted in the environment summarized in Table S1.

### S2 Continued Pretraining Implementation Details

- Table S2 summarizes the hyperparameters shared across BioMamba continued pretraining runs, and
- Table S3 provides per-scale implementation details.

Table S1: Experimental environment for continued pretraining and evaluation. Item Version / Specification

GPU NVIDIA H100 80 GB HBM3 GPU driver 580.82.07 CUDA 12.8 Operating system Ubuntu 24.04.1 LTS (Kernel 5.14.0) Python 3.12.3 PyTorch 2.10.0+cu128 mamba-ssm 2.3.0 transformers 4.57.1 datasets 4.0.0

Table S2: Shared continued pretraining hyperparameters. Parameter Value Description

Epochs 3 Training epochs Weight decay 0.1 AdamW weight decay Maximum sequence length 1,024 Tokens per sequence Gradient clipping 1.0 Maximum gradient norm Precision bfloat16 Mixed precision training Seed 42 Random seed

Warmup-stable-decay scheduler Warmup ratio 0.10 Linear warmup Stable ratio 0.65 Constant learning-rate plateau Decay ratio 0.20 Cosine decay Minimum learning-rate ratio 0.1 Minimum learning-rate ratio

Advanced techniques Exponential moving average Enabled Decay = 0.999 Layer-wise learning-rate decay 0.90–0.95 Lower layers adapt more conservatively

Table S3: Per-scale continued pretraining configuration, combining optimization (batch, gradient accumulation, effective batch size, layer-wise learning-rate decay, maximum learning rate, total run steps) and resource usage (number of GPUs, peak memory per GPU). The base learning rate was set conservatively and scaled by model size to reduce catastrophic forgetting during continued pretraining.

Scale Batch/GPU Accum Eff. BS LR decay Max. LR Steps # GPUs Peak mem.

130M 32 8 256 0.90 1.5e-6 5,967 1× H100 ∼40 GB 370M 48 5 240 0.95 4.5e-7 6,363 1× H100 ∼76 GB 780M 32 8 256 0.95 3.0e-7 5,967 1× H100 ∼65 GB

- 1.3B 16 16 256 0.95 2.5e-7 5,967 1× H100 ∼65 GB
- 2.7B 6 8 192 0.95 2.0e-7 7,953 4× H100 ∼80 GB

### S3 Evaluation Details and Public Checkpoint Sources

The exact public checkpoints used for the contextual comparison in the main text are listed in

- Table S4. Internal evaluations followed the fixed PubMed, Wikipedia, and C4 validation sets described in the main Methods section, whereas public-checkpoint comparisons used the same raw text with each model’s native tokenizer.

Loss computation. Perplexity was computed from next-token cross-entropy with padding masked via ignore_index = −100. Non-finite losses were excluded from averaging.

- Table S4: Exact public checkpoint sources used for the contextual comparison in the main text.

Model Public checkpoint

BioGPT microsoft/biogpt BioGPT-Large microsoft/BioGPT-Large BioGPT-Large-PubMedQA microsoft/BioGPT-Large-PubMedQA BioMedLM stanford-crfm/BioMedLM Meditron3-Gemma2-2B OpenMeditron/Meditron3-Gemma2-2B Bio-Medical-Llama-3.2-1B ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025 Gemma3-1B (fine-tuned) kunjcr2/gemma3_finetune

S4 Supervised Fine-Tuning Details

- Table S5 summarizes the selected supervised fine-tuning hyperparameters for each BioMamba scale.
- Table S6 provides class-specific yes/no F1 breakdowns for BioASQ and PubMedQA across scales.

- Table S5: Supervised fine-tuning hyperparameters. “BioASQ share” denotes the proportion of BioASQ data in the training mixture.

Model Learning rate

Epochs Effective batch size

Weight decay

BioASQ share

130M 3e-5 3 128 0.01 0.65 370M 3e-5 2 64 0.01 0.70 780M 2e-5 5 64 0.03 0.65

- 1.3B 7e-6 3 32 0.03 0.70
- 2.7B 2e-5 5 32 0.01 0.65

### S5 MIMIC-IV Clinical SFT and Evaluation Details

This section provides full details for the MIMIC-IV clinical generation experiments reported in the main text.

- Table S6: Class-specific yes/no F1 for BioASQ and PubMedQA across model scales. M2+SFT

= Mamba2+SFT (the public Mamba2 base checkpoint after supervised fine-tuning); BM+SFT = BioMamba+SFT (the counterpart after continued pretraining and the same supervised fine-tuning). Accuracy and macro-F1 are reported in the main text (Table 12); this table shows only the yes/no F1 breakdown to avoid repetition.

###### Benchmark Scale Arm Yes F1 No F1

M2+SFT 0.513 0.558 BM+SFT 0.768 0.500

130M

M2+SFT 0.862 0.585 BM+SFT 0.850 0.591

370M

BioASQ 13B

M2+SFT 0.847 0.609 BM+SFT 0.835 0.721

780M

- 1.3B

M2+SFT 0.877 0.720 BM+SFT 0.893 0.769

- 2.7B

M2+SFT 0.927 0.852 BM+SFT 0.927 0.852

M2+SFT 0.667 0.472 BM+SFT 0.727 0.595

130M

M2+SFT 0.736 0.639 BM+SFT 0.746 0.633

370M

PubMedQA

M2+SFT 0.737 0.636 BM+SFT 0.754 0.667

780M

- 1.3B

M2+SFT 0.737 0.619 BM+SFT 0.782 0.697

- 2.7B

M2+SFT 0.783 0.703 BM+SFT 0.800 0.745

#### S5.1 Data source and splits

The evaluation corpus is MIMIC-IV-Note v2.2 [34], derived from the broader MIMIC-IV resource [35], and contains 331,793 de-identified discharge summaries from 145,914 unique patients (average note length: 10,551 characters). Data were split by subject_id at a 9:1 ratio (seed = 42), yielding approximately 131,323 training patients (298,614 notes) and 14,591 test patients (33,179 notes).

##### S5.2 SFT data construction Two supervised fine-tuning tasks were constructed from the discharge notes:

Note completion. The first 50% of each note serves as the instruction; the remainder is the response. Filtering requires ≥100 characters for the prefix and ≥50 characters for the continuation (max_context_chars = 3,000; max_response_chars = 2,000). Up to 20,000 samples were retained.

Discharge summary generation. Admission sections (chief complaint, history of present illness, past medical history, medications, allergies, social and family history, physical examination, and laboratory results) form the instruction. Discharge sections (hospital course, assessment and plan, discharge diagnosis, discharge medications, discharge instructions, and discharge condition) form the response. The same character limits and filters apply, yielding approximately 12,893 valid samples.

The combined dataset contains 32,893 samples (90% train / 10% validation). On average each sample contains 1,023 total tokens, of which 139 (13.6%) are supervised response tokens. Instruction masking sets the loss label of instruction tokens to −100.

#### S5.3 SFT hyperparameters

Table S7 reports the Mamba2+SFT and BioMamba+SFT configurations at 130M, 370M, 780M, and

- 1.3B after hyperparameter tuning. The 2.7B configurations for both arms were selected by a small sweep over learning rates and epochs following the same conservative pattern; the exact selected recipes and training logs are in the public code release (Supplementary Section S15).

#### S5.4 Evaluation protocol

Evaluation used 500 randomly sampled test notes per task from the patient-level test split. Both tasks generated up to 128 tokens with greedy decoding (temperature = 0). The maximum context length was 4,000 characters. ROUGE scores were computed using the rouge-score library with stemming enabled.

#### S5.5 BioMamba+SFT tuning observations

A key finding during hyperparameter tuning is that validation loss does not reliably predict ROUGE quality for BioMamba+SFT models. Lower learning rates often produce higher validation loss but better ROUGE scores. Different model sizes require different learning rates: smaller models

- Table S7: MIMIC-IV clinical SFT configuration for both arms at 130M–1.3B. Mamba2+SFT uses the baseline recipe; BioMamba+SFT uses lower learning rates and more epochs to preserve domain knowledge acquired during continued pretraining. All runs use AdamW with weight decay 0.01, cosine learning-rate schedule, bf16 precision, gradient clipping 1.0, and early stopping with patience 5; Mamba2+SFT uses 10% warmup throughout. The 2.7B recipe for both arms is in the public code release (Supplementary Section S15).

Recipe Size Batch/GPU Grad Accum

Effective BS

Learning Rate

Epochs Warmup

130M 8 4 32 2e-5 3 0.10 370M 4 8 32 2e-5 3 0.10 780M 4 8 32 2e-5 3 0.10

Mamba2+SFT

1.3B 2 16 32 2e-5 3 0.10

130M 8 4 32 1e-5 5 0.10 370M 16 2 32 1e-5 5 0.10 780M 16 2 32 7.5e-6 6 0.12

BioMamba+SFT

1.3B 16 2 32 9e-6 6 0.12

(130M–370M) perform best with 1e-5, whereas the 780M and 1.3B models require rates between 7.5e-6 and 9e-6. This suggests that BioMamba models have already internalized domain knowledge and are more sensitive to learning-rate magnitude during clinical fine-tuning.

### S6 Additional MIMIC-IV Clinical Generation Case Studies

To complement the representative main-text case studies, we present four additional qualitative examples from the 1.3B model comparing three training stages: Mamba2 (pretrained only), Mamba2+SFT (supervised fine-tuning without continued pretraining), and BioMamba+SFT (continued pretraining followed by supervised fine-tuning). All examples use greedy decoding with a maximum of 128 new tokens.

#### S6.1 Additional note completion examples

In the note completion task, the model receives the first 50% of a discharge note and generates a continuation.

Laboratory results continuation. The input ends with a partial complete blood count. Table S8 shows that the base model degenerates into repetitive entries, Mamba2+SFT produces correct but incomplete results, and BioMamba+SFT generates three distinct lab panels (basic metabolic panel, minerals, and liver function tests).

Post-operative course continuation. The input describes post-operative pain management. Table S9 shows that BioMamba+SFT exactly reproduces the standardized preadmission medication statement used in MIMIC notes and generates a well-structured medication list, achieving ROUGE-1 of 16.43 (+25% over Mamba2+SFT).

- Table S8: Additional note-completion case: laboratory results continuation. Input ends with “...WBC-6.2 RBC-3.23* Hgb-9.7* Hct-30.6* MCV-95 MCH-30.1”. Source Text R-1

Reference MCHC-31.8 RDW-15.2 Plt ___ → Glucose-70 UreaN-40* Creat-2.4* Na-145 K-4.5 Cl-115* HCO3-21* → Brief Hospital Course: ___ year old woman with IDDM, HTN, CKD stage IV. . .

—

Mamba2 MCHC-32.2* RDW-0.9*... then degenerates into repetitive meaningless entries (“BUN-14.1*” “BIL-0.9*” repeated).

3.96

Mamba2+SFT MCHC-31.8, generates WBC differential (Neuts/Lymphs/Monos), Glucose-137* UreaN-25* Creat-1.9* — correct format but missing electrolytes.

5.17

BioMamba+SFT MCHC-31.0, then generates complete BMP (Glucose-137* UreaN-24* Creat-1.6* Na-141 K-4.2 Cl-102 HCO3-26), minerals (Calcium-8.8 Phos3.5 Mg-1.9), and liver function (ALT-16 AST-20 AlkPhos-76).

8.58

- Table S9: Additional note-completion case: post-operative course continuation. Input ends with “...His pain was well controlled on minimal amounts of oxycodone. He took Colace and senna”.

Source Text R-1 R-2 Reference to prevent constipation. Incision was intact without redness

— —

or drainage. He was ambulating well. He received teaching for all of his transplant medications. Medications on Admission: 1. Atorvastatin 10 mg PO DAILY 2. Ferrous Sulfat. . .

Mamba2 “On postop day 4, he was discharged home on metoprolol tartrate, dulcolax suppository, and oxycodone” — repeats the discharge description twice and gives an incomplete medication list.

11.47 0.93

Mamba2+SFT “He was discharged home on postop day 4 in stable condition.”

13.15 4.71

→ generates reasonable admission and discharge medication lists with transplant-appropriate drugs (Cellcept, Solu-medrol).

BioMamba+SFT “He was discharged home on postop day 3 in stable condition.”

###### 16.43 9.95

→ exactly reproduces: “Medications on Admission: The Preadmission Medication list is accurate and complete.” → generates a structured medication list (Amlodipine, Aspirin, Atorvastatin, Calcitriol, Docusate, Furosemide).

#### S6.2 Additional discharge-summary generation examples

In the discharge summary generation task, the model receives structured admission sections (chief complaint, history of present illness, physical exam, labs, etc.) and generates discharge content.

Diverticulitis surgery discharge. Table S10 shows a clear difference in document planning: BioMamba+SFT spontaneously generates a structured Hospital Course narrative, the most clinically central section of a discharge summary, whereas Mamba2+SFT remains closer to laboratory-style continuation.

Table S10: Additional discharge-summary case: diverticulitis surgery discharge. Input: Service: SURGERY; Chief Complaint: Abdominal pain and failure to thrive related to diverticulitis; Procedure: Low anterior resection, mobilization of splenic flexure...

Source Text R-1 R-2 Reference DISCHARGE MEDICATIONS: 1. Acetaminophen 325–650mg

— —

PO Q4H:PRN pain 2. Aspirin 81mg PO DAILY 3. Depakote 250mg BID 4. Levothyroxine 75mcg PO DAILY 5. OxycoDONE ___ mg PO Q4H:PRN pain. . .

Mamba2 Generates repetitive meaningless lab values (“pH-7.4* O2-95% CO2-38% Hct-38.0*” repeated multiple times).

2.86 0.0

Mamba2+SFT Generates lab results in correct format, but lacks discharge medications and Hospital Course narrative.

8.14 0.91

BioMamba+SFT First generates lab results, then produces “Patient is a ___ year old woman with a history of diverticulitis who presented with abdominal pain and failure to thrive. . . ”, spontaneously creating a Hospital Course narrative that accurately summarizes the admission reason and surgical procedure.

21.55 4.35

Orthopedic discharge instructions. Table S11 illustrates the repetition problem: Mamba2+SFT falls into a loop repeating “Please have your INR checked” six or more times, whereas BioMamba+SFT generates concise, non-repetitive instructions covering four distinct categories.

#### S6.3 Summary of qualitative findings

Table S12 summarizes the capability progression across the three training stages observed in the main-text and supplementary case studies.

These additional case studies confirm that the quantitative ROUGE improvements reported in the main text reflect meaningful differences in clinical generation quality. BioMamba+SFT generates more comprehensive clinical content (multi-panel lab results, medication reconciliation, Hospital Course narrative), selects medications aligned with standard care patterns, and produces betterstructured discharge documents with less repetitive degeneration. These qualitative advantages originate from domain knowledge acquired during continued pretraining on PubMed literature, which is then effectively leveraged through clinical supervised fine-tuning.

- Table S11: Additional discharge-summary case: orthopedic discharge instructions. Input: Service: ORTHOPAEDICS; Allergies: Penicillins / Sulfa / Aspirin / Iodine / NSAIDs / Codeine / Gluten / Shellfish... (extensive allergy list). Source Text R-1 R-2

Reference HOSPITAL COURSE: ...tolerating a regular diet and feeling well... afebrile with stable vital signs... pain was adequately controlled on an oral regimen. . .

— —

Mamba2 Repeatedly generates a generic template (“Home ___, dressing changes as instructed, wound checks, staple removal at two weeks”).

6.50 0.0

Mamba2+SFT Generates DVT prophylaxis (coumadin 4 weeks) and INR monitoring, but falls into repetition (“Please have your INR checked on ___” repeated 6+ times).

11.76 3.22

BioMamba+SFT “DVT Prophylaxis: Please take coumadin for 2 weeks.” → “Resume your home medications unless otherwise directed.” → “Follow up with your PCP.” → “Please follow up with your outpatient physical therapy appointment.” — concise, wellorganized discharge instructions.

14.75 5.42

- Table S12: Qualitative capability comparison across training stages based on 1.3B model case studies.

Capability Mamba2 Mamba2+SFT BioMamba+SFT

Clinical terminology accuracy

Frequent misuse (e.g., “RRR” to all systems)

Generally correct Accurate and compre-

hensive Repetition avoidance Severe degeneration Occasional repetition Minimal repetition Lab result coverage Disorganized format Correct format, incom-

Correct format, multipanel coverage

plete coverage

Discharge document structure

Cannot generate Generates basic framework

Structured with appropriate content

Medication appropriateness

Irrelevant medications Partially appropriate Clinically standard regimens

Hospital Course narrative

Cannot generate Fragmented Coherent clinical narrative

### S7 Cross-Scale Data-Mixture Ablation at 370M

To verify that the selected 80%/10%/10% PubMed/C4/Wikipedia mixture is not specific to the 130M pilot, we additionally compared the paper recipe against a PubMed-only control at the 370M scale under matched optimizer and scheduler settings (Table S13). The PubMed-only control worsens Wikipedia-style held-out perplexity by 69% and C4 perplexity by 15% relative to the 80%/10%/10% recipe, and is also slightly worse on held-out PubMed (9.32 → 9.91). This pattern is consistent with the 130M pilot, where the 80%/10%/10% mixture already slightly outperformed PubMed-only on PubMed (8.41 vs. 8.47; Table 13); the gap is larger at 370M, indicating that domain-only continued pretraining incurs a larger forgetting penalty at scale even on in-domain text. This confirms that the mixture is load-bearing at larger scale and not an artifact of pilot selection.

Table S13: 370M cross-scale validation of the 80%/10%/10% continued pretraining mixture against a PubMed-only control. PubMed-only continued pretraining worsens Wikipedia-style held-out perplexity by +69% and C4 perplexity by +15% relative to the 80%/10%/10% recipe, and is also slightly worse on held-out PubMed (9.32 → 9.91), consistent with the 130M pilot pattern (Table 13).

370M CPT recipe C4 PPL (↓) Wikipedia PPL (↓)

PubMed PPL (↓)

80%/10%/10% (paper recipe) 18.67 16.96 9.32 PubMed only 21.45 28.72 9.91 ∆ (PubMed-only − paper) +2.78 (+15%) +11.76 (+69%) +0.59

### S8 Out-of-Domain Evaluation: Protocol and Per-Task Grid

The six out-of-domain multiple-choice benchmarks referenced in Table 4 of the main text—LAMBADA, HellaSwag, ARC-Easy, ARC-Challenge, PIQA, and OpenBookQA—were scored by continuation log-likelihood at a fixed evaluation budget, using each benchmark’s standard split (LAMBADA, HellaSwag, OpenBookQA: 5,000, 5,000, 500 test items; ARC-Easy, ARC-Challenge, PIQA: 570, 299, 1,838 items respectively). The same evaluation script was used for Mamba2 and BioMamba at every scale. Table S14 reports the full per-task accuracy grid (5 scales × 6 tasks × 2 arms, 60 cells total) that underlies the summary in Table 4 of the main text. All per-task ∆ = (BioMamba − Mamba2) lie inside ±3pp at every scale, and no task shows a systematic regression in either direction. The 370M LAMBADA row is noticeably lower than the 130M and 780M rows for both arms (Mamba2 18.70%; BioMamba 19.62%); this is a known pipeline artifact at that particular 370M checkpoint and affects the two arms identically, so the BioMamba−Mamba2 comparison (+0.92pp) is unaffected. The evaluation script is released with the training and evaluation code (Supplementary Section S15).

### S9 Matched-Budget SFT Control at 1.3B

To verify that the BioMamba+SFT advantage is not an artifact of the more conservative BioMamba SFT schedule, we additionally fine-tuned Mamba2-1.3B with the exact BioMamba SFT recipe

- Table S14: Full per-task OOD accuracy grid (%) for Mamba2 base and BioMamba across five scales and six benchmarks. Mean is the unweighted average across the six tasks. Boldface marks the arm with the higher accuracy within each (scale, task) cell; a tie is not bolded.

Scale Arm LAMBADAHellaSwag ARC-E ARC-C PIQA OBQA Mean 130M

Mamba2-130M 53.18 32.00 42.63 27.42 64.53 25.00 40.79 BioMamba-130M 51.08 31.68 41.93 28.09 64.85 25.80 40.57

370M

Mamba2-370M 18.70 38.88 46.67 27.09 69.21 27.80 38.06 BioMamba-370M 19.62 39.16 46.14 28.76 69.80 26.20 38.28

780M

Mamba2-780M 55.52 45.80 52.98 31.10 72.52 31.60 48.26 BioMamba-780M 54.52 46.00 53.51 30.10 71.93 30.40 47.74

- 1.3B

Mamba2-1.3B 51.22 49.34 59.65 35.12 74.32 30.60 50.04

- BioMamba-1.3B 53.58 49.64 60.35 34.78 74.76 30.60 50.62

2.7B

Mamba2-2.7B 63.86 52.82 62.98 35.45 76.01 34.20 54.22

- BioMamba-2.7B 63.82 53.28 63.16 34.78 76.06 34.80 54.32

reported in Table S7 (lower learning rate, more epochs, same effective batch size). Table S15 shows that the BioMamba schedule does not transfer the BioMamba gain to Mamba2; instead, it hurts Mamba2 relative to its own recipe on both completion and discharge generation. This is the expected behavior if BioMamba+SFT is leveraging information acquired during continued pretraining rather than exploiting an unequal optimization effort.

- Table S15: Matched-budget 1.3B MIMIC-IV control: Mamba2-1.3B trained with the exact BioMamba SFT recipe (Table S7). The BioMamba schedule worsens Mamba2 relative to its own recipe, which rules out unequal SFT budgets as a source of the BioMamba+SFT advantage. ROUGE %, 500 test samples, greedy decoding.

1.3B arm (recipe) Completion R1 / R-2 / R-L

Discharge R1 / R-2 / R-L

Paper Mamba2+SFT 7.93 / 3.28 / 5.76 9.99 / 3.70 / 7.04 Paper BioMamba+SFT 8.11 / 3.33 / 5.89 10.11 / 3.69 / 7.04

Mamba2 @ BioMamba recipe 7.56 / 2.77 / 5.28 9.00 / 2.58 / 6.14 ∆ vs. paper BioMamba+SFT −0.55/ − 0.56/ − 0.61 −1.11/ − 1.11/ − 0.90 ∆ vs. paper Mamba2+SFT −0.37/ − 0.51/ − 0.48 −0.99/ − 1.12/ − 0.90

### S10 Three-Seed Reproduction on BioASQ and PubMedQA

To assess the sensitivity of BioMamba+SFT QA accuracy to random seed and to interpret the single-question BioASQ deltas at 370M and 2.7B, we reproduced BioMamba+SFT at every scale across seeds {13,42,97} (Table S16). Reproduced means stay within seed noise of the paper values on BioASQ (σ ≤ 0.021) and match or exceed the paper on PubMedQA (σ ≤ 0.032). Because the BioASQ test split contains only 82 items, a single-question difference corresponds to 1.22 percentage points, which matches the magnitude of the 370M and 2.7B deltas reported in Table 12.

- Table S16: Three-seed reproduction of BioMamba+SFT accuracy on BioASQ and PubMedQA across all five scales (seeds {13,42,97}). Paper columns report the single-seed values reported in the main text. ∆ columns report the difference between the three-seed mean and the paper value. BioASQ n = 82; PubMedQA n = 200.

BioASQ PubMedQA Scale 3-seed

mean ± std

Paper ∆ (pp) 3-seed mean ± std

Paper ∆ (pp)

130M 0.683 ± 0.000 0.683 +0.00 0.630 ± 0.003 0.630 +0.00 370M 0.780 ± 0.012 0.780 +0.00 0.685 ± 0.013 0.660 +2.50 780M 0.817 ± 0.021 0.793 +2.44 0.678 ± 0.023 0.675 +0.30

- 1.3B 0.841 ± 0.012 0.854 −1.22 0.713 ± 0.032 0.700 +1.30
- 2.7B 0.898 ± 0.007 0.902 −0.40 0.730 ± 0.012 0.730 +0.00

- Table S17: Raw correct/total counts and 95% Wilson binomial confidence intervals for the BioMamba+SFT accuracy values reported in Table 12. BioASQ test-set size n = 82; PubMedQA test-set size n = 200. Confidence-interval endpoints are in percentage points.

BioASQ PubMedQA Scale Correct / to-

95% Wilson CI Correct / total (acc.)

95% Wilson CI

tal (acc.)

130M 56/82 (68.29%) [57.6, 77.4] 126/200 (63.00%) [56.1, 69.4] 370M 64/82 (78.05%) [67.9, 85.6] 132/200 (66.00%) [59.2, 72.2] 780M 65/82 (79.27%) [69.3, 86.6] 135/200 (67.50%) [60.7, 73.6]

- 1.3B 70/82 (85.37%) [76.1, 91.4] 140/200 (70.00%) [63.3, 75.9]
- 2.7B 74/82 (90.24%) [81.9, 95.0] 146/200 (73.00%) [66.5, 78.7]

### S11 Item-Level Error Analysis on BioASQ

To complement the variance analysis, we ran a per-example wrong-index analysis comparing Mamba2+SFT and BioMamba+SFT on the BioASQ test split at every scale (Table S18). At every scale, BioMamba fixes at least as many items as it breaks: the fixes/breaks ratios are 9:5 at 130M, 6:5 at 370M, 2:0 at 780M, 4:1 at 1.3B, and 5:3 at 2.7B. Most errors that remain are shared between the two arms—at 780M, 100% of BioMamba’s errors are also made by Mamba2, and at

- 1.3B the share is 11 of 12—so residual disagreements at larger scales are confined to a handful of genuinely hard items rather than indicating a regression attributable to continued pretraining.

Table S18: BioASQ item-level wrong-index analysis across five scales, computed on the three-seed reproduction runs (Supplementary Table S16) rather than the single-seed values in Table 12. “Shared wrong” counts items that both arms get wrong; “BM fixes” counts items Mamba2+SFT gets wrong but BioMamba+SFT gets correct; “BM breaks” counts items Mamba2+SFT gets correct but BioMamba+SFT gets wrong; fixes minus breaks therefore corresponds to the three-seed accuracy delta (in items on the 82-item test split), not to the paper’s single-seed ∆.

Scale Shared wrong BM fixes BM breaks BM errors shared with M2

130M 22 9 5 22 / 27 (81%) 370M 15 6 5 15 / 20 (75%) 780M 17 2 0 17 / 17 (100%)

- 1.3B 11 4 1 11 / 12 (92%)
- 2.7B 5 5 3 5 / 8 (63%)

### S12 NLI Factuality Protocol and Auto-Flagged Candidate List

The NLI-ensemble factuality proxy reported in the main text (Table 10) uses two MNLI checkpoints (roberta-large-mnli and microsoft/deberta-v2-xlarge-mnli), applied across 500 2.7B MIMIC discharge generations per arm and per decoding setting. For each (generation, source) pair we scored entailment, neutral, and contradiction probabilities under each MNLI checkpoint. A generation is flagged as contradictory only when both checkpoints independently assign contradiction probability > 0.7 (unanimous-agreement rule, equivalent to an AND over the two scorers); this is the tiebreak-free specialization of plain majority voting for a two-model ensemble and is conservative, since contradiction is the minority class. Alternative rules—e.g. averaging the two contradiction probabilities and thresholding at 0.7, or OR-agreement with any one checkpoint flagging contradictionyield higher baseline contradiction rates but do not change the relative ordering of the two arms at the decoding settings tested. The source passed to the NLI scorers is the input admission text only—chief complaint, history of present illness, past medical history, medications, allergies, social and family history, physical examination, and laboratory/imaging summaries—i.e. the same sections used as the instruction for discharge summary generation. Discharge-section text (hospital course, discharge diagnosis, discharge medications, discharge instructions) is never passed to the scorer as

source, so this is an input-grounding check rather than a comparison against the reference discharge. The hypothesis is the model’s 128-token continuation.

We additionally exported a clinician-review candidate list, in which auto-flagged sentences are organized into four categories—factual (verifiable clinical content contradicted by the source), omission (important source content missing from the generation), fabrication (plausible-looking content with no source support), and safety (medication- or dosage-related generations). This list is a proxy for factuality review rather than a clinician audit, and the auto-flagging pipeline is intended as a precursor to clinician involvement rather than a replacement for it. A copy of the list is available from the authors upon request.

### S13 MIMIC-IV Discharge Section-Header F1 (1.3B and 2.7B)

The lightweight section-header F1 diagnostic (Table S19) is a structural rather than factual proxy, measuring how often generations contain the discharge section headers (Hospital Course, Discharge Medications, Discharge Instructions, etc.) that a reference discharge note contains. We report this at 1.3B and 2.7B under greedy and both nucleus settings; BioMamba leads or ties Mamba2 across all six (scale × decoding) cells tested. The main-text diagnostic table (Table 9) already covers distinct-n and repeated-4-gram rate at 1.3B; greedy is omitted there because under argmax both arms collapse to similar low-diversity outputs, making these metrics uninformative between arms.

Table S19: Discharge section-header F1 (a lightweight structure diagnostic, not a factuality metric). BioMamba leads or ties Mamba2 across all six tested (scale × decoding) cells.

Scale Decoding M2+SFT F1 BM+SFT F1 ∆ (BM − M2)

- 1.3B greedy 0.285 0.293 +0.008

- 1.3B nucleus-0.9 0.259 0.260 +0.001

- 1.3B nucleus-0.95 0.190 0.235 +0.045
- 2.7B greedy 0.168 0.196 +0.028

- 2.7B nucleus-0.9 0.170 0.188 +0.019

- 2.7B nucleus-0.95 0.152 0.158 +0.006

### S14 Bootstrap Confidence Intervals and Paired Bootstrap at 2.7B

For the 2.7B MIMIC-IV row in Table 6, we report two complementary bootstrap analyses with 1000 resamples each over the 500 test samples. Table S20 shows 95% percentile-bootstrap confidence intervals for per-arm ROUGE-1 and ROUGE-L; Table S21 shows paired-bootstrap significance on per-sample ROUGE-1 differences between BioMamba+SFT and Mamba2+SFT. On both tasks, BioMamba-2.7B+SFT lies above Mamba2-2.7B+SFT on ROUGE-1 and ROUGE-L (Table S20), and the paired bootstrap gives P(∆ > 0) = 1.00 for completion and P(∆ > 0) = 0.997 for discharge (Table S21), indicating that the 2.7B advantage is not within test-sample noise. The per-arm absolute

CIs in Table S20 overlap between the two arms; the paired-bootstrap analysis in Table S21 is therefore the more informative significance test because it exploits within-sample pairing.

- Table S20: Per-arm 95% percentile-bootstrap confidence intervals for ROUGE-1 and ROUGE-L at

- 2.7B on MIMIC-IV clinical generation (500 test samples, 1000 resamples; greedy decoding).

- 2.7B arm Task R-1 R-1 95% CI R-L R-L 95% CI

Mamba2+SFT completion 7.16 [6.72, 7.63] 4.98 [4.72, 5.30] BioMamba+SFT completion 7.57 [7.16, 8.02] 5.30 [4.97, 5.62]

Mamba2+SFT discharge 8.87 [8.17, 9.63] 6.21 [5.66, 6.76] BioMamba+SFT discharge 9.39 [8.69, 10.09] 6.46 [5.95, 7.07]

- Table S21: Paired-bootstrap significance of the 2.7B BioMamba+SFT advantage over Mamba2+SFT on per-sample ROUGE-1 differences (500 test samples, 1000 resamples; greedy decoding).

Task Mean ∆ R-1 95% CI P(∆ > 0)

Note completion +0.41 [+0.22, +0.60] 1.00 Discharge generation +0.52 [+0.14, +0.88] 0.997

### S15 Reproducibility Details

This section provides additional reproducibility information.

Base-model identifiers. The five base Mamba2 checkpoints used for continued pretraining are state-spaces/mamba2-130m, state-spaces/mamba2-370m, state-spaces/mamba2-780m, state-spaces/mamba2-1.3b, and state-spaces/mamba2-2.7b. All internal comparisons reuse the shared GPT-NeoX tokenizer from state-spaces/mamba-2.8b-hf (vocabulary size 50,280).

Corpus snapshots. The continued pretraining corpus was loaded from Hugging Face with fixed repository, config, and revision identifiers: PubMed/MEDLINE from cyrilzakka/pubmed-medline (default, revision 432681e19469e93e6c42878d5f41fec400974fb8; the dataset card records date accessed as 2025-05-02), Wikipedia from wikimedia/wikipedia (20231101.en, revision b04c8d1ceb2f5cd4588862100d08de323dccfbaa), and C4 from allenai/c4 (en, revision 1588ec454efa1a09f29cd18ddd04fe05fc8653a2). All three sources were tokenized with the shared tokenizer and filtered to remove near-duplicates and sequences with unreadable characters before truncation/padding to 1,024 tokens. For the fixed held-out perplexity sets, we sampled 1,000 sequences per source with seed 42 from the same PubMed and C4 pools used for continued pretraining but excluded them from training; for the Wikipedia-style held-out evaluation we used the Wikitext-103 raw split (wikitext-103-raw-v1), as described in Section 2.

MIMIC-IV preprocessing and splits. MIMIC-IV-Note v2.2 was split by subject_id (9:1 ratio, seed = 42), yielding approximately 131,323 training patients and 14,591 test patients; quantitative

evaluation used 500 randomly sampled held-out notes per task. Preprocessing filters (minimum lengths, maximum characters) are reported in Supplementary Section S5.

Model-selection criteria. For continued pretraining, the checkpoint with the lowest validation loss was kept for downstream evaluation. For BioMamba+SFT and Mamba2+SFT, the checkpoint with the best downstream ROUGE on the 10% validation split was selected, because validation loss was not a reliable proxy for ROUGE in the BioMamba setting (Section 2; Supplementary Section S4). For QA fine-tuning, the checkpoint with the best validation accuracy was kept.

Prompt templates. Figure S1 shows the exact prompt templates used for yes/no QA fine-tuning, note completion, and discharge summary generation.

Yes/No QA (BioASQ / PubMedQA).

Question: <question> Context: <context> Answer (yes or no):

###### Note completion (MIMIC-IV).

Complete the following discharge note. Beginning of note: <first 50% of the discharge note> Continuation: Discharge summary generation (MIMIC-IV).

Generate the discharge sections for the following admission. Admission sections: <chief complaint, HPI, PMH, medications, allergies,

social/family history, physical exam, labs> Discharge sections:

Figure S1: Prompt templates used for BioASQ/PubMedQA yes/no question-answering fine-tuning, MIMIC-IV note completion, and MIMIC-IV discharge summary generation. Instruction masking was applied during training so that only response tokens contributed to the loss.

Public release. The five BioMamba checkpoints (130M / 370M / 780M / 1.3B / 2.7B) and the full training and evaluation code are publicly available at https://github.com/lingyue404/ BioMamba; the released checkpoints are also collected at https://huggingface.co/collections/ zmzfpc/biomamba. The release is under the Apache-2.0 license, matching the upstream Mamba2 checkpoints. The repository includes training configurations, preprocessing scripts, and evaluation scripts.

