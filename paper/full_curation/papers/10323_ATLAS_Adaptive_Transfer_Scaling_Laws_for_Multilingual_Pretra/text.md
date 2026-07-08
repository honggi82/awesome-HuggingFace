## arXiv:2510.22037v2[cs.CL]25Feb2026

[Figure 1]

ATLAS : ADAPTIVE TRANSFER SCALING LAWS FOR MULTILINGUAL PRETRAINING, FINETUNING, AND DECODING THE CURSE OF MULTILINGUALITY

Shayne Longpre∗1 Sneha Kudugunta2 Niklas Muennighoff3 I-Hung Hsu4 Isaac Caswell5 Alex Pentland1 Sercan Ö. Arık4 Chen-Yu Lee4 Sayna Ebrahimi5

1MIT 2University of Washington 3Stanford University 4Google Cloud AI 5Google DeepMind

ABSTRACT

Scaling laws research has focused overwhelmingly on English—yet the most prominent AI models explicitly serve billions of international users. In this work, we undertake the largest multilingual scaling laws study to date, totaling 774 multilingual training experiments, spanning 10M-8B model parameters, 400+ training languages and 48 evaluation languages. We introduce the ADAPTIVE TRANSFER SCALING LAW (ATLAS) for both monolingual and multilingual pretraining, which outperforms existing scaling laws’ out-of-sample generalization often by more than 0.3 R2. Our analyses of the experiments shed light on multilingual learning dynamics, transfer properties between languages, and the curse of multilinguality. First, we derive a cross-lingual transfer matrix, empirically measuring mutual benefit scores between 38 × 38 = 1444 language pairs. Second, we derive a language-agnostic scaling law that reveals how to optimally scale model size and data when adding languages without sacrificing performance. Third, we identify the computational crossover points for when to pretrain from scratch versus finetune from multilingual checkpoints. We hope these findings provide the scientific foundation for democratizing scaling laws across languages, and enable practitioners to efficiently scale models—beyond English-first AI.

1 INTRODUCTION

Scaling laws research has focused overwhelmingly on English (Kaplan et al., 2020; Hoffmann et al., 2022; Li et al., 2025a). However, most of the major closed and open models now explicitly target massively multilingual uses (OpenAI, 2025; Anthropic, 2025; Google DeepMind, 2025; DeepSeekAI, 2024; Team OLMo et al., 2024). Nonetheless, multilingual scaling laws research in the public sphere is limited. Among proprietary lab reports, only Llama-3 briefly discuss their multilingual scaling laws, though they train on only 8% non-English tokens (Dubey et al., 2024). Prominent public work investigates scaling laws for data mixing (Goyal et al., 2024; Ye et al., 2024; Ge et al., 2024a), scaling laws for machine translation (Fernandes et al., 2023; Gordon et al., 2021), multilingual instruction tuning/adaptation (Shaham et al., 2024; Lai et al., 2024; Weber et al., 2024), and only a couple of recent rigorous contributions investigate scaling for smaller multilingual models (<45M and <1.2B respectively) (Chang et al., 2024; He et al., 2024).

Extending these prior works, we seek to fill the ample remaining knowledge gaps with comprehensive examinations of the following research questions. First, we explore how the properties of different languages’ scaling laws differ. Second, we measure the cross-lingual transfer benefits between 38 languages. To our knowledge, this represents the most comprehensive available empirical resource to explicitly measure language transfer across 38 × 38 = 1444 language pairs. Third, we succeed in modeling the curse of multilinguality—the phenomena where adding languages to the training mixture can degrade loss for each language, due to limited model capacity. Lastly, we measure when it is more efficient to pretrain from scratch, or finetune from a general-purpose multilingual checkpoint, resolving an unaddressed practical question. In pursuing these research ques-

∗Correspondence: slongpre@media.mit.edu

###### EN

###### FR RU

10B

ModelSize(N)

1B

100M

Mono-Mono Multi-Mono Multi-Unimax

10M

1M

###### ZH

###### HI

SW

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

10B

ModelSize(N)

1B

100M

10M

1M

0.1B 1B 10B 100B 1T

0.1B 1B 10B 100B 1T

0.1B 1B 10B 100B 1T

Data Size (D)

Data Size (D)

Data Size (D)

- Figure 1: Optimal Scaling Trajectories for English, French, Russian, Chinese, Hindi, and Swahili. The law for [monolingual vocabulary, monolingual training]=(—), the law for [multilingual vocabulary, monolingual training]=(- - -), and the law for [multilingual vocabulary, unimax training]=(···). We find (1) per-language optimal scaling trajectories are similar, (2) there is a compute efficiency tax for training with multilingual vocabularies or training sets (especially for English), and (3) as Hindi and Swahili observe data repetition their curves slope upward from diminishing returns.

tions, we develop new tools and estimates for multilingual practitioners, as well as the ADAPTIVE TRANSFER SCALING LAW, which significantly outperforms prior work across several dimensions of generalization. In total, our pretraining and finetuning experiments span 774 independent experiments on MADLAD-400 (Kudugunta et al., 2024), for models sized 10M − 2B, and evaluating 48 languages. Our main contributions include:

- 1. The ADAPTIVE TRANSFER SCALING LAW that offers better multilingual generalization to larger/unseen N,D,C, and training mixes M, than prior work (Table 1). In Figure 1 we also estimate a compute efficiency tax between multilingual and monolingual training.
- 2. A 38 × 38 CROSS-LINGUAL TRANSFER MATRIX, providing the largest resource for empirically measured transfer benefits/interference between languages (Figure 2).
- 3. A scaling law for the curse of multilinguality, that informs practitioners how much to scale (N,D) to accommodate expansions in a models’ language coverage (Figure 5).
- 4. A general pretrain vs finetune formula, that informs practitioners whether it is more efficient to pretrain from scratch or begin from a multilingual Unimax checkpoint (Figure 7).

- 2 EXPERIMENTAL SETUP

Datasets and Evaluation We use the MADLAD-400 dataset (Kudugunta et al., 2024), a popular CommonCrawl-based dataset with the most expansive coverage of languages, totaling over 400. The MADLAD-400 authors prioritized multilingual-specific curation for this pretraining corpus, including language detection, filtering, and preprocessing. For 50 languages, chosen to represent a range of language families, scripts, and degrees of resourcefulness, we partition a random test set, to evaluate vocabulary-insensitive loss (Tao et al., 2024) for fairer cross-lingual comparisons. Following (Muennighoff et al., 2024; He et al., 2024), we evaluate the fit of our parametric scaling laws using R2 calculated on held-out test sets partitioned to assess specific dimensions of generalization: R2(N) for the largest model sizes, R2(D) for the largest token ranges, R2(C) for the largest compute runs, and R2(M) for unique training language mixtures not seen at training (more details in Subsection B.3). We separate these dimensions in response to prior work that has demonstrated the importance of rigorous test-set splits in scaling laws research (Li et al., 2025a).

Model Training We pretrain models from 10M to 8B parameters, using hyperparameter choices similar to those trained in Kudugunta et al. (2024) (with updates as prescribed by recent work). In this work, we train both monolingual and multilingual models, varying the model size N, training tokens D, multilingual data proportions, and the total training compute C. In particular, we focus on

training monolingual models, bilingual models, and massively multilingual Unimax models (Chung et al., 2023). most experiments center around scale, token, and mixture variations on these languages: English, French, Chinese, Hindi, Swahili, Russian, Spanish, Portuguese, Japanese, though certain experiments evaluate across 50 languages. We use a 64k Sentence Piece Model vocabulary (Kudo, 2018). Across experiments we pretrain 280 monolingual models, 240 bilingual models, 120 multilingual mixtures, and we finetune 130 monolingual models from Unimax checkpoints, totaling over 750 independent training runs. Full experimental details, including the specific language choices, hyperparameters, and training mixtures are provided in full in Section B.

Table 1: The R2 evaluation metrics for the fitted scaling laws, holding out separate dimensions of generalization: the largest model sizes N, most training tokens D, most compute C, and unique multilingual training mixtures M. In both monolingual and multilingual settings, we average R2 across languages=[EN, FR, RU, ZH, HI, SW], including [ES, DE] for the multilingual setting. We find ADAPTIVE TRANSFER SCALING LAW outperforms prior work in Monolingual and Multilingual settings. We ablate the use of the terms for Dother and transfer languages ( K

Di).

t

SCALING LAWS R2 R2(N) R2(D) R2(C) R2(M)

CHINCHILLA SCALING LAW (Hoffmann et al., 2022) 0.94 0.68 0.94 0.90 – DATA-CONSTRAINED SCALING LAW (Muennighoff et al., 2024) 0.93 0.78 0.93 0.88 – [Ours] ATLAS (Dt ONLY) 0.92 0.88 0.91 0.88 –

M.ONO

CHINCHILLA SCALING LAW (Hoffmann et al., 2022) 0.64 -0.99 0.72 0.66 0.61 MULTILINGUAL SCALING LAW (He et al., 2024) 0.67 -0.65 0.73 0.67 0.70 [Ours] ATLAS (Dt ONLY) 0.70 -0.75 0.80 0.72 0.64 [Ours] ATLAS (Dt + Dother) 0.98 0.89 0.97 0.97 0.66 [Ours] ATLAS (Dt + Dother + K

M.ULTI

Di) 0.98 0.89 0.96 0.98 0.82

t

- 3 ADAPTIVE SCALING LAWS FOR MONOLINGUAL & MULTILINGUAL SETTINGS

Challenges with existing scaling laws for multilingual modeling. In this section, we discuss our attempts to precisely model different monolingual and multilingual pretraining experiments. Scaling laws for English are typically based on Chinchilla (Hoffmann et al., 2022)—which we refer to as the CHINCHILLA SCALING LAW (CSL). Hoffmann et al. (2022)’s two power-law terms (Equation (1)), allows us to separate how loss scales model size N and data size D. E is the irreducible loss (representing the entropy of natural text), (A, B) are learned coefficients and and (α, β) are scaling exponents for for model size and data size respectively. Beyond English, languages may have different scaling laws (Arnett & Bergen, 2025), and language resources are often limited, requiring a DATA-CONSTRAINED SCALING LAW (DCSL) (Muennighoff et al., 2024) to account for diminishing returns after multiple epochs. However, DCSL requires ample data both before and after one epoch to accommodate its two-stage fitting process. For high-resource languages (English, French, Chinese), it is costly to collect ample data beyond one epoch, and for low/mid-resource languages (even Hindi, Hebrew, and Swahili) it can be very difficult to collect sufficient observations below one epoch.1 As such, even for monolingual modeling, we require a scaling law that is data repetition-aware, and robust to different collection allowances for (N,D).

For modeling multilingual mixtures, monolingual scaling laws can be used (either with D representing the total or target language data), or He et al. (2024) introduce MULTILINGUAL SCALING LAW (MSL), which expresses the loss for target language t using N,D and the sampling ratio for the target languages’ language family in the training mixture. However, each of these solutions only accounts for one of: the sampled target language data Dt, the total data altogether Dt + Dother, or a set of the target and likely positive transfer languages Dt + i Di. We posit that a better model would separate and learn to weight these independent contributions. In summary, none of the existing multilingual solutions account for (a) multi-epoch data repetition, or (b) the cross-lingual transfer effects beyond the target’s language family.

1In MADLAD-400 these languages have < 0.3% English tokens. At standard batch sizes, even frequently evaluating (every 1k) isn’t enough to collect many (sometimes any) observations before 1 epoch is reached.

B Deffβ

A Nα

L(N,Deff) = E +

+

+ τother Sλother

#### Deff = Sλt

τi Sλi

(Dother;Uother) Other Languages

(Dt;Ut) Monolingual

#### +

(Di;Ui)

i∈K

Transfer Languages

(1)

(2)

Sλ(D;U) =

D, D ≤ U (≤ 1epoch) U 1 + 1−exp(−λλ(D/U−1)) , D > U (> 1epoch)

(3)

The ADAPTIVE TRANSFER SCALING LAW. To resolve these challenges, we introduce the ADAPTIVE TRANSFER SCALING LAW (ATLAS), a simpler variant of DCSL that is repetition-aware, introduces fewer additional parameters (for the monolingual variant), is fit in one stage, and is adaptable to an open-ended number of languages that would benefit from an explicit cross-lingual term. In Equation (1) the core scaling law formula includes the standard parameters governing the irreducible loss and N,D scaling: E,A,B,α,β. Its effective data exposure term Deff (Equation (2)) unpacks data sources into three terms: a Monolingual term for the target language data Dt, an optional Transfer Language term for up to |Kt| languages we wish to learn independent transfer coefficients for τi, and an Other Languages remainder term that sums all training tokens not already accounted for in the first two terms Dother = Dtot − Dt − K

Di. The latter two terms are optionally added for multilingual modeling, and Kt can be adapted as any combination of the languages with presumed highest positive transfer to the target language, or languages with the greatest representation

t

in the experiment training mixture. We found the latter was most effective and selected the |Kt| = 3 most highly co-sampled languages along target language t across mixtures. For each term, we apply a saturation function (Equation (3)), ensuring a smooth decay on the effective data for each subsequent epoch where the training tokens have surpassed that languages’ unique tokens U. The new parameters introduced over standard scaling laws are: the repetition parameter λ, which is shared across each data source, the transfer weights τi for each language in Kt (which are initialized from language transfer scores derived in Section 4), and the transfer weight τother for the remainder of language tokens.

Research Question: How do scaling laws differ by language, and by monolingual vs multilingual training mixtures?

Monolingual scaling behavior is consistent in form, and multilingual variants exhibit a variable-sized compute-efficiency tax. Figure 1 compares optimal scaling trajectories across sixvariable resourced languages, under three regimes: monolingual vocabulary + monolingual training, multilingual vocabulary + monolingual training, and multilingual vocabulary + unimax training. Across languages, the monolingual curves are nearly parallel, indicating similar exponents and comparable returns to additional data and parameters (see full law parameters in Table C.1). Both multilingual vocabulary and Unimax training shift the frontier upwards, evidencing a computeefficiency tax relative to the monolingual-vocabulary/monolingual-training setting. This is most pronounced for English, indicating it benefits less from language transfer than other languages do from English. For Hindi and Swahili, the right tail bends upward, consistent with diminishing returns from severe data repetition after many epochs. These qualitative trends motivate a single functional form that remains stable across languages while accounting for vocabulary/training-regime effects.

Research Question: How well can our scaling laws capture unique monolingual constraints, and complex multilingual cross-lingual transfer dynamics?

ATLAS outperforms prior scaling laws, with a more robust fit across held-out axes in monolingual and multilingual settings. Table 1 evaluates fitted laws by holding out the largest model sizes N, token ranges D, compute C = 6ND, and held-out language mixtures M, reporting R2 averaged over available languages. In the monolingual setting, all scaling laws perform comparably across most dimensions of generalization. However, when generalizing to the largest size of model, laws that model data repetition outperform those that don’t. ATLAS provides the strongest generalization to greater model sizes: R2(N) = 0.88 versus 0.68 (CSL) and 0.78 (DCSL), respectively.

In the multilingual setting, monolingual scaling laws can only observe their target data tokens, and as such perform adequately, but not well. In particular, they are unable to generalize to the largest

models R2(N) at all. This is also the case for MSL, although it obtains better generalization to unseen language mixture in R2(M) = 0.69 > 0.64 by virtue of modeling the whole language family. By separating the sources of data, ATLAS is able to significantly outperform the other laws across all dimensions, frequently achieving > 0.9 fit. However, our ablation of the data terms shows that only by adding the Transfer Language term is the model able to outperform MSL and achieve a better multilingual mixture generalization R2(M) = 0.82. In summary, ATLAS offers a simple framework to adaptively model cross-lingual transfer, and enables reliable extrapolation across out-of-sample dimensions in both monolingual and multilingual settings—addressing a critical gap where existing approaches fall short.

- 4 HOW DO LANGUAGES BENEFIT OR INTERFERE WITH EACH OTHER? Research Question: Which languages synergize or interfere most with one another’s performance?

Indo-Iranian

Vietic

Tai

Sinitic

Romance

Benue-Congo

Germanic

Japanese

Common Turkic

Slavic

Semitic

Buyeo

South-Central Dravidian

Malayo-Polynesian

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | |-0.2| |0.2| |0.2| |0.2| |0.1| |-0.1| |-0.2| |-0.1| |-0.0| |-0.1| |-0.1| |0.1| |0.2| |-0.0| |-0.1| |-0.1| |-0.1| |-0.0| |0.3| |0.3| |0.0| |0.2| |0.1| |-0.1| |0.3| |0.1| |-0.0| |-0.1| |-0.0| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.2| | | |-0.2| |-0.2| |-0.2| |-0.3| |-0.1| |-0.5| |-0.3| |-0.3| |-0.3| |-0.3| |-0.2| |-0.3| |-0.3| |-0.2| |-0.2| |-0.3| |-0.1| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.3| |-0.1| |-0.2| |0.3| |-0.3| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.1| |-0.2| | | |0.4| |0.1| |0.5| |-0.2| |-0.2| |-0.3| |-0.2| |-0.2| |-0.2| |-0.1| |-0.1| |-0.2| |-0.2| |-0.1| |-0.2| |-0.2| |-0.1| |-0.1| |0.4| |0.3| |-0.1| |-0.2| |0.0| |-0.0| |-0.2| |-0.2| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.1| |-0.2| |0.2| | | |0.1| |0.2| |-0.1| |-0.3| |-0.2| |-0.3| |-0.2| |-0.3| |-0.2| |-0.2| |-0.2| |-0.3| |-0.2| |-0.3| |-0.3| |-0.1| |-0.1| |-0.1| |0.2| |-0.1| |-0.0| |0.2| |-0.1| |-0.2| |-0.3| |-0.3| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.3| |-0.1| |0.0| | | |-0.1| |-0.2| |-0.5| |-0.3| |-0.1| |-0.2| |-0.4| |-0.1| |-0.2| |-0.3| |-0.3| |-0.2| |-0.2| |-0.3| |-0.1| |-0.2| |-0.2| |-0.0| |-0.2| |-0.3| |-0.1| |-0.1| |-0.2| |-0.3| |-0.3| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.1| |-0.2| |0.5| |0.3| |0.1| | | |-0.1| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.1| |-0.0| |0.0| |0.3| |-0.2| |-0.2| |-0.1| |-0.1| |-0.2| |-0.2| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-1.0| |-0.9| |-1.4| |-1.0| |-1.1| |-1.0| | | |-1.6| |-0.6| |-1.2| |-0.5| |-1.3| |-1.2| |-1.2| |-1.4| |-0.6| |-0.4| |-0.6| |-1.3| |-0.9| |-1.0| |-1.2| |-1.2| |-1.2| |-1.4| |-1.2| |-1.2| |-1.2| |-1.4| |-1.3| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.2| |-0.0| |0.0| |-0.0| |-0.0| |-0.0| | | |-0.2| |-0.1| |-0.2| |-0.3| |-0.2| |-0.2| |-0.1| |-0.1| |-0.1| |-0.1| |-0.3| |-0.1| |-0.0| |-0.0| |-0.0| |-0.1| |-0.3| |-0.0| |-0.1| |-0.1| |-0.2| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.6| |-1.0| |-0.7| |-0.7| |-0.8| |-0.9| |-0.1| |-1.3| | | |-0.5| |-0.5| |-1.3| |-0.9| |-0.9| |-1.3| |-0.5| |-0.4| |-0.5| |-1.2| |-0.9| |-0.8| |-0.8| |-0.8| |-0.9| |-1.3| |-0.7| |-0.9| |-0.9| |-1.3| |-1.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.7| |-0.1| |-0.6| |-0.0| |-0.1| |-0.0| |-0.1| |-0.3| |-0.1| | | |-0.1| |-0.3| |-0.1| |-0.1| |-0.3| |-0.2| |-0.1| |-0.1| |-0.3| |-0.0| |-0.0| |-0.0| |-0.0| |-0.1| |-0.3| |-0.7| |-0.1| |-0.1| |-0.3| |-0.3| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.8| |-0.9| |-1.0| |-0.8| |-0.9| |-0.8| |-0.4| |-1.1| |-0.3| |-0.5| | | |-1.0| |-0.9| |-0.8| |-0.5| |-0.5| |-0.4| |-0.4| |-0.6| |-0.9| |-0.9| |-0.8| |-0.8| |-0.9| |-0.6| |-0.9| |-0.9| |-0.9| |-0.5| |-0.5| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.2| |-0.0| |-0.0| |-0.1| |-0.0| |-0.1| |-0.3| |-0.2| |-0.1| |-0.2| | | |-0.1| |-0.1| |-0.2| |-0.1| |-0.1| |-0.2| |-0.3| |-0.0| |-0.0| |-0.0| |-0.0| |-0.2| |-0.3| |-0.0| |-0.1| |-0.2| |-0.2| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.0| |-0.3| |-0.2| |-0.2| |0.1| |-0.2| |-0.2| |-0.3| |-0.3| |-0.3| |-0.2| |-0.3| | | |0.3| |-0.3| |-0.2| |-0.2| |-0.3| |-0.3| |-0.2| |-0.2| |-0.2| |-0.2| |0.1| |-0.3| |-0.2| |-0.2| |-0.2| |-0.3| |-0.3| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.0| |-0.2| |-0.2| |-0.2| |0.1| |-0.2| |-0.2| |-0.3| |-0.2| |-0.2| |-0.2| |-0.3| |0.3| | | |-0.3| |-0.2| |-0.1| |-0.2| |-0.3| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.3| |-0.2| |-0.2| |-0.2| |-0.2| |-0.3| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.1| |-0.0| |-0.0| |-0.1| |-0.0| |-0.1| |-0.3| |-0.1| |-0.2| |-0.1| |-0.3| |-0.1| |-0.1| | | |-0.1| |-0.1| |-0.1| |-0.3| |-0.0| |-0.0| |-0.0| |-0.0| |-0.1| |-0.3| |-0.0| |-0.1| |-0.1| |-0.3| |-0.3| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-1.0| |-0.9| |-1.0| |-1.0| |-0.6| |-1.1| |-0.4| |-0.6| |-0.4| |-0.6| |-0.4| |-0.5| |-0.9| |-0.9| |-0.5| | | |-0.4| |-0.4| |-0.5| |-1.0| |-1.0| |-1.1| |-1.0| |-0.9| |-0.5| |-1.0| |-0.4| |-0.6| |-0.5| |-0.5| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-1.2| |-1.2| |-1.2| |-1.2| |-1.2| |-1.2| |-0.6| |-1.4| |-0.6| |-1.4| |-1.3| |-1.4| |-1.2| |-1.2| |-1.4| |-1.3| | | |-0.6| |-1.4| |-1.2| |-1.2| |-1.2| |-1.2| |-1.2| |-1.4| |-1.2| |-1.2| |-1.2| |-1.4| |-1.4| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-1.0| |-1.1| |-1.0| |-1.0| |-1.1| |-1.1| |-0.4| |-1.3| |-0.5| |-0.5| |-0.5| |-1.3| |-1.1| |-1.1| |-1.2| |-0.5| |-0.4| | | |-1.3| |-1.0| |-1.0| |-1.0| |-1.0| |-1.1| |-1.3| |-1.0| |-1.1| |-1.1| |-1.3| |-0.6| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.1| |-0.0| |-0.0| |-0.1| |-0.0| |-0.1| |-0.3| |-0.1| |-0.1| |-0.1| |-0.3| |-0.1| |-0.1| |-0.2| |-0.1| |-0.1| |-0.1| | | |-0.0| |-0.0| |-0.0| |-0.0| |-0.1| |-0.3| |-0.0| |-0.1| |-0.1| |-0.2| |-0.1| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.2| |-0.1| |-0.1| |-0.2| |-0.1| |-0.2| |-0.2| |-0.2| |-0.2| |-0.3| |-0.3| |-0.2| |-0.2| |-0.3| |-0.2| |-0.2| |-0.3| |-0.3| | | |0.4| |-0.1| |-0.1| |-0.2| |-0.3| |-0.1| |-0.2| |-0.2| |-0.3| |-0.3| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.2| |-0.1| |-0.1| |-0.1| |-0.1| |-0.2| |-0.3| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |0.4| | | |-0.1| |-0.1| |-0.2| |-0.2| |-0.1| |-0.1| |-0.2| |-0.2| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.2| |0.4| |0.2| |-0.1| |0.2| |-0.1| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.1| |-0.1| | | |0.1| |-0.2| |-0.2| |-0.1| |-0.1| |-0.2| |-0.2| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.2| |0.2| |0.3| |-0.1| |0.0| |-0.2| |-0.3| |-0.3| |-0.2| |-0.3| |-0.3| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.3| |-0.2| |-0.1| |-0.1| |-0.1| | | |-0.2| |-0.3| |-0.1| |-0.1| |-0.2| |-0.2| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.1| |0.1| |-0.1| |0.1| |-0.2| |-0.2| |-0.2| |-0.4| |-0.3| |-0.3| |-0.2| |-0.3| |-0.2| |-0.2| |-0.3| |-0.2| |-0.2| |-0.3| |-0.3| |-0.2| |-0.2| |-0.2| |-0.1| | | |-0.3| |-0.2| |-0.1| |-0.2| |-0.3| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.1| |-0.0| |-0.0| |-0.1| |-0.1| |-0.1| |-0.3| |-0.1| |-0.2| |-0.2| |-0.3| |-0.1| |-0.1| |-0.2| |-0.1| |-0.1| |-0.1| |-0.3| |-0.0| |-0.0| |-0.0| |-0.0| |-0.1| | | |0.2| |-0.1| |-0.1| |-0.2| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.1| |-0.1| |-0.0| |-0.0| |-0.0| |-0.2| |-0.2| |-0.2| |-0.1| |-0.2| |-0.2| |-0.1| |-0.1| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.0| |-0.0| |0.1| |-0.0| |0.1| |-0.0| | | |0.2| |-0.1| |-0.2| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.0| |-0.2| |-0.1| |-0.1| |-0.2| |-0.1| |-0.2| |-0.4| |-0.2| |-0.2| |-0.2| |-0.4| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.3| |-0.2| |-0.1| |-0.1| |-0.1| |-0.2| |-0.3| |-0.1| | | |-0.2| |-0.3| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.1| |0.3| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.4| |-0.3| |-0.3| |-0.3| |-0.3| |-0.2| |-0.3| |-0.3| |-0.3| |-0.2| |-0.3| |-0.3| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.3| |-0.2| |-0.2| | | |-0.3| |-0.3| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.1| |-0.0| |-0.0| |-0.1| |-0.0| |-0.1| |-0.3| |-0.1| |-0.1| |-0.1| |-0.3| |-0.1| |-0.1| |-0.3| |-0.2| |-0.1| |-0.1| |-0.3| |-0.0| |-0.0| |-0.0| |-0.0| |-0.1| |-0.3| |-0.0| |-0.1| |-0.1| | | |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.0| |-0.1| |-0.0| |-0.0| |-0.1| |-0.0| |-0.1| |-0.1| |-0.1| |-0.1| |-0.1| |-0.2| |0.0| |0.1| |-0.2| |-0.2| |-0.1| |-0.1| |-0.3| |0.2| |0.2| |-0.0| |-0.0| |-0.1| |-0.2| |-0.0| |0.1| |-0.0| |-0.3| | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

guenruesfrdeptzhviswkotrnosvbnpsurfilidmscaitroarhehruktethja

TargetLanguage

en ru es fr de pt zh vi ja sw ko tr no sv bn gu ps ur fil id ms ca it ro ar he hr uk te th

Source Language

- Figure 2: The CROSS-LINGUAL TRANSFER MATRIX, depicting the measured Language Transfer Score across 30 × 30 language pairs. Positive scores indicate more positive transfer, negative scores more interference, during bilingual co-training. The dashed boxes indicate the top-5 source languages for each target language. Refer to Appendix B.6 for full details, and Figure C.2 for the larger 38 × 38 matrix. While English is the best source language for many of the languages, we find language similarity is highly predictive of these scores.

When allocating multilingual training mixtures, practitioners need to know which source language(s) provide the most positive benefit, in co-training, to the target language(s) they are optimizing for. Throughout this paper, we use the terms transfer or interference to refer to the positive

Linguistic Similarity

Diff. Family & Script

0.5

Diff. Family, Same Script

Same Family & Script

Same Family, Diff. Script

Perfect Symmetry

0.0

TransferScore(LanguageBA)

0.5

1.0

1.5

1.5 1.0 0.5 0.0 0.5

Transfer Score (Language A B)

0.5

0.0

LanguageTransferScore

0.5

1.0

1.5

Same Family, Same Script

Diff. Family, Same Script

Same Family, Diff. Script

Diff. Family, Diff. Script

Linguistic Similarity

- Figure 3: Left: A language transfer scatter plot, comparing score symmetry: is language A as helpful to B as vice versa? Points cluster near the diagonal, indicating strong symmetry. The most synergistic and symmetric pairs almost exclusively share both language family and script. Greater linguistic distance correlates with increased asymmetry and reduced positive transfer. Right: The impact of linguistic similarity (via language family or script) on transfer scores. The box spans the inter-quartile range, with the median line at the center. We find the differences between each group are statistically significant (p < .001), suggesting that sharing either a language family or a script independently contribute greater positive cross-lingual transfer.

(or negative) inductive transfer between the training signal of two or more tasks (Caruana, 1997). We construct, to our knowledge, the most comprehensive matrix of language-to-language transfer scores, to assist practitioners in selecting training languages.

σbi(Lt(dmono)) − 2dmono dmono

Language Transfer Score (s → t) = BTSs→t = −

To do this, we measure how much training in a source language s affects the test loss on a target language t. We define a Bilingual Transfer Score (BTS) as the relative training efficiency of a bilingual model (s,t) = (50%,50%) compared to the monolingual model t at reaching the same loss level.2 dmono denotes a pre-defined target step (42B tokens), Lt(d) computes the monolingual models’ loss at step d, and σbi finds the number of tokens for the bilingual model to reach that loss. Because BTSs→t > 0 is the zero-centered, when it is = 0 there is no transfer, > 0 there is positive transfer, and < 0 there is negative interference, leading to the bilingual model taking more than double the steps dmono the monolingual model took to reach the target loss Lt(dmono). Refer to Subsection B.5 for full experimental details and methodology.

In Figure 2, we plot the normalized BTS scores between 30 × 30 language pairs (or the full 38 × 38 in Figure C.2), spanning language families and scripts, on 2B parameter models. Where prior work has developed comprehensive resources for measuring syntactic/phonological distances between languages (Khan et al., 2025), or measured transfer scores with smaller models specifically between high and low-resource languages (Protasov et al., 2024), our work offers among the most expansive and rigorous, fully-symmetric transfer matrices, to our knowledge. It contains many compelling observations. For instance, as one might expect, notable positive transfer exists between related languages, such as between Spanish, Catalan, and Portuguese, or Swedish and Norwegian, or Indonesian and Malay. The top-5 most helpful source languages to co-train with per target are highlighted, with English appearing the most (19 of 30 instances), followed by French (16 of 30), Spanish (13 of 30), and even Hebrew (11 of 30). In other cases, low-resource languages such as Urdu or Pashto have significant negative transfer with all other languages considered.

2We measure BTSs→t directly for 80 language pairs, then estimate it using other training signals (with high fidelity R2 = 0.85), as detailed in Subsection B.6.

0.8

0.8

0.7

0.7

RelativeScore

RelativeScore

0.6

0.6

0.5

0.5

0.4

0.4

0.3

0.3

0.2

0.2

50

50

0.1

0.1

40

40

Num Languages (K)

Num Languages (K)

0.0

0.0

30

30

0B

20

20

25B

2B

50B

10

10

4B

75B

6B

100B

ModelSize(N)

Tokens (Dtot)

0

0

8B

125B

- Figure 4: We empirically measure the relative degradation in loss (y-axis), as compared to a monolingual model of the same (N,D), from adding pretraining languages (z-axis). Left: We fix D = 25B tokens, and vary the model size N. Right: We fix N = 2B parameters, and vary the tokens D. The points are real empirical observations, averaged across languages. The mesh-grid is a surface estimate, using a cubic spline. Target language loss is most affected by the number of training languages, but this loss penalty declines for larger models with more capacity.

Language script, followed by language family are strongly correlated with positive transfer scores. In Figure 3, we correlate the transfer scores from the full language matrix Figure 2. Our analysis confirms a clear and statistically significant link between transfer performance and the similarity of the source and target language. Specifically, we found sharing a script or family introduced statistically significant shifts in the mean of the transfer score (always p < .001). These findings corroborate (He et al., 2024)’s scaling laws that grouped data by language family. This effect is strongest for shared scripts. As shown in Figure 3, language pairs that share the same writing system (e.g. Latin) exhibit dramatically improved transfer scores compared to pairs with disparate scripts (e.g., Latin and Cyrillic), with a mean score of −0.23 versus −0.39. The larger effect size for script suggests that the ability to share surface-level representations and subword vocabularies is a primary mechanism for positive transfer, more so than deeper grammatical or lexical similarities alone. In Section 5 and Subsection C.2 we explore how N,D affect language transfer.

Language transfer scores are often symmetric within the same language family and script, but surprisingly, they cannot be assumed to be reciprocal otherwise. Next, we examine the extent to which language transfer is symmetric, that is, whether the benefit from language A to B is correlated to that from B to A. As illustrated in Figure 3 (left), we find a surprising triangle pattern, with a Pearson correlation of r = −0.11. This implies two things. First, and perhaps most importantly, across all language pairs, there is no significant correlation, and because we observe language A is helpful to language B, we cannot assume the same in reverse. This corroborates findings in Li et al. (2025b) that altruistic languages don’t always yield mutual benefits. Second, there does appear to be a clear structure to the scatter: language pairs that share family and script (blue) cluster mostly in the top right quadrant and more tightly around the identity line, whereas language pairs with differing script and family (red) are simultaneously less symmetric, and less synergistic. For instance, pairs (French, Spanish) and (Russian, Ukrainian) share highly symmetric transfer scores, whereas (Chinese, Farsi) and (Russian, Vietnamese) are highly asymmetric. We hope these findings, as well as the Figure 2 transfer matrix, will be directly applicable to practitioners selecting language mixtures based on empirical measurements, rather than intuition.

- 5 THE CURSE OF MULTILINGUALITY—MODEL CAPACITY CONSTRAINTS

Research Question: Is the “curse of multilinguality” measurable—the phenomenon where adding languages to the training mixture can degrade loss of each language, due to limited model capacity?

A model’s capacity can hinder its ability to learn multiple languages at once—known as the curse of multilinguality (Conneau et al., 2019; Chang et al., 2024). We can empirically measure this relationship, between N, D, and the number of training languages K, by running experiments with variable K. Full experimental details and scaling law derivations are provided in Subsection B.7.

Modeling the Relationship between K, N, and D. In Figure 4 we examine the relative loss increase, over a monolingual baseline, from varying the number of training languages K, and either the model size N (left plot) or total training tokens Dtot (right plot). For simplicity, we don’t explicitly model token repetition, and we sample tokens from each language uniformly—we believe this is a reasonable assumption for models designed to serve all K languages. We observe three phenomena. First, the number of training languages K has far more impact on the relative loss than N or D. Relative loss appears to grow monotonically as K increases. Second, both increasing model size N and total tokens Dtot mitigate this penalty. However, computing the partial along the fitted surface shows |∂S/∂ log N| > |∂S/∂ log D|; indicating N delivers larger marginal gains than Dtot. Consequently, maintaining performance as K grows requires scaling both data and model size, but scaling the model size has a greater effect. To understand the relationship among these variables more precisely, we model per–target–language loss as

Kϕ Nα

L(K,N,Dt) = L∞ + A

Kψ Dtβ

+ B

,

where, under even sampling the total tokens across languages are Dtot = K Dt. We tested several scaling law variations for this research question, but settled on this one as (a) it retains Chinchillastyle power-law decay properties that cleanly separate model capacity from data (it also reduces to Chinchilla when K=1), (b) it makes the role of the number of languages explicit and interpretable, and (c) it achieved a robust R2 ≥ 0.87, indicating a strong fit. The exponent ϕ captures how capacity requirements grow with the number of languages, while ψ captures how data requirements change with multilinguality. ψ < 0 indicates positive transfer (sublinear data needs per language as K increases), and ψ > 0 implies negative transfer/interference.

We fit the scaling law for each of 8 different languages, as well as the combination of all, achieving similar coefficients, and > 0.8 R2 for each. Using the scaling law fit on all language data we find ϕ = 0.11 and ψ = −0.04, indicating a mild capacity-driven curse of multilinguality, tempered by positive transfer across languages. In other words, as languages are added to the training mixture, a positive ϕ means the loss will decline from limited model capacity, however, a negative ψ means that less data per language is needed. (Though the total amount of data increases, as we sample from new languages too.) This provides clear, measurable evidence of positive cross-lingual transfer.

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

Language scaling

K 2 K K 4 K K 8 K K 16 K

(N/N)Modelmultiplier

compute-optimal frontier

D =D r1+ / =D r0.728 N =N r / =N r0.243 C =C r / +1+ / =C r0.970

(x7.52, x1.96)

(x4.54, x1.66)

(x2.74, x1.4)

(x1.66, x1.18)

0 1 2 3 4 5 6 7 8

Token multiplier (D tot/Dtot)

- Figure 5: Iso-loss from expanding language coverage: When scaling the number of languages a model serves from K → r·K, we can estimate how much they need to increase model size

Estimating the Iso-Loss Frontier, and Compute-Optimal scaling of N,D when adding languages: K → rK. Next, we examine the practical case where a practitioner wants to retrain a new model, increasing the number of languages it serves from K to rK, without hindering the performance the original model achieved on the existing languages. To accommodate these new languages, they will need to scale C with some combination of N → N′ and Dtot → Dtot′ . We can compute

N′/N and/or training tokens Dtot′ /Dtot, without degrading any of their languages’ loss. We derive the compute optimal scaling equations. The iso-losses indicate that a practitioner should expand their compute budget by C · r0.97 to expand their language coverage by r.

ψ D′β

ϕ

ψ

ϕ

N′α + B(Kr)

Dβ = A(Kr)

. From this we can derive the closed-form expression for how to scale (N,Dtot) when increasing K to rK

this iso-loss by equating the loss terms: L(K,N,Dtot) = AK

Nα + BK

(as derived in Subsection B.7):

N∗(rK) N∗(K)

Dt∗(rK) D∗ t (K)

Dtot∗ (rK) D∗ tot(K)

= rϕ/α,

= rψ/β,

We can also derive the minimal increase in the compute budget:

= r1+ψ/β.

C′ C

N′Dtot′ NDtot

⋆

⋆

Dtot′ Dtot

′

= r1+ϕ/α+ψ/β .

= N

=

N

In Figure 5 we plot the iso-loss frontiers, and compute-optimal allocations, when scaling a model from K to r · K languages. It shows in closed-form how a practitioner should scale Dtot, N, and by extension their compute budget C to accommodate additional languages, without degrading their loss. For instance, we find expanding to 4 · K languages requires a practitioner to expand Dtot by 2.74, and the model size N by 1.4. Incidentally, evenly sampling across languages, this actually corresponds to using sampling 1 − (2.74/4) = 32% less data per language. Although there are fewer tokens in each target language (by 32%), the total tokens has risen by 2.74, and the positive transfer prevents any potential loss degradation.

EN

JA

PT

DE

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | |144|B| | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | |145|B| | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | |14|8B| | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | |16|0B| | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.15

0.10

1.0

Difference

0.05

Loss

0.9

0.00

0.05

0.8

0.10

###### RU

###### VI

###### ES

ZH

1.05

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | |189B| | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | |234|B| | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | |239|B| | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | |283B| | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.15

1.00

0.10

Difference

0.95

0.05

Loss

0.90

0.00

0.85

0.05

0.80

0.10

0 100 200 300 400 500

0 100 200 300 400 500

0 100 200 300 400 500

0 100 200 300 400 500

Tokens 1e9

Tokens 1e9

Tokens 1e9

Tokens 1e9

- Figure 6: For eight languages, we plot the loss curves for 2B parameter models pretraining monolingually from scratch (···), finetuning monolingually from the Unimax base model (- - -), and the difference between these losses (—). We annotate the number of tokens at which the pretrained loss becomes better than the finetuning loss. The difference between the interpolated pretraining curve and finetuning curve. When the loss difference reaches zero, pretraining from scratch has surpassed finetuning using equal compute. Depending on the token (or compute) budget, it is usually more effective to finetune a multilingual checkpoint if there are < 144B tokens, and pretrain from scratch if the budget accommodates ≥ 283B tokens.

- 6 PRETRAIN OR FINETUNE?

Research Question: When training a model for target language t, is it more effective to pretrain from scratch, or finetune a general-purpose massively multilingual checkpoint?

Prior scaling law work focuses on compute efficiency with respect to pretraining from scratch. However, in reality, practitioners who aim to optimize for a target language t may have the option to start from multilingual public checkpoints or pretrain one multilingual checkpoint to serve as the starting point for many downstream models. Consider a practitioner with a compute budget C. In Figure 6, we plot the interpolated loss curves for the model pretrained from scratch, the model finetuned from a checkpoint, and the difference between these losses, for several languages. We find that while the warm-started finetuned model performs better at first, pretraining from scratch eventually surpasses its performance after 144B − 283B tokens, depending on the language. Note that we leave the reason for these convergence differences to future work, without a clear hypothesis. Though English pretraining converges the fastest, it is also allocated a 5% sampling rate within Unimax, whereas each of the other languages are allocated 1.4%.

From the number of training tokens D we can also infer the inflection point which decides whether it is more computationally efficient (C = 6ND) to pretrain from scratch or finetune from the multilingual checkpoint. Using our range of model size experiments, we can estimate a best fit power law in Figure 7, relating N to C. We find log(C) = 1113708 × N1.65 to hold consistently across languages. For simplicity, we choose not to account for data repetition in this model. Practitioners may use this to estimate whether pretraining or finetuning will yield greater performance for their compute budget. Note one limitation to consider: not all multilingual base models are trained with the same mixture or for sufficiently longand these factors would impact the pretrain vs finetuning intersection points. We choose a widely used Unimax mixture trained for 1B tokens (Chung et al., 2023). We believe this is a

- 1018

- 1019

- 1020

- 1021

- 1022

English Russian Spanish French

ComputeIntersection(C)

C=10283128 N1.54

German

Portuguese

Vietnamese

Chinese

Japanese

Hindi

Power fit

0 1 2 3 4 Model Size (N) [×10^9]

- Figure 7: In terms of N, we model the compute budget C required for a model pretrained from scratch to outperform the model finetuned from the generic Unimax multilingual checkpoint. We estimate this relationship as C = 10283128 × N1.65.

useful heuristic, with reasonable assumptions, to determine the optimal choice between pretraining and finetuning for a given compute budget.

- 7 CONCLUSION

In this work, we answer several scaling questions on training multilingual models, validating our findings on scales up to 8B model parameters. First, we introduce a novel scaling law formulation called Adaptive Transfer Scaling Law (ATLAS) that significantly improves scaling law fit by more than 0.3 R2 for multilingual mixtures. To aid our analysis, we also derive transfer scores for 38 langauages. We find that English is often a strong positive transfer language, alongside languages with shared script and language families, though we note that this transfer is often asymmetric. However, there is still a compute-efficiency tax when we train multilingual models. We quantify this "curse of multilinguality" and provide a mathematical trade-off between model capacity and language coverage: we find that the performance penalty from adding languages is mitigated more effectively by scaling model size (N) rather than training tokens (D). Finally, we quantitatively answer a critical practical question for practitioners: determining the compute crossover point between pretraining and finetuning. We find that depending on the language, for a budget under 144-283B tokens, finetuning a general-purpose multilingual checkpoint is more efficient. Collectively, these contributions provide valuable guidelines for practitioners determining data mixtures, language coverage and computational requirements for creating multilingual foundation models.

- 8 ACKNOWLEDGEMENTS

We thank Luke Zettlemoyer, Catherine Arnett and Stella Biderman for helpful discussions on the paper. We thank Biao Zhang and Xavier Garcia for the technical discussions and feedback on early directions.

REFERENCES

Armen Aghajanyan, Lili Yu, Alexis Conneau, Wei-Ning Hsu, Karen Hambardzumyan, Susan Zhang, Stephen Roller, Naman Goyal, Omer Levy, and Luke Zettlemoyer. Scaling laws for generative mixed-modal language models. In International Conference on Machine Learning, pp. 265–279. PMLR, 2023.

Ibrahim M Alabdulmohsin, Behnam Neyshabur, and Xiaohua Zhai. Revisiting neural scaling laws in language and vision. Advances in Neural Information Processing Systems, 35:22300–22312, 2022.

Alon Albalak, Yanai Elazar, Sang Michael Xie, Shayne Longpre, Nathan Lambert, Xinyi Wang, Niklas Muennighoff, Bairu Hou, Liangming Pan, Haewon Jeong, et al. A survey on data selection for language models. arXiv preprint arXiv:2402.16827, 2024.

Dario Amodei, Sundaram Ananthanarayanan, Rishita Anubhai, Jingliang Bai, Eric Battenberg, Carl Case, Jared Casper, Bryan Catanzaro, Qiang Cheng, Guoliang Chen, et al. Deep speech 2: End-toend speech recognition in english and mandarin. In International conference on machine learning, pp. 173–182. PMLR, 2016.

Anthropic. System card: Claude opus 4 & claude sonnet 4. Technical report, Anthropic, May 2025. URL https://www-cdn.anthropic.com/ 4263b940cabb546aa0e3283f35b686f4f3b2ff47.pdf.

Newsha Ardalani, Carole-Jean Wu, Zeliang Chen, Bhargav Bhushanam, and Adnan Aziz. Understanding scaling laws for recommendation models. arXiv preprint arXiv:2208.08489, 2022.

Catherine Arnett and Benjamin Bergen. Why do language models perform worse for morphologically complex languages? In Proceedings of the 31st International Conference on Computational Linguistics, pp. 6607–6623, 2025.

Michele Banko and Eric Brill. Scaling to very very large corpora for natural language disambiguation. In Proceedings of the 39th annual meeting of the Association for Computational Linguistics, pp. 26–33, 2001.

Yamini Bansal, Behrooz Ghorbani, Ankush Garg, Biao Zhang, Colin Cherry, Behnam Neyshabur, and Orhan Firat. Data scaling laws in nmt: The effect of noise and architecture. In International Conference on Machine Learning, pp. 1466–1482. PMLR, 2022.

Ankur Bapna, Isaac Caswell, Julia Kreutzer, Orhan Firat, Daan van Esch, Aditya Siddhant, Mengmeng Niu, Pallavi Baljekar, Xavier Garcia, Wolfgang Macherey, Theresa Breiner, Vera Axelrod, Jason Riesa, Yuan Cao, Mia Xu Chen, Klaus Macherey, Maxim Krikun, Pidong Wang, Alexander Gutkin, Apurva Shah, Yanping Huang, Zhifeng Chen, Yonghui Wu, and Macduff Hughes. Building Machine Translation Systems for the Next Thousand Languages. arXiv e-prints, art. arXiv:2205.03983, May 2022.

David Brandfonbrener, Nikhil Anand, Nikhil Vyas, Eran Malach, and Sham Kakade. Loss-to-loss

prediction: Scaling laws for all datasets. arXiv preprint arXiv:2411.12925, 2024. Rich Caruana. Multitask learning. Machine learning, 28(1):41–75, 1997. Tyler Chang, Catherine Arnett, Zhuowen Tu, and Ben Bergen. When is multilinguality a curse? lan-

guage modeling for 250 high-and low-resource languages. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 4074–4096, 2024.

Yangyi Chen, Binxuan Huang, Yifan Gao, Zhengyang Wang, Jingfeng Yang, and Heng Ji. Scaling laws for predicting downstream performance in llms. arXiv preprint arXiv:2410.08527, 2024.

Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2818–2829, 2023.

Hyung Won Chung, Noah Constant, Xavier Garcia, Adam Roberts, Yi Tay, Sharan Narang, and Orhan Firat. Unimax: Fairer and more effective language sampling for large-scale multilingual pretraining. arXiv preprint arXiv:2304.09151, 2023.

Aidan Clark, Diego de Las Casas, Aurelia Guy, Arthur Mensch, Michela Paganini, Jordan Hoffmann, Bogdan Damoc, Blake Hechtman, Trevor Cai, Sebastian Borgeaud, et al. Unified scaling laws for routed language models. In International conference on machine learning, pp. 4057– 4086. PMLR, 2022.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised cross-lingual representation learning at scale. arXiv preprint arXiv:1911.02116, 2019.

DeepSeek-AI. Deepseek-v3 technical report, 2024. URL https://arxiv.org/abs/2412. 19437.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Patrick Fernandes, Behrooz Ghorbani, Xavier Garcia, Markus Freitag, and Orhan Firat. Scaling laws for multilingual neural machine translation. arXiv preprint arXiv:2302.09650, 2023.

Elias Frantar, Carlos Riquelme, Neil Houlsby, Dan Alistarh, and Utku Evci. Scaling laws for sparsely-connected foundation models. arXiv preprint arXiv:2309.08520, 2023.

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pp. 10835–10866. PMLR, 2023.

Ce Ge, Zhijian Ma, Daoyuan Chen, Yaliang Li, and Bolin Ding. Bimix: A bivariate data mixing law for language model pretraining. arXiv preprint arXiv:2405.14908, 2024a.

Ce Ge, Zhijian Ma, Daoyuan Chen, Yaliang Li, and Bolin Ding. Data mixing made efficient: A bivariate scaling law for language model pretraining. arXiv preprint arXiv:2405.14908, 2024b.

Behrooz Ghorbani, Orhan Firat, Markus Freitag, Ankur Bapna, Maxim Krikun, Xavier Garcia, Ciprian Chelba, and Colin Cherry. Scaling laws for neural machine translation. arXiv preprint arXiv:2109.07740, 2021.

Google DeepMind. Gemini 2.5 deep think model card. Technical report, Google DeepMind, August 2025. URL https://storage.googleapis.com/deepmind-media/ Model-Cards/Gemini-2-5-Deep-Think-Model-Card.pdf. Model card published/updated August 1, 2025.

Mitchell A Gordon, Kevin Duh, and Jared Kaplan. Data and parameter scaling laws for neural machine translation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 5915–5922, 2021.

Naman Goyal, Cynthia Gao, Vishrav Chaudhary, Peng-Jen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, Marc’Aurelio Ranzato, Francisco Guzmán, and Angela Fan. The flores-101 evaluation benchmark for low-resource and multilingual machine translation. 2021.

Sachin Goyal, Pratyush Maini, Zachary C Lipton, Aditi Raghunathan, and J Zico Kolter. Scaling laws for data filtering–data curation cannot be compute agnostic. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22702–22711, 2024.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

Alexander Hägele, Elie Bakouch, Atli Kosson, Loubna Ben Allal, Leandro Von Werra, and Martin Jaggi. Scaling laws and compute-optimal training beyond fixed training durations. arXiv preprint arXiv:2405.18392, 2024.

Tatsunori Hashimoto. Model performance scaling with multiple data sources. In International Conference on Machine Learning, pp. 4107–4116. PMLR, 2021.

Yifei He, Alon Benhaim, Barun Patra, Praneetha Vaddamanu, Sanchit Ahuja, Parul Chopra, Vishrav Chaudhary, Han Zhao, and Xia Song. Scaling laws for multilingual language models. arXiv preprint arXiv:2410.12883, 2024.

Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B Brown, Prafulla Dhariwal, Scott Gray, et al. Scaling laws for autoregressive generative modeling. arXiv preprint arXiv:2010.14701, 2020.

Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Diamos, Heewoo Jun, Hassan Kianinejad, Md Mostofa Ali Patwary, Yang Yang, and Yanqi Zhou. Deep learning scaling is predictable, empirically. arXiv preprint arXiv:1712.00409, 2017.

Jacob Hilton, Jie Tang, and John Schulman. Scaling laws for single-agent reinforcement learning. arXiv preprint arXiv:2301.13442, 2023.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024.

Andy L Jones. Scaling scaling laws with board games. arXiv preprint arXiv:2104.03113, 2021. Feiyang Kang, Yifan Sun, Bingbing Wen, Si Chen, Dawn Song, Rafid Mahmood, and Ruoxi Jia.

Autoscale: Scale-aware data mixing for pre-training llms. arXiv preprint arXiv:2407.20177, 2024.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. 2020.

Aditya Armaan Khan, Mason Stephen Shipton, David Anugraha, Kaiyao Duan, Phuong H Hoang, Eric Khiu, A Seza Do˘gruöz, and Annie Lee. Uriel+: Enhancing linguistic inclusion and usability in a typological and multilingual knowledge base. In Proceedings of the 31st International Conference on Computational Linguistics, pp. 6937–6952, 2025.

T Kudo. Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. arXiv preprint arXiv:1808.06226, 2018.

Sneha Kudugunta, Isaac Caswell, Biao Zhang, Xavier Garcia, Derrick Xin, Aditya Kusupati, Romi Stella, Ankur Bapna, and Orhan Firat. Madlad-400: A multilingual and document-level large audited dataset. Advances in Neural Information Processing Systems, 36, 2024.

Wen Lai, Mohsen Mesgar, and Alexander Fraser. Llms beyond english: Scaling the multilingual capability of llms with cross-lingual feedback. In Findings of the Association for Computational Linguistics ACL 2024, pp. 8186–8213, 2024.

Margaret Li, Sneha Kudugunta, and Luke Zettlemoyer. (mis) fitting: A survey of scaling laws. arXiv preprint arXiv:2502.18969, 2025a.

Zihao Li, Shaoxiong Ji, Hengyu Luo, and Jörg Tiedemann. Rethinking multilingual continual pretraining: Data mixing for adapting llms across languages and resources. arXiv preprint arXiv:2504.04152, 2025b.

Qian Liu, Xiaosen Zheng, Niklas Muennighoff, Guangtao Zeng, Longxu Dou, Tianyu Pang, Jing Jiang, and Min Lin. Regmix: Data mixture as regression for language model pre-training. arXiv preprint arXiv:2407.01492, 2024.

Shayne Longpre, Gregory Yauney, Emily Reif, Katherine Lee, Adam Roberts, Barret Zoph, Denny Zhou, Jason Wei, Kevin Robinson, David Mimno, et al. A pretrainer’s guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity. arXiv preprint arXiv:2305.13169, 2023.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id= Bkg6RiCqY7.

Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, and Colin A Raffel. Scaling data-constrained language models. Advances in Neural Information Processing Systems, 36, 2024.

OpenAI. Gpt-5 system card. Technical report, OpenAI, August 2025. URL https://cdn. openai.com/gpt-5-system-card.pdf.

Vitaly Protasov, Elisei Stakovskii, Ekaterina Voloshina, Tatiana Shavrina, and Alexander Panchenko. Super donors and super recipients: Studying cross-lingual transfer between highresource and low-resource languages. In Proceedings of the Seventh Workshop on Technologies for Machine Translation of Low-Resource Languages (LoResMT 2024), pp. 94–108, 2024.

Haoran Que, Jiaheng Liu, Ge Zhang, Chenchen Zhang, Xingwei Qu, Yinghao Ma, Feiyu Duan, Zhiqi Bai, Jiakai Wang, Yuanxing Zhang, et al. D-cpt law: Domain-specific continual pre-training scaling law for large language models. Advances in Neural Information Processing Systems, 37: 90318–90354, 2024.

Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jeanbaptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

Teven Le Scao, Thomas Wang, Daniel Hesslow, Lucile Saulnier, Stas Bekman, M Saiful Bari, Stella Biderman, Hady Elsahar, Niklas Muennighoff, Jason Phang, et al. What language model to train if you have one million gpu hours? arXiv preprint arXiv:2210.15424, 2022.

Uri Shaham, Jonathan Herzig, Roee Aharoni, Idan Szpektor, Reut Tsarfaty, and Matan Eyal. Multilingual instruction tuning with just a pinch of multilinguality. In Findings of the Association for Computational Linguistics ACL 2024, pp. 2304–2317, 2024.

Luisa Shimabucoro, Ahmet Ustun, Marzieh Fadaee, and Sebastian Ruder. A post-trainer’s guide to multilingual training data: Uncovering cross-lingual transfer dynamics. arXiv preprint arXiv:2504.16677, 2025.

Mustafa Shukor, Louis Bethune, Dan Busbridge, David Grangier, Enrico Fini, Alaaeldin El-Nouby, and Pierre Ablin. Scaling laws for optimal data mixtures. arXiv preprint arXiv:2507.09404, 2025.

Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari Morcos. Beyond neural scaling laws: beating power law scaling via data pruning. Advances in Neural Information Processing Systems, 35:19523–19536, 2022.

Chaofan Tao, Qian Liu, Longxu Dou, Niklas Muennighoff, Zhongwei Wan, Ping Luo, Min Lin, and Ngai Wong. Scaling laws with vocabulary: Larger models deserve larger vocabularies. arXiv preprint arXiv:2407.13623, 2024.

Yi Tay, Mostafa Dehghani, Samira Abnar, Hyung Won Chung, William Fedus, Jinfeng Rao, Sharan Narang, Vinh Q Tran, Dani Yogatama, and Donald Metzler. Scaling laws vs model architectures: How does inductive bias influence scaling? arXiv preprint arXiv:2207.10551, 2022.

Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, Nathan Lambert, Dustin Schwenk, Oyvind Tafjord, Taira Anderson, David Atkinson, Faeze Brahman, Christopher Clark, Pradeep Dasigi, Nouha Dziri, Michal Guerquin, Hamish Ivison, Pang Wei Koh, Jiacheng Liu, Saumya Malik, William Merrill, Lester James V. Miranda, Jacob Morrison, Tyler Murray, Crystal Nam, Valentina Pyatkin, Aman Rangapur, Michael Schmitz, Sam Skjonsberg, David Wadden, Christopher Wilhelm, Michael Wilson, Luke Zettlemoyer, Ali Farhadi, Noah A. Smith, and Hannaneh Hajishirzi. 2 olmo 2 furious, 2024. URL https://arxiv.org/abs/2501.00656.

Alexander Arno Weber, Klaudia Thellmann, Jan Ebert, Nicolas Flores-Herr, Jens Lehmann, Michael Fromm, and Mehdi Ali. Investigating multilingual instruction-tuning: Do polyglot models demand for multilingual instructions? In EMNLP, 2024.

Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy S Liang, Quoc V Le, Tengyu Ma, and Adams Wei Yu. Doremi: Optimizing data mixtures speeds up language model pretraining. Advances in Neural Information Processing Systems, 36, 2024.

Jiasheng Ye, Peiju Liu, Tianxiang Sun, Yunhua Zhou, Jun Zhan, and Xipeng Qiu. Data mixing laws: Optimizing data mixtures by predicting language modeling performance. arXiv preprint arXiv:2403.16952, 2024.

Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12104–12113, 2022.

# Appendix

### Table of Contents

- A Extended Related Work 17
- B Experimental Details 18

- B.1 Language Choice . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- B.2 Experiment Design . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- B.3 Experimental Details: Scaling Laws Evaluation . . . . . . . . . . . . . . . . . . 19
- B.4 Experimental Details: Language Finetuning . . . . . . . . . . . . . . . . . . . . 19
- B.5 Experimental Details: Language Transfer . . . . . . . . . . . . . . . . . . . . . 20
- B.6 Estimating Bilingual Transfer Score (BTS) . . . . . . . . . . . . . . . . . . . . 24
- B.7 Experimental Details: Multilingual Capacity . . . . . . . . . . . . . . . . . . . 24

- C Extended Results 27

- C.1 Extended Results: Scaling Laws . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- C.2 Extended Results: Language Transfer . . . . . . . . . . . . . . . . . . . . . . . 28

- A EXTENDED RELATED WORK

Scaling Laws Scaling laws are used to study the behavior of deep learning systems on scaling training data and/or compute. Various researchers have observed scaling properties of generalization error with training data size and model capacity (Banko & Brill, 2001; Hestness et al., 2017; Amodei et al., 2016). Specifically, in the context of LLMs, Kaplan et al. (2020) explicitly propose a power law for the relationship between the loss L, number of parameters in the language model N, and the number of dataset tokens D. Later, Hoffmann et al. (2022) proposed a different formula:

A Nα

L(N,D) = E +

B Dβ

+

(4)

We build upon this form throughout the paper to fit our scaling laws. Other researchers, too, have built upon these scaling laws to study various aspects of scaling, such as neural network architectures (Clark et al., 2022; Tay et al., 2022; Frantar et al., 2023; Gu & Dao, 2023; Scao et al., 2022) or transfer learning (Henighan et al., 2020). Researchers have also investigated scaling properties in different domains of deep learning such as neural machine translation (Ghorbani et al., 2021; Gordon et al., 2021), vision language models like CLIP (Cherti et al., 2023; Henighan et al., 2020), vision (Alabdulmohsin et al., 2022; Zhai et al., 2022), reinforcement learning (Hilton et al., 2023; Jones, 2021; Gao et al., 2023), and recommendation systems (Ardalani et al., 2022).

Scaling Laws for Data Mixtures Many prior works (Hoffmann et al., 2022) assume a fixed dataset distribution to study the effect of scaling on model performance. However, empirically, dataset composition can have a significant impact on quality (Longpre et al., 2023; Albalak et al., 2024; Sorscher et al., 2022). Hashimoto (2021) studies the relationship between dataset composition and model loss, finding a simple scaling law quantifying this relationship. Bansal et al. (2022) study the impact of data quality and noise on architecture, finding that synthetic data has a different exponent. Fernandes et al. (2023) study scaling for multilingual neural translation (Bapna et al., 2022), but experiments are restricted to 2-3 language pairs. Aghajanyan et al. (2023), in a similar vein, propose bimodal scaling laws for multimodal foundation models (Reid et al., 2024). Other researchers have considered different aspects of dataset composition that can affect model scaling: Muennighoff et al. (2024) study the effect of data repetition on scaling, while Goyal et al. (2024) consider how the effect of mixing data pools of varying quality changes with scale.

Foundation models involve training models on a mixture of datasets with the aim of achieving a certain validation loss on various validation sets and downstream datasets of interest. Based on the observation that optimal data mixture changes with scale, several recent works have attempted to characterize the change in the scaling law equation when either the train or test distribution changes.

Some researchers predict the optimal data mixture by first ablating mixtures using smaller models, and second, training a larger scaled-up model using the found optimal mixture (Ye et al., 2024; Xie et al., 2024; Liu et al., 2024). However, it is unclear if this optimal data mixture can extrapolate to a much larger magnitude, with some evidence in computer vision that this is not the case (Goyal et al., 2024) - i.e., the optimal data mixture changes with scale.

Some researchers have described scaling laws for different downstream evaluation sets for the same model (Dubey et al., 2024; Chen et al., 2024), while others (Brandfonbrener et al., 2024) have derived power law relationships between models trained on different distributions and the same model on different test sets.

Recently, several researchers have attempted to predict the final loss of a model for a given data mixture with scaling laws. Ge et al. (2024b) use the observation of a logarithmic shift in scaling pattern to predict loss on subsets of the Pile weighted in various proportions on a fixed model size. Ye et al. (2024), too, fit a scaling law for data mixtures based on a single scale. Kang et al. (2024) empirically show that the optimal mixture for a model changes with scale, and propose a method that addresses this on GPT-2 scale models. Que et al. (2024) use continual pre-training as a tool to develop their scaling law. Shukor et al. (2025), too, develop scaling laws that account for the transfer between data mixtures changing at scale for English LLMs and multimodal models for less than 10 domains.

Multilingual Scaling Laws Prior work has also investigated multilingual transfer effects specifically in pretraining (Chang et al., 2024) and post-training (Shimabucoro et al., 2025), though on a smaller scale of models. He et al. (2024) is the work closest to ours, where the authors attempt to describe scaling laws for multilingual LLMs. A crucial distinction from our work is that we account for individual-language transfer learning in our scaling laws, and achieve a better resulting fit (Table 1).

- B EXPERIMENTAL DETAILS

To understand multilingual scaling characteristics we run 774 separate experiments, across different model scales, and language mixtures. In Table B.1 we summarize the experiment segments, using the notation below, that defines sets of languages or scales.

- • Lmono (7): {en, fr, ru, zh, hi, sw, yo}
- • Lpairs (10): {en, fr, ru, zh, hi, de, es, pt, ja, vi}
- • Ltarget (15): en/X for X ∈ Lpairs + {es/pt, fr/zh, hi/ru, de/ja, hi/vi}
- • Leval (48): complete evaluation set (all 48 languages)
- • Cexp (12): {4×4, 6, 8×2, 12, 16, 24, 32, 50}
- • Sfull (20): {0, 1, 2, 4, 6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 42, 46}
- • Spart (11): {0, 4, 8, 12, 16, 20, 24, 28, 34, 42, 46}
- • Smin (7): {0, 8, 16, 24, 34, 42, 46}

To see the model sizes defined by the scales in Sfull, Spart, and Smin, see Table B.3. We select these scale ranges to empirically observe models from 10M to 8B parameters. Subsets of languages, or

subsets of the full scale (Spart, or Smin) are chosen to ensure the experiment number is sufficiently tractable for each research question, but also computationally feasible given our resources.

- Table B.1: An overview of experiment configurations in this work. We enumerate the experiment types: <Lang> for monolingual scaling, Unimax as a massively multilingual baseline, Language Pairs to measure language-to-language transfer, Capacity to measure the curse of multilinguality, or model capacity constraints on learning new languages, and Finetunes to understand how finetuning from a massively multilingual model compares to pretraining from scratch. We use a mix of Monolingual and Multilingual vocabularies, and training data. In the LANGUAGES and SCALES columns we use parentheses to show the number of language mixtures and number of scales run. Symbols such as Lmono (7) and Sfull (20) are defined above.

EXPERIMENT TAG VOCAB. TRAIN-DATA LANGUAGES SCALES # EXPS

Mono <Lang> Monolingual Monolingual Lmono (7) Sfull (20) 7 × 20 = 140 Unimax Multilingual Multilingual Lunimax Sfull (20) 20 Multi <Lang> Multilingual Monolingual Lmono (7) Spart (11) 10 × 7 = 70 Language Pairs Multilingual Monolingual Leval (48) 34 50 Language Pairs Multilingual Bilingual Lpairs (10) 34 90 Language Pairs Multilingual Bilingual Ltarget (15) Spart (11) 10 × 15 = 150 Capacity Multilingual Uniform Sampling Cexp (12) Spart (11) 10 × 12 = 120 Finetunes Multilingual Monolingual Leval (48) 34 50 Finetunes Multilingual Monolingual Lmono (7)∪Lpairs (10) Smin (7) 7 × 12 = 84

###### Total 774

- B.1 LANGUAGE CHOICE

The fifty languages we selected come from the intersection of languages covered by both Flores-200 and MADLAD-400. They were selected to cover a wide range of language families, scripts, and resource levels. To get this collection, we sorted all languages in the Flores-MADLAD intersection by the number of characters, and then selected every six languages. We then went over this list and

perturbed the index up or down if it was too typologically similar to an already-selected language. For example, Language 48 was Afrikaans, but there were already several Germanic languages on the list, so we opted to take language 47 instead (Telugu).

- B.2 EXPERIMENT DESIGN

Pretraining Hyperparameter Details We rely on the hyperparameter settings refined by prior work to obtain reliable performance across model sizes. We also conducted initial experiments, varying batch sizes, learning rates, optimizer, and dropout to ensure our settings were relatively robust. In table B.2 we enumerate and explain all our hyperparameter choices, citing related work.

Train & Test Data We use two test sets. For each we measure the vocabulary-insensitive loss proposed in Tao et al. (2024) in order to fairly evaluate across languages.

- 1. MADLAD-400 Test We isolate a test set from each of our 50 evaluation languages in MADLAD-400 (Kudugunta et al., 2024). In initial experiments we find stable metrics over random seeds requires sampling roughly (sequence length x num instances) 2048∗10000 = 20,480,000 tokens. For each language we randomly sample the minimum of either 20M tokens or 20% of the total tokens for low resource languages min(20M tokens, 20%).
- 2. Flores-101 We use Flores-101 (Goyal et al., 2021), a language-parallel set of 101 highquality translations, as a comparable test set. We extract the monolingual sequences the test set, using those to compute sequence losses, comparable to MADLAD-400 Test.

Vocabulary Details We use a vocabulary trained by Kudugunta et al., with a 64000 sized vocabulary, T = 100 and 99.9995% character coverage.

- B.3 EXPERIMENTAL DETAILS: SCALING LAWS EVALUATION

Typically, to evaluate fitted scaling laws, prior work has adopted the R2 metric, for it’s scaleinvariant measure of a laws’ explanatory power over the observed data. However, the choice of what data to hold-out as the test set can have a significant impact on the performance and it’s interpretation (Li et al., 2025a). For this reason, and because multilingual scaling laws are used to measure more than just the predictive power of the law to larger scales, we use a set of different R2 metrics to measure different objectives. These are:

- 1. R2 Hold-out a randomly selected 20% of the data points from the data. This gives an impression of the models’ ability to fit the variance, without measuring particular dimensions of fit.
- 2. R2(D) Hold-out the training runs with training tokens with the top 20% of D tokens.
- 3. R2(N) — Hold-out a couple of scales of model sizes, including the largest: N = 660M and N = 8B parameters.
- 4. R2(C) Hold-out the largest compute parts of the training runs, where C = 6ND FLOPs. This will include mainly a combination of the larger model sizes, and longer training runs.
- 5. R2(M) Hold-out all mixtures that are not monolingual, bilingual, or unimax. This allows the scaling law to get a baseline sense of each interacting language, but it has to infer the rest of the mixture scaling properties.

These metric variants allow us to unpack specifically where the scaling laws perform well, and for what purposes they are reliable.

- B.4 EXPERIMENTAL DETAILS: LANGUAGE FINETUNING

In Section 6 we experiment with continuous pretraining on a single language, starting from a massively multilingual model’s checkpoint. For clarity, we refer to this as finetuning, to distinguish it from pretraining from scratch.

We begin by training massively multilingual models, using UNIMAX (Chung et al., 2023) language sampling across all 420 languages in MADLAD-400. Chung et al. (2023) demonstrate this is an

- Table B.2: The pretraining hyperparameters used across experiments. We detail the batch size scheduling, the vocabulary choice, as well as fine-grained hyperparameter choices. Each choice is justified and grounded in prior work, discussed in the Explanation column.

PARAMETER VALUE EXPLANATION General Training Parameters

Learning Rate Scheduler WSD We adopt the Warmup Stable Decay (WSD) learning rate schedule, designed to efficiently study data-model scaling law without extensive retraining (Hu et al., 2024; Hägele et al., 2024)

Optimizer AdamW Commonly used in scaling law experiments (Loshchilov & Hut-

ter, 2019; Hoffmann et al., 2022) Base Learning Rate 2e-4 Following (Muennighoff et al., 2024) and confirmation it works

well in initial experiments. Warmup Steps 1000 Standard practice, as in (Longpre et al., 2023) Decay Period 10% Following (Hu et al., 2024; Hägele et al., 2024) this is the min-

imum well performing decay length. Sequence Length 2048 As in (Longpre et al., 2023; Muennighoff et al., 2024). Training Steps 30k+ We vary the steps according to experiments, to always ensure

we are well above likely chinchilla compute optimal ranges.

Dropout 0.1 While English scaling laws work tends to use a dropout of 0.0, we see little notable difference, except for lower resource languages, with repeating epochs.

Batch Size Schedule Batch Size (<150M params) 256 Hu et al. (2024); Muennighoff et al. (2024) both find the

- Batch Size (<1B params) 512

- Batch Size (<2B params) 1024 Batch Size (2B+ params) 2048

optimal batch size increases with model size. We adopt a slightly larger rising batch size than Muennighoff et al. (2024), following initial empirical results.

Vocabulary

Vocab Size 64k Vocab (Unimax) Vocabulary by Kudugunta et al., with T = 100 and 99.9995%

character coverage. Vocab (Monolingual) Same settings as above, but trained only on sentences from the

language being considered.

effective sampling strategy to perform well across languages, as is the objective with many large multilingual models. The UNIMAX sampling rate per language is documented in Table B.4 Each of these models we train for 1T tokens, to reflect real-world practices.

Using these Unimax checkpoints, across model scales, we then conduct continuous monolingual pretraining (which we call finetuning here) on all 48 languages separately in Leval (48). While finetuning, we also observe the loss across all Leval (48)languages as well. We use the same hyperparameters as discussed for pretraining, but reset the learning schedule, and train for at least another 30k+ steps. These empirical results allow us to compare (across a range of N,D,C) the loss on language t between a model finetuned from a Unimax checkpoint, and a model pretrained monolingually from scratch on language t.

- B.5 EXPERIMENTAL DETAILS: LANGUAGE TRANSFER

- In Section 4 we measure how much training on source language s is beneficial for the test loss on the target language t. A language transfer score tells a practitioner the extent to which training with each language would help or hinder their target language t. There are two ways in which we can measure this language transfer:

1. BILINGUAL TRANSFER SCORE. This score measures how many training tokens a 50/50 (s, t) bilingual model needs, relative to a monolingual target language t baseline, to reach the same validation loss Lt. A positive score indicates co-training with the source language has positive transfer, whereas a negative score indicates it hinders convergence.

- Table B.3: The SCALE value mapped to the dimensions of the model, and the parameter count. The model dimensions, borrowed from Muennighoff et al. (2024), report the number of attention heads, number of layers, embed size, feedforward size, and key-value size. Our model may be slightly different sizes than prior work based on our vocabulary size (64000 + 512 special tokens).

SCALE HEADS LAYERS EMBED FFW KV PARAMETERS

- 0 4 3 128 512 32 9,044,352

- 1 7 4 224 896 32 17,662,848

- 2 7 5 288 1,152 32 24,847,776

- 3 7 6 448 1,792 32 45,763,200

- 4 8 8 512 2,048 64 66,588,672

- 5 9 9 576 2,304 64 84,939,840

- 6 10 10 640 2,560 64 106,830,080

- 7 10 13 640 2,560 64 126,492,800

- 8 10 16 640 2,560 64 146,155,520

- 9 12 12 768 3,072 64 162,800,640

- 10 12 15 768 3,072 64 191,114,496

- 11 12 18 768 3,072 64 219,428,352

- 12 14 14 896 3,584 64 237,646,080

- 13 14 16 896 3,584 64 263,337,984

- 14 14 18 896 3,584 64 289,029,888

- 15 16 16 1,024 4,096 64 334,512,128

- 16 16 18 1,024 4,096 64 368,068,608

- 17 16 20 1,024 4,096 64 401,625,088

- 18 10 18 1,280 5,120 128 554,457,600

- 19 10 21 1,280 5,120 128 633,104,640

- 20 11 18 1,408 5,632 128 661,807,872

- 21 10 24 1,280 5,120 128 711,751,680

- 22 11 21 1,408 5,632 128 756,970,368

- 23 12 19 1,536 6,144 128 816,345,600

- 24 11 24 1,408 5,632 128 852,132,864

- 25 12 22 1,536 6,144 128 929,596,416

- 26 12 25 1,536 6,144 128 1,042,847,232

- 27 14 20 1,792 7,168 128 1,143,245,824

- 28 14 23 1,792 7,168 128 1,297,391,872

- 29 14 26 1,792 7,168 128 1,451,537,920

- 30 16 22 2,048 8,192 128 1,608,560,640

- 31 17 22 2,176 8,704 128 1,807,137,536

- 32 16 25 2,048 8,192 128 1,809,893,376

- 33 16 28 2,048 8,192 128 2,011,226,112

- 34 17 25 2,176 8,704 128 2,034,422,912

- 35 18 24 2,304 9,216 128 2,187,122,688

- 36 17 28 2,176 8,704 128 2,261,708,288

- 37 18 28 2,304 9,216 128 2,526,870,528

- 38 18 32 2,304 9,216 128 2,866,618,368

- 39 20 26 2,560 10,240 128 2,891,514,880

- 40 20 30 2,560 10,240 128 3,310,955,520

- 41 20 34 2,560 10,240 128 3,730,396,160

- 42 21 36 2,688 10,752 128 4,335,303,168

- 43 22 36 2,816 11,264 128 4,749,364,224

- 44 23 36 2,944 11,776 128 5,182,299,648

- 45 24 36 3,072 12,288 128 5,634,109,440

- 46 28 40 3,584 14,336 128 8,452,190,208

- 47 32 42 4,096 16,384 128 11,538,702,336

- 48 32 47 4,352 17,408 128 14,314,315,520

- 49 36 44 4,608 18,432 128 15,245,973,504

- 50 32 47 4,608 18,432 128 15,821,655,552

- 51 32 47 4,864 19,456 128 17,402,920,192

- 52 40 47 5,120 20,480 128 18,982,943,616

- Table B.4: The Unimax language sampling rates adapted from Chung et al. (2023). Languages are listed in order of their percentage sampling rate, which sum to 100.

LANG % LANG % LANG % LANG % LANG % LANG %

- en 5.00e+00 af 1.42e+00 be 1.42e+00 fil 1.42e+00 gl 1.42e+00 te 1.42e+00 de 1.42e+00 es 1.42e+00 fr 1.42e+00 hr 1.42e+00 is 1.42e+00 kk 1.42e+00 ml 1.42e+00 mr 1.42e+00 ru 1.42e+00 sr 1.42e+00 ta 1.42e+00 az 1.42e+00 hi 1.42e+00 id 1.42e+00 it 1.42e+00 lv 1.42e+00 ms 1.42e+00 nl 1.42e+00 pl 1.42e+00 pt 1.42e+00 sq 1.42e+00 sv 1.42e+00 tr 1.42e+00 vi 1.42e+00 ca 1.42e+00 et 1.42e+00 hu 1.42e+00 iw 1.42e+00 ro 1.42e+00 sl 1.42e+00 th 1.42e+00 zh 1.42e+00 ar 1.42e+00 cs 1.42e+00 fa 1.42e+00 fi 1.42e+00 ja 1.42e+00 ko 1.42e+00 lt 1.42e+00 sk 1.42e+00 uk 1.42e+00 bg 1.42e+00 da 1.42e+00 el 1.42e+00 no 1.42e+00 mk 1.38e+00 bn 1.34e+00 eu 1.34e+00 ka 1.19e+00 mn 1.09e+00 bs 1.03e+00 uz 1.02e+00 ur 8.19e-01 sw 7.19e-01 ne 6.87e-01 kaa 6.76e-01 kn 6.72e-01 gu 6.43e-01 si 5.89e-01 cy 5.14e-01

- eo 5.06e-01 la 4.64e-01 hy 4.47e-01 ky 4.37e-01 tg 4.28e-01 ga 4.25e-01 mt 4.06e-01 my 3.95e-01 km 3.35e-01 tt 3.14e-01 so 2.93e-01 ps 2.52e-01 ku 2.50e-01 pa 2.38e-01 rw 2.29e-01 lo 2.06e-01 dv 1.83e-01 ha 1.78e-01 ckb 1.73e-01 fy 1.68e-01 lb 1.63e-01 mg 1.54e-01 ug 1.52e-01 am 1.50e-01 gd 1.48e-01 ht 1.27e-01 grc 1.25e-01 jv 1.12e-01 tk 1.09e-01 hmn 1.09e-01 sd 1.05e-01 mi 9.77e-02 yi 9.55e-02 ba 9.42e-02 fo 9.24e-02 ceb 9.12e-02 or 9.07e-02 kl 8.12e-02 xh 7.21e-02 su 7.20e-02 ny 6.97e-02 sm 6.94e-02 sn 6.68e-02 co 6.67e-02 pap 6.57e-02 zu 6.46e-02 ig 6.31e-02 yo 6.00e-02 st 5.70e-02 haw 5.38e-02 as 5.07e-02 oc 4.93e-02 cv 4.66e-02 lus 4.61e-02 tet 4.14e-02 gsw 4.04e-02 sah 4.01e-02 br 3.29e-02 rm 2.52e-02 sa 2.23e-02 bo 2.23e-02 om 2.22e-02 se 2.12e-02 ce 1.70e-02 cnh 1.58e-02 ilo 1.49e-02 hil 1.44e-02 udm 1.39e-02 os 1.26e-02 lg 1.21e-02 ti 1.12e-02 vec 1.10e-02 ts 9.73e-03 tyv 9.66e-03 kbd 9.23e-03 ee 8.25e-03 iba 7.66e-03 av 7.57e-03 kha 7.57e-03 to 7.51e-03 tn 7.33e-03 nso 7.08e-03 fj 7.02e-03 zza 6.60e-03 ak 6.23e-03 ada 6.08e-03 otq 5.86e-03 dz 5.69e-03 bua 5.44e-03 cfm 5.41e-03 ln 5.39e-03 chm 5.36e-03 gn 5.23e-03 krc 5.21e-03 wa 5.11e-03 hif 4.79e-03 yua 4.32e-03 srn 4.26e-03 war 4.03e-03 rom 3.99e-03 bik 3.94e-03 sg 3.89e-03 lu 3.87e-03 ady 3.73e-03 kbp 3.68e-03 syr 3.51e-03 ltg 3.49e-03 myv 3.48e-03 iso 3.43e-03 kac 3.43e-03 bho 3.38e-03 ay 3.30e-03 kum 3.10e-03 qu 3.06e-03 pag 3.02e-03 ngu 2.97e-03 ve 2.94e-03 pck 2.88e-03 zap 2.86e-03 tyz 2.83e-03 hui 2.73e-03 bbc 2.65e-03 tzo 2.65e-03 tiv 2.55e-03 ksd 2.52e-03 gom 2.50e-03 min 2.47e-03 ang 2.46e-03 nhe 2.45e-03 bgp 2.45e-03 nzi 2.37e-03 nnb 2.29e-03 nv 2.28e-03 bci 2.26e-03 kv 2.25e-03 new 2.21e-03 mps 2.19e-03 alt 2.18e-03 meu 2.15e-03 bew 2.13e-03 fon 2.08e-03 iu 2.08e-03 abt 2.07e-03 mgh 2.05e-03 tvl 2.02e-03 dov 2.00e-03 tlh 1.96e-03 ho 1.96e-03 kw 1.92e-03 mrj 1.92e-03 meo 1.89e-03 crh 1.89e-03 mbt 1.87e-03 emp 1.85e-03 ace 1.85e-03 ium 1.85e-03 mam 1.81e-03 gym 1.74e-03 mai 1.72e-03 crs 1.70e-03 pon 1.69e-03 ubu 1.68e-03 quc 1.62e-03 gv 1.57e-03 kj 1.49e-03 btx 1.48e-03 ape 1.46e-03 chk 1.45e-03 rcf 1.44e-03 shn 1.42e-03 tzh 1.41e-03 mdf 1.39e-03 ppk 1.38e-03 ss 1.37e-03 gag 1.31e-03 cab 1.27e-03 kri 1.25e-03 seh 1.23e-03 ibb 1.23e-03 tbz 1.21e-03 bru 1.21e-03 enq 1.20e-03 ach 1.17e-03 cuk 1.16e-03 kmb 1.15e-03 wo 1.14e-03 kek 1.12e-03 qub 1.11e-03 tab 1.11e-03 bts 1.07e-03 kos 1.06e-03 rwo 1.05e-03 cak 1.05e-03 tuc 1.02e-03 bum 1.01e-03 gil 9.73e-04 stq 9.65e-04 tsg 9.47e-04 quh 9.39e-04 mak 9.37e-04 arn 9.35e-04 ban 9.06e-04 jiv 8.84e-04 sja 8.49e-04 yap 8.33e-04 tcy 8.26e-04 toj 8.20e-04 twu 8.15e-04 xal 8.08e-04 amu 8.05e-04 rmc 8.04e-04 hus 7.83e-04 nia 7.77e-04 kjh 7.73e-04 bm 7.64e-04 guh 7.64e-04 mas 7.64e-04 acf 7.56e-04 dtp 7.46e-04 ksw 7.25e-04 bzj 7.22e-04 din 7.17e-04 zne 7.15e-04 mad 6.99e-04 msi 6.84e-04 mag 6.60e-04 mkn 6.57e-04 kg 6.54e-04 lhu 6.37e-04 ch 6.23e-04 qvi 5.78e-04 mh 5.59e-04 djk 5.52e-04 sus 5.21e-04 mfe 5.20e-04 srm 5.12e-04 dyu 5.08e-04 ctu 5.07e-04 gui 5.03e-04 pau 5.02e-04 inb 4.88e-04 bi 4.71e-04 mni 4.59e-04 guc 4.43e-04 jam 4.39e-04 wal 4.38e-04 jac 4.35e-04 bas 4.30e-04 gor 4.18e-04 skr 4.15e-04 nyu 4.14e-04 noa 4.08e-04 sda 4.08e-04 gub 4.07e-04 nog 4.04e-04 teo 4.01e-04 tdx 3.92e-04 sxn 3.81e-04 rki 3.76e-04 nr 3.74e-04 frp 3.61e-04 alz 3.60e-04 taj 3.51e-04 lrc 3.50e-04 cce 3.22e-04 rn 3.21e-04 jvn 3.14e-04 hvn 3.08e-04 nij 3.07e-04 dwr 2.99e-04 izz 2.79e-04 msm 2.78e-04 bus 2.73e-04 ktu 2.66e-04 chr 2.52e-04 maz 2.39e-04 tzj 2.22e-04 suz 2.15e-04 knj 2.15e-04 bim 1.99e-04 gvl 1.98e-04 bqc 1.98e-04 tca 1.97e-04 pis 1.92e-04 laj 1.83e-04 qxr 1.82e-04 niq 1.80e-04 ahk 1.80e-04 shp 1.78e-04 hne 1.75e-04 spp 1.71e-04 koi 1.67e-04 quf 1.53e-04 agr 1.39e-04 tsc 1.31e-04 mqy 1.27e-04 gof 1.27e-04 gbm 1.25e-04 miq 1.22e-04 dje 1.21e-04 awa 1.19e-04 qvz 1.10e-04 tll 1.03e-04 raj 1.02e-04 kjg 1.00e-04 quy 9.24e-05 cbk 8.52e-05 akb 8.48e-05 oj 8.46e-05 ify 8.39e-05 cac 8.03e-05 brx 7.64e-05 qup 7.48e-05 ff 6.97e-05 ber 6.68e-05 tks 6.55e-05 trp 6.46e-05 mrw 6.46e-05 adh 6.40e-05 smt 6.16e-05 ffm 5.93e-05 qvc 5.87e-05 ann 5.40e-05 kaa_Latn 5.26e-05 nut 4.62e-05 kwi 4.34e-05 msb 4.20e-05 el_Latn 4.09e-05 doi 3.40e-05 dln 2.97e-05 hi_Latn 2.48e-05 ctd_Latn 1.03e-05 ru_Latn 6.95e-06 te_Latn 4.84e-06 ber_Latn 4.74e-06 az_RU 3.23e-06 ta_Latn 2.29e-06 tly_IR 2.15e-06 nan_Latn_TW 1.94e-06 ml_Latn 1.80e-06 zxx_xx_dtynoise 1.65e-06 gom_Latn 1.28e-06 bg_Latn 9.21e-07 kn_Latn 6.18e-07 zh_Latn 5.69e-07 cr_Latn 2.59e-07 bn_Latn 1.83e-07 gu_Latn 1.69e-07 sat_Latn 1.51e-07 ndc_ZW 1.31e-07 kmz_Latn 1.01e-07 ms_Arab 6.00e-08 ms_Arab_BN 4.29e-08

2. FINETUNING ADAPTATION SCORE. This score captures the effect on the target language’s loss Lt, when a massively multilingual model is finetuned only on the source language s.

- B.5.1 BILINGUAL TRANSFER SCORE (BTS)

The Bilingual Transfer Score asks how data-efficient a bilingual learner is, relative to a purely monolingual one, on some target language t. To estimate this, we pretrain two models, a monolingual one on t and a bilingual one that samples tokens equally (50/50) from the source and target languages (s, t). We then measure how many more training tokens it takes the bilingual model to attain the same validation loss Lt as the monolingual model. This “distance” between the learning curves is measured throughout training, at different reference horizons—or, number of tokens trained on. In Figure 2 we use models with 2B parameters, and a reference horizon of d = 42B tokens, as it roughly equates to 10,000 pretraining steps for our 2B parameter models, and learning curves have stabilized. In Figure C.1 we measure how the BTS varies by model size, and by number of training tokens seen.

We are able to compute the bilingual transfer scores, across every combination of pairings between 10 languages: English, Russian, Spanish, French, German, Portuguese, Chinese, Vietnamese, Japanese, and Hindi. These languages were chosen to give a distribution of high- and mid-resource languages from a variety of language families. This totals 10 monolingual experiments (one for each language), and C(10,2) = 45 bilingual experiments, which provide the results for a 10x10 grid of Bilingual Transfer Scores.

In more detail, let dmono be the number of tokens trained on language t by a monolingual model. This model attains test loss of Lt(dmono) after dmono tokens on language t. We record the number of training tokens dbi at which the bilingual model reaches the same loss, Lt(dbi) = Lt(dmono). We define the Bilingual Transfer Score (BTS) as the signed step surplus

BTSs→t = −

σbi(Lt(dmono)) − 2dmono dmono

where dbi = σbi(Lt(dmono)). σbi maps a loss value to the number of steps taken by the bilingual model to reach that loss. In short, this score measures the relative training efficiency of the bilingual model compared to a monolingual model at reaching the same loss level for language t. Subtracting 2dmono and dividing by dmono simply centers the value on 0, and forces positive transfer to give a positive value. A value of 0, implies neutral transfer, as the multilingual model requires exactly the same number of steps in the target language t as the monolingual model, a value below 0 implies negative transfer, as the multilingual model is slower to reach the loss Lt(dmono), and a value above 0 implies positive transfer, as the bilingual model reaches the target loss in < 2dmono. Note that for Figure 2 these scores are anchored on 2B parameter models, trained for d = 42B tokens, for consistency. But as model size increases, or tokens trained decreases, the language transfer scores improve monotonically.

- B.5.2 FINETUNING ADAPTATION SCORE (FAS)

Complementary to BTS, the Finetuning Adaptation Score measures the instantaneous impact of continued exposure to a single language on all others while holding model capacity fixed. Specifically, it measures how finetuning a massively multilingual model on source language s affects the performance of target language t. Because we only need to train on each source language once, and evaluate on all languages, it is less computationally expensive than BTS, which requires training on every pair of languages. As such, we are able to derive 38 ∗ 38 = 1444 (s, t) comparisons from 38 finetuning experiments.

We start from a shared multilingual checkpoint: a Unimax pretrained model, trained for 1T tokens. For each source language s, we continue pre-training (for simplicity we denote this as finetune) exclusively on this language, and evaluate at regular intervals on every target language’s t∈L validation set. Let Ls→t(d) be the validation loss on t after d additional updates on s, and let Lunimaxt denote the baseline loss of the shared Unimax checkpoint, before finetuning has begun. The Finetun-

ing Adaptation Score (FAS) aggregates the reduction in loss over a common time window [0,dmax]:

dmax

1 dmax

Lunimaxt − Ls→t(d) dd.

FASs→t =

0

Positive values indicate that exposing the model to s accelerates learning or yields immediate gains on t (“helpful transfer”), whereas negative values capture negative interference. Normalising by dmax makes the score comparable across language pairs and training horizons.

Because finetuning on language s can cause unpredictable behavior in the validation loss of a target language t (it might immediately increase, or decrease then eventually increase), we start with deriving a series of features about the finetuning loss curve. We calculate the following features for a finetuning loss curve:

- • Baseline Loss Deviation (δs→t). For each language pair s ̸= t we quantify the deviation in validation loss between the baseline Lunimaxt and the loss Ls→t on the final step:

δs→t = Ls→t(dmax) − Lunimaxt (dmax) δs→t < 0 means fine–tuning on s helps t; δs→t > 0 means it hurts.

- • Adaptation Gain (gl). This measures the area normalized reduction in loss on language l from finetuning on language l. This allows us to capture the relative learning dynamics of both the source and target language.

dmax

1 dmax

Lunimaxl − Ls→t(d) ds.

gl =

0

gl > 0 indicates net improvement; large values imply that l benefits greatly from additional supervision.

For each language transfer pair (s, t), we compose the following feature vector xs→t. As languages differ in intrinsic difficulty, we normalize each feature.

gl − µg σg

δs→t − µδ,t σδ,t

, δ˜s→t =

, (5)

g˜l =

For adaptation gain, we use the global mean and standard deviation (µg,σg), and for baseline loss deviation we compute the mean and standard deviation (µδ,t,σδ,t) across all source languages s for each fixed target language t.

- B.6 ESTIMATING BILINGUAL TRANSFER SCORE (BTS)

In the Bilingual Transfer Score experiments we empirically measured language transfer scores ystrue→t for a subset Pobs ⊂ L×L. We construct a training set from these 90 experiments, using them as the gold labels.

xs→t = g ˜s, g˜t, g˜s − g˜t, δ˜s→t ⊤, ys→t = ystrue→t, (6)

We fit a random-forest regressor φθ : R4 → R with 300 trees and unlimited depth. The predictive performance is assessed by k-fold cross-validation (k =5), obtaining an R2 = 0.85 and Spearman correlation of ρ = 0.88. After training on all observed pairs, φ is invoked on every unlabelled (s,t)∈L×L, yielding predicted BTS values, that estimate the language transfer shown in Figure 2.

- B.7 EXPERIMENTAL DETAILS: MULTILINGUAL CAPACITY

- B.7.1 MULTILINGUAL CAPACITY SCALING LAWS

- In Section 5 we aim to understand how a model’s capacity hinders multilingual learning—also termed the curse of multilinguality (Conneau et al., 2019; Chang et al., 2024). To do this, we train models of various sizes, and with a range of language mixtures, shown in Table B.5. For simplicity, languages are always evenly sampled within their mixtures, even if some languages have more available text than others. We train models on these mixtures, and evaluate the validation loss of all other languages, throughout training.

- Table B.5: The set of experiments designed to measure the curse of multilinguality—or how language performance is impacted by both the number of training languages and the model size. We defined VERSION mixtures with NUM languages, ranging between 4 and 50. For mixtures with 4 languages, we define a few different versions oriented to languages that might be more likely to be grouped together. This set of experiments are summarized in the Capacity row of Table B.1.

VERSION NUM LANGUAGES (ALPHABETICAL)

4v0 4 de, es, fr, pt 4v1 4 de, en, es, fr 4v2 4 hi, ja, vi, zh 4v3 4 en, hi, pt, ru 6v0 6 de, en, es, fr, pt, ru 8v0 8 de, en, es, fr, pt, ru, vi, zh 8v1 8 de, en, fr, hi, ja, ru, vi, zh 12v0 12 de, en, es, fr, hi, it, ja, pl, pt, ru, vi, zh 16v0 16 de, en, es, fr, hi, id, it, ja, nl, pl, pt, ru, sv, tr, vi, zh 24v0 24 cs, da, de, en, es, fa, fi, fr, hi, hu, id, it, ja, nl, pl, pt, ro, ru, sv, th, tr, uk, vi, zh 32v0 32 ar, bg, ca, cs, da, de, el, en, es, fa, fi, fr, hi, hu, id, it, ja, ko, lt, nl, no, pl, pt, ro, ru, sk, sv, th, tr,

uk, vi, zh

50v0 50 af, ar, ba, bm, bn, ca, ckb, cy, de, ee, el, en, es, fa, fil, fr, gn, gu, ha, he, hi, hr, id, it, ja, jv, kk,

ko, mr, ms, my, nl, no, om, pl, ps, pt, ro, ru, sm, sv, sw, te, th, tr, uk, ur, vi, yo, zh

For a given target language t we have empirical loss scores across 11 differently sized models (from 10M to 8B), and across each mixture that contains the language in training, including monolingual, bilingual, and the multilingual mixtures shown in Table B.5.

We model each language’s loss as L(N,D,K), where N is the model size, D are the number of training tokens for the target language, and K is the number of languages in the training mixture (evenly sampled). We define L(N,D,K) as follows:

L(N,D,K) = L∞ +

AKϕ Nα

+

B Kψ Dβ

(7)

This law preserves the well-validated power-law behavior of monolingual scaling laws, while introducing K into both the capacity term (AKϕ/Nα) and the data term (BKψ/Dβ). The exponent ϕ captures how capacity requirements grow with the number of languages, while ψ captures how data requirements change with multilinguality: ψ < 0 indicates positive transfer (sublinear per-language data needs as K increases), ψ = 0 implies no net transfer in the data term, and ψ > 0 implies negative transfer/interference. Under even sampling, the total tokens across languages are Dtot = K ·D, in which case the data term can be rewritten as B Kψ+β/Dtotβ .

- B.7.2 ADDING LANGUAGES: K TO K ∗ r

Practitioners may wish to change the number of languages supported by a model, from K to K ∗ r. If the model size and training tokens remain unchanged, it will lead to a degradation in loss across languages, as the same compute budget is allocated across more languages. When adding languages, the practitioner may wish to know how much they need to scale the model (N, D) and the training compute C to maintain the same performance per language, as before. We call this an iso-loss, as it approximately preserves the loss values, with a change in K.

Compute-Optimality for Equation (7). As we increase K to K ∗ r we would like to understand the new N′,D′ that minimize training compute on the iso-loss surface. Let r = K′/K. Minimizing

- C′ ∝ N′D′ subject to A(Kr)ϕ

BKψ Dβ yields the closed-form optimum:

B(Kr)ψ D′β =

AKϕ Nα

N′α +

+

N′ N

⋆

′

= rϕ/α, D

D

⋆

′ tot

= rψ/β, D

Dtot

⋆

= r1+ψ/β.

These expressions show a clear separation of roles: ϕ/α governs the parameter burden of adding languages, whereas ψ/β governs the data burden (with ψ < 0 reducing the per-language data requirement due to positive transfer). Since the compute budget is in terms of total tokens Dtot = K · D, and C ∝ NDtot.

Adding languages: K to K ∗r — Finding the compute multiplier. At the iso-loss optimum, the compute multiplier is

C′ C

N′Dtot′ NDtot

⋆

⋆

Dtot′ Dtot

′

= r1+ϕ/α+ψ/β .

= N

=

N

Intuitively, αϕ measures the extra parameter capacity per added language, while ψβ measures the extra (or reduced, if ψ < 0) per-language token budget. But since compute is defined with total tokens

Dtot, then we consider the additional factor of r from enlarging the language inventory. Adding languages: K to K ∗ r — Finding the Iso-loss curve. Let s ≡ N

′

′

D denote the multipliers for model size and (per-target-language) data when we grow the language inventory from K to K′ = rK. Define the baseline term contributions

N and t ≡ D

AKϕ/Nα AKϕ/Nα + BKψ/Dβ

wN ≡

, wD = 1 − wN.

The iso-loss condition equates the sum of the two error terms before and after the change in K:

A(Kr)ϕ (Ns)α

B(Kr)ψ (Dt)β

AKϕ Nα

BKψ Dβ

+

=

+

.

Dividing both sides by AKϕ/Nα + BKψ/Dβ yields the normalized iso-loss constraint

rϕ wN s−α + rψ wD t−β = 1. (8) Solving for t given s. Rearranging equation 8 gives

1/β

rψ wD 1 − rϕ wN s−α

t =

.

The denominator must be positive, which implies the feasibility condition

sα > rϕ wN.

- As sα ↓ rϕwN the denominator tends to 0+ and t → ∞; as s → ∞, the denominator tends to 1 and t → (rψwD)1/β. Solving for s given t. By symmetry,

1/α

rϕ wN 1 − rψ wD t−β

, with feasibility tβ > rψ wD.

s =

Total-token form. Under even sampling, Dtot = KD and Dtot′ = K′D′ = rKD′, so

1/β

D′ D

Dtot′ Dtot

rψ wD 1 − rϕ wN s−α

.

= r

= r

Compute-optimal frontier (starting from a compute-optimal (K,N,D)). Suppose the baseline (K,N,D) is compute-optimal for its loss level, i.e., it minimizes C ∝ ND subject to AKϕ/Nα + BKψ/Dβ = const. A standard Lagrange multiplier argument then yields

AKϕ Nα

BKψ Dβ ⇐⇒ wN =

β α + β

α α + β

α

= β

, wD =

.

′ tot

After changing K → rK, the compute along the iso-loss curve is C′/C = st (or C′/C = s · D

Dtot

if compute is defined with total tokens). Minimizing this product subject to equation 8 again via Lagrange multipliers gives the stationarity condition

α rϕ wN s−α = β rψ wD t−β.

Table C.1: A summary of the monolingual scaling laws for each language The columns are: the language, the language family, the language script, its number of unique tokens |D| in MADLAD400 (with the percentage of unique tokens as compared to English), and the derived scaling law.

LANGUAGE FAMILY SCRIPT |D| (% EN) SCALING LAW

Monolingual Vocabulary, Monolingual Training English Indo-european Latin 2.8T (100.0%) 0.67 + Ne60..1841 + De80..2541 French Indo-european Latin 363B (13.01%) 0.76 + eN100..6411 + eD130..6424 Russian Indo-european Cyrillic 705B (25.26%) 0.82 + eN100..6307 + eD130..6328 Chinese Sino-tibetan Hans 125B (4.48%) 1.08 + Ne80..2046 + eD100..4637 Hindi Indo-european Devanagari 7.9B (0.28%) 0.52 + Ne60..0339 + De80..0339 Swahili Niger-congo: atlantic congo Latin 770M (0.03%) 0.00 + Ne40..1326 + De50..5126

Multilingual Vocabulary, Monolingual Training

English Indo-european Latin 2.8T (100.0%) 0.83 + Ne90..9163 + eD130..6306 French Indo-european Latin 363B (13.01%) 0.66 + Ne70..1245 + De80..9445 Russian Indo-european Cyrillic 705B (25.26%) 0.76 + Ne70..9749 + De90..9149 Chinese Sino-tibetan Hans 125B (4.48%) 1.18 + Ne80..8749 + eD100..4990 Hindi Indo-european Devanagari 7.9B (0.28%) 0.63 + Ne60..9943 + De80..6943 Swahili Niger-congo: atlantic congo Latin 770M (0.03%) 0.00 + Ne40..9930 + De60..4330

Multilingual Vocabulary, Unimax Training

English Indo-european Latin 2.8T (100.0%) 0.00 + Ne30..0316 + De30..1516 French Indo-european Latin 363B (13.01%) 0.00 + Ne30..6319 + De30..9419 Russian Indo-european Cyrillic 705B (25.26%) 0.00 + Ne30..9020 + De40..2820 Chinese Sino-tibetan Hans 125B (4.48%) 0.39 + Ne40..5821 + De50..4721 Hindi Indo-european Devanagari 7.9B (0.28%) 0.00 + Ne30..4618 + De30..8918 Swahili Niger-congo: atlantic congo Latin 770M (0.03%) 0.00 + Ne30..5218 + De30..7418

Combining with equation 8 and substituting the compute-optimal weights above yields the unique minimum-compute point on the new frontier:

N′ N

⋆

′

= rϕ/α, D

D

⋆

′ tot

= rψ/β, D

Dtot

⋆

= r1+ψ/β.

- At this point each error component is individually restored to its baseline magnitude:

A(Kr)ϕ/N′α = AKϕ/Nα and B(Kr)ψ/D′β = BKψ/Dβ, so the weights (wN,wD)—and hence the balance of capacity/data bottlenecks—remain unchanged. Because the frontier is convex in (log s,log t) and log C′ = log s + log t is linear, this stationary point is the global compute minimum along the iso-loss curve.

C EXTENDED RESULTS

- C.1 EXTENDED RESULTS: SCALING LAWS

In Table C.1 we illustrate the fitted scaling laws for each language. Specifically, we show the scaling law parameters for each language when trained with a monolingual vocabulary on monolingual data, when trained with a multilingual vocabulary on monolingual data, and when trained with a multilingual vocabulary on the Unimax multilingual mixture. These results match up with those presented in Figure 1.

1.0

0.5

0.5

0.0

TransferScore

TransferScore

0.5

0.0

1.0

0.5

1.5

Score=0.39 1.31 D

Score= 0.11+0.08 N

1.0

0.2 0.4 0.6 0.8 1.0

0 2 4 6 8

Training Tokens (D) [×1011]

Model Size (N) [×109]

- Figure C.1: Left: X-axis is training tokens (D), Y-axis is language-transfer score. Right: X-axis is model size (N), Y-axis is language-transfer score.

- C.2 EXTENDED RESULTS: LANGUAGE TRANSFER

Research Question: How does language transfer change with larger models, or when you train for longer?

We further investigated how language transfer evolves with training data (D) and model size (N) in Figure C.1. Our analysis of transfer scores over the course of training reveals that transfer dynamics are established early. The performance gap between synergistic pairs (e.g., es → pt) and interfering pairs (e.g., en → zh) appears within the initial stages of training and remains largely consistent, suggesting that the fundamental compatibility of languages is not significantly altered by longer training. However, the impact of model scale is more pronounced. As illustrated in Figure C.1, we observe a clear trend in which larger models are more effective in facilitating cross-lingual transfer. For synergistic pairs, the positive transfer score increases modestly with model capacity. More importantly, for challenging pairs that exhibit interference in smaller models, larger models are significantly better at mitigating this negative effect, bringing the score closer to zero. This is similar to what has been observed while training larger and larger foundation models (Team et al., 2024). Given that transfer dynamics change at scale, it is an important factor to consider when designing multilingual foundation models that are performant for all languages being trained on.

Indo-Iranian

Vietic Celtic

Tai

Sinitic

Romance

Volta-Niger

Benue-Congo

Germanic

Japanese

Common Turkic

Slavic

Semitic

Buyeo

South-Central Dravidian

Hellenic

Malayo-Polynesian

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | |-0.2| |0.2| |0.2| |0.2| |0.1| |-0.1| |-0.2| |-0.1| |-0.0| |-0.1| |-0.1| |-0.1| |-0.1| |-0.0| |0.1| |0.2| |0.2| |-0.0| |-0.1| |-0.0| |-0.1| |-0.0| |-0.1| |-0.1| |-0.0| |0.3| |0.3| |0.0| |0.2| |0.1| |-0.1| |0.3| |0.1| |-0.0| |-0.1| |-0.0| |-0.1| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.2| | | |-0.2| |-0.2| |-0.2| |-0.3| |-0.1| |-0.5| |-0.3| |-0.3| |-0.3| |-0.2| |0.0| |-0.3| |-0.3| |-0.2| |-0.3| |-0.2| |-0.3| |-0.2| |-0.3| |-0.2| |-0.3| |-0.2| |-0.3| |-0.1| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.3| |-0.1| |-0.2| |0.3| |-0.3| |-0.2| |-0.1| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.1| |-0.2| | | |0.4| |0.1| |0.5| |-0.2| |-0.2| |-0.3| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.1| |-0.1| |-0.0| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.1| |-0.2| |-0.2| |-0.1| |-0.1| |0.4| |0.3| |-0.1| |-0.2| |0.0| |-0.0| |-0.2| |-0.2| |-0.2| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.1| |-0.2| |0.2| | | |0.1| |0.2| |-0.1| |-0.3| |-0.2| |-0.3| |-0.2| |-0.2| |-0.1| |-0.3| |-0.2| |-0.2| |-0.2| |0.2| |-0.2| |-0.1| |-0.1| |-0.3| |-0.3| |-0.2| |-0.3| |-0.3| |-0.1| |-0.1| |-0.1| |0.2| |-0.1| |-0.0| |0.2| |-0.1| |-0.2| |-0.3| |-0.3| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.3| |-0.1| |0.0| | | |-0.1| |-0.2| |-0.5| |-0.3| |-0.1| |-0.2| |-0.2| |-0.3| |-0.4| |-0.2| |-0.1| |-0.2| |-0.2| |-0.3| |-0.2| |-0.4| |-0.3| |-0.2| |-0.2| |-0.2| |-0.3| |-0.1| |-0.2| |-0.2| |-0.0| |-0.2| |-0.3| |-0.1| |-0.1| |-0.2| |-0.3| |-0.3| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.1| |-0.2| |0.5| |0.3| |0.1| | | |-0.1| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.1| |-0.0| |0.0| |0.3| |-0.2| |-0.2| |-0.1| |-0.1| |-0.2| |-0.2| |-0.2| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-1.0| |-0.9| |-1.4| |-1.0| |-1.1| |-1.0| | | |-1.6| |-0.6| |-1.2| |-0.5| |-0.6| |-1.4| |-1.3| |-1.3| |-1.2| |-1.2| |-1.2| |-1.4| |-0.4| |-1.4| |-0.6| |-1.4| |-0.4| |-0.6| |-1.3| |-0.9| |-1.0| |-1.2| |-1.2| |-1.2| |-1.4| |-1.2| |-1.2| |-1.2| |-1.4| |-1.3| |-0.5| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.2| |-0.0| |0.0| |-0.0| |-0.0| |-0.0| | | |-0.2| |-0.1| |-0.2| |-0.1| |-0.3| |-0.3| |-0.2| |-0.2| |-0.2| |-0.1| |-0.1| |-0.1| |-0.3| |-0.1| |-0.2| |-0.1| |-0.1| |-0.3| |-0.1| |-0.0| |-0.0| |-0.0| |-0.1| |-0.3| |-0.0| |-0.1| |-0.1| |-0.2| |-0.2| |-0.1| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.6| |-1.0| |-0.7| |-0.7| |-0.8| |-0.9| |-0.1| |-1.3| | | |-0.5| |-0.5| |-0.5| |-1.3| |-1.3| |-1.2| |-0.9| |-0.9| |-0.9| |-1.3| |-0.4| |-1.2| |-0.5| |-1.3| |-0.4| |-0.5| |-1.2| |-0.9| |-0.8| |-0.8| |-0.8| |-0.9| |-1.3| |-0.7| |-0.9| |-0.9| |-1.3| |-1.2| |-0.4| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.7| |-0.1| |-0.6| |-0.0| |-0.1| |-0.0| |-0.1| |-0.3| |-0.1| | | |-0.1| |-0.1| |-0.3| |-0.3| |-0.3| |-0.1| |-0.1| |-0.1| |-0.3| |-0.1| |-0.3| |-0.2| |-0.3| |-0.1| |-0.1| |-0.3| |-0.0| |-0.0| |-0.0| |-0.0| |-0.1| |-0.3| |-0.7| |-0.1| |-0.1| |-0.3| |-0.3| |-0.1| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.8| |-0.9| |-1.0| |-0.8| |-0.9| |-0.8| |-0.4| |-1.1| |-0.3| |-0.5| | | |-0.5| |-0.6| |-1.0| |-0.6| |-0.9| |-0.8| |-0.9| |-0.5| |-0.4| |-1.2| |-0.5| |-0.5| |-0.4| |-0.4| |-0.6| |-0.9| |-0.9| |-0.8| |-0.8| |-0.9| |-0.6| |-0.9| |-0.9| |-0.9| |-0.5| |-0.5| |-0.4| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-1.0| |-1.1| |-1.0| |-1.0| |-1.1| |-1.1| |-0.4| |-1.2| |-0.4| |-0.5| |-0.5| | | |-0.8| |-1.1| |-0.6| |-1.1| |-1.1| |-1.1| |-0.5| |-0.4| |-1.3| |-0.5| |-0.6| |-0.4| |-0.4| |-0.6| |-1.0| |-1.0| |-1.1| |-1.0| |-1.1| |-0.6| |-1.0| |-1.1| |-1.1| |-0.5| |-0.5| |-0.4| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |0.3| |-0.0| |-0.0| |-0.1| |-0.0| |-0.1| |-0.3| |-0.1| |-0.2| |-0.1| |-0.2| | | |-0.3| |-0.3| |-0.1| |-0.1| |-0.1| |-0.2| |-0.1| |-0.3| |-0.1| |-0.3| |-0.1| |-0.1| |-0.3| |-0.0| |-0.0| |-0.0| |-0.0| |-0.1| |-0.3| |-0.0| |-0.1| |-0.1| |-0.2| |-0.1| |-0.1| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.2| |-0.0| |-0.0| |-0.1| |-0.0| |-0.1| |-0.3| |-0.2| |-0.1| |-0.2| |-0.1| |-0.3| | | |-0.2| |-0.1| |-0.1| |-0.2| |-0.2| |-0.1| |-0.3| |-0.1| |-0.2| |-0.1| |-0.2| |-0.3| |-0.0| |-0.0| |-0.0| |-0.0| |-0.2| |-0.3| |-0.0| |-0.1| |-0.2| |-0.2| |-0.2| |-0.1| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.1| |-0.0| |-0.0| |-0.1| |-0.0| |-0.1| |-0.3| |-0.1| |-0.2| |-0.1| |-0.2| |-0.3| |-0.3| | | |-0.1| |-0.1| |-0.1| |-0.2| |-0.1| |-0.3| |-0.1| |-0.3| |-0.1| |-0.1| |-0.3| |-0.0| |-0.0| |-0.0| |-0.0| |-0.1| |-0.3| |-0.0| |-0.1| |-0.1| |-0.3| |-0.2| |-0.1| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.0| |-0.3| |-0.2| |-0.2| |0.1| |-0.2| |-0.2| |-0.3| |-0.3| |-0.3| |-0.2| |-0.2| |-0.3| |-0.3| |-0.3| | | |0.3| |-0.2| |-0.3| |-0.2| |-0.4| |-0.2| |-0.3| |-0.2| |-0.3| |-0.3| |-0.2| |-0.2| |-0.2| |-0.2| |0.1| |-0.3| |-0.2| |-0.2| |-0.2| |-0.3| |-0.3| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.0| |-0.2| |-0.2| |-0.2| |0.1| |-0.2| |-0.2| |-0.3| |-0.2| |-0.2| |-0.2| |-0.2| |-0.3| |-0.3| |-0.3| |0.3| | | |0.1| |-0.3| |-0.2| |-0.4| |-0.2| |-0.3| |-0.1| |-0.2| |-0.3| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.3| |-0.2| |-0.2| |-0.2| |-0.2| |-0.3| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.1| |-0.3| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.4| |-0.3| |-0.3| |-0.3| |-0.3| |-0.3| |-0.1| |-0.3| |-0.2| |-0.2| | | |-0.3| |-0.2| |-0.4| |-0.2| |-0.3| |-0.2| |-0.2| |-0.3| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |0.1| |0.1| |-0.3| |-0.3| |-0.2| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.1| |-0.0| |-0.0| |-0.1| |-0.0| |-0.1| |-0.3| |-0.1| |-0.2| |-0.1| |-0.1| |-0.3| |-0.3| |-0.3| |-0.1| |-0.1| |-0.1| | | |-0.1| |-0.3| |-0.1| |-0.3| |-0.1| |-0.1| |-0.3| |-0.0| |-0.0| |-0.0| |-0.0| |-0.1| |-0.3| |-0.0| |-0.1| |-0.1| |-0.3| |-0.3| |-0.1| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-1.2| |-1.2| |-1.2| |-1.2| |-1.2| |-1.2| |-1.3| |-1.4| |-1.4| |-1.4| |-1.4| |-1.4| |-1.4| |-1.4| |-1.4| |-1.2| |-1.2| |-1.2| |-1.4| | | |-1.4| |-1.4| |-1.4| |-0.8| |-1.3| |-1.4| |-1.2| |-1.2| |-1.2| |-1.2| |-1.2| |-1.4| |-1.2| |-1.2| |-1.2| |-1.4| |-1.4| |-0.4| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.1| |-0.0| |-0.1| |-0.1| |-0.0| |-0.1| |-0.3| |-0.1| |-0.1| |-0.1| |-0.1| |-0.3| |-0.3| |-0.2| |-0.1| |-0.2| |-0.1| |-0.1| |-0.1| | | |-0.1| |-0.2| |-0.1| |-0.1| |-0.3| |-0.1| |-0.1| |-0.0| |-0.1| |-0.1| |-0.0| |-0.0| |-0.1| |-0.1| |-0.2| |-0.1| |-0.1| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-1.0| |-0.9| |-1.0| |-1.0| |-0.6| |-1.1| |-0.4| |-0.6| |-0.4| |-0.6| |-0.4| |-0.4| |-0.5| |-0.5| |-0.5| |-0.9| |-0.9| |-0.6| |-0.5| |-0.4| |-0.6| | | |-0.6| |-0.4| |-0.4| |-0.5| |-1.0| |-1.0| |-1.1| |-1.0| |-0.9| |-0.5| |-1.0| |-0.4| |-0.6| |-0.5| |-0.5| |-0.4| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.1| |-0.0| |-0.0| |-0.1| |-0.0| |-0.1| |-0.3| |-0.1| |-0.1| |-0.1| |-0.1| |-0.3| |-0.3| |-0.3| |-0.1| |-0.1| |-0.1| |-0.2| |-0.1| |-0.3| |-0.2| | | |-0.1| |-0.1| |-0.3| |-0.0| |-0.0| |-0.0| |-0.0| |-0.1| |-0.3| |-0.0| |-0.1| |-0.1| |-0.3| |-0.2| |-0.1| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-1.2| |-1.2| |-1.2| |-1.2| |-1.2| |-1.2| |-0.6| |-1.4| |-0.6| |-1.4| |-1.3| |-1.3| |-1.4| |-1.4| |-1.4| |-1.2| |-1.2| |-1.2| |-1.4| |-0.4| |-1.4| |-1.3| |-1.4| | | |-0.6| |-1.4| |-1.2| |-1.2| |-1.2| |-1.2| |-1.2| |-1.4| |-1.2| |-1.2| |-1.2| |-1.4| |-1.4| |-0.4| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-1.0| |-1.1| |-1.0| |-1.0| |-1.1| |-1.1| |-0.4| |-1.3| |-0.5| |-0.5| |-0.5| |-0.5| |-1.3| |-1.3| |-1.3| |-1.1| |-1.1| |-1.1| |-1.2| |-0.4| |-1.3| |-0.5| |-1.3| |-0.4| | | |-1.3| |-1.0| |-1.0| |-1.0| |-1.0| |-1.1| |-1.3| |-1.0| |-1.1| |-1.1| |-1.3| |-0.6| |-0.4| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.1| |-0.0| |-0.0| |-0.1| |-0.0| |-0.1| |-0.3| |-0.1| |-0.1| |-0.1| |-0.1| |-0.3| |-0.3| |-0.3| |-0.1| |-0.1| |-0.1| |-0.2| |-0.1| |-0.3| |-0.1| |-0.3| |-0.1| |-0.1| | | |-0.0| |-0.0| |-0.0| |-0.0| |-0.1| |-0.3| |-0.0| |-0.1| |-0.1| |-0.2| |-0.1| |-0.1| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.2| |-0.1| |-0.1| |-0.2| |-0.1| |-0.2| |-0.2| |-0.2| |-0.2| |-0.3| |-0.2| |-0.2| |-0.3| |-0.2| |-0.2| |-0.2| |-0.2| |-0.3| |-0.2| |-0.3| |-0.2| |-0.2| |-0.2| |-0.3| |-0.3| | | |0.4| |-0.1| |-0.1| |-0.2| |-0.3| |-0.1| |-0.2| |-0.2| |-0.3| |-0.3| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.2| |-0.1| |-0.1| |-0.1| |-0.1| |-0.2| |-0.3| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.3| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |0.4| | | |-0.1| |-0.1| |-0.2| |-0.2| |-0.1| |-0.1| |-0.2| |-0.2| |-0.2| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.2| |0.4| |0.2| |-0.1| |0.2| |-0.1| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.1| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.1| |-0.1| | | |0.1| |-0.2| |-0.2| |-0.1| |-0.1| |-0.2| |-0.2| |-0.2| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.2| |0.2| |0.3| |-0.1| |0.0| |-0.2| |-0.3| |-0.3| |-0.2| |-0.3| |-0.2| |-0.2| |-0.3| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.3| |-0.2| |-0.2| |-0.2| |-0.3| |-0.2| |-0.1| |-0.1| |-0.1| | | |-0.2| |-0.3| |-0.1| |-0.1| |-0.2| |-0.2| |-0.2| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.1| |0.1| |-0.1| |0.1| |-0.2| |-0.2| |-0.2| |-0.4| |-0.3| |-0.3| |-0.2| |-0.2| |-0.3| |-0.3| |-0.3| |-0.2| |-0.2| |-0.1| |-0.3| |-0.2| |-0.4| |-0.2| |-0.2| |-0.2| |-0.3| |-0.3| |-0.2| |-0.2| |-0.2| |-0.1| | | |-0.3| |-0.2| |-0.1| |-0.2| |-0.3| |-0.2| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.1| |-0.0| |-0.0| |-0.1| |-0.1| |-0.1| |-0.3| |-0.1| |-0.2| |-0.2| |-0.2| |-0.3| |-0.3| |-0.3| |-0.1| |-0.1| |-0.1| |-0.2| |-0.1| |-0.1| |-0.1| |-0.3| |-0.1| |-0.1| |-0.3| |-0.0| |-0.0| |-0.0| |-0.0| |-0.1| | | |0.2| |-0.1| |-0.1| |-0.2| |-0.2| |-0.1| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.1| |-0.1| |-0.0| |-0.0| |-0.0| |-0.2| |-0.2| |-0.2| |-0.1| |-0.2| |-0.1| |-0.2| |-0.2| |-0.1| |-0.1| |-0.1| |0.2| |-0.2| |-0.2| |-0.1| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.0| |-0.0| |0.1| |-0.0| |0.1| |-0.0| | | |0.2| |-0.1| |-0.2| |-0.2| |-0.1| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.0| |-0.2| |-0.1| |-0.1| |-0.2| |-0.1| |-0.2| |-0.4| |-0.2| |-0.2| |-0.2| |-0.2| |-0.3| |-0.4| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.4| |-0.2| |-0.2| |-0.2| |-0.2| |-0.3| |-0.2| |-0.1| |-0.1| |-0.1| |-0.2| |-0.3| |-0.1| | | |-0.2| |-0.3| |-0.2| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.1| |0.3| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.4| |-0.3| |-0.3| |-0.3| |-0.3| |-0.3| |-0.3| |-0.3| |-0.2| |-0.3| |-0.2| |-0.3| |-0.2| |-0.4| |-0.3| |-0.3| |-0.2| |-0.3| |-0.3| |-0.2| |-0.2| |-0.2| |-0.2| |-0.2| |-0.3| |-0.2| |-0.2| | | |-0.3| |-0.3| |-0.2| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |0.0| |-0.1| |-0.0| |-0.0| |-0.1| |-0.0| |-0.1| |-0.3| |-0.1| |-0.1| |-0.1| |-0.1| |-0.3| |-0.3| |-0.3| |-0.1| |-0.1| |-0.1| |-0.3| |-0.1| |-0.3| |-0.2| |-0.3| |-0.1| |-0.1| |-0.3| |-0.0| |-0.0| |-0.0| |-0.0| |-0.1| |-0.3| |-0.0| |-0.1| |-0.1| | | |-0.2| |-0.1| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-0.0| |-0.1| |-0.0| |-0.0| |-0.1| |-0.0| |-0.1| |-0.1| |-0.1| |-0.1| |-0.1| |-0.1| |-0.1| |-0.2| |-0.2| |0.0| |0.1| |0.1| |-0.2| |-0.1| |-0.1| |-0.2| |-0.2| |-0.1| |-0.1| |-0.3| |0.2| |0.2| |-0.0| |-0.0| |-0.1| |-0.2| |-0.0| |0.1| |-0.0| |-0.3| | | |-0.1| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |-1.2| |-1.2| |-1.2| |-1.2| |-1.2| |-1.2| |-1.4| |-1.4| |-1.4| |-1.4| |-1.4| |-1.4| |-1.4| |-1.4| |-1.4| |-1.2| |-1.2| |-1.2| |-1.4| |-1.3| |-1.4| |-1.4| |-1.4| |-1.4| |-1.4| |-1.4| |-1.2| |-1.2| |-1.2| |-1.2| |-1.2| |-1.4| |-1.2| |-1.2| |-1.2| |-1.4| |-1.4| | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

cyguyoenruesfrdeptzhviswkokktrafnosvelbnckbfamrpsurfilidmscaitroarhehruktethja

TargetLanguage

en ru es fr de pt zh vi ja sw ko cy kk tr af no sv el bn ckb fa gu mr ps ur fil id ms ca it ro ar he hr uk te th yo

Source Language

- Figure C.2: The language transfer scores measure the benefit to the target language of (co-)training with the source language. We bold the top-5 source languages for each target language. The top left 9x9 are the Bilingual Transfer Score computed directly, whereas the rest are estimated from the Finetuning Adaptation Score. While English is the best source language for many of of languages, we find language similarity is highly predictive of these scores.

