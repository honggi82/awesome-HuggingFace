# arXiv:2511.14295v2[cs.CL]9Dec2025

## AraLingBench A Human-Annotated Benchmark for Evaluating Arabic Linguistic Capabilities of Large Language Models

###### Mohamad Zbib∗

King Abdullah University of Science and Technology (KAUST) American University of Beirut (AUB) mohamad.zbib@kaust.edu.sa

###### Hasan Abed Al Kader Hammoud∗

King Abdullah University of Science and Technology (KAUST) hasanabedalkader.hammoud@kaust.edu.sa

###### Sina Mukalled

American University of Beirut (AUB)

###### Nadine Rizk

American University of Beirut (AUB)

###### Fatima Karnib

American University of Beirut (AUB)

###### Issam Lakkis

American University of Beirut (AUB)

###### Ammar Mohanna

American University of Beirut (AUB)

###### Bernard Ghanem

King Abdullah University of Science and Technology (KAUST)

###### Abstract

We present AraLingBench, a fully human annotated benchmark for evaluating the Arabic linguistic competence of large language models (LLMs). The benchmark spans five core categories: grammar, morphology, spelling, reading comprehension, and syntax, through 150 expert-designed multiple choice questions that directly assess structural language understanding. Evaluating 35 Arabic and bilingual LLMs reveals that current models demonstrate strong surface level proficiency but struggle with deeper grammatical and syntactic reasoning. AraLingBench highlights a persistent gap between high scores on knowledgebased benchmarks and true linguistic mastery, showing that many models succeed through memorization or pattern recognition rather than authentic comprehension. By isolating and measuring fundamental linguistic skills, AraLingBench provides a diagnostic framework for developing Arabic LLMs. The full evaluation code is publicly available on GitHub.

∗Equal contribution. This work was done during Mohammad’s internship at KAUST.

###### :ﺔﻠﻤﺠﻟا ﻲﻓ ﻲﻌﻳﺪﺒﻟا ﻦﺴﺤﻤﻟا ﻮﻫ ﺎﻣّ ؟"ﺮﻜﻨﻤﻟا ﻦﻋ ﻢﻫﺎﻬﻨﻳو ،فوﺮﻌﻤﻟﺎﺑ ﻢﻫﺮﻣﺄﻳ"

###### :ﺔﻠﻤﺟ ﻲﻓ "مﺎﺣدزا" ﺔﻤﻠﻛ بﺮﻋا "قاﻮﺳﻷا ﻲﻓٌﺪﻳﺪﺷمﺎﺣدزا ﻢﺣدﺰﻳ"ٌ

What is the rhetorical (stylistic) device used in the sentence: "He enjoins what is right and forbids what is wrong"?

###### ؟"خَ ﺬَ َﺑ" ﻞﻌﻔﻟا ﻦﻣ ﻞﻋﺎﻔﻟا ﻢﺳا ﻮﻫ

What is the active participle (ism al-fāʿil) of the verb "badhakha" (to be luxurious or extravagant)?

Parse (analyze the grammatical function of) the word (crowding) in the sentence: “There is heavy crowding in the markets”.

[Figure 1]

#### AraLingBench

###### ﲆﻋ ﺎﻬﻛﺮﺗ ﻲﺘﻟا ﻒﺳﻮﻳ ﺔﻗرﻮﻟ ﻖﻤﻋﻷا ﺔﻟﻻﺪﻟا ﺎﻣ ﺎﻨﻫ ثﺪﺤﻳ ﺎﻣّﻞﻛ" ﺎﻬﻴﻠﻋ ﺐﺘُﻛ ﻲﺘﻟاو ﺔﻟوﺎﻄﻟا ؟ﻲﺋاوﺮﻟا ﺪﻬﺸﻤﻟا قﺎﻴﺳ ﻲﻓ "ﺔﻓﺪﺻ ﺲﻴﻟ

###### ؟"ءاﺮﻤﺳ" ﻰّﻨﺜﻤﻟ ﺎًﻴﺋﻼﻣإ ﺔﺤﻴﺤﺼﻟا ﺔﺑﺎﺘﻜﻟا ﻲﻫ ﺎﻣ

What is the correct spelling of the dual form of “samrāʾ” (dark-skinned woman)?

What is the deeper significance of Youssef’s note left on the table, which read “Everything that happens here is not a coincidence,” in the context of the novel’s scene?

- Figure 1: Sample Questions from AraLingBench. Example items illustrating the five linguistic categories: grammar, morphology, spelling, reading comprehension, and syntax. Each question targets a distinct aspect of Arabic linguistic competence and is crafted by expert annotators to assess genuine linguistic understanding.

###### 1 Introduction

Natural language processing (NLP) in Arabic has seen remarkable progress in recent years, supported by the growing availability of large language models (LLMs) trained specifically for Arabic or Arabic-English bilingual contexts. These models range from general-purpose systems to domain-focused variants (Al-Khalifa et al., 2025; Inoue et al., 2021), and collectively have advanced Arabic NLP in tasks of generation, reasoning, and understanding. However, this rapid growth in model development has not been accompanied by an equally systematic approach to evaluation. The field still lacks reliable methods to assess how well these models truly understand Arabic at the linguistic level.

Existing benchmarks such as BALSAM (Almatham et al., 2025), CamelEval (Qian et al., 2024), and 3LM (Boussaha et al., 2025) offer a broad coverage of reasoning, comprehension, and task-specific evaluation. However, they share a common limitation: their strong emphasis on factual recall and problem-solving often comes at the expense of testing fundamental language competence. Datasets like EXAMS and ArabicMMLU primarily assess academic or general knowledge, rarely examining whether models can correctly handle grammatical agreement, morphological derivation, or orthographic conventions. Even recent efforts that incorporate dialectal content, such as DialectalArabicMMLU (Altakrori et al., 2025), remain largely within this knowledge-oriented paradigm. Consequently, the community lacks a benchmark that isolates and measures the linguistic foundations underlying genuine Arabic language understanding.

Arabic presents unique linguistic challenges that differ substantially from those in English and other IndoEuropean languages. It is characterized by complex morphology, a rich system of inflection and derivation, and flexible syntax that allows multiple valid word orders. Thus, mastery of Arabic requires competence across several interrelated skills presented in Figure 1: grammar (Nahw), morphology (Sarf ), orthography (Imlaa), reading comprehension (Fahm al-logha), and syntactic structure (Tarkib Lughawi). However, despite their central importance to the language, these aspects are often assumed rather than directly tested in current evaluation practices.

To address this gap, we introduce AraLingBench, a human-annotated benchmark specifically designed to evaluate Arabic LLMs on their core linguistic competence. The benchmark consists of 150 multiple-choice questions authored by experts, distributed evenly across the five categories above. Each question is written

and reviewed by trained Arabic linguists to ensure linguistic validity, pedagogical clarity, and a focus on reasoning grounded in the language itself rather than factual recall. Figure 1 shows representative examples of these categories, illustrating how AraLingBench isolates different dimensions of Arabic linguistic ability.

We evaluate more than thirty Arabic and bilingual LLMs using AraLingBench and uncover consistent patterns across models. While overall performance correlates with results from general benchmarks such as ArabicMMLU, many high-scoring models achieve these results through surface-level pattern recognition or retrieval-based shortcuts rather than true linguistic understanding. These models often perform well on spelling and reading comprehension, but struggle with grammar, morphology, and syntax the very skills that underlie authentic Arabic proficiency.

By focusing on the linguistic core of Arabic understanding, AraLingBench provides a framework to diagnose where models succeed and where they fall short. It enables researchers to distinguish between models that merely reproduce fluent Arabic text and those that internalize its grammatical and morphological structure. In doing so, AraLingBench moves Arabic NLP evaluation beyond knowledge assessment toward a more complete picture of language understanding.

Contributions. This work makes three main contributions:

- 1. We introduce AraLingBench, a fully human-annotated benchmark dedicated to evaluating fundamental Arabic linguistic competence across grammar, morphology, spelling, reading comprehension, and syntax.
- 2. We conduct a large-scale evaluation of more than 30 Arabic and bilingual LLMs, revealing systematic weaknesses in grammatical and morphological reasoning despite strong performance on general benchmarks.
- 3. We analyze cross-benchmark relationships and show that AraLingBench captures a distinct dimension of model ability, separating genuine linguistic understanding from surface-level or retrieval-based performance.

###### 2 Related Work

The rapid development of Arabic large language models (LLMs) has transformed the landscape of natural language processing for Arabic-speaking communities, creating an urgent need for more comprehensive and linguistically grounded evaluation frameworks. We organize this review into three main areas: (1) the evolution of Arabic language models, (2) the expansion of Arabic evaluation benchmarks, and (3) how AraLingBench addresses the persistent gap in linguistic evaluation.

###### 2.1 Arabic Language Models

The development of Arabic language models has progressed through several architectural generations. The encoder-based era began with AraBERT (Antoun et al., 2020), which adapted BERT’s architecture for Arabic text, followed by specialized variants such as MARBERT (Abdul-Mageed et al., 2021) for dialectal processing, ARBERT (Panboonyuen, 2025) for social media content, and CAMeLBERT (Inoue et al., 2021). These models established strong baselines for Arabic NLP tasks but remained limited to discriminative applications such as classification and sequence labeling.

The transition to generative architectures marked a major turning point for Arabic NLP. AraGPT2 introduced Arabic text generation capabilities, paving the way for larger bilingual and multilingual systems. JAIS (Sengupta et al., 2023) (13B–30B parameters) stands among the most ambitious bilingual Arabic–English efforts, while ALLaM (Bari et al., 2025) (7B–13B) emphasizes instruction tuning and task generalization. Similarly, AceGPT (Huang et al., 2024) focuses on cross-lingual transfer learning built upon LLaMA-2, while models such as Hala (Hammoud et al., 2025) leverage synthetic bilingual data for fine-tuning.

Recent contributions have diversified further. Yehia (Navid-AI, 2025) has consistently ranked among topperforming Arabic models across multiple benchmarks in our evaluation. Fanar (Team et al., 2025), developed

Table 1: Comparison of major Arabic language models showing architecture types, parameter scales, and training approaches.

Model Size Architecture Training Approach

AraBERT (Antoun et al., 2020) 110M / 335M Encoder (BERT) Pretrained from scratch MARBERT (Abdul-Mageed et al., 2021) ∼163M Encoder (BERT) Pretrained from scratch CAMeLBERT (Inoue et al., 2021) 110M Encoder (BERT) Pretrained from scratch JABER (Ghaddar et al., 2021) ∼125M Encoder (BERT) Pretrained from scratch AraGPT2 (Antoun et al., 2021) 125M–1.5B Decoder (GPT-2) Pretrained from scratch JAIS (Sengupta et al., 2023) 13B/30B/70B Decoder-only Transformer Pretrained from scratch; +Chat SFT ALLaM (Bari et al., 2025) 7B / 13B / 34B / 70B Decoder-only Transformer Continued pretraining + SFT AceGPT(Huang et al., 2024) 7B / 13B Decoder Continued pretraining (Llama-2) + SFT Hala (Hammoud et al., 2025) 350M/700M/1.2B/9B Decoder SFT on synthetic bilingual supervision Atlas-Chat (Shang et al., 2025) 9B Decoder Dialect-focused SFT Yehia (Navid-AI, 2025) 7B Decoder Instruction tuning (SFT/DPO) Fanar (Team et al., 2025) 9B Decoder Continued pretraining (Gemma-2-9B) + SFT SUHAIL (AI, 2025) 14B Decoder Inst. tuning / LoRA on multilingual base ArabianGPT (Koubâa et al., 2024) 1.5B Decoder Continued pretraining + SFT Jais-Adapted (Inception, 2024) 13B Decoder Instruction tuning (from Llama-2)

###### Notes:

- – Pretrained from scratch: Trained on Arabic data from initialization.
- – Continued pretraining: Further trained from a multilingual base model.
- – SFT: Supervised fine-tuning.
- – LoRA: Low-rank adaptation for efficient fine-tuning.

by QCRI, benefits from extensive computational resources, while domain-specialized systems such as AtlasChat (Shang et al., 2025) and ArabianGPT target dialectal and regional variants, including Moroccan Darija and Gulf Arabic. Collectively, these developments illustrate the growing maturity and variety of approaches in the Arabic LLM ecosystem.

- Table 1 summarizes major Arabic language models, outlining their architectures, scales, and training strategies. This taxonomy highlights the diversity of design choices—ranging from models trained from scratch on Arabic data to those adapted from multilingual foundations through continued pretraining or supervised fine-tuning.

2.2 Evaluation Benchmarks for Arabic

Alongside advances in model development, the Arabic evaluation ecosystem has grown substantially, now encompassing over 40 distinct benchmarks (Alzubaidi et al., 2025). Most of these benchmarks are knowledgeintensive, designed to test factual reasoning or domain expertise rather than linguistic competence. ArabicMMLU (Koto et al., 2024) leads this trend, featuring over 14,000 multiple-choice questions across academic and professional subjects. EXAMS (Hardalov et al., 2020) extends this with standardized test material, while 3LM (Boussaha et al., 2025) bridges Arabic, STEM, and coding domains.

Multi-task evaluation platforms have also emerged to support more holistic assessment. ORCA (Elmadany et al., 2023) provides a unified framework for comparing models across diverse NLP tasks. AlGhafa (Almazrouei et al., 2023) introduces balanced test sets that reflect both classical and modern Arabic, and BALSAM (Almatham et al., 2025) has evolved into a community benchmark platform featuring standardized protocols and public leaderboards.

In addition, domain-specific benchmarks now target specialized forms of reasoning and linguistic variation. For instance, ArabLegalEval (Hijazi et al., 2024) tests legal reasoning, MedArabiQ (Daoud et al., 2025) evaluates medical knowledge, and several financial datasets assess economic understanding. Cultural and dialectal benchmarks, including Palm (Alwajih et al., 2025), AraDiCE (Mousi et al., 2025), and ArabCulture (Sadallah et al., 2025), explore models’ sensitivity to regional variation and sociocultural context.

- Table 2 provides an overview of key Arabic benchmarks, their focus areas, and whether they directly target linguistic skills. Notably, none of the existing benchmarks explicitly evaluate core linguistic competence, which defines the unique contribution of AraLingBench.

Table 2: Comparison of Arabic LLM evaluation benchmarks.

Benchmark Year Primary Focus Type Source Ling.

ArabicMMLU (Koto et al., 2024) 2024 Knowledge (MMLU) MC Native No EXAMS (Hardalov et al., 2020) 2020 Knowledge (Exams) MC Native No AlGhafa (Almazrouei et al., 2023) 2023 Multi-task NLP MC Mixed No ORCA (Elmadany et al., 2023) 2023 Multi-task NLP Mixed Mixed No BALSAM (Almatham et al., 2025) 2025 Platform Mixed Mixed No 3LM (Boussaha et al., 2025) 2025 STEM + Code Mixed Mixed No ArabLegalEval (Hijazi et al., 2024) 2024 Legal Mixed Mixed No MedArabiQ(Daoud et al., 2025) 2025 Medical Mixed Mixed No CamelEval (Qian et al., 2024) 2024 Instruction Open Mixed No AraDiCE (Mousi et al., 2025) 2024 Dialectal + Cultural Mixed Mixed No PalmX (Alwajih et al., 2025) 2025 Cultural MC Mixed No ACVA (Huang et al., 2024) 2023 Cultural Values MC Mixed No AraLingBench (Ours) 2025 Linguistic MC Native Yes

Notes:

- – MC: Multiple-choice.
- – Open: Open-ended.
- – Mixed: Various formats.
- – Ling.: Focuses on core linguistic capabilities.

###### 2.3 Positioning AraLingBench

AraLingBench directly addresses the gap left by existing evaluation benchmarks by focusing on the linguistic dimension of Arabic understanding. Unlike prior benchmarks that assume linguistic competence as a prerequisite for reasoning, AraLingBench makes it the explicit target of evaluation.

The benchmark systematically assesses five fundamental categories—grammar, morphology, spelling and orthography, reading comprehension, and syntax that collectively define Arabic linguistic proficiency. Each category contains carefully constructed questions that isolate specific linguistic phenomena, allowing for precise diagnosis of model strengths and weaknesses. Because all items are authored and reviewed by human experts, AraLingBench provides a high-quality, interpretable resource for evaluating whether models have acquired genuine linguistic competence or not.

In this way, AraLingBench complements existing benchmarks while filling a critical gap: it re-centers linguistic understanding as the foundation of Arabic NLP evaluation.

###### 3 AraLingBench Construction

###### 3.1 Data Collection Process

The construction of AraLingBench followed a rigorous multi-stage process designed to ensure both high quality and linguistic validity. We employed a team of five Arabic linguistics experts from the American University of Beirut, all with advanced training/degrees in Arabic grammar, morphology, and syntax. The process unfolded through four distinct phases:

- Phase 1: Question Generation. Each expert authored original question–answer pairs across the five benchmark categories. While experts could consult reference materials such as textbooks and exam archives for inspiration, they were required to compose novel questions. This approach ensured that all items reflected authentic linguistic phenomena without reproducing existing materials.
- Phase 2: Difficulty and Diversity Filtering. A separate group of native Arabic speakers (non-experts) reviewed the questions to assess clarity and perceived difficulty. Questions were retained if they met two conditions: (1) they were sufficiently challenging to the validation group, indicating non-trivial difficulty, and (2) they represented a diverse range of linguistic phenomena and formats.

- Phase 3: Expert Quality Control. A senior Arabic linguist reviewed all candidate questions for accuracy, phrasing, and category alignment. Items were refined to guarantee unambiguous wording and exactly one correct answer. When a question could belong to multiple categories, it was assigned to the one most closely aligned with its primary linguistic focus.
- Phase 4: Difficulty Annotation. Three independent annotators rated each question on a three-point scale ({1, 2, 3}) corresponding to Easy, Medium, and Hard. The final difficulty label was determined by majority vote.

This process produced 150 carefully validated questions evenly distributed across the five linguistic categories (30 per category), forming a compact, fully human-authored multiple choice benchmark explicitly focused on core Arabic linguistic skills for LLM evaluation.

###### 3.2 Benchmark Statistics

- Figure 2 presents key statistics for AraLingBench. The dataset maintains perfect category balance, with 30 questions per category. The difficulty distribution comprises 50 Easy (33.3%), 74 Medium (49.3%), and 26 Hard (17.3%) items—an intentional skew toward medium difficulty to maximize discriminative power while preserving a range of challenge levels.

Regarding format, 125 questions (83.3%) use four-choice multiple-choice format, while 25 (16.7%) use three options. The correct answer distribution shows moderate variation across positions (A: 34.0%, B: 27.3%, C: 26.0%, D: 12.7%), reflecting natural variation in item design without systematic positional bias.

###### Category Balance

###### Difficulty Distribution

35

100

| |30 30 30 30 30|
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

| |50 (33.3%)<br><br>74 (49.3%)<br><br>26 (17.3%)|
|---|---|
| | |
| | |
| | |
| | |

30

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

80

NumberofQuestions

NumberofQuestions

25

| | |
|---|---|
| | |

60

20

| | |
|---|---|
| | |

15

40

10

| | |
|---|---|
| | |

20

5

0

0

Easy

Medium

Hard

ReadingComprehension

SyntaxStructure

Spelling

Morphology

Grammar

###### Options Distribution

Correct Answer Distribution

140

125 (83.3%)

60

120

51 (34.0%)

50

###### NumberofQuestions

NumberofQuestions

100

41 (27.3%) 39 (26.0%)

40

80

30

60

19 (12.7%)

20

40

25 (16.7%)

10

20

0

0

3 4

A B C D

Number of Options per Question

Correct Answer Position

- Figure 2: Overview of AraLingBench. Category balance, difficulty distribution, question formats, and answer position frequencies. The benchmark maintains balanced coverage across linguistic categories and difficulty levels.

###### 4 Experimental Evaluation

We evaluate over 30 Arabic and bilingual large language models on AraLingBench to assess their linguistic competence. Our analysis is structured around four key research questions, each probing a distinct dimension of model performance and benchmark validity.

###### 4.1 Evaluation Setup

Model Selection. We evaluated 35 models drawn from the Open Arabic LLM Leaderboard, spanning sizes from 350M to 70B parameters. The set includes Arabic-specific models (Hala, Fanar, Yehia), bilingual systems (JAIS, ALLaM), and multilingual base models adapted for Arabic (Qwen2.5, Phi-4). This diversity enables comprehensive comparison across architectures, training strategies, and parameter scales.

Evaluation Protocol. All models were tested using zero-shot prompting in a standardized multiple-choice format. Each question was presented fully in Arabic with answer options labeled A–D. Models produced a single-letter response, which was automatically matched to the gold label. No few-shot examples or chainof-thought prompting were used, ensuring a uniform and fair evaluation setting.

Metrics. We report accuracy (%) per linguistic category and compute overall average accuracy across all five categories. Accuracy provides a direct and interpretable measure of linguistic competence and allows straightforward comparison with existing Arabic benchmarks.

###### 4.2 RQ1: Do Models Exhibit Balanced Linguistic Competence?

Motivation. True Arabic linguistic competence requires mastery across spelling, syntax, morphology, grammar, and reading comprehension. We analyze whether models achieve balanced proficiency or exhibit specialized strengths and weaknesses.

- Results. Table 3 reports per-category performance for all evaluated models. Several trends emerge:

- 1. Performance Stratification: Models cluster into three clear tiers. Top performers (Yehia-7B, ALLaM-7B) achieve 72–74% average accuracy. Mid-tier models (Fanar, Qwen2.5-14B variants) reach 55–62%, while smaller or less specialized models remain below 50%.
- 2. Category Difficulty: Spelling and Reading Comprehension are the easiest (median ∼58–60%), while Syntax proves most challenging (median ∼48%). This aligns with linguistic theory—surfacelevel orthographic patterns are easier to internalize than deep syntactic structures.
- 3. Unbalanced Competence: Even the strongest models show substantial variance across categories. Yehia-7B attains 86.7% on Spelling but only 53.3% on Syntax—a 33-point gap—indicating asymmetric skill development.
- 4. Morphology Challenge: Despite Arabic’s morphological richness, models consistently underperform in this category (median ∼60%). High performers like Yehia reach only 80%, suggesting persistent limitations in representing derivational and inflectional patterns.

- Figure 3 visualizes category-level performance variability. Wide interquartile ranges (15–20 percentage points) indicate substantial heterogeneity in how architectures and training regimes affect each linguistic dimension.

Interpretation. Overall, current Arabic LLMs lack balanced linguistic competence. Most models prioritize surface-level tasks (spelling, lexical retrieval) over structural understanding (syntax, morphology). This imbalance likely reflects training corpus composition, where correct orthography and frequent lexical patterns are abundant but explicit grammatical or morphological signals are sparse.

Model Spelling Syntax Morphology Grammar Reading Comp. Average Yehia-7B-preview 86.7 53.3 80.0 80.0 70.0 74.0 ALLaM-7B-Instruct-preview 86.7 60.0 73.3 73.3 76.7 74.0 Yehia-7B-Reasoning-preview 80.0 50.0 80.0 76.7 73.3 72.0 Yehia-7B-DPO-Reasoning-preview 80.0 50.0 80.0 76.7 73.3 72.0 Yehia-7B-SFT-Reasoning-preview 76.7 36.7 66.7 76.7 73.3 66.0 tempmotacilla-cinerea-0308 63.3 60.0 60.0 60.0 70.0 62.7 Qwen2.5-Lumen-14B 70.0 56.7 63.3 60.0 60.0 62.0 Saka-14B 66.7 56.7 63.3 60.0 60.0 61.3 SUHAIL-14B-preview 60.0 60.0 70.0 63.3 53.3 61.3 lambda-qwen2.5-14b-dpo-test 70.0 60.0 60.0 60.0 56.7 61.3 Qwen2.5-14B 60.0 46.7 60.0 70.0 66.7 60.7 Qwen2.5-14B-Gutenberg-1e-Delta 70.0 56.7 60.0 60.0 56.7 60.7 Rombos-LLM-V2.6-Qwen-14b 70.0 60.0 60.0 56.7 56.7 60.7 Fanar-1-9B-Instruct 60.0 43.3 73.3 63.3 60.0 60.0 Qwen2.5-14B-Instruct 66.7 56.7 60.0 56.7 53.3 58.7 Qwen3-8B-Base 60.0 60.0 60.0 60.0 43.3 56.7 Hala-9B 63.3 40.0 63.3 46.7 60.0 54.7 emirati-14b-v2 63.3 46.7 60.0 56.7 46.7 54.7 Qwen2.5-7B-Instruct-abliterated-v2 53.3 53.3 43.3 56.7 60.0 53.3 SILMA-9B-Instruct-v1.0 53.3 53.3 66.7 56.7 36.7 53.3 T.E-8.1 43.3 43.3 60.0 50.0 66.7 52.7 Marco-LLM-AR-V2 53.3 46.7 66.7 56.7 36.7 52.0 recoilme-gemma-2-9B-v0.4 56.7 40.0 63.3 53.3 46.7 52.0 Qwen2.5-7B-Instruct 50.0 50.0 40.0 50.0 66.7 51.3 Qwen2.5-7B-Instruct-Uncensored 46.7 40.0 46.7 50.0 66.7 50.0 Josiefied-Qwen2.5-7B 43.3 50.0 40.0 46.7 60.0 48.0 Qwen2.5-7B-Instruct-abliterated 46.7 40.0 40.0 46.7 66.7 48.0 SauerkrautLM-Nemo-12b-Instruct 46.7 40.0 46.7 43.3 56.7 46.7 Phi-4-mini-instruct 46.7 43.3 53.3 43.3 43.3 46.0 Qwen2-7B-Instruct 50.0 46.7 36.7 43.3 50.0 45.3 Marco-LLM-AR-V4 53.3 33.3 40.0 50.0 36.7 42.7 Qwen2.5-3B-Instruct 36.7 50.0 26.7 50.0 36.7 40.0 Hala-1.2B 26.7 43.3 30.0 36.7 56.7 38.7 Hala-350M 36.7 43.3 30.0 46.7 36.7 38.7 Hala-700M 43.3 36.7 23.3 33.3 53.3 38.0

- Table 3: Model performance across AraLingBench categories. Top-performing models reach 72–74% accuracy but show large intra-category variance, with Syntax persistently posing the greatest challenge, top models are highlited in red.

###### Distribution of Model Accuracy Across Categories

100

80

Accuracy(%)

60

40

20

0

Spelling Syntax Morphology Grammar Reading Comprehension

- Figure 3: Category-level accuracy distribution. Models perform best on Spelling and Reading Comprehension, with Syntax remaining the most difficult8category.

###### 4.3 RQ2: How Do Linguistic Skills Correlate?

Motivation. Understanding how linguistic categories relate helps determine whether Arabic competence develops holistically or through independent skill acquisition. Strong correlations suggest shared underlying representations, whereas weak ones imply category-specific learning.

Results. Figure 4 presents the correlation matrix across the five linguistic categories. Several clear relationships emerge:

- 1. Grammar–Morphology: High correlation (r = 0.83) indicates that these two skills develop in tandem. Models capable of accurate morphological parsing tend to perform well in grammatical reasoning, reflecting their shared dependence on word structure and inflection.
- 2. Spelling–Grammar: A similarly high correlation (r = 0.86) underscores the link between orthographic accuracy and grammatical knowledge. Correct spelling often depends on syntactic case or agreement, showing that orthography and grammar are mutually reinforcing.
- 3. Spelling–Reading Comprehension: A moderate correlation (r ≈ 0.51) suggests that lexical precision supports text comprehension, though understanding sentences extends beyond word-level recognition.
- 4. Syntax Independence: Syntax exhibits the weakest correlations with other categories (r ≈ 0.13− 0.40), suggesting that syntactic reasoning relies on distinct mechanisms. This aligns with linguistic theory: syntactic competence requires compositional and hierarchical understanding that does not emerge directly from surface-level distributional patterns.

###### Category Correlation Matrix

1.00

|[Figure 2]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

[Figure 3]

Spelling

1.00

0.75

0.50

CorrelationCoefficient

Syntax

0.47 1.00

0.25

Morphology

0.00

0.83 0.40 1.00

0.25

Grammar

0.86 0.43 0.83 1.00

0.50

0.75

Reading Comprehension

0.51 0.13 0.39 0.49 1.00

1.00

Spelling SyntaxMorphology Grammar Reading Comprehension

- Figure 4: Inter-category correlations. Grammar and Morphology show the strongest relationship (r = 0.80), while Syntax remains comparatively independent, suggesting distinct representational mechanisms.

Interpretation. These correlations indicate that Arabic linguistic competence in LLMs emerges through partially overlapping clusters of skills rather than as a single unified construct. Morphology and grammar

- Table 4: Cross-benchmark comparison. Performance of major Arabic LLMs across eight benchmarks shows limited predictive power between knowledge-based and linguistic evaluations.

Model AlGhafa ArabicMMLU EXAMS MadinahQA AraTrust ALRAGE ArbMMLU-HT AraLingBench Navid-AI/Yehia-7B-preview 70.8 64.9 52.1 54.4 87.5 76.6 53.4 74.0 ALLaM-7B-Instruct-preview 69.5 64.9 51.6 54.2 86.9 76.8 52.8 74.0 Yehia-7B-Reasoning-preview 75.2 66.3 52.7 55.0 80.8 73.3 55.3 72.0 Hala-9B 78.3 65.6 53.8 70.4 89.6 – 61.4 54.7 Fanar-1-9B-Instruct 76.4 65.8 52.7 73.4 88.3 77.0 58.6 60.0 Qwen2.5-14B-Instruct 72.3 60.0 53.6 35.6 86.1 78.9 55.7 58.7 Qwen2.5-7B-Instruct 65.6 52.3 39.7 62.7 80.7 77.4 40.3 51.3 Hala-1.2B 59.2 48.6 43.4 41.6 71.7 – 44.2 38.7 Hala-700M 55.5 45.9 40.6 34.7 65.2 – 39.4 38.0 Hala-350M 51.4 41.2 36.9 34.5 52.1 – 35.4 38.7

form a tightly linked subsystem, while syntax stands apart as a specialized ability that may require targeted inductive biases—such as hierarchical attention or structural modeling—beyond those learned from general pretraining.

###### 4.4 RQ3: Does General Benchmark Performance Predict Linguistic Competence?

Motivation. Arabic LLMs are often ranked by performance on general benchmarks such as ArabicMMLU or EXAMS. We examine whether such rankings reflect genuine linguistic understanding or merely general reasoning ability.

- Results. Table 4 compares model scores across AraLingBench and seven established Arabic benchmarks. Several trends are evident:

- 1. Predictive Alignment: High performance on ArabicMMLU does not reliably translate to strong results on AraLingBench. For instance, Hala-9B attains 65.6% on ArabicMMLU but only 54.7% on AraLingBench, showing that factual or reasoning benchmarks may not capture linguistic depth, in fact, the top performers on ArabicMMLU are not necessarily the top performers on AraLingBench.
- 2. Synthetic Training Effects: Models heavily tuned on synthetic instruction data (e.g., Hala family) achieve high scores on knowledge benchmarks but underperform on linguistic ones, suggesting overreliance on memorized or templated patterns.
- 3. Instruction-Tuning Benefits: Models fine-tuned with real instruction data (Yehia, ALLaM) demonstrate stronger alignment between general and linguistic competence, indicating that linguistically grounded supervision improves generalization.
- 4. Domain Mismatch: Some models excel on specialized domains (e.g., legal or medical QA) yet perform poorly on basic linguistic tasks, reflecting narrow domain optimization without underlying language mastery.

Interpretation. Figure 5 highlights nuanced relationships across benchmarks. AraLingBench correlates strongly with language understanding benchmarks such as ArabicMMLU (r = 0.884) though the correlation varies across individual MMLU categories, EXAMS (r = 0.784), and AraTrust (r = 0.751), confirming that linguistic competence underpins general reasoning performance. However, negative correlation with retrievalaugmented evaluation (ALRAGE, r = −0.539) reveals that reliance on retrieval can substitute for genuine understanding. These findings indicate that while linguistic ability transfers positively to most NLP tasks, some optimization pathways especially synthetic or retrieval-heavy ones decouple apparent performance from true linguistic competence.

###### 4.5 RQ4: Does Question Difficulty Align with Model Performance?

Motivation. We test whether human-annotated difficulty levels correspond to model performance patterns and whether difficulty scales consistently across models and linguistic categories.

Results. Figure 6 shows accuracy distributions across Easy, Medium, and Hard questions.

###### Correlation Between AraLingBench and Other Arabic Benchmarks

###### PearsonCorrelationCoefficient

1.0

0.884

0.784

0.8

0.751

0.750

0.699

0.6

0.471

0.4

0.2

0.0

0.2

0.4

Strong correlation (r=0.7)

Moderate correlation (r=0.5)

0.6

-0.539

AlGhafa ArabicMMLU EXAMS MadinahQA AraTrust ALRAGE ArbMMLU-HT

Benchmark

- Figure 5: Cross-benchmark correlations. Pearson coefficients between AraLingBench and seven major Arabic benchmarks reveal strong alignment with language understanding tasks but weak or negative correlation with retrieval-augmented systems.

- 1. Non-Monotonic Scaling: Performance does not degrade consistently with difficulty. Easy questions yield a median accuracy of 58%, Medium 50%, but Hard questions recover to 54%.
- 2. Model-Dependent Difficulty: Certain models (e.g., Qwen3-8B-Base) perform better on Hard items (73.1%) than Medium ones (50.0%), indicating a mismatch between human and model difficulty perception.
- 3. Category Interactions: Difficulty effects vary by category. Hard Morphology questions can be easier for models than Medium ones, while Syntax maintains a consistent downward trend with difficulty.
- 4. Stable High-Performer Trends: Leading models (Yehia, ALLaM) exhibit only modest degradation across levels (76% → 69%), suggesting robust linguistic generalization.

Easy Medium Hard

0

20

40

60

80

100

Accuracy(%)

Distribution of Model Accuracy Across Difficulty Levels

- Figure 6: Performance by difficulty level. Model accuracy does not decrease monotonically with annotated difficulty; Hard questions occasionally yield higher accuracy than Medium ones.

###### Model Performance Heatmap by Category

|86.7|60.0|73.3|73.3|76.7|
|---|---|---|---|---|
|86.7|53.3|80.0|[Figure 4]<br><br>80.0|70.0|
|80.0|50.0|80.0|76.7|73.3|
|80.0|50.0|80.0|76.7|73.3|
|76.7|36.7|66.7|76.7|73.3|
|63.3|60.0|60.0|60.0|70.0|
|70.0|56.7|63.3|60.0|60.0|
|60.0|60.0|70.0|63.3|53.3|
|70.0|60.0|60.0|60.0|56.7|
|66.7|56.7|63.3|60.0|60.0|
|60.0|46.7|60.0|70.0|66.7|
|70.0|56.7|60.0|60.0|56.7|
|70.0|60.0|60.0|56.7|56.7|
|60.0|43.3|73.3|63.3|60.0|
|66.7|56.7|60.0|56.7|53.3|
|60.0|60.0|60.0|60.0|43.3|
|63.3|46.7|60.0|56.7|46.7|
|63.3|40.0|63.3|46.7|60.0|
|53.3|53.3|43.3|56.7|60.0|
|53.3|53.3|66.7|56.7|36.7|
|43.3|43.3|60.0|50.0|66.7|
|56.7|40.0|63.3|53.3|46.7|
|53.3|46.7|66.7|56.7|36.7|
|50.0|50.0|40.0|50.0|66.7|
|46.7|40.0|46.7|50.0|66.7|
|43.3|50.0|40.0|46.7|60.0|
|46.7|40.0|40.0|46.7|66.7|
|46.7|40.0|46.7|43.3|56.7|
|46.7|43.3|53.3|43.3|43.3|
|50.0|46.7|36.7|43.3|50.0|
|53.3|33.3|40.0|50.0|36.7|
|36.7|50.0|26.7|50.0|36.7|
|36.7|43.3|30.0|46.7|36.7|
|26.7|43.3|30.0|36.7|56.7|
|43.3|36.7|23.3|33.3|53.3|

ALLaM-7B-Instruct-preview Yehia-7B-preview

Yehia-7B-Reasoning-preview Yehia-7B-DPO-Reasoning-preview

Yehia-7B-SFT-Reasoning-preview tempmotacilla-cinerea-0308 Qwen2.5-Lumen-14B SUHAIL-14B-preview lambda-qwen2.5-14b-dpo-test Saka-14B

Qwen2.5-14B Qwen2.5-14B-Gutenberg-1e-Delta

100

|[Figure 5]| |
|---|---|
| | |
| | |
| | |
| | |

Rombos-LLM-V2.6-Qwen-14b

80

Fanar-1-9B-Instruct Qwen2.5-14B-Instruct

Accuracy(%)

Qwen3-8B-Base emirati-14b-v2

60

Hala-9B Qwen2.5-7B-Instruct-abliterated-v2

40

SILMA-9B-Instruct-v1.0

T.E-8.1 recoilme-gemma-2-9B-v0.4

20

Marco-LLM-AR-V2 Qwen2.5-7B-Instruct

0

Qwen2.5-7B-Instruct-Uncensored Josiefied-Qwen2.5-7B-Instruct-abliterated-v2

Qwen2.5-7B-Instruct-abliterated

SauerkrautLM-Nemo-12b-Instruct Phi-4-mini-instruct Qwen2-7B-Instruct

Marco-LLM-AR-V4 Qwen2.5-3B-Instruct

Hala-350M

Hala-1.2B Hala-700M

Spelling SyntaxMorphologyGrammarReading Comprehension

- Figure 7: Model performance heatmap across the five AraLingBench linguistic categories. Accuracy values are shown for 35 evaluated models, sorted by weighted average performance. Color intensity ranges from red (low) through yellow (moderate) to green (high).

##### Model Performance Heatmap by Difficulty

|76.0|74.3|69.2|
|---|---|---|
|76.0|74.3|69.2|
|78.0|[Figure 6]<br><br>71.6|61.5|
|78.0|71.6|61.5|
|72.0|63.5|61.5|
|68.0|64.9|46.2|
|70.0|62.2|46.2|
|68.0|59.5|53.8|
|66.0|63.5|46.2|
|66.0|64.9|42.3|
|64.0|63.5|50.0|
|68.0|60.8|46.2|
|64.0|63.5|46.2|
|64.0|56.8|61.5|
|62.0|60.8|46.2|
|58.0|50.0|73.1|
|62.0|52.7|46.2|
|64.0|52.7|42.3|
|54.0|50.0|61.5|
|58.0|54.1|42.3|
|64.0|51.4|34.6|
|58.0|55.4|30.8|
|56.0|50.0|50.0|
|56.0|47.3|53.8|
|58.0|45.9|46.2|
|50.0|45.9|50.0|
|58.0|45.9|34.6|
|56.0|41.9|42.3|
|52.0|45.9|34.6|
|44.0|47.3|42.3|
|40.0|44.6|42.3|
|32.0|39.2|57.7|
|36.0|39.2|42.3|
|40.0|31.1|57.7|
|46.0|28.4|50.0|

ALLaM-7B-Instruct-preview Yehia-7B-preview

Yehia-7B-Reasoning-preview Yehia-7B-DPO-Reasoning-preview

Yehia-7B-SFT-Reasoning-preview tempmotacilla-cinerea-0308

Saka-14B SUHAIL-14B-preview

Rombos-LLM-V2.6-Qwen-14b lambda-qwen2.5-14b-dpo-test Qwen2.5-Lumen-14B

Qwen2.5-14B Qwen2.5-14B-Gutenberg-1e-Delta

100

Fanar-1-9B-Instruct Qwen2.5-14B-Instruct

|[Figure 7]| |
|---|---|
| | |
| | |
| | |
| | |

80

### Accuracy(%)

Qwen3-8B-Base emirati-14b-v2

60

Hala-9B Qwen2.5-7B-Instruct-abliterated-v2

40

SILMA-9B-Instruct-v1.0

20

T.E-8.1 recoilme-gemma-2-9B-v0.4

0

Marco-LLM-AR-V2 Qwen2.5-7B-Instruct

Qwen2.5-7B-Instruct-Uncensored Josiefied-Qwen2.5-7B-Instruct-abliterated-v2

Qwen2.5-7B-Instruct-abliterated

SauerkrautLM-Nemo-12b-Instruct Phi-4-mini-instruct Qwen2-7B-Instruct

Marco-LLM-AR-V4 Qwen2.5-3B-Instruct

Hala-350M

Hala-1.2B Hala-700M

Easy Medium Hard

- Figure 8: Model performance heatmap across AraLingBench difficulty levels (Easy, Medium, Hard) for 35 evaluated models, sorted by weighted average performance. Color intensity ranges from red (low accuracy) through yellow (moderate) to green (high). 13

- 4.6 Detailed Performance Visualization We present two heatmaps to highlight model behavior across categories and difficulty levels.

Category-Level Patterns. Figure 7 shows model accuracy across the five linguistic categories. Top models (Yehia-7B-preview, ALLaM-7B-Instruct-preview, 74.0%) perform consistently well, especially in Spelling (up to 86.7%), suggesting effective instruction tuning. Mid-range models (50–62%) show uneven performance, with strengths in one area offsetting weaknesses in another. For example, SILMA-9B-Instruct-v1.0 scores 66.7% in Morphology despite a 53.3% average. Variability is highest in Reading Comprehension and Grammar, reflecting diverse alignment strategies. Models below 50% struggle most with Morphology and Syntax (some below 30%), showing the challenge of Arabic’s complex structure. Even top models hit a ceiling in Syntax (best: 60.0%).

Difficulty-Level Patterns. Figure 9 presents performance across Easy, Medium, and Hard questions. Top models show stable trends with only slight drops (about 7 points from Easy to Hard). Mid-range models show irregularities: Qwen3-8B-Base scores 58.0% (Easy), 50.0% (Medium), and 73.1% (Hard), indicating nonlinear difficulty perception. Hard items may depend on memorized patterns or domain-specific knowledge, while Medium ones require reasoning less present in pretraining. Low-performing models show unpredictable scaling, sometimes scoring near chance on Easy but higher on Hard. This suggests a mismatch between human and model-perceived difficulty.

Overall, models acquire surface skills more reliably than deep structural understanding, and human difficulty ratings do not always predict actual model challenge.

###### Difficulty Correlation Matrix

1.00

|[Figure 8]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

[Figure 9]

0.75

Easy

1.00

CorrelationCoefficient

0.50

|0.89|1.00|
|---|---|
|0.28|0.31|

0.25

Medium

0.00

0.25

0.50

Hard

1.00

0.75

1.00

Easy Medium Hard

- Figure 9: Difficulty-level correlations. Strong positive relationships (r > 0.65) indicate consistent model ranking despite non-monotonic accuracy patterns.

Interpretation. The non-monotonic scaling pattern reveals that model difficulty diverges from humanperceived complexity. Hard questions may feature constructions or vocabulary frequent in pretraining corpora, making them easier for models, while Medium questions often require integrative reasoning that transformer models handle less effectively. This underscores the need to calibrate benchmark difficulty using both human annotation and pilot testing on representative models.

- 4.7 Summary of Experimental Findings Our evaluation yields four main insights:

- 1. Arabic LLMs display highly uneven linguistic competence, excelling in surface-level abilities (spelling, comprehension) but struggling with deeper structural understanding (syntax, morphology).
- 2. Linguistic skills correlate moderately but not uniformly: grammar and morphology form a tightly coupled subsystem, while syntax remains largely independent.
- 3. General benchmark success does not guarantee linguistic competence. Although overall correlations are strong (r > 0.75), certain training regimes especially retrieval-heavy or synthetic setups inflate benchmark scores without improving true linguistic understanding.
- 4. Human-assigned difficulty labels only partially align with model performance, highlighting the need to jointly consider cognitive and data-driven measures of challenge.

Taken together, these findings establish AraLingBench as a crucial complement to existing Arabic evaluation suites. It isolates fundamental linguistic understanding, exposing competence gaps that remain invisible in knowledge-oriented benchmarks.

###### 5 Conclusion

We introduced AraLingBench, a fully human annotated benchmark designed to evaluate the fundamental linguistic competence of Arabic large language models. By focusing on five core categories grammar, morphology, spelling, reading comprehension, and syntax AraLingBench isolates the linguistic foundations of Arabic understanding that current knowledge-based benchmarks overlook.

Our large-scale evaluation of more than 30 models reveals that high performance on general benchmarks does not necessarily indicate genuine language understanding, as many models continue to struggle with grammatical and morphological reasoning. AraLingBench offers a complementary diagnostic perspective, enabling researchers to distinguish between superficial fluency and true linguistic competence.

We release the benchmark to support the development of Arabic LLMs that not only generate fluent text but also demonstrate authentic mastery of the language’s structure and logic.

###### References

Muhammad Abdul-Mageed, AbdelRahim Elmadany, and El Moatez Billah Nagoudi. ARBERT & MARBERT: Deep bidirectional transformers for Arabic. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.), ACL, pp. 7088–7105, Online, August 2021. ACL. doi: 10.18653/v1/2021.acl-long.551. URL https://aclanthology.org/2021.acl-long.551/.

ZeroOne AI. Suhail-14b-preview, 2025. URL https://huggingface.co/01-ZeroOne/ SUHAIL-14B-preview.

Shahad Al-Khalifa, Nadir Durrani, Hend Al-Khalifa, and Firoj Alam. The landscape of arabic large language models. Commun. ACM, 68(10):54–61, September 2025. ISSN 0001-0782. doi: 10.1145/3737453. URL https://doi.org/10.1145/3737453.

Rawan Nasser Almatham, Kareem Mohamed Darwish, Raghad Al-Rasheed, Waad Thuwaini Alshammari, Muneera Alhoshan, Amal Almazrua, Asma Al Wazrah, Mais Alheraki, Firoj Alam, Preslav Nakov, et al. Balsam: A platform for benchmarking arabic large language models. In ArabicNLP, pp. 258–277, 2025.

Ebtesam Almazrouei, Ruxandra Cojocaru, Michele Baldo, Quentin Malartic, Hamza Alobeidli, Daniele Mazzotta, Guilherme Penedo, Giulia Campesan, Mugariya Farooq, Maitha Alhammadi, Julien Launay, and Badreddine Noune. AlGhafa evaluation benchmark for Arabic language models. In Hassan

Sawaf, Samhaa El-Beltagy, Wajdi Zaghouani, Walid Magdy, Ahmed Abdelali, Nadi Tomeh, Ibrahim Abu Farha, Nizar Habash, Salam Khalifa, Amr Keleg, Hatem Haddad, Imed Zitouni, Khalil Mrini, and Rawan Almatham (eds.), ArabicNLP, pp. 244–275, Singapore (Hybrid), December 2023. ACL. doi: 10.18653/v1/2023.arabicnlp-1.21. URL https://aclanthology.org/2023.arabicnlp-1.21.

Malik H. Altakrori, Nizar Habash, Abdelhakim Freihat, Younes Samih, Kirill Chirkunov, Muhammed AbuOdeh, Radu Florian, Teresa Lynn, Preslav Nakov, and Alham Fikri Aji. Dialectalarabicmmlu: Benchmarking dialectal capabilities in arabic and multilingual language models. arXiv preprint arXiv:2510.27543v1, 2025. URL https://arxiv.org/abs/2510.27543v1.

Fakhraddin Alwajih, Abdellah El Mekki, Samar Mohamed Magdy, AbdelRahim A. Elmadany, Omer Nacar, El Moatez Billah Nagoudi, Reem Abdel-Salam, Hanin Atwany, Youssef Nafea, Abdulfattah Mohammed Yahya, Rahaf Alhamouri, Hamzah A. Alsayadi, Hiba Zayed, Sara Shatnawi, Serry Sibaee, Yasir Echchammakhy, Walid Al-Dhabyani, Marwa Mohamed Ali, Imen Jarraya, Ahmed Oumar El-Shangiti, Aisha Alraeesi, Mohammed Anwar AL-Ghrawi, Abdulrahman S. Al-Batati, Elgizouli Mohamed, Noha Taha Elgindi, Muhammed Saeed, Houdaifa Atou, Issam Ait Yahia, Abdelhak Bouayad, Mohammed Machrouh, Amal Makouar, Dania Alkawi, Mukhtar Mohamed, Safaa Taher Abdelfadil, Amine Ziad Ounnoughene, Anfel Rouabhia, Rwaa Assi, Ahmed Sorkatti, Mohamedou Cheikh Tourad, Anis Koubaa, Ismail Berrada, Mustafa Jarrar, Shady Shehata, and Muhammad Abdul-Mageed. Palm: A culturally inclusive and linguistically diverse dataset for Arabic LLMs. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), ACL, pp. 32871–32894, Vienna, Austria, July 2025. ACL. ISBN 979-889176-251-0. doi: 10.18653/v1/2025.acl-long.1579. URL https://aclanthology.org/2025.acl-long. 1579/.

Ahmed Alzubaidi, Shaikha Alsuwaidi, Basma El Amel Boussaha, Leen AlQadi, Omar Alkaabi, Mohammed Alyafeai, Hamza Alobeidli, and Hakim Hacid. Evaluating arabic large language models: A survey of benchmarks, methods, and gaps. arXiv preprint arXiv:2510.13430v2, 2025. URL https://arxiv.org/ abs/2510.13430v2.

Wissam Antoun, Fady Baly, and Hazem Hajj. AraBERT: Transformer-based model for Arabic language understanding. In Hend Al-Khalifa, Walid Magdy, Kareem Darwish, Tamer Elsayed, and Hamdy Mubarak (eds.), OSACT, pp. 9–15, Marseille, France, May 2020. European Language Resource Association. ISBN 979-10-95546-51-1. URL https://aclanthology.org/2020.osact-1.2/.

Wissam Antoun, Fady Baly, and Hazem Hajj. AraGPT2: Pre-trained transformer for Arabic language generation. In Nizar Habash, Houda Bouamor, Hazem Hajj, Walid Magdy, Wajdi Zaghouani, Fethi Bougares, Nadi Tomeh, Ibrahim Abu Farha, and Samia Touileb (eds.), WANLP, pp. 196–207, Kyiv, Ukraine (Virtual), April 2021. ACL. URL https://aclanthology.org/2021.wanlp-1.21/.

M Saiful Bari, Yazeed Alnumay, Norah A. Alzahrani, Nouf M. Alotaibi, Hisham Abdullah Alyahya, Sultan AlRashed, Faisal Abdulrahman Mirza, Shaykhah Z. Alsubaie, Hassan A. Alahmed, Ghadah Alabduljabbar, Raghad Alkhathran, Yousef Almushayqih, Raneem Alnajim, Salman Alsubaihi, Maryam Al Mansour, Saad Amin Hassan, Dr. Majed Alrubaian, Ali Alammari, Zaki Alawami, Abdulmohsen AlThubaity, Ahmed Abdelali, Jeril Kuriakose, Abdalghani Abujabal, Nora Al-Twairesh, Areeb Alowisheq, and Haidar Khan. ALLam: Large language models for arabic and english. In ICLR, 2025. URL https://openreview.net/forum?id=MscdsFVZrN.

Basma El Amel Boussaha, Leen Al Qadi, Mugariya Farooq, Shaikha Alsuwaidi, Giulia Campesan, Ahmed Alzubaidi, Mohammed Alyafeai, and Hakim Hacid. 3LM: Bridging Arabic, STEM, and code through benchmarking. In Kareem Darwish, Ahmed Ali, Ibrahim Abu Farha, Samia Touileb, Imed Zitouni, Ahmed Abdelali, Sharefah Al-Ghamdi, Sakhar Alkhereyf, Wajdi Zaghouani, Salam Khalifa, Badr AlKhamissi, Rawan Almatham, Injy Hamed, Zaid Alyafeai, Areeb Alowisheq, Go Inoue, Khalil Mrini, and Waad Alshammari (eds.), ArabicNLP, pp. 42–63, Suzhou, China, November 2025. ACL. ISBN 979-8-89176-3524. URL https://aclanthology.org/2025.arabicnlp-main.4/.

Mouath Abu Daoud, Chaimae Abouzahir, Leen Kharouf, Walid Al-Eisawi, Nizar Habash, and Farah E. Shamout. Medarabiq: Benchmarking large language models on arabic medical tasks. arXiv preprint arXiv:2505.03427v2, 2025. URL https://arxiv.org/abs/2505.03427v2.

AbdelRahim Elmadany, ElMoatez Billah Nagoudi, and Muhammad Abdul-Mageed. ORCA: A challenging benchmark for Arabic language understanding. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Findings of ACL, pp. 9559–9586, Toronto, Canada, July 2023. ACL. doi: 10.18653/v1/2023. findings-acl.609. URL https://aclanthology.org/2023.findings-acl.609/.

Abbas Ghaddar, Yimeng Wu, Ahmad Rashid, Khalil Bibi, Mehdi Rezagholizadeh, Chao Xing, Yasheng Wang, Duan Xinyu, Zhefeng Wang, Baoxing Huai, Xin Jiang, Qun Liu, and Philippe Langlais. Jaber: Junior arabic bert. ArXiv, abs/2112.04329, 2021. URL https://api.semanticscholar.org/CorpusID: 244954747.

Hasan Abed Al Kader Hammoud, Mohammad Zbeeb, and Bernard Ghanem. Hala technical report: Building arabic-centric instruction & translation models at scale. arXiv preprint arXiv:2509.14008v1, 2025. URL https://arxiv.org/abs/2509.14008v1.

Momchil Hardalov, Todor Mihaylov, Dimitrina Zlatkova, Yoan Dinkov, Ivan Koychev, and Preslav Nakov. EXAMS: A multi-subject high school examinations dataset for cross-lingual and multilingual question answering. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (eds.), EMNLP, pp. 5427–5444, Online, November 2020. ACL. doi: 10.18653/v1/2020.emnlp-main.438. URL https://aclanthology. org/2020.emnlp-main.438/.

Faris Hijazi, Somayah Alharbi, Abdulaziz AlHussein, Harethah Shairah, Reem Alzahrani, Hebah Alshamlan, George Turkiyyah, and Omar Knio. ArabLegalEval: A multitask benchmark for assessing Arabic legal knowledge in large language models. In Nizar Habash, Houda Bouamor, Ramy Eskander, Nadi Tomeh, Ibrahim Abu Farha, Ahmed Abdelali, Samia Touileb, Injy Hamed, Yaser Onaizan, Bashar Alhafni, Wissam Antoun, Salam Khalifa, Hatem Haddad, Imed Zitouni, Badr AlKhamissi, Rawan Almatham, and Khalil Mrini (eds.), ArabicNLP, pp. 225–249, Bangkok, Thailand, August 2024. ACL. doi: 10.18653/v1/2024. arabicnlp-1.20. URL https://aclanthology.org/2024.arabicnlp-1.20/.

Huang Huang, Fei Yu, Jianqing Zhu, Xuening Sun, Hao Cheng, Song Dingjie, Zhihong Chen, Mosen Alharthi, Bang An, Juncai He, Ziche Liu, Junying Chen, Jianquan Li, Benyou Wang, Lian Zhang, Ruoyu Sun, Xiang Wan, Haizhou Li, and Jinchao Xu. AceGPT, localizing large language models in Arabic. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 8139–8163, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.450. URL https://aclanthology.org/2024.naacl-long.450/.

Inception. Jais family model card. 2024. URL https://huggingface.co/inceptionai/ jais-family-30b-16k-chat/blob/main/README.md.

Go Inoue, Bashar Alhafni, Nurpeiis Baimukan, Houda Bouamor, and Nizar Habash. The interplay of variant, size, and task type in Arabic pre-trained language models. In Nizar Habash, Houda Bouamor, Hazem Hajj, Walid Magdy, Wajdi Zaghouani, Fethi Bougares, Nadi Tomeh, Ibrahim Abu Farha, and Samia Touileb (eds.), Proceedings of the Sixth Arabic Natural Language Processing Workshop, pp. 92–104, Kyiv, Ukraine (Virtual), April 2021. Association for Computational Linguistics. URL https://aclanthology. org/2021.wanlp-1.10/.

Fajri Koto, Haonan Li, Sara Shatnawi, Jad Doughman, Abdelrahman Sadallah, Aisha Alraeesi, Khalid Almubarak, Zaid Alyafeai, Neha Sengupta, Shady Shehata, Nizar Habash, Preslav Nakov, and Timothy Baldwin. ArabicMMLU: Assessing massive multitask language understanding in Arabic. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Findings of ACL 2024, pp. 5622–5640, Bangkok, Thailand, August 2024. ACL. doi: 10.18653/v1/2024.findings-acl.334. URL https://aclanthology.org/2024. findings-acl.334/.

Anis Koubâa, Adel Ammar, Lahouari Ghouti, Omar Najar, and Serry Sibaee. Arabiangpt: Native arabic gpt-based large language model. ArXiv, abs/2402.15313, 2024. URL https://api.semanticscholar. org/CorpusID:267897726.

Basel Mousi, Nadir Durrani, Fatema Ahmad, Md. Arid Hasan, Maram Hasanain, Tameem Kabbani, Fahim Dalvi, Shammur Absar Chowdhury, and Firoj Alam. AraDiCE: Benchmarks for dialectal and cultural capabilities in LLMs. In Owen Rambow, Leo Wanner, Marianna Apidianaki, Hend Al-Khalifa, Barbara Di Eugenio, and Steven Schockaert (eds.), COLING, pp. 4186–4218, Abu Dhabi, UAE, January 2025. ACL. URL https://aclanthology.org/2025.coling-main.283/.

Navid-AI. Yehia 7b preview. https://huggingface.co/Navid-AI/Yehia-7B-preview, 2025. Teerapong Panboonyuen. Albert: Advanced localization and bidirectional encoder representations from

transformers for automotive damage evaluation, 2025. URL https://arxiv.org/abs/2506.10524. Zhaozhi Qian, Faroq Altam, Muhammad Alqurishi, and Riad Souissi. Cameleval: Advancing culturally aligned arabic language models and benchmarks, 2024. URL https://arxiv.org/abs/2409.12623.

Abdelrahman Sadallah, Junior Cedric Tonga, Khalid Almubarak, Saeed Almheiri, Farah Atif, Chatrine Qwaider, Karima Kadaoui, Sara Shatnawi, Yaser Alesh, and Fajri Koto. Commonsense reasoning in Arab culture. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 7695–7710, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 9798-89176-251-0. doi: 10.18653/v1/2025.acl-long.380. URL https://aclanthology.org/2025.acl-long. 380/.

Neha Sengupta, Sunil Kumar Sahu, Bokang Jia, Satheesh Katipomu, Haonan Li, Fajri Koto, William Marshall, Gurpreet Gosal, Cynthia Liu, Zhiming Chen, Osama Mohammed Afzal, Samta Kamboj, Onkar Pandit, Rahul Pal, Lalit Pradhan, Zain Muhammad Mujahid, Massa Baali, Xudong Han, Sondos Mahmoud Bsharat, Alham Fikri Aji, Zhiqiang Shen, Zhengzhong Liu, Natalia Vassilieva, Joel Hestness, Andy Hock, Andrew Feldman, Jonathan Lee, Andrew Jackson, Hector Xuguang Ren, Preslav Nakov, Timothy Baldwin, and Eric Xing. Jais and jais-chat: Arabic-centric foundation and instructiontuned open generative large language models. arXiv preprint arXiv:2308.16149v2, 2023. URL https: //arxiv.org/abs/2308.16149v2.

Guokan Shang, Hadi Abdine, Yousef Khoubrane, Amr Mohamed, Yassine Abbahaddou, Sofiane Ennadir, Imane Momayiz, Xuguang Ren, Eric Moulines, Preslav Nakov, Michalis Vazirgiannis, and Eric Xing. Atlaschat: Adapting large language models for low-resource Moroccan Arabic dialect. In Hansi Hettiarachchi, Tharindu Ranasinghe, Paul Rayson, Ruslan Mitkov, Mohamed Gaber, Damith Premasiri, Fiona Anting Tan, and Lasitha Uyangodage (eds.), LoResLM, pp. 9–30, Abu Dhabi, United Arab Emirates, January 2025. ACL. URL https://aclanthology.org/2025.loreslm-1.2/.

Fanar Team, Ummar Abbas, Mohammad Shahmeer Ahmad, Firoj Alam, Enes Altinisik, Ehsannedin Asgari, Yazan Boshmaf, Sabri Boughorbel, Sanjay Chawla, Shammur Chowdhury, Fahim Dalvi, Kareem Darwish, Nadir Durrani, Mohamed Elfeky, Ahmed Elmagarmid, Mohamed Eltabakh, Masoomali Fatehkia, Anastasios Fragkopoulos, Maram Hasanain, Majd Hawasly, Mus’ab Husaini, Soon-Gyo Jung, Ji Kim Lucas, Walid Magdy, Safa Messaoud, Abubakr Mohamed, Tasnim Mohiuddin, Basel Mousi, Hamdy Mubarak, Ahmad Musleh, Zan Naeem, Mourad Ouzzani, Dorde Popovic, Amin Sadeghi, Husrev Taha Sencar, Mohammed Shinoy, Omar Sinan, Yifan Zhang, Ahmed Ali, Yassine El Kheir, Xiaosong Ma, and Chaoyi Ruan. Fanar: An arabic-centric multimodal generative ai platform, 2025. URL https://arxiv.org/abs/2501.13944.

