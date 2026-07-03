## arXiv:2505.17894v2[cs.CL]21Aug2025

# Mutarjim: Advancing Bidirectional Arabic-English Translation with a Small Language Model

##### Khalil Hennara, Muhammad Hreden, Mohamed Motasim Hamed, Zeina Aldallal, Sara Chrouf, and Safwan AlModhayan

[Figure 1]

Khobar, Saudi Arabia hennara,hreden,hamed,aldallal,chrouf,safwan@misraj.ai

##### Abstract

We introduce Mutarjim, a compact yet powerful language model for bidirectional Arabic-English translation. While large-scale LLMs have shown impressive progress in natural language processing tasks, including machine translation, smaller models. Leveraging this insight, we developed Mutarjim based on Kuwain-1.5B Hennara et al. [2025], a language model tailored for both Arabic and English. Despite its modest size, Mutarjim outperforms much larger models on several established benchmarks, achieved through an optimized two-phase training approach and a carefully curated, high-quality training corpus.. Experimental results show that Mutarjim rivals models up to 20 times larger while significantly reducing computational costs and training requirements. We also introduce Tarjama-25, a new benchmark designed to overcome limitations in existing Arabic-English benchmarking datasets, such as domain narrowness, short sentence lengths, and English-source bias. Tarjama-25 comprises 5,000 expert-reviewed sentence pairs and spans a wide range of domains, offering a more comprehensive and balanced evaluation framework. Notably, Mutarjim achieves state-of-the-art performance on the English-to-Arabic task in Tarjama-25, surpassing even significantly larger and proprietary models like GPT-4o mini. We publicly release Tarjama-25 to support future research and advance the evaluation of Arabic-English translation systems.

### 1 Introduction

Machine translation (MT), a core task in natural language processing (NLP), has made great progress with the rise of Large Language Models (LLMs). However, Arabic machine translation (AMT) is considered a big challenge due to many factors and characteristics of the Arabic language, such as grammar and morphology complexity. Lexical, syntactic, and semantic problems arise when translating the meaning of Arabic words into English and vice versa Baligh & Mohammed [2022]. Despite recent advances in NLP, the Arabic language still lags behind other high-resource languages in terms of translation quality. Existing Arabic-English systems are either limited in their capabilities or are part of larger multilingual models that, while capable of handling many languages, often underperform on Arabic-specific tasks. These models are also computationally demanding, limiting their practicality in low-resource or real-time settings. Consequently, there is increasing interest in developing smaller, task-specific models that balance performance with efficiency while effectively modeling Arabic’s linguistic complexity.

[Figure 3]

[Figure 4]

ChrF++ scores COMET scores

Figure 1: Performance of various models on Tarjama-25 sorted by model size, our newly introduced benchmark for Arabic-English translation, evaluated using two metrics: ChrF++ (left) and COMET (right).

In this paper, we introduce Mutarjim, a task-specific small language model optimized for ArabicEnglish translation. Mutarjim is built on Kuwain-1.5B Hennara et al. [2025], an Arabic-centric decoder-only model. Mutarjim is trained in two stages: a translation-oriented large-scale pretraining phase and a targeted fine-tuning stage using high-quality parallel corpora. This tailored training approach enables Mutarjim to deliver competitive translation quality and faster inference times. In benchmark evaluations, Mutarjim outperforms models with more than 30 billion parameters, including proprietary systems like GPT-4o mini in both accuracy and efficiency, as shown in Figure 1.

To facilitate robust evaluation and future research, we also present Tarjama-25, a new benchmark dataset for bidirectional Arabic-English translation. Tarjama-25 addresses key limitations of existing datasets, such as short sentence length, English source bias, and limited domain diversity. It comprises 5,000 pairs of expert-curated sentences from diverse domains, with an equal number of examples sourced from Arabic and English originals. All pairs are used to evaluate both Arabicto-English and English-to-Arabic translation, providing a comprehensive and realistic benchmark for bidirectional translation performance.

Our contributions can be summarized as follows:

- • We introduce Mutarjim, a compact but powerful decoder-only model specifically optimized for Arabic-English translation.
- • We present Tarjama-25, a new benchmark for Arabic-English translation, featuring:

- – Longer and more natural sentence structures;
- – Balanced translation directionality, with an equal number of source texts originally written in Arabic and English;
- – Broad domain coverage, spanning general, medical, legal, technological, and other fields;
- – Careful curation to eliminate contamination from large-scale pre-training corpora, ensuring fair and unbiased evaluation.
- – Expert-reviewed and human-corrected translations to ensure high linguistic quality and fidelity.

- • We perform extensive evaluations using multiple standard benchmarks, including WMT24++ Deutsch et al. [2025], IWSLT2017 Cettolo et al. [2017], and our newly introduced Tarjama-

- 25, and compare Mutarjim against a range of open-source and proprietary models, using automatic metrics BLEU, chrF++, and COMET Rei et al. [2020].
- • We publicly release both the Tarjama-25 ∗ benchmark and its accompanying evaluation toolkit † as open-source resources to promote transparency, reproducibility, and further progress in Arabic machine translation research.

The rest of the paper is structured as follows: Section 2 reviews the related works and theoretical foundations relevant to our study. Section 3 describes the dataset creation process in the two phases. Section 4 outlines our benchmarking setup. Section 5 introduces Mutarjim model with the training methodology. Section 6 details our experiments and results. Section 7 explains our evaluation strategy. Finally, Section 8 concludes the paper and suggests directions for future research.

### 2 Background

Machine translation has undergone significant evolution, progressing from rule-based approaches necessitating extensive development and maintenance toward statistical and neural paradigms. The development of encoder-decoder and, more recently, decoder-only architectures has played a pivotal role in advancing the field. In this section, we review the major contributions that form the foundation of modern neural MT, focusing on models relevant to Arabic and multilingual translation.

#### 2.1 Encoder-Decoder Models

Encoder-decoder Transformer models form the basis of modern neural machine translation (NMT) systems, evolving from general-purpose architectures to specialized, multilingual frameworks. These models vary in scale, language coverage, and specialization, balancing broad applicability with performance in specific languages such as Arabic. This section reviews key models, highlighting their contributions and limitations, particularly in the context of multilingual and Arabic-focused translation.

The Text-to-Text Transfer Transformer (T5) Raffel et al. [2023] introduced a unified text-to-text framework, trained on the English-only C4 dataset with model sizes ranging from 60M to 11B parameters. Although not designed for translation, the flexible architecture of T5 laid the groundwork for subsequent multilingual models. Its primary limitation, the lack of multilingual support, restricts its direct applicability to NMT tasks. Based on T5, mT5 Xue et al. [2021] extends the text-to-text framework to 101 languages using the multilingual mC4 dataset. However, Arabic constitutes only

- 1.66% of its pre-training data, constraining its effectiveness for Arabic translation. Despite strong zero-shot transfer capabilities, mT5 struggles with language-specific nuances, prompting further specialization in later models. Similarly, Aya-101 Üstün et al. [2024] adapts the mT5-XXL model (13B parameters) using instruction tuning in 100+ languages. This broad multilingual tuning improves the model’s ability to translate languages with limited training data, known as low-resource languages. However, its large size results in high training and inference costs, making it less practical for deployment compared to smaller, more specialized models. In particular, mBART Liu et al. [2020] is a 680M parameter model trained using multilingual denoising pre-training and includes 2.87

∗https://huggingface.co/datasets/Misraj/Tarjama-25 †https://github.com/misraj-ai/Mutarjim-evaluation

billion Arabic tokens in its corpus. To address broader language coverage, NLLB-200 Costa-Jussà et al. [2022a] is a 3.3B parameter dense model trained on over 18 billion sentence pairs spanning 200 languages. It achieves strong translation performance, particularly in low-resource settings, and is known for its relative robustness against hallucinations. However, its effectiveness diminishes in domain-specific texts such as Islamic or medical content, where it struggles to maintain accuracy and relevance. In response to the need for Arabic-specific solutions, TURJUMAN Nagoudi et al.

- [2022], built directly on AraT5 Nagoudi et al. [2021], provides a toolkit for translating 20 source languages into MSA.

2.2 Decoder-Only Models

The recent shift toward decoder-only language models has reshaped the machine translation landscape, particularly through their use in autoregressive generation. Unlike traditional encoderdecoder architectures, decoder-only models handle both the source and target text within a single sequence and rely on large-scale pre-training.

A major advantage of decoder-only models lies in their unified architecture for both understanding and generation tasks, enabling efficient transfer and scalability. Prompt-based translation using large models such as GPT-4 Achiam et al. [2023] has shown promising results, especially when effective prompting strategies are applied He et al. [2024]; Lu et al. [2023]; Xi et al. [2022]; Zhu et al.

- [2023]; Agrawal et al. [2022]; Vilar et al. [2022]. These approaches often involve feeding a few translation examples directly into the prompt, allowing the model to generalize with minimal supervision. Beyond prompting, compact decoder-only models have gained traction due to their efficiency. For example, BigTranslate Yang et al. [2023] and PARADIGM Xu et al. [2023] explore fine-tuning small open-source LLMs using parallel corpora, achieving competitive translation performance at a fraction of computational cost. Models like Tower Alves et al. [2024] and GemmaX2-28 Cui et al. [2025] employ two-phase training: multilingual pre-training followed by fine-tuning conducted according to instructions, enabling domain-specific translation capabilities. Multilingual decoder-only models, such as XALMA Xu et al. [2024], scale this approach further, incorporating 50 languages and using MoE layers and adapter modules to manage multilingual transfer.

Within the Arabic domain, general purpose decoder-only models, such as Allam Bari et al. [2024], AceGPT Huang et al. [2023], and Silma Team [2024], are trained primarily in Arabic-centric corpora with objectives to cover various downstream tasks. However, these models are typically not specialized in translation and often lack the fine-grained bilingual alignment necessary for a high-quality Arabic-English translation. Specialized Arabic decoder-only translation models have also emerged. One such model is Lahjawi Hamed et al. [2025], a cross-dialect translation system that demonstrated strong performance in both MSA and dialectal Arabic. Lahjawi is specifically undergoing targeted fine-tuning for cross-dialect translation.

Given the trade-off between translation quality, inference speed, and resource efficiency, in this work, we adopt the decoder-only paradigm. Mutarjim follows a two-stage training strategy: largescale monolingual pre-training followed by supervised fine-tuning using high-quality Arabic-English parallel data. Our choice reflects a growing shift toward compact, efficient, and Arabic-optimized decoder-only systems for machine translation.

### 3 Data

Our training data consists exclusively of bilingual Arabic-English corpora, combining proprietary and open-source sources. For pre-training, we utilize large-scale, domain-diverse parallel datasets to expose the model to a wide range of linguistic patterns and translation contexts. Fine-tuning focuses on higher-quality data, combining carefully filtered open-source corpora with proprietary datasets curated for accuracy, fluency, and domain relevance. This composition ensures broad coverage during pre-training while emphasizing translation accuracy, fluency, and domain specificity during fine-tuning.

#### 3.1 Pre-training Data

Our pre-training corpus comprises approximately 10 billion tokens of bilingual Arabic-English data, used to continue pre-training kuwain-1.5 the base model and improve its performance in translation tasks. Data are sourced from the OPUS platform Tiedemann [2016], where we exclusively select Arabic-English parallel corpora, supplemented with proprietary datasets curated internally to improve domain diversity and coverage.

To improve data quality, we applied a set of pre-processing steps, including the removal of sentence pairs with fewer than three tokens, as such samples often lack meaningful context. We also filtered out misaligned examples in which the target sentence is not in the correct language. Pairs with substantial length mismatches are excluded to reduce the risk of partial or noisy translations. Finally, we performed deduplication to eliminate repeated sentence pairs and reduce redundancy in the training data. These filtering steps help maintain a reasonably clean and consistent corpus for pre-training.

#### 3.2 Fine-tuning Data

The fine-tuning procedure utilized a precisely selected corpus of approximately 6 million ArabicEnglish parallel sentence pairs. This corpus exhibits substantial diversity and underwent careful filtering procedures to ensure high translation fidelity. The dataset was derived from two principal sources:

- • One portion of the data incorporates translations originally produced by a state-of-the-art LLM. Subsequently, expert linguists inspected a representative subset of these outputs to confirm their accuracy and fluency. To promote stronger Arabic fluency in the model, we emphasized Arabic-centric samples, where Arabic is the source language, at a 2:1 ratio compared to English-centric samples. This approach not only enhances the model’s ability to understand and generate Arabic text but also helps preserve the cultural and linguistic richness of the language.
- • The remaining portion comprises high-quality filtered data from OPUS. We applied a combination of automatic and manual review processes, including human inspection of representative subsets. Datasets exhibiting recurring issues such as out-of-context sentences, hallucinations, misinformation, or poor alignment were excluded to maintain overall data integrity.

The fine-tuning dataset was designed to align with the domain categories introduced in our bench-

mark (Section 4), ensuring broad and realistic coverage across cultural, legal, scientific, healthcare, religious, and technical domains. We prioritized the inclusion of authentic Arabic source material and maintained a balanced representation across domains to mitigate distributional bias. This targeted curation enables the model to generalize effectively across diverse topics while preserving high translation quality.

### 4 Tarjama-25: Bidirectional Arabic-English Translation Benchmark

#### 4.1 Motivation and Development

Modern MT systems face persistent evaluation challenges due to the limitations of existing benchmarks. To address these gaps, we introduce Tarjama-25, a comprehensive benchmark specifically designed for both Arabic–to–English and English–to–Arabic translation tasks. The current landscape of translation evaluation reveals several critical shortcomings: most publicly available datasets are English-centric (i.e., English is the source language), lacking authentic bidirectional content; benchmarks tend to contain predominantly short sentences (typically 6–30 words), which underutilizes the capacity of modern language models designed to process substantially longer input sequences; and domain-specific coverage remains limited. Furthermore, potential data contamination from web-scale pre-training and insufficient representation of language-specific characteristics, particularly for Arabic texts, pose additional challenges. To address these challenges, we developed Tarjama-25 through a comprehensive data collection and validation pipeline:

- • We began by collecting 30,000 sentences from authentic Arabic and English sources, each ranging from 50 to 100 words long, ensuring broad domain coverage across scientific, technical, healthcare, cultural, and general interest topics. Half of the data was originally written in Arabic, and the other half in English.
- • The 30,000 sentences were initially translated using state-of-the-art machine translation systems to create parallel sentence pairs.
- • From these, 5,000 pairs of sentences were selected for detailed human refinement. Professional translators reviewed and corrected each selected pair to ensure linguistic accuracy and fluency. The final selection maintains a balanced distribution in all domains (Figure 2).
- • Finally, domain experts conducted an additional review to validate the accuracy and contextual relevance of the translations within their respective fields.

This careful multi-stage process ensures high-quality, human-validated translations with a balanced source language distribution and rich domain diversity, making Tarjama-25 a robust and realistic benchmark for bidirectional Arabic-English translation evaluation.

#### 4.2 Findings and Recommendations

Tarjama-25 distinguishes itself through authentic source content in both languages, diverse text lengths, extensive domain coverage, and a strong focus on language-specific subtleties. Our preliminary evaluations reveal that many current MT models, despite their strong performance on existing

[Figure 10]

Figure 2: Domain Coverage in Tarjama-25 Benchmark

benchmarks, face significant challenges with Tarjama-25. Detailed evaluation results are presented in Section 7.

Based on our findings, we recommend:

- 1. Development of language-specific authentic benchmarks;
- 2. Greater emphasis on domain-specific translation capability;
- 3. Integration of cultural and linguistic nuances in evaluation metrics;
- 4. Regular benchmark updates to reflect evolving language use.
- 5 Method

For Mutarjim, we build on Kuwain-1.5B Hennara et al. [2025], a decoder-only bilingual ArabicEnglish small language model designed for efficiency in resource-constrained environments. Our approach adopts standard LLM training methodologies commonly used in the field. These methodologies comprise two main phases: pre-training and fine-tuning. To improve translation performance, we introduce targeted modifications within this framework. The pre-training phase is designed to develop a robust bilingual representation, a foundation for the subsequent fine-tuning stage focused specifically on translation tasks.

#### 5.1 Pre-training Phase

Following the successful approaches of recent works such as GemmaX Cui et al. [2025] and Tower Alves et al. [2024] in continuing pre-training for translation tasks, we further pre-trained our model on English-Arabic parallel data using a next-token prediction objective.

To facilitate the learning process, we introduce two special tokens to our model: <|English|> and <|Arabic|>. We formatted the data as shown on the left side of Figure 3, where English sentences begin with the token <|English|> and Arabic sentences with <|Arabic|>. All pre-training data

[Figure 12]

[Figure 13]

Figure 3: Illustration of the two data formats used in Mutarjim: (Left) pre-training stream data format; (Right) fine-tuning data sample.

consist of paired Arabic-English sentences structured according to this format. During training, the model sees both sentences and is trained to predict the next token over the entire input. To prevent unidirectional translation bias, we randomly select the order of the sentences in each pair. This encourages the model to develop robust bidirectional translation capabilities without favoring a specific source language.

#### 5.2 Fine-tuning Phase

The fine-tuning phase follows the same format as pre-training, adding a newline between the two sentences for improved structural clarity, as illustrated on the right side of Figure 3. However, unlike the pre-training stage, we apply causal masking to the input sentence so that the model is only trained on generating the target sentence from the source, while still using the same next-token prediction objective.

We exclusively use high-quality, human-curated parallel data for this phase to ensure translation accuracy. The model is trained for two epochs over a total of 3 billion tokens, balancing sufficient exposure to high-quality examples with the need to avoid overfitting. We carefully monitor both training phases to maintain translation quality and prevent performance degradation. Detailed training specifications, including learning rates, batch sizes, and other hyperparameters, are provided in the appendix B.

### 6 Experiment and Results

To thoroughly evaluate the effectiveness of Mutarjim, we conducted a series of experiments aimed at gaining deeper insights into the challenges and dynamics of Arabic-English translation. Our evaluation focuses on three core aspects. First, we compare unidirectional and bidirectional training setups to assess whether a single model trained in both directions (Arabic–to–English and English–to–Arabic) compromises performance relative to dedicated unidirectional models. Second, we examine the contribution of the continued pre-training phase in enhancing translation quality and improving the model’s generalization across domains. Third, we analyze the effect of context length during fine-tuning to understand how sentence length influences performance, particularly when the evaluation samples differ in length from those seen during training. These experiments are conducted using the WMT24++ benchmark, providing a consistent and challenging evaluation framework.

#### 6.1 Unidirectional vs. Bidirectional Translation Performance

To assess the impact of directional training on translation quality, we compared unidirectional and bidirectional versions of Mutarjim, focusing on how specializing in a single translation direction affects performance relative to a multitask setup. We investigated the performance trade-offs between unidirectional models—Mutarjim-AR2EN (Arabic–to–English) and Mutarjim-EN2AR (English–to–Arabic)—and the bidirectional model Mutarjim-Bi. The unidirectional variants were each trained for 3 epochs, while the bidirectional variant was trained for 2 epochs on the combined data. Table 1 presents evaluation results using COMET Rei et al. [2020] and chrF++ metrics on the WMT24++ benchmark Deutsch et al. [2025]. Despite being exposed to more diverse data, the bidirectional model showed a slight decrease in performance. Unidirectional models consistently outperformed the bidirectional model, with Mutarjim-AR2EN achieving a COMET score 3.16 points higher than Mutarjim-Bi for Arabic-to-English translation. Ultimately, the choice of model depends on application needs: Mutarjim-Bi offers greater efficiency and flexibility through multitask support, while the unidirectional variants deliver higher translation accuracy for specific directions. Given the compact size of our model (1.5B parameters), the computational cost difference between approaches remains modest.

Arabic → English English → Arabic COMET chrF++ COMET chrF++

Model Training

Mutarjim-Bi Bidirectional 79.73 50.27 72.86 47.04 Mutarjim-AR2EN Unidirectional 82.89 54.89 — Mutarjim-EN2AR Unidirectional — — 75.46 48.04

- Table 1: Performance comparison between bidirectional (Mutarjim-Bi) and unidirectional (Mutarjim-EN2AR or AREN) translation models on WMT24++.

6.2 The Impact of Continued Pre-training Phase

We evaluated the impact of continued pre-training on translation performance, aiming to determine whether translation-specific pre-training could yield meaningful gains over direct fine-tuning. Although our base model, Kuwain, was initially trained on a substantial corpus, making it a viable candidate for direct fine-tuning, we explore whether targeted continued pre-training on bilingual data enhances downstream translation quality, following recent successes in domain-adaptive pretraining Cui et al. [2025]; Alves et al. [2024].

- Table 2 presents a comparison between models trained with and without the additional translationfocused pre-training phase. Models benefiting from this additional phase consistently outperform their counterparts trained solely through fine-tuning, as reflected in both COMET and chrF++ scores. The gains are evident in both the Arabic–to–English and English–to–Arabic directions, underscoring the general effectiveness of this strategy in translation tasks. While this approach may not be cost-effective for larger models, it remains computationally feasible for smaller architectures like our 1.5B parameter models.

Arabic → English English → Arabic COMET chrF++ COMET chrF++

Model

Without Additional Pre-training Mutarjim-AR2EN 74.30 42.17 — Mutarjim-EN2AR — — 61.91 34.89

With Additional Pre-training Mutarjim-AR2EN 82.89 54.89 — Mutarjim-EN2AR — — 75.46 48.04

- Table 2: Effect of additional translation-specific pre-training on model performance, evaluated on WMT24++ test set.

6.3 Context Length Effect

We conducted two independent fine-tuning experiments to evaluate the impact of input length distributions on translation performance. In the first experiment (e1), we fine-tuned the pre-trained Mutarjim model using samples containing more than 30 words, aiming to improve the model’s performance on longer sentences. While this enhanced fluency on long-form content, we observed performance degradation on shorter inputs, with increased hallucinations and irrelevant continuations.

To address this, we performed a second, separate fine-tuning experiment (e2) using the same base model but modifying the training set to include an additional 15% of short samples (ranging from 2 to 30 words). This experiment sought to balance the model’s ability across varying sequence lengths. We evaluated both versions on the WMT24++ test set. As shown in Table 3, the second experiment (e2) led to improved performance in both directions of translation, confirming the benefit of including shorter sequences in the training data.

Model

Arabic → English English → Arabic COMET chrF++ COMET chrF++

- Experiment 1 (Long Inputs Only)

- Mutarjim-AR2EN-e1 73.62 48.57 — —

- Mutarjim-EN2AR-e1 — — 69.07 43.40

Experiment 2 (Mixed Length Inputs) Mutarjim-AR2EN-e2 74.22 50.84 — —

- Mutarjim-EN2AR-e2 — — 73.56 46.05

- Table 3: Evaluation of models fine-tuned with different input length distributions on the WMT24++ test set.

### 7 Evaluation

To contextualize the performance of Mutarjim, we compare it against a diverse set of strong decoder-only models that support Arabic and are widely recognized for their translation capabilities. These include general-purpose language models such as AceGPT-8BHuang et al. [2023], ALLam-

7BBari et al. [2024], C4AI-7BCohere For AI [2024], Cohere-8B Aryabumi et al. [2024], Cohere-32B Aryabumi et al. [2024], Gemma2-27B Team et al. [2024], Silma-9B Team [2024], and Yehia-7B Navid-AI [2025]. Furthermore, we include multilingual translation-specialized models such as XALMA-13B-Group8 Xu et al. [2024], LLaMAX3-8B-Alpaca Lu et al. [2024], and GemmaX-9B Cui et al. [2025]. To provide a closer baseline in terms of the architecture and size of the model, we also evaluate against NLLB-3.3B Team et al. [2022], an encoder–decoder model known for its effectiveness in low-resource translation tasks and its widespread adoption in Arabic-English translation. We evaluated the performance of our model compared to a range of strong baseline models across three established benchmarks: WMT24++, IWSLT2017, and our newly proposed benchmark Tarjama-25. For all benchmarks, we evaluated translation quality using widely adopted metrics, BLEU, chrF++, and COMET, to ensure a comprehensive and fair assessment.

The results for each benchmark are reported in their respective tables: Tarjama-25 in Table 4, WMT24++ in Table 5, and IWSLT2017 in Table 6. For consistency, all models are listed in the tables in order of model size. To ensure a fair comparison, we employ model-specific prompts during evaluation, as illustrated in Appendix D. To streamline the evaluation pipeline and accelerate inference, we utilize VLLM Kwon et al. [2023]‡, which enables efficient batched decoding across decoder-only models.

Although being the smallest among the evaluated models, Mutarjim achieves state-of-the-art performance on the Tarjama-25 benchmark for the Arabic-to-English direction in all evaluation metrics, and leads in the English–to–Arabic direction when measured by the BLEU score. It closely trails the much larger GPT-4o-mini model in COMET and chrF++ with only a narrow margin. These results highlight Mutarjim’s competitive effectiveness despite its compact size, demonstrating its strength in both translation quality and efficiency.

Model performance varies noticeably on Tarjama-25 compared to existing benchmarks. For example, while GPT-4o-mini excels on WMT24++ and IWSLT2017, its relative performance declines on Tarjama-25. This highlights how standard benchmarks may overlook challenges in domainspecific and bidirectional translation. Tarjama-25 helps expose these gaps, offering a more realistic and rigorous assessment of real-world translation capabilities.

Another key observation is the consistent performance gap observed in most models between Arabicto-English and English-to-Arabic translation, with the former generally yielding better results. This trend is visually illustrated in Figure 1, where the disparity, particularly in the chrF++ metric, is pronounced. Several factors may contribute to this asymmetry, including Arabic’s rich morphology and syntactic flexibility, which allow for multiple valid translations that current metrics may fail to recognize. Furthermore, the predominance of English-centric training data in many models may hinder their ability to generate fluent and accurate Arabic output.

Notably, Mutarjim demonstrates balanced performance in both translation directions, which we attribute to its Arabic-centric training strategy. This indicates that training with authentic Arabic source data can help mitigate directional bias and improve overall translation fidelity.

‡https://docs.vllm.ai/en/stable/

Arabic → English English → Arabic COMET Chrf++ Bleu COMET Chrf++ Bleu Mutarjim 1.5B 82.63 74.66 55.28 83.41 68.67 43.71 NLLBCosta-Jussà et al. [2022b] 3.3B 67.06 40.50 24.38 81.27 59.69 30.32 c4ai Cohere For AI [2024] 7B 80.93 67.24 43.34 79.10 55.96 25.18 Yehia Navid-AI [2025] 7B 73.31 56.77 32.14 74.97 50.32 20.67 ALLamBari et al. [2024] 7B 72.90 56.88 31.01 75.41 51.24 20.54 Cohere Aryabumi et al. [2024] 8B 81.20 67.16 42.72 82.50 58.46 26.26 AceGPT Huang et al. [2023] 8B 80.71 65.63 38.67 78.39 50.67 20.02 LLaMAX3 Lu et al. [2024] 8B 77.72 54.95 27.86 56.76 33.25 7.63 SILMA Team [2024] 9B 64.36 37.84 15.67 58.01 27.71 5.62 GemmaX Cui et al. [2025] 9B 69.63 43.42 19.96 66.94 37.66 9.98 XALMA Xu et al. [2024] 13B 73.37 46.96 21.57 66.36 29.88 6.64 Gemma 2 Team et al. [2024] 27B 80.81 70.42 42.78 42.20 3.52 3.08 Cohere Aryabumi et al. [2024] 32B 82.44 73.10 51.16 82.09 63.29 32.25 GPT-4o mini Hurst et al. [2024] - 83.67 76.08 54.24 83.36 66.36 38.52

Model Size

###### Table 4: Performance comparison of bidirectional (Arabic-English) translation models on the Tarjama-25 benchmark in terms of COMET, Chrf++, and Bleu.

Arabic → English English → Arabic COMET Chrf++ Bleu COMET Chrf++ Bleu Mutarjim 1.5B 72.99 52.27 19.26 75.46 48.04 17.99 NLLBCosta-Jussà et al. [2022b] 3.3B 76.71 50.13 25.50 77.75 45.89 16.03 c4ai Cohere For AI [2024] 7B 79.27 54.91 26.35 72.45 44.32 14.19 Yehia Navid-AI [2025] 7B 72.72 47.58 15.39 72.23 41.12 10.69 ALLamBari et al. [2024] 7B 72.00 46.80 15.01 71.89 41.45 10.41 Cohere Aryabumi et al. [2024] 8B 78.89 54.06 24.96 74.80 44.95 14.08 AceGPT Huang et al. [2023] 8B 78.18 52.25 21.21 73.65 40.55 11.37 LLaMAX3 Lu et al. [2024] 8B 75.91 48.18 18.89 57.31 28.83 4.03 SILMA Team [2024] 9B 71.33 38.96 16.44 60.54 26.75 4.97 GemmaX Cui et al. [2025] 9B 77.82 50.80 22.67 70.21 38.81 9.83 XALMA Xu et al. [2024] 13B 76.84 48.65 19.34 69.19 33.23 7.54 Gemma 2 Team et al. [2024] 27B 72.79 51.09 16.59 54.00 32.66 4.77 Cohere Aryabumi et al. [2024] 32B 79.77 57.05 27.98 72.74 47.13 15.84 GPT-4o mini Hurst et al. [2024] - 83.29 58.24 29.23 82.32 50.03 20.48

Model Size

###### Table 5: Performance comparison of bidirectional (Arabic-English) translation models on the WMT24++ benchmark in terms of COMET, Chrf++, and Bleu.

Arabic → English English → Arabic COMET Chrf++ Bleu COMET Chrf++ Bleu Mutarjim 1.5B 82.89 54.89 31.00 79.76 44.21 12.74 NLLBCosta-Jussà et al. [2022b] 3.3B - - - - - c4ai Cohere For AI [2024] 7B 83.99 56.64 33.64 77.41 40.50 9.14 Yehia Navid-AI [2025] 7B 75.58 47.38 15.93 76.22 38.41 6.65 ALLamBari et al. [2024] 7B 75.64 37.36 5.89 75.25 46.54 14.79 Cohere Aryabumi et al. [2024] 8B 83.60 55.83 31.71 79.05 42.36 9.10 AceGPT Huang et al. [2023] 8B 81.72 52.83 26.26 79.62 40.23 9.25 LLaMAX3 Lu et al. [2024] 8B 81.04 49.17 24.28 67.79 30.17 4.18 SILMA Team [2024] 9B 78.55 47.57 24.28 69.59 30.03 5.11 GemmaX Cui et al. [2025] 9B 82.06 53.30 30.25 76.17 37.17 7.10 XALMA Xu et al. [2024] 13B 80.06 49.04 24.10 76.41 36.99 7.13 Gemma 2 Team et al. [2024] 27B - - - 48.56 22.28 1.57 Cohere Aryabumi et al. [2024] 32B 84.30 59.02 35.37 74.63 43.53 8.93 GPT-4o mini Hurst et al. [2024] - 86.37 60.48 36.86 87.14 47.63 15.50

Model Size

- Table 6: Performance comparison of bidirectional (Arabic-English) translation models on the IWSLT-2017 benchmark in terms of COMET, Chrf++, and Bleu.

### 8 Conclusion

In this work, we introduce Mutarjim, an efficient and compact small language model, optimized for bidirectional Arabic-English machine translation while providing rich and accurate output. We also present a new benchmark Tarjama-25, a diverse and representative dataset for bidirectional ArabicEnglish MT evaluation. Our evaluation and experiments demonstrate that Mutarjim achieves competitive performance against larger models while requiring significantly fewer computational resources. The model’s compact architecture enables deployment in resource-constrained environments without sacrificing translation quality. Future work will focus on scaling up the model architecture and training on larger multilingual datasets to support translation between Arabic and multiple languages, including French, Turkish, and Japanese, to create a comprehensive multilingual translation system while maintaining efficiency.

### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Sweta Agrawal, Chunting Zhou, Mike Lewis, Luke Zettlemoyer, and Marjan Ghazvininejad. Incontext examples selection for machine translation. arXiv preprint arXiv:2212.02437, 2022.

Duarte M Alves, José Pombal, Nuno M Guerreiro, Pedro H Martins, João Alves, Amin Farajian, Ben Peters, Ricardo Rei, Patrick Fernandes, Sweta Agrawal, et al. Tower: An open multilingual large language model for translation-related tasks. arXiv preprint arXiv:2402.17733, 2024.

Viraat Aryabumi, John Dang, Dwarak Talupuru, Saurabh Dash, David Cairuz, Hangyu Lin, Bharat Venkitesh, Madeline Smith, Jon Ander Campos, Yi Chern Tan, et al. Aya 23: Open weight releases to further multilingual progress. arXiv preprint arXiv:2405.15032, 2024.

Babaali Baligh and Salem Mohammed. Arabic machine translation: A panoramic survey. SSRN Electronic Journal, 01 2022. doi: 10.2139/ssrn.4312742.

M Saiful Bari, Yazeed Alnumay, Norah A Alzahrani, Nouf M Alotaibi, Hisham A Alyahya, Sultan AlRashed, Faisal A Mirza, Shaykhah Z Alsubaie, Hassan A Alahmed, Ghadah Alabduljabbar, et al. Allam: Large language models for arabic and english. arXiv preprint arXiv:2407.15390, 2024.

Mauro Cettolo, Marcello Federico, Luisa Bentivogli, Jan Niehues, Sebastian Stüker, Katsuhito Sudoh, Koichiro Yoshino, and Christian Federmann. Overview of the IWSLT 2017 evaluation campaign. In Proceedings of the 14th International Conference on Spoken Language Translation, pp. 2–14, Tokyo, Japan, December 14-15 2017. International Workshop on Spoken Language Translation. URL https://aclanthology.org/2017.iwslt-1.1.

Cohere For AI. c4ai-command-r-07-arabic-2025, 2024. URL https://huggingface.co/CohereFor AI/c4ai-command-r-08-2024.

Marta R Costa-Jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, et al. No language left behind:

- Scaling human-centered machine translation. arXiv preprint arXiv:2207.04672, 2022a.

Marta R Costa-Jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, et al. No language left behind:

- Scaling human-centered machine translation. arXiv preprint arXiv:2207.04672, 2022b.

Menglong Cui, Pengzhi Gao, Wei Liu, Jian Luan, et al. Multilingual machine translation with open large language models at practical scale: An empirical study. arXiv preprint arXiv:2502.02481, 2025.

Daniel Deutsch, Eleftheria Briakou, Isaac Caswell, Mara Finkelstein, Rebecca Galor, Juraj Juraska, Geza Kovacs, Alison Lui, Ricardo Rei, Jason Riesa, et al. Wmt24++: Expanding the language coverage of wmt24 to 55 languages & dialects. arXiv preprint arXiv:2502.12404, 2025.

Mohamed Motasim Hamed, Muhammad Hreden, Khalil Hennara, Zeina Aldallal, Sara Chrouf, and Safwan AlModhayan. Lahjawi: Arabic cross-dialect translator. In Proceedings of the 4th Workshop on Arabic Corpus Linguistics (WACL-4), pp. 12–24, 2025.

Zhiwei He, Tian Liang, Wenxiang Jiao, Zhuosheng Zhang, Yujiu Yang, Rui Wang, Zhaopeng Tu, Shuming Shi, and Xing Wang. Exploring human-like translation strategy with large language models. Transactions of the Association for Computational Linguistics, 12:229–246, 2024.

Khalil Hennara, Sara Chrouf, Mohamed Motaism Hamed, Zeina Aldallal, Omar Hadid, and Safwan AlModhayan. Kuwain 1.5 b: An arabic slm via language injection. arXiv preprint arXiv:2504.15120, 2025.

Huang Huang, Fei Yu, Jianqing Zhu, Xuening Sun, Hao Cheng, Dingjie Song, Zhihong Chen, Abdulmohsen Alharthi, Bang An, Juncai He, et al. Acegpt, localizing large language models in arabic. arXiv preprint arXiv:2309.12053, 2023.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, and Luke Zettlemoyer. Multilingual denoising pre-training for neural machine translation. Transactions of the Association for Computational Linguistics, 8:726–742, 2020.

Hongyuan Lu, Haoran Yang, Haoyang Huang, Dongdong Zhang, Wai Lam, and Furu Wei. Chainof-dictionary prompting elicits translation in large language models. arxiv e-prints, page. arXiv preprint arXiv:2305.06575, 2023.

Yinquan Lu, Wenhao Zhu, Lei Li, Yu Qiao, and Fei Yuan. Llamax: Scaling linguistic horizons of llm by enhancing translation capabilities beyond 100 languages. arXiv preprint arXiv:2407.05975, 2024.

El Moatez Billah Nagoudi, AbdelRahim Elmadany, and Muhammad Abdul-Mageed. Arat5: Textto-text transformers for arabic language generation. arXiv preprint arXiv:2109.12068, 2021.

El Moatez Billah Nagoudi, AbdelRahim Elmadany, and Muhammad Abdul-Mageed. Turjuman: A public toolkit for neural arabic machine translation, 2022. URL https://arxiv.org/abs/2206

.03933. Navid-AI. Yehia 7b preview. https://huggingface.co/Navid-AI/Yehia-7B-preview, 2025. Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi

Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer, 2023. URL https://arxiv.org/abs/1910.10683.

Ricardo Rei, Craig Stewart, Ana C Farinha, and Alon Lavie. Comet: A neural framework for mt evaluation. arXiv preprint arXiv:2009.09025, 2020.

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.

NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and Jeff Wang. No language left behind: Scaling human-centered machine translation, 2022. URL https://arxiv.org/abs/ 2207.04672.

Silma Team. Silma. 2024. URL https://www.silma.ai.

Jörg Tiedemann. OPUS – parallel corpora for everyone. In Proceedings of the 19th Annual Conference of the European Association for Machine Translation: Projects/Products, Riga, Latvia, May 30–June 1 2016. Baltic Journal of Modern Computing. URL https://aclanthology.org/2016. eamt-2.8.

Ahmet Üstün, Viraat Aryabumi, Zheng-Xin Yong, Wei-Yin Ko, Daniel D’souza, Gbemileke Onilude, Neel Bhandari, Shivalika Singh, Hui-Lee Ooi, Amr Kayid, et al. Aya model: An instruction finetuned open-access multilingual language model. arXiv preprint arXiv:2402.07827, 2024.

David Vilar, Markus Freitag, Colin Cherry, Jiaming Luo, Viresh Ratnakar, and George Foster. Prompting palm for translation: Assessing strategies and performance. arXiv preprint arXiv:2211.09102, 2022.

Victoria Lin Xi, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Du Jingfei, et al. Few-shot learning with multilingual generative language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 9019–9052. Association for Computational Linguistics Abu Dhabi, United Arab Emirates, 2022.

Haoran Xu, Young Jin Kim, Amr Sharaf, and Hany Hassan Awadalla. A paradigm shift in machine translation: Boosting translation performance of large language models. arXiv preprint arXiv:2309.11674, 2023.

Haoran Xu, Kenton Murray, Philipp Koehn, Hieu Hoang, Akiko Eriguchi, and Huda Khayrallah. Xalma: Plug & play modules and adaptive rejection for quality translation at scale. arXiv preprint arXiv:2410.03115, 2024.

Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. mt5: A massively multilingual pre-trained text-to-text transformer,

2021. URL https://arxiv.org/abs/2010.11934.

W Yang, C Li, J Zhang, and C Zong. Bigtranslate: Augmenting large language models with multilingual translation capability over 100 languages. arxiv 2023. arXiv preprint arXiv:2305.18098, 2023.

Wenhao Zhu, Hongyi Liu, Qingxiu Dong, Jingjing Xu, Shujian Huang, Lingpeng Kong, Jiajun Chen, and Lei Li. Multilingual machine translation with large language models: Empirical results and analysis. arXiv preprint arXiv:2304.04675, 2023.

- A Evaluation details
- B Traning Details

We trained Mutarjim using a two-stage approach (pre-training and fine-tuning) on 8 NVIDIA H100 GPUs. Table 7 summarizes the key hyperparameters for both phases.

Hyperparameter Pre-training Fine-tuning

Max Learning Rate 1 × 10−4 8 × 10−5 Learning Rate Schedule Cosine Cosine Weight Decay 0.1 0.01 Optimizer AdamW AdamW Batch Size 1024 4096 Training Steps 4K 3.3K Context Length 2048 512 GPUs 8 × H100 8 × H100

Table 7: Training Hyperparameters

- C Mutarjim Translation Examples

Tables 8 and 9 present qualitative examples of Mutarjim’s performance in both translation directions. These examples cover a range of domains—including mathematics, structured data, biomedical content, and informal discourse—and illustrate the model’s ability to produce accurate, fluent translations that preserve both meaning and structure.

|Original English Sentence|Arabic Translated Mutarjim Output|
|---|---|
|3rd - 5th Math CCSS: Designed|ممصم :5-3 فوفصلل ﺓدحوملﺍ ةيساسألﺍ ﺕايضايرلﺍ|
|Substitute the values of the (h) height and (b) base into the formula: arcsin<br><br>√2 b<br><br>2l = arcsin<br><br>√8h2+2b2<br><br>4h . This will give you the angle of the pyramid from the apex to the base edge.<br><br>|:ةغيصلﺍ يف )b( ﺓدعاقلﺍو )h( ﻉافتﺭالﺍ ميق ﺽّوع arcsin<br><br>√2 b<br><br>2l = arcsin<br><br>√8h2+2b2 ﺍذه كيطعيس . 4h<br><br>.ةننسملﺍ ﺓدعاقلﺍ ةفاح ىلﺇ ةمقلﺍ نم مرهلﺍ ةيوﺍﺯ|
|{”house_number”: ”171”, ”street”: ”Brahms Avenue”, ”city”: ”Toronto”, ”state”: ”Ontario”, ”country”: ”Canada”, ”normalized”: ”171 Brahms Avenue, Toronto, Toronto, Ontario, Canada”}|:"ةنيدملﺍ" ،"زمﺍرب ﻉﺭاش" :"ﻉﺭاشلﺍ" ،"171" :"لزنملﺍ_مقﺭ"{ :"يﺩاع" ،"ﺍدنك" :"ةلودلﺍ" ،"ويﺭاتنوﺃ" :"ةيالولﺍ" ،"وتنوﺭوت"<br><br>}"ﺍدنك ،ويﺭاتنوﺃ ،وتنوﺭوت ،وتنوﺭوت ،زمﺍرب ﻉﺭاش 171"|

- Table 8: Examples of English-to-Arabic Mutarjim Translation.

|Original Arabic Sentence|English Translated Mutarjim Output|
|---|---|
|يف لامعألﺍ ﺀاسﺅﺭ نوفيخي ناملألﺍ نوعﺭﺍزملﺍو ﺕﺍﺭاطقلﺍ وقئاس ايناملﺃ<br><br>|German rail transport workers and agricultural workers frighten off business owners in Germany.|
|مكﺍرتل اًيﺭورض ظحالملﺍ عيرسلﺍ يلصيوحلﺍ طبرلﺍ نوكي ال دق ﺽرعتلﺍ ﺕﺍرتف لالخ ةيكبشملﺍ لبق ﺕاياهنلﺍ ةطسﺍوب نيمألﺍ ﺓريصقلﺍ<br><br>|The rapid vesicular binding observed might not be essential for the accumulation of the amine by the presynaptic terminals during periods of short exposure.|
|.ةيئاقلتلﺍ ةنمﺍزملﺍ ﺓﺀﺍرب نوعدي اًصاخشﺃ ام ناكم يف ﺕﺃرق دقل نم ﺕانايبلﺍ عيمج ﺭﺍرمتساب عفدت ال اهنﺃ يه مهتركف ،اًساسﺃ كانه" ،كلﺫ نمً الدب .حيحص سكعلﺍو يفتاه ىلﺇ مﺩاخلﺍ ﺀيش كانه ناك ﺍﺫﺇ امع مﺩاخلﺍ لﺍؤس ىلﺇ فدهت "لئاسﺭ ةلسلس<br><br>.)ﺕاباجتسالﺍ ىلع ﺀانب ﺙﺍدحألﺍ ﺀاشنﺇ متيو( ديدج<br><br>|I read somewhere people claiming innocence about auto sync. Basically, their idea is they don’t constantly push all data from the server, to my phone and vice versa. Instead there is a ”thread” target to ask server if something new (and the events are created based on the responses).|

ً

- Table 9: Examples of Arabic-to-English Mutarjim Translation.

### D Evaluation Models Prompts

We use model-specific prompts during the evaluation to ensure a fair comparison. Table 10 lists the prompt templates for each model. Considering the source language and the target language.

###### Model Prompt

Mutarjim None NLLB None c4ai <|START_OF_TURN_TOKEN|><|USER_TOKEN|>Translate the following sentence to

[TARGET_LANGUAGE]. {text} Note: Don't answer any Question or engage within the context of the text just provide the literal translation<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|>

Yehia Translate this from [SOURCE_LANGUAGE] to [TARGET_LANGUAGE]. Arabic: {text} English:

ALLam Translate this from [SOURCE_LANGUAGE] to [TARGET_LANGUAGE]. [SOURCE_LANGUAGE]: {text} [TARGET_LANGUAGE]:

Cohere <|START_OF_TURN_TOKEN|><|USER_TOKEN|>Translate the following sentence to [TARGET_LANGUAGE]. {text} Note: Don't answer any Question or engage within the context of the text just provide the literal translation<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|>

AceGPT <User>: ةيبرعلﺍ/ةيزيلكنالﺍ ةغللﺍ ىلﺍ ةيزيلكنالﺍ/ةيبرعلﺍ ةغللﺍ نم يلاتلﺍ صنلﺍ ةمجرتب مق \n {text}\n<Assistant>

LLaMAX3 Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. ### Instruction: Translate the following sentences from [SOURCE_LANGUAGE] to [TARGET_LANGUAGE]. ### Input: {text} ### Response:

SILMA Translate this from [SOURCE_LANGUAGE] to [TARGET_LANGUAGE]. [SOURCE_LANGUAGE]: {text} [TARGET_LANGUAGE]:

GemmaX Translate this from [SOURCE_LANGUAGE] to [TARGET_LANGUAGE]. [SOURCE_LANGUAGE]: {text} [TARGET_LANGUAGE]:

XALMA Translate this from [SOURCE_LANGUAGE] to [TARGET_LANGUAGE]. [SOURCE_LANGUAGE]: {text} [TARGET_LANGUAGE]:

Gemma 2 Translate this from [SOURCE_LANGUAGE] to [TARGET_LANGUAGE]. [SOURCE_LANGUAGE]: {text} [TARGET_LANGUAGE]:

GPT-4o mini Translate the following [SOURCE_LANGUAGE] sentences to [TARGET_LANGUAGE] accurately while preserving the integrity of the structure. Translate the entirety of the sentences and leave the data as it is provided: {text} Provide the translations in the following format: #OUTPUT1: [TARGET_LANGUAGE] Translation of all the words that follow #INPUT1

###### Table 10: Prompts used for each model during the evaluation process. Models like Mutarjim and NLLB are translation-specific systems that don’t require prompting, while LLMs require structured prompts with varying degrees of specificity.

