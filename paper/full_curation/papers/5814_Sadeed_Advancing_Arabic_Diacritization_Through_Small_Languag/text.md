## arXiv:2504.21635v2[cs.CL]21Aug2025

[Figure 1]

# Sadeed: Advancing Arabic Diacritization Through Small Language Model

##### Zeina Aldallal, Sara Chrouf, Khalil Hennara, Mohamed Motasim Hamed, Muhammad Hreden, and Safwan AlModhayan

[Figure 2]

Khobar, Saudi Arabia aldallal,chrouf,hennara,hamed,hreden,safwan@misraj.ai

##### Abstract

Arabic text diacritization remains a persistent challenge in natural language processing due to the language’s morphological richness. In this paper, we introduce Sadeed‡, a novel approach based on a fine-tuned decoder-only language model adapted from Kuwain 1.5B Hennara et al. [2025], a compact model originally trained on diverse Arabic corpora. Sadeed is fine-tuned on carefully curated, high-quality diacritized datasets, constructed through a rigorous data-cleaning and normalization pipeline. Despite utilizing modest computational resources, Sadeed achieves competitive results compared to proprietary large language models and outperforms traditional models trained on similar domains. Additionally, we highlight key limitations in current benchmarking practices for Arabic diacritization. To address these issues, we introduce SadeedDiac-25, a new benchmark designed to enable fairer and more comprehensive evaluation across diverse text genres and complexity levels. Together, Sadeed and SadeedDiac-25 provide a robust foundation for advancing Arabic NLP applications, including machine translation, text-to-speech, and language learning tools.

### 1 Introduction

Diacritization, known as ’tashkı¯l’ in Arabic, plays a crucial role in disambiguating text. It serves as the primary vocalization system for distinguishing between words that share the same consonantal structure but differ in meaning and pronunciation. Table 1 illustrates this concept with examples of Arabic words that are visually identical but convey different meanings when diacritized. The importance of diacritization goes beyond mere disambiguation. It is essential to improve various Arabic NLP tasks, including Text-to-Speech (TTS) synthesis Madhfar & Qamar [2021]; Almanea [2021], machine translation Diab et al. [2007]; Zakraoui et al. [2021]; Fadel et al. [2019b], morphological analysis Almanea [2021]; Habash et al. [2016], and Part-of-Speech (POS) tagging Almanea [2021]; Sabtan [2021]. By resolving ambiguities, diacritization significantly improves the accuracy and effectiveness of these downstream tasks.

However, the field of Arabic diacritization faces several challenges:

‡Sadded ( ): This term conveys the notion of sound judgment, precision, or correctness. Phonetically, it can be rendered in English as /sæ"di:d/.

##### Arabic word Pronunciation Part of speech English Translation

/kUlIb@/ Verb Was turned /k2l2b@/ Verb Turn

/k2ll2b@/ Verb Flip

/k2lb/ Noun Heart /kUll2b/ Noun Volatile

/kUlUb/ Noun Plural of heart

- Table 1: Comparison of meanings and pronunciations using diacritics. The table illustrates differences in phonetic transcriptions and their associated meanings, highlighting how diacritics affect pronunciation and meaning.

- • Data Scarcity: There is a notable lack of labeled data. Modern Arabic writing often omits diacritics for efficiency, as adding them can require multiple keystrokes per character, making the process time-consuming and labor-intensive.
- • Writing Styles: The Arabic Text is typically categorized into Classical Arabic (CA) and Modern Standard Arabic (MSA). CA refers to historical manuscripts, while MSA represents contemporary writing styles. Although most available diacritization datasets are in Classical Arabic, training diacritization models predominantly on CA data degrades their performance in Modern Standard Arabic.
- • Contextual Dependence: Accurate diacritization often requires understanding the full sentence context, a factor sometimes considered in existing models. Table 2 illustrates how Arabic diacritics can change based on the context of subsequent words in a sentence, dramatically altering meaning and grammatical function.
- • Benchmark Limitations: Current benchmarks often focus exclusively on either CA or MSA, lacking a comprehensive representation of both styles or the diversity of text genres. Additionally, some widely-used benchmarks have been found to contain errors in data splitting and labeling.

In addition to these well-known challenges, there are other less frequently mentioned issues, such as the ambiguity related to punctuation. This problem can significantly affect the correct diacritization of Arabic text, as illustrated in Table 3, which provides examples of how punctuation can alter the diacritization and meaning of sentences.

Arabic word Diacritic form Pronunciation Sentence English Transla-

tion َ

|لبقتسﺍ<br><br>|َلَ بْقتْسﺍ /ist2qb2l@/ َمَْوي ِةَسَﺭْدَمْلﺍ ُريِدُم َلَ بْقَ تْسﺍ َنِيقِّوَفَ ُتمْلﺍ َﺏالطلﺍ سْمﺃ <br><br>The school principal received the outstanding students yesterday.|
|---|---|
| |َلِبْقتْسﺍ /istUqbil@/ َمَْوي ِةَسَﺭْدَمْلﺍ ُريِدُم َلَ بُْقتْسﺍ ُهَمَاقﺃ  ٍرِيبَك لْفَ ح ِب سْمﺃ <br><br>َهل ُﺏالطلﺍ<br><br>The school principal was received yesterday with a grand celebration organized by the students for him.|
|فيرشلﺍ ثيدحلﺍ بتك<br><br>|فيِرشلﺍ ثيِ َدحلﺍ بَ تَك /k2 t2b@/ /aladi@/ /a-arif@/<br><br>َفيِرشلﺍ َثيِ َدحلﺍ َبَ تَك ِنْرَقلﺍ يف يِﺭاَخُ بلﺍ ُماَم لﺍ يِرْجِهلﺍ ِثِ لاثلﺍ<br><br>Imam Al-Bukhari wrote the Noble Hadith in the third century AH|
| |فيِرشلﺍ ثيِ َدحلﺍ بُ تُك /kUtUbU/ /aladthi/ /a-arifi/<br><br>ِفيِرشلﺍ ِثيِ َدحلﺍ ُبُ تُك ِﺕَاه م ِب ُفَ رُْعت ٌةتِس ِبُ تُك لﺍ<br><br>The books of Nobel Hadith are six known as the Mothers of Books.|

ّ ّ ِ

ُ

ٍ ِ ُ ّ ّ

َ ّ َ َ ّ

ِ ّ

ِ ّ ّ ِ ّ ِ ُ ّ

ّ ّ

- Table 2: illustrates how diacritics in Arabic can change based on the context of subsequent words in a sentence, dramatically altering meaning and grammatical function

Category Sentence English Translation َ ّ َ َ

|Lack of punctuation|لَمَعْلﺍَو ،ٌ ُفُك َديِ َدجْلﺍ َفظَوُمْلﺍ ّنﺇ" :ُريِدُمْلﺍ َلَاق ".َكِلَﺫَ رِكُْني ْنﺃ  ٌدَحﺃ  ُعيِطَ تْ َسي اَل .ٍةَعْرُ سِب ُمّدَقََتي<br><br>The director said, ”The new employee is competent, and the work is progressing quickly. No one can deny that.”|
|---|---|
| |لَمَعْلﺍَو ٌ ُفُك َديِ َدجْلﺍ َفظَوُمْلﺍ نﺇ ُريِدُمْلﺍ َلَاق َكِلَﺫَ رِكُْني ْنﺃ  ٌدَحﺃ  ُعيِطَ تْ َسي اَل ٍةَعْرُ سِب ُمّدَقََتي<br><br>The director said that the new employee is competent and the work is progressing quickly no one can deny that.<br><br>|
|Punctuation differences|!ﺀاَمسلﺍ يف اَم لَ مْجﺃ  اَم How beautiful is what’s in the sky!|
| |؟ﺀاَمسلﺍ يف اَم لَ مْجﺃ  اَم What is the most beautiful thing in the sky?|

َ

ُ ّ َ ّ َ

َ

ّ َ ِ َ ِ ّ َ ِ ُ

- Table 3: Punctuation can cause diacritization ambiguities in Arabic by altering sentence structure and grammatical roles, as seen in shifts between case endings and changes in word function (e.g., verb vs. adjective).

To address these challenges, we introduce Sadeed, a compact and task-specific Arabic diacritization model fine-tuned on high-quality, noise-free diacritized text. Sadeed is adapted from KuwainHennara

et al. [2025], a decoder-only small language model (SLM) pre-trained on diverse Arabic corpora. Optimized for accurate performance on clean, structured input, Sadeed demonstrates competitive results compared to proprietary models. We also propose SadeedDiac-25„ a novel open-source benchmark. SadeedDiac-25 incorporates both Classical Arabic and Modern Standard Arabic texts, carefully curated and reviewed by experts to avoid the pitfalls of previous benchmarks. This comprehensive approach aims to provide a more accurate and reliable evaluation framework for Arabic diacritization systems.

In summary, our main contributions are:

- • Sadeed: a compact decoder-only language model fine-tuned specifically for Arabic diacritization. Despite its small size and limited training resources, Sadeed achieves competitive results against proprietary models and outperforms traditional models trained in-domain.
- • SadeedDiac-25: We present a novel, comprehensive benchmark for Arabic diacritization, which includes:

- – A balanced representation of both Classical Arabic (CA) and Modern Standard Arabic (MSA) texts.
- – Carefully curated and expert-reviewed data to ensure accuracy and reliability.

- • Error Analysis of existing benchmarks: We conduct a thorough analysis of existing benchmarks Abbad and Xiong Abbad & Xiong [2020] and CATT Alasmary et al. [2024], identifying errors in data splitting, labeling, and content.
- • Diacritization Dataset: We also release a high-quality, publicly available dataset derived from the Tashkeela Corpus Zerrouki & Balla [2017], which has been rigorously filtered and cleaned to ensure consistency and reliability. While it is compatible with the commonly used Fadel test set Fadel et al. [2019a] for evaluation, our dataset significantly expands the training pool, offering approximately 1 million diacritized examples and approximately 53 million words. This resource is intended to support the development and benchmarking of Arabic diacritization models at scale.

Together, these contributions provide a more robust foundation for advancing research in Arabic diacritization. They also highlight the importance of tailored benchmarks, clean training data, and domain-specific modeling in achieving reliable and reproducible performance in this linguistically complex task.

### 2 Related work

Previous research has examined a broad array of methodologies for the Arabic diacritization task, evolving from rule-based approaches Nelken & Shieber [2005]; Pasha et al. [2014]; Alnefaie & Azmi [2017] to traditional machine learning models Darwish et al. [2017a]; Abdelali et al. [2016] and, more recently, to sophisticated deep learning architectures. In this section, we focus on these advanced methods, categorizing them into two primary classes: Recurrent-based Models and Transformer-based Models, which have proven most effective in addressing the complexities of Arabic diacritization.

#### 2.1 Recurrent-based Models

Recurrent neural networks, particularly LSTM and Bidirectional LSTM (BiLSTM) architectures, have demonstrated significant success in modeling long-range dependencies within sequential data, a crucial aspect of the diacritization task. These models have demonstrated the ability to model the contextual information necessary for accurate diacritization effectively. The following section reviews a range of influential studies that adopt such models: Several deep learning models based on feedforward neural networks (FFNNs) and recurrent neural networks (RNNs) that are proposed in Fadel

- et al. [2019a], integrating techniques such as 100-hot encoding, embeddings, conditional random

fields (CRF), and Bayesian Neural Gates (BNG). The models are trained on an adapted version of the Tashkeela corpus, with additional data from the Classical Arabic section of Tashkeela and the Holy Quran. Evaluation is conducted on the original "Fadel Tashkeela" test set as well as datasets from the Al-Shamela Library and the MSA section of Tashkeela. A sequence-to-sequence model utilizing a BiLSTM-CRF architecture is proposed in Al-Thubaity et al. [2020], adopting a characterlevel input-output framework for predicting diacritics. The model is trained on four distinct datasets: the King Abdulaziz City for Science and Technology Text-to-Speech (KACST TTS) dataset, the Holy Quran, Sahih Al-Bukhari, and the Penn Arabic Treebank (ATB). Evaluation is performed on a representative subset of this combined corpus. A hierarchical architecture incorporating cross-level attention mechanisms is introduced in AlKhamissi et al. [2020], separating processing at the word and character levels. This model is trained and evaluated on the "Fadel Tashkeela" dataset. In Darwish et al. [2020], a feature-rich BiLSTM model is proposed, emphasizing the critical role of integrating linguistic features for diacritization. The model is trained on two distinct corpora: a 9.7 million-token corpus for Modern Standard Arabic (MSA) and a 2.7 million-token corpus for Classical Arabic (CA). The results demonstrate the significant impact of training data size on the model’s performance, underscoring the importance of large, high-quality datasets for accurate diacritization. In Abbad & Xiong [2020], a model with four BiLSTM layers is introduced. The model is trained on a subset of the Abbad training set, with performance evaluation conducted using the first file from the Abbad test set as the benchmark. A recent model, 2SDiac Bahar et al. [2023], combines bidirectional LSTM (BiLSTM) and self-attention mechanisms with a Guided Learning strategy. This model is trained on the Tashkeela corpus and Arabic Treebanks (ATB) and evaluated on the Tashkeela, ATB, and WikiNews datasets.

These studies collectively underscore the central role of recurrent architectures—particularly BiLSTMbased models—in advancing Arabic diacritization, highlighting their capacity to effectively capture contextual dependencies and generalize across diverse linguistic domains.

#### 2.2 Transformer-based Models

Transformer-based Models have emerged as a powerful paradigm in Arabic diacritization, leveraging the strengths of large pre-trained language models and transfer learning techniques. Recent studies have demonstrated the efficacy of these approaches in significantly improving diacritization accuracy. In Madhfar & Qamar [2021], three deep learning models are proposed for Arabic text diacritization: a baseline LSTM model, an encoder-decoder model inspired by Neural Machine Translation (NMT), and an efficient encoder-only model. These models were trained and evaluated on the Tashkeela corpus. The results show that the encoder-only model outperforms the other architectures, highlighting its efficiency and effectiveness in handling the diacritization task. Furthermore, Skiredj & Berrada [2024] introduced a method called PTCAD, witch is (Pre-FineTuned Token Classification for Arabic Diacritization), which integrates transfer learning with token classification. This approach, trained and validated on the "Abbad Tashkeela" and "Fadel Tashkeela datasets, highlights the increasing trend of applying transfer learning techniques to improve the performance of Arabic diacritization tasks. In addition, Alasmary et al. [2024] introduced fine-tuned transformer architectures, both encoder-only and encoder-decoder, using a character-based BERT model. This model employs the Noisy-Student method to enhance performance and is further fine-tuned on the Tashkeela dataset, with evaluations conducted on Wikinews and a newly created CATT dataset. Lastly, the study Kharsa et al. [2024] presents a BERT-based model pre-trained on extensive Arabic corpora and fine-tuned specifically for diacritization. This model is trained and evaluated on the

"Abbad Tashkeela" and "Fadel Tashkeela" datasets, Their results underscore the effectiveness of leveraging pre-trained contextual embeddings for improved diacritization accuracy.

As seen across the reviewed studies, there exists a wide range of training algorithms, architectural choices, and methodological strategies for Arabic diacritization. We have emphasized the training and evaluation datasets employed in each case, as these decisions significantly influence the reported outcomes. By analyzing these aspects, we identify recurring issues—particularly in the use of overlapping data between training and test sets. In Section 6, we delve deeper into these concerns, surveying commonly used benchmarks and highlighting problematic evaluation practices that may compromise the reliability of comparative results.

### 3 Diacritization Dataset

Sadeed training dataset leverages both the Tashkeela corpus Zerrouki & Balla [2017] and the Arabic Treebank (ATB-3) Maamouri et al. [2008]. Tashkeela is the largest publicly available corpus for the diacritization task, comprising approximately 75 million words, predominantly in classical Arabic. Only about 1.15% of the text is in Modern Standard Arabic (MSA). In contrast, ATB-3 includes around 300,000 words, primarily in MSA and sourced from news articles.

Several cleaned and preprocessed versions of these datasets exist; however, many are constrained either by limited size—such as the "Fadel Tashkeela" dataset Fadel et al. [2019a]—or by flaws in the cleaning and segmentation process. For instance, the "Abbad" version of Tashkeela Abbad & Xiong [2020] exhibits sentence truncation that disrupts contextual continuity, a factor critical for achieving accurate diacritization.

To optimize the dataset for this task, we implemented a rigorous preprocessing pipeline:

Text Cleaning: As noted by Fadel Fadel et al. [2019a], the Tashkeela corpus Zerrouki & Balla [2017] suffers from various issues related to text quality and diacritization consistency. To address these challenges, we applied a comprehensive preprocessing pipeline aimed at cleaning the data, correcting diacritization errors, and normalizing the Arabic text. While our process is similar to Fadel’s, we aimed to maintain the original text’s integrity by preserving non-Arabic characters and other symbols that may occur in Arabic texts, as Kuwain is capable of handling such data.

The preprocessing pipeline began by applying the same rigorous cleaning function used in the pretraining of the base model Kuwain Hennara et al. [2025], followed by additional normalization steps to ensure consistency in diacritization and eliminate common textual errors. Specifically, we unified the diacritization style by omitting the suku¯n (absence of a vowel) on elongation letters and the definite article l¯am when followed by a sun letter. Furthermore, we corrected the diacritization of frequently occurring words—particularly stop words—that typically appear with incomplete or inconsistent vocalization, despite having a single possible diacritization such as and .

Finally, we addressed the issue of iltiqa’ assakinayn (the occurrence of two adjacent consonants without an intervening vowel), which is inconsistently handled across the corpus—some texts apply the phonological rule correctly, while others do not. To resolve this inconsistency, we automatically adjusted the vowelization of the first consonant based on standard Arabic phonological rules.

Text Chunking: We segmented the corpus into coherent chunks (samples) of 50-60 words each. To achieve this, we employed a hierarchical approach that prioritized splitting and rejoining text chunks using multiple separators (end-of-sentence punctuation marks, line breaks, quotation marks, parentheses, and commas). This method was designed to preserve the syntactical dependencies of the chunk words as much as possible. By using this hierarchical method, we aimed to avoid splitting the text at punctuation marks that might disrupt the syntactical coherence within each chunk. For example, we prioritized splitting at stronger punctuation marks or natural breaks in the text before resorting to splitting at commas, which often connect closely related clauses or phrases.

Dataset filtering: The dataset filtering process involves two main steps to optimize the training samples with fully diacritized words. Initially, we excluded examples with more than two words lacking diacritical marks, preserving 93% of the dataset. Additionally, we removed examples containing three or more words with partial diacritics. After this filtering, only 10.7% of the sentences in the dataset contain at most two partially diacritized words, ensuring a high standard of diacritic completeness. To effectively use Fadel Fadel et al. [2019a] test set, which is already a split of Tashkeela Zerrouki & Balla [2017], we eliminated overlapping examples with this test set. This process reduced the overlap to only 0.4% of examples, ensuring that sentences in each sample have at most two overlapping words with Fadel test set.

The resulting dataset comprises 1,042,698 examples totaling approximately 53 million words. It is normalized, contains minimal missing diacritics, and chunked to preserve syntactic and contextual dependencies. This dataset can be used as training data alongside Fadel’s test set without the risk of data leakage, thus providing a solid foundation for models’ training and evaluation. The data is made publicly available∗.

### 4 Benchmark

Existing benchmarks for Arabic diacritization present notable structural and linguistic limitations that hinder comprehensive evaluation.

First, we observed a clear divide among the benchmarks: some focused exclusively on Classical Arabic (CA), such as Fadel’s Fadel et al. [2019a] and Abbad’s Abbad & Xiong [2020], both of which are subsets of the Tashkeela corpus Zerrouki & Balla [2017], while others targeted only Modern Standard Arabic (MSA), such as CATT Alasmary et al. [2024] and WikiNews Darwish et al. [2017b]. This separation fails to capture the full spectrum of Arabic language usage, limiting the generalizability of models trained on these datasets.

Moreover, several of these corpora exhibit notable flaws. Abbad Tashkeela Abbad & Xiong [2020], for instance, suffers from segmentation issues—discussed earlier—which disrupt contextual coherence. The CATT benchmark Alasmary et al. [2024] contains numerous spelling and grammatical errors; Section 6.2 provides a detailed analysis. As for WikiNews Darwish et al. [2017b], its primary limitation lies in being derived exclusively from news articles, offering limited domain diversity. Additionally, it contains a large number of named entities, many of which show inconsistent diacritization, particularly in the final letters, often defaulting to suk¯un inappropriately.

To address these shortcomings, we introduce SadeedDiac-25—a comprehensive benchmark that

∗https://huggingface.co/datasets/Misraj/Sadeed_Tashkeela

###### unifies both Modern Standard Arabic (MSA) and Classical Arabic (CA) within a single dataset. Our benchmark spans a wide range of topics and writing styles, capturing linguistic diversity across domains such as sports, politics, religion, and culinary arts, as illustrated in Figure 1.

[Figure 10]

Figure 1: Benchmark Category Statistics

The development of SadeedDiac-25 followed a rigorous and systematic process to ensure highquality data. As illustrated in Figure 1, we sourced texts from a variety of domains to achieve broad linguistic and topical coverage. The benchmark comprises 1,200 paragraphs. Of this data, 50% is in Modern Standard Arabic (MSA)—including 454 paragraphs of originally curated content and 146 paragraphs from WikiNews Darwish et al. [2017b]. These MSA samples range from 40 to 50 words in length. The remaining 50% represents Classical Arabic (CA), consisting of 600 paragraphs drawn from the Fadel test set Fadel et al. [2019a]. The original content included in the benchmark was curated through the following multi-stage process:

- • Data Collection: Text was sourced from a diverse range of web articles spanning various domains to ensure topic variability and linguistic richness.
- • Initial Diacritization: The collected text was automatically diacritized using a proprietary large language model. This initial step significantly accelerated the overall process by providing a strong baseline that experts could efficiently refine, rather than starting from unannotated text.
- • Expert Review: A two-stage expert review process was applied to ensure the quality and accuracy of the diacritization:

- – First Stage: Two independent experts reviewed the auto-diacritized text, correcting any errors or inconsistencies.
- – Second Stage: Each expert then reviewed the other’s corrections to cross-validate the annotations and resolve any remaining discrepancies or ambiguities.

This rigorous pipeline ensured that the benchmark content is both linguistically accurate and representative of real-world Arabic usage across various contexts.

By conducting the diacritization process in-house and incorporating expert review, we ensure that the dataset remains entirely novel to existing models, as they have not been exposed to the diacritized form of our data. This directly addresses the concern raised in the CATT Alasmary et al. [2024] regarding the potential familiarity of language models with benchmark datasets. Our approach effectively prevents data leakage and results in a genuinely unseen evaluation set, thereby enabling a more reliable and accurate assessment of models’ true diacritization capabilities. SadeedDiac-25 is publicly available∗ to support further research and development in Arabic diacritization and to facilitate fair and reproducible evaluation of diacritization models.

### 5 Model

Sadeed is a fine-tuned variant of Kuwain Hennara et al. [2025], a small language model (SLM) specifically tailored for Arabic using an original language injection technique. The fine-tuning process was carefully designed to optimize the model for the task of Arabic diacritization. As part of this process, we reformulated diacritization as a Question-Answering (QA) task, enabling more focused and efficient training by leveraging the model’s generative capabilities in a structured manner. To prepare the model for fine-tuning, we applied a consistent template transformation across the entire training dataset, as illustrated in Figure 3. This step was crucial in adapting the general-purpose Kuwain model to the specialized task of diacritization. The complete workflow of our approach is presented in Figure 2.

[Figure 12]

Figure 2: Sadded pipeline, it shows how we get sadeed from our model kuwain-1.5

This fine-tuned version of Kuwain, which we call Sadeed, demonstrates the potential of adapting compact models for specific NLP tasks. The resulting model offers a balance between performance and efficiency, making it particularly suitable for Arabic diacritization tasks. For detailed information about the training methodology and hyperparameters, please refer to Appendix A.

During inference, non-diacritized text is input into the model using the provided template in the inputs field. The model then generates the corresponding diacritized text for the input. As a generative model, the output generally matches the input. However, there are instances where the generated text differs from the original. This discrepancy, known as hallucination, occurs because

∗https://huggingface.co/datasets/Misraj/SadeedDiac-25

[Figure 14]

[Figure 15]

(a) (b)

Figure 3: (a) is The System prompts we use to fine-tune our models. (b) An example from our dataset after applying the template

the model generates the most probable tokens based on the given context. To correct the resulting text, we compare the input with the output. We remove any added words, restore missing words, and replace altered words in the output with their non-diacritized versions. To identify these discrepancies, we use the Needleman-Wunsch Likic [2008] algorithm for sequence alignment. The resulting text is a diacritized version of the original, with some words remaining undiacritized due to being hallucinated. see Appendix B for examples of the model’s hallucinations and Appendix C for detailed examples of the model’s output over various text types.

### 6 Analysis of Existing Benchmarks and Training Practices

Despite recent advances in Arabic diacritization, the field still faces critical challenges related to dataset design, benchmark reliability, and training methodologies. Many existing datasets and benchmarks suffer from issues such as inconsistent annotation, errors, and content overlapping between training and test sets. These issues undermine the validity of the evaluation results and complicate fair comparisons between models. In this section, we examine the prevailing practices in dataset usage, analyze benchmark construction, and identify limitations that could hinder progress in the field.

#### 6.1 Overlap Analysis Between Fadel and Abbad Datasets

To ensure the validity and integrity of model evaluation, it is essential to examine potential overlaps between training and test data. In this subsection, we analyze the degree of content overlap between two widely used diacritized Arabic datasets: Fadel Tashkeela Fadel et al. [2019a] and Abbad Tashkeela Abbad & Xiong [2020]. Our analysis reveals substantial overlap in both the training and benchmark subsets of these datasets. This overlap calls into question their joint use in training and evaluation pipelines, as it can lead to data leakage, inflated performance results, and a distorted view of a model’s true generalization capabilities. We quantify the overlap and discuss its implications for benchmark design and best practices in training.

Since the Fadel and Abbad samples differ in length, with Abbad examples being generally shorter, we segmented the Fadel samples using punctuation markers, without accounting for context, to match the segmentation strategy used for Abbad. This allowed for a consistent comparison between the two datasets, focusing on the recurrence of contextual structures rather than individual word repetitions. Similarity is calculated as the ratio of the number of words in the overlapping segments

to the total number of words in the original sample.

- • Overlap (Identical Samples) refers to:

- – In the Fadel set: Instances where all segments of a Fadel sample are found in the Abbad set, resulting in a similarity score of 1.0.
- – In the Abbad set: Instances where an Abbad sample exactly matches any segment derived from a Fadel sample.

- • Overlap (Samples with Similarity > 0.5) is calculated only for Fadel samples. It includes cases where more than 50% of the words in a Fadel sample appear in segments that overlap with the Abbad dataset. This measure does not apply to Abbad samples, as each consists of a single segment that either exactly matches part of a Fadel sample or does not match at all.

- Table 4 provides an overview of the overlap between the Abbad (Train + Validation) set and the Fadel test set. It reveals that 6065 samples from the Abbad training set (0.2%) are identical to those in the Fadel test set. In contrast, 865 samples from the Fadel test set (34.6%) are fully present in the Abbad training set. Furthermore, 1703 samples from the Fadel test set (68.12%) exhibit a similarity score greater than 0.5 when compared to the Abbad training set.

Dataset Total Samples Overlap (Identical Samples)

Overlap (Similarity > 0.5)

Abbad Train 2,940,000 6065 (0.2%) Fadel Test 2,500 865 (34.6%) 1703 (68.12%)

- Table 4: Comparison of datasets showing the total number of samples, overlap of identical samples (with percentage), and overlap of samples with similarity greater than 0.5 (with percentage).
- Table 5 shows that 657 samples from the Fadel training set (1.25%) are completely identical to those from the Abbad test set. Similarly, 7045 samples from the Abbad test set (4.69%) are fully present in the Fadel training set. Furthermore, 1,345 samples from the Fadel training set (2.56%) exhibit a similarity score greater than 0.5 compared to the Abbad test set.

Dataset Total Samples Overlap (Identical Samples)

Overlap (Similarity > 0.5)

Fadel Train 52,500 657 (1.25%) 1345 (2.56%) Abbad Test 150,015 7045 (4.69%) —

- Table 5: Comparison of datasets showing the total number of samples, overlap of identical samples (with percentage), and overlap of samples with similarity greater than 0.5 (with percentage).

This overlap raises serious methodological concerns for model evaluation in Arabic diacritization tasks. Researchers should be cautious when using these datasets together in a training-evaluation pipeline, as significant content duplication could artificially inflate performance metrics and provide misleading assessments of model generalization. Notably, such overlaps have been observed in the evaluation setup of the PTCAD model Skiredj & Berrada [2024], where the model is trained and evaluated on both Abbad and Fadel datasets. By providing our cleaned version of the Tashkeela

dataset, which removes problematic overlaps and enables fair evaluation on the Fadel benchmark, we effectively address these data leakage concerns.

#### 6.2 CATT Benchmark Analysis

We critically examined the CATT benchmark introduced in Alasmary et al. [2024], which presents the Character-based Arabic Tashkeel Transformer (CATT) model and claims state-of-the-art performance on its own curated benchmark. While the proposed CATT model demonstrates strong results, our analysis reveals substantial limitations in both the benchmark dataset and the evaluation methodology, raising concerns about the generalizability and reliability of the reported performance. One of the most critical issues with the CATT dataset is the complete removal of punctuation marks. Punctuation plays a vital role in defining sentence boundaries and guiding syntactic and semantic interpretation—both of which are essential for accurate diacritization. Without punctuation, the model loses key contextual cues that help disambiguate words and phrases based on sentence structure. This limitation significantly hinders the model’s ability to leverage broader sentence-level context, thereby constraining its effectiveness and potentially inflating its reported performance in controlled settings.

A linguistics expert conducted a detailed review of 30% of the CATT evaluation data, identifying several distinct categories of diacritization errors (see Appendix D for a comprehensive tabulation). These errors can be classified as follows:

- • Diacritization ambiguity: Occurs in situations such as short sentences that prevent correct diacritization, often in passive voice constructions.
- • Partially diacritized: Some letters in certain words are not diacritized.
- • Erroneous diacritization: Incorrect diacritization.
- • Misplaced diacritics: Incorrect placement of vowels. This category includes false-diacritization, where vowels are transposed.

The limitations identified in CATT benchmark pose significant methodological concerns, potentially resulting in inaccurate evaluation outcomes. By presenting our carefully curated benchmark, developed through a rigorous and systematic process, we effectively address these critical shortcomings, providing researchers with a more reliable and robust evaluation framework.

### 7 Evaluation

To thoroughly assess the performance of our Sadeed model, we conducted an extensive series of experiments using well-established benchmarks in Arabic diacritization. This section reports the results of these evaluations, offering a detailed comparison between Sadeed and state-of-the-art models across multiple datasets and evaluation metrics. To ensure transparency and reproducibility, we have made the evaluation code † publicly available.

†https://github.com/misraj-ai/Sadeed

#### 7.1 Evaluation on Fadel Benchmark

We first evaluated Sadeed on the widely-used Fadel Fadel et al. [2019a] test split to establish a clear comparison with existing state-of-the-art models. Although the Fadel dataset is considered the most reliable benchmark for classical Arabic, it generally lacks consistency in handling cases of adjacent consonants without intervening vowels (iltiqa¯‘ as-s¯akinayn). To address this limitation, we systematically the vowelization of the first consonant based on standard Arabic phonological rules. This refined and phonologically consistent version of the Fadel test set has been made publicly available to facilitate accurate and reliable benchmarking.

Table 6 presents our evaluation results in terms of Word Error Rate (WER) and Diacritic Error Rate (DER), comparing Sadeed to existing state-of-the-art models. To ensure a fair and transparent comparison, we report performance metrics on both the original and the phonologically corrected versions of the Fadel test set.

Including No Diacritic Excluding No Diacritic w/case ending w/o case ending w/case ending w/o case ending DER WER DER WER DER WER DER WER D3 AlKhamissi et al. [2020] 1.83 5.34 1.48 3.11 2.09 5.08 1.69 3.00

Paper

SUKOUN Kharsa et al. [2024] 1.16 3.34 0.96 1.96 1.23 3.03 1.00 1.77 PTCAD Skiredj & Berrada [2024] 1.1 4.19 - - - - - -

Fadel Fadel et al. [2019b] 1.78 5.38 1.39 3.04 2.05 5.17 1.60 2.96 Sadeed- Fadel original 1.6814 4.4914 0.7889 1.8256 1.5758 4.1469 0.7761 1.7955 Sadeed- Fadel corrected 1.2386 2.9375 0.7632 1.7376 1.1432 2.6286 0.7518 1.7115

Table 6: Performance comparison of Arabic diacritization models on the Fadel dataset in terms of Diacritic Error Rate (DER) and Word Error Rate (WER)

As shown in Table 6, Sadeed achieves state-of-the-art (SOTA) performance on the Fadel dataset in terms of Word Error Rate (WER), particularly when the metric excludes non-diacritized characters in the test set. This notable improvement over previous models underscores the robustness of our approach in handling the intricate challenges of Arabic diacritization.We further argue that our Diacritic Error Rate (DER) also constitutes SOTA performance, particularly when considering other models ware trained on a combination of the Abbad and Fadel datasets—datasets that exhibit significant overlap, as detailed in Section 6.1. This overlap raises concerns about inflated performance metrics due to data leakage. For a more comprehensive discussion, refer to Section 8.

#### 7.2 Evaluation on WikiNews Benchmark

To ensure a fair comparison of the performance of our model, we evaluated Sadeed on WikiNews dataset Darwish et al. [2017b], which is widely used for assessing diacritization models on Modern Standard Arabic. This evaluation is particularly important, given that a subset of WikiNews data was also incorporated into our Sadeed-Daic-25 benchmark. By including this assessment, we aim to provide a consistent reference point against existing approaches. Table 7 presents the performance of Sadeed model in comparison to previously reported results in this dataset.

The results presented in Table 7 show that Sadeed achieves competitive performance in the WikiNews dataset Darwish et al. [2017b], although it does not outperform the model proposed by Darwish

- et al. [2020], nor match its performance on the Fadel dataset. This discrepancy can probably be

Including No Diacritic Excluding No Diacritic

w/case ending w/o case ending w/case ending w/o case ending DER WER DER WER DER WER DER WER 2SDiac Bahar et al. [2023] 11.6 33.2 9.9 22.7 13.2 32.2 11.1 22.1

Paper

CATT Alasmary et al. [2024] 5.963 20.060 3.631 11.310 - - - FRRNN Darwish et al. [2020] 3.7 6.0 0.9 2.9 - - - -

Sadeed 5.2464 14.6352 3.1107 8.4400 3.5895 12.2182 1.8500 5.8926

- Table 7: Performance comparison of Arabic diacritization models on the WikiNews dataset in terms of Diacritic Error Rate (DER) and Word Error Rate (WER)

attributed to Sadeed’s limited exposure to Modern Standard Arabic (MSA) during training. In contrast, other models, such as that of Darwish et al. [2017a], were trained on datasets that share the same distribution as WikiNews, giving them a natural advantage. These findings suggest that increasing the amount of MSA-specific training data could significantly enhance model performance on MSA benchmarks. However, this remains a challenge because of the scarcity of publicly available diacritized MSA corpora.

- 7.3 Evaluation on the SadeedDiac-25 Benchmark

A key contribution to our work is the introduction of SadeedDiac-25, a new benchmark specifically designed to offer a fresh, diverse, and rigorous evaluation set for Arabic diacritization. This benchmark addresses key limitations in existing datasets by providing a more comprehensive and representative assessment of model performance across different varieties of Arabic.

To evaluate the effectiveness of current models in Arabic diacritization, we conducted an extensive benchmarking study using SadeedDiac-25. The evaluation covers both our model, Sadeed, and leading proprietary large language models, Claude 3.7 Sonnet, GPT-4, and Gemini-Flash 2.0, and prominent open source Arabic models, namely Aya-8B Aryabumi et al. [2024], ALLaM-7B-Instruct Bari et al. [2024], Yehia-7B Navid-AI [2025], Jais-13B Sengupta et al. [2023], Gemma-2-9BTeam et al. [2024], and SILMA-9B-Instruct-v1.0Team [2024].

Each model was prompted to diacritize the texts sampled from SadeedDiac-25. Since raw model output often contains hallucinations, such as missing, altered, or inserted words, we applied the Needleman-Wunsch alignment algorithm to automatically correct structural discrepancies while preserving the diacritization generated by the models.

- Table 8 presents a comparative overview of the performance of the evaluated models on the SadeedDiac25 benchmark, offering insights into their relative strengths and limitations in this task.

The results in Table 8 reveal several important trends. Claude 3.7 Sonnet consistently achieved the best performance across all evaluation metrics, producing the fewest diacritization errors and exhibiting minimal hallucination, highlighting its superior generalization and robustness on Arabic text. Compared to open source Arabic models, Sadeed demonstrated the strongest performance, outperforming all other open models by a significant margin and achieving results that are competitive with leading proprietary models. However, the primary limitation of Sadeed lies in its hallucination rate: approximately 7.19 points of its total 9.92 WER are attributed to hallucinated

With Case Ending Without Case Ending

Model

Hallucinations DER WER DER WER

###### Claude-3-7-Sonnet-Latest 1.3941 4.6718 0.7693 2.3098 0.821

GPT-4 3.8645 5.2719 3.8645 10.9274 1.0242 gemini-flash-2.0 3.1926 7.9942 2.3783 5.5044 1.1713

Sadeed 7.2915 13.7425 5.2625 9.9245 7.1946 aya-23-8B 25.6274 47.4908 19.7584 40.2478 5.7793

ALLaM-7B-Instruct 50.3586 70.3369 39.4100 67.0920 36.5092 Yehia-7B 50.8801 70.2323 39.7677 67.1520 43.1113 jais-13B 78.6820 99.7541 60.7271 99.5702 61.0803

gemma-2-9b 78.8560 99.7928 60.9188 99.5895 86.8771 SILMA-9B-Instruct-v1.0 78.6567 99.7367 60.7106 99.5586 93.6515

Table 8: Performance comparison of Arabic diacritization models on the SadeedDiac-25 dataset in terms of Diacritic Error Rate (DER) and Word Error Rate (WER)

words, likely a consequence of its relatively small model size.

It is also worth noting that Sadeed’s results on the SadeedDiac-25 benchmark are lower than its performance on the Fadel test set. This discrepancy can be attributed to the more comprehensive nature of SadeedDiac-25, which covers both Modern Standard Arabic (MSA) and Classical Arabic (CA) texts. Since Sadeed was trained on very limited MSA data, its performance tends to decline when faced with MSA-dominant content, highlighting the need for broader training exposure to achieve consistent results across different Arabic varieties.

The broader set of Arabic open-source models exhibited noticeably bad performance. While they generally understand the diacritization task, they have not been explicitly trained or fine-tuned for it, leading to inconsistent and often suboptimal outputs. Among these, Aya-8B stood out as the best-performing open-source model after Sadeed. Although its performance shows higher variance across samples, it produced fewer hallucinations compared to other models, suggesting that Aya-8B could achieve substantial gains if fine-tuned specifically for diacritization.

Overall, these results highlight the critical role of task-specific training in achieving high diacritization accuracy, and reinforce the value of specialized models such as Sadeed for this challenging and linguistically nuanced task.

### 8 Discussion & Limitations

#### 8.1 Dataset Contamination and Its Implications

The studies by Kharsa et al. [2024] and Skiredj & Berrada [2024] utilize the Fadel Fadel et al. [2019a] and Abbad Abbad & Xiong [2020] datasets for model training, but face a significant challenge due to substantial overlap between these datasets, as both derive from the same Tashkeela Zerrouki & Balla [2017] source. Our in-depth analysis in Section 6.1 reveals high contamination between these benchmarks, raising concerns about previously reported results. This overlap suggests potential inflation of performance metrics, as models may inadvertently be tested on data similar to their training sets. This contamination underscores the critical need for careful dataset curation and highlights the possibility of overly optimistic results in previous research, emphasizing the impor-

tance of our SadeedDiac-25 benchmark, which provides a truly independent and diverse evaluation resource for Arabic diacritization models.

#### 8.2 Challenges with Modern Standard Arabic (MSA)

Our model demonstrates strong overall performance but shows limitations with Modern Standard Arabic (MSA) text due to insufficient MSA training data. While excelling in Classical Arabic diacritization, it underperforms on MSA content in our SadeedDiac-25 benchmark. This performance gap validates our benchmark’s effectiveness in identifying improvement areas for current diacritization models. To address this limitation, we are expanding our dataset with carefully diacritized MSA texts following our benchmark’s rigorous methodology. This effort aims to enhance our model’s capabilities across the full spectrum of Arabic language variants.

#### 8.3 Importance of Rigorous Benchmark Creation

We urge researchers to exercise rigorous standards when creating Arabic diacritization benchmarks. This task requires a deep understanding of Arabic grammatical rules beyond mere linguistic intuition. Our analysis reveals that even established benchmarks may contain errors significantly impacting model evaluation. We recommend:

- • Multi-stage review processes involving expert linguists and grammarians
- • Diverse datasets spanning Classical and Modern Standard Arabic, with varied topics and writing styles

Maintaining high standards for benchmark creation will advance the field toward more accurate and reliable Arabic language processing tools.

#### 8.4 Limitations

Our study encountered several significant limitations that merit attention. The primary challenge lies in model hallucinations, particularly with non-Arabic words. Addressing these issues necessitates the use of advanced techniques, such as constrained decoding, to ensure greater accuracy and coherence in predictions. Moreover, resources limitations restricted scaling to larger parameters and performing extended training with high-quality data, which potentially limited the model’s performance. However, increasing the model’s size could improve capabilities but introduce computational and efficiency challenges. A notable constraint is the limited availability of openly accessible, highquality diacritized datasets, particularly for Modern Standard Arabic (MSA). This data scarcity poses a significant barrier to developing models that generalize well across different genres and real-world MSA applications. In summary, these limitations highlight both the computational and linguistic challenges in Arabic diacritization, emphasizing the need for better resources, architectural optimization, and targeted techniques to improve model robustness, fairness, and generalizability in future work.

### References

Hamza Abbad and Shengwu Xiong. Multi-components system for automatic arabic diacritization. In Joemon M. Jose, Emine Yilmaz, João Magalhães, Pablo Castells, Nicola Ferro, Mário J. Silva, and Flávio Martins (eds.), Advances in Information Retrieval, pp. 341–355, Cham, 2020. Springer International Publishing. ISBN 978-3-030-45439-5.

Ahmed Abdelali, Kareem Darwish, Nadir Durrani, and Hamdy Mubarak. Farasa: A fast and furious segmenter for Arabic. In John DeNero, Mark Finlayson, and Sravana Reddy (eds.), Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Demonstrations, pp. 11–16, San Diego, California, June 2016. Association for Computational Linguistics. doi: 10.18653/v1/N16-3003. URL https://aclanthology.org/N16-3003.

Abdulmohsen Al-Thubaity, Atheer Alkhalifa, Abdulrahman Almuhareb, and Waleed Alsanie. Arabic diacritization using bidirectional long short-term memory neural networks with conditional random fields. IEEE Access, 8:154984–154996, 2020.

Faris Alasmary, Orjuwan Zaafarani, and Ahmad Ghannam. Catt: Character-based arabic tashkeel transformer, 2024. URL https://arxiv.org/abs/2407.03236.

Badr AlKhamissi, Muhammad N ElNokrashy, and Mohamed Gabr. Deep diacritization: Efficient hierarchical recurrence for improved arabic diacritization. arXiv preprint arXiv:2011.00538, 2020.

Manar M. Almanea. Automatic methods and neural networks in arabic texts diacritization: A comprehensive survey. IEEE Access, 9:145012–145032, 2021. doi: 10.1109/ACCESS.2021.312297 7.

Rehab Alnefaie and Aqil M. Azmi. Automatic minimal diacritization of arabic texts. Procedia Computer Science, 117:169–174, 2017. ISSN 1877-0509. doi: https://doi.org/10.1016/j.procs.20 17.10.106. URL https://www.sciencedirect.com/science/article/pii/S1877050917321634. Arabic Computational Linguistics.

Viraat Aryabumi, John Dang, Dwarak Talupuru, Saurabh Dash, David Cairuz, Hangyu Lin, Bharat Venkitesh, Madeline Smith, Jon Ander Campos, Yi Chern Tan, et al. Aya 23: Open weight releases to further multilingual progress. arXiv preprint arXiv:2405.15032, 2024.

Parnia Bahar, Mattia Di Gangi, Nick Rossenbach, and Mohammad Zeineldeen. Take the hint: Improving arabic diacritization with partially-diacritized text, 2023. URL https://arxiv.org/ abs/2306.03557.

M Saiful Bari, Yazeed Alnumay, Norah A Alzahrani, Nouf M Alotaibi, Hisham A Alyahya, Sultan AlRashed, Faisal A Mirza, Shaykhah Z Alsubaie, Hassan A Alahmed, Ghadah Alabduljabbar, et al. Allam: Large language models for arabic and english. arXiv preprint arXiv:2407.15390, 2024.

Kareem Darwish, Hamdy Mubarak, and Ahmed Abdelali. Arabic diacritization: Stats, rules, and hacks. In Nizar Habash, Mona Diab, Kareem Darwish, Wassim El-Hajj, Hend Al-Khalifa, Houda Bouamor, Nadi Tomeh, Mahmoud El-Haj, and Wajdi Zaghouani (eds.), Proceedings of the Third Arabic Natural Language Processing Workshop, pp. 9–17, Valencia, Spain, April 2017a. Association for Computational Linguistics. doi: 10.18653/v1/W17-1302. URL https://aclanthology.org/W17-1302.

Kareem Darwish, Hamdy Mubarak, and Ahmed Abdelali. Arabic diacritization: Stats, rules, and hacks. In Nizar Habash, Mona Diab, Kareem Darwish, Wassim El-Hajj, Hend Al-Khalifa, Houda Bouamor, Nadi Tomeh, Mahmoud El-Haj, and Wajdi Zaghouani (eds.), Proceedings of the Third Arabic Natural Language Processing Workshop, pp. 9–17, Valencia, Spain, April 2017b. Association for Computational Linguistics. doi: 10.18653/v1/W17-1302. URL https://aclanthology.org/W17-1302.

Kareem Darwish, Ahmed Abdelali, Hamdy Mubarak, and Mohamed Eldesouki. Arabic diacritic recovery using a feature-rich bilstm model, 2020. URL https://arxiv.org/abs/2002.01207.

Mona Diab, Mahmoud Ghoneim, and Nizar Habash. Arabic diacritization in the context of statistical machine translation. In Proceedings of Machine Translation Summit XI: Papers, 2007.

Ali Fadel, Ibraheem Tuffaha, Bara’ Al-Jawarneh, and Mahmoud Al-Ayyoub. Arabic text diacritization using deep neural networks, 2019a. URL https://arxiv.org/abs/1905.01965.

Ali Fadel, Ibraheem Tuffaha, Bara’ Al-Jawarneh, and Mahmoud Al-Ayyoub. Neural arabic text diacritization: State of the art results and a novel approach for machine translation. In Proceedings of the 6th Workshop on Asian Translation, pp. 215–225. Association for Computational Linguistics, 2019b. doi: 10.18653/v1/d19-5229. URL http://dx.doi.org/10.18653/v1/D19-5229.

Nizar Habash, Anas Shahrour, and Muhamed Al Khalil. Exploiting arabic diacritization for high quality automatic annotation. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16), pp. 4298–4304, 2016.

Khalil Hennara, Sara Chrouf, Mohamed Motaism Hamed, Zeina Aldallal, Omar Hadid, and Safwan AlModhayan. Kuwain 1.5 b: An arabic slm via language injection. arXiv preprint arXiv:2504.15120, 2025.

Ruba Kharsa, Ashraf Elnagar, and Sane Yagi. Bert-based arabic diacritization: A state-of-theart approach for improving text accuracy and pronunciation. Expert Systems with Applications, 248:123416, 2024. ISSN 0957-4174. doi: https://doi.org/10.1016/j.eswa.2024.123416. URL https://www.sciencedirect.com/science/article/pii/S0957417424002811.

Vladimir Likic. The needleman-wunsch algorithm for sequence alignment. Lecture given at the 7th Melbourne Bioinformatics Course, Bi021 Molecular Science and Biotechnology Institute, University of Melbourne, pp. 1–46, 2008.

Mohamed Maamouri, Seth Kulick, and Ann Bies. Diacritic annotation in the Arabic treebank and its impact on parser evaluation. In Nicoletta Calzolari, Khalid Choukri, Bente Maegaard, Joseph Mariani, Jan Odijk, Stelios Piperidis, and Daniel Tapias (eds.), Proceedings of the Sixth International Conference on Language Resources and Evaluation (LREC’08), Marrakech, Morocco, May 2008. European Language Resources Association (ELRA). URL http://www.lrec-conf.org/proceedings/lrec2008/pdf/706_paper.pdf.

Mokthar Ali Hasan Madhfar and Ali Mustafa Qamar. Effective deep learning models for automatic

diacritization of arabic text. IEEE Access, 9:273–288, 2021. doi: 10.1109/ACCESS.2020.3041676. Navid-AI. Yehia 7b preview. https://huggingface.co/Navid-AI/Yehia-7B-preview, 2025. Rani Nelken and Stuart M. Shieber. Arabic diacritization using weighted finite-state transducers.

In Kareem Darwish, Mona Diab, and Nizar Habash (eds.), Proceedings of the ACL Workshop on

Computational Approaches to Semitic Languages, pp. 79–86, Ann Arbor, Michigan, June 2005. Association for Computational Linguistics. URL https://aclanthology.org/W05-0711.

Arfath Pasha, Mohamed Al-Badrashiny, Mona Diab, Ahmed El Kholy, Ramy Eskander, Nizar Habash, Manoj Pooleery, Owen Rambow, and Ryan Roth. MADAMIRA: A fast, comprehensive tool for morphological analysis and disambiguation of Arabic. In Nicoletta Calzolari, Khalid Choukri, Thierry Declerck, Hrafn Loftsson, Bente Maegaard, Joseph Mariani, Asuncion Moreno, Jan Odijk, and Stelios Piperidis (eds.), Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14), pp. 1094–1101, Reykjavik, Iceland, May 2014. European Language Resources Association (ELRA). URL http://www.lrec-conf.org/procee dings/lrec2014/pdf/593_Paper.pdf.

Yasser Sabtan. Arabic part-of-speech tagging using a combined rule-based and data-driven approach. Digital Scholarship in the Humanities, 36(3):719–735, 2021.

Neha Sengupta, Sunil Kumar Sahu, Bokang Jia, Satheesh Katipomu, Haonan Li, Fajri Koto, William Marshall, Gurpreet Gosal, Cynthia Liu, Zhiming Chen, Osama Mohammed Afzal, Samta Kamboj, Onkar Pandit, Rahul Pal, Lalit Pradhan, Zain Muhammad Mujahid, Massa Baali, Xudong Han, Sondos Mahmoud Bsharat, Alham Fikri Aji, Zhiqiang Shen, Zhengzhong Liu, Natalia Vassilieva, Joel Hestness, Andy Hock, Andrew Feldman, Jonathan Lee, Andrew Jackson, Hector Xuguang Ren, Preslav Nakov, Timothy Baldwin, and Eric Xing. Jais and jais-chat: Arabiccentric foundation and instruction-tuned open generative large language models, 2023. URL https://arxiv.org/abs/2308.16149.

Abderrahman Skiredj and Ismail Berrada. Arabic text diacritization in the age of transfer learning: Token classification is all you need, 2024. URL https://arxiv.org/abs/2401.04848.

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.

Silma Team. Silma. 2024. URL https://www.silma.ai. Jezia Zakraoui, Moutaz Saleh, Somaya Al-Maadeed, and Jihad Mohamed Alja’am. Arabic machine

translation: A survey with challenges and future directions. IEEE Access, 9:161445–161468, 2021. Taha Zerrouki and Amar Balla. Tashkeela: Novel corpus of arabic vocalized texts, data for auto-

diacritization systems. Data in Brief, 11, 02 2017. doi: 10.1016/j.dib.2017.01.011.

### Appendix

### A Training Details

The fine-tuning process for Sadeed employed the standard next-token prediction methodology, with the system prompt and embedding tokens masked. Given the compact nature of the Kuwain model, we designed a concise yet effective fine-tuning regimen. Table 9 presents the detailed hyperparameters used in our training process.

Parameter Value Training Epochs 3 Learning Rate Schedule Cosine decay Learning Rate 5e-6 Batch Size 1024 Weight Decay 0.01 Warm-up Steps 30 Optimizer AdamW Max Sequence Length 512

Table 9: Training Hyperparameters for Sadeed Model

The model was trained on 8 A100 GPUs. During fine-tuning, we monitored the validation loss to prevent overfitting and implemented early stopping with a patience of 3 evaluation steps. The best checkpoint was selected based on the lowest validation loss achieved during training.

### B Model Hallucinations Examples

In this section, we highlight examples of model hallucination observed during the initial diacritization process. Table 10 presents side-by-side comparisons between the raw model outputs and the corrected versions obtained through the Needleman-Wunsch alignment algorithm.

# Text Model’s Output Final Output

- 1 نورمثتسملﺍ بسك ةشيشلﺍ يهاقم

نوك هتسملﺍ ﺽرمو

َنيِرِمَْثتْسُمْلﺍ ُبْسَك ِةَشيِّشلﺍ يِهاَقَم َنيُكِ َْهتْسُمْلﺍ ُﺽَرَمَو

نورمثتسملﺍ ُبْسَك ِةَشيِّشلﺍ يِهاَقَم نوك هتسملﺍ ُﺽَرَمَو

- 2 ةنماكلﺍ ةيئانغلﺍ ةبهوملﺍ ريثك لﺍ ظحال ﺏﺭ لواحف ،نوسكاج ةلئاع يف نﺃ ،نوسكاج فيﺯوج ،ﺓرسألﺍ ةيئانغ ةكرش عم دقعب هﺩالوﺃ لجسي

َةَ نِماَكْلﺍَةّيِ ئَ َانِغْلﺍَةَ بِهْوَمْلﺍُرِيثَكْ لﺍ َظَحاَل ّﺏَﺭُ َلَوَاحَف ،نوُفْسِكاَج ِ َةلِ ئَاع يِف ْنﺃ  ،نوُفْسِكاَج فيِﺯوُج ،ِﺓَرْسأُ ْلﺍ ٍةّيِ ئَ َانِغ ٍةَكِرَش َعَم ٍدْقَعِ ب ُهَﺩاَلْوﺃ  َلِّجَ ُسي

َةَ نِماَكْلﺍَةّيِ ئَ َانِغْلﺍَةَ بِهْوَمْلﺍُرِيثَكْ لﺍ َظَحاَل ّﺏَﺭُ َلَوَاحَف ،نوسكاج ِ َةلِ ئَاع يِف ْنﺃ  ،نوسكاج فيِﺯوُج ،ِﺓَرْسأُ ْلﺍ ٍةّيِ ئَ َانِغ ٍةَكِرَش َعَم ٍدْقَعِ ب ُهَﺩاَلْوﺃ  َلِّجَ ُسي

- 3 عقوم يف ﺭوشنم ريرقت حضوﺃو نﺃ صصختملﺍ ينقتلﺍ تيجﺩاغنﺇ

ع ِقْوَمِ يِف ٌﺭوُشْنَم ٌريِرَْقت َح َضْوﺃ َو ّنَ ﺃ  ِرِّصَ تْخُمْلﺍ ّيِ ِنْق ّتلﺍَ ِتيِجْﺩاَغْنﺇ ٍﺓَديِدَج ٍةّ َِينَْقت ِﺭَاِبتْخاِب ﺃَُدَْبت كرِبْشَ يف

ع ِقْوَمِ يِف ٌﺭوُشْنَم ٌريِرَْقت َح َضْوﺃ َو ّنَ ﺃ  صصختملﺍ ّيِ ِنْق ّتلﺍَ ِتيِجْﺩاَغْنﺇ ٍﺓَديِدَج ٍةّ َِينَْقت ِﺭَاِبتْخاِب ﺃَُدَْبت كوبسيف

ﺓديدج ةينقت ﺭابتخاب ﺃدبت كوبسيف ىلع ﺓﺩوجوملﺍ كلتب ةهيبش صصقلل مﺍرجتسنﺇ

َىَلع ِﺓَﺩوُجْوَمْلﺍ َكِْلِتب ٍةَهيِبَش ِصَصِقِْلل مﺍَزج ِنْﺇ

َىَلع ِﺓَﺩوُجْوَمْلﺍ َكِْلِتب ٍةَهيِبَش ِصَصِقِْلل مﺍرجتسنﺇ

###### Table 10: Examples of hallucination correction in Arabic text. In Sentence 1, the grammatical case ending is corrected to align with the change in word type caused by the hallucinated word. In Sentences 2 and 3, the hallucinated words were foreign terms.

### C Examples from Sadeed

This section provides multiple examples of Sadeed model output. Table 11 shows the results without errors.

None Diacritized text Diacritzed Text

َ َ َ ُ ّ َ ِ َ َ َ َ ِ ِ َ َ ِ َ ّ َ َ ّ َ َ َ ِ َ َ َ ِ ِ َ

|لصي ىتح هب نوهقفتي ال مهنﺃ ملعﺃ هللﺍو ينعي ،مهيقﺍرت ﺯواجي ال نﺁرقلﺍ نﺅرقي مهنأب مهف هيف لصحي مل بلقلﺍ ىلﺇ لصي مل ﺍﺫإف ،بلقلﺍ ىلﺇ عجﺍﺭ مهفلﺍ نأل مهبولق ىلﺇ يذلﺍ وهو ،طقف ةعومسملﺍ فورحلﺍو ﺕﺍوصألﺍ لحم دنع فقي امنإو ،لاح ىلع :مالسلﺍو ﺓالصلﺍ هيلع هلوق نم اضيﺃ مدقت امو ،مهفي ال نمو مهفي نم هيف كرتشي .هرخﺁ ىلﺇ اعﺍزتنﺍ ملعلﺍ ضبقي ال هللﺍ نﺇ<br><br>|ىّتَح ِهِ ب َنوُهّقَفََتي اَل ْمُ هّنﺃ  ُمَلْعﺃ  هللﺍَو ينَْعي ،ْمُ َهيِقﺍَ َرت ُﺯِوَاجُي اَل َنﺁْرُقْلﺍ َنُﺅَ رَْقي ْمُ هّنأ ِب ْلُصْحَي ْمل ِبْلَقْلﺍ َىلﺇ ْلِصَ ي ْمل ﺍَﺫإَ ف ، ِبْلَقْلﺍ َىلﺇ ٌع ِجﺍَﺭ َمْهَفْلﺍ ّنأل ْمِهِب ُوُلق َىلﺇ َلِصَ ي<br><br>َوُهَو ، ْطَ َقف ِةَعوُمْسَمْلﺍ ِفوُ ُرحْلﺍَو ِﺕَﺍوْصأْلﺍ ّلَ حَم َدْنِع ُفِ َقي َامّنإَو ،ٍلاَح َىَلع ٌمْهَ ف ِهيِف ُﺓاَ لّصلﺍ ِهَْيَلع ِهلْ َوق ْنِم اًضْيﺃ  َمّدَ َقت اَمَو ،ُمَهَْفي اَل ْنَمَو ُمَهَْفي ْنَم ِهيِف ُكَِرتْ َشي يِذلﺍ .ِهِرِخﺁ َىلﺇ ًاعﺍَزِتْنﺍ َمْلِعْلﺍ ُضِبَْقي اَل هللﺍ ّنﺇ :ُماَ لّسلﺍَو|
|---|---|
|نع هركﺫ مﺍوعلﺍ نب ﺩابع مهنم نإف ديﺯ نب ليمج نع كلﺫ يف امهفلاخ نم امﺃو ميهﺍربإو ﺩوﺍﺩ نب يلع نب دمحم انثدح امك يﺭاصنألﺍ ديﺯ نب بعك تعمس لاق ليمج ليمج انثدح مﺍوعلﺍ نب ﺩابع نع يطسﺍولﺍ ناميلس نب ديعس انثدح الاق ﺩوﺍﺩ يبﺃ نب )563 / 1( يبنلﺍ نﺃ ﺙدحي يﺭاصنألﺍ ديﺯ نب بعك تعمس لاق يئاطلﺍ ديﺯ نب<br><br>|ْنَع ُهَ رَ َكﺫ مﺍّوَعْلﺍ َنْ ب َﺩاّبَع ْمُ هْنِم ّنإَ ف ٍدْيَﺯ ِنْ ب ليِمَج ْنَع َكِلَﺫ يف اَمُهَ َفلاَخ ْنَم اّمﺃ َو ُ ميِهﺍَرْبإَو ﺩُوﺍَﺩ ِنْ ب ّيَِلع ُنْ ب ُدّ َمُحم َاَنثّدَح َامَك ّيِﺭاَصْنأْلﺍ ٍدْيَﺯ َنْ ب َبْعَك َتْعِمَس َلَاق ليِمَج ُليِمَج َاَنثّدَح مﺍّوَعْلﺍ ِنْ ب ِﺩاّبَع ْنَع ّيِطِسَﺍوْلﺍ َناَمَْيلُس ُنْ ب ُديِعَس َاَنثّدَح اَلَاق َﺩُوﺍَﺩ يبﺃ  ُنْ ب )563 / 1( ّيبّنلﺍ ّنﺃ  ُﺙِّدَ حُي ّيِﺭاَصْنأْلﺍ ٍدْيَﺯ َنْ ب َبْعَك ُتْعِمَس َلَاق ّيئاّطلﺍ ٍدْيَﺯ ُنْ ب|
|تبلغ يبتك نﺃ ركفﺃ نﺃ نوﺩ ناكم لكب ﺓرشتنملﺍ هضﺍرغﺃ نم قيضلاب رعشﺃ ﺕﺃدب هﺍﺭﺃ يذلﺍ يملاع نيب زيمت ال يتلﺍ ةلفطلﺍ ﺽﺍرغﺃ اضيﺃ يللمت لمش امبﺭو .هاضوف دعبو ،ﺓﺩالولﺍ دعب ام ﺏائتكاب يتباصﺇ دنع ﺍﺀوس رمألﺍ ﺩﺍﺯو ،اهدلﺍو ملاعو امظنم ،ةطرفم نﺯولﺍ يف ﺓﺩايﺯ نم تيناع نﺃ<br><br>|ْتَ َبَلغ يُبتُك ّنﺃ َ رّ َكفﺃ ْنﺃ  َنوُﺩ ٍناَكَم ّلُكِ ب ِﺓَرِشَ تْ ُنمْلﺍ ِهِضﺍرْغﺃ  ْنِم ِقيِّضلاِبُ رُعْشﺃ  ُﺕﺃَدَ ب ُهﺍَﺭﺃ  يِذلﺍ يِ َملَاع َنْ َيب ُزَِّيُمت اَل يتّلﺍ ِ َةلْفِّطلﺍ َﺽﺍرْغﺃ  اًضْيﺃ  يِلَّلَمت َلِمَش َامّبُﺭَو .ُهاَضَ َوف َدَْعبَو ،ِﺓَﺩاَلِوْلﺍ َدَْعب اَم ِﺏاَ ئِتْكاِب يَتباَصﺇ َدْنِع ﺍﺀوُس رْمأْلﺍ َﺩﺍَﺯَو ،َاهِدِ لﺍَو ِملَاعَو اًمّظَ نُم ،ٍةَطِرْفُم ِنْﺯَوْلﺍ يف ٍﺓَﺩَايِﺯ ْنِم ُتَْينَاع ْنﺃ |
|ﺓرشقتملﺍ ﺏوبحلﺍ نم رخﺁ ﻉون يﺃ وﺃ لماكلﺍ حمقلﺍ قئاقﺭ وﺃ ﺓﺭذلﺍ قئاقﺭ نم ﺏوك ةلﺍﺯإب يموق ،ﺝاجدلﺍ حئﺍرش نم لطﺭ 1 ةمسقم ،رشوك لﺍ حلم نم ﺓريغص ةقعلم 2 تيﺯ ﺫﺍﺫﺭ ةصاصملﺍ ﺩﺍوعﺃ نم ﺓوبع 1 بئﺍرلﺍ نبللﺍ نم ﺏوك 1 ضيبألﺍ رتولﺍ -2 .ةجﺭﺩ 573 ﺓﺭﺍرح ةجﺭﺩ ىلع نرفلﺍ ينخس -1 ريضحتلﺍ ةقيرط يهطلﺍ|ِﺏُوُبحْلﺍ َنِم َرَخﺁ ﻉَْون ّيﺃ  ْوﺃ  لِماَكْلﺍ حْمَقْلﺍ ِقِ ئَاقَﺭ ْوﺃ  ِﺓَﺭّذلﺍ ِقِ ئَاقَﺭ ْنِم ٌﺏوُك ،ﺝاَجّدلﺍ حئﺍَرش ْنِم ٌلْطِﺭ 1 ٌةَمّسَقُم ،رِ شوُكْ لﺍ حْلِم ْنِم ٌﺓَريِغَص ٌةَقَعْلِم 2 ِﺓَرِّشَقَ ُتمْلﺍ ِةَصاَصِ ْملﺍ ِﺩَﺍوْعﺃ  ْنِم ﺓُوبع 1 ِبِ ئﺍّرلﺍ ِنَ بّللﺍ َنِم ٌﺏوُك 1 ِضَيْبأْلﺍ َرتَوْلﺍ ةلﺍَﺯإب يِمُوق .ًةَجَﺭَﺩ 573 ٍﺓَﺭﺍَرَح ِةَجَﺭَﺩ َىَلع َنْرُفْلﺍ ينخس -1 ِريِضْحّتلﺍُةَقيِرَط يِهطْلﺍ ِتْيَﺯ ُﺫﺍَﺫَﺭ<br><br>-2|
|"ﺕﺍﺭاسم ﺓﺭﺩابم" يف ةيﺭﺍﺩإلﺍ نوؤشلﺍ لوؤسم ،دمحﺃ ﺝاح رمع حضوي :ﺕﺍﺭﺩابم ىلع تفوسوركيام ةكرش عم نواعتلاب ﺕﺍونس 5 ذنم نولمعي مهنﺃ ،دعب نع ميلعتلل ةيفرعم ةمدخ ميدقت ربع لمعلﺍ قوسب مهطبﺭو يلحملﺍ عمتجملﺍ ﺕائف نيكمت يف ماهسإلﺍ<br><br>.ةلماك<br><br>|" ٍﺕﺍَﺭاَسَم ِﺓَﺭَﺩَابُم" يف ِةّيِﺭﺍَﺩإْلﺍ ِنوُؤّشلﺍ ُلوُؤْسَم ،دَ مْحﺃ  ّﺝاَح َرُمع ُح ّضُوي : ٌﺕﺍَﺭَﺩَابُم َىَلع َتفوُسوُ ركياَم ِةَكِرَش َعَم ِنُواَعّتلاِب ٍﺕَﺍَونَس 5 ُذْنُم َنُولَمَْعي مُ هّنﺃ  ،ٍدُْعب ْنَع ميِلْعّتِلل<br><br>ٍةَمْدِخ ِميِدَْقت َرَْبع لَمَعْلﺍ ِقوُسِب مِهِطْبَﺭَو ّيِّ َلحَمْلﺍ عَمَ تْجُمْلﺍ ِﺕاَ ئِف ِنيِكَْمت يف ماَهْسإْلﺍ<br><br>.ٍ َةلِماَكٍةّيِفِرْعَم|
|يف هديس هل نﺫأف هملكف ،انالف ملكي ال نﺃ هللاب فلح ﺍدبع نﺃ ول تيﺃﺭﺃ :تلق مﺃ وسكي مﺃ معطيﺃ ؟كلام ىلﺇ بحﺃ كلﺫ يﺃ ،موصلﺍ وﺃ ﺓوسك لﺍ وﺃ ماعطلﺍ دي يف ناك ﺍﺫﺇ ماعطإلﺍو ﺓوسك لﺍ ىلع ﺭدقي وهو موصي نﺃ هل ﺯوجي لهو ،موصي مايصلﺍ كلام يل لاق :لاق ؟هسفن نع وسكي وﺃ معطي نﺃ هديس هل نﺫأف لام دبعلﺍ يبلق يف لوقي ناكو هنع ﺃزجﺃ معطأف ،هديس هل نﺫﺃ نإو ماعطإلﺍ نم يدنع نيبﺃ .ﺀيش هنم<br><br>|ُهُدِّيَس ُ َهل َنِﺫ َ ف ُهَمّلَكَ ف ،اًناَ ُلف َمِّلَكُ ي اَل ْنﺃ  هللاِب َفَ لَح ﺍًدْبَع ّنﺃ  َْول َتْيﺃ َﺭﺃ  : ُتُْلق ْمﺃ  وُسْكَ ي ْمﺃ  ُمِعْطُ يﺃ  ؟كِلاَم َىلﺇ ّبَحﺃ  َكِلَﺫ ّيﺃ  ،مْوّصلﺍ ِوﺃ  ِﺓَوْسِكْ لﺍ ِوﺃ  ماَعّطلﺍ يف ِدَ ي يف َناَك ﺍَﺫﺇ ماَعْطإْلﺍَو ِﺓَوْسِكْ لﺍ َىَلع ُﺭِدَْقي َوُهَو َموُصَ ي ْنﺃ  َهل ُﺯُوجَي ْلَهَو ،ُموُصَ ي ُمَايِّصلﺍ ٌكِلاَم يِل َلَاق :َلَاق ؟ِهِسَْفن ْنَعَوُسْكَ ي ْوﺃ  َمِعْطُ ي ْنﺃ  ُهُدِّيَسُ َهل َنِﺫ َ ف ٌلاَم ِدْبَعْلﺍ يبَْلق يف ُلوُ َقي َناَكَوُهْنَع ﺃ َزْجﺃ  َمَعْط َ ف ،ُهُدِّيَسُ َهل َنِﺫﺃ  ْنإَو ماَعْطإْلﺍ َنِم يِدْنِع َنْيبﺃ  .ﺀْيَشُهْنِم|

ِ َ َ َ ٍ َ ِ َ َ ُ َ ِ

َ ِ َ َ َ ُ ِ َ ِ َ ُ ِ َ

ُ َ َ

ّ َ ِ َ

ِ ً ُ َ َ ِ

ٍ

ِ ِ ِ ُ

ِ َ َ ِ َ َ ِ ِ َ

ِ َ ُ ِ َ َ َ ِ َ ِ ِ ِ

ِ ِ َ

َ ِ ّ َ َ

ٍ ُ ُ ِ َ ِ َ ِ ِ ِ ُ

ِ ِ ِ ُ ٌ

###### Table 11: Output samples from Sadeed

### D CATT Benchmark Examples

Table 12 illustrates some of errors found in the CATT benchmark, showing examples of incorrect diacritizations, their corrections, and the corresponding error classes.

CATT GT Correction or Alternative Error class

ِ ّ َ ِ ّ َ ّ ُ ّ َ

ِ ّ َ ِ ّ َ ّ ُ ّ َ

|ْنﺃ  هللﺍ اً لِ  اَس عِيَمجْلﺍ َنِم ِﺕاَكيِرْبتلﺍَو ينَاهتلﺍ يِمِﺯَاحْلﺍ ىقَ َلَتيَو َﺓَﺩاَعسلﺍَو َةَكَ َرْبلﺍ ُمِهيِف َﺡَ رْطَ ي|ْنﺃ  هللﺍ اً لِ  اَس عِيَمجْلﺍ َنِم ِﺕاَكيِرْبتلﺍَو َينَاهتلﺍ يِمِﺯَاحْلﺍ ىقَ َلَتيَو َﺓَﺩاَعسلﺍَو َةَكَ َرْبلﺍ ُمِهيِف َﺡَ رْطَ ي|Partially diacritized|
|---|---|---|
|َنِم يﺃ  لِمَاحِل ُنِكُْمي ِة ِينوُ ُرتْكِ ل ْلﺍ ِﺓَريِش تلﺍ َىَلع ِلوُصُ حْلاِبَو ْ َيبُﺩَو ِةيِﺩوُعسلﺍ ِﺕﺍَﺭاَطَم َرَْبع ُرِ شَاُبمْلﺍ ُلوُخ دلﺍ ِنْ َيَتريِش تلﺍ َاِيكُْرِتب ْلُوبْنَطْسَو<br><br>|َنِم يﺃ  لِمَاحِل ُنِكُْمي ِة ِينوُ ُرتْكِ ل ْلﺍ ِﺓَريِش تلﺍ َىَلع ِلوُصُ حْلاِبَو َيبُﺩَو ِةيِﺩوُعسلﺍ ِﺕﺍَﺭاَطَم َرَْبع ُرِ شَاُبمْلﺍ ُلوُخ دلﺍ ِنْ َيَتريِش تلﺍ<br><br>َاِيكُْرِتب َلُوبْنَطْسَو|Partially diacritized|
|َﺏَرَض ﺏ ْلﺍ نﺇ اً لِ  َاق ِ َةلِ  اَعْلﺍ َنِم ٌﺏرَقُم ُرَخﺁ ٌصْخَش َﺙدَ حَتَو اَهِعُولُض ْنِمًةتِسَو َاهَنَانْسﺃ  َرَسَكَو َاَهتَمُجْمُج َمشَهَو َﺓَريِغصلﺍُهَ َتْنبﺍ|َﺏَرَض َﺏ ْلﺍ نﺇ اً لِ  َاق ِ َةلِ  اَعْلﺍ َنِم ٌﺏرَقُم ُرَخﺁ ٌصْخَش َﺙدَ حَتَو اَهِعُولُض ْنِمًةتِسَو َاهَنَانْسﺃ  َرَسَكَو َاَهتَمُجْمُج َمشَهَو َﺓَريِغصلﺍُهَ َتْنبﺍ<br><br>|Erroneous Diacritization|
|َةيِفﺍَرِتْحالﺍ َةَ  ِيْبلﺍ ُمُ َهل ْﺕَ رفَوَو ِمْلسلﺍِّ َتْقَو ُﺓَﺭﺍَﺯِوْلﺍ ُمُ هْتَ رَ َكﺫ ﺍَﺫﺇ َةَمِيلسلﺍَةِينِْهْملﺍ<br><br>|َةيِفﺍَرِتْحالﺍ َةَ  ِيْبلﺍ ُمُ َهل ْﺕَ رفَوَو ِمْلسلﺍِّ َتْقَو ُﺓَﺭﺍَﺯِوْلﺍ ُمُ هْتَ رَ َكﺫ ﺍَﺫﺇ َةَمِيلسلﺍَةِينَ ِهْملﺍ|Erroneous Diacritization|
|يِرْيَصُ خْلﺍ ُدْهَ فُﺭُوتْكدلﺍ ِﺕَانِطْرَسُمْلﺍ ِﺙَاحْبﺃ ُ ملَاعَوُﺫَاتْسﺃ َفَشَك<br><br>ِﺕﺍَﺭِّدَخُمْلﺍَو ُنوُجَاتْبِكْ لﺍ َاُهبِّبَ ُسي يت لﺍ ِﺭﺍَرْض ْلﺍ َنِم ٍ َةلْمُج ْنَع َنِينِمْدُمْلﺍَو َنيِطاَ َعُتمِْلل<br><br>|يِرْيَصُ خْلﺍ ُدْهَ فُﺭُوتْكدلﺍ ِﺕَانِطْرَسُمْلﺍ ِﺙَاحْبﺃ ُ ملَاعَوُﺫَاتْسﺃ َفَشَك ُﺕﺍَﺭِّدَخُمْلﺍَو ُنوُجَاتْبِكْ لﺍ َاُهبِّبَ ُسي يت لﺍ ِﺭﺍَرْض ْلﺍ َنِم ٍ َةلْمُج ْنَع َنِينِمْدُمْلﺍَو َنيِطاَ َعُتمِْلل<br><br>|Erroneous Diacritization|
|ْنِبوُﺭ يِلاَ ُغتُْرْبلﺍ ِبَخَ تْ ُنمْلﺍ ِبِعاَل َعَم َعيِقْوتلﺍ ِلاَ ِلهْلﺍ يِﺩَان مَتﺃ  ِﺙاَ َلث َﺓدُِمل يِزِيلِجْن لﺍ ْنُوِتبْمَاهْرَفْلُو ِقيَِرف ْنِم اًمِﺩَاق ْزيِفِين ٍﺕَﺍَونَس|ْنِبوُﺭ يِلاَ ُغتُْرْبلﺍ ِبَخَ تْ ُنمْلﺍ ِبِعاَل َعَم َعيِقْوتلﺍ ِلاَ ِلهْلﺍ يِﺩَان مَتﺃ  ِﺙاَ َلث ِﺓدُِمل يِزِيلِجْن لﺍ ْنُوِتبْمَاهْرَفْلُو ِقيَِرف ْنِم اًمِﺩَاق ْزيِفِين ٍﺕَﺍَونَس<br><br>|Erroneous Diacritization|
|ٍﺓَدْجَن ِةيِﺭْوَﺩ ْنِم ُﺡاَ لسلﺍِّ َقِرُس<br><br>|ٍﺓَدْجَن ِةيِﺭْوَﺩ ْنِم ﺡاَ لسلﺍِّ َقَرَس|Diacritization ambiguity|
|ىَلوََتتَوٍةَعِّ َوَنتُمٍةيِﺩاَصِتْقﺍ ٍﺕَاعاَطِق ِقْلَخ يف ﺍًﺭْوَﺩ َايﺍَدْس ُبَعَْلت َامَك ِﺕَانَاَيْبلﺍ ِلَاجَم يف ِةيِﺩوُعسلﺍ ِﺭِﺩَﺍوَكْ لﺍ ِريِوْطَ ت مَاَهم|ىَلوََتتَوٍةَعِّ َوَنتُمٍةيِﺩاَصِتْقﺍ ٍﺕَاعاَطِق ِقْلَخ يف ﺍًﺭْوَﺩ َايﺍَدَس ُبَعَْلت َامَك ِﺕَانَاَيْبلﺍ ِلَاجَم يف ِةيِﺩوُعسلﺍ ِﺭِﺩَﺍوَكْ لﺍ ِريِوْطَ ت مَاَهم<br><br>|Erroneous Diacritization|
|ِنْ يَخوُﺭاصلﺍٌةَصﺍوَغ ْتَ َقلْطﺃ  ِةِيلاَمشلﺍ ِةيِﺭوُكْ لﺍ َِةلاَكَوْلﺍ ِبَسَ ح ِبَو ِﺩاَ ِلْبلﺍ ِقْرَش يف ِةَعِقَﺍوْلﺍ ُوبْنيِس ِةَنيِدَم لِحَﺍوَس َِةلَابِق ْنِم|ِنْ يَخوُﺭاصلﺍٌةَصﺍوَغ ْتَ َقلْطﺃ  ِةِيلاَمشلﺍ ِةيِﺭوُكْ لﺍ َِةلاَكَوْلﺍ ِبَسَ ح ِبَو ِﺩاَ ِلْبلﺍ ِقْرَش يف ِةَعِقَﺍوْلﺍ ُوبْنيِس ِةَنيِدَم لِحَﺍوَس َِةلَاُبق ْنِم|Erroneous Diacritization|
|يف ِةَسِﺩاسلﺍ ِﺓرَمِْلل ِم لاَعْلﺍ ِﺱ َك ي ف ُكِﺭاَ ُشي َرَضْخ ْلﺍ نﺃ  ُ رَكْذُ ي َكيِسْكِمْلﺍَو ﺍَدَْنلُوبَو َنِيتْنَجْﺭ ْلﺍ مُضَ ت ٍةيَِوق ٍةَعُومَْجم يف ِهِخيِﺭَات<br><br>|يف ِةَسِﺩاسلﺍ ِﺓرَمِْلل ِم لاَعْلﺍ ِﺱ َك ي ف ُكِﺭاَ ُشي َرَضْخ ْلﺍ نﺃ  ُ رُكْذَ ي َكيِسْكِمْلﺍَو ﺍَدَْنلُوبَو َنِيتْنَجْﺭ ْلﺍ مُضَ ت ٍةيَِوق ٍةَعُومَْجم يف ِهِخيِﺭَات|Diacritization ambiguity|
|لْكَشِب ِةِّيناَطَرّسلﺍ َاياَ َلخْلﺍ لْتَ ِقل اًمّمَصُم اًسوُرَْيف ُﺝاَ لِعْلﺍ ُدِمَ تَْعيَو ِناَطَرّسِلل ِةّيِعَاَنمْلﺍ ِمْسِجْلﺍ َةباَجِتْسﺍ ميْخَضت َعَم ّي اَ ِقتْنﺍ|لْكَشِب ِةِّيناَطَرّسلﺍ َاياَ َلخْلﺍ لْتَ ِقل اًمّمَصُم اًسوُرَْيف ُﺝاَ لِعْلﺍ ُدِمَ تَْعيَو ِناَطَرّسِلل ِةّيِعَاَنمْلﺍ ِمْسِجْلﺍ َةباَجِتْسﺍ مْيِخْضَ ت َعَم ّي اَ ِقتْنﺍ<br><br>|Misplaced diacritics|
|ُةّيِزَمْلﺍَو َرب ُوتْكﺃ يف ِةّيِزَمْلﺍ ِهِذَه َﺭَاِبتْخﺍ ًاَنَلع ْﺏاَسْ تﺍَو ْﺕﺃ َدَ ب ِ َةِلبْقُمْلﺍ عِيباَس ْلﺍ يف َنيِمِدْخَ تْسُمْلﺍ ِةّفاَكِ ل ًةَحَاتُم ُنوُكَ تَس|ُةّيِزَمْلﺍَو َرب ُوتْكﺃ يف ِةّيِزَمْلﺍ ِهِذَه َﺭَاِبتْخﺍ ًاَنَلع ْﺏاَسْ تﺍَو ْﺕﺃ َدَ ب ِ َةِلبْقُمْلﺍ عِيباَس ْلﺍ يف َنيِمِدْخَ تْسُمْلﺍ ِةّفاَكِ ل ًةَحَاتُم ُنوُكَ تَس|Partially diacritized|
|َعيِمَج ِةَ ُعُمجْلﺍ َمَْوي ٍﺩﺍَدَْغب يف ُة ّيِكيِرْم ْلﺍ ُﺓَﺭاَفِّسلﺍ ِﺕَدَشَانَو ِﺕﺍَرّتَوّتلﺍ يِمَاَنت ِبَ بَسِب ِﺭْوَفْلﺍ َىَلع ِقَﺍرِعْلﺍ َﺓَﺭَﺩاَغُم َنِينِطَﺍوُمْلﺍ<br><br>|َعيِمَج ِةَ ُعُمجْلﺍ َمَْوي َﺩﺍَدَْغب يف ُة ّيِكيِرْم ْلﺍ ُﺓَﺭاَفِّسلﺍ ِﺕَدَشَانَو ِﺕﺍَرّتَوّتلﺍ يِمَاَنت ِبَ بَسِب ِﺭْوَفْلﺍ َىَلع ِقَﺍرِعْلﺍ َﺓَﺭَﺩاَغُم َنِينِطَﺍوُمْلﺍ|Erroneous Diacritization|
|يف ِنﺍَ َريطلﺍ ِرِكﺍَذَ ت َﺭاَعْسﺃ  ِناَفِعاَضُ ي ِرْشَعْلﺍ ُﺓَ رْ ُمع َو ِرْطِفْلﺍ ُﺓَﺯاَجﺇ ا يِجِيلَخ ُصَخْﺭ ْلﺍ ُتْيَوُكْ لﺍَو ِةَكَ  لْمَمْلﺍ<br><br>|يف ِنﺍَ َريطلﺍ ِرِكﺍَذَ ت َﺭاَعْسﺃ  ِناَفِعاَضُ ي ِرْشَعْلﺍ ُﺓَ رْ ُمع َو ِرْطِفْلﺍ ُﺓَﺯاَجﺇ ا يِجِيلَخ ُصَخْﺭ ْلﺍ ُتْيَوُكْ لﺍَو .ِةَكَ  لْمَمْلﺍ<br><br>|Punctuation-induced diacritization ambiguity|
| |يف ِنﺍَ َريطلﺍ ِرِكﺍَذَ ت َﺭاَعْسﺃ  ِناَفِعاَضُ ي ِرْشَعْلﺍ ُﺓَ رْ ُمع َو ِرْطِفْلﺍ ُﺓَﺯاَجﺇ ا يِجِيلَخ ِصَخْﺭ ْلﺍ ، ِتْيَوُكْ لﺍَو ِةَكَ  لْمَمْلﺍ|Punctuation-induced diacritization ambiguity|
|َمْ َوْيلﺍ ُهَرَ َشن ِزَ ْكرَمْلﺍ ِﺏاَسِح َىَلع ْكيِفﺍَرْجوُفْنﺇ يف ُ زَ ْكرَمْلﺍ َحَصَ نَو ِريِوْط ّتلﺍَو ِفﺍَدْه ْلﺍ ِقيِقْحَت ْنَع ِلُﺯَانّتلﺍ ِوﺃ  ِﺱ ْيلﺍ مَدَع ِﺓَﺭوُرَضِ ب<br><br>ِﺕﺍّذِلل ِّرِمَ تْسُمْلﺍ<br><br>|َمْ َوْيلﺍ ُهَرْ َشن ،ِزَ ْكرَمْلﺍ ِﺏاَسِح َىَلع ْكيِفﺍَرْجوُفْنﺇ يف ُ زَ ْكرَمْلﺍ َحَصَ نَو ِريِوْط ّتلﺍَو ِفﺍَدْه ْلﺍ ِقيِقْحَت ْنَع ِلُﺯَانّتلﺍ ِوﺃ  ِﺱ ْيلﺍ مَدَع ِﺓَﺭوُرَضِ ب<br><br>ِﺕﺍّذِلل ِّرِمَ تْسُمْلﺍ<br><br>|Punctuation-induced diacritization ambiguity|
| |ُهَرَ َشن ). وﺃ ؛( ِزَ ْكرَمْلﺍ ِﺏاَسِح َىَلع ْكيِفﺍَرْجوُفْنﺇ يفُ زَ ْكرَمْلﺍ َحَصَ نَو ِفﺍَدْه ْلﺍ ِقيِقْحَت ْنَع ِلُﺯَانّتلﺍ ِوﺃ  ِﺱ ْيلﺍ مَدَع ِﺓَﺭوُرَضِ ب َمْ َوْيلﺍ ِﺕﺍّذِلل ِّرِمَ تْسُمْلﺍ ِريِوْطّتلﺍَو|Punctuation-induced diacritization ambiguity|

َ ّ َ ّ َ

َ ّ َ

ّ َ ٍّ ِ ّ َ ّ َ

ٍّ ِ ّ َ ّ َ ّ َ ّ َ ّ ُ ّ ُ ّ َ

ّ َ ّ ُ ّ ُ ّ َ

ّ َ ّ َ ّ َ ّ َ ّ َ ّ َ ّ َ

ّ َ ّ َ ّ َ ّ َ ّ َ ّ َ

ّ َ ِ ّ َ ّ َ ّ َ

ّ َ ِ ّ َ

ّ َ ّ َ ّ ُ ّ ُ ِ ُ

ّ ُ ّ ُ ِ ُ ِ ّ َ

ِ ّ َ

ِّ ّ َ ّ َ ّ َ ِ ّ

ِّ ّ َ ّ َ ّ َ ِ ّ

ّ َ ّ َ ّ َ ّ َ ِ

ّ َ ّ َ ِ

ِ ّ َ ّ ُ ّ َ ّ َ ّ َ ّ َ ّ َ ّ َ

ِ ّ َ ّ ُ ّ َ

ّ َ ّ َ ّ َ ّ َ ّ َ

ِ ِ ِ ّ َ ّ َ َ ِ ّ َ

ِ ِ

ِ ّ َ ّ َ َ ِ ّ َ

ّ ُ ّ َ ِ ٍ َ َ ِ َ

ّ ُ ّ َ ِ

ٍ َ َ ِ َ

َ َ ِ ِ ٍ ِ َ ُ ِ َ

َ َ ِ ِ ِ ٍ ِ

َ َ ُ ِ َ

ِ ِ َ

ِ ِ َ ِ َ

ِ َ ُ َ

ُ َ

ِ ّ َ ّ

ِ ّ َ ّ

ِ ّ َ ّ

ِ

ِ

َ َ َ ِ َ

َ َ َ ِ َ

ِ َ َ ِ

َ َ

###### Table 12: In the table provided, the red color indicates the presence of a diacritical error in the word, while the green color denotes the corrected diacritical marking. The orange color represents cases where multiple valid forms are allowed depending on the context or according to different linguistic schools.

These mistakes range from minor diacritization inconsistencies to more significant issues in data splitting and labeling. For instance, some words are incorrectly diacritized, while others show inconsistent diacritization patterns across similar contexts. Moreover, there are cases where the benchmark’s ground truth differs from the correct diacritization according to Arabic linguistic rules. These errors can lead to inaccurate assessments of model performance and potentially misleading conclusions about the state of Arabic diacritization technology.

