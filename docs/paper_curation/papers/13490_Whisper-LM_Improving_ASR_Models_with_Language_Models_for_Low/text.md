# arXiv:2503.23542v1[cs.CL]30Mar2025

## WHISPER-LM: IMPROVING ASR MODELS WITH LANGUAGE MODELS FOR LOW-RESOURCE LANGUAGES

#### Xabier de Zuazo*, Eva Navas†, Ibon Saratxaga‡, and Inma Hern´aez Rioja§

HiTZ - University of the Basque Country - UPV/EHU, Spain

March, 2025

### ABSTRACT

Automatic speech recognition systems have undoubtedly advanced with the integration of multilingual and multitask models such as Whisper, which have shown a promising ability to understand and process speech across a wide range of languages. Despite their robustness, these models often fall short in handling the linguistic distinctions of minority languages. This study addresses this gap by integrating traditional and novel language models with fine-tuned Whisper models to raise their performance in less commonly studied languages. Through rigorous fine-tuning and evaluation across multiple datasets, we demonstrate substantial improvements in word error rate, particularly in low-resource scenarios. Our approach not only does take advantage of the extensive data Whisper was pre-trained on, but also complements its linguistic adaptability by incorporating language models. We obtained improvements up to 51% for in-distribution datasets and up to 34% for out-of-distribution sentences using statistical language models, while large language models provided moderate but consistently robust improvement across diverse linguistic contexts. The findings reveal that, while the integration reliably benefits all model sizes, the extent of improvement varies, highlighting the importance of optimized language model parameters. Finally, we emphasize the importance of selecting appropriate evaluation parameters when reporting the results using transformer-based ASR models. In summary, this research clears the way for more inclusive ASR technologies that perform better across languages by enriching their linguistic knowledge. For further implementation details of this study, the technical documentation and source code are available at http://www.github.com/hitz-zentroa/whisper-lm.5

### 1 Introduction

In order to enable computers to comprehend and analyze human speech, automatic speech recognition (ASR) technology attempts to convert spoken language into text. The recent development and use of unsupervised and weakly supervised learning techniques have significantly expedited progress in ASR. One of these systems, Whisper, is a multilingual and multitask ASR model that has shown impressive performance in processing and recognizing speech in a wide range of languages and domains [1]. This innovation is further strengthened by the growing volume of pre-training data, which includes up to 680,000 hours of tagged audio to enhance generalization and robustness across datasets.

However, despite its effectiveness, Whisper and similar models often struggle with linguistic characteristics and grammatical peculiarities of low-resource languages, necessitating further refinement to optimize performance. This motivates the integration of language models (LMs), which have historically played a vital role in enhancing linguistic

∗xabier.dezuazo@ehu.eus †eva.navas@ehu.eus ‡ibon.saratxaga@ehu.eus §inma.hernaez@ehu.eus 5This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which

this version may no longer be accessible.

coherence and context handling in ASR systems. By embedding language models, we aim to bridge the gap between the phonetic richness captured by Whisper’s encoder and the unique syntactic and semantic structures these languages exhibit.

Beyond closing the linguistic gap, a key achievement of this work is to improve the out-of-distribution (OOD) robustness, as ASR applications often exhibit worse performance when confronted with unseen or real-world data. To that end, we integrate language models into Whisper not through prompting but by merging their internal scores at inference time. This method augments Whisper’s outputs with an external language prior and effectively mitigates the domain-shift issues. In addition, we propose the Effective Robustness of Relative Error Reduction (noted ERER), a novel metric that quantifies how consistently the performance of a model scales from in-distribution to OOD scenarios. As our experiments show, both n-gram and large language models can substantially improve OOD performance under this framework, offering a more robust path toward more powerful ASR models across a variety of domains.

Additionally, the complexity and variability of real-world applications highlight the need to critically evaluate the impact of various model configuration parameters, such as those used at evaluation time. Despite not being very studied, these parameters considerably influence the ASR models’ reported results when extended to novel or challenging linguistic environments. This paper seeks to methodically explore these aspects, exhaustively analyzing how different evaluation-time settings within the Whisper framework affect model performance across diverse datasets.

Our primary contribution is to demonstrate improved accuracy and robustness in speech recognition for low-resource languages through the integration of language models with weakly supervised acoustic ASR systems. This includes discarding possible data leakage by a sentence-level overlap analysis. We also provide a detailed exploration of the evaluation parameters that prominently affect the final results. This study marks the first documented instance of integrating Whisper with different language models, including an exhaustive comparison of their respective contributions. This methodology incorporates rigorous fine-tuning experiments supplemented by language model integration, offering insights into the synergistic combination of contemporary deep learning ASR models with both traditional and novel language models.

### 2 Related Work

The integration of language models into ASR systems, while not new, remains highly relevant in the era of deep learning-driven speech recognition technologies. The profound advancements in ASR, notably through deep recurrent neural networks [2], attention mechanisms [3], and transformer models [4], have pushed the limits of what these systems can achieve. These advances underscore the evolution of ASR from traditional methods to modern approaches that integrate complex neural network architectures, addressing challenges in various applications ranging from industrial sectors to military and cultural applications [5]–[9], and even more sophisticated fields like cognitive neuroscience [10]–[13]. In fact, multiple studies demonstrate that ASR systems can surpass human-level accuracy in certain environments, which prepares the path for their extensive deployment in real-world applications [14], [15].

Research on robust ASR systems often involves semi-supervised and self-supervised learning approaches, which use unlabeled data to improve learning efficacy [15]–[28]. These methods have proven particularly effective in escalating the ASR systems to handle multiple languages and dialects, which is essential to create inclusive technologies that cover the global linguistic diversity [29]–[35]. There are even some initial efforts to introduce semantic information into self-supervised encoders using unlabeled data [36].

Furthermore, the field has explored various strategies to adapt ASR systems to low-resource settings, often leveraging transfer learning and data augmentation techniques [37]–[39]. A remarkable example of integrating LLMs with ASR models for low-resource languages is the work by Vasquez-Correa et al. [40], who explored the use of language models to select between different transcriptions provided by Zipformer ASR systems [41]. This represents an initial, albeit shallow, integration of LLMs in ASR, suggesting promising future pathways for deeper synergies between these technologies. These strategies are vitally important for improving speech recognition accuracy in languages that lack extensive training corpora.

On the other hand, more recent techniques use weakly supervised pre-training methods to exploit large datasets with noisy, imprecise labeling [42]. This approach is particularly advantageous when labeled data is scarce or of low quality, which is often true with multilingual speech transcriptions. Whisper models, developed using weakly supervised pretraining, are excellent in handling multilingual speech recognition tasks. This competence positions Whisper models as prime candidates for further improvement through the integration of language models, especially to boost performance in low-resource languages.

A considerable amount of research has also been committed to the application of Whisper for various speech recognition tasks, highlighting its adaptability and the extensive scope of its training data [1], [43], [44]. Along the way,

multiple approaches and techniques to fine-tune Whisper-based models have been developed [45]–[50], and a more exhaustive evaluation of the models has been performed [51]. In addition, the challenge of enhancing ASR performance for low-resource languages has been addressed by integrating Whisper with techniques to mitigate the impact of data imbalance, such as weighted cross-entropy for language discrimination [52], or by encoding the structured information into a sequence to allow multitask learning [53].

In this rich context of ongoing research and development, our work contributes to a deeper understanding of language model integration within the Whisper framework and its impact on ASR performance across diverse linguistic environments. The work is heavily based on previous approaches to integrate language models with neuronal ASR models [15], [54]. Our method leverages Whisper’s foundational strengths, augmenting them with linguistic models to augment its performance and robustness in multiple languages. Specifically, we focused our contribution on four Iberian languages: three low-resource languages (Basque, Galician, and Catalan) and one high-resource language (Spanish) [55]. In this regard, for our specific case, the Catalan language should be considered more a mediumsized language being more present online [56], [57] and, indeed, it has considerably higher representation in both the pre-training and fine-tuning datasets used here.

Besides, efforts to reproduce and improve Whisper models through open-source platforms and public datasets have gained traction, with initiatives like ESPnet [58] and subsequent improvements in Whisper-style model training [59], [60]. These efforts align with the open science movement, which promotes the transparency and accessibility of advanced models and training pipelines [61]–[65]. In line with these open science principles, all of our material is released with an open-source license for free use and to encourage further research.

### 3 Methodology

This section outlines the methodologies used to improve the performance of multilingual Whisper models. We detail a structured fine-tuning process to enhance Whisper’s performance across multiple languages. We then describe how language models are integrated at inference time to further increase accuracy and robustness. This includes an analysis of sentence leakage, and an ablation study to assess the impact of various evaluation parameters.

#### 3.1 Fine-Tuning Process

We employed a standard fine-tuning process to adapt the multilingual Whisper models to improve performance in specific languages. Each model size, ranging from Tiny to Large-V3, was individually fine-tuned using the Hugging Face’s community Whisper Fine-Tuning Event code6. Fine-tuning was carried out on separate models for four distinct languages and seven different model sizes. This involved initializing the models with the pre-trained multilingual Whisper models provided by OpenAI (vanilla) and then fine-tuning them using the corresponding language subset from the Common Voice version 13.0 dataset [66], including both the train and validation splits. This practice, recommended in the documentation for low-resource languages7, provides a larger training corpus than using only the train partition. However, it may lead to slightly optimistic results in the fine-tuning scores on the Common Voice dataset. In spite of that, as our study is focused on later language model integration benefits, this will not affect our final analysis. Ultimately, we ended with one fine-tuned model per language and size.

For homogeneity and fairness, parameters shared between languages were chosen based on the original scripts, and some adjustments derived from preliminary tests to improve performance while ensuring efficient training times. The learning rate is one of the most critical hyperparameters for Whisper model fine-tuning. Following the advice of Jong Wook Kim8, one of the authors of the Whisper paper, we started with a learning rate that is approximately 40 times lower than the pre-training rate. Then, we adjusted the final learning rate based on initial training tests on the datasets and languages we used. The main idea was to use hyperparameters that worked well enough across languages to homogenize the fine-tuning processes and have an equivalent cross-language evaluation. These were the final learning rates used for each model size: Tiny models used a learning rate of 3.75 · 10−5, Base models 2.5 · 10−5, and Small to Large-V3 models 1 · 10−5. Based on our experience, starting with a learning rate of 1 · 10−5 gives better results with the largest models. Besides, we employed a linear learning rate scheduler with a warm-up phase of 500 steps, which is essential for stabilizing the training process early on. For all model sizes, the optimizer used was AdamW, with betas set to (0.9, 0.999) and epsilon to 1 · 10−08 [67]. The batch sizes were set based on our GPU’s capacity, ranging from 256 for Tiny models to 32 for Large-V3 models, with gradient accumulation set to 2 for the largest models to compensate for the smaller batch size. Evaluation batch sizes were set correspondingly, from 128 for Tiny

- 6 https://github.com/huggingface/community-events
- 7https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event#data
- 8https://github.com/vasistalodagala/whisper-finetune/blob/master/README.md#hyperparameter-tuning

to 16 for Large-V3 models. Fine-tuning steps were adjusted to the complexity and size of the models: 5,000 steps for small models and up to 20,000 for the largest models. This ensured that each model was adequately trained to converge to optimal performance, reaching a good improvement plateau during training. A detailed summary of the hyperparameters used for fine-tuning each model size is provided in Table 1.

Table 1: Summary of fine-tuning hyperparameters for the vanilla Whisper models.

|Model Size|Learning Rate Batch Size Eval Batch Size Gradient Acc. Steps<br><br>|
|---|---|
|Tiny Base Small Medium Large-V1..V3|3.75 · 10−5 256 128 1 5,000<br><br>2.5 · 10−5 128 64 1 5,000 1 · 10−5 64 32 1 5,000 1 · 10−5 64 32 1 10,000 1 · 10−5 32 16 2 20,000<br><br>|

#### 3.2 Using Whisper with N-Gram Language Models

Integrating n-gram language models with Whisper at inference time improves accuracy by combining the neural model’s output with statistical language patterns captured by n-grams. This integration involves adjusting the log probabilities from Whisper’s output based on the probabilities of the n-gram model.

The core of our integration involves modifying Whisper’s beam search process. Precisely, we adjust the log probabilities of candidate tokens generated by Whisper using scores from a statistical language model. Equation 1 governs this adjustment, which has been previously used in the literature, including for Deep Speech models [14], [68].

Q(c|x) = log(Pacoustic(c|x)) + α · log(PLM(c)) + β · word count(c) (1)

In this equation, Q(c|x) represents the unnormalized log probability (or score) of the sequence c of candidate tokens, conditioned on the audio input x. log(Pacoustic(c|x)) is the sequence’s log probability from Whisper (acoustic model) given the audio input x, log(PLM(c)) is the log probability from the n-gram model, and word count(c) counts the words in c. The parameters α and β, which scale the language model’s and the sentence length’s influence, respectively, are optimized for each language using a systematic approach detailed below. Notably, this equation is applied only at word boundaries. In addition, to prevent premature evaluation of short sequences, we introduce a length threshold based on the shortest recordings in the datasets. Specifically, language model scoring is applied only to sequences containing at least four Whisper tokens. These measures ensure that the language model’s influence is applied when sufficient contextual information is present, enhancing the reliability of our results and avoiding repetitions and hallucinations.

As the n-gram language model, we selected KenLM for its ability to work successfully with an enormous linguistic corpus and for using Kneser-Ney smoothing to obtain relatively low perplexity. Furthermore, KenLM’s efficient querying of n-gram probabilities is essential to maintain performance in ASR applications [69], [70]. Based on previous work [14], [21], [68], [71], we will restrict our approach to 5-gram language models for simplicity and to include a long enough linguistic context.

Concerning the parameter values to use, proper optimization of the α and β in Equation 1 is indispensable to integrate the Whisper neural model with it smoothly. We employed a Bayesian optimization approach using the Optuna framework [72] for this optimization process. This process involves optimizing the parameters α and β to minimize the Character Error Rate (CER) [73] or the Word Error Rate (WER) [74], depending on the characteristics of the language. For our optimization, Optuna uses its default single-objective sampler, the Tree-structured Parzen Estimator (TPESampler) [75]. The TPESampler models the probability distribution of parameters using two Gaussian Mixture Models, one for parameters leading to better objective values and another for the rest. The sampler then selects new parameters by maximizing the ratio of these two models, concentrating the search on areas of the parameter space that lead to improved performance outcomes. For the search process to be more feasible, we have constrained the range of values of α and β parameters to be between 0 and 5.

Our objective function for the optimization performed here measures the transcription error (WER in our case) of predictions made by the Whisper model, adjusted using the 5-gram model scores. For each trial, a different value of α and β is suggested, and they are used to transcribe the train and validation dataset examples. It is crucial to avoid using the test split at this stage, as the model is prone to rapidly overfitting to it, leading to overly optimistic results in the optimized split. Due to the slowness of the search process, we parallelized it by loading up to 32 model instances in the same GPU at the same time and, hence, having multiple trials running simultaneously.

The optimization was conducted in over 100 trials, systematically exploring the parameter space. The number of trials is essential: making too few trials will not improve the results, and too many trials may lead to overfitting to the dataset, performing worse in new datasets, and not generalizing correctly. Also, for efficiency, all optimizations were performed using only the Tiny models, and the resulting parameters were then reused for the other model sizes. This approach of reusing the parameters for the other model sizes will probably not provide the best results in the large models. Still, in our experiments, it works well enough for an initial approach and is recommended due to the resource-intensive nature of the optimization process.

To summarize, this method will improve the fine-tuned models by adapting the Whisper model to account for linguistic characteristics captured by the language models, improving overall performance across different linguistic environments. That is to say, it ensures that the Whisper decoder’s knowledge, which is general to all languages, is complemented by specific linguist statistical knowledge of the language models.

#### 3.3 Using Whisper with Large Language Models

Following a similar approach to the integration of n-gram models, we extended our methodology to incorporate Large Language Models (LLMs) with Whisper, utilizing the same adjustment equation (Equation 1). This integration aims to take advantage of the broader contextual knowledge of LLMs to further improve Whisper’s performance, especially in terms of generalizability and robustness. For an efficient integration process, we made several adaptations: the range of parameter values (α and β) was constrained to the interval [0, 3], and a subset of 4000 randomized recordings from both training and validation splits was used for a more manageable computation. This integration was performed in parallel across 7 GPUs to accelerate the optimization, which involved 100 trials, similar to the n-gram model integration. We leveraged specific LLMs optimized for particular languages: Latxa 7B v1.2 for Basque [76] (based on Llama 2 [64]), Carballo Cerebras 1.3B for Galician [77] (based on FLOR 6.3B [78]), and FLOR 6.3B for both Catalan and Spanish [78] (based on BLOOM-7.1B [79] and Chinchilla [80]). These models were selected due to their extensive pre-training on large, diverse corpora and their effectiveness in linguistic applications relevant to the respective languages. To integrate them, we feed partial transcripts generated by Whisper into the LLM and collect the output logits for the next token. We then apply a softmax to these logits, extract the highest log probability, and use it as the language model score. Specifically, if

ℓ = logits(c)[−1], (2) represents the final token logits after processing the partial transcript c with the LLM, then the score is given by

PLM(c) = max softmax(ℓ) , (3)

which we hypothesize serves as a reasonable proxy for the LLM’s confidence in the partial transcript. Actually, this provides a simple and effective measure of the LLM’s trust in the next token. While this does not directly represent the sentence probability, we expect it to serve as a practical approximation for overall sentence reliability, helping to rescore the Whisper outputs at each decoding step.

#### 3.4 Corpora Leakage Analysis

To ensure the integrity of our evaluations, particularly when assessing out-of-distribution datasets, we measured the degree of sentence overlap between our evaluation datasets and the corpora used to train the language models. This analysis helps confirm that a likely text data leakage is not skewing our results. Our n-gram language model corpora creation is detailed in Section 4.1.2. For the LLM language models’ corpora, we have used Latxa Corpus v1.1 [76] for Basque and CorpusNOS´ [81] for Galician. The corpus used for the Catalan and Spanish LLM model is not publicly available, complicating the leakage assessments for these languages. For Catalan, we recreated the complete original, non-public LLM corpus by compiling a dataset that includes CATalog 1.0 [82]9, CaWaC [83], [84]10, mC4 [85]11, Wiki-40B [86]12, and a recent Wikipedia dump as of 2025-02-1413. Similarly, for Spanish, we successfully recreated 98% of the original non-public LLM corpus by compiling a dataset that includes mC4 [85], Wiki-40B [86], legal corpora [87]14, the parts available from the biomedical corpora [88]15, and a recent Wikipedia dump as of 2025-02-14.

- 9https://huggingface.co/datasets/projecte-aina/CATalog
- 10https://zenodo.org/records/4519349
- 11https://huggingface.co/datasets/allenai/c4
- 12https://huggingface.co/datasets/google/wiki40b
- 13https://dumps.wikimedia.org/
- 14https://zenodo.org/records/5495529
- 15https://github.com/PlanTL-GOB-ES/lm-biomedical-clinical-es

Nevertheless, to ensure a fair comparison, all sentences from both the evaluation datasets and the language model corpora in this analysis have been normalized using the methods described in Section 4.1.2. Additionally, the Whisper text normalizer has been applied, removing punctuation, lowercasing, diacritic normalization, and other linguistic standardizations before sentence comparison [1].

#### 3.5 Ablation Study of Evaluation Parameters

This study evaluates the impact of various evaluation options on the performance of the ASR using the Whisper vanilla models. We examine how changing specific parameters affects the WER by disabling them individually from our default configuration. To methodologically assess the impact of these parameters, we conducted a series of experiments in which each parameter varied while others remained in our selected settings. In simple terms, we disabled each parameter individually over our baseline. Furthermore, we will use the original multilingual OpenAI models, not the fine-tuned ones. This will adequately justify the default parameters used in the rest of our study.

The parameters studied here are the following:

- • Beam size: Beam search is used in decoding to consider multiple hypotheses at each step. A default beam size of 5 is utilized based on Whisper’s suggestion to balance performance and computational efficiency. We will compare it by disabling the beam search, also known as greedy decoding.
- • Diacritics: Taking the diacritics into account when evaluating the model can substantially alter the reported results. We ignored diacritics to homogenize language differences and possible encoding format inconsistencies between and within datasets.
- • Timestamps: While timestamps are important for applications like subtitle generation, they are not the focus of our study. Instead, we assess whether enabling timestamps in the output affects the transcription accuracy.
- • Language: Specifying the language of the audio can guide the model’s recognition process, mainly when working with non-English languages and out-of-domain datasets. Initially, we hypothesized this parameter may gain importance for low-resource languages.
- • Temperature: This parameter controls the randomness of the predictions, with a higher temperature introducing more variability into the output. By default, it has a range of values from 0.0 to 1.0, incremented by 0.2. This is known as a temperature scheduler (i.e., ‘(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)‘) and reflects a progressive increase typically used for long-sentence scenarios [1]. In the context of Whisper, the temperature scheduler is used to mitigate common decoding issues with long-form audio contexts. In our particular case, we disabled it by default because we are working with sentence-level audio recordings, and we want to enable beam search for the language models. Still, we want to measure to what extent this parameter affects the scores in our datasets, just in case it is relevant to our particular scenario.

Our default configuration includes a beam size of 5, diacritics removed, timestamps excluded, the language provided to the model, and a single temperature value of 0.0.

### 4 Experimental Setup

This section presents the experimental framework used to evaluate the models, detailing the datasets, model configurations, and the experimental environment employed.

#### 4.1 Datasets

This study utilizes a diverse set of datasets to evaluate the performance of the models across various languages. The datasets are selected to test both in-distribution (ID) and out-of-distribution (OOD) generalization ability.

#### 4.1.1 Audio Datasets

The models are fine-tuned on the Common Voice version 13.0 dataset, which is used in the speech recognition community in recent work due to its diversity and size [43], [47]. These fine-tuned models are later evaluated on several other datasets to measure generalization across different acoustic and linguistic environments.

As for in-distribution evaluation, all models are evaluated using the Common Voice 13.0 dataset’s test split to measure the ID WER.

Concerning the out-of-distribution Evaluation, for language-specific assessments, the models are tested on the following datasets:

Pre-Train Fine-Tune Evaluation

gl:9h eu:21h 1 , 8 8 3 h

23h 26h

435h 30h 25h

49h 50h

1,6 4 3 h

1 1 , 1 0 0 h

Basque (eu) Galician (gl) Catalan (ca) Spanish (es)

Figure 1: Distribution of dataset hours across different training phases.

- • Basque: Evaluated on AhoMyTTS (non-public) [89] and OpenSLR-76 [90]. The AhoMyTTS is especially challenging because it is not in the public domain and uses unique sentences that are unavailable online. Therefore, it may serve as a control of possible leakages in the language model’s corpora.
- • Galician: Evaluated on OpenSLR-77 [90], and Google’s FLEURS [91].
- • Catalan: Evaluated on OpenSLR-69 [90], and Google’s FLEURS.
- • Spanish: Evaluated on Google’s FLEURS and Facebook’s Multilingual LibriSpeech (MLS) [92].

Further details of the datasets used for fine-tuning and evaluation are in Table 2.

|Dataset Short Name Language Split<br><br>|Recordings Hours|
|---|---|
|Common Voice 13 CV13 Basque train+validation Common Voice 13 CV13 Galician train+validation Common Voice 13 CV13 Catalan train+validation Common Voice 13 CV13 Spanish train+validation|17,509 25.9 17,348 23.1<br><br>1,063,345 1,643.1 296,037 434.8<br><br>|
|Common Voice 13 CV13 Basque test AhoMyTTS AhoMyTTS Basque -<br><br>OpenSLR-76 SLR76 Basque Common Voice 13 CV13 Galician test FLEURS Fleurs Galician all<br><br>OpenSLR-77 SLR77 Galician Common Voice 13 CV13 Catalan test FLEURS Fleurs Catalan all OpenSLR-69 SLR69 Catalan Common Voice 13 CV13 Spanish test FLEURS Fleurs Spanish all Multilingual LibriSpeech (MLS) MLS Spanish test<br><br><br>|6,591 10.5 590 0.8<br>7,136 13.9 6,546 9.4 3,497 10.3 5,587 10.3<br><br><br>16,380 28.1<br><br>3,644 11.9<br>4,240 9.4<br><br><br>15,708 26.8 4,112 13.3 2,385 10.0|

Table 2: Details of the datasets used for fine-tuning (top) and evaluation (bottom).

In addition to that, Figure 1 shows the distribution of dataset hours across different training stages: pre-training, finetuning, and evaluation. The pre-training hours per language, taken from the original Whisper paper [1], show Catalan’s representation as medium-sized compared to other languages. During fine-tuning, Catalan is notably over-represented, surpassing even Spanish in terms of hours, which could influence model adaptability. However, for evaluation, the datasets are balanced across languages, suggesting that the precision of our final scores will likely be more equitable and unbiased across different languages. In any case, based on the data distribution shown, Basque and Galician can be considered under-resourced in this experiment.

#### 4.1.2 Textual Corpora for Language Models

For the creation of the n-gram language models, each language’s corpus is capped at 27 million sentences to maintain comparability and manageability. The main sources of these corpora are as follows: EusCrawl 1.0 [93] for Basque, SLI GalWeb Corpus [94] for Galician, Catalan Textual Corpus [83], [84] for Catalan, and Spanish LibriSpeech MLS [92] for Spanish. If needed, a recent Wikipedia dump and the Opus [95] corpus are appended to the initial corpora until it reaches 27 million sentences. Subsequently, the number of sentences for the corpus for each language is identical, making the model comparison fairer. Besides, all texts within these corpora have been segmented and normalized to ensure uniformity in language modeling processes, following established methods in the field [71].

#### 4.2 Model Configurations

This study utilizes the series of official multi-lingual Whisper models, ranging from Tiny to Large-V3, which we have fine-tuned to improve performance in specific languages. These models are available in the official OpenAI Hugging Face repository16. Their underlying structure remains consistent across models, and no internal changes were made to the original models. Basically, based on OpenAI’s Whisper transformer encoder-decoder architecture [96], these models are adapted to handle specific languages, improving their proficiency without changing the model’s constructional design.

#### 4.3 Experimental Environment

The study leveraged substantial computing resources to ensure the efficient processing of models and data.

Regarding the hardware, as this experiment has been extended over time, the fine-tuning, optimization, and posterior evaluation tasks have been performed in different server instances as our infrastructure evolved. The initial finetuning and n-gram LM evaluations were performed on an NVIDIA A100-SXM4-80GB GPU, supported by 16 AMD EPYC™7513 CPU cores with 128 MB of RAM memory. For the ablation study, we used 8 AMD EPYC™75F3 CPU cores with a single NVIDIA A100-SXM4-80GB GPU. Lastly, all the LLM optimization and evaluation processes have been completed in 196 Intel®Xeon Platinum 8480C CPU cores with 7 NVIDIA H100 80GB HBM3 GPUs working in parallel.

As for the software, the models were developed and evaluated using a suite of software tools and libraries:

- • Hugging Face library: utilized for model fine-tuning through the scripts from the Whisper language finetuning community event6 and for loading the LLM tokenizers and models.
- • Whisper ASR: we utilized the official OpenAI’s Whisper implementation for the language model integration and all evaluation tasks17.
- • Common Voice Utils: used to preprocess the corpora for the language models by segmenting and normalizing the texts in the different languages [71]18.
- • KenLM library: used to construct robust 5-gram language models trained for each language19 together with its Python package for the final integration.
- • Optuna framework: for the optimization of the language model parameters20.
- • JiWER: used for measuring the CER and WER, providing accurate and reliable evaluation metrics21.

#### 4.4 Evaluation Metrics

In this study, we primarily utilize the WER [74] to assess the accuracy of the fine-tuned Whisper models across different datasets and languages. The WER is a standard metric in speech recognition that measures the performance of an automatic speech recognition system. Particularly, it calculates the errors in the transcribed text, which is analogous to accuracy in more traditional classification tasks. Although WER is a widely used metric, comparing systems across diverse languages and datasets can sometimes be challenging. For this reason, in this work, we have included additional measures derived from WER to facilitate more consistent comparisons across these varied conditions.

- 16https://huggingface.co/openai
- 17https://github.com/openai/whisper/ (version v20231117)
- 18https://github.com/ftyers/commonvoice-utils/
- 19https://kheafield.com/code/kenlm/
- 20https://optuna.org/
- 21https://jitsi.github.io/jiwer/

Relative Error Reduction (RER). In response to these challenges, we will employ the RER metric [1], [97], [98], calculated using Equation 4. This metric helps contextualize improvements by calculating the error reduction relative to a baseline, providing a simpler view of performance changes across diverse conditions. By focusing on relative rather than absolute improvements, RER also aligns better with the diminishing returns often observed in neural models as their performance increases [99].

WERwith intervention WERbaseline × 100% (4)

RER = 1 −

Effective Robustness of Relative Error Reduction (ERER). To compare the robustness of different fine-tuning and language modeling approaches in speech recognition, we adapt the concept of effective robustness from Taori et al. [100] to the ASR domain, following a method similar to that in the Whisper paper [1].

In Taori et al., effective robustness is defined as in Equation 5, where acc1(f) denotes the model’s accuracy on an in-distribution (ID) test set, acc2(f) is its accuracy on an out-of-distribution (OOD) test set under some distribution shift, and ξ(·) is a baseline function that predicts acc2 given acc1. A distribution shift is understood as a change in data properties not covered by the ID test set. Since higher accuracy corresponds to better performance, a model lying exactly on the baseline has ρ(f) = 0, while a model exceeding the expected OOD performance has ρ(f) > 0.

ρ(f) = acc2(f) − ξ acc1(f) (5)

With regard to ASR models, we typically use the WER metric, where lower values are better. However, to maintain consistency with the ”higher is better” paradigm from Taori et al., we instead measure RER. Specifically, let RERID(f) be the relative error reduction on an ID dataset compared to a reference system, and REROOD

(f) the relative error reduction on the i-th OOD dataset. We then define ERER in Equation 6, where N is the total number of OOD datasets and f denotes the model. Intuitively, ERER(f) measures how much better (or worse) a model performs under OOD conditions relative to its ID performance.

i

ERER(f) =

N

1 N

i=1

(f) − RERID(f) , (6)

##### REROOD

i

In practice, the function ξ from Taori et al. [100] is conceptualized as the direct ID performance, assuming that the expected baseline OOD performance should ideally match the ID performance. This simplification reflects our empirical observation that OOD performance generally tracks ID performance closely but is shifted by a constant factor reflecting the generalization gap.

To clarify, the ERER intuitively captures the relative improvement or degradation in model performance when subjected to new or unseen data scenarios, providing a direct measure of a model’s robustness beyond conventional accuracy metrics. An ERER value close to zero indicates that the model’s OOD performance scales with its ID performance in a predictable way, suggesting balanced robustness. Negative ERER values imply that the model underperforms on OOD data relative to what one would expect, given its ID results. Conversely, positive ERER values, though less common, imply that the model exceeds its expected performance under a distribution shift. From a robustness standpoint, the ideal scenario is an ERER of zero or above, indicating that the model’s OOD performance keeps pace with (or even surpasses) its ID performance, thereby maintaining uniformly high performance across both ID and OOD conditions.

The RER and ERER metrics ensure that our evaluations are both exhaustive and easily understandable, facilitating a better understanding of the effectiveness of the proposed fine-tuning and language modeling interventions. Nevertheless, the detailed WER scores for each model configuration and dataset are provided in Appendix A, allowing further inspection.

#### 4.5 Statistical Significance Analysis

To evaluate the statistical significance of the differences observed in our results, we employ the Wilcoxon signed-rank test [101]. This non-parametric test is ideal for our analysis as it does not assume a normal data distribution, making it well-suited for real-world datasets that may be skewed or non-uniform. The test is used to determine if the median of the differences between paired observations, such as WERs across different models or configurations, significantly deviates from zero, indicating a true effect rather than a result of random variation [102].

The Wilcoxon signed-rank test has been applied in two scenarios within our study:

- 1. Method-level significance: To assess the overall influence of the different methodological interventions (such as fine-tuning, LM integration, or ablated parameters) on model performance. In this case, we will compare the dataset-level WER scores across methods. This analysis will help us assess the impact of the method employed.
- 2. Sentence-level significance: To analyze the impact of individual result values across specific datasets and model sizes, ensuring a fine-grained evaluation of performance variations. In this method, we will compare individual sentence-level WERs across the different methods. These comparisons will be displayed in the results tables as a superscript in the scores.

For statistical comparisons, we juxtapose the fine-tuned model results against the baseline vanilla models and subsequently compare the outcomes of n-gram and large language model integrations against their respective fine-tuned baselines. This approach mirrors the sequential application of each method to the models, reflecting the incremental improvements aimed at each step. Unless otherwise stated, results will be considered statistically significant at a p-value threshold of less than 0.001, often referred to as very highly significant or extremely significant. This strict criterion helps ensure that the observed differences are highly unlikely to be due to chance, thereby confirming the efficacy of the tested approaches. This methodological rigor is critical for drawing reliable conclusions from our experiments and posterior analysis.

### 5 Results

The following subsections provide and analyze the results of the fine-tuning process, the integration of language models with the fine-tuned models, the corpora leakage analysis, and, finally, the parameter ablation study of vanilla models.

#### 5.1 Performance Improvements

This subsection discusses the improvements observed when integrating language models with Whisper for various languages and datasets, comparing the performance before and after the integration of language models.

#### 5.1.1 Fine-Tuning Results

As shown in Table 3, fine-tuning Whisper models undoubtedly improves WER across most scenarios in low and medium-resource languages. The most prominent gains are seen in ID datasets, where improvements are as high as 76% for the Medium and Large models in the Basque language. Notably, improvements for OOD datasets are somewhat lower but still substantial, peaking at 68% improvement for the Medium model in Basque in the AhoMyTTS dataset. In general, the upgrades seem to affect all the model sizes, from the Tiny to Large models, with an overall positive trend. For details on the WER values from which these improvements were calculated, refer to Appendix A.1.

- Table 3: The relative error reduction for fine-tuned models compared to vanilla models, with ID datasets listed at the top and OOD datasets at the bottom. Values indicating the highest improvements are marked in bold, and negative reductions are highlighted in red. Significance levels are indicated as follows: pa < 0.05, pb < 0.01, pc < 0.001, no-superscript meaning no significance.

|Language Dataset<br><br>|Tiny Base Small Medium Large L-V2 L-V3|
|---|---|
|Basque CV13 Galician CV13 Catalan CV13 Spanish CV13|+69%c +73%c +75%c +76%c +76%c +74%c +73%c +56%c +64%c +66%c +68%c +65%c +64%c +64%c +73%c +71%c +65%c +69%c +71%c +70%c +59%c +40%c +35%c +23%c +23%c +19%c +14%c -2%<br><br>|
|Basque AhoMyTTS Basque SLR76 Galician Fleurs Galician SLR77 Catalan Fleurs Catalan SLR69 Spanish Fleurs Spanish MLS<br><br>|+61%c +66%c +66%c +68%c +64%c +60%c +62%c +59%c +63%c +62%c +62%c +60%c +58%c +58%c +43%c +48%c +47%c +41%c +29%c +25%c +14%c +46%c +55%c +50%c +52%c +46%c +48%c +49%c +48%c +39%c +11%c 0%c -10% -24%c -48%c +57%c +50%c +32%c +32%c +28%c +21%c +8%c<br><br>-1%c -6%c -17%c -9%c -15%c -10%c -12%c<br>-11%c -14%c -11%c -16%c -12%c -31%c -40%c<br>|

In contrast, the results for the high-resource language in OOD datasets are mixed. For instance, fine-tuning negatively impacts Spanish models when evaluating them with datasets like MLS, where WER worsened by up to 40% in the

Large-V3 model configuration. This trend affects smaller models less, but still, there is a slightly worsening effect, and no model size shows improvements in OOD datasets for Spanish. This probably means the pre-trained model decoder already has considerable Spanish linguistic knowledge. Consequently, during fine-tuning, it accidentally overfits to the dataset and loses the ability to generalize to other contexts.

Similarly, medium-resource languages like Catalan exhibit uneven improvements. This is particularly true in one OOD dataset, where larger models do not consistently yield better results, occasionally showing degradation in performance, as with the Large-V3 model in the FLEURS dataset. Nevertheless, the small models still benefit from fine-tuning, with a peak at 57% in the Tiny model with the OpenSLR-69 dataset, and the net result is very positive.

Altogether, the changes produced by the fine-tuning method are statistically significant, confirming the effectiveness of fine-tuning across various languages and model sizes (W = 261.0, p = 1.07 · 10−11). Moreover, most of the individual performance improvements for each dataset and model size are also statistically significant. For detailed significance levels, refer to the annotations in the Table 3 (pa < 0.05, pb < 0.01, pc < 0.001, no-superscript indicating no significance).

#### 5.1.2 N-Gram Language Model Integration Results

In this section, an n-gram language model is integrated with the fine-tuned Whisper model. As previously stated, the LM weight optimization process was performed only using the Tiny models of each language, whose parameters were later reused for the other model sizes. Hence, the smallest model results are the most relevant for our analysis. Nonetheless, all results of all the model sizes will be reported and analyzed.

Overall, integrating language models after the fine-tuning demonstrates general positive results. Table 4 shows the final results when comparing the scores with the fine-tuned models without a language model. The improvement percentages listed are incremental, indicating additional gains on top of those achieved through fine-tuning alone. For instance, a +37% improvement in the Tiny model for Basque means a further 37% error reduction beyond the initial 69% improvement from fine-tuning. The improvement is compared with fine-tuned models as a baseline, even for the cases where the models do not clearly improve, like in Spanish. In other words, we added the language models to the fine-tuned models in all cases, even if they were not clearly better than the vanilla models. Notwithstanding these results, if the goal is to obtain the best performance, attaching the language model to the original models might be a better approach for high-resource languages (see Appendix A.2 for further details).

- Table 4: The relative error reduction for fine-tuned models with a 5-gram language model compared with the previous fine-tuned results without the language model. ID datasets are listed at the top, and OOD datasets are listed at the bottom. Values indicating the highest improvements are marked in bold, and negative reductions are highlighted in red. Significance levels are indicated as follows: pa < 0.05, pb < 0.01, pc < 0.001, no-superscript meaning no significance.

|Language Dataset<br><br>|Tiny Base Small Medium Large L-V2 L-V3|
|---|---|
|Basque CV13 Galician CV13 Catalan CV13 Spanish CV13|+37%c +45%c +50%c +50%c +49%c +48%c +51%c +32%c +39%c +40%c +39%c +35%c +33%c +38%c +14%c +14%c +11%c 0%c +4%c +7%c +7%c +22%c +24%c +25%c +23%c +19%c +19%c +18%c<br><br>|
|Basque AhoMyTTS Basque SLR76 Galician Fleurs Galician SLR77 Catalan Fleurs Catalan SLR69 Spanish Fleurs Spanish MLS<br><br>|+9%c +29%c +23%c +34%c +30%c +31%c +30%c +21%c +22%c +22%c +22%c +20%c +20%c +23%c<br><br>+14%c +15%c +8%c -3%a -2%c -1%c -4%<br><br>+21%c +20%c +20%c +16%c +16%c +15%c +14%c +13%c +12%c +12%c +5%c -1%c +1%c +7%c +19%c +18%c +19%c +13%c +13%c +13%c +12%c<br><br>+7%c +5%c -8% +1%c 0%c 0%b -1%b<br><br>+4%c +9%c -2%c -9%c -5%c -17% -10%|

The results show remarkable improvements in ID datasets, with the Large-V3 model in the Basque Common Voice dataset improving by up to 51%. The improvements for OOD datasets reach up to 34%, as observed in the Medium model for the Basque language in the FLEURS dataset.

Contrary to fine-tuning results, adding language models still benefits the smaller models of high-resource languages, such as Spanish, or, alternatively, it helps mitigate some of the performance declines observed during fine-tuning. Nonetheless, as it happens with fine-tuning, it still does not consistently benefit large models. For example, when tested on Spanish OOD datasets like FLEURS and MLS, the Large, Large-V2, and Large-V3 models often show minimal to negative improvements. This could be indicative of the limitations of the current LM parameters used, which were

optimized based on the Tiny models and may not scale correctly across larger model architectures. Additionally, the quality of corpora used for training the language models might also be influencing these scores.

With regard to the statistical effect of n-grams integration, the differences are very highly significant (W = 186.0, p = 9.95 · 10−13), demonstrating the value of this method. Most of the individual score changes are also statistically significant, as shown in Tables 4’s superscripts, with only a few exceptions in high-resource or large models.

To summarize, smaller models benefit from LM integration in all the cases tested here, suggesting that language models can raise performance where fine-tuning alone may not suffice.

#### 5.1.3 Large Language Model Integration Results

With respect to the integration of large language models with the fine-tuned Whisper models, the scores consistently show performance improvements across all tested languages and datasets, albeit with more modest gains compared to n-gram LM integrations. Table 5 displays the relative error reductions achieved by adding LLMs to the fine-tuned models.

- Table 5: The relative error reduction for fine-tuned models with a large language model compared with the fine-tuned results without any language model. ID datasets are listed at the top, and OOD datasets are listed at the bottom. Values indicating the biggest improvements are marked in bold, and values that improved over 5-gram LM results are underlined. Significance levels are indicated as follows: pa < 0.05, pb < 0.01, pc < 0.001, no-superscript meaning no significance.

|Language Dataset<br><br>|Tiny Base Small Medium Large L-V2 L-V3|
|---|---|
|Basque CV13 Galician CV13 Catalan CV13 Spanish CV13|+18%c +20%c +23%c +19%c +16%c +16%c +16%c +8%c +9%c +7%c +6%c +1%c +3%c +4%c<br><br>+6%c +6%c +5%c +1%c +3%c +4%c +3%c<br><br>+7%c +8%c +6%c +6%c +5%c +4%c +4%c<br><br><br>|
|Basque AhoMyTTS Basque SLR76 Galician Fleurs Galician SLR77 Catalan Fleurs Catalan SLR69 Spanish Fleurs Spanish MLS|+18%c +17%c +19%c +17%c +16%b +16%b +15%a +15%c +15%c +15%c +12%c +11%c +11%c +11%c<br><br>+9%c +8%c +7%c +5%c +4%c +3%c +2%c<br><br>+9%c +8%c +7%c +4%c +5%c +4%c +7%c<br><br>+9%c +9%c +7%c +5%c +4%c +5%c +4%c<br><br><br>+8%c +10%c +8%c +5%c +4%c +7%c +4%c +4%c +4%c +2%b +1%c +2%c +1%a 0%b +8%c +12%c +9%c +3%c +8%c +7%c +2%c<br><br>|

The improvements were more prominent in the Basque language, where the Small model size reduced WER by up to 23% in the ID dataset. This superior performance in Basque could be attributed to the Latxa model’s robust adaptation to the language, possibly improved by its extensive pre-training on a large and adequately curated Basque corpus. Although the overall improvements are lower than those seen with n-gram LMs, this initial integration of LLMs demonstrates promising directions for improving ASR systems for low-resource languages.

Expanding on this, the addition of LLMs improved performance across both ID and OOD datasets much more evenly than adding n-gram LMs did. In some cases, the improvements in OOD datasets slightly surpassed those in ID, indicating strong generalization ability. For instance, the Basque language models showed consistent improvements, with OOD datasets such as AhoMyTTS and OpenSLR-76 exhibiting improvements pretty close to those in the ID dataset.

As with the previous methods, the integration of large language models also results in statistically significant differences in performance (W = 0.0, p = 1.71 · 10−15). In this case, all the performance improvements achieved through LLM integration are statistically significant at least at the pa < 0.05 level, with most scores exhibiting very high statistical significance, indicated by pc < 0.001. This consistency in statistical significance emphasizes the robustness of LLM integration across different model configurations and datasets.

Moreover, in contrast to the first methods, where high-resource languages such as Spanish showed some divergent or negative trends with fine-tuning and n-gram LM addition, the integration of LLMs into these models resulted in consistent improvements. This suggests that LLMs, with their broader and deeper understanding of language, can help overcome some of the overfitting issues seen before in fine-tuning and n-gram LM integrations.

0 Method

FT

Robustness(ERER)

−10

FT+LM

FT+LLM

−20

−30

−40

−50

−60

−70

Tiny Base Small Medium Large L-V2 L-V3

Model Size

Figure 2: Effective robustness of RER by model size.

0 Method

FT

Robustness(ERER)

−10

FT+LM

FT+LLM

−20

−30

−40

−50

−60

−70

Basque Galician Catalan Spanish

Language

Figure 3: Effective robustness of RER by language.

- 5.1.4 Comparison of Method Robustness In this subsection, we further analyze the robustness of the various methods employed.

- Figure 2 provides a visual analysis of the effective robustness of RER (ERER) by model size. It reveals a trend where increasing model size tends to decrease robustness when only fine-tuning is applied, indicating a risk of dataset overfitting. This tendency toward decreasing robustness, even though much more subtle, is also present in the LM results. However, this trend does not persist with the addition of LLMs, which uniformly improve robustness across all model sizes.
- Figure 3 extends this analysis by comparing methods across different languages. It illustrates that while the transition from fine-tuning alone to integrating LMs does not uniformly improve robustness across languages, as in Basque and Galician, the addition of LLMs consistently elevates robustness, notably outperforming other methods. This suggests the greater effectiveness of LLMs, which appear to better mitigate language-specific challenges. Interestingly, Catalan, which demonstrated lower robustness during initial fine-tuning, shows significant improvement upon the addition of LMs.

Unlike the variations observed with fine-tuning and n-gram LMs, LLMs contribute to more stable improvements across different testing scenarios. The effective robustness analysis confirms that LLM integration not only does increase the baseline performance but does so consistently across varied environments and contexts. These insights highlight the considerable variability between languages and emphasize the importance of considering model size when developing robust ASR systems.

As we continue to refine the integration techniques and as the LLMs themselves evolve, we anticipate further improvements in ASR performance, particularly for languages that have traditionally been underrepresented in speech technology resources. The promising results in both ID and OOD settings highlight the capacity of LLMs to contribute to the robustness and accuracy of multilingual models.

#### 5.2 Results of Corpora Leakage Analysis

Regarding the results from our data leakage analysis presented in Table 6, sentence overlap in the ID datasets is notably high, exceeding 75% for most languages in the n-gram language models and ranging from 18% to 84% for the large language models. This high overlap in Common Voice datasets is partly due to the inclusion of sentences from Wikipedia in the recording prompts22, which is also a common source for corpora used in language model training. The Common Voice Catalan dataset presents a notable exception, showing lower overlap due to their proactive efforts to generate genuine, diverse sentences through collaborations with authors, publishers, and public entities, avoiding reliance on web-crawled data [103]. In contrast to the ID datasets, the OOD datasets exhibit minimal to no leakage (less than 15%), suggesting that the robustness of the LLMs is not merely a result of memorization. Curiously, using TTS datasets for ASR evaluations proves valuable, as they overlap less with today’s language model corpora. Overall, these findings reinforce our confidence that the observed improvements in OOD datasets are genuine and not influenced by memorized sentences.

#### 5.3 Ablation Study of the Impact of Evaluation Parameters

The ablation study focuses on understanding the impact of various evaluation parameters on the WER across different languages and Whisper model sizes.

22https://common-voice.github.io/community-playbook/sub_pages/text.html

- Table 6: Percentage of sentences from evaluation datasets found in the training corpora of n-gram and large language models.

|Language Dataset|LM Corpus LLM Corpus<br><br>|
|---|---|
|Basque CV13 Galician CV13 Catalan CV13 Spanish CV13|79.82% 36.34% 79.04% 36.40%<br><br>7.39% 17.53% 77.09% 83.87%<br><br>|
|Basque AhoMyTTS Basque SLR76 Galician Fleurs Galician SLR77 Catalan Fleurs Catalan SLR69 Spanish Fleurs Spanish MLS<br><br>|13.39% 11.19% 0.25% 0.27% 0.00% 0.00% 0.88% 0.81% 0.00% 0.00% 0.33% 0.92% 0.00% 0.00% 0.00% 4.23%|

50 Model Size

Tiny Base Small

0

RER(%)

Medium

Large

−50

Large-V2 Large-V3

−100

Beam Search Diacritics Timestamps Language Temperature

Parameter

- Figure 4: The averaged RER across different model sizes to study the impact of various evaluation parameters on the WER. Negative values indicate performance decreases when changing from our selected baseline.

- Figure 4 provides an overview of how the removal or change of specific parameters affects the WER across all languages and datasets. Each bar in the plot displays the mean RER value and its standard deviation, with negative values indicating a deterioration in performance and positive values indicating an improvement over the established baseline configuration: a beam size of 5, diacritics removed, timestamps excluded, the language provided to the model, and a temperature of 0. Detailed WER values used for these calculations can be found in Appendix A.3. The statistical test results revealed significant differences for most parameters, as follows:

The removal of the language specification has the biggest impact, worsening the WER by as much as 58% for larger model configurations, with significant statistical evidence (W = 0.0, p = 5.36 · 10−15). This substantial degradation highlights the importance of specifying the language to maintain high transcription accuracy and indicates the model’s difficulties in recognizing languages with lower resources. Likewise, disabling beam search results in an average decrease in WER ranging from 3% to 17%, with strong statistical support for these findings (W = 6.0, p = 2.12 · 10−15), underscoring the effectiveness of beam search in enhancing model precision across various sizes.

As expected, incorporating diacritics during evaluation consistently has a negative impact on WER, a change that is statistically significant (W = 0.0, p = 1.71 · 10−15). That said, disabling them might not always be a sensible decision, depending on specific use cases. Meanwhile, including timestamps shows a variable impact with an overall trend of slightly decreasing performance, depending on the model size and specific dataset conditions. This effect, too, is statistically significant (W = 857.0, p = 3.49 · 10−5). This variability indicates that including timestamp information may negatively impact the scores, especially for the smaller models, and it is advisable to disable it if not needed. For example, the Base model shows a particular sensitivity to this parameter. During the initial pre-training of the models, the timestamped examples were probably more scarce and may have biased the model slightly to produce worse results.

Interestingly, disabling the temperature scheduler results in negligible changes to the WER across any model size, with most of the RER values remaining at or near 0%. Indeed, this parameter does not statistically influence the WER (W = 680.5, p = 0.59), indicating an insignificant impact on performance across different model sizes and conditions. This is particularly relevant to our study as it supports our decision to disable this feature when integrating the language models. The absence of changes due to the temperature scheduler can be attributed to the nature of our datasets, which consist of short sentences. It is important to note that the impact of this parameter might vary in scenarios involving longer or more complex audio recordings, where temperature control could help mitigate repetitive or overly simplistic generation patterns often seen in longer sequences.

As a result, our chosen parameter configuration demonstrates the best overall performance, as deviations from this setup typically result in increased WERs.

### 6 Discussion

This study has demonstrated that the integration of statistical language models with the Whisper system can remarkably reduce word error rates across a variety of languages and datasets. Particularly in minority languages, these improvements suggest that while Whisper’s decoder is phonetically robust, it may not sufficiently capture the full range of linguistic characteristics necessary for optimal performance without additional help. This is evident from the substantial improvements observed when language models are integrated, highlighting a shortcoming in the ability of the model’s native decoder to function effectively as a language model, particularly at the grammatical levels of low-resource languages.

Furthermore, the variability in improvements, especially in high-resource languages and larger model architectures, points to a complex interaction between model size, language resources, and the parameters of the language model. The observed dependency of the language models’ effectiveness on the model size and dataset characteristics underscores the importance of the context in applying these improvements. Remarkably, the weighting α and β parameters for the language model integration appear to have some relationship with the language characteristics or the dataset resources used, as the good results obtained by their reuse imply. In particular, our evaluations indicate that the Large-V3 models consistently outperform other versions, with the Large-V2 also showing superior performance than Large-V1. Accordingly, we recommend the use of Large-V3 for those seeking the most accurate Whisper model, addressing the concerns raised in recent discussions23.

Alongside these findings, the integration of LLMs with Whisper, while giving more modest improvements, demonstrates a clear increase in robustness. This robustness, along with expected future improvements as LLMs are better adapted to specific languages, indicates that they could play an important role in evolving ASR systems towards greater linguistic generalizability and reliability. Currently, there seems to exist a trade-off between using traditional language models for performance and employing LLMs for stronger robustness. In environments where specific use cases dominate or where performance optimization is critical, traditional LMs may still be preferable. Conversely, for broader, more generalized applications, especially with high-resource languages or larger models where statistical LMs struggle to improve the results, LLMs start to give promising results. Moreover, the validation provided by the corpora leakage analysis further strengthens our confidence in these improvements. It ensures that the improvements we observe are due to effective model integration, not simply a result of overfitting or memorization.

On the other hand, the ablation study conducted as part of this research has also underlined the critical nature of parameter settings in model evaluation. Adjusting these parameters can lead to considerable variances in model performance, indicating Whisper’s sensitivity to configuration settings. This highlights an essential aspect of deploying Whisper in diverse linguistic settings: the configuration used to evaluate model performance needs to be consistent and well-considered to ensure reliability in the results.

### 7 Conclusion and Future Work

The findings from this study contribute to the aged but growing body of knowledge on the application of traditional language models in the field of speech recognition, particularly in enhancing the Whisper ASR models across diverse linguistic environments. We have shown that language model integration can improve performance, particularly in low-resource languages where Whisper’s native decoding power may be insufficient. Although, despite the overall positive trend, the effectiveness of these models varies for the largest models with high-resource languages, indicating a need for more customized approaches to language model integration.

23https://deepgram.com/learn/whisper-v3-results

The introduction of large language models into the Whisper framework has also highlighted a trade-off between immediate performance boosts and the robustness of model outputs. While n-gram language models offer better improvements, LLMs provide a more stable and consistent improvement across different datasets, including in-distribution and out-of-distribution scenarios. This suggests that LLMs may be particularly valuable in environments where generalization and stability are priorities. We also hypothesize that LLMs may shine in processing long-form content, leveraging their broader contextual understanding to capture more complex language patterns than n-gram models can provide. An obvious next step would be to investigate the combined use of n-gram and large language models to conceivably merge the capabilities of both methods.

Looking ahead, it is essential to explore the optimized use of language models further, mainly how these adjustments affect different languages and model sizes. Experimenting with a broader range of parameter and hyperparameter values adjusted to specific linguistic and acoustic characteristics may produce further improvements in model performance. Moreover, expanding the scope of these improvements to include other types of language models and decoding strategies, like end-to-end approaches, could provide deeper insights into the optimal configurations for various use cases. Finally, while this work focused on a limited set of languages and datasets due to time and resource constraints, evaluating these methods across a wider range of languages, out-of-domain conditions, and acoustic models remains an important next step to confirm broader applicability.

This research proposes an initial approach to using language models in modern speech recognition technologies. It involves bringing back traditional, widely tested integration techniques and applying them to new, advanced models. By continuing to refine these approaches and adapting the integration strategies as models evolve, it may be possible to improve further the accuracy and reliability of novel systems like Whisper across all languages, not just those well-represented in training data.

### 8 Acknowledgments

The authors express their gratitude to both the DIPC Supercomputing Center and the technological management body of the Basque Government, EJIE, for their technical and human support. Additionally, this research was partially financed by the Spanish Ministry for Digital Transformation and of Civil Service and the EU-funded NextGenerationEU Recovery, Transformation, and Resilience Plan (ILENIA project, 2022/TL22/00215335). We also express our gratitude to Carlos Dom´ınguez for his constant assistance, personal feedback, and ideas for further research, and to Andr´es Pi˜neiro-Mart´ın, whose insights unknowingly planted the initial seed for this project idea.

### References

- [1] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, Robust speech recognition via large-scale weak supervision, 2022. arXiv: 2212.04356 [eess.AS]. [Online]. Available: https://arxiv. org/abs/2212.04356.
- [2] A. Graves, A.-r. Mohamed, and G. Hinton, Speech recognition with deep recurrent neural networks, 2013. arXiv: 1303.5778 [cs.NE]. [Online]. Available: https://arxiv.org/abs/1303.5778.
- [3] W. Chan, N. Jaitly, Q. V. Le, and O. Vinyals, Listen, attend and spell, 2015. arXiv: 1508.01211 [cs.CL]. [Online]. Available: https://arxiv.org/abs/1508.01211.
- [4] Q. Zhang, H. Lu, H. Sak, A. Tripathi, E. McDermott, S. Koo, and S. Kumar, Transformer transducer: A streamable speech recognition model with transformer encoders and RNN-T loss, 2020. arXiv: 2002.02562 [eess.AS]. [Online]. Available: https://arxiv.org/abs/2002.02562.
- [5] P. Karmakar, S. W. Teng, and G. Lu, Thank you for attention: A survey on attention-based artificial neural networks for automatic speech recognition, 2021. arXiv: 2102.07259 [cs.SD]. [Online]. Available: https: //arxiv.org/abs/2102.07259.
- [6] G. Hinton et al., “Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups,” IEEE Signal Processing Magazine, vol. 29, no. 6, pp. 82–97, 2012. DOI: 10.1109/MSP. 2012.2205597.
- [7] Y. Zhang, M. Pezeshki, P. Brakel, S. Zhang, C. L. Y. Bengio, and A. Courville, Towards end-to-end speech recognition with deep convolutional neural networks, 2017. arXiv: 1701.02720 [cs.CL]. [Online]. Available: https://arxiv.org/abs/1701.02720.
- [8] S. Alharbi, M. Alrazgan, A. Alrashed, T. Alnomasi, R. Almojel, R. Alharbi, S. Alharbi, S. Alturki, F. Alshehri, and M. Almojil, “Automatic speech recognition: Systematic literature review,” IEEE Access, vol. 9, pp. 131858–131876, 2021. DOI: 10.1109/ACCESS.2021.3112535.

- [9] J. Li, Recent advances in end-to-end automatic speech recognition, 2022. arXiv: 2111.01690 [eess.AS]. [Online]. Available: https://arxiv.org/abs/2111.01690.
- [10] A. D´efossez, C. Caucheteux, J. Rapin, O. Kabeli, and J.-R. King, “Decoding speech perception from noninvasive brain recordings,” Nature Machine Intelligence, vol. 5, no. 10, pp. 1097–1107, Oct. 2023, ISSN: 2522-

5839. DOI: 10.1038/s42256-023-00714-5. [Online]. Available: https://doi.org/10.1038/s42256023-00714-5.

- [11] J. Millet, C. Caucheteux, P. Orhan, Y. Boubenec, A. Gramfort, E. Dunbar, C. Pallier, and J.-R. King, “Toward a realistic model of speech processing in the brain with self-supervised learning,” in Proc. 36th NeurIPS, ser. NIPS ’22, New Orleans, LA, USA: Curran Associates Inc., 2024, ISBN: 9781713871088.
- [12] R. Antonello, A. Vaidya, and A. G. Huth, Scaling laws for language encoding models in fMRI, 2024. arXiv: 2305.11863 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2305.11863.
- [13] M. Sato, K. Tomeoka, I. Horiguchi, K. Arulkumaran, R. Kanai, and S. Sasai, Scaling law in neural data: Noninvasive speech decoding with 175 hours of EEG data, 2024. arXiv: 2407.07595 [q-bio.NC]. [Online]. Available: https://arxiv.org/abs/2407.07595.
- [14] D. Amodei et al., “Deep Speech 2: End-to-end speech recognition in english and mandarin,” in Proc. ICML, ser. Proc. of Machine Learning Research, vol. 48, PMLR, Jun. 2016, pp. 173–182. [Online]. Available: https: //Proc..mlr.press/v48/amodei16.html.
- [15] A. Baevski, H. Zhou, A. Mohamed, and M. Auli, “Wav2vec 2.0: A framework for self-supervised learning of speech representations,” in Proc. NeurIPS, ser. NIPS ’20, Vancouver, BC, Canada: Curran Associates Inc., 2020, ISBN: 9781713829546.
- [16] S. Schneider, A. Baevski, R. Collobert, and M. Auli, “Wav2vec: Unsupervised pre-training for speech recognition,” in Interspeech, 2019, pp. 3465–3469. DOI: 10.21437/Interspeech.2019-1873.
- [17] A. Baevski, S. Schneider, and M. Auli, Vq-wav2vec: Self-supervised learning of discrete speech representations, 2020. arXiv: 1910.05453 [cs.CL]. [Online]. Available: https://arxiv.org/abs/1910.05453.
- [18] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, HuBERT: Selfsupervised speech representation learning by masked prediction of hidden units, 2021. arXiv: 2106.07447 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2106.07447.
- [19] S. Chen et al., “WavLM: Large-scale self-supervised pre-training for full stack speech processing,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1505–1518, Oct. 2022, ISSN: 1941-0484. DOI: 10.1109/jstsp.2022.3188113. [Online]. Available: http://dx.doi.org/10.1109/JSTSP.2022. 3188113.
- [20] A. Conneau, A. Baevski, R. Collobert, A. Mohamed, and M. Auli, “Unsupervised cross-lingual representation learning for speech recognition,” in Interspeech, 2021, pp. 2426–2430. DOI: 10.21437/Interspeech.2021329.
- [21] A. Babu et al., “XLS-R: Self-supervised cross-lingual speech representation learning at scale,” in Interspeech, 2022, pp. 2278–2282. DOI: 10.21437/Interspeech.2022-143.
- [22] S.-w. Yang et al., “SUPERB: Speech processing universal performance benchmark,” in Interspeech, 2021, pp. 1194–1198. DOI: 10.21437/Interspeech.2021-1775.
- [23] A. Mohamed et al., “Self-supervised speech representation learning: A review,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1179–1210, 2022. DOI: 10.1109/JSTSP.2022.3207050.
- [24] X. Chang et al., “An exploration of self-supervised pretrained representations for end-to-end speech recognition,” in Proc. ASRU, 2021, pp. 228–235. DOI: 10.1109/ASRU51503.2021.9688137.
- [25] Y. Peng, S. Arora, Y. Higuchi, Y. Ueda, S. Kumar, K. Ganesan, S. Dalmia, X. Chang, and S. Watanabe, A study on the integration of pre-trained SSL, ASR, LM and SLU models for spoken language understanding, 2022. arXiv: 2211.05869 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2211.05869.
- [26] Y.-A. Chung, Y. Zhang, W. Han, C.-C. Chiu, J. Qin, R. Pang, and Y. Wu, W2v-BERT: Combining contrastive learning and masked language modeling for self-supervised speech pre-training, 2021. arXiv: 2108.06209 [cs.LG]. [Online]. Available: https://arxiv.org/abs/2108.06209.
- [27] A. Baevski, W.-N. Hsu, A. Conneau, and M. Auli, Unsupervised speech recognition, 2022. arXiv: 2105.11084 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2105.11084.
- [28] A. H. Liu, W.-N. Hsu, M. Auli, and A. Baevski, Towards end-to-end unsupervised speech recognition, 2022. arXiv: 2204.02492 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2204.02492.
- [29] A. Narayanan, A. Misra, K. C. Sim, G. Pundak, A. Tripathi, M. Elfeky, P. Haghani, T. Strohman, and M. Bacchiani, “Toward domain-invariant speech recognition via large scale training,” in Proc. SLT, 2018, pp. 441–

##### 447. DOI: 10.1109/SLT.2018.8639610.

- [30] W. Chan, D. Park, C. Lee, Y. Zhang, Q. Le, and M. Norouzi, SpeechStew: Simply mix all available speech recognition data to train one large neural network, 2021. arXiv: 2104.02133 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2104.02133.
- [31] Y. Zhang et al., “BigSSL: Exploring the frontier of large-scale semi-supervised learning for automatic speech recognition,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1519–1532, Oct. 2022, ISSN: 1941-0484. DOI: 10.1109/jstsp.2022.3182537. [Online]. Available: http://dx.doi.org/10. 1109/JSTSP.2022.3182537.
- [32] B. Li, R. Pang, T. N. Sainath, A. Gulati, Y. Zhang, J. Qin, P. Haghani, W. R. Huang, M. Ma, and J. Bai, Scaling end-to-end models for large-scale multilingual ASR, 2021. arXiv: 2104.14830 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2104.14830.
- [33] Y. Zhang et al., Google USM: Scaling automatic speech recognition beyond 100 languages, 2023. arXiv: 2303.01037 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2303.01037.
- [34] S. Communication et al., SeamlessM4T: Massively multilingual & multimodal machine translation, 2023. arXiv: 2308.11596 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2308.11596.
- [35] S. Communication et al., Seamless: Multilingual expressive and streaming speech translation, 2023. arXiv: 2312.05187 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2312.05187.
- [36] D. Xu et al., Introducing semantics into speech encoders, 2022. arXiv: 2211.08402 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2211.08402.
- [37] D. Wang and T. F. Zheng, Transfer learning for speech and language processing, 2015. arXiv: 1511.06066 [cs.CL]. [Online]. Available: https://arxiv.org/abs/1511.06066.
- [38] S. Khare, A. Mittal, A. Diwan, S. Sarawagi, P. Jyothi, and S. Bharadwaj, “Low resource ASR: The surprising effectiveness of high resource transliteration,” in Interspeech, 2021, pp. 1529–1533. DOI: 10.21437/ Interspeech.2021-2062.
- [39] L. Meng, J. Xu, X. Tan, J. Wang, T. Qin, and B. Xu, MixSpeech: Data augmentation for low-resource automatic speech recognition, 2021. arXiv: 2102.12664 [cs.CL]. [Online]. Available: https://arxiv.org/ abs/2102.12664.
- [40] J. C. V´asquez-Correa, A. Alvarez, H. Arzelus, S. A. Moreno Acevedo, A. Gonz´´ alez-Docasal, and J. M. Mart´ınDo˜nas, “The Vicomtech speech transcription systems for the albayz´ın 2024 bilingual Basque-Spanish speech to text (bbs-s2t) challenge,” in IberSPEECH, 2024, pp. 305–309. DOI: 10.21437/IberSPEECH.2024-64.
- [41] Z. Yao, L. Guo, X. Yang, W. Kang, F. Kuang, Y. Yang, Z. Jin, L. Lin, and D. Povey, “Zipformer: A faster and better encoder for automatic speech recognition,” in Proc. ICLR, 2024. [Online]. Available: https:// openreview.net/forum?id=9WD9KwssyT.
- [42] Z.-H. Zhou, “A brief introduction to weakly supervised learning,” National Science Review, vol. 5, no. 1, pp. 44–53, Aug. 2017, ISSN: 2095-5138. DOI: 10.1093/nsr/nwx106. [Online]. Available: https://doi. org/10.1093/nsr/nwx106.
- [43] S. Gandhi, P. von Platen, and A. M. Rush, Distil-Whisper: Robust knowledge distillation via large-scale pseudo labelling, 2023. arXiv: 2311.00430 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2311. 00430.
- [44] H. Shao, B. Liu, W. Wang, X. Gong, and Y. Qian, DQ-Whisper: Joint distillation and quantization for efficient multilingual speech recognition, 2024. arXiv: 2305.10788 [cs.SD]. [Online]. Available: https://arxiv. org/abs/2305.10788.
- [45] Z. Zhou, J. Shin, L. Zhang, S. Gurudu, M. Gotway, and J. Liang, “Fine-tuning convolutional neural networks for biomedical image analysis: Actively and incrementally,” in Proc. CVPR, 2017, pp. 4761–4772. DOI: 10. 1109/CVPR.2017.506.
- [46] C. Sicard, V. Gillioz, and K. Pyszkowski, “Spaiche: Extending state-of-the-art ASR models to Swiss German dialects,” in Proc. SwissText, Association for Computational Linguistics, Jun. 2023, pp. 76–83. [Online]. Available: https://aclanthology.org/2023.swisstext-1.8/.
- [47] V. Timmel, C. Paonessa, R. Kakooee, M. Vogel, and D. Perruchoud, Fine-tuning Whisper on low-resource languages for real-world applications, 2024. arXiv: 2412.15726 [cs.CL]. [Online]. Available: https: //arxiv.org/abs/2412.15726.
- [48] Y. Liu, X. Yang, and D. Qu, “Exploration of Whisper fine-tuning strategies for low-resource ASR,” EURASIP Journal on Audio, Speech, and Music Processing, vol. 2024, no. 1, p. 29, Jun. 2024, ISSN: 1687-4722. DOI: 10.1186/s13636-024-00349-3. [Online]. Available: https://doi.org/10.1186/s13636-02400349-3.
- [49] W. Liu, Y. Qin, Z. Peng, and T. Lee, Sparsely shared LoRA on Whisper for child speech recognition, 2024. arXiv: 2309.11756 [eess.AS]. [Online]. Available: https://arxiv.org/abs/2309.11756.

- [50] P. Xie and K. Chen, MADGF: Multi-agent data generation framework, 2024. arXiv: 2310.17953 [cs.SD]. [Online]. Available: https://arxiv.org/abs/2310.17953.
- [51] S. Gandhi, P. von Platen, and A. M. Rush, ESB: A benchmark for multi-domain end-to-end speech recognition,

2022. arXiv: 2210.13352 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2210.13352.

- [52] A. Pi˜neiro-Mart´ın, C. Garc´ıa-Mateo, L. Docio-Fernandez, M. del Carmen L´opez-P´erez, and G. Rehm, “Weighted cross-entropy for low-resource languages in multilingual speech recognition,” in Interspeech, 2024, pp. 1235–1239. DOI: 10.21437/Interspeech.2024-734.
- [53] A. Pi˜neiro-Mart´ın, C. Garc´ıa-Mateo, L. Doc´ıo-Fern´andez, and M. del Carmen L´opez-P´erez, “Whisper meets FalAI: From speech recognition to end-to-end spoken language understanding,” in IberSPEECH, 2024, pp. 171–175. DOI: 10.21437/IberSPEECH.2024-35.
- [54] A. Gulati et al., “Conformer: Convolution-augmented transformer for speech recognition,” in Interspeech, 2020, pp. 5036–5040. DOI: 10.21437/Interspeech.2020-3015.
- [55] F. Gaspari, A. Gr¨utzner-Zahn, G. Rehm, O. Gallagher, M. Giagkou, S. Piperidis, and A. Way, “Digital language equality: Definition, metric, dashboard,” in European Language Equality: A Strategic Agenda for Digital Language Equality. Springer International Publishing, 2023, pp. 39–73, ISBN: 978-3-031-28819-7. DOI: 10.1007/978-3-031-28819-7_3. [Online]. Available: https://doi.org/10.1007/978-3-03128819-7_3.
- [56] P. Gerrand, “Catalan’s presence on the internet (1993–2018),” in The Rise of Catalan Identity: Social Commitment and Political Engagement in the Twentieth Century. Springer International Publishing, 2019, pp. 261– 270, ISBN: 978-3-030-18144-4. DOI: 10.1007/978-3-030-18144-4_17. [Online]. Available: https: //doi.org/10.1007/978-3-030-18144-4_17.
- [57] M. Villegas Montserrat, O. L´opez de Lacalle Lekuona, I. Baucells de la Pe˜na, and B. Calvo Figueras, “Transferencia de tareas basada en implicaci´on textual para la clasificaci´on de textos en catal´an en escenarios de pocos datos,” Procesamiento del lenguaje natural, vol. 71, pp. 165–177, 2023.
- [58] S. Watanabe et al., “ESPnet: End-to-end speech processing toolkit,” in Interspeech, 2018, pp. 2207–2211. DOI: 10.21437/Interspeech.2018-1456.
- [59] Y. Peng et al., Reproducing Whisper-style training using an open-source toolkit and publicly available data,

2023. arXiv: 2309.13876 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2309.13876.

- [60] Y. Peng et al., “OWSM v3.1: Better and faster open whisper-style speech models based on e-branchformer,” in Interspeech, 2024, pp. 352–356. DOI: 10.21437/Interspeech.2024-1194.
- [61] S. Zhang et al., OPT: Open pre-trained transformer language models, 2022. arXiv: 2205.01068 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2205.01068.
- [62] Z. Liu et al., LLM360: Towards fully transparent open-source LLMs, 2023. arXiv: 2312.06550 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2312.06550.
- [63] H. Touvron et al., LLaMA: Open and efficient foundation language models, 2023. arXiv: 2302 . 13971 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2302.13971.
- [64] H. Touvron et al., Llama 2: Open foundation and fine-tuned chat models, 2023. arXiv: 2307.09288 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2307.09288.
- [65] A. Grattafiori et al., The Llama 3 herd of models, 2024. arXiv: 2407.21783 [cs.AI]. [Online]. Available: https://arxiv.org/abs/2407.21783.
- [66] R. Ardila, M. Branson, K. Davis, M. Kohler, J. Meyer, M. Henretty, R. Morais, L. Saunders, F. Tyers, and G. Weber, “Common Voice: A massively-multilingual speech corpus,” eng, in Proc. LREC, ELRA, May 2020, pp. 4218–4222, ISBN: 979-10-95546-34-4. [Online]. Available: https://aclanthology.org/2020.lrec1.520/.
- [67] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. ICLR, 2019. [Online]. Available: https://openreview.net/forum?id=Bkg6RiCqY7.
- [68] A. Hannun et al., Deep Speech: Scaling up end-to-end speech recognition, 2014. arXiv: 1412.5567 [cs.CL]. [Online]. Available: https://arxiv.org/abs/1412.5567.
- [69] K. Heafield, “KenLM: Faster and smaller language model queries,” in Proc. WMT, Association for Computational Linguistics, Jul. 2011, pp. 187–197. [Online]. Available: https://www.aclweb.org/anthology/ W11-2123.
- [70] K. Heafield, I. Pouzyrevsky, J. H. Clark, and P. Koehn, “Scalable modified Kneser-Ney language model estimation,” in Proc. ACL (Volume 2: Short Papers), Association for Computational Linguistics, Aug. 2013, pp. 690–696. [Online]. Available: https://www.aclweb.org/anthology/P13-2121.

- [71] F. M. Tyers and J. Meyer, What shall we do with an hour of data? Speech recognition for the un- and underserved languages of Common Voice, 2021. arXiv: 2105.04674 [cs.CL]. [Online]. Available: https:// arxiv.org/abs/2105.04674.
- [72] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, “Optuna: A next-generation hyperparameter optimization framework,” in Proc. 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, Anchorage, AK, USA: Association for Computing Machinery, 2019, pp. 2623–2631, ISBN:

9781450362016. DOI: 10.1145/3292500.3330701. [Online]. Available: https://doi.org/10.1145/ 3292500.3330701.

- [73] D. Klakow and J. Peters, “Testing the correlation of word error rate and perplexity,” Speech Communication, vol. 38, no. 1, pp. 19–28, 2002, ISSN: 0167-6393. DOI: https://doi.org/10.1016/S01676393(01)00041-3. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S0167639301000413.
- [74] Y.-Y. Wang, A. Acero, and C. Chelba, “Is word error rate a good indicator for spoken language understanding accuracy,” in Proc. ASRU, 2003, pp. 577–582. DOI: 10.1109/ASRU.2003.1318504.
- [75] S. Watanabe, Tree-structured Parzen estimator: Understanding its algorithm components and their roles for better empirical performance, 2023. arXiv: 2304.11127 [cs.LG]. [Online]. Available: https://arxiv. org/abs/2304.11127.
- [76] J. Etxaniz, O. Sainz, N. Miguel, I. Aldabe, G. Rigau, E. Agirre, A. Ormazabal, M. Artetxe, and A. Soroa, “Latxa: An open language model and evaluation suite for Basque,” in Proc. ACL (Volume 1: Long Papers), Association for Computational Linguistics, Aug. 2024, pp. 14952–14972. DOI: 10.18653/v1/2024.acllong.799. [Online]. Available: https://aclanthology.org/2024.acl-long.799/.
- [77] P. Gamallo, P. Rodr´ıguez, I. de-Dios-Flores, S. Sotelo, S. Paniagua, D. Bardanca, J. R. Pichel, and M. Garcia, Open generative large language models for Galician, 2024. arXiv: 2406.13893 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2406.13893.
- [78] S. Da Dalt, J. Llop, I. Baucells, M. Pamies, Y. Xu, A. Gonzalez-Agirre, and M. Villegas, “FLOR: On the effectiveness of language adaptation,” in Proc. LREC-COLING, ELRA and ICCL, May 2024, pp. 7377–7388. [Online]. Available: https://aclanthology.org/2024.lrec-main.650/.
- [79] B. Workshop et al., BLOOM: A 176B-parameter open-access multilingual language model, 2023. arXiv: 2211.05100 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2211.05100.
- [80] J. Hoffmann et al., “Training compute-optimal large language models,” in Proc. NeurIPS, ser. NIPS ’22, New Orleans, LA, USA: Curran Associates Inc., 2022, ISBN: 9781713871088.
- [81] I. de-Dios-Flores, S. P. Su´arez, C. C. P´erez, D. B. Outeiri˜no, M. Garcia, and P. Gamallo, “CorpusNOS:´ A massive Galician corpus for training large language models,” in Proc. PROPOR Vol. 1, Association for Computational Lingustics, Mar. 2024, pp. 593–599. [Online]. Available: https://aclanthology.org/2024. propor-1.66/.
- [82] J. Palomar-Giner et al., “A CURATEd CATalog: Rethinking the extraction of pretraining corpora for midresourced languages,” in Proc. LREC-COLING, ELRA and ICCL, May 2024, pp. 335–349. [Online]. Available: https://aclanthology.org/2024.lrec-main.31/.
- [83] N. Ljubeˇsi´c and A. Toral, “CaWaC – a web corpus of Catalan and its application to language modeling and machine translation,” in Proc. LREC, ELRA, May 2014, pp. 1728–1732. [Online]. Available: https:// aclanthology.org/L14-1647/.
- [84] J. Armengol-Estap´e, C. P. Carrino, C. Rodriguez-Penagos, O. de Gibert Bonet, C. Armentano-Oller, A. Gonzalez-Agirre, M. Melero, and M. Villegas, Are multilingual models the best choice for moderately underresourced languages? a comprehensive assessment for catalan, 2021. arXiv: 2107.07903 [cs.CL].
- [85] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,” Journal of Machine Learning Research, vol. 21, no. 140, pp. 1–67, 2020. [Online]. Available: http://jmlr.org/papers/v21/20-074.html.
- [86] M. Guo, Z. Dai, D. Vrandecic, and R. Al-Rfou, “Wiki-40b: Multilingual language model dataset,” in Proc. LREC, 2020.
- [87] A. Guti´errez-Fandi˜no, J. Armengol-Estap´e, A. Gonzalez-Agirre, and M. Villegas, Spanish legalese language model and corpora, 2021. arXiv: 2110.12201 [cs.CL].
- [88] C. P. Carrino, J. Llop, M. P`amies, A. Guti´errez-Fandi˜no, J. Armengol-Estap´e, J. Silveira-Ocampo, A. Valencia, A. Gonzalez-Agirre, and M. Villegas, “Pretrained biomedical language models for clinical NLP in Spanish,” in Proc. 21st Workshop on Biomedical Language Processing, Association for Computational Linguistics, May 2022, pp. 193–199. DOI: 10.18653/v1/2022.bionlp-1.19. [Online]. Available: https: //aclanthology.org/2022.bionlp-1.19.

- [89] I. Sainz, D. Erro, E. Navas, I. Hern´aez, J. Sanchez, I. Saratxaga, and I. Odriozola, “Versatile speech databases for high quality synthesis for Basque,” in Proc. LREC, European Language Resources Association (ELRA), May 2012, pp. 3308–3312. [Online]. Available: https://aclanthology.org/L12-1014/.
- [90] O. Kjartansson, A. Gutkin, A. Butryna, I. Demirsahin, and C. Rivera, “Open-Source High Quality Speech Datasets for Basque, Catalan and Galician,” in Proc. SLTU and CCURL, European Language Resources association (ELRA), May 2020, pp. 21–27, ISBN: 979-10-95546-35-1. [Online]. Available: https://www. aclweb.org/anthology/2020.sltu-1.3.
- [91] A. Conneau, M. Ma, S. Khanuja, Y. Zhang, V. Axelrod, S. Dalmia, J. Riesa, C. Rivera, and A. Bapna, FLEURS: Few-shot learning evaluation of universal representations of speech, 2022. arXiv: 2205.12446 [cs.CL]. [Online]. Available: https://arxiv.org/abs/2205.12446.
- [92] V. Pratap, Q. Xu, A. Sriram, G. Synnaeve, and R. Collobert, “MLS: A large-scale multilingual dataset for speech research,” in Interspeech, 2020, pp. 2757–2761. DOI: 10.21437/Interspeech.2020-2826.
- [93] M. Artetxe, I. Aldabe, R. Agerri, O. Perez-de-Vi˜naspre, and A. Soroa, “Does corpus quality really matter for low-resource languages?” In Proc. EMNLP, Association for Computational Linguistics, Dec. 2022, pp. 7383–

7390. DOI: 10.18653/v1/2022.emnlp-main.499. [Online]. Available: https://aclanthology.org/ 2022.emnlp-main.499/.

- [94] R. Agerri, X. G´omez Guinovart, G. Rigau, and M. A. Solla Portela, “Developing new linguistic resources and tools for the Galician language,” in Proc. LREC, ELRA, May 2018. [Online]. Available: https:// aclanthology.org/L18-1367/.
- [95] M. Aulamo and J. Tiedemann, “The OPUS resource repository: An open package for creating parallel corpora and machine translation services,” in Proc. 22nd Nordic Conference on Computational Linguistics, Link¨oping University Electronic Press, Sep. 2019, pp. 389–394. [Online]. Available: https://aclanthology.org/ W19-6146/.
- [96] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems, vol. 30, Curran Associates, Inc., 2017. [Online]. Available: https://Proc..neurips.cc/paper_files/paper/2017/file/ 3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.
- [97] Z. Zhang, Z. Shao, X. Chen, K. Wang, and J. Qian, “Quasi-weighted least squares estimator for data reconciliation,” Computers & Chemical Engineering, vol. 34, no. 2, pp. 154–162, 2010, ISSN: 0098-1354. DOI: https://doi.org/10.1016/j.compchemeng.2009.09.007. [Online]. Available: https://www. sciencedirect.com/science/article/pii/S0098135409002464.
- [98] W. de Vries, M. Wieling, and M. Nissim, “DUMB: A benchmark for smart evaluation of Dutch models,” in Proc. EMNLP, Association for Computational Linguistics, Dec. 2023, pp. 7221–7241. DOI: 10.18653/v1/ 2023.emnlp-main.447. [Online]. Available: https://aclanthology.org/2023.emnlp-main.447/.
- [99] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei, Scaling laws for neural language models, 2020. arXiv: 2001.08361 [cs.LG]. [Online]. Available: https://arxiv.org/abs/2001.08361.
- [100] R. Taori, A. Dave, V. Shankar, N. Carlini, B. Recht, and L. Schmidt, “Measuring robustness to natural distribution shifts in image classification,” in Advances in Neural Information Processing Systems, vol. 33, Curran Associates, Inc., 2020, pp. 18583–18599. [Online]. Available: https://Proc..neurips.cc/paper_ files/paper/2020/file/d8330f857a17c53d217014ee776bfd50-Paper.pdf.
- [101] F. Wilcoxon, “Individual comparisons by ranking methods,” Biometrics Bulletin, vol. 1, no. 6, pp. 80–83, 1945, ISSN: 00994987. [Online]. Available: http://www.jstor.org/stable/3001968 (visited on 06/22/2024).
- [102] G. Santafe, I. Inza, and J. A. Lozano, “Dealing with the evaluation of supervised classification algorithms,” Artificial Intelligence Review, vol. 44, no. 4, pp. 467–508, Dec. 2015, ISSN: 1573-7462. DOI: 10.1007/ s10462-015-9433-y. [Online]. Available: https://doi.org/10.1007/s10462-015-9433-y.
- [103] C. Armentano-Oller, M. Marimon, and M. Villegas, “Becoming a high-resource language in speech: The Catalan case in the common voice corpus,” in Proc. LREC-COLING, ELRA and ICCL, May 2024, pp. 2142–

##### 2148. [Online]. Available: https://aclanthology.org/2024.lrec-main.193/.

### A Detailed Performance Results of Whisper Models

#### A.1 Vanilla, Fine-Tuned, and Language Model WER Scores

Table 7: WER scores for Whisper vanilla models without any modifications or fine-tuning.

|Language Dataset<br><br>|Tiny Base Small Medium Large L-V2 L-V3|
|---|---|
|Basque CV13 Galician CV13 Catalan CV13 Spanish CV13<br><br>|97.93 92.86 71.72 58.27 50.84 43.20 38.85<br><br>51.48 46.31 29.37 20.31 17.84 15.26 12.46<br>52.94 41.48 25.22 17.48 15.98 14.94 13.67 27.69 18.39 9.70 6.38 5.81 5.16 4.38<br>|
|Basque AhoMyTTS Basque SLR76 Galician Fleurs Galician SLR77 Catalan Fleurs Catalan SLR69 Spanish Fleurs Spanish MLS|91.65 84.91 67.66 56.81 48.20 41.16 36.67 93.55 91.82 72.59 61.72 55.01 48.87 45.92 48.04 40.53 24.94 16.48 14.87 12.41 10.06 52.12 46.75 31.76 24.26 21.28 19.77 17.20<br><br>40.74 27.68 14.20 8.57 7.42 6.18 5.68<br><br>41.26 29.95 16.21 10.38 9.09 7.90 7.86<br><br><br>27.70 22.02 17.46 15.78 15.44 15.15 15.01 17.33 11.45 6.58 4.62 4.34 3.71 2.92<br><br>|

Table 8: WER scores of fine-tuned models.

|Language Dataset|Tiny Base Small Medium Large L-V2 L-V3|
|---|---|
|Basque CV13 Galician CV13 Catalan CV13 Spanish CV13|30.26 24.82 17.72 14.03 12.11 11.26 10.52 22.69 16.71 9.94 6.44 6.27 5.50 4.55 14.55 11.85 8.78 5.40 4.69 4.52 5.56 16.61 11.90 7.32 4.92 4.69 4.43 4.46<br><br>|
|Basque AhoMyTTS Basque SLR76 Galician Fleurs Galician SLR77 Catalan Fleurs Catalan SLR69 Spanish Fleurs Spanish MLS|35.57 29.17 23.12 18.22 17.59 16.62 13.96 38.55 34.36 27.51 23.19 22.15 20.54 19.14 27.41 21.00 13.24 9.68 10.60 9.26 8.61<br><br>27.96 21.06 15.75 11.55 11.50 10.37 8.81<br><br>21.32 16.81 12.68 8.58 8.16 7.69 8.38 17.63 15.04 11.04 7.01 6.55 6.26 7.20<br><br>28.02 23.26 20.41 17.25 17.69 16.61 16.87<br><br><br>19.25 13.04 7.28 5.34 4.86 4.87 4.08|

- Table 9: WER scores of fine-tuned models with the n-gram language model.

|Language Dataset<br><br>|Tiny Base Small Medium Large L-V2 L-V3|
|---|---|
|Basque CV13 Galician CV13 Catalan CV13 Spanish CV13|18.99 13.70 8.84 7.07 6.17 5.88 5.15 15.34 10.16 5.97 3.90 4.05 3.68 2.80 12.54 10.18 7.77 5.41 4.49 4.19 5.16 12.94 8.99 5.50 3.77 3.78 3.57 3.67<br><br>|
|Basque AhoMyTTS Basque SLR76 Galician Fleurs Galician SLR77 Catalan Fleurs Catalan SLR69 Spanish Fleurs Spanish MLS<br><br>|32.38 20.69 17.77 12.09 12.35 11.49 9.74 30.59 26.76 21.38 18.01 17.66 16.43 14.83 23.59 17.94 12.21 9.95 10.82 9.38 8.94 22.20 16.79 12.62 9.69 9.70 8.80 7.59 18.56 14.82 11.21 8.13 8.23 7.65 7.83 14.35 12.38 8.96 6.09 5.71 5.45 6.32 25.93 22.14 22.01 17.15 17.76 16.67 17.01 18.50 11.81 7.44 5.81 5.09 5.69 4.49|

- Table 10: WER scores of fine-tuned models with the large language model.

|Language Dataset|Tiny Base Small Medium Large L-V2 L-V3|
|---|---|
|Basque CV13 Galician CV13 Catalan CV13 Spanish CV13<br><br>|24.73 19.96 13.60 11.40 10.15 9.43 8.87 20.77 15.20 9.24 6.08 6.21 5.31 4.37 13.70 11.14 8.37 5.34 4.55 4.35 5.38 15.41 10.98 6.89 4.65 4.48 4.26 4.30|
|Basque AhoMyTTS Basque SLR76 Galician Fleurs Galician SLR77 Catalan Fleurs Catalan SLR69 Spanish Fleurs Spanish MLS<br><br>|29.28 24.14 18.77 15.18 14.85 14.02 11.84 32.77 29.25 23.27 20.51 19.68 18.29 17.01 25.01 19.25 12.27 9.21 10.16 8.97 8.40<br><br>25.58 19.41 14.62 11.07 10.97 9.91 8.20 19.39 15.30 11.85 8.15 7.82 7.29 8.02<br><br>16.24 13.51 10.14 6.67 6.26 5.85 6.92<br><br>26.89 22.32 19.99 17.03 17.34 16.45 16.81<br><br>17.71 11.51 6.60 5.20 4.49 4.54 3.98<br><br><br>|

#### A.2 Language Model Integration for Vanilla Spanish Models

Scores in Table 11 reflect the performance of vanilla Whisper models integrated with an n-gram language model directly, without fine-tuning. The results generally are similar to those reported in Table 4, with some improvements and declines being more pronounced. For further details, Table 12 shows the WER scores of the vanilla models with the language model.

Table 11: The relative error reduction for vanilla Spanish models with a 5-gram language model compared with the original vanilla results without the language model. Negative reductions are highlighted in red.

|Language Dataset<br><br>|Tiny Base Small Medium Large L-V2 L-V3|
|---|---|
|Spanish CV13 Spanish Fleurs Spanish MLS<br><br>|+16% +20% +23% +22% +22% +20% +25% +6% +5% +1% -3% -4% -1% -7% +9% +3% -16% -63% -52% -29% -47%|

- Table 12: WER scores of vanilla with the n-gram language model for Spanish.

|Language Dataset<br><br>|Tiny Base Small Medium Large L-V2 L-V3|
|---|---|
|Spanish CV13 Spanish Fleurs Spanish MLS<br><br>|23.35 14.63 7.52 4.96 4.53 4.14 3.30 26.08 20.92 17.35 16.29 16.04 15.24 16.04 15.85 11.16 7.61 7.54 6.58 4.80 4.28|

#### A.3 Ablation Study WER Scores

- Table 13: WER scores in the ablation study results for the Basque language.

|Parameter Dataset<br><br>|Tiny Base Small Medium Large L-V2 L-V3|
|---|---|
|Baseline CV13 No Beam Size CV13 Diacritics CV13 Timestamps CV13 No Language CV13 Temp. Scheduler CV13|97.93 92.86 71.72 58.27 50.84 43.20 38.85<br><br>105.52 98.45 83.90 65.70 57.92 50.29 42.98<br><br>98.03 92.93 71.83 58.32 50.93 43.26 38.90<br><br><br>101.67 98.75 71.72 58.09 50.47 43.44 39.10 136.73 116.79 112.40 93.08 72.92 68.22 41.67 105.31 96.03 72.17 58.30 50.82 43.21 38.87<br><br>|
|Baseline AhoMyTTS<br><br>No Beam Size AhoMyTTS Diacritics AhoMyTTS Timestamps AhoMyTTS No Language AhoMyTTS Temp. Scheduler AhoMyTTS|91.65 84.91 67.66 56.81 48.20 41.16 36.67 103.68 92.90 73.78 64.11 54.78 45.27 39.59<br><br>91.97 85.64 68.14 57.67 48.90 41.94 37.27 100.66 87.81 67.56 56.40 47.81 41.33 36.54 118.92 99.55 87.24 73.77 56.52 46.43 37.13<br><br>93.63 84.94 67.66 56.81 48.20 41.16 36.67<br><br>|
|Baseline SLR76 No Beam Size SLR76 Diacritics SLR76 Timestamps SLR76 No Language SLR76 Temp. Scheduler SLR76|93.55 91.82 72.59 61.72 55.01 48.87 45.92<br><br>101.32 97.20 78.16 67.34 59.85 54.34 48.20 93.77 92.12 72.97 62.08 55.40 49.21 46.28 98.85 98.46 72.65 62.00 55.00 48.85 46.15<br><br>126.77 103.16 96.94 80.50 64.62 57.74 47.58 95.15 90.22 72.43 61.67 55.01 48.87 45.92<br><br>|

- Table 14: WER scores in the ablation study results for the Galician language.

#### Parameter Dataset Tiny Base Small Medium Large L-V2 L-V3

|Baseline CV13 No Beam Size CV13 Diacritics CV13 Timestamps CV13 No Language CV13 Temp. Scheduler CV13<br><br>|51.48 46.31 29.37 20.31 17.84 15.26 12.46 59.72 53.33 31.91 21.61 19.19 16.60 12.86 55.66 50.51 32.90 23.03 20.10 17.53 14.64 54.85 49.13 30.08 20.33 17.99 15.25 12.36 66.79 60.08 51.36 50.33 50.96 48.26 22.47 51.21 45.74 29.46 20.39 17.87 15.27 12.46|
|---|---|
|Baseline Fleurs No Beam Size Fleurs Diacritics Fleurs Timestamps Fleurs No Language Fleurs Temp. Scheduler Fleurs|48.04 40.53 24.94 16.48 14.87 12.41 10.06<br><br>54.33 43.68 26.97 17.54 15.73 13.25 10.26 51.77 44.26 28.14 18.96 17.03 14.47 12.10<br><br>49.03 41.45 25.08 16.68 14.72 12.38 10.06<br><br><br>58.05 51.41 43.72 38.18 40.49 33.35 13.72 46.36 40.17 24.95 16.48 14.87 12.41 10.06<br><br>|
|Baseline SLR77 No Beam Size SLR77 Diacritics SLR77 Timestamps SLR77 No Language SLR77 Temp. Scheduler SLR77<br><br>|52.12 46.75 31.76 24.26 21.28 19.77 17.20 59.15 51.74 34.43 25.48 22.44 21.16 17.30 55.53 50.22 34.48 26.41 22.93 21.40 18.81 54.46 49.21 31.98 24.06 21.06 19.60 16.79 64.57 59.62 47.66 41.57 46.69 39.78 21.73 51.88 46.67 31.77 24.27 21.28 19.77 17.20|

- Table 15: WER scores in the ablation study results for the Catalan language.

Parameter Dataset Tiny Base Small Medium Large L-V2 L-V3

|Baseline CV13 No Beam Size CV13 Diacritics CV13 Timestamps CV13 No Language CV13 Temp. Scheduler CV13<br><br>|52.94 41.48 25.22 17.48 15.98 14.94 13.67 65.85 50.82 29.59 19.29 17.33 16.43 14.33 54.13 42.53 25.94 18.06 16.46 15.35 14.11 57.82 45.92 26.03 17.80 15.86 15.28 13.75 74.81 58.13 35.63 27.18 23.88 24.51 16.59 52.03 41.10 25.23 17.45 15.98 14.94 13.67|
|---|---|
|Baseline Fleurs No Beam Size Fleurs Diacritics Fleurs Timestamps Fleurs No Language Fleurs Temp. Scheduler Fleurs<br><br>|40.74 27.68 14.20 8.57 7.42 6.18 5.68 51.04 34.07 16.52 9.22 8.01 6.77 5.95 42.08 28.80 14.81 8.97 7.77 6.45 5.96 42.90 29.30 14.51 8.39 7.37 6.27 5.67 57.32 40.61 26.86 15.05 15.36 10.24 6.97 40.16 27.17 14.20 8.57 7.40 6.18 5.68|
|Baseline SLR69 No Beam Size SLR69 Diacritics SLR69 Timestamps SLR69 No Language SLR69 Temp. Scheduler SLR69|41.26 29.95 16.21 10.38 9.09 7.90 7.86<br><br>51.58 35.80 18.55 11.11 9.53 8.63 8.02<br><br>42.66 31.08 17.06 10.96 9.67 8.34 8.39<br><br><br>42.76 31.13 16.28 10.34 8.89 8.03 7.79 54.55 39.15 18.86 13.71 11.38 10.41 8.93 41.25 29.53 16.21 10.38 9.09 7.90 7.83<br><br>|

- Table 16: WER scores in the ablation study results for the Spanish language.

#### Parameter Dataset Tiny Base Small Medium Large L-V2 L-V3

|Baseline CV13 No Beam Size CV13 Diacritics CV13 Timestamps CV13 No Language CV13 Temp. Scheduler CV13<br><br>|27.69 18.39 9.70 6.38 5.81 5.16 4.38 38.01 23.54 11.31 6.99 6.31 5.72 4.52<br>28.67 19.23 10.33 6.93 6.36 5.63 4.82 31.27 20.23 9.86 6.47 5.85 5.42 4.30<br>29.94 19.74 10.31 6.88 6.24 5.53 4.60 28.25 18.56 9.76 6.40 5.83 5.21 4.38<br>|
|---|---|
|Baseline Fleurs No Beam Size Fleurs Diacritics Fleurs Timestamps Fleurs No Language Fleurs Temp. Scheduler Fleurs<br><br>|27.70 22.02 17.46 15.78 15.44 15.15 15.01<br><br>32.04 24.05 18.03 15.87 15.56 15.32 14.95<br><br>28.30 22.44 17.74 15.97 15.63 15.33 15.14 31.25 79.01 17.64 15.71 15.06 14.59 14.94<br><br>33.19 27.37 18.77 15.78 15.44 15.15 15.02 27.70 22.02 17.46 15.78 15.44 15.15 14.96<br><br><br>|
|Baseline MLS No Beam Size MLS Diacritics MLS Timestamps MLS No Language MLS Temp. Scheduler MLS|17.33 11.45 6.58 4.62 4.34 3.71 2.92<br><br>20.73 13.26 7.24 5.22 4.48 3.70 2.85<br><br>18.81 12.67 7.53 5.41 5.08 4.42 3.86<br><br><br>18.20 11.96 7.05 4.48 4.22 3.93 2.80 17.53 11.53 6.77 4.88 4.79 3.90 2.93 17.33 11.45 6.58 4.62 4.21 3.56 2.89<br><br>|

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

Language

Basque

- 0
- 1
- 2
- 3
- 4

Galician

βParameter

Catalan Spanish

0 1 2 3 4 5

α Parameter

- Figure 5: LM optimization trials with better scores being more opaque.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 1 2 3 4 5

- 0
- 1
- 2
- 3
- 4

Language

Basque

Galician

Catalan Spanish

α Parameter

βParameter

Figure 6: LLM optimization trials with better scores being more opaque.

A.3.1 Optimization of Parameter Value Ranges in Language Model Integration

In this section, we present the optimization trials for the weighting parameters used to integrate the n-gram and large language models with the acoustic model results. Figures 5 and 6 illustrate these optimization processes by mapping the α and β values against the resulting WER for each trial in the validation split. The opacity of each marker represents its corresponding WER score: markers with a very low WER are more opaque, indicating better performance, while markers with a WER approaching 15% are nearly transparent, denoting lower performance.

In Figure 5, we observe that trials for lower-resource languages often present higher β values, indicating a greater reliance on sentence length adjustments within the n-gram model framework. These languages are more widely spread across the plot, suggesting a diverse range of effective parameter configurations. In contrast, higher-resource languages are clustered around lower parameter values, typically within the [0..1] range for both α and β, indicating less variability in their effective optimization settings.

- Figure 6 reveals an outstanding trend where the β parameter generally holds much lower values or appears to be nearly irrelevant. This pattern suggests that, due to their extensive context understanding, large language models do not benefit as much from modifications based on sentence length as their n-gram counterparts do. This implies that their intrinsic knowledge may already account for contextual length internally, rendering additional length-based adjustments unnecessary.

The results of these optimization trials show that some characteristics of the languages may influence the parameter values. We hope the observed value ranges will serve as a useful benchmark for refining the optimization process for additional languages in the future. However, since our study was limited to only four languages, our ability to generalize these findings across a broader linguistic spectrum remains constrained.

