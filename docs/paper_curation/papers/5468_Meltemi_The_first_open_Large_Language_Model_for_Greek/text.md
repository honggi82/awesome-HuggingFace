Meltemi: The ﬁrst open Large Language Model for Greek

arXiv:2407.20743v1[cs.CL]30Jul2024

Leon Voukoutis, Dimitris Roussis, Georgios Paraskevopoulos, Sokratis Soﬁanopoulos, Prokopis Prokopidis, Vassilis Papavasileiou, Athanasios Katsamanis, Stelios Piperidis, Vassilis Katsouros

Institute for Speech and Language Processing, Athena Research Center Artemidos 6 & Epidavrou, Athens, Greece vsk@athenarc.gr

Abstract

We describe the development and capabilities of Meltemi 7B, the ﬁrst open Large Language Model for the Greek language. Meltemi 7B has 7 billion parameters and is trained on a 40 billion token Greek corpus. For the development of Meltemi 7B, we adapt Mistral, by continuous pretraining on the Greek Corpus. Meltemi 7B contains upto-date information up to September 2023. Furthermore, we have translated and curated a Greek instruction corpus, which has been used for the instruction-tuning of a chat model, named Meltemi 7B Instruct. Special care has been given to the alignment and the removal of toxic content for the Meltemi 7B Instruct. The developed models are evaluated on a broad set of collected evaluation corpora, and examples of prompts and responses are presented. Both Meltemi 7B and Meltemi 7B Instruct are available1 under the Apache 2.0 license.

# 1 Introduction

Large Language Models (LLMs) have emerged as potent tools, that enable user-facing Artiﬁcial Intelligence (AI) applications. Their success lies in their ability to understand and generate natural language, which provides a natural and intuitive avenue for users to interact with them. Nevertheless, their training requires the dedication of massive amounts of resources, both in terms of computing power, as well as data collection and curation. Thus, LLM development progress has

[Figure 1]

1https://huggingface.co/ilsp/

favored major languages (e.g. English, Chinese), because their large user bases justify the signiﬁcant upfront investment.

There is a growing need to overcome this barrier to entry because the commoditization of AI is rapidly driving its evolution from a novelty to a core utility2; akin to the evolution of the internet during the late nineties. The impact of native, widely available, high-performant AI foundation models thus, is a mandate that should be incorporated into the strategic plan of local stakeholders.

Continual pretraining of LLMs has emerged as a signiﬁcant area of research aimed at exploiting existing models by extending the model to languages other than English. Aguila-7Bˇ (Aguilaˇ , 2023) is a 7B parameters model based on Falcon-7b (Almazrouei et al., 2023) and further trained with a mixture of Spanish, Catalan and English data. Using a similar strategy, LAION (2023) released LeoLM-7B, a German foundation model created by applying continual pretraining on Llama2-7B. Finally, similar work has been performed for Japanese with Levine et al. (2024) and TokyoTech-LLM (2024), both being foundation models created by applying continual pretraining on Mistral-7B.

In this paper, we present the development of Meltemi 7B and Meltemi 7B Instruct, the ﬁrst dedicated open LLM for the Greek language, and the corresponding instruction following model. Furthermore, we develop a comprehensive benchmark for LLM evaluation in the Greek language.

[Figure 2]

2https://www2.deloitte.com/us/ en/pages/consulting/articles/ the-future-of-ai.html

This benchmark covers a wide range of tasks, i.e., translation, dialogue, reasoning, and consists of translated and curated versions of existing test sets. The development of LLMs brings to light a necessity for a plan to be formulated for their maintenance and continuous provision. We initiate this discussion for the environmentally and economically sustainable deployment, continuous data integration, and updating of the developed models.

# 2 Our Method

LLMs require signiﬁcant amounts of data in order to learn how to generate ﬂuent texts as well as capture culture-speciﬁc knowledge. Therefore, adapting an LLM for the Greek language requires addressing its underrepresentation in datasets comprising trillions of tokens for other languages, such as RedPajama3. Notable approaches to tackle this challenge include the creation of synthetic data for a low-resource language via the translation of English corpora, as for example in Jais, a bilingual model pretrained for Arabic and English from scratch (Sengupta et al., 2023). There have also been several approaches to extend the pretraining of English-centric LLMs, such as the Sabiá models for Portuguese (Pires et al., 2023), BgGPT for Bulgarian (INSAIT, 2024).

Existing resources for the Greek language which have been processed for direct use in LLMs include CulturaX (Nguyen et al., 2023) and the HPLT corpus (de Gibert et al., 2024).

In the subsections that follow, we detail out strategy for creating and evaluating Meltemi:

- • Acquiring, collecting, ﬁltering, preprocessing, and deduplicating large amounts of Greek texts from various diverse sources.
- • Extending the original tokenizer with Greek tokens and ﬁne-tuning the newly initialized embeddings on a smaller subcorpus.
- • Continually pretraining Mistral 7B (Jiang et al., 2023) with a mix of Greek, English, and Parallel (English-Greek) data.
- • Imbuing the new foundation model with chat capabilities using the ORPO (Hong et al.,

2024) algorithm and translated preference data.

[Figure 3]

3https://github.com/togethercomputer/ RedPajama-Data

• Evaluating Meltemi and Meltemi 7B Instruct on Greek benchmarks, including novel, existing, and synthetically generated test sets.

2.1 Pretraining Data

As mentioned earlier, LLMs require signiﬁcant amounts of pretraining data in order to form a strong foundation model which can be then ﬁnetuned and aligned to human preferences. More data is not always better; data quality is also of utmost importance. In our work, we set the goal of sourcing as much Greek monolingual texts as possible, as well as to ensure that the pretraining data mix is of high quality, by using various ﬁltering and pre-processing procedures.

Our continual pretraining approach adapts an existing model for a new natural language using data which induce a strong distribution shift to the model, as it has already been trained with dissimilar data4. Taking into consideration this large distribution shift which could potentially lead to catastrophic forgetting, our approach involves re-warming and re-decaying the learning rate (Ibrahim et al., 2024), and utilizing both Greek and English monolingual data in the continual pretraining data mix.

Regarding the Greek monolingual data, we leveraged original Greek monolingual texts from various sources, including Wikipedia, ELRCSHARE (Lösch et al., 2021), EUR-LEX & MultiEUR-LEX (Chalkidis et al., 2019, 2021), MaCoCu (Bañón et al., 2022), CLARIN-EL (Gavriilidou et al., 2023), parliamentary proceedings (Erjavec et al., 2022), full texts (i.e., theses, dissertations, etc.) and abstracts from Greek academic repositories, as well as pre-ﬁltered resources originally compiled from the web, such as CulturaX (Nguyen et al., 2023).

In order to create an English monolingual corpus, we decided to use the English counterparts of Wikipedia and EUR-LEX, as well as English full texts and abstracts from academic records in Greek repositories. Additionally, we included the AutoMathText dataset (Zhang et al., 2024), which is a collection of math-related documents originating from web data, papers on arXiv, and code/notebooks on GitHub. These documents have undergone an automatic selection process using Qwen-72B (Bai et al., 2023) which resulted in

[Figure 4]

4Note that there is no published information regarding the data that Mistral 7B has been trained on, but it is evident that it has been predominantly trained on English texts.

a score between 0 and 1, reﬂecting the relevancy to the mathematical domain and the educational value of each document. In our work, we select all documents with a score greater than or equal to 0.70.

Furthermore, we also include augmented English-Greek translation data in the pretraining corpus, randomly sampled for each translation direction, i.e., EN-EL and EL-EN. Adding translation data to the continual pretraining corpus improves translation quality (Alves et al., 2024), while there also has been limited empirical evidence which suggests that LLMs address multilingualism by ﬁrst translating queries into English, process them using English and their multilingual knowledge, and then translate the responses back into the original language (Zhao et al., 2024). The parallel data corpus is a diverse mixture of selected high-quality datasets which we have acquired in previous work (Lösch et al., 2021; Roussis et al., 2022a,b), or are available in ELRC-SHARE & OPUS (Tiedemann, 2012).

In order to preprocess and clean the monolingual texts, we ﬁrst had to ensure that they shared a common format and, thus, we extracted textual content from documents that were stored in PDF and HTML formats, we converted each textual document (or datapoint) to a JSON object which includes the document’s textual content, as well as other metadata such as the text’s language, its number of words, its dataset of origin, any source URLs (if available), etc.

After extensive manual inspection of each source dataset, we decided to use different preprocessing and ﬁltering pipelines, as we observed large variations in data quality, readiness, and cleanliness, which were also connected with the way the dataset was originally acquired. For example, we removed documents from an inhouse list of -approximately 500- Greek blacklisted URLs, if metadata about the source URL were available, as in CulturaX. Also, in documents extracted from PDFs, we used regex patterns to remove textual parts with glued words or many consecutive single alphabetic characters, which are usually artifacts of the PDF extraction process.

Other rule-based ﬁlters that were used include:

- • Removal of small documents (<300 characters or <6 words)
- • Removal of documents with extremely long words (at least one with >60 characters)

- • Removal of documents with at least 2 bad words from a curated in-house list with Greek bad words and phrases.
- • Removal of documents containing the "lorem ipsum" substring.
- • Removal of noisy parallel sentences in the EN-EL parallel dataset using an in-house ﬁltering pipeline (Papavassiliou et al., 2018; Roussis and Papavassiliou, 2022).

Our ﬁltering pipeline also made use of more sophisticated tools, such as Monocleaner 5, which assigns a ﬂuency score in a text’s paragraphs using a 7-gram KenLM language model (SánchezCartagena et al., 2018). A threshold of 0.7 was set for the score given by Monocleaner for documents such as the ones extracted from PDFs of academic records. These scores were already available for MaCoCu, while 5-gram KenLM language models trained on Wikipedia were used in CulturaX (Nguyen et al., 2023). Likewise, parallel corpora were ﬁltered using a LASER (Artetxe and Schwenk, 2018, 2019) threshold of 1.06 where available, as well as with a threshold of 0.7 of the score calculated using BiCleaner AI (ZaragozaBernabeu et al., 2022) full model for EnglishGreek, which is a tool for detecting noisy sentence pairs in parallel corpora.

Finally, we deduplicated the pretraining corpus by utlizing the MinHashLSH near-deduplication method (Broder, 1997; Leskovec et al., 2020) which has been commonly used for deduplicating large datasets at scale (Penedo et al., 2023; Nguyen et al., 2023; Tokpanov et al., 2024; Penedo et al., 2024). In particular, we followed a 2-stage process:

- • Intra-dataset dedulication: First, we used MinHashLSH to identify duplicates within each dataset which was not known to be already deduplicated.
- • Cross-dataset dedulication: Then, we removed duplicates across the whole corpus which emerged after concatenating all of the datasets with the application of MinHashLSH.

[Figure 5]

5https://github.com/bitextor/ monocleaner

[Figure 6]

Subcorpus No. of Tokens Percentage Greek 43,383,244,502 79.5% English 10,538,413,259 19.3% Parallel 633,816,023 1.2%

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Total 54,555,473,784 100%

[Figure 19]

Table 1: Continual Pretraining Corpus.

[Figure 20]

Tokenizer Model

Fertility English Mistral 7B 32,000 6.80 1.49 Meltemi 7B 61,362 1.52 1.44

Vocab. Size

Fertility Greek

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Table 2: Tokenizer Statistics.

In particular, we adapted the implementation in the text-dedup6 repository for our case by using 5-gram subsets, a MinHash signature of 128, and a Jaccard similarity threshold of 0.8; a choice of parameters which has been based on the approach used for deduplicating CulturaX (Nguyen et al., 2023), although other choices have also been shown to work well.

In order to identify duplicates in the parallel corpora, we followed a different approach. Initially, we converted all sentence pairs to lowercase and removed digits and punctuation. Afterwards, we removed duplicate sentence pairs based on either the source or the target sentence, by ensuring no Greek or English sentence appears more than once in the combined parallel dataset (Roussis and Papavassiliou, 2022).

The resulting high-quality continual pretraining corpus consists of approximately 54.5 billion tokens as listed in Table 1.

2.2 Tokenizer and Embeddings Expansion

In this section, we present our experiments and methodology towards extending the original tokenizer of Mistral 7B to facilitate efﬁcient handling of Greek textual data.

The original tokenizer of Mistral 7B v0.1 (Jiang et al., 2023) is comprised of 32,000 tokens and is generally inefﬁcient for Greek texts. To ascertain this, we conducted preliminary tests on diverse Greek and English corpora (approximately 2M words each) and calculated the average amount of tokens produced by the tokenizer for each word, also known as fertility. The fertility of a tokenizer is an extremely important quantity, as higher values directly correlate with higher training and inference costs; a phenomenon observed for various underrepresented natural languages (Csaki et al., 2023).

As we can observe in Table 2, the original tokenizer of Mistral 7B produces approximately 4.4

[Figure 32]

6https://github.com/ChenghaoMou/ text-dedup/

more tokens per word for Greek texts. This would mean that, on average, 4.4 times more computing resources would be needed for training and inference. Therefore, increasing the vocabulary size of the original tokenizer proved to be a vital step in signiﬁcantly reducing training costs and increasing inference speed.

Following the expansion of the initial Mistral7B tokenizer to 61,362 tokens we expanded the embedding layer of the base model to match that number and then rounded the size up to a multiple of 8 for computational efﬁciency.

2.3 Foundation Model Training

Our pretraining procedure is split into two stages. We ﬁrst loosely train only the freshly initialized embeddings, keeping therest of the parameters frozen and subsequently unfreeze the rest of the model, warm restart it and continue with training all the parameters. For both stages of this procedure we utilized 8x NVIDIA H100 GPUs and Deepspeed Zero (Rajbhandari et al., 2020) for distributed training.

- In stage 1 the embeddings for each new token are ﬁrst initialized from the average of the embeddings of the tokens that would be retrieved by the initial tokenizer, when prompted to encode the given token. The same procedure was followed for the LM head respectively. Subsequently the added embeddings and LM head were trained on a small subset of the dataset with all other parameters frozen, to be brought in line with the rest of the weights. For this stage we trained for 2,500 steps with a maximum learning rate of 2.5e−4, a linear warmup over 250 steps, followed by a cosine decay to a minimum of 2.5e−5 and an effective batch size of 1.5M tokens. The optimizer used was the AdamW optimizer (Loshchilov and Hutter, 2017) with β1 = 0.9,β2 = 0.999,ǫ = 10−5 and gradient clipping set to 1.0.
- In stage 2 we warm restart the model, training over 24,800 steps with an effective batch size of 4.5M tokens. We set a maximum learning rate of 2.5e−5 that is reached after a linear warmup over

248 steps. We employ a cosine decay schedule of the learning rate down to a minimum of 2.5e−6, where it plateaus over the ﬁnal 10% of training steps. The optimizer used was the AdamW optimizer (Loshchilov and Hutter, 2017) with β1 = 0.9,β2 = 0.95,ǫ = 10−5 and gradient clipping set to 1.0.

- 2.4 Instruction Tuning

In order to create the Meltemi 7B Instruct version of our model, we used ORPO (Hong et al., 2024) to align our model with human preferences. The implementation is based on the TRL library from Huggingface (von Werra et al., 2020) and partially on the Alignment Handbook repository (Tunstall et al., 2023). In particular, we translated a mix of 12 preference datasets from Huggingface and carefully curated them by:

- • Using preference data in which the chosen response has a high rating (no ties).
- • Fixing formatting errors and ﬁltering out cases with translation inconsistencies, excessive Unicode.
- • Adding appropriate Greek system messages to each dataset.

Finally, we used the high quality translated Greek preference dataset comprising 89,730 preference triplets in Greek and 7,342 preference triplets in English from the initial datasets to mirror the initial training distribution, to create Meltemi 7B Instruct. Following common practice, we use special tokens to format the dataset for chat use, by denoting different roles: system messages, user messages, and assistant messages. We randomly apply custom Greek system messages to data without a pre-existing system role, while we also utilize RAG-speciﬁc, CoT-speciﬁc, mathematics-speciﬁc, and coding-speciﬁc system messages for the corresponding data.

Training on the 97,072 preference data ran for 2 epochs over 2 days on 4x NVIDIA RTX A6000 GPUs utilizing Deepspeed Zero stage 3, with a maximum learning rate of 1e−6, that is reached after 110 steps of linear warmup followed by cosine decay to 0.

- 2.5 Evaluation

Our evaluation suite includes Greek machinetranslated versions of established English bench-

marks for language understanding and reasoning, publicly available QA benchmarks targetting Greek (Belebele (ell) (8-shot), Greek Medical Multiple Choice QA (15-shot)) , and a novel benchmark with questions extracted from past medical exams as described in Table 3. We also evaluate the effect of the continual pretraining on the English capabilities of the model on the OpenLLMLeaderboard tasks. We implemented our evaluation procedure based on a fork of the lighteval framework (Fourrier et al., 2023). We follow the same experimental setup as the OpenLLMLeaderboard for the English tasks and their machine-translated Greek counterparts.

We report our evaluation results in Table 4. We see that Meltemi-7B enhances performance across all Greek test sets by a +20.2% average improvement. On the other side, the model performs worse than Mistral-7B for the English tasks, trailing by 6%. The difference in style between Greek and English data the model was initially trained on and the capacity of a 7B model, has impacted the performance in English tasks. This issue is partially ﬁxed after the models alignment with human preferences.

The same tendency has been observed for similar efforts in other languages. LAION (2023) report that LeoLM-7B achieved an average improvement of +4.8% for German benchmarks compared to the base Llama-2 model, while it achieved lower average scores by -2.6% on English benchmarks. Swallow-MS-7b-v0.1 (TokyoTech-LLM, 2024), has shown an average improvement of +8% on Japanese benchmarks versus its base model, and lower average scores of -5.3% on English benchmarks.

# 3 Discussion and Conclusions

In this paper we have extended Mistral-7B using continuous pretraining to create Meltemi-7B, the ﬁrst open Large Language Model for the Greek language. Meltemi was further tuned on instruction data using state-of-the-art preference optimization techniques, yielding the chat model Meltemi 7B Instruct. For evaluation we have created a benchmark suite in Greek by translating and curating a set of popular English benchmarks and including a novel dataset targeted towards medical question answering. Our evaluation indicates that Meltemi 7B Instruct achieves state-of-the-art results, indicating a 20.2% absolute performance

[Figure 33]

Name # Ex. Description ARC Greek 7.78K MT of ARC (Clark et al., 2018), a dataset of science exam questions

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

(with typically four answer options) partitioned into a Challenge and an Easy Set of 2.6K and 5.2 questions.

Truthful QA Greek 817 Edited MT of Truthful QA (Lin et al., 2022), a dataset of questions that are crafted so that some humans would answer wrongly due to a false belief or misconception.

[Figure 39]

[Figure 40]

HellaSwag Greek 59.8K MT of the HellaSwag dataset (Zellers et al., 2019) for commonsense NLI.

[Figure 41]

[Figure 42]

MMLU Greek 15.9K MT of the MMLU dataset (Hendrycks et al., 2021) of multiplechoice questions from 57 tasks including elementary mathematics, history, computer science, law, etc. Belebele (ell) 900 The Greek part of Belebele (Bandarkar et al., 2023), a multiplechoice machine reading comprehension dataset covering 122 language variants.

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Greek Medical Multiple Choice QA

2.03K Multiple choice questions extracted from past medical exams of the Greek National Acadenic Recognition and Information Center available at https://www.doatap.gr.

[Figure 47]

[Figure 48]

[Figure 49]

Table 3: Greek evaluation datasets

increase for the Greek language averaged across metrics and datasets over Mistral 7B, with a moderate 6% drop for the English benchmarks.

We hope that this effort furthers the discussion about the need for open-source LLMs for mid and low-resource languages, which take into account cultural aspects, local laws and customs. We acknowledge that this requires a signiﬁcant economic and environmental investment. For reference, the development of Meltemi 7B required an energy consumption of 2300 kWh. While continual pretraining helps to reduce the cost of extending existing models to new languages and domains, we want to further investigate techniques for efﬁcient model adaptation. Furthermore, in the future we want to explore the adaptation of larger models for Greek, with a focus on sustainability, and also explore multimodality, thus creating a suite of open and multimodal models for the Greek language.

# Acknowledgements

The authors wish to thank AWS and GRNET, especially Nikiforos Botis and Panos Louridas, for their ongoing support and helping us attain the required training infrastructure. We also acknowledge the support of the Luleå University of Technology for providing computational resources, thus supporting part of our experiments. Many thanks to Olga Yannoutsou for her help in

the translation of one of the evaluation datasets. Special thanks to all members of the Institute for Language and Speech Processing, Athena RC who participated in the internal evaluations and especially to Dimitris Galanis, Maria Giagkou, Sotiris Kotitsas and Dimitris Pappas for their valuable advice. Finally we are grateful to our colleagues in Xanthi who provided us access to four NVIDIA RTX A6000 GPUs for internal validation.

# References

Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, Mérouane Debbah, Étienne Gofﬁnet, Daniel Hesslow, Julien Launay, Quentin Malartic, Daniele Mazzotta, Badreddine Noune, Baptiste Pannier, and Guilherme Penedo. 2023. The Falcon Series of Open Language Models. arXiv e-prints, page arXiv:2311.16867.

Duarte M Alves, José Pombal, Nuno M Guerreiro, Pedro H Martins, João Alves, Amin Farajian, Ben Peters, Ricardo Rei, Patrick Fernandes, Sweta Agrawal, et al. 2024. Tower: An open multilingual large language model for translation-related tasks. arXiv preprint arXiv:2402.17733.

Mikel Artetxe and Holger Schwenk. 2018. Margin-based parallel corpus mining with mul-

[Figure 50]

Models

[Figure 51]

[Figure 52]

Task Mistral-7B-v0.1 Meltemi-7B Meltemi-7B-Instruct ARC-C Greek 27.22 47.17 40.8

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

TruthfulQA:MC2 Greek 44.93 45.19 53.8

[Figure 60]

[Figure 61]

[Figure 62]

HellaSwag Greek 35.20 65.75 63.7 MMLU Greek 28.35 42.45 45.9 Belebele (ell) 35.77 68.66 75.5

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Greek Medical MC QA 27.77 48.12 48.0 Average Greek 33.20 52.89 54.6

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

ARC-C 59.98 54.26 56.6 TruthfulQA:MC2 42.15 40.60 51.2

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

HellaSwag 83.31 79.60 78.3

[Figure 86]

[Figure 87]

[Figure 88]

MMLU 64.16 56.86 57.0 Winogrande 78.37 73.16 70.2

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

GSM8K (pass@1) 34.50 22.13 32.8 Average English 60.41 54.43 57.7

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

Table 4: Evaluation Results.

tilingual sentence embeddings. arXiv preprint arXiv:1811.01136.

Mikel Artetxe and Holger Schwenk. 2019. Massively multilingual sentence embeddings for zero-shot cross-lingual transfer and beyond. Transactions of the association for computational linguistics, 7:597–610.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe, Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke Zettlemoyer, and Madian Khabsa. 2023. The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants. arXiv:2308.16884.

Marta Bañón, Miquel Esplà-Gomis, Mikel L. Forcada, Cristian García-Romero, Taja Kuzman, Nikola Ljubeši´c, Rik van Noord, Leopoldo Pla Sempere, Gema Ramírez-Sánchez, Peter Rupnik, Vít Suchomel, Antonio Toral, Tobias van der Werff, and Jaume Zaragoza. 2022. MaCoCu: Massive collection and curation of monolingual and bilingual data: focus on under-resourced languages. In Proceedings of the 23rd Annual Conference of the European Association for Machine Translation, pages 303–

304, Ghent, Belgium. European Association for Machine Translation.

A Broder. 1997. On the resemblance and containment of documents. In Proceedings of the Compression and Complexity of Sequences 1997, page 21.

Ilias Chalkidis, Manos Fergadiotis, and Ion Androutsopoulos. 2021. Multieurlex – a multi-lingual and multi-label legal document classiﬁcation dataset for zero-shot cross-lingual transfer. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.

Ilias Chalkidis, Manos Fergadiotis, Prodromos Malakasiotis, and Ion Androutsopoulos. 2019. Large-scale multi-label text classiﬁcation on EU legislation. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 6314–6322, Florence, Italy. Association for Computational Linguistics.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv:1803.05457v1.

Zoltan Csaki, Pian Pawakapan, Urmish Thakker, and Qiantong Xu. 2023. Efﬁciently adapting pretrained language models to new languages. arXiv preprint arXiv:2311.05741.

Tomaž Erjavec, Maciej Ogrodniczuk, Petya Osenova, Nikola Ljubeši´c, Kiril Simov, Andrej Panˇcur, Michał Rudolf, Matyáš Kopp, Starkaður Barkarson, Steinþór Steingrímsson, Ça˘grı Çöltekin, Jesse de Does, Katrien Depuydt, Tommaso Agnoloni, Giulia Venturi, María Calzada Pérez, Luciana D. de Macedo, Costanza Navarretta, Giancarlo Luxardo, Matthew Coole, Paul Rayson, Vaidas Morkeviˇcius, Tomas Krilaviˇcius, Roberts Darundeﬁnedis, Orsolya Ring, Ruben van Heusden, Maarten Marx, and Darja Fišer. 2022. The ParlaMint corpora of parliamentary proceedings. Lang. Resour. Eval., 57(1):415–448.

Clémentine Fourrier, Nathan Habib, Thomas Wolf, and Lewis Tunstall. 2023. Lighteval: A lightweight framework for llm evaluation.

Maria Gavriilidou, Stelios Piperidis, Dimitrios Galanis, Kanella Pouli, Penny Labropoulou, Juli Bakagianni, Iro Tsiouli, Miltos Deligiannis, Athanasia Kolovou, Dimitris Gkoumas, Leon Voukoutis, and Katerina Gkirtzou. 2023. The CLARIN:EL infrastructure: Platform, Portal, K-Centre. In Selected papers from the CLARIN Annual Conference 2023.

Ona de Gibert, Graeme Nail, Nikolay Arefyev, Marta Bañón, Jelmer van der Linde, Shaoxiong Ji, Jaume Zaragoza-Bernabeu, Mikko Aulamo, Gema Ramírez-Sánchez, Andrey Kutuzov, et al. 2024. A New Massive Multilingual Dataset for High-Performance Language Technologies. arXiv:2403.14009.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Jiwoo Hong, Noah Lee, and James Thorne. 2024. Reference-free monolithic preference optimization with odds ratio. arXiv preprint arXiv:2403.07691.

Adam Ibrahim, Benjamin Thérien, Kshitij Gupta, Mats L Richter, Quentin Anthony, Timothée Lesort, Eugene Belilovsky, and Irina Rish. 2024. Simple and scalable strategies to continually pre-train large language models. arXiv preprint arXiv:2403.08763.

INSAIT. 2024. BgGPT-7B, a Bulgarian language model. https://huggingface.co/ tokyotech-llm/Swallow-MS-7b-v0.

1. Accessed: (12 July 2024).

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

LAION. 2023. LeoLM: Igniting GermanLanguage LLM Research. https:// laion.ai/blog/leo-lm/. Accessed: (12 July 2024).

Jure Leskovec, Anand Rajaraman, and Jeffrey David Ullman. 2020. Mining of massive data sets. Cambridge university press.

Aaron Levine, Connie Huang, Chenguang Wang, Eduardo Batista, Ewa Szymanska, Hongyi Ding, Hou Wei Chou, Jean-François Pessiot, Johanes Effendi, Justin Chiu, Kai Torben Ohlhus, Karan Chopra, Keiji Shinzato, Koji Murakami, Lee Xiong, Lei Chen, Maki Kubota, Maksim Tkachenko, Miroku Lee, Naoki Takahashi, Prathyusha Jwalapuram, Ryutaro Tatsushima, Saurabh Jain, Sunil Kumar Yadav, Ting Cai, Wei-Te Chen, Yandi Xia, Yuki Nakayama, and Yutaka Higashiyama. 2024. Rakutenai-7b: Extending large language models for japanese.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. TruthfulQA: Measuring how models mimic human falsehoods. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3214–3252, Dublin, Ireland. Association for Computational Linguistics.

Andrea Lösch, Valérie Mapelli, Khalid Choukri, Maria Giagkou, Stelios Piperidis, Prokopis Prokopidis, Vassilis Papavassiliou, Miltos Deligiannis, Aivars Berzins, Andrejs Vasiljevs, et al. 2021. Collection and Curation of Language Data within the European Language Resource Coordination (ELRC). In Qurator.

Ilya Loshchilov and Frank Hutter. 2017. Fixing weight decay regularization in adam. CoRR, abs/1711.05101.

Thuat Nguyen, Chien Van Nguyen, Viet Dac Lai, Hieu Man, Nghia Trung Ngo, Franck Dernoncourt, Ryan A Rossi, and Thien Huu Nguyen. 2023. CulturaX: A cleaned, enormous, and multilingual dataset for large language models in 167 languages. arXiv preprint arXiv:2309.09400.

Vassilis Papavassiliou, Sokratis Soﬁanopoulos, Prokopis Prokopidis, and Stelios Piperidis. 2018. The ILSP/ARC submission to the WMT 2018 parallel corpus ﬁltering shared task. In Proceedings of the Third Conference on Machine Translation: Shared Task Papers, pages 928–933, Belgium, Brussels. Association for Computational Linguistics.

Guilherme Penedo, Hynek Kydlíˇcek, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf, et al. 2024. The ﬁneweb datasets: Decanting the web for the ﬁnest text data at scale. arXiv preprint arXiv:2406.17557.

Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Hamza Alobeidli, Alessandro Cappelli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. 2023. The reﬁnedweb dataset for falcon llm: Outperforming curated corpora with web data only. Advances in Neural Information Processing Systems, 36:79155–79172.

Ramon Pires, Hugo Abonizio, Thales Sales Almeida, and Rodrigo Nogueira. 2023. Sabiá: Portuguese large language models. In Intelligent Systems, pages 226–240, Cham. Springer Nature Switzerland.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. 2020. Zero: memory optimizations toward training trillion parameter models. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, SC ’20. IEEE Press.

Dimitrios Roussis and Vassilis Papavassiliou. 2022. The ARC-NKUA submission for the English-Ukrainian general machine translation shared task at WMT22. In Proceedings of the Seventh Conference on Machine Translation (WMT), pages 358–365, Abu Dhabi, United

Arab Emirates (Hybrid). Association for Computational Linguistics.

Dimitrios Roussis, Vassilis Papavassiliou, Prokopis Prokopidis, Stelios Piperidis, and Vassilis Katsouros. 2022a. SciPar: A collection of parallel corpora from scientiﬁc abstracts. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 2652–2657, Marseille, France. European Language Resources Association.

Dimitrios Roussis, Vassilis Papavassiliou, Sokratis Soﬁanopoulos, Prokopis Prokopidis, and Stelios Piperidis. 2022b. Constructing parallel corpora from COVID-19 news using MediSys metadata. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 1068–1072, Marseille, France. European Language Resources Association.

Víctor M. Sánchez-Cartagena, Marta Bañón, Sergio Ortiz-Rojas, and Gema Ramírez-Sánchez. 2018. Prompsit’s submission to WMT 2018 Parallel Corpus Filtering shared task. In Proceedings of the Third Conference on Machine Translation, Volume 2: Shared Task Papers, Brussels, Belgium. Association for Computational Linguistics.

Neha Sengupta, Sunil Kumar Sahu, Bokang Jia, Satheesh Katipomu, Haonan Li, Fajri Koto, Osama Mohammed Afzal, Samta Kamboj, Onkar Pandit, Rahul Pal, et al. 2023. Jais and Jais-chat: Arabic-Centric Foundation and Instruction-Tuned Open Generative Large Language Models. arXiv preprint arXiv:2308.16149.

Jörg Tiedemann. 2012. Parallel data, tools and interfaces in OPUS. In Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC’12), pages 2214–2218, Istanbul, Turkey. European Language Resources Association (ELRA).

Yury Tokpanov, Beren Millidge, Paolo Glorioso, Jonathan Pilault, Adam Ibrahim, James Whittington, and Quentin Anthony. 2024. Zyda: A 1.3 t dataset for open language modeling. arXiv preprint arXiv:2406.01981.

TokyoTech-LLM. 2024. The Swallow-MS-7bv0.1 model. https://huggingface.

co/INSAIT-Institute/ BgGPT-7B-Instruct-v0.2. Accessed: (12 July 2024).

Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Shengyi Huang, Kashif Rasul, Alexander M. Rush, and Thomas Wolf. 2023. The Alignment Handbook. https://github.com/huggingface/ alignment-handbook.

Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, and Shengyi Huang. 2020. TRL: Transformer Reinforcement Learning. https://github.com/huggingface/ trl.

Jaume Zaragoza-Bernabeu, Gema RamírezSánchez, Marta Bañón, and Sergio Ortiz Rojas. 2022. Bicleaner AI: Bicleaner goes neural. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 824–831, Marseille, France. European Language Resources Association.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. HellaSwag: Can a machine really ﬁnish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4791–4800, Florence, Italy. Association for Computational Linguistics.

Yifan Zhang, Yifan Luo, Yang Yuan, and Andrew Chi-Chih Yao. 2024. Automathtext: Autonomous data selection with language models for mathematical texts. arXiv preprint arXiv:2402.07625.

Yiran Zhao, Wenxuan Zhang, Guizhen Chen, Kenji Kawaguchi, and Lidong Bing. 2024. How do large language models handle multilingualism? arXiv preprint arXiv:2402.18815.

Aguila.ˇ 2023. Introducing Aguila,ˇ a new open-source LLM for Spanish and Catalan. https://huggingface.co/ projecte-aina/aguila-7b. Accessed: (12 July 2024).

# A Examples of queries and responses

User: Πώς πιστεύεις θα είναι η ζωή σε 100 χρόνια;

Meltemi: Ως μοντέλο γλώσσας ΑΙ, δεν μπορώ να προβλέψω το μέλλον με βεβαιότητα ή προκατάληψη. Ωστόσο, εδώ είναι μερικές πιθανές προβλέψεις που βασίζονται στο τρέχον επίπεδο τεχνολογικής προόδου και στην κοινωνική ανάπτυξη:

Η τεχνητή νοημοσύνη μπορεί να γίνει πιο προηγμένη και ενσωματωμένη στην καθημερινή μας ζωή. Οι εικονικοί βοηθοί και οι chatbot μπορούν ενδεχομένως να αντικαταστήσουν πολλές εργασίες ρουτίνας και ακόμη και ορισμένες θέσεις εργασίας. Μπορεί επίσης να δούμε την εμφάνιση αυτόνομων μηχανών που μπορούν να εκτελούν πολύπλοκες λειτουργίες σε βιομηχανίες όπως η κατασκευή, η γεωργία και οι μεταφορές...

User: Ποιος ήταν ο πρώτος που πραγματοποίησε πρακτικά, το 1803, την πρόωση των πλοίων με ατμοκίνηση;

Meltemi: Ο πρώτος που πραγματοποίησε πρακτικά την προώση των πλοίων με ατμοκίνηση ήταν ο Ρόμπερτ Φούλτον.

User: Τα εμβόλια-υπομονάδες περιέχουν α. πρωτεΐνες με αντιγονική δράση. β. γυμνό DNA του μικροοργανισμού. γ. γενετικά τροποποιημένα βακτήρια. δ. αδρανοποιημένους ιούς.

Meltemi: α. User: Ποιος ήταν ο πρώτος που πραγματοπο-

ίησε πρακτικά, το 1803, την πρόωση των πλοίων με ατμοκίνηση;

Meltemi: Ο πρώτος που πραγματοποίησε πρακτικά την προώση των πλοίων με ατμοκίνηση ήταν ο Ρόμπερτ Φούλτον.

User: Από πόσους στίχους αποτελείται ο Ερωτόκριτος;

Meltemi: Ο Ερωτόκριτος αποτελείται από 10.012 στίχους.

User: Τι είναι τα συνελικτικά νευρωνικά δίκτυα; Meltemi: Τα Συνελικτικά Νευρωνικά Δίκτυα

(CNN) είναι ένας τύπος αλγόριθμου μηχανικής μάθησης που είναι ιδιαίτερα κατάλληλος για εργασίες αναγνώρισης εικόνας και

βίντεο. Είναι ένα υποσύνολο των νευρωνικών δικτύων που χρησιμοποιούν φίλτρα και στρώματα συγκέντρωσης για να εξάγουν χαρακτηριστικά από εικόνες ή ακολουθίες εισόδου βίντεο, καθιστώντας τα εξαιρετικά χρήσιμα στην ταξινόμηση εικόνων, τον εντοπισμό αντικειμένων, την ανίχνευση προσώπου, ακόμη και τη συμπίεση δεδομένων.

User: Ποιος ήταν ο Κατσουσίκα Χοκουσάι;

Meltemi: Ο Κατσουσίκα Χοκουσάι ήταν Ιάπωνας καλλιτέχνης που έζησε από το 1760 έως το 1849 και θεωρείται ένας από τους κορυφαίους καλλιτέχνες της εποχής ΄Εντο στην Ιαπωνία. Είναι γνωστός για τις ξυλογραφίες του, συμπεριλαμβανομένης της διάσημης σειράς ‘Τριάντα ΄Εξι ΄Οψεις του Φούτζι’.

