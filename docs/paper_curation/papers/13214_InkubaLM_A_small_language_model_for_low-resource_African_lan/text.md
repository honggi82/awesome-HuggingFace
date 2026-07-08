# arXiv:2408.17024v2[cs.CL]3Sep2024

## InkubaLM: A small language model for low-resource African languages

### Atnafu Lambebo Tonja1,2,∗, Bonaventure F. P. Dossou1,3,∗, Jessica Ojo1,3, Jenalea Rajab1,5, Fadel Thior1, Eric Peter Wairagala1, Anuoluwapo Aremu1, Pelonomi Moiloa1, Jade Abbott1, Vukosi Marivate1,4, Benjamin Rosman1,5

1 Lelapa AI, 2 MBZUAI, 3 McGill University & Mila Quebec AI Institute, 4 DSFSI - University of Pretoria, 5 RAIL Lab - University of the Witwatersrand ∗ Equal Contributions

### Abstract

High-resource language models often fall short in the African context, where there is a critical need for models that are efficient, accessible, and locally relevant, even amidst significant computing and data constraints. This paper introduces InkubaLM, a small language model with 0.4 billion parameters, which achieves performance comparable to models with significantly larger parameter counts and more extensive training data on tasks such as machine translation, question-answering, AfriMMLU, and the AfriXnli task. Notably, InkubaLM outperforms many larger models in sentiment analysis and demonstrates remarkable consistency across multiple languages. This work represents a pivotal advancement in challenging the conventional paradigm that effective language models must rely on substantial resources. Our model and datasets are publicly available 1 to encourage research and development on lowresource languages.

### 1 Introduction

The field of Natural Language Processing (NLP) has witnessed transformative growth with the advent of Large Language Models (LLMs). These models, characterized by billions of parameters, have set new benchmarks in tasks such as language translation, sentiment analysis, and more sophisticated applications like creative writing and conversational AI. Notable examples include models like GPT-3 (Brown et al., 2020) and BERT (Devlin et al., 2018), which have influenced academic research and have seen extensive adoption across industries. These models are central to enhancing user interactions and automating content generation, leading to widespread use in consumer applications like chatbots, virtual assistants, and recommendation systems.

1https://huggingface.co/lelapa

LLMs have predominantly been developed and trained on high-resource languages with extensive datasets (Minaee et al., 2024), such as English, Chinese, and Spanish, leaving low-resource languages, particularly in Africa, at a significant disadvantage. The main challenge is the scarcity of quality textual data, as LLMs require vast data to train effectively. This data is often fragmented, non-standardized, or non-existent for low-resource languages, and the absence of essential tools like part-of-speech taggers or annotated datasets limits progress. With its over 2,000 languages, Africa exemplifies this challenge, as many of these languages are underrepresented in digital resources and NLP research (Nekoto et al., 2020). Additionally, the computational resources required to train and deploy LLMs are frequently scarce in many African regions, limiting the ability of researchers and developers to leverage these powerful tools for use within their communities (Orife et al., 2020).

Developing models that are easy to refine, finetune, explore, and deploy in cost-effective ways on limited hardware is essential. While opensource models have made strides in bridging the language gap, more efforts are needed to create models that are not only efficient but also locally relevant. InkubaLM, which can move 250 times its weight, inspired by the strength and adaptability of the dung beetle, exemplifies this approach by offering a smaller yet powerful model designed to empower African communities. Accompanied by two datasets, InkubaLM represents a pioneering initiative to distribute the computational load and enable access to NLP tools such as Machine Translation, Sentiment Analysis, Named Entity Recognition (NER), Parts of Speech Tagging (POS), Question Answering, and Topic Classification for their languages.

Our contributions are as follows: (1) we introduce InkubaLM, the first open-source small multilingual language model for African languages, and

(2) we introduce the first instruction dataset for five NLP tasks in African languages. (3) we are also releasing the monolingual dataset used for training in 5 African languages to encourage further research on low-resource languages.

### 2 Related Work

- 2.1 Low-resource Languages

Given the vast linguistic diversity and the scarcity of digital resources for many languages, particularly in Africa, researchers have been exploring various approaches to make LLMs more inclusive and effective for these languages.

One approach to addressing the challenges in low-resource settings involves multilingual models and cross-lingual transfer learning. Models like Multilingual BERT (mBERT) (Devlin et al., 2018), XLM-R (Conneau et al., 2019), and Llama 3 (Dubey et al., 2024) have been trained on data from multiple languages, including some low-resource languages. These models can leverage shared representations across languages, allowing for improved performance even in cases where labeled data is scarce. However, the effectiveness of these models in truly low-resource languages remains limited due to the small amount of training data available for these languages.

In addition to general-purpose multilingual models (Qin et al., 2024), there has been a push toward developing specialized models tailored to specific low-resource languages (Hedderich et al., 2021). These models often utilize transfer learning techniques, where a pre-trained model on a high-resource language is fine-tuned on a smaller dataset from a low-resource language (Nekoto et al., 2020). This approach improves the model’s performance and reduces the computational resources required, making it more feasible for use in resourceconstrained environments.

Despite these advancements, significant challenges remain in making LLMs effective for lowresource languages. Issues such as linguistic bias, model interpretability, and the ethical implications of deploying these models in diverse cultural contexts remain concerning. This paper focuses on developing a culturally aware, efficient, low-resource model that can operate effectively despite limited data and computational power.

- 2.2 Small language models Zhang et al. (2024) introduced TinyLlama, a 1.1

billion parameter language model pre-trained on 1 trillion tokens. Despite its compact size, TinyLlama leverages techniques such as FlashAttention to deliver strong performance across various tasks, outperforming many models within its class. Building on this, the authors developed TinyLlama v1.1, which includes specialized versions of models tailored for math, code, and Chinese language tasks, showcasing improved results through a multi-stage pretraining process.

In a related effort to optimize model efficiency, the OneBit framework (Xu et al., 2024) marks a significant advancement in the quantization of large language models (LLMs) to 1-bit representations. This method drastically reduces computational and memory requirements, enabling the deployment of LLMs on devices with limited resources. Departing from traditional quantization techniques that use 4-bit or 8-bit compression, OneBit achieves an impressive compression ratio while maintaining a balanced trade-off between size reduction and model accuracy across various tasks.

Moving towards resource-efficient training, Inheritune (Sanyal et al., 2024) develops smaller base language models by inheriting layers from a more extensive reference model and training on a significantly reduced dataset. This approach was exemplified using a 1.5 billion parameter model derived from a more significant 3 billion parameter model. Despite training on only 1 billion tokens—just 0.1% of the original dataset—the resulting model performed comparably to others trained on significantly larger datasets, highlighting its effectiveness in low-data regimes.

Focusing on on-device processing, MobiLlama (Thawakar et al., 2024) is a 0.5 billion parameter small language model (SLM) optimized explicitly for resource-constrained devices. MobiLlama is designed for energy efficiency, low memory usage, and faster inference times, making it ideal for on-device applications. The researchers employed a parameter-sharing technique across transformer layers, enabling the model to retain high accuracy while minimizing training and deployment costs. MobiLlama was evaluated across nine benchmarks, consistently outperforming comparable models, especially in efficiency on low-end hardware.

Regarding model performance in specific tasks, Lepagnol et al. (2024) conducted a study on small language models (SLMs) in zero-shot text classification. Their research involved testing models ranging from 77M to 40B parameters across 15

diverse datasets. The results showed that these smaller models not only match but sometimes surpass the performance of their larger counterparts, creating an open-source repository documenting their methodologies.

Scaria et al. (2024) investigated the capacity of small language models to learn, retain, and unlearn noise patterns. Their study involved models like Olmo 1B, Qwen1.5 1.8B, Gemma 2B, and Phi2

- 2.7B. It revealed that while these models could learn and even eliminate noise, their performance varied significantly depending on the type of noise introduced, particularly at the character level.

Zhu et al. (2024) introduced LLaVA-Phi, an efficient multi-modal assistant that harnesses the power of the small language model Phi-2 to facilitate multi-modal dialogues. Despite having only

- 2.7 billion parameters, LLaVA-Phi demonstrated commendable performance across benchmarks, including visual comprehension and reasoning tasks. The model opens new avenues for applications in time-sensitive environments and systems requiring real-time interaction, proving the potential of smaller language models in sophisticated tasks while maintaining greater resource efficiency.

For natural language processing tasks, Brei et al. (2024) addressed the challenge of translating natural language into SPARQL queries using SLMs. Models such as BART and M2M100 were employed across datasets like QALD and CoyPu, achieving solid results in SPARQL translation, though T5 models struggled with accuracy.

In another approach to model efficiency, Song et al. (2024) focused on achieving sparse activation in SLMs. They developed a new attribution metric to overcome the limitations of existing sparse activation techniques, successfully achieving an 80% sparsification ratio with minimal accuracy loss comparable to larger models.

Lastly, in speech synthesis, Lemerle et al. (2024) introduced the Small-E model, a compact language model enhanced with linear attention. Their work set a new benchmark in zero-shot voice cloning, demonstrating the strong capabilities of small models in this specialized area.

#### 2.3 LLMs for African languages

Recent research on LLMs for African languages focuses on several key aspects: the creation of linguistic resources, the development of adapted models, and the improvement of the performance of LLMs for these languages. Adelani et al. (2024a)

introduce IrokoBench, a benchmark dataset for 16 African languages, which uncovers significant performance gaps between high-resource and lowresource languages and underscores the importance of developing LLMs tailored to African languages, especially given the disparity between open and proprietary models. ? present a novel approach with AfriBERTa, a multilingual language model specifically trained on low-resource African languages, showing that “small data” approaches can outperform traditional models like mBERT and XLM-R without relying on high-resource languages. Building on the challenges of low-resource languages, Joshua (2024) address the hallucination issues in LLMs, particularly GPT-3.5 turbo, when processing Yoruba by employing RetrievalAugmented Generation (RAG) techniques, significantly enhancing the models’ accuracy and cultural relevance. Similarly, Lawal et al. (2024) explore the performance of LLMs in educational settings, particularly in understanding and generating Yoruba primary education science content, and reveal a need for more targeted language-specific models due to the underperformance of existing LLMs in this context. Focusing on another lowresource language, Azime et al. (2024) enhance the LLAMA-2-Amharic model by integrating taskspecific and generative datasets, demonstrating improved performance in various NLP tasks and contributing to the growing body of resources for lowresource languages.

### 3 Languages

Over 3,000 languages are spoken in Africa, a continent known for its rich cultural diversity. From this language, Swahili, Hausa, and Yoruba are widely spoken languages in Africa, with over 218 million speakers. In contrast, isiZulu and isiXhosa are widely spoken languages in South Africa, with over 22 million speakers. We focus on these five languages due to the availability of corpora and because they are the top widely spoken African languages.

#### 3.1 Hausa

The Hausa language is spoken in Northern Nigeria and parts of Cameroon, Chad, Ivory Coast, Ghana, Benin, Togo, Sudan and Niger. The language belongs to the Chadic branch of the Afro-Asiatic language family, boasting about 90M first and secondlanguage speakers. Hausa has between 23-25 con-

sonants and 10 vowels. It also heavily loan words from Arabic and has several dialectal variants.

#### 3.2 isiZulu

isiZulu is predominantly spoken in South Africa and some parts of Zimbabwe and has about 14M first language speakers. The language belongs to the Nguni branch of the Niger-Congo language family, with 51 consonants, five vowels, and 15 click sounds. isiZulu also uses three level tones: low, mid, and high tones, which are distinctive in the language.

#### 3.3 isiXhosa

isiXhosa is spoken predominantly in South Africa, parts of Zimbabwe, and Lesotho. It is classified under the Nguni Languages branch of the NigerCongo language family. The language has 8M native speakers and about 11M second language speakers in South African areas of Eastern Cape, Western Cape, Northern Cape, and Gauteng. isiXhosa is the second most spoken Bantu language in South Africa. The language has 58 consonants, which include 18 click consonants, ten vowels, and two tones, which are rarely marked and use the Latin script.

#### 3.4 Swahili

Swahili, also known as KiSwahili, is the most widely spoken language on the African continent, with about 150 million first- and second-language speakers. It is also an official language in Tanzania, Rwanda, Kenya, and Uganda and one of the three official languages of the East African Community (EAC). The language has 30 letters, including 24 Latin letters without characters (q and x) and six additional consonants (ch, dh, gh, ng’, sh, th) unique to Swahili pronunciation.

#### 3.5 Yoruba

Over 40 million native speakers of Yoruba speak it in South-Western Nigeria and parts of Togo, Benin, and Ghana. It has 18 consonants, seven oral vowels, five nasal vowels, and syllabic nasals. The Yoruba language is tonal, with three tones: low, mid, and high. The tonal marks and underdots are referred to as diacritics and are needed for a word’s correct pronunciation. The language’s sentence structure is Subject-Verb-Object.

### 4 Dataset

We present two datasets: Inkuba-Mono (for monolingual pre-training) and Inkuba-Instruct (for instruction fine-tuning) for five languages listed in Section 3.

#### 4.1 Inkuba-Mono Dataset

Inkuba-Mono2 is a monolingual dataset collected from open-source repositories in the five languages to train the InkubaLM model. We collected opensource datasets for these five African languages from repositories on Hugging Face3, Github4 and Zenodo5. After pre-processing and combining the raw dataset, we used 2.4 billion tokens to train the InkubaLM models. Table 1 shows Inkuba-Mono dataset statistics.

|Language<br><br>|Number of sentences|Tokens|
|---|---|---|
|Hausa<br><br>|10.5 M|345 M|
|Yoruba|2.2 M|759.5 M<br><br>|
|Swahili<br><br>|44.3 M|1.2 B<br><br>|
|isiZulu<br><br>|9 M<br><br>|172.7 M|
|isiXhosa<br><br>|3 M|62.5 M|

|African only<br><br>|69.3 M|1.9 B|
|---|---|---|
|English<br><br>|18.2 M<br><br>|409.9 M|
|French<br><br>|5.2 M|174 M|

Total 93.2 M 2.4 B

Table 1: Dataset statistics for Inkuba-Mono dataset. M stands for Millions, and B stands for Billions.

#### 4.2 Inkuba-Instruct Dataset

The Inkuba-Instruct6 dataset is a comprehensive multilingual instruction dataset, combining several open-source downstream datasets designed to support a range of natural language processing tasks in these African languages. Our instruction dataset focused on six tasks for each of the five languages: Machine Translation, Sentiment Analysis, Named Entity Recognition (NER), Parts of Speech Tagging (POS), Question Answering, and Topic Classification. Table 3 summarizes the datasets and their sources we used for each task, and Table 2 summarizes the statistics of the instruction dataset of each language. It is important to note that we use English only for Machine Translation as a pivot language from and to the African languages.

- 2https://huggingface.co/datasets/lelapa/

Inkuba-Mono

- 3https://huggingface.co/datasets
- 4https://github.com/
- 5https://zenodo.org/
- 6https://huggingface.co/datasets/lelapa/

Inkuba-instruct

[Figure 1]

Figure 1: Example of the Swahili Topic Classification dataset converted into an Instruction dataset.

|Language|Number of samples<br><br>|
|---|---|
|Hausa|5.8 M|
|Yoruba<br><br>|6.4 M|
|Swahili|62.41 M<br><br>|
|isiZulu|16.20 M|
|isiXhosa|25.35 M<br><br>|
|English|95.42 M|

Table 2: Dataset statistics for different languages. English is used as a pivot language for Machine Translation from (eng → xxx) and to (xxx → eng) African languages.

We created prompt templates for these tasks in English and manually translated them into the five African languages. We built the instruction datasets for the machine translation task in two directions (xxx → eng and eng → xxx, where xxx represents the African language). Regarding the Topic Classification and Sentiment Analysis tasks, we translated and used the labels in the respective target languages (i.e., if the label is ‘politics’, then for Swahili, we use the Swahili translation of ‘politics’; if we switch to Hausa, we use the Hausa translation of ‘politics’). We did not perform this mapping for tasks such as NER and POS, as the labels are language agnostic. After generating the instruction inputs and targets for each task and language, we merged them and added a ‘task’ column to make it easier to filter later. We split into ‘train’, ‘dev’ and ‘test’ sets. Across all languages, merging tasks, we created a training instruction dataset of 148M sam-

ples, a validation set of 65M samples, and a testing set of size 55M samples. In Figure 1, we show an example of how we converted the Swahili Topic Classification dataset into an instruction dataset.

### 5 InkubaLM

InkubaLM is the first decoder-only lightweight African language model with a 0.4B parameter model trained from scratch with an autoregressive language modeling objective for these five African languages. During training, we also included English and French datasets due to their prevalence in many African regions and the tendency for natural African languages to code-mix.

InkubaLM architecture and hyperparameters follow existing work (Thawakar et al., 2024), with a slight modification by introducing multilingual capability and implementing custom Flash Attention (Dao et al., 2022) to enhance efficiency. This technique allows us to optimize the utilization of the compute resources, resulting in improved performance and efficiency during the training and inference process. During training, we incorporated Fully Sharded Data Parallel (FSDP) to efficiently utilize multi-GPU and multi-node setups. Table 4 shows the InkubaLM-0.4B model architecture and hyperparameters used during training.

We employ Byte Pair Encoding (BPE) (Gage, 1994) to train a multilingual tokenizer with a vocabulary size of 61788. The figure below shows different public models’ training data and model sizes. In Figure 2, we compared the training dataset

|Task<br><br>|Datasets<br><br>|Total Size (# samples)|
|---|---|---|
|Machine Translation<br><br>|WMT-22-African (Team et al., 2022), Mafand-MT (Adelani et al., 2022a), Menyo-20k (Adelani et al., 2021)|359 M|
|NER<br><br>|MasakhaNER2 (Adelani et al., 2022b), Hausa VoA NER (Hedderich et al., 2020), isiXhosa NER Corpus (Eiselen, 2016)|64k|
|POS|MasakhaPOS (Dione et al., 2023)<br><br>|6.5k|
|Question-Answering<br><br>|AfriQA (Ogundepo et al., 2023)|4.45k|
|Topic Classification<br><br>|SIB-200 (Adelani et al., 2023a), MasakhaNEWS (Adelani et al., 2023b), Hausa News Classification (Hedderich et al., 2020)|22.8k|
|Sentiment Analysis|AfriSenti (Muhammad et al., 2023a,b), NaijaSenti (Muhammad et al., 2022), Swahili-Tweet-Sentiment|46.62k|

Table 3: Sources of downstream datasets for the five African languages.

Figure 2: Training data(in billions of token) vs. Model size (in billions of parameters)

|Hyperparameter|Value|
|---|---|
|Total Parameters|0.422B<br><br>|
|Hidden Size<br><br>|2048|
|Intermediate Size (MLPs)|5632|
|Number of Attention Heads<br><br>|32|
|Number of Hidden Layers<br><br>|8|
|RMSNorm ϵ|1 × 10−5<br><br>|
|Max Seq Length<br><br>|2048|
|Vocab Size<br><br>|61788|

Table 4: InkubaLM-0.4B architecture and hyperparameters.

the least data from the compared models.

The environmental impact of training the InkubaLM-0.4B model was considered and measured using the machine learning impact calculator 7. The training process on 8 A100 NVIDIA GPUs over 16 days resulted in an estimated carbon emission of 53.76 kg of CO2 equivalent. The training was performed on the Google Cloud Platform (GCP) in the Asia-Southeast1-C region, where the carbon footprint was carefully monitored to assess the environmental impact.

size used for models ranging from 0.4B up to 8B in terms of parameters. As shown in the Figure, our model is the smallest and has been trained using

7https://mlco2.github.io/impact#compute

### 6 Evaluation

#### 6.1 Evaluation Tool

To evaluate models, we use the EleutherAI LM Evaluation Harness tool (Gao et al., 2024) —- a popular evaluation framework that supports a wide range of zero- and few-shot evaluation tasks on autoregressive language models. In all evaluations, we use zero-shots and prompts in these languages; Section 6.4 discusses our prompt designs.

#### 6.2 Model Selection

We select open-source base models to evaluate their performance on tasks discussed in Section 6.3. We compare the performance of small, big, and multilingual models with our InkubaLM model. For small models, we use SmolLM (Allal et al., 2024) with 1.7B parameters and MobiLlama (Thawakar et al., 2024) with 1B parameters, for Big models we use Gemma (Team et al., 2024) with 7B parameters and LLaMa 3 (Dubey et al., 2024) with 8B parameters and for multilingual models, we use BLOOMZ with 7B parameters and lola_v18 with 7.4B parameters.

#### 6.3 Tasks

We evaluate models with the Inkuba-Instruct (Section 4.2) and IrokoBench (Adelani et al., 2024b) datasets. From the Inkuba-Instruct dataset, we select sentiment analysis and machine translation tasks, as discussed in Section 4.2; these tasks are created by combining task datasets from different open-source platforms. IrokoBench is a humantranslated benchmark dataset that includes languages from various geographical regions of Africa. We use AfriXnli, a human-translated dataset for African languages from the English portion of XNLI (Conneau et al., 2018), and AfriMMLU, a human-translated dataset for the African language from MMLU (Hendrycks et al., 2020).

#### 6.4 Prompts

For the Inkuba-Instruct dataset, we explore three prompts: (1) Multiple Prompts (direct): the model is prompted using four slightly different prompts at random curated for each task in five languages. (2) Single prompt – English (English): the model is prompted using one English prompt curated for each task (3) Single prompt – Native (native): the model is prompted using one prompt in the language curated for each task. Table

8https://huggingface.co/dice-research/lola_v1

5 shows different prompt samples we use for initial evaluation. We follow the same template for the IrokoBench dataset used in their work. We report results for all tasks using native prompts.

### 7 Results

#### 7.1 Sentiment Analysis

The sentiment analysis results, presented in Table 6, highlight the performance of various language models, including InkubaLM-0.4B, on three African languages: Swahili (swa), Hausa (hau), and Yoruba (yor). These results were obtained in a zero-shot setting using English prompts.

#### 7.1.1 Key Observations

- • InkubaLM: achieves an average score of 30.93, and is particularly impressive in Swahili, where it scores 42.47. This score is the highest among all the models tested, indicating that InkubaLM is particularly welltuned for Sentiment Analysis in Swahili.
- • Comparison with Other Models:

- – SmolLM-1.7B: achieves a lower average score of 28.80. This suggests that InkubaLM-0.4B’s optimizations may be more effective for certain African languages, particularly Swahili.
- – MobiLlama-1B: This model outperforms InkubaLM-0.4B on average, with a score of 34.87, showing stronger performance in Hausa and Yoruba. However, it falls short in Swahili compared to InkubaLM-0.4B.
- – Gemma-7B and LLaMa 3-8B: These models show mixed performance, with Gemma-7B excelling in Hausa but underperforming in Swahili. LLaMa 3-8B shows consistent, moderate performance across all languages.
- – BLOOMZ-7B and lola_v1-7.4B: These models generally perform worse, particularly in Swahili, where their scores are significantly lower than those of InkubaLM-0.4B.

##### 7.1.2 Analysis The results indicate that InkubaLM is highly competitive in Sentiment Analysis, especially in Swahili. Its performance in this task demonstrates the effectiveness of the model’s architecture and

|Tasks<br><br>|Prompts|langs<br><br>|prompt|
|---|---|---|---|
|Sentiment|Tafadhali tambua mawazo yaliyoonyeshwa kwenye matini haya kwa kutegemea miongozo ifuatayo: Chanya: —, Hasi: —, Wastani: — {inputs} Output:<br><br>|swa<br><br>|Native|
|MT (swa-eng)<br><br>|Tafsiri zifuatazo kutoka kwa Swahili hadi English. {inputs} Output:|swa<br><br>|Native|
|MT (eng-swa)<br><br>|Tafsiri zifuatazo kutoka kwa English hadi Swahili. {inputs} Output:|swa|Native|
|Sentiment|Please identify the sentiment reflected in this text based on the following guidelines: Positive: —, Negative: —, Neutral: — {inputs} Output:|swa|English<br><br>|
|MT (eng-swa)<br><br>|Translate the following from Swahili into English. {inputs} Output:<br><br>|swa|English|

Table 5: Sample prompt templates used for machine translation and sentiment analysis tasks

Model swa hau yor AVG Prompt LLMs in English Language

InkubaLM-0.4B 42.47 22.25 28.08 30.93 SmolLM-1.7B 26.09 31.97 28.36 28.80 MobiLlama-1B 37.2 34.53 32.89 34.87 Gemma-7B 14.42 36.16 26.17 25.58 LLaMa 3-8B 19.48 32.44 29.77 27.23 BLOOMZ-7B 17.26 33.81 32.99 28.02 lola_v1-7.4B 14.4 26.71 28.16 22.42

Table 6: Sentiment Analysis results using prompt in English. The above results are zero-shot results only

training optimizations, such as using Flash Attention and including multilingual capabilities. Despite its smaller size, InkubaLM outperforms or matches larger models’ performance in key areas, making it a strong contender for tasks involving African languages.

#### 7.2 Machine Translation

The machine translation results are split into two tasks: translating from English to African languages (Table 7) and from African languages to English (Table 8). The performance is measured using the BLEU score, a widely used metric for evaluating machine translation quality.

- 7.2.1 English to African Languages (Zero-shot)

• InkubaLM achieves an average BLEU score of 8.15, with its highest performance in isiZulu (21). The model shows moderate performance in the other languages, with BLEU (Papineni et al., 2002) scores ranging from 3.44 in Swahili to 7.0 in Xhosa.

#### • Comparison with other models:

- – SmolLM-1.7B and MobiLlama-1B: Both models achieve an average BLEU score of 6.47. While SmolLM-1.7B performs better in Swahili, it significantly underperforms in isiZulu compared to InkubaLM-0.4B.
- – Gemma-7B and LLaMa 3-8B: These models generally underperform in comparison to InkubaLM-0.4B, especially in isiZulu, where they achieve much lower BLEU scores.
- – lola_v1-7.4B: Despite having the highest average BLEU score (9.78), lola_v17.4B shows inconsistent performance across different languages, with particularly high scores in isiZulu but lower scores in other languages.

#### 7.2.2 Analysis

InkubaLM-0.4B demonstrates strong performance in the English to African language translation task, particularly excelling in isiZulu. Its competitive average BLEU score suggests that the model is wellsuited for specific languages, even when compared to larger models. However, its performance across different languages is variable, indicating that further refinement could enhance its overall translation capabilities while the model is optimized for particular tasks and languages.

InkubaLM-0.4B 3.44 4.67 3.49 7.0 21 8.15 SmolLM-1.7B 13.08 3.24 7.36 8.23 2.21 6.47 MobiLlama-1B 1.78 3.24 7.36 8.23 2.21 6.47 Gemma-7B 3.46 2 3.75 2.95 1.99 2.80 LLaMa 3-8B 3.17 4.17 5.11 5.45 2.83 3.82 BLOOMZ-7B 3.28 1.73 3.85 3 2.76 2.90 lola_v1-7.4B 5.63 14.5 9.3 8.87 9.7 9.78

- Table 7: Zero-shot BLEU score for English to African translation results using prompts in the African language.

Model swa hau yor xho zul AVG Prompt LLMs in African Language InkubaLM-0.4B 5.29 4.82 5.71 10.06 8.06 6.78 SmolLM-1.7B 0.72 6.24 16.97 0.73 2.64 5.46 MobiLlama-1B 1.16 3.42 7.35 2.96 4.72 3.92 Gemma-7B 1.24 0.81 5.58 0.76 0.11 1.790 LLaMa 3-8B 17.38 5.54 20.24 26.26 15.89 17.06 BLOOMZ-7B 0.99 11.26 5.21 3.75 15.91 7.42 lola_v1-7.4B 7.89 8.73 2.20 19.03 14.47 10.46

- Table 8: Zero-shot BLEU score for African to English translation results using prompts in African language.

- 7.2.3 African Languages to English (Zero-shot)

- • InkubaLM-0.4B shows moderate performance with an average BLEU score of 6.78, with notable strengths in Xhosa (10.06) and Swahili (5.29).
- • Comparison with other models:

- – LLaMa 3-8B stands out with a significantly higher average BLEU score of 17.06, particularly excelling in Xhosa and Swahili. This model outperforms InkubaLM-0.4B across all languages.
- – BLOOMZ-7B and lola_v1-7.4B also perform well, with lola_v1-7.4B achieving the second-highest average score (10.46).
- – SmolLM-1.7B and MobiLlama-1B: These models show mixed results, with relatively lower scores in several languages compared to InkubaLM-0.4B.

- 7.2.4 Analysis The results for African languages to English translation indicate that InkubaLM-0.4B performs reasonably well but is outperformed by larger models such as LLaMa 3-8B and lola_v1-7.4B. The variability in scores suggests that while InkubaLM0.4B is effective in some instances, particularly in translating Xhosa and Swahili, it may benefit from further tuning and optimization to improve its overall translation performance in this direction.

7.3 AfriMMLU

Table 9 presents the F1 scores for various models on the AfriMMLU task, using prompts in the five African languages. The models evaluated include InkubaLM-0.4B, SmolLM-1.7B, MobiLlama-1B, Gemma-7B, LLaMa 3-8B, BLOOMZ-7B, and lola_v1-7.4B.

#### 7.3.1 Key Observations

- • InkubaLM-0.4B: This model achieves an average F1 score of 26.16 across the five languages, with its best performance in Hausa (29.4) and Xhosa (27.4). It performs consistently across all languages, showing balanced capabilities.
- • Comparison with other models:

- – Gemma-7B: This model outperforms all others, with the highest average F1 score of 30.28. It shows particularly strong performance in Swahili (34) and Zulu (32.4), indicating its effectiveness across various African languages.
- – LLaMa 3-8B: With an average F1 score of 28.12, this model also performs well, closely following Gemma-7B, and is particularly strong in Swahili (30.4) and Hausa (29.6).
- – BLOOMZ-7B: This model achieves an average F1 score of 24.2, with consistent but lower scores across all languages

- compared to InkubaLM-0.4B and the topperforming models.
- – SmolLM-1.7B and MobiLlama-1B: These models have lower average F1 scores of 21.88 and 22.28, respectively. They show weaker performance across most languages, particularly in Xhosa and Yoruba.
- – lola_v1-7.4B: Despite being a larger model, lola_v1-7.4B achieves a relatively low average F1 score of 23.0, with inconsistent performance, particularly lower in Yoruba (18.8) and Zulu (21).

#### 7.3.2 Analysis

The results indicate that InkubaLM-0.4B performs competitively on the AfriMMLU task, with an average F1 score that positions it in the mid-range among the models tested. While it does not outperform the better models such as Gemma-7B or LLaMa 3-8B, InkubaLM-0.4B shows consistent performance across all languages, making it a reliable option for tasks involving African languages.

The top-performing model, Gemma-7B, demonstrates the highest average F1 score, suggesting that larger model sizes, combined with practical multilingual training, can significantly enhance performance on complex tasks like AfriMMLU. However, the performance of InkubaLM-0.4B, despite its smaller size, highlights the effectiveness of its architecture and training optimizations, such as Flash Attention and multilingual capabilities.

Given these results, while larger models dominate raw performance, InkubaLM-0.4B offers a balanced trade-off between model size and effectiveness, particularly in resource-constrained settings where computational efficiency is a priority and data resources are limited.

7.4 AfriXnli

Table 10 presents the F1 scores for various models on the AfriXnli task, using prompts in the five African languages. The models evaluated include InkubaLM-0.4B, SmolLM-1.7B, MobiLlama-1B, Gemma-7B, LLaMa 3-8B, BLOOMZ-7B, and lola_v1-7.4B.

#### 7.4.1 Key Observations

• InkubaLM-0.4B: InkubaLM-0.4B achieves an average F1 score of 33.47, with consistent performance across all languages, ranging

from 32.8 in Xhosa to 35.3 in Hausa. This consistency indicates the model’s balanced capability across the different African languages.

#### • Comparison with other models:

- – SmolLM-1.7B: This model achieves an average F1 score of 33.09, performing similarly to InkubaLM-0.4B but slightly lower on average.
- – MobiLlama-1B: With an average F1 score of 33.66, MobiLlama-1B slightly outperforms InkubaLM-0.4B, showing strong performance across all languages with scores consistently above 32.5.
- – Gemma-7B: This model achieves a significantly higher average F1 score of 37.5, with particularly strong performance in Swahili (39.3) and Hausa (38.8), indicating its robustness across these languages.
- – LLaMa 3-8B: With an average F1 score of 37.7, LLaMa 3-8B also performs well, closely following Gemma-7B, and shows strong results across all languages.
- – BLOOMZ-7B: This model stands out with the highest average F1 score of 41.52, demonstrating exceptional performance in Swahili (45.8) and Yoruba (45.2).
- – lola_v1-7.4B: This model achieves an average F1 score of 34.13, performing comparably to InkubaLM-0.4B, with relatively consistent results across all languages.

#### 7.4.2 Analysis

The results indicate that InkubaLM-0.4B is a strong performer in the AfriXnli task, achieving a consistent F1 score across all evaluated African languages. While it does not achieve the highest average score, its performance is close to that of other similarly sized models like SmolLM-1.7B and MobiLlama-1B.

The top-performing models, Gemma-7B, LLaMa 3-8B, and BLOOMZ-7B, demonstrate the benefits of larger model sizes. BLOOMZ-7B outperforms all others with an impressive average F1 score of 41.52. This suggests that while InkubaLM-0.4B is effective, larger models can offer significant performance advantages, espe-

InkubaLM-0.4B 25 29.4 24.8 27.4 24.2 26.16 SmolLM-1.7B 20.2 21.8 23.4 19.2 24.8 21.88 MobiLlama-1B 19.8 20.2 24.2 25.2 22 22.28 Gemma-7B 34 29 29 27 32.4 30.28 LLaMa 3-8B 30.4 29.6 28 24.6 28 28.12 BLOOMZ-7B 26 22.2 25.6 23.2 24 24.2 lola_v1-7.4B 32.44 21.2 18.8 21.6 21 23.0

Table 9: F1 score for AfriMMLU results using prompt in African language.

Model swa hau yor xho zul AVG Prompt LLMs in African Language InkubaLM-0.4B 33 35.3 33.16 32.8 33.1 33.47 SmolLM-1.7B 33.33 33.66 32.33 33.33 32.83 33.09 MobiLlama-1B 34.66 33.66 32.5 33.5 34 33.66 Gemma-7B 39.3 38.8 36.5 37.2 35.7 37.5 LLaMa 3-8B 39.5 36 38.1 38.1 36.8 37.7 BLOOMZ-7B 45.8 36.5 45.2 39.8 40.3 41.52 lola_v1-7.4B 36.83 33.5 33.16 33.83 33.33 34.13

Table 10: AfriXnli results using prompt in African language.

cially in tasks like AfriXnli that require nuanced language understanding.

Overall, while InkubaLM-0.4B may not match the performance of the largest models, it offers a good balance of size and performance, making it a competitive option for African language processing tasks within resource-constrained environments.

### 8 Conclusion

InkubaLM-0.4B has proven a reliable and efficient model tailored for African language processing, delivering robust performance across various tasks. Despite being a smaller model with only 0.4 billion parameters, it consistently holds its own against much larger models in tasks like Sentiment Analysis, Machine Translation, AfriMMLU, and AfriXnli. Its balanced performance across five key African languages (Swahili, Hausa, Yoruba, Xhosa, and Zulu) demonstrates the model’s robustness and versatility, particularly in low-resource settings where such capabilities are critical.

In Sentiment Analysis, InkubaLM-0.4B stands out with its top performance in Swahili, outperforming all other models in this language. Similarly, in the Machine Translation task, the model shows strong results in isiZulu, further underscoring its ability to handle diverse linguistic contexts. The model’s design, which incorporates Flash Attention and supports multilingual capabilities, plays a significant role in its efficiency, allowing it to achieve competitive results without requiring extensive computational resources. This efficiency

is critical in resource-constrained environments, where maximizing performance while minimizing computational costs is crucial.

While InkubaLM-0.4B may not always match the peak performance of the largest models like BLOOMZ-7B or LLaMa 3-8B, its consistent delivery of reliable results across all evaluated languages makes it a valuable tool for a wide range of applications. Despite its smaller size, the model’s ability to maintain competitive performance across various tasks highlights its potential as a go-to solution for African language processing, particularly in environments where resources are limited, and efficiency is paramount.

The imagination of NLP has traditionally been constrained by an emphasis on written histories and extensive text-based corpora, typically controlled by a few highly resourced institutions. SLMs like InkubaLM represent a significant step forward in exploring new possibilities beyond these conventional assumptions. In Africa, developing models we can train within the limitations of available computational resources and data is crucial. Such empowerment is essential for enabling local communities to address context-specific challenges, including broadened access to digital products and necessary services that would otherwise be out of reach.

### 9 Limitations

The InkubaLM model has been trained on multilingual datasets but does have some limitations. It

can understand and generate content in five African languages: Swahili, Yoruba, Hausa, isiZulu, and isiXhosa, as well as English and French. While it can generate text on various topics, the resulting content may not always be entirely accurate, logically consistent, or free from biases found in the training data. Additionally, the model may sometimes use different languages when generating text. Nonetheless, this model is intended to be a foundational tool to aid research in African languages.

### 10 Ethical Considerations and Risks

InkubaLM is a small LM developed for five African languages. The model is evaluated only in sentiment analysis, machine translation, AfriMMLU, and AfriXNLI tasks and has yet to cover all possible evaluation scenarios. Similar to other language models, it is impossible to predict all of InkubaLM’s potential outputs in advance, and in some cases, the model may produce inaccurate, biased, or objectionable responses. Therefore, before using the model in any application, the users should conduct safety testing and tuning tailored to their intended use.

### Acknowledgments

We thank Microsoft AI4Good lab9 for the compute credits to train the above model. This work would not have been possible without your sponsorship.

### References

David I Adelani, Dana Ruiter, Jesujoba O Alabi, Damilola Adebonojo, Adesina Ayeni, Mofe Adeyemi, Ayodele Awokoya, and Cristina España-Bonet. 2021. The effect of domain and diacritics in yor\ub\’aenglish neural machine translation. arXiv preprint arXiv:2103.08647.

David Ifeoluwa Adelani, Jesujoba Oluwadara Alabi, Angela Fan, Julia Kreutzer, Xiaoyu Shen, Machel Reid, Dana Ruiter, Dietrich Klakow, Peter Nabende, Ernie Chang, et al. 2022a. A few thousand translations go a long way! leveraging pre-trained models for african news translation. arXiv preprint arXiv:2205.02022.

David Ifeoluwa Adelani, Hannah Liu, Xiaoyu Shen, Nikita Vassilyev, Jesujoba O. Alabi, Yanke Mao, Haonan Gao, and Annie En-Shiun Lee. 2023a. Sib-200: A simple, inclusive, and big evaluation dataset for topic classification in 200+ languages and dialects. Preprint, arXiv:2309.07445.

9https://www.microsoft.com/en-us/research/ group/ai-for-good-research-lab/

David Ifeoluwa Adelani, Marek Masiak, Israel Abebe Azime, Jesujoba Oluwadara Alabi, Atnafu Lambebo Tonja, Christine Mwase, Odunayo Ogundepo, Bonaventure F. P. Dossou, Akintunde Oladipo, Doreen Nixdorf, Chris Chinenye Emezue, Sana Sabah al azzawi, Blessing K. Sibanda, Davis David, Lolwethu Ndolela, Jonathan Mukiibi, Tunde Oluwaseyi Ajayi, Tatiana Moteu Ngoli, Brian Odhiambo, Abraham Toluwase Owodunni, Nnaemeka C. Obiefuna, Shamsuddeen Hassan Muhammad, Saheed Salahudeen Abdullahi, Mesay Gemeda Yigezu, Tajuddeen Gwadabe, Idris Abdulmumin, Mahlet Taye Bame, Oluwabusayo Olufunke Awoyomi, Iyanuoluwa Shode, Tolulope Anu Adelani, Habiba Abdulganiy Kailani, Abdul-Hakeem Omotayo, Adetola Adeeko, Afolabi Abeeb, Anuoluwapo Aremu, Olanrewaju Samuel, Clemencia Siro, Wangari Kimotho, Onyekachi Raphael Ogbu, Chinedu E. Mbonu, Chiamaka I. Chukwuneke, Samuel Fanijo, Jessica Ojo, Oyinkansola F. Awosan, Tadesse Kebede Guge, Sakayo Toadoum Sari, Pamela Nyatsine, Freedmore Sidume, Oreen Yousuf, Mardiyyah Oduwole, Ussen Kimanuka, Kanda Patrick Tshinu, Thina Diko, Siyanda Nxakama, Abdulmejid Tuni Johar, Sinodos Gebre, Muhidin Mohamed, Shafie Abdi Mohamed, Fuad Mire Hassan, Moges Ahmed Mehamed, Evrard Ngabire, , and Pontus Stenetorp. 2023b. Masakhanews: News topic classification for african languages. ArXiv.

David Ifeoluwa Adelani, Graham Neubig, Sebastian Ruder, Shruti Rijhwani, Michael Beukman, Chester Palen-Michel, Constantine Lignos, Jesujoba Oluwadara Alabi, Shamsuddeen Hassan Muhammad, Peter Nabende, Cheikh M. Bamba Dione, Andiswa Bukula, Rooweither Mabuya, Bonaventure F. P. Dossou, Blessing K. Sibanda, Happy Buzaaba, Jonathan Mukiibi, Godson Kalipe, Derguene Mbaye, Amelia Taylor, Fatoumata Kabore, Chris C. Emezue, Anuoluwapo Aremu, Perez Ogayo, Catherine W. Gitau, Edwin Munkoh-Buabeng, Victoire Memdjokam Koagne, Allahsera Auguste Tapo, Tebogo Macucwa, Vukosi Marivate, Elvis Mboning, Tajuddeen R. Gwadabe, Tosin P. Adewumi, Orevaoghene Ahia, Joyce Nakatumba-Nabende, Neo L. Mokono, Ignatius M Ezeani, Chiamaka Ijeoma Chukwuneke, Mofetoluwa Adeyemi, Gilles Hacheme, Idris Abdulmumin, Odunayo Ogundepo, Oreen Yousuf, Tatiana Moteu Ngoli, and Dietrich Klakow. 2022b. Masakhaner 2.0: Africa-centric transfer learning for named entity recognition. ArXiv, abs/2210.12391.

David Ifeoluwa Adelani, Jessica Ojo, Israel Abebe Azime, Jian Yun Zhuang, Jesujoba O. Alabi, Xuanli He, Millicent Ochieng, Sara Hooker, Andiswa Bukula, En-Shiun Annie Lee, Chiamaka Chukwuneke, Happy Buzaaba, Blessing Sibanda, Godson Kalipe, Jonathan Mukiibi, Salomon Kabongo, Foutse Yuehgoh, Mmasibidi Setaka, Lolwethu Ndolela, Nkiruka Odu, Rooweither Mabuya, Shamsuddeen Hassan Muhammad, Salomey Osei, Sokhar Samb, Tadesse Kebede Guge, and Pontus Stenetorp. 2024a. Irokobench: A

new benchmark for african languages in the age of large language models. Preprint, arXiv:2406.03368.

David Ifeoluwa Adelani, Jessica Ojo, Israel Abebe Azime, Jian Yun Zhuang, Jesujoba O Alabi, Xuanli He, Millicent Ochieng, Sara Hooker, Andiswa Bukula, En-Shiun Annie Lee, et al. 2024b. Irokobench: A new benchmark for african languages in the age of large language models. arXiv preprint arXiv:2406.03368.

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Leandro von Werra, and Thomas Wolf. 2024. Smollm blazingly fast and remarkably powerful.

Israel Abebe Azime, Mitiku Yohannes Fuge, Atnafu Lambebo Tonja, Tadesse Destaw Belay, Aman Kassahun Wassie, Eyasu Shiferaw Jada, Yonas Chanie, Walelign Tewabe Sewunetie, and Seid Muhie Yimam. 2024. Enhancing amharic-llama: Integrating task specific and generative datasets. In 5th Workshop on African Natural Language Processing.

Felix Brei, Johannes Frey, and Lars-Peter Meyer. 2024. Leveraging small language models for text2sparql tasks to improve the resilience of ai assistance. arXiv preprint arXiv:2405.17076.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. CoRR, abs/2005.14165.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Unsupervised cross-lingual representation learning at scale. CoRR, abs/1911.02116.

Alexis Conneau, Guillaume Lample, Ruty Rinott, Adina Williams, Samuel R Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. Xnli: Evaluating crosslingual sentence representations. arXiv preprint arXiv:1809.05053.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. BERT: pre-training of deep bidirectional transformers for language understanding. CoRR, abs/1810.04805.

Cheikh M. Bamba Dione, David Ifeoluwa Adelani, Peter Nabende, Jesujoba Alabi, Thapelo Sindane, Happy Buzaaba, Shamsuddeen Hassan Muhammad, Chris Chinenye Emezue, Perez Ogayo, Anuoluwapo Aremu, Catherine Gitau, Derguene Mbaye, Jonathan Mukiibi, Blessing Sibanda, Bonaventure F. P. Dossou, Andiswa Bukula, Rooweither Mabuya, Allahsera Auguste Tapo, Edwin Munkoh-Buabeng, Victoire Memdjokam Koagne, Fatoumata Ouoba Kabore, Amelia Taylor, Godson Kalipe, Tebogo Macucwa, Vukosi Marivate, Tajuddeen Gwadabe, Mboning Tchiaze Elvis, Ikechukwu Onyenwe, Gratien Atindogbe, Tolulope Adelani, Idris Akinade, Olanrewaju Samuel, Marien Nahimana, Th’eog‘ene Musabeyezu, Emile Niyomutabazi, Ester Chimhenga, Kudzai Gotosa, Patrick Mizha, Apelete Agbolo, Seydou Traore, Chinedu Uchechukwu, Aliyu Yusuf, Muhammad Abdullahi, and Dietrich Klakow. 2023. MasakhaPOS: Part-of-speech tagging for typologically diverse African languages. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10883–10900, Toronto, Canada. Association for Computational Linguistics.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur,

Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaoqing Ellen Tan, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aaron Grattafiori, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alex Vaughan, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Franco, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, Danny Wyatt, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Firat Ozgenel, Francesco Caggioni, Francisco Guzmán, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Govind Thattai, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, He-

len Suk, Henry Aspegren, Hunter Goldman, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Karthik Prasad, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kun Huang, Kunal Chawla, Kushal Lakhotia, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Maria Tsimpoukelli, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikolay Pavlovich Laptev, Ning Dong, Ning Zhang, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Rohan Maheswari, Russ Howes, Ruty Rinott, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Kohler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vítor Albiero, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaofang Wang, Xiaojian Wu, Xiaolan Wang, Xide Xia, Xilun Wu, Xinbo Gao, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yuchen Hao, Yundi Qian, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, and Zhiwei Zhao. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Roald Eiselen. 2016. Government domain named entity recognition for south african languages. In Proceedings of the 10th Language Resources and Evaluation Conference (LREC).

Philip Gage. 1994. A new algorithm for data compression. The C Users Journal, 12(2):23–38.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2024. A framework for few-shot language model evaluation.

Michael A. Hedderich, David Adelani, Dawei Zhu, Jesujoba Alabi, Udia Markus, and Dietrich Klakow. 2020. Transfer learning and distant supervision for multilingual transformer models: A study on African languages. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2580–2591, Online. Association for Computational Linguistics.

Michael A. Hedderich, Lukas Lange, Heike Adel, Jannik Strötgen, and Dietrich Klakow. 2021. A survey on recent approaches for natural language processing in low-resource scenarios. Preprint, arXiv:2010.12309.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300.

Adejumobi Monjolaoluwa Joshua. 2024. Improving question-answering capabilities in large language models using retrieval augmented generation (RAG): A case study on yoruba culture and language. In 5th Workshop on African Natural Language Processing.

Olanrewaju Israel Lawal, Olubayo Adekanmbi, and Anthony Soronnadi. 2024. Contextual evaluation of LLM’s performance on primary education science learning contents in the yoruba language. In 5th Workshop on African Natural Language Processing.

Théodor Lemerle, Nicolas Obin, and Axel Roebel. 2024. Small-e: Small language model with linear attention for efficient speech synthesis. arXiv preprint arXiv:2406.04467.

Pierre Lepagnol, Thomas Gerald, Sahar Ghannay, Christophe Servan, and Sophie Rosset. 2024. Small language models are good too: An empirical study of zero-shot classification. arXiv preprint arXiv:2404.11122.

Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, and Jianfeng Gao. 2024. Large language models: A survey. Preprint, arXiv:2402.06196.

Shamsuddeen Muhammad, Idris Abdulmumin, Abinew Ayele, Nedjma Ousidhoum, David Adelani, Seid Yimam, Ibrahim Ahmad, Meriem Beloucif, Saif Mohammad, Sebastian Ruder, et al. 2023a. Afrisenti: A twitter sentiment analysis benchmark for african languages. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 13968–13981.

Shamsuddeen Hassan Muhammad, Idris Abdulmumin, Seid Muhie Yimam, David Ifeoluwa Adelani, Ibrahim Sa’id Ahmad, Nedjma Ousidhoum, Abinew Ayele, Saif M Mohammad, and Meriem Beloucif. 2023b. Semeval-2023 task 12: Sentiment analysis for african languages (afrisenti-semeval). arXiv preprint arXiv:2304.06845.

Shamsuddeen Hassan Muhammad, David Ifeoluwa Adelani, Sebastian Ruder, Ibrahim Sa’id Ahmad, Idris Abdulmumin, Bello Shehu Bello, Monojit Choudhury, Chris Chinenye Emezue, Saheed Salahudeen Abdullahi, Anuoluwapo Aremu, Alípio Jorge, and Pavel Brazdil. 2022. NaijaSenti: A Nigerian Twitter sentiment corpus for multilingual sentiment analysis. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 590–602, Marseille, France. European Language Resources Association.

Wilhelmina Nekoto, Vukosi Marivate, Tshinondiwa Matsila, Timi E. Fasubaa, Tajudeen Kolawole, Taiwo Fagbohungbe, Solomon Oluwole Akinola, Shamsuddeen Hassan Muhammad, Salomon Kabongo, Salomey Osei, Sackey Freshia, Rubungo Andre Niyongabo, Ricky Macharm, Perez Ogayo, Orevaoghene Ahia, Musie Meressa, Mofe Adeyemi, Masabata Mokgesi-Selinga, Lawrence Okegbemi, Laura Jane Martinus, Kolawole Tajudeen, Kevin Degila, Kelechi Ogueji, Kathleen Siminyu, Julia Kreutzer, Jason Webster, Jamiil Toure Ali, Jade Z. Abbott, Iroro Orife, Ignatius Ezeani, Idris Abdulkabir Dangana, Herman Kamper, Hady Elsahar, Goodness Duru, Ghollah Kioko, Espoir Murhabazi, Elan Van Biljon, Daniel Whitenack, Christopher Onyefuluchi, Chris Emezue, Bonaventure Dossou, Blessing K. Sibanda, Blessing Itoro Bassey, Ayodele Olabiyi, Arshath Ramkilowan, Alp Öktem, Adewale Akinfaderin, and Abdallah Bashir. 2020. Participatory research for low-resourced machine translation: A case study in african languages. CoRR, abs/2010.02353.

Odunayo Ogundepo, Tajuddeen R. Gwadabe, Clara E. Rivera, Jonathan H. Clark, Sebastian Ruder, David Ifeoluwa Adelani, Bonaventure F. P. Dossou, Abdou Aziz DIOP, Claytone Sikasote, Gilles Hacheme, Happy Buzaaba, Ignatius Ezeani, Rooweither Mabuya, Salomey Osei, Chris Emezue, Albert Njoroge Kahira, Shamsuddeen H. Muhammad, Akintunde Oladipo, Abraham Toluwase Owodunni, Atnafu Lambebo Tonja, Iyanuoluwa Shode, Akari Asai, Tunde Oluwaseyi Ajayi, Clemencia Siro, Steven Arthur, Mofetoluwa Adeyemi, Orevaoghene Ahia, Aremu Anuoluwapo, Oyinkansola Awosan, Chiamaka Chukwuneke, Bernard

Opoku, Awokoya Ayodele, Verrah Otiende, Christine Mwase, Boyd Sinkala, Andre Niyongabo Rubungo, Daniel A. Ajisafe, Emeka Felix Onwuegbuzia, Habib Mbow, Emile Niyomutabazi, Eunice Mukonde, Falalu Ibrahim Lawan, Ibrahim Said Ahmad, Jesujoba O. Alabi, Martin Namukombo, Mbonu Chinedu, Mofya Phiri, Neo Putini, Ndumiso Mngoma, Priscilla A. Amuok, Ruqayya Nasir Iro, and Sonia Adhiambo. 2023. Afriqa: Cross-lingual openretrieval question answering for african languages. Preprint, arXiv:2305.06897.

Iroro Orife, Julia Kreutzer, Blessing K. Sibanda, Daniel Whitenack, Kathleen Siminyu, Laura Martinus, Jamiil Toure Ali, Jade Z. Abbott, Vukosi Marivate, Salomon Kabongo, Musie Meressa, Espoir Murhabazi, Orevaoghene Ahia, Elan Van Biljon, Arshath Ramkilowan, Adewale Akinfaderin, Alp Öktem, Wole Akin, Ghollah Kioko, Kevin Degila, Herman Kamper, Bonaventure Dossou, Chris Emezue, Kelechi Ogueji, and Abdallah Bashir. 2020. Masakhane - machine translation for africa. In 1st AfricaNLP Workshop Proceedings, AfricaNLP@ICLR 2020, Virtual Conference, Formerly Addis Ababa Ethiopia, 26th April 2020.

Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318.

Libo Qin, Qiguang Chen, Yuhang Zhou, Zhi Chen, Yinghui Li, Lizi Liao, Min Li, Wanxiang Che, and Philip S. Yu. 2024. Multilingual large language model: A survey of resources, taxonomy and frontiers. Preprint, arXiv:2404.04925.

Sunny Sanyal, Sujay Sanghavi, and Alexandros G Dimakis. 2024. Pre-training small base lms with fewer tokens. arXiv preprint arXiv:2404.08634.

Nicy Scaria, Silvester John Joseph Kennedy, and Deepak Subramani. 2024. Can small language models learn, unlearn, and retain noise patterns? arXiv preprint arXiv:2407.00996.

Jifeng Song, Kai Huang, Xiangyu Yin, Boyuan Yang, and Wei Gao. 2024. Achieving sparse activation in small language models. arXiv preprint arXiv:2406.06562.

Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. 2024. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295.

NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti,

John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and Jeff Wang. 2022. No language left behind: Scaling human-centered machine translation. Preprint, arXiv:2207.04672.

Omkar Thawakar, Ashmal Vayani, Salman Khan, Hisham Cholakal, Rao M Anwer, Michael Felsberg, Tim Baldwin, Eric P Xing, and Fahad Shahbaz Khan. 2024. Mobillama: Towards accurate and lightweight fully transparent gpt. arXiv preprint arXiv:2402.16840.

Yuzhuang Xu, Xu Han, Zonghan Yang, Shuo Wang, Qingfu Zhu, Zhiyuan Liu, Weidong Liu, and Wanxiang Che. 2024. Onebit: Towards extremely low-bit large language models. arXiv preprint arXiv:2402.11295.

Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. 2024. Tinyllama: An open-source small language model. arXiv preprint arXiv:2401.02385.

Yichen Zhu, Minjie Zhu, Ning Liu, Zhicai Ou, Xiaofeng Mou, and Jian Tang. 2024. Llava-phi: Efficient multimodal assistant with small language model. arXiv preprint arXiv:2401.02330.

