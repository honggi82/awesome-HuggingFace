# arXiv:2405.06694v1[cs.CL]7May2024

## SUTRA: SCALABLE MULTILINGUAL LANGUAGE MODEL ARCHITECTURE

A PREPRINT

#### Abhijit Bendale*

Two Platforms abhijit@two.ai

Michael Sapienza Two Platforms michael@two.ai

Steven Ripplinger Two Platforms steven@two.ai

Pranav Mistry Two Platforms pranav@two.ai

Simon Gibbs

Two Platforms simon@two.ai

Jaewon Lee Two Platforms jaewon@two.ai

May 14, 2024

### ABSTRACT

In this paper, we introduce SUTRA, multilingual Large Language Model architecture capable of understanding, reasoning, and generating text in over 50 languages. SUTRA’s design uniquely decouples core conceptual understanding from language-specific processing, which facilitates scalable and efficient multilingual alignment and learning. Employing a Mixture of Experts framework both in language and concept processing, SUTRA demonstrates both computational efficiency and responsiveness. Through extensive evaluations, SUTRA is demonstrated to surpass existing models like GPT-3.5, Llama2 by 20-30% on leading Massive Multitask Language Understanding (MMLU) benchmarks for multilingual tasks. SUTRA models are also online LLMs that can use knowledge from the internet to provide hallucination-free, factual and up-to-date responses while retaining their multilingual capabilities. Furthermore, we explore the broader implications of its architecture for the future of multilingual AI, highlighting its potential to democratize access to AI technology globally and to improve the equity and utility of AI in regions with predominantly non-English languages. Our findings suggest that SUTRA not only fills pivotal gaps in multilingual model capabilities but also establishes a new benchmark for operational efficiency and scalability in AI applications.

Keywords Multilingual · Online · Large Language Models

### 1 Introduction

Recent advancements in Large Language Models (LLMs) have predominantly focused on a limited set of data-rich languages, with training datasets being notably skewed towards English. This skew results in a significant bias, rendering LLMs less capable of understanding, processing, and generating text in languages with substantial speaker populations. Notably, languages such as Hindi, Arabic, Bengali, and Japanese, each with over 250 million speakers, constitute less than 3% of the data typically used for training these models. This enduring bias represents a substantial challenge, as it is not feasible for LLMs to underperform in languages spoken by vast numbers of people. In light of these disparities, our work seeks to bridge the gap between market demands and the current capabilities of LLMs. We strive to mitigate linguistic inequality inherent in recent multilingual Instruction-Following Task (IFT) models. Our goal is to develop a model adept at performing downstream tasks in any supported language, eliminating the need for multilingual speakers to default to English for prompts.

The advent of Large Language Models (LLMs) like GPT-3.5, BERT, and others has revolutionized the field of artificial intelligence, offering unprecedented capabilities in natural language understanding and generation [Brown et al., 2020,

∗Correspondence can be addressed to abhijit@two.ai

[Figure 1]

- Figure 1: SUTRA is a novel multilingual large language model architecture that is trained by decoupling concept learning from language learning. The input is processed through a multilingual concept encoder, followed by the concept model and finally through a multilingual concept decoder to generate the output response.

Devlin et al., 2018]. These models have been instrumental in a variety of applications, ranging from conversational agents to complex decision support systems. However, the vast majority of these models predominantly cater to English, which is not only limiting in terms of linguistic diversity but also in accessibility and utility across different geographic and cultural contexts [Jia et al., 2019].

Addressing the challenge, multilingual LLMs have been developed, but these models often suffer from significant trade-offs between performance, efficiency, and scalability, particularly when extending support across a broader spectrum of languages [Conneau et al., 2020]. The most common approach has been to train large universal models capable of understanding multiple languages. Yet, these models, such as BLOOM and Llama2, typically underperform in languages that are less represented in the training data due to the difficulty of balancing language-specific nuances [Smith et al., 2021, Zhang et al., 2020]. The development of SUTRA was motivated by the inherent limitations in existing multilingual LLMs. On the one hand there are language-specific LLMs like HyperClova in Korean or OpenHaathi in Hindi. Scaling and managing such models is not only costly, but challenging due to the exponential data and training requirements. Each time a new base model is created, it would require fine-tuning for many different languages. On the other hand large traditional LLMs like BLOOM and Llama2 struggle on multilingual tasks, as they have to balance learning core multilingual capabilities and skills, often resulting in confusion between languages. For example, when asking GPT a question in Korean, one might notice how formal and informal tones are often misplaced. SUTRA was developed to address two main challenges of existing multilingual LLMs: the high computational/scaling costs of language-specific models, and the difficulties larger models face with multilingual tasks (leading to language confusion).

In response to these limitations, we introduce SUTRA (Sanskrit for "thread"), a transformative approach in the architecture of multilingual LLMs. SUTRA uniquely separates the process of concept learning from language learning, as illustrated in Figure 1. SUTRA is a novel multilingual large language model architecture that is trained by decoupling concept learning from language learning. This architecture enables the core model to focus on universal languageagnostic concepts while leveraging specialized neural machine translation (NMT) mechanisms for language-specific processing, thus preserving linguistic nuances without compromising the model’s scalability or performance [Wu et al., 2019]. SUTRA employs a Mixture of Experts (MoE) strategy, enhancing the model’s efficiency by engaging only the relevant experts based on the linguistic task at hand [Shazeer et al., 2017]. Furthermore, SUTRA models are internet-connected and hallucination-free models that understand queries, browse the web, and summarize information to provide the most current answers, without loosing their multilingual capabilities. A combination of multilingual skills, online connectivity, and efficiency in language generation incorporated by SUTRA models promises to redefine the landscape of multilingual language modeling.

In the subsequent sections, we will outline the architecture of SUTRA, our training methodology, and present a comprehensive evaluation that demonstrates its superiority over contemporary multilingual models on several benchmarks, including the Massive Multitask Language Understanding (MMLU) tasks [Hendrycks et al., 2021]. By effectively decoupling concept learning from language processing, SUTRA sets a new paradigm in the development of LLMs, promising broader accessibility and enhanced performance across diverse linguistic landscapes.

The paper is organized as follows: First, we discuss related work in the context of SUTRA. Next, we describe the architecture and training methodology adopted. We then discuss the data used for training and provide both an evaluation

of SUTRA’s multilingual as well as online cabailities. Finally, we discuss how to build more inclusive LLMs for the benefit of a wider community.

### 2 Related Work

Large Language Models & Multilinguality: The field of Large Language Models (LLMs) has witnessed substantial advancements, particularly through the development of models such as GPT-3 [Brown et al., 2020] and BERT [Devlin et al., 2018], which have set new benchmarks in language understanding and generation. These models utilize vast amounts of data to learn complex patterns and generate coherent text, but their primary limitation has been a focus largely on English language data. In response to the need for supporting global linguistic diversity, research has expanded into multilingual LLMs. Pioneering works like mBERT [Devlin et al., 2018] and XLM-R [Conneau et al., 2020] have demonstrated significant potential in learning representations that generalize across languages. However, these models often face challenges in balancing performance across languages, especially for those less represented in training datasets [Conneau et al., 2020]. Further, as the number of languages increases, the scalability and efficiency of these models often degrade, necessitating more specialized architectures to handle the diversity of languages effectively [Smith et al., 2021].

Neural Machine Translation: Neural Machine Translation (NMT) has been integral to the progress in multilingual model performance. Early NMT systems were limited by the complexity of their architectures and the quality of their translations, especially in low-resource languages [Wu et al., 2019]. Recent studies have revisited the core challenges of machine translation in the context of advanced Large Language Models (LLMs). The work by Koehn and Knowles [2017] offers insights into the ongoing relevance of challenges such as domain mismatch, rare word prediction, and translation of long sentences, even as LLMs have shown significant improvements in these areas. Additionally, a study by Son and Kim [2023] explored the translation performance of LLMs from the user’s perspective, highlighting their potential to enhance the translation of long sentences while also identifying persistent challenges around domain mismatch and rare word prediction. The work by Wu et al. [2016] on Google’s neural machine translation system has also served as a benchmark for progress in this field, bridging the gap between human and machine translation. Recently, the work by Costa-jussà et al. [2022] showed that the Mixture of Experts architecture can be used effectively in the context of Neural Machine Translation and have considerable gains in translation performance on various low-resource languages.

Mixture of Experts: Mixture of Experts (MoE) has emerged as a promising architecture for managing the computational costs associated with scaling up large language models (LLMs). Recent studies have explored the benefits of MoE in this context. Zhou et al. [2022] proposed a Mixture-of-Experts with Expert Choice Routing, which enables dynamic allocation of data among different experts, allowing each expert to focus on its expertise and achieve model sparsity. Similarly, Zoph [2022] investigated the design of effective sparse expert models, highlighting the importance of carefully balancing the number and size of experts to optimize performance. Additionally, Ott et al. [2022] introduced the OPT family of open pre-trained transformer language models, which leverage MoE to achieve significant improvements in efficiency and scalability compared to dense models. Furthermore, Zheng et al. [2019] explored the application of MoE in the context of Chinese idiom datasets, demonstrating the potential of this approach to enhance language understanding tasks. These studies collectively suggest that MoE can serve as an effective choice for building highly capable and computationally efficient LLMs.

Multimodal LLMs: Researchers have also explored the potential of multimodal Large Language Models that can process and generate content across different modalities, such as text, images, and video. For example, the work by Dai et al. [2019] has investigated the use of multimodal models for tasks like image captioning and visual question answering, demonstrating their ability to leverage cross-modal information to enhance performance. Similarly, the study by Nichols and Warnow [2008] has explored the application of multimodal models in the context of computational linguistic phylogeny, highlighting their potential to uncover insights from diverse data sources. Additionally, the recent advancements in the field of multimodal machine translation, as discussed by Birch [2021], have shown the benefits of integrating visual information into language models to improve translation quality.

Online LLMs: Modern Large Language Models like Llama2, GPT-3.5, and GPT-4 have been engineered as comprehensive, open-domain chatbots capable of engaging in extended dialogues on a variety of topics. Yet, they face a significant limitation: their data is time-locked, leading to a cutoff date for knowledge. Due to this, these models sometimes generate responses that are plausible yet factually incorrect, diminishing the reliability of their output as noted by Vu et al. [2023] and Press et al. [2022] and such inaccuracies are often linked to outdated information embedded in the model’s parameters. A detailed list of knowledge cutoff dates for major models is shown in Table 1. While this can be somewhat rectified through additional training with human feedback or by incorporating knowledge-intensive tasks, scaling these solutions to accommodate real-time updates, such as changes in stock prices, remains challenging [Komeili

#### Model Name Creator License Context Window Training Cut-off Date

GPT-4 OpenAI Proprietary 8k Sep 2021 GPT-3.5 Turbo OpenAI Proprietary 16k Sep 2021 Llama 2 Chat (70B) Meta Open 4k Sep 2022 Mixtral 8x7B Instruct Mistral Open 32K Sep 2023 SUTRA Two Platforms Proprietary 32K Up-to-Date

Table 1: Comparison of various AI models for their knowledge cut-off dates. The knowledge cutoff represents the latest point at which the language model was updated with new information, beyond which it lacks any further data or recent developments. Online models like SUTRA have the ability to continuously learn and reason from recent data.

et al., 2021]. In-context learning presents a promising alternative, allowing for the incorporation of real-time data directly into the model’s prompts to guide response generation. Although there are ongoing efforts to enhance LLMs with internet search results, effectively leveraging this external data to improve the accuracy of LLM outputs is still under development. In this context, SUTRA stands out by presenting a structured approach for response augmentation, providing the ability to learn, reason, and interpret information from various knowledge sources.

### 3 SUTRA Approach

#### 3.1 What is SUTRA?

SUTRA is a novel multilingual large language model architecture that is trained by decoupling concept learning from language learning. Inspired by how humans learn, SUTRA decouples core concept learning from language learning, making it scalable and easier to reach large number of languages. Humans first understand the world through concepts and then gradually learn their native language. Once fluent in one language, they learn new languages without having to re-learn common core concepts. Similarly, central to our approach is the innovative strategy of separating concept learning from language learning. This enables the core LLM capabilities to operate within a conceptual or latent space, while the heavy lifting of tokenization and translation is handled by specialized encoders and decoders inspired from Neural Machine Translation. This approach makes training of LLMs more scalable, whilst making it easier to reach a larger number of languages.

Our training methodology unfolds in three phases: concept learning, language learning and language alignment.

- • Concept Learning: Initially, the core concept model undergoes training to grasp concepts within a small set of languages, setting a solid foundation for understanding basic concepts and skills.
- • Language Learning: In parallel, we train specialized Neural Machine Translation (NMT) based encoders and decoders, alongside a multilingual tokenizer, specifically designed to master multi-language translation and ensure concept consistency across languages.
- • Language Alignment: Finally, we perform language alignment, merging concept understanding with linguistic proficiency.

In the inference stage, SUTRA employs a structured path: Input is processed through an NMT Encoder, followed by the Concept Model, and finally through the NMT Decoder to produce the output.

#### 3.2 Architecture

The architecture of our model, referred to herein as SUTRA, is built upon the foundational principles of the transformer architecture as delineated by Vaswani et al. [2017]. Our model retains the enhancements specified by Jiang et al. [2023], with the critical adaptation that it facilitates an extended dense context length of up to 32k tokens. Moreover, we have employed MoE layers, enabling selective activation of experts and leading to efficiency in computation and memory consumption, as shown in Figure 2. The key architectural parameters of SUTRA are encapsulated in Table 2.

Given an input x, the output yielded by the Expert Mixture module is the sum of each expert network’s contribution, modulated by the gating network. Formally, for n experts {E0,E1,...,En−1}, the resultant output is:

n−1

G(x)i · Ei(x) (1)

i=0

[Figure 2]

- Figure 2: Expert Mixture Layer Configuration. Input vectors are routed to a subset of the available experts, specifically 2 out of 8, by a specialized router. The aggregate output of this layer is the sum of the individual outputs, each weighted accordingly. Each expert comprises a feedforward module similar to those found in conventional transformer models.

#### Parameter Name Parameter Value Parameter Name Parameter Value

Model Dim 1024 Context Window 8K, 32K LLM Layers 32 Batch Size 1M Tokens Attention Heads 32 FFN Dim 14336 # of Experts 8 Language Enc. Attn Heads 16 # of top Experts 2 Language Dec. Attn Heads 16

Table 2: The above table shows some selected model parameters for SUTRA.

where G(x)i represents the gating function’s output, producing an n-dimensional vector corresponding to the i-th expert’s activation, while Ei(x) delineates the i-th expert network’s output. The model capitalizes on sparsity by disregarding inactive experts, thereby conserving computational resources. Several mechanisms for constructing the gating function G(x) exist [Clark et al., 2022, Hazimeh et al., 2021, Zhou et al., 2022]; however, our implementation opts for the efficient approach of selecting the Top-K values from a linear projection, followed by a softmax operation [Shazeer et al., 2017]:

G(x) = Softmax(TopK(xWg)) (2)

in which TopK preserves the highest K values from the logit vector ℓ ∈ Rn, assigning them as −∞ otherwise. The choice of K, indicative of the active experts per token, is a strategic hyperparameter influencing the computational expenditure per token. Elevating the number of experts n while maintaining a constant K allows for an augmentation of the model’s capacity without a proportionate rise in computational overhead. This strategy creates a distinction between the model’s total parameters or the sparse parameter count, and the activeparameter count, which relates directly to K and determines the number of parameters actively employed per token.

#### 3.3 Training Data

The key to our language training strategy lies in leveraging linguistic commonalities during the language learning phase. For example, Hindi has a lot more commonalities in terms of semantics, grammar and cultural context with Gujarati or Bengali as compared to Nordic languages.

The limited multilingual capabilities of large language models (LLMs) stem from an uneven data distribution favoring a handful of well-resourced languages. Multilingual data in machine translation is task-specific and misses key training areas for LLMs, such as conversation, summarization, and instruction-following. To address this, the SUTRA dataset includes over 100 million conversations in real and synthetically translated pairs across various languages, supplemented by publicly available datasets for a comprehensive training landscape. Past research has demonstrated synthetic data’s role in fostering reasoning, code generation, and task complexity learning in LLMs, as well as enhancing cross-lingual transfer with multilingual synthetic data [Lai et al., 2023, Whitehouse et al., 2023]. Following this insight, we adopt a methodical use of abundant data from languages like English to facilitate concept learning. During the language learning and alignment phases of multilingual training, we employ a combination of real and synthetic data to bolster

[Figure 3]

- Figure 3: The same concepts (umbrella, house, dog) when expressed in different languages (English, Hindi) can be mapped to quite different embedding vectors (left). In order to achieve multilingual encoders and decoders which map concepts in different languages to a common concept space, the these embedding vectors need to be aligned (middle). It can be seen that after the multilingual concept alignment stage, the same concepts (umbrella, house, dog) are now mapped to similar embedding vectors, even though they are expressed in different languages (right). Our specialized Neural Machine Translation (NMT) based encoders and decoders, can apply the same principle to master multi-language translation and ensure concept consistency across languages.

[Figure 4]

- Figure 4: Conversation Data Topic Distribution. In the following plot we are are showing topic distribution of over 1M sampled conversations. Inspection of cluster centroids reveals that this is a rich and diverse data covering wide range of topics.

and broaden our training framework. An illustraion and description of the matching and alignment process is shown in Figure 3.

In Figure 4 we show the topic distribution of over 1M sampled conversations. Inspection of cluster centroids reveals that this is a rich and diverse data covering a wide range of topics.

The purpose-built multi-language tokenizers efficiently represent each language. They are trained on cross-lingual data and finely tuned with the base model, setting a new benchmark in multilingual language modeling. One of the most critical aspects of having good performance of conversational LLMs is high-quality instruction fine-tuning (IFT) datasets. Majority of IFT datasets are in English. We use Neutral Machine Translation (NMT) to translate the

Dataset Number # Users Avg. Turns / Sample Avg. Tokens / Prompt Anthropic HH 338,704 143 2.3 19 OpenAssistant 66,497 13,500 - 37 Chatbot Arena 33,000 13,383 2.1 53 SUTRA dataset 10M+ 350,000 15 120

- Table 3: In this table, statistics of various leading conversation datasets are shown such as Anthropic HH [Bai et al., 2022], OpenAssistant Conversations [Köpf et al., 2023], LMSys [Chiang et al., 2024] and the SUTRA dataset. The tokens are counted using Llama2 tokenizer [Touvron et al., 2023] for public datasets and for SUTRA dataset using SUTRA’s tokenizer. One of the key aspects of our dataset is having long term and multi-turn conversations.

instructions, inputs and outputs from different datasets, to ensure balanced representation across tasks, in multiple Indian and non-English languages. Overall, we prepare more than 100M training samples from languages like English, Hindi, Tamil, Korean etc. with wide ranging datasets such as our internal SUTRA dataset, as well as open-source FLAN-v2, OpenAssistant and wikiHow. The translated examples are filtered to retain high-quality examples. Note that our internal data includes long-term and multi-turn conversational data, which helps to tune it towards better human-AI conversations and interactions. A comparison and detailed description of the dataset is shown in Table 3.

4 Training Multilingual Tokenizers

Tokenization, a critical step in NLP pipeline, involves converting text into a sequence of tokens, where each token represents a subword or word. Although English specific tokenizers can generate text in non-English languages, they don’t capture language specific nuances and are highly inefficient in other languages, especially non-Romanized languages. More specifically for Indian languages like Hindi, Gujarati, or Tamil, we note that tokenizers from leading LLMs like Llama-2, Mistral, and GPT-4 consume 4.5X to 8X more tokens compared to English, as shown in Table 4.

Model/Tok. English Hindi Gujarati Tamil Korean

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

SUTRA 19 18 23 22 20 GPT-4 18 82 137 180 100 Llama2 22 98 200 163 68 Mistral 19 93 165 157 53

- Table 4: Number of tokens per language for models (Note: Lower is better). SUTRA Models use these multilingual tokenizers to get quality, performance, and efficiency.

A key step in adding language specific skills is decreasing the average number of tokens a word is split into (also known as token fertility) by a language model on non-english text. This makes inferencing efficient as well as semantically meaningful. We train the sentence-piece tokenizer from a large corpus of multi-language dataset of 500K+ documents, which is then merged with a pre-trained english tokenizer to increase the vocabulary size. Text generated with our tokenizers lead to 80% to 200% reduction in overall tokens consumed across languages, which is critical for bringing down the cost of inferencing when deploying these models for cost-sensitive use-cases.

- 5 Multilingual MMLU

#### 5.1 Massive Multitask Language Understanding

We evaluate our model on a variety of NLU and NLG tasks. To test the knowledge and reasoning capabilities of the model, we evaluate on the machine-translated version of the benchmarks such as MMLU [Hendrycks et al., 2021]. The Massive Multitask Language Understanding (MMLU) benchmark is a comprehensive and challenging evaluation

Model Name en hi gu ta ko bn pa mr te ml ar ja SUTRA 77 68 67 67 67 68 67 69 68 68 67 75 Okapi - 26 27 26 - 26 - 26 26 25 27 mT0 - 32 29 29 - 31 - 31 29 29 31 mT0X - 31 29 27 - 30 - 29 27 27 31 Aya - 39 33 21 - 36 - 36 39 32 38 -

- Table 5: The above table shows comparison with recent purpose built multilingual language models such as those proposed by Üstün et al. [2024], Lai et al. [2023]. SUTRA provides strong multilingual performance compared to many leading purpose built multilingual language models by significant margin.

framework designed to test the capabilities of Large Language Models (LLMs) across a wide array of tasks. It was created with the goal of pushing the boundaries of what LLMs can understand and how well they can adapt to various domains of knowledge. The benchmark covers 57 subjects across STEM, the humanities, the social sciences, and more. It ranges in difficulty from an elementary level to an advanced professional level, and it tests both world knowledge and problem solving ability. Subjects range from traditional areas, such as mathematics and history, to more specialized areas like law and ethics. The granularity and breadth of the subjects makes the benchmark ideal for identifying a model’s blind spots. This diversity ensures that models are not only proficient in a broad spectrum of topics but also capable of generalizing their understanding to new and unseen domains. The MMLU evaluates models on their ability to answer multiple-choice questions, requiring nuanced comprehension and the application of reasoning, which collectively serve as a measure of an LLM’s depth of knowledge and its interpretive skills.

#### 5.2 Extending MMLU to Multiple Languages

To assess our models’ effectiveness in various tasks and across multiple languages, we developed a multilingual evaluation suite that broadens the scope of evaluation linguistically. We utilized the multilingual assessment framework suggested by Lai et al. [2023] and Üstün et al. [2024], with certain distinctions. Notably, while Okapi uses a 25-shot evaluation, our methodology employs a 5-shot evaluation as per the original benchmark by Hendrycks et al. [2021]. We anticipate that a 5-shot evaluation, offering fewer examples, presents a more challenging benchmark. Recognizing the existence of over 200 major languages globally, our evaluation focuses on three distinct language groups: English, Korean, Japanese, Arabic, and Indian Languages. Although this selection is not exhaustive, it encompasses a significant portion of linguistic diversity, enabling thorough analysis of the models’ multilingual capabilities. These languages represent a substantial demographic, accounting for more than half of the global population as primary or secondary speakers. Additionally, they are key languages in global business, ensuring our evaluation has broad relevance.

#### 5.3 Consistent Performance across Languages

The SUTRA model demonstrates a notable consistency in linguistic performance across a variety of languages, as evidenced by the MMLU benchmark results. It exhibits a minimal performance deviation from its English language results to other languages such as Hindi, Gujarati, and Arabic, highlighting its robust multilingual capabilities critical for applications on a global scale.

Superior concept and language modeling underpin the SUTRA model’s ability to maintain performance levels across different languages, distinguishing it from other leading models, including GPT-4, GPT-3.5, and Llama2. Many existing model architectures (including purpose built multilanguage models) experience a pronounced decline in performance in non-English languages, often regressing to baseline random chance performance, as detailed in Table 5. Note that random chance performance is at 25% on the MMLU benchmark. In contrast, SUTRA consistently achieves stable scores across languages, setting it apart, particularly in languages that are less commonly represented in language models, such as Hindi, Gujarati, Tamil, and Korean. The SUTRA model, therefore, not only excels in individual language performance but also promotes a more universal, language-agnostic approach to AI. It serves as a robust solution for international businesses, educational platforms, and cross-cultural communication, setting a new benchmark for LLMs in a multi-lingual, interconnected world.

#### 5.4 Comparing with leading models for Multilingual Performance

For our evaluation, we use multiple state of the art models and compare their performance on the multilingual MMLU benchmark, as shown in Table 6. We considered multiple leading models such as GPT-4 and GPT-3.5 from OpenAI,

#### Language SUTRA LL3 70B LL2 70B GPT 4 GPT 3.5 Mixt. 8x22B Mixt. 8x7B HCX PPLX

English 77 82 63 86 70 77 70 66 62 Hindi 68 64 31 71 39 38 35 39 32 Korean 67 60 38 72 51 56 46 54 40 Gujarati 67 54 29 61 35 29 29 36 26 Tamil 67 52 29 44 30 34 29 33 27 Bengali 68 58 27 73 36 37 33 – – Punjabi 67 55 26 71 34 29 30 – – Marathi 69 62 25 66 32 36 32 – – Telugu 68 53 24 62 32 32 28 – – Arabic 67 60 48 80 49 48 39 – – Japanese 75 70 56 80 57 60 51 – –

- Table 6: The table shows multilingual performance of various leading models on MMLU benchmark for multiple languages. SUTRA has competitive performance in English while maintaining strong multilingual performance in other languages. Many leading language models’ MMLU scores for non-english languages falls close to random chance (25% is random chance on MMLU task).

Mixtral-8x7b from Mistral, Llama2-13b, Llama2-70b and Llama3-70b from Meta, sonar-medium from Perplexity, HyperClovaX from Naver, and Airavata Model from Sarvam AI. Of these, GPT-4, GPT-3.5, Mixtral, Llama series and Perplexity are generic models i.e. they were not trained to optimize for specific languages. HyperClovaX was specifically trained to optimize performance on the Korean language, whilst Airavata was specifically trained to optimize performance in Hindi.

Overall, the evaluation results demonstrate that our SUTRA models can match and even outperform GPT-3.5 and Llama-7b on TWO-related use cases, particularly for providing natural and engaging responses across languages. Although GPT-4 is still state-of-the-art in terms of performance, cost continues to be a major hindrance for wide-scale deployment in cost-sensitive markets. Surpassing GPT-3.5 multilingual performance by 20-30% on the leading MMLU benchmark, SUTRA models excel in comprehending and generating responses across numerous languages. We find that SUTRA does well even compared to models that were specifically optimized for a particular language, showing promise for the approach followed by SUTRA, as shown in Table 7. More detailed results showing MMLU scores across groups of categories such as STEM, humanities etc. are listed in Table 8.

Language Organization Model Name MMLU SUTRA MMLU

Hindi Sarvam Airavata 35 68 Korean Naver HyperClovaX 54 67 Arabic Inception / MBZUAI Jais 34 67 Japanese Rakuten Rakuten-7B 61 75

- Table 7: The above table shows comparison of language specific LLMs for multiple languages such as Hindi [Gala et al., 2024], Korean [Son et al., 2024], Arabic [Sengupta et al., 2023] and Japanese [Group et al., 2024]. The selected models are best performing models on respective languages, as they were purposely built and tuned for those languages. Shown on the right is MMLU score for SUTRA on respective languages. From the performance numbers it is evident that the concept and language modeling approach followed by SUTRA yields superior multilingual performance.

### 6 Quantitative Evaluation for Real-Time Queries

SUTRA models are connected, up-to-date, and hallucination-free models that provide factual responses with a conversational tone. They are online LLMs that use, infer, and process real-time knowledge from the internet and leverage it to provide the most up-to-date information when forming responses. SUTRA-Online models can accurately respond to time-sensitive queries, extending its knowledge beyond a static training corpus. Online models can therefore accurately answer questions like "Who won the game last night” or “What’s the most popular movie right now?”.

We evaluated the SUTRA models using the Fresh Prompt framework [Vu et al., 2023], developed by Google for assessing online LLMs [Press et al., 2022], and discovered that SUTRA-Online models surpass the competing search

#### Language STEM Social Sci. Humanities Other ∼Average

English 69.78 80.83 76.08 79.07 76.24 Hindi 61.67 74.17 69.23 68.93 67.81 Gujarati 62.78 70.83 66.92 66.79 66.40 Marathi 59.72 75.42 73.46 71.07 68.95 Bengali 60.83 74.17 66.54 73.57 68.07 Tamil 61.39 71.67 64.62 70.71 66.58 Punjabi 59.72 73.75 69.62 68.21 67.02 Korean 57.78 72.08 67.69 70.30 66.14 Arabic 59.72 75.83 63.85 68.93 66.32 Japanese 66.67 81.67 76.15 78.57 74.91

- Table 8: SUTRA quantitative MMLU results across a subset of supported languages for fine-grained tasks on the MMLU benchmark.

Model Name Knowledge Cut. FreshLLM Dataset all all fast slow never ≥ 2022 1-hop

Google Search Up to Date 04/26/2023 39.6 48.9 32 46.4 68.3 37.9 55.6 GPT-3.5 2021 04/26/2023 26 26.1 4 15.2 58.7 5.1 28 Google DeepMind Up to Date 04/26/2023 56 62.5 46.4 60.8 80.2 57 68.7 Perplexity AI Up to Date 04/26/2023 52.2 57.2 38.4 53.6 79.4 47.7 63.8 SUTRA-Online Up to Date 04/15/2024 56 63.8 47.7 61.6 88.7 59.1 70.4

- Table 9: Performance Comparison of Language Models for handling fresh (realtime queries) with valid premise according to freshness LLM benchmark from Vu et al. [2023]

engine-augmented models from Google, as well as OpenAI’s GPT-3.5 and Perplexity AI. The benchmark contains exhaustive questions covering various nuanced online scenarios covering never-changing, in which the answer almost never changes; slow-changing, in which the answer typically changes over the course of several years; fast-changing, in which the answer typically changes within a year or less. SUTRA performed well across majority of these scenarios, as shown in Table 9.

### 7 Discussion and Conclusion

Looking ahead, the SUTRA paves the way for the development of phonetic models (approach for SUTRA-Dhvanim), which benefits from the clear separation between concept modeling and language learning. By replacing the NMT decoder with a phonetic decoder, we enable the generation of phonetic responses for more seamless integration with speech models. Our next frontier for optimization is to examine the accuracy and performance impact of structured sparsity and int4 precision, which could significantly reduce SUTRA’s GPU memory footprint and yield with improvements in latency.

This research has introduced SUTRA, a state-of-the-art multilingual conversational language model, showcasing its superior ability to handle multiple languages with remarkable efficiency and performance. SUTRA is already proficient in 31 languages across multiple tasks, as detailed in Table 10, and is being extended to support over 50 languages. Unlike its predecessors, which struggle with the nuanced requirements of multi-language understanding, SUTRA exhibits a robust proficiency that is evident across a range of linguistic contexts. This is particularly notable in its application to languages with fewer resources available for model training, which traditionally lag in performance metrics. The innovative architecture of SUTRA, with its decoupled concept and language processing, allows for a scalable and flexible approach to language model training. This not only opens the door for more equitable representation of less commonly spoken languages but also ensures that the quality of interaction remains high across all languages. The efficient tokenization strategy of SUTRA, reducing token fertility for non-English languages, also points to potential cost reductions in deploying AI in multi-language environments, a notable consideration for global accessibility.

In conclusion, SUTRA sets a new precedent for multilingual language models by delivering high performance and efficiency without sacrificing linguistic diversity. Its architecture, which mirrors human cognitive development by separating concept understanding from linguistic expression, allows for a more natural and extensive language

#### Language ISO Code Language ISO Code

English en Korean ko French fr Japanese ja Italian it Thai th Spanish es Arabic ar German de Persian fa Portuguese pt Vietnamese vi Hindi hi Indonesian id Bengali bn Turkish tr Marathi mr Polish pl Telugu te Russian ru Tamil ta Ukranian uk Gujarati gu Dutch nl Kannada kn Greek el Malayalam ml Punjabi pa Assamese as Urdu ur Odia or

- Table 10: Although SUTRA can support more than 50 languages, the languages listed in the table above are the ones we have tested across number of tasks. Support for additional languages will be released in next versions of SUTRA

comprehension. This breakthrough bears significant implications for the global adoption and application of AI, paving the way for more inclusive and equitable access to technology across language barriers.

### References

Tom B. Brown et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional

transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. Roger Jia et al. Bias in multilingual models: The case for linguistic equity in ai. In Conference on Neural Information Processing Systems (NeurIPS), 2019. Alexis Conneau et al. Unsupervised cross-lingual representation learning at scale. arXiv preprint arXiv:1911.02116,

- 2020.

Linda Smith et al. Can multilingual models transfer for less resourced languages? Language Resources and Evaluation,

- 2021.

Yiming Zhang et al. Improving multilingual models with language-clustered vocabularies. arXiv preprint arXiv:2007.07680, 2020.

Yonghui Wu et al. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2019.

Noam Shazeer et al. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint

arXiv:1701.06538, 2017. Dan Hendrycks et al. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2021. Philipp Koehn and Rebecca Knowles. Six challenges for neural machine translation. arXiv preprint arXiv:1706.03872,

2017. Jungwoo Son and Byeongil Kim. Translation performance from the user’s perspective of large language models and neural machine translation systems. Information, 14(10):574, 2023.

Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.

Marta R Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, et al. No language left behind: Scaling human-centered machine translation. arXiv preprint arXiv:2207.04672, 2022.

Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M Dai, Quoc V Le, James Laudon, et al. Mixture-of-experts with expert choice routing. Advances in Neural Information Processing Systems, 35: 7103–7114, 2022.

Barret Zoph. Designing effective sparse expert models. IEEE International Parallel and Distributed Processing Symposium (IPDPS), 2022.

Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, P Sai Koura, Abhinav Sridhar, Tao Wang, and Luke Zettlemoyer. Opt: Open pre-trained transformer language models. 2022.

Chujie Zheng, Minlie Huang, and Aixin Sun. Chid: A large-scale chinese idiom dataset for cloze test. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 778–787, 2019. doi:10.18653/v1/P19-1075. URL https://doi.org/10.18653/v1/p19-1075.

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G Carbonell, Quoc Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860, 2019.

Johanna Nichols and Tandy Warnow. Tutorial on computational linguistic phylogeny. Language and Linguistics

Compass, 2(5):760–820, 2008. Alexandra Birch. Neural Machine Translation. Cambridge University Press, 2021. Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou,

Quoc Le, and Thang Luong. Freshllms: Refreshing large language models with search engine augmentation, 2023. Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. Measuring and narrowing the

compositionality gap in language models. arXiv preprint arXiv:2210.03350, 2022. Mojtaba Komeili, Kurt Shuster, and Jason Weston. Internet-augmented dialogue generation. arXiv preprint

arXiv:2107.07566, 2021. Ashish Vaswani et al. Attention is all you need. Advances in neural information processing systems, 30, 2017. Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las

Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

Aidan Clark, Diego De Las Casas, Aurelia Guy, Arthur Mensch, Michela Paganini, Jordan Hoffmann, Bogdan Damoc, Blake Hechtman, Trevor Cai, Sebastian Borgeaud, et al. Unified scaling laws for routed language models. In International Conference on Machine Learning, pages 4057–4086. PMLR, 2022.

Hussein Hazimeh, Zhe Zhao, Aakanksha Chowdhery, Maheswaran Sathiamoorthy, Yihua Chen, Rahul Mazumder, Lichan Hong, and Ed Chi. Dselect-k: Differentiable selection in the mixture of experts with applications to multi-task learning. Advances in Neural Information Processing Systems, 34:29335–29347, 2021.

Viet Dac Lai, Chien Van Nguyen, Nghia Trung Ngo, Thuat Nguyen, Franck Dernoncourt, Ryan A Rossi, and Thien Huu Nguyen. Okapi: Instruction-tuned large language models in multiple languages with reinforcement learning from human feedback. arXiv preprint arXiv:2307.16039, 2023.

Chenxi Whitehouse, Monojit Choudhury, and Alham Fikri Aji. Llm-powered data augmentation for enhanced crosslingual performance. arXiv preprint arXiv:2305.14288, 2023.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, Abdullah Barhoum, Nguyen Minh Duc, Oliver Stanley, Richárd Nagyfi, et al. Openassistant conversations–democratizing large language model alignment. arXiv preprint arXiv:2304.07327, 2023.

Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Hao Zhang, Banghua Zhu, Michael Jordan, Joseph E Gonzalez, et al. Chatbot arena: An open platform for evaluating llms by human preference. arXiv preprint arXiv:2403.04132, 2024.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Ahmet Üstün, Viraat Aryabumi, Zheng-Xin Yong, Wei-Yin Ko, Daniel D’souza, Gbemileke Onilude, Neel Bhandari, Shivalika Singh, Hui-Lee Ooi, Amr Kayid, et al. Aya model: An instruction finetuned open-access multilingual language model. arXiv preprint arXiv:2402.07827, 2024.

Jay Gala, Thanmay Jayakumar, Jaavid Aktar Husain, Aswanth Kumar M, Mohammed Safi Ur Rahman Khan, Diptesh Kanojia, Ratish Puduppully, Mitesh M. Khapra, Raj Dabre, Rudra Murthy, and Anoop Kunchukuttan. Airavata: Introducing hindi instruction-tuned llm. arXiv preprint arXiv: 2401.15006, 2024.

Guijin Son, Hanwool Lee, Sungdong Kim, Seungone Kim, Niklas Muennighoff, Taekyoon Choi, Cheonbok Park, Kang Min Yoo, and Stella Biderman. Kmmlu: Measuring massive multitask language understanding in korean. arXiv preprint arXiv:2402.11548, 2024.

Neha Sengupta, Sunil Kumar Sahu, Bokang Jia, Satheesh Katipomu, Haonan Li, Fajri Koto, Osama Mohammed Afzal, Samta Kamboj, Onkar Pandit, Rahul Pal, et al. Jais and jais-chat: Arabic-centric foundation and instruction-tuned open generative large language models. arXiv preprint arXiv:2308.16149, 2023.

Rakuten Group, Aaron Levine, Connie Huang, Chenguang Wang, Eduardo Batista, Ewa Szymanska, Hongyi Ding, Hou Wei Chou, Jean-François Pessiot, Johanes Effendi, et al. Rakutenai-7b: Extending large language models for japanese. arXiv preprint arXiv:2403.15484, 2024.

### About Two Platforms

Two Platforms (TWO) is a tech startup that aims to redefine Human-AI Interaction, and is at the forefront of the next generation of AI that is visual and immersive. TWO is building consumer AI apps and services powered by its proprietary Gen-AI models. TWO is headquartered in Silicon Valley with offices in Seoul and Mumbai.

