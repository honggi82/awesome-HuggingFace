# CKnowEdit: A New Chinese Knowledge Editing Dataset for Linguistics, Facts, and Logic Error Correction in LLMs

Jizhan Fang1* Tianhe Lu1,* Yunzhi Yao1 Ziyan Jiang1 Xin Xu2 Huajun Chen1 Ningyu Zhang1† 1Zhejiang University 2University of California, San Diego {fangjizhan, yyztodd, zhangningyu}@zju.edu.cn xinxucs@ucsd.edu

arXiv:2409.05806v4[cs.CL]1Jun2025

## Abstract

Chinese, as a linguistic system rich in depth and complexity, is characterized by distinctive elements such as ancient poetry, proverbs, idioms, and other cultural constructs. However, current Large Language Models (LLMs) face limitations in these specialized domains, highlighting the need for the development of comprehensive datasets that can assess, continuously update, and progressively improve these culturally-grounded linguistic competencies through targeted training optimizations. To address this gap, we introduce CKnowEdit, the first-ever Chinese knowledge editing dataset designed to correct linguistic, factual, and logical errors in LLMs. We collect seven types of knowledge from a wide range of sources, including classical texts, idioms, and content from Baidu Tieba Ruozhiba, taking into account the unique polyphony, antithesis, and logical structures inherent in the Chinese language. By analyzing this dataset, we highlight the challenges current LLMs face in mastering Chinese. Furthermore, our evaluation of state-of-the-art knowledge editing techniques reveals opportunities to advance the correction of Chinese knowledge1.

## 1 Introduction

The reliance on static training data and the lack of explicit knowledge representation in Large Language Models often lead to issues such as hallucinations, bias, and offensive outputs (Zhao et al., 2023; Huang et al., 2023a; Liu et al., 2023; Sun et al., 2024b; Chen, 2024). These limitations become particularly pronounced when LLMs operate in complex domains or languages, such as Chinese. As shown in Figure 1, Chinese is a highly complex and linguistically unique system and presents three

*Equal contribution and shared co-first authorship. †Corresponding Author.

- 1Code and dataset are available at https://github.com/

zjunlp/EasyEdit.

distinct challenges for LLMs versus Indo-European languages (Luelsdorff, 1994; Matthiessen, 2023; Xu et al., 2023; Wen et al., 2023): (i) Linguistic Complexity: Characters intricately blend shape, sound, and meaning through composition and contextual pronunciation shifts, while flexible grammar and cultural elements (poetry, idioms, etc.) evolved over millennia. (ii) Culture-Laden Facts: Untranslatable contexts in specific facts like geographical/historical terms. (iii) LanguageSpecific Logic: Context-dependent reasoning patterns that rely on implicit connectors and topic prominence over subject-predicate structures, often leading to misalignment in logical chain extraction.

In this work, we propose to correct Chinese knowledge errors in LLMs via knowledge editing (Yao et al., 2023; Wang et al., 2023b; Zhang et al.,

- 2024a; Hu et al., 2024; Ni et al., 2023; Wei et al.,
- 2024b; Wang et al., 2024c; Padmanabhan et al.,

- 2023; Qiao et al., 2024; Chen et al., 2024; Li et al.,
- 2024; Hase et al., 2024; Wu et al., 2024a; Wang et al., 2024c). Nevertheless, current research on knowledge editing predominantly concentrates on English-language factual knowledge (Cao et al., 2021; Meng et al., 2022; Wu et al., 2024b), derived from Wikipedia, which introduces an Anglo-centric bias. Recently, there have been some multilingual datasets (Wang et al., 2023a; Xie et al., 2024; Wei et al., 2024a; Nie et al., 2024) attempting to explore editing methods for different languages. However, these datasets are often created by translating the English corpus into another language, and translation (Vanmassenhove et al., 2019; Berman and Venuti, 2021) has been shown failing to capture the intricate linguistic features and cultural nuances inherent to special language, resulting in a loss of lexical richness and diversity. Meanwhile, these works are primarily designed to assess the coherence of current editing methods between different languages and are not suitable for research on language-specific (a.k.a. Chinese) knowledge edit-

[Figure 1]

Figure 1: Examples of data from each subcategory in CKnowEdit, with detailed explanations provided in §2.

ing methods or for understanding LLMs’ representation of specific languages.

To help address the three major challenges mentioned above and mitigate some existing deficiencies in the current editing datasets, we construct a new Chinese dataset, CKnowEdit, which takes into account language-specific characteristics, ensuring that the data is not only linguistically accurate but also culturally matched. To ensure the quality and diversity of CKnowEdit, we collect data from a variety of sources, including classical literature, modern colloquialisms, and Baidu Tieba Ruozhiba (Bai et al., 2024) (a popular Chinese online forum renowned for its abundance of logic puzzles and brainteasers, highly suitable for evaluating the reasoning capabilities). As a result, we organize CKnowEdit into 3 major categories, including Linguistic, Facts and Logic corresponding to the three major challenges and 10 subcategories, as shown in Figure 1.

To benchmark the effectiveness of knowledge editing methods on CKnowEdit, we evaluate five representative methods on four models. Departing from traditional knowledge editing evaluations that

rely on token/logit-level measurements through teacher-forcing automation (Yao et al., 2023), we implement open-ended text generation to evaluate edited models under more realistic and demanding conditions and utillize LLM-as-a-judge paradiam to effectively evaluate. The results demonstrate the challenges presented by the dataset and underscore the need for more sophisticated Chinese knowledge editing approaches in the future. Our major contributions are as follows:

- • We propose a new knowledge editing dataset, CKnowEdit, which is uniquely characterized by its Chinese linguistic features and cultural depth, comprehensively exploring the language’s distinctiveness and the challenges it poses to LLMs from three perspectives.
- • We report the empirical results of recent knowledge editing baselines on CKnowEdit, revealing their limitations when applied to Chinese literature.
- • We further explore the challenges of Chinese knowledge editing and the struggles faced by

existing models in understanding Chinese language and culture.

## 2 Criteria for Knowledge Sourcing

### 2.1 Chinese Linguistics

Chinese linguistics studies the phonetics, vocabulary, semantics and grammar of the Chinese language, the linguistic knowledge in CKnowEdit is categorized into five subtypes. Each subtype of knowledge presents unique challenges for LLMs.

Pinyin Pinyin Notation serves as the official romanization system for Standard Mandarin Chinese, utilizing the Latin alphabet to represent Chinese characters phonetically. In Chinese, the phenomenon of polyphonic characters is widespread. As shown in Figure 1, the character ‘ ’ (six) is pronounced ‘Liù’ in most cases, but in ‘ ’ (a city) it is pronounced ‘Lù.’ This inherent ambiguity in grapheme-phoneme mapping poses challenges for LLMs, especially when dealing with low-frequency characters with multiple pronunciations, which are also included in CKnowEdit.

[Figure 2]

[Figure 3]

Ancient Poetry Ancient Poetry constitutes an essential component of Chinese classical literature, which significantly differs from Modern Vernacular Chinese, particularly in semantic constructs and graphological conventions. Additionally, Ancient Poetry adhere to extremely strict requirements for format and rhythm, where every character must be precise and cannot be altered or omitted. This form of ancient language commonly embedded in the parameters of large language models, poses a significant challenge to their memory and processing capabilities.

Classical Chinese Words in Classical Chinese often carry greatly different meanings compared to Modern Chinese. And the same character may represent distinct concepts based on context. As shown in Figure 1, the ‘ ’ ( means ‘safety’ in Modern Chinese ) can denote ‘to nurture’, ‘to stabilize’ or function as an interrogative term (‘where/how’) in classical texts. This semantic divergence poses unique challenges for language models trained on Modern Chinese data, particularly when processing context-sensitive interpretations of polysemous characters in classical literature.

[Figure 4]

Idiom Directly comprehending Chinese idioms or interpreting them literally often leads to a loss of their true meaning. In fact, the actual meaning

of many idioms can be entirely opposite to their literal interpretation, such as the idiom which literal meaning is July’s flowing fire contrary to the true meaning. LLMs’ statistical learning paradigms struggle to resolve these interpretative gaps, particularly when processing idioms whose surface forms actively contradict their established semantic values in linguistic praxis.

[Figure 5]

Proverb Proverbs often use modern expressions with clear literal meanings, but their actual significance usually depends on metaphorical understanding. While these proverbs maintain consistent core meanings, LLMs struggle to apply them appropriately across different real-life situations.

### 2.2 Factual Knowledge

History and Geographical knowledge in CKnowEdit covers key events and historical figures, regional landscapes, and unique local cultures across China. However, mainstream LLMs demonstrate notable gaps in their understanding of factual knowledge related to China’s history and geography (Sun et al., 2024a).

### 2.3 Chinese language-specific logic trap

Phonetic Misunderstand Figure 1 demonstrates a typical Chinese phonetic misunderstanding involving the polyphonic character ‘ ’. When pronounced as ‘zhˇang’, it combines with the preceding ‘ ’ to form ‘ ’ (team leader), suggesting the illogical meaning ‘The vaccinated team leader has died’. However, ‘ ’ actually functions as an adjective meaning ‘was long’ which pronounced as ‘cháng’, and ‘ ’ simply means ‘queue", indicating that ‘Today’s vaccination queue was extremely long’. This highlights how LLMs’ pronunciation disambiguation failures can lead to semantic misinterpretations, even with proper word segmentation.

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Reasoning Error When meeting complex reasoning tasks in the Chinese language, LLMs may commit reasoning errors, hence CKnowEdit has incorporated such data into its considerations.

Wordplay This type of logical fallacy often arises from word segmentation errors or ambiguous terms being misinterpreted as unintended meanings, thereby distorting the original semantic content of the textual components within a sentence. As illustrated in Example 1, the LLM misinterpreted (Bluetooth earphones) through erroneous word segmentation as ‘ - ’ (literally

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Figure 2: Overview of CKnowEdit construction. A full sample of CKnowEdit is shown in Figure 7 and 8.

[Figure 15]

Figure 3: The statistics of CKnowEdit.

‘blue tooth-ear device’), forcing a literal interpretation within a physiological context (teeth and ears), ultimately producing semantically absurd outputs.

## 3 The Construction of CKnowEdit

### 3.1 Data Preprocess

Data Collection As described in §2, we classify the data types in CKnowEdit into 3 major categories and 10 subcategories, as also illustrated in Figure 1. Data collection is conducted based on this classification. We crawled authentic and diverse Chinese corpora and collected 11,981 raw data entries initially. All our data collectors adhere to the copyright and licensing terms of each data source website and all the data collected are freely available for academic research. Detailed addresses of each source website can be found in Appendix A.

Data Filtering As shown in Figure 2, we first convert all the collected raw data into queries and pose them to the Qwen-7b-chat (Yang et al., 2024) model as a baseline. We then retain only those questions that the model answered incorrectly, discarding those it answered correctly. The filtering

process ensures CKnowEdit remains challenging and justifies the necessity of applying knowledge editing techniques. To maintain data quality, we conduct a manual review of all collected responses.

### 3.2 Data Annotation

Prompt-target Construction The queries after filtering are used as the prompt field in the data. But for target field: fixed/data-provided answers are used directly; open-ended explanations (e.g., interpret logic errors) are generated by GPT-4. Moreover, all answers generated by GPT-4 undergo meticulous manual review and correction to ensure their accuracy.

In-scope Construction Effective model editing requires consistent behavioral adjustments across all examples within the editing scope. Beyond correcting the primary target knowledge, related in-scope information conveying similar concepts should also be updated. We therefore assess two distinct generalization capabilities: weak and robust generalization. Specifically, we evaluate the weak generalization effect by rephrasing the prompt, such as rephrasing ‘Please complete the following ancient poem’ as ‘The next line of the following ancient poem is...’. Robust generalization is measured through two approaches: (i) Context Transfer: This involves transferring the same knowledge or language pattern to a different application scenario to see if the edited model has truly learned the knowledge. For example, in classical Chinese, the character ‘ ’ means ‘nurture’ in the phrase ‘ ’ (provides sustenance). We then ask the edited model about the mean-

[Figure 16]

[Figure 17]

ing of ‘ ’ in ‘ ’ (the elderly are supported) where it still means ‘nurture’. (ii) Logical SingleHop: We present the edited model with a question that requires one additional reasoning step beyond the original prompt. For example, if the original prompt is ‘Please complete the following ancient poem A’ (with the correct answer being B), the portability field would then be ‘What is the line before B?’

[Figure 18]

[Figure 19]

Out-scope Construction A successful edit should adjust the targeted knowledge locally while leaving unrelated knowledge unaffected. However, current approaches that modify internal model parameters often introduce knowledge conflicts and distortions (Li et al., 2023). Unlike other knowledge editing datasets that adopt entirely unrelated knowledge for locality evaluation, we construct our locality field by selecting somewhat related knowledge (e.g., sharing the same subject) to the target knowledge but containing distinct factual information. This approach provides stricter evaluation of editing side-effects while posing greater challenges for language models.

### 3.3 Dataset Statistics

Finally, we distilled 1,854 samples from 11,981 raw entries to form CKnowEdit. Regarding the three main knowledge classifications in CKnowEdit, the largest proportion is attributed to linguistic data accounts for 48.40% and the Logic reasoning data accounts for 45.63% because we found that knowledge that is highly characteristic of the Chinese language poses significant challenges for current LLMs. The specific quantity and proportion of each data category are shown in the Figure 3.

### 3.4 Quality Assurance

After constructing CKnowEdit, we implement a comprehensive quality assurance process to ensure data reliability. We hire professional NLP annotators to review all the fields within the dataset. The quality assurance process involved five steps: (1) Task Setup: The dataset is split into

- 3 fields—prompt-target, generalization, and locality—each assigned to separate teams. (2) Team Training: Team members are trained to understand their assigned field’s purpose and follow standardized review workflows. (3) Guideline Calibration: We conduct a trial review on a random 20% of data to fine-adjust the review process. (4) Dual Review: Each field is independently reviewed by two anno-

tators. A field is considered acceptable only if both annotators’ conclusions matched our own, and they identified no issues with the field both the question and the ground truth. (5) Resolution of Discrepancies: For any fields that failed at the step 4, the authors discussed whether to retain, discard, or correct them, depending on the nature of the identified issues.

### 3.5 Field and Usage Specifications

As shown in Figure 7, ’prompt’ and ’target new’ fields constitute the target knowledge we aim to edit. ’target old’ represents the original incorrect response generated by Qwen-7B-Chat during our data filtering phase.

With the continuous advancement of LLMs, their capabilities are growing increasingly powerful. As a result, some data in CKnowEdit may already be correctly answered by more capable models. However, this does not diminish the significance of CKnowEdit for editing purposes. During usage, ’target new’ and ’target old’ fields can be swapped to construct counterfactual data (as is commonly done in many editing datasets). This approach not only preserves the essence of the editing operation but also maintains the unique Chinese characteristics of this dataset.

4 Experiments

### 4.1 Experiment Settings

Models and Editing Methods To better evaluate the editing effectiveness on CKnowEdit, we select 4 advanced LLMs that are widely used in the Chinese community: Qwen-7B-Chat, Qwen27B-Instruct (Yang et al., 2024), DeepSeek-LLM7B-Chat (DeepSeek-AI, 2024) and Baichuan2-7BChat (Baichuan, 2023). Among them, Qwen-7BChat is the original model used for data collection, providing baseline performance. We investigate diverse model editing methods, including FT-M (Zhang et al., 2024a), AdaLoRA (Zhang

- et al., 2023), ROME, GRACE and AlphaEdit (Fang
- et al., 2024). All the experiments are conducted by EasyEdit (Wang et al., 2024b). All models are deployed and edited on 1 to 2 NVIDIA A800 GPUs.

Evaluation Unlike conventional knowledge editing evaluations that use token/logit-level metrics with teacher-forcing automation, we adopt openended text generation to assess edited models in more practical and challenging scenarios. While

[Figure 20]

- Figure 4: Main results. We do not report the locality of Ancient Poetry, Proverbs, Idioms and Facts Knowledge because it is challenging to find out-scope knowledge that is both relevant to and distinct from the target knowledge when we construct the locality field. The corresponding numerical results are presented in the Table1.

some studies (Deng et al., 2024) under similar setups use metrics like ROUGE-L or semantic similarity, we find these metrics often fail to reflect true text quality. For instance, ROUGE-L is heavily skewed by text length—shorter reference texts paired with longer model outputs lead to unreliable scores. A recent study (Yang et al., 2025) has also revealed specific inadequacies in traditional evaluation methods.

Inspired by MT-Bench (Zheng et al., 2023b) which reveals that strong LLM judges like GPT-4o can align closely with human preferences, We customize prompts and evaluation processes for each knowledge category’s unique characteristics, enabling GPT-4o to serve as evaluator. For each evaluation metric, we provide GPT-4o (gpt-4o-202408-06) with the corresponding question, edited model’s response, and the reference answer. GPT-

- 4o then assigns a score from 1 to 10 based on the

relevance between the model’s response and the reference answer. For detailed evaluation procedures and templates, refer to Figure 9 to 13.

Metrics We employ 4 key knowledge editing evaluation metrics: (1) Edit Success (ES) : This metric measures how well the edits align LLMs’ responses with the expected outcomes. (2) Generalization (Gen) : The metric helps to assess the weak generalization of the editing. (3) Portability (Por) : This measures the model’s capability to apply corrected knowledge to new but related prompts, assessing the robust generalization of the editing across contexts. (4) Locality (Loc) : This metric ensures that edits do not inadvertently affect unrelated areas of the model’s knowledge base.

### 4.2 Main Results

Methods Comparison AdaLoRA achieves the highest Edit Success in over 70% of cases across

[Figure 21]

- Figure 5: The format of the indicators in the figure is: data type-matric, for example, Lin-ES (linguistics-ES) represents the editing success rate of the language in the linguistic data category. The results of ROME are shown in the Figure 14.

- 4 models, outperforming AlphaEdit and FT-M, which excel in 4 and 3 instances respectively but remain suboptimal overall. For Generalization and Portability metrics, AdaLoRA dominates with nearly 70% and 86% top scores, respectively, while AlphaEdit consistently performs suboptimally. These results demonstrate that AdaLoRA achieves the best editing performance, contrasting with prior findings (Zhang et al., 2024a).

We believe the reason is that CKnowEdit’s focus on editing long-text patterns and evaluating long-text generation differs fundamentally from prior studies. Traditional approaches like ROME edit models via localized parameter tweaks to precisely overwrite single factual knowledge as discrete triplet (s-r-o). While effective for closedform tasks (e.g., token-level teacher forcing evaluation task), this approach disrupts the generative distribution needed for coherent open-ended text. In contrast, AdaLoRA adaptively adjusts multiple modules (like attention heads and FFN layers), allowing the model to implicitly learn taskspecific patterns (e.g., long-range dependencies). By holistically adjusting parameters linked to target knowledge, AdaLoRA preserves contextual consistency, aligning edits with the broader language generation process.

Data Types Comparison The editing performance on the Ancient Poetry is notably poor across all knowledge types, especially for Portability, where almost all models and methods achieve scores below 1. As described in §2, Chinese ancient poetry poses significant challenges to the memorization capabilities of LLMs. This stems from two linguistic specificities: (1) Rare characters: Many obscure characters in poetry appear infrequently in training data, leading to weak semantic

representation and context modeling; (2) Distribution shift: The syntactic structures and vocabulary differ markedly from modern Chinese, making patterns harder to capture. Combined, these factors cause strong prior biases from modern Chinese during next token prediction. When generating text with modern-style prefixes or the current token is common in modern Chinese, models increasingly misalign subsequent token distributions.

Additionally, the poor performance on Classical Chinese highlights the need for more advanced editing methods to handle its rich syntax, semantics, and context-dependency, particularly in addressing nuances like polysemy and homophony, which are less common in English.

4.3 Why do we need an editing dataset that is highly characteristic of Chinese?

The Irreplaceability of Chinese To better illustrate the unique characteristics of Chinese and its irreplaceability in conveying Chinese knowledge, we selected 100 data samples from each of the three knowledge categories in CKnowEdit. These samples were first translated into English, then edited using AdaLoRA and ROME on four baseline models. The results were then translated back into Chinese and evaluated. The AdaLoRA results are shown in Figure 5.

It can be observed that in linguistic knowledge editing tasks, the results of English editing differ significantly from those of Chinese editing, often failing to produce precise edits. This is because the literal translation of Chinese linguistic knowledge into English frequently loses the original meaning, aesthetic value, correct structure, and language patterns, leading to significant deviations between the model’s edited responses and the correct answers. For example, in the case of classi-

cal poetry editing shown in Figure 6a), the model can successfully edit the English target. However, when translating back into Chinese, current translation software or LLMs generally learn the language patterns of modern Chinese, thus unable to translate a sentence of English back into classical poetry.

In factual tasks, the results of English editing are generally on par with those of Chinese editing. This aligns with intuition, as factual knowledge is less dependent on the linguistic medium, and literal translations do not significantly alter the intended meaning.

In logical tasks, English editing performs even slightly better than Chinese editing. This is because many logic traps unique to the Chinese language, which are challenging for LLMs, are often lost during the translation process, reducing their logical complexity in the English version.

## 5 Related Work

- 5.1 Knowledge Editing Methods

Current knowledge editing approaches can be categorized into two main types: preserving LMs’ parameters or modifying LMs’ parameters. Preservative methods incorporate external memory or additional trainable parameters: SERAC (Mitchell

- et al., 2022b) and IKE (Zheng et al., 2023a) leverage a counterfactual model and a multi-fact prompt, respectively, as external working memory. CaliNET (Dong et al., 2022), T-Patcher (Huang et al.,

- 2023b), GRACE (Hartvigsen et al., 2024), and WISE (Wang et al., 2024a) introduce extra trainable parameters. The locate-and-edit approaches have to locate the relevant neurons and then modify those parameters. Representative studies are KN (Dai et al., 2022), ROME (Meng et al., 2022), MEMIT (Meng et al., 2023) and NSE (Jiang et al.,
- 2024). Additionally, meta-learning approaches utilize a hyper-network to generate the weights for layers in LLMs, including KE (Cao et al., 2021), MEND (Mitchell et al., 2022a), and MALMEN (Tan et al., 2023).

5.2 Knowledge Editing Datasets

Existing knowledge editing datasets have largely centered on English-language texts, such as ZsRE (Cao et al., 2021), Counterfact (Meng et al., 2022), KnowEdit (Zhang et al., 2024a), MQuAKE (Zhong

- et al., 2023) and . Some research (Deng et al., 2024; Rosati et al., 2024; Wu et al., 2024b) has also introduced the concept of evaluating knowledge editing through unstructured text and long-form content, but these efforts have been predominantly limited to English. In a more inclusive direction, recent academic initiatives have broadened the scope of these datasets to include a multilingual dimension (Xie et al., 2024; Wei et al., 2024a; Wu et al., 2024a; Nie et al., 2024).

Language Functional Area Offset Similar to the human brain, neuron parameter regions for different languages in LLMs often don’t overlap (Zhang et al., 2024b), creating natural barriers for cross-language knowledge editing and generalization. Previous studies (Wang et al., 2023a) show that when editing knowledge in English and testing its generalization in Chinese, performance sometimes drops – even for factual knowledge where the English-Chinese gap is relatively small. As shown in Figure 6(b), our tests on Qwen2-7B-Instruct reveal this limitation: the model struggles to generalize English-edited knowledge to Chinese, whether for factual geography or linguistically complex tasks. For instance, while the model correctly answers classical poetry question in English, it fails completely when the original question is posed.

### 4.4 Human Evaluation

To verify the effectiveness of our designed automatic GPT-4 score for CKnowEdit evaluation, from the outputs of 4 baseline models edited by 5 different methods (totaling 20 output categories), we select 70 samples per category for human evaluation conducted by our 5 contracted annotators under rigorous assessment standards. From the human evaluation results, the overall correlation coefficient across all 4 metrics between the automatic and human evaluation is 0.70, indicating a high consistency between GPT-4 scores and human preferences.

## 6 Conclusion

In this work, we created a new, high-quality Chinese knowledge editing dataset, CKnowEdit, which is rich in Chinese linguistic characteristics and linguistic value. This dataset comprehensively evaluates the performance of current mainstream editing methods on leading Chinese LLMs across three knowledge types: linguistics, facts, and logic. Furthermore, we adopted an evaluation approach that better aligns with real-world application requirements. To date, most existing mothods and LLMs

[Figure 22]

- Figure 6: a) shows case where data is directly translated from Chinese to English, and the model’s responses is translated back to Chinese. Part b) includes two cases that after editing target knowledge in English, query are asked directly in Chinese to test cross-language generalization.

still can not edit the Chinese characteristic knowledge well.

## Limitations

Imbalanced Distribution of Data Types Since the original intention of CKnowEdit is to study knowledge with distinctive Chinese linguistic characteristics, and Chinese linguistic knowledge or logical knowledge better reflect these features, the quantity of these two types of knowledge in CKnowEdit is significantly greater than that of factual knowledge.

Experimental Setup Due to computational resource constraints, all experiments in this study were conducted solely under the single-edit setting, without investigating batch edit or sequential edit scenarios.

LLM-as-a-judge The use of GPT-4 as an evaluator for other LLMs has become a widely adopted practice in the field. While employing GPT-4 to evaluate itself may introduce inherent biases, its assessments of other models still provide valuable reference points. It’s important to note that the challenge of effectively evaluating LLMs extends beyond our specific tasks - the research community continues to actively investigate robust evaluation methodologies across various applications. As an interim solution, we advise readers to interpret GPT-4 evaluation scores with appropriate caution, recognizing both their utility and limitations.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (No. 62206246, No. NSFCU23B2055, No. NSFCU19B2027), the Fundamental Research Funds for the Central Uni-

versities (226-2023-00138), Yongjiang Talent Introduction Programme (2021A-156-G), Tencent AI Lab Rhino-Bird Focused Research Program (RBFR2024003), Ningbo Natural Science Foundation (2024J020), Information Technology Center and State Key Lab of CAD&CG, Zhejiang University. We gratefully acknowledge the support of Zhejiang University Education Foundation Qizhen Scholar Foundation.

## References

Yuelin Bai, Xinrun Du, Yiming Liang, Yonggang Jin, Ziqiang Liu, Junting Zhou, Tianyu Zheng, Xincheng Zhang, Nuo Ma, Zekun Wang, Ruibin Yuan, Haihong Wu, Hongquan Lin, Wenhao Huang, Jiajun Zhang, Wenhu Chen, Chenghua Lin, Jie Fu, Min Yang, Shiwen Ni, and Ge Zhang. 2024. Coig-cqia: Quality is all you need for chinese instruction fine-tuning. Preprint, arXiv:2403.18058.

Baichuan. 2023. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305.

Antoine Berman and Lawrence Venuti. 2021. Translation and the trials of the foreign. In The translation studies reader, pages 247–260. Routledge.

Nicola De Cao, Wilker Aziz, and Ivan Titov. 2021. Editing factual knowledge in language models. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 711 November, 2021, pages 6491–6506. Association for Computational Linguistics.

Canyu Chen, Baixiang Huang, Zekun Li, Zhaorun Chen, Shiyang Lai, Xiongxiao Xu, Jia-Chen Gu, Jindong Gu, Huaxiu Yao, Chaowei Xiao, Xifeng Yan, William Yang Wang, Philip Torr, Dawn Song, and Kai Shu. 2024. Can editing llms inject harm? CoRR, abs/2407.20224.

Huajun Chen. 2024. Large knowledge model: Perspectives and challenges. Data Intelligence, 6(3):587–620.

Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. 2022. Knowledge neurons in pretrained transformers. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 8493– 8502. Association for Computational Linguistics.

DeepSeek-AI. 2024. Deepseek llm: Scaling opensource language models with longtermism. arXiv preprint arXiv:2401.02954.

Jingcheng Deng, Zihao Wei, Liang Pang, Hanxing Ding, Huawei Shen, and Xueqi Cheng. 2024. Unke: Unstructured knowledge editing in large language models. arXiv preprint arXiv:2405.15349.

Qingxiu Dong, Damai Dai, Yifan Song, Jingjing Xu, Zhifang Sui, and Lei Li. 2022. Calibrating factual knowledge in pretrained language models. In Findings of the Association for Computational Linguistics: EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 5937–5947. Association for Computational Linguistics.

Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Xiang Wang, Xiangnan He, and Tat seng Chua. 2024. AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models. arXiv, page 2410.02355.

Tom Hartvigsen, Swami Sankaranarayanan, Hamid Palangi, Yoon Kim, and Marzyeh Ghassemi. 2024. Aging with grace: Lifelong model editing with discrete key-value adaptors. Advances in Neural Information Processing Systems, 36.

Peter Hase, Thomas Hofweber, Xiang Zhou, Elias Stengel-Eskin, and Mohit Bansal. 2024. Fundamental problems with model editing: How should rational belief revision work in llms? arXiv preprint arXiv:2406.19354.

Xinshuo Hu, Dongfang Li, Baotian Hu, Zihao Zheng, Zhenyu Liu, and Min Zhang. 2024. Separate the wheat from the chaff: Model deficiency unlearning via parameter-efficient module operation. In Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada, pages 18252– 18260. AAAI Press.

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. 2023a. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. CoRR, abs/2311.05232.

Zeyu Huang, Yikang Shen, Xiaofeng Zhang, Jie Zhou, Wenge Rong, and Zhang Xiong. 2023b. Transformerpatcher: One mistake worth one neuron. In The

Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Houcheng Jiang, Junfeng Fang, Tianyu Zhang, An Zhang, Ruipeng Wang, Tao Liang, and Xiang Wang. 2024. Neuron-level sequential editing for large language models. Preprint, arXiv:2410.04045.

Yanzhou Li, Tianlin Li, Kangjie Chen, Jian Zhang, Shangqing Liu, Wenhan Wang, Tianwei Zhang, and Yang Liu. 2024. Badedit: Backdooring large language models by model editing. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Zhoubo Li, Ningyu Zhang, Yunzhi Yao, Mengru Wang, Xi Chen, and Huajun Chen. 2023. Unveiling the pitfalls of knowledge editing for large language models. arXiv preprint arXiv:2310.02129.

Yang Liu, Yuanshun Yao, Jean-Francois Ton, Xiaoying Zhang, Ruocheng Guo, Hao Cheng, Yegor Klochkov, Muhammad Faaiz Taufiq, and Hang Li. 2023. Trustworthy llms: A survey and guideline for evaluating large language models’ alignment. arXiv preprint arXiv:2308.05374.

Philip A Luelsdorff. 1994. The Prague School of structural and functional linguistics, volume 41. John Benjamins Publishing.

Christian MIM Matthiessen. 2023. System in Systemic Functional Linguistics: A system-based theory of language. University of Toronto Press.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in gpt. Advances in Neural Information Processing Systems, 35:17359–17372.

Kevin Meng, Arnab Sen Sharma, Alex J. Andonian, Yonatan Belinkov, and David Bau. 2023. Massediting memory in a transformer. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, and Christopher D. Manning. 2022a. Fast model editing at scale. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.

Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D Manning, and Chelsea Finn. 2022b. Memorybased model editing at scale. In International Conference on Machine Learning, pages 15817–15831. PMLR.

Shiwen Ni, Dingwei Chen, Chengming Li, Xiping Hu, Ruifeng Xu, and Min Yang. 2023. Forgetting before learning: Utilizing parametric arithmetic for knowledge updating in large language models. CoRR, abs/2311.08011.

Ercong Nie, Bo Shao, Zifeng Ding, Mingyang Wang, Helmut Schmid, and Hinrich Schütze. 2024. Bmike53: Investigating cross-lingual knowledge editing with in-context learning. arXiv preprint arXiv:2406.17764.

Shankar Padmanabhan, Yasumasa Onoe, Michael J. Q. Zhang, Greg Durrett, and Eunsol Choi. 2023. Propagating knowledge updates to lms through distillation. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Shanbao Qiao, Xuebing Liu, and Seung-Hoon Na. 2024. Distillmike: Editing distillation of massive in-context knowledge editing in large language models. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pages 7639–7654. Association for Computational Linguistics.

Domenic Rosati, Robie Gonzales, Jinkun Chen, Xuemin Yu, Melis Erkan, Yahya Kayani, Satya Deepika Chavatapalli, Frank Rudzicz, and Hassan Sajjad. 2024. Long-form evaluation of model editing. arXiv preprint arXiv:2402.09394.

Jiaxing Sun, Weiquan Huang, Jiang Wu, Chenya Gu, Wei Li, Songyang Zhang, Hang Yan, and Conghui He. 2024a. Benchmarking chinese commonsense reasoning of llms: From chinese-specifics to reasoning-memorization correlations. Preprint, arXiv:2403.14112.

Lichao Sun, Yue Huang, Haoran Wang, Siyuan Wu, Qihui Zhang, Chujie Gao, Yixin Huang, Wenhan Lyu, Yixuan Zhang, Xiner Li, et al. 2024b. Trustllm: Trustworthiness in large language models. arXiv preprint arXiv:2401.05561.

Chenmien Tan, Ge Zhang, and Jie Fu. 2023. Massive editing for large language models via meta learning. CoRR, abs/2311.04661.

Eva Vanmassenhove, Dimitar Shterionov, and Andy Way. 2019. Lost in translation: Loss and decay of linguistic richness in machine translation. In Proceedings of Machine Translation Summit XVII: Research Track, pages 222–232.

Jiaan Wang, Yunlong Liang, Zengkui Sun, Yuxuan Cao, and Jiarong Xu. 2023a. Cross-lingual knowledge editing in large language models. arXiv preprint arXiv:2309.08952.

Peng Wang, Zexi Li, Ningyu Zhang, Ziwen Xu, Yunzhi Yao, Yong Jiang, Pengjun Xie, Fei Huang, and Huajun Chen. 2024a. Wise: Rethinking the knowledge memory for lifelong model editing of large language models. Preprint, arXiv:2405.14768.

Peng Wang, Ningyu Zhang, Bozhong Tian, Zekun Xi, Yunzhi Yao, Ziwen Xu, Mengru Wang, Shengyu Mao, Xiaohan Wang, Siyuan Cheng, Kangwei Liu, Yuansheng Ni, Guozhou Zheng, and Huajun Chen. 2024b.

EasyEdit: An easy-to-use knowledge editing framework for large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pages 82–93, Bangkok, Thailand. Association for Computational Linguistics.

Song Wang, Yaochen Zhu, Haochen Liu, Zaiyi Zheng, Chen Chen, et al. 2023b. Knowledge editing for large language models: A survey. arXiv preprint arXiv:2310.16218.

Yiwei Wang, Muhao Chen, Nanyun Peng, and Kai-Wei Chang. 2024c. Deepedit: Knowledge editing as decoding with constraints. CoRR, abs/2401.10471.

Zihao Wei, Jingcheng Deng, Liang Pang, Hanxing Ding, Huawei Shen, and Xueqi Cheng. 2024a. Mlake: Multilingual knowledge editing benchmark for large language models. arXiv preprint arXiv:2404.04990.

Zihao Wei, Liang Pang, Hanxing Ding, Jingcheng Deng, Huawei Shen, and Xueqi Cheng. 2024b. Stable knowledge editing in large language models. CoRR, abs/2402.13048.

Chaojie Wen, Xudong Jia, and Tao Chen. 2023. Improving extraction of chinese open relations using pre-trained language model and knowledge enhancement. Data Intelligence, 5(4):962–989.

Xiaobao Wu, Liangming Pan, William Yang Wang, and Anh Tuan Luu. 2024a. AKEW: assessing knowledge editing in the wild. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, pages 15118–15133. Association for Computational Linguistics.

Xiaobao Wu, Liangming Pan, William Yang Wang, and Anh Tuan Luu. 2024b. Updating language models with unstructured facts: Towards practical knowledge editing. arXiv preprint arXiv:2402.18909.

Jiakuan Xie, Pengfei Cao, Yuheng Chen, Yubo Chen, Kang Liu, and Jun Zhao. 2024. Memla: Enhancing multilingual knowledge editing with neuron-masked low-rank adaptation. arXiv preprint arXiv:2406.11566.

Liang Xu, Anqi Li, Lei Zhu, Hang Xue, Changtai Zhu, Kangkang Zhao, Haonan He, Xuanwei Zhang, Qiyue Kang, and Zhenzhong Lan. 2023. Superclue: A comprehensive chinese large language model benchmark. arXiv preprint arXiv:2307.15020.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu,

Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. 2024. Qwen2 technical report. arXiv preprint arXiv:2407.10671.

Wanli Yang, Fei Sun, Jiajun Tan, Xinyu Ma, Qi Cao, Dawei Yin, Huawei Shen, and Xueqi Cheng. 2025. The mirage of model editing: Revisiting evaluation in the wild. Preprint, arXiv:2502.11177.

Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, and Ningyu Zhang. 2023. Editing large language models: Problems, methods, and opportunities. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 10222–10240. Association for Computational Linguistics.

Ningyu Zhang, Yunzhi Yao, Bozhong Tian, Peng Wang, Shumin Deng, Mengru Wang, Zekun Xi, Shengyu Mao, Jintian Zhang, Yuansheng Ni, Siyuan Cheng, Ziwen Xu, Xin Xu, Jia-Chen Gu, Yong Jiang, Pengjun Xie, Fei Huang, Lei Liang, Zhiqiang Zhang, Xiaowei Zhu, Jun Zhou, and Huajun Chen. 2024a. A comprehensive study of knowledge editing for large language models. CoRR, abs/2401.01286.

Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. 2023. Adaptive budget allocation for parameter-efficient fine-tuning. In The Eleventh International Conference on Learning Representations.

Zhihao Zhang, Jun Zhao, Qi Zhang, Tao Gui, and Xuanjing Huang. 2024b. Unveiling linguistic regions in large language models. Preprint, arXiv:2402.14700.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2023. A survey of large language models. CoRR, abs/2303.18223.

Ce Zheng, Lei Li, Qingxiu Dong, Yuxuan Fan, Zhiyong Wu, Jingjing Xu, and Baobao Chang. 2023a. Can we edit factual knowledge by in-context learning? In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 4862– 4876. Association for Computational Linguistics.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023b. Judging llm-as-a-judge with mt-bench and chatbot arena. Preprint, arXiv:2306.05685.

Zexuan Zhong, Zhengxuan Wu, Christopher D Manning, Christopher Potts, and Danqi Chen. 2023.

Mquake: Assessing knowledge editing in language models via multi-hop questions. arXiv preprint arXiv:2305.14795.

## A Data Source Website

This section provides a detailed overview of the data types and their corresponding data source websites:

- (1) Ancient Poetry: https://zhuanlan.zhihu.com/p/414484867
- (2) Proverbs: http://www.360doc.com/content/19/0218/

14/39098269_815762159.shtml,

http://www.360doc.com/content/19/0312/ 16/5784427_820995624.shtml,

http://www.360doc.com/content/19/0126/ 14/55773589_811408910.shtml

- (3) Idioms: https://zhuanlan.zhihu.com/p/599709230
- (4) Pinyin notation: https://zhuanlan.zhihu.com/p/599709230
- (5) Classical Chinese: https://zhuanlan.zhihu.com/p/

622859964,

https://www.bilibili.com/read/ cv20279857/,

https://wyw.hwxnet.com/search.do?wd= %E9%84%99&x=0&y=0

(6) Geographical and History: https://baijiahao.baidu.com/s?id= 1682950669904608106&wfr=spider&for=pc, https://www.sohu.com/a/419822319_ 100941,

https://www.jingyanben.com/ qitawendang/125282.html?page=1,

http://www.360doc.com/content/20/0613/ 11/7254176_918223750.shtml

(7) Logic Error: https://github.com/Leymore/ruozhiba, https://docs.qq.com/sheet/

DUlZ6aURhamdwb1RO?tab=BB08J2

[Figure 23]

Figure 7: An example from Classical Chinese type.

Model Knowledge Type (Pre-edit) FT-M AdaLoRA ROME GRACE AlphaEdit

Pinyin 1.22 / 0.76 / 0.68 / 8.53 6.33 / 6.08 / 6.27 / 8.55 9.52 / 8.90 / 7.51 / 5.66 7.22 / 7.16 / 6.20 / 7.37 6.18 / 5.64 / 5.73 / 8.06 6.75 / 6.82 / 5.88 / 6.30 Classical Chinese 2.93 / 3.52 / 3.53 / 5.96 3.71 / 3.79 / 4.23 / 6.26 6.72 / 6.33 / 5.64 / 3.99 2.88 / 3.72 / 3.26 / 4.61 2.81 / 3.77 / 3.28 / 6.05 6.31 / 6.82 / 4.11 / 5.94

Idiom 6.77 / 6.91 / 6.55 / - 6.70 / 6.66 / 6.79 / - 8.77 / 8.34 / 8.05 / - 8.48 / 8.18 / 7.19 / - 6.60 / 6.68 / 6.86 / - 9.12 / 8.79 / 7.97 / Proverb 5.38 / 5.10 / 6.22 / - 5.31 / 5.51 / 6.36 / - 8.13 / 7.79 / 7.58 / - 7.85 / 7.73 / 7.02 / - 5.40 / 5.39 / 6.42 / - 8.38 / 8.36 / 7.50 / -

Qwen-7B-Chat

Ancient Poetry 2.10 / 1.63 / 0.54 / - 1.85 / 1.19 / 0.70 / - 7.35 / 6.13 / 0.26 / - 3.62 / 2.33 / 0.54 / - 1.49 / 1.29 / 0.49 / - 3.41 / 1.66 / 0.18 / -

Fact 2.88 / 3.20 / 3.91 / - 3.03 / 2.51 / 4.03 / - 7.33 / 6.50 / 5.61 / - 4.34 / 3.20 / 3.88 / - 3.17 / 2.94 / 3.81 / - 3.03 / 3.74 / 3.29 / Logic 4.59 / 4.81 / 5.30 / 7.09 5.63 / 5.78 / 6.29 / 6.94 8.22 / 7.28 / 6.93 / 7.19 5.43 / 4.95 / 5.77 / 6.32 5.56 / 5.67 / 6.21 / 6.96 5.83 / 5.13 / 6.25 / 6.97

Pinyin 1.75 / 1.19 / 1.02 / 8.04 6.24 / 6.58 / 3.09 / 0.83 8.80 / 8.59 / 7.57 / 4.22 6.80 / 7.29 / 6.10 / 6.91 7.03 / 6.14 / 6.20 / 8.05 6.22 / 6.59 / 6.25 / 7.16 Classical Chinese 4.87 / 5.42 / 5.25 / 6.92 7.42 / 7.57 / 6.51 / 0.91 6.61 / 7.55 / 6.24 / 3.13 7.77 / 7.13 / 5.66 / 6.01 4.58 / 5.56 / 5.29 / 7.06 8.51 / 7.77 / 5.69 / 6.42

Idiom 9.04 / 9.11 / 7.46 / - 6.80 / 7.16 / 5.27 / - 9.33 / 9.31 / 8.50 / - 8.12 / 8.01 / 7.60 / - 9.02 / 9.14 / 7.71 / - 8.58 / 8.30 / 7.91 / Proverb 6.79 / 6.75 / 6.26 / - 7.33 / 7.68 / 6.35 / - 8.90 / 8.82 / 8.06 / - 7.85 / 7.53 / 7.45 / - 6.76 / 6.76 / 7.24 / - 7.70 / 7.68 / 7.51 / -

Qwen2-7B-Instruct

Ancient Poetry 4.84 / 2.10 / 0.79 / - 7.66 / 6.79 / 0.28 / - 8.69 / 7.94 / 0.65 / - 5.34 / 2.75 / 0.97 / - 4.84 / 2.10 / 1.03 / - 6.64 / 3.84 / 0.64 / -

Fact 4.31 / 4.31 / 4.91 / - 6.97 / 6.42 / 1.97 / - 7.73 / 7.33 / 6.48 / - 4.71 / 4.50 / 5.30 / - 4.30 / 4.23 / 4.75 / - 6.57 / 4.86 / 5.35 / Logic 5.06 / 5.00 / 5.04 / 8.08 7.13 / 5.13 / 4.11 / 3.00 9.36 / 8.29 / 7.71 / 7.78 7.55 / 7.32 / 7.24 / 7.70 7.12 / 7.10 / 7.41 / 7.78 7.88 / 7.49 / 7.53 / 7.91

Pinyin 1.00 / 0.72 / 0.16 / 5.76 7.47 / 6.57 / 4.54 / 2.78 8.02 / 8.04 / 5.61 / 3.62 5.30 / 5.01 / 4.32 / 5.35 5.12 / 4.94 / 4.14 / 4.95 5.20 / 5.27 / 4.55 / 5.21 Classical Chinese 2.88 / 3.51 / 3.25 / 6.31 4.19 / 4.03 / 3.47 / 5.03 4.29 / 4.51 / 3.90 / 6.50 4.40 / 4.03 / 3.25 / 4.99 5.12 / 4.94 / 4.14 / 4.95 5.40 / 5.44 / 3.68 / 6.04

Idiom 8.09 / 8.72 / 6.80 / - 9.27 / 9.06 / 7.11 / - 8.88 / 8.73 / 7.56 / - 8.76 / 7.56 / 7.33 / - 8.36 / 7.06 / 6.33 / - 9.03 / 8.94 / 7.97 / Proverb 6.79 / 6.89 / 6.91 / - 8.38 / 8.33 / 7.56 / - 8.24 / 8.42 / 7.83 / - 8.37 / 8.36 / 7.35 / - 6.82 / 7.29 / 7.02 / - 8.18 / 8.41 / 7.75 / -

DeepSeek-LLM-7B-Chat

Ancient Poetry 2.02 / 1.86 / 0.43 / - 4.82 / 6.05 / 0.20 / - 8.77 / 7.48 / 0.34 / - 3.33 / 3.09 / 0.37 / - 2.34 / 1.70 / 0.23 / - 4.07 / 3.07 / 0.52 / -

Fact 2.63 / 1.89 / 3.21 / - 8.26 / 8.40 / 5.74 / - 6.43 / 6.54 / 5.99 / - 4.02 / 4.32 / 3.01 / - 2.51 / 2.80 / 3.21 / - 4.20 / 3.37 / 3.96 / Logic 4.39 / 4.56 / 4.10 / 7.62 7.25 / 6.34 / 5.94 / 6.02 8.43 / 7.36 / 7.33 / 7.66 6.72 / 6.63 / 6.98 / 5.36 6.38 / 6.35 / 7.06 / 7.56 7.07 / 6.67 / 7.04 / 7.61

Pinyin 0.32 / 0.07 / 0.04 / 5.30 5.42 / 4.14 / 5.24 / 4.07 8.69 / 5.69 / 6.57 / 4.03 4.92 / 2.61 / 5.43 / 5.09 5.20 / 2.84 / 5.02 / 5.39 3.33 / 2.35 / 3.25 / 3.27 Classical Chinese 2.76 / 3.03 / 2.85 / 5.68 8.34 / 8.13 / 6.41 / 1.95 5.65 / 5.78 / 4.06 / 4.20 1.74 / 2.64 / 1.90 / 2.78 2.55 / 3.14 / 2.91 / 5.82 7.44 / 6.81 / 3.57 / 4.71

Idiom 8.16 / 7.98 / 6.74 / - 7.98 / 8.08 / 6.84 / - 9.28 / 9.29 / 7.75 / - 7.60 / 6.24 / 6.60 / - 8.33 / 7.72 / 6.61 / - 7.94 / 7.15 / 6.71 / Proverb 6.87 / 6.46 / 6.57 / - 7.38 / 6.94 / 6.47 / - 8.67 / 8.61 / 7.82 / - 7.54 / 7.74 / 6.71 / - 6.79 / 6.67 / 6.63 / - 8.30 / 7.78 / 6.46 / -

Baichuan2-7B-Chat

Ancient Poetry 1.78 / 1.52 / 0.22 / - 3.39 / 2.95 / 0.51 / - 7.51 / 7.00 / 0.45 / - 1.51 / 1.34 / 0.30 / - 1.61 / 1.40 / 0.19 / - 2.75 / 1.07 / 0.00 / -

Fact 2.25 / 2.86 / 3.28 / - 6.90 / 7.13 / 4.31 / - 8.19 / 7.57 / 5.66 / - 3.77 / 3.10 / 3.27 / - 2.21 / 2.75 / 3.22 / - 4.77 / 4.74 / 4.04 / Logic 4.62 / 4.93 / 5.17 / 7.00 5.36 / 5.39 / 6.14 / 6.76 6.42 / 5.94 / 6.39 / 6.97 5.63 / 5.31 / 4.09 / 6.98 4.65 / 4.71 / 5.96 / 6.80 5.93 / 5.03 / 5.96 / 7.07

#### Table 1: Numerical results (Edit Success / Generalization / Portability / Locality) of pre-edit and post-edit with 5 knowledge editing methods for 4 baseline LLMs.

[Figure 24]

Figure 8: An example from Classical Chinese type(translated by English).

[Figure 25]

Figure 9: Evaluation process of CKnowEdit.

[Figure 26]

#### Figure 10: An example of evaluation.

[Figure 27]

#### Figure 11: An example of evaluation.

[Figure 28]

#### Figure 12: An example of evaluation prompt.

[Figure 29]

#### Figure 13: An example of evaluation prompt(translated by English).

[Figure 30]

#### Figure 14: . The results of ROME.

