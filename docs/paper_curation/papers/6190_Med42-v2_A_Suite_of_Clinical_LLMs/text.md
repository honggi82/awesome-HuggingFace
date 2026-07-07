arXiv:2408.06142v1[cs.CL]12Aug2024

MED42-V2: A SUITE OF CLINICAL LLMS

Clément Christophe∗, Praveen K Kanithi∗, Tathagata Raha Shadab Khan, Marco AF Pimentel M42 Abu Dhabi, UAE {cchristophe, pkanithi, traha, skhan, mpimentel}@m42.ae

ABSTRACT

Med42-v2 introduces a suite of clinical large language models (LLMs) designed to address the limitations of generic models in healthcare settings. These models are built on Llama3 architecture and ﬁne-tuned using specialized clinical data. They underwent multi-stage preference alignment to effectively respond to natural prompts. While generic models are often preference-aligned to avoid answering clinical queries as a precaution, Med42-v2 is speciﬁcally trained to overcome this limitation, enabling its use in clinical settings. Med42-v2 models demonstrate superior performancecompared to the original Llama3 models in both 8B and 70B parameter conﬁgurations and GPT-4 across various medical benchmarks. These LLMs are developed to understand clinical queries, perform reasoning tasks, and provide valuable assistance in clinical environments. The models are now publicly available at https://huggingface.co/m42-health.

- 1 INTRODUCTION

Large language models (LLMs) have revolutionized natural language processing, demonstrating remarkable capabilities across various domains (Achiam et al., 2023; Gemini et al., 2023; Anthropic, 2024). However, their application in specialized ﬁelds like healthcare has been limited due to the need for domain-speciﬁc knowledge and adherence to strict ethical and safety guidelines. The medical sector, in particular, requires models that can understand complex clinical terminology, reason through medical scenarios, and provide accurate, context-appropriate responses.

Despite the advancements, generic models face signiﬁcant limitations when applied to healthcare settings. These include concerns about hallucinations and fabrications, biases and knowledge gaps, and risks about data privacy and ethics (Thirunavukarasu et al., 2023; Li et al., 2023). Such limitations reduce their effectiveness in aiding diagnostic processes (de Souza et al., 2023; Hirosawa et al.,

- 2023), interpreting medical literature (Bagde et al., 2023; Cascella et al., 2023), generating patient education materials (Ali et al., 2023), and assisting in clinical guidelines and decision support systems.

To address these challenges, we introduce a second revision of Med42 (Christophe et al., 2024) called Med42-v2, a suite of clinical large language models designed to overcome the limitations of generic models in healthcare settings. Built on the Llama3 architecture (Dubey et al., 2024) and ﬁne-tuned with specialized clinical data, Med42-v2 models undergo multi-stage preference alignment to effectively respond to natural prompts. Unlike generic models, which are often preferencealigned to avoid answering clinical queries as a precaution, Med42-v2 is speciﬁcally trained to engage with clinical queries, making it suitable for various stakeholders in healthcare, including clinicians, patients, and providers. Med42-v2 demonstrates superior performance compared to the original Llama3 models in both 8B and 70B parameter conﬁgurations (Table 1) across various medical benchmarks, excelling in understanding clinical queries, performing reasoning tasks, and providing valuable assistance in clinical environments.

The key contributions of this work are:

[Figure 2]

∗Equal contribution

[Figure 4]

Base Model Finetuned Aligned Release Date

[Figure 5]

- Med42-Llama2-70B Llama2-70B ✔ ✘ October 2023

[Figure 6]

- Med42-Llama3-8B1 Llama3-8B ✔ ✔ June 2024 Med42-Llama3-70B2 Llama3-70B ✔ ✔ June 2024 Med42-Llama3.1-8B Llama3.1-8B ✔ ✔ August 2024 Med42-Llama3.1-70B Llama3.1-70B ✔ ✔ August 2024

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

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Table 1: Overview of Med42 and Med42-v2 suite of models.

- • A suite of clinical LLMs (Med42-v2) built on Llama3 architecture, ﬁne-tuned with specialized medical instruction data;
- • A multi-stage preference alignment process to enhance the clinically ﬁne-tuned model’s ability to meet user expectations in healthcare settings;
- • Empirical evidence demonstrating Med42-v2’s superior performance over original Llama3 models in both 8B and 70B parameter conﬁgurations across various medical benchmarks.

- 2 METHOD

The development of Med42-v2 follows a two-stage training process designed to create specialized clinical large language models (LLMs) that can effectively handle medical queries and tasks. Our approach builds upon the foundational capabilities of the Llama3 and Llama3.1 model families, enhancing them with domain-speciﬁc knowledge and alignment to clinical use cases. Our training methodology consists of two primary stages:

- • Instruction Fine-tuning: In this initial stage, we ﬁne-tune models from the Llama3 and Llama3.1 families using carefully curated clinical datasets. This process aims to improve the models with specialized medical knowledge.
- • Preference Alignment: The second stage focuses on aligning the models’ outputs with human preferences to ensure they can follow user instructions while safeguarding against unethical or biased behavior.

The following subsections detail each stage of our training process, highlighting the techniques and considerations involved in creating the Med42-v2 suite of models.

- 2.1 CLINICAL FINE-TUNING STAGE

The clinical ﬁne-tuning stage is an important step in adapting large language models for specialized medical applications. This phase aims to enhance the model’s understanding and generation capabilities in clinical contexts, reducing the apprehension to answer medical-related questions and improving its relevance and accuracy for healthcare-related tasks.

Datasets: To construct a training dataset tailored for clinical applications, we curated a diverse collection of resources speciﬁcally focused on medical and biomedical domains. Recognizing the importance of real-world usability beyond simple question-answering, we added examples demonstrating chain-of-thought reasoning as well as chat interactions. This addition was aimed at maximizing the model’s reasoning capabilities and its effectiveness in conversational settings. To further enhance the model’s generalizability and linguistic understanding, we incorporated a carefully selected subset of data from a general domain, comprising 26.5% of the ﬁnal training dataset. This hybrid approach is designed to optimize the model’s performance across both specialized medical content and broader linguistic tasks.

Table 5 provides a detailed breakdown of the various data subsets included in our study, along with their respective sample sizes.

[Figure 28]

Hyperparameter Llama3/3.1 8B Llama3/3.1 70B

[Figure 29]

GPU setup 2 x 8 H100s 6 x 8 H100s LR scheduler Linear warmup - Cosine Linear warmup - Cosine

Maximum LR 5 × 10−6 5 × 10−6 Optimizer AdamW AdamW Beta (0.9, 0.95) (0.9, 0.95) Weight decay 0.01 0.01 Number of steps 5,321 3,549 Tokens per step 262,144 393,216

[Figure 30]

Table 2: Hyperparameters for the clinical ﬁne-tuning stage

Training Methodology: We employ the classic auto-regressive loss for ﬁne-tuning. Loss is backpropagated only on output tokens. This approach ensures that the model learns to generate appropriate responses and not learn to generate the prompts. To maximize training speed and usage of the models context length, we concatenated all of our training samples into chunks of 8192 tokens.

Prompt Format: As we are ﬁne-tuning the Instruct versions of Llama3 and Llama3.1, we adhere to their established prompt format, which includes system, assistant, and user ﬁelds.

Training Process: Each model was ﬁne-tuned for two epochs over our curated dataset. The exact hyperparameters used in this process are detailed in Table 2, providing full transparency for reproducibility.

- 2.2 PREFERENCE-ALIGNMENT STAGE

Preference alignment is a crucial step in developing large language models that can effectively meet user needs and expectations. This process involves adjusting the model’s outputs to align with human preferences. However, obtaining direct human feedback at scale is challenging and resourceintensive. To address this, we employed open-access preference datasets created with AI feedback, allowing for more efﬁcient and scalable alignment.

Datasets: For our preference alignment phase, we utilized two primary datasets: the UltraFeedback dataset (Tunstall et al., 2023) and the Snorkel-DPO dataset (SnorkelAI, 2023). The UltraFeedback dataset is a comprehensive collection of AI preferences on various topics and tasks. The Snorkel-DPO dataset was created through an iterative process. Prompts were exclusively selected from UltraFeedback, without including external LLM responses. For each prompt, ﬁve response variations were generated using the Mistral-7B-Instruct-v0.23 model. These responses were then reranked using PairRM (Jiang et al., 2023) to identify the top (chosen) and bottom (rejected) responses. This process was repeated across three sets of 20,000 prompts, reﬁning both the LLM and the dataset responses through three iterations. This method ensured a comprehensive and structured dataset for training purposes, improving with each iteration.

Training Methodology: We employed Direct Preference Optimization (DPO) (Rafailov et al.,

- 2024) to align our clinically ﬁne-tuned checkpoints with preference data. This approach was chosen over more complex reinforcement learning algorithms (Ouyang et al., 2022) due to its stability and scalability. We used DPO implementation from Huggingface Alignment Handbook library (Tunstall et al.) to train all our models.

Training Process: We followed an iterative alignment approach (Tran et al., 2023) using the multi-stage data as described earlier. For the ﬁrst iteration, we used UltraFeedback data and SnorkelDPO-stage-1 data. The second and third iterations utilized Snorkel-DPO-stage-2 and Snorkel-DPOstage-3 data, respectively. In each iteration, the model resulting from the previous iteration served as a reward model, leading to progressive performance improvements. The exact hyperparameters used in this process are detailed in Table 3.

[Figure 31]

3https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

[Figure 33]

Hyperparameter Llama3/3.1 8B Llama3/3.1 70B

[Figure 34]

GPU setup 2 x 8 H100s 4 x 8 H100s DPO-Beta 0 1 0 01 LR scheduler Linear warmup - Cosine Linear warmup - Cosine

Maximum LR 1 × 10−6 1 × 10−6 Optimizer RMSprop RMSprop Weight decay 0.0 0.0 Batch size 256 128 Maximum length 4096 4096 Epochs (stage 1-3) 1 1

[Figure 35]

- Table 3: Hyperparameters for the preference alignment stage. The same set of hyperparameters is consistently applied across all three stages of alignment.

3 BENCHMARKS

To assess the performance of the ﬁne-tuned language models, following previous works (Singhal et al., 2023; Chen et al., 2023; Toma et al., 2023), we used Eleuther AI’s evaluation harness framework(Gao et al., 2023) to computetheir zero-shotperformanceacross variouscommonly-used medical benchmarks. These contain medical exam questions and research datasets with multiplechoice answers, and include: MMLU (medical subset) (Hendrycks et al., 2021), MMLU-Pro (Wang et al., 2024), MedMCQA (Pal et al., 2022), MedQA (Jin et al., 2020), USMLE (Nori et al., 2023a; Han et al., 2023), PubmedQA (Jin et al., 2019), ToxiGen (Hartvigsen et al., 2022). All datasets are in the English language and all questions containing images were excluded. The harness framework has been updated to include chat templates. Additionally, our log-likelihood calculations are over the entire response sequence instead of just the ﬁrst token.

- Table 4 presents a comparison of our medically-aligned models with various clinical and generalpurpose LLMs. The latest version of Med42 shows signiﬁcant improvements over its previous iteration across multiple benchmarks. In particular, the larger Med42 models (70B) exceed the performance of other leading general-purpose and domain-speciﬁc models, even outperforming proprietary models like GPT-4.0 (Achiam et al., 2023) on all datasets. This suggests that targeted medical instruction and alignment enhance the model’s clinical knowledge and reasoning abilities.

These ﬁndings consistently demonstrate that larger models perform better on these tasks, in line with general trends in language model scaling. However, the performance gains are less signiﬁcant on safety-focused benchmarks like ToxiGen. Moreover, models such as Med42 and OpenBioLLM exhibit enhanced performance on these benchmarks compared to their base Llama3-Instruct versions. This highlights the advantages of speciﬁc medical instruction and alignment in improving the models’ clinical expertise and analytical capabilities.

It’s worth noting that these results represent zero-shot performance. Prior research has indicated that prompting techniques, such as Medprompt (Nori et al., 2023b), or integration with search functionalities can yield even higher accuracy rates. For instance, Med-Gemini has achieved a 91.2% accuracy on benchmarks like MedQA (Saab et al., 2024).

LLMs are designed to excel across a diverse set of tasks, leveraging their conversationalcapabilities. This versatility is crucial for their application in various clinical tasks. Our future work will focus on evaluating these capabilities in a clinical setting in detail.

- 4 CONCLUSIONS AND LIMITATIONS

In conclusion, we introduced Med42-v2, a suite of clinical large language models built on the Llama3 architecture and ﬁne-tuned with specialized clinical data. Med42-v2 also employs a multistage preference alignment process, enabling it to effectively handle clinical queries. Our empirical results show that Med42-v2 outperforms the original Llama3 models in both 8B and 70B parameter conﬁgurations and GPT-4 across various medical benchmarks.

However, utilizing clinical LLMs in real-world settings can present several limitations. Despite improvements, Med42-v2 may not entirely be free from issues like hallucinations, biases, and ethical

[Figure 37]

MedMCQA

[Figure 38]

MMLU-Pro

PubmedQA

ToxiGen

MedQA USMLE

MMLU

Model

Avg.

[Figure 39]

[Figure 40]

Mistral-7B-Instruct-v0.3 33.8 64.6 46.3 49.3 50.4 42.8 86.2 53.3 Llama3-8B-Instruct 48.2 72.9 59.7 61.6 60.4 69.8 78.5 64.4 Llama3.1-8B-Instruct 49.9 73.4 58.4 62.0 68.2 76.2 82.3 67.2 JSL-MedLlama-3-8B-v2.0 46.9 75.9 59.7 59.9 60.6 75.0 74.3 64.6 Med42-Llama3-8B 54.3 75.8 61.3 62.8 67.0 68.4 81.5 67.3 Med42-Llama3.1-8B 54.2 73.6 59.7 63.2 69.9 72.2 83.8 68.1

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Gemma-2-9B 49.9 78.8 56.2 60.9 66.8 39.4 70.5 60.4 Falcon-11B 26.3 62.2 43.8 43.1 44.1 58.0 68.9 49.5 Gemma-2-27B 55.8 81.3 60.2 65.7 71.5 51.4 69.3 65.0 Mixtral-8x7B-Instruct 46.9 75.6 54.1 58.4 67.1 63.2 83.5 64.1 BiMediX (Eng) 49.7 74.9 61.1 65.1 66.4 77.8 43.2 62.6 Phi-3-Medium-128k-instruct 58.2 81.4 61.5 69.0 73.9 46.4 86.6 68.1

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

Mixtral-8x22B-Instruct 55.6 80.7 61.4 67.2 76.1 62.2 87.1 70.0 Llama3-70B-Instruct 64.2 86.0 72.0 78.9 83.6 71.8 87.6 77.7 Llama3.1-70B-Instruct 64.6 87.4 71.9 78.6 93.4 76.6 91.3 80.5 OpenBioLLM-70B 64.2 90.4 73.2 76.9 79.0 73.2 91.3 78.3

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

- Med42-Llama2-70B 51.5 76.7 60.9 61.5 71.9 64.6 88.8 68.0

[Figure 96]

[Figure 97]

- Med42-Llama3-70B 64.4 87.1 73.2 79.1 83.8 78.8 90.3 79.5 Med42-Llama3.1-70B 66.1 86.8 72.4 80.4 94.5 77.6 90.4 81.2

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

Mistral-Large-Instruct-2407 66.4 87.5 68.3 75.9 85.8 56.2 91.1 75.9 GPT-4.0† - 87.0 69.5 78.9 84.1 75.2 - 78.9 Llama3.1-405B-Instruct 70.2 89.3 75.8 81.9 95.5 74.6 90.7 82.6

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

- Table 4: Performance of Med42-v2 models on key closed-ended medical benchmark (zero-shot) evaluations. We compare the performance with that of competing models, and we boldface and underline the best and second best-performing models (respectively) in each of the three model-size equivalence classes. †Performance results for GPT-4.0 have been reported in Nori et al. (2023a).

[Figure 134]

concerns, which are particularly critical in the medical ﬁeld. The reliance on high-quality, domainspeciﬁc data means that any gaps or biases in the training data could impact the model’s effectiveness. To address these concerns, our future work involves developing a new evaluation framework to assess the clinical utility of LLMs by testing them on real-world use cases. This framework will focus on evaluating clinical data understanding, safety, and reasoning capabilities, providing a more comprehensive understanding of how these models perform in practical, high-stakes environments. By rigorously testing LLMs in real-world scenarios, we aim to identify and mitigate potential risks, ensuring that models like Med42-v2 can be safely and effectively integrated into healthcare settings.

REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Stephen R Ali, Thomas D Dobbs, Hayley A Hutchings, and Iain S Whitaker. Using chatgpt to write patient clinic letters. The Lancet Digital Health, 5(4):e179–e181, 2023.

Mohammed Altaf. Medical instruction 120k, 2023. URL https://t.ly/Pn5lg.

Anthropic. Introducing the next generation of Claude — anthropic.com. https://t.ly/JQLxz,

2024. [Accessed 09-07-2024].

Hiroj Bagde, Ashwini Dhopte, Mohammad Khursheed Alam, and Rehana Basri. A systematic review and meta-analysis on chatgpt and its utilization in medical and dental research. Heliyon, 9

(12), 2023. Asma Ben Abacha and Dina Demner-Fushman. A question-entailment approach to question answering. BMC Bioinform., 20(1):511:1–511:23, 2019. URL https://t.ly/U1FFw.

Asma Ben Abacha, Wen-wai Yim, Yadan Fan, and Thomas Lin. An empirical study of clinical note generation from doctor-patient encounters. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pp. 2291– 2302, Dubrovnik, Croatia, May 2023. Association for Computational Linguistics. URL https://aclanthology.org/2023.eacl-main.168.

Marco Cascella, Jonathan Montomoli, Valentina Bellini, and Elena Bignami. Evaluating the feasibility of chatgpt in healthcare: an analysis of multiple clinical and research scenarios. Journal of medical systems, 47(1):33, 2023.

Zeming Chen, Alejandro Hernández Cano, Angelika Romanou, Antoine Bonnet, Kyle Matoba, Francesco Salvi, Matteo Pagliardini, Simin Fan, Andreas Köpf, Amirkeivan Mohtashami, et al. Meditron-70b: Scaling medical pretraining for large language models. arXiv preprint arXiv:2311.16079, 2023.

Clément Christophe, Praveen K Kanithi, Prateek Munjal, Tathagata Raha, Nasir Hayat, Ronnie Rajan, Ahmed Al-Mahrooqi, Avani Gupta, Muhammad Umar Salman, Gurpreet Gosal, et al. Med42–evaluating ﬁne-tuning strategies for medical llms: Full-parameter vs. parameter-efﬁcient approaches. arXiv preprint arXiv:2404.14779, 2024.

Lucas Lacerda de Souza, Felipe Paiva Fonseca, Manoela Domingues Martins, Oslei Paes de Almeida, Helder Antônio Rebelo Pontes, Fábio Luiz Coracin, Márcio Ajudarte Lopes, Syed Ali Khurram, Alan Roger Santos-Silva, Ahmed Hagag, et al. Chatgpt and medicine: a potential threat to science or a step towards the future? Journal of Medical Artiﬁcial Intelligence, 6, 2023.

Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations, 2023.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPoﬁ, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 12 2023. URL https://zenodo.org/records/10256836.

Team Gemini, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Tianyu Han, Lisa C Adams, Jens-Michalis Papaioannou, Paul Grundmann, Tom Oberhauser, Alexander Löser, Daniel Truhn, and Keno K Bressem. Medalpaca–an open-source collection of medical conversational ai models and training data. arXiv preprint arXiv:2304.08247, 2023.

Thomas Hartvigsen, Saadia Gabriel, Hamid Palangi, Maarten Sap, Dipankar Ray, and Ece Kamar. Toxigen: A large-scale machine-generated dataset for adversarial and implicit hate speech detection. arXiv preprint arXiv:2203.09509, 2022.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021.

Takanobu Hirosawa, Yukinori Harada, Masashi Yokose, Tetsu Sakamoto, Ren Kawamura, and Taro Shimizu. Diagnostic accuracy of differential-diagnosis lists generated by generative pretrained transformer 3 chatbot for clinical vignettes with common chief complaints: a pilot study. International journal of environmental research and public health, 20(4):3378, 2023.

Dongfu Jiang, Xiang Ren, and Bill Yuchen Lin. Llm-blender: Ensembling large language models with pairwise comparison and generative fusion. In Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics (ACL 2023), 2023.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. arXiv preprint arXiv:2009.13081, 2020.

Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. Pubmedqa: A dataset for biomedical research question answering. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 2567–2577, 2019.

Matt Gardner Johannes Welbl, Nelson F. Liu. Crowdsourcing multiple choice science questions. 2017.

Neema Kotonya and Francesca Toni. Explainable automated fact-checking for public health claims. arXiv preprint arXiv:2010.09926, 2020.

Nathan Lambert, Lewis Tunstall, Nazneen Rajani, and Tristan Thrush. Huggingface h4 stack exchange preference dataset, 2023. URL https://t.ly/2sE9E.

Hanzhou Li, John T Moon, Saptarshi Purkayastha, Leo Anthony Celi, Hari Trivedi, and Judy W Gichoya. Ethics of large language models in medicine and medical research. The Lancet Digital Health, 5(6):e333–e335, 2023.

Wing Lian, Bleys Goodson, Eugene Pentland, Austin Cook, Chanvichet Vong, and "Teknium". Openorca: An open dataset of gpt augmented ﬂan reasoning traces. https://https://huggingface.co/Open-Orca/OpenOrca, 2023.

Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. The ﬂan collection: Designing data and methods for effective instruction tuning. arXiv preprint arXiv:2301.13688, 2023.

Harsha Nori, Nicholas King, Scott Mayer McKinney, Dean Carignan, and Eric Horvitz. Capabilities of gpt-4 on medical challenge problems, 2023a.

Harsha Nori, Yin Tat Lee, Sheng Zhang, Dean Carignan, Richard Edgar, Nicolo Fusi, Nicholas King, Jonathan Larson, Yuanzhi Li, Weishung Liu, Renqian Luo, Scott Mayer McKinney, Robert Osazuwa Ness, Hoifung Poon, Tao Qin, Naoto Usuyama, Chris White, and Eric Horvitz. Can generalist foundation models outcompete Special-Purpose tuning? case study in medicine. November 2023b.

OpenChat, 2023. URL https://t.ly/UN-Hu.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35: 27730–27744, 2022.

Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Medmcqa: A largescale multi-subject multi-choice dataset for medical domain question answering. In Gerardo Flores, George H Chen, Tom Pollard, Joyce C Ho, and Tristan Naumann (eds.), Proceedings of the Conference on Health, Inference, and Learning, volume 174 of Proceedings of Machine Learning Research, pp. 248–260. PMLR, 07–08 Apr 2022. URL https://proceedings.mlr.press/v174/pal22a.html.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.

Khaled Saab, Tao Tu, Wei-Hung Weng, Ryutaro Tanno, David Stutz, Ellery Wulczyn, Fan Zhang, Tim Strother, Chunjong Park, Elahe Vedadi, Juanma Zambrano Chaves, Szu-Yeu Hu, Mike Schaekermann, Aishwarya Kamath, Yong Cheng, David G T Barrett, Cathy Cheung, Basil Mustafa, Anil Palepu, Daniel McDuff, Le Hou, Tomer Golany, Luyang Liu, Jean-Baptiste Alayrac, Neil Houlsby, Nenad Tomasev, Jan Freyberg, Charles Lau, Jonas Kemp, Jeremy Lai, Shekoofeh Azizi, Kimberly Kanada, Siwai Man, Kavita Kulkarni, Ruoxi Sun, Siamak Shakeri, Luheng He, Ben Caine, Albert Webson, Natasha Latysheva, Melvin Johnson, Philip Mansﬁeld, Jian Lu, Ehud Rivlin, Jesper Anderson, Bradley Green, Renee Wong, Jonathan Krause, Jonathon Shlens, Ewa Dominowska, S M Ali Eslami, Katherine Chou, Claire Cui, Oriol Vinyals, Koray Kavukcuoglu, James Manyika, Jeff Dean, Demis Hassabis, Yossi Matias, Dale Webster, Joelle Barral, Greg Corrado, Christopher Semturs, S Sara Mahdavi, Juraj Gottweis, Alan Karthikesalingam, and Vivek Natarajan. Capabilities of gemini models in medicine. April 2024.

Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chafﬁn, Arnaud Stiegler, Teven Le Scao, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, Ryan Teehan, Tali Bers, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M. Rush. Multitask prompted training enables zero-shot task generalization, 2022.

Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Le Hou, Kevin Clark, Stephen Pfohl, Heather Cole-Lewis, Darlene Neal, et al. Towards expert-level medical question answering with large language models. arXiv preprint arXiv:2305.09617, 2023.

SnorkelAI. snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset · Datasets at Hugging Face — huggingface.co. https://tinyurl.com/2ze38278, 2023. [Accessed 07-08-2024].

Arun James Thirunavukarasu, Darren Shu Jeng Ting, Kabilan Elangovan, Laura Gutierrez, Ting Fang Tan, and Daniel Shu Wei Ting. Large language models in medicine. Nature medicine, 29(8):1930–1940, 2023.

Augustin Toma, Patrick R Lawler, Jimmy Ba, Rahul G Krishnan, Barry B Rubin, and Bo Wang. Clinical camel: An open expert-level medical language model with dialogue-based knowledge encoding. arXiv preprint arXiv:2305.12031, 2023.

Hoang Tran, Chris Glaze, and Braden Hancock. Iterative dpo alignment. Technical report, Snorkel AI, 2023.

Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Shengyi Huang, Kashif Rasul, Alvaro Bartolome, Alexander M. Rush, and Thomas Wolf. The Alignment Handbook. URL https://github.com/huggingface/alignment-handbook.

Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, YounesBelkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, Nathan Sarrazin, Omar Sanseviero, Alexander M. Rush, and Thomas Wolf. Zephyr: Direct distillation of lm alignment, 2023.

David Vilares and Carlos Gómez-Rodríguez. HEAD-QA: A healthcare dataset for complex reasoning. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 960–966, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1092. URL https://www.aclweb.org/anthology/P19-1092.

Lucy Lu Wang, Kyle Lo, Yoganand Chandrasekhar, Russell Reas, Jiangjiang Yang, Doug Burdick, Darrin Eide, Kathryn Funk, Yannis Katsis, Rodney Michael Kinney, Yunyao Li, Ziyang Liu, William Merrill, Paul Mooney, Dewey A. Murdick, Devvret Rishi, Jerry Sheehan, Zhihong Shen, Brandon Stilson, Alex D. Wade, Kuansan Wang, Nancy Xin Ru Wang, Christopher Wilhelm,

Boya Xie, DouglasM. Raymond, Daniel S. Weld, Oren Etzioni, and Sebastian Kohlmeier. CORD19: The COVID-19 open research dataset. In Proceedings of the 1st Workshop on NLP for COVID-19 at ACL 2020, Online, July 2020. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/2020.nlpcovid19-acl.1.

Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Gary Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Maitreya Patel, Kuntal Kumar Pal, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Shailaja Keyur Sampat, Savan Doshi, Siddhartha Mishra, Sujan Reddy, Sumanta Patro, Tanay Dixit, Xudong Shen, Chitta Baral, Yejin Choi, Noah A. Smith, Hannaneh Hajishirzi, and Daniel Khashabi. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks, 2022.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. MMLU-Pro: A more robust and challenging Multi-Task language understanding benchmark. June 2024.

Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners, 2022.

Wen-wai Yim, Yujuan Fu, Asma Ben Abacha, Neal Snider, Thomas Lin, and Meliha Yetisgen. Aci-bench: a novel ambient clinical intelligence dataset for benchmarking automatic visit note generation. Nature Scientiﬁc Data, 2023.

A APPENDIX

[Figure 140]

Dataset # Samples Mixture ratio (%) Medical domain

[Figure 141]

MedMCQA (Pal et al., 2022) 180,462 13.92 Medical Flashcards (Han et al., 2023) 30,106 2.32 StackExchange† (Lambert et al., 2023) 64,246 4.96 MedQA (USMLE) (Jin et al., 2020) 11,290 0.87 CORD-19 (Wang et al., 2020) 17,721 1.37 PubMedQA (Jin et al., 2019) 499 0.04 HeadQA‡ (Vilares & Gómez-Rodríguez, 2019) 2,657 0.20 MediQA (Han et al., 2023) 1,950 0.15 SciQ (Johannes Welbl, 2017) 11,679 0.90 PubMed Causal (Han et al., 2023) 2,169 0.17 OpenGPT (OpenChat, 2023) 66,026 5.09 MedQUAD (Ben Abacha & Demner-Fushman, 2019) 14,553 1.12 MMLU$ (Hendrycks et al., 2021) 244 0.02 Niv2* (Wang et al., 2022) 11,447 0.88 Pubhealth (Kotonya & Toni, 2020) 9,804 0.76 Medical-Instruction (Altaf, 2023) 120,000 9.26 ACI-Bench (Yim et al., 2023) 87 0.01 MTS-Dialog (Ben Abacha et al., 2023) 2602 0.20

Total 952,942 73.50 General domain

[Figure 142]

SlimOrca T0 (Lian et al., 2023; Sanh et al., 2022) 109,235 8.43 SlimOrca Flan (Lian et al., 2023; Longpre et al., 2023) 109,169 8.42 SlimOrca CoT (Lian et al., 2023; Wei et al., 2022) 74,172 5.72 Ultrachat (Ding et al., 2023) 50,953 3.93

Total 343,529 26.50

[Figure 143]

† The following categories were included: “academia", “bioinformatics”, “biology", “cogsci", “ﬁtness", “health". ‡ Only samples in English were used. $ The following subjects were included: “anatomy", “clinical knowledge", “college medicine", “nutrition", “medical genetics", “professional medicine", “college biology", “high-school biology", “virology", “high-school psychology", “human sexuality", “human aging", and “professional psychology".

* Samples from 47 tasks (from over 1,000 tasks) related to science, healthcare and medicine were included.

- Table 5: Summary of subsets of the data used for ﬁne-tuning the models. Note that medical-domain data correspond to 73.5% of the entire dataset.

