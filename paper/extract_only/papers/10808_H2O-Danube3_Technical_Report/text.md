# arXiv:2407.09276v1[cs.CL]12Jul2024

## H2O-Danube3 Technical Report

Pascal Pfeiffer Philipp Singer Yauhen Babakhin Gabor Fodor Nischay Dhankhar Sri Satish Ambati

H2O.ai {firstname.lastname, sri}@h2o.ai

### 1 Abstract

We present H2O-Danube3, a series of small language models consisting of H2O-Danube3-4B, trained on 6T tokens and H2O-Danube3-500M, trained on 4T tokens. Our models are pre-trained on high quality Web data consisting of primarily English tokens in three stages with different data mixes before final supervised tuning for chat version. The models exhibit highly competitive metrics across a multitude of academic, chat, and fine-tuning benchmarks. Thanks to its compact architecture, H2O-Danube3 can be efficiently run on a modern smartphone, enabling local inference and rapid processing capabilities even on mobile devices. We make all models openly available under Apache 2.0 license further democratizing LLMs to a wider audience economically.

Danube3 model collection: https://huggingface.co/collections/h2oai/h2o-danube3-6687a993641452457854c609

### 2 Introduction

Small language models have taken a pivotal place in today’s open source language model landscape particularly aiming at efficient inference on consumer hardware and edge devices also allowing for full offline applications. Additionally, smaller models have proven to be particularly useful after fine-tuning them for specific tasks, such as sequence classification, question answering, or token classification even outpacing previously used encoder/decoder models such as those stemming from BERT and its derivatives [4, 7].

We extend previous research in this area [2, 3, 5, 10–12, 14, 15] and present H2O-Danube3, a series of small language models consisting of H2O-Danube3-4B, trained on 6T tokens and H2O-Danube3-500M, trained on 4T tokens based on incremental research and training efforts [11]. In this report, we present an overview of the models, detailing their architecture, training procedures, and fine-tuning processes. We offer extensive evaluations using a diverse range of benchmarks, encompassing both standard academic metrics, chat benchmarks, and fine-tuning benchmarks.

Results show that H2O-Danube3 exhibits competitive benchmarks across all dimensions, expanding the repertoire of open source small language models. We hope our work can further democratize language models to a wider audience and that our models can play a pivotal role for various use cases such as (1) chatbot applications, (2) RAG applications, (3) fine-tuning for specific use cases such as classification, (4) research or (5) on-device offline applications. To demonstrate the potential, we also present H2O AI Personal GPT1, an iOS application allowing to run H2O-Danube3 fully offline on a modern phone device.

1https://h2o.ai/platform/danube/personal-gpt/

Technical Report, work in progress.

[Figure 1]

[Figure 2]

[Figure 3]

Figure 1: Data stages for H2O-Danube3-4B. The model is trained over three different stages with different data mixes. The first data stage consist of 90.6% of web data which is gradually decreasing to 81.7% at the second stage, and to 51.6% at the third stage. The first two stages include the majority of the tokens: 4.6T and 1.35T tokens respectively, while the third stage comprises of 0.05T tokens.

### 3 Model architecture

H2O-Danube3 is a family of decoder only LLM models that use the general Llama model architecture adopting core principles from Llama 2 [13] and Mistral [8] with custom parameters determining the shape of each layer and total parameter count. We use the Mistral tokenizer with a vocabulary size of 32,000 and train our model up to a context length of 8,192. We make use of Grouped Query Attention [1] and optimize towards parameter and compute efficiency resulting in a wide architecture (see Table 1). In total, H2O-Danube3-4B consists of 3.96B trainable parameters. In addition, we release H2O-Danube3-500M with 500M trainable parameters for edge devices with limited compute or for custom fine-tuning tasks that require low memory footprint or high throughput at low cost.

Table 1: Key model parameters.

Parameters 500M 4B Layers 16 24 Hidden size 1536 3840 Intermediate size 4096 10240 Num heads 16 32 Num KV heads 8 8 Head size 96 120 Vocab size 32000 32000 RoPE theta 100000 100000

### 4 Training

Models are primarily trained on English text in three stages with different data mixes. At each stage, we gradually decrease the percentage of noisy web data in favor of higher quality data. The first data stage consist of 90.6% of web data which is gradually decreasing to 81.7% at the second stage, and to 51.6% at the third stage. Simultaneously, the share of instruct data, Wikipedia,

academic texts, synthetic texts and other higher quality textual data is increasing. The first two stages include the majority of the tokens: 4.6T and 1.35T tokens respectively (2.8T and 1.15T tokens for H2O-Danube3-500M), while third stage comprises of 0.05T tokens. The data distribution across stages is presented in Figure 1.

We also provide chat fine-tuned versions H2O-Danube3-4B-Chat and H2O-Danube3-500M-Chat. We utilize H2O LLM Studio2, an Apache 2.0 open-source framework and no-code GUI for fine-tuning LLMs. We tune the base model using supervised fine-tuning (SFT) on input/output conversational pairs. We mask the prompt loss, and use a custom prompt format. Hyperparameters were optimized iterating over multiple experiments.

### 5 Evaluation

In this section, we present evaluation of H2O-Danube3 across a variety of dimensions, focusing on

(1) academic benchmarks, (2) chat benchmarks and (3) fine-tuning benchmarks.

Academic benchmarks. We evaluate H2O-Danube3 on a wide range of benchmarks and compare it with other existing open-source language models which have a similar number of parameters, specifically Qwen/Qwen1.5-4B-Chat, stabilityai/stablelm-zephyr-3b and microsoft/Phi-3-mini-4k-instruct. We also compare to our previous model h2oai/h2o-danube2-1.8b-chat. To evaluate the models, we use the Language Model Evaluation Harness framework3 [6]. H2O-Danube3-4B shows very competitive and consistent results across all reported benchmarks (see Table 2). It is the best-in-class model for the knowledge based CommonsenseQA benchmark and PhysicsQA and achieves a strong accuracy of 50.14% on the math centered benchmark GSM8K. In all other benchmarks, H2O-Danube3-4B ranks second only after Phi-3-mini-4k-instruct which is well known for its outstanding reasoning capabilities and strong benchmark scores. Notably, H2O-Danube3-4B scores over 80% on 10-shot hellaswag benchmark, closing the gap to much larger models. The smaller H2O-Danube3-500M is evaluated against the same benchmarks and compared to similar sized Qwen2-0.5B-Instruct (see Table 3). Our model scores highest in eight out of twelve benchmarks and we consider it a new well rounded model for this parameter count.

- Table 2: Academic benchmarks. Academic benchmark results, compared to openly-available models of similar size and trained on general English text data. We compare the instruction finetuned models h2oai/h2o-danube2-1.8b-chat, h2oai/h2o-danube3-4b-chat, Qwen/Qwen1.5-4B-Chat, stabilityai/stablelm-zephyr-3b, microsoft/Phi-3-mini-4k-instruct. To evaluate the models, we use the Language Model Evaluation Harness framework [6].

Benchmark Metric Danube2 Danube3 Qwen1.5 StableLM Phi3

| |1.8B 4B<br><br>|4B 3B 4B|
|---|---|---|
|ARC-c 25-shot Hellaswag 10-shot MMLU 5-shot TruthfulQA 0-shot mc2 Winogrande 5-shot GSM8K 5-shot ARC-e 25-shot BBH 3-shot CoT CommonsenseQA 3-shot CoQA 0-shot F1 PIQA 3-shot SciQ 3-shot|43.69 58.96<br><br>73.91 80.36 37.83 54.74 40.53 47.79 69.30 76.48 32.30 50.18<br>74.92 83.84 30.39 38.92 54.30 79.52 68.30 77.23 78.67 82.64 95.70 97.10<br>|42.15 47.70 63.91 69.46 73.71 80.62 54.03 44.98 69.43 44.88 46.40 57.72 66.22 65.59 70.80<br><br>3.63 52.46 77.48 73.44 72.10 87.29 21.03 36.77 71.42 76.09 75.76 77.81 61.94 70.86 79.75 76.61 77.42 78.35 95.40 94.80 97.60<br><br>|
|Average|58.32 68.98|57.07 63.21 76.01|

2https://github.com/h2oai/h2o-llmstudio 3commit e5e5ee0cb629c9c88165292d1b4bf34623392d33

- Table 3: Academic benchmarks for smaller models. Academic benchmark results, compared to openly-available models of similar size and trained on general English text data. We compare the instruction fine-tuned models h2oai/h2o-danube3-500m-chat and Qwen/Qwen2-0.5B-Instruct. To evaluate the models, we use the Language Model Evaluation Harness framework [6].

Benchmark Metric Danube3 Qwen2 0.5B 0.5B

ARC-c 25-shot 39.25 32.00 Hellaswag 10-shot 61.02 49.11 MMLU 5-shot 26.33 43.88 TruthfulQA 0-shot mc2 39.96 39.28 Winogrande 5-shot 61.72 56.99 GSM8K 5-shot 16.00 34.12 ARC-e 25-shot 71.84 62.12 BBH 3-shot CoT 25.14 18.98 CommonsenseQA 3-shot 19.57 52.74 CoQA 0-shot F1 48.02 54.89 PIQA 3-shot 74.70 68.72 SciQ 3-shot 95.40 92.60

Chat benchmarks. Evaluating chat and instruct fine-tuned LLMs remains a critical challenge and can most reliably be conducted by large scale human assessment. In order to give an initial evaluation of our chat model, we resort to MT-Bench [16] and WildBench-v2 [9] benchmarks. They represent a collection of multi-turn questions across different categories followed by GPT-4 judgement which assigns a score from 1 to 10 for each model’s response. Results are presented in Table 4 showing that H2O-Danube3-4B-Chat is surpassing other similar sized models while Phi-3-mini takes the top spot. The 500M parameter version of the model H2O-Danube3-500M-Chat shows results that are comparable to Qwen2-0.5B-Instruct (see Table 5).

We additionally conducted multiple internal evaluations and show them in the same tables. First, we performed a blind evaluation of chat performance (excluding the 500M models) following the idea of Chat Arena4. This involved presenting users with random pairs of models and allowing them to prompt and vote on output preference (A better, B better, both bad, both good), followed by calculating an ELO score using MLE and bootstrapping. Second, we utilized an internal RAG (Retrieval-Augmented Generation) benchmark5 to assess the models performance in question-answering tasks based on long PDF documents. We calculated an accuracy score for each model by comparing its generated responses to the ground truth answers.

- Table 4: Chat benchmarks. H2O-Danube3-4B-Chat consistently performs very well across all benchmarks, surpassing other similar sized models and outperforming our previous Danube2 release, while Phi-3-mini takes the top spot.

Benchmark Metric Danube2 Danube3 Qwen1.5 StableLM Phi3 1.8B 4B 4B 3B 4B

|MT-Bench Turn 1<br>MT-Bench Turn 2 MT-Bench Average<br>|6.41 7.28<br><br>4.88 5.69<br>5.64 6.49<br>|6.68 7.10 8.38<br><br>5.33 5.74 7.58<br>6.00 6.42 7.98<br><br><br>|
|---|---|---|
|WildBench-v2 Raw score<br><br>|4.65 5.54|4.87 5.51 6.47|
|Internal Voting ELO|1531<br><br>|1466 1435 1564|
|RAG Benchmark Accuracy|66.88 73.37|67.53 68.18 73.37|

- 4https://chat.lmsys.org/
- 5https://github.com/h2oai/enterprise-h2ogpte/tree/main/rag_benchmark

- Table 5: Chat benchmarks for smaller models. The 500M parameter version of the model H2O-Danube3-500M-Chat shows results that are comparable to Qwen2-0.5B-Instruct model. In particular, they achieve a close MT-Bench average score (H2O-Danube3-500M-Chat being better in the 1st turn), while H2O-Danube3-500M-Chat produces better results on Wild-Bench benchmark.

Benchmark Metric Danube3 Qwen2 0.5B 0.5B

| | |
|---|---|
|MT-Bench Turn 1<br>MT-Bench Turn 2 MT-Bench Average<br>|4.16 3.78<br><br>2.40 2.76<br>3.28 3.27<br><br><br>|
|WildBench-v2 Raw score<br><br>|3.36 3.11|
|Internal RAG Benchmark Accuracy|44.16 50.00|

Fine-tuning benchmarks. A common application of small language models is their fine-tuning for various use cases to optimize performance on specific tasks. To that end, we also evaluate the different models’ capability to be easily adaptable to new tasks, here focusing on text classification observed frequently across various use cases in businesses and applications.

We employ the following process for our fine-tuning benchmarks. We utilize H2O LLM Studio offering out-of-the-box for tuning language models for classification feeding the final token logit distribution in a custom head for classification. For all models and datasets, we use the same settings with LoRA (r = 16, α = 32) and same hyperparameters (bs = 1, epochs = 1, lr = 1e − 4, diff_lr = 1e − 05, max_length = 8192). These settings are commonly used default settings in the field and we aim at particularly evaluating the default performance of models after tuning. We look at the following datasets, that all can be found on Hugging Face:

- • stanfordnlp/imdb: binary sentence classification of imdb movie reviews
- • knowledgator/Scientific-text-classification: classification of scientific texts into 10 most frequent classes, random 50-50 split
- • ccdv/arxiv-classification: long context classification of arxiv papers into 11 classes
- • ccdv/patent-classification: long context classification of patents into 9 classes

- Table 6 highlights the results for individual datasets and models, we always report the accuracy taking the highest probability class. We can see, that all small language models show excellent performance on text classification tasks after fine-tuning. Even small 500M parameter models can be highly competitive, exemplifying the utility of fine-tuning such models for specific use cases. Overall, H2O-Danube3-4B takes a leading spot in all benchmarks. These results can be seen as baseline results based on default hyperparameter settings. More extensive parameter sweeps would potentially improve results of all models at hand, and might also alter the order of performance. We plan on investigating such fine-tuning performance more extensively in the future.

- Table 6: Fine-tuning benchmarks. All models show excellent performance on various classification tasks after fine-tuning with H2O-Danube3-4B taking a top spot in most benchmarks.

Dataset Danube2 Danube3 Danube3 Qwen1.5 Qwen2 StableLM Phi3

| |1.8B 0.5B 4B|4B 0.5B 3B 4B<br><br>|
|---|---|---|
|Arxiv Imdb Patent Scientific|0.864 0.863 0.873 0.968 0.959 0.971 0.721 0.708 0.727 0.868 0.846 0.872<br><br>|0.877 0.874 0.865 0.869 0.970 0.959 0.969 0.967 0.717 0.707 0.719 0.712 0.875 0.855 0.867 0.870|
|Average|0.855 0.844 0.861|0.86 0.849 0.855 0.854|

- Table 7: Model quantization. This table summarizes different quantized versions of H2O-Danube3-4B-Chat showing the trade-off between size and quality of the models. Results indicate that quantized models reduced to 4 bits exhibit minimal loss in benchmark performance.

Quant method Model size MT-Bench Perplexity

F16 7.92 GB 6.43 6.17 Q8_0 4.21 GB 6.49 6.17 Q6_K 3.25 GB 6.37 6.20 Q5_K_M 2.81 GB 6.25 6.24 Q4_K_M 2.39 GB 6.31 6.37 Q3_K_M 1.94 GB 5.87 6.99 Q2_K 1.51 GB 3.71 9.42

### 6 Model quantization

To facilitate the use of our models on edge devices, we introduce quantized versions of H2O-Danube3-4B-Chat and H2O-Danube3-500M-Chat. They are available in the H2O-Danube3 Hugging Face collection and contain GGUF format model files that were quantized using the llama.cpp6 framework.

- Table 7 summarizes different quantized versions of H2O-Danube3-4B-Chat. It shows the tradeoff between size and quality of different quantization methods. Columns in the table represent quantization method, size of the model in gigabytes, MT-Bench [16] benchmark score, and perplexity metric on WikiText-2 dataset (as reported in a perplexity test from llama.cpp). Results suggest that we can reduce the model size by a factor of 3.3 (4-bit quantization) keeping the quality of the model almost the same, but going to 3-bit quantization already decreases the performance significantly.

### 7 Conclusions

We introduce H2O-Danube3, a series of small language models consisting of H2O-Danube3-4B and H2O-Danube3-500M released open source under Apache 2.0. Our models show competitive performance compared to popular models of similar size across a wide variety of benchmarks including (1) academic benchmarks, (2) chat benchmarks, as well as (3) fine-tuning benchmarks. H2O-Danube3 is built on our continuous efforts to contribute to the growing ecosystem of open source small language models. We are confident that our models can play a pivotal role in a wide range of applications, from typical chatting and fine-tuning for specific use cases to on-device offline applications on mobile phones or edge devices.

### References

- [1] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023.
- [2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [3] Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pages 2397–2430. PMLR, 2023.
- [4] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.
- [5] Marah Abdin et al. Phi-3 technical report: A highly capable language model locally on your phone, 2024. 6https://github.com/ggerganov/llama.cpp

- [6] Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, September 2021.
- [7] Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. Deberta: Decoding-enhanced bert with disentangled attention. arXiv preprint arXiv:2006.03654, 2020.
- [8] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.
- [9] Bill Yuchen Lin, Yuntian Deng, Khyathi Chandu, Faeze Brahman, Abhilasha Ravichander, Valentina Pyatkin, Nouha Dziri, Ronan Le Bras, and Yejin Choi. Wildbench: Benchmarking llms with challenging tasks from real users in the wild, 2024.
- [10] Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen Lai, Yuandong Tian, Igor Fedorov, Yunyang Xiong, Ernie Chang, Yangyang Shi, Raghuraman Krishnamoorthi, et al. Mobilellm: Optimizing sub-billion parameter language models for on-device use cases. arXiv preprint arXiv:2402.14905, 2024.
- [11] Philipp Singer, Pascal Pfeiffer, Yauhen Babakhin, Maximilian Jeblick, Nischay Dhankhar, Gabor Fodor, and Sri Satish Ambati. H2o-danube-1.8b technical report, 2024.
- [12] Stability AI Language Team. Introducing stable lm 2 1.6b. Last accessed on 2024-01-22. https: //stability.ai/news/introducing-stable-lm-2.
- [13] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [14] Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. Tinyllama: An open-source small language model. arXiv preprint arXiv:2401.02385, 2024.
- [15] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.
- [16] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv:2306.05685, 2023.

