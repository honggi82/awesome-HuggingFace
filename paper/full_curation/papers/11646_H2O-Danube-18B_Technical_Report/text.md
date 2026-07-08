# arXiv:2401.16818v2[cs.CL]15Apr2024

## H2O-Danube-1.8B Technical Report

Philipp Singer∗ Pascal Pfeiffer∗ Yauhen Babakhin Maximilian Jeblick Nischay Dhankhar Gabor Fodor Sri Satish Ambati

H2O.ai {firstname.lastname, sri}@h2o.ai

### 1 Abstract

We present H2O-Danube, a series of small 1.8B language models consisting of H2O-Danube-1.8B, trained on 1T tokens, and the incremental improved H2O-Danube2-1.8B trained on an additional 2T tokens. Our models exhibit highly competitive metrics across a multitude of benchmarks and, as of the time of this writing, H2O-Danube2-1.8B achieves the top ranking on Open LLM Leaderboard for all models below the 2B parameter range. The models follow core principles of LLama 2 and Mistral, and we leverage and refine various techniques for pre-training large language models. We additionally release chat models trained with supervised fine-tuning followed by direct preference optimization. We make all models openly available under Apache 2.0 license further democratizing LLMs to a wider audience economically.

Danube2 base model: https://huggingface.co/h2oai/h2o-danube2-1.8b-base Danube2 chat model: https://huggingface.co/h2oai/h2o-danube2-1.8b-chat

### 2 Introduction

Research over the past few years has significantly enhanced language models’ capabilities, making them pivotal in tasks like text and code generation, question answering, translation, summarization, and more [44]. Most state-of-the-art large language models (LLMs) leverage decoder attention architectures [43] popularized by the series of GPT models [7, 36, 37] exemplifying the benefits of pre-training such models on extensive text corpora.

Scaling laws for LLMs suggest that performance scales by factors such as model and dataset size, as well as computational resources for training [27]. This has led to the development of a plethora of models, ranging in size to optimize performance given certain data and compute constraints; notable representatives are: Falcon [35], Llama 2 [42], Qwen [3], Mistral [24], or Mixtral [25].

Despite the trend towards larger models, smaller LLMs have taking an important place in today’s landscape allowing for efficient inference on consumer hardware and edge devices. While larger models often times excel across various generic tasks [3, 24, 42], fine-tuning smaller models for specific tasks can enable competitive performance with benefits of model size and inference speed [16], a concept also proven by the success of BERT and its derivatives [13, 19].

In this report, we want to extend previous research in this area [3, 5, 41, 49, 50] and present a series of models based on incremental research and training efforts. We release all models with open weights under Apache 2.0. The first part describes the initial H2O-Danube-1.8B model, as trained on 1T tokens, and a separation Section 7 describes H2O-Danube2-1.8B, a continued modeling effort trained on additional 2T tokens. In order to transparently elaborate our incremental insights, the first part is identical to an earlier version of this report2, while the second part highlights new insights of the second iteration.

∗The first two authors contributed equally. 2https://arxiv.org/abs/2401.16818v1

Technical Report, work in progress.

Fundamentally, H2O-Danube follows a decoder LLM architecture adopting core principles from Llama 2 [42] and Mistral [24]. The models are trained on a combination of, but not limited to, web documents, encyclopedia and public knowledge databases, excluding coding data. H2O-Danube2-1.8B is trained on a a more diverse mix of data over multiple data stages. Compared to recent models released in this parameter range [3, 41, 49], our models demonstrate to be highly competitive across various benchmarks. As of this writing, H2O-Danube2-1.8B is the highest ranked open model on the Hugging Face Open LLM Leaderboard3 for models below the 2B range. Alongside the base modes, we release chat variants, enhanced with supervised fine-tuning on instruction data and preference data optimization (DPO).

### 3 Model architecture

We adjust the Llama 2 architecture [42] for a total of around 1.8b parameters with a hidden size of 2,560, an intermediate size of 6,912, and a total of 24 hidden layers. We use the original Llama 2 tokenizer with a vocabulary size of 32,000 and train our model up to a context length of 16,384 (see Section 4). In the following, we elaborate more details about the architecture of H2O-Danube-1.8B.

Sliding window. We adopt the sliding window approach for local attention popularized by Mistral [24] as implemented in FlashAttention-2 [12]. For training, we use a fixed sliding window of 4,096.

Rotary Positional Embedding. To model dependencies of elements at different positions of a sequence, we use the Rotary Positional Embedding (RoPE) technique as introduced by Su et al. [40] and successfully being applied in multiple popular foundation models [24, 42].

Grouped-query attention. For reducing the memory bandwidth overhead, we utilize grouped-query attention [1], setting our architecture to use 32 attention heads and 8 key-value heads.

Further details. We rely on root mean square layer normalization (RMSNorm) [48] separately for pre- and post-normalization to stabilize training as commonly used in modern LLMs [42]. We do not use bias within linear layers nor tie word embeddings.

### 4 Training

We train on a single node consisting of 8xH100 GPUs. With Distributed Data Parallel (DDP), each GPU holds a full copy of the model. For finding good settings for our training routine and hyperparameters, we conducted initial experiments on smaller subsets of the data and model sizes up to 500M parameters.

Among other findings, these initial experiments showed, that for higher token throughput and compute efficiency, we can iteratively increase the sequence length during the training using a constant sliding window of 4,096 (see Section 3). Out of the 1T tokens in total, we train subsequently on

- • 700B tokens with a sequence length of 2,048,
- • 100B tokens with a sequence length of 4,096,
- • 100B tokens with a sequence length of 8,192,
- • 100B tokens with a sequence length of 16,384.

We employ recent advances in 8-bit floating-point (FP8) calculations on the Hopper architecture to further speed up the training. For this, we cast all linear layers in the Grouped-Query Attention and in the Multi-Layer Perceptron to FP8. The lm_head layer remains in bfloat16 precision to ensure training stability.

We use AdamW optimizer [30] with β1 = 0.9 and β2 = 0.95 as well as a cosine learning rate scheduler (see Figure 1). We warm up the learning rate for ∼2.36B tokens to a maximum of 2e − 4 and then decay it to a minimum of 1e − 5. Our total batch size across GPUs is ∼1.18M tokens, weight decay is 1.e − 1 and gradient clipping threshold is set to 1.0. With these settings, we achieved an average throughput of 292.7k tokens/s on the single node during training.

3https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Figure 1: Training logs. Training (top left) and validation (top right) cross-entropy loss, learning rate schedule (bottom left) and sequence length (bottom right). X-axis shows the number of tokens that have been trained up to the step.

### 5 Results

We evaluate H2O-Danube-1.8B on a wide range of benchmarks and compare it with other existing open-source language models which have a similar number of parameters. Such models include TinyLlama with 1.1B parameters [49], Falcon with 1.3B parameters [35], OPT with 1.3B and 2.7B parameters [50], Cerebras-GPT with 1.3B and 2.7B parameters [14], Pythia-deduped with 1.4B and 2.8B parameters [5], Qwen with 1.8B parameters [3], and most recent Stable LM 2 with 1.6B parameters [41]. The majority of these models have Apache 2.0 license, however, Stable LM 2 and Qwen require additional conditions for commercial use and are marked with an asterisk in Table 1.

To evaluate the models, we use the Language Model Evaluation Harness framework [17]. Specifically, we use the version of the framework that is utilized in Open LLM Leaderboard [4]. We report model capabilities across a wide variety of benchmark domains: commonsense reasoning, world knowledge, reading comprehension and an aggregated Open LLM Leaderboard benchmark.

Commonsense Reasoning. In Table 1, we present six standard common sense reasoning benchmarks in 0-shot setting: ARC easy and challenge [9], HellaSwag [47], OpenBookQA [31], PIQA [6], Winogrande [39].

World Knowledge. We evaluate 5-shot performance on TriviaQA [26] which represents a closedbook question answering benchmark. Results are presented in Table 1.

Reading Comprehension. We report 0-shot performance on BoolQ [8] in Table 1.

Open LLM Leaderboard. It evaluates models on six key benchmarks which test a variety of reasoning and general knowledge across a wide range of fields in 0-shot and few-shot settings. It consists of the following benchmarks: ARC challenge (25-shot) [9], HellaSwag (10-shot) [47], MMLU (5-shot) [20], TruthfulQA (0-shot) [29], Winogrande (5-shot) [39], GSM8k (5-shot) [10]. Results are presented in Table 2

For each model in Table 1 we report its number of parameters and the total number of tokens it was trained on. H2O-Danube-1.8B achieves good results across all the commonsense reasoning, world knowledge and reading comprehension benchmarks compared to other models of a similar size. The closest competitors are Qwen and Stable LM 2 models. H2O-Danube-1.8B shows better performance than Qwen on all the benchmarks except for BoolQ. Qwen model has the same 1.8B parameters but was trained on 2.2 times more tokens – 2.2T. At the same time, H2O-Danube-1.8B is slightly worse than Stable LM 2 on the majority of the benchmarks, while Stable LM 2 was trained on four times more tokens – 2T tokens for 2 epochs. Also, it is important to note that neither Qwen nor Stable LM 2 models have the Apache 2.0 license requiring additional conditions for commercial use.

Similarly, H2O-Danube-1.8B, Qwen and Stable LM 2 are the strongest models on Open LLM Leaderboard (see Table 2) having comparable results on the majority of the benchmarks except for MMLU and GSM8k. A potential explanation for such a behavior might be specifically tailored data that was used for training of Qwen and Stable LM 2 models improving some particular benchmarks, for example, Qwen used gsm8k-ScRel dataset [46] for better math reasoning.

### 6 Chat Fine-Tuning

One of the most common use cases for LLMs evolves around instructing and chatting. We thus also provide a chat fine-tuned version H2O-Danube-1.8B-Chat released under Apache 2.0. We utilize H2O LLM Studio4, an Apache 2.0 open-source framework and no-code GUI for fine-tuning LLMs.

#### 6.1 Supervised fine-tuning

As first step, we tune the base model using supervised fine-tuning (SFT) on input/output conversational pairs. In detail, we combine the following datasets totalling 157k samples: OpenOrca [33] following the steps outlined in Orca [32], MetaMathQA [45], UltraChat200k [15, 22], and Oasst2 [34].

We train all layers of the model for a single epoch using a learning rate of 1e − 5, a batch size of 8, and using cosine learning rate scheduling with a short warmup. We use the full pre-trained context length of 16,384, mask the prompt loss, and use a custom prompt format. Hyperparameters were optimized iterating over multiple experiments.

4https://github.com/h2oai/h2o-llmstudio

- Table 1: Commonsense reasoning, world knowledge and reading comprehension benchmarks. H2O-Danube-1.8B exhibits consistently good results across all the benchmarks compared to other models of a similar size. It shows better performance than Qwen on all the benchmarks except for BoolQ, being of the same size but trained on 2.2 times fewer tokens. Stable LM 2 slightly outperforms H2O-Danube-1.8B on the majority of the benchmarks, but was trained on four times the number of tokens. Moreover, neither Qwen nor Stable LM 2 models have the Apache 2.0 license requiring additional conditions for commercial use.

Model Size Tokens ARC-e ARC-c Bool HS OB PIQA Triv Wino

acc_n acc_n acc acc_n acc_n acc_n em acc

TinyLlama 1.1B 3.00T 55.35 30.12 57.80 59.18 36.00 73.18 28.78 58.88 Falcon 1.3B 0.35T 57.32 32.00 62.84 61.52 35.20 74.65 27.27 60.77

- 1.3B 0.18T 50.93 29.52 57.71 53.73 33.40 72.52 15.67 59.59
- 2.7B 0.18T 54.34 31.31 60.31 60.61 35.20 74.81 22.38 60.85

OPT

- 1.3B 0.03T 45.88 25.26 59.36 38.37 29.20 66.76 05.49 52.01
- 2.7B 0.05T 52.53 27.30 59.20 48.84 32.00 70.89 12.41 55.96

Cerebras

- 1.4B 0.30T 56.57 29.86 56.76 54.40 33.20 72.36 18.46 56.20
- 2.8B 0.30T 58.96 32.68 64.19 59.45 35.60 73.78 24.39 58.17

Pythia

Qwen* 1.8B 2.20T 58.25 34.98 67.13 58.82 33.40 72.85 23.92 58.96 Stable LM 2* 1.6B 4.00T 67.63 39.08 75.60 68.78 40.00 76.39 33.84 63.30 H2O-Danube 1.8B 1.00T 62.29 35.84 65.81 68.20 37.60 76.93 38.99 61.96

- Table 2: Open LLM Leaderboard. For each model in the table we report all the individual benchmarks, the average score and the average score without GSM8k benchmark. H2O-Danube-1.8B shows the results similar to Qwen and Stable LM 2 models on the majority of the benchmarks apart from GSM8k and MMLU. It can be explained by the data used for model training, for example, Qwen used gsm8k-ScRel dataset [46] for the better math reasoning.

Model Size ARC HS MMLU TQA Wino GSM Average Average 25-shot 10-shot 5-shot 0-shot 5-shot 5-shot no GSM

TinyLlama 1.1B 33.87 60.31 26.04 37.32 59.51 01.44 36.42 43.41 Falcon 1.3B 35.07 63.56 25.28 35.96 62.04 00.53 37.07 44.38

- 1.3B 29.52 54.53 24.96 38.71 59.75 00.15 34.60 41.49
- 2.7B 33.96 61.43 25.43 37.43 61.96 00.23 36.74 44.04

OPT

- 1.3B 26.28 38.54 26.59 42.70 53.43 00.23 31.30 37.51
- 2.7B 29.10 49.29 25.17 41.37 54.14 00.45 33.25 39.81

Cerebras

- 1.4B 32.68 54.96 25.56 38.66 57.30 00.83 35.00 41.83
- 2.8B 36.26 60.66 26.78 35.56 60.22 00.83 36.72 43.90

Pythia

Qwen* 1.8B 37.71 58.87 46.37 39.41 61.72 24.41 44.75 48.82 Stable LM 2* 1.6B 43.52 70.45 39.08 36.81 65.82 17.36 45.51 51.14 H2O-Danube 1.8B 39.68 69.75 25.97 33.63 64.17 02.05 39.21 46.64

#### 6.2 DPO

We follow SFT, by direct preference optimization (DPO) [38] using a combination of the following datasets: UltraFeedback Binarized [11], Orca DPO Pairs [23] and Distilabel Math Preference DPO [2]. The DPO model is trained using LoRA [21] with r = 4, alpha =16 for one epoch using a batch size of 2 with a learning rate of 1e − 5 using cosine learning rate decay, and beta = 0.2 for DPO loss.

Afterwards, we run a final DPO fine-tune using Oasst2 [34] dataset building preference pairs from ranks where the chosen answer is the lowest rank, and the rejected answer is the highest one, limited to only English conversations totalling around 5k samples. The training run uses similar hyperparameters as the previous one, just a lower learning rate of 3e − 6.

#### 6.3 Evaluation

Evaluating chat and instruct fine-tuned LLMs remains a critical challenge and can most reliably be conducted by large scale human assessment. In order to give an initial evaluation of our chat model, we resort to MT-Bench, a collection of multi-turn questions across different categories followed by judgement by GPT4 [51]. We keep all categories apart from coding which is out of scope for H2O-Danube-1.8B. Each model is run with repetition_penalty = 1.1 and temperature = 0.0 to reduce randomness and a more fair comparison between models.

We compare our results to popular chat models below 2B parameters and highlight them in Table 3 showing that H2O-Danube-1.8B-Chat is exhibiting strong results across categories, particularly for natural language tasks as focused on here. For single turn conversations, H2O-Danube-1.8B-Chat is the best model for five out of seven categories, and on average on-par with Stablelm 2. For turn 2, we can see that it is comparable to Qwen 2, while Stablelm 2 outperforms other models.

We make the intermediate sft version5 as well as the final DPO model weights6 available online. We plan on exploring further improvements for the chat version in the future, working on SFT as well as improved DPO. Particularly, we aim at enhancing preference data with multi turn conversations. We also hope for the open source community to further fine-tune our models for various use cases.

Additionally, we also evaluate chat models on commonsense reasoning, world knowledge, reading comprehension and aggregated Open LLM Leaderboard benchmarks. Similarly as for base models, we report 0-shot benchmark results of the chat versions of H2O-Danube-1.8B, TinyLlama, Qwen and

- 5https://huggingface.co/h2oai/h2o-danube-1.8b-sft
- 6https://huggingface.co/h2oai/h2o-danube-1.8b-chat

Stable LM 2 in Table 4, and Open LLM Leaderboard results are available in Table 5. We show that H2O-Danube-1.8B-Chat and Stablelm-2-Zephyr perform better than Qwen-Chat and TinyLlamaChat models on the majority of the benchmarks while being on par between each other. Only exceptions are, again, MMLU and GSM8k benchmarks. As we mentioned in Section 5, one of the potential explanations for the worse H2O-Danube-1.8B performance might be a specifically tailored data that was used for training of Qwen and Stable LM 2 base models to optimize those benchmarks.

- Table 3: Mt-bench chat benchmark. Both turn 1 and 2 evaluations for mt-bench (ex. coding category) highlight the excellent performance of H2O-Danube-1.8B-Chat, particularly for single turn conversations showing the highest Mt-bench scores for multiple categories and the average.

TinyLlama-1.1B-Chat Qwen-1.8B-Chat Stablelm-2-Zephyr-1.6B H2O-Danube-1.8B-Chat

- Turn 1

Extraction 2.50 4.70 5.20 3.40 Humanities 8.15 6.60 9.20 8.90 Math 1.20 2.40 3.50 3.80 Reasoning 3.10 3.50 3.50 4.30 Roleplay 5.60 6.70 6.80 7.05 STEM 6.60 6.50 7.35 8.10 Writing 8.65 9.20 9.35 9.35

Average 5.11 5.66 6.41 6.41

- Turn 2

Extraction 1.20 2.40 4.80 3.20 Humanities 8.10 7.30 9.70 8.90 Math 1.40 1.60 1.60 1.50 Reasoning 2.30 3.90 3.20 2.70 Roleplay 5.60 6.70 6.95 6.90 STEM 4.60 5.80 6.80 6.10 Writing 2.70 4.80 7.50 3.10

Average 3.70 4.64 5.79 4.63

- Table 4: Commonsense reasoning, world knowledge and reading comprehension benchmarks for chat models. H2O-Danube-1.8B-Chat outperforms TinyLlama-Chat and Qwen-Chat models, and is on-par with Stablelm-2-Zephyr on all 0-shot benchmarks for commonsense reasoning.

Model Size ARC-e ARC-c Bool HS OB PIQA Triv Wino

acc_n acc_n acc acc_n acc_n acc_n em acc

TinyLlama-1.1B-Chat 1.1B 54.34 33.36 60.83 60.39 37.20 74.59 29.04 59.91 Qwen-1.8B-Chat 1.8B 49.28 32.94 67.74 54.73 34.60 71.82 19.04 59.43 Stablelm-2-Zephyr-1.6B 1.6B 60.35 39.25 80.18 68.85 39.60 74.92 31.46 64.48 H2O-Danube-1.8B-Chat 1.8B 67.51 39.25 77.89 67.60 39.20 76.71 36.29 65.35

- Table 5: Open LLM Leaderboard for chat models. H2O-Danube-1.8B-Chat outperforms TinyLlama-Chat, and shows similar results to Qwen-Chat and Stablelm-2-Zephyr models apart from GSM8k and MMLU, as also already imminent from results on base models discussed in Table 2.

Model Size ARC HS MMLU TQA Wino GSM Average

25-shot 10-shot 5-shot 0-shot 5-shot 5-shot

TinyLlama-1.1B-Chat 1.1B 36.01 61.05 25.04 37.86 60.77 02.35 37.18 Qwen-1.8B-Chat 1.8B 36.95 54.34 44.55 43.70 58.88 19.26 42.94 Stablelm-2-Zephyr-1.6B 1.6B 43.69 69.34 41.85 45.21 64.09 35.18 49.89 H2O-Danube-1.8B-Chat 1.8B 41.47 68.02 33.49 40.82 64.40 15.54 43.96

### 7 H2O-Danube2-1.8B

In our effort to grow the ecosystem of permissive open-source foundation models, we publish a new set of models called H2O-Danube2-1.8B. The base model was initialized from H2O-Danube-1.8B and trained for additional 2T tokens. This second iteration of H2O-Danube is the result of extensive experimentation on smaller models, and significantly improves the performance.

The most significant changes that we have made compared to H2O-Danube-1.8B include:

- • Removal of sliding window attention and change of the maximum context length to 8,192. By doing so, we effectively improve the long context behavior of the model while keeping memory footprint similar.
- • Change the tokenizer to Mistral which showed superior performance in our experimentation. Instead of fully re-training the embedding and head layers, we re-map the matching tokens and only randomly re-initialize the new tokens.
- • We improve the quality of underlying training data by applying heuristics as well as small models (GBM and BERT) predicting the quality of respective input samples.
- • Training the model in three stages with different data mixes. At each stage, we gradually decrease the percentage of noisy web data in favor of higher quality data. The first data stage consist of 84.5% of web data which is gradually decreasing to 72.8% at the second stage, and to 55.5% at the third stage. Simultaneously, the share of instruct data, Wikipedia, academic texts and other higher quality textual data is increasing. The first two stages include the majority of the tokens: 1T and 0.95T tokens respectively, while third stage comprises of 0.05T tokens. The data distribution across stages is presented in Figure 2.

Given these adjustments and the continuous training of 2T additional tokens, we were able to significantly improve the performance of H2O-Danube. Since H2O-Danube-1.8B release, there were a couple of new open-weights released in the small models space. For the comparison of base models, we will be using the leading models from Open LLM Leaderboard [4] in the category of ∼1.5B parameters (up to 2B parameters); namely, Phi-1.5 [28], Qwen1.5-1.8B [3] and StableLM2-1.6B [41]. We are also comparing to Gemma-2B [18] with 2.5B parameters. We report OpenLLM Leaderboard results in Table 6. We can see, that in comparison to the first iteration reported in Table 2, we can improve on all benchmarks significantly. As of this writing, H2O-Danube2-1.8B7 is the highest scoring open model as measured by the average used for the official ranking.

On top of an improved base model, we were also able to develop better chat models following the concepts as described in Section 6. We make the intermediate sft version8 as well as the final DPO model weights9 available online. The final MT-Bench across all categories and as calculated in the official repository results in a score of 6.23 for the first turn, 5.34 for the second turn, and a final average score of 5.79.

- Table 6: Danube2 Open LLM Leaderboard. For each model in the table we report all the individual benchmarks and the average score. H2O-Danube2-1.8B achieves state-of-the-art results on this Leaderboard on the average of all benchmarks.

Model Size ARC HS MMLU TQA Wino GSM Average

25-shot 10-shot 5-shot 0-shot 5-shot 5-shot

Stable LM 2 1.6B 43.34 70.45 38.95 36.78 64.56 17.44 45.25 Gemma-2B 2.5B 48.46 71.65 41.68 33.13 66.77 17.36 46.51 Qwen1.5 1.8B 37.88 61.42 46.71 39.43 60.30 33.59 46.55 Phi-1.5 1.3B 52.90 63.79 43.89 40.89 72.22 12.43 47.69 H2O-Danube 1.8B 39.42 69.58 25.94 33.86 64.48 01.44 39.12 H2O-Danube2 1.8B 43.34 72.95 40.20 38.01 68.03 29.80 48.72

- 7https://huggingface.co/h2oai/h2o-danube2-1.8b-base
- 8https://huggingface.co/h2oai/h2o-danube2-1.8b-sft
- 9https://huggingface.co/h2oai/h2o-danube2-1.8b-chat

[Figure 5]

[Figure 6]

[Figure 7]

Figure 2: Data stages for Danube2. The model is trained over three different stages with different data mixes. The first data stage consist of 84.5% of web data which is gradually decreasing to 72.8% at the second stage, and to 55.5% at the third stage. The first two stages include the majority of the tokens: 1T and 0.95T tokens respectively, while third stage comprises of 0.05T tokens.

### 8 Conclusions

We introduce H2O-Danube, a series of new open-source pre-trained foundation model with 1.8B parameters including H2O-Danube-1.8B trained on 1T tokens and an improved second iteration H2O-Danube2-1.8B trained on additional 2T tokens from diverse sources. The Apache 2.0 license allows for commercial use and for further fine-tuning by the community. We also release a SFT + DPO fine-tuned chat versions, exhibiting state-of-the art results in commonsense reasoning, world knowledge and reading comprehension benchmarks. We show that H2O-Danube-1.8B-Chat outperforms other models of a similar size on multiple benchmarks. H2O-Danube is our first contribution to the growing ecosystem of permissive open-source foundation models and we strive to continue publishing high quality foundation models and chat fine-tunes in the near future. Notably, small models can be used on consumer hardware further democratizing LLMs to a wider audience economically.

### References

- [1] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023.
- [2] argilla. Distilabel math preference dpo, 2023. Last accessed on 2024-01-15. https://huggingface. co/datasets/argilla/distilabel-math-preference-dpo.
- [3] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [4] Edward Beeching, Clémentine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Rajani, Omar Sanseviero, Lewis Tunstall, and Thomas Wolf. Open llm leaderboard. https://huggingface. co/spaces/HuggingFaceH4/open_llm_leaderboard, 2023.
- [5] Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pages 2397–2430. PMLR, 2023.

- [6] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, pages 7432–7439, 2020.
- [7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [8] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044, 2019.
- [9] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.
- [10] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [11] Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. Ultrafeedback: Boosting language models with high-quality feedback, 2023.
- [12] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022.
- [13] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.
- [14] Nolan Dey, Gurpreet Gosal, Zhiming, Chen, Hemant Khachane, William Marshall, Ribhu Pathria, Marvin Tom, and Joel Hestness. Cerebras-gpt: Open compute-optimal language models trained on the cerebras wafer-scale cluster, 2023.
- [15] Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. arXiv preprint arXiv:2305.14233, 2023.
- [16] Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, and Tushar Khot. Specializing smaller language models towards multi-step reasoning. arXiv preprint arXiv:2301.12726, 2023.
- [17] Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, September 2021.
- [18] Thomas Mesnard Gemma Team, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, Pouya Tafti, Léonard Hussenot, and et al. Gemma. 2024.
- [19] Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. Deberta: Decoding-enhanced bert with disentangled attention. arXiv preprint arXiv:2006.03654, 2020.
- [20] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Xiaodong Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300,

- 2020.

[21] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685,

- 2021.

- [22] HuggingFaceH4. ultrachat_200k, 2023. Last accessed on 2024-01-15. https://huggingface.co/ datasets/HuggingFaceH4/ultrachat_200k.
- [23] Intel. Orca dpo pairs, 2023. Last accessed on 2024-01-15. https://huggingface.co/datasets/ Intel/orca_dpo_pairs.
- [24] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

- [25] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mixtral of experts, 2024.
- [26] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551, 2017.
- [27] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.
- [28] Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need ii: phi-1.5 technical report, 2023.
- [29] Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods, 2022.
- [30] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [31] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. arXiv preprint arXiv:1809.02789, 2018.
- [32] Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. Orca: Progressive learning from complex explanation traces of gpt-4. arXiv preprint arXiv:2306.02707, 2023.
- [33] Open-Orca. Openorca, 2023. Last accessed on 2024-01-15. https://huggingface.co/datasets/ Open-Orca/OpenOrca.
- [34] OpenAssistant. oasst2, 2023. Last accessed on 2024-01-15. https://huggingface.co/datasets/ OpenAssistant/oasst2.
- [35] Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb dataset for falcon llm: outperforming curated corpora with web data, and web data only. arXiv preprint arXiv:2306.01116, 2023.
- [36] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.
- [37] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
- [38] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290, 2023.
- [39] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.
- [40] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [41] Stability AI Language Team. Introducing stable lm 2 1.6b. Last accessed on 2024-01-22. https: //stability.ai/news/introducing-stable-lm-2.
- [42] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [43] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [44] Junjie Ye, Xuanting Chen, Nuo Xu, Can Zu, Zekai Shao, Shichun Liu, Yuhan Cui, Zeyang Zhou, Chao Gong, Yang Shen, et al. A comprehensive capability analysis of gpt-3 and gpt-3.5 series models. arXiv preprint arXiv:2303.10420, 2023.

- [45] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023.
- [46] Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, Chang Zhou, and Jingren Zhou. Scaling relationship on learning mathematical reasoning with large language models, 2023.
- [47] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.
- [48] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019.
- [49] Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. Tinyllama: An open-source small language model. arXiv preprint arXiv:2401.02385, 2024.
- [50] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.
- [51] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv:2306.05685, 2023.

