# arXiv:2412.19638v1[cs.AI]27Dec2024

## Xmodel-2 Technical Report

Wang Qun Liu Yang Lin Qingquan Qu Zhijiu Jiang Ling Xiaoduo AI Lab {wangqun,liuyangfoam,quzhijiu}@xiaoduotech.com

### Abstract

Xmodel-2 is a 1.2-billion-parameter large language model designed specifically for reasoning tasks. Its architecture enables different model scales to share a unified set of hyperparameters, allowing for extensive experimentation on smaller models and seamless transfer of optimal configurations to larger models. To maximize training efficiency and stability, Xmodel-2 employs the WSD learning rate scheduler from MiniCPM. Pretrained on 1.5 trillion tokens from diverse sources, Xmodel-2 achieves state-of-the-art performance in complex reasoning and agent-based tasks, while maintaining low training costs. These results highlight the potential of efficient model design and training strategies in advancing reasoning capabilities. Model checkpoints and code are publicly available on GitHub at https://github.com/XiaoduoAILab/Xmodel-2.

Xmodel-2 Qwen2.5

40

Qwen2

MiniCPM

35

ComplexReasoningAvgScore

InternLM2.5

30

Phi-1.5

StableLM-2-zephyr

25

Fox-1

SmolLM

20

Llama-3.2

DCLM

15

MobiLlama

TinyLLaMA1.1 OLMo

10

OpenELM

1.0 1.2 1.4 1.6 1.8 2.0 Model Size (B)

Figure 1: Average Scores on Complex Reasoning Benchmarks (GSM8K, MATH, BBH, MMLU, HumanEval and MBPP).

### 1 Introduction

Large language models (LLMs) have made significant strides in natural language understanding, demonstrating impressive performance across a range of tasks. However, they still face challenges when tackling complex reasoning tasks. Effective reasoning is crucial for applications such as automated customer service and scientific discovery [AI4Science and Quantum, 2023]. While larger models typically show improved reasoning capabilities, they also require more computational resources, longer training times, and higher energy consumption.

Xmodel-2 is a 1.2B-parameter model designed to balance reasoning power and training efficiency. It excels in complex reasoning, code generation, and agent-based interactions. Unlike other models, Xmodel-2 incorporates an innovative architecture based on Tensor Programs [Yang et al., 2022] [Yang

- et al., 2023] , enabling models of different scales to share the same set of hyperparameters. This approach allows for extensive hyperparameter search on smaller models, with the best configurations transferred seamlessly to larger models, enhancing both efficiency and performance.

To accelerate training and ensure stable convergence, Xmodel-2 uses the Warmup-Stable-Decay (WSD) learning rate scheduler from MiniCPM [Hu et al., 2024]. Pretrained on 1.5 trillion tokens, Xmodel-2 is capable of processing diverse inputs, such as text and code, which strengthens its performance in complex reasoning tasks.

Our contributions are as follows:

- 1. Our Xmodel-2 is open-source, aimed at improving accessibility for researchers in language model research. We believe its state-of-the-art performance and compact size make it an ideal platform for both researchers and practitioners.
- 2. We advanced the decay phase by applying the WSD learning rate scheduler and exploring data ratio search, addressing a gap in the literature and achieving significant improvements in reasoning performance.
- 3. We conducted a focused evaluation of Xmodel-2 ’s agent capabilities, demonstrating its strong potential for real-world applications such as customer service and task automation.

### 2 Pretraining

This chapter provides a detailed overview of the pretraining process for Xmodel-2. We begin with a description of the model architecture, followed by an explanation of the data distribution across the stable training and decay stages, and conclude with an outline of the overall training procedure.

- 2.1 Model Architecture We adopt an architecture similar to LLama 2 [Touvron et al., 2023], with the following configuration:

Hidden size Intermediate size Attention heads KV heads Layers Context Len 1536 3840 24 8 48 4096

Table 1: Model configuration for Xmodel-2.

Tokenizer: Unlike most large models that use the BPE tokenizer, Xmodel-2 employs a custom Unigram tokenizer [Kudo, 2018] with a vocabulary size of 65,280 tokens.

Embedding Sharing: In small language models (SLMs), the embedding layer constitutes a significant portion of the total parameters. To improve efficiency, we implement embedding sharing, which reduces the parameter count by 0.1B.

Deep-and-Thin Architecture: The importance of a deep and thin architecture for SLMs is emphasized by [Liu et al., 2024], a concept that aligns with our observations.

Grouped-Query Attention: To optimize training and inference efficiency, we adopt Grouped-Query Attention (GQA) [Ainslie et al., 2023], which utilizes 24 attention heads and 8 key-value (KV) heads.

#### 2.2 Training Stages

The training of the Xmodel-2 base model consists of two key stages: the Stable Training Stage and the Decay Stage.

Stable Training Stage: In this phase, we train on approximately 1.5 trillion tokens (see Figure 2 for data distribution), primarily sourced from open datasets. The training follows the optimal configuration identified through model tuning experiments, using the WSD LRS [Hu et al., 2024], with a batch size of 3.93 million tokens and a maximum learning rate of 0.01.

Decay Stage: This stage combines the pretraining data with high-quality supervised fine-tuning (SFT) data. We apply exponential annealing to the WSD learning rate scheduler, following the formula f(s − T) = 0.5(s−S)/T, where T is set to 5000 steps (20 billion tokens), allowing the learning rate to gradually decrease during the final training phase.

#### 2.3 Data Ratio Optimization in the Decay Stage

Previous work [Ye et al., 2024] demonstrated that data ratio experiments on small models can effectively transfer to larger models. However, their focus was mainly on cosine learning rate decay and pretraining data ratio optimization. MiniCPM [Hu et al., 2024] emphasized the benefits of incorporating SFT data during the decay stage, but lacked detailed exploration of data ratios under WSD learning rate schedulers.

In contrast, our work explores the interaction between SFT data and domain-specific pretraining data during the WSD decay phase. We framed the data ratio search around two key questions: the overall proportion of SFT data and the distribution of categories within SFT. This approach significantly reduces the search space, enabling efficient optimization with fewer trials.

Through over 400 trials, we identified that the optimal SFT data ratio falls between 60% and 69%, with the precise value depending on the internal composition of the SFT-mixed dataset. We also observed that Chain-of-Thought datasets may enhance logical reasoning Suzgun et al. [2022b], while instruction-formatted datasets in mathematics and code outperform pretraining-format data in complex reasoning tasks.

#### 2.4 Training Data Distribution

- Figure 2 shows the data distribution across training stages, including CC_Chn (Chinese corpus), FineWeb-Edu [Penedo et al., 2024], Dolma [Soldaini et al., 2024] (English corpora), and Code Pretrain datasets like StarCoder [Li et al., 2023a] and The Stack [Kocetkov et al., 2022]. The decay stage incorporates diverse data, such as EvolInstruct [Xu et al., 2023], OssInstruct [Wei et al., 2024], and UltraChat [Ding et al., 2023].

The SFT-Mixed Dataset is composed of five distinct categories: Mathematics, Code, Logic, Knowledge, and Commonsense. Chain-of-Thought (CoT) data is categorized under Logic. To improve generalization, SFT prompts were diversified via rule-based transformations, though multilingual alignment and domain-specific data were excluded for future exploration. The SFT data underwent multiple rounds of SimHash deduplication with a bucket size of 1 million, improving performance by

- 1.7% compared to non-deduplicated data.

Data ratio experiments revealed the effectiveness of instruction-formatted SFT data during the annealing phase, leading us to allocate 64% to SFT data. These adjustments, combined with optimized data mixing and processing, improved complex reasoning performance by 29.31% compared to our baseline.

- 2.5 Training Loss

- Figure 3 presents the training loss curve on the FineWeb-Edu dataset [Penedo et al., 2024]. The initial drop corresponds to increasing the batch size from 2M to 4M, which likely replicates the stabilizing effect of a reduced learning rate [Smith et al., 2018]. The second drop reflects the impact of the learning rate decay phase.

CC_Chn

| |
|---|

SFT_logic Dolma

SFT_knowledge CC_Chn multilang_pretrain Code Pretrain

SFT_code

Code Pretrain

| |
|---|

20.0

6.4 4.5

Dolma

4.5

| |
|---|

12.1

3.6

open-web-math-train algebraic-stack-train

3.4 2.1

11.0

26.0

Book MultiLangWiki

1.6 1.6 1.1

| |
|---|

wiki

17.9

| |
|---|

book

22.4 18.5

FineWeb-Edu

41.0

SFT_commensense

SFT_math

FineWeb-Edu

Figure 2: Data mixture of different training stages.The left side represents the stable training phase, and the right side represents the decay phase.

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

3.0

2.8

2.6

Loss

2.4

2.2

2.0

0 200 400 600 800 1000 1200 1400 Tokens (B)

Figure 3: Loss curve for Xmodel-2-1.2B.

### 3 Results

We compared Xmodel-2 with recent decoder-only models containing 1–2 billion parameters, as identified in [Lu et al., 2024]. The baselines include TinyLLaMA [Zhang et al., 2024], InternLM2 [Cai et al., 2024], Qwen2 [Yang et al., 2024], MiniCPM [Hu et al., 2024], Llama 3 [Grattafiori

- et al., 2024], Phi-1.5 [Li et al., 2023b], StableLM-2 [Bellagente et al., 2024], OLMo [Groeneveld et al., 2024], MobiLlama [Thawakar et al., 2024], and SmolLM [Allal et al., 2024].Our experiments demonstrate that Xmodel-2 achieves state-of-the-art (SOTA) performance among 1B-parameter models, demonstrating the effectiveness of our training strategies and optimized data ratios, especially in commonsense reasoning, complex reasoning and agent-based tasks.

#### 3.1 Commonsense Reasoning

We evaluate Xmodel-2 on various commonsense reasoning benchmarks using the Language Model Evaluation Harness [Gao et al., 2021], which includes: ARC-Challenge [Clark et al., 2018], ARCEasy [Clark et al., 2018], BoolQ [Clark et al., 2019], HellaSwag [Zellers et al., 2019], OpenBookQA [Mihaylov et al., 2018], PiQA [Bisk et al., 2019], SciQ [Welbl et al., 2017], TriviaQA [Joshi et al.,

2017], and Winogrande [Sakaguchi et al., 2021]. For fairness and reproducibility, all models were evaluated in the same environment using raw accuracy metrics.

Table 2 presents the zero-shot evaluation results, models in green perform worse than Xmodel-2, while models in red outperform Xmodel-2. In this evaluation, Xmodel-2 outperforms the majority of baseline models and demonstrates competitive performance.

#### Model ARC-c ARC-e Boolq HS. OB. PiQA SciQ Wino. Avg

MobiLlama-1B 28.24 61.53 60.92 46.74 21.80 75.14 88.20 59.27 55.23 TinyLLaMA1.1-1.1B 30.97 61.66 55.99 46.70 25.20 72.63 89.30 59.43 55.24 OLMo-1B 28.67 63.34 61.74 46.97 25.00 75.03 87.00 59.98 55.97 OpenELM-1.1B 28.84 62.37 63.58 48.36 25.40 74.76 90.60 61.72 56.95 Llama-3.2-1B 31.31 65.36 63.73 47.78 26.40 74.48 91.50 61.01 57.70 MiniCPM-1.2B 36.86 70.29 67.92 49.91 23.60 74.43 91.80 60.77 59.45 Fox-1-1.6B 34.73 69.91 71.77 46.33 24.60 75.24 93.20 60.77 59.57 InternLM2.5-1.8B 35.24 66.37 79.82 46.99 22.00 73.29 94.90 62.67 60.16 Qwen2-1.5B 33.11 66.41 72.60 48.57 27.00 75.57 94.60 65.75 60.45 StableLM-2-zephyr-1.6B 36.52 66.79 80.00 53.26 26.80 74.86 88.00 64.09 61.29 SmolLM-1.7B 43.43 76.47 65.93 49.58 30.00 75.79 93.20 60.93 61.92 Qwen2.5-1.5B 41.21 75.21 72.97 50.15 31.80 75.90 94.30 63.61 63.14 DCLM-1B 41.30 74.79 71.41 53.59 32.20 76.93 94.00 66.22 63.81 Phi-1.5-1.3B 44.80 76.22 74.95 47.96 38.60 76.66 93.30 72.93 65.68

Xmodel-2-1.2B 39.16 71.55 74.65 47.45 29.20 74.81 93.60 63.93 61.79 Table 2: Zero-shot performance on Commonsense Reasoning tasks.

#### 3.2 Complex Reasoning

To evaluate the complex reasoning abilities of Xmodel-2, we conducted tests using several wellestablished benchmarks, including GSM8K [Cobbe et al., 2021], MATH [Hendrycks et al., 2021b], BBH [Suzgun et al., 2022a], MMLU [Hendrycks et al., 2021a], HumanEval [Chen et al., 2021], and MBPP [Austin et al., 2021]. The first four benchmarks were assessed using the Language Model Evaluation Harness [Gao et al., 2021], while the last two were evaluated with the Code Generation LM Evaluation Harness [Ben Allal et al., 2022]. The results are presented in Table 3.

Model GSM8K MATH BBH MMLU HumanEval MBPP Avg 5-shot 4-shot 3-shot 0-shot pass@1 pass@1

OpenELM-1.1B 0.45 1.06 6.62 25.52 8.54 6.80 8.16 OLMo-1B 2.35 1.46 25.60 24.46 5.49 0.20 9.93 TinyLLaMA1.1-1.1B 2.50 1.48 25.57 25.35 1.83 3.40 10.02 MobiLlama-1B 1.97 1.54 25.76 25.26 7.93 5.40 11.31 DCLM-1B 4.93 2.14 30.70 46.43 8.54 6.80 16.59 Llama-3.2-1B 6.60 1.78 31.44 36.63 14.63 22.20 18.88 SmolLM-1.7B 7.51 3.18 29.21 27.73 21.34 31.80 20.13 Fox-1-1.6B 34.34 7.94 28.75 39.55 14.02 9.00 22.27 StableLM-2-zephyr-1.6B 41.32 10.12 32.71 41.30 25.61 19.40 28.41 Phi-1.5-1.3B 32.15 3.18 28.81 41.75 36.59 35.40 29.65 InternLM2.5-1.8B 27.90 16.68 41.76 46.30 27.40 29.60 31.61 MiniCPM-1.2B 40.11 10.98 35.42 43.99 43.90 36.80 35.20 Qwen2-1.5B 57.62 22.90 33.05 55.11 20.73 30.40 36.64 Qwen2.5-1.5B 62.40 28.28 43.99 59.72 5.49 40.00 39.98

Xmodel-2-1.2B 55.88 25.50 48.40 48.87 29.88 29.20 39.62 Table 3: Performance on Complex Reasoning tasks.

- 3.3 Agent Capabilities

We evaluate Xmodel-2’s performance on four agent tasks using the ReAct prompting technique [Yao et al., 2023b]. These tasks include HotpotQA [Yang et al., 2018], FEVER [Thorne et al., 2018], AlfWorld [Shridhar et al., 2021], and WebShop [Yao et al., 2023a]. We use EM(Exact Match) as the evaluation metric in FEVER and HotpotQA, and success rate in AlfWorld and WebShop.

To accomplish FEVER[Thorne et al., 2018] and HotpotQA[Yang et al., 2018], the agent retrieves information from Wikipedia. In FEVER, the agent verifies the truth of a claim via multiple-choice questions, while in HotpotQA[Yang et al., 2018], the agent reasons across multiple documents to answer complex, open-ended questions. For AlfWorld[Shridhar et al., 2021], the agent interacts with an environment of 25 containers, performing actions like retrieving or manipulating objects. This task requires spatial reasoning and decision-making. Finally, in WebShop[Yao et al., 2023a], the agent navigates a simulated e-commerce environment to search, customize, and purchase items. This tests the agent’s ability to search efficiently and make decisions within real-world e-commerce constraints. These tasks pose significant challenges for small language models (SLMs) due to their requirements for complex reasoning, multi-step decision-making, and real-world interaction. The results are summarized in Table 4.

Model HotpotQA FEVER AlfWorld WebShop Avg EM EM success rate success rate

OLMo-1B 2.67 18.58 0.00 0.00 4.32 Phi-1.5 1.3B 3.54 17.56 2.24 0.80 6.04 DCLM-1B 4.92 24.39 0.75 0.00 7.52 MobiLlama-1B 0.00 30.43 0.00 0.00 7.61 TinyLLaMA1.1-1.1B 2.11 28.77 0.00 0.20 7.77 OpenELM-1-1B 2.70 28.37 0.00 0.40 7.87 StableLM-2-zephyr-1.6B 1.44 20.81 8.96 2.20 8.35 SmolLM-1.7B 2.28 31.31 0.00 0.60 8.55 Fox-1-1.6B 5.37 30.88 0.00 0.60 9.21 Llama-3.2-1B 4.87 27.67 8.21 3.20 10.99 Qwen2.5-1.5B 13.53 27.58 5.97 0.60 11.92 MiniCPM-1.2B 11.00 36.57 1.60 1.00 12.52 InternLM2.5-1.8B 12.84 34.02 2.99 1.00 12.71

Xmodel-2-1.2B 13.70 40.00 0.78 2.20 14.21 Table 4: Performance on Agent tasks.

- 4 Case Study

- 4.1 Calibration

The pre-trained Xmodel-2-1.2B model exhibits strong calibration, with predicted confidence aligning closely to actual correctness probabilities. Figure 4 illustrates the calibration plot, where the x-axis represents confidence bins (log-probabilities) for A/B/C/D choices, and the y-axis shows accuracy within each bin. The dotted diagonal indicates perfect calibration.

#### 4.2 Post-training Scaling Law

We explored the Post-training Scaling Law of Xmodel-2 on the Wikitext-2 dataset, focusing on how test-time loss changes as the number of prompt tokens increases. This analysis reveals that as the context token count grows, the model’s prediction accuracy for the next token improves, with loss and token index following a power-law relationship. Figure 5 shows a consistent decrease in perplexity, with diminishing returns captured by the fitted curve:

##### L(t) = b + (t/c)a ; a ∼ −0.575,b ∼ 1.772,c ∼ 32.840

Calibration curve(model=pre-train)

1.0

Accuracy: 0.46 ECE: 0.060

0.8

0.6

P(correct)

0.4

0.2

0.0

0.0 0.2 0.4 0.6 0.8 1.0 P(answer)

Figure 4: Calibration plot for the pre-trained Xmodel-2-1.2B model on the MMLU dataset.

This suggests that, akin to OpenAI’s approach of using test-time to enhance model performance, increasing context length leads to more accurate token predictions, as described by the Post-training Scaling Law.

### 5 Conclusions

This paper introduced Xmodel-2, a 1.2-billion-parameter model optimized for reasoning tasks. By leveraging the maximal update parametrization (µP), Warmup-Stable-Decay (WSD) learning rate scheduler, and data ratio optimization during the decay phase, Xmodel-2 showed significant improvements in complex reasoning capabilities. Most notably, Xmodel-2 achieved state-of-the-art performance in agent-based evaluations within the 1-2B parameter range, highlighting its strong potential for real-world applications such as e-commerce customer service and task automation.

### References

thesis with large language models, 2021. URL https://arxiv.org/abs/2108.07732.

M. R. AI4Science and M. A. Quantum. The impact of large language models on scientific discovery: a preliminary study using gpt-4, 2023. URL https://arxiv.org/abs/2311.07361.

- M. Bellagente, J. Tow, D. Mahan, D. Phung, M. Zhuravinskyi, R. Adithyan, J. Baicoianu, B. Brooks,
- N. Cooper, A. Datta, M. Lee, E. Mostaque, M. Pieler, N. Pinnaparju, P. Rocha, H. Saini, H. Teufel, N. Zanichelli, and C. Riquelme. Stable lm 2 1.6b technical report, 2024.

J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón, and S. Sanghai. Gqa: Training generalized multi-query transformer models from multihead checkpoints, 2023.

L. Ben Allal, N. Muennighoff, L. Kumar Umapathi, B. Lipkin, and L. von Werra. A framework for the evaluation of code generation models. https://github.com/bigcode-project/ bigcode-evaluation-harness, 2022.

L. B. Allal, A. Lozhkov, E. Bakouch, L. von Werra, and T. Wolf. Smollm - blazingly fast and remarkably powerful, 2024.

J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le, and C. Sutton. Program syn-

Y. Bisk, R. Zellers, R. L. Bras, J. Gao, and Y. Choi. Piqa: Reasoning about physical commonsense in

Post-training Scaling Law

Xmodel-2

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

Xmodel-2 fit: a=-0.575, b=1.772, c=32.840

loss

0 100 200 300 400 500 test time token index

Figure 5: Post-training Scaling Law for Xmodel-2-1.2B on the Wikitext-2 dataset.

natural language. In AAAI Conference on Artificial Intelligence, 2019. URL https://api. semanticscholar.org/CorpusID:208290939.

Z. Cai, M. Cao, H. Chen, K. Chen, K. Chen, X. Chen,

- X. Chen, Z. Chen, Z. Chen, P. Chu, X. Dong, H. Duan, Q. Fan, Z. Fei, Y. Gao, J. Ge, C. Gu, Y. Gu, T. Gui, A. Guo, Q. Guo, C. He, Y. Hu, T. Huang, T. Jiang, P. Jiao, Z. Jin, Z. Lei, J. Li, J. Li, L. Li, S. Li, W. Li, Y. Li, H. Liu, J. Liu, J. Hong, K. Liu, K. Liu, X. Liu, C. Lv, H. Lv, K. Lv, L. Ma, R. Ma, Z. Ma, W. Ning, L. Ouyang, J. Qiu, Y. Qu, F. Shang,
- Y. Shao, D. Song, Z. Song, Z. Sui, P. Sun, Y. Sun,

- H. Tang, B. Wang, G. Wang, J. Wang, J. Wang,

- R. Wang, Y. Wang, Z. Wang, X. Wei, Q. Weng,

- F. Wu, Y. Xiong, C. Xu, R. Xu, H. Yan, Y. Yan,

- X. Yang, H. Ye, H. Ying, J. Yu, J. Yu, Y. Zang, C. Zhang, L. Zhang, P. Zhang, P. Zhang, R. Zhang,

S. Zhang, S. Zhang, W. Zhang, W. Zhang, X. Zhang, Z. Zhou, J. Zhuo, Y. Zou, X. Qiu, Y. Qiao, and

- X. Zhang, H. Zhao, Q. Zhao, X. Zhao, F. Zhou,

- D. Lin. Internlm2 technical report, 2024.

M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra,

- E. Morikawa, A. Radford, M. Knight, M. Brundage,

M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba. Evaluating large language models trained on code, 2021. URL https://arxiv.org/ abs/2107.03374.

C. Clark, K. Lee, M.-W. Chang, T. Kwiatkowski,

- M. Collins, and K. Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions, 2019.

P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge, 2018.

- K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

N. Ding, Y. Chen, B. Xu, Y. Qin, Z. Zheng, S. Hu, Z. Liu, M. Sun, and B. Zhou. Enhancing chat language models by scaling high-quality instructional conversations, 2023. URL https://arxiv.org/ abs/2305.14233.

- L. Gao, J. Tow, S. Biderman, S. Black, A. DiPofi, C. Foster, L. Golding, J. Hsu, K. McDonell, N. Muennighoff, J. Phang, L. Reynolds, E. Tang, A. Thite, B. Wang, K. Wang, and A. Zou. A framework for few-shot language model evaluation, Sept. 2021. URL https://doi.org/10.5281/ zenodo.5371628.

A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, A. Yang, A. Fan, A. Goyal, A. Hartshorn, A. Yang, A. Mitra, A. Sravankumar, A. Korenev, A. Hinsvark, A. Rao, A. Zhang,

- A. Rodriguez, A. Gregerson, A. Spataru, B. Roziere,
- B. Biron, B. Tang, B. Chern, C. Caucheteux,
- C. Nayak, C. Bi, C. Marra, C. McConnell, C. Keller, C. Touret, C. Wu, C. Wong, C. C. Ferrer, C. Nikolaidis, D. Allonsius, D. Song, D. Pintz, D. Livshits,
- D. Wyatt, D. Esiobu, D. Choudhary, D. Mahajan, D. Garcia-Olano, D. Perino, D. Hupkes,
- E. Lakomkin, E. AlBadawy, E. Lobanova, E. Dinan, E. M. Smith, F. Radenovic, F. Guzmán, F. Zhang,

- G. Synnaeve, G. Lee, G. L. Anderson, G. Thattai, G. Nail, G. Mialon, G. Pang, G. Cucurell,
- H. Nguyen, H. Korevaar, H. Xu, H. Touvron,
- I. Zarov, I. A. Ibarra, I. Kloumann, I. Misra, I. Evtimov, J. Zhang, J. Copet, J. Lee, J. Geffert, J. Vranes,
- J. Park, J. Mahadeokar, J. Shah, J. van der Linde, J. Billock, J. Hong, J. Lee, J. Fu, J. Chi, J. Huang, J. Liu, J. Wang, J. Yu, J. Bitton, J. Spisak, J. Park,

- J. Rocca, J. Johnstun, J. Saxe, J. Jia, K. V. Alwala, K. Prasad, K. Upasani, K. Plawiak, K. Li,
- K. Heafield, K. Stone, K. El-Arini, K. Iyer, K. Malik, K. Chiu, K. Bhalla, K. Lakhotia, L. RantalaYeary, L. van der Maaten, L. Chen, L. Tan, L. Jenkins, L. Martin, L. Madaan, L. Malo, L. Blecher,
- L. Landzaat, L. de Oliveira, M. Muzzi, M. Pasupuleti, M. Singh, M. Paluri, M. Kardas, M. Tsimpoukelli, M. Oldham, M. Rita, M. Pavlova, M. Kambadur, M. Lewis, M. Si, M. K. Singh, M. Hassan, N. Goyal, N. Torabi, N. Bashlykov, N. Bogoychev, N. Chatterji, N. Zhang, O. Duchenne, O. Çelebi, P. Alrassy, P. Zhang, P. Li, P. Vasic, P. Weng, P. Bhargava, P. Dubal, P. Krishnan, P. S. Koura, P. Xu, Q. He, Q. Dong, R. Srinivasan, R. Ganapathy,

- R. Calderer, R. S. Cabral, R. Stojnic, R. Raileanu,

- R. Maheswari, R. Girdhar, R. Patel, R. Sauvestre,

- R. Polidoro, R. Sumbaly, R. Taylor, R. Silva,

- R. Hou, R. Wang, S. Hosseini, S. Chennabasappa,
- S. Singh, S. Bell, S. S. Kim, S. Edunov, S. Nie,

- S. Narang, S. Raparthy, S. Shen, S. Wan, S. Bhosale,

- S. Zhang, S. Vandenhende, S. Batra, S. Whitman,

- S. Sootla, S. Collot, S. Gururangan, S. Borodinsky, T. Herman, T. Fowler, T. Sheasha, T. Georgiou, T. Scialom, T. Speckbacher, T. Mihaylov,
- T. Xiao, U. Karn, V. Goswami, V. Gupta, V. Ramanathan, V. Kerkez, V. Gonguet, V. Do, V. Vogeti,

- V. Albiero, V. Petrovic, W. Chu, W. Xiong, W. Fu,
- W. Meers, X. Martinet, X. Wang, X. Wang, X. E. Tan, X. Xia, X. Xie, X. Jia, X. Wang, Y. Goldschlag, Y. Gaur, Y. Babaei, Y. Wen, Y. Song, Y. Zhang, Y. Li, Y. Mao, Z. D. Coudert, Z. Yan, Z. Chen, Z. Papakipos, A. Singh, A. Srivastava, A. Jain, A. Kelsey, A. Shajnfeld, A. Gangidi, A. Victoria, A. Goldstand, A. Menon, A. Sharma, A. Boesenberg, A. Baevski, A. Feinstein, A. Kallet, A. Sangani, A. Teo, A. Yunus, A. Lupu, A. Alvarado, A. Caples, A. Gu, A. Ho, A. Poulton, A. Ryan, A. Ramchandani, A. Dong, A. Franco, A. Goyal, A. Saraf,

- A. Chowdhury, A. Gabriel, A. Bharambe, A. Eisenman, A. Yazdan, B. James, B. Maurer, B. Leonhardi, B. Huang, B. Loyd, B. D. Paola, B. Paranjape, B. Liu, B. Wu, B. Ni, B. Hancock, B. Wasti,
- B. Spence, B. Stojkovic, B. Gamido, B. Montalvo,
- C. Parker, C. Burton, C. Mejia, C. Liu, C. Wang, C. Kim, C. Zhou, C. Hu, C.-H. Chu, C. Cai, C. Tindal, C. Feichtenhofer, C. Gao, D. Civin, D. Beaty,
- D. Kreymer, D. Li, D. Adkins, D. Xu, D. Testuggine, D. David, D. Parikh, D. Liskovich, D. Foss, D. Wang, D. Le, D. Holland, E. Dowling, E. Jamil,

E. Montgomery, E. Presani, E. Hahn, E. Wood, E.-T. Le, E. Brinkman, E. Arcaute, E. Dunbar, E. Smothers, F. Sun, F. Kreuk, F. Tian, F. Kokkinos, F. Ozgenel, F. Caggioni, F. Kanayet, F. Seide, G. M. Florez, G. Schwarz, G. Badeer, G. Swee, G. Halpern,

- G. Herman, G. Sizov, Guangyi, Zhang, G. Lakshminarayanan, H. Inan, H. Shojanazeri, H. Zou,
- H. Wang, H. Zha, H. Habeeb, H. Rudolph, H. Suk,

- H. Aspegren, H. Goldman, H. Zhan, I. Damlaj,
- I. Molybog, I. Tufanov, I. Leontiadis, I.-E. Veliche,

I. Gat, J. Weissman, J. Geboski, J. Kohli, J. Lam, J. Asher, J.-B. Gaya, J. Marcus, J. Tang, J. Chan,

- J. Zhen, J. Reizenstein, J. Teboul, J. Zhong, J. Jin,

- J. Yang, J. Cummings, J. Carvill, J. Shepard, J. McPhie, J. Torres, J. Ginsburg, J. Wang, K. Wu, K. H. U, K. Saxena, K. Khandelwal, K. Zand, K. Matosich,
- K. Veeraraghavan, K. Michelena, K. Li, K. Jagadeesh, K. Huang, K. Chawla, K. Huang, L. Chen,
- L. Garg, L. A, L. Silva, L. Bell, L. Zhang, L. Guo, L. Yu, L. Moshkovich, L. Wehrstedt, M. Khabsa,
- M. Avalani, M. Bhatt, M. Mankus, M. Hasson, M. Lennie, M. Reso, M. Groshev, M. Naumov, M. Lathi, M. Keneally, M. Liu, M. L. Seltzer,

- M. Valko, M. Restrepo, M. Patel, M. Vyatskov,

- M. Samvelyan, M. Clark, M. Macey, M. Wang, M. J. Hermoso, M. Metanat, M. Rastegari, M. Bansal,
- N. Santhanam, N. Parks, N. White, N. Bawa,

- N. Singhal, N. Egebo, N. Usunier, N. Mehta, N. P. Laptev, N. Dong, N. Cheng, O. Chernoguz, O. Hart,
- O. Salpekar, O. Kalinli, P. Kent, P. Parekh, P. Saab,
- P. Balaji, P. Rittner, P. Bontrager, P. Roux, P. Dollar, P. Zvyagina, P. Ratanchandani, P. Yuvraj, Q. Liang,

- R. Alao, R. Rodriguez, R. Ayub, R. Murthy,

- R. Nayani, R. Mitra, R. Parthasarathy, R. Li,

- R. Hogan, R. Battey, R. Wang, R. Howes, R. Rinott,
- S. Mehta, S. Siby, S. J. Bondu, S. Datta, S. Chugh,

- S. Hunt, S. Dhillon, S. Sidorov, S. Pan, S. Mahajan, S. Verma, S. Yamamoto, S. Ramaswamy,

- S. Lindsay, S. Lindsay, S. Feng, S. Lin, S. C. Zha, S. Patil, S. Shankar, S. Zhang, S. Zhang, S. Wang, S. Agarwal, S. Sajuyigbe, S. Chintala, S. Max, S. Chen, S. Kehoe, S. Satterfield, S. Govindaprasad,

- S. Gupta, S. Deng, S. Cho, S. Virk, S. Subramanian,

- S. Choudhury, S. Goldman, T. Remez, T. Glaser,
- T. Best, T. Koehler, T. Robinson, T. Li, T. Zhang,

- T. Matthews, T. Chou, T. Shaked, V. Vontimitta, V. Ajayi, V. Montanez, V. Mohan, V. S. Kumar, V. Mangla, V. Ionescu, V. Poenaru, V. T. Mihailescu,

- V. Ivanov, W. Li, W. Wang, W. Jiang, W. Bouaziz,
- W. Constable, X. Tang, X. Wu, X. Wang, X. Wu,
- X. Gao, Y. Kleinman, Y. Chen, Y. Hu, Y. Jia, Y. Qi,
- Y. Li, Y. Zhang, Y. Zhang, Y. Adi, Y. Nam, Yu, Wang, Y. Zhao, Y. Hao, Y. Qian, Y. Li, Y. He, Z. Rait,
- Z. DeVito, Z. Rosnbrick, Z. Wen, Z. Yang, Z. Zhao, and Z. Ma. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

D. Groeneveld, I. Beltagy, P. Walsh, A. Bhagia, R. Kinney, O. Tafjord, A. H. Jha, H. Ivison, I. Magnusson, Y. Wang, S. Arora, D. Atkinson, R. Authur, K. R. Chandu, A. Cohan, J. Dumas, Y. Elazar, Y. Gu, J. Hessel, T. Khot, W. Merrill, J. Morrison, N. Muennighoff, A. Naik, C. Nam, M. E. Peters, V. Pyatkin, A. Ravichander, D. Schwenk, S. Shah, W. Smith, E. Strubell, N. Subramani,

- M. Wortsman, P. Dasigi, N. Lambert, K. Richardson, L. Zettlemoyer, J. Dodge, K. Lo, L. Soldaini,
- N. A. Smith, and H. Hajishirzi. Olmo: Accelerating the science of language models, 2024. URL https://arxiv.org/abs/2402.00838.

D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt. Measuring

massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021a.

D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. Measuring mathematical problem solving with the math dataset, 2021b. URL https://arxiv.org/abs/ 2103.03874.

- S. Hu, Y. Tu, X. Han, C. He, G. Cui, X. Long, Z. Zheng,

- Y. Fang, Y. Huang, W. Zhao, X. Zhang, Z. L. Thai, K. Zhang, C. Wang, Y. Yao, C. Zhao, J. Zhou, J. Cai,
- Z. Zhai, N. Ding, C. Jia, G. Zeng, D. Li, Z. Liu, and M. Sun. Minicpm: Unveiling the potential of small language models with scalable training strategies, 2024. URL https://arxiv.org/abs/2404. 06395.

M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension, 2017.

D. Kocetkov, R. Li, L. B. Allal, J. Li, C. Mou, C. M. Ferrandis, Y. Jernite, M. Mitchell, S. Hughes, T. Wolf, D. Bahdanau, L. von Werra, and H. de Vries. The stack: 3 tb of permissively licensed source code, 2022. URL https://arxiv.org/abs/ 2211.15533.

- T. Kudo. Subword regularization: Improving neural network translation models with multiple subword candidates, 2018. URL https://arxiv. org/abs/1804.10959.

- R. Li, L. B. Allal, Y. Zi, N. Muennighoff, D. Kocetkov, C. Mou, M. Marone, C. Akiki, J. Li, J. Chim, Q. Liu, E. Zheltonozhskii, T. Y. Zhuo, T. Wang, O. Dehaene, M. Davaadorj, J. Lamy-Poirier, J. Monteiro, O. Shliazhko, N. Gontier, N. Meade, A. Zebaze, M.-H. Yee, L. K. Umapathi, J. Zhu, B. Lipkin, M. Oblokulov, Z. Wang, R. Murthy, J. Stillerman,
- S. S. Patel, D. Abulkhanov, M. Zocca, M. Dey, Z. Zhang, N. Fahmy, U. Bhattacharyya, W. Yu,

- S. Singh, S. Luccioni, P. Villegas, M. Kunakov,

- F. Zhdanov, M. Romero, T. Lee, N. Timor, J. Ding, C. Schlesinger, H. Schoelkopf, J. Ebert, T. Dao, M. Mishra, A. Gu, J. Robinson, C. J. Anderson,

- B. Dolan-Gavitt, D. Contractor, S. Reddy, D. Fried, D. Bahdanau, Y. Jernite, C. M. Ferrandis, S. Hughes,

- T. Wolf, A. Guha, L. von Werra, and H. de Vries. Starcoder: may the source be with you!, 2023a.

- Y. Li, S. Bubeck, R. Eldan, A. D. Giorno, S. Gunasekar, and Y. T. Lee. Textbooks are all you need ii: phi-1.5 technical report, 2023b.
- Z. Liu, C. Zhao, F. Iandola, C. Lai, Y. Tian, I. Fedorov,

- Y. Xiong, E. Chang, Y. Shi, R. Krishnamoorthi, L. Lai, and V. Chandra. Mobilellm: Optimizing subbillion parameter language models for on-device use cases, 2024. URL https://arxiv.org/abs/ 2402.14905.
- Z. Lu, X. Li, D. Cai, R. Yi, F. Liu, X. Zhang, N. D. Lane, and M. Xu. Small language models: Survey, measurements, and insights, 2024. URL https: //arxiv.org/abs/2409.15790.

- T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering, 2018.

- G. Penedo, H. Kydlíˇcek, L. B. allal, A. Lozhkov, M. Mitchell, C. Raffel, L. V. Werra, and T. Wolf. The fineweb datasets: Decanting the web for the finest text data at scale, 2024. URL https://arxiv. org/abs/2406.17557.

- K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9): 99–106, 2021.

M. Shridhar, X. Yuan, M.-A. Côté, Y. Bisk, A. Trischler, and M. Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning, 2021. URL https://arxiv.org/abs/ 2010.03768.

S. L. Smith, P.-J. Kindermans, C. Ying, and Q. V. Le. Don’t decay the learning rate, increase the batch size, 2018. URL https://arxiv.org/abs/1711. 00489.

- L. Soldaini, R. Kinney, A. Bhagia, D. Schwenk,

- D. Atkinson, R. Authur, B. Bogin, K. Chandu, J. Dumas, Y. Elazar, V. Hofmann, A. H. Jha, S. Kumar, L. Lucy, X. Lyu, N. Lambert, I. Magnusson, J. Morrison, N. Muennighoff, A. Naik, C. Nam, M. E. Peters, A. Ravichander, K. Richardson, Z. Shen,
- E. Strubell, N. Subramani, O. Tafjord, P. Walsh, L. Zettlemoyer, N. A. Smith, H. Hajishirzi, I. Beltagy, D. Groeneveld, J. Dodge, and K. Lo. Dolma: an open corpus of three trillion tokens for language model pretraining research, 2024. URL https://arxiv.org/abs/2402.00159.

- M. Suzgun, N. Scales, N. Scharli, S. Gehrmann, Y. Tay, H. W. Chung, A. Chowdhery, Q. V. Le, E. H. hsin Chi, D. Zhou, and J. Wei. Challenging big-bench tasks and whether chain-of-thought can solve them. In Annual Meeting of the Association for Computational Linguistics, 2022a. URL https://api. semanticscholar.org/CorpusID:252917648.

M. Suzgun, N. Scales, N. Schärli, S. Gehrmann, Y. Tay, H. W. Chung, A. Chowdhery, Q. V. Le, E. H. Chi, D. Zhou, and J. Wei. Challenging big-bench tasks and whether chain-of-thought can solve them, 2022b.

O. Thawakar, A. Vayani, S. Khan, H. Cholakal, R. M. Anwer, M. Felsberg, T. Baldwin, E. P. Xing, and F. S. Khan. Mobillama: Towards accurate and lightweight fully transparent gpt, 2024.

J. Thorne, A. Vlachos, C. Christodoulopoulos, and A. Mittal. Fever: a large-scale dataset for fact extraction and verification, 2018. URL https: //arxiv.org/abs/1803.05355.

- H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. C. Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal,

- A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M.-A. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten,

- R. Silva, E. M. Smith, R. Subramanian, X. E. Tan,

B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur,

- S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.

- Y. Wei, Z. Wang, J. Liu, Y. Ding, and L. Zhang. Magicoder: Empowering code generation with ossinstruct, 2024. URL https://arxiv.org/abs/ 2312.02120.

- J. Welbl, N. F. Liu, and M. Gardner. Crowdsourcing multiple choice science questions, 2017.

- C. Xu, Q. Sun, K. Zheng, X. Geng, P. Zhao, J. Feng, C. Tao, and D. Jiang. Wizardlm: Empowering large language models to follow complex instructions, 2023. URL https://arxiv.org/abs/ 2304.12244.

A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C. Zhou, C. Li, C. Li, D. Liu, F. Huang, G. Dong, H. Wei, H. Lin, J. Tang, J. Wang, J. Yang, J. Tu, J. Zhang,

- J. Ma, J. Yang, J. Xu, J. Zhou, J. Bai, J. He, J. Lin,
- K. Dang, K. Lu, K. Chen, K. Yang, M. Li, M. Xue, N. Ni, P. Zhang, P. Wang, R. Peng, R. Men, R. Gao, R. Lin, S. Wang, S. Bai, S. Tan, T. Zhu, T. Li, T. Liu, W. Ge, X. Deng, X. Zhou, X. Ren, X. Zhang,

- X. Wei, X. Ren, X. Liu, Y. Fan, Y. Yao, Y. Zhang,
- Y. Wan, Y. Chu, Y. Liu, Z. Cui, Z. Zhang, Z. Guo, and Z. Fan. Qwen2 technical report, 2024. URL https://arxiv.org/abs/2407.10671.

- G. Yang, E. J. Hu, I. Babuschkin, S. Sidor, X. Liu, D. Farhi, N. Ryder, J. Pachocki, W. Chen, and J. Gao. Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer, 2022. URL https://arxiv.org/abs/2203.03466.

G. Yang, D. Yu, C. Zhu, and S. Hayou. Tensor programs vi: Feature learning in infinite-depth neural networks, 2023. URL https://arxiv.org/abs/ 2310.02244.

Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. W. Cohen,

- R. Salakhutdinov, and C. D. Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering, 2018. URL https://arxiv.org/ abs/1809.09600.
- S. Yao, H. Chen, J. Yang, and K. Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents, 2023a. URL https://arxiv.org/abs/2207.01206.

S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao. React: Synergizing reasoning and acting in language models, 2023b. URL https://arxiv.org/abs/2210.03629.

J. Ye, P. Liu, T. Sun, Y. Zhou, J. Zhan, and X. Qiu. Data mixing laws: Optimizing data mixtures by predicting language modeling performance, 2024. URL https://arxiv.org/abs/2403.16952.

R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi. Hellaswag: Can a machine really finish your sentence?, 2019.

P. Zhang, G. Zeng, T. Wang, and W. Lu. Tinyllama: An open-source small language model, 2024.

### 6 Appendix : Model Wind Tunnel Experiments

Before pre-training, we conducted wind tunnel experiments on two small models: nano (6M) and tiny (54M) to validate our training strategy and data ratio. Key experiments included a µP hyperparameter search and data ratio optimization, which confirmed the strategy’s suitability for Xmodel-2.

#### 6.1 µP Hyperparameter Search

We observed that µP hyperparameters remained stable across model scales. Using Bayesian optimization, we optimized four key hyperparameters: scale_emb, dim_model_base, scale_depth, and learning_rate on the nano model with the C4 dataset. The search explored 300 configurations, compared to 570,000 in a grid search. Results showed:

- • Optimal Hyperparameters: learning_rate between 0.01 and 0.02, and dim_model_base below 256.
- • Loss Patterns: Loss below 4.1 concentrated around specific hyperparameters, indicating stable performance (Figure 6).

#### Hyperparameter Range Options Step Size

scale_emb [2, 20] - 1 dim_model_base - {32, 64, 128, 256, 512, 1024} scale_depth [1, 5] - 0.1 learning_rate [0.001, 0.1] - 0.001

Table 5: Hyperparameter search ranges for nano model.

#### Name Specific Operation

| | |
|---|---|
|Embedding Output Scaling<br><br>|Multiply the output of the embedding by scale_emb|
|Residual Connection Scaling<br><br>|Scale the output tensor of a block before adding to each residual connection in each layer by scale_depth/√num_layers<br><br>|
|Initialization of Tensors|Set the initialization standard deviation of each twodimensional tensor parameter to init_std/ dm/dbase, and set other parameters’ initialization to 0.1<br><br>|
|Learning Rate Scaling of Tensors|Adjust the learning rate of each two-dimensional tensor parameter to 1/(dm/dbase) times the learning rate of other parts (or the overall learning rate)<br><br>|
|LM Head Scaling|Adjust the output logits to 1/(dm/dbase) times the original value|

Table 6: List of operations used when applying tensor program techniques.

Learning Rate vs Loss

Scale Embedding vs Loss

4.05

4.05

4.00

4.00

3.95

3.95

3.90

3.90

Loss

Loss

3.85

3.85

|[Figure 1]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

3.80

3.80

4.00

3.75

3.75

3.95

3.90

0.00 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 Learning Rate

2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 Scale Embedding

Loss

Dim Model Base vs Loss

Scale Depth vs Loss

3.85

4.05

4.05

3.80

4.00

4.00

3.75

3.95

3.95

3.90

3.90

Loss

Loss

3.85

3.85

3.80

3.80

3.75

3.75

0 200 400 600 800 1000 Dim Model Base

1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 Scale Depth

Figure 6: Grid search over the µP parameterization spaces.

