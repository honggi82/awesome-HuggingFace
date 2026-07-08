## Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning

Dawid J. Kopiczko1 Sagar Vaze2 Tijmen Blankevoort3 Yuki M. Asano1

# arXiv:2602.11149v1[cs.CL]11Feb2026

### Abstract

Supervised fine-tuning (SFT) on chain-of-thought data is an essential post-training step for reasoning language models. Standard machine learning intuition suggests that training with more unique training samples yields better generalization. Counterintuitively, we show that SFT benefits from repetition: under a fixed update budget, training for more epochs on smaller datasets outperforms single-epoch training on larger datasets. On AIME’24/25 and GPQA benchmarks, Olmo37B trained for 128 epochs on 400 samples outperforms the equivalent 1 epoch on 51200 samples by 12–26 percentage points, with no additional catastrophic forgetting. We find that training token accuracy reliably signals when repetition has saturated; improvements from additional epochs plateau at full memorization, a pattern consistent across all settings. These findings provide a practical approach for reasoning SFT, where scaling epochs with token accuracy as a stopping criterion can replace expensive undirected data scaling. We pose the repetition advantage, where full memorization coincides with improved generalization, as a new open problem for the community in understanding the training dynamics of large language models. Code is available at: https://github.com/ dkopi/data-repetition.

### 1. Introduction

Modern language model training proceeds through distinct stages: pretraining on internet-scale data to acquire world knowledge, mid-training on curated corpora to extend capabilities, and post-training to shape model behavior (Guo et al., 2025; Team OLMo, 2025; Yang et al., 2025). For reasoning-focused models, post-training typically begins with supervised fine-tuning (SFT) on long Chain-ofThought (CoT) demonstrations, often distilled from

1University of Technology Nuremberg 2Mistral AI 3NVIDIA. Correspondence to: <dj.kopiczko@gmail.com>. Preprint. February 12, 2026.

|Pretraining| |
|---|---|
| | |

|Supervised Fine-Tuning (SFT) Stage<br><br>|x 2-5|
|---|
<br><br>Common approach<br><br>maximize data size few epochs<br><br>|x 20-50<br><br>|
|---|
<br><br>Our approach<br><br>small random subset many epochs<br><br>~8x less compute|
|---|

|SFT| |
|---|---|
| | |

RL

Figure 1. Illustration of our approach to supervised fine-tuning in a modern LLM training pipeline. Instead of maximizing dataset size and training for few epochs, we train for many epochs on a small random subset of SFT data, substantially reducing compute while improving downstream reasoning performance.

more capable models, where reasoning traces can span thousands of tokens before reaching a final answer. This SFT step, analogous to behavioral cloning in reinforcement learning (Osa et al., 2018), primes the model for subsequent stages such as reinforcement learning from human feedback (Ouyang et al., 2022) or reinforcement learning with verifiable rewards (Guo et al., 2025; Shao et al., 2024). Unlike pretraining data, which can be scraped at scale from the web, high-quality long-CoT demonstrations require either expensive human annotation or careful distillation from larger models, including generation, filtering, and validation of long reasoning traces. As a result, the question of how to best utilize limited SFT data is practically important.

The common assumption in machine learning would suggest that training with more unique training samples yields better generalization. Under i.i.d. sampling, each new example provides independent information about the data distribution, and generalization bounds in statistical learning theory typically improve with dataset size. This principle manifests practically throughout the field – data augmentation techniques are widely used to artificially expand effective dataset size when real data is limited (Hern´andez-Garc´ıa & K¨onig, 2018; Shorten & Khoshgoftaar, 2019), and the success of large language models has been attributed in significant part to training on ever-larger unique corpora. Following this logic, modern post-training pipelines employ millions of SFT samples (Team OLMo, 2025).

In this paper, we show that this might not only be suboptimal, but that, actually, a reverse pattern can be observed for the SFT stage for a pretrained LLM, see Figure 1. Under a

###### Accuracy

###### Pass@n

Termination Rate

34

64

85

1248163264128256

1248163264128256

1248163264128256

36 38

69 72

82 85

39 37 39

72 70 69

85 80 86

- 38 39 42 41 38
- 39 34 37 39

67 68 69 74

82 75 86 89

Epochs

72 71 74 73 68

74 74 84 85 88

31 33 36 39 39 36

62 64 63 68 70 64

49 53 59 70 71 65

17 22 26 27 29 31 28

49 49 53 56 56 58 54

28 31 33 37 42 43 42

17 19 18 19 21 22 23 23

40 44 43 44 47 48 46 48

22 24 22 24 28 29 32 32

13 14 14 11 13 15 16 16 17

33 36 33 36 37 41 43 38 40

18 17 17 15 16 19 21 21 24

200

400

800

200

400

800

200

400

800

1600

3200

6400

1600

3200

6400

1600

3200

6400

12800

25600

51200

12800

25600

51200

12800

25600

51200

Samples

Samples

Samples

- Figure 2. Scaling epochs versus scaling data for Olmo3-7B trained on long-CoT SFT data, averaged across AIME’24, AIME’25, and GPQA benchmarks. Each diagonal represents a fixed update budget, where epochs × samples is constant. Within any diagonal, moving toward fewer samples and more epochs consistently improves accuracy and pass@n, with gains diminishing around 32–64 epochs. Termination rate correlates strongly with accuracy and may be a primary driver of performance gains, as models that fail to terminate cannot produce a final answer.

fixed update budget, training for more epochs on smaller datasets outperforms training on larger datasets. The gains are not marginal. The performance and termination rates, i.e., the model’s ability to successfully conclude reasoning with a final answer, both scale with epoch count and saturate together, suggesting that sufficient repetition of the same data is required for models to fully internalize the demonstrated reasoning structure.

We find that this convergence is tightly linked to training set memorization. Performance improvements plateau once models achieve near-perfect next-token prediction accuracy on the training data, even as validation loss continues to rise. This relationship holds across all models we test, all benchmarks, and different training datasets, making train token accuracy a practical stopping criterion for scaling epochs. Despite this apparent overfitting, we observe no additional catastrophic forgetting compared to single-epoch training on large datasets. Our main contributions are:

- • Phenomenon. We demonstrate that under a fixed update budget, scaling epochs on smaller datasets substantially outperforms scaling unique samples.
- • Dynamics. We identify training token accuracy as a reliable stopping criterion for epoch scaling, with performance gains plateauing once models reach full memorization, and we show that multi-epoch training on small datasets causes no additional catastrophic forgetting compared to large datasets.
- • Factors. We show how training data properties, such

as teacher model size in distillation and the correctness of data samples, affect the repetition advantage.

While we provide a practical heuristic for exploiting the repetition advantage, we pose explaining this phenomenon in long-CoT SFT as a novel, open problem for the community.

### 2. Scaling Epochs on a Fixed Update Budget

To investigate whether data repetition can substitute for data scaling in supervised fine-tuning, we conduct controlled experiments varying the number of epochs and unique samples while holding total gradient updates, and all other parameters, constant. We train base checkpoints of two recent language models on chain-of-thought data and evaluate on challenging reasoning benchmarks.

#### 2.1. Preliminaries

Supervised fine-tuning adapts a pretrained language model to target behaviors by training on demonstration data. Given input-output pairs (x,y) where y = (y1,...,yT) is a target sequence, SFT minimizes the cross-entropy loss over nexttoken predictions:

T

log pθ(yt | x,y<t) (1)

L(θ) = −

t=1

In practice, the loss is typically masked to exclude input tokens, applying only to the response.

Throughout this work, we use update budget B to denote the

###### AIME25 Accuracy@16 AIME24 Accuracy@16 GPQA Accuracy@4 AIME25 Pass@16 AIME24 Pass@16 GPQA Pass@4

256

1.0

|[Figure 1]| |
|---|---|
| | |
| | |
| | |
| | |

128

64

Olmo37B

32

Epochs

0.8

16

8

NormalizedScore

4

0.6

- 1

- 2

256

128

0.4

64

Qwen38B

32

Epochs

16

0.2

8

4

- 1

- 2

0.0

200

400

800

200

400

800

200

400

800

200

400

800

200

400

800

200

400

800

1600

3200

6400

1600

3200

6400

1600

3200

6400

1600

3200

6400

1600

3200

6400

1600

3200

6400

12800

25600

51200

12800

25600

51200

12800

25600

51200

12800

25600

51200

12800

25600

51200

12800

25600

51200

Samples

Samples

Samples

Samples

Samples

Samples

- Figure 3. The repetition advantage is consistent across models, benchmarks, and evaluation metrics. Heatmaps show normalized scores for Olmo3-7B (top) and Qwen3-8B (bottom) on AIME’24, AIME’25, and GPQA, evaluated with both Accuracy@n and Pass@n. Each diagonal corresponds to a fixed update budget (epochs × samples), and in all settings, performance improves when moving along a diagonal toward fewer samples and more epochs.

total number of gradient updates during training, which for batch size one is equal to the number of epochs multiplied by the number of unique samples. Comparing configurations at equal update budgets isolates the effect of data repetition from differences in total optimization steps.

#### 2.2. Experimental Setup

Models. We use the Qwen3-4B, Qwen3-8B (Yang et al., 2025), and Olmo3-7B (Team OLMo, 2025) base models. These are pretrained checkpoints prior to any instruction tuning, providing a clean starting point for studying SFT dynamics. For training and evaluation, we use the default chat template for each model.

Dataset. We use the Dolci SFT 7B1 dataset from the Olmo3 post-training pipeline, which contains distilled longCoT demonstrations spanning math, coding, precise instruction following, and general conversation (Team OLMo, 2025). We apply several filters: we keep only the first conversation turn, retain samples containing complete reasoning traces (verified by presence of <think> and </think> tags), and remove samples exceeding 10k tokens when tokenized with the Olmo tokenizer. From the filtered data, we randomly sample nested training splits of increasing size: 200, 400, 800, 1.6k, 3.2k, 6.4k, 12.8k, 25.6k, 51.2k samples, constructed so that each smaller split is a subset of the next larger one. We hold out 1000 random samples as a validation set for analysis.

1https://huggingface.co/datasets/allenai/ Dolci-Think-SFT-7B

Evaluation. We evaluate on three challenging reasoning benchmarks: AIME 2024, AIME 2025, and GPQA. AIME (AIME, 2025) is a mathematical reasoning benchmark consisting of 30 competition problems per year, requiring multistep reasoning across algebra, geometry, number theory, and combinatorics; each answer is an integer from 0 to 999. GPQA (Rein et al., 2024) is a graduate-level multiplechoice benchmark with expert-written questions in biology, physics, and chemistry, where the model must reason through the problem before selecting from four options. For each problem in these benchmarks, we append an instruction requesting the final answer in \boxed{} format for straightforward extraction.

We report three metrics: Acc@n, the accuracy averaged over n independent generations per problem; Pass@n, the fraction of problems solved in at least one of n attempts; and Termination, the fraction of generations that conclude with an end-of-sequence token rather than being truncated. We sample up to 30k tokens per generation to accommodate extended reasoning traces. For AIME we generate 16 responses per problem, while for GPQA, 4 responses due to its larger test set. We use recommended sampling parameters from each model’s technical report and vLLM (Kwon et al., 2023) for efficient inference.

Training. We load models in bfloat16, use Unsloth optimized kernels (Daniel Han & team, 2023), and the 8-bit Adam optimizer (Dettmers et al., 2022) with a cosine learning rate schedule. Warmup is set to 10% of the total update budget for each run. We use a batch size of one, following recent findings that small batch sizes achieve equal or better per-token performance (Marek et al., 2025). We mask

- Table 1. Performance at a fixed update budget of B = 51, 200 gradient updates, showing configurations up to 16 epochs. All rows within each model use equivalent update budget but vary the epochs-to-samples ratio. For all three models, 16 epochs on 3,200 samples substantially outperforms 1 epoch on 51,200 samples across all benchmarks.

Model Epochs Samples GPQA AIME’24 AIME’25 Avg@4 Pass@4 Avg@16 Pass@16 Avg@16 Pass@16

- 1 51.2k 11.5 23.7 17.7 46.7 22.3 50.0
- 2 25.6k 14.8 29.3 28.1 66.7 24.8 46.7 4 12.8k 20.2 38.9 33.3 73.3 29.2 50.0 8 6.4k 29.7 51.5 44.4 73.3 35.4 66.7

Olmo3-7B

###### 16 3.2k 34.0 62.1 42.3 80.0 39.2 63.3

- 1 51.2k 21.6 38.9 12.3 36.7 11.2 30.0
- 2 25.6k 25.4 42.4 14.0 46.7 17.3 43.3 4 12.8k 35.6 56.6 20.6 66.7 20.0 40.0 8 6.4k 41.9 61.6 30.6 70.0 27.7 56.7

Qwen3-8B

###### 16 3.2k 51.0 72.7 30.6 76.7 31.2 63.3

- 1 51.2k 13.1 29.8 6.5 26.7 5.6 30.0
- 2 25.6k 21.1 38.9 13.5 36.7 14.0 36.7 4 12.8k 29.7 52.5 18.1 36.7 17.7 40.0 8 6.4k 40.5 64.1 23.3 43.3 23.5 43.3

Qwen3-4B

16 3.2k 39.3 68.7 19.2 43.3 18.8 40.0

the input prompt and compute cross-entropy loss only on response tokens. We conduct a learning rate sweep for each model using 1 epoch on 51,200 samples and select the bestperforming rate based on benchmark accuracy, then use that learning rate for all subsequent runs. Each configuration is run on a single H100 94GB GPU for up to 24 hours.

Experimental grid. We train models across dataset sizes from 200 to 51,200 samples and epoch counts from 1 to 256, subject to a maximum update budget of 51,200. For example, the 200-sample split is trained for up to 256 epochs, while the 51,200-sample split is trained for only 1 epoch. Each configuration is trained independently from the base checkpoint with its own warmup and learning rate schedule, rather than evaluating intermediate checkpoints from a single extended run. This design ensures that for any given update budget, we can compare multiple configurations trading off epochs against unique samples.

#### 2.3. Results

Figure 2 presents heatmaps of accuracy, pass rate, and termination rate across all combinations of epochs and dataset sizes for Olmo3-7B, averaged over benchmarks. Figure 3 presents normalized scores for each benchmark separately, for Olmo3-7B and Qwen3-8B models. Table 1 provides detailed per-benchmark results for all models at a fixed update budget of 51,200 gradient updates, showing training runs up to 16 epochs. Figures with all configurations can be found in Appendix B.

We can see a clear and consistent pattern:

Fewer unique samples repeated more times yields substantially better performance than training on more data for fewer epochs.

For example, at a budget of 51,200 updates, Olmo3-7B trained for 32 epochs on 1,600 samples reaches an average 39% accuracy across benchmarks, compared to 17% for a single epoch on 51,200 samples. The same pattern appears across benchmarks and models: on Figure 3 all top performances are clearly in the top part of the samples×epochs pyramid. The gains diminish around 32–64 epochs, suggesting a saturation point beyond which additional repetition provides limited benefit. We investigate this saturation in Sec. 4.1

### 3. Impact of Training Data

The previous experiments establish the repetition advantage on a general-purpose SFT dataset spanning diverse domains, but whether this phenomenon depends on properties of the training data remains unclear. In this section, we vary data characteristics while keeping the model fixed to Olmo3-7B.

We construct math-focused datasets by distilling long chainof-thought solutions from various Qwen3 models. We use problems from the NuminaMath-TIR2 dataset (Li et al., 2024) as prompts and generate solutions using reasoning checkpoints of Qwen3-0.6B and Qwen3-8B as teacher models. We split each distilled dataset into nested subsets from 200 to 25,600 samples and train across the same epoch-

2https://huggingface.co/datasets/AI-MO/ NuminaMath-TIR

- Table 2. Impact of teacher model size on the repetition advantage. Olmo3-7B is trained on math data distilled from Qwen3-0.6B and Qwen3-8B teachers, with results averaged across AIME’24, AIME’25, and GPQA. The repetition advantage persists for both teachers. With the weaker 0.6B teacher, increasing the update budget from 6.4k to 25.6k leads to lower peak performance, echoing the degradation observed in weak-to-strong generalization.

Teacher Budget B Ep Samples Avg@n Pass@n

Qwen3-0.6B

- 6.4k 1 6.4k 2.2 13.7

- 6.4k 2 3.2k 4.8 19.4 6.4k 4 1.6k 6.9 22.3 6.4k 8 800 11.4 34.1 6.4k 16 400 20.6 46.5 6.4k 32 200 21.6 54.0

- 25.6k 1 25.6k 3.8 16.7

- 25.6k 2 12.8k 5.4 19.1 25.6k 4 6.4k 6.5 25.9 25.6k 8 3.2k 10.4 29.8 25.6k 16 1.6k 20.8 49.2 25.6k 32 800 20.2 49.5

Qwen3-8B

- 6.4k 1 6.4k 10.6 36.1

- 6.4k 2 3.2k 17.3 39.7 6.4k 4 1.6k 22.1 51.0 6.4k 8 800 25.4 53.6 6.4k 16 400 26.1 55.0 6.4k 32 200 24.5 51.0

- 25.6k 1 25.6k 13.3 36.5

- 25.6k 2 12.8k 18.5 40.0 25.6k 4 6.4k 24.1 49.9 25.6k 8 3.2k 33.3 64.9 25.6k 16 1.6k 35.5 66.6 25.6k 32 800 30.1 63.0

sample grid as before.

- 3.1. Teacher Model Quality.

Table 2 compares results when training on data distilled from the 0.6B and 8B teachers, for budgets of B = 25,600 and B = 6,400. The repetition advantage persists in both settings, with epoch scaling improving performance more reliably than data scaling regardless of teacher size.

The interaction between epochs and data differs between teachers, however. With the smaller 0.6B teacher, the average performance degrades with additional samples; the highest average pass rate for B = 6,400 is 54.0%, while for B = 25,600 it’s 49.5%. This pattern echoes findings in weak-to-strong generalization (Burns et al., 2024), where student models trained on weaker teacher data can initially exceed teacher performance but degrade with prolonged exposure.

With the larger 8B teacher, the pattern is similar to the previous experiments on the Dolci SFT 7B dataset. The model reaches higher absolute performance after sufficient number of epochs, and the performance improves when

Table 3. Training on incorrect reasoning traces does not harm performance. Olmo3-7B is trained on positive and negative trajectories distilled from Qwen3-8B, with a fixed update budget of B = 6.4k. The repetition advantage holds regardless of trajectory correctness. Surprisingly, training on negatives often matches or exceeds training on positives, with higher peak performance on GPQA and AIME’24. A and P denote Accuracy@n and Pass@n respectively, while Ep stands for the number of epochs.

Ep Subset GPQA AIME’24 AIME’25 A@4 P@4 A@16 P@16 A@16 P@16

- 1 Neg. 4.5 13.1 15.0 56.7 13.3 36.7 Pos. 3.0 9.1 13.3 33.3 15.0 40.0

- 2 Neg. 6.6 17.7 22.1 66.7 19.8 46.7 Pos. 6.2 15.2 20.2 60.0 19.4 46.7

4 Neg. 11.9 26.8 30.2 73.3 27.5 50.0 Pos. 10.4 25.3 27.3 66.7 26.0 46.7

8 Neg. 18.7 42.4 35.0 73.3 30.8 66.7 Pos. 19.1 43.4 36.9 80.0 29.2 53.3

16 Neg. 29.3 54.0 40.0 80.0 33.3 66.7 Pos. 23.4 51.5 37.3 80.0 34.2 63.3

32 Neg. 28.5 55.6 35.0 70.0 31.2 70.0 Pos. 16.4 41.4 38.8 76.7 27.3 53.3

scaling up the number of data samples. In this case, the highest average pass rate for B = 6,400 is 55.0%, while for B = 25,600 it’s 66.6%. These results suggest that:

Teacher quality determines whether data scaling remains beneficial, while the repetition advantage itself is robust to teacher choice.

#### 3.2. Negative Trajectories

If the repetition advantage depends on learning from correct reasoning, training on incorrect traces should degrade performance or exhibit different scaling dynamics. We define negative trajectories as chain-of-thought samples where the model’s final answer is incorrect.

To test this, we take the data distilled from the Qwen3-8B teacher and partition it by correctness of the final answer. Samples with correct answers form the positive set; those with incorrect answers form the negative set. We construct nested splits from 200 to 6,400 samples for each and train Olmo3-7B across the same epoch-sample grid.

From Table 3 we find that:

Training on negative trajectories does not degrade performance.

The epoch scaling advantage persists with the same pattern as before. Moreover, the top performance on AIME’24 and GPQA is on-par and even slightly higher when training on

###### GPQA

###### AIME'24

###### AIME'25

40

40

40

Epochs

30

35

Accuracy(%)

- 1

- 2

30

30

4 8

20

25

16 32 Base

20

10

20

70 80 90 100

70 80 90 100

70 80 90 100

Train Set Token Acc. (%)

Train Set Token Acc. (%)

Train Set Token Acc. (%)

- Figure 4. Relationship between training set memorization and downstream performance for Olmo3-7B. Points are colored by epoch count; within each epoch group, variation reflects different dataset sizes. Token accuracy on train set increases primarily with epochs rather than total updates. Across all benchmarks, performance gains plateau once models approach full memorization, suggesting that token accuracy can serve as a stopping criterion for epoch scaling. The initial token accuracy of the base model is marked with the vertical line.

negatives than positives, reaching 40.0% versus 38.8% on AIME’24 and 29.3% versus 23.4% on GPQA. One possible explanation is that negative trajectories come from harder problems where the teacher failed, and exposure to difficult reasoning attempts benefits the student even when the final answer is wrong.

### 4. Probing the Repetition Advantage

Having demonstrated that the repetition advantage is robust across models, benchmarks, and training data sources, we now attempt to understand what drives this phenomenon. We return to the Olmo3-7B models trained on the Dolci dataset from Section 2 and examine several training dynamics, including memorization, termination behavior, and classical overfitting metrics, searching for signals that might explain why epoch scaling outperforms data scaling. While we identify correlates of improved performance, we do not find a definitive causal mechanism. We present these observations as empirical characterizations that may guide future investigation into the underlying causes.

#### 4.1. Memorization signals convergence.

We first investigate training set memorization as a potential indicator of convergence. During SFT, we measure token accuracy on a fixed 200-sample training subset, computing the fraction of response tokens where the model’s top nexttoken prediction matches the target. Figure 4 plots this metric against downstream accuracy for Olmo3-7B. Token accuracy increases primarily with epoch count rather than total gradient updates: models trained for 16 epochs achieve near-perfect memorization regardless of whether they see 200 or 3,200 unique samples. Across all three benchmarks, performance improvements plateau once models approach full memorization. Table 4 shows this pattern across all three

models, revealing that the smaller model memorizes faster and peaks at lower epoch counts, possibly due to higher optimal learning rate than larger models. This relationship suggests a practical stopping criterion for epoch scaling:

Saturation of token accuracy on training data marks convergence.

#### 4.2. Termination correlates with performance.

A notable pattern in Figure 2 is the strong correlation between termination rate and accuracy. Single-epoch models terminate only 24% of generations, while 32-epoch models approach the rate of 89%. This correlation likely reflects a causal relationship, where models that fail to terminate cannot produce a final answer, directly limiting their measured accuracy. The increase in termination rate with epoch count suggests that:

Repeated exposure helps models internalize not just the reasoning patterns but also the structural convention of concluding long reasoning chains.

This behavioral convergence appears to require sufficient repetition, as even models trained on 51,200 unique samples fail to reliably terminate when trained for only one epoch.

#### 4.3. Overfitting paradox.

A natural concern with multi-epoch training is overfitting. Figure 5 examines this for Olmo3-7B. As epochs increase, train loss approaches zero while validation loss rises substantially. We also measure prediction entropy on the validation set, defined as the average token-level entropy H = − i pi log pi of the model’s output distribution. This metric decreases with epoch count, indicating the model

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

40

40

40

AverageAccuracy(%)

35

35

35

Epochs

- 1

- 2

30

30

30

4 8

25

25

25

16 32 Base

20

20

20

15

15

15

10

10

10

0.0 0.2 0.4 0.6 0.8 1.0 1.2

1.00 1.25 1.50 1.75 2.00 2.25

0.4 0.6 0.8 1.0

Train Loss

Validation Loss

Val. Prediction Entropy

- Figure 5. Training dynamics for Olmo3-7B showing the relationship between loss, entropy, and downstream performance averaged over AIME’24, AIME’25, and GPQA. Points are colored by epoch count; within each group, variation reflects dataset size. As epochs increase, train loss approaches zero while validation loss rises, the classical signature of overfitting in terms of the train-validation gap. Prediction entropy also decreases, showing increased model confidence in predictions that diverge from the validation distribution. Despite these indicators, downstream accuracy improves with epoch count. Vertical lines mark base model metrics.

grows more confident in predictions that diverge from the validation distribution. By standard metrics, the model is overfitting. Yet, downstream accuracy improves monotonically with epoch count, suggesting that:

Validation loss on held-out SFT data is not a reliable metric for reasoning performance.

One interpretation is that multi-epoch training elicits latent capabilities already present in the pretrained model, rather than teaching genuinely new skills. The model becomes confident in its own reasoning patterns, which differ from the validation trajectories but nonetheless transfer to heldout benchmarks. This view aligns with recent work on entropy minimization in fine-tuning (Agarwal et al., 2025) and suggests that SFT may function more as capability elicitation than capability acquisition.

4.4. Catastrophic Forgetting

Beyond overfitting, multi-epoch training on small datasets risks catastrophic forgetting, where the model may lose general capabilities while specializing to the narrow training distribution. To evaluate this, we measure performance on MMLU (Hendrycks et al., 2021), a broad knowledge benchmark spanning 57 subjects. Unlike our reasoning benchmarks, MMLU is evaluated by comparing the model’s probability assignments to answer choices rather than generating full responses. We use 5-shot prompting following the standard protocol.

- Figure 6 compares two training strategies matched by total gradient updates: scaling epochs on a fixed 200-sample dataset versus scaling dataset size with a single epoch. Both approaches cause some forgetting relative to the base model,

However:

Epoch scaling leads to less catastrophic forgetting than data scaling.

Combined with the large improvement in reasoning accuracy, epoch scaling offers a strictly better tradeoff.

### 5. Related Work

Data repetition and scaling laws in pretraining. Scaling laws for language model pretraining characterize how validation loss improves predictably with increased model size, total training tokens, and compute (Kaplan et al., 2020; Hoffmann et al., 2022). While these laws are agnostic to whether tokens are unique or repeated, they have commonly been interpreted as motivating the heuristic that, when available, additional fresh data is preferable to revisiting the same corpus.

More directly, recent work studies pretraining in dataconstrained regimes where training necessarily becomes multi-epoch. Muennighoff et al. (2023) propose dataconstrained scaling laws that explicitly model the decreasing marginal value of repeated tokens, and empirically find that repeating a fixed corpus for a small number of epochs (on the order of a few passes) can be nearly as effective for loss as training on equivalently-sized fresh tokens, while the returns from further repetition decay sharply. Relatedly, recent work on diffusion language models shows that, in data-constrained pretraining regimes, extensive data repetition can be beneficial, with diffusion objectives extracting substantially more value per unique token than autoregressive training (Ni et al., 2025).

- as expected when fine-tuning on domain-specific data.

Our work contrasts with this pretraining-focused literature

Table 4. Relationship between training set memorization and average downstream performance at a fixed update budget of B = 51,200. Token accuracy measures the fraction of response tokens where the model’s top prediction matches the training target. Performance improves with epoch count until models reach full memorization, after which gains plateau or degrade.

Model Epochs Acc@n Pass@n Token Acc.

- 1 17.2 40.1 75.7
- 2 22.6 47.5 78.8 4 27.6 54.1 84.1 8 36.5 63.8 92.2

Olmo3-7B

16 38.5 68.5 98.0 32 38.8 73.7 100.0 64 38.9 69.0 100.0

128 38.4 71.5 100.0

1 15.0 35.2 76.6 2 18.9 44.1 79.4 4 25.4 54.4 84.2 8 33.4 62.8 91.6

Qwen3-8B

16 37.6 70.9 97.7 32 36.2 61.6 100.0 64 34.4 62.1 100.0

128 30.4 61.4 100.0

1 8.4 28.8 76.6 2 16.2 37.4 81.5 4 21.8 43.1 88.5 8 29.1 50.3 96.1

Qwen3-4B

16 25.7 50.7 99.9 32 22.9 49.7 100.0 64 18.7 40.3 100.0

128 12.4 37.6 100.0

Single-Epoch Multi-Epoch (200 samples)

- 62

- 63

- 64

- 65

MMLUAccuracy

40

ReasoningAccuracy

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

30

20

200 1,000 10,000 50,000

Update Steps

Figure 6. Catastrophic forgetting under epoch scaling versus data scaling for Olmo3-7B. Multi-epoch training on 200 samples is compared against single-epoch training on increasingly large datasets, matched by total update steps. Both approaches exhibit forgetting as measured by MMLU accuracy, with epoch scaling causing less degradation. Combined with the large improvement in reasoning accuracy, measured on AIME’24/’25 and GPQA benchmarks, epoch scaling offers a strictly better tradeoff.

by showing that the “avoid repetition” heuristic does not transfer to supervised fine-tuning on long chain-of-thought data; on the contrary, repetition substantially improves convergence and downstream performance.

Multi-epoch SFT in post-training practice. Although single-pass training is often treated as the default in instruction tuning, many recent training pipelines perform supervised fine-tuning for multiple epochs as part of post-training, often without isolating epoch count as a studied variable. Examples include: 1) Olmo 3 reports training on SFT data, consisting of over 2M samples, for two epochs (Team OLMo, 2025). 2) DeepSeek-R1 similarly includes an SFT phase that fine-tunes its base model for 2-3 epochs on a large curated set prior to reinforcement learning (Guo et al., 2025). 3) Llama-3 trains SFT for “multiple epochs” (at Meta AI, 2024). 4) LIMO trains for 15 epochs on a curated reasoning set (Ye et al., 2025), while 5) Muennighoff et al. (2025) train an instruct model on long-CoT data for 5 epochs. Across these releases, epoch counts are typically presented as recipe details rather than as ablated design choices. Our work provides a controlled, compute-matched comparison of epoch scaling versus unique-data scaling in long-CoT SFT, showing that multi-epoch training can be a strictly better strategy

even when additional training tokens are available.

Memorization, overfitting, and training dynamics. Classic results in deep learning challenge the view that memorization necessarily harms generalization. Arpit et al. (2017) show that deep networks tend to learn simple patterns before memorizing noise. For language modeling specifically, Tirumala et al. (2022) study exact memorization throughout training and characterize how memorization depends on model size, dataset size, and optimization choices. Complementing these empirical findings, Feldman (2019) provides a theoretical perspective arguing that memorization can be necessary for generalization on long-tailed data distributions. We connect to this literature by showing that, in long-CoT supervised fine-tuning, downstream gains from repetition saturate when the model reaches near-perfect token-level accuracy on the training demonstrations.

### 6. Conclusion

We show that supervised fine-tuning on long chain-ofthought data can defy standard machine learning intuition.

Under a fixed update budget, training for more epochs on smaller datasets substantially outperforms training on larger datasets, and this repetition advantage holds across models, benchmarks, and training data sources studied in this work. Despite its robustness, the mechanism underlying the repetition advantage remains poorly understood. While training token accuracy provides a practical stopping signal for epoch scaling, the optimal dataset size is data- and model-dependent, and principled criteria for selecting it a priori remain elusive.

We argue that explaining why memorization under repetition improves generalization in reasoning SFT is an important open problem. More broadly, our results suggest that both epoch count and dataset size should be treated as first-class decision variables in reasoning SFT, rather than defaulting to single-epoch training on the largest available dataset.

### References

Agarwal, S., Zhang, Z., Yuan, L., Han, J., and Peng, H. The unreasonable effectiveness of entropy minimization in LLM reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum? id=UfFTBEsLgI.

AIME. Aime problems and solutions, 2025. URL https://artofproblemsolving.com/wiki/ index.php/AIME_Problems_and_Solutions.

Arpit, D. et al. A closer look at memorization in deep networks, 2017. URL https://arxiv.org/abs/ 1706.05394.

- at Meta AI, L. T. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

Burns, C., Izmailov, P., Kirchner, J. H., Baker, B., Gao, L., Aschenbrenner, L., Chen, Y., Ecoffet, A., Joglekar, M., Leike, J., Sutskever, I., and Wu, J. Weak-to-strong generalization: eliciting strong capabilities with weak supervision. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

Daniel Han, M. H. and team, U. Unsloth, 2023. URL http://github.com/unslothai/unsloth.

Dettmers, T., Lewis, M., Shleifer, S., and Zettlemoyer, L. 8-bit optimizers via block-wise quantization. 9th International Conference on Learning Representations, ICLR, 2022.

Feldman, V. Does learning require memorization? a short tale about a long tail, 2019. URL https://arxiv. org/abs/1906.05271.

Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., et al. Deepseek-r1 incentivizes reasoning in LLMs through reinforcement learning. Nature, 645:633–638, 2025. doi: 10.1038/s41586-025-09422-z.

Hendrycks, D. et al. Measuring massive multitask language understanding. In International Conference on Learning Representations (ICLR), 2021. URL https://arxiv. org/abs/2009.03300.

Hern´andez-Garc´ıa, A. and K¨onig, P. Data augmentation instead of explicit regularization. CoRR, abs/1806.03852, 2018. URL http://arxiv.org/ abs/1806.03852.

Hoffmann, J. et al. Training compute-optimal large language models, 2022. URL https://arxiv.org/ abs/2203.15556.

Kaplan, J. et al. Scaling laws for neural language models, 2020. URL https://arxiv.org/abs/2001. 08361.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Li, J., Beeching, E., Tunstall, L., Lipkin, B., Soletskyi, R., Huang, S. C., Rasul, K., Yu, L., Jiang, A., Shen, Z., Qin, Z., Dong, B., Zhou, L., Fleureau, Y., Lample, G., and Polu, S. Numinamath tir. [https://huggingface.co/AI-MO/

NuminaMath-TIR](https://github.com/ project-numina/aimo-progress-prize/ blob/main/report/numina_dataset.pdf),

- 2024.

Marek, M., Lotfi, S., Somasundaram, A., Wilson, A. G., and Goldblum, M. Small batch size training for language models: When vanilla SGD works, and why gradient accumulation is wasteful. In The Thirty-ninth Annual Conference on Neural Information Processing Systems,

- 2025. URL https://openreview.net/forum? id=52Ehpe0Lu5.

Muennighoff, N., Yang, Z., Shi, W., Li, X. L., Fei-Fei, L., Hajishirzi, H., Zettlemoyer, L., Liang, P., Cand`es, E., and Hashimoto, T. s1: Simple test-time scaling, 2025. URL https://arxiv.org/abs/2501.19393.

Muennighoff, N. et al. Scaling data-constrained language models, 2023. URL https://arxiv.org/abs/ 2305.16264.

Ni, J., Liu, Q., Dou, L., Du, C., Wang, Z., Yan, H., Pang, T., and Shieh, M. Q. Diffusion language models are super data learners. arXiv preprint arXiv:2511.03276, 2025.

Osa, T., Pajarinen, J., Neumann, G., Bagnell, J. A., Abbeel, P., and Peters, J. An algorithmic perspective on imitation learning. In Foundations and Trends in Robotics, volume 7, pp. 1–179. Now Publishers, 2018.

Ouyang, L. et al. Training language models to follow instructions with human feedback, 2022. URL https: //arxiv.org/abs/2203.02155.

Rein, D. et al. GPQA: A graduate-level google-proof Q&A benchmark. In Proceedings of the First Conference on Language Modeling (COLM), 2024. URL https:// openreview.net/forum?id=Ti67584b98.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., and Guo, D. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

Shorten, C. and Khoshgoftaar, T. M. A survey on Image Data Augmentation for Deep Learning. Journal of Big Data, 6(1):60, July 2019. ISSN 2196-1115. doi: 10.1186/ s40537-019-0197-0.

Team OLMo. Olmo 3, 2025. URL https://arxiv. org/abs/2512.13961.

Tirumala, K. et al. Memorization without overfitting: Analyzing the training dynamics of large language models, 2022. URL https://arxiv.org/abs/2205. 10770.

Yang, A. et al. Qwen3 technical report, 2025. URL https: //arxiv.org/abs/2505.09388.

Ye, Y., Huang, Z., Xiao, Y., Chern, E., Xia, S., and Liu, P. LIMO: Less is more for reasoning, 2025. URL https: //arxiv.org/abs/2502.03387.

### A. Hyperparameters

Hyperparameter Value GPUs 1 H100 94GB per run Optimizer 8-bit Adam Weight Decay 0.0

- Beta1 0.9
- Beta2 0.999 Grad. Norm Clipping 1.0 Batch Size 1 LR Scheduler Cosine Warmup Steps 10% of all updates RNG Seed 42

| |Olmo3 7B Qwen3 8B Qwen3 4B|
|---|---|
|Learning Rate<br><br>|2e-5 2e-5 3e-5|

### B. Full Results.

#### B.1. Dolci Dataset

Figures 7–9 show results on the Dolci dataset for three model backbones. Across all models, training for more epochs on smaller datasets consistently outperforms training on larger datasets for fewer epochs. Performance gains saturate once models approach full memorization, mirroring the convergence behavior discussed in Section 4.1.

Heatmaps for Olmo3 7B

AIME 2024 Acc@16

AIME 2025 Acc@16

GPQA Acc@4

40

30

32

256

256

256

42 44

34 34

33 37

128

128

128

46 42 42

- 37 33 37 35
- 37 34 38

34 35 37

64

64

64

46 35 40 42

35 34 35 39

32

32

32

Epochs

43 45 46 46 42

- 31 34 35 38 39 35
- 32 36 42 37 39

38 35 39 40 34

16

16

16

40 38 46 47 47 44

24 28 26 33 32 30

8

8

8

- 20 23 24 26 27 29 30 28
- 21 29 34 35 36 40 33

17 24 29 28 32 31 29

12 14 15 18 18 20 20

4

4

4

21 25 22 21 24 24 25 25

8 10 9 11 12 13 14 15

- 1

- 2

- 1

- 2

- 1

- 2

17 17 16 14 15 20 18 20 18

19 19 20 17 17 18 21 19 22

5 5 5 4 7 7 10 9 11

200

400

800

200

400

800

200

400

800

1600

3200

6400

1600

3200

6400

1600

3200

6400

12800

25600

51200

12800

25600

51200

12800

25600

51200

Pass@16

Pass@16

Pass@4

77

53

62

256

256

256

80 83

67 67

60 65

128

128

128

83 83 77

63 60 67

69 67 64

64

64

64

80 80 77 83

57 57 67 67

- 64 66 68 69 62
- 65 67 63 71

32

32

32

Epochs

87 83 80 83 80

67 63 73 67 63

16

16

16

77 80 80 80 80 73

67 60 60 63 70 67

43 53 49 60 60 52

8

8

8

73 73 70 77 77 73 73

47 47 60 57 57 63 50

26 27 29 35 34 38 39

4

4

4

63 63 63 63 67 67 63 67

37 43 43 43 47 53 47 47

19 27 22 26 26 24 27 29

- 1

- 2

- 1

- 2

- 1

- 2

50 50 53 57 53 67 57 53 47

37 43 33 40 40 37 47 37 50

13 14 13 11 18 18 25 23 24

200

400

800

200

400

800

200

400

800

1600

3200

6400

1600

3200

6400

1600

3200

6400

12800

25600

51200

12800

25600

51200

12800

25600

51200

Samples

Samples

Samples

- Figure 7. Dolci dataset results for Olmo3-7B. Scaling epochs on smaller datasets yields higher downstream accuracy than scaling the number of unique samples under a fixed update budget.

Heatmaps for Qwen3 8B

###### AIME 2024 Acc@16

###### AIME 2025 Acc@16

GPQA Acc@4

- 21 24
- 22

- 20 22 19 17 21 18 20

- 25 29 29 29 28 28

27 25 28 30 31

- 26 27 29 29

- 23 28 27
- 24 23

- 21

40

256

256

256

38 44

128

128

128

23 22 26

- 42 41 44 42 43 42

47 49 50 49 51

44 47 49 52

- 43 47 51

64

64

64

23 26 27 28

32

32

32

Epochs

25 24 29 28 31

16

16

16

23 25 27 29 30 31

8

8

8

18 19 19 18 19 20 21

29 28 27 27 31 33 36

4

4

4

15 15 13 13 10 10 14 14

15 15 15 15 16 13 15 17

20 22 18 17 20 21 27 25

- 1

- 2

- 1

- 2

- 1

- 2

11 12 13 10 9 6 6 6 12

12 13 14 12 11 8 7 10 11

15 18 16 14 16 15 17 19 22

200

400

800

200

400

800

200

400

800

1600

3200

6400

1600

3200

6400

1600

3200

6400

12800

25600

51200

12800

25600

51200

12800

25600

51200

Pass@16

Pass@16

Pass@4

60

43

68

256

256

256

60 60

47 50

63 74

128

128

128

57 47 63

50 50 47

- 67 65 69 62 67 62

73 75 77 72 73

72 68 72 75

- 68 74 76

64

64

64

60 63 60 57

60 57 57 53

32

32

32

Epochs

63 57 63 60 77

50 50 53 63 63

16

16

16

53 63 70 67 63 70

53 53 63 60 57 57

8

8

8

53 50 53 53 60 60 67

43 47 40 37 40 43 40

50 47 46 44 53 49 57

4

4

4

47 37 40 33 37 40 33 47

40 33 33 37 37 43 33 43

43 43 38 33 40 38 44 42

- 1

- 2

- 1

- 2

- 1

- 2

33 33 37 20 23 27 20 30 37

27 33 33 33 23 20 33 27 30

34 37 34 33 35 33 37 36 39

200

400

800

200

400

800

200

400

800

1600

3200

6400

1600

3200

6400

1600

3200

6400

12800

25600

51200

12800

25600

51200

12800

25600

51200

Samples

Samples

Samples

- Figure 8. Results for distillation from a Qwen3-8B teacher. Stronger teachers increase overall performance but do not eliminate the repetition advantage.

200

400

800

1600

3200

6400

12800

25600

51200

1248163264128256

Epochs

- 8 7 7 3 4 5 8 5 6

- 8 10 11 10 13 10 15 14

14 14 15 12 18 21 18

- 17 20 20 21 23 23
- 18 19 15 20 19

12 12 15 17

- 9 11 13

- 9 8

4

AIME 2024 Acc@16

200

400

800

1600

3200

6400

12800

25600

51200

1248163264128256

7 7 6 5 5 7 7 6 6

- 10 12 13 11 11 12 10 14

13 15 15 10 18 19 18

18 20 21 18 24 24

18 18 17 18 19

- 11 16 16 16

10 11 13

10 7

5

AIME 2025 Acc@16

200

400

800

1600

3200

6400

12800

25600

51200

1248163264128256

13 9 9 7 7 12 11 12 13

16 12 14 12 17 17 21 21

24 23 24 23 30 31 30

- 34 35 38 44 40 41
- 35 36 37 41 39

29 30 34 36

24 29 30

24 22

19

GPQA Acc@4

200

400

800

1600

3200

6400

12800

25600

51200

Samples

1248163264128256

Epochs

30 20 23 17 17 17 20 20 27

20 27 27 30 37 37 30 37

37 37 43 33 40 50 37

43 50 40 50 47 43

40 47 30 43 43

33 27 33 43

27 37 30

30 33

23

Pass@16

200

400

800

1600

3200

6400

12800

25600

51200

Samples

1248163264128256

27 20 20 20 30 30 23 23 30

40 23 40 30 30 37 33 37

40 37 43 33 47 37 40

47 43 43 40 50 43

37 43 40 50 40

33 40 37 40

33 27 33

40 30

23

Pass@16

200

400

800

1600

3200

6400

12800

25600

51200

Samples

1248163264128256

30 23 22 17 16 27 27 25 30

39 30 29 26 33 35 41 39

- 46 44 45 42 50 54 53

64 66 67 71 69 64

63 68 67 72 69

59 60 68 66

57 57 58

54 49

- 47

Pass@4

Heatmaps for Qwen3 4B

- Figure 9. Dolci dataset results for Qwen3-8B. The repetition advantage persists across dataset sizes, with gains plateauing at higher epoch counts.

#### B.2. Qwen3 Distills

Figures 10 and 11 examine the effect of teacher model size in distillation. While stronger teachers improve absolute performance, the repetition advantage remains robust: multi-epoch training on smaller distilled datasets consistently outperforms scaling unique samples.

Heatmaps for Olmo3 7B (Qwen3 0.6B Distill)

AIME 2024 Acc@16

AIME 2025 Acc@16

GPQA Acc@4

7

11

8

1248163264128

1248163264128

1248163264128

17 17

- 17 20 15 15 10

21 24 22 19

23 22 22

- 18 21

14 19

23 21 19

19 20 20

18 18 21 18

17 19 22 25

Epochs

15 15 12 14 12

10 8 8 8 9

9 8 9 7 7 6

11 12 10 10 8 10

- 1 2 2 1 2 1 1 3
- 2 1 2 2 2 2 3
- 3 4 3 4 5 4

6 5 4 5 4 5 5

9 8 6 6 8 7 8

3 5 3 3 3 2 3 3

6 5 5 4 5 3 6 6

200 400 800 1600 3200 6400 12800 25600

200 400 800 1600 3200 6400 12800 25600

200 400 800 1600 3200 6400 12800 25600

Pass@16

Pass@16

Pass@4

27

33

27

1248163264128

1248163264128

1248163264128

43 43

47 47

35 47

63 60 57

53 50 43

45 49 48

57 50 53 47

43 47 47 50

42 43 48 51

Epochs

37 50 43 43 33

40 43 37 40 33

24 24 22 22 23

30 27 27 27 30 30

33 30 33 30 27 37

- 8 5 7 8 8 6 11
- 9 12 8 10 16 11

30 20 27 17 20 20 17

33 33 27 27 30 23 30

17 20 17 17 13 13 13 13

27 20 27 20 30 23 20 27

3 6 5 6 8 5 5 10

200 400 800 1600 3200 6400 12800 25600

200 400 800 1600 3200 6400 12800 25600

200 400 800 1600 3200 6400 12800 25600

Samples

Samples

Samples

- Figure 10. Results for distillation from a Qwen3-0.6B teacher. Despite weaker teacher signals, repetition continues to improve downstream accuracy.

Heatmaps for Olmo 7B (Qwen3 8B Distill)

AIME 2024 Acc@16

###### AIME 2025 Acc@16

GPQA Acc@4

21

- 19 25 30 27 29 28

- 28 32 30 34 33

31 28 34 33

- 29 31 30
- 30 29

- 20

5

1248163264128

1248163264128

1248163264128

- 28 35 31 41 41

34 33 38 39

34 33 37

- 29 30

- 8 15 14 21 26

12 17 24 34

11 18 23

- 9 19

Epochs

19 23 26 27 32 32

- 4 5 4 6 7 7 7
- 5 8 9 12 11 13

16 18 18 19 21 21 25

14 18 18 21 24 20 24

12 10 12 12 16 12 16 15

11 12 12 14 12 16 16 19

1 3 3 3 4 4 4 6

200

400

800

200

400

800

200

400

800

1600

3200

6400

1600

3200

6400

1600

3200

6400

12800

25600

12800

25600

12800

25600

Pass@16

Pass@16

Pass@4

60

47

15

1248163264128

1248163264128

1248163264128

70 73

57 60

21 48

73 67 73

53 57 67

26 39 49

70 70 77 73

57 57 57 63

27 38 52 63

Epochs

70 70 70 77 80

50 57 60 60 63

23 34 31 42 52

60 73 73 77 73 73

43 60 53 50 47 50

16 19 20 26 24 26

53 53 57 53 57 57 57

37 47 40 43 47 40 47

10 12 9 17 16 18 17

37 33 43 40 40 53 50 50

37 37 37 33 37 43 43 43

5 9 8 10 10 12 10 16

200

400

800

200

400

800

200

400

800

1600

3200

6400

1600

3200

6400

1600

3200

6400

12800

25600

12800

25600

12800

25600

Samples

Samples

Samples

- Figure 11. Results for distillation from a Qwen3-8B teacher. Stronger teachers increase overall performance but do not eliminate the repetition advantage.

#### B.3. Qwen3 8B Distill; Pos. vs Neg.

Figures 12 and 13 separate distilled samples by correctness. The repetition advantage is substantially stronger when training on correct reasoning traces, while incorrect samples reduce overall performance and weaken the gains from repetition.

Heatmaps for Olmo 7B (Qwen3 8B Distill; Positive)

AIME 2024 Acc@16

AIME 2025 Acc@16

GPQA Acc@4

39

27

16

12481632

12481632

12481632

39 37

31 34

19 23

33 35 37

31 30 29

12 17 19

Epochs

25 26 30 27

22 22 27 26

8 8 10 10

14 16 19 19 20

14 17 17 20 19

4 4 4 6 6

9 12 13 14 13 13

12 12 13 13 15 15

2 2 2 2 4 3

200 400 800 1600 3200 6400

200 400 800 1600 3200 6400

200 400 800 1600 3200 6400

Pass@16

Pass@16

Pass@4

77

53

41

12481632

12481632

12481632

77 80

60 63

42 52

77 67 80

60 57 53

25 35 43

Epochs

60 70 67 67

53 53 53 47

19 20 23 25

47 50 47 53 60

37 43 43 40 47

8 11 12 15 15

37 37 37 40 43 33

33 33 33 37 43 40

6 6 5 7 11 9

200 400 800 1600 3200 6400

200 400 800 1600 3200 6400

200 400 800 1600 3200 6400

Samples

Samples

Samples

- Figure 12. Results using only correct (positive) distilled samples from a Qwen3-8B teacher. Repetition yields consistent gains until memorization saturates.

200 400 800 1600 3200 6400

12481632

Epochs

11 11 11 14 14 15

18 20 21 21 22

25 28 28 30

33 34 35

33 40

35

AIME 2024 Acc@16

200 400 800 1600 3200 6400

12481632

12 13 14 14 16 13

22 20 20 20 20

24 26 25 28

32 29 31

31 33

31

AIME 2025 Acc@16

200 400 800 1600 3200 6400

12481632

- 2 1 3 3 3 5
- 3 4 5 5 7

8 9 9 12

15 19 19

25 29

29

GPQA Acc@4

200 400 800 1600 3200 6400

Samples

12481632

Epochs

50 37 43 43 47 57

57 60 57 53 67

73 73 70 73

73 73 73

73 80

70

Pass@16

200 400 800 1600 3200 6400

Samples

12481632

37 40 37 33 40 37

43 43 43 43 47

50 47 50 50

57 63 67

63 67

70

Pass@16

200 400 800 1600 3200 6400

Samples

12481632

7 5 10 8 11 13

11 11 15 16 18

22 24 24 27

33 40 42

54 54

56

Pass@4

Heatmaps for Olmo 7B (Qwen3 8B Distill; Negative)

- Figure 13. Results using incorrect (negative) distilled samples from a Qwen3-8B teacher. Overall performance is lower, and the repetition advantage is substantially diminished.

