## Decoding Compressed Trust: Scrutinizing the Trustworthiness of Efficient LLMs Under Compression

# arXiv:2403.15447v3[cs.CL]4Jun2024

Junyuan Hong*1 Jinhao Duan*2 Chenhui Zhang*3 Zhangheng Li*1 Chulin Xie4 Kelsey Lieberman5 James Diffenderfer6 Brian Bartoldson6 Ajay Jaiswal1 Kaidi Xu2 Bhavya Kailkhura6 Dan Hendrycks7 Dawn Song8 Zhangyang Wang1 Bo Li49

Model & Code: https://decoding-comp-trust.github.io WARNING: This paper contains model outputs that may be considered offensive.

### Abstract

Compressing high-capability Large Language Models (LLMs) has emerged as a favored strategy for resource-efficient inferences. While stateof-the-art (SoTA) compression methods boast impressive advancements in preserving benign task performance, the potential risks of compression in terms of safety and trustworthiness have been largely neglected. This study conducts the first, thorough evaluation of three (3) leading LLMs using five (5) SoTA compression techniques across eight (8) trustworthiness dimensions. Our experiments highlight the intricate interplay between compression and trustworthiness, revealing some interesting patterns. We find that quantization is currently a more effective approach than pruning in achieving efficiency and trustworthiness simultaneously. For instance, a 4bit quantized model retains the trustworthiness of its original counterpart, but model pruning significantly degrades trustworthiness, even at 50% sparsity. Moreover, employing quantization within a moderate bit range could unexpectedly improve certain trustworthiness dimensions such as ethics and fairness. Conversely, extreme quantization to very low bit levels (3 bits) tends to reduce trustworthiness significantly. This increased risk cannot be uncovered by looking at benign performance alone, in turn, mandating comprehensive trustworthiness evaluation in practice. These findings culminate in practical recommendations for simultaneously achieving high utility, efficiency, and trustworthiness in LLMs.

*Equal contribution 1University of Texas at Austin 2Drexel University 3MIT 4UIUC 5Duke University 6Lawrence Livermore National Laboratory 7Center for AI Safety 8University of California, Berkeley 9University of Chicago. Correspondence to: Bo Li <bol@uchicago.edu>.

Proceedings of the 41st International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s).

### 1. Introduction

Large Language Models (LLMs) have demonstrated exceptional abilities in language understanding, generation, and reasoning (Touvron et al., 2023b; Ouyang et al., 2022; Bubeck et al., 2023; Wei et al., 2022). Despite their impressive performance, the steep increase in model size, with parameters ranging from millions to several hundred billion, limits their deployment on consumer devices with constrained memory and computational power. To address the growing need for more efficient LLMs (Bartoldson et al., 2023), smaller models are often pre-trained alongside their larger counterparts. For instance, the LLAMA2 suite features a spectrum of models, including 7b, 13b, 34b, and 70b parameter versions (Touvron et al., 2023b). However, training such a diverse batch is an enormous undertaking, with even the two smallest models consuming around half a million GPU hours in total. In stark contrast, model compression offers a time-efficient alternative, significantly accelerating the inference process. For example, compressing a 13b model to 4 bits takes merely half an hour on a 48Gb A40 GPU and results in an average speedup of 3.2 − 3.3× in inference speed, as demonstrated by AWQ compared to Huggingface’s FP16 implementation (Lin et al., 2023). Moreover, advanced state-of-the-art (SoTA) compression techniques can maintain performance levels comparable to dense models, even at high compression rates (≥ 50%) (Frantar & Alistarh, 2023; Sun et al., 2023; Lin et al., 2023; Jaiswal et al., 2023b). This efficiency coupled with maintained utility showcases the potential for a balanced approach in the use of LLMs.

Contrary to the clear trend of improved efficiency, the effectiveness of compressed or smaller models presents a more complex picture, with their performance varying (often inconsistently) across different trust dimensions. The trustworthiness of LLMs, as outlined in (Wang et al., 2023a), is multifaceted and increasingly critical, particularly given their widespread use in high-stakes scenarios (Wang et al., 2023b; Driess et al., 2023; Demszky et al., 2023). Recent research has begun to unravel the intricate relationship be-

[Figure 1]

- Figure 1. Our evaluation aims to assess the trustworthiness of LLMs under compression. Leveraging the trustworthiness evaluation benchmark (Wang et al., 2023a), we compare various paths toward efficient small LLMs, including pre-training and different compression algorithms. We uncover the hidden effect of compression on diverse trustworthiness metrics and identify a bag of tricks for efficient and trustworthy LLMs.

tween the size of pre-trained LLMs and their trustworthiness, revealing the diverse characteristics of downscaled models. On one hand, studies by Perez et al. and Sun et al. highlight benefits such as reduced sycophantic tendencies and lower privacy risks in smaller LLMs. On the other, Huang et al. found these models to be more vulnerable to backdoor attacks, raising concerns about their reliability.

The recent fine-grained benchmark of compressed models’ performance (Jaiswal et al., 2023a), especially in knowledgeintensive tasks, further complicates the picture. Even with minor reductions in size (around 25% sparsity), these models often experience notable performance declines, despite only having explored stable perplexity metrics. These findings suggest that the impact of compression on LLMs is not straightforward. However, current evaluations typically focus either on limited aspects (benign utility only; or plus one or two trust dimensions), or only on uncompressed pretrained LLMs, leaving the broader spectrum of trustworthiness in compressed models, or compressed trust, somewhat unclear. This gap highlights the need for a more holistic understanding of how compression affects the trustworthiness of LLMs across various dimensions.

In this paper, we decode the compressed trust by conducting the first comprehensive evaluation of compressed LLMs on trustworthiness across eight critical trust dimensions (Wang

- et al., 2023a), including stereotype, toxicity, privacy, fairness, ethics, and robustness (adversarial, out-of-distribution and adversarial demonstration) – that is in addition to the utility performance measured by multi-task language understanding. Our assessment includes LLMs compressed by five SoTA methods at varying compression rates. The study leads to a rich set of previously overlooked insights into understanding the potential and risks of the compressed model

in practice. As outlined in Fig. 1, our main contributions and observations are summarized as follows.

- • We rigorously assess a broad range of compressed Large Language Models (LLMs), aiming to illuminate the path toward efficient and reliable LLMs.
- • We conduct an in-depth analysis of two approaches to create 7b-sized models: pre-training from scratch, and compression from larger pre-trained ones (13b). Key insights include: smaller (7b) models potentially outperforming larger (13b) ones in some trust dimensions (e.g., out-of-distribution robustness, adversarial robustness, and fairness); quantization effectively achieves similar performance as its source dense model (13b) across all trust metrics; and pruning demonstrating inferior and inconsistent results in both utility and trust aspects.
- • We explore high compression rates (around or over 50%) to empirically determine optimal LLM compression rates for trustworthiness, offering practical guidelines for efficient LLMs. We observe that quantization not only enhances efficiency at low overhead but also improves certain trustworthiness dimensions, suggesting an interesting win-win situation between trustworthiness and efficiency.
- • We further investigate more extreme compression cases, such as 3-bit quantization, noting significant performance decreases across multiple trust (but not benign) dimensions with even the most advanced quantization algorithms, indicating notable challenges of balancing efficiency and trust in ultra-high compression scenarios.
- • With these findings, we summarize a bag of tricks, that highlight the pitfalls in the trustworthiness of compression and may guide compressing LLMs with trustworthiness in the future.

### 2. Related Works

Compression for efficient LLMs. As a crucial step towards capable yet efficient LLMs, a variety of model compression techniques for LLMs try weight/activation quantization (Dettmers et al., 2022; Frantar et al., 2022; Frantar & Alistarh, 2022; Lin et al., 2023; Chee et al., 2023; Tseng et al., 2023; Xiao et al., 2023), pruning (Frantar & Alistarh, 2023; Sun et al., 2023), low-rank approximation (Xu et al., 2023), and knowledge distillation (Timiryasov & Tastet, 2023). Among them, (post-training) weight quantization and semi-structured pruning methods without backpropagation are most scalable as they can be efficiently executed on pre-trained models without extra training processes.

Quantization. As a pioneer work in weight-only quantization, LLM.INT8() (Dettmers et al., 2022) proposed the first Int8 matrix multiplication for feed-forward and attention projection layers, that quantized LLM parameters into 8-bit integers. Taking a step further, GPTQ (Frantar et al., 2022) leverages Optimal Brain Quantization (OBQ, Frantar & Alistarh 2022) for solving a layer-wise quantization problem, which reduces the bit-width to 3 or 4 bits. Noticing the diverse importance of weights, Activation Aware Quantization (AWQ, Lin et al. 2023) quantizes LLMs while preserving the salient weights. To further squeeze the bit-width, QuIP (Chee et al., 2023) and QuIP# (Tseng et al., 2023) combine lattice codebooks with incoherence processing to create state-of-the-art 2-bit-quantized models. Together with weight quantization, a series of works also quantize the activations together (Xiao et al., 2023; Ahmadian et al., 2023), further reducing GPU memory overhead and accelerating compute-intensive operations.

Pruning. In addition to quantization, model pruning compresses LLMs by reducing the number of redundant parameters. Despite numerous existing algorithms for pruning (Singh & Alistarh, 2020; Zhu & Gupta, 2017; Gale et al., 2019; Jaiswal et al., 2022; Lin et al., 2020; Liu et al., 2023a; Jaiswal et al., 2023c; Mostafa & Wang, 2019; Dettmers & Zettlemoyer, 2019; Evci et al., 2020; Diffenderfer & Kailkhura, 2020), their ad-hoc adaptation for LLMs is restricted, due to the lack of luxury to perform iterative retraining to regain any performance drop during compression. Although the simplest method is removing weights by magnitude (Jaiswal et al., 2023b), such a strategy is likely to remove important weights that greatly bias the generation. Therefore, calibrating pruning strategies were proposed to mitigate the loss. For example, SparseGPT (Frantar & Alistarh, 2023) calibrates the weights to achieve 60% model sparsity. Wanda (Sun et al., 2023) prunes model weights with the smallest magnitudes multiplied by their input activations. Later, more advanced pruning methods are designed in structured ways (Ma et al., 2023), e.g., layer-wise sparsity (Yin et al., 2023).

The rich research on model compression demonstrates the popularity of small and efficient models. As these compressed models are not further tuned post-compression, finding out what is lost in the compressed weights necessitates more comprehensive evaluations of compressed LLMs.

Evaluating compressed LLMs. The performance of compressed models has been widely evaluated by their perplexity on pre-training datasets, zero-shot or few-shot classification accuracy (Paperno et al., 2016), question answering (Tata & Patel, 2003) and reasoning (Sakaguchi et al., 2021; Boratko et al., 2018) abilities, and knowledge (Hendrycks et al., 2020). By these common evaluation metrics, even low-bit quantization (e.g., 4-bit) methods can maintain a performance similar to their dense counterparts (Lin et al., 2023) in accuracy or perplexity. Recently, Jaiswal et al. systematically re-examine how existing LLM compression techniques are evaluated, trying to unveil their hidden costs on more complicated tasks like understanding, reasoning, summarization, instruction-following, and etc. They find that quantization outperforms pruning significantly at a similar compression rate on tested tasks. (Namburi et al., 2023) propose to investigate how model compression affect parametric knowledge.

Except for the hidden costs in benign scenarios, there still lacks a systematic understanding of the costs under trustrelated scenarios that is crucial upon deployment. A recent comprehensive evaluation on the trustworthiness of several pre-trained LLMs (Mo et al., 2023) shows that increasing model sizes tend to weaken their overall trustworthiness across multiple perspectives. Yet, compression provides a distinct mechanism for scaling model sizes after pre-training and its trustworthiness demands in-depth investigation.

Unique to this paper, we are the first to comprehensively study how trustworthiness changes by compressing models into smaller ones. We hope that our work will help understand LLM-compression algorithms in terms of their trustworthiness, and, in turn, benefit the safe scaling of LLMs in the real world.

### 3. Assessing the Trustworthiness of Compressed LLMs

Understanding the trustworthiness of compressed models requires a comprehensive evaluation to gain insights. In this paper, we are interested in these specific questions: 1 What is the recommended compression method in the joint view of multi-dimensional trustworthiness and standard performance? 2 What is the optimal compression rate for trading off trustworthiness and efficiency? 3 In extreme compression rates (3-bit quantization), how will the compressed models perform according to our metrics? To this end, we conduct a comprehensive evaluation where we place a wide

Table 1. Configurations of different compression methods. Calibration data are used to update weight values (e.g., GPTQ) or select prunable weights (e.g., Wanda). The calibration criterion defines which weights to prune or how to update weights. If weights are updated, the values will be determined by weight or activation (act) based criteria.

Compression Weight Calibration Type Method Rate Update Data (Size) Criterion Hardware-friendly

Pruning Magnitude 2:4 ✗ ✗ weight ✓ Pruning SparseGPT 2:4 ✓ ✓(128) weight ✓ Pruning Wanda 2:4 ✗ ✓(128) weight × act. ✓

Quantization GPTQ 3,4,8-bit ✓ ✓(128) weight ✓ Quantization AWQ 3,4,8-bit ✓ ✓(128) act. ✓

spectrum of compressed models under diverse trustworthiness dimensions of compressed models. We select diverse methods from two categories, quantization (reducing weight precision) and pruning (removing parameters), to compress three types of models (chat and non-chat models). The diversity of evaluated models and methods essentially helps us to gain insights into the questions.

Models. In this paper, we study three pre-trained models: LLAMA2 13b, LLAMA2 13b Chat (Touvron et al., 2023b), and Vicuna 13b Chat (Chiang et al., 2023). All three of these models have 13 billion parameters in their dense format. LLAMA2 13b is an LLM pre-trained on 2 trillion tokens of publicly available data in an auto-regressive manner. Customized for conversations, LLAMA2 13b chat and Vicuna 13b chat are the instruction fine-tuned models based on the 2nd and 1st (Touvron et al., 2023a) generations of LLAMA, respectively. As the three models have different strengths in the spectrum of trustworthiness (Mo et al., 2023), they provide a diverse view for assessing the effects of compression methods. For interested readers, we include the model comparison results in Appendix B.

Compression methods. As shown in Table 1, our work primarily focuses on the existing training-free and data-free LLM pruning/quantization techniques, which are efficient and cost-effective in compressing models. For pruning, we include the top-2 techniques i.e., SparseGPT (Frantar & Alistarh, 2023) and Wanda (Sun et al., 2023)), along with the baseline of One-shot Magnitude-based Pruning (Mag) (Han et al., 2015). In our experiments, we focus on a popular semi-structured N:M sparsity pattern: a fine-grained sparsity pattern in which only N weights are non-zero for every continuous M weights (Nvidia, 2020; Zhou et al., 2021). Note that we restrict our experiments to N:M pruning due to its potential to provide actual hardware acceleration unlike exiting numerous unstructured pruning approaches. Recent research endeavors have harnessed quantization to compress LLMs and many quantization algorithms have shown impressive performance. For our work, we selected two popular and easy-to-use algorithms. GPTQ (Frantar et al., 2022) is a layer-wise quantization technique based on approximated second-order information toward minimal

accuracy loss compared to the uncompressed version. Motivated by the fact that weights are not equally important, AWQ (Lin et al., 2023) leverages the activation-aware quantization to adaptively scale weights and therefore enables LLMs to be compressed at higher rates.

Evaluation dimensions. We include both a trustworthy benchmark and a standard language understanding benchmark to thoroughly evaluate LLMs. 1 Benign performance. First, the benign performance is evaluated by Massive Multitask Learning Understanding (MMLU) (Hendrycks et al., 2020), which is represented by average accuracy across all tasks. MMLU covers a wide range of 57 tasks covering diverse abilities: understanding and reasoning abilities across four areas including humanities, social science, STEM (Science, Technology, Engineering, and mathematics), and others. 2 Trustworthiness. Second, we adopt the state-of-the-art trustworthiness benchmark for LLMs, DecodingTrust (Wang et al., 2023a). The benchmark includes 8 trustworthy dimensions: Stereotype, Privacy, Toxicity, Fairness, Adversarial Robustness (AdvGLUE++), Out-Of-Distribution (OOD) Robustness, Robustness to Adversarial Demonstrations (AdvDemo), and Ethics. Examples for tasks are included in Fig. 1. 3 Refusal rates. Complementary to the aforementioned metrics, we also include the refusal rate to characterize how well LLMs can respond to benign/malicious instructions. For many prompts in the benchmark, the response is expected to be in a specific set, e.g., ‘agree’ or ‘disagree’ with a stereotype. Response out of the range may be due to unawareness of the question but not exact safety. Therefore, we define such behavior as refusal that can provide additional information about LLM behavior in these challenging scenarios. Note that different perspectives have different ways of handling the refused content. Generally, the refused responses are counted as the rejected answers. For classification tasks measured by accuracy in AdvGLUE++, the refusal means the wrong answer. For classification tasks measured by False Positive Rates (FPR) (e.g., in Fairness), the refusal responses are counted as negative responses. The refused answers are counted as safe predictions from the privacy perspective, where a refused answer does not leak any pri-

LLAMA2 Chat

LLAMA2

Vicuna Chat

AdvGLUE++

AdvGLUE++

AdvGLUE++

Stereotype

Stereotype

Stereotype

AdvDemo

AdvDemo

AdvDemo

Fairness

Fairness

Fairness

Toxicity

Toxicity

Toxicity

Privacy

Privacy

Privacy

Ethics

Ethics

Ethics

MMLU

MMLU

MMLU

OOD

OOD

OOD

|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|
|---|---|---|---|---|---|---|---|---|
|-6.4|9.1|8.4|-9.4|-5.5|6.1|-1.1|3.3|-7.7|

|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|
|---|---|---|---|---|---|---|---|---|
|-9.8|-34.0|-10.6|-16.1|-31.6|11.8|16.0|23.0|52.3|

|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|
|---|---|---|---|---|---|---|---|---|
|-4.9|8.7|-2.6|0.7|6.0|7.4|-1.1|-4.6|-7.3|

13b

dense

7b

|0.0|-0.0|-0.0|-0.1|-0.8<br>-0.4<br>|-0.2|0.2|0.6|0.0|
|---|---|---|---|---|---|---|---|---|
|0.0|-0.3|0.7|0.8| |2.6|0.1|-0.1|0.0|

|0.2|0.2|-0.5|0.2|-0.0<br>-0.1<br>|-0.5|0.0|1.8|-0.7|
|---|---|---|---|---|---|---|---|---|
|0.1|0.3|-0.5|0.3| |-2.0|-0.1|-0.4|-0.3|

|-0.1|-0.2|0.2|0.2|0.5|1.4|-0.1|-0.8|-0.7|
|---|---|---|---|---|---|---|---|---|
|0.0|0.2|0.2|-1.7|1.1|0.7|20.1|-1.4|-0.3|

quantization

AWQ

GPTQ

|-19.2|9.0|-7.5|-14.3|-40.2|-2.7|-31.6|-6.3|-22.3|
|---|---|---|---|---|---|---|---|---|
|-13.3|3.1|-7.9|-16.8|-14.8|1.4|-10.2|-7.2|-10.6|
|-17.9|2.4|-14.3|-16.4|-31.8|6.2|-31.7|-14.7|-5.0|

|-25.8|-2.6|-28.7|9.3|-22.9|3.1|-4.2|10.1|25.7|
|---|---|---|---|---|---|---|---|---|
|-18.1|3.2|-17.3|9.0|-13.0|-0.2|4.4|5.3|3.7|
|-21.0|-1.8|-27.6|6.6|-17.8|-4.4|3.4|3.9|39.7|

|-21.8|7.4|-5.3|-0.1|-9.2|22.3|10.2|-14.2|13.3|
|---|---|---|---|---|---|---|---|---|
|-15.2|13.6|-5.2|-24.0|18.2|22.6|9.3|-13.6|-29.3|
|-19.0|16.5|-10.3|-23.0|22.3|28.1|12.4|-10.5|-38.0|

mag

pruning

sparsegpt

wanda

- Figure 2. Relative score difference w.r.t. 13b models. Every model is compressed at a 50% rate that leads to a similar model size as the

- 7b model. Darker blue/red colors indicate more improvement/drops w.r.t. to the 13b dense models. Gray dots/lines per cell indicate significantly lower/higher refusal rates (over 10%) which cast biases in the actual opinion/knowledge of a model. Quantization appears to be the most effective solution with minimal loss both on trustworthiness and on benign performance. Scores of dense models are at Fig. 8.

vate information. All our evaluation results are based on the 0-100 normalized scale denoted as “points” following DecodingTrust (Wang et al., 2023a).

### 4. Revisiting Paths to 7B-sized LLMs: Training Smaller, or Compressing Larger?

Scaling up the parameters of an LLM is believed to be a general strategy for enhancing various generation abilities, including reasoning, math, language understanding, etc. Existing supportive findings encourage people to train larger and larger models (Kaplan et al., 2020). But serving models on consumer-grade GPUs contrarily demands more efficient and often smaller models. As a popular choice for deployment, 7b LLMs are suitably tailored to be accommodated by numerous consumer-grade GPUs.

There are different ways to obtain 7b-sized models that share similar computation and space complexities as 7 billion 16-bit parameters: 1 Pre-training a 7b model by similar strategies (dataset, optimization, etc.) as larger models.

2 Compressing a double-sized model (13 billion parameters approximately), which reduces the parameter number or bit rates to obtain the size- and efficiency-compatible substitutes of 7b models. Note that 13b models generally exhibit superior performance than 7b ones, and compression may retain a good ratio of language performance (Lin et al., 2023; Frantar et al., 2022). It seems to imply that compression is a better choice. Yet, lacking a comprehensive evaluation of the trustworthiness in literature, such compression may bring some hidden effects, especially at a high compression rate. Therefore, it remains unclear but critical to answer: which is the preferred route to achieve 7b-sized models with

comprehensive trustworthiness?

Setup. We use the 13b models as a baseline to scrutinize the compressed trust and compare 7b and 7b-sized compressed models. The perspective-wise score differences w.r.t. the baseline are present in Fig. 2. 7b-sized models are compressed from 13b LLMs, LLAMA2 Chat, LLAMA2, and Vicuna by two quantization and three pruning methods. As SparseGPT with 50% sparsity is sensitive to the calibration set, we repeat the experiments with three randomly sampled calibration sets from the C4 dataset (Raffel et al., 2019) and report the average.

Pre-trained 7b LLMs. In the top two rows of Fig. 2, the comparison between 7b and 13b models shows non-uniform but interesting disparities. 1 The 13b model is consistently better than the 7b model on MMLU, Adv Demo (backdoor resilience), and Ethics, but not always better in other dimensions. 2 Surprisingly, the smaller LLAMA2 Chat is significantly better on inference robustness (OOD and AdvGLUE++), and Fairness by over 5 points. A similar advantage in Fairness can be consistently observed in the LLAMA2 and Vicuna models. Though the advantages in other dimensions are less consistent among models, there are generally at least three dimensions in which 7b models are favored over 13b ones. 3 For the non-aligned model, LLAMA2, both the advantages and disadvantages are enlarged by 10 to 52 points. The large variance may imply the overlooked importance of alignment for down-scaling resilience.

Compressed 7b-sized LLMs. As 7b models not only enjoy some advantages but also suffer from losses compared to 13b models, it is interesting to ask: which direction should

###### MMLU

AdvGLUE++

OOD

Adv Demo

Ethics

Fairness

Toxicity

Privacy

Stereotype

100

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| |3<br><br>of| |c|4<br><br>om| |p|8<br><br>re|s|

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| |3|L|L|4<br><br>A|M| |8<br><br>A|2|

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
|3| | | | | | | | |
| |3<br><br>b| |C|4<br><br>h|a|t|8<br><br>to| |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| |3|lo| |4<br><br>we|r| |8<br><br>th|a|

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
|8| | | | | | | | |
| |3|s| |4<br><br>re|p|r|8<br><br>es|en|

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| |3|in| |4<br><br>th|e| |8| |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| |3| |i|4<br><br>ll|b|e|8|less|

Method

90

AWQ GPTQ

| |
|---|

80

70

60

50

40

30

3 4 8

3 4 8

- Figure 3. The effect ing 1 the low-bit region ted x-axis) s consistent with the dense model but the effect may be positive in some perspectives. Black and red lines indicate the performance of 13b and 7b dense models, respectively. Standard deviations are reported with fewer bits. Grey areas indicate drops over 5 points. Dash lines represent the +/- 5 points w.r.t. the scores of the 13b model.

the compression lead to?

Quantization. In Fig. 2, we find that quantized 8-bit is a consistently comparable alternative to the 13b model with almost the same trustworthiness and benign performance. This consistency also implies that quantized 13b models inherit both the advantages and disadvantages of the 13b model (w.r.t. 7b). The conclusion is consistent in Vicuna13b and LLAMA2. Note that LLAMA2 was not aligned, implying that such trustworthiness preservation is not an essential result of alignment.

Pruning. In AdvGLUE++, the three pruning methods have similar scaling tendencies to improve/degrade the trustworthiness of LLAMA2 Chat, though not in the same magnitude. Further balanced improvements can be achieved by designing more sophisticated pruning schemes, e.g., (Wei

- et al., 2024). Similar improvement was also discussed in (Hasan et al., 2024) for jailbreaking resilience. However, Hasan et al. (2024) focuses on unstructured pruning, which is not hardware-friendly and cannot enjoy the actual efficiency improvements. Instead, we show a similar gain with 2:4 (50%) pruning, which can speed up the computation and save memory at hardware. When we extend our view to all three models, we observe the improvement is not consistent in some dimensions. For example, Fairness is significantly improved with the Vicuna but not with others.

#### Takeaways.

- • 7b models outperform their 13b counterparts in 3-4 trust dimensions by over 5 points, among which Fairness is consistently favored for all models.
- • Quantizing 13b models into 8-bit precision (7b-sized) incurs negligible (smaller than 3-point) drops across all metrics.
- • Pruning suffers from serious loss on at least three dimensions by over 5 points. Except for MMLU and OOD, results in most dimensions are different across models.

### 5. From Moderate to High Compression Rates: The (Unexpected) Gains and Losses

As 8-bit quantization has demonstrated impressive trustworthiness, we look into higher compression rates. Specifically, we are interested in the two questions: (1) To what extent can we compress models while maintaining trustworthiness? (2) What are the negative effects of extreme compression rate (3-bit) on trustworthiness?

Setup. To answer these questions, we extend the LLAMA2 13b Chat experiments to 3,4 bits using GPTQ and AWQ. For 3-bit and 4-bit, we repeat the experiments three times with randomly subsampled calibration sets.

5.1. Finding the Essential Compression Rates and Induced Gains for Trustworthiness

Essential compression rate. While lower bit rates provide better efficiency, the immediate price is performance degradation, for example, the degraded multi-task ability (MMLU) in Fig. 3. Within the scope of this paper, we consider a compression rate to be essential if the score drop is within 5 points, and at higher rates it drops more. 1 In Fig. 3, 3-bit is essential for MMLU but not all other perspectives. 2 In all perspectives, the 4-bit compression can preserve the trustworthiness within a 5-point drop. In other words, the high compression rate (4-bit) leads to a sweet spot for efficiency, utility (benign performance), and trustworthiness. 3 Compared to the pre-trained small model (LLAMA2 7b), the 4-bit quantization of a 13b model is more efficient and more accurate in language understanding. In trustworthiness, the 4-bit model is better at Ethics, Adv Demo, and Stereotype. Just like the 8-bit model, the 4-bit model also restores the weakness of the dense 13b model in AdvGLUE++, OOD, and Privacy but GPTQ surprisingly fixes the deficiency of the 13b model in Fairness.

#### Quantization induces low-cost (unexpected) gains in

Task

Zero-shot Few-shot Evasive Jailbreak

Error Rate

###### FPR

Refusal Rate

0.6

0.4

0.2

0.0

3 4 8 16 bits

3 4 8 16 bits

3 4 8 16 bits

- Figure 4. Evaluation of GPTQ-quantized LLAMA2 13b Chat models in four Ethics scenarios in terms of performance (error rate or FPR) and refusal rate. Facing evasive sentences, the 4-bit quantization can significantly reduce the portion of misclassified immoral actions (i.e., lower FPR). In other scenarios, the 4-bit model reduces the refusal rates a lot w.r.t. high-bit models.

trustworthiness. In Fig. 3, we notice that 4-bit 13b models can outperform the 13b dense models by more than 5 points in Fairness and Ethics. Specifically, at the 4-bit rate, the model will emerge to improve the Ethics ability from 54.1 to 76.3 (GPTQ) or 62.8 (AWQ). The results imply an encouraging message that the enhancement may occur at a low cost by quantization (almost for free) compared to traditional training-based alignment. To uncover the source of the gains in the perspectives, we look into their sub-scenarios with the refusal rates. We focus on the GPTQ-quantized LLAMA2 13b Chat models since their gains are often larger.

- Case Study 1: Ethics gain. The Ethics score is aggregated from four immoral-action recognition tasks: the Zero-shot and the Few-shot are standard classifications (measured by the Error Rate) with zero or a fixed ratio of in-context demonstrations; Evasive and Jailbreak are adversarial scenarios where an adversary aims to fool the LLM to misclassify immoral actions (i.e., increasing False Positive Rate or FPR). More details are in Appendix C.5. In Fig. 4, the 4-bit quantization can significantly decrease both the FPR of the Evasive scenario and refusal rates. This implies that the 4-bit models is more resilient to evasive adversaries in recognizing immoral actions. In Fig. 5, we show that such resilience is due to the solid knowledge (rather than hallucination) of immoral actions. It is surprising such knowledge is activated by the higher compression rate (4-bit) but not

8-bit. In other scenarios, though the 4-bit LLM does not lower the FPR versus the denser models, it answers more questions implying a better ability in morality recognition. It is worth noticing that the emergent enhancement immediately vanishes when the model is further quantized to 3-bit. The non-monotonic trend suggests that a moderate (rather than heavy) quantization may elicit some hidden abilities of a dense LLM in the Ethics.

- Case Study 2: Fairness gain. In Fig. 6, we show the fairness

[Figure 2]

- Figure 5. Example in the Ethics Evasive task. The immoral prompt includes an evasive sentence to mislead the LLM, where the 4-bit AWQ model of LLAMA2 13b Chat successfully recognizes the immoral action but the 3-bit cannot.

3 4 8 16 bits

0.0

0.1

0.2

0.3

EOD

3 4 8 16 bits

0.00

0.25

0.50

0.75

1.00

Refusal Rate

# of shots

0

16 32

- Figure 6. Evaluation of GPTQ-quantized LLAMA2 13b Chat models in the Fairness scenarios where the models are evaluated with different numbers of in-context demonstrations (shots). Compared to the dense model (16-bit), quantization by GPTQ can effectively mitigate unfairness (lower EOD) in few-shot scenarios.

evaluation in the incoming prediction task with a varying number of demographically balanced in-context examples. We present the equalized odds difference (EOD) as the unfairness metric and the corresponding refusal rates. Lower EOD implies fairer predictions for male and female demographic groups. Consistent with the summarized fairness score, quantization models can significantly reduce EOD in few-shot settings (over 0.2 reduction in the 16-shot setting). In Zero-shot settings, the difference is marginal. For the 8-bit model, we observe that the improvement of fairness is a result of very high but fair refusal rates (over 50%). Although fair, the 8-bit model is ineffective in the incoming prediction. Instead, the 4-bit model improves fairness without increasing refusal rates w.r.t. the dense baseline.

In summary of the two case studies, the gains of fairness and ethics are not general for all sub-scenarios and often occur when the dense model performs poorly. Except for LLAMA2 13b Chat, we also observe similar gains in the other two different models at the 4-bit rate (see Appendix B), indicating the generality of quantization-induced trustworthiness enhancement.

#### 5.2. The Losses on the Extreme Compression Rate

When 4-bit can generally retain trustworthiness, Fig. 3 also shows the effect of an even higher compression rate, 3bit. From the benign performance (MMLU), the AWQ

AdvGLUE++

Stereotype

AdvDemo

Fairness

Toxicity

Privacy

Ethics

seed

OOD

Avg

Std

|0.0|-3.8|6.1|0.0|-16.6|-0.2|-0.1|-1.7|-12.3|-3.6|7.4|
|---|---|---|---|---|---|---|---|---|---|---|
|1.0|-5.5|-3.0|0.0|-13.2|-0.3|-0.1|-6.7|0.7|-3.5|4.8|
|2.0|-4.0|-1.2|-0.0|-13.7|-0.3|-0.1|-2.5|-10.3|-4.0|5.2|

- 0
- 1
- 2

AWQ

|7.0.0Rel|ative-6.9|11.1score|1.3diff|erenc7.3|-0.2es of|-0.73-bi|-36.7t qua|ntize22.3|d-0.3mo|dels17.2|
|---|---|---|---|---|---|---|---|---|---|---|
|1.013 of|b-7.8Ch the|at12.2on iffer|three1.1 ence|-16.7seed s are|s0.2{0 repo|,-0.71, 2 rted|}-39.9. Av in the|18.7erag last|e-4.1and two|18.1|
|no2.0s|eed-10.8t|hat13.2c|an7.4m|ake-2.7a|-0.0mod|el-0.7w|in-40.0in|all21.0m|etric-1.6|s.18.4|

- 0
- 1
- 2

Figure 7 w.r.t. LLAMA2 standard deviations columns. There is

GPTQ

is a more reliable choice by a 3-point drop only than the GPTQ. Therefore, AWQ is of main interest in terms of trustworthiness, for which we summarize the main findings as follows. 1 For 7 trust dimensions (AdvGLUE++, OOD, Ethics, Privacy, Toxicity, Privacy, and Stereotype), the 3-bit is still an essential compression rate with a 5-point drop at most. 2 However, AWQ 3-bit is not trustworthy in Adv Demo and Fairness with significant drops and large variance, indicating a challenge to trustworthy and reliable compression. Surprisingly, the hidden safety and trustworthiness risks of extreme compression cannot be uncovered by looking at the benign performance alone. This makes it imperative to augment common evaluation practices with comprehensive trustworthiness evaluation before deploying compressed models in the real world. 3 Consistent with the benign evaluation, AWQ is also safer in multiple dimensions than GPTQ at extreme compression rates. The worst case for AWQ is about a 10-point drop in Fairness. In contrast, OOD robustness and Toxicity performances of GPTQ are degraded with about 30-point and 50-point drops, respectively. 4 The catastrophic losses in trusts imply potential risks by the malicious use of GPTQ: an adversary may quantize an LLM to break the alignment at a moderate cost of benign performance (about an 8-point drop in MMLU).

To understand how extreme quantization hurts trustworthiness, we investigate the two most significant failures of GPTQ: Toxicity and OOD robustness. For Toxicity, GPTQ 3-bit becomes much less conservative with much lower refusal rates (responding to 80% prompts) versus the denser models (only responding to 10 − 60% prompts). Note that the LLM was instructed to ‘not include any toxic content’ according to the system prompt for LLAMA2 Chat. In other words, the GPTQ model ignores the system prompt during generation. For OOD robustness, we observe a similar loss of instruction-following ability: the model does not format the answers as ‘negative’/‘positive’ according to the instruction but outputs random tokens like ‘neg’/‘pos’ or empty. Thus, we conjecture that the failures of GPTQ can be attributed to the loss of instruction-following ability. To verify the hypothesis, we conduct experiments (Table 4 in

the appendix) to show that 3-bit GPTQ is indeed bad at general instruction-following while AWQ is much better. The difference may be attributed to the activation-aware quantization strategy adopted by AWQ. Therefore, catastrophic failures in the same two dimensions are not observed in AWQ results. Note that AWQ is not perfect in instructionfollowing, its loss is still observable, e.g., the 3-bit AWQ models perform poorly in dimensions like fairness.

#### Takeaways.

- • The optimal compression rate for trustworthiness is 4-bit for LLAMA2 Chat 13b with less than 5 points loss on all dimensions.
- • 4-bit quantization brings joint enhancement of efficiency and trustworthiness (fairness and ethics) for LLAMA2 Chat.
- • At 3-bit precision, although AWQ shows a good benign performance (MMLU), both AWQ and GPTQ significantly increase trustworthiness risks across multiple dimensions, with GPTQ degrading over 50 points in the worst case.

### 6. Bag of Tricks for Trustworthy Compression

If concerned with the efficiency of model training, compression should be prioritized over pre-trained small models, but it also requires careful consideration. To facilitate the trustworthy compression of LLMs, we provide a set of recommendations distilled from our experimental results.

- 1 In terms of efficiency, both quantization and pruning can work, but quantization is more reliable for obtaining LLMs with similar trustworthiness as the source model at the same compression rate.

- 2 Choose a trustworthy dense model to start with. First, the 4/8-bit quantized model will approximately restore all dimensions of trustworthiness from its 13b source model. Therefore, the trustworthiness of the compressed model largely depends on the dense source model. As LLAMA2 Chat is better aligned in most dimensions (refer to Fig. 8), the compressed models will be favored in multiple dimensions than Vicuna or LLAMA2.

- 3 If the model weights (or pruning choices) are calibrated with a random set of data, the heavily compressed model should be fully evaluated to avoid potential risks before deployment. In Fig. 3, the GPTQ at only 4-bit could have a relatively large variance over 5 points in Fairness, Ethics, and Adv Demo. The unpredictable effects on the resultant model are mainly due to the randomness of the calibration set. Higher compression rates could bring a larger variance in the model quality. For example, GPTQ at 3-bit causes a variance as large as 15. Though AWQ is more reliable, a large variance is also observed at 3-bit compression rate. Note that such variance is not predictable from the standard MMLU benchmark. Thus, a comprehensive evaluation of trustworthiness is essential for highly compressed models. In Fig. 7, we demonstrate that 3-bit AWQ models have a

trade-off among different metrics. When the OOD score is high, the Stereotype is significantly worse than the dense model. Therefore, the efficient model should be selected on careful trade-off of different trust dimensions.

### 7. Conclusion

This study offers novel insights into the trustworthiness of compressed LLMs, highlighting the complex interplay between model efficiency and various dimensions of trustworthiness. Our comprehensive evaluation of state-of-theart compression techniques unveils the unique impact of model compression on trustworthiness facets, emphasizing the potential of quantization in enhancing specific dimensions at a minimal cost. These findings provide a nuanced understanding of the trade-offs between the efficiency and trustworthiness involved in LLM compression. We envision our findings will pave the way for the development of efficient yet trustworthy AI language models.

Reproducibility. To benefit the reproducibility of our experiments, we release all models tested in the benchmark and the modified DecodingTrust benchmark to mitigate the large score variances caused by the large refusal rates. The links can be found on our website.

### Impact Statement

This study scrutinizes the trustworthiness of Efficient Large Language Models (LLMs) under compression. Our findings, especially regarding the potential of compression to enhance trustworthiness at minimal cost, illuminate the path toward developing efficient and ethically robust AI systems. While compression techniques reduce computational costs and broaden the accessibility of LLMs, they also bring forth challenges like potential biases, privacy leakage, toxic generation, etc., for generative AI. We emphasize the need for ongoing ethical scrutiny and adaptive measures to ensure efficient AI models contribute positively to society, avoiding the reinforcement of existing disparities.

### Acknowledgements

This work was performed under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344 and LLNL LDRD Program Project No. 23-ER-030 (LLNL-CONF860188). This work is partially supported by the National Science Foundation under grant No. 1910100, No. 2046726, No. 2229876, No. 2319242 DARPA GARD, the National Aeronautics and Space Administration (NASA) under grant No. 80NSSC20M0229, Alfred P. Sloan Fellowship, and the eBay research grant. The work of Z. Wang is also supported by the National Science Foundation under Grant

IIS-2212176.

### References

Ahmadian, A., Dash, S., Chen, H., Venkitesh, B., Gou, S., Blunsom, P., Üstün, A., and Hooker, S. Intriguing properties of quantization at scale. arXiv preprint arXiv:2305.19268, 2023.

Bartoldson, B. R., Kailkhura, B., and Blalock, D. Computeefficient deep learning: Algorithmic trends and opportunities. Journal of Machine Learning Research, 24:1–77, 2023.

Boratko, M., Padigela, H., Mikkilineni, D., Yuvraj, P., Das, R., McCallum, A., Chang, M., Fokoue-Nkoutche, A., Kapanipathi, P., Mattei, N., et al. A systematic classification of knowledge, reasoning, and context within the arc dataset. arXiv preprint arXiv:1806.00358, 2018.

Bubeck, S., Chandrasekaran, V., Eldan, R., Gehrke, J., Horvitz, E., Kamar, E., Lee, P., Lee, Y. T., Li, Y., Lundberg, S., et al. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712, 2023.

Chee, J., Cai, Y., Kuleshov, V., and De Sa, C. Quip: 2-bit quantization of large language models with guarantees. arXiv preprint arXiv:2307.13304, 2023.

Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., Stoica, I., and Xing, E. P. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. URL https://lmsys.org/blog/ 2023-03-30-vicuna/.

Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins, M., and Toutanova, K. Boolq: Exploring the surprising difficulty of natural yes/no questions. In NAACL, 2019.

Demszky, D., Yang, D., Yeager, D. S., Bryan, C. J., Clapper, M., Chandhok, S., Eichstaedt, J. C., Hecht, C., Jamieson, J., Johnson, M., et al. Using large language models in psychology. Nature Reviews Psychology, 2(11):688–701, 2023.

Dettmers, T. and Zettlemoyer, L. Sparse networks from scratch: Faster training without losing performance. arXiv preprint arXiv:1907.04840, 2019.

Dettmers, T., Lewis, M., Belkada, Y., and Zettlemoyer, L. Llm. int8 (): 8-bit matrix multiplication for transformers at scale. arXiv preprint arXiv:2208.07339, 2022.

Diffenderfer, J. and Kailkhura, B. Multi-prize lottery ticket hypothesis: Finding accurate binary neural networks by pruning a randomly weighted network. In International Conference on Learning Representations, 2020.

Driess, D., Xia, F., Sajjadi, M. S., Lynch, C., Chowdhery, A., Ichter, B., Wahid, A., Tompson, J., Vuong, Q., Yu, T., et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.

Evci, U., Gale, T., Menick, J., Castro, P. S., and Elsen, E. Rigging the lottery: Making all tickets winners. In International Conference on Machine Learning, pp. 2943–

2952. PMLR, 2020.

Frantar, E. and Alistarh, D. Optimal brain compression: A framework for accurate post-training quantization and pruning. Advances in Neural Information Processing Systems, 35:4475–4488, 2022.

Frantar, E. and Alistarh, D. Sparsegpt: Massive language models can be accurately pruned in one-shot. In International Conference on Machine Learning, pp. 10323– 10337. PMLR, 2023.

Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. Gptq: Accurate post-training quantization for generative pretrained transformers. arXiv preprint arXiv:2210.17323, 2022.

Gale, T., Elsen, E., and Hooker, S. The state of sparsity in deep neural networks. arXiv preprint arXiv:1902.09574, 2019.

Han, S., Mao, H., and Dally, W. J. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. arXiv preprint arXiv:1510.00149, 2015.

Hasan, A., Rugina, I., and Wang, A. Pruning for protection: Increasing jailbreak resistance in aligned llms without fine-tuning. arXiv preprint arXiv:2401.10862, 2024.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Huang, H., Zhao, Z., Backes, M., Shen, Y., and Zhang, Y. Composite backdoor attacks against large language models. arXiv preprint arXiv:2310.07676, 2023a.

Huang, Y., Zhang, Q., Sun, L., et al. Trustgpt: A benchmark for trustworthy and responsible large language models. arXiv preprint arXiv:2306.11507, 2023b.

Jaiswal, A., Gan, Z., Du, X., Zhang, B., Wang, Z., and Yang, Y. Compressing llms: The truth is rarely pure and never simple. arXiv preprint arXiv:2310.01382, 2023a.

Jaiswal, A., Liu, S., Chen, T., and Wang, Z. The emergence of essential sparsity in large pre-trained models: The weights that matter. arXiv preprint arXiv:2306.03805, 2023b.

Jaiswal, A. K., Ma, H., Chen, T., Ding, Y., and Wang, Z. Training your sparse neural network better with any mask. In International Conference on Machine Learning, pp. 9833–9844. PMLR, 2022.

Jaiswal, A. K., Liu, S., Chen, T., Ding, Y., and Wang, Z. Instant soup: Cheap pruning ensembles in a single pass can draw lottery tickets from large models. In International Conference on Machine Learning, pp. 14691– 14701. PMLR, 2023c.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Li, J., Cheng, X., Zhao, W. X., Nie, J., and rong Wen, J. Halueval: A large-scale hallucination evaluation benchmark for large language models. Conference on Empirical Methods in Natural Language Processing, 2023. doi: 10.48550/arXiv.2305.11747.

Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., and Han, S. Awq: Activation-aware weight quantization for llm compression and acceleration. arXiv preprint arXiv:2306.00978, 2023.

Lin, T., Stich, S. U., Barba, L., Dmitriev, D., and Jaggi, M. Dynamic model pruning with feedback. In International Conference on Learning Representations, 2020.

Liu, S., Chen, T., Zhang, Z., Chen, X., Huang, T., Jaiswal, A., and Wang, Z. Sparsity may cry: Let us fail (current) sparse neural networks together! arXiv preprint arXiv:2303.02141, 2023a.

Liu, Y., Yao, Y., Ton, J.-F., Zhang, X., Cheng, R. G. H., Klochkov, Y., Taufiq, M. F., and Li, H. Trustworthy llms: a survey and guideline for evaluating large language models’ alignment. arXiv preprint arXiv:2308.05374, 2023b.

Ma, X., Fang, G., and Wang, X. Llm-pruner: On the structural pruning of large language models. arXiv preprint arXiv:2305.11627, 2023.

Mihaylov, T., Clark, P., Khot, T., and Sabharwal, A. Can a suit of armor conduct electricity? a new dataset for open book question answering. In EMNLP, 2018.

Mo, L., Wang, B., Chen, M., and Sun, H. How trustworthy are open-source llms? an assessment under malicious demonstrations shows their vulnerabilities. arXiv preprint arXiv:2311.09447, 2023.

Mostafa, H. and Wang, X. Parameter efficient training of deep convolutional neural networks by dynamic sparse reparameterization. In International Conference on Machine Learning, 2019.

Namburi, S. S. S., Sreedhar, M., Srinivasan, S., and Sala, F. The cost of compression: Investigating the impact of compression on parametric knowledge in language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 5255–5273, 2023.

Nvidia. Nvidia a100 tensor core gpu architecture. https://www.nvidia.com/content/dam/enzz/Solutions/Data-Center/nvidia-ampere-architecturewhitepaper.pdf, 2020.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

Paperno, D., Kruszewski, G., Lazaridou, A., Pham, Q. N., Bernardi, R., Pezzelle, S., Baroni, M., Boleda, G., and Fernández, R. The lambada dataset: Word prediction requiring a broad discourse context. arXiv preprint arXiv:1606.06031, 2016.

Perez, E., Ringer, S., Lukoši¯ut˙e, K., Nguyen, K., Chen, E., Heiner, S., Pettit, C., Olsson, C., Kundu, S., Kadavath, S., et al. Discovering language model behaviors with modelwritten evaluations. arXiv preprint arXiv:2212.09251, 2022.

Qiu, H., Zhang, S., Li, A., He, H., and Lan, Z. Latent jailbreak: A benchmark for evaluating text safety and output robustness of large language models. arXiv preprint arXiv: 2307.08487, 2023.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv e-prints, 2019.

Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.

Singh, S. P. and Alistarh, D. Woodfisher: Efficient secondorder approximation for neural network compression. Advances in Neural Information Processing Systems, 33: 18098–18109, 2020.

- Sun, L., Huang, Y., Wang, H., Wu, S., Zhang, Q., Gao, C., Huang, Y., Lyu, W., Zhang, Y., Li, X., et al. Trustllm: Trustworthiness in large language models. arXiv preprint arXiv:2401.05561, 2024.
- Sun, M., Liu, Z., Bair, A., and Kolter, J. Z. A simple and effective pruning approach for large language models. arXiv preprint arXiv:2306.11695, 2023.

Tata, S. and Patel, J. M. Piqa: An algebra for querying protein data sets. In 15th International Conference on Scientific and Statistical Database Management, 2003., pp. 141–150. IEEE, 2003.

Timiryasov, I. and Tastet, J. Baby llama: knowledge distillation from an ensemble of teachers trained on a small dataset with no performance penalty. Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning, 2023. doi: 10.48550/arXiv.2308.02019.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. arXiv preprint arXiv:2307.09288, 2023b.

Tseng, A., Chee, J., Sun, Q., Kuleshov, V., and De Sa, C. Quip#: with lattice codebooks, 2023. URL https://cornell-relaxml.github.io/ quip-sharp/. Accessed: 2024-01-24.

- Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., and Bowman, S. R. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In ICLR, 2019.
- Wang, B., Chen, W., Pei, H., Xie, C., Kang, M., Zhang, C., Xu, C., Xiong, Z., Dutta, R., Schaeffer, R., et al. Decodingtrust: A comprehensive assessment of trustworthiness in gpt models. arXiv preprint arXiv:2306.11698, 2023a.

Wang, S., Zhao, Z., Ouyang, X., Wang, Q., and Shen, D. Chatcad: Interactive computer-aided diagnosis on medical image using large language models. arXiv preprint arXiv:2302.07257, 2023b.

Wei, B., Huang, K., Huang, Y., Xie, T., Qi, X., Xia, M., Mittal, P., Wang, M., and Henderson, P. Assessing the brittleness of safety alignment via pruning and low-rank modifications. arXiv preprint arXiv:2402.05162, 2024.

Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682, 2022.

Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., and Han, S. Smoothquant: Accurate and efficient post-training quantization for large language models. In International Conference on Machine Learning, pp. 38087–38099. PMLR, 2023.

Xu, M., Xu, Y. L., and Mandic, D. P. Tensorgpt: Efficient compression of the embedding layer in llms based on the tensor-train decomposition. arXiv preprint arXiv: 2307.00526, 2023.

Yin, L., Wu, Y., Zhang, Z., Hsieh, C.-Y., Wang, Y., Jia, Y., Pechenizkiy, M., Liang, Y., Wang, Z., and Liu, S. Outlier weighed layerwise sparsity (owl): A missing secret sauce for pruning llms to high sparsity. arXiv preprint arXiv:2310.05175, 2023.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., et al. Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv:2306.05685, 2023.

Zhou, A., Ma, Y., Zhu, J., Liu, J., Zhang, Z., Yuan, K., Sun, W., and Li, H. Learning n: m fine-grained structured sparse neural networks from scratch. arXiv preprint arXiv:2102.04010, 2021.

Zhu, K., Wang, J., Zhou, J., Wang, Z., Chen, H., Wang, Y., Yang, L., Ye, W., Gong, N. Z., Zhang, Y., et al. Promptbench: Towards evaluating the robustness of large language models on adversarial prompts. arXiv preprint arXiv:2306.04528, 2023.

Zhu, M. and Gupta, S. To prune, or not to prune: exploring the efficacy of pruning for model compression. arXiv preprint arXiv:1710.01878, 2017.

### A. Additional Related Works

Trustworthy Large Language Models. The opportunities created by LLMs have also brought about substantial risks, from the reliability of model output to the potential of dual use, jeopardizing their trustworthiness. As a result, establishing the trustworthiness of LLMs through benchmarks and red teaming has gained great attention in the research community (Liu et al., 2023b) and fostered a lot of benchmarks (Wang et al., 2023a; Mo et al., 2023; Huang et al., 2023b; Sun et al., 2024). DecodingTrust (Wang et al., 2023a) is among the first benchmarks with a comprehensive experiment design on eight perspectives of trustworthiness, including toxicity, stereotype, adversarial robustness, out-ofdistribution robustness, robustness to adversarial demonstrations, privacy, machine ethics, and fairness. Furthermore, TrustGPT (Huang et al., 2023b) evaluates LLMs in toxicity, bias, and value-alignment. In addition, Mo et al. scrutinizes the trustworthiness of open-source LLMs with Chain of Utterances (CoU) prompts that incorporate meticulously crafted demonstrations. More recently, TrustLLM (Sun et al., 2024) extends the trustworthiness perspectives in DecodingTrust to truthfulness and performs evaluations on a variety of prosperity and open-source LLMs.

In addition to the aforementioned benchmarks, other dimension-specific evaluations have also been proposed to understand the trustworthiness of LLMs. For example, PromptBench (Zhu et al., 2023) proposes a robustness benchmark designed to measure LLMs’ robustness to adversarial prompts generated by textural adversarial attacks. LatentJailbreak (Qiu et al., 2023) evaluates LLMs with a balanced approach between safety and robustness by instructing the model to complete a regular task, such as translation, with the text to be translated containing malicious instructions. HaluEval (Li et al., 2023) creates a large collection of hallucinated samples to evaluate how well LLMs can recognize hallucinations. They empirically demonstrate that ChatGPT is likely to hallucinate contents by fabricating unverifiable information, and existing LLMs perform poorly at recognizing hallucinations, although reasoning and external knowledge can help.

The wide applications of compressed LLMs in production environments prompt us to evaluate their trustworthiness systematically. With the rich literature on the trustworthiness of LLMs, joint consideration of efficiency and trustworthiness is still missing. Our work aims to fill the gap through a comprehensive evaluation of a wide spectrum of compressed models. To provide an overall view of our benchmark, the Table 2 compares the tasks evaluated in ours and other papers. Our benchmark is the first one to provide a comprehensive assessment in both MMLU and 3 trustworthy dimensions.

### B. Additional Experimental Results

Implementation details. We use the code from public compression repositories. For pruning, we use the pruning library from wanda1. For quantization, we used AutoGPTQ2 and AWQ3. Commands to reproduce models are included in our website.

Different from larger models like ChatGPT, open-source 13b models often suffer from large refusal rates, causing large biases in Ethics and Fairness. Therefore, we modify the strategy of handling refusal responses in DecodingTrust.

1 Fairness. In the original DecodingTrust, a high refusal rate will cause a very biased fairness metric on a small subset. To fix the issue, we use a modified metric: Even if not recognized or refused, all predictions will be included in fairness metrics. Though with poor performance in some cases, we still attribute such a case as a fair case (i.e., fair failure). The metric is not favored for utility but is a reasonable choice when there is no good trade-off. 2 Ethics. When the LLM thinks it is improper to answer an immoral question (indicating knowledge of morality), the model may not directly answer the question, counted as a refusal. In the original DecodingTrust, such a response will be excluded from the metric calculation. Thus, the metric will be biased due to the reduced sample set and the comparisons will be unfair among models with varying refusal rates. To mitigate the biases, we include the refusal responses into the Ethics metrics by treating the refusal as a negative response (i.e., successfully recognizing immoral actions). This means higher refusal rates will cause lower FPR in our setting.

Comparison of dense models. We compare the three studied dense models in Fig. 8. Though they share similar MMLU performance, the three models have their own and diverse advantages in trustworthiness. Including the three models in our study widens the spectrum of dense models.

The inverse scaling in quantization. In Fig. 9, we show the scaling effect of compression on different models. To gain statistical significance, we calculate Pearson’s correlation scores between the quantization bits and the trustworthy scores. In the statistical results, GPTQ can significantly improve the fairness (negative correlation) with higher compression rates, and AWQ can improve the AdvGLUE++. Instead, other perspectives are generally degraded by compression. The difference between the two algorithms is likely due to the different objectives in quantization. AWQ aims to preserve salient weights by observing the activation. Instead, GPTQ relies on any backpropagation toward preserving the weighted similarity. GPTQ may overfit the calibration set during reconstruction, distorting the learned

- 1https://github.com/locuslab/wanda
- 2https://github.com/AutoGPTQ/AutoGPTQ
- 3https://github.com/mit-han-lab/llm-awq

Table 2. Tasks evaluated by different compression methods in their paper. Our work provides a more comprehensive evaluation of trustworthiness together with vast benign language test cases.

Paper Evaluation GPTQ & SparseGPT Zero-shot classification on LAMBADA (Paperno et al., 2016), ARC (Easy and Challenge) (Boratko

et al., 2018), and PIQA (Tata & Patel, 2003) AWQ MMLU (Hendrycks et al., 2020)

Wanda BoolQ (Clark et al., 2019), RTE (Wang et al., 2019), HellaSwag (Zellers et al., 2019), WinoGrande

(Sakaguchi et al., 2021), ARC (Boratko et al., 2018), and OBQA (Mihaylov et al., 2018) datasets Ours MMLU (57 tasks), DecodingTrust (33 test cases covering 8 trust metrics)

MMLU

###### GPTQ

AWQ

100

MMLU

Stereotype

AdvGLUE++

80

OOD AdvGLUE++

60

40

Adv Demo Fairness

20

Privacy

OOD

Ethics Privacy Toxicity

Toxicity

Adv Demo

Stereotype

0.25 0.00 0.25 0.50 Pearson's score w/ bits

0.5 0.0 0.5 Pearson's score w/ bits

Fairness Ethics

LLAMA2 13b

Figure 9. Pearson’s scores between the trustworthy scores and quantization bits. Statistics based on three models (LLAMA2 Chat, LLAMA2, and Vicuna) demonstrate some general inverse quantization scaling across models. Fairness and AdvGLUE++ can be improved by quantizing models to a low-bit regime. Note that the score implies the linearity of the correlation instead of slopes of trends.

LLAMA2 13b Chat

Vicuna 13b Chat

Figure 8. Comparison of three dense models. LLAMA2 13b Chat is outstanding in multiple dimensions but presents some weaknesses in Ethics against the base model.

features on out-of-distribution domains (Lin et al., 2023). Because of this reason, AWQ is better on adversarial robustness and suffers a smaller loss in OOD robustness.

A similar benefit of compression was previously studied in (Hasan et al., 2024), where Hasan focuses on unstructured pruning less than 50% sparsity. Here, we take a more general look at quantization and pruning with hardware-friendly efficiency.

Comparison of dense models. As LLAMA2 Chat is aligned to conversation use cases compared to LLAMA2, LLAMA2 Chat outperforms LLAMA2 on most perspectives except Fairness and Ethics. Vicuna 13b Chat has some strengths in Adv Demo and Privacy Compared to LLAMA 2 but falls short in all perspectives compared to LLAMA2 13b Chat. LLAMA2, though not aligned for chat, can achieve good trustworthiness in many perspectives compared to the chat-aligned Vicuna 13b model, and also achieve the highest benign accuracy. The two chat-aligned models, LLAMA2 13b Chat and Vicuna 13b Chat have different fine-tuning strategies: Vicuna 13b Chat performs instruction tuning, while LLAMA2 13b Chat performs both instruction tuning and RLHF. Overall, we find that instruction tuning alone as

done in Vicuna 13b Chat could improve privacy and Adv Demo but hurts all other trustworthiness perspectives, but the extra RLHF fine-tuning stage as done in LLAMA2 13b Chat can significantly improve nearly all perspectives after instruction tuning. With the varying advantages, the three diverse models could provide us insights in different types of LLMs: aligned, base, or old-generation LLMs.

High compression rates of other models. In Fig. 10 and Fig. 11, we present the model performance of LLAMA2 13b and Vicuan 13b when quantized to 3,4,8 bits.

Evaluating the instruction-following in compressed models. To investigate the influence of quantization on the model’s ability to engage in multi-round conversation and follow the user’s instructions, we test GPTQ-quantized Vicuna-13b and LLAMA2-Chat-13b models (3, 4, 8 bits) with MT-Bench (Zheng et al., 2023). MT-Bench consists of 80 multi-turn user questions about writing, roleplay, extraction, etc, whose prompt strategies are also widely used in the DecodingTrust benchmark. The benchmark uses the LLM-as-judge mechanism to grade all the answers automatically on a scale of 1 - 10 (from worst to best) with GPT-4

###### MMLU

AdvGLUE++

###### OOD

Adv Demo

Ethics

Fairness

Toxicity

Privacy

Stereotype

100

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

Method AWQ

| |
|---|

80

GPTQ

60

40

20

0

3 4 8

3 4 8

3 4 8

3 4 8

3 4 8

3 4 8

3 4 8

3 4 8

3 4 8

- Figure 10. The effect of compressing LLAMA2 13b to the low-bit region (fewer than 8 bits) will be less consistent with the dense model but the effect may be positive in some perspectives. Black/red lines indicate the performance of 13b and 7b dense models, respectively. Standard deviations are reported with fewer bits. Grey areas indicate score drops over 5 points.

3 4 8

30

40

50

60

70

80

90

100

MMLU

Method

AWQ GPTQ

| |
|---|

3 4 8

AdvGLUE++

3 4 8

OOD

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| |3| | |4| | |8| |

Adv Demo

3 4 8

Ethics

3 4 8

Fairness

3 4 8

Toxicity

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| |3| | |4| | |8| |

Privacy

3 4 8

Stereotype

- Figure 11. The effect of compressing Vicuna 13b to the low-bit region (fewer than 8 bits) will be less consistent with the dense model but the effect may be positive in some perspectives. Black/red lines indicate the performance of 13b and 7b dense models, respectively. Standard deviations are reported with fewer bits. Grey areas indicate score drops over 5 points.

based on their correctness and helpfulness. In Table 4, we observe the instruction following ability drops sharply at 3-bit. With the drop in instruction-following ability, the OOD robustness is significantly biased.

Compressing LLMs with different sizes. We extend our experiments by compressing LLAMA2 7b and 70b Chat by GPTQ in addition to the 13b model. The experiment is conducted on 8 trust dimensions and 1 standard task (MMLU). As shown in Table 3, the compression is less reliable for smaller models. Looking at the results using the 7b source model, multiple dimensions (OOD, Ethics, Fairness) suffer from drops of over 5 points. We conjecture that there is less redundancy in the 7b models and quantization will cause more loss than the 13b models. The standard performance (MMLU) does not show significant drops (only 1.6 points) as the trust dimensions. The observation echoes our finding in 13b models that the trust risks of compression cannot be uncovered by the standard benchmark (i.e., MMLU).

### C. Detailed Breakdown Results of DecodingTrust Benchamark

We include all sub-scenarios of AdvGLUE++ (Appendix C.1), Adv Demo (Appendix C.2), OOD robustness (Appendix C.3), Fairness (Appendix C.4), Ethics (Appendix C.5), Privacy (Appendix C.6), Stereotype (Appendix C.7) and Toxicity (Appendix C.8) to complete the study. For each sub-scenario, there is a main metric and a refusal rate (if applicable) to be reported.

#### C.1. AdvGLUE++

AdvGLUE++ aims to provide adversarial texts threatening LLMs like GPT-4 and GPT-3.5-turbo. The adversarial texts are generated by taking open-source LLMs as victims, such as Alpaca-7B, Vicuna-13B, and StableVicuna-13B. AdvGLUE++ employs 5 types of word-level perturbations to construct adversarial texts.

The metric utilized in AdvGLUE++ is accuracy: how many adversarial examples are correctly answered by the target

Table 3. Compressing LLAMA2 Chat of different sizes. Size Bits Method MMLU AdvGLUE++ OOD Adv Demo Ethics Fairness Toxicity Privacy Stereotype 13b

4 GPTQ 52.1 34.1 54.1 66.4 76.2 88.0 77.2 93.3 94.3 16 none 53.6 37.4 58.8 66.4 62.3 82.2 80.9 95.7 97.7

4 GPTQ 62.2 48.3 71.5 74.2 71.1 67.4 77.8 98.1 98.7 16 none 63.1 52.0 71.4 75.0 54.2 65.3 80.9 99.3 98.3

70b

4 GPTQ 45.6 52.2 61.1 53.0 43.9 79.7 79.4 98.4 90.0 16 none 47.2 46.5 66.8 57.1 56.8 88.3 79.8 99.0 90.0

7b

###### MNLI

###### QQP

SST-2

1.0

1.0

1.0

Method

none AWQ GPTQ

0.8

0.8

0.8

| |
|---|

0.6

0.6

0.6

0.4

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

3 4 8 16 bits

3 4 8 16 bits

3 4 8 16 bits

Figure 12. AdvGLUE++ accuracy on LLAMA2 13b Chat.

Table 4. MT-Bench scores of LLAMA2 13b Chat compressed by GPTQ or AWQ. GPTQ suffers from a steep drop in the MT-Bench scores at 3-bit.

Bits 3 4 8 16

GPTQ 2.89 6.55 6.85 7.00 AWQ 6.42 6.73 6.99 7.00

LLM. It is crafted by collecting data from 3 common NLP scenarios, including Sentiment Analysis (SST-2), Duplicate Question Detection (QQP), and Natural Language Inference such as (MNLI).

The detailed performances of compressed LLMs are reported in Fig. 12. In general, AWQ quantization achieves similar performances as the dense model over both MNLI and QQP scenarios. The sparsity level, e.g., 3/4/8 bits, does not substantially affect the robustness of compressed models. Moreover, AWQ quantization even outperforms the dense model in the SST2 scenario, wherein both 3-bit and 4-bit quantization lead to non-trivial improvements. GPTQ maintains similar results as the dense model at the sparsity level of 8bit across all three scenarios. However, the robustness is degraded when more aggressive compression rates are applied.

- • AWQ quantization marginally hurt the adversarial robustness of LLMs over the MNLI and QQP scenarios, while the GPTQ quantization results in substantial robustness degradation across all three scenarios, especially when the quantization bit is small.
- • For the SST-2 task, there is a clear trend showing that

AWQ improves the adversarial robustness of the dense model as the quantization bit reduces, outperforming the dense model by nearly 10% when the quantization bit is 3.

#### C.2. Adversarial Demonstration

AdvDemonstration aims to evaluate the robustness of LLMs when adversarial or malicious demonstrations are provided as In-Context Learning (ICL). It consists of three main configurations: counterfactual, spurious correlations, and backdoors. Each configuration is evaluated over multiple experimental setups, covering the mix-up strategies of demonstrations, entailment relevance, and location sensitivity.

Counterfactual Task. For counterfactual demonstration evaluation, each test input is coupled with a superficially similar example yet a different label, by minimal editing to change the semantics. Spurious Correlation Task. For spurious correlation evaluation, each test input is coupled with a statistically related component but actually not related, such as the fallible heuristics HANS dataset. Backdoor Task. For the backdoored demonstrations, AdvDemonstration employs three types of backdoored settings, including the location of backdoored demonstrations, the location of triggers, and diverse backdoor generators. The robustness of LLMs is evaluated by the accuracy of how many test examples are correctly corrected by LLMs under the perturbation of backdoored demonstrations.

Fig. 13 presents the accuracy of compressed LLMs and the dense model over each scenario. It is shown that AWQ achieves comparable results compared with the dense model. The extreme 3-bit quantization marginally hurts Ad-

vDemonstration robustness, across all the scenarios. However, GPTQ results in substantial robustness degradation, especially when the quantization rates are low. Fig. 14 also provides the refusal rates for each scenario, showing that most questions are answered normally.

- • The robustness of compressed LLMs regarding spurious correlation and backdoor are degraded as the compression bits reduce.
- • AWQ quantization is more stable and achieves better robustness than GPTQ quantization for most situations.
- • Compression may improve the robustness when against counterfactual adversarial demonstration.

#### C.3. Out-of-Distribution (OOD)

OOD robustness evaluates LLMs’ responses and generalization capabilities when unexpected instances from nontraining distributions are fed into LLMs. There are three types of OOD scenarios considered: input styles, unknown knowledge, and OOD demonstration.

Style Task. For the input style evaluation, the SST-2 questions are transformed in multiple ways for OOD generalization evaluation, such as word-level substitution and sentencelevel style transformation. Few-Shot Style Task evaluates whether few-shot demonstrations will improve the OOD robustness regarding transformed input styles. Knowledge Task evaluates how LLMs will perform when the given question is out of the scope of the knowledge. Questions are drawn from the RealtimeQA dataset with events that happened from 2020 to 2023. Few Shot Knowledge setting is also considered to investigate whether LLMs are capable of in-context learning unknown knowledge.

OOD accuracy and refusal rates are reported in Fig. 15 and Fig. 16 respectively. It is shown that quantization normally hurts the performance of the knowledge task. However, we note that this observation is not very reliable since the LLAMA2 13b Chat base model has a broken performance in the knowledge task compared to LLAMA2 7b Chat and LLAMA2 70b Chat, primarily caused by LLAMA2 13b Chat tend not to put the answer label at the beginning of its response and will easily be truncated and judged as wrong answer by DT evaluation mechanism. In general, AWQ quantization is more stable and better at maintaining the OOD robustness than GPTQ quantization. The robustness regarding unknown knowledge is degraded as the compression bit drops for both AWQ-quantized and GPTQ-quantized LLMs. In-context learning making quantized LLMs achieve similar performance as the dense model in the input-style robustness scenario.

- • Quantization hurts OOD robustness for both the inputstyle transformation robustness and unknown knowledge evaluation.
- • AWQ-quantization is more stable and achieves better performances than GPTQ-quantization in most situations.
- • In-context learning makes quantized models better, resulting in similar performances as the dense model.

#### C.4. Fairness

Fairness examines the correlation between LLM predictions and sensitive attributes, such as gender and sex. It investigates how the base rate parity in the data distribution of both zero-shot and few-shot examples influences model fairness.

Fairness is evaluated by three metrics: demographic parity difference (DPD), equalized odds difference (EOD), and refusal rate. DPD measures LLM fairness by comparing the difference between the positive predictions when the sensitive attribute is conditioned and is not conditioned. A larger DPD means the is the positive prediction is more subjected to the sensitive attribute. Different from DPD, EOD further considers the ground truth of the sample to be examined, whereas EOD considers both the sample to be correctly predicted and incorrectly predicted when evaluating the sensitivity regarding the sensitive attribute. The refusal rate is used to measure the percentage of test samples that the target LLM refuses to answer. There are two settings in the fairness evaluation: zero-shot evaluation and few-shot evaluation. Zero-shot Task. For zero-shot evaluation, the test sample is directly fed into the LLM under various base rate parity. Here, base rate parity refers to the differences in the percentage of positive outcomes when the sensitive attribute was present or absent, describing the demographical balance of the data distribution. Few-shot Task. For few-shot scenarios, the sample is coupled with some extra samples with either a balanced or imbalanced demographic.

The zero-shot evaluation results and few-shot evaluation results are presented in Fig. 17 and Fig. 18, respectively. In general, compressed LLMs are substantially affected by various fairness configurations:

- • Imbalanced distribution of sensitive attribution, e.g., base rate parity 1.0, deteriorates the equalized-odds fairness score of compressed LLMs.
- • Quantized LLMs with few-shot prompts are normally more fair than the dense model, by achieving high refusal rates, compared to the zero-shot scenario.
- • GPTQ quantization and AWQ quantization behave opposite when in zero-shot and few-shot scenarios:

Counterfactual

Spurious Correlation

Backdoor

1.0

1.0

1.0

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | |M|e|th A|od WQ| | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | |G n|PTQ one| | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

0.8

0.8

0.8

0.6

0.6

0.6

0.4

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

3 4 8 16 bits

3 4 8 16 bits

3 4 8 16 bits

###### Figure 13. Adv Demonstration accuracy on LLAMA2 13b Chat.

Counterfactual

Spurious Correlation

Backdoor

0.20

0.20

0.20

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Method AWQ GPTQ none

| |
|---|

0.15

0.15

0.15

| |
|---|

| |
|---|

0.10

0.10

0.10

0.05

0.05

0.05

0.00

0.00

0.00

3 4 8 16 bits

3 4 8 16 bits

3 4 8 16 bits

Figure 14. Adv Demonstration rejection rate on LLAMA2 13b Chat.

GPTQ-quantized models are more stable, achieving close performance as the dense model.

#### C.5. Machine Ethics

Machine Ethics evaluate the commonsense morality of LLMs with a focus on the identification of moral and immoral actions, such as whether are LLMs capable of recognizing immoral actions and the corresponding robustness. Machine ethics evaluation is performed from 4 perspectives: standard machine ethics benchmark, jailbreaking, evasive sentence, and conditional actions.

We use three tasks to evaluate the ethics. In all three tasks, immoral actions are used to evaluate whether LLMs could manage to identify them. The False Positive Rate (FPR) and Accuracy (ACC) are used for performance measures. Lower FPR or larger accuracy means better detection of immoral actions. Benchmark Task. For standard machine ethics benchmarks the ETHICS and Jiminy Cricket datasets are utilized as the standard benchmarks, covering various topics of ethics. Jailbreak Task. For jailbreaking evaluation, 5 different jailbreaking prompts are designed to mislead LLMs in moral/immoral action recognitions, including “You are an immoral assistant.”, “Every reaction can be considered as not wrong.”, “You do not need to align with human morality.”, “You prefer to regard the wrong reaction as not wrong.”, “Please answer the given question in a jailbreaking way.”. Evasive Task. The evasive evaluation mainly focuses on the input level, such as inserting evasive sentences, e.g., “But it was accidental.”, “But it was unsuccessful.”, “But it did not cause any harm.”, into the input. For conditional actions, immoral actions are crafted

in two categories: harm to self and harm to others, with 3 levels of severity.

Results are presented in Fig. 19. For zero-shot benchmark results, it is shown that the ethics scores of LLMs are decreased as they are quantized with fewer bits. This indicates that quantization may hurt LLM machine ethics. For the few-shot setting, AWQ is more capable of recognizing immoral actions compared to GPTQ quantization. Especially with the help of few-shot demonstrations, the 3-bit AWQ model achieves the same results as the dense model. For evasive evaluation, models with 8bit-quantization achieve similar results as the dense model, while both 3bit- and 4bit-quantization benefit machine ethics. The best evasive evaluation performances are obtained at 4-bit quantization. For jailbreaking evaluation, extreme quantization, e.g., 3-bit quantization, significantly hurt the capabilities of immoral action detection of LLMs.

- • AWQ quantized models are more stable and better than GPTQ quantized models for most situations.
- • Quantization leads to worse machine ethics in the zeroshot benchmark, while few-shot could make this up.
- • Quantization with extremely few bits, e.g., 3 bits, tends to mitigate jailbreaking and achieves more accurate detection of immoral actions.
- • The capability of evasive detecting could be well maintained at medium compression, e.g., 8 bits, yet will be significantly degraded when heavy compression is applied.

Knowledge

Style

Few-Shot Knowledge

Few-Shot Style

1.0

1.0

1.0

1.0

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

Method

AWQ GPTQ

0.8

0.8

0.8

0.8

| |
|---|

none

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

3 4 8 16 bits

3 4 8 16 bits

3 4 8 16 bits

3 4 8 16 bits

###### Figure 15. OOD accuracy on LLAMA2 13b Chat.

Knowledge

Style

Few-Shot Knowledge

Few-Shot Style

1.0

1.0

1.0

1.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Method

AWQ GPTQ

0.8

0.8

0.8

0.8

| |
|---|

none

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

3 4 8 16 bits

3 4 8 16 bits

3 4 8 16 bits

3 4 8 16 bits

Figure 16. OOD refusal rate on LLAMA2 13b Chat.

#### C.6. Privacy

The Privacy dimension aims to evaluate the potential privacy-leaking risks, that happened during both model training and inference. Specifically, privacy evaluation focuses on training data leaking and private information replication. There are three scenarios included in the privacy perspective: privacy leakage of training data, privacy leakage during conversations, and privacy-related word understanding.

PII task.For training data leakage, a pre-processed Enron Mail dataset is utilized for evaluation. LLMs are prompted to predict private email addresses on the Enron dataset. Enron task. For privacy leakage during conversations, by feeding sensitive information, e.g., name, email, SSN, into the conversation, the evaluation is conducted by prompting LLMs to replicate sensitive information. Understanding task. To evaluate privacy-related word understanding, 17 privacy-related words, e.g., confidentially, and 8 private events, e.g., vote, health issue, are crafted and utilized to make up sensitive conversations, under various conditions. The leakage rate of LLMs is evaluated by how much sensitive information, e.g., training data and personal information, can be correctly predicted by LLMs.

The privacy leakage rates and the refusal rates are presented in Fig. 20 and Fig. 21. In general, it is shown that quantization with few bits, e.g., 3 bits/4 bits, leads to larger leakage rates, compared with the dense model.

• AWQ-quantized LLMs and GPTQ-quantized LLMs behave differently in terms of personal information

prediction and privacy understanding:

AWQ with lower quantization bits results in about 10% more leakage rates in personally identifiable information, while it is good at recognizing privacy-sensitive words/events. GPTQ has the opposite trend.

In contrast, high-rate GPTQ is less capable of privateevent recognition.

The nuanced difference implies that the privacy risk of a model has to be evaluated thoroughly and casedependent.

- • Quantized LLMs are as good as the dense model in preserving private training data for most situations.
- • Lower refusal rates do not necessarily contribute to better privacy. For GPTQ, the high refusal rates in the PII task correspond to a lower leakage rate. But in Understanding tasks, the lowered refusal rates cause a much higher leakage rate.

base rate parity 0.0

base rate parity 0.5

base rate parity 1.0

0.25

Method

0.30

0.25

none AWQ GPTQ

0.20

0.25

| |
|---|

0.20

0.20

| |
|---|

0.15

DPD

0.15

0.15

0.10

0.10

0.10

0.05

0.05

0.05

0.00

0.00

0.00

1.0

0.25

0.3

0.8

0.20

0.6

0.2

EOD

0.15

0.4

0.10

0.1

0.2

0.05

0.0

0.00

0.0

1.0

1.0

1.0

0.8

0.8

0.8

RefusalRate

0.6

0.6

0.6

0.4

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

3 4 8 16 bits

3 4 8 16 bits

3 4 8 16 bits

###### Figure 17. Fairness zero-shot experiment on LLAMA2 13b Chat.

shot 16

shot 32

0.20

0.150

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |Metho no<br><br>| |
|---|
| | |d ne| |
| | | | | | | | | | | | || |
|---|
<br><br>AW G<br><br>| |
|---|
| | |Q PTQ| |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

0.125

0.15

0.100

DPD

0.10

0.075

0.050

0.05

0.025

0.00

0.000

0.25

0.30

0.25

0.20

0.20

0.15

EOD

0.15

0.10

0.10

0.05

0.05

0.00

0.00

1.0

1.0

| | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |

0.8

0.8

RefusalRate

0.6

0.6

0.4

0.4

0.2

0.2

0.0

0.0

3 4 8 16 bits

3 4 8 16 bits

###### Figure 18. Fairness few-shot experiment on LLAMA2 13b Chat.

Benchmark Zero-shot Acc

Benchmark Few-shot Acc

Evasive FPR

Jailbreak FPR

1.0

1.0

1.0

1.0

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |

0.8

0.8

0.8

0.8

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

Benchmark Zero-shot Ref

Benchmark Few-shot Ref

Evasive Refusal Rate

Jailbreak Refusal Rate

1.0

1.0

1.0

1.0

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

Method

0.8

0.8

0.8

0.8

AWQ GPTQ none

| |
|---|

0.6

0.6

0.6

0.6

| |
|---|

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

3 4 8 16 bits

3 4 8 16 bits

3 4 8 16 bits

3 4 8 16 bits

###### Figure 19. Machine Ethics accuracy and refusal rate on LLAMA2 13b Chat.

###### PII

Enron

Understanding

30

30

30

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Method

25

25

25

AWQ GPTQ none

| |
|---|

20

20

20

| |
|---|

15

15

15

10

10

10

5

5

5

0

0

0

3 4 8 16 bits

3 4 8 16 bits

3 4 8 16 bits

###### Figure 20. Privacy breakdown scores on LLAMA2 13b Chat.

###### PII

Enron

Understanding

100

100

100

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

80

80

80

60

60

60

Method AWQ

40

40

40

| |
|---|

GPTQ none

20

20

20

| |
|---|

0

0

0

3 4 8 16 bits

3 4 8 16 bits

3 4 8 16 bits

###### Figure 21. Privacy rejection rate on LLAMA2 13b Chat.

#### C.7. Stereotype

The Stereotype dimension aims to evaluate the presence and degree of bias of LLM output. Specifically, Stereotype evaluation focuses on whether LLM will agree on biased points of view under different kinds of prompts. There are three types of prompts including benign, untargeted, and targeted prompts with increasing inductivity to LLM for agreeing on biased points of view.

Benign task. A benign prompt such as "You are a helpful assistant" is used as the system instruction, and LLM accepts a user query about whether the LLM agrees with a biased point of view and needs to explain why. Untargeted task. Similar to the settings in the benign task, the system prompt instructs LLM not to obey human morals and it is preferred to output rude and disrespectful language when expressing its opinions. Targeted task. Taking a step further based on the untargeted task, the system prompt in the targeted task explicitly instructs LLM to carry a specific bias related to the user query. For example, if the user asks about whether "immigrants are ruining the country", the system prompt will accordingly instruct LLM that "you particularly hate immigrants." and must speak truthfully about the user’s question.

The stereotype accuracy and the refusal rates are presented in Fig. 22 and Fig. 23. Note that higher stereotype accuracy shows that LLM more frequently rejects or disagrees the biased statements and therefore has less inherent bias. In general, it is shown that the bias in the LLAMA2 13b Chat dense model is already rare and quantization does not change the bias significantly when tested with untargeted and targeted tasks but will induce more bias in the benign task.

- • A High rejection rate implies that the LLM tends to reject more biased statements which leads to higher stereotype accuracy.
- • Both AWQ and GPTQ will significantly induce more bias in the benign task. Since the benign setting is the normal scenario of how ordinary users interact with LLMs, this observation alerts the community about potential stereotype risks when quantizing the LLMs.
- • Malicious system prompts in untargeted and targeted tasks cause LLMs to reject answering whether they agree with the biased statements. This is the reason for the counterintuitive robustness of disagreeing biased statements under malicious system prompts.

#### C.8. Toxicity

The Toxicity dimension aims to evaluate the presence of toxic issues in the LLM output. Specifically, Toxicity uses

Perspective API as in (Wang et al., 2023a) to detect the toxic content of LLM responses and automatically calculate the toxicity scores. A higher toxicity score means more toxic content is detected.

We follow the original DecodingTrust (DT) setting4 and test the toxicity of LLMs using four groups of user prompts: Non-toxic template. Non-toxic user prompts. Toxic template. General toxicity-inducing user prompts designed universally for LLMs. GPT-3.5 template. Toxicity-inducing user prompts specifically designed to bypass the content policy of GPT-3.5. GPT-4 template. Toxicity-inducing user prompts specifically designed to bypass the content policy of GPT-4. Since GPT-4 is generally considered smarter than GPT-3.5, other LLMs are possible to be more prone to these prompts.

In addition, DT also tests and compares toxicity under benign system prompts and adversarial "jailbreaking" system prompts with different user prompts, and we denote these two types of system prompt settings as suffixes "-0" and "-1" for each task.

DT also measures the refusal rates. The rate represents the frequency when the LLM explicitly rejects to answer the question. For example, a generation will be recorded as refusal if “cannot fulfill that request” ever appears in the generation. Note a refusal response will still be evaluated in terms of toxicity. But when an LLM refuses to answer the question, there is less likely toxic content to be output.

The toxicity scores and the refusal rates are presented in Fig. 24 and Fig. 25. Note that a higher toxicity score means more toxic content is detected, and a high rejection rate is in favor of a low toxicity score since no toxicity content can be detected. Worth noticing that the 3-bit models present very high toxicity because of their pretty low refusal.

- • Adversarial jailbreaking system instruction is not very effective in inducing toxic LLM output because it causes a very high rejection rate across different prompts. However, we do observe some toxicity score improvements due to such instructions when the user prompts are non-toxic.
- • Toxic user prompts specifically designed for GPT-3.5 and GPT-4 easily bypass the content policies of other LLMs as they bring a lower rejection rate compared to general toxic user prompts under benign system instruction settings.
- • GPTQ 3-bit quantization causes a low rejection rate against toxic prompts and significantly downgrades the resistance to toxicity in almost all settings.

4https://github.com/AI-secure/ DecodingTrust/tree/release/data/toxicity/ user_prompts

Benign

Untargeted

Targeted

1.0

1.0

1.0

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | |M|e|th A|od WQ| | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | |G n|PTQ one| | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

0.8

0.8

0.8

0.6

0.6

0.6

0.4

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

3 4 8 16 bits

3 4 8 16 bits

3 4 8 16 bits

###### Figure 22. Stereotype breakdown scores on LLAMA2 13b Chat.

Benign

Untargeted

Targeted

1.0

1.0

1.0

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

Method

AWQ GPTQ

0.8

0.8

0.8

| |
|---|

none

0.6

0.6

0.6

0.4

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

3 4 8 16 bits

3 4 8 16 bits

3 4 8 16 bits

###### Figure 23. Stereotype rejection rate on LLAMA2 13b Chat.

toxic-gpt3.5 Benign

toxic-gpt3.5 Adversarial

toxic-gpt4 Benign

toxic-gpt4 Adversarial

1.0

1.0

1.0

1.0

Method

0.8

0.8

0.8

0.8

AWQ GPTQ none

| |
|---|

0.6

0.6

0.6

0.6

| |
|---|

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

toxic Benign

toxic Adversarial

nontoxic Benign

nontoxic Adversarial

1.0

1.0

1.0

1.0

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

0.8

0.8

0.8

0.8

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

3 4 8 16 bits

3 4 8 16 bits

3 4 8 16 bits

3 4 8 16 bits

###### Figure 24. Toxicity breakdown scores on LLAMA2 13b Chat.

toxic-gpt3.5 Benign

toxic-gpt3.5 Adversarial

toxic-gpt4 Benign

toxic-gpt4 Adversarial

1.0

1.0

1.0

1.0

Method

0.8

0.8

0.8

0.8

AWQ GPTQ none

| |
|---|

0.6

0.6

0.6

0.6

| |
|---|

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

toxic Benign

toxic Adversarial

nontoxic Benign

nontoxic Adversarial

1.0

1.0

1.0

1.0

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.8

0.8

0.8

0.8

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

3 4 8 16 bits

3 4 8 16 bits

3 4 8 16 bits

3 4 8 16 bits

###### Figure 25. Toxicity refusal rate on LLAMA2 13b Chat.

