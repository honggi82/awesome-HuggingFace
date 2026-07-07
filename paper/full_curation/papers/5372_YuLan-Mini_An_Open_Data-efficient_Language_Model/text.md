YuLan-Mini: An Open Data-efficient Language Model
Yiwen Hu∗, Huatong Song∗
Jia Deng, Jiapeng Wang, Jie Chen
Kun Zhou, Yutao Zhu, Jinhao Jiang, Zican Dong
Wayne Xin Zhao†, Ji-Rong Wen
Gaoling School of Artificial Intelligence
Renmin University of China
batmanfly@gmail.com, jrwen@ruc.edu.cn
Abstract
Effective pre-training of large language models (LLMs) has been challenging
due to the immense resource demands and the complexity of the technical pro-
cesses involved. This paper presents a detailed technical report on YuLan-Mini,
a highly capable base model with 2.42B parameters that achieves top-tier per-
formance among models of similar parameter scale. Our pre-training approach
focuses on enhancing training efficacy through three key technical contributions:
an elaborate data pipeline combines data cleaning with data schedule strategies,
a robust optimization method to mitigate training instability, and an effective
annealing approach that incorporates targeted data selection and long context
training. Remarkably, YuLan-Mini, trained on 1.08T tokens, achieves perfor-
mance comparable to industry-leading models that require significantly more data.
To facilitate reproduction, we release the full details of the data composition
for each training phase. Project details can be accessed at the following link:
https://github.com/RUC-GSAI/YuLan-Mini.
1
Introduction
In recent years, large language models (LLMs) [Zhao et al., 2023, Dubey et al., 2024, Qwen-
Team, 2024] have significantly advanced the frontier of AI technology. Unlike traditional machine
learning methods, which are often specialized, LLMs excel across a diverse range of domains and
tasks, showcasing their potential as versatile generalists. Typically, LLMs are developed through a
combination of pre-training and post-training techniques. It is widely recognized that pre-training is
crucial for building the foundational capabilities of the LLMs [OpenAI, 2023, Touvron et al., 2023,
DeepSeek-AI et al., 2024, Yang et al., 2024b].
For transformer language models, the prevailing pre-training approach involves next-token predic-
tion over large-scale unlabeled texts. Although this approach is conceptually straightforward, its
implementation is technically complex. First, researchers must design an effective and efficient
data pipeline to support pre-training, typically involving data collection, data cleaning, data mixing,
and data curriculum. It is widely recognized that “data” is the most crucial element in enhancing
model capabilities. Second, given that LLMs contain a vast number of parameters, the training
process is challenging to stabilize and optimize. Common issues such as loss spikes or gradient
explosions can occur during training, potentially leading to a failed process. Despite the availability
of extensive model checkpoints released by industry companies, the core technical details often
remain undisclosed in public reports. As a result, we know very little about how top-tier language
models are developed in the industry.
∗Team leaders.
†Correspondence to Wayne Xin Zhao.
1
arXiv:2412.17743v2  [cs.CL]  24 Dec 2024

10
23
Approximate FLOPs
40
45
50
55
60
65
70
Avg performance (%)
YuLan-Mini-2.4B
Qwen2-1.5B
 
 
Qwen2.5-0.5B
OLMo2-7B
Gemma2-9B
Qwen2.5-1.5B
 
Qwen2.5-3B
Qwen2-7B
Qwen2.5-7B
SmolLM2-1.7B
Llama3-8B
Gemma2-2.6B
Figure 1: Performance comparison of YuLan-Mini against other base models, based on the average
scores across eight benchmarks: GSM8K, MATH-500, HumanEval, MBPP, MMLU, ARC-Challenge,
HellaSwag, and CEval. Floating Point Operations (FLOPs) are estimated using the scaling law
formula C = 6ND proposed by Kaplan et al. [2020], where N is the model size and D is the size of
the dataset. The models with a size larger than 3B are plotted in gray.
Fortunately, the research community has made significant efforts to enhance the availability of
data resources and the openness of training methodologies for LLM pre-training [Allal et al., 2024,
Groeneveld et al., 2024, Zhang et al., 2024a]. First, well-curated datasets have been released to
support the data preparation required for LLM pre-training. Additionally, various open research
publications have documented the overall training procedures, providing a foundational understanding
of LLM pre-training [Hu et al., 2024, Li et al., 2024b]. These contributions offer basic technical
approaches and essential resources for pre-training an LLM.
Despite these advancements, open LLMs—those with fully disclosed technical details—still face
two main limitations. First, most of these models tend to underperform compared to their industry
counterparts due to constraints in data and computational resources. While some open LLMs achieve
performance levels comparable to industry models, they also require similar amounts of resources,
making them difficult to replicate within the research community. Therefore, developing competitive
LLMs with limited training resources remains a challenge, particularly in university-level laboratories.
Given these constraints, we aim to advance the openness of LLM pre-training by significantly
enhancing both the performance ceiling and training efficiency of open models. Specifically, we focus
on developing relatively small-scale language models, that is, models with a parameter scale ranging
from 1B to 3B with a restricted compute budget.3 Our goal is to build a small yet powerful language
model using only publicly available data, while sharing experiences or insights on improving training
efficiency with limited computational resources.
In this paper, we present a comprehensive technical report on the development of a highly capable
2.42B-parameter language model (the base model) that achieves top-tier performance among models
of similar parameter scale. Building upon the transformer architecture, we devise a data-efficient
pre-training approach for LLMs. Our approach includes three major contributions to enhance training
efficacy: (1) an elaborately designed data pipeline that combines data cleaning with data schedule
strategies; (2) a systematic optimization method that can effectively mitigate training instability;
(3) an effective annealing approach that integrate targeted data selection and long context training.
We explore a variety of techniques to enhance the performance of YuLan-Mini. In particular,
we extensively leverage synthetic data for model training, including o1-like long-thought data.
3Some papers refer to pre-trained language models of this size as “small language models”. However, due to
the technical similarities, we continue to refer to them as large language models in this paper.
2

Additionally, we investigate multiple factors that may contribute to training instability. We provide
two versions of the checkpoints, supporting 4K and 28K contexts, respectively.4
To demonstrate the effectiveness of our pre-trained base model, we conduct extensive experiments on
a variety of benchmarks, and compare it with a few competitive base models from both research and
industry. Experimental results show that our base model, YuLan-Mini, can achieve very promising
results among these compared models. For instance, it (the 28K version) achieves scores of 37.80 on
MATH-500 (four-shot), 64.00 on HumanEval (zero-shot), and 49.10 on MMLU (five-shot). Figure 1
presents a comparison of YuLan-Mini with other industry models.5
To facilitate reproduction, we report the complete training details for YuLan-Mini, and also
release the data composition for all training phases (Appendix E). More supporting resources can
be accessed at our project link: https://github.com/RUC-GSAI/YuLan-Mini.
2
Overall Pre-Training Configuration
In this section, we will provide an overview of the pre-training configuration, introducing its key
components and the algorithms involved in the process. For a more detailed discussion of the major
contributions made in this work, please refer to Section 3, Section 4, and Section 5.
2.1
Model Architecture
Our model is based on a decoder-only transformer with a tall and narrow architecture, inspired by
previous studies [Liu et al., 2024c, Hu et al., 2024]. It comprises a total of 2.42B parameters, of which
2.23B are non-embedding parameters. The hyperparameter configurations for our model architecture
are provided in Table 1. Additionally, we re-parameterize each weight matrix of different modules
with an extra learnable parameter [Nishida et al., 2024], enhancing the model’s training stability
(discussed in Section 3). Next, we briefly introduce the main components in our architecture.
Embedding tying
We utilize embedding tying [Press and Wolf, 2017] to reduce the model’s
parameter size and stabilize training. In our preliminary experiments, we find that sharing the
embedding and unembedding matrices improves model convergence. Furthermore, when these
matrices are not shared, they often necessitate different initialization strategies, which we will discuss
in Section 3.
Pre-RMSNorm
Layer normalization (LN) has been shown to enhance numerical stability and
accelerate learning speed [Ba et al., 2016]. We integrate Pre-LN into our model architecture to
improve convergence stability and speed compared to Post-LN [Xiong et al., 2020]. Regarding the
form of normalization, we opt for RMSNorm over the conventional LayerNorm, as it conserves CUDA
memory while attaining a comparable effect [Zhang and Sennrich, 2019].
SwiGLU
Our model introduces non-linearity using a gated linear unit (GLU) with the Swish
activation function, known as SwiGLU Shazeer [2020]. This method effectively captures complex
data relationships and has proven to be effective in relatively small language models, as demonstrated
by [Liu et al., 2024c].
Attention mechanism
We adopt the grouped-query attention (GQA, Ainslie et al. [2023]), which
enables the model to reduce KV cache usage while maintaining high performance. Specifically, we
employ 30 heads for query attention and 6 groups for key-value heads. We opt not to make the
KV head size divisible by 8 since small language models rarely require tensor parallelism during
inference.
Rotary Embedding
We adopt rotary positional embedding (ROPE) to capture the positional
information in our model, since it integrates absolute and relative positioning in an unified way.
4Due to resource constraints, we were only able to train a model with up to 28K context.
5We select eight popular benchmarks to cover math, coding, general, and language capabilities. This figure is
primarily intended to illustrate the training efficacy of our model, rather than to serve as an accurate capacity
ranking of existing models.
3

Table 1: Hyperparameter settings of diffrent models. rffn is the ratio of the feed-forward network’s
hidden size to the model’s hidden size. The definition of the symbols is available at Table 8
Model
nlayers
dmodel
rffn
nheads
nkv_heads
LLaMA-3.2-3B
28
3,072
2.7
24
8
Phi-3-mini-4k-instruct
32
3,072
2.7
32
32
MiniCPM-2B
40
2,304
2.5
36
36
MiniCPM3-4B
62
2,560
2.5
40
40
Qwen2.5-1.5B
28
1,536
5.8
12
2
MobileLLM-1B
54
1,280
2.8
20
5
YuLan-Mini
56
1,920
2.5
30
6
Table 2: Compression rate of different tokenizers. Higher values indicate more effective compression.
Tokenizer
Vocabulary Size
Web
Chinese
Math
Code
Gemma2-2B
256,000
4.928
3.808
2.865
3.309
Qwen2.5
151,936
4.935
3.956
2.890
3.881
LLaMA-3.1
128,000
4.994
3.263
3.326
3.911
MiniCPM-2.4B
122,753
4.753
4.273
2.739
3.052
Phi-3.5-mini
100,352
4.311
1.914
2.654
3.110
MiniCPM-1.2B
73,440
4.631
4.042
2.696
3.017
YuLan-Mini
99,000
4.687
4.147
2.716
3.033
+ Dropout
99,000
4.687
4.146
2.715
3.031
During the stable training stage, we set the parameter θ to 10,000, and increase it to 49 000 during the
annealing stage to extend the context length to 28,672 (28K) tokens using adjusted base frequency
(ABF).
2.2
Tokenizer
Tokenization is a critical preprocessing step that splits input text into sequences of tokens. Below, we
provide details of our tokenizer.
Vocabulary size
Generally, the vocabulary size should be chosen to balance its effects on the
model’s parameter size and efficiency. We adopt the three approaches proposed by Dagan et al.
[2024] to balance the compute budget and vocabulary capacity, yielding a final vocabulary size of
around 99,000. For simplicity, we reuse the Byte Pair Encoding (BPE) tokenizer of MiniCPM [Hu
et al., 2024]. Specifically, we truncate the vocabulary by applying the corresponding BPE merge
rules to reduce the number of tokens. We also heuristically remove rare domain-specific tokens,
while add some reserved tokens in the vocabulary. The statistics of the modified vocabulary and the
compression rate are shown at Table 2. The test set for the tokenization experiments is sourced from
a diverse array of datasets, as detailed in Section 6.3. Overall, our tokenization method achieves a
well-balanced compression rate across different domains.
BPE-dropout
Existing sub-word tokenization methods prevent the language models from under-
standing the alphabetic composition of a token. To mitigate this issue, BPE-dropout [Provilkov et al.,
2020] has been proposed to help the model better learn the internal representation of a token, enabling
it to more effectively capture possible subwords within a word. Specifically, we use a relatively low
dropout rate of 0.2, and applying the dropout method results in only a slight increase in the number
of tokens (0.07%), as shown in Table 2.
Digit tokenization
Digit tokenization plays a crucial role in mathematical tasks, including numerical
calculation and complex reasoning. We follow the common practice of splitting numbers into
individual digits [Bi et al., 2024, Yang et al., 2023]. Although other methods, such as three-digit
4

tokenization, may achieve higher compression rates, using individual-digit tokenization typically
leads to improved numerical calculation accuracy [Wang et al., 2024a].
2.3
Training Data Preparation
Data serves as the foundation for developing the model’s capabilities, and we employ specially
designed strategies for collecting and preparing the training dataset. Next, we briefly describe the
general procedure for data preparation. A more detailed and comprehensive description of the data
pipeline is provided in Section 4.
Data collection and selection
To ensure reproducibility, our pre-training data is primarily sourced
from open-source pretraining datasets and synthetically generated data. The main open-source
datasets include FineWeb-Edu [Lozhkov et al., 2024a], the-stack-v2 [Lozhkov et al., 2024b], open-
web-math [Paster et al., 2024], Chinese-FineWeb-Edu [Opencsg], and OpenCoder-LLM [Huang
et al., 2024]. The entire pre-training dataset has undergone rigorous preprocessing, with 1.08T tokens
for training. Among them are 481B English web data, 138B general English knowledge, 227B
code pre-training data, 16.7B code instruction data, 93.8B mathematics pre-training data, 15.5B
mathematics instruction data, and 108B Chinese data.
Data schedule
Using the WSD scheduling method [Hu et al., 2024], the training process is divided
into three main stages: warmup, stable training, and annealing. The warmup stage uses 10B tokens,
the stable training stage utilizes 990B tokens, and the annealing stage uses 80B tokens. To better
manage the training process, we divide the entire training trajectory into 27 consecutive curriculum
phases, each consisting of 40B tokens. When transitioning between these curriculum phases, the
dataset proportions are slightly adjusted based on the model’s performance on various benchmarks
and the perplexity (PPL) of validation texts. However, the internal data distribution of each curriculum
phase cannot be modified once it has been scheduled for training. During the annealing stage, the
proportion of instruction data and long context data is increased.
2.4
Model Optimization
For model optimization, hyperparameters are crucial for training stability and model performance.
Specifically, we adopt the WSD learning rate scheduler [Hu et al., 2024]. Maintaining a constant
learning rate during the stable training stage eliminates the necessity to specify an ending step, as
required by the cosine scheduler. This approach facilitates continuing pre-training from the last
checkpoint during stable training. It also allows for more flexible data preparation: we can prepare the
data while the preceding curriculum phase is running. Additionally, we estimate an optimal annealing
ratio of 8% for the stable training stage using the scaling law of learning rate annealing [Tissue et al.,
2024].
For training stability, we combine a parameter initialization approach akin to µP [Dey et al., 2023b,
Hu et al., 2024, Yang et al., 2022] with WeSaR re-parameterization [Nishida et al., 2024], using a
relatively large global learning rate of 0.01. The rationale behind adopting a large learning rate is
the expectation that the model will possess greater potential for enhancement during the annealing
stage. We set the AdamW hyper-parameters as follows: β1 = 0.9, β2 = 0.95, ϵ = 10−15, with the
weight_decay of 0.1 and the z-loss coefficient of 10−4 [de Brébisson and Vincent, 2016]. We
use a variance of 5 × 10−5 for initialization. As found by Wortsman et al. [2024], extending the
warm-up ratio enhances training stability, so we linearly warm up the model over 10B tokens. We
use a batch size of 4.12M tokens with a sequence length of 4,096, extending the context length
during the annealing stage while keeping the total token count in the batch size unchanged. We avoid
using gradient accumulation to prevent numerical precision error of bfloat16. Detailed analysis of
training stability can be found in Section 3.
2.5
Training Infrastructure
We build a simple yet efficient training framework based on the HuggingFace Trainer and other
open-source libraries (DeepSpeed, flash-attention, and liger-kernel).
5

0
5
10
15
20
25
Number of steps (×104)
1
2
4
8
(a) Training loss.
0
5
10
15
20
25
Number of steps (×104)
0.0
0.2
0.4
0.6
0.8
1.0
(b) Gradient norm.
Figure 2: Training loss and gradients during pre-training process.
Specifically, we first use ZeRO-1 [Rajbhandari et al., 2020] data parallelism provided by DeepSpeed
intergration and then switch to ZeRO-2 after confirming that it does not cause training divergence in
our model.6 We also leverage Flash Attention [Dao et al., 2022, Dao, 2024] and a triton kernel library
liger-kernel [Hsu et al., 2024] to accelerate training processes. By employing fused kernels, we
achieve a 30% reduction in training time and up to 70% savings in CUDA memory.7 We further
optimize the balance between CUDA memory usage and training time by adjusting the number
of layers through the activation checkpointing function. For enhanced training efficiency, we use
bfloat16 precision for both model parameters and NCCL communications. The model’s FLOPs
utilization (MFU) is estimated at 51.57%.
Regarding the hardware setup, we initially employ a 56 A800-GPU cluster managed by the SLURM
system [Yoo et al., 2003]. We later reduce the number of GPUs to 48 by transitioning the distributed
optimizer to a universal checkpoint [Lian et al., 2024]. To maximize device utilization, we perform
tokenization and packing asynchronously. Given the modest size of our cluster, the likelihood
of encountering NCCL failures is relatively low. Therefore, after assessing the advantages and
disadvantages, we decide to store a checkpoint every hour and implement automatic restarts.
For efficient evaluation, we utilize LLMBox [Tang et al., 2024b] to integrate vLLM [Kwon et al.,
2023] for generative tasks and employ KV cache scheduling for multiple-choice tasks. For a detailed
description of the evaluation setup and results, please refer to Section 5.
3
Training Stability
Training stability is crucial for the efficient pre-training of LLMs. Under normal conditions, loss
trajectories are expected to decrease smoothly and remain near their anticipated values consistently,
such as those predicted by the scaling law curve, even in the presence of minor perturbations. However,
models that are improperly initialized or trained with unsuitable architectures or hyper-parameters
may experience marginal stability or outright instability. In a marginally stable network, even a
minor abnormal perturbation from the data can push it into a non-steady state. If the network can
self-correct, it will experience temporary loss spikes; otherwise, training may diverge.
Training stability issue exists even in training relatively language models. For example, as observed
empirically by Wortsman et al. [2024], under the same learning rate, the smaller the model, the larger
the order of magnitude of the model logits, which is a typical factor for training instability. General
methods, such as replacing the data that leads to spikes or reducing the learning rate, usually only
palliate superficial problems.
Despite significant efforts in the literature to mitigate training instability [Takase et al., 2023, Yang
et al., 2022, Wortsman et al., 2024], prior studies typically focus on individual techniques or conduct
6https://github.com/microsoft/DeepSpeed/issues/6351
7Fused kernels include: SelfAttention, RMSNorm, RoPE, SwiGLU, FusedLinearCrossEntropy, and AdamW.
torch.compile is also enabled in our implementation.
6

relatively small-scale experiments. There remains a lack of systematic investigation into the effects
of various potential techniques in large-scale pre-training experiments. During our pre-training
process, we encounter severe training instability issues, prompting us to conduct an in-depth study
on how to address this problem effectively. Our primary approach involves combining a µP-like
initialization [Dey et al., 2023a] with a re-parametrization method [Nishida et al., 2024] to adjust the
learning rate and stabilize training. In the following, we present a detailed approach for maintaining
training stability.
3.1
Exploring the Hidden States Variability and Training Instability
To effectively mitigate training instability, it is crucial to examine potential indicators of abnormal
states. Generally, loss, gradient, and hidden states are three interconnected factors that reflect the
dynamics of training. Among these indicators, loss provides surface-level clues about instability, while
gradients and hidden states often reveal deeper underlying factors that contribute to a pathological
state. We begin by conducting a preliminary experiment to assess the impact of different indicators
on training instability. We then analyze the potential reasons theoretically.
3.1.1
Preliminary Experiment on Indicators
We conduct a preliminary experiment to showcase the effects of tracking our indicators based on
hidden states during training.
Training setup
Since it is resource-intense to perform extensive experiments on our model, we
explore the training dynamics by conducting surrogate experiment with a small proxy model of 0.2B
with similar architecture. We employ a relatively large learning rate of 0.01, to expose potential
instabilities within the model. We keep this baseline model setup in the subsequent experiment, which
we elaborate on in Appendix B. Specifically, our optimization goal is to achieve optimal performance
while ensuring that the training process does not result in divergent loss or an increasing trend in
gradient norm.
Indicators setup
In large-scale training, distributed optimizers are often used, which means that the
gradients of different modules may be distributed across various data parallel ranks. This distribution
makes it inefficient to directly obtain the gradients. As a result, we primarily track each module’s
weight matrix and hidden states (i.e., their outputs). Specifically, we record the mean and variance
of the weights and hidden states, as well as the root mean square (RMS), which is calculated using
the follow formula RMS =
p
Var + Mean2. Note we consider the outputs of various modules in the
transformer (i.e., FFN, Attention, RMSNorm) as hidden states.
Empirical findings
Figure 3a illustrates our findings on the relationship between training instability
and exploding hidden states. Across all layers, there is a consistent upward trend in both hidden states
and gradient norms as the number of training steps increases. Interestingly, these intermediate trends
are difficult to detect in the early stages of pre-training when focusing solely on the loss. Moreover,
in addition to the temporal dimension, indicators related to hidden states also show an increasingly
divergent trend across the depth of the model (i.e., as the number of layers increases). The ratio of
the variance of the last layer to that of the first layer grows linearly, indicating a potential future
explosion in hidden states and gradient norms. From these results, we can see that hidden states
play a significant role in affecting training stability. Based on these observations, our core idea is to
monitor and adjust hidden states to maintain training stability.
In addition to monitoring hidden states, other indicators can help detect abnormal optimization
issues. In our experiments, we also examine another stability indicator known as token embedding
variability (TEV) [Chung et al., 2024], which measures the variance of data entries within a vector.
By comparing the TEV of two input vectors, we can quantify how differently they are distributed.
If the input vectors follow distributions that are significantly different from each other, the output
vectors may of course exhibit large fluctuations. These output vectors are essentially a special kind of
hidden states and therefore can detect training instability. However, since we are already recording
hidden states in our experiments, we did not use TEV as an additional indicator.
7

0
2500
5000
7500
10000
Training Steps
1
10
Layer  1
Layer  7
Layer 13
Layer 19
Layer 25
Layer 30
Grad Norm
(a) Exploding hidden states.
0
2500
5000
7500
10000
Training Steps
1
10
(b) Convergent hidden states.
0
2000 4000 6000 8000
Training steps
101
103
105
107
109
1011
Attention scores
Attention scores
Loss
2
3
4
6
10
Loss
(c) Loss prediction failure.
Figure 3: Comparison of training dynamics between divergent and convergent trial. The y-axis
denotes the value of the hidden states variance and gradient norm on a log-scale. Both trials have
consistent loss, but different trends of hidden states variance and gradient norm.
3.1.2
Theoretical Analysis and Empirical Evidence on Exploding Hidden States
In this part, we first formalize the hidden states of transformer, and then derive three potential reasons
for exploding hidden states theoretically. We also showcase corresponding empirical evidence, to
verify the effectiveness of using hidden states as a training stability indicator.
Formalization of hidden states
To better analyze how hidden states can be utilized to indicate
training stability, we begin by formally defining the hidden states in our model. Since Yulan-Mini is
a multi-layer transformer model, and the hidden states of the l-th layer, denoted by zl ∈Rd, can be
specified as follows:
zl = yl + FFN(RMSNorm(yl)),
yl = xl + MHA(RMSNorm(xl)),
where xl ∈Rd denotes the input of the l-th layer. With sub-layer inputs u = RMSNorm(xl) and
v = RMSNorm(yl), the FFN and MHA are defined as:
FFN(u) = W2F(W1u),
(1)
MHA(v) = concath
i=1[headi(v)]Wo.
(2)
It is worth noting that we focus on discussing a simplified version of the FFN layer above. Modern
LLAMA-like architectures typically utilize GLU-style non-linearities, which we discuss in Section 3.2.
Building on this definition, we specifically focus on the change in the variance of hidden states,
as we empirically observe that it tends to increase during the training process. If not properly
addressed, this increasing trend could lead to training instability. Considering the relationship
var(a + b) = var(a) + var(b). We begin by analyzing the abnormal increase in variance of hidden
states that arises inherently from residual connections. Furthermore, we demonstrate how layer
normalization can contribute to the variance of hidden states increase, particularly when the inputs
deviate significantly from their normal range. Finally, we show that certain abnormal updates in layer
normalization are actually driven by the growing mean of attention logits.
Exploding hidden states due to residual connection
Figure 3a illustrates the exponential growth
trend of the variance in the hidden states. To understand the underlying cause, we express the hidden
states in terms of the model’s weights and inputs:
var(zl) = var(yl) + var(FFN(RMSNorm(yl))),
var(yl) = var(xl) + var(MHA(RMSNorm(xl))).
For ease of analysis, we first assume that:
x, y ∼N(0, σ2).
(3)
Under this assumption, we can obtain var(u) = var(v) = 1. In this case, we can express the variance
as the following form:
var(zl) = var(xl) + var(FFN(u)) + var(MHA(v)),
(4)
8

0
2000
4000
6000
8000
Training steps
0
1
2
3
4
Variance of LayerNorm output
Layer 1
Layer 2
Layer 3
Layer 4
Layer 5
Layer 6
Figure 4: Variance of LN output of each layers.
0
2000
4000
6000
8000
Training steps
101
102
103
104
105
Mean of attention scores
Mean of attention scores
Variance of LayerNorm output
0.1
1
Variance of LayerNorm output
Figure 5: Attention scores explodes before LN.
which means, the hidden states will grow by the variance of MHA and FFN in each layer:
var(headi(v)) = var(softmax(Z)V) · dmodel · var(Wv) < dmodel · var(Wv),
(5)
var(FFN) = dffn · dmodel · var(W1) · var(W2),
(6)
var(MHA) = var(head(v)) · dmodel · var(Wo) < d2
model · var(Wv) · var(Wo),
(7)
where Z denotes the scaled attention scores. The base dimensionality dmodel of LLMs are often
large (e.g., 1,920 in our model). Additionally, the vanilla setup in Hugging Face uses a default
initialization standard deviation of 0.02 for the weight matrices. When training the proxy model,
this initialization leads to significantly large variance values, exacerbated by the effects of the
aforementioned Attention and FFN modules. As a result, the gradient norm becomes very large. A
detailed derivation of the attention head can be found in Takase et al. [2023]. Common mitigation
strategies include initializing the weights with a very small variance, typically inversely proportional
to dmodel, which we will discuss in Section 3.2.
Exploding hidden states due to layer normalization
According to prior studies [Xiong et al.,
2020, Takase et al., 2023, Wortsman et al., 2024], the variance of the input to layer normalization
should not be significantly smaller than 1, as this can lead to an increase in the gradient norm:

∂RMSNorm(x)
∂x

2
= O
 √
d
∥x∥2
!
.
To investigate the impact of layer normalization, we conduct an experiment examining the variance
of the outputs from the LN layer in our model (Figure 4), where the embedding layer is followed
by layer normalization. We empirically find, if the embedding layer is initialized with the default
variance, we must multiply its output by a scaling factor to ensure the input to the layer normalization
has a variance of 1. Therefore, we apply a scaling factor of scale_embed = 10 for model training.
For experiments that do not employ embedding tying, we directly set the initialization standard
deviation of the embeddings to 1.
Exploding hidden states due to attention scores
When applying the scaled embedding method
mentioned above, we have empirically observed that attention scores can also explode, resulting in
the explosions of hidden states. We first examine the calculation process of attention scores, which is
formally defined as:
S = XT WT
QWKX.
Then, we calculate the gradient of the attention scores with respect to the query and key matrices as
follows:
∂S
∂WQ
= XXT WT
K,
(8)
∂S
∂WK
= WT
QXXT .
(9)
9

0
10
20
30
40
50
LAMBADA
ALL (D=80B)
+ WeSaR
+ Depth µP
+ Cerebras µP (LR=0.01)
 QK LayerNorm & + Weight Decay
+ QK LayerNorm
Baseline (LR=0.001, D=20B)
33.36
29.37
28.93
27.57
23.06
22.10
21.86
33.36
29.37
28.93
27.57
23.06
22.10
21.86
Time (h)
4.07
3.87
3.87
3.57
4.79
3.57
Figure 6: Ablation experiments on training instability mitigation methods are conducted. We report
the average of LAMBADA accuracy of the last three checkpoints of the training and the estimated
running time on our 48 A800-GPU cluster. Divergent gradient norm or spiking loss trajectories are
shown in red bars, and convergent training is shown in green.
From the above derivation, we find that the query-key multiplication term involved can lead to an
unbounded gradient, potentially causing severe self-excitation. In such cases, the mean of the attention
scores gradually increases as the number of layers increases, eventually resulting in exploding hidden
states. Therefore, it is essential to monitor and regularize the attention scores during training.
3.2
Training Instability Mitigation Methods
After discussing the potential causes of training instability, we will introduce mitigation methods to
enhance training stability.
3.2.1
Scaled Initialization and Scaling Factor
As discussed in Section 3.1.2, employing a carefully designed initialization method along with appro-
priate scaling factors is a key strategy for addressing training instability. In this regard, we explore the
initialization strategies proposed by Megatron-LM [Shoeybi et al., 2020] and BLOOM [Scao et al.,
2022]. Specifically, we initialize W2 of the FFN layer (Equation (1)) and Wo of the Attention layer
(Equation (2)) in accordance with N(0, σ2
base/(2nlayers)), and initialize the remaining parameters
according to N(0, σ2
base). Here, we set the initialization standard deviation as σbase =
p
2/(5d).
To analyze the effect of the above initialization strategy, we plug the initialization standard deviation
into Equation (6) and (7), and obtain the following formulas:
var(FFN) = rffn · d2
model ·
2
5dmodel
·
1
5dmodel · nlayers
=
1
5nlayers
,
var(MHA) = var(head(v)) · dmodel ·
1
5dmodel · nlayers
<
2
25nlayers
.
Substituting the obtained results into Equation (4), we can measure the growth of the hidden states’
variance throughout the entire network as:
var(z) −var(x) = nlayersvar(FFN) + nlayersvar(MHA) < 7/25,
(10)
which successfully regularizes the growing trend of hidden states.
In the modern LLaMA architecture, it is more common to use a GLU-style of non-linearity, which can
be denoted as FFN(u) = [F(uWgate) ⊙(uWup)] · Wdown. We initialize Wup and Wgate the same
way as W1, and Wdown same as W2. The above derivation in Equation (10) still holds empirically
in this setting.
10

Table 3: Comparison of the used hyperparameter settings for training stability, where the detailed
explanation for the variables are in Table 8. We include SI [Takase et al., 2023] for comparison,
MiniCPM [Hu et al., 2024], CerebrasGPT [Dey et al., 2023a]. The definition of the symbols is
available at Table 8 .
Method
SI
MiniCPM
CerebrasGPT
YuLan-Mini
Scale Embedding Output
1
12
10
10
Scale MHA equation
1/√dhead
1/√dhead
1/dhead
1/√dhead
Scale Residual Connection
1
1.4
√nlayers
1
1.4
√nlayers
QKV Weights LR
ηbase
ηbase/mwidth
ηbase/mwidth
ηbase/mwidth
QKV σ Init
σ2
base
σ2
base/mwidth
σ2
base/mwidth
σ2
base/mwidth
O Weights LR
ηbase
ηbase/mwidth
ηbase/mwidth
ηbase/mwidth
O σ Init
σ2
base
2nlayers
σ2
base/mwidth
σ2
base
2mwidth·nlayers
σ2
base
2mwidth·nlayers
FFN1 Weights LR
ηbase
ηbase/mwidth
ηbase/mwidth
ηbase/mwidth
FFN1 σ Init
σ2
base
σ2
base/mwidth
σ2
base/mwidth
σ2
base/mwidth
FFN2 Weights LR
ηbase
ηbase/mwidth
ηbase/mwidth
ηbase/mwidth
FFN2 σ Init
σ2
base
2nlayers
σ2
base/mwidth
σ2
base
2mwidth·nlayers
σ2
base
2mwidth·nlayers
Scale Output logits
1
1/mwidth
1/mwidth
1
By initializing in this manner, we empirically find that while there may still be a tendency for the
hidden states at each layer to gradually increase, this increase remains within a reasonable range.
3.2.2
Maximal Update Parametrization
In the above, we have explored the importance and effectiveness of parameter initialization through
theoretical analysis and empirical experiments. However, when we migrate the training configuration
selected on the 0.05B proxy model to the model of the target size, instability still exists.
To ensure hyper-parameter consistency across different model scales, the Maximal Update
Parametrization (µP) has been proposed [Yang et al., 2022, 2024c], including width scaling and
depth scaling, which facilitate transferring the hyperparameters from smaller models to the training
of larger models. Typical use of this strategy can be found in CerebrasGPT [Dey et al., 2023a] and
MiniCPM [Hu et al., 2024]. In particular, they have found that the optimal learning rate remains
quite stable during migration. Compared to the basic method in Section 3.2.1, µP provides a more
systematic approach for setting the initialization and scaling factor.
We apply µP, which takes into account embedding parameters, model depth, and model width, for
parameter initialization. Additionally, we conduct a comprehensive parameter search on proxy model
to identify the optimal configuration, exploring parameters such as batch size and learning rate.
3.2.3
Mitigating Instability through Re-Parametrization
When using µP, we find that spikes in loss still occur under large learning rates. We speculate this is
because, although µP alleviates instability during the initial stages of training, it may still deviate
from a stable state during prolonged updates at large learning rates.
Therefore, we apply a simple yet effective method WeSaR proposed by Nishida et al. [2024], which
empirically decouples the update of the gradient norm from the gradient direction. This is achieved
by re-parametrizing the matrix weights W with an additional learnable parameter α ∈R as:
W = αf
W.
We find the above WeSaR method to be effective in addressing the reasons of exploding hidden states
highlighted in the previous analysis, as illustrated in Figure 3b. It is likely due to re-parametrization,
11

which distributes the gradient of a single weight across multiple new re-parametrized weights, thereby
reducing the abnormal updates caused by excessively large gradients.
Re-parametrization is applied to matrices other than Layer Normalization as shown in Figure 6.
Specifically, we initialize f
W ∼N(0, σ2) and set α = 1/γ. In this setting, it is straightforward to
verify that W ∼N(0, (σ/γ)2), which satisfies the scaled initialization requirements described in
Section 3.2.1.
3.3
Discussion on Other Training Stabilization Methods
During our training process, we thoroughly explore and utilize various training stabilization tech-
niques. Below, we provide a brief introduction to these methods.
3.3.1
Warmup Based Methods
To ensure the model transitions smoothly from its initial state to a stable training phase, we empirically
find that employing learning rate warmup and sequence length warmup is often effective, which are
detailed below.
Learning rate warmup
Learning rate warmup involves gradually increasing the learning rate from
a small initial value (e.g., 0) to the max learning rate in TLR steps. Wortsman et al. [2024] suggests
that a longer learning rate warmup can reduce sensitivity to the learning rate, as measured by training
stability across different learning rates. We empirically verify this conclusion and find increasing TLR
indeed enhances training stability. For our final training, we set TLR = 2,433, which approximately
corresponds to 10 billion tokens of data.
Sequence length warmup
Sequence length warmup starts training with short sequences (e.g., 64
tokens) and gradually increases their length within the steps of TSL, which is typically set to a few
multiples of TLR [Li et al., 2022]. The rationale behind this approach is that longer sequence lengths
contribute significantly to extreme gradient variance, particularly in the early stages of training. In
our experiments, we also observe similar fluctuations in loss during long context training (especially
in the 27-th curriculum phase). However, since we have stabilized the training using other methods
and this approach requires additional preparation of the data, we ultimately decided not to adopt it.
3.3.2
Module Based Methods
In this part, we introduce module-based methods which regularize the model states by adjusting
specific components in it.
QK LayerNorm
QK LayerNorm and its variants (e.g., QKV LayerNorm or capped QK LayerNorm)
have have been shown to effectively mitigate the growth of attention logits [Rybakov et al., 2024],
which we also have identified in Section 3.1.2. We highlight the effectiveness of QK LayerNorm
because it directly addresses the exponential growth of gradients caused by the interaction of hidden
states (QKT ), whereas some other methods only attempt to control the downstream instability.
Our empirical study, which is shown in Figure 7a and 7b, demonstrates the advantages of QK
LayerNorm in terms of training stability. However, it significantly slows down the calculation in
training: with the same acceleration configuration, using QK LayerNorm increases the training time
by 34%. Considering that the previously mentioned methods have already demonstrated stability in
our preliminary experiments, we ultimately decided not to use QK LayerNorm.
Embedding tying
Embedding tying aims to share the weights of embedding and unembedding
(i.e., lm_head) parameters [Press and Wolf, 2017]. Our experiments demonstrate that the utilization
of embedding sharing enables faster convergence and more stable training, and there is no significant
degradation in training performance.
Z-loss
Z-loss was originally proposed to alleviate the shift and scale of logits in classification
tasks [de Brébisson and Vincent, 2016]. Subsequently, it has been introduced to LLM and MoE
training to mitigate the growth of the logits layer [Chowdhery et al., 2023, Zoph et al., 2022]. It adds an
auxiliary term related to the softmax normalizer log Z to the original loss: L = lm_loss + ζ log2 Z.
12

0
2000
4000
6000
8000
10000
Training steps
1
10
Avg LN variance w/o QK LayerNorm
Avg attention logits w/o QK LayerNorm
Avg LN variance w/o QK LayerNorm
Avg attention logits w/ QK LayerNorm
(a) Variance of attention values and LN outputs
0
2000
4000
6000
8000
10000
Training steps
10
1
100
101
Grad Norm
Loss w/o QK LayerNorm
Loss w/ QK LayerNorm
Grad Norm w/o QK LayerNorm
Grad Norm w/ QK LayerNorm
1
10
Loss
(b) Gradient norm and loss trajectory
Figure 7: The curves of attention value and LN output variances (left) and gradient norm and loss
(right). After using QK LayerNorm, we prevent the explosion of attention logits and gradients,
keeping the LN output stable around 1 and the loss consistent.
In our experiments, we set the coefficient ζ = 10−4 to encourage the logits to be close to 0. Although
ablation studies did not show significant effects, we incorporate it into the final training.
3.3.3
Numerical Optimization Based Methods
In addition, we consider using several commons methods to reduce abnormal updates during opti-
mization, as described below.
Weight decay
To prevent abnormal model weights due to large gradient updates, weight decay
functions by subtracting a penalty term from the weights during the update step, rather than directly
modifying the gradients. Formally, we denote the AdamW update without learning rate or weight
decay as:
∆= α ˆmt/(
p
ˆvt + ϵ).
(11)
Then at update step t, the AdamW update with weight decay is given by θ →θ −stη(∆−λθ),
where λ is the weight decay coefficient, st is learning rate schedule and η is the max learning
rate. Previous work has recommended using an independent weight decay for updates, expressed as
θ →θ−st(η∆−λ′θ), which is claimed to be applicable to a wider range of learning rates [Loshchilov
and Hutter, 2019, Wortsman et al., 2024]. In the PyTorch implementation, this approach can be
achieved by tuning the weight decay coefficient λ in conjunction with the maximum learning rate,
following the relationship λ′ = η · λ.
Optimizer hyper-parameter
In the update of AdamW (Equation (11)), ˆmt and ˆvt represent the
first and second gradient moment exponential moving averages (EMA), respectively. If the gradient
is of the same order of magnitude as ϵ, then the update value ∆will be significantly reduced due to ϵ,
which empirically leads to training instability inherent in embedding layer. A direct solution is to
reduce ϵ from the default value of 10−8 to 10−15. Generally speaking, this method can alleviate the
divergence caused by abnormal embedding gradient values in larger-scale models [Wortsman et al.,
2024, Molybog et al., 2023].
Numerical stability
In practice, paying close attention to numerical stability is crucial, as it can
be an important source of training instability. In large-scale model training, float32 often suffers
from low computational efficiency. Although float16 offers comparable precision with higher
computational efficiency, it has a limited numerical representation range (e.g., maximum positive
number that can be represented is 65,504). Therefore, bfloat16 has been proposed as a trade-off
between precision and representation range. It largely alleviates the training instability caused by
exceeding the representable range. However, in practice, bfloat16 introduces precision problems
compared to float16. In experiments conducted by Lee et al. [2024] using bfloat16 with 188
13

Table 4: Statistical information of the entire pre-training corpus for YuLan-Mini. The data during the
annealing process is detailed in Table 5. For model reproducibility, all curated datasets are placed in
Appendix D, and the remaining synthetic data we generated is open-sourced.
Type
Source
Volume
Web Pages
FineWeb-Edu, DCLM, Chinese-FineWeb-Edu
559.76B
Math (Pretrain)
AutoMathText, Proof-Pile-2, OpenWebMath Pro
85.00B
Code (Pretrain)
the-stack-v2, StarCoder
202.44B
General Knowledge
arXiv, StackExchange, English News
121.87B
Books
CBook, Gutenberg, LoC-PD-Books
52.13B
Encyclopedia
Wikipedia, Baidu-Baike
14.80B
Open-Source Instruction
SlimOrca, OpenMathInstruct-1, JiuZhang3.0
11.64B
Synthetic Pretrain Data (Ours)
Synthetic document (seed: AutoMathText, LeetCode)
8.76B
Synthetic Instruction (Ours)
Reasoning (seed: MetaMathQA, DeepMind Math, ...)
23.52B
Total
-
1,080B
random seeds, 18 runs diverged, whereas using float32 under the same configuration resulted in
all runs converging normally. To mitigate precision issues with bfloat16, Gemma [Mesnard et al.,
2024] find that shifting the RMSNorm weight from 1 to 0 helps, considering that bfloat16 has
symmetric numerical precision around 0 but greater inaccuracies near 1.
Value clipping
To further limit the gradient within certain range, we utilize a gradient clipping of 1.
We find using a smaller limit does not help stabilize the training. In addition, initializing the LLM in
accordance with “3-σ” rule with nn.init.trunc_normal_ may be helpful for numerical stability.
4
Data Pipeline
To pre-train an effective LLM, it is crucial to develop a robust data pipeline that comprehensively
encompasses the key steps for curating the pre-training data. These steps include data collection,
filtering, selection, mixing, and curriculum design. Below, we describe each step of the data pipeline
in detail.
4.1
Data Collection
For pre-training data preparation, we primarily reference the data configuration of Yulan-3 [Zhu
et al., 2024] and Llama-3-SynE [Chen et al., 2024], which encompasses a wide-ranging and diverse
collection of data such as web pages, encyclopedias, books, mathematical corpora, code, general
knowledge, and synthetic data. Table 4 presents an overall summary regarding the composition of
our training data.
4.2
Data Filtering
As we aim for a data-efficient training approach, data quality is crucial to the final model’s perfor-
mance. For this purpose, we implement a thorough data cleaning process to remove low-quality texts
(Figure 8).
De-duplication
Data de-duplication is a crucial step in standard LLM training practices, as previous
research has demonstrated that duplicate data can significantly degrade model performance [Tirumala
et al., 2023]. We use the MinHash algorithm implemented by the Yulan-GARDEN library [Sun et al.,
2024] to deduplicate the training data.
Heuristic filtering
We adopt heuristic methods to filter the data, some of which are listed as
follows:
• All: we remove the documents containing fewer than 20 tokens.
14

Data Collection
De-duplication
Heuristic
Filtering
Topic-based
Text Recall
Decontamination
Model-based
Quality Scoing
        Math Doc
         Competition
         Code Doc
         Math CoT
          Program
          Generated
          OSS-Instruct
          Science CoT
Mathematics
Coding
Science
        Formal
         Mathematics
Data Filtering Pipeline
Synthetic Generation of Reasoning Data
Figure 8: Illustration of our data filtering pipeline and synthetic generation for reasoning data. The
filtering pipeline consists of six steps starting from data collection. Synthetic data generation includes
both pretraining data (above the horizontal line) and instruction data (below the line).
• Code: we apply filtering criteria based on code metrics (e.g., average line length, alphabetic
characters ratio, and keyword statistics) similar to DeepSeek-Coder [Guo et al., 2024].
• Synthetic data: we remove responses that are garbled or contain repeated content. For math
texts, we remove response that do not contain an hightlited answer part (e.g., $box{}$).
Topic-based text recall
To enhance the model’s capabilities in specialized areas, it is essential to
include ample knowledge documents related to mathematics, code, and reasoning. For this purpose,
we extract relevant documents from unused web pages by training fasttext [Bojanowski et al.,
2017] and TinyBert [Jiao et al., 2020] classifiers specifically tailored to these categories. From the
FineWeb-Edu [Lozhkov et al., 2024a] and DCLM [Li et al., 2024b] web corpus, we extract 10.4B
math text tokens, 1.11B code text tokens, and 1.01B reasoning text tokens. which are directly used for
training or serve as seed data for synthesizing instruction data. Furthermore, we reuse the synthesized
science data (1.5B) from Llama-3-SynE [Chen et al., 2024], which covers an extensive range of
disciplines, such as math and physics.
Model-based quality scoring
For general web page data and mathematical pre-training data, we
use the fineweb-edu-scorer released by FineWeb-Edu for data scoring. For Python code data, we
use the python-edu-scorer released by FineWeb-Edu. To avoid language models favoring highly
technical pages like arXiv abstracts and submitted papers, these two classifiers focus on knowledge at
the elementary and middle school levels. Following the methodology of Penedo et al. [2024], we
conduct quality assessments on all Python code data, most mathematical data, and web page data
using scoring tools. We exclude data with scores of 1 and 2 and then heuristically sort data with
scores from 3 to 5 (as detailed in Section 4.5).
Decontamination
To ensure the fairness of comparison, we perform decontamination based on the
selected evaluation benchmarks. Initially, we tokenize both the training set and the benchmarks that
require decontamination, such as GSM8K [Cobbe et al., 2021], MATH [Hendrycks et al., 2021b],
HumanEval [Chen et al., 2021], and ARC [Yadav et al., 2019]. Next, we divide all the benchmarks
using n-gram tokens to create a contamination set. We use tokens rather than words to form n-gram
segment, which achieves a higher level of decontamination in the domains of mathematics and code.
Additionally, we exclude 20-gram segments that occur more than four times, as they are typically
not relevant to the questions or solutions. Ultimately, the contamination set comprises 1,917,428
tuples. For each training document, if more than 10% of its generated 20-grams are present in the
contamination set, we exclude that document from the final pre-training set.
15

4.3
Synthetic Generation of Reasoning Data
Reasoning is widely recognized as one of the most desirable capabilities for LLMs [Huang and Chang,
2023]. However, unlike basic skills such as knowledge memorization, reasoning is more challenging
to achieve and enhance in LLMs. One potential reason for this difficulty is the scarcity of high-quality
texts containing logical or complex reasoning in real-world datasets. To address this limitation, a
common approach is to generate data samples related to reasoning, such as chain-of-thought data,
which focus on demonstrating the thought processes likely to lead to the final solution rather than
directly producing question-answer pairs [Wei et al., 2022]. To enrich diversity and coverage, we
consider a broad range of domains for synthetic data generation, as introduced below.
4.3.1
Math Reasoning
Mathematical reasoning data is relatively scarce in pre-training corpora, yet it is crucial for enhancing
model capabilities [Wei et al., 2022]. To address this gap, we generate a diverse range of mathematical
reasoning data, including documents, instructions, and formal mathematics data.
Mathematical documents
For mathematical documents, we generate descriptive content spanning
various difficulty levels and thematic styles. It includes explanations of mathematical concepts and
science education materials for primary school levels, as well as lecture scripts, tutorials, educational
articles, and problem sets suitable for high school and college-level content. We primarily source
math-related seed data from mathematical pre-training corpora, such as OpenWebMath [Paster et al.,
2024], and from a self-compiled math dataset using classifiers described in Section 4.2.
Chain-of-thought reasoning
For instructional data, we employ three approaches for text synthesis.
First, we use Qwen2.5-Math-7B-Instruct [Yang et al., 2024b] to generate thought processes (e.g.,
chain-of-thought) for existing problems found in open-source datasets, such as Orca-Math [Mitra et al.,
2024b], MetaMathQA [Yu et al., 2024], AMPS-Math [Hendrycks et al., 2021b], and NuminaMath [LI
et al., 2024]. Second, following the method of JiuZhang3.0 [Zhou et al., 2024b], we utilize a
finetuned Qwen2-Math-7B-Instruct [Ding et al., 2024b] to automatically generate new math
problems (without the thought process) and then annotate the solutions in the same manner as the
first approach. To obtain more extensive thought processes, we select more challenging data, such as
from the NuminaMath dataset, and utilize slow-thinking model QwQ-32B-Preview [Qwen-Team,
2024] for distillation. We obtain the long-form thought data from our o1-reproduction project “Slow
thinking with LLMs” [Jiang et al., 2024, Min et al., 2024]. This data aims to enhance the mathematical
capacities of our base model.
Formal mathematical reasoning
We also incorporate reasoning data from formal mathematics,
such as formal theorem proving, which has been shown to enhance performance on general mathemat-
ical tests like GSM8K and MATH. Specifically, we collect Lean tactic (e.g., intro, simp) dataset like
DeepSeek-Prover [Xin et al., 2024] and its associated prover states to train the model in generating
proof tactics. For the Lean GitHub dataset [Wu et al., 2024], we concatenate together the reasoning
steps in each document to form a long reasoning chain. Additionally, inspired by LIME [Wu et al.,
2021], we augment the Lean Workbook [Ying et al., 2024] datasets with three reasoning primitives:
(1) Deduction: Statebefore, Tactic →Stateafter; (2) Abduction: Stateafter, Tactic →Statebefore; and (3)
Induction: Statebefore, Stateafter →Tactic. Rather than directly predicting the next proof tactics, we
train the model to predict the previous or next state based on the proof tactics.
Program generated numerical reasoning
LLMs perform reasoning in natural language, which is
flexible but does not allow for the verification of the correctness and necessity of each step. To enhance
the model’s basic numerical abilities, we select subsets of addition, subtraction, multiplication,
division, and remainder operations from the DeepMind-Math dataset [Saxton et al., 2019]. Our
goal is to transform simple mathematical expressions (e.g., “What is 0.079 - 162?”) into a
corresponding calculation procedure consisting of individual steps. This enables the model to learn
calculations in a manner similar to chain-of-thought reasoning. This approach is mainly applicable to
simple and limited mathematical computations and is suitable only for the early and middle stages of
training. Given that manually writing conversion code is time-consuming, we also utilize an agentic
framework for the automatic generation of code.
16

4.3.2
Code Reasoning
For code reasoning, we primarily synthesize two types of data: programming competition problems
and real-world programming tasks. We also generate long-form reasoning thought data with slow-
thinking models.
Competition code synthesis through ICL
To enhance existing programming competition datasets,
like those from LeetCode8, we utilize the in-context learning (ICL) method by leveraging a small
number of demonstrations. This approach allows us to generate additional examples, thereby
expanding and diversifying the data. By introducing a broader range of challenging programming
problems, we enrich the dataset significantly.
OSS-Instruct
Additionally, we generate real-world programming tasks and their corresponding
solutions using the OSS-Instruct method, as detailed in previous work [Wei et al., 2024]. This process
is guided by carefully crafted prompts (see Appendix C), ensuring that the generated tasks are both
relevant and applicable to real-world scenarios. This approach significantly enhances the practical
utility of the dataset.
4.3.3
Scientific Reasoning
Scientific reasoning is crucial for expanding the capabilities of LLMs. To acquire scientific reasoning
data, we consider the following two approaches.
Scientific chain-of-thought reasoning
In Section 4.2, we train a scientific classifier to extract
documents related to various scientific fields from the web page data of FineWeb-Edu and DCLM.
We utilize these scientific data as the seed data, and apply a synthesis method similar to that used for
generating math reasoning data to create science-related questions and answers [Chen et al., 2024].
Scientific problems with slow-thinking processes
We gather more difficult scientific questions
from college entrance examinations and camel-ai, covering subjects such as physics, chemistry, and
biology. Then we use QwQ-32B-Preview to answer questions and obtain pairs of answers to difficult
scientific data questions.
4.3.4
Reflection
An important aspect of reasoning ability is the capacity to reflect on and backtrack from the current
state [Shinn et al., 2023]. In this work, we explore the generation of reflection data to further enhance
the model’s reasoning capabilities. First, we sample mathematical problems and collect both positive
and negative responses by comparing them to the golden label. We then employ a powerful model,
Qwen2.5-Math-7B-Instruct, to identify the first error in the negative response and truncate the
content that follows. Instead of merely concatenating the positive response with the truncated negative
one, we create error analyses and transitional statements to seamlessly and effectively connect the
two responses.
The prompts used for data synthesis are provided in Appendix C.
4.4
Data Mix
Our training process is divided into three main stages: warmup, stable training, and annealing.
During the warmup and stable training stages, the dataset composition is as follows: 60% general
English data (consisting of 45% from web and 15% from books, papers and other relevant sources),
20% code data, 10% math data, and 10% general Chinese data. In the later stage of stable training,
a small amount (<5%) of instruction data is introduced. We maintain a relatively consistent data
distribution across different curriculum phases, and will slightly adjust the data proportions according
to the the perplexity performance of the model on various benchmarks. We strive to avoid large shifts
in data distribution, as significant changes can cause the loss to spike suddenly. To ensure training
stability and account for testing inaccuracies, the change in data distribution between two consecutive
phases is kept within 3%.
8https://huggingface.co/datasets/greengerong/leetcode
17

1
3
0
20
40
60
80
100
%
11
13
15
17
19
21
23
25
27
General-Pretrain
General-SFT
Math-Pretrain
Math-SFT
Code-Pretrain
Code-SFT
Web
Chinese
Curriculum Phase
Figure 9: The data mixture proportion of math, code, and general data. We keep the proportion of
web data unchanged in stable stage, and then gradually decrease it in annealing stage. The entire
process is divided into into three major stages: warmup, stable training, and annealing (i.e., beginning
after the dashed line).
During the annealing stage, the proportion of instruction data is increased to 19.19% in total: code-
related instruction data constitutes approximately 11%, math-related instruction data accounts for
about 7%, and general instruction data amounts to around 1%. Additionally, the proportion of long
context data (i.e., those exceeding 8K tokens) is also increased, occupying approximately 14.21% of
the tokens used.
4.5
Data Curriculum
Following previous studies [Kim and Lee, 2024, Chen et al., 2024], we adopt a curriculum-
based method to prepare training data throughout the process. We use quality classifiers, such
as fineweb-edu-scorer, to evaluate content based on educational difficulty, assigning higher
scores to text content suitable for primary and secondary school levels. Manual inspection reveals
that, for math and code pre-training data, lower scores often indicate higher difficulty. Conversely,
for English webpage data, we find that staged training based on educational level scores significantly
impacts the original distribution. Therefore, we train on math and code data in order of increasing
difficulty, but do not apply curriculum learning to webpage data.
Throughout the entire training process, we continuously monitor the model’s performance, as detailed
in Section 6.3. We save a checkpoint and conduct an evaluation each 4B training tokens. For each
40B tokens, we reassess and adjust the data ratio when transitioning between training phases based
on the model’s overall performance in that phase. For example, if the model’s performance on
the HumanEval benchmark does not improve or declines after a stage, we may consider slightly
increasing the amount of code data in the subsequent stage.
Additionally, to help the model adapt to a relatively high proportion of instruction data in the annealing
stage, we gradually increase the amount of instruction data. However, throughout the entire stable
training stage, this proportion will not exceed 5%.
We present the data distribution scheduled in the training curriculum in Figure 9, and the detailed
data composition for each phase are presented in Appendix E.
18

5
Annealing
As demonstrated in previous studies [Hu et al., 2024], the annealing stage is particularly effective in
boosting model performance. We also implement effective training strategies during the annealing
stage, which are described below.
5.1
Optimization Setting
In the annealing stage, we improve the performance of the model by using a high-quality data set and
equip the model with long context processing capacities. Thus, we mainly consider the two settings,
i.e., learning rate annealing and context window extension.
Learning rate annealing
Since we use the WSD scheduler, in the annealing stage, the learning
rate will gradually decrease from that in the stable training stage. We need to select an appropriate
learning rate annealing function. We investigate the performance of several typical learning rate
annealing functions [Hägele et al., 2024], such as linear annealing, cosine annealing, 1-sqrt annealing.
We empirically find that 1-sqrt performs the best, so we choose 1-sqrt as our learning rate annealing
function, which is defined as follows:
f(n; N, Nannealing) =
 
1 −
s
n −(N −Nannealing)
Nannealing
!
,
where n, N and Nannealing denote the current number of steps, the total number of steps and the
number of annealing steps, respectively.
Following the work by Hu et al. [2024], we estimate the optimal annealing ratio to be 8%, i.e., 80
billion tokens. We maintain the same batch size used during stable training, i.e., 4 million tokens.
The learning rate is decreased from 10−2 to 5.22 × 10−5 over a span of 18,802 steps. Subsequently,
the learning rate is held constant at 5.22 × 10−5 for the final 772 steps.
Context Window Extension
Previous research [Chen et al., 2023] has demonstrated that LLMs can
hardly process texts exceeding their context windows due to the out-of-distribution (OOD) rotation
angles in RoPE. To achieve the context window extension, increasing the base frequency of RoPE
to migrate the OOD rotation angles and continual pre-training has been an effective method [Xiong
et al., 2024]. Consequently, during the annealing stage, we increase the base frequency of RoPE θ
from 10,000, employed during stable training, to 490,000 and train the model on long texts. This
adjustment successfully extends the context length from 4,096 (4K) tokens to 28,672 (28K) tokens.
5.2
Data Selection for Annealing Stage
It is particularly important to select high-quality data during the annealing stage [Hu et al., 2024]. As
the learning rate decreases, the model can rapidly enhance its performance by leveraging high-quality
data. We consider selecting high-quality data for the annealing stage using the following methods:
• We include a variety of high-quality data sources, particularly synthetic reasoning data
discussed in Section 4.3.
• We use a gradient-based data selection method, which accelerates and improves the LESS
method [Xia et al., 2024], combining the method InsTag [Lu et al., 2024] for constructing a
diversified target set.
In particular, we incorporate formal mathematical reasoning (theorem proving in Lean) and advanced
reasoning data (o1-like thought data) to improve the model’s performance on challenging math
benchmarks, e.g., MATH-500, which have been shown in Table 6.
We present the final data composition for the annealing stage in Table 5.
5.3
Long Context Training
During the annealing stage of the final 80B tokens, we adjust the base frequency of RoPE from
10,000 to 490,000 and train on long sequences to extend the context length from 4,096 tokens to
19

Table 5: Detailed information of the training data in the annealing stage.
Domain
Type
Dataset
Volume
Mix
Pretrain
FineWeb-Edu, CBook, arXiv
64.65B
Math
(1) CoT
Deepmind-Math, MathInstruct
3.07B
(2) Long CoT
Numina, AMPS, Platypus
0.61B
(3) Formal math
Lean-GitHub, Lean-WorkBook, DeepSeek-Prover-V1
0.10B
(4) Curated
Tulu v3, MathInstruct
1.42B
Code
(1) CoT
OSS-Instruct (seed: the-Stack-v2), OpenCoder-LLM
6.66B
(2) Curated
LeetCode, XCoder-80K
2.39B
Science
(1) Long CoT
Camel-ai
0.04B
(2) Curated
EvolKit-20k, Celestia, Supernova
1.06B
Total
-
-
80B
28,672 tokens. We avoid training with long contexts in earlier stages because the computational
cost of self-attention layers increases quadratically with sequence length, making it prohibitively
expensive [Dubey et al., 2024].
When training on long contexts, we observe a decline in the model’s performance on short-text
benchmarks. To enhance the long-text capacities and preserve the short-text capacities, we carefully
design the mixture of data. We upample books and concatenated GitHub code texts [Liu et al.,
2024b] as long context data to capture long-term dependencies, while using high-quality short texts
to preserve short-text capabilities. Additionally, inspired by previous studies Ding et al. [2024a], Gao
et al. [2024], we also apply masked cross-document attention that prevents attention across different
documents to preserve short-context capabilities.
5.4
Other Strategies
Packing
Since the training data during the annealing stage includes some instruction data, using a
traditional simple packing method for pre-training data could result in instruction data being split,
thereby compromising its effectiveness. To address this, we propose a packing strategy designed
to maintain training efficiency while minimizing the disruption of instruction data. This strategy
involves different packing methods based on data type. Pre-training data is directly spliced, whereas
for instruction data, if it is divided into two sequences, the remaining part of the previous sequence is
padded directly, and this instruction data serves as the beginning of the second sequence. Subsequently,
any redundant padding tokens are replaced with pre-training data tokens. By including the instruction
data, our main goal is to learn the reasoning process rather than focusing solely on the question-and-
answer format. Therefore, we employ the same data processing method used in pre-training, which
directly includes question-answer pairs without relying on a chat template. When calculating the loss,
the instruction and response are treated as a single document, and the loss for the instruction is not
masked.
Checkpoint merging
Following the approach used in LLaMA3 [Dubey et al., 2024], we combine
the last few checkpoints during the annealing stage to produce the final pre-trained model. While this
strategy might result in a slight reduction in certain specific capabilities (e.g., GSM8K), it generally
leads to a more well-rounded model.
6
Evaluation
In this section, we conduct the evaluation experiments to verify the effectiveness of our base model
YuLan-Mini. We first set up the experiments for evaluation, and then present the experiment results.
20

6.1
Experimental Setup
6.1.1
Evaluation Benchmarks
For a comprehensive evaluation of LLMs performance, we select the benchmarks from the following
aspects.
• Language
comprehension:
We
select
the
widely-used
English
benchmarks
MMLU [Hendrycks et al., 2021a], LAMBADA [Kazemi et al., 2023] and RACE [Lai et al.,
2017], along with the Chinese benchmarks CMMLU [Li et al., 2024a] and CEval [Huang
et al., 2023], to evaluate the bilingual comprehension capabilities of the LLM. These
benchmarks span various domains, such as history, science, and culture.
• Code generation: We select Humaneval [Chen et al., 2021] and MBPP [Austin et al., 2021]
to assess the capability of LLMs to generate accurate code snippets for natural language
problems.
• Mathematical reasoning:
We utilize GSM8K [Cobbe et al., 2021] and MATH-
500 [Hendrycks et al., 2021b, Lightman et al., 2024] to evaluate the mathematical rea-
soning capabilities of LLMs. These benchmarks range from basic arithmetic to advanced
mathematical problems.
• Logical reasoning: We assess the logical reasoning capabilities of LLMs using ARC-
E [Yadav et al., 2019], ARC-C [Yadav et al., 2019], which provide a comprehensive
evaluation of logical reasoning across various knowledge domains.
• Commonsense reasoning:
We evaluate the LLM’s commonsense reasoning ability
using WinoGrande [Sakaguchi et al., 2021], HellaSwag [Zellers et al., 2019], Sto-
ryCloze [Mostafazadeh et al., 2016] which test the understanding and utilization of daily
commonsense knowledge.
• Long context understanding: We employ RULER [Hsieh et al., 2024] to evaluate the long
context understanding ability, which measures its performance change as the sequence
length increases. We perform evaluations of applicable models within a context length of
28K tokens.
6.1.2
Baseline Models
To ensure a comprehensive evaluation, we select several small LLMs with comparable scales (i.e.,
base models ranging from 0.5 to 3B, including embedding sizes) as baselines for comparison:
• MiniCPM-2.4B [Hu et al., 2024]: MiniCPM-2.4B is pre-trained on 1.06T tokens and also
employs the annealing training strategy. Despite its small size, it exhibits impressive
performance in general tasks while supporting deployments with limited hardware resource.
• Qwen series models [Qwen-Team, 2024, Yang et al., 2024a]: We select Qwen2-1.5B,
Qwen2.5-0.5B, and Qwen2.5-1.5B for comparison. The Qwen series of small LLMs have
been pre-trained on 18T tokens, and the training details have not been fully publicly released.
They demonstrate strong performance in both general and domain-specific tasks.
• StableLM2-1.6B [Bellagente et al., 2024]: StableLM2-1.6B is a small LLM proposed by
StabilityAI. It has been pre-trained on a mixture of open-source datasets, which utilizes
several small LLMs to determine the training data proportion.
• SmolLM2-1.7B [Allal et al., 2024]: SmolLM2-1.7B is developed by HuggingFace TB
Research based on its collected high-quality pre-training corpus, which has been trained on
11T tokens, and maintains a good balance between speed and accuracy.
• Llama3.2-3B [Dubey et al., 2024]: Llama3.2-3B is developed by MetaAI, which is trained
on up to 9T tokens. It further distills the knowledge from LLaMA3.1-8B and 70B models
by using their logits during the pretraining stage.
• Gemma2-2.6B [Team, 2024]: Gemma2-2.6B is developed by Google, which is trained on
2T tokens, mainly including web documents, code, and mathematical text.
21

Table 6: Performance on math, code, and long context benchmarks. Results marked with * are
cited from their official paper or report. The best and second best results are bold and underlined,
respectively.
Models
Model
Size
# Train
Tokens
Context
Length
MATH
500
GSM
8K
Human
Eval
MBPP
RACE
Middle
RACE
High
RULER
MiniCPM
2.6B
1.06T
4K
15.00
53.83
50.00∗
47.31
56.61
44.27
N/A
Qwen-2
1.5B
7T
128K
22.60
46.90∗
34.80∗
46.90∗
55.77
43.69
60.16
Qwen2.5
0.5B
18T
128K
23.60
41.60∗
30.50∗
39.30∗
52.36
40.31
49.23
Qwen2.5
1.5B
18T
128K
45.40
68.50∗
37.20∗
60.20∗
58.77
44.33
68.26
Gemma2
2.6B
2T
8K
18.30∗
30.30∗
19.50∗
42.10∗
-
-
N/A
StableLM2
1.7B
2T
4K
-
20.62
8.50
17.50
56.33
45.06
N/A
SmolLM2
1.7B
11T
8K
11.80
-
23.35
45.00
55.77
43.06
N/A
Llama3.2
3.2B
9T
128K
7.40
-
29.30
49.70
55.29
43.34
77.06
YuLan-Mini
2.4B
1.04T
4K
32.60
66.65
61.60
66.70
55.71
43.58
N/A
2.4B
1.08T
28K
37.80
68.46
64.00
65.90
57.18
44.57
51.48
6.1.3
Implementation Details
To comprehensively compare the performance of different LLMs, we employ diverse evaluation
settings and design specific methods for guaranteeing the fairness and efficiency.
• Zero-shot and few-shot settings: Following existing work [Qwen-Team, 2024], For LAM-
BADA, HumanEval, MBPP, RACE, StoryCloze and RULER, we adopt the zero-shot setting.
For GSM8K and MATH, we adopt the 4-shot setting. For MMLU, CMMLU, WinoGrande
and CEval, we adopt the 5-shot setting. For HellaSwag, we adopt the 10-shot setting. For
ARC-E, ARC-C, we adopt the 25-shot setting.
• Chain-of-Thought (CoT): For GSM8K and MATH, we follow previous work [Qwen-Team,
2024] that uses CoT prompting to facilitate the LLM to perform step-by-step reasoning.
Considering the potential performance variance caused by CoT prompts, we utilize both the
short ones provided by the original dataset and the long ones generated by kimi-k0-math.
For each model, we evaluate the performance using both prompt types, and select the one
yielding the higher score as the result.
• Evaluation metrics: For QA tasks, we employ maj@1 for GSM8K and MATH, pass@1
for HumanEval and MBPP, and accuracy of the model response for remaining generation
tasks. For multiple-choice questions, we primarily evaluate based on the accuracy of the
generated answer, which is determined by selecting the choice with the lowest perplexity.
However, for ARC-E and ARC-C, we utilize normalized accuracy [Brown et al., 2020]. To
accurately measure the performance of MATH-500, we further use gpt-4o-mini to verify
the correctness of the results generated by all models and conducted manual checks.
• Maximum length: For GSM8K and MATH, since CoT prompting may result in longer
outputs, we set the maximum generation length to 596 for short context (i.e., 4K) models
and 2,048 for long context models. For HumanEval and MBPP, we set the maximum
generation length to 400. For other generative tasks, we set it to 128 for efficiency.
• Evaluation framework: For the majority of tasks, we employ LLMBox [Tang et al., 2024b] to
assess performance. Specifically, for generation tasks, we enable vLLM [Kwon et al., 2023].
However, to ensure reproducibility, we utilize EvalPlus [Liu et al., 2024a] for HumanEval
and MBPP.
Despite our considerable efforts, fully reproducing the results of these baseline models as originally
reported remains challenging, due to the lack of detailed evaluation setup information. For a fair
comparison, we report the performance results of the baselines as provided in their official technical
reports.
22

Table 7: Performance on commonsense reasoning benchmarks. Results marked with * are cited from
their official paper or report.
Models
LAMBADA
MMLU
CMMLU
CEval
Hella
Swag
Wino
Grande
Story
Cloze
ARC-e
ARC-c
MiniCPM-2.6B
61.91
53.37
48.97
48.24
67.92
65.74
78.51
55.51
43.86
Qwen2-1.5B
64.68
55.90
70.76
71.94
66.11
66.14
77.60
62.21
42.92
Qwen2.5-0.5B
52.00
47.50
52.17
54.27
50.54
55.88
71.67
56.10
39.51
Qwen2.5-1.5B
62.12
60.71
67.82
69.05
67.18
64.48
76.80
71.51
53.41
Gemma2-2.6B
-
52.20∗
-
28.00∗
74.60∗
71.50∗
-
-
55.70∗
StableLM2-1.7B
66.15
40.37
29.29
26.99
69.79
64.64
78.56
54.00
40.78
SmolLM2-1.7B
67.42
51.91
33.46
35.10
72.96
67.40
79.32
44.82
35.49
Llama3.2-3B
69.08
63.40
44.44
44.49
75.62
67.48
76.80
70.12
48.81
YuLan-Mini
64.72
51.79
48.35
51.47
68.65
67.09
76.37
69.87
50.51
65.67
49.10
45.45
48.23
67.22
67.24
75.89
67.47
49.32
6.2
Main Results
The experimental results of different models on the specified benchmarks are shown in Table 6 and
Table 7. Furthermore, we have selected 4 benchmarks from each of the two tables to construct
Figure 1. Based on these results, we can identify the following key observations:
Superior training efficacy
Overall, YuLan-Mini achieves highly competitive performance com-
pared to leading small industry models, despite being trained on a significantly smaller corpus (1.08T
tokens). To ensure successful pre-training with relatively limited data, we meticulously designed the
data pipeline, effectively mitigated training instability, and implemented annealing training. Addi-
tionally, most of our training data comes from open-source and synthetic datasets, demonstrating that
with careful data cleaning, selection, and scheduling, we can develop a robust base model even with
limited resources in a university-level laboratory setting. This highlights the superior data efficiency
of our pre-training approach.
Excellence in mathematical and coding
On specific benchmarks for mathematical reasoning
(MATH-500 and GSM8K) and coding generation (HumanEval and MBPP), YuLan-mini achieves
leading average performance. This consistent superiority can be mainly attributed to the use of
high-quality pre-training corpus and reasoning synthetic data (e.g., formal mathematics reasoning
problems and o1-like long thought data). Our core idea is to extend the types of reasoning data and
enhance the complex reasoning capacities of our base model, which leads to large improvements on
mathematical benchmarks.
Strong general capability
Beyond specialized tasks, YuLan-mini also demonstrates strong per-
formance on various general benchmarks, spanning from language modeling and commonsense
reasoning, highlighting the versatility of the model. It indicates that our pre-training approach well
balances the learning of diverse abilities, resulting in a robust general-purpose foundation model.
This success can be attributed to our data mixture and curriculum-based adjustment strategies, which
carefully balance the initial proportion of math, coding, and general knowledge related corpora, while
gradually enhancing them during the annealing stage to better develop advanced capabilities.
Moderate long context capability
Due to limited computing resources, YuLan-Mini has had
limited exposure to long context training samples. As a result, its ability to model long contexts is
not yet on par with state-of-the-art LLMs, as reflected in the results from the RULER benchmark.
Additionally, due to a lack of GPU resources, our current approach can only extend the context
window up to 28K. This limitation could be addressed with additional resources to support further
development.
6.3
Evaluating Model Performance during Pre-Training
During pre-training, it is crucial to continuously evaluate the model’s performance to monitor
for any unstable or abnormal training issues. However, existing benchmarks rely on advanced
23

0
100
200
300
400
500
Trained Tokens (B)
0.0
2.5
5.0
7.5
10.0
12.5
15.0
17.5
20.0
HumanEval (0-shot, pass@k=1)
HumanEval (0-shot, pass@k=1)
Code PPL
1.0
1.2
1.4
1.6
1.8
2.0
Code PPL
(a) Performance curve on HumanEval.
0
100
200
300
400
500
Trained Tokens (B)
2
4
6
8
10
12
14
16
GSM-8K(8-shot)
GSM-8K(8-shot)
Math PPL
1.2
1.4
1.6
1.8
2.0
2.2
2.4
2.6
Math PPL
(b) Performance curve on GSM8K.
Figure 10: Performance comparison using perplexity (PPL) and accuracy-based metrics to monitor
the code generation and math reasoning abilities of YuLan-Mini.
abilities (e.g., instruction following), which often develop with sufficient data training. Thus, the
model’s performance tends to remain at a low level on these benchmarks in the early stages, and
directly evaluating the model’s performance on specific validation sets would not provide an accurate
assessment.
To address this, we have designed two monitoring strategies for different stages of training. In the early
stages, we assess the model’s performance primarily through perplexity measures on the constructed
validation datasets and LAMBADA benchmark. In the later stages, we shift to using performance on
selected benchmarks (e.g., HumanEval and GSM8K) for more comprehensive evaluation. Next, we
introduce how to construct the validation set for perplexity measurement at early stage of pre-training.
To comprehensively evaluate the key abilities of our model, we create four validation sets from the
following aspects, namely English understanding, Chinese understanding, code generation, and math
reasoning. The detailed data composition is as follows.
• English understanding: We randomly select 2,118 samples from FineWeb-Edu and compute
the perplexity for ability evaluation.
• Chinese understanding: We randomly select 1,679 samples from Chinese-FineWeb-Edu for
computing the perplexity.
• Code generation: We randomly select 2,067 samples from a widely-used code instruction
datasets, Python-Code-Instructions-18k-Alpaca for perplexity evaluation.9
• Math reasoning: We randomly sample 1,499 open-ended questions from MathInstruct [Yue
et al., 2024] for perplexity.
Once the advanced capabilities are well-developed, we can directly monitor the model’s performance
by evaluating it on the selected benchmarks.
7
Conclusion
In this paper, we have introduced YuLan-Mini, a highly capable base model with 2.42B parameters.
We provide all the essential technical details and resources necessary to reproduce our model, includ-
ing improvements to the training method for enhanced stability and efficiency, as well as tokenized
data sources organized with a specially developed training curriculum. Extensive experiments have
demonstrated the effectiveness of YuLan-Mini, showing that it achieves performance comparable
to its industry counterparts of a similar parameter scale. Our primary contribution is enabling the
reproduction of competitive language models in a highly data-efficient manner, making it feasible
9https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca
24

for university-level laboratories. Additionally, by aligning the training data with intermediate check-
points, our approach facilitates in-depth research on LLMs, such as exploring how model capacities
develop during pre-training.
In future work, we plan to release an instruct version of YuLan-Mini. Furthermore, we aim to
extend YuLan-Mini to other architectures and training methods and also explore its specialization in
professional domains (e.g., math and coding).
Acknowledgment
We sincerely thank Weizheng Lu and Xu Han for their assistance with this work. We encourage more
researchers to collaborate in uncovering the pre-training secrets of LLMs.
References
Gretel AI. Synthetically generated reasoning dataset (gsm8k-inspired) with enhanced diversity using gretel
navigator and meta-llama/meta-llama-3.1-405b. https://huggingface.co/gretelai/synthetic-gsm8k-reflection-
405b, 9 2024.
Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai.
GQA: training generalized multi-query transformer models from multi-head checkpoints. In Houda Bouamor,
Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 4895–4901. Association for
Computational Linguistics, 2023. doi: 10.18653/V1/2023.EMNLP-MAIN.298. URL https://doi.org/
10.18653/v1/2023.emnlp-main.298.
Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Lewis Tunstall, Agustín Piqueres,
Andres Marafioti, Cyril Zakka, Leandro von Werra, and Thomas Wolf. Smollm2 - with great data, comes
great performance, 2024.
Jacob Austin, Augustus Odena, Maxwell I. Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen
Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le, and Charles Sutton. Program synthesis with large language
models. CoRR, abs/2108.07732, 2021. URL https://arxiv.org/abs/2108.07732.
Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen Marcus McAleer, Albert Q.
Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language model for mathematics. In
The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11,
2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=4WnqRR915j.
Lei Jimmy Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. CoRR, abs/1607.06450, 2016.
URL http://arxiv.org/abs/1607.06450.
Edward Beeching, Shengyi Costa Huang, Albert Jiang, Jia Li, Benjamin Lipkin, Zihan Qina, Kashif Rasul, Ziju
Shen, Roman Soletskyi, and Lewis Tunstall. Numinamath 7b cot, 2024. URL http://faculty.bicmr.
pku.edu.cn/~dongbin/Publications/numina_dataset.pdf.
Marco Bellagente, Jonathan Tow, Dakota Mahan, Duy Phung, Maksym Zhuravinskyi, Reshinth Adithyan,
James Baicoianu, Ben Brooks, Nathan Cooper, Ashish Datta, Meng Lee, Emad Mostaque, Michael Pieler,
Nikhil Pinnaparaju, Paulo Rocha, Harry Saini, Hannah Teufel, Niccoló Zanichelli, and Carlos Riquelme.
Stable LM 2 1.6b technical report. CoRR, abs/2402.17834, 2024. doi: 10.48550/ARXIV.2402.17834. URL
https://doi.org/10.48550/arXiv.2402.17834.
Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong,
Qiushi Du, Zhe Fu, Huazuo Gao, Kaige Gao, Wenjun Gao, Ruiqi Ge, Kang Guan, Daya Guo, Jianzhong
Guo, Guangbo Hao, Zhewen Hao, Ying He, Wenjie Hu, Panpan Huang, Erhang Li, Guowei Li, Jiashi Li,
Yao Li, Y. K. Li, Wenfeng Liang, Fangyun Lin, Alex X. Liu, Bo Liu, Wen Liu, Xiaodong Liu, Xin Liu,
Yiyuan Liu, Haoyu Lu, Shanghao Lu, Fuli Luo, Shirong Ma, Xiaotao Nie, Tian Pei, Yishi Piao, Junjie Qiu,
Hui Qu, Tongzheng Ren, Zehui Ren, Chong Ruan, Zhangli Sha, Zhihong Shao, Junxiao Song, Xuecheng
Su, Jingxiang Sun, Yaofeng Sun, Minghui Tang, Bingxuan Wang, Peiyi Wang, Shiyu Wang, Yaohui Wang,
Yongji Wang, Tong Wu, Y. Wu, Xin Xie, Zhenda Xie, Ziwei Xie, Yiliang Xiong, Hanwei Xu, R. X. Xu,
Yanhong Xu, Dejian Yang, Yuxiang You, Shuiping Yu, Xingkai Yu, B. Zhang, Haowei Zhang, Lecong Zhang,
Liyue Zhang, Mingchuan Zhang, Minghua Zhang, Wentao Zhang, Yichao Zhang, Chenggang Zhao, Yao
Zhao, Shangyan Zhou, Shunfeng Zhou, Qihao Zhu, and Yuheng Zou. Deepseek LLM: scaling open-source
language models with longtermism. CoRR, abs/2401.02954, 2024. doi: 10.48550/ARXIV.2401.02954. URL
https://doi.org/10.48550/arXiv.2401.02954.
25

Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. Enriching word vectors with subword
information. Transactions of the Association for Computational Linguistics, 5:135–146, 2017. ISSN
2307-387X.
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen
Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter,
Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark,
Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models
are few-shot learners, 2020. URL https://arxiv.org/abs/2005.14165.
Jie Chen, Zhipeng Chen, Jiapeng Wang, Kun Zhou, Yutao Zhu, Jinhao Jiang, Yingqian Min, Wayne Xin Zhao,
Zhicheng Dou, Jiaxin Mao, Yankai Lin, Ruihua Song, Jun Xu, Xu Chen, Rui Yan, Zhewei Wei, Di Hu,
Wenbing Huang, and Ji-Rong Wen. Towards effective and efficient continual pre-training of large language
models. CoRR, abs/2407.18743, 2024. doi: 10.48550/ARXIV.2407.18743. URL https://doi.org/10.
48550/arXiv.2407.18743.
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, Jared Kaplan, Harri
Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael
Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov,
Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such,
Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen
Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain,
William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan
Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob
McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language
models trained on code. CoRR, abs/2107.03374, 2021. URL https://arxiv.org/abs/2107.03374.
Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large
language models via positional interpolation. CoRR, abs/2306.15595, 2023. doi: 10.48550/ARXIV.2306.
15595. URL https://doi.org/10.48550/arXiv.2306.15595.
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul
Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha
Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prab-
hakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael
Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk
Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito,
David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani
Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor
Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang,
Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck,
Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways. J. Mach. Learn.
Res., 24:240:1–240:113, 2023. URL https://jmlr.org/papers/v24/22-1144.html.
Woojin Chung, Jiwoo Hong, Na Min An, James Thorne, and Se-Young Yun. Stable language model pre-training
by reducing embedding variability. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors,
Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024,
Miami, FL, USA, November 12-16, 2024, pages 10852–10863. Association for Computational Linguistics,
2024. URL https://aclanthology.org/2024.emnlp-main.606.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert,
Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to
solve math word problems. CoRR, abs/2110.14168, 2021. URL https://arxiv.org/abs/2110.14168.
Gautier Dagan, Gabriel Synnaeve, and Baptiste Rozière. Getting the most out of your tokenizer for pre-
training and domain adaptation. In Forty-first International Conference on Machine Learning, ICML 2024,
Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=
ZFYBnLljtT.
Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. In The Twelfth
International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. Open-
Review.net, 2024. URL https://openreview.net/forum?id=mZn2Xyh9Ec.
Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-
efficient exact attention with io-awareness. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave,
26

K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Confer-
ence on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, Novem-
ber 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/
67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html.
Alexandre de Brébisson and Pascal Vincent. The z-loss: a shift and scale invariant classification loss belonging
to the spherical family. CoRR, abs/1604.08859, 2016. URL http://arxiv.org/abs/1604.08859.
DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Deng, Chong
Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fuli Luo, Guangbo
Hao, Guanting Chen, Guowei Li, Hao Zhang, Hanwei Xu, Hao Yang, Haowei Zhang, Honghui Ding, Huajian
Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jin Chen, Jingyang
Yuan, Junjie Qiu, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Lean Wang, Lecong Zhang, Lei Xu,
Leyi Xia, Liang Zhao, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui
Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qihao Zhu, Qinyu Chen, Qiushi
Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruizhe Pan, Runxin Xu, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan
Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu,
Shunfeng Zhou, Size Zheng, Tao Wang, Tian Pei, Tian Yuan, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wei
An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi,
Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaosha Chen, Xiaotao Nie, and Xiaowen Sun.
Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. CoRR, abs/2405.04434,
2024. doi: 10.48550/ARXIV.2405.04434. URL https://doi.org/10.48550/arXiv.2405.04434.
Nolan Dey, Gurpreet Gosal, Zhiming Chen, Hemant Khachane, William Marshall, Ribhu Pathria, Marvin
Tom, and Joel Hestness. Cerebras-gpt: Open compute-optimal language models trained on the cerebras
wafer-scale cluster. CoRR, abs/2304.03208, 2023a. doi: 10.48550/ARXIV.2304.03208. URL https:
//doi.org/10.48550/arXiv.2304.03208.
Nolan Dey, Gurpreet Gosal, Zhiming, Chen, Hemant Khachane, William Marshall, Ribhu Pathria, Marvin
Tom, and Joel Hestness. Cerebras-GPT: Open Compute-Optimal Language Models Trained on the Cerebras
Wafer-Scale Cluster, April 2023b. URL http://arxiv.org/abs/2304.03208. arXiv:2304.03208 [cs].
Hantian Ding, Zijian Wang, Giovanni Paolini, Varun Kumar, Anoop Deoras, Dan Roth, and Stefano Soatto.
Fewer truncations improve language modeling. In Forty-first International Conference on Machine Learning,
ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024a. URL https://openreview.net/
forum?id=kRxCDDFNpp.
Yuyang Ding, Xinyu Shi, Xiaobo Liang, Juntao Li, Qiaoming Zhu, and Min Zhang. Unleashing Reasoning
Capability of LLMs via Scalable Question Synthesis from Scratch, October 2024b. URL http://arxiv.
org/abs/2410.18693. arXiv:2410.18693 [cs].
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi
Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurélien Rodriguez,
Austen Gregerson, Ava Spataru, Baptiste Rozière, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte
Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret,
Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song,
Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego
Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael
Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme
Nail, Grégoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo
Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet,
Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer
Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao
Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia,
Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, and
et al. The llama 3 herd of models. CoRR, abs/2407.21783, 2024. doi: 10.48550/ARXIV.2407.21783. URL
https://doi.org/10.48550/arXiv.2407.21783.
Tianyu Gao, Alexander Wettig, Howard Yen, and Danqi Chen. How to train long-context language models
(effectively). CoRR, abs/2410.02660, 2024. doi: 10.48550/ARXIV.2410.02660. URL https://doi.org/
10.48550/arXiv.2410.02660.
Dirk Groeneveld, Iz Beltagy, Evan Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh
Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khy-
athi Raghavi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, Tushar
Khot, William Merrill, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E.
27

Peters, Valentina Pyatkin, Abhilasha Ravichander, Dustin Schwenk, Saurabh Shah, Will Smith, Emma
Strubell, Nishant Subramani, Mitchell Wortsman, Pradeep Dasigi, Nathan Lambert, Kyle Richardson,
Luke Zettlemoyer, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah A. Smith, and Hannaneh Hajishirzi.
Olmo: Accelerating the science of language models.
In Lun-Wei Ku, Andre Martins, and Vivek
Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 15789–
15809. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.ACL-LONG.841. URL
https://doi.org/10.18653/v1/2024.acl-long.841.
Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y. Wu,
Y. K. Li, Fuli Luo, Yingfei Xiong, and Wenfeng Liang. Deepseek-coder: When the large language model meets
programming - the rise of code intelligence. CoRR, abs/2401.14196, 2024. doi: 10.48550/ARXIV.2401.14196.
URL https://doi.org/10.48550/arXiv.2401.14196.
Alexander Hägele, Elie Bakouch, Atli Kosson, Loubna Ben Allal, Leandro von Werra, and Martin Jaggi. Scaling
laws and compute-optimal training beyond fixed training durations. CoRR, abs/2405.18392, 2024. doi:
10.48550/ARXIV.2405.18392. URL https://doi.org/10.48550/arXiv.2405.18392.
Xiaotian Han, Yiren Jian, Xuefeng Hu, Haogeng Liu, Yiqi Wang, Qihang Fan, Yuang Ai, Huaibo Huang, Ran
He, Zhenheng Yang, and Quanzeng You. Infimm-webmath-40b: Advancing multimodal pre-training for
enhanced mathematical reasoning. CoRR, abs/2409.12568, 2024. doi: 10.48550/ARXIV.2409.12568. URL
https://doi.org/10.48550/arXiv.2409.12568.
Conghui He, Zhenjiang Jin, Chao Xu, Jiantao Qiu, Bin Wang, Wei Li, Hang Yan, Jiaqi Wang, and Dahua Lin.
Wanjuan: A comprehensive multimodal dataset for advancing english and chinese large models. CoRR,
abs/2308.10755, 2023. doi: 10.48550/ARXIV.2308.10755. URL https://doi.org/10.48550/arXiv.
2308.10755.
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Stein-
hardt. Measuring massive multitask language understanding. In 9th International Conference on Learn-
ing Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021a. URL
https://openreview.net/forum?id=d7KBjmI3GmQ.
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song,
and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Joaquin
Vanschoren and Sai-Kit Yeung, editors, Proceedings of the Neural Information Processing Systems
Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, vir-
tual, 2021b. URL https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/
be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html.
Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang,
and Boris Ginsburg. RULER: what’s the real context size of your long-context language models? CoRR,
abs/2404.06654, 2024. doi: 10.48550/ARXIV.2404.06654. URL https://doi.org/10.48550/arXiv.
2404.06654.
Pin-Lun Hsu, Yun Dai, Vignesh Kothapalli, Qingquan Song, Shao Tang, Siyu Zhu, Steven Shimizu, Shivam
Sahni, Haowen Ning, and Yanning Chen. Liger kernel: Efficient triton kernels for LLM training. CoRR,
abs/2410.10989, 2024. doi: 10.48550/ARXIV.2410.10989. URL https://doi.org/10.48550/arXiv.
2410.10989.
Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang
Huang, Weilin Zhao, Xinrong Zhang, Zhen Leng Thai, Kai Zhang, Chongyi Wang, Yuan Yao, Chenyang
Zhao, Jie Zhou, Jie Cai, Zhongwu Zhai, Ning Ding, Chao Jia, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and
Maosong Sun. Minicpm: Unveiling the potential of small language models with scalable training strategies.
CoRR, abs/2404.06395, 2024. doi: 10.48550/ARXIV.2404.06395. URL https://doi.org/10.48550/
arXiv.2404.06395.
Jie Huang and Kevin Chen-Chuan Chang. Towards reasoning in large language models: A survey. In Anna
Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki, editors, Findings of the Association for Computational
Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023, pages 1049–1065. Association for Computational
Linguistics, 2023. doi: 10.18653/V1/2023.FINDINGS-ACL.67. URL https://doi.org/10.18653/v1/
2023.findings-acl.67.
Siming Huang, Tianhao Cheng, J. K. Liu, Jiaran Hao, Liuyihan Song, Yang Xu, J. Yang, J. H. Liu, Chenchen
Zhang, Linzheng Chai, Ruifeng Yuan, Zhaoxiang Zhang, Jie Fu, Qian Liu, Ge Zhang, Zili Wang, Yuan Qi,
Yinghui Xu, and Wei Chu. OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models.
CoRR, abs/2411.04905, 2024. doi: 10.48550/ARXIV.2308.10755. URL https://doi.org/10.48550/
arXiv.2411.04905.
28

Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng
Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, and Junxian He. C-eval: A multi-level multi-discipline
chinese evaluation suite for foundation models. In Advances in Neural Information Processing Systems, 2023.
Jinhao Jiang, Zhipeng Chen, Yingqian Min, Jie Chen, Xiaoxue Cheng, Jiapeng Wang, Yiru Tang, Haoxiang
Sun, Jia Deng, Wayne Xin Zhao, Zheng Liu, Dong Yan, Jian Xie, Zhongyuan Wang, and Ji-Rong Wen.
Technical report: Enhancing llm reasoning with reward-guided tree search. CoRR, abs/2411.11694, 2024.
doi: 10.48550/ARXIV.2411.11694. URL https://doi.org/10.48550/arXiv.2411.11694.
Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. TinyBERT:
Distilling BERT for Natural Language Understanding, October 2020. URL http://arxiv.org/abs/1909.
10351. Issue: arXiv:1909.10351 1097 citations (Semantic Scholar/arXiv) [2023-07-31] arXiv:1909.10351
[cs].
Marek Kadlcík, Michal Stefánik, Ondrej Sotolár, and Vlastimil Martinek. Calc-x and calcformers: Em-
powering arithmetical chain-of-thought through interaction with symbolic systems. In Houda Bouamor,
Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in
Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 12101–12108.
Association for Computational Linguistics, 2023.
doi: 10.18653/V1/2023.EMNLP-MAIN.742.
URL
https://doi.org/10.18653/v1/2023.emnlp-main.742.
J.
Kaplan,
Sam
McCandlish,
T.
Henighan,
Tom
B.
Brown,
Benjamin
Chess,
Rewon
Child,
Scott Gray,
Alec Radford,
Jeff Wu,
and Dario Amodei.
Scaling Laws for Neu-
ral
Language
Models.
ArXiv,
January
2020.
URL
https://www.semanticscholar.
org/paper/Scaling-Laws-for-Neural-Language-Models-Kaplan-McCandlish/
e6c561d02500b2596a230b341a8eb8b921ca5bf2.
Mehran Kazemi, Najoung Kim, Deepti Bhatia, Xin Xu, and Deepak Ramachandran. LAMBADA: backward
chaining for automated reasoning in natural language.
In Anna Rogers, Jordan L. Boyd-Graber, and
Naoaki Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 6547–6568.
Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.ACL-LONG.361. URL https:
//doi.org/10.18653/v1/2023.acl-long.361.
Jisu Kim and Juhwan Lee. Strategic data ordering: Enhancing large language model performance through
curriculum learning. CoRR, abs/2405.07490, 2024. doi: 10.48550/ARXIV.2405.07490. URL https:
//doi.org/10.48550/arXiv.2405.07490.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao
Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention.
In Jason Flinn, Margo I. Seltzer, Peter Druschel, Antoine Kaufmann, and Jonathan Mace, editors, Proceedings
of the 29th Symposium on Operating Systems Principles, SOSP 2023, Koblenz, Germany, October 23-26,
2023, pages 611–626. ACM, 2023. doi: 10.1145/3600006.3613165. URL https://doi.org/10.1145/
3600006.3613165.
Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. Race: Large-scale reading comprehension
dataset from examinations, 2017. URL https://arxiv.org/abs/1704.04683.
Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester
James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D.
Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith,
Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. Tulu 3: Pushing frontiers in open language
model post-training. CoRR, abs/2411.15124, 2024. doi: 10.48550/ARXIV.2411.15124. URL https:
//doi.org/10.48550/arXiv.2411.15124.
Joonhyung Lee, Jeongin Bae, Byeongwook Kim, Se Jung Kwon, and Dongsoo Lee. To FP8 and back again:
Quantifying the effects of reducing precision on LLM training stability. CoRR, abs/2405.18710, 2024. doi:
10.48550/ARXIV.2405.18710. URL https://doi.org/10.48550/arXiv.2405.18710.
Conglong Li, Minjia Zhang, and Yuxiong He. The stability-efficiency dilemma: Investigating sequence
length warmup for training GPT models. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave,
K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Confer-
ence on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, Novem-
ber 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/
aac02401755a65904cf977a33136af4a-Abstract-Conference.html.
29

Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin.
CMMLU: measuring massive multitask language understanding in chinese. In Lun-Wei Ku, Andre Martins,
and Vivek Srikumar, editors, Findings of the Association for Computational Linguistics, ACL 2024, Bangkok,
Thailand and virtual meeting, August 11-16, 2024, pages 11260–11285. Association for Computational
Linguistics, 2024a. doi: 10.18653/V1/2024.FINDINGS-ACL.671. URL https://doi.org/10.18653/
v1/2024.findings-acl.671.
Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Yitzhak Gadre, Hritik Bansal, Etash Ku-
mar Guha, Sedrick Keh, Kushal Arora, Saurabh Garg, Rui Xin, Niklas Muennighoff, Reinhard Heckel, Jean
Mercat, Mayee Chen, Suchin Gururangan, Mitchell Wortsman, Alon Albalak, Yonatan Bitton, Marianna
Nezhurina, Amro Abbas, Cheng-Yu Hsieh, Dhruba Ghosh, Josh Gardner, Maciej Kilian, Hanlin Zhang, Rulin
Shao, Sarah M. Pratt, Sunny Sanyal, Gabriel Ilharco, Giannis Daras, Kalyani Marathe, Aaron Gokaslan,
Jieyu Zhang, Khyathi Raghavi Chandu, Thao Nguyen, Igor Vasiljevic, Sham M. Kakade, Shuran Song, Sujay
Sanghavi, Fartash Faghri, Sewoong Oh, Luke Zettlemoyer, Kyle Lo, Alaaeldin El-Nouby, Hadi Pouransari,
Alexander Toshev, Stephanie Wang, Dirk Groeneveld, Luca Soldaini, Pang Wei Koh, Jenia Jitsev, Thomas Kol-
lar, Alexandros G. Dimakis, Yair Carmon, Achal Dave, Ludwig Schmidt, and Vaishaal Shankar. Datacomp-lm:
In search of the next generation of training sets for language models. CoRR, abs/2406.11794, 2024b. doi:
10.48550/ARXIV.2406.11794. URL https://doi.org/10.48550/arXiv.2406.11794.
Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul,
Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample,
and Stanislas Polu. Numinamath. [https://github.com/project-numina/aimo-progress-prize]
(https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_
dataset.pdf), 2024.
Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc
Marone, Christopher Akiki, Jia Li, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas
Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João Monteiro, Oleh Shliazhko, Nicolas
Gontier, Nicholas Meade, Armel Zebaze, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Benjamin
Lipkin, Muhtasham Oblokulov, Zhiruo Wang, Rudra Murthy V, Jason T. Stillerman, Siva Sankalp Patel,
Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Nour Fahmy, Urvashi Bhattacharyya, Wenhao
Yu, Swayam Singh, Sasha Luccioni, Paulo Villegas, Maxim Kunakov, Fedor Zhdanov, Manuel Romero, Tony
Lee, Nadav Timor, Jennifer Ding, Claire Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra,
Alex Gu, Jennifer Robinson, Carolyn Jane Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy,
Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf, Arjun
Guha, Leandro von Werra, and Harm de Vries. Starcoder: may the source be with you! Trans. Mach. Learn.
Res., 2023, 2023. URL https://openreview.net/forum?id=KoFOg41haE.
Wing Lian, Guan Wang, Bleys Goodson, Eugene Pentland, Austin Cook, Chanvichet Vong, and "Teknium".
Slimorca: An open dataset of gpt-4 augmented flan reasoning traces, with verification, 2023. URL https:
//https://huggingface.co/Open-Orca/SlimOrca.
Xinyu Lian, Sam Ade Jacobs, Lev Kurilenko, Masahiro Tanaka, Stas Bekman, Olatunji Ruwase, and Minjia
Zhang. Universal checkpointing: Efficient and flexible checkpointing for large scale distributed training.
CoRR, abs/2406.18820, 2024. doi: 10.48550/ARXIV.2406.18820. URL https://doi.org/10.48550/
arXiv.2406.18820.
Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John
Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference
on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL
https://openreview.net/forum?id=v8L0pN6EOi.
Haohan Lin, Zhiqing Sun, Yiming Yang, and Sean Welleck. Lean-star: Learning to interleave thinking and
proving. CoRR, abs/2407.10040, 2024. doi: 10.48550/ARXIV.2407.10040. URL https://doi.org/10.
48550/arXiv.2407.10040.
Jiawei Liu, Songrun Xie, Junhao Wang, Yuxiang Wei, Yifeng Ding, and Lingming Zhang. Evaluating language
models for efficient code generation. In First Conference on Language Modeling, 2024a. URL https:
//openreview.net/forum?id=IBCBMeAhmC.
Xiaoran Liu, Kai Lv, Qipeng Guo, Hang Yan, Conghui He, Xipeng Qiu, and Dahua Lin. Longwanjuan:
Towards systematic measurement for long text quality. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung
Chen, editors, Findings of the Association for Computational Linguistics: EMNLP 2024, Miami, Florida,
USA, November 12-16, 2024, pages 5709–5725. Association for Computational Linguistics, 2024b. URL
https://aclanthology.org/2024.findings-emnlp.327.
30

Zechun Liu, Changsheng Zhao, Forrest N. Iandola, Chen Lai, Yuandong Tian, Igor Fedorov, Yunyang Xiong,
Ernie Chang, Yangyang Shi, Raghuraman Krishnamoorthi, Liangzhen Lai, and Vikas Chandra. Mobilellm:
Optimizing sub-billion parameter language models for on-device use cases. In Forty-first International
Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024c.
URL https://openreview.net/forum?id=EIGbXbxcUQ.
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In 7th International Conference on
Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. URL
https://openreview.net/forum?id=Bkg6RiCqY7.
Anton Lozhkov, Loubna Ben Allal, Leandro von Werra, and Thomas Wolf. Fineweb-edu, May 2024a. URL
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu.
Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang,
Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, Tianyang Liu, Max Tian, Denis Kocetkov, Arthur Zucker, Younes
Belkada, Zijian Wang, Qian Liu, Dmitry Abulkhanov, Indraneil Paul, Zhuang Li, Wen-Ding Li, Megan
Risdal, Jia Li, Jian Zhu, Terry Yue Zhuo, Evgenii Zheltonozhskii, Nii Osae Osae Dade, Wenhao Yu, Lucas
Krauß, Naman Jain, Yixuan Su, Xuanli He, Manan Dey, Edoardo Abati, Yekun Chai, Niklas Muennighoff,
Xiangru Tang, Muhtasham Oblokulov, Christopher Akiki, Marc Marone, Chenghao Mou, Mayank Mishra,
Alex Gu, Binyuan Hui, Tri Dao, Armel Zebaze, Olivier Dehaene, Nicolas Patry, Canwen Xu, Julian J.
McAuley, Han Hu, Torsten Scholak, Sébastien Paquet, Jennifer Robinson, Carolyn Jane Anderson, Nicolas
Chapados, and et al. Starcoder 2 and the stack v2: The next generation. CoRR, abs/2402.19173, 2024b. doi:
10.48550/ARXIV.2402.19173. URL https://doi.org/10.48550/arXiv.2402.19173.
Keming Lu, Hongyi Yuan, Zheng Yuan, Runji Lin, Junyang Lin, Chuanqi Tan, Chang Zhou, and Jingren
Zhou. #instag: Instruction tagging for analyzing supervised fine-tuning of large language models. In The
Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024.
OpenReview.net, 2024. URL https://openreview.net/forum?id=pszewhybU9.
Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane
Rivière, Mihir Sanjay Kale, Juliette Love, Pouya Tafti, Léonard Hussenot, Aakanksha Chowdhery, Adam
Roberts, Aditya Barua, Alex Botev, Alex Castro-Ros, Ambrose Slone, Amélie Héliou, Andrea Tacchetti,
Anna Bulanova, Antonia Paterson, Beth Tsai, Bobak Shahriari, Charline Le Lan, Christopher A. Choquette-
Choo, Clément Crepy, Daniel Cer, Daphne Ippolito, David Reid, Elena Buchatskaya, Eric Ni, Eric Noland,
Geng Yan, George Tucker, George-Cristian Muraru, Grigory Rozhdestvenskiy, Henryk Michalewski, Ian
Tenney, Ivan Grishchenko, Jacob Austin, James Keeling, Jane Labanowski, Jean-Baptiste Lespiau, Jeff
Stanway, Jenny Brennan, Jeremy Chen, Johan Ferret, Justin Chiu, and et al. Gemma: Open models based on
gemini research and technology. CoRR, abs/2403.08295, 2024. doi: 10.48550/ARXIV.2403.08295. URL
https://doi.org/10.48550/arXiv.2403.08295.
Yingqian Min, Zhipeng Chen, Jinhao Jiang, Jie Chen, Jia Deng, Yiwen Hu, Yiru Tang, Jiapeng Wang,
Xiaoxue Cheng, Huatong Song, Wayne Xin Zhao, Zheng Liu, Zhongyuan Wang, and Ji-Rong Wen.
Imitate, explore, and self-improve: A reproduction report on slow-thinking reasoning systems. arXiv
preprint arXiv:2412.09413, abs/2412.09413, 2024.
doi: 10.48550/ARXIV.2412.09413.
URL https:
//doi.org/10.48550/arXiv.2412.09413.
Arindam Mitra, Luciano Del Corro, Guoqing Zheng, Shweti Mahajan, Dany Rouhana, Andrés Codas, Yadong
Lu, Weige Chen, Olga Vrousgos, Corby Rosset, Fillipe Silva, Hamed Khanpour, Yash Lara, and Ahmed
Awadallah. Agentinstruct: Toward generative teaching with agentic flows. CoRR, abs/2407.03502, 2024a.
doi: 10.48550/ARXIV.2407.03502. URL https://doi.org/10.48550/arXiv.2407.03502.
Arindam Mitra, Hamed Khanpour, Corby Rosset, and Ahmed Awadallah. Orca-math: Unlocking the potential
of slms in grade school math. CoRR, abs/2402.14830, 2024b. doi: 10.48550/ARXIV.2402.14830. URL
https://doi.org/10.48550/arXiv.2402.14830.
Igor Molybog, Peter Albert, Moya Chen, Zachary DeVito, David Esiobu, Naman Goyal, Punit Singh Koura,
Sharan Narang, Andrew Poulton, Ruan Silva, Binh Tang, Diana Liskovich, Puxin Xu, Yuchen Zhang, Melanie
Kambadur, Stephen Roller, and Susan Zhang. A theory on adam instability in large-scale machine learning.
CoRR, abs/2304.09871, 2023. doi: 10.48550/ARXIV.2304.09871. URL https://doi.org/10.48550/
arXiv.2304.09871.
Nasrin Mostafazadeh, Nathanael Chambers, Xiaodong He, Devi Parikh, Dhruv Batra, Lucy Vanderwende, Push-
meet Kohli, and James Allen. A corpus and evaluation framework for deeper understanding of commonsense
stories, 2016. URL https://arxiv.org/abs/1604.01696.
31

Kosuke Nishida, Kyosuke Nishida, and Kuniko Saito. Initialization of large language models via reparameteriza-
tion to mitigate loss spikes. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Proceedings
of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL,
USA, November 12-16, 2024, pages 22699–22714. Association for Computational Linguistics, 2024. URL
https://aclanthology.org/2024.emnlp-main.1264.
OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023. doi: 10.48550/ARXIV.2303.08774. URL
https://doi.org/10.48550/arXiv.2303.08774.
Opencsg.
Opencsg/chinese-fineweb-edu
·
Datasets
at
Hugging
Face.
https://huggingface.co/datasets/opencsg/chinese-fineweb-edu.
Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. Openwebmath: An open dataset of
high-quality mathematical web text. In The Twelfth International Conference on Learning Representations,
ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/
forum?id=jKHmjlpViu.
Guilherme Penedo, Hynek Kydlíˇcek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel,
Leandro Von Werra, and Thomas Wolf. The FineWeb Datasets: Decanting the Web for the Finest Text Data
at Scale, October 2024. URL http://arxiv.org/abs/2406.17557. arXiv:2406.17557.
Ofir Press and Lior Wolf. Using the output embedding to improve language models. In Mirella Lapata, Phil
Blunsom, and Alexander Koller, editors, Proceedings of the 15th Conference of the European Chapter of the
Association for Computational Linguistics, EACL 2017, Valencia, Spain, April 3-7, 2017, Volume 2: Short
Papers, pages 157–163. Association for Computational Linguistics, 2017. doi: 10.18653/V1/E17-2025. URL
https://doi.org/10.18653/v1/e17-2025.
Ivan Provilkov, Dmitrii Emelianenko, and Elena Voita. Bpe-dropout: Simple and effective subword regularization.
In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault, editors, Proceedings of the 58th Annual
Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020, pages 1882–
1892. Association for Computational Linguistics, 2020. doi: 10.18653/V1/2020.ACL-MAIN.170. URL
https://doi.org/10.18653/v1/2020.acl-main.170.
Qwen-Team. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm.github.io/
blog/qwen2.5/.
Qwen-Team. Qwq: Reflect deeply on the boundaries of the unknown, November 2024. URL https://qwenlm.
github.io/blog/qwq-32b-preview/.
Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: memory optimizations toward
training trillion parameter models. In Christine Cuicchi, Irene Qualters, and William T. Kramer, editors,
Proceedings of the International Conference for High Performance Computing, Networking, Storage and
Analysis, SC 2020, Virtual Event / Atlanta, Georgia, USA, November 9-19, 2020, page 20. IEEE/ACM, 2020.
doi: 10.1109/SC41405.2020.00024. URL https://doi.org/10.1109/SC41405.2020.00024.
Oleg Rybakov, Mike Chrzanowski, Peter Dykas, Jinze Xue, and Ben Lanir. Methods of improving LLM training
stability. CoRR, abs/2410.16682, 2024. doi: 10.48550/ARXIV.2410.16682. URL https://doi.org/10.
48550/arXiv.2410.16682.
Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: an adversarial
winograd schema challenge at scale. Commun. ACM, 64(9):99–106, 2021. doi: 10.1145/3474381. URL
https://doi.org/10.1145/3474381.
David Saxton, Edward Grefenstette, Felix Hill, and Pushmeet Kohli. Analysing mathematical reasoning abilities
of neural models. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA,
USA, May 6-9, 2019. OpenReview.net, 2019. URL https://openreview.net/forum?id=H1gR5iR5FX.
Teven Le Scao, Thomas Wang, Daniel Hesslow, Lucile Saulnier, Stas Bekman, M. Saiful Bari, Stella Biderman,
Hady Elsahar, Niklas Muennighoff, Jason Phang, Ofir Press, Colin Raffel, Victor Sanh, Sheng Shen, Lintang
Sutawika, Jaesung Tae, Zheng Xin Yong, Julien Launay, and Iz Beltagy. What Language Model to Train
if You Have One Million GPU Hours?, November 2022. URL http://arxiv.org/abs/2210.15424.
arXiv:2210.15424 [cs].
Noam Shazeer. GLU variants improve transformer. CoRR, abs/2002.05202, 2020. URL https://arxiv.org/
abs/2002.05202.
32

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: lan-
guage agents with verbal reinforcement learning. In Alice Oh, Tristan Naumann, Amir Globerson, Kate
Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems
36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA,
USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/
1b44b878bb782e6954cd888628510e90-Abstract-Conference.html.
Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro.
Megatron-lm: Training multi-billion parameter language models using model parallelism, 2020. URL
https://arxiv.org/abs/1909.08053.
Damien Sileo. Scaling synthetic logical reasoning datasets with context-sensitive declarative grammars. In Yaser
Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Proceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, pages
5275–5283. Association for Computational Linguistics, 2024. URL https://aclanthology.org/2024.
emnlp-main.301.
Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben
Bogin, Khyathi Raghavi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Harsh Jha,
Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighoff,
Aakanksha Naik, Crystal Nam, Matthew E. Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang
Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Pete Walsh, Luke Zettlemoyer, Noah A. Smith,
Hannaneh Hajishirzi, Iz Beltagy, Dirk Groeneveld, Jesse Dodge, and Kyle Lo. Dolma: an open corpus
of three trillion tokens for language model pretraining research. In Lun-Wei Ku, Andre Martins, and
Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 15725–
15788. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.ACL-LONG.840. URL
https://doi.org/10.18653/v1/2024.acl-long.840.
Yiding Sun, Feng Wang, Yutao Zhu, Wayne Xin Zhao, and Jiaxin Mao. An integrated data processing framework
for pretraining foundation models. In Grace Hui Yang, Hongning Wang, Sam Han, Claudia Hauff, Guido
Zuccon, and Yi Zhang, editors, Proceedings of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval, SIGIR 2024, Washington DC, USA, July 14-18, 2024, pages 2713–2718.
ACM, 2024. doi: 10.1145/3626772.3657671. URL https://doi.org/10.1145/3626772.3657671.
Sho Takase, Shun Kiyono, Sosuke Kobayashi, and Jun Suzuki. Spike no more: Stabilizing the pre-training of
large language models. CoRR, abs/2312.16903, 2023. doi: 10.48550/ARXIV.2312.16903. URL https:
//doi.org/10.48550/arXiv.2312.16903.
Liping Tang, Nikhil Ranjan, Omkar Pangarkar, Xuezhi Liang, Zhen Wang, Li An, Bhaskar Rao, Linghao Jin,
Huijuan Wang, Zhoujun Cheng, Suqi Sun, Cun Mu, Victor Miller, Xuezhe Ma, Yue Peng, Zhengzhong Liu,
and Eric P. Xing. Txt360: A top-quality llm pre-training dataset requires the perfect blend, 2024a.
Tianyi Tang, Hu Yiwen, Bingqian Li, Wenyang Luo, ZiJing Qin, Haoxiang Sun, Jiapeng Wang, Shiyi Xu,
Xiaoxue Cheng, Geyang Guo, Han Peng, Bowen Zheng, Yiru Tang, Yingqian Min, Yushuo Chen, Jie
Chen, Ranchi Zhao, Luran Ding, Yuhao Wang, Zican Dong, Xia Chunxuan, Junyi Li, Kun Zhou, Xin
Zhao, and Ji-Rong Wen. LLMBox: A Comprehensive Library for Large Language Models. In Yixin
Cao, Yang Feng, and Deyi Xiong, editors, Proceedings of the 62nd Annual Meeting of the Association
for Computational Linguistics (Volume 3: System Demonstrations), pages 388–399, Bangkok, Thailand,
2024b. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-demos.37. URL https:
//aclanthology.org/2024.acl-demos.37.
Zhengyang Tang, Xingxing Zhang, Benyou Wang, and Furu Wei. Mathscale: Scaling instruction tuning
for mathematical reasoning. In Forty-first International Conference on Machine Learning, ICML 2024,
Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024c. URL https://openreview.net/forum?id=
Kjww7ZN47M.
Gemma Team. Gemma. 2024. doi: 10.34740/KAGGLE/M/3301. URL https://www.kaggle.com/m/3301.
Kushal Tirumala, Daniel Simig, Armen Aghajanyan, and Ari Morcos. D4: improving LLM pretraining
via document de-duplication and diversification. In Alice Oh, Tristan Naumann, Amir Globerson, Kate
Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems
36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA,
USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/
a8f8cbd7f7a5fb2c837e578c75e5b615-Abstract-Datasets_and_Benchmarks.html.
Howe Tissue, Venus Wang, and Lu Wang. Scaling law with learning rate annealing. CoRR, abs/2408.11029,
2024. doi: 10.48550/ARXIV.2408.11029. URL https://doi.org/10.48550/arXiv.2408.11029.
33

Shubham Toshniwal, Ivan Moshkov, Sean Narenthiran, Daria Gitman, Fei Jia, and Igor Gitman.
Openmathinstruct-1: A 1.8 million math instruction tuning dataset. CoRR, abs/2402.10176, 2024. doi:
10.48550/ARXIV.2402.10176. URL https://doi.org/10.48550/arXiv.2402.10176.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Bap-
tiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave,
and Guillaume Lample. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971,
2023. doi: 10.48550/arXiv.2302.13971. URL https://doi.org/10.48550/arXiv.2302.13971.
Dixuan Wang, Yanda Li, Junyuan Jiang, Zepeng Ding, Guochao Jiang, Jiaqing Liang, and Deqing Yang.
Tokenization matters! degrading large language models through challenging their tokenization. CoRR,
abs/2405.17067, 2024a. doi: 10.48550/ARXIV.2405.17067. URL https://doi.org/10.48550/arXiv.
2405.17067.
Ke Wang, Houxing Ren, Aojun Zhou, Zimu Lu, Sichun Luo, Weikang Shi, Renrui Zhang, Linqi Song, Mingjie
Zhan, and Hongsheng Li. Mathcoder: Seamless code integration in llms for enhanced mathematical reasoning.
In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11,
2024. OpenReview.net, 2024b. URL https://openreview.net/forum?id=z8TW0ttBPp.
Yejie Wang, Keqing He, Dayuan Fu, Zhuoma Gongque, Heyang Xu, Yanxu Chen, Zhexu Wang, Yujia Fu,
Guanting Dong, Muxi Diao, Jingang Wang, Mengdi Zhang, Xunliang Cai, and Weiran Xu. How do your code
llms perform? empowering code instruction tuning with really good data. In Yaser Al-Onaizan, Mohit Bansal,
and Yun-Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language
Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, pages 14027–14043. Association for
Computational Linguistics, 2024c. URL https://aclanthology.org/2024.emnlp-main.777.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le,
and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Sanmi Koyejo,
S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information
Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022,
New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_
files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html.
Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, and Lingming Zhang. Magicoder: Empowering code
generation with oss-instruct. In Forty-first International Conference on Machine Learning, ICML 2024,
Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=
XUeoOBid3x.
Mitchell Wortsman, Peter J. Liu, Lechao Xiao, Katie E. Everett, Alexander A. Alemi, Ben Adlam, John D.
Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, Jeffrey Pennington, Jascha Sohl-Dickstein, Kelvin
Xu, Jaehoon Lee, Justin Gilmer, and Simon Kornblith. Small-scale proxies for large-scale transformer
training instabilities. In The Twelfth International Conference on Learning Representations, ICLR 2024,
Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=
d8w0pmvXbZ.
Yuhuai Wu, Markus N. Rabe, Wenda Li, Jimmy Ba, Roger B. Grosse, and Christian Szegedy. LIME: learning
inductive bias for primitives of mathematical reasoning. In Marina Meila and Tong Zhang, editors, Proceedings
of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event,
volume 139 of Proceedings of Machine Learning Research, pages 11251–11262. PMLR, 2021. URL
http://proceedings.mlr.press/v139/wu21c.html.
Zijian Wu, Jiayu Wang, Dahua Lin, and Kai Chen.
Lean-github: Compiling github LEAN repositories
for a versatile LEAN prover. CoRR, abs/2407.17227, 2024. doi: 10.48550/ARXIV.2407.17227. URL
https://doi.org/10.48550/arXiv.2407.17227.
Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. LESS: selecting
influential data for targeted instruction tuning. In Forty-first International Conference on Machine Learning,
ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. URL https://openreview.net/
forum?id=PG5fV50maR.
Huajian Xin, Daya Guo, Zhihong Shao, Zhizhou Ren, Qihao Zhu, Bo Liu, Chong Ruan, Wenda Li, and Xiaodan
Liang. Deepseek-prover: Advancing theorem proving in llms through large-scale synthetic data. CoRR,
abs/2405.14333, 2024. doi: 10.48550/ARXIV.2405.14333. URL https://doi.org/10.48550/arXiv.
2405.14333.
34

Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan,
Liwei Wang, and Tie-Yan Liu. On layer normalization in the transformer architecture. In Proceedings
of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event,
volume 119 of Proceedings of Machine Learning Research, pages 10524–10533. PMLR, 2020. URL
http://proceedings.mlr.press/v119/xiong20b.html.
Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang, Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi
Rungta, Karthik Abinav Sankararaman, Barlas Oguz, Madian Khabsa, Han Fang, Yashar Mehdad, Sharan
Narang, Kshitiz Malik, Angela Fan, Shruti Bhosale, Sergey Edunov, Mike Lewis, Sinong Wang, and Hao
Ma. Effective long-context scaling of foundation models. In Kevin Duh, Helena Gómez-Adorno, and Steven
Bethard, editors, Proceedings of the 2024 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), NAACL 2024, Mexico
City, Mexico, June 16-21, 2024, pages 4643–4663. Association for Computational Linguistics, 2024. doi: 10.
18653/V1/2024.NAACL-LONG.260. URL https://doi.org/10.18653/v1/2024.naacl-long.260.
Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, and Bill Yuchen
Lin. Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing. CoRR,
abs/2406.08464, 2024. doi: 10.48550/ARXIV.2406.08464. URL https://doi.org/10.48550/arXiv.
2406.08464.
Vikas Yadav, Steven Bethard, and Mihai Surdeanu. Quick and (not so) dirty: Unsupervised selection of
justification sentences for multi-hop question answering. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun
Wan, editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing
and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong
Kong, China, November 3-7, 2019, pages 2578–2589. Association for Computational Linguistics, 2019. doi:
10.18653/V1/D19-1260. URL https://doi.org/10.18653/v1/D19-1260.
Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian
Wang, Dong Yan, Fan Yang, Fei Deng, Feng Wang, Feng Liu, Guangwei Ai, Guosheng Dong, Haizhou
Zhao, Hang Xu, Haoze Sun, Hongda Zhang, Hui Liu, Jiaming Ji, Jian Xie, Juntao Dai, Kun Fang, Lei
Su, Liang Song, Lifeng Liu, Liyun Ru, Luyao Ma, Mang Wang, Mickel Liu, MingAn Lin, Nuolan Nie,
Peidong Guo, Ruiyang Sun, Tao Zhang, Tianpeng Li, Tianyu Li, Wei Cheng, Weipeng Chen, Xiangrong
Zeng, Xiaochuan Wang, Xiaoxi Chen, Xin Men, Xin Yu, Xuehai Pan, Yanjun Shen, Yiding Wang, Yiyu
Li, Youxin Jiang, Yuchen Gao, Yupeng Zhang, Zenan Zhou, and Zhiying Wu. Baichuan 2: Open large-
scale language models. CoRR, abs/2309.10305, 2023. doi: 10.48550/ARXIV.2309.10305. URL https:
//doi.org/10.48550/arXiv.2309.10305.
An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li,
Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang,
Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang
Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng
Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao
Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei,
Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu
Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report. CoRR, abs/2407.10671, 2024a.
doi: 10.48550/ARXIV.2407.10671. URL https://doi.org/10.48550/arXiv.2407.10671.
An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu,
Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru
Zhang. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. CoRR,
abs/2409.12122, 2024b. doi: 10.48550/ARXIV.2409.12122. URL https://doi.org/10.48550/arXiv.
2409.12122.
Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub
Pachocki, Weizhu Chen, and Jianfeng Gao. Tensor programs V: tuning large neural networks via zero-
shot hyperparameter transfer. CoRR, abs/2203.03466, 2022. doi: 10.48550/ARXIV.2203.03466. URL
https://doi.org/10.48550/arXiv.2203.03466.
Greg Yang, Dingli Yu, Chen Zhu, and Soufiane Hayou. Tensor programs VI: feature learning in infinite
depth neural networks. In The Twelfth International Conference on Learning Representations, ICLR 2024,
Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024c. URL https://openreview.net/forum?id=
17pVDnpwwl.
Huaiyuan Ying, Zijian Wu, Yihan Geng, Jiayu Wang, Dahua Lin, and Kai Chen. Lean workbook: A large-scale
lean problem set formalized from natural language math problems. CoRR, abs/2406.03847, 2024. doi:
10.48550/ARXIV.2406.03847. URL https://doi.org/10.48550/arXiv.2406.03847.
35

Andy B. Yoo, Morris A. Jette, and Mark Grondona. SLURM: simple linux utility for resource management. In
Dror G. Feitelson, Larry Rudolph, and Uwe Schwiegelshohn, editors, Job Scheduling Strategies for Parallel
Processing, 9th International Workshop, JSSPP 2003, Seattle, WA, USA, June 24, 2003, Revised Papers,
volume 2862 of Lecture Notes in Computer Science, pages 44–60. Springer, 2003. doi: 10.1007/10968987\_3.
URL https://doi.org/10.1007/10968987_3.
Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li,
Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language
models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria,
May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=N8N0hgNDRt.
Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth:
Building math generalist models through hybrid instruction tuning. In The Twelfth International Conference
on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL
https://openreview.net/forum?id=yLClGs770I.
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really
finish your sentence? In Anna Korhonen, David R. Traum, and Lluís Màrquez, editors, Proceedings of the
57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August
2, 2019, Volume 1: Long Papers, pages 4791–4800. Association for Computational Linguistics, 2019. doi:
10.18653/V1/P19-1472. URL https://doi.org/10.18653/v1/p19-1472.
Biao Zhang and Rico Sennrich.
Root mean square layer normalization.
In Hanna M. Wal-
lach, Hugo Larochelle, Alina Beygelzimer, Florence d’Alché-Buc, Emily B. Fox, and Roman Gar-
nett, editors, Advances in Neural Information Processing Systems 32: Annual Conference on Neu-
ral Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC,
Canada, pages 12360–12371, 2019.
URL https://proceedings.neurips.cc/paper/2019/hash/
1e8a19426224ca89e83cef47f1e7f53b-Abstract.html.
Ge Zhang, Scott Qu, Jiaheng Liu, Chenchen Zhang, Chenghua Lin, Chou Leuang Yu, Danny Pan, Esther Cheng,
Jie Liu, Qunshu Lin, Raven Yuan, Tuney Zheng, Wei Pang, Xinrun Du, Yiming Liang, Yinghao Ma, Yizhi Li,
Ziyang Ma, Bill Y. Lin, Emmanouil Benetos, Huan Yang, Junting Zhou, Kaijing Ma, Minghao Liu, Morry Niu,
Noah Wang, Quehry Que, Ruibo Liu, Sine Liu, Shawn Guo, Soren Gao, Wangchunshu Zhou, Xinyue Zhang,
Yizhi Zhou, Yubo Wang, Yuelin Bai, Yuhan Zhang, Yuxiang Zhang, Zenith Wang, Zhenzhu Yang, Zijian Zhao,
Jiajun Zhang, Wanli Ouyang, Wenhao Huang, and Wenhu Chen. Map-neo: Highly capable and transparent
bilingual large language model series. CoRR, abs/2405.19327, 2024a. doi: 10.48550/ARXIV.2405.19327.
URL https://doi.org/10.48550/arXiv.2405.19327.
Yifan Zhang, Yifan Luo, Yang Yuan, and Andrew Chi-Chih Yao. Automathtext: Autonomous data selection with
language models for mathematical texts. CoRR, abs/2402.07625, 2024b. doi: 10.48550/ARXIV.2402.07625.
URL https://doi.org/10.48550/arXiv.2402.07625.
Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen
Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang,
Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. A survey of
large language models. CoRR, abs/2303.18223, 2023. doi: 10.48550/ARXIV.2303.18223. URL https:
//doi.org/10.48550/arXiv.2303.18223.
Tianyu Zheng, Ge Zhang, Tianhao Shen, Xueling Liu, Bill Yuchen Lin, Jie Fu, Wenhu Chen, and Xiang
Yue. Opencodeinterpreter: Integrating code generation with execution and refinement. In Lun-Wei Ku,
Andre Martins, and Vivek Srikumar, editors, Findings of the Association for Computational Linguistics, ACL
2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pages 12834–12859. Association for
Computational Linguistics, 2024. doi: 10.18653/V1/2024.FINDINGS-ACL.762. URL https://doi.org/
10.18653/v1/2024.findings-acl.762.
Fan Zhou, Zengzhi Wang, Qian Liu, Junlong Li, and Pengfei Liu. Programming every example: Lifting pre-
training data quality like experts at scale. CoRR, abs/2409.17115, 2024a. doi: 10.48550/ARXIV.2409.17115.
URL https://doi.org/10.48550/arXiv.2409.17115.
Kun Zhou, Beichen Zhang, Jiapeng Wang, Zhipeng Chen, Wayne Xin Zhao, Jing Sha, Zhichao Sheng, Shijin
Wang, and Ji-Rong Wen. Jiuzhang3.0: Efficiently improving mathematical reasoning by training small
data synthesis models. CoRR, abs/2405.14365, 2024b. doi: 10.48550/ARXIV.2405.14365. URL https:
//doi.org/10.48550/arXiv.2405.14365.
Yutao Zhu, Kun Zhou, Kelong Mao, Wentong Chen, Yiding Sun, Zhipeng Chen, Qian Cao, Yihan Wu, Yushuo
Chen, Feng Wang, Lei Zhang, Junyi Li, Xiaolei Wang, Lei Wang, Beichen Zhang, Zican Dong, Xiaoxue
Cheng, Yuhan Chen, Xinyu Tang, Yupeng Hou, Qiangqiang Ren, Xincheng Pang, Shufang Xie, Wayne Xin
36

Zhao, Zhicheng Dou, Jiaxin Mao, Yankai Lin, Ruihua Song, Jun Xu, Xu Chen, Rui Yan, Zhewei Wei,
Di Hu, Wenbing Huang, Ze-Feng Gao, Yueguo Chen, Weizheng Lu, and Ji-Rong Wen. Yulan: An open-
source large language model. CoRR, abs/2406.19853, 2024. doi: 10.48550/ARXIV.2406.19853. URL
https://doi.org/10.48550/arXiv.2406.19853.
Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William
Fedus. Designing effective sparse expert models. CoRR, abs/2202.08906, 2022. URL https://arxiv.
org/abs/2202.08906.
37

A
Definition of Variables
We list the detailed definition of all the used variables from Section 3 in Table 8.
Table 8: Definition of the variables for computing the hyperparameters.
Variables
Meaning
nlayers
The num of model’s layers
nheads
The num of model’s attention heads
nkv_heads
The num of model’s kv-heads used in GQA
dmodel
Model dimension, i.e., hidden size
dhead
Dimension of attention head
dffn
The hidden size of feed-forward network
σbase
Initialization standard deviation for each matrix
ηbase
Learning rate, i.e., max learning rate
dmodel_proxy
dmodel for proxy model, i.e., the 0.05B model
mwidth
Width scaling factor in µP, i.e., dmodel/dmodel_proxy
B
Training Stability
B.1
Experiment Setup
We employ a relatively large learning rate with the intention of revealing the instability within the
model. Unless otherwise specified, a trial learning rate of 0.01 is adopted. We sample 20B tokens
from our pre-training dataset, ensuring consistency with the source used in the final training.
The detailed architecture settings are listed in Table 9. The 0.2B proxy model is utilized for general
exploratory experiments, whereas the 0.05B and 0.4B model are employed for µP experiments. The
latter proxy model has the same number of layers as the final model.
Table 9: Small proxy models used to explore the training dynamics.
Model
LR
nlayers
dmodel
dffn
nheads
nkv_heads
YuLan-Mini
0.01
56
1920
4800
30
6
Proxy model (0.05B)
0.01
32
256
640
2
2
Proxy model (0.2B)
0.01
30
576
1536
9
3
Proxy model (0.4B)
0.01
56
576
1536
9
3
C
Prompts for Generating Synthetic Data
C.1
Math
Prompt for Mathematical Documents Synthesis A
## Instruction
Please gain inspiration from the following content to create a lecture script for a college-level mathematics course.
## Content
{Content Placeholder}
Begin without using titles or introductions.
38

Prompt for Mathematical Documents Synthesis B
Write an educational piece suited for college students related to the following text snippet:
{Content Placeholder}
Do not just list concepts, but develop each one in detail before moving to the next, as we prioritize depth of understand-
ing and comprehensive exploration of the subject matter over breadth. Focus on:
- Rigor: Ensure in-depth coverage of the concepts/sections.
- Engagement: Write with an academic, professional and engaging tone that captivates interest.
- Application: Incorporate specific, practical examples, such as proofs in calculus or critical dates and figures in history.
Do not include a title or an introduction, simply write the content without headlines and introductory phrases.
Prompt for Mathematical Documents Synthesis C
Write an educational piece suited for middle school students related to the following text snippet:
{Content Placeholder}
Do not just list concepts, but develop each one in detail before moving to the next, as we prioritize depth of understand-
ing and comprehensive exploration of the subject matter over breadth. Focus on:
- Rigor: Ensure in-depth coverage of the concepts/sections.
- Engagement: Write with an academic, professional and engaging tone that captivates interest.
- Application: Incorporate specific, practical examples, such as proofs in calculus or critical dates and figures in history.
Do not include a title or an introduction, simply write the content without headlines and introductory phrases.
Prompt for Mathematical Documents Synthesis D
## Instruction
Please gain inspiration from the following content to draft a mathematics textbook chapter suitable for college students:
## Content
{Content Placeholder}
Write in a clear, structured manner that is easy for students to follow and understand.
Prompt for Mathematical Documents Synthesis E
## Instruction
Please gain inspiration from the following content to draft a mathematics textbook chapter suitable for middle school students:
## Content
{Content Placeholder}
Write in a clear, structured manner that is easy for students to follow and understand.
Prompt for Mathematical Documents Synthesis F
## Instruction
Please gain inspiration from the following content to design a problem set with solutions.
## Content
{Content Placeholder}
## Guidelines
- Formulate a series of problems that test understanding and application of the concepts.
- Provide detailed solutions for each problem, explaining the reasoning and calculations involved.
- Vary the difficulty of the problems to cater to students with different levels of proficiency.
The problem set should challenge students to apply their knowledge in practical scenarios.
39

Prompt for Mathematical Instruction Synthesis
You are exceptionally skilled at crafting high-quality math problems and offering precise solutions.
Please gain inspiration from the following random math content to create a high-quality math problem and solve it step
by step with clear logic. Present your output in two distinct sections:
[Problem Description] and [Solution]
Math content for inspiration:
{Seed Content Placeholder}
Guidelines for each section:
1. [Problem Description]: This should be **completely self-contained**, providing all the contextual information one needs to
understand and solve the problem.
2. [Solution]: Offer a comprehensive, **correct** solution that accurately addresses the [Problem Description] you provided step by
step with clear logic. Please ensure that the Solution only involves answering the Problem, **without addressing the requirements I
provided.**
C.2
Code
Prompt for Code Instruction Synthesis A
You are a teaching assistant helping to create a Python programming task from a given code snippet. You must provide the best
response to the Python programming task, including reasoning thought and reference solutions.
[Code Snippet]
{Code Placeholder}
Your response must have these parts:
[Task]
{Create an independent and detailed Python programming task}
[Analysis]
{Analyze the task and reason about the given task step by step}
[Solution]
{Write a high-quality reference solution in a self-contained script that solves the task}
Prompt for Code Instruction Synthesis B
You are exceptionally skilled at crafting high-quality Python programming problems and offering precise solutions.
Please gain inspiration from the following random code snippet to create a high-quality programming problem. Present your output in
two distinct sections:
[Problem Description] and [Solution]
Code snippet for inspiration:
{Code Placeholder}
Guidelines for each section:
1. [Problem Description]: This should be **completely self-contained**, providing all the contextual information one needs to
understand and solve the problem. Assume common programming knowledge, but ensure that any specific context, variables, or code
snippets pertinent to this problem are explicitly included.
2. [Solution]: Offer a comprehensive, **correct** solution that accurately addresses the [Problem Description] you provided. Please
ensure that the Solution only involves answering the Problem, **without addressing the requirements I provided!**
40

C.3
Science
Prompt for Scientific QA Synthesis
Instruction
Please gain inspiration from the following {Discipline Placeholder} content to create a high-quality {Discipline Placeholder} problem
and solution. Present your output in two distinct sections: [Problem] and [Solution].
{Discipline Placeholder} Content
{Seed Snippet Placeholder}
Guidelines
[Problem]: This should be **completely self-contained**, providing all the contextual information one needs to understand and solve
the problem.
[Solution]: Present a comprehensive, step-by-step solution that solves the problem **correctly** and educates the student, around
250-350 words long. Clearly articulate the reasoning and methods used at each step, providing insight into the problem-solving
process. Take care to format any equations properly using LaTeX or appropriate notation.
Prompt for Topic Labeling
I am categorizing a series of articles according to the following 11 topics. Next, I will give you an article, please select only one topic
that the article is the most related to:
[Topics]: {Topic List Placeholder}
[Article]: {Web Page Content Placeholder}
Please only return the most related topic:
C.4
Data Selection
Prompt for Instruction Tag Labeling
Please identify the relevant tags representing the user’s intentions in the following Problem and Solution. Focus on the reasoning
behind the solution. Please ONLY respond with tags in a Python list.
Problem:
{Question Placeholder}
Solution:
{Answer Placeholder}
D
Open-Source Datasets Used during Pre-training
Table 10: Comprehensive list of all open-source datasets used. For datasets that are only available via
links, we also offer additional guidance on our project website https://github.com/RUC-GSAI/
YuLan-Mini.
Domain
Dataset
General
chinese-fineweb-edu [Opencsg]
llm360-txt360 [Tang et al., 2024a]
wanjuan [He et al., 2023]
dclm [Li et al., 2024b]
fineweb-edu [Penedo et al., 2024]
dolma [Soldaini et al., 2024]
tulu3 [Lambert et al., 2024]
magpie-reasoning-150k [Xu et al., 2024]
Code
the-stack-v2 [Lozhkov et al., 2024b]
starcoderdata [Li et al., 2023]
opencoder-llm [Huang et al., 2024]
longwanjuan-github [Liu et al., 2024b]
mathcodeinstruct [Wang et al., 2024b]
codefeedback-filtered-instruction [Zheng et al., 2024]
xcoder-80k [Wang et al., 2024c]
41

Math
proof-pile-2 [Azerbayev et al., 2024]
automathtext [Zhang et al., 2024b]
open-web-math-pro [Zhou et al., 2024a]
deepmind-math [Saxton et al., 2019]
orca-math [Mitra et al., 2024b]
metamathqa [Yu et al., 2024]
numina [Beeching et al., 2024]
silmorca [Lian et al., 2023]
scalequest-math [Ding et al., 2024b]
infimm-webmath-40b [Han et al., 2024]
lean-star [Lin et al., 2024]
lean-github [Wu et al., 2024]
lean-workbook [Ying et al., 2024]
lean-deepseek-v1 [Xin et al., 2024]
ape210k [Kadlcík et al., 2023]
mathinstruct [Yue et al., 2024]
openmathinstruct-1 [Toshniwal et al., 2024]
mathscaleqa-2m [Tang et al., 2024c]
orca-agentinstruct [Mitra et al., 2024a]
fol-nli [Sileo, 2024]
gretel-math-gsm8k-v1 [AI, 2024]
E
Detailed Data Composition by Training Phases
The specific composition of data for each course stage is shown in Table 12, where black represents
English web pages and general content, green represents Chinese, blue represents code, and red
represents mathematics. The first 10 billion tokens of Phase 1 are used during the warm-up stage.
The next 30 billion tokens of Phase 1, along with Phases 2 through 25, are employed in the stable
training stage. Phases 26 and 27 constitute the annealing stage.
Table 12: Detailed data composition by training curriculum phases.
Phase
Data composition by phase (in billions of tokens)
1
dclm (1.80), fineweb-edu (16.20), english-books (1.60),
pes2o (0.80), arxiv (1.20), wikipedia (0.40), dolma (1.24),
cicg-news (0.76), cn-baike (0.39), mnbvc-news (0.08), cn-book (0.24),
cn-legal-case-law (0.36), zhihu-qa (0.12), the-stack-v2 (4.91),
starcoder (2.92), smollm-python (0.20), proof-pile-2 (1.52),
automathtext (1.12), open-web-math-pro (0.20), cosmopedia (1.01),
mathtext (0.12)
2
dclm (1.80), fineweb-edu (16.20), english-books (1.60),
pes2o (0.80), arxiv (1.20), wikipedia (0.40), dolma (1.24),
cicg-news (0.76), cn-baike (0.39), mnbvc-news (0.08), cn-book (0.24),
cn-legal-case-law (0.36), zhihu-qa (0.12), the-stack-v2 (4.90),
starcoder (2.92), smollm-python (0.20), proof-pile-2 (1.52),
automathtext (1.12), open-web-math-pro (0.20), cosmopedia (1.02),
mathtext (0.12)
3
dclm (1.80), fineweb-edu (16.20), english-books (1.60),
pes2o (0.80), arxiv (1.20), wikipedia (0.40), dolma (1.24),
cicg-news (0.76), cn-baike (0.39), mnbvc-news (0.08), cn-book (0.24),
cn-legal-case-law (0.36), zhihu-qa (0.12), the-stack-v2 (4.86),
starcoder (2.92), smollm-python (0.20), proof-pile-2 (1.52),
automathtext (1.12), open-web-math-pro (0.20), cosmopedia (1.02),
mathtext (0.12)
42

4
dclm (1.80), fineweb-edu (16.20), english-books (1.60),
pes2o (0.80), arxiv (1.20), wikipedia (0.40), dolma (1.24),
cicg-news (0.76), cn-baike (0.39), mnbvc-news (0.08), cn-book (0.24),
cn-legal-case-law (0.36), zhihu-qa (0.12), the-stack-v2 (4.14),
starcoder (2.92), smollm-python (0.20), proof-pile-2 (1.52),
automathtext (1.12), open-web-math-pro (0.24), cosmopedia (0.98),
mathtext (0.12)
5
dclm (1.80), fineweb-edu (16.20), english-books (1.60),
pes2o (0.80), arxiv (1.20), wikipedia (0.40), dolma (1.24),
cicg-news (0.76), cn-baike (0.39), mnbvc-news (0.08), cn-book (0.24),
cn-legal-case-law (0.36), zhihu-qa (0.12), the-stack-v2 (4.90),
starcoder (2.92), smollm-python (0.20), proof-pile-2 (1.54),
automathtext (1.14), open-web-math-pro (0.24), cosmopedia (0.94),
mathtext (0.12)
6
dclm (1.80), fineweb-edu (16.20), english-books (1.60),
pes2o (0.80), arxiv (1.20), wikipedia (0.40), dolma (1.24),
cicg-news (0.76), cn-baike (0.39), mnbvc-news (0.08), cn-book (0.24),
cn-legal-case-law (0.36), zhihu-qa (0.12), the-stack-v2 (4.90),
starcoder (2.92), smollm-python (0.20), proof-pile-2 (1.54),
automathtext (1.16), open-web-math-pro (0.26), cosmopedia (0.82),
mathtext (0.12), metamathqa (0.02), orca-math (0.02),
yulan-mini-syn-math-inst (0.04)
7
dclm (1.80), fineweb-edu (16.20), english-books (1.60),
pes2o (0.80), arxiv (1.20), wikipedia (0.40), dolma (1.24),
cicg-news (0.76), cn-baike (0.39), mnbvc-news (0.08), cn-book (0.24),
cn-legal-case-law (0.36), zhihu-qa (0.12), the-stack-v2 (4.90),
starcoder (2.92), smollm-python (0.20), proof-pile-2 (1.60),
automathtext (1.17), open-web-math-pro (0.28), cosmopedia (0.77),
mathtext (0.12), metamathqa (0.01), orca-math (0.01),
yulan-mini-syn-math-inst (0.02)
8
dclm (1.80), fineweb-edu (16.20), english-books (1.60),
pes2o (0.80), arxiv (1.20), wikipedia (0.40), dolma (1.24),
cicg-news (0.76), cn-baike (0.39), mnbvc-news (0.08), cn-book (0.24),
cn-legal-case-law (0.36), zhihu-qa (0.12), the-stack-v2 (4.90),
starcoder (2.92), smollm-python (0.20), proof-pile-2 (1.64),
automathtext (1.17), open-web-math-pro (0.32), cosmopedia (0.53),
fineweb-math (0.16), mathtext (0.12), metamathqa (0.01),
orca-math (0.01), yulan-mini-syn-math-inst (0.02)
9
dclm (1.80), fineweb-edu (16.20), english-books (1.20), pes2o (0.80),
arxiv (1.20), wikipedia (0.40), dolma (1.24), cosmopedia-v2 (0.40),
cicg-news (0.76), cn-baike (0.39), mnbvc-news (0.08),
cn-book (0.24), cn-legal-case-law (0.36), zhihu-qa (0.12),
the-stack-v2 (4.86), starcoder (2.92), smollm-python (0.20),
proof-pile-2 (1.64), automathtext (1.17), open-web-math-pro (0.32),
cosmopedia (0.33), fineweb-math (0.16), mathtext (0.12),
metamathqa (0.01), orca-math (0.01), yulan-mini-syn-math-inst (0.02),
yulan-mini-syn-math-doc (0.22)
10
dclm (1.80), fineweb-edu (16.20), english-books (1.00), pes2o (0.80),
arxiv (1.20), wikipedia (0.40), dolma (1.24), cosmopedia-v2 (0.60),
cicg-news (0.76), cn-baike (0.39), mnbvc-news (0.08), cn-book (0.24),
cn-legal-case-law (0.36), zhihu-qa (0.12), the-stack-v2 (4.85),
starcoder (2.92), smollm-python (0.20), yulan-mini-syn-code-inst (0.03),
proof-pile-2 (1.64), automathtext (1.17), open-web-math-pro (0.32),
cosmopedia (0.29), fineweb-math (0.20), mathtext (0.12),
metamathqa (0.01), orca-math (0.01), yulan-mini-syn-math-inst (0.02),
yulan-mini-syn-math-doc (0.22)
43

11
dclm (1.80), fineweb-edu (16.20), english-books (0.82), pes2o (0.80),
arxiv (1.20), wikipedia (0.40), dolma (1.24), cosmopedia-v2 (0.78),
cicg-news (0.76), cn-baike (0.39), mnbvc-news (0.08), cn-book (0.24),
cn-legal-case-law (0.36), zhihu-qa (0.12), the-stack-v2 (4.56),
starcoder (2.92), smollm-python (0.20), mnbvc-code (0.04),
yulan-mini-syn-code-inst (0.28), proof-pile-2 (1.64),
automathtext (0.93), open-web-math-pro (0.41), cosmopedia (0.16),
fineweb-math (0.33), dclm-math (0.12), mathtext (0.12),
metamathqa (0.01), orca-math (0.01), yulan-mini-syn-math-inst (0.02),
yulan-mini-syn-math-doc (0.25)
12
dclm (1.80), fineweb-edu (16.20), english-books (0.82), pes2o (0.80),
arxiv (1.20), wikipedia (0.40), dolma (1.24), cosmopedia-v2 (0.78),
cicg-news (0.76), cn-baike (0.39), mnbvc-news (0.08), cn-book (0.24),
cn-legal-case-law (0.36), zhihu-qa (0.12), the-stack-v2 (4.44),
starcoder (2.92), smollm-python (0.20), mnbvc-code (0.16),
yulan-mini-syn-code-inst (0.28), proof-pile-2 (1.64),
automathtext (0.63), open-web-math-pro (0.41), cosmopedia (0.03),
fineweb-math (0.50), dclm-math (0.11), mathtext (0.12),
basic-math-10m (0.04), metamathqa (0.01), orca-math (0.01),
yulan-mini-syn-math-inst (0.06), yulan-mini-syn-math-doc (0.44)
13
dclm (1.80), fineweb-edu (16.20), english-books (0.82), pes2o (0.80),
arxiv (1.20), wikipedia (0.40), dolma (1.24), cosmopedia-v2 (0.78),
cicg-news (0.76), cn-baike (0.39), mnbvc-news (0.08), cn-book (0.24),
cn-legal-case-law (0.36), zhihu-qa (0.12), the-stack-v2 (4.44),
starcoder (2.92), smollm-python (0.20), mnbvc-code (0.16),
yulan-mini-syn-code-inst (0.28), proof-pile-2 (1.64),
automathtext (0.58), open-web-math-pro (0.41), fineweb-math (0.51),
dclm-math (0.15), mathtext (0.12), basic-math-10m (0.04),
metamathqa (0.01), yulan-mini-syn-math-inst (0.08),
yulan-mini-syn-math-doc (0.44)
14
dclm (1.80), fineweb-edu (16.20), english-books (0.82), pes2o (0.80),
arxiv (1.20), wikipedia (0.40), dolma (1.24), cosmopedia-v2 (0.78),
cicg-news (0.76), cn-baike (0.33), mnbvc-news (0.08),
cn-book (0.24), cn-legal-case-law (0.36), zhihu-qa (0.12),
the-stack-v2 (4.44), starcoder (2.92), smollm-python (0.20),
mnbvc-code (0.16), yulan-mini-syn-code-inst (0.28),
proof-pile-2 (1.64), automathtext (0.57), open-web-math-pro (0.41),
cosmopedia (0.02), fineweb-math (0.51), dclm-math (0.15),
mathtext (0.12), basic-math-10m (0.04), metamathqa (0.01),
yulan-mini-syn-math-inst (0.09), yulan-mini-syn-math-doc (0.44)
15
dclm (1.80), fineweb-edu (16.20), english-books (0.82), pes2o (0.80),
arxiv (1.20), wikipedia (0.40), dolma (1.24), cosmopedia-v2 (0.78),
cicg-news (0.76), cn-baike (0.27), mnbvc-news (0.08),
cn-book (0.24), cn-legal-case-law (0.36), zhihu-qa (0.12),
the-stack-v2 (4.37), starcoder (2.32), smollm-python (0.20),
mnbvc-code (0.83), yulan-mini-syn-code-inst (0.28),
proof-pile-2 (1.34), automathtext (0.65), open-web-math-pro (0.41),
cosmopedia (0.02), fineweb-math (0.42), dclm-math (0.40),
mathtext (0.12), basic-math-10m (0.04), metamathqa (0.01),
yulan-mini-syn-math-inst (0.13), yulan-mini-syn-math-doc (0.46)
44

16
dclm (1.80), fineweb-edu (16.20), english-books (0.82), pes2o (0.80),
arxiv (1.20), wikipedia (0.40), dolma (1.24), cosmopedia-v2 (0.78),
cicg-news (0.76), cn-baike (0.27), mnbvc-news (0.10), cn-book (0.24),
cn-legal-case-law (0.36), zhihu-qa (0.12), the-stack-v2 (4.27),
starcoder (2.12), smollm-python (0.20), mnbvc-code (1.13),
yulan-mini-syn-code-inst (0.28), proof-pile-2 (1.03),
automathtext (0.68), open-web-math-pro (0.36), cosmopedia (0.02),
fineweb-math (0.46), dclm-math (0.51), mathtext (0.12),
basic-math-10m (0.04), yulan-mini-syn-math-inst (0.20),
yulan-mini-syn-math-doc (0.57)
17
dclm (1.80), fineweb-edu (16.20), english-books (0.82), pes2o (0.80),
arxiv (1.20), wikipedia (0.40), dolma (1.24), cosmopedia-v2 (0.78),
cicg-news (0.76), cn-baike (0.27), mnbvc-news (0.10), cn-book (0.24),
cn-legal-case-law (0.36), zhihu-qa (0.12), the-stack-v2 (4.45),
starcoder (2.12), smollm-python (0.20), mnbvc-code (1.13),
yulan-mini-syn-code-inst (0.28), proof-pile-2 (0.55),
automathtext (0.68), cosmopedia (0.02), fineweb-math (0.46),
dclm-math (1.02), mathtext (0.12), basic-math-10m (0.04),
yulan-mini-syn-math-inst (0.36), yulan-mini-syn-math-doc (0.57)
18
dclm (1.80), fineweb-edu (16.20), english-books (0.82), pes2o (0.80),
arxiv (1.20), wikipedia (0.40), dolma (1.24), cosmopedia-v2 (0.78),
cicg-news (0.76), cn-baike (0.27), mnbvc-news (0.10), cn-book (0.24),
cn-legal-case-law (0.36), zhihu-qa (0.12), the-stack-v2 (4.45),
starcoder (2.12), smollm-python (0.20), mnbvc-code (1.13),
yulan-mini-syn-code-inst (0.31), opencoder-llm-math-web (1.54),
automathtext (0.00), cosmopedia (0.02), fineweb-math (0.17),
dclm-math (0.82), mathtext (0.12), basic-math-10m (0.04),
yulan-mini-syn-math-inst (0.52), yulan-mini-syn-math-doc (0.56)
19
dclm (1.80), fineweb-edu (16.20), english-books (0.82), pes2o (0.80),
arxiv (1.20), wikipedia (0.40), dolma (1.24), cosmopedia-v2 (0.78),
cicg-news (0.76), cn-baike (0.02), mnbvc-news (0.02), cn-book (0.24),
cn-legal-case-law (0.36), zhihu-qa (0.12), the-stack-v2 (4.41),
starcoder (2.12), smollm-python (0.20), mnbvc-code (1.06),
opencoder-llm-annealing (0.08), yulan-mini-syn-code-inst (0.31),
opencoder-llm-math-web (0.38), cosmopedia (0.02), fineweb-math (0.10),
dclm-math (0.82), infimm-webmath (1.23), mathtext (0.12),
basic-math-10m (0.04), yulan-mini-syn-math-inst (0.55),
yulan-mini-syn-math-doc (0.56)
20
dclm (1.80), fineweb-edu (16.20), english-books (0.82),
pes2o (0.80), arxiv (1.20), wikipedia (0.40), dolma (0.84),
opencoder-llm-fineweb-corpus (0.40), cosmopedia-v2 (0.78),
cicg-news (0.76), mnbvc-news (0.02), cn-book (0.26),
cn-legal-case-law (0.36), zhihu-qa (0.12), the-stack-v2 (4.33),
starcoder (2.12), smollm-python (0.20), mnbvc-code (0.92),
opencoder-llm-annealing (0.16), yulan-mini-syn-code-inst (0.45),
opencoder-llm-math-web (0.24), cosmopedia (0.02), fineweb-math (0.04),
dclm-math (0.56), infimm-webmath (1.62), mathtext (0.12),
basic-math-10m (0.01), yulan-mini-syn-math-inst (0.67),
yulan-mini-syn-math-doc (0.53)
45

21
dclm (1.80), fineweb-edu (16.20), english-books (0.82),
pes2o (0.80), arxiv (1.20), wikipedia (0.30), dolma (0.84),
opencoder-llm-fineweb-corpus (0.50), cosmopedia-v2 (0.78),
cicg-news (0.76), mnbvc-news (0.02), cn-book (0.26),
cn-legal-case-law (0.36), zhihu-qa (0.12), the-stack-v2 (4.19),
starcoder (2.12), smollm-python (0.20), mnbvc-code (0.80),
opencoder-llm-annealing (0.41), yulan-mini-syn-code-inst (0.45),
opencoder-llm-math-web (0.24), cosmopedia (0.02), dclm-math (0.23),
infimm-webmath (2.00), mathtext (0.12), yulan-mini-syn-math-inst (0.70),
yulan-mini-syn-math-doc (0.53)
22
dclm (1.80), fineweb-edu (16.20), english-books (0.82), pes2o (0.80),
arxiv (1.20), dolma (0.84), opencoder-llm-fineweb-corpus (0.80),
cosmopedia-v2 (0.78), cicg-news (0.76), mnbvc-news (0.02),
cn-book (0.26), cn-legal-case-law (0.36), zhihu-qa (0.12),
the-stack-v2 (4.60), starcoder (1.10), smollm-python (0.20),
mnbvc-code (1.13), opencoder-llm-sft-s1 (0.20), opencoder-llm-annealing (0.38),
yulan-mini-syn-code-inst (0.56), opencoder-llm-math-web (0.24),
cosmopedia (0.02), dclm-math (0.22), infimm-webmath (1.98),
mathtext (0.06), yulan-mini-syn-math-inst (0.78),
yulan-mini-syn-math-doc (0.53)
23
dclm (1.80), fineweb-edu (16.20), english-books (0.82), pes2o (0.80),
arxiv (1.20), dolma (0.84), opencoder-llm-fineweb-corpus (0.80),
cosmopedia-v2 (0.78), cicg-news (0.76), mnbvc-news (0.02),
cn-book (0.26), cn-legal-case-law (0.36), zhihu-qa (0.12),
the-stack-v2 (4.16), starcoder (0.80), mnbvc-code (0.85),
opencoder-llm-sft-s1 (0.20), opencoder-llm-sft-s2 (0.15),
opencoder-llm-annealing (1.20), yulan-mini-syn-code-inst (0.56),
opencoder-llm-math-web (0.24), cosmopedia (0.02), dclm-math (0.22),
infimm-webmath (2.06), mathtext (0.04), lean (0.02),
yulan-mini-syn-math-inst (0.92), yulan-mini-syn-math-doc (0.56)
24
dclm (1.80), fineweb-edu (16.20), english-books (0.82), pes2o (0.80),
arxiv (1.20), dolma (0.84), opencoder-llm-fineweb-corpus (0.80),
cosmopedia-v2 (0.78), cicg-news (0.76), mnbvc-news (0.02),
cn-book (0.26), cn-legal-case-law (0.36), zhihu-qa (0.12),
the-stack-v2 (4.07), starcoder (0.80), mnbvc-code (0.85),
opencoder-llm-sft-s1 (0.25), opencoder-llm-sft-s2 (0.08),
opencoder-llm-annealing (1.27), yulan-mini-syn-code-inst (0.56),
opencoder-llm-math-web (0.24), cosmopedia (0.02), dclm-math (0.19),
infimm-webmath (2.11), lean (0.04), yulan-mini-syn-math-inst (0.96),
yulan-mini-syn-math-doc (0.56)
25
dclm (1.80), fineweb-edu (16.20), english-books (0.82), pes2o (0.80),
arxiv (1.20), dolma (0.84), opencoder-llm-fineweb-corpus (0.80),
cosmopedia-v2 (0.78), cicg-news (0.76), mnbvc-news (0.02),
cn-book (0.26), cn-legal-case-law (0.36), zhihu-qa (0.12),
the-stack-v2 (4.11), starcoder (0.80), mnbvc-code (0.78),
opencoder-llm-sft-s1 (0.25), opencoder-llm-sft-s2 (0.08),
opencoder-llm-annealing (1.30), ioccc (0.00), yulan-mini-syn-code-inst (0.54),
opencoder-llm-math-web (0.24), cosmopedia (0.02), fineweb-math (0.20),
dclm-math (0.17), infimm-webmath (2.11), lean (0.04),
yulan-mini-syn-math-inst (0.93), yulan-mini-syn-math-doc (0.45)
46

26
(Decay)
dclm (1.62), fineweb-edu (14.58), english-books (0.74), pes2o (0.72),
arxiv (1.08), dolma (0.48), opencoder-llm-fineweb-corpus (1.00),
cosmopedia-v2 (0.70), cicg-news (0.68), wizardlm-evol-instruct-v2-196k (0.04),
less-data (0.04), claude-3-opus-claude-3.5-sonnnet-9k (0.01),
slimorca (0.16), tulu-v3.1-mix-preview-4096-olmoe (0.20),
supernova (0.03), magpie-reasoning-150k (0.09), spurline (0.01),
celestia (0.04), mnbvc-news (0.01), cn-book (1.00),
zhihu-qa (0.05), chinese-porety (0.03), the-stack-v2 (3.16),
starcoder (0.18), mnbvc-code (0.70), opencoder-llm-sft-s1 (0.62),
opencoder-llm-annealing (1.09), magicoder-oss (0.06),
textbook-quality-programming (0.06), yulan-mini-syn-code-inst (3.00),
code-290k-sharegpt (0.08), evol-codealpaca-v1 (0.04),
magicoder-evol-instruct-110k (0.04), mathcodeinstruct
(0.03), codefeedback-filtered-instruction (0.08),
python-code-23k-sharegpt (0.01), evol-instruct-code-80k-v1 (0.03),
codeexercise-python-27k (0.02), xcoder-80k (0.04),
leetcode-solution-python (0.00), tachibana (0.03),
opencoder-llm-math-web (0.24), cosmopedia (0.12), infimm-webmath (1.33),
ape210k (0.01), polytope (0.03), yulan-mini-syn-math-inst (1.44),
yulan-mini-syn-math-doc (0.58), mammothmathinstruct (0.04),
openmathinstruct-1 (0.35), fol-nli (0.12), mathscaleqa-2m (0.28)
27
(Decay)
dclm (1.44), fineweb-edu (12.96), english-books (1.48), pes2o (0.64),
arxiv (0.96), dolma (0.20), opencoder-llm-fineweb-corpus (1.08),
cosmopedia-v2 (0.70), cicg-news (0.61), wizardlm-evol-instruct-v2-196k (0.04),
long-cot (0.65), slimorca (0.04), tulu-v3.1-mix-preview-4096-olmoe (0.25),
evolkit-20k (0.02), orca-agentinstruct (0.49), transcript (0.01),
spurline (0.01), titanium (0.02), celestia (0.06), cn-book (1.40),
zhihu-qa (0.05), ruozhiba (0.00), chinese-porety (0.04),
the-stack-v2 (1.50), mnbvc-code (0.38), opencoder-llm-sft-s1 (0.16),
opencoder-llm-annealing (1.13), magicoder-oss (0.11),
textbook-quality-programming (0.05), longwanjuan-github (1.82),
yulan-mini-syn-code-inst (3.75), code-290k-sharegpt (0.04),
evol-codealpaca-v1 (0.03), magicoder-evol-instruct-110k (0.03),
mathcodeinstruct
(0.02), codefeedback-filtered-instruction (0.01),
self-oss-instruct-sc2-exec-filter-50k (0.02), xcoder-80k (0.04),
tulu-code (0.02), codefuse-evol-instruct-clean
(0.03),
proof-pile-2 (0.60), opencoder-llm-math-web (0.45),
cosmopedia (0.02), dclm-math (0.06), infimm-webmath (1.34),
yulan-mini-syn-math-inst (1.72), yulan-mini-syn-math-doc (0.25),
mammothmathinstruct (0.02), openmathinstruct-1 (0.02),
tulu-math (0.14), tulu-math-grade (0.03), tulu-algebra (0.02),
fol-nli (0.12), reasoning-0.01 (0.02), gretel-math-gsm8k-v1 (0.01),
mathscaleqa-2m (0.40)
47
