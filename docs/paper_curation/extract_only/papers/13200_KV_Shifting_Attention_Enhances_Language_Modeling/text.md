# arXiv:2411.19574v2[cs.CL]5Dec2024

## KV Shifting Attention Enhances Language Modeling

Mingyu Xu Baichuan Inc.

Wei Cheng Baichuan Inc.

Bingning Wang ∗ Baichuan Inc.

Weipeng Chen Baichuan Inc.

#### Abstract

The current large language models are mainly based on decode-only structure transformers, which have great in-context learning (ICL) capabilities. It is generally believed that the important foundation of its ICL capability is the induction heads mechanism, which requires at least two layers attention. In order to more efficiently implement the ability of the model’s induction, we revisit the induction heads mechanism and proposed a KV shifting attention. We theoretically prove that the KV shifting attention reducing the model’s requirements for the depth and width of the induction heads mechanism. Our experimental results demonstrate that KV shifting attention is beneficial to learning induction heads and language modeling, which lead to better performance or faster convergence from toy models to the pre-training models with more than 10 B parameters.

#### 1 Introduction

Transformer-based Large language models (Vaswani, 2017) have demonstrated remarkable capabilities in in-context learning (ICL), largely attributed to the underlying mechanisms of induction heads (Elhage et al., 2021; Olsson et al., 2022). These mechanisms enable the models to identify and leverage repeating patterns, which is crucial in ICL (Song et al., 2024; Crosbie & Shutova, 2024) and multi-step reasoning (Sanford et al., 2024b).

Although there are many works based on transformer for analysis of induction heads, there are few works that utilize analysis of induction heads to modify transformers to enhance their ability to learn induction heads.

To achieve this goal, we revisit the induction head mechanism and analyze the required depth and width for induction heads. Building on the perspective, we propose KV shifting attention, a novel approach designed to simplify and enhance the induction process. By decoupling keys and values in the attention mechanism, KV shifting attention reduces the structural requirements for depth and width, enabling single-layer transformers to effectively perform induction tasks.

Through theoretical analysis and empirical validation, we demonstrate that KV Shifting attention achieves comparable or superior performance to conventional multi-layer transformers. Moreover, its bias towards learning induction leads to more efficient and effective language modeling across diverse scales, from toy models to pre-trained models with billions of parameters.

In summary, our contributions are as follows:

- • We proposed a novel KV shifting attention which accelerates learning ability for induction heads.
- • We theoretically analyze that KV shifting attention can effectively represent induction heads and learn induction heads from induction data.
- • We apply KV shifting attention in large language pre-training, and demonstrate its effectiveness experimentally.

∗Corresponding author, daniel@baichuan-inc.com

The following sections of the article will be arranged in the following order: Section 2 introduces our motivation and methods, Section 3 analyzes the KV shifting attention, Section 4 introduces our experiments on a large language model, and the last few Sections discuss, introduce relevant work, and summarize.

#### 2 Method

###### 2.1 Motivation

Induction heads Our story begins with induction heads. Induction heads mechanism (Elhage et al., 2021; Olsson et al., 2022) is a circuit whose function is to find the latest same previous instances of the current token (call it A) and the token that came after it (call it B), then use B as current token’s next token prediction. (e.g. forming the sequence [A][B]... [A] → [B]). It is generally believed that the implementation of the induction heads mechanism often requires two heads belonging to different layers. For a detailed proof that a layer of transformer cannot implement induction heads, readers can refer to Sanford et al. (2024a). So, can we make slight adjustments to the attention so that a single layer of attention can achieve the mechanism of induction heads?

Virtual attention heads Another interesting perspective proposed in the article (Elhage et al., 2021) is the concept of virtual attention heads. It demonstrates the cooperation among attentions in different layers. For simplicity, we consider the attention of only two adjacent layers and ignore the presence of residual connections and MLP. The formal expression is:

X1 = Ah1X0Wovh1, X2 = Ah2X1Wovh2, (1)

where X0 ∈ RN×D is the input, Xl ∈ RN×D is the output of lth layers, Ahl ∈ RN×N is the attention weights of lth layer, Wovhl = WvhiWohl ∈ RD×D, Wvhl ∈ RD×D is the v projection, Wohl ∈ RD×D is the o projection of lth layer, l ∈ 1,2, n ∈ R is context length, d is the dimension of hidden states. Then virtual attention heads is:

X2 = Ah2X1Wovh2 = (Ah2Ah1)X0(Wovh1Wovh2). (2)

Through virtual attention heads, models can achieve complex functions by combining simple attention heads.

With causal mask, there is an interesting things when virtual attention heads do function like induction heads.

Property 1 With causal mask and j ≥ i, we have

N

(Ah2Ah1)j,i+1 =

Ahj,2kAkh,1i+1 =

∑

k=1

j

Ahj,2kAkh,1i+1 (3)

∑

k=i+1

According to Property 1, from the perspective of virtual attention heads, it is difficult for the model to indirectly utilize j tokens to focus on the (i + 1)th token through ith token. In other words, in order for the (i + 1)th token to be output by future tokens using the induced heads mechanism, it must first integrate the information of the ith token into its hidden states, even if the information of the ith token is useless for predicting the (i + 2)th token. In other words, this imposes certain requirements on the dimensionality of hidden states. I will further explain in the next section that the induction heads mechanism requires a certain width.

###### 2.2 KV shifting attention

From a more general perspective, induction heads means obtaining information about some tokens by focusing on it surrounding tokens. In order to make it easier for the transformer to learn the mechanism of inductions heads, we can unbind the key and value of the ith

token. When current token attention to the ith token’s key, it can get the jth token’s value, j ∈ {i − 1,i,i + 1}. From a different perspective, current token can obtain the value of the ith token by focusing on the keys of the {i + 1,i,i − 1} tokens.

However, if we directly use the combination of jth token’s value (j ∈ {i − 1,i,i + 1}), it may breaking the causal mask, because (i + 1)th token’s value can’t be computed when ith token try to do next token prediction. So we can only let ith token’s key connect to (i − 1)th and ith token’s value. And if we do similar operation to value, which means let ith token’s value connect to (i − 1)th and ith token’s key. We can find that we can pay attention to the key of ith token, and then get the value of its near token without breaking causal mask.

Therefore, we propose the following KV shifting attention. For simplicity, we only formalize the single head attention as follow:

Q, K,V = XWQ, XWK, XWV (4) Kˆ,Vˆ = α1K + α2Shift(K), β1V + β2Shift(V) (5)

Output = Softmax(QKˆT · M/σ)VWˆ o, (6) where X ∈ RN×D is hidden states, WQ,WK,WV,WO ∈ RD×D, α1, α2, β1, β2 ∈ R are learnable parameters, σ = √D, M ∈ RD×D is the causal mask, Shift(·) means discarding the last token and padding zero at the beginning. Compared to the original attention, 4 learnable parameters have been added. In the case of multi head attention, α1, α2, β1, β2 are learned per head, so the additional parameters is 4h. The additional calculation caused by Eq 5 is O(ND), which is much smaller than O(ND2 + N2D) in Eq. 6.1 We provide the training and inference code for PyTorch implementation in the Appendix F. In order to ensure that the initialization of KV shifting does not affect the initial optimization state, we select α1 and β1 from U(0,1) and let α2 = 1 − α1, β2 = 1 − β1.

#### 3 Analysis

In this section, we will analyze the KV shifting attention. The first section will examine how KV shifting attention has a better ability to characterize induction heads compared to vanilla attention. The second section is to learn induction heads on the toy model and analyze their dynamic process. The third section provides an exploration of the capability boundary of KV shifting attention.

###### 3.1 Better representation for induction heads

The KV shifting attention reduced not only the depth but also width requirements of the transformers for forming induction heads mechanism. Improving the requirement for depth is very intuitive, while improving the requirement for width is the question we left in section

- 2.1. To strictly illustrate this point, we first modeling KV heads attention do induction heads. Then we use the theorem from previous work.

Definition 1 (Induction heads) We define the induction heads machine modeling by IH :

L∈N+ RL×D  → RD with Alibi RPE (Press et al.) as follows: IH(x) =

L−1

softmax xLxsT−1/σ − m|L − s|

### ∑

xs. (7)

i

s=2

Which has the ability to implement induction heads for any long context, when T > 0 and m > 0, which infinitely approach 0. In practice, transformers processed in a finite length, so only a very small σ and m can to be taken. 2

- 1In current popular group queries attention Ainslie et al. (2023) for large language models, the additional parameters is 4h1, where h1 refers to the number of KV pairs, not the number of heads, and the additional calculation caused by Eq. (5) is O(Nh1d1), where d1 is the head dims.
- 2We use infinite precision transformers in this article.

Previous work (Wang et al., 2024) has shown that a two-layer transformer (Alibi RPE, without FFN) can be used to approximate IH, the detail is as follow:

- Theorem 1 (Modify from Wang et al. (2024)). There exists a constant C > 0 and a two-layer single-head transformer TF(without FFNs), with D = 2d, WK(1,1) = WQ(1,1) = 0, p(2) = m, (p(i) means the Alibi bias in ith layers), and ∥WK(2,1)∥, ∥WQ(2,1)∥ ≤ O(1,1/σ), such that

sup

L∈N+

∥IH − TF∥L,∞ ≤ O(e−p(1)) (8)

If we use KV shifting attention, we can get better estimation as follow

- Theorem 2 There exists a one-layer single-head KV shifting attention KVSA, with D = d, such that

IH = KVSA. (9)

The proof in the case of using KV shifting attention is relatively simple if we are familiar with the induction heads. We include the proofs of Theorem 1 and Theorem 2 in Appendix A and Appendix B. From Theorem 1 and Theorem 2, we can find that KV shifting attention can represent or approximate induction heads with less depth and less width. In addition, since the copy operation in the first layer of the vanilla transformer introduces noise due to Alibi’s bias, the final upper bound is bounded by a quantity related to Alibi’s bias. And the KV shifting attention, due to the absence of this noise, Eq. 9 takes an equal sign. 3

For the theoretical upper limit of the transformer structure in induction heads or more generalized tasks, readers can refer to Sanford et al. (2024b). I believe that after replacing the standard Transformer with KV shifting attention, although there may not be a margin improvement in the theory bound, it is still possible to achieve a constant improvement. Next, we will leave the field of representation and enter the field of practice, which will provide experimental support for some of the assertion in this section.

###### 3.2 Great bias when learning induction heads

Various depth The architecture of the model adopts the same architecture as Llama (Touvron et al., 2023)4 with approximately 20M no-embedding parameters. We used a huge vocabulary with 8000 tokens to randomly generate sentences and ensure that the sequences in them satisfy the condition that when the jth token is the same as the ith token, then the (j + 1)th token is the same as the ith token (i < j). We present the accuracy of jth next token prediction’s accuracy in Figure 1a.

- From Figure 1a, we can see that it is indeed difficult for a one-layer standard transformer to learn the ability of induction. And the 2-layer transformer and 1-layer KV shifting attention can perfectly learn the ability of induction. At the same time, KV shifting attention has a bias towards better learning induction, and its convergence speed is much faster than two-layer transformers. At the same time, we also found that increasing the model depth to layer 4 for Vallina does not make the model learn induction faster.

Various width Hidden size for the experiment in Figure 1a is set to 1024, which is a relatively relaxed setting. As we analyzed in sections 2.1 and 2.3. Existing attention may require additional width to perform well on induction heads, so as the dimensionality of hidden states decreases, it will become increasingly difficult for standard attention to learn

- 3Theorems 1 and 2 only provide constructive upper bounds for implementing induction heads. A

more rigorous statement would be to prove that the lower bound of Theorem 1 is smaller than the upper bound of Theorem 2.

- 4These is a slightly different from the experiments in (Elhage et al., 2021), which use the attention-

only structure that is better alignment with theory. But in order to better fit the actual scenarios of large language models, we chose the llama architecture.

[Figure 6]

[Figure 7]

(a) Various depth (b) Various width

- Figure 1: On the left, as the training step size increases, the accuracy of induction varies among different models. In this setting, the only difference between Vanilla and KV shifting attention is the calculation of key and value. The total parameters of Vanilla and KV shifting attention with one layers is the same. And the parameters of Vanilla with 2 layers is twice. On the right is the induction accuracy with different hidden size. There are two layers in Vanilla model, and one layer in KV shifting attention, which means Vanilla model has two times parameters than KV shifting attention.

induction heads. Therefore, we conducted pressure testing with hidden states=8. The results are shown in Figure 1b.

- From Figure 1b, we find that the learning ability of standard attention is very poor, and even failed to cover up one of the answers mentioned in the previous text.

Although this is a toy task, people may think that the current model has a large dimension and can do the induction task well. But induction may also be done in some implicit way in language modeling. This limitation in width will result in the model considering a limited number of different implicit inductions in parallel, or introducing noise in superposition. Undoubtedly, this will have an negative impact on language modeling.

Learning from induction data Regarding how traditional two-layer transformers learn about induced heads, the analyzing it is a very complex. Previous work (Bietti et al., 2024; Wang et al., 2024; Chen et al., 2024) has provided analysis under simplified conditions. We also follow their work and provide an analysis of the dynamic process of KV shifting attention in learning induced heads. We use the following simplified conditions: removed the residual connection, MLP, normalization, position embedding and use tie embedding and the each component of the embedding of each token is from independently and identically

distributed N(0, 1d), Wq,Wk,Wv,Wo = I, and we assume the sentence’s length is T + 1 and vocabulary size is T (T ≥ 3) and every token only appearance once except the last token

appearance twice and calculate cross entropy loss only when predicting the next token on the last token. We analyzed the learning process of the four additional variables α1, α2, β1, and β2 introduced in KV shifting attention. We have:

- Theorem 3 Under the simplified condition we describe, and as d approx ∞, learning induction heads by KV shifting attention is equivalent to:

ea2β1+β2/S ea2β1+β2/S + 2eβ1/S+β2a2 + e2a1β1+a2β2 + O(T)

), (10)

min L = −log(

where a2 = eα2/S, a1 = eα1/S, S = eα2 + 2eα1 + O(T), all O(·) here is greater than or equal to 0. The detail proof is in Appendix C. The dynamic process of optimization is interesting. We consider O(T) as a constant and plot the contour lines and gradient directions of −L for α2 = 1 − α1 and β2 = 1 − β1 in Figure 2.

[Figure 9]

[Figure 10]

[Figure 11]

(a) O(T) = 0 (b) O(T) = 10 (c) O(T) = 100

- Figure 2: Contour lines and gradient decent derection of L. We simplified O(T) as a constant, and α2 = 1 − α1 and β2 = 1 − β1. Induction heads means (α1, β1) = (0,1).

From Figure 2, as O(T) increases, the contour lines become sparse, which means the convergence speed slows down. In addition, when O(T) is relatively small, the direction of GD may undergo a small non monotonic learning process. When O(T) is relatively large, the direction of GD is relatively consistent. In practice, we often have many attention heads, (α1, β1) of some heads are closer to (0,1) during initialization, making it much easier for them to learn induction heads.

When T = 2,5 training data become limitation, such as [A] [B] [A] → [B]. The model only needs to predict the previous token to achieve low loss on the training data, but this is not an induction head. Learning such a head cannot achieve [A] [B] [C] [A] → [B].

In Bietti et al. (2024), they think global bigrams are learned first, then the induction head is formed by learning appropriate memories in a top-down fashion. But obviously, in KV shifting attention, induction heads become very easy to learn, and even with good initialization, the model has induction capability. But it is difficult to have appropriate initialization to obtain a certain level of bigrams capability without training.

- 3.3 Can KV shifting attention learn n-gram better?

In addition to induction ability, an important part of language modeling is n-gram In this section, we test the model’s ability to learn n-grams. We randomly generated approximately 200 pairs of x1 and x2, and then we randomly generated x3 for each pair. The model needs to accurately predict x3 when seeing x1 and x2. The results are shown in Figure 3. Obviously, the KV shifting attention do not enhance the model’s learning ability for this 3-gram tasks, but it also does not weaken the learning ability for this 3-gram tasks.

[Figure 12]

(a) 50M (b) 0.4M (c) 0.8K

[Figure 13]

[Figure 14]

- Figure 3: Accuracy of learning 3-gram text using models of different sizes. In this experiments, there are 50M parameters model with 4 layers, 0.4M parameters model with 2 layers, 0.8K parameters model with 1 layer.

We must emphasize here that, the motivation of our KV shifting attention is to reduce the width and depth required for induction, we cannot expect KV shifting attention to greatly improve the memory ability of the model. In addition, n-gram tasks or some Markova data can actually be simulated with just one layer transformer with vanilla attention (Rajaraman et al., 2024). Therefore, the performance of Figure 3 is completely different from that of Figure 1a.

5This is actually no longer applicable to Theorem 3.

Next, we will leave the field of toy models and validate the effectiveness of KV shifting attention on large-scale models. 6

#### 4 Experiments

###### 4.1 Setting

In this section, we evaluate KV shifting attention across two models trained from scratch: a 2.9B/19B parameters within an architecture similar to Llama2 (Touvron et al., 2023). The experiments trained from scratch are conducted on Nvidia H800-80G GPUs, while others are conducted on Nvidia A100-80G GPUs.

Model Configuration Since the baseline of the model trained from scratch is for production environments, not just for this paper, it uses Group query attention (GQA) (Ainslie et al.,

- 2023) to reduce memory usage during inference and employs a larger vocabulary (48000) to cope with more multilingual environments. For this reason, our experiment on KV shfting attention, which was trained from scratch, was also based on the baseline. Instead of training the model from scratch, we used Llama’s original vocabulary size of 36000 and employed standard Multi head attention (MHA). The detail configuration is as shown in Table 6. 78

Table 1: Model Configuration.

PARAMETERS 1.5B 2.9B 6.7B 13B 19B HIDDEN SIZE 2,048 2,560 4,096 5,120 6,144

LAYERS 28 32 32 40 48 HEAD NUMBER 16 20 32 40 48 KV NUMBER 16 4 32 40 4

FFN SIZE 5,504 8,704 11,008 13,824 16,384 MAX LENGTH 2,048 4,096 2,048 2,048 12,288

TOTAL TOKENS 10B 500B 10B 10B 200B VOCAB SIZE 36,000 48,000 36,000 36,000 48,000

Datasets Due to some commercial reasons and data limitations, we used non-public private data for pre training, which means that the models are trained on our own dataset. Our data collection and filtering methods are similar to FineWeb-edu (Penedo et al., 2024). When computing resources are available, we will use open-source data, such as RedPajama1T(Weber et al.) to train two models for comparison, one is the baseline and the other is the KV shifting attention.

Hyperparameters We used a constant learning rate with a linear warmup of 1000 steps. The learning rate for 1.4B / 3B / 7B / 13B / 19B model is 2e-4 / 8e-4 / 2e-4 / 2e-4 / 2e-4, the batch size is 1M/16M/1M/2M/3M9. For optimization, we apply the AdamW optimizer with β1 = 0.9 and β = 0.95, and weight decay = 0.1.

Evaluation To validate the performance of different models, we used some benchmarks for the 3B and 19B models that were trained more tokens, including Lambada(Paperno et al., 2016), Winogrande(Sakaguchi et al., 2021), Hellaswag(Zellers et al., 2019), ARC(Clark et al., 2018), CMMLU(Li et al., 2023a), MMLU(Hendrycks et al.), Math(Hendrycks et al., 2021).

- 6There are some additional small-scale experiments on multi-hop and math, which is in the

Appendix H and Appendix I.

- 7The Vanilla-2.9B can be download from https://huggingface.co/xumingyu16/Baseline_2.9B.
- 8The KV shifting-2.9B can be download from https://huggingface.co/xumingyu16/KV_shifting_2.9B. 91M here means 1,048,576 = 1,024 × 1,024, while elsewhere in the paper it refers to 1,000,000

- Table 2: Main results. We trained four models on 2.9B and 19B parameters respectively, with the 2.9B model having a total training token count of 500B and the 19B model having a total training token count of 200B. ARC-E is short for ARC-easy, and ARC-C if short for ARC-Challenge.

###### MODEL TOKENS LAMBADA WINOGRANDE HELLASWAG ARC-E ARC-C CMMLU MMLU MATH AVERAGE

|VANILLA - 2.9B<br><br>340B 420B 500B|52.92 52.09 42.70 27.45 25.97 28.51 29.43 0.80 52.80 54.85 43.68 28.96 26.02 34.77 30.34 1.20 51.66 54.06 44.49 36.20 27.90 38.22 37.26 1.80<br><br>|32.48 34.08 36.45|
|---|---|---|
|KV SHITING - 2.9B<br><br>340B 420B 500B<br><br>|55.44 53.91 42.87 36.74 30.04 34.51 36.20 2.00 51.91 54.78 43.83 36.66 31.91 37.24 34.30 1.80 54.51 55.33 44.52 39.02 30.89 40.78 40.88 2.60|36.46 36.55 38.57<br><br>|
|VANILLA - 19B<br><br>160B 180B 200B<br><br>|59.93 48.22 48.25 30.34 24.56 39.12 39.22 1.80 58.80 48.07 47.78 31.28 25.99 40.80 39.34 2.60<br><br>60.88 49.01 47.36 33.25 25.78 42.92 42.68 2.60<br><br><br>|36.43 36.83 38.06|
|KV SHIFTING - 19B<br><br>160B 180B 200B|61.93 48.46 48.28 31.25 25.06 42.10 42.87 2.00<br><br>60.20 47.67 48.16 32.45 26.55 43.38 40.49 3.00<br><br>62.35 48.38 48.42 33.28 29.32 42.40 43.29 3.20<br>|37.74 37.74 38.83<br><br>|

While for the models that were trained less token, we could only look at their loss curve. Due to computation limitations, almost experiments are only run once except there is an additional notion.

###### 4.2 Main Result

We pre trained language models at two scales with 2.9B and 19B parameters, and the experimental results are shown in Table 2. The results indicate that KV Shifting attention achieved better performance than baseline at various scales and training token numbers. In addition, we also plotted the training loss of them, as shown in Figure 4.

[Figure 17]

[Figure 18]

(a) 2.9B (b) 19B

- Figure 4: Training loss curve. We train 2.9B model with 500B tokens, and 19B models with 200B tokens.

In Table 2, we can find that KV shifting attent KV Shifting introduces a bias that is more suitable for language modeling, accelerating the convergence of the model. And it seems that both 2.9B models are converge on the benchmark Lambda, and KV shifting attention can achieve better results. Here we slightly argue that KV shifting attention can achieve better performance than the vanilla model when they both converge, although it may take several TB data for a 2.9B model to converge.

###### 4.3 Metrics for evaluation

In this section, we conducted mmlu evalutaion under three condition, few shot (5 - shot), zero shot and cloze (zero shot). The setting of cloze is followed by Waleffe et al. (2024), which intends to break away from the format of standard multiple-choice and directly measure the knowledge. The test results are shown in Table 3. It can be seen that compared to vanilla, KV shifting attention not only enhances the ability of in-context learning, but also accelerates the

model’s learning of world knowledge. Moreover, the format of multiple-choice questions may be advantageous for KV shifting attention. The model can easily use context to compare the possibilities of various options and select the option with the highest probability.

- Table 3: We compare vanilla and KV shifting at 2.9B model with 500B training tokens by using different evaluation metric.

BENCHMARK VANILLA KV SHIFTING

CLOZE ZERO FEW CLOZE ZERO FEW MMLU 30.41 33.14 37.26 32.17 37.13 40.88

In addition, under different evaluation metric, KV shifting attention achieved better results, which reflects the robust of KV shifting attention.

###### 4.4 Robust Experiment

To verify the robustness of the model, we used different random seeds for 1.5B model, specifically, we set random number seeds for model initialization and data sampling, and conducted five experiments. The Vanilla and KV shifting attention in each experiment using the same re initialization and data. As shown in Figure 5a, although the training loss is quite shaky, it can be seen that under different random seeds, KV shifting attention is always better than vanilla.

[Figure 20]

[Figure 21]

[Figure 22]

(a) Various Seeds (b) Various LR (c) LR=1e-2

Figure 5: Training loss of 1.5B parameters model among random seeds and learning rate (LR).

In addition, we also conducted experiments on different learning rates, including 1e-4, 2e-4, 1e-3, and 1e-2, as shown in Figure 5b. It can be observed that KV shifting attention achieves better results than Vanilla at different learning rates. And when the learning rate is set to 1e-2, Vanilla has already diverged, while the loss of KV shifting attention has not yet diverged as shown in Figure 5c10. This suggests that the optimization space for KV shifting attention may be flatter. Shifting KV provides a smooth key and value during model initialization, which may make the optimization process smoother. This also partially explains why in Figure 4, the 2.9B KV shifting attention leads more in terms of loss, because the 2.9B model is trained with a large learning rate11.

###### 4.5 Scaling Experiment

Firstly, we plotted the training loss curves at the 1.5B, 6.7B, and 13B parameters as shown in Figure 6. We found that under different settings, KV shifting attention achieved better results compared to vanilla model.

Afterwards, we also followed Kaplan et al. (2020) and used WebText (Radford et al.) as the validation set to draw a scaling law for vanilla and KV shifting attention. As shown

- 10To confirm this, we attempted 5 random seeds under the condition of LR=1e-2, and the all results of each experiment are Vanilla divergence and KV shifting convergence.
- 11We adopt a large learning rate in practice, because Lobacheva et al. suggests large learning rates improve generalization

[Figure 24]

[Figure 25]

[Figure 26]

(a) 1.5B Parameters (b) 6.7B Parameters (c) 13B Parameters

- Figure 6: Training loss comparison between different size. All models are trained on 10B tokens. The batch size for 1.5B and 6.7B model is 0.5M, for 13B is 1M, so the total steps of 13B model is half of others.

in Figure 7a12, KV shifting attention also has excellent scaling properties and outperforms baseline at every parameter scale. And we continued to train the 1.5B model to 30B tokens, which we calculate the validation set loss every 1000 steps, but the difference in validation loss still does not decrease, as shown in Figure 7b.

[Figure 27]

(a) Scaling law (b) 1.5B Parameters

[Figure 28]

- Figure 7: Validation loss across different size and training tokens. For scaling law, while others in this paper means the total parameters. models are trained on 10B tokens and calculate the final checkpoint’s validation loss.

###### 4.6 Learnable parameter analysis

One worth studying is how these learnable parameters will change in a pre trained model. We conducted research on the 2.9B model, and due to our initialization, αi and βi are independent random variables, but as training progressed, they became coupled with each other. As shown in the Table 4, we have counted the number of whether α1 ≤ α2 and β1 ≤ β2 in each KV pair.

- Table 4: We calculate the number of whether α1 ≤ α2 and β1 ≤ β2 in each KV pair in 2.9B model with 500B token, the total numbers is 128.

α1 > α2 α1 ≤ α2

β1 > β2 50 17 β1 ≤ β2 9 52

As shown in the Table 4, we find that the numbers on the diagonal dominate, which means that KV pairs with same relative size relationships between α1 and α2, β1 and β2, has become

12In this figure, the parameters is the no-embedding parameters, followed by Kaplan et al. (2020), while others in this paper mean total parameters.

the majority. And we find that the diagonal was already 47,43 when training 20B tokens. This indicates that for most heads, the model tends to use the key and value of the same token.

The heads in the upper right focus on the key of the (i − 1)th token to obtain information about the ith token, while the heads in the lower left focus on the key of the ith token to obtain information about the (i − 1)th token. And they are not symmetrical. We speculate that this is because the model can easily obtain information about the (i − 1)th token by interacting with ith token under the causal mask, as the (i − 1)th token token may contain some information about the (i − 1)th token token to some extent. Therefore, the lower left corner will be relatively small.

Besides, we find that for the trained 2.9B model with 500B tokens, ∑i αi and ∑i αi is away from 1, although they are 1 when initializing. However, a common gating mechanism

is to use activation functions to control ∑i αi = 1 and ∑i αi = 1 during the training (e.g. α1 = Sigmoid(a), β1 = Sigmoid(b), α2 = 1 − α1, β = 1 − α2, where a, b ∈ R is the learnable parameters, we call it KV shifting gate). In addition, during initialization, these parameters are all between 0 to 1, but as training progresses, some parameters become negative and far from zero.13 Do we need to keep the model between 0 to 1 during the training process (e.g. αi = min(max(αi,0),1), βi = min(max(βi,0),1), we call it KV shifting 0 to 1.)?

gateFor this purpose, we conducted experiments on the 1.5B parameters model with 10B token, as shown in Figure 8a. Using more controls does not make the model learn better. Allowing α and β to have a wider range of degrees of freedom may enable the model to learn richer features. For example, Elhage et al. (2021) discovered the presence of "anti-copying prefix-search" heads in vanilla model. Although we don’t know what its function is, if we restrict β2 ≥ 0, it is likely to limit the generation of this kind of heads.

[Figure 30]

[Figure 31]

[Figure 32]

(a) Variant (b) Ablation (c) More shifting

- Figure 8: Further experiments are conducted on a 1.5B model, where we trained 10B tokens.

###### 4.7 Further experiments

To further validate the effectiveness of the proposed KV shifting attention, we conducted the following ablation experiments. Firstly, in the experiment of ablating the shifting of K and V, as shown in Figure 8b, we find that the shifting of k and v plays an important role. It can be inferred that obtaining the value of the (i − 1)th token or the value of the (i + 1)th token by focusing on the key of the ith token is important in language modeling. If K shifting or V shifting is not used, the model needs to use two layers of attention to indirectly implement this operation.

In addition, we also performed longer shifts on k and v, extending the shift between i − 1 and i to [i − 2,i] and [i − 3,i]14. The experimental results are shown in Figure 8c. The results have shown that using a longer shift window does not improve performance, although increases computational complexity. From the perspective of induction heads, using our

- 13e.g. α1 = 0.08, α2 = 0.43, β1 = 0.34, β2 = −0.15 in the 3th KV pairs of the 17th layers. The previous token’s key to dominate the current token’s key, while subtracting the previous token’ value from the current token’s value.
- 14KV shifting 1 is the KV shifting we used in our paper. For KV shifting 2, we randomly initialize α1, α2 from U(0,1), and then use 1 − α1 − α2 as α3. It is similar for β and KV shifting 3.

current shifting window size is sufficient. If someone want to expand to longer windows, it may need to carefully design it. In this paper, we will no longer attempt more refined designs, as current lightweight designs are sufficient for better learning of induction heads and improving language modeling.

#### 5 Discussion

Overall, based on previous research on the induction heads mechanism in the Transformer model, we have designed an attention mechanism that is more suitable for learning induction heads. This can not only reduce the demand for width, but also reduce the demand for layers to learn induction heads. At the same time, in large-scale language model pre training, the use of KV shifting attention can achieve better results than baseline in many experimental settings, which implicitly demonstrates the importance of learning induction heads for language modeling.

Another noteworthy fact is that the previous research (Elhage et al., 2021; Akyürek et al., 2024) think Transformers to outperform LSTMs at in-context learning on natural text, with induction heads as a major explanation. Our experiments indicate that existing transformer mechanisms still have not fully unleashed the potential of learning induction heads. By making slight modifications to the original architecture, the model can learn induction heads much faster than before.

If we are not ready to completely overturn the Transformer architecture, it may be interesting to delve into its underlying mechanisms, identify important features, and modify the Transformer to make it easier to implement certain important functions. For example, by modifying transformer to make it easy to learn parity check, which might be very challenging (Wies et al., 2023).

Due to our lightweight modifications, the KV shifting attention can easily be compatible with existing training inference frameworks. We believe that there is not only one way for models to better learn induction heads. But some more complex operations may be difficult to adapt to under existing training or inference acceleration frameworks.

From the perspective of operation, the fusion of adjacent information in neural network has a long history, such as Wu et al. (2018) in vision, Zhang et al. (2021) in video, Li et al. (2023b) in speech, and Peng et al. (2023) in text. From the perspective of cognitive psychology, this aggregation of neighboring information also occurs in humans (Todorovic, 2008). How to better utilize the information of neighboring tokens may be a consideration in language modeling, especially for pure attention with permutation invariance. Currently, many LLMs use Rotary positional embedding (Su et al., 2024) to better utilize the information of neighboring tokens.

Besides, using KV shifting attention may be much easier to locate the induction heads, which can be helpful for mechanistic interpretability. Of course, if we start from making the model more interpretable, we can also impose some constraints on the original Transformer (Friedman et al., 2024), but this may not improve the performance of the model like KV shifting attention.

#### 6 Limitation

Due to limitations in computing resources and other limitations, many experiments cannot be repeated many times, and it is not convenient for us to use open-source datasets or open source our datasets. But our robustness experiments have convinced us that KV shifting attention can achieve better results in many experimental settings, and we have also open-source our pertained model.15 In theory, we provide a way for KV shifting attention to learn induction heads under relaxed conditions. However, in more complex multi-layer

15We are delighted that after our paper was published on Arxiv, someone has implemented it in open source and achieved better results than Vanilla. https://github.com/erogol/BlaGPT.

transformer conditions, how to learn induction heads and more complex reasoning skills remains a challenging problem.

#### 7 Related works

Induction Heads Induction heads, introduced by (Olsson et al., 2022; Elhage et al., 2021), are specialized attention mechanisms that identify patterns in sequences, enabling large language models to predict subsequent tokens based on previous patterns. There are numerous studies(Bansal et al., 2023; Conmy et al., 2023; Wang et al.; Ren et al., 2024; Todd et al.) starting from induction heads to investigate the interpretability of large models, in order to enhance their transparency. Another part of works(Bietti et al., 2024; Wang et al., 2024; Sanford et al., 2024b; Chen et al., 2024) are to study how the model learns induction heads from a theoretical perspective. Previous studies motivated us greatly, and in this work, we modify the attention to make the model better learn induction heads.

Model Structure and Language Modeling Over the past many years, people have attempted various structures to enhance model’s language modeling abilities. From early RNN (Mikolov et al., 2010) and LSTM(Sundermeyer et al., 2012) to the dominant transformer(Vaswani, 2017) today. Transformers have quadratic complexity, and many efforts have been made to improve them, such as RWKV (Peng et al., 2023), Mamba (Gu & Dao, 2023), RetNet (Sun et al., 2023). However, transformers still have many excellent properties that cannot be replaced temporarily, especially their ability to retrieve and replicate previous information (Jelassi et al.; Alman & Yu, 2024). Currently, popular language models (Achiam et al., 2023; Dubey et al., 2024; Yang et al., 2024) still use Transformers as architecture. Recently, there have been some efforts to modify transformers to enhance modeling capabilities, such as reducing the noise of attention (Ye et al., 2024a) or reducing attention to unneeded elements (Leviathan et al., 2024). Our work is to slightly modify the attention to enhance its ability to learn induction heads, which potentially improves language modeling.

#### 8 Conclusion

In this work, we analyzed that induction heads have certain requirements for both the width and depth of the transformer, which motivated us to implement the induction mechanism more efficiently. For stronger expressiveness and faster convergence for the learning induction heads, we proposed the KV shifting attention, enhancing the language modeling ability of the decode-only structure transformers. We conducted extensive experiments to verified the effectiveness of KV shifting attention, including pre training of 2.9B and 19B parameter models. We hope this work can provide inspiration for achieving more powerful language modeling.

#### 9 Acknowledgments

Thank you to Qingyu Zhang for discussing with me and providing valuable suggestions. And I would like to thank countless predecessors for their research and open sourcing of the code, which saved me a lot of time.

#### References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebron, F., and Sanghai, S. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 4895–4901, 2023.

Akyürek, E., Wang, B., Kim, Y., and Andreas, J. In-context language learning: Architectures and algorithms. In Forty-first International Conference on Machine Learning, 2024.

Alman, J. and Yu, H. Fundamental limitations on subquadratic alternatives to transformers. arXiv preprint arXiv:2410.04271, 2024.

Bansal, H., Gopalakrishnan, K., Dingliwal, S., Bodapati, S., Kirchhoff, K., and Roth, D. Rethinking the role of scale for in-context learning: An interpretability-based case study at 66 billion scale. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11833–11856, 2023.

Bietti, A., Cabannes, V., Bouchacourt, D., Jegou, H., and Bottou, L. Birth of a transformer: A memory viewpoint. Advances in Neural Information Processing Systems, 36, 2024.

Chen, S., Sheen, H., Wang, T., and Yang, Z. Unveiling induction heads: Provable training dynamics and feature learning in transformers. arXiv preprint arXiv:2409.10559, 2024.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Conmy, A., Mavor-Parker, A., Lynch, A., Heimersheim, S., and Garriga-Alonso, A. Towards automated circuit discovery for mechanistic interpretability. Advances in Neural Information Processing Systems, 36:16318–16352, 2023.

Crosbie, J. and Shutova, E. Induction heads as an essential mechanism for pattern matching in in-context learning. arXiv preprint arXiv:2407.07011, 2024.

Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai,

- Y., Chen, A., Conerly, T., DasSarma, N., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., and Olah, C. A mathematical framework for transformer circuits. Transformer Circuits Thread, 2021. https://transformercircuits.pub/2021/framework/index.html.

Friedman, D., Wettig, A., and Chen, D. Learning transformer programs. Advances in Neural Information Processing Systems, 36, 2024.

Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Jelassi, S., Brandfonbrener, D., Kakade, S. M., et al. Repeat after me: Transformers are better than state space models at copying. In Forty-first International Conference on Machine Learning.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Leviathan, Y., Kalman, M., and Matias, Y. Selective attention improves transformer. arXiv preprint arXiv:2410.02703, 2024.

Li, H., Zhang, Y., Koto, F., Yang, Y., Zhao, H., Gong, Y., Duan, N., and Baldwin, T. Cmmlu: Measuring massive multitask language understanding in chinese. arXiv preprint arXiv:2306.09212, 2023a.

Li, Y., Wu, Y., Li, J., and Liu, S. Accelerating transducers through adjacent token merging. arXiv preprint arXiv:2306.16009, 2023b.

Lobacheva, E., Pokonechny, E., Kodryan, M., and Vetrov, D. Large learning rates improve generalization: But how large are we talking about? In NeurIPS 2023 Workshop on Mathematics of Modern Machine Learning.

Lozhkov, A., Ben Allal, L., von Werra, L., and Wolf, T. Fineweb-edu, May 2024. URL

###### https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu.

Mikolov, T., Karafiát, M., Burget, L., Cernocky,` J., and Khudanpur, S. Recurrent neural network based language model. In Interspeech, volume 2, pp. 1045–1048. Makuhari, 2010.

Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Johnston, S., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., and Olah, C. In-context learning and induction heads. Transformer Circuits Thread, 2022. https://transformer-circuits.pub/2022/incontext-learning-and-induction-heads/index.html.

Paperno, D., Kruszewski, G., Lazaridou, A., Pham, N.-Q., Bernardi, R., Pezzelle, S., Baroni, M., Boleda, G., and Fernández, R. The lambada dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1525–1534, 2016.

Penedo, G., Kydlícek, H., Allal, L. B., and Wolf, T. Fineweb: Decanting the web for the finest text data at scale. HuggingFace. Accessed: Jul, 12, 2024.

Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., Biderman, S., Cao, H., Cheng, X., Chung, M., Grella, M., et al. Rwkv: Reinventing rnns for the transformer era. arXiv preprint arXiv:2305.13048, 2023.

Press, O., Smith, N., and Lewis, M. Train short, test long: Attention with linear biases

enables input length extrapolation. In International Conference on Learning Representations. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. Language models are

unsupervised multitask learners. Rajaraman, N., Bondaschi, M., Ramchandran, K., Gastpar, M., and Makkuva, A. V. Transformers on markov data: Constant depth suffices. arXiv preprint arXiv:2407.17686, 2024. Ren, J., Guo, Q., Yan, H., Liu, D., Zhang, Q., Qiu, X., and Lin, D. Identifying semantic

induction heads to understand in-context learning. arXiv preprint arXiv:2402.13055, 2024. Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y. Winogrande: An adversarial

winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021. Sanford, C., Hsu, D., and Telgarsky, M. One-layer transformers fail to solve the induction heads task. arXiv preprint arXiv:2408.14332, 2024a.

Sanford, C., Hsu, D., and Telgarsky, M. Transformers, parallel computation, and logarithmic depth. In Forty-first International Conference on Machine Learning, 2024b.

Song, J., Xu, Z., and Zhong, Y. Out-of-distribution generalization via composition: a lens through induction heads in transformers. arXiv preprint arXiv:2408.09503, 2024.

Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., and Wei, F. Retentive network: A successor to transformer for large language models. arXiv preprint arXiv:2307.08621, 2023.

Sundermeyer, M., Schlüter, R., and Ney, H. Lstm neural networks for language modeling. In Interspeech, volume 2012, pp. 194–197, 2012.

Todd, E., Li, M., Sharma, A. S., Mueller, A., Wallace, B. C., and Bau, D. Function vectors in

large language models. In The Twelfth International Conference on Learning Representations. Todorovic, D. Gestalt principles. Scholarpedia, 3(12):5345, 2008. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra,

S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Vaswani, A. Attention is all you need. Advances in Neural Information Processing Systems, 2017.

Waleffe, R., Byeon, W., Riach, D., Norick, B., Korthikanti, V., Dao, T., Gu, A., Hatamizadeh, A., Singh, S., Narayanan, D., et al. An empirical study of mamba-based language models. arXiv preprint arXiv:2406.07887, 2024.

Wang, K. R., Variengien, A., Conmy, A., Shlegeris, B., and Steinhardt, J. Interpretability in the wild: a circuit for indirect object identification in gpt-2 small. In The Eleventh International Conference on Learning Representations.

Wang, M., Yu, R., Wu, L., et al. How transformers implement induction heads: Approximation and optimization analysis. arXiv preprint arXiv:2410.11474, 2024.

Weber, M., Fu, D. Y., Anthony, Q., Oren, Y., Adams, S., Alexandrov, A., Lyu, X., Nguyen, H.,

Yao, X., Adams, V., et al. Redpajama: an open dataset for training large language models. Wies, N., Levine, Y., and Shashua, A. Sub-task decomposition enables learning in sequence

to sequence tasks. In The Eleventh International Conference on Learning Representations, 2023.

Wu, B., Wan, A., Yue, X., Jin, P., Zhao, S., Golmant, N., Gholaminejad, A., Gonzalez, J., and Keutzer, K. Shift: A zero flop, zero parameter alternative to spatial convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 9127–9135, 2018.

Xu, M., Men, X., Wang, B., Zhang, Q., Lin, H., Han, X., and weipeng chen. Base of roPE bounds context length. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=EiIelh2t7S.

Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C., Li, C., Li, C., Liu, D., Huang, F., et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.

Ye, T., Dong, L., Xia, Y., Sun, Y., Zhu, Y., Huang, G., and Wei, F. Differential transformer. arXiv preprint arXiv:2410.05258, 2024a.

Ye, T., Xu, Z., Li, Y., and Allen-Zhu, Z. Physics of language models: Part 2.1, grade-school math and the hidden reasoning process. arXiv preprint arXiv:2407.20311, 2024b.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4791–4800, 2019.

Zhang, H., Hao, Y., and Ngo, C.-W. Token shift transformer for video classification. In Proceedings of the 29th ACM International Conference on Multimedia, pp. 917–925, 2021.

#### A Proof of Theorem 1

Proof. We consider two-layer single-head transformer without FFN, where the first layer has the residual block, while the second layer does not have the residual block.

0 0 Id×d 0

xs 0

and take WV(1) =

We first embed each token into RD as

, then the s-th output token of the first layer is

xs ys =

xs ∑τs−=11 softmax −p(1)(s − 1 − τ) xτ

.

Then for the second layer, we choose p(2) = m,

WQ(2,1) =

0 0 Id×d 0

,WK(2,1) =

0 0 0 W⋆

,WV(2,1) =

Id×d 0

0 0 ∈ RD×D,

and the projection WO(2) = (Id×d,0d×d) ∈ Rd×D. Then the last output token of the second layer is

L−1

### ∑

softmax(x⊤L ys/σ − m|L − s|)xs

s=2

By Lemma D.1, for any L ∈ N+

##### ∥IH2 − TF∥L,∞

= sup

∥IH(XL) − TF−1(XL)∥∞

XL

L−1

### ∑

softmax(x⊤L ys/σ − m|L − s|)xs

=

s=2

L−1

### ∑

softmax(x⊤L xs−1/σ − m|L − s|)xs

−

∞

s=2

L−1

### ∑

|softmax(x⊤L ys/σ − m|L − s|)

≤

s=2

− softmax(x⊤L xs−1/σ − m|L − s|)| ≤ 2sup

|x⊤L ys/σ − x⊤L xs−1/σ| ≤ 2∥x⊤L /σ∥1 sup

s

∥ys − xs−1∥∞ ≤ 2∑

s

s−1

softmax −p(1)(s − 1 − τ) xτ − xs−1

### ∑

|I/σ| sup

s

∞

τ=1

i,j

s−1

softmax −p(1)((s − 1 − τ))

≤ 2∥I/σ∥1,1 sup

###### − es−1

s

τ=1

1

1

∑sτ−=20 e−p(1)τ − 1 < 4∥I/σ∥1,1

= 4∥I/σ∥1,1 sup

s

e−p(1)

1 − e−p(1) ≤ O e−p(1) .

#### B Proof of Theorem 2

If we set α1 = 0, α2 = 1, β0 = 1/σ, β2 = 0, p = m (the bias of Alibi position embedding), Wq = Wk = Wo = Wv = I ∈ Rd×d, we can get the proof.

#### C Proof of Theorem 3

We consider embeddings uk ∈ Rd with i.i.d. Gaussian N(0, d1) entries. We recall a few facts:

- - (Norm) We have ui⊤ui = 1 + O(1/√d). Because ui⊤ui is a scaled chi-squared distribution, with mean = 1, and variance 2/d.

- - (Near-orthogonality) For i ̸= j, we have ui⊤uj = O(1/√d). To see this, denoting ui = d−1/2(u˜ik)k, where u˜ik are the normalized entries of ui, note that we have

###### √

1 √d

dui⊤uj =

d

### ∑

u˜iku˜jk → N(0,1),

k=1

by the central limit theorem, since for each k, the quantities u˜iku˜jk are zero-mean, unitvariance, i.i.d. random.

Although we can describe it more accurately, in (10) we actually only used when d tends towards positive infinity, ui⊤ui = 1 and ui⊤uj = 0. Assume the sequence is x1, x2,..., xi, xi+1,..., xT, xT+1, where xT+1 = xi, and x1, x2,..., xT are different.16 Now let’s start our proof:

Calculate attention score The last token attend to the kth token, the attention score aˆk before softmax can be calculate as: If k = i, aˆk = xi⊤(α1xi + α2xi−1) = α1 + O(α1√+dα2 ); If k = i + 1, aˆk = xi⊤(α1xi+1 + α2xi) = α2 + O(α1√+dα2 ); If k = T + 1, aˆk = xi⊤(α1xi + α2xn) = α1 + O(α1√+dα2 ); Other else, aˆk = xi⊤(α1xk + α2xk−1) = O(α1√+dα2 ). So, the attention score ak after softmax can be calculate as: If k = i, ak = eα1O(e

α1√+dα2 ))/S;

α1√+dα2 ))/S;

If k = i + 1, ak = eα2O(e

α1√+dα2 ))/S;

If k = T + 1, ak = eα1O(e

α1√+dα2 ))/S, where S = 2eα1O(e

α1√+dα2 )) + eα2O(e

α1√+dα2 )) + O(Te

α1√+dα2 ).

Other else, ak = O(e

Calculate logits Now we calculate the logits of predict kth token, donate as lk′: If k′ = i, lk′ = ∑kT=+11 xi⊤(β1xk + β2xk−1)ak = (β1 + O(β1√+dβ2 ))ai + (β2 + O(β1√+dβ2 ))ai+1 + (β1 + O(β1√+dβ2 ))aT+1 + O(T)O(β1√+dβ2 )O(e

α1√+dα2 ))/S = (2β1eα1 + 2eα1O(β1√+dβ2 ) + β2eα2 + eα2O(β1√+dβ2 ) + O((β1+√βd2)T))O(e

α1√+dα2 ))/S. (Expand according to k=i, i+1, T+1, other else.)

16Here, we assume i > 1 and i < T, if not, there will be a slight difference in proof.

If k′ = i +1, lk′ = ∑kT=+11 xi⊤+1(β1xk + β2xk−1)ak = O(β1√+dβ2 )ai +(β1 +O(β1√+dβ2 ))ai+1 +(β2 + O(β1√+dβ2 ))ai+2 + O(β1√+dβ2 )aT+1 + O(T)O(β1√+dβ2 )O(e

α1√+dα2 ))/S = (2eα1O(β1√+dβ2 ) + β1eα2 + eα2O(β1√+dβ2 )+ β2 +O((β1+√βd2)T))O(e

α1√+dα2 ))/S (Expand according to k=i, i+1, i+2, T+1, other

else.) If k′ = T, lk′ = ∑kT=+11 xT⊤(β1xk + β2xk−1)ak = O(β1√+dβ2 )ai + O(β1√+dβ2 )ai+1 + (β1 + O(β1√+dβ2 ))aT + (β2 + O(β1√+dβ2 ))aT+1 + O(T)O(β1√+dβ2 )O(e

α1√+dα2 ))/S = (eα1O(β1√+dβ2 ) + eα2O(β1√+dβ2 ) + β1 + β2eα1 + O((β1+√βd2)T))O(e

α1√+dα2 ))/S (Expand according to k=i, i+1, T,

T+1, other else.) If k′ = i − 1, lk′ = ∑kT=+11 xi⊤−1(β1xk + β2xk−1)ak = (β1 + O(β1√+dβ2 ))ai−1 + (β2 + O(β1√+dβ2 ))ai + O(β1√+dβ2 )ai+1 + O(β1√+dβ2 )aT+1 + O(T)O(β1√+dβ2 )O(e

α1√+dα2 ))/S = (β1 + β2eα1 + eα2O(β1√+dβ2 ) + 2eα1O(β1√+dβ2 ) +O((β1+√βd2)T))O(e

α1√+dα2 ))/S (Expand according to k=i-

1, i, i+1, T+1, other else.) Other else, lk′ = ∑kT=+11 xk⊤′(β1xk + β2xk−1)ak = O(β1√+dβ2 )ai + O(β1√+dβ2 )ai+1 + (β1 + O(β1√+dβ2 ))ak′ + (β2 + O(β1√+dβ2 ))ak′+1 + O(T)O(β1√+dβ2 )O(e

α1√+dα2 ))/S = (eα1O(β1√+dβ2 ) + eα2O(β1√+dβ2 ) + β1 + β2 + O((β1+√βd2)T))O(e

α1√+dα2 ))/S (Expand according to k=i, i+1, k′, k′+1,

other else.) As d tending towards positive infinity, we summay the logits of ith token in Table 5:

|predict tokens<br><br>|logit|
|---|---|
|i - 1 i i+1 T other else<br><br>|(β1 + β2eα2)/S (2β1eα1 + β2eα2)/S (β1eα2 + β2)/S (β1 + β2eα2)/S (β1 + β2)/S|

Table 5: Logits summary, when d tend to ∞, where S = 2eα1 + eα2 + O(T)

Calculate loss We denote eα1/S as a1, eα2/S as a2, than we can get:

eli+1 ∑iT=1 eli

) (11)

Loss = −log(

ea2β1+β2/S ea2β1+β2/S + 2eβ1/S+β2a2 + e2a1β1+a2β2 + O(T)

) (12)

= −log(

#### D Python script for generating induction data

This is a Python script that generates induction data. The model will randomly select a number from mid val to max val as a token. If this token has already appeared, the next token will be the same as the previous one. When the sequence length is less than 512, 0 will be added. If there is no induction data in the sequence, it will be regenerated. The predict positions returned by this function are the positions where the accuracy of the induction is calculated.

def generate_array(length=512, min_val=1, mid_val=10,max_val =8000): lis = list(range(mid_val+1,max_val))

random.shuffle(lis) lis = lis[:length] array = [] predict_positions = [] di = {} while len(array)<length:

x = random.choice(lis) if (len(array) == 0) or (x!=array[-1]):

if x not in di: if len(array) >0:

di[array[-1]] = x di[x] = -1 array.append(x)

else: predict_positions.append(len(array)) array.append(x) array.append(di[x]) return array + [0]*(512 -len(array)), predict_positions

[0:1] if len(predict_positions) == 0: return generate_array(length , min_val , mid_val ,max_val)

#### E Python code for drawing a streamline diagram

We provide a Python code for drawing streamline diagrams in Figure 2.

import torch import numpy as np import matplotlib.pyplot as plt def f(alpha ,beta ,ot = 0):

- alpha1 = alpha

- alpha2 = 1 - alpha

- beta1 = beta

- beta2 = 1 - beta

- a1 = torch.exp(alpha1)/(2* torch.exp(alpha1)+torch.exp(alpha2)+ot)

- a2 = torch.exp(alpha2)/(2* torch.exp(alpha1)+torch.exp(alpha2)+ot) S = (2*torch.exp(alpha1)+torch.exp(alpha2)+ot) target = a2*beta1 + (beta2/S)

- no1 = (beta1/S) + beta2*a2

- no2 = 2*a1*beta1 + a2+beta2 exp_target = torch.exp(target)

- exp_no1 = torch.exp(no1)

- exp_no2 = torch.exp(no2) exp_ot = torch.exp(ot) denominator = exp_target + 2 * exp_no1 + exp_no2 + exp_ot prob_ratio = exp_target / denominator loss = -torch.log(prob_ratio) return loss

X,Y = np.meshgrid(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000))

- X = torch.tensor(X, requires_grad=True)

- Y = torch.tensor(Y, requires_grad=True)

- Z = f(X,Y,0.00001 ,0)

- Z.sum().backward()

- grad_X = X.grad

- grad_Y = Y.grad fig , ax = plt.subplots() strm = ax.streamplot(

X.detach().numpy(), Y.detach().numpy(),

-grad_X.detach().numpy(), -grad_Y.detach().numpy() )

levels = np.arange(0, 10, 0.05) contours = plt.contour(

X.detach().numpy(), Y.detach().numpy(), Z.detach().numpy(), levels=levels , colors=’black’

) ax.set_xlabel(r’$\alpha_1$’) ax.set_ylabel(r’$\beta_1$’) plt.show()

#### F Pytorch code for KV shifting attention

We provide the following Python code that can easily implement KV shifting attention with rotary embedding. In this example, we used convolution operation to perform shifting operations.

from torch import nn import torch.nn.functional as F from flash_attn import flash_attn_func def custom_convolution(U, K):

bs, seq , h, d = U.shape h, w = K.shape padding = (w - 1, 0) U_padded = F.pad(U, (0, 0, 0, 0, *padding)) # (bs, seq+w-1, h, d

)

- U_unfolded = U_padded.unfold(1, w, 1) # (bs, seq+w-1, h, d, w) K_expanded = K.unsqueeze (0).unsqueeze (0).unsqueeze(-2) # (1, 1,

h, 1, w)

- V_unfolded = U_unfolded * K_expanded # (bs, seq , h, d, w) V = V_unfolded.sum(dim=-1) # (bs, seq , h, d) return V

def __init__(self): K = torch.rand(self.num_kv_heads ,1) V = torch.rand(self.num_kv_heads ,1) self.K = nn.Parameter(torch.cat([K,1-K],dim=1)) self.V = nn.Parameter(torch.cat([V,1-V],dim=1))

def foward(self ,q,k,v): k = custom_convolution(k, self.K) v = custom_convolution(v, self.V) q, k= self.rotary_emb(q, k, seqlen_offset =0) attn_outputs = flash_attn_func(

q, k, v, causal=True

)

The following code can be used for inference:

if past_key_value is None: k = custom_convolution(k, self.K) v = custom_convolution(v, self.V) self.last_k = k[:,-1:] self.last_v = v[:,-1:]

else:

self.last_k ,k = k, self.K[:,:1]*self.last_k + self.K[:,1:]*k self.last_v ,v = v, self.V[:,:1]*self.last_v + self.V[:,1:]*v

#### G Experimental setup details

We conducted toy models for induction heads on 8 Nvidia A100-80G GPUs, with 512 samples per GPU. The learning rate is 2e − 4 with 1000 steps warm-up. For large language

model, due to privacy reasons, we are unable to provide a detailed description of the training data here. Our training data contains a large amount data from Fineweb-edu (Lozhkov et al., 2024), as well as some other filtered data. We have listed the parameters of our models of various sizes below, some of which have also been mentioned in the main text. We used a larger RoPE’s base here because previous study (Xu et al., 2024) has shown that the longer the context length, the larger the base required, while the default base=10,000 is relatively small, even for 2048 windows. Therefore, a relatively large value has been uniformly set here.

Table 6: Configuration.

PARAMETERS 1.5B 2.9B 6.7B 13B 19B HIDDEN SIZE 2,048 2,560 4,096 5,120 6,144

LAYERS 28 32 32 40 48 HEAD NUMBER 16 20 32 40 48 KV NUMBER 16 4 32 40 4

FFN SIZE 5,504 8,704 11,008 13,824 16,384

MAX LENGTH 2,048 4,096 2048 2,048 12,288 TOTAL TOKENS 10B 500B 10B 10B 200B

VOCAB SIZE 36,000 48,000 36,000 36,000 48,000

GPU A100-80G H800-80G A100-80G A100-80G A800-80G GPU NUMBERS 64 512 64 128 240 CONTEXT LENGTH 2,048 4,096 2,048 2,048 12,288 BATCH SIZE PER GPU 8 8 8 8 1

LEARNING RATE 2E-4 8E-4 2E-4 2E-4 2E-4 LEARNING RATE SHEDULE CONSTANT

WARM-UP STEPS 1,000 600 1,000 1,000 3,000

OPTIMIZER ADAM WITH α1 = 0.9,α2 = 0.95, WEIGHT DECAY=0.1 ROPE’S BASE 100,000

#### H Hop k

In this section, we evaluate the a multi-layered form of induction heads, namely multi-hop. We followed the code of Sanford et al. (2024b) and only changed the attention of vanilla to KV shifting. The experimental results are shown in Figure 9. Obviously, KV shifting has a very good bias for learning multi-hop.

[Figure 45]

[Figure 46]

[Figure 47]

(a) Vanilla (Origin) (b) Vanilla (Reproduce by us) (c) KV shifting attention

- Figure 9: Comparison of Error Rates under Hop k Tasks. The smaller the error, the better the performance.

Taking the pink line (L=5) as an example, the performance of the vanilla model will significantly degrade when the hop number exceeds 8. However, the KV shifting attention still has a small error when the hop count reaches 16. This powerful ability to perform implicit reasoning implies that KV shifting attention may achieve better results in mathematical or reasoning abilities. Therefore, we present the experimental results of mathematical ability in the next section.

#### I Grade-School Math

As a more direct manifestation of induction ability, learning math problems is a natural experiment. At the same time, in order to eliminate the influence of complex syntax, test data leakage, etc. We follow Ye et al. (2024b) and conduct experiments on the iGSM dataset. Due to the code of Ye et al. (2024b) haven’t been released yet, we start from code of https://github.com/kaminocode/iGSM. We use the similar hyper-parameters, except the context length which we set as 1024 for all experiments and the learning rate which we set as 2e-4. 17

In addition, as we are using an open-source code implementation. For simplicity, there is no final summary of the answer. So the evaluation of accuracy is based on the fact that as long as some step calculates the question asked and answers the correct answer, it is considered to be answered correctly, even if additional calculation steps are performed after answering the correct answer.

Table 7: Experiments on iGSM. Tr X - Te Y means train with the numbers of operation no greater than X and test with the numbers of operation as Y.

MODEL TR12-TE15 TR21-TE24 VANILLA 0.8154 0.8711

KV SHIFTING 0.8909 0.9062

The result in Table 7 shown the enormous potential of KV shifting attention. We expect KV shifting attention to enhance the reasoning ability of the model by improving its ability in basic induction heads.

We have included this section in the appendix because our experiment was not as thorough as Ye et al. (2024b). We only tested the accuracy without delving deeper into the analysis, such as the reasons for mistake like Ye et al. (2024b). Besides, conducting more in-depth experiments is beyond the scope of this article.

#### J QKV shifting attention?

If we also shift Q in attention, it doesn’t look like an ablation experiment, after all, it adds an extra part compared to the KV shifting attention. On the other hand, shifting Q is also far from our motivation. So we won’t mention this section in the main text.

But in order to better compare with some baseline methods which use the similar operation, such as Peng et al. (2023) which we discussed in the discussion section, they can shift all the information of Adjacent tokens. So we conduct shifting experiments on all QKV, and the shifting is also per head.

We trained a model for QKV shifting with 19B parameters. Due to some machine malfunctions, the last checkpoint of our QKV shifting attention is saved with 180B training tokens, and we don’t plan to continue training until 200B. The loss curve is shown in Figure 10. From the loss curve, it seems that the two are similar. Then we evaluated the benchmarks as shown in Figure 10. It can be seen that the performance of shifting QKV is not as good as shifting KV, and even not as good as the vanilla. From the perspective of induction heads, the shifting of Q is difficult to contribute to the formation of the induction heads mechanism.

Finally, I will ask myself and answer a question when I recall Zhang et al. (2021). In their Figure 3, they try different position for token shift, such as "after Add", "before Norm", "after Norm", "before Add". From the perspective of our article, placing it in the place of K and V

17In our experiment, if lr=2e-3 is used, Vanilla’s performance will be quite poor (In Tr12-Te15, Vanilla will get approximately 0.5, while KV shifting will get approximately 0.7). For parameter tuning is not the most important thing I have at hand, I will try to find the most suitable hyper-parameters for each model in the future.

[Figure 50]

[Figure 51]

[Figure 52]

(a) Loss (b) MMLU (c) CMMLU

- Figure 10: QKV shifting attention vs KV shifting attention in model with 19B parameters.

is the most suitable for forming induction heads. Shifting elsewhere can not alleviate the width requirement of induction heads.

