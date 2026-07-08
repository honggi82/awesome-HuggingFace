# arXiv:2306.11987v2[cs.LG]22Jun2023

## Training Transformers with 4-bit Integers

Haocheng Xi, Changhao Li, Jianfei Chen, and Jun Zhu Tsinghua University {xihc20,lichangh20}@mails.tsinghua.edu.cn, {jianfeic,dcszj}@tsinghua.edu.cn

### Abstract

Quantizing the activation, weight, and gradient to 4-bit is promising to accelerate neural network training. However, existing 4-bit training methods require custom numerical formats which are not supported by contemporary hardware. In this work, we propose a training method for transformers with all matrix multiplications implemented with the INT4 arithmetic. Training with an ultra-low INT4 precision is challenging. To achieve this, we carefully analyze the specific structures of activation and gradients in transformers to propose dedicated quantizers for them. For forward propagation, we identify the challenge of outliers and propose a Hadamard quantizer to suppress the outliers. For backpropagation, we leverage the structural sparsity of gradients by proposing bit splitting and leverage score sampling techniques to quantize gradients accurately. Our algorithm achieves competitive accuracy on a wide range of tasks including natural language understanding, machine translation, and image classification. Unlike previous 4-bit training methods, our algorithm can be implemented on the current generation of GPUs. Our prototypical linear operator implementation is up to 2.2 times faster than the FP16 counterparts and speeds up the training by up to 35.1%. Our code is available at https://github.com/xijiu9/Train_Transformers_with_INT4.

### 1 Introduction

Training neural networks is computationally demanding. Training with low-precision arithmetic (a.k.a., fully quantized training or FQT) is promising to improve computational and memory efficiency. FQT methods add some quantizers and dequantizers in the original full-precision computational graph, and replace expensive floating-point operations with cheap low-precision ones.

Research in FQT aims to reduce the training numerical precision, without sacrificing much convergence speed or accuracy. The required numerical precision has been reduced from FP16 [32] to FP8 [53, 45], INT32+INT8 [3] and INT8+INT5 [7]. FP8 training is implemented in Nvidia’s H100 GPU with Transformer Engine [34], achieving impressive speedup for the training of large-scale transformers. Recently, the training numerical precision has been pushed down to 4 bits. Sun et al. [46] successfully trained several modern networks with INT4 activation/weights and FP4 gradients; and Chmiel et al. [8] propose a custom 4-bit logarithmic numerical format to further improve the accuracy. However, these 4-bit training methods cannot be directly utilized for acceleration as they require custom numerical formats which are not supported on contemporary hardware.

There are significant optimization challenges to train neural networks at an extremely low 4-bit level. First, the non-differentiable quantizers in forward propagation make the loss landscape rugged, where gradient-based optimizers can easily stuck at local optima [30]. Second, gradients are only computed approximately in low-precision. Such imprecise gradients slow down the training process and even cause the training to be unstable or diverge.

In this work, we propose a novel INT4 training algorithm for a class of popular neural networks, transformers [51]. All the costly linear operations for training transformers can be written in a matrix

Preprint. Under review.

multiplication (MM) form. This MM form allows us to design more flexible quantizers, which better approximate FP32 matrix multiplications by utilizing specific structures of the activations, weights, and gradients in transformers. Our quantizers leverage advances in the field of randomized numerical linear algebra (RandNLA) [14].

For forward propagation, we find that outliers in the activation are the main reason for accuracy degradation. To suppress outliers, we propose a Hadamard quantizer, which quantizes a transformed version of the activation matrix. The transformation is a block diagonal Hadamard matrix, which spreads the information carried in outliers to its nearby entries of the matrix and thus reduces the numerical range of the outliers.

For backpropagation, we exploit the structural sparsity of activation gradients. We find that the gradients of a few tokens are extremely large. Meanwhile, the gradients for the rest majority of the tokens are very small, even smaller than the quantization residuals of larger gradients. Rather than computing these small gradients, it is better to save the computational resource for calculating the residuals of the larger gradients. To utilize such sparsity, we propose bit splitting, which split the gradient of each token into higher 4 bits and lower 4 bits. Then, we choose the most informative gradients by leverage score sampling, which is an importance sampling technique for RandNLA.

Combining quantization techniques for forward and backward propagation, we propose an algorithm that uses INT4 MMs for all linear operations in transformers. We evaluate our algorithm for training transformers on a wide variety of tasks, including natural language understanding, question answering, machine translation, and image classification. Our algorithm achieves competitive or superior accuracy compared with existing works on 4-bit training [46, 8]. Moreover, our algorithm is compatible with contemporary hardware like GPUs, since it does not require custom numerical formats like FP4 or logarithm formats. Our prototypical quantization + INT4 MM operator implementation is up to

- 2.2 times faster than the FP16 MM baseline, and it speeds up the training by up to 35.1%.

### 2 Related Work

Fully Quantized Training Fully quantized training (FQT) [32, 53, 45, 3, 15, 1, 56, 64, 28, 29, 58, 67] methods accelerate training by quantizing the activations, weights, and gradients to low-precision, so linear and nonlinear operators during training can be implemented with low-precision arithmetic. Researches on FQT design novel numerical formats and quantization algorithms which better approximate full-precision tensors. The current research frontier is 4-bit FQT. FQT is challenging due to the vast numerical range of the gradient and the optimization issues of training quantized networks from scratch. Due to these challenges, existing 4-bit FQT algorithms [46, 8] still have ∼1-2.5% accuracy drop on several tasks, and they cannot support contemporary hardware.

Other Efficient Training Methods Mixture-of-experts [42] improves the model capacity without increasing the training budget. Structural dropout [21, 17] exploits computationally efficient ways to regularize the model. Efficient attention [26, 10] reduces the quadratic time complexity for computing attention. Distributed training systems [38, 22] reduce training time by leveraging more computational resources. Our work on reducing numerical precision is orthogonal with these directions.

### 3 Forward Propagation

Neural network training is an iterative optimization procedure with stochastic gradients computed by forward and back propagation. We accelerate forward and back propagation with 4-bit integer (INT4) arithmetic. We first describe the forward propagation of our training procedure. The forward propagation can be formulated as a composition of linear and non-linear (GeLU, normalization, softmax, etc.) operators. In our training procedure, we accelerate all the linear operators with INT4 arithmetic and leave all the less-computationally-intensive non-linear operators in the 16-bit floatingpoint (FP16) format. All linear operations in transformers can be written in a matrix multiplication (MM) form. For ease of presentation, we consider the acceleration of the following simple matrix multiplication throughout this paper:

Z = XW⊤,where Z ∈ RN×C,X ∈ RN×Dand W ∈ RC×D. (1)

The most predominant use case of such MM is the fully-connected layer. Consider a transformer with an input shape of (batch size S, sequence length T, dimensionality D). The fully-connected layer can be written as Eq. (1) where X is the activation for N = ST tokens, and W is the weight matrix. For attention layers, batch matrix multiplications (BMMs) might be required. Our proposed techniques can be applied to BMMs, and we leave the discussion of BMMs in Appendix. A.1.

#### 3.1 Learned Step Size Quantization

To accelerate training, the forward propagation must be computed with integer arithmetic. We leverage the learned step size quantizer (LSQ) [16] for this purpose. LSQ is a static quantization method whose quantization scale does not depend on the input, and is thus cheaper than dynamic quantization methods [23], which need to compute the quantization scale dynamically per iteration.

Given a FP matrix X, LSQ quantizes X to integer with

ints

(X) := ⌊clamp(X/sX,−QN,QP)⌉, (2)

X

where sX is a learnable scalar parameter, clamp restricts its input to the range [−QN,QP], ⌊·⌉ is a rounding operation, and X/sX is computed elementwise. The resultant matrix takes values from {−QN,−QN + 1,...,QP}. Since we aim to perform INT4 MMs, we set QN = QP = 7. The integer matrix can be dequantized back to FP through float(ints

(X)) = sXints

(X) ≈ X. With LSQ, Eq. (1) can be computed approximately as Y = XW⊤ ≈ sXsWints

X

X

(W)⊤ , where the INT4 MM ints

(X)ints

X

W

(W)⊤ can be implemented efficiently on hardware.

(X)ints

X

W

Remark: Quantization-aware training (QAT) [9, 62, 66, 23, 12, 11, 43, 59, 44, 48, 63, 2, 18, 54] is an inference acceleration technique which trains networks with quantizers inserted in the forward propagation graph, so the trained network can perform efficiently during inference. QAT can compress activation/weights to extremely low precision (e.g. 1-2 bits). It is tempting to think that directly applying a quantizer for QAT to FQT can lead to similar low activation/weights bit-width. However, even only quantizing the forward propagation for FQT is much more challenging than QAT because: (1) QAT requires a converged full-precision model as initialization [16] and/or as a teacher model for knowledge distillation [2]; (2) QAT can adopt expensive multi-stage training pipelines without worrying about the convergence speed [31], while FQT algorithm must converge as fast as fullprecision training algorithms to be useful; (3) QAT may approximate the discrete quantizer with continuous functions during training [19], which cannot be implemented with integer arithmetic. Due to these challenges, it is still an open problem to do FQT with 4-bit activations/weights.

#### 3.2 Activation Outliers

Simply applying LSQ for FQT with 4-bit activation/weights leads to accuracy degradation due to activation outliers [57]. As shown in Fig. 1(a), activations have some outlier entries, which are much larger in magnitude than other entries. In this case, the step size sX poses a trade-off between quantization granularity and representable numerical range. If sX is large, we can represent the outliers well at the expense of representing most other entries in a very coarse manner. On the other hand, if sX is small, we have to truncate the entries outside the range [−QNsX,QPsX]. Unfortunately, the transformers tend to store information in these outliers, and such truncation would seriously harm accuracy (see Sec. 5.2 for details). The outlier problem is particularly significant when the training task is to fine-tune a pre-trained model on some new downstream tasks, since the pre-train model contains more outliers [57] than random initialization.

There exists some works to handle activation outliers for post-training quantization (PTQ). Outlier Suppression [55] discover that LayerNorms amplify outliers, and propose Gamma Migration and Token-Wise Clipping to solve this issue and achieves 6-bit BERT PTQ without too much degradation. SmoothQuant [57] migrates the quantization difficulty of activation outliers to weights and achieves 8-bit PTQ for large language models, such as OPT-175B. Outlier Channel Splitting [65] duplicates channels containing outliers with small overhead on the size of the network. However, these methods mainly focus on PTQ or QAT, and seldom successfully deal with ultra-low 4-bit training.

| | |
|---|---|
| | |
| | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

Figure 1: Histogram of activation of the linear-1-2 layer in a BERT-base-uncased model. (a) Original activation distribution; (b) Hadamard-transformed activation distribution.

Figure 2: (a) The distribution of gradient norm along the token dimension. (b) The cumulative sum of the top X values as a percentage of the sum of all norms along the token dimension.

#### 3.3 Hadamard Quantization

We propose a Hadamard quantizer (HQ) to solve the outlier problem. Its main idea is to quantize the matrices in another linear space which has fewer outliers.

The outliers in activation matrices form a feature-wise structure [57]. They are typically concentrated on a few dimensions, i.e., only a few columns of X are significantly larger than others. Hadamard transform [47] is a linear transformation, which can amortize the outliers into other entries. Specifically, the Hadamard transform Hk is a 2k × 2k matrix, where

H0 = [1], Hk = √12 [Hk−1 Hk−1;Hk−1 − Hk−1].

Hadamard matrices are orthogonal and symmetric: Hk = H⊤k = H−k 1, so HkHk = I,∀k ≥ 0. Consider any coordinate row vector1 e⊤i ∈ R2

k

, we have e⊤i Hk = 2−k/212k,∀i, where 12k = (1,1,...,1) is a 2k-dimensional all-one-vector. This demonstrates the extreme case when a single outlier dominates all the rest dimensions. In this case, Hadamard transformation effectively turns the vector into a quantization-friendly all-one-vector. The practical effect of the Hadamard transform on suppressing activation outliers is demonstrated in Fig. 1(b).

HQ uses a block-diagonal transformation matrix H ∈ RD×D: H = BlockDiag(Hk,...,Hk), where D is a multiple of 2k. To suppress outliers, we quantize a transformed version of X and W:

(WH)H⊤. Combining the quantized matrices, we get

X = (XH)H⊤ ≈ sXints

#### (XH)H⊤, W = (WH)H⊤ ≈ sWints

X

W

Y = XW⊤ ≈ sXsWintsX (XH) H⊤HintsW H⊤W⊤ = sXsWintsX (XH) intsW H⊤W⊤ , (3)

where the inverse transformations cancel with each other, and the MM can be implemented as:

|Procedure HQ-MM<br><br>1. Compute XH and H⊤W⊤ in FP16.<br>2. Quantize the resultant matrices to INT4 by LSQ.<br>3. Multiply the two INT4 matrices.<br>4. Dequantize the resultant INT32 matrix to FP16 by multiplying sXsW.<br>|
|---|

For time complexity, Step 1 takes O(2kN(D + C)) FP16 multiply-accumulates (MACs); Step 2 and Step 4 takes O(N(D + C)) FP16 MACs in total; and Step 3 takes O(NDC) INT4 MACs. Comparing with the plain LSQ Eq. (2), the amount of FP16 MACs increases by 2k times, from O(N(D + C)) to O(2kN(D + C)). However, our HQ-MM is still much cheaper than an FP16 MM given 2k ≪ D and 2k ≪ C. The number k shows a tradeoff between the ability to suppress outliers and computation complexity. Larger k allows for amortizing the outlier within a larger horizon, at the cost of being more expensive. We propose an adaptive algorithm to choose k for each activation depending on the outlier scale, as discussed in Appendix A.5. The typical value is k = 5, while the dimensionality C and D ranges from 768 to 4096.

1A vector which i-th dimension is 1, and all other dimensions are 0.

### 4 Backpropagation

We now consider accelerating the backpropagation of the linear layer with INT4 operations. The linear operator HQ-MM defined in Eq. (3) has four inputs: activation X, weight W, and step sizes sX, sW. Given the output gradient ∇YL w.r.t. some loss function L, we need to compute the gradient of all four inputs. We discuss the computation of activation/weight gradients in this section, and left the discussion of step size gradients to Appendix A.3. For simplicity, we omit L and simply use ∇Y to denote the gradient in the following text.

By the straight-through estimator ⌊x⌉′ = 1 [5] and the chain rule, we have

∇W = sX ∇⊤YXˆ ◦ IW H⊤, ∇X = sWIX ◦ ∇YWHˆ ⊤, (4)

where we define Xˆ = ints

#### (XH), Wˆ = ints

(WH), IX = I(−QN ≤ X/sX ≤ QP), and IW = I(−QN ≤ W/sW ≤ QP). For computing the gradients, three types of matrix multiplications are required:

X

W

- 1. The element-wise multiplication ◦ of a 0/1 matrix IX (or IW) with another INT4 (or INT32) matrix. This operation has low time complexity.
- 2. The multiplication of an INT32 matrix with an FP16 block-wise Hadamard matrix sWH⊤, which also has low-time complexity, as discussed in Sec. 3.3.
- 3. The multiplication of the FP16 gradient ∇Y with an INT4 matrix Xˆ or Wˆ , which we will accelerate by quantizing ∇Y to INT4.

In the rest of this section, we will discuss quantization methods to compute the “type 3” MMs ∇⊤YXˆ and ∇YWˆ . We quantize ∇Y dynamically for each MM, while Xˆ and Wˆ have been already calculated in forward propagation in Section. 3. We start by discussing the structure of the gradient.

- 4.1 Structural Sparsity of Gradients

We note that the gradient matrix ∇Y tends to be very sparse along the training process. Furthermore, the sparsity has a structure: few rows (i.e., tokens) of ∇Y have large entries, while most other rows are close to an all-zero vector. We illustrate this by plotting the histogram of per-row norm ∥(∇Y)i,:∥ for all the rows i in Fig. 2.

Such a structural sparsity arises from the heavy overparameterization [61] of modern neural networks. During almost the entire training process, the network operates in the overparameterized scheme [33], where it can fit most training data well, except for a few hard examples. Therefore, the (activation) gradient will be close to zero for well-fitted data points. We find that for pretraining tasks, such structural sparsity quickly emerges after only a few training epochs. For fine-tuning tasks, the gradient is always sparse during the whole training process.

#### 4.2 Bit Splitting and Leverage Score Sampling

Here, we discuss how to design gradient quantizers to accurately compute the MMs during backpropagation by leveraging structural sparsity. The high-level idea is that many rows of the gradient are so small that they have little impact on the parameter gradient, yet they waste abundant computation. On the other hand, the large rows cannot be accurately represented with INT4. We drop some small rows and use the saved computation to represent large rows more accurately.

First, we propose bit splitting (BS), which splits a full-precision matrix as higher and lower 4 bits:

∇Y ≈ s↑∇↑Y + s↓∇↓Y, (5)

where s↑,s↓ are two floating-point scalars, and ∇↑Y, ∇↓Y are INT4 matrices representing the higher and lower 4 bits, respectively. BS can be implemented by first quantizing ∇Y to INT4 as ∇Y ≈

s↑∇↑Y and then quantize the residual to INT4 as ∇Y − s↑∇↑Y ≈ s↓∇↓Y. BS can be viewed as an INT8 representation of a matrix, where ∇↑Y and ∇↓Y are the higher and lower 4 bits of the INT8 representation. Next, we discuss how to compute the weight and activation gradient.

Weight Gradient As discussed earlier, weight gradient involves the matrix multiplication ∇⊤YXˆ , where ∇Y ∈ RN×C and Xˆ is an N × D INT4 matrix. By Eq. (5):

⊤

⊤ + s↓∇↓Y

⊤)Xˆ = ∇↕Y

∇⊤YXˆ ≈ (s↑∇↑Y

X↕, (6)

where we define ∇↕Y = [s↑∇↑Y;s↓∇↓Y]⊤ ∈ R2N×C and Xˆ ↕ = [Xˆ ;Xˆ ] to be a 2N × D INT4 matrix. Eq. (6) represents the product of an INT8 ∇⊤Y and an INT4 Wˆ , and can be implemented by two INT4 MMs ∇↑Y

⊤Xˆ and ∇↓Y

⊤Xˆ . Such MM is rather accurate since ∇Y is represented with 8 bits.

However, comparing to a naïve quantization of ∇Y to INT4, BS doubles the amount of INT4 operations for MM. We propose leverage score sampling (LSS) to cut the operations of Eq. (5) by

half, to the same amount as the naïve MM s↑∇↑Y

Xˆ . Noticing that the MM Eq. (6) can be written as the sum of 2N rank-1 matrices:

2N

2N

⊤

⊤ :,iX↕i =

∇↕Y

∇↕Y

X↕ =

, (7)

∇Wi

i=1

i=1

⊤ :,iX↕i . Due to the sparsity of ∇Y, the matrices ∇Wi

= ∇↕Y

where ∇Wi

differ in magnitude and small matrices can be discarded without having a big influence on the result. Our proposed LSS assigns each ∇Wi

a probability pi ∈ [0,1],i = 1,··· ,2N, that satisfies

2N i=1 pi = N. We define random masks mi ∼ Bern(pi) and mask matrix M˜ , and approximate it as

2N

mi pi ∇↕Y

⊤

⊤MX˜ ↕ =

⊤ :,iX↕i ,where M˜ = diag m1

∇↕Y

X↕ ≈ ∇↕Y

p1 ,..., m

p2N ,

2N

i=1

⊤

##### ⊤MX˜ ↕ = ∇↕Y

⊤

E M ˜ X↕ = ∇↕Y

which is an unbiased approximation since E ∇↕Y

#### X↕.

In expectation, there are only N nonzero mis. Therefore, LSS reduces the cost of MM by half. For LSS to be accurate, we minimize its variance. We have:

Proposition 4.1. (LSS variance for weight gradient)

2N

2N

1 − pi pi ∥∇↕Yi,:∥2∥X↕i,:∥2,where Var[X] := E[∥X − EX∥]2F .

mi pi ∇↕Y

⊤ :,iX↕i =

Var

i=1

i=1

The coefficient ci := ∥∇↕Yi,:∥∥X↕i,:∥ is called the leverage score, which can be easily computed in low time complexity. When pi ∝ ci, the variance attends its minimum due to Cauchy inequality:

2N

2N

2N

2N

2N

c2i pi

c2i pi

1 pi ∥∇↕Yi,:∥2∥X↕i,:∥2 =

ci)2,

pi ≥ (

=

i=1

i=1

i=1

i=1

i=1

where the equality holds when pi ∝ ci. Intuitively, LSS can approximate the MM Eq. (7) well with significantly lower computational cost when the leverage scores {ci} are diverse, which is indeed the case as shown in Fig. 2.

Define M↑ to be the top-left N × N submatrix of M˜ and M↓ to be the bottom-right one, we have

⊤MX˜ ↕ = s↑∇↑Y

⊤M˜ ↑Xˆ + s↓∇↓Y

⊤M˜ ↓Xˆ ,

∇↕Y

which can be implemented by two INT4 MMs with sampled rows/columns. Putting everything together, we propose the following MM procedure to compute the weight gradient:

|Procedure LSS-MM<br><br>1. Quantize ∇Y with BS to obtain ∇↑Y and ∇↓Y in INT4.<br>2. Compute the leverage score ∥∇↕Yi,:∥∥X↕i,:∥ in FP16.<br>3. Sample the masks {mi}.<br>4. Sample rows of ∇Y and Xˆ given the masks {mi}.<br>5. Compute INT4 MMs ∇↑Y<br><br>⊤M˜ ↑Xˆ and ∇↓Y<br><br>⊤M˜ ↓Xˆ ,<br><br>6. Dequantize and sum up the resultant INT32 matrices to obtain the FP16 result ∇↕Y<br><br><br>⊤MX˜ ↕.|
|---|

Table 1: Results on language model fine-tuning, transformer pretraining, and vision transformers fine-tuning and pretraining. Standard deviation is reported as subscript. FT refers to Fine-tuning, and PT refers to Pre-training. For WMT the result of 25.4 is result of Ultra-Low, not INT8.

BASESLINES 4-BIT TRAINING METHODS DATASET TRAIN TYPE MODEL METRIC NAME FP INT8 LSQ+LUQ HQ+LSS

BERT-BASE AVG 82.670.24 81.450.13 75.290.52 80.810.31

GLUE-DEV FT

BERT-LARGE AVG 84.570.42 82.740.24 55.932.47 82.250.58 SQUAD V1 FT BERT-BASE F1 88.320.30 88.420.20 85.750.31 87.600.25 SQUAD V2 FT BERT-BASE F1 76.040.68 75.630.07 71.020.41 74.630.18

ADVERSARIAL QA FT BERT-BASE F1 40.990.38 40.170.58 31.850.30 38.700.77

SWAG FT BERT-BASE ACC 79.840.10 79.180.19 70.791.20 77.490.16 CONLL FT BERT-BASE ACC 93.380.08 93.130.14 87.630.39 91.900.48

WMT PT TRANSFORMER-BASE BLEU 27.5 25.4(ULTRA LOW) 27.17 -

SACREBLEU 26.5 - - 25.57 CIFAR10 FT

VIT-B/32

TOP1 ACC 98.770.03 98.590.02 97.760.10 98.360.05 VIT-L/32 98.98 98.76 98.38 98.47

VIT-B/32

TOP1 ACC 91.940.11 90.990.07 88.630.085 89.780.06 VIT-L/32 93.07 92.2 90.97 91.13

CIFAR100 FT

81.88 80.42 77.25 79.18 VIT-L/32 81.62 81.3 77.41 80.06 VIT-L/16 84.55 83.05 82.4 82.61

VIT-B/32

IMAGENET1K FT

TOP1 ACC

PT DEIT-SMALL TOP1 ACC 73.1 70.95 69.96 69.18

As M˜ only has N non-zero elements in expectation, the two matrix multiplications in Step 5 take about 2NCD INT4 MACs, which aligns with the cost of the naïve MM s↑∇↑Y

Xˆ . The overhead of

all the other steps is O(NC + ND) in total. Activation Gradient Similar to the previous discussion, the gradient of input can be written as

Wˆ = ˆI↕∇↕Y

∇YWˆ ≈ (s↑∇↑Y + s↓∇↓Y)Wˆ = s↑∇↑Y

Wˆ + s↓∇↓Y

#### W ˆ , (8)

where we define ∇↕Y = [s↑∇↑Y;s↓∇↓Y] ∈ R2N×C and ˆI↕ = [I I] to be a N × 2N INT4 matrix, I is a N × N identity matrix. The original product can also be implemented by two INT4 MMs ∇↑Y

#### Wˆ and ∇↓Y

Wˆ . But different from weight gradients, we now focus on ˆI↕∇↕Y in Eq. (8) and do leverage score sampling on this MM. A detailed discussion can be found in Appendix B.2, and we only present the leverage score here. Similarly, we write the MM as the sum of 2N smaller multiplications:

2N

2N

mi pi

ˆI↕∇↕Y =

ˆI↕:,i∇↕Yi ≈

∇Yi

,

i=1

i=1

= ˆI↕:,i∇↕Yi and associate the probability pi and Bernoulli mask mi ∼ Bern(pi)

where we define ∇Yi

with the i multiplication. The leverage score for activation gradient is ci := ∥∇↕Yi∥, and the variance attains minimum when pi ∝ ci. More details about the algorithm can be found at Appendix. A.3 On the implementation side, once the mask {mi} is known, we can decompose the MM Eq. (8) as two

INT4 MMs: ˆI↕M˜ ∇↕Y

W ˆ = s↑M˜ ↑∇↑Y

Wˆ + s↓M˜ ↓∇↓Y

Wˆ .

### 5 Experiments

We evaluate our INT4 training algorithm on a wide variety of tasks including language model fine-tuning, machine translation, and image classification. We implement our proposed HQ-MM and LSS-MM algorithms with CUDA and cutlass2, and the implementation details can be found in Appendix A. We replace all the floating-point linear operators with our INT4 implementation except simply using LSQ for embedding layers, and leaving the last classifier layer in full precision. We adopt default architectures, optimizers, schedulers, and hyper-parameters for all the evaluated models.

#### 5.1 Converged Model Accuracy

We compare the accuracy of the converged model on various tasks in Table 1. The compared methods include full-precision training (FP), INT8 training [3](INT8), FP4 training [46] (“Ultra-low”), 4-bit

2https://github.com/NVIDIA/cutlass

Figure 4: Comparison of basic FP16 MM, HQ, and LSS operators.

Figure 3: CoLA performance under different methods using different bits.

Figure 5: SpeedUp of our INT4 training algorithm compared with FP16 PyTorch AMP on (a) Bert-Large (b) Gpt2-base.

- (a) Comparison of forward methods.
- (b) Comparison of backward methods.

logarithm quantization [8] with LSQ for activations and weights (LSQ+LUQ), and our algorithm which utilizes HQ for forward and LSS for backpropagation (HQ+LSS). Ultra-low does not have a public implementation, so we only report its performance from its original paper on the machine translation task. Except for the large machine translation task and the task of large vision transformers, we repeat each run by three times and report the standard deviation as subscripts in tables. We do not include any kind of knowledge distillation or data augmentation.

Language model fine-tuning: We use the pretrained BERT-base-uncased and BERT-large-uncased [24] model, and evaluate the performance of our method on GLUE dev-set [52], SQUAD [40], SQUADv2 [39], Adversarial QA [4], CoNLL-2003 [41] and SWAG [60] datasets. We present the average result of bert-base-uncased and bert-large-uncased model on the GLUE dataset. The full results are listed in Appendix C.2. Compared with LSQ+LUQ, our method achieves 5.5% improvement of accuracy on average for the bert-base model and achieves > 25% improvement of accuracy on average for the bert-large model. We further show the result on the SQUAD, SQUAD 2.0, Adversarial QA, CoNLL-2003, and SWAG datasets. On all of the tasks, compared with LSQ+LUQ, our method achieves better performance. We improve by 1.8% and 3.6% on SQUAD and SQUAD 2.0 compared to LSQ+LUQ, respectively. On the more difficult Adversarial QA, we improve by 6.8% on F1 score. On SWAG we improve by 6.7% and on CoNLL-2003 we improve by 4.2% accuracy.

Machine translation: We also apply our method for pretraining. We train a Transformer-base [51] model on WMT 14 En-De dataset [6] for machine translation. Note that we reproduce this experiment with Fairseq’s recipe 3, which reports the SacreBleu score (26.5 for FP) [36], while Ultra-low and LUQ report the more optimistic original BLEU score (27.5 for FP) [35]. Our HQ+LSS has about 1.0% BLEU degradation, which is smaller than 2.1% of Ultra-low and higher than 0.3% reported in the LUQ paper. Nevertheless, HQ+LSS still performs comparably with existing methods for this pretraining task, and it supports contemporary hardware.

Image Classification: We load ViT checkpoints pretrained on ImageNet21k [13], and fine-tune it on CIFAR-10, CIFAR-100 [27], and ImageNet1k. We use ViT-B/32 and ViT-L/32 for CIFAR datasets and use ViT-B/32, ViT-L/32 and ViT-L/16 for ImageNet1k. On CIFAR10 we achieve < 0.5% accuracy degradation, while LSQ+LUQ has 1% degradation for ViT-B/32 and 0.6% degradation for ViT-L/32. On CIFAR100, INT8 already has ∼ 1% accuracy degradation, which shows its difficulty. We improve by 1.1% accuracy for ViT-B/32 and 0.2% accuracy for ViT-L/32 compared with LSQ+LUQ. On ImageNet1k, we improve by 2% accuracy for ViT-B/32, 2.6% accuracy for ViT-L/32 and 0.2% for ViT-L/32 compared with LSQ+LUQ. We further test the effectiveness of our algorithm for pretraining a DeiT-Small model [50] on ImageNet1K, where HQ+LSS can still converge to similar accuracy level compared to LSQ+LUQ, while being more hardware friendly.

#### 5.2 Ablation Study

Here, we conduct ablation studies to show the effectiveness of our forward and backward methods independently on the challenging CoLA dataset. To study the effectiveness of different quantizers for forward propagation, we leave backpropagation in FP16. The result is shown in Fig. 3(a). We first validate the claim in Sec. 3.2 that outliers are the main cause of accuracy degradation in quantized forward propagation. We test an “outlier” method which maintains 1% largest activation entries in

3https://github.com/facebookresearch/fairseq

FP. The “outlier” method achieves good performance, which proves that outliers are indeed the most significant challenge of the transformer’s forward quantization. The hardware-unfriendly “outlier” method serves as an upper bound of methods to handle outliers. Our HQ outperforms LSQ by better handling the outliers and achieves comparable results to maintaining the outliers.

We also investigated whether more granular quantizers, such as per-token quantization or per-channel quantization could be used to quantify outliers, or whether existing methods like SmoothQuant [57] could be used for INT4 FQT. The results are listed in Appendix C.3, and we find that without HQ, none of these methods achieve good accuracy under 4-bit quantization, and the result of HQ is not strongly affected when more granular quantization methods are applied.

For backpropagation, we compare a simple minimax quantizer [3], LUQ [8] and our LSS, and leave forward propagation in FP16. The minimax quantizer divides the numerical range from the minimum to the maximum into equally large quantization bins. The result is shown in Fig. 3(b). While the bit-width is higher than 2, our LSS achieves results that are comparable and even slightly higher than LUQ. Meanwhile, LSS is more hardware friendly as it requires only INT4 arithmetic.

#### 5.3 Computational and Memory Efficiency

Finally, we demonstrate the potential of our method to accelerate neural network training by evaluating our prototypical implementation discussed in Appendix A.6. We emphasize that our implementation is not fully optimized. For example, the backward computation requires an INT4 MM in the form of Y = AB, while cutlass only supports Y = AB⊤, so explicit transpose is required. We also do not fuse the linear operators with nonlinearities and normalizations. Therefore, the results cannot fully reflect the potential of INT4 training algorithms. A fully optimized implementation requires heavy engineering, which exceeds the scope of our paper.

Operator Speed: We compare the throughput of our proposed HQ-MM (HQ), LSS for computing weight gradient (LSSWeight), LSS for computing activation gradient (LSSAct), and their average throughput (INT4) with a baseline tensor-core FP16 GEMM implementation (FP16) provided by cutlass in Fig. 4 on an Nvidia RTX 3090 GPU which has a peak throughput at 142 FP16 TFLOPs and 568 INT4 TFLOPs. As the matrix size grows, the overhead of quantization diminishes and our INT4 operators can be up to 2.2 times faster compared with FP16 MM. We further analyze the quantization overhead for each operator in Appendix C.5.

Training Throughput: We compare the training throughput of the FP16 PyTorch AMP and our INT4 training algorithm for training BERT [24] and GPT [37]-style language models on a system of 8 Nvidia A100 GPUs. We vary the hidden layer size, intermediate fully-connected layer size, and batch size, and plot the speedup of INT4 training in Fig. 5. Our INT4 training algorithm can achieve up to 35.1% speedup for BERT-style models and up to 26.5% speedup for GPT-style models. The training time can be found in Appendix C.4.

### 6 Conclusions

We propose a hardware-friendly INT4 training method for transformers. By analyzing the properties of MMs in transformers, we propose HQ and LSS methods to quantize activations and gradients while preserving accuracy. On several important tasks, our method performs comparably or better than existing INT4 methods. Our work can be potentially extended beyond transformers to other MM-only architectures, such as MLP-Mixer [49], graph neural networks [25], and recurrent neural networks [20]. We leave it as a future direction.

Broader Impacts: Our algorithm can improve efficiency and reduce the energy consumption of training neural networks, which helps reduce the carbon footprint caused by deep learning. However, our efficient training algorithm might also facilitate the development of large language models with safety concerns for human beings; and malicious AI applications such as fake content generation.

Limitations: The main limitation of this work is that it can only accelerate models with a large portion of matrix multiplications (linear layers), but can not accelerate convolution layers. Moreover, the proposed method cannot yet work well for those extremely large models such as OPT-175B. To the best of our knowledge, even INT8 training is still an open problem for these large models.

### References

- [1] Menachem Adelman and Mark Silberstein. Faster neural network training with approximate tensor operations. arXiv preprint arXiv:1805.08079, 2018.
- [2] Haoli Bai, Wei Zhang, Lu Hou, Lifeng Shang, Jing Jin, Xin Jiang, Qun Liu, Michael Lyu, and Irwin King. Binarybert: Pushing the limit of bert quantization. arXiv preprint arXiv:2012.15701, 2020.
- [3] Ron Banner, Itay Hubara, Elad Hoffer, and Daniel Soudry. Scalable methods for 8-bit training of neural networks. In Advances in Neural Information Processing Systems, pages 5145–5153, 2018.
- [4] Max Bartolo, Alastair Roberts, Johannes Welbl, Sebastian Riedel, and Pontus Stenetorp. Beat the ai: Investigating adversarial human annotation for reading comprehension. Transactions of the Association for Computational Linguistics, 8:662–678, 2020.
- [5] Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.
- [6] Ondˇrej Bojar, Christian Buck, Christian Federmann, Barry Haddow, Philipp Koehn, Johannes Leveling, Christof Monz, Pavel Pecina, Matt Post, Herve Saint-Amand, et al. Findings of the 2014 workshop on statistical machine translation. In Proceedings of the ninth workshop on statistical machine translation, pages 12–58, 2014.
- [7] Jianfei Chen, Yu Gai, Zhewei Yao, Michael W Mahoney, and Joseph E Gonzalez. A statistical framework for low-bitwidth training of deep neural networks. In Advances in neural information processing systems, 2020.
- [8] Brian Chmiel, Ron Banner, Elad Hoffer, Hilla Ben Yaacov, and Daniel Soudry. Logarithmic unbiased quantization: Practical 4-bit training in deep learning. arXiv preprint arXiv:2112.10769, 2021.
- [9] Jungwook Choi, Zhuo Wang, Swagath Venkataramani, Pierce I-Jen Chuang, Vijayalakshmi Srinivasan, and Kailash Gopalakrishnan. Pact: Parameterized clipping activation for quantized neural networks. arXiv preprint arXiv:1805.06085, 2018.
- [10] Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. In International Conference on Learning Representations, 2020.
- [11] Zhen Dong, Zhewei Yao, Yaohui Cai, Daiyaan Arfeen, Amir Gholami, Michael W Mahoney, and Kurt Keutzer. Hawq-v2: Hessian aware trace-weighted quantization of neural networks. arXiv preprint arXiv:1911.03852, 2019.
- [12] Zhen Dong, Zhewei Yao, Amir Gholami, Michael Mahoney, and Kurt Keutzer. Hawq: Hessian aware quantization of neural networks with mixed-precision. ICCV, 2019.
- [13] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.
- [14] Petros Drineas and Michael W Mahoney. Randnla: randomized numerical linear algebra. Communications of the ACM, 59(6):80–90, 2016.
- [15] Mario Drumond, LIN Tao, Martin Jaggi, and Babak Falsafi. Training dnns with hybrid block floating point. In Advances in Neural Information Processing Systems, pages 453–463, 2018.
- [16] Steven K Esser, Jeffrey L McKinstry, Deepika Bablani, Rathinakumar Appuswamy, and Dharmendra S Modha. Learned step size quantization. In International Conference on Learning Representations, 2019.

- [17] Angela Fan, Edouard Grave, and Armand Joulin. Reducing transformer depth on demand with structured dropout. In International Conference on Learning Representations, 2019.
- [18] Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Sharpness-aware minimization for efficiently improving generalization. arXiv preprint arXiv:2010.01412, 2020.
- [19] Ruihao Gong, Xianglong Liu, Shenghu Jiang, Tianxiang Li, Peng Hu, Jiazhen Lin, Fengwei Yu, and Junjie Yan. Differentiable soft quantization: Bridging full-precision and low-bit neural networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4852–4861, 2019.
- [20] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.
- [21] Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In European conference on computer vision, pages 646–661. Springer, 2016.
- [22] Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V Le, Yonghui Wu, et al. Gpipe: Efficient training of giant neural networks using pipeline parallelism. Advances in neural information processing systems, 32, 2019.
- [23] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for efficient integer-arithmetic-only inference. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2704–2713, 2018.
- [24] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of NAACL-HLT, pages 4171–4186, 2019.
- [25] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.
- [26] Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In International Conference on Learning Representations, 2019.
- [27] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. Technical report, 2009.
- [28] Hamed F Langroudi, Zachariah Carmichael, and Dhireesha Kudithipudi. Deep learning training on the edge with low-precision posits. arXiv preprint arXiv:1907.13216, 2019.
- [29] Hamed F Langroudi, Zachariah Carmichael, David Pastuch, and Dhireesha Kudithipudi. Cheetah: Mixed low-precision hardware & software co-design framework for dnns on the edge. arXiv preprint arXiv:1908.02386, 2019.
- [30] Zechun Liu, Zhiqiang Shen, Shichao Li, Koen Helwegen, Dong Huang, and Kwang-Ting Cheng. How do adam and training strategies help bnns optimization. In International Conference on Machine Learning, pages 6936–6946. PMLR, 2021.
- [31] Zechun Liu, Zhiqiang Shen, Marios Savvides, and Kwang-Ting Cheng. Reactnet: Towards precise binary neural network with generalized activation functions. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIV 16, pages 143–159. Springer, 2020.
- [32] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, et al. Mixed precision training. In International Conference on Learning Representations, 2018.
- [33] Preetum Nakkiran, Gal Kaplun, Yamini Bansal, Tristan Yang, Boaz Barak, and Ilya Sutskever. Deep double descent: Where bigger models and more data hurt. Journal of Statistical Mechanics: Theory and Experiment, 2021(12):124003, 2021.

- [34] Nvidia. Transformer Engine. https://github.com/NVIDIA/TransformerEngine, 2023. Online; accessed 23 January 2023.
- [35] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318, 2002.
- [36] Matt Post. A call for clarity in reporting bleu scores. arXiv preprint arXiv:1804.08771, 2018.
- [37] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI Blog, 1(8):9, 2019.
- [38] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020.
- [39] Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don’t know: Unanswerable questions for squad. arXiv preprint arXiv:1806.03822, 2018.
- [40] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. arXiv preprint arXiv:1606.05250, 2016.
- [41] Erik F Sang and Fien De Meulder. Introduction to the conll-2003 shared task: Languageindependent named entity recognition. arXiv preprint cs/0306050, 2003.
- [42] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.
- [43] Sheng Shen, Zhen Dong, Jiayu Ye, Linjian Ma, Zhewei Yao, Amir Gholami, Michael W Mahoney, and Kurt Keutzer. Q-bert: Hessian based ultra low precision quantization of bert. arXiv preprint arXiv:1909.05840, 2019.
- [44] Sheng Shen, Zhen Dong, Jiayu Ye, Linjian Ma, Zhewei Yao, Amir Gholami, Michael W Mahoney, and Kurt Keutzer. Q-bert: Hessian based ultra low precision quantization of bert. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8815–8821, 2020.
- [45] Xiao Sun, Jungwook Choi, Chia-Yu Chen, Naigang Wang, Swagath Venkataramani, Vijayalakshmi Viji Srinivasan, Xiaodong Cui, Wei Zhang, and Kailash Gopalakrishnan. Hybrid 8-bit floating point (hfp8) training and inference for deep neural networks. In Advances in Neural Information Processing Systems, pages 4901–4910, 2019.
- [46] Xiao Sun, Naigang Wang, Chia-Yu Chen, Jiamin Ni, Ankur Agrawal, Xiaodong Cui, Swagath Venkataramani, Kaoutar El Maghraoui, Vijayalakshmi Viji Srinivasan, and Kailash Gopalakrishnan. Ultra-low precision 4-bit training of deep neural networks. In Advances in Neural Information Processing Systems, volume 33, 2020.
- [47] James Joseph Sylvester. Lx. thoughts on inverse orthogonal matrices, simultaneous signsuccessions, and tessellated pavements in two or more colours, with applications to newton’s rule, ornamental tile-work, and the theory of numbers. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, 34(232):461–475, 1867.
- [48] Hanlin Tang, Xipeng Zhang, Kai Liu, Jianchen Zhu, and Zhanhui Kang. Mkq-bert: Quantized bert with 4-bits weights and activations. arXiv preprint arXiv:2203.13483, 2022.
- [49] Ilya O Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, et al. Mlpmixer: An all-mlp architecture for vision. Advances in neural information processing systems, 34:24261–24272, 2021.
- [50] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In International conference on machine learning, pages 10347–10357. PMLR, 2021.

- [51] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [52] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018.
- [53] Naigang Wang, Jungwook Choi, Daniel Brand, Chia-Yu Chen, and Kailash Gopalakrishnan. Training deep neural networks with 8-bit floating point numbers. In Advances in Neural Information Processing Systems, pages 7675–7684, 2018.
- [54] Zheng Wang, Juncheng B Li, Shuhui Qu, Florian Metze, and Emma Strubell. Squat: Sharpnessand quantization-aware training for bert. arXiv preprint arXiv:2210.07171, 2022.
- [55] Xiuying Wei, Yunchen Zhang, Xiangguo Zhang, Ruihao Gong, Shanghang Zhang, Qi Zhang, Fengwei Yu, and Xianglong Liu. Outlier suppression: Pushing the limit of low-bit transformer language models. arXiv preprint arXiv:2209.13325, 2022.
- [56] Shuang Wu, Guoqi Li, Feng Chen, and Luping Shi. Training and inference with integers in deep neural networks. In International Conference on Learning Representations, 2018.
- [57] Guangxuan Xiao, Ji Lin, Mickael Seznec, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models. arXiv preprint arXiv:2211.10438, 2022.
- [58] Yukuan Yang, Lei Deng, Shuang Wu, Tianyi Yan, Yuan Xie, and Guoqi Li. Training highperformance and large-scale deep neural networks with full 8-bit integers. Neural Networks, 125:70–82, 2020.
- [59] Ofir Zafrir, Guy Boudoukh, Peter Izsak, and Moshe Wasserblat. Q8bert: Quantized 8bit bert. In 2019 Fifth Workshop on Energy Efficient Machine Learning and Cognitive Computing-NeurIPS Edition (EMC2-NIPS), pages 36–39. IEEE, 2019.
- [60] Rowan Zellers, Yonatan Bisk, Roy Schwartz, and Yejin Choi. Swag: A large-scale adversarial dataset for grounded commonsense inference. arXiv preprint arXiv:1808.05326, 2018.
- [61] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning (still) requires rethinking generalization. Communications of the ACM, 64(3):107– 115, 2021.
- [62] Dongqing Zhang, Jiaolong Yang, Dongqiangzi Ye, and Gang Hua. LQ-Nets: Learned quantization for highly accurate and compact deep neural networks. In The European Conference on Computer Vision (ECCV), September 2018.
- [63] Wei Zhang, Lu Hou, Yichun Yin, Lifeng Shang, Xiao Chen, Xin Jiang, and Qun Liu. Ternarybert: Distillation-aware ultra-low bit bert. arXiv preprint arXiv:2009.12812, 2020.
- [64] Xishan Zhang, Shaoli Liu, Rui Zhang, Chang Liu, Di Huang, Shiyi Zhou, Jiaming Guo, Yu Kang, Qi Guo, Zidong Du, et al. Adaptive precision training: Quantify back propagation in neural networks with fixed-point numbers. arXiv preprint arXiv:1911.00361, 2019.
- [65] Ritchie Zhao, Yuwei Hu, Jordan Dotzel, Chris De Sa, and Zhiru Zhang. Improving neural network quantization without retraining using outlier channel splitting. In International conference on machine learning, pages 7543–7552. PMLR, 2019.
- [66] Aojun Zhou, Anbang Yao, Yiwen Guo, Lin Xu, and Yurong Chen. Incremental network quantization: Towards lossless cnns with low-precision weights. International Conference on Learning Representations, 2017.
- [67] Feng Zhu, Ruihao Gong, Fengwei Yu, Xianglong Liu, Yanfei Wang, Zhelong Li, Xiuqi Yang, and Junjie Yan. Towards unified int8 training for convolutional neural network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1969–1979, 2020.

### A Implementation Details

In this section, we present some works that need to be done to actually accelerate the training process on hardware.

#### A.1 BMM in Attention

In attention, there are batch matrix multiplications (BMMs) that need to be dealt with. We now show that our method for MMs can be extended to BMMs.

Consider the following BMM product:

T = BMM(Q,K⊤),

where we define T ∈ RB×N×P,Q ∈ RB×N×M,K ∈ RB×P×M. The Hadamard matrix is defined as :

Hˆ = RepeatB(H) = RepeatB(BlockDiag(Hk,...,Hk)), where Hˆ ∈ RB×M×M,H ∈ RM×M,Hk ∈ R2

k×2k. In this case,

T ≈ BMM BMM(Q,Hˆ ),BMM(K,Hˆ )⊤ , which verifies that our HQ can be applied to BMMs. For backward, the gradient of weight and activation can be calculated by the straight-through estimator ⌊x⌉′ = 1 and the chain rule:

∇Q = sQ BMM(∇⊤T,Kˆ ) ◦ IQ H⊤, ∇K = sKIK ◦ BMM(∇T,Qˆ )H⊤ = sKBMM(IK ◦ ∇T,Qˆ )H⊤,

where we define sQ ∈ RB,sk ∈ RB being the batch step size, Kˆ = ints

BMM(K,Hˆ ) , Qˆ = ints

K

BMM(Q,Hˆ ) , IQ = I(−QN ≤ Q/sQ ≤ QP), and IK = I(−QN ≤ K/sK ≤ QP).

Q

Similar to Sec. 4.2, we only focus on BMM(∇⊤T,Kˆ ) and ∇T, since we do leverage sampling on them.

For BMM(∇⊤T,Kˆ ), we define the sample probability pi and sample the M˜ in the same way as MMs. The matrix can be computed as BMM(BMM(∇↕T

⊤

,Hˆ˜ ),Kˆ ↕), where Hˆ˜ is defined as CONCAT(H˜ 1,··· ,H˜ B), ∇↕T

⊤

and Kˆ ↕ follows the same definition of Eq. 6and the leverage score is cb,i := ∥∇↕Tb,i,:∥∥K↕b,i,:∥ for 0 ≤ b ≤ B,0 ≤ i ≤ 2M.

For ∇T, similarly, can be viewed as ∇T = BMM(ˆI↕,∇↕T),where we define ∇↕Y = CONCAT([s↑b∇↑Tb;s↓b∇↓Tb]) ∈ RB×2N×P, ˆI↕ = CONCAT([I I]) ∈ RB×N×2N, s↑b,∇↑Tb,s↓b,∇↓Tb follows the definition of Eq.5. So it can be computed as BMM(BMM(ˆI↕,Hˆ˜ ),∇↕T), where Hˆ˜ is defined as CONCAT(H˜ 1,··· ,H˜ B), and the leverage score is cb,i := ∥∇↕Tb,i,:∥ for 0 ≤ b ≤ B,0 ≤ i ≤ 2M, which verifies that our LSS can be applied to BMM.

#### A.2 Computing Leverage Score

In the previous discussion, we find the optimal sample probability pi that can minimize the variance of the gradient. However, it is likely for the proportional pi is larger than one, which is invalid for the Bernoulli distribution. Accordingly, we propose an algorithm to solve this issue.

Define the probability array as

2N

P = [p01,··· ,p02N],

i=1

p0i = N,

we first clamp the array to p1i ∈ [0,1]. In this case, 2i=1N p1i ≤ N, so we scale the pi which is smaller than 1 to make sure their sum is again N. However, this will probably introduce some more elements

larger than 1, so we cycle through the above operations until all the pi ∈ [0,1]. This process will certainly stop, since if after the scaling operation, no element is larger than 1, then we get a valid distribution. Otherwise, the number larger than 1 is reduced by at least one, thus the process will halt after at most O(N) times.

#### A.3 Learning Quantizer Parameters

In this section, we discuss the detail of how to calculate the gradient of activation and quantization step size.

For gradient of activation, the coefficient ci := ∥∇↕Yi∥ is the leverage score for activation gradient, and the variance achieves its minimum When pi ∝ ci by the Cauchy Inequality.

Putting everything together, we propose the following MM procedure to compute activation gradient:

|Procedure LSS-MM<br><br>1. Quantize ∇Y with BS to obtain ∇↑Y and ∇↓Y in INT4.<br>2. Compute the leverage score ∥∇↕Yi∥ in FP16.<br>3. Sample the masks {mi}.<br>4. Sample rows of ∇Y given the masks {mi}.<br>5. Compute IM˜ ↑∇↑Y and IM˜ ↓∇↓Y by discard some of its rows.<br>6. Compute INT4 MMs IM˜ ↑∇↑Y<br><br>Wˆ and IM˜ ↓∇↓Y<br><br>Wˆ .<br><br>7. Dequantize and sum up the resultant INT32 matrices to obtain the FP16 result ˆI↕∇↕Y<br><br><br>Wˆ .|
|---|

The two matrix multiplications in Step 5 take about 2NCD INT4 MACs in expectation. For the quantization step sizes. Following the chain rule, we have

∇sW = g(sW)∇⊤YXˆ ◦ δW(sW), ∇sX = g(sX)∇YWˆ ◦ δX(sX),

where we define g(sW) = 1/ QpNW, g(sX) = 1/ QpNX, NW and NX being the number of elements of weight and activation, δX(sX) = ints

(X) − IX ◦ (X/sX), and δW(sW) = ints

X

(W) − IW ◦ (W/sW).

W

, the most expensive MMs are ∇⊤YXˆ and ∇YWˆ , which are already calculated through Eq. (7) and Eq. (8) during previous calculations, so it does not require extra computation. The elementwise multiplication with δX(sX) and δW(sW) requires minor computation.

Notice that for computing ∇sW

and ∇sX

#### A.4 Cold Start Problem

There is a cold start problem. When the model is trained from scratch (i.e., from a random initialization), distributions of weights and activations can change rapidly in the early stage of optimization. In this case, jointly optimizing the quantization step size and the weights would cause the training to be unstable. As a remedy, we do not learn the step size in the first few iterations, and use a heuristic rule to dynamically set the step size for each tensor X to 2mean(X)/ Qp in each iteration.

#### A.5 Choose hadamard matrix size

For the hadamard matrix, let the hadamard matrix to be H ∈ RD×D: H = BlockDiag(Hk,...,Hk), where D is a multiple of 2k. We first define

X¯ k = sXints

(XH)H⊤, W¯ = sWints

#### (WH)H⊤,

X

W

where X¯ and W¯ can be viewed as an approximation of X and W. Then, we define the quantization error to be MSE(X¯ ,X) × MSE(W¯ ,W). We search for the optimal k that can minimize this quantization error. For fine-tuning tasks, once the hadamard matrix size has been calculated, we fix it through the training process. For the pre-training task, since the distribution shifts greatly as we train the model, we empirically define a time when we re-initialize the hadamard matrix size and the LSQ step size. Usually, we do this when the first 2 epochs finish.

#### A.6 GPU Implementation

In the previous discussion, we get to know HQ-MM and LSS-MM from an algorithm level, nevertheless it is not enough to actually implement it on hardware. In this section, we will delve deeper into hardware implementation details as well as extra limitations.

HQ-MM can be divided into 5 parts: Hadamard matrix multiplication, Quantize, Data Pack, INT4 GEMM, and Dequantize.

For the Hadamard matrix multiplication process, since it can be interpreted as a half float matrix multiplication process where the two matrices involved in the operation are input/weight matrix and hadamard matrix, respectively, we implement it in Python, because PyTorch MM uses CublassGemm and is more efficient then CutlassGemm.

In the quantize process, we quantize input/weight into INT4 data respectively, and also preserve a corresponding FP16 version for the LSQ Back Propagation process to use.

In the previous discussion, we assume the quantize part of HQ-MM is quantizing the resultant matrices to INT4, however, the smallest representation unit of data is INT8. As a result, we actually use INT8 data type to represent quantized data and pack two adjacent data into one data using (data[1] << 4)|(data[0]&15) in the data packing process, which means we use one INT8 data to represent two adjacent INT4 data. With both input matrices’ data packed in this way, we then use cutlass tensor-core INT4 GEMM to do the matrix multiplication.

For the GEMM process, we choose Nvidia CutlassGemm because it’s the most efficient open-source operator library we can find. We use INT4 Tensor Core Gemm for our implementation and it requires the two input matrices A&B to be RowMajor and ColMajor, respectively. Since the default Pytorch tensor is RowMajor, we have to use Transpose+Contiguous operations to make it ColMajor, which is very time-consuming and needs further optimization in the future.

Finally, we dequantize the INT GEMM result back into FP16 output using a dequantize kernel, which is the final output of the forward kernel.

As compared, LSS-MM is more complicated, and can be divided into 7 parts: Quantization of higher lower 4-bit, Leverage Score Calculating, Sampling, Data Pack, INT4 GEMM, Dequantize, and LSQ Back Propagation.

In the Quantize process, we fuse the quantize operation of higher 4-bit and lower 4-bit into a single kernel for acceleration. In the Leverage Score Calculating process, we use the quantized INT8 data to calculate the score and scale up it in the final because integer arithmetic is far more efficient than float arithmetic.

In the sampling process, we sample out rows/columns given the previously calculated leverage score. Note that in Section. A.2, we repeat our proposed algorithm for several loops to sample out specific elements, which is effective but not efficient. According to experiments, however, we notice that simply selecting elements whose leverage score is bigger than 0 can also work well, even better than our proposed algorithm in some cases. So in real quantization implementation, we just sample out rows/ columns whose Euclidean norm is bigger than 0 to accelerate our training process.

Pack, Gemm, and Dequantize processes are as similar as before. It’s worth noting that for Int4 Tensor Core Gemm, suppose two input matrices have shape M × K and K × N, K needs to be a multiple of 32 so that the Tensor core Gemm address can be aligned. We do not need to consider this in the Forward Propagation process because the input data shape always satisfies. However, in the Back Propagation process, the matrix shape may not meet the requirement after sampling. As a result, we need zero_padding the sampled matrix so that K can be a multiple of 32.

Finally, we utilize the dequantized data to do the LSQ Back Propagation. We also fuse all operations into a single Cuda kernel for acceleration, and the metric remains.

Besides the component of HQ-MM and LSS-MM , there is still something that needs to be mentioned.

- 1. We omit the Quantization and Leverage Score Calculating process in LSSinput, and use the same value as LSSWeight to accelerate the training process.
- 2. For Element-Wise kernel, we set block size as 256, grid size as input.numel()/256. For Reduction kernels like sum and min/max, we set block size as 32, grid size as RowNum,

reducing elements in each row to the first 32 elements. We find this setting to be most efficient through experiments.

- B Proofs. In this section, we present the proofs of the leverage score.

#### B.1 Proof of Proposition. 4.1

- Proposition B.1. (LSS variance for weight gradient)

Var

2N

2N

mi pi ∇↕Y

⊤ :,iX↕i =

i=1

i=1

1 − pi pi ∥∇↕Yi,:∥2∥X↕i,:∥2.

Proof.

2N

1 pi

⊤ :,iX↕i )

(mi∇↕Z

V ar(∇W) = V ar

i=1

C

D

2N

1 pi

⊤ j,iX↕i,k)

mi∇↕Z

(

= V ar

j=1

i=1

k=1

C

D

2N

pi(1 − pi) p2i

⊤ j,iX↕i,k)

∇↕Z

V ar (

=

j=1

i=1

k=1

2N

C

D

2

1 − pi pi

⊤ j,i

2

∇↕Z

X↕i,k

=

(

).

i=1

j=1

k=1

| |
|---|

So that

which proves.

2N

V ar(∇W) =

i=1

2N

=

i=1

C

2

1 pi − 1)(

⊤ j,i

∇↕Z

(

j=1

D

X↕i,k

)(

k=1

2

) (9)

1 pi − 1)∥∇↕Z

⊤ :,i∥2∥X↕i,:∥2, (10)

(

###### B.2 Proof of Activation Leverage Score in Sec. 4.2 we divide the matrix multiplication into the sum of 2N smaller multiplications:

2N

2N

ˆI↕:,i∇↕Yi =

ˆI↕∇↕Y =

i=1

i=1

∇ˆYi

, (11)

= ˆI↕:,i∇↕Yi. We assigns each ∇Yi

where we define ∇ˆYi

a probability pi ∈ [0,1],i = 1,··· ,2N, that satisfies 2i=1N pi = N. We define random masks mi ∼ Bern(pi), and define M˜ = diag m1

p1 ,..., m

p2N , and make an unbiased estimation:

2N

2N

ˆI↕∇↕Y ≈ ˆI↕M˜ ∇↕Y =

i=1

mi pi ∇↕Yi.

Define M↑ to be the top-left N × N submatrix of M and M↓ to be the bottom-right one, we have

ˆI↕M˜ ∇↕Y = s↑IM˜ ↑∇↑Y + s↓IM˜ ↓∇↓Y,

In this case, IM˜ ↑∇↑Y and IM˜ ↓∇↓Y both only have parts of its rows being non zero, and the rest rows are zeros since they are discarded. Then, when we multiply it by Wˆ , there are half of rows being

zeros in IM˜ ↑∇↑Y

#### Wˆ and IM˜ ↓∇↓Y

Wˆ . So there’s no need to calculate them, and we successfully cut off half of the computation in this case. Now focus on the variance that

- Proposition B.2. (LSS variance for activation gradient)

2N

2N

1 − pi

ˆI↕:,i∇↕Yi =

pi ∥∇↕Yi∥2. Proof.

Var

i=1

i=1

2N

1 pi

(miˆI↕:,iX↕i )

V ar(∇X) = V ar

i=1

C

D

2N

1 pi

miˆI↕j,i∇↕Yi,k)

(

= V ar

j=1

i=1

k=1

2N

D

C

pi(1 − pi) p2i

ˆI↕j,i∇↕Yi,k)

=

V ar (

i=1

j=1

k=1

2N

C

D

1 − pi pi

(ˆI↕j,i)2(∇↕Yi,k)2

=

i=1

j=1

k=1

2N

D

C

1 pi − 1)

(ˆI↕j,i)2)(

(∇↕Yi,k)2

(

=

i=1

j=1

k=1

2N

1 pi − 1)∥ˆI↕:,i∥2∥∇↕Yi∥2

(

=

i=1

2N

1 pi − 1)∥∇↕Yi∥2.

(

=

i=1

| |
|---|

In this way, the coefficient ci := ∥∇↕Yi∥ is the leverage score.

- C Experiments. In this section, we present more details for experiments in Sec. 5.

#### C.1 Experiments setup

For the GLUE, QA, SWAG, and CONLL tasks, we implement our algorithm based on https:

//github.com/huggingface/transformers. For the machine translation task, we implement our algorithm based on https://github.com/facebookresearch/fairseq. For the ViT fine-tuning task, we implement our algorithm based on https://github.com/jeonsworld/ViT-pytorch. For the deit pretraining task, we implement our algorithm based on https://github.com/ facebookresearch/deit.

We employed NVIDIA GeForce RTX 3090 for running most of the experiments, while the NVIDIA A40 was utilized to evaluate the performance of BERT-Large and ViT-L. Furthermore, we conducted runtime measurements using the NVIDIA T4, 3090, and A100 GPUs.

Table 2: GLUE results on BERT-base-uncased and BERT-large uncased. FP refers to full precision training, INT8 refers to INT8 training, LSQ + LUQ refers to learned step size quantization for forward and logarithmic unbiased quantization for backward, and HQ + LSS refers to Hadamard quantization for forward and leverage score sampling for backward.

QUANTIZATION METHODS MODEL DATASET FP INT8 LSQ+LUQ HQ+LSS

COLA 56.890.64 56.150.94 18.763.58 52.461.46 STSB 88.140.73 87.050.38 84.310.29 87.770.30

RTE 64.801.26 62.271.26 56.800.92 62.451.08

MRPC 88.610.66 86.850.76 86.230.67 86.540.83 SST2 92.720.06 92.370.17 90.370.46 92.490.29 QNLI 91.520.22 90.920.24 87.330.48 90.530.23

BERT-BASE

QQP 91.090.11 90.570.05 89.260.03 89.800.05 MNLI 84.520.22 84.100.08 81.790.18 83.590.12 MNLI-MM 84.680.20 84.490.31 82.220.33 83.750.28

COLA 60.330.49 58.801.52 0.000.00 53.461.17 STSB 87.592.39 86.530.20 83.080.41 87.570.78

RTE 71.121.80 63.711.26 53.060.72 64.620.78

MRPC 91.060.28 87.571.47 82.560.59 87.620.51 SST2 93.980.17 93.750.63 83.940.69 93.520.40 QNLI 92.260.05 92.290.29 63.1813.10 91.530.38

BERT-LARGE

QQP 91.040.63 90.710.00 75.6212.44 90.770.02 MNLI 86.710.19 85.820.08 33.421.38 85.860.10

MNLI-MM 86.410.35 85.870.14 33.541.55 85.820.07

- C.2 GLUE results

In this section, we present the detailed result of fine-tuning the GLUE dataset on BERT-base-uncased and BERT-large-uncased.

On BERT-base, on STSB, SST2, QNLI, and QQP, HQ+LSS only has < 0.5% accuracy degradation. On the most challenging tasks CoLA and RTE, our accuracy degradation is much smaller compared to LSQ+LUQ. On QQP and MNLI, our method achieves < 1.3% degradation, while LSQ + LUQ has ≥ 1.8% degradation. The trend is that the more difficult the task is, the more significant our advantage over LSQ+LUQ.

On BERT-large, the improvement is significant. On CoLA, QNLI, and MNLI, the accuracy improvement compared with LSQ+LUQ > 30%. On other datasets like SST2 and QQP, the accuracy improvement is > 10%. On RTE the accuracy improvement is > 5%, and on STSB and MRPC the improvement is > 3%.

We suspect that for those challenging tasks, there is more information stored in the outliers, which results in a larger gap between our method and LSQ+LUQ.

- C.3 More Granular Quantization Methods

In this section, in Table 4, we show that the more granular quantization methods, such as per-token quantization and per-channel quantization, or smoothing techniques, such as SmoothQuant, do not work under the 4-bit FQT setting. Meanwhile, combining these methods with HQ will not bring significant improvement.

We find that LSQ is beneficial for all of these more granular quantization methods under low-bit settings, which highlights the importance of LSQ. Meanwhile, we also notice that the smoothquant will even harm the result of LSQ when the bit-width is low. Our explanation is that the motivation of LSQ is to learn a trade-off between outliers and inliers, while smoothquant aims to sacrifice the

- Table 3: Experiments on GPT2-base and Bert-large. Total time spent for epoch 1-5 are reported. TRAINING METHODS

MODEL (HIDDEN_SIZE, INTERMIDIATE_SIZE, BATCH_SIZE) FP16 HQ+LSS SPEEDUP

(2560, 10240, 2048) 15.094S 18.949S −25.5% (4096, 16384, 1280) 32.016S 30.594S 4.4%

(5120, 20480, 960) 47.418S 39.482S 16.7% (7680, 30720, 600) 95.832S 67.253S 29.8% (8960, 35840, 480) 128.441S 83.388S 35.1% (9600, 38400, 160) 161.114S 114.325S 29.0%

BERT-LARGE

(12800, 51200, 100) 326.265S 255.966S 21.5% (14400, 57600, 96) 409.291S 346.354S 15.3%

(2560, 10240, 1536) 17.253S 22.037S −27.7% (4096, 16384, 960) 35.937S 35.694S ~ (5120, 20480, 768) 52.723S 46.548S 11.7% (7680, 30720, 260) 113.855S 92.548S 18.7% (8960, 35840, 200) 150.680S 114.881S 23.8% (9600, 38400, 180) 172.182S 126.540S 26.5%

GPT2-BASE

(12800, 51200, 112) 320.757S 236.433S 26.3%

[Figure 1]

Figure 6: Time proportion for each part in HQ-MM and LSS-MM operator.

precision of inliers in order to exactly maintain the information of outliers. When the bitwidth is high, this is not a problem, since there are still enough bits to quantize the inliers. But when the bitwidth is low, such sacrifice will cause severe problems since the inlier information is discarded.

#### C.4 Large Language Model Operator Speed

In this section, we show that our hardware-friendly INT4 training method can really accelerate the training process on Large Language Models. We run distributed training on a system of 8 A100 cards and our implementation uses distributed data parallel training with zero-3, gradient checkpointing, and optimizer offloading.

We experimented with two architectures: BERT-Large and GPT2-base. We vary the network width and batch size to make full utilization of the GPU memory and show the end-to-end performance for fine-tuning these models on the SuperGLUE RTE dataset in Table 3.

#### C.5 More experiments on Operator Speed

Time proportion We examine the proportion of time for each part of computation in HQ-MM and LSS-MM operator in Fig. 6 when the shapes of input matrices vary. In HQ, hadamard means multiplying the input matrix with the Hadamard matrix, pack means packing input data into INT4 data, gemm means the matrix multiplication of two INT4 matrices. In LSSWeight, quantize corresponds to

[Figure 2]

70

FP16 INT4 HQ LSSWeight

60

| |
|---|

| |
|---|

50

| |
|---|

LSSAct

40

Tflops

30

20

10

0

(4608,5120,6144)(5120,6144,8192)(6144,6144,9216)(7168,6656,8704)(8192,7680,9728)(15360,8704,10752)

Matrix size (M,N,K)

Figure 7: Real quantization performance on Nvidia T4.

[Figure 3]

400

FP16 INT4 HQ LSSWeight LSSAct

350

| |
|---|

| |
|---|

300

| |
|---|

250

| |
|---|

Tflops

200

150

100

50

0

(4608,5120,6144)(5120,6144,8192)(6144,6144,9216)(7168,6656,8704)(8192,7680,9728)(15360,8704,10752)

Matrix size (M,N,K)

Figure 8: Real quantization performance on Nvidia A100.

- Table 4: Comparison of different quantization methods, quantize the activation and weight into the same bit-width from 2 to 8. Per-token refers to quantize activation per-token, while Per-channel refers to quantize weight per-channel.

Quantize Bits quantization methods 2 3 4 5 6 7 8

Per-tensor 0 0 0 0 0 50.2 54.6 Per-token 0 0 0 0 31.4 52.8 56 Per-channel 0 0 0 0 0 51.9 56.7

smoothquant 0 0 0 0 0 49.4 57.7 Per-token + Per-channel + smoothquant 0 0 0 0 40.7 55.7 56.7

LSQ 0 9.16 24.2 37.3 39.6 45.3 51.4 Per-token + LSQ 0 15.3 27.8 31.6 42.9 46 54.4

Per-channel + LSQ 0 8 23.9 29.3 40 45.5 50.7 smoothquant + LSQ 0 0 0 0 49.6 54.9 57

Per-token + Per-channel + smoothquant + LSQ 0 0 0 0 28.8 52.4 55.2

HQ 0 45.2 54.6 54.2 56.5 57.4 58.4 HQ + Per-token + Per-channel 0 48.4 54.1 54.9 55 56 56

HQ + Per-token + Per-channel + smoothquant 0 0 46.6 54.9 55.9 55.8 56.5

the quantization of higher and lower 4-bit, leverage means computing leverage score, sample means sample out rows/columns given the leverage score, dequantize is the process of dequantizing INT data back into FP16 data, and LSQ is the backpropagation process of LSQ method. In LSSAct, we ignore quantize and leverage process, using the same value as LSSWeight for saving time, other processes share the same meaning with LSSWeight. Note that our implementation is not fully optimized, and optimizations like operator fusion can further improve the performance.

Operator Speed on more GPUs On an Nvidia RTX 3090 GPU with a Cuda capability of sm_86., we show the comparison of FP16 MM, HQ, and LSS operators in Section 5.3 as well as time proportion in each operator in Figure. 6. We also adjust our hardware implementation and test its performance on Nvidia T4 GPU and Nvidia A100 GPU, which have Cuda capability of sm_75 and sm_80 , respectively. The result is shown in Fig. 7 and Fig. 8.

