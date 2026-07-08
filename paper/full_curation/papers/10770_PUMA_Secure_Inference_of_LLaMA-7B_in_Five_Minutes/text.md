arXiv:2307.12533v3[cs.CR]26Sep2023

[Figure 1]

PUMA: SECURE INFERENCE OF LLAMA-7B IN FIVE MINUTES

[Figure 2]

A PREPRINT

Ye Dong Ant Group dongye.dong@antgroup.com

Wen-jie Lu Ant Group juhou.lwj@antgroup.com

Yancheng Zheng Ant Group zhengyancheng.zyc@antgroup.com

Haoqi Wu Ant Group haoqi.whq@antgroup.com

Derun Zhao Ant Group zhaoderun.zdr@antgroup.com

Zhicong Huang Ant Group zhicong.hzc@antgroup.com

Cheng Hong Ant Group vince.hc@antgroup.com

Jin Tan Ant Group tanjin.tj@antgroup.com

Tao Wei Ant Group lenx.wei@antgroup.com

Wenguang Chen Ant Group yuanben.cwg@antgroup.com

September 27, 2023

ABSTRACT

With ChatGPT as a representative, tons of companies have began to provide services based on large Transformers models. However, using such a service inevitably leak users’ prompts to the model provider. Previous studies have studied secure inference for Transformer models using secure multiparty computation (MPC), where model parameters and clients’ prompts are kept secret. Despite this, these frameworks are still limited in terms of model performance, efﬁciency, and deployment. To address these limitations, we propose framework PUMA to enable fast and secure Transformer model inference. Our framework designs high quality approximations for expensive functions such as GeLU and softmax, and signiﬁcantly reduce the cost of secure inference while preserving the model performance. Additionally, we design secure Embedding and LayerNorm procedures that faithfully implement the desired functionality without undermining the Transformer architecture. PUMA is about 2× faster than the state-of-the-art framework MPCFORMER(ICLR 2023) and has similar accuracy as plaintextmodels without ﬁne-tuning(which the previousworks failed to achieve). PUMA can even evaluate LLaMA-7B in around 5 minutes to generate 1 token. To our best knowledge, this is the ﬁrst time that a model with such a parameter size is able to be evaluated under MPC. PUMA has been open-sourced in the Github repository of SecretFlow-SPU1.

- 1 Introduction

Pre-trained Transformer models (Vaswani et al., 2017) have attracted much attentions for their high performance in practical tasks (Radford & Narasimhan, 2018; Zhuge et al., 2021) and been widely in Deep Learning as a Service (DLaaS) paradigm (Soifer et al., 2019). However, these services can raise privacy concerns, such as in the case of ChatGPT (Brown et al., 2020), which requires either users to reveal their private prompts to the service provider or the service provider to release their proprietary trained weights to users.

[Figure 3]

1https://github.com/secretflow/spu/tree/main/examples/python/ml/flax_llama7b

One solution to address the privacy concerns of Transformer models service is Secure Multi-Party Computation (MPC) (Shamir, 1979; Yao, 1986; Goldreich et al., 1987), which can keep data and model weights private during inference. (Hao et al., 2022; Li et al., 2023; Akimoto et al., 2023; Liang et al., 2023; Liu & Liu, 2023) have proposed various ways to support secure Transformer models inference, but these approaches still have one or several of the following drawbacks:

High inference cost. Non-linear functions like GeLU and softmax are challenge to design in MPC. (Hao et al., 2022) computes these non-linear functions in a faithful way. e.g., they design GeLU using tanh based on general MPC exponentiation method proposed by (Rathee et al., 2021). But these general methods are quite expensive in terms of computation and communication, and only tested under small bitwidth (e.g. below 32).

Retraining required. To reduce the cost of non-linear functions, several works (Li et al., 2023; Akimoto et al., 2023; Liu & Liu, 2023) suggested to approximate GeLU and softmax using simpler functions like ReLU and quadratics. These functions are up to an order of magnitude cheaper in MPC, but would introduce utility loss to the Transformer model. As a result, they require an extra step of model retraining (ﬁne-tuning). However, retraining is unfriendly for data-limited participants, and might not achieve satisfactory performance (Kumar et al., 2022).

Incompatible architectures. (Li et al., 2023; Liang et al., 2023) proposed to modify the architecture of Transformer models to further accelerate secure inference, e.g., decompose the embedding procedure or reorganize the linear layers. Worsely, (Li et al., 2023) does not support secure LayerNorm and simulated the costs using BatchNorm, resulting in incorrect secure inference results. These modiﬁcations are in conﬂicts with existing plaintext Transformer systems, and would lead to deployment obstacles.

To summarize, in the ﬁeld of MPC Transformer inference, achieving both model performance and efﬁciency is challenging, and people may ask the following question:

Could pre-trained large transformer models be securely and efﬁciently evaluated with similar accuracy as in plaintext, without further retraining ?

To address this challenge, we propose the PUMA framework, which is a fast and accurate end-to-end secure Transformer inference framework. Our contributions can be summarized as follows:

- • New Approximations for Non-linear Functions. We propose more accurate and faster approximations for the expensive non-linear functions (e.g., GeLU and softmax) in Transformer models. Different from existing works, we design the approximations based on the specialized properties of these non-linear functions to achieve both accuracy and efﬁciency.
- • Faster and More Accurate Secure Inference. We make extensive experiments on 6 transformer models and 4 datasets, the results show that PUMA’s precision is similar to plaintext ones’ and is about 2× faster than MPCFORMER (note that MPCFORMER does not achieve similar precision as PUMA). PUMA can even evaluate LLaMA-7B in around 5 minutes to generate one word. To our best knowledge, this is the ﬁrst time that such a large language model is able to be evaluated under MPC.
- • End-to-End Framework compatible with plaintext. We design and implement all the layers required by Transformer (including the Embedding and LayerNorm layers that are missing in other works) in MPC. This allows us to load and securely evaluate the pre-trained plaintext Transfomer models (e.g. downloaded from Hugging face) easily. To our best knowledge, PUMA is the ﬁrst open-sourced MPC solution that supports accurate inference of pre-trained Transformer models without further modiﬁcations such as re-training.

Organization. We summarize the related work in § 2 and present the background in § 3. We give PUMA’s high-level view and concrete design in § 4. We analyze the experimental results in § 5 and conclude this work in § 6.

- 2 Related Work

Secure Multiparty Computation (MPC) (Yao, 1986; Goldreich et al., 1987) enables distrusted parties to jointly compute a function while keeping their inputs private, and secure deep learning inference using MPC has gained much attention due its high privacy protection. These works operate in a variety of models and architectures, including twoparty setting (Mohassel & Zhang, 2017; Liu et al., 2017; Mishra et al., 2020; Huang et al., 2022; Patra et al., 2021; Rathee et al., 2020), three-party setting (Wagh et al., 2019; Mohassel & Rindal, 2018; Wagh et al., 2020; Kumar et al., 2019; Patra & Suresh, 2020; Tan et al., 2021; Dong et al., 2023), or four-partysetting (Byali et al., 2020; Dalskov et al., 2021). However, most of these approaches only consider secure inference of convolutional/deep neural networks, and cannotbe directly extended to supportTransformermodels. Recently several research works (Hao et al., 2022; Li et al., 2023; Akimoto et al., 2023; Liang et al., 2023; Liu & Liu, 2023) have proposed MPC-based secure inference solutions

for Transformer models, but these approaches still have limitations in terms of model performance, efﬁciency, and deployment. Among these works, MPCFORMER (Li et al., 2023) is the only one that have been open-sourced, it is based on CrypTen (Knott et al., 2021) which is a three-party framework that uses a non-colluding third party to produce correlated randomness for the client and server. Also their three-party model with non-colluding assumption has the highest concrete efﬁciency among different MPC settings. So we mainly compare our proposed framework PUMA with MPCFORMER under the same three-party setting.

- 3 Background

- 3.1 Notations

The main used notations are as follows: Pi represents the i-th computing party, i ∈ {0,1,2}. The uppercase bold letter X is used for matrices, and the lowercase bold letter x denotes vectors. x[i] denotes the i-th element of vector

x, while lowercase letter x is used for scalar values. Z2ℓ denotes the discrete ring modulo 2ℓ, R denotes real numbers. · is used for 2-out-of-3 replicated secret sharing (Araki et al., 2016; Mohassel & Rindal, 2018).

- 3.2 Transformer Model

Transformer models have achieved remarkable success in language understanding (Radford & Narasimhan, 2018; Devlin et al., 2019; Yang et al., 2019; Touvron et al., 2023), vision understanding (Zhuge et al., 2021; Dong et al., 2022; Chen et al., 2021), and etc. Two popular variants are Bert (Bidirectional Encoder Representations from Transformers) (Devlin et al., 2019) and GPT (Generative Pre-Trained models) (Radford & Narasimhan, 2018). A Transformer model (Vaswani et al., 2017) mainly consists of Embedding, Attention, Feed-Forward Network, and LayerNorm sub-layers:

Attention. Given inputs (Q,K,V), the Attention function is computed as Attention(Q,K,V) = softmax(Q·KT + M) · V, where M can be viewed as a bias matrix. Besides, (Vaswani et al., 2017) proposed Multi-Head Attention to jointly attend to information from different representation subspaces at different positions.

Feed-Forward Network (FFN). FFN is applied to each position separately and identically. This consists of two linear transformations with an activation in between, and the most commonly used activation function is GeLU. Given input x and parameters {W1,b1,W2,b2}, FFN can be formalized as FFN(x) = W2GeLU(W1x + b1) + b2. Note that the parameters of linear transformations are different from layer to layer.

LayerNorm. Given vector x ∈ Rn, LayerNorm is deﬁned as: LayerNorm(x)[i] = γ · x[√i]−σµ + β, where (γ,β) are trained parameters, µ =

[Figure 6]

[Figure 7]

n i=1 x[i]

[Figure 8]

n , and σ = ni=1(x[i] − µ)2.

- 3.3 2-out-of-3 Replicated Secret Sharing

- A secret value x ∈ Z2ℓ is shared by three random values x0,x1,x2 ∈ Z2ℓ with x = x0 +x1 + x2 (mod 2ℓ). In 2-outof-3 replicated secret sharing (denoted as · -sharing), party Pi gets x i = (xi,xi+1). Without special declaration,

we compute in Z2ℓ and omit (mod 2ℓ) for brevity. In the case of ℓ > 1 (e.g., ℓ = 64) which support arithmetic operations (e.g., +, −, and ·), we refer to this type as Arithmetic Sharing and use notation · . Boolean Sharing ( · B) refers to ℓ = 1 where (+,−) and · are respectively replaced by bit-wise ⊕ and ∧.

Addition. Let (c1, c2, c3) be public constants, and ( x , y ) be two secret-shared values. Then, c1x + c2y + c3 can be computed as (c1x0 + c2y0 + c3,c1x1 + c2y1,c1x2 + c2y2) where Pi can compute its share locally. When (c1 = 1,c2 = 1,c3 = 0), we get x + y .

Multiplication. In secure multiplication protocol ΠMul, given two shared values x and y , parties follows steps: i) First, Pi computes zi = xiyi + xi+1yi + xiyi+1 locally, ii) Parties then perform re-sharing by letting Pi sends zi′ = αi + zi to Pi−1, where α0 + α1 + α2 = 0 (Pi can generate αi in the setup phase as Mohassel & Rindal (2018)). iii) Finally, {(z0′,z1′),(z1′ ,z2′),(z2′,z0′)} form x · y .

Underlying Protocols. In addition to addition and multiplication, PUMA relies on several other underlying protocols: boolean-arithmetic multiplication (ΠMul

), square ΠSquare, equality test (ΠEq), less than (ΠLT), reciprocal (ΠRecip), maximum (ΠMax), and reciprocal of square root (ΠrSqrt), from the state-of-the-art works. We employ them in a blackbox manner, and only enumerate the inputs and outputs of these protocols as follows:

BA

- • z = ΠMul

BA

( b B, x ), s.t. z = b · x

- • z = ΠSquare( x ), s.t. z = x2
- • z B = ΠEq( x , y ), s.t. z = 1{x = y}
- • z B = ΠLT( x , y ), s.t. z = 1{x < y}

- • z = ΠRecip( x ), s.t. z = 1/x
- • z = ΠrSqrt( x ), s.t. z = 1/√x

[Figure 10]

- • z = ΠMax( x ), s.t. z = maximum(x)

1{e} returns 1 that when condition e is true, and 0 otherwise. For detailed protocol constructions, please refer to (Mohassel & Rindal, 2018; Lu et al., 2020; Keller, 2020).

Fixed-Point Representation & Truncation. Real numbers has to be encoded into ﬁxed-point numbers before represented in ﬁnite rings/ﬁelds. To avoid overﬂow, ΠfTrunc has to be used after each ﬁxed-point multiplication to truncate the least f bits securely. For simpler description, we include ΠfTrunc in ΠMul and ΠSquare by default and and do not explicitly mention it in our protocol designs. The above operations can be easily extended to vectors and matrices, and we use the same notation for vector and matrix operations for simplicity. For more details, please refer to (Mohassel & Rindal, 2018; Wagh et al., 2020).

Threat Model. Following previous works (Mohassel & Rindal, 2018; Li et al., 2023), PUMA is secure against a semihonest adversary that corrupts no more than one of the three computing parties. Semi-honest means such an adversary will follow the protocol speciﬁcations, but may try to learn other’s private information during the protocol. Please note that PUMA cannot defend against attacks based on inference results, and the mitigation of such attacks (e.g., differential privacy (Abadi et al., 2016)) falls outside the scope of this study.

- 4 Secure Design of PUMA

In this section, we ﬁrst present an overview of PUMA, and present the protocols for secure GeLU , softmax, embedding, and LayerNorm used by PUMA. Note that the linear layers such as matrix multiplication are straightforward in replicated secret sharing, so we mainly describe our protocols for non-linear layers in this manuscript.

- 4.1 Overview of PUMA

To achieve secure inference of Transformer models, PUMA deﬁnes three kinds of roles: one model owner, one client, and three computing parties. The model owner and the client provide their models or inputs to the computing parties (i.e., P0, P1, and P2) in a secret-shared form, then the computing parties execute the MPC protocols and send the results back to the client. Note that the model owner and client can also act as one of the computing party, we describe them separately for generality. e.g., when the model owner acts as P0, the client acts as P1, a third-party dealer acts as P2, the system model becomes the same with MPCFORMER (Li et al., 2023).

During the secure inference process, a key invariant is maintained: For any layer, the computing parties always start with 2-out-of-3 replicated secret shares of the previous layer’s output and the model weights, and end with 2-out-of-3 replicated secret shares of this layer’s output. As the shares do not leak any information to each party, this ensures that the layers can be sequentially combined for arbitrary depths to obtain a secure computation scheme for any Transformer-based model.

- 4.2 Protocol for Secure GeLU

Most of the current approaches view the GeLU function as a composition of smaller functions and try to optimize each piece of them, making them to miss the chance of optimizing the private GeLU as a whole. Given the GeLU function:

x 2 · 1 + tanh

GeLU(x) =

[Figure 11]

[Figure 12]

2 π · x + 0.044715 · x3

[Figure 13]

, (1)

≈ x · sigmoid(0.071355 · x3 + 1.595769 · x)

these approaches (Hao et al., 2022; Wang et al., 2022) focus either on designing approximate protocols for function tanh or using existing general MPC protocols of exponentiation and reciprocal for sigmoid.

However, none of current approaches have utilized the fact that GeLU function is almost linear on the two sides (i.e., GeLU(x) ≈ 0 for x < −4 and GeLU(x) ≈ x for x > 3). Within the short interval [−4,3] of GeLU, we suggest a piece-wise approximation of low-degree polynomials is a more efﬁcient and easy-to-implement choice for its secure

- Algorithm 1 Secure GeLU Protocol ΠGeLU

[Figure 16]

Input: Pi holds the 2-out-of-3 replicate secret share x i for i ∈ {0,1,2} Output: Pi gets the 2-out-of-3 replicate secret share y i for i ∈ {0,1,2}, where y = GeLU(x).

- 1: P0, P1, and P2 jointly compute

- b0 B = ΠLT( x ,−4), ⊲ b0 = 1{x < −4}
- b1 B = ΠLT( x ,−1.95), ⊲ b1 = 1{x < −1.95}
- b2 B = ΠLT(3, x ), ⊲ b2 = 1{3 < x}

and compute z0 B = b0 B ⊕ b1 B, z1 B = b1 B ⊕ b2 B ⊕ 1, and z2 B = b2 B. Note that z0 = 1{−4 ≤ x < −1.95}, z1 = 1{−1.95 ≤ x ≤ 3}, and z2 = 1{x > 3}.

- 2: Jointly compute x2 = ΠSquare( x ), x3 = ΠMul( x , x2 ), x4 = ΠSquare( x2 ), and x6 = ΠSquare( x3 ).
- 3: Computing polynomials F0(x) and F1(x) based on { x , x2 , x3 , x4 , x6 } as equation (2) securely.
- 4: return y = ΠMul

( z0 B, F0(x) ) + ΠMul

( z1 B, F1(x) ) + ΠMul

( z2 B, x ).

BA

BA

BA

[Figure 17]

protocol. Concretely, our piece-wise low-degree polynomials are shown as equation (2):

0, x < −4

 

- F0(x), −4 ≤ x < −1.95
- F1(x), −1.95 ≤ x ≤ 3 x, x > 3

GeLU(x) =

, (2)



where polynomials F0() and F1() are computed by library numpy.ployﬁt2 as equation (3). Surprsingly, the above simple poly ﬁt works very well and our max error < 0.01403, median error < 4.41e−05, and mean error < 0.00168.

- F0(x) = −0.011034134030615728x3 − 0.11807612951181953x2 −0.42226581151983866x− 0.5054031199708174
- F1(x) = 0.0018067462606141187x6 − 0.037688200365904236x4

 

(3)



+0.3603292692789629x2 + 0.5x + 0.008526321541038084

Formally, given secret input x , our secure GeLU protocol ΠGeLU is constructed as algorithm 1.

- 4.3 Protocol for Secure Softmax

In the function Attention(Q,K,V) = softmax(Q · KT + M) · V, the key challenge is computing function softmax. For the sake of numerical stability, the softmax function is computed as

exp(x[i] − x¯ − ǫ) i exp(x[i] − x¯ − ǫ)

, (4)

softmax(x)[i] =

[Figure 18]

where x¯ is the maximum element of the input vector x. For the normal plaintext softmax, ǫ = 0. For a two-dimension matrix, we apply equation (4) to each of its row vector.

Formally, our detailed secure protocol Πsoftmax is illustrated in algorithm 2, where we propose two optimizations:

• For the ﬁrst optimization, we set ǫ in equation 4 to a tiny and positive value, e.g., ǫ = 10−6, so that the inputs to exponentiation in equation 4 are all negative. We exploit the negative operands for acceleration. Particularly, we compute the exponentiation using the Taylor series (Tan et al., 2021) with a simple clipping

0, x < Texp (1 + 2xt )2

negExp(x) =

(5)

t

, x ∈ [Texp,0].

[Figure 19]

Indeed, we apply the less-than for the branch x < Texp The division by 2t can be achieved using ΠtTrunc since the input is already negative. Also, we can compute the power-of-2t using t-step sequences of square function

Πsquare and ΠfTrunc. Suppose our MPC program uses 18-bit ﬁxed-point precision. Then we set Texp = −14 given exp(−14) < 2−18, and empirically set t = 5.

[Figure 20]

2https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html

- Algorithm 2 Secure softmax Protocol Πsoftmax

[Figure 23]

Input: Pi holds the 2-out-of-3 replicate secret share x i for i ∈ {0,1,2}, and x is a vector of size n. Output: Pi gets the 2-out-of-3 replicate secret share y i for i ∈ {0,1,2}, where y = softmax(x).

- 1: P0, P1, and P2 jointly compute b B = ΠLT(Texp, x ) and the maximum x ¯ = ΠMax( x ).
- 2: Parties locally computes x ˆ = x − x ¯ − ǫ, and jointly compute z0 = 1 + ΠtTrunc( x ˆ ).
- 3: for j = 1,2,...,t do
- 4: zj = ΠSquare( zj−1 ).
- 5: end for
- 6: Parties locally compute z = ni=1 z[i] and jointly compute 1/z = ΠRecip( z ).
- 7: Parties jointly compute z/z = ΠMul( z , 1/z )
- 8: return y = ΠMul

( b B, z/z ).

BA

[Figure 24]

• Our second optimization is to reduce the number of divisions, which ultimately saves computation and communication costs. To achieve this, for a vector x of size n, we have replaced the operation Div(x,Broadcast(y)) with x · Broadcast(y1), where y = ni=1 x[i]. By making this replacement, we effectively reduce n divisions to just one reciprocal operation and n multiplications. This optimization is particularly beneﬁcial in the case of the softmax operation. The y1 in the softmax operation is still large enough to maintain sufﬁcient accuracy under ﬁxed-point values. As a result, this optimization can signiﬁcantly reduce the computational and communication costs while still providing accurate results.

[Figure 25]

[Figure 26]

- 4.4 Protocol for Secure Embedding

The current secure embedding procedure described in (Li et al., 2023) necessitates the client to generate a one-hot vector using the token id locally. This deviates from a plaintext Transformer workﬂow where the one-hot vector is generated inside the model. As a result, they have to carefully strip off the one-hot step from the pre-trained models, and add the step to the client side, which could be an obstacle for deployment.

To address this issue, we propose a secure embedding design as follows. Assuming that the token id ∈ [n] and all embedding vectors are denoted by E = (eT1 ,eT2 ,...,eTn), the embedding can be formulated as eid = E[id]. Given (id,E) are in secret-shared fashion, our secure embedding protocol ΠEmbed works as follows:

- • The computing parties securely compute the one-hot vector o B after receiving id from the client. Speciﬁcally, o[i] B = ΠEq(i, id ) for i ∈ [n].
- • The parties can compute the embedded vector via eid = ΠMul

BA

( E , o B), where does not require secure truncation.

In this way, our ΠEmbed does not require explicit modiﬁcation of the workﬂow of plaintext Transformer models, at the cost of more ΠEq and ΠMul

BA

operations.

- 4.5 Protocol for Secure LayerNorm

Recall that given a vector x of size n, LayerNorm(x)[i] = γ · x[√i]−σµ + β, where (γ,β) are trained parameters, µ =

[Figure 27]

[Figure 28]

n i=1 x[i]

n , and σ = ni=1(x[i] − µ)2. In MPC, the key challenge is the evaluation of the divide-square-root x[i]−µ

[Figure 29]

√σ formula. To securely evaluate this formula, CrypTen sequentially executes the MPC protocols of square-root,

[Figure 30]

[Figure 31]

reciprocal, and multiplication. However, we observe that x[√i]−σµ is equal to (x[i] − µ) · σ−1/2. And in the MPC side, the costs of computing the inverse-square-root σ−1/2 is similar to that of the square-root operation (Lu et al., 2020). Besides, inspired by the second optimization of § 4.3, we can ﬁrst compute σ−1/2 and then Broadcast(σ−1/2) to support fast and secure LayerNorm(x). And our formal protocol ΠLayerNorm is shown in algorithm 3.

[Figure 32]

[Figure 33]

- 5 Experimental Evaluations

Implementation. We implement PUMA on top of SecretFlow-SPU (Ma et al., 2023) in C++ and Python. We encode the data in a ﬁxed-point form under ring Z264 with 18-bit fractional part. Our experiments are run on 3 Alibaba Cloud ecs.g7.8xlarge servers with 32 vCPU and 128GB RAM each. The CPU model is Intel Xeon(Ice Lake) Platinum 8369B

- Algorithm 3 Secure LayerNorm Protocol ΠLayerNorm

[Figure 36]

Input: Pi holds the 2-out-of-3 replicate secret share x i for i ∈ {0,1,2}, and x is a vector of size n. Output: Pi gets the 2-out-of-3 replicate secret share y i for i ∈ {0,1,2}, where y = LayerNorm(x).

- 1: P0, P1, and P2 compute µ = n1 · ni=1 x[i] and σ = ni=1 ΠSquare( x − µ )[i].

[Figure 37]

- 2: Parties jointly compute σ−1/2 = ΠrSqrt( σ ).
- 3: Parties jointly compute c = ΠMul(( x − µ ), σ−1/2 )
- 4: return y = ΠMul( γ , c ) + β .

[Figure 38]

Table 1: Performance on GLUE benchmark of Bert-Base, Roberta-Base, and Bert-Large on CoLA, RTE, and QNLI, Matthews correlation is reported for CoLA. Accuracy is reported for other datasets.

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Model Bert-Base Roberta-Base Bert-Large TASK CoLA RTE QNLI CoLA RTE QNLI CoLA RTE QNLI

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

CPU 0.616 0.700 0.916 0.629 0.805 0.920 0.686 0.755 0.922 PUMA 0.613 0.700 0.916 0.618 0.805 0.918 0.690 0.747 0.918

[Figure 50]

[Figure 51]

[Figure 52]

CPU @ 2.70GHz. We evaluate PUMA on Ubuntu 20.04.6 LTS with Linux kernel 5.4.0-144-generic. Our bandwidth is about 5Gbps and round trip time is about 1ms.

Models & Datasets. We evaluate PUMA on seven NLP models: Bert-Base, Roberta-Base, and BertLarge (Devlin et al., 2019); GPT2-Base, GPT2-Medium, and GPT2-Large (Radford & Narasimhan, 2018); and LLaMA-7B (Touvron et al., 2023). We measure the Bert performance for three NLP tasks over the datasets of Corpus of Linguistic Acceptability (CoLA), Recognizing Textual Entailment (RTE), Stanford Question Answering Dataset (QNLI) from GLUE benchmarks (Wang et al., 2019), and GPT2 performance on Wikitext-103 V1 (Merity et al., 2016).

Baseline. We compare PUMA to the most similar prior work MPCFORMER (Li et al., 2023). But for fair comparison, we have the following considerations: i) As MPCFORMER neither supports loading pretrained transformer models nor implements LayerNorm faithfully3, we cannot achieve meaningful secure inference results using their framework. Therefore, we compare our performance to that of plaintext (ﬂoating-point) to show our precision guarantee. ii) MPCFORMER with Quad approximations requires retraining the modiﬁed models. As PUMA does not require retraining, we compare our cost to that of MPCFORMER without Quad approximations. Also, we re-run MPCFORMER in our environment.

- 5.1 Precision

We compare our secure model inference performance to that of plaintext (ﬂoating-point) in Table 1 and 2 to show our precision guarantee.

In Table 1, we show the Matthews correlation/accuracy of plaintext and PUMA on the Bert-Base, Roberta-base, and Bert-Large. We observe that the accuracy achieved by PUMA matches the accuracy of the plaintext Flax code. Specifically, the accuracy difference does not exceed 0.011 over all datasets. Moreover, in Table 2, we also compare our perplexity on dataset Wikitext-103 V1 with the plaintext baseline on GPT2 models. The results are similar and the perplexity differences do not exceed 0.02 over all models.

The above accuracy and perplexity advantages experimentally validate that our protocols are numerically precise.

[Figure 53]

3As MPCFORMER does not support loading pre-trained Transformer models, we did an experiment in plaintext Bert-Base that replaced LayerNorm with BatchNorm as MPCFORMER did. This resulted in a signiﬁcant drop in the MCC score for CoLA task from 0 616 to −0 020. On the contrary, PUMA achieves an MCC score of 0 613.

Table 2: Perplexity of GPT2-Base, GPT2-Medium, and GPT2-Large on Wikitext-103 V1. Model GPT2-Base GPT2-Medium GPT2-Large

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

CPU 16.284 12.536 10.142 PUMA 16.284 12.540 10.161

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

- Table 3: Costs of Bert-Base, Roberta-Base, and Bert-Large for one sentence of length 128. Time is in seconds and Communication (Comm. for short) is in GB, which is the same for the following tables.

[Figure 69]

[Figure 70]

Model Bert-Base Roberta-Base Bert-Large Costs Time Comm. Time Comm. Time Comm.

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

MPCFORMER 55.320 12.089 57.256 12.373 141.222 32.577 PUMA 33.913 10.773 41.641 11.463 73.720 27.246 Improv. 1.631× 1.122× 1.375× 1.079× 1.916× 1.195×

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

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

- Table 4: Costs of GPT2-Base, GPT2-Medium, and GPT2-Large. The input sentence is of length 32, all of the costs are for generating 1 token.

[Figure 93]

[Figure 94]

Model GPT2-Base GPT2-Medium GPT2-Large Costs Time Comm. Time Comm. Time Comm.

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

MPCFORMER 34.889 4.999 73.078 11.766 129.095 22.522 PUMA 15.506 3.774 30.272 7.059 54.154 11.952 Improv. 2.250× 1.325× 2.414× 1.667× 2.383× 1.884×

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

5.2 Inference Costs

We compare PUMA’s inference cost to that of MPCFORMER. The costs are for processing one input sentence: i) For Bert models the input sentence is of length 128. ii) For GPT2 models the input length is 32 and generate 1 new word.

On the 3 Bert models in Table 3, PUMA is 1.375 ∼ 1.916× faster than MPCFORMER, and is 1.079 ∼ 1.195× more communication-efﬁcient. For the GPT2 models in Table 4, PUMA is 2.250 ∼ 2.414× faster than MPCFORMER, and is 1.325 ∼ 1.884× more communication-efﬁcient.

We observe that PUMA’s improvementsincrease as the model size grows, particularly for the GPT2 models. This trend is because our specialized optimizations are more effective when processing large-scale evaluations.

5.3 Scalability

In this subsection, we measure the costs of evaluating PUMA on Bert-Base and GPT2-Base models for batched inputs, varying-length inputs, and varying-length outputs (only for GPT2-Base). We also compare our costs to those of MPCFORMER to demonstrate our improvements.

Input Length Evaluation. Table 5 shows our costs on varying-length inputs, we evaluate Bert-Base on inputs of length {64,128,256}, and GPT2-Base on inputs of length {16,32,64}. For Bert-Base, PUMA is 1.631 ∼ 1.837× faster, and for GPT2-Base, PUMA is 1.744 ∼ 2.686× faster.

Output Length Evaluation. Fig 1 presents our costs on varying-length outputs for GPT2-Base. Our improvements against MPCFORMER range from 1.279 ∼ 2.700×.

We observe in Table 5 and Fig 1 that for GPT-2, our efﬁciency gains decrease with more input/output tokens. This is because PUMA introduces extra one-hot embedding costs (as described in 4.4). We should emphasize again that PUMA is compatible with plaintext models, and could achieve a similar accuracy as plaintext models while MPCFORMER could not.

- Table 5: Costs of Bert-Base and GPT2-Base for different input length (denoted as #Input). The input lengths for Bert-Base and GPT2-Base are respectively {64,128,256} and {16,32,64}. GPT2-Base generates 1 token.

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

#Input 64/16 128/32 256/64 Costs Time Comm. Time Comm. Time Comm.

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

MPCFORMER 36.354 5.707 55.320 12.089 112.453 29.927 PUMA 21.141 4.881 33.913 10.773 61.210 26.004 Improv. 1.720× 1.169× 1.631× 1.122× 1.837× 1.151×

[Figure 128]

Bert

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

MPCFORMER 29.695 4.011 34.889 4.999 43.344 7.318 PUMA 11.056 1.875 15.506 3.777 24.860 7.821 Improv. 2.686× 2.139× 2.250× 1.324× 1.744× 0.936×

[Figure 141]

GPT2

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

500

MPCFormer Time

Puma Time

400

300

Time(s)

200

100

2 4 8 16

Figure 1: Runtime of GPT2-Base for generating different output tokens, the input length is of length 32.

- Table 6: Costs of the secure inference of LLaMA-7B, #Input denotes the length of input sentence and #Output denotes the number of generated tokens.

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

(#Input, #Output) (4,1) (8,1) (8,2) Costs Time Comm. Time Comm. Time Comm. PUMA 122.004 0.907 200.473 1.794 364.527 3.857

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

- 5.4 Evaluating LLaMA-7B in Five Minutes.

Our protocols are already complete for evaluating any Transformer-based models including LLaMA-7B. Unfortunately, existing serialization libraries such as Protobuf (Varda, 2008) and FlatBuffers (van Oortmerssen, 2014) only support data trunks with size up to 2GB, which is not sufﬁcient for large MPC tasks. To address this problem, we propose an optimization to SecretFlow-SPU. Concretely, the system could automatically divide and serialize overly large secret-shared structures into smaller chunks when communicating or performing I/O operations.

We evaluated the large language model LLaMA-7B using PUMA under 3 Alibaba Cloud ecs.r7.32xlarge servers, each has 128 threads and 1TB RAM, with 20GB bandwidth, 0.1ms round-trip-time. As shown in Table 6, PUMA can support secure inference of LLaMA-7B with reasonable costs. For example, given an input sentence of 8 tokens, PUMA can output one token in around 200 seconds with communication costs of 1.794 GB. To our knowledge, this is the ﬁrst time that LLaMA-7B has been evaluated using MPC. Moreover, PUMA can generate the same tokens exactly as plaintext LLaMA-7B, see Appendix for an example.

- 6 Conclusion

We propose an efﬁcient MPC framework PUMA for secure inference on Transformer models based on replicated secret sharing. To reduce the costs of secure inference, we approximate expensive functions with accurate polynomials and propose secure Embedding and LayerNorm protocols to support end-to-end secure inference. Although the inference cost is still quite high, we successfully make it one step closer to solving users’ privacy concerns in Transformer-based DLaaS. We believe that by combining PUMA with quantization methods and hardware accelerations in the future, secure inference of large Transformer models in seconds is no longer impossible.

References

Martin Abadi, Andy Chu, Ian Goodfellow, H Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC conference on computer and communications security, pp. 308–318, 2016.

Y. Akimoto, K. Fukuchi, Y. Akimoto, and J. Sakuma. Privformer: Privacy-preserving transformer with mpc. In 2023 IEEE 8th European Symposium on Security and Privacy (EuroSP), pp. 392–410, Los Alamitos, CA, USA, 2023. IEEE Computer Society. doi:10.1109/EuroSP57164.2023.00031. URL https://doi.ieeecomputersociety.org/10.1109/EuroSP57164.2023.00031.

Toshinori Araki, Jun Furukawa, Yehuda Lindell, Ariel Nof, and Kazuma Ohara. High-throughput semi-honest secure three-partycomputationwith an honest majority. In Proceedingsof the 2016 ACM SIGSACConference on Computer and Communications Security, pp. 805–817, 2016.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020.

Megha Byali, Harsh Chaudhari, Arpita Patra, and Ajith Suresh. Flash: Fast and robust framework for privacypreserving machine learning. Proc. Priv. Enhancing Technol., 2020(2):459–480, 2020.

Hanting Chen, Yunhe Wang, Tianyu Guo, Chang Xu, Yiping Deng, Zhenhua Liu, Siwei Ma, Chunjing Xu, Chao Xu, and Wen Gao. Pre-trained image processing transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12299–12310, 2021.

Anders Dalskov, Daniel Escudero, and Marcel Keller. Fantastic four: Honest-majority four-party secure computation with malicious security. In 30th {USENIX} Security Symposium ({USENIX} Security 21), 2021.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. ArXiv, abs/1810.04805, 2019.

Xiaoyi Dong, Jianmin Bao, Ting Zhang, Dongdong Chen, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, and Nenghai Yu. Bootstrapped masked autoencoders for vision bert pretraining. In European Conference on Computer Vision, pp. 247–264. Springer, 2022.

Ye Dong, Chen Xiaojun, Weizhan Jing, Li Kaiyun, and Weiping Wang. Meteor: Improved secure 3-party neural network inference with reducing online communication costs. In Proceedings of the ACM Web Conference 2023, WWW ’23, pp. 2087–2098, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9781450394161.

O. Goldreich, S. Micali, and A. Wigderson. How to play any mental game. In Proceedings of the Nineteenth Annual ACM Symposium on Theory of Computing, STOC ’87, pp. 218–229, New York, NY, USA, 1987. Association for Computing Machinery. ISBN 0897912217.

Meng Hao, Hongwei Li, Hanxiao Chen, Pengzhi Xing, Guowen Xu, and Tianwei Zhang. Iron: Private inference on transformers. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=deyqjpcTfsG.

Zhicong Huang, Wen jie Lu, Cheng Hong, and Jiansheng Ding. Cheetah: Lean and fast secure Two-Party deep neural network inference. In 31st USENIX Security Symposium (USENIX Security 22), pp. 809–826, Boston, MA, August

2022. USENIX Association. ISBN 978-1-939133-31-1. Marcel Keller. Mp-spdz: A versatile framework for multi-party computation. In Proceedings of the 2020 ACM SIGSAC conference on computer and communications security, pp. 1575–1590, 2020.

- B. Knott, S. Venkataraman, A.Y. Hannun, S. Sengupta, M. Ibrahim, and L.J.P. van der Maaten. Crypten: Secure multi-party computation meets machine learning. In arXiv 2109.00984, 2021.

Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, and Percy Liang. Fine-tuning can distort pretrained features and underperform out-of-distribution. arXiv preprint arXiv:2202.10054, 2022.

Nishant Kumar, Mayank Rathee, Nishanth Chandran, Divya Gupta, Aseem Rastogi, and Rahul Sharma. Cryptﬂow: Secure tensorﬂow inference. arXiv preprint arXiv:1909.07814, 2019.

Dacheng Li, Hongyi Wang, Rulin Shao, Han Guo, Eric Xing, and Hao Zhang. MPCFORMER: FAST, PERFORMANT AND PRIVATE TRANSFORMER INFERENCE WITH MPC. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=CWmvjOEhgH-.

Zi Liang, Pinghui Wang, Ruofei Zhang, Lifeng Xing, Nuo Xu, and Shuo Zhang. Merge: Fast private text generation, 2023.

Jian Liu, Mika Juuti, Yao Lu, and Nadarajah Asokan. Oblivious neural network predictions via minionn transformations. In Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, pp. 619–631, 2017.

Xuanqi Liu and Zhuotao Liu. Llms can understand encrypted prompt: Towards privacy-computingfriendly transformers, 2023.

Wen-jie Lu, Yixuan Fang, Zhicong Huang, Cheng Hong, Chaochao Chen, Hunter Qu, Yajin Zhou, and Kui Ren. Faster secure multiparty computation of adaptive gradient descent. In Proceedings of the 2020 Workshop on PrivacyPreserving Machine Learning in Practice, PPMLP’20, pp. 47–49, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450380881.

Junming Ma, Yancheng Zheng, Jun Feng, Derun Zhao, Haoqi Wu, Wenjing Fang, Jin Tan, Chaofan Yu, Benyu Zhang, and Lei Wang. SecretFlow-SPU: A performant and User-Friendly framework for Privacy-Preserving machine learning. In 2023 USENIX Annual Technical Conference (USENIX ATC 23), pp. 17–33, Boston, MA, July 2023. USENIX Association. ISBN 978-1-939133-35-9. URL https://www.usenix.org/conference/atc23/presentation/ma.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models, 2016. Pratyush Mishra, Ryan Lehmkuhl, Akshayaram Srinivasan, Wenting Zheng, and Raluca Ada Popa. Delphi: A cryp-

tographic inference service for neural networks. In 29th {USENIX} Security Symposium ({USENIX} Security 20), pp. 2505–2522, 2020.

Payman Mohassel and Peter Rindal. Aby3: A mixed protocol framework for machine learning. In Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security, pp. 35–52, New York, NY, USA, 2018. Association for Computing Machinery. ISBN 9781450356930. doi:10.1145/3243734.3243760. URL https://doi.org/10.1145/3243734.3243760.

Payman Mohassel and Yupeng Zhang. Secureml: A system for scalable privacy-preservingmachine learning. In 2017 IEEE Symposium on Security and Privacy (SP), pp. 19–38. IEEE, 2017.

Arpita Patra and Ajith Suresh. Blaze: blazing fast privacy-preserving machine learning. arXiv preprint arXiv:2005.09042, 2020.

Arpita Patra, Thomas Schneider, Ajith Suresh, and Hossein Yalame. {ABY2. 0}: Improved {Mixed-Protocol} secure

{Two-Party} computation. In 30th USENIX Security Symposium (USENIX Security 21), pp. 2165–2182, 2021. Alec Radford and Karthik Narasimhan. Improving language understanding by generative pre-training. 2018. Deevashwer Rathee, Mayank Rathee, Nishant Kumar, Nishanth Chandran, Divya Gupta, Aseem Rastogi, and Rahul

Sharma. Cryptﬂow2: Practical 2-party secure inference. New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450370899. URL https://doi.org/10.1145/3372297.3417274.

Deevashwer Rathee, Mayank Rathee, Rahul Kranti Kiran Goli, Divya Gupta, Rahul Sharma, Nishanth Chandran, and

Aseem Rastogi. Sirnn: A math library for secure rnn inference. arXiv preprint arXiv:2105.04236, 2021. Adi Shamir. How to share a secret. Communications of the ACM, 22(11):612–613, 1979. Jonathan Soifer, Jason Li, Mingqin Li, Jeffrey Zhu, Yingnan Li, Yuxiong He, Elton Zheng, Adi Oltean, Maya Mosyak,

Chris Barnes, et al. Deep learning inference service at microsoft. In 2019 USENIX Conference on Operational Machine Learning (OpML 19), pp. 15–17, 2019.

Sijun Tan, Brian Knott, Yuan Tian, and David J Wu. Cryptgpu: Fast privacy-preserving machine learning on the gpu. arXiv preprint arXiv:2104.10949, 2021.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efﬁcient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Wouter van Oortmerssen. Flatbuffers: a memory efﬁcient serialization library. Web Page. androiddevelopers. googleblog. com/2014/06/ﬂatbuffers-memory-efﬁcient. html, 2014.

Kenton Varda. Protocol buffers: Google’s data interchange format. Google Open Source Blog, Available at least as early as Jul, 72:23, 2008.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

Sameer Wagh, Divya Gupta, and Nishanth Chandran. Securenn: 3-party secure computation for neural network training. Proceedings on Privacy Enhancing Technologies, 2019(3):26–49, 2019.

Sameer Wagh, Shruti Tople, Fabrice Benhamouda, Eyal Kushilevitz, Prateek Mittal, and Tal Rabin. Falcon: Honestmajority maliciously secure framework for private deep learning. arXiv preprint arXiv:2004.02229, 2020.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=rJ4km2R5t7.

Yongqin Wang, G. Edward Suh, Wenjie Xiong, Benjamin Lefaudeux, Brian Knott, Murali Annavaram, and Hsien-Hsin S. Lee. Characterization of mpc-based private inference for transformer-based models. In 2022 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), pp. 187–197, 2022. doi:10.1109/ISPASS55109.2022.00025.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38–45, Online, October 2020. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/2020.emnlp-demos.6.

Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V. Le. Xlnet: Generalized autoregressive pretraining for language understanding. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, Red Hook, NY, USA, 2019. Curran Associates Inc.

AndrewChi-Chih Yao. How to generate and exchangesecrets. In 27th AnnualSymposiumon Foundationsof Computer Science (sfcs 1986), pp. 162–167. IEEE, 1986.

Mingchen Zhuge, Dehong Gao, Deng-Ping Fan, Linbo Jin, Ben Chen, Haoming Zhou, Minghui Qiu, and Ling Shao. Kaleido-bert: Vision-language pre-training on fashion domain. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12647–12657, 2021.

- A Details of Experimental Models

In this section, we present the architecture of the experimental models in brief. For more details, please refer to HuggingFace Transformers library (Wolf et al., 2020).

- • Bert-Base: Bert-Base is the base version of the Bert model and consists of 12 Transformer encoder layers, 768 hidden size, and 12 heads. It has 110 million parameters and is trained on a large corpus of unlabeled text data.
- • Roberta-Base: Similar to Bert-base, Roberta-base is a base version of the Roberta model. It comprises 12 Transformer layers, 768 hidden size, and 12 heads. It has around 125 million parameters.
- • Bert-Large: Bert-Large is an extended version of Bert-base with 24 Transformer encoder layers, 1024 hidden size, and 16 heads. It has approximately 340 million parameters, making it more powerful and capable of capturing complex language patterns.
- • GPT2-Base: GPT2-Base is the base version of the Gpt2 model and consists of 12 Transformer decoder layers, 768 hidden size, and 12 heads. It has 117 million parameters and is trained on a large corpus of text data. GPT2-Base is mainly used for tasks involving text generation and language understanding.
- • GPT2-Medium: GPT2-Medium comprises 24 Transformer decoder layers, 1024 hidden size, and 16 heads. And it has approximately 345 million parameters.
- • GPT2-Large: GPT2-Large is the largest variant of the GPT2 model, featuring 36 Transformer decoder layers, 1280 hidden size, and 16 heads. It has approximately 774 million parameters.

- B PUMA for LLaMA-7B

Unlike GPT-2 and Bert, LLaMA uses SiLU instead of GeLU, we can approximate SiLU using similar piece-wise low-degree polynomials with different coefﬁcients. The full polynomials could be found in flax_llama7b.py .

In Figure 2, we show the output tokens of LLamA-7B (with ﬁxed randomness) given the prompt: Q: What is the largest animal? It can be seen that our PUMA outputs the same tokens as LLaMA-7B does in plaintext for generating more than 20 tokens.

Plaintext

Prompt: Q: What is the largest animal? Outputs: A: The largest animal is the blue whale. Q: What is the smallest animal? A: The smallest animal is the bee.

PUMA

Prompt: Q: What is the largest animal? Outputs: A: The largest animal is the blue whale. Q: What is the smallest animal? A: The smallest animal is the bee.

Figure 2: Outputs of LLaMA-7B in plaintext and PUMA.

