#### Physics in Next-token Prediction

Hongjun An† Yiliang Song† Xuelong Li*

### arXiv:2411.00660v2[cs.LG]16Nov2024

###### Abstract

We discovered the underlying physics in Nexttoken Prediction (NTP). We identified the law of information conservation within NTP and proposed the First Law of Information Capacity (IC-1), demonstrating that the essence of intelligence emergence in auto-regressive models is fundamentally a process of information transfer. We also introduced Landauer’s Principle into NTP, formulating the Second Law of Information Capacity (IC-2), which establishes the relationship between auto-regressive model training and energy consumption. Additionally, we presented several corollaries, which hold practical significance for production practices. Finally, we demonstrate the consistency between the Law of Information Capacity and the Scaling Law for Neural Language Models, the Knowledge Capacity Scaling Laws, and the Scaling Laws for Precision.

###### 1. Introduction

Currently, the state-of-the-art (SOTA) artificial intelligence models predominantly employ an auto-regressive architecture. Utilizing the Next-token Prediction (NTP) approach, these models seamlessly integrate various modalities, including text, images, audio, and video. This remarkable versatility and intelligence are reshaping human production and lifestyle in profound ways.

However, beneath this promising landscape looms a substantial “scientific dark cloud.” Guided by the Scaling Law (Kaplan et al., 2020), researchers are relentlessly constructing ever-larger training datasets and expending increasing amounts of energy to train larger auto-regressive models in pursuit of more “intelligent” systems. Why do big data and immense computational power lead to the emergence of “intelligent”? What is the essence behind this phenomenon? Where does the Scaling Law ultimately lead? What awaits

†Equal contribution. The authors are with the Institute of Artificial Intelligence (TeleAI), China Telecom, P. R. China.

*Correspondence to: Xuelong Li <xuelong li@ieee.org>. Preprint. Copyright 2024 by the author(s).

|The 1st Law of Information Capacity<br><br>N = D(H −L)<br><br>η:<br><br>N:<br><br>D: H: L:<br>E0:<br><br><br>kB:<br><br>T:<br><br>information capacity of model parameter size (in bits) number of trained tokens<br><br>entropy of the entire dataset<br><br>average cross-entropy training loss minimum energy required to train the Boltzmann constant<br><br>the Kelvin temperature of radiator<br><br>The 2nd Law of Information Capacity<br><br>0<br><br>( ln2)<br><br>B<br><br>E =N k T<br><br>Note: Information capacity is the ratio of information content to data volume, quantifies the informationconveying capability of a unit of data.|
|---|

Figure 1. We propose the First Law of Information Capacity (IC-1) and the Second Law of Information Capacity (IC-2), reveals the principle of information conservation and the energy relationship within NTP.

us on the horizon?

In this paper, we delve deeply into these questions and uncover the underlying physics in NTP. We identify the law of information conservation (Hawking, 2014) within NTP and derive the First Law of Information Capacity (IC-1), demonstrating that the essence of intelligence emergence in auto-regressive models is fundamentally a process of information transfer. Additionally, we introduce Landauer’s Principle (Landauer, 1961) into NTP, formulating the Second Law of Information Capacity (IC-2), which reveals the relationship between the training process of auto-

regressive models and the principles of energy in the real world. Based on our findings, we derive several corollaries that can effectively guide our daily practices and production activities. Finally, we validate the compatibility and complementarity of our insights with existing theories, attesting to the rationality of our theoretical framework.

The contribution of this paper can be summarized as follows:

- • We identified the conservation law of information in NTP and proposed the IC-1.
- • We introduced the Landauer’s Principle into NTP and proposed the IC-2.
- • Based on IC-1 and IC-2, we derived several corollaries that can guide practical applications in human production.
- • we demonstrate the consistency between our findings and the Scaling Law for Neural Language Models (Kaplan et al., 2020), the Knowledge Capacity Scaling Laws (Allen-Zhu & Li, 2024), and the Scaling Laws for Precision (Kumar et al., 2024).

###### 2. The First Law of Information Capability: the Conservation of Information in NTP

###### 2.1. Preliminaries

In 2023, Rae conceptualized the process of NTP as a mechanism for compressing information of dataset (Rae, 2023), elucidating the relationship between the compression process and the emergence of intelligence: given a token vocabulary V and a dataset D = {x1,x2,...,x|D|}(xi ∈ V), our goal is to transmit D from A to B token-by-token as accurately as possible. During transmission, both parties can share an encoding and decoding function f. In this section, we will demonstrate that the intelligence level of function f is related to its ability to compress D.

2.1.1. BASELINE: NON-INTELLIGENT TRANSMISSION

|D|

|D|

|zf

−log P(xt+1|x1:t)

t | =

I(D) =

0

t=1

t=1

= |D|H(D).

- 2.1.2. INTELLIGENT TRANSMISSION BASED ON COMPRESSION

When auto-regressive models, such as large language models (LLMs), are applied to predict the next token, at each iteration, they input x1:t and predict P(xt+1|x1:t,fa). So the total cost required by this method to transmit D is at least I(D|fa) (Eq. 3).

I(D|fa) =

|D|−1

t=0

−log P(xt+1|x1:t,fa). (3)

It is noteworthy that the Eq. 3 is in exact correspondence with the objective loss function, cross-entropy loss, which is optimised during the training phase (Eq. 4).

ℓ(fa) =

1 |D|

|D|−1

t=0

−log P(xt+1|x1:t,fa)

=

1 |D|

I(D|fa).

(4)

Thus, the model training process, characterized by the reduction of the loss function, can be regarded as a compression process of the dataset D. The higher the compression, the more intelligent of the model becomes.

- 2.2. A Derivation of the First Law of Information Capability

(2)

If we conduct a meticulous examination of Eq. 2 and Eq. 4, we may uncover an even more intriguing insight. That is, when fa is sufficiently powerful, I(D|fa) is absolutely likely to be smaller than I(D). Does this imply that some information disappears into thin air (Eq. 5)?

Assuming we have transmitted Dt = {x1:t}(t < |D|) and are about to transmit xt+1, for a non-intelligent f0, according to information theory (Shannon, 1948), the length of the code zf

t+1 = f0(xt+1|x1:t) is at least −log P(xt+1|x1:t) (Eq. 1), which is its self-information.

0

|zf

t+1| = I(xt+1|x1:t) = −log P(xt+1|x1:t), (1)

0

where the initial condition is P(x1|x0) = P(x1). Thus, the total cost required by this method is as indicated by I(D) (Eq. 2).

I(D) − I(D|fa) = ? (5)

In fact, from the perspective of physics, information is conserved (Hawking, 2014), these pieces of information do not vanish; rather, they are transferred into the model fa (Eq. 6).

I(fa+) = I(D) − I(D|fa), (6)

where I(fa+) represents the effective information stored in fa pertaining to task D. We introduce the concept of

Information Capacity (Li & Zhao, 2021), denoted by η, defined as the ratio of effective information I(fa+) to its data size N (approximately equal to the number of parameter size, measured in bits). Thus, we obtain Eq. 7

I(fa+) N

. (7)

η =

After performing equivalent transformations to the Eq. 7, we arrive at the First Law of Information Capacity (IC-1), as shown in Eq. 8.

ηN = D(H − L). (8)

where L = ℓ(fa) denotes the average cross-entropy loss, D = |D| denotes the number of tokens in dataset D, and H = H(D) denotes the entropy of the entire dataset.

Therefore, the process of model training is essentially a process of compressing dataset D, leading to a reduction in the loss function, the transfer of information to model fa, and an increase in its information capacity η. This transfer is driven by the model training process, particularly the back propagation algorithm (Rumelhart et al., 1986). The energy for this transfer comes from electrical resources in the real world (Sec. 3).

2.3. A Dynamic Perspective to the Process of Intelligence Emergence

Currently, Eq. 8 remains static. In particular, within our derivation, D represents the total number of tokens in the dataset, H denotes the overall entropy of the dataset, and L is the overall average loss after training has concluded. From the perspective of the law of conservation of information, it is imperative that conservation be observed not only at the terminal state, but throughout the entirety of the dynamic training process. Therefore, we will restate the meanings of the variables in Eq. 8:

- • H: the overall entropy of the dataset, is a constant.
- • N: parameter size of the model, measured in bits. Once the model architecture is established, it will become a fixed constant.
- • D: the token numbers that has been trained. This is a variable that monotonically increases as training progresses.
- • η,L: η is the information capacity of the model, and L is the dynamic average cross-entropy loss. Both change dynamically as D increases.

It is therefore possible to describe the entirety of the model training process from a dynamic perspective based on IC-1:

Initial State. Training has not yet begun, with no information transfer, thus η = 0.

Training State. As training progresses, D gradually increases, leading to a decrease in L. To satisfy the equation, η must inevitably increase. During this dynamic process, the information gradually transfers to the fa, prompting the model to learn.

Terminal State. When η = ηmax (determined by the model architecture), the information that the model parameters can store reaches saturation. At this point, continuing the training process does not enable the model to learn more tokens; L converges, and training concludes.

###### 3. The Second Law of Information Capacity: the Energy Relationship in NTP

In Sec. 2.2, the IC-1 indicates that the learning process in auto-regressive models is fundamentally an information transfer process. The driving force behind this transfer comes from the back-propagation (Rumelhart et al., 1986) algorithm, while the energy is sourced from the power of the physical world. One might wonder, what is the minimum amount of energy required to complete this information transfer process?

In 1961, Landauer proposed that the energy required to erase a single bit is at least kBT ln2, known as the Landauer’s Principle (Landauer, 1961). Therefore, according to Eq. 8, when we transfer information I(fa+), at least energy E0 = I(fa+)kBT ln2 must be consumed. Thus, the training process of auto-regressive models establishes an energy relationship with the physical world. We can derive the Second Law of Information Capacity (IC-2) (Eq. 9).

E0 = ηN(kBT ln2), (9)

where E0 is the minimum energy required to complete the information transfer, kB ≈ 1.38 × 10−23J/K is the Boltzmann constant, and T is the temperature of the heat reservoir in Kelvin.

###### 4. Corollaries to Guide Practice 4.1. Estimating the Entropy of the Dataset

From Sec. 2.3, it can be inferred that when the model training process is in its initial state, η ≈ 0. From this, we can derive the following corollary: the Data Entropy Estimation Theorem (Corollary 4.1).

Corollary 4.1. The entropy of the dataset is approximately equal to the initial loss of the model training. Proof.

ηN = D(H − L),D > 0,η ≈ 0,

(10)

⇒ H ≈ L.

It is crucial to acknowledge that this corollary is exclusively applicable for estimation purposes. This is due to the fact that the model itself incorporates initial information following its initialisation, which ultimately leads to the emergence of η > 0.

###### 4.2. Estimating the Quality of the Dataset

When the number of tokens in the dataset is fixed, a higher entropy of the dataset indicates that it can provide more information for the model to learn. Therefore, we can consider entropy H as a metric for evaluating dataset quality, with the assessment method outlined in Corollary 4.1.

###### 4.3. Matching Suitable Model Size with Dataset Size

In the practice of large model production, determining the appropriate model size, the amount of training data required, and the duration of training is a critical issue. In previous work, the Knowledge Capacity Scaling Laws indicated that current large language models (LLMs) generally store only 2 bits of information per parameter (Allen-Zhu & Li, 2024). For models using the FP16/BF16 or INT8 formats to store parameters, the information capacity η is approximately 0.125 ∼ 0.25. Additionally, according to the technical reports of well-known large models (Kaplan et al., 2020), it has been inferred that the entropy of datasets typically ranges from 10 to 15 approximately. Thus, if N and L are given, the required dataset size D can be estimated; if D and L are given, the required model size can be estimated.

Sec. 5.1 indicates that the IC-1 is compatible with the Scaling Law proposed by the OpenAI team in 2020 (Kaplan et al., 2020). Their advantage lies in providing power-law relationships between L and N, L and D, as well as L and C (computation cost) respectively through extensive experiments. However, these power-law relationships are validated only for large models based on the Transformer architecture. From the IC-1, we cannot derive the relationships between L and N or D respectively; we can only calculate their corresponding values from a macro perspective. Nevertheless, the IC-1 is applicable to all auto-regressive models, regardless of architecture, because its underlying principle is based on the physical principle of information conservation. Therefore, we are compatible and complementary with the OpenAI’s Scaling Law.

###### 4.4. Identify the Energy Limits Required for Training Auto-regressive Models

As chip technology advances and autore-gressive model learning algorithms are optimized, the energy required to

train models achieving similar intelligence levels is expected to decline. Additionally, future developments may lead to a shift from GPUs to quantum computers, which could further reduce energy consumption. Nevertheless, regardless of technological advancement, there is an inherent limitation to the amount of energy that can be consumed for the transmission of information, known as Landauer’s Limit, which is represented by IC-2 (Eq. 9).

- 4.5. The conditions for lossless quantization

Quantizing the weights of autoregressive models to achieve economic goals is a common practice. However, achieving lossless quantization requires certain conditions. Suppose the model weights before quantization are b-bit with information capacity η, and after quantization, they are b′-bit with information capacity η′. To achieve lossless quantization, the following conditions must be met Eq. 11:

ηb ≤ η′b′. (11)

Since η′ ≤ 1, we can derive the necessary condition for lossless quantization as Eq. 12:

ηb ≤ b′. (12)

- 5. Consistency with Existing Theories

5.1. Consistency with the Scaling Law of Neural Language Models

In this section, we will substitute the experimental data from (Kaplan et al., 2020) into the IC-1 to verify the consistency between the IC-1 and the Scaling Law of Neural Language Models. According to Sec. 2.3, we can know that the H is a constant.

When L is fixed, it indicates that model training has stopped. At this point, for a given set of N and D—that is, for a specific model trained on a defined number of tokens—η becomes a constant. Consequently, (H − L)/η is also a constant, resulting in N being proportional to D.

H − L η

N = kD,k =

;

⇒ N ∝ D.

(13)

When D is fixed, it implies that the same number of tokens is being trained. If the model structure is altered to increase N, the change in the information capacity η cannot be directly determined. Consequently, the relationship between L and N cannot be directly established (Eq. 14).

- 0

10

- 1

10

-1

10

-2

10

-3

10

-4

10

-5

10

-6

10

-7

10

-8

10

- 10
- 11

10 10

10 12

10 13

10 14

10 15

010

110

210

310

410

510

10 6

10 7

10 8

10 9

Figure 2. The plotted curve of f(x) = x0.095 − x0.076. When 0 < x ≪ 1015, 0 < f(x) ≪ 10.

When N is fixed, it implies that the model is deterministic. If D is increased, meaning that the model is trained on a larger number of tokens, η will also increase correspondingly. However, the relationship between the change rate of the two cannot be directly determined, and thus the relationship between L and D also cannot be established directly (Eq. 14).

nearly equal, with the difference in exponents attributed to observational errors.

5.4 × 1013 D ≈

8.8 × 1013

.

1 16N

Thus, we can deduce that:

N D

L = H − η

. (14)

In summary, to verify whether the IC-1 aligns with the empirical formula measured by the OpenAI team in 2020, we can approach the analysis from the perspective of fixed L. Here is the empirical formula (Eq. 15 and Eq. 16):

###### N ≈ 26.08D.

This is consistent with the IC-1. Furthermore, when η = 0, meaning the model is untrained, we have L0 = H ≈ 10 (as visually inferred from Figure 2 in the paper (Kaplan et al., 2020)); when the model converges, L ∈ [2,7]. Thus, we obtain:

L =

D 5.4 × 1013

−0.095

, (15)

L =

1 16N

8.8 × 1013

−0.076

. (16)

The factor of 161 in the Eq. 16 arises because the parameter count in the original paper refers to the quantity of FP/BF16

numbers, whereas N in this paper is expressed in bits. By setting (15)=(16), we obtain:

5.4 × 1013 D

0.095

=

8.8 × 1013

1 16N

0.076

.

By constructing the function f(x) = x0.095 − x0.076 and plotting its curve (Fig. 2), it becomes evident that when 0 < x ≪ 1015, 0 < f(x) ≪ 10. Therefore, it can be inferred that the bases on both sides of the equation are

H − L k ∈

η =

10 − 7 26.08

,

10 − 3 26.08

= [0.115,0.268].

The information capacity η falls within a normal range (< 1), thus indicating that the IC-1 is consistent with OpenAI’s empirical formula.

###### 5.2. Consistency with the Knowledge Capacity Scaling Laws

As noted in Sec. 4.3, the information capacity of LLMs in the Knowledge Capacity Scaling Laws should generally be less than 0.125 ∼ 0.25 (Allen-Zhu & Li, 2024), which is consistent with the value of [0.115,0.268] in Sec. 5.1.

###### 5.3. Consistency with the Scaling Laws for Precision

(Kumar et al., 2024) found that as the model is trained on more data, the degradation caused by post-training quantization increases, ultimately making additional pre-training

data detrimental. This is, in fact, an inevitable consequence of the First Law of Information Capacity. As the model is trained on more data, the information capacity η of the model will gradually increase. When condition 12 is violated, quantization of the model will damage the information learned by the model, leading to negative effects.

###### 6. Conclusion

This paper has revealed the fundamental physical principles that underpin NTP by establishing the IC-1 and the IC-2. These laws elucidate not only the conservation of information and the energy requirements in auto-regressive model training, but also offer practical corollaries that can guide the development and training of intelligent systems. By aligning our findings with existing theories, we have demonstrated the broad applicability and significance of our theoretical framework, thereby paving the way for more efficient and sustainable advancements in artificial intelligence.

###### References

Allen-Zhu, Z. and Li, Y. Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws, 2024.

Hawking, S. W. Information preservation and weather forecasting for black holes, 2014.

Kaplan, J., McCandlish, S., Henighan, T., et al. Scaling Laws for Neural Language Models, 2020.

Kumar, T., Ankner, Z., Spector, B. F., et al. Scaling Laws for Precision, 2024.

Landauer, R. Irreversibility and heat generation in the computing process. IBM Journal of Research and Development, 5(3):183–191, 1961.

Li, X. and Zhao, B. Video distillation. SCIENTIA SINICA Informationis, 51(5):695–734, 2021.

Rae, J. Compression for AGI, 2023. URL https://www. youtube.com/watch?v=dO4TPJkeaaU. Accessed: 2024-10-23.

Rumelhart, D. E., Hinton, G. E., and Williams, R. J. Learning representations by back-propagating errors. Nature, 323:533–536, 1986.

Shannon, C. E. A Mathematical Theory of Communication. The Bell System Technical Journal, 27(3):379–423, 1948.

