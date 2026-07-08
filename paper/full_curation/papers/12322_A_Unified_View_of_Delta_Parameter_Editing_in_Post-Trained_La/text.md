# arXiv:2410.13841v1[cs.LG]17Oct2024

## A UNIFIED VIEW OF DELTA PARAMETER EDITING IN POST-TRAINED LARGE-SCALE MODELS

Qiaoyu Tang1,2, Le Yu3, Bowen Yu3 ∗, Hongyu Lin1∗, Keming Lu3, Yaojie Lu1, Xianpei Han1, Le Sun1

1Chinese Information Processing Laboratory, Institute of Software, Chinese Academy of Sciences 2University of Chinese Academy of Sciences 3Alibaba Group {tangqiaoyu2020,hongyu,luyaojie,xianpei,sunle}@iscas.ac.cn {chuanyi.yl,yubowen.ybw,lukeming.lkm}@alibaba-inc.com

ABSTRACT

Post-training has emerged as a crucial paradigm for adapting large-scale pretrained models to various tasks, whose effects are fully reflected by delta parameters (i.e., the disparity between post-trained and pre-trained parameters). While numerous studies have explored delta parameter properties via operations like pruning, quantization, low-rank approximation, and extrapolation, a unified framework for systematically examining these characteristics has been lacking. In this paper, we propose a novel perspective based on Riemann sum approximation of the loss function to elucidate delta parameter editing operations. Our analysis categorizes existing methods into three classes based on their post-editing performance: competitive, decreased, and improved, explaining how they are expressed by the Riemann sum approximation term and how they alter the model performance. Extensive experiments on both visual and language models, including ViT, LLaMA 3, Qwen 2, and Mistral, corroborate our theoretical findings. Furthermore, we introduce extensions to existing techniques like DARE and BitDelta, highlighting their limitations in leveraging the properties of delta parameters and reorganizing them into general expressions to enhance the applicability and effectiveness of delta parameter editing in post-trained models.

1 INTRODUCTION

With the remarkable success of large-scale pre-trained models, post-training has emerged as the de facto standard paradigm for effective adaptations to various tasks (Han et al., 2024; Xin et al., 2024; Dodge et al., 2020; Zhao et al., 2023). Conceptually, post-training optimizes the parameters of pre-trained backbone on task-specific data, endowing models with diverse abilities like visual recognition (Chen et al., 2022; Sandler et al., 2022), instruction following (Rafailov et al., 2023; Ethayarajh et al., 2024), and mathematical reasoning (Luo et al., 2023; Tong et al., 2024). It has been noted that the impact of post-training is fully manifested in the delta parameters, which are defined as the difference between parameters of pre-trained and post-trained models (Ilharco et al., 2023; Yu et al., 2024).

Due to the inherent correlations between delta parameters and post-training, significant efforts have been made to investigate the properties of delta parameters through various editing operations in recent years. For instance, studies like DARE (Yu et al., 2024) and DELLA-Merging (Deep et al., 2024) showed that models can achieve comparable performance with only a small fraction of delta parameters, highlighting their extreme redundancy. BitDelta (Liu et al., 2024) demonstrated that delta parameters could be quantized to 1 bit with modest performance compromise. Twin-Merging (Lu et al., 2024) and TIES-Merging (Yadav et al., 2023) discovered that most of the benefits of posttraining can be retained after executing singular value decomposition and magnitude-based pruning on delta parameters. EXPO (Zheng et al., 2024) observed that cheaply extrapolating delta parameters with a suitable scaling factor can even enhance the performance. However, a comprehensive

∗ Corresponding authors.

framework for systematically discussing delta parameter characteristics and theoretically explaining how different operations impact model performance remains lacking.

In this work, we make a pioneering effort to provide a unified view of delta parameter editing in post-trained large-scale models. We formulate the editing operations of delta parameters based on Riemann sum approximation of the loss difference. By mathematically analyzing existing editing operations’ loss change, we elucidate why certain operations result in competitive, decreased, or improved performance. Specifically, we verify that: 1) methods such as DARE and DELLA-Merging can well keep the approximation term to zero through the random drop and rescale processes, ensuring equal loss between the edited and post-trained models and achieving competitive performance; 2) techniques including BitDelta, Twin-Merging, and TIES-Merging often result in decreased performance, with a positive approximation term introduced by quantization, low-rank approximation, and magnitude-based pruning; 3) EXPO-like methods, by extrapolating delta parameters, produce negative loss changes on alignment data, resulting in better-aligned models. To validate our theoretical analysis, extensive experiments are conducted on large-scale visual models (ViT (Radford et al., 2021)) and language models (LLaMA 3 (Dubey et al., 2024), and Mistral (Jiang et al., 2023)), and the results strongly support our analysis.

Besides understanding existing delta parameter editing techniques in the proposed view, we further present several extensions to provide more general formats. Firstly, we introduce a factor to handle the dropped parameters in DARE, effectively expanding methods like DARE. Secondly, we extend the scope of quantification-based methods like BitDelta, identifying a broader area for effective quantification beyond reducing magnitude diversity to a single value. Finally, we identify that extrapolation is not the key to the success of EXPO-like methods. Instead, we should determine whether to use extrapolation or interpolation based on the direction of the approximation term. Experimental results also demonstrate the effectiveness of the proposed extensions.

2 RELATED WORK

- 2.1 POST-TRAINING OF LARGE-SCALE MODELS

In recent years, with the rapid development of large-scale models, post-training has become an essential process for adapting the pre-trained backbone to a variety of tasks (Xin et al., 2024; Dodge et al., 2020; Zhao et al., 2023). Post-training realizes the adaptation via adjusting the pre-trained backbone’s parameters through full fine-tuning (Dosovitskiy et al., 2021; Liu et al., 2021; Devlin et al., 2019; Radford et al., 2018) or parameter-efficient fine-tuning (He et al., 2023; Houlsby et al., 2019; Li & Liang, 2021; Hu et al., 2022; Han et al., 2024) algorithms. It is straightforward to conclude that the effectiveness of post-training can be perfectly denoted by the delta parameters, which represent the difference between post-trained and pre-trained parameters (Ilharco et al., 2023; Yu et al., 2024). Given the close correlations between delta parameters and the post-training process, investigating the properties of delta parameters becomes particularly important. In this paper, we present a novel perspective to illustrate delta parameter characteristics of post-trained models.

- 2.2 DELTA PARAMETER EDITING FOR POST-TRAINED MODELS

Existing delta parameter editing techniques can be generally categorized as three aspects according to their post-editing performance, including competitive, decreased, and improved performance.

Delta Parameter Editing with Competitive Performance. DARE (Yu et al., 2024) is a widely used approach to edit delta parameters without compromising the model performance. Technically, DARE can eliminate most (90% or even 99%) of the delta parameters with the random drop and rescale operations. Inspired by DARE, DELLA-Merging (Deep et al., 2024) presented a magnitudeaware drop to replace the random drop for achieving better performance, which ranks delta parameters by their magnitude and assigns higher dropout probabilities to those with lower ranks (i.e., corresponding to lower magnitudes). Yu et al. (2024) and Deep et al. (2024) explained that DARE and DELLA-Merging can work because they are able to approximate the original embeddings based on only a small fraction of delta parameters, thus maintaining the model performance.

Delta Parameter Editing with Decreased Performance. BitDelta (Liu et al., 2024) quantized delta parameters to only 1 bit according to the average magnitude scalar and sign bits. Twin-Merging (Lu

et al., 2024) applied singular value decomposition (Klema & Laub, 1980) on delta parameters to extract exclusive knowledge for each specific task. TIES-Merging (Yadav et al., 2023) retained delta parameters with the largest magnitudes for reducing redundancy. All the above methods yield slightly worse results after executing the corresponding quantization, low-rank approximation, or pruning operations.

Delta Parameter Editing with Improved Performance. EXPO (Zheng et al., 2024) extrapolated delta parameters calculated by two relatively weaker models with an appropriate scaling factor to construct a stronger model, which can enhance the model performance.

It can be concluded that current approaches utilizes distinct operations for editing delta parameter, lacking a comprehensive analysis of whether these editing operations are suitable and why different operations cause various influence on the model performance. In this work, we make the first attempt to introduce a unified view of delta parameter editing in post-training, which is supported both theoretically and empirically.

- 3 PRELIMINARIES

- 3.1 NOTATIONS

Delta Parameters During Post-Training. Let WPRE ∈ Rd×k denote the parameters of a pretrained model, where d and k represent the output and input dimensions. A post-trained model with

parameters WPOST ∈ Rd×k can be derived from the pre-trained backbone, yielding delta parameters ∆W = WPOST − WPRE ∈ Rd×k. As delta parameters denote the alterations of parameters during the post-training process, investigating the characteristics of delta parameters can provide a deeper understanding of post-training.

Delta Parameter Editing. Let F represent the delta parameter editing function. The edited parameters ∆ WEdit = F(∆W) is then combined with WPRE to obtain the final edited parameter WEdit = WPRE + ∆ WEdit. Existing delta parameter editing methods can be categorized into three types based on their effects on model performance, i.e., competitive, decreased, and improved performance. These methods employ various techniques including pruning, quantization, low-rank approximation, and extrapolation. Notable works in this field include DARE, BitDelta, Twin-Merging, TIES-Merging, and EXPO, which are investigated in this paper.

- 3.2 A UNIFIED VIEW OF DELTA PARAMETER EDITING

In this work, we introduce a unified view of delta parameter editing during the post-training process based on Riemann sum approximation. Specifically, we represent the changes caused by existing editing methods by ∆ W and aim to investigate their effects on performance via analyzing the loss difference. To better analyze the changes in loss, we introduce the Riemann sum approximation, which corresponds to the difference in loss made by the editing operation as follows,

1

∆L = L(WPOST + ∆ W) − L(WPOST) =

∇L(WPOST + t∆ W) · ∆ W dt

0

C−1

C−1

1 C

c C

1 C

⟨∇Lc,∆ W⟩,

≈

⟨∇L(WPOST +

∆ W),∆ W⟩ =

c=0

c=0

(1)

where L(W) : Rd×k → R denotes the loss function of a model with parameters W ∈ Rd×k, ∇L(W) is the gradient of the loss function at W, and ⟨·,·⟩ denotes the Frobenius inner product. C denotes the number of subdivisions of the interval [0,1]. This expansion provides a linear approximation of the loss function in the neighborhood of WPOST, allowing the analysis of the impact of parameter changes on the model performance. In most cases, the loss difference can reflect the influence on performance, with a positive value indicating deterioration, zero indicating stability, and a negative value indicating improvement. In section 4, section 5, and section 6, we respectively discuss editing operations that cause competitive, decreased, and improved performance, and derive the format of these operations when organizing them into the proposed unified paradigm.

To validate our theoretical analysis and the proposed extensions, we conducted experiments on LLaMA-3-8B-Instruct (Dubey et al., 2024), Mistral-7B-Instruct-v0.3 (Jiang et al., 2023), and ViTB-32 (Radford et al., 2021). We evaluate text models on 8 tasks: 25-shot ARC Challenge (Clark et al., 2018), 5-shot GSM8K (Cobbe et al., 2021), 10-shot HellaSwag (Zellers et al., 2019), zero-shot HumanEval (Chen et al., 2021), zero-shot IFEval (Zhou et al., 2023), 5-shot MMLU (Hendrycks et al., 2020), zero-shot TruthfulQA (Lin et al., 2021), and 5-shot Winogrande (Sakaguchi et al., 2021), and evaluate vision models on 8 tasks: Cars (Krause et al., 2013), DTD (Cimpoi et al., 2014), EuroSAT (Helber et al., 2019), GTSRB (Stallkamp et al., 2011), MNIST (LeCun et al., 2010), RESISC45 (Cheng et al., 2017), SUN397 (Xiao et al., 2016), and SVHN (Netzer et al., 2011).

- 4 UNIFYING EDITING OPERATIONS WITH COMPETITIVE PERFORMANCE

As a widely-used approach for delta parameter editing, DARE (Yu et al., 2024) presents the random drop and rescale process to remove 90% or even 99% delta parameters without compromising the model performance. Following this line, many follow-up works have been proposed. For example, DELLA-Merging (Deep et al., 2024) modifies the drop operation in DARE from random to magnitude-aware. In this section, we select DARE for analysis because it is the most representative method among those that can retain the original model performance after editing delta parameters.

- 4.1 EXPRESS DARE WITH APPROXIMATION TERM Mathematically, the editing process of delta parameters in DARE is denoted by

WDARE = WPOST + ∆ WDARE = WPRE + ∆W + ∆ WDARE

(2)

1 1 − p · (1 − M) ⊙ ∆W = WPRE +

1 1 − p · (1 − M) ⊙ ∆W,

= WPRE + 0 · M ⊙ ∆W +

where p ∈ R represents the drop rate and ⊙ denotes the element-wise Hadamard product. M ∼ Bernoulli(p,∆W) ∈ Rd×k is a mask matrix sampled from Bernoulli distribution according to p, whose shape is identical to that of ∆W. From Equation (2), we can derive that

p − M

1 − p ⊙ ∆W. (3) Referring to Equation (1), we obtain

∆ WDARE =

C−1

k

d

p − Mij 1 − p · ∆Wij · ∇Lcij

1 C

∆LDARE ≈

c=0

j=1

i=1

  p

 .

(4)

C−1

1 C

∆Wij · ∇Lcij −

∆Wij · ∇Lcij

1 − p ·

=

c=0

Mij=0

Mij=1

Due to the vast number of parameters in large-scale models, we can use the Law of Large Numbers to approximate the summations by their expected values. Additionally, because of the randomness of the drop operation in DARE, Mij and ∆Wij ·∇Lcij can be considered approximately independent random variables. It is straightforward to deduce that

- Mij=0

∆Wij · ∇Lcij ≈ (1 − p) ·

d

i=1

k

j=1

∆Wij · ∇Lcij,

- Mij=1

∆Wij · ∇Lcij ≈ p ·

d

k

i=1

j=1

Substituting Equation (5) into Equation (4), we derive

C−1

1 C

p 1 − p · (1 − p) − p) ·

∆LDARE ≈ (

c=0

∆Wij · ∇Lcij.

(5)

d

k

∆Wij · ∇Lcij = 0. (6)

i=1

j=1

To this end, we can conclude that after editing delta parameters with DARE, the loss L(WDARE) is approximately equal to L(WPOST), independent of the specific dataset, explaining why DARE can achieve competitive performance even when most delta parameters are eliminated.

To verify the above analysis, we used the DARE method to construct models on LLaMA3-8BInstruct and computed the approximation term on the GSM8K dataset. The results are shown in Figure 3. We use the drop-only (w/o rescale) as the reference. As can be seen, models with DARE constructed consistently achieved lower average loss, and with a smaller drop rate, the approximation term calculated across different parts of the model remained relatively small. This validates our theoretical derivation above.

- 4.2 EXTENSION OF DARE

We further present a more general format of delta parameter editing operations that can achieve competitive performance. In particular, instead of dropping delta parameters, we introduce a term k to adjust them and rescale the remaining ones with (1 − k · p)/(1 − p). Similar to the deduction in Equation (2) to Equation (6), we obtain

1 − k · p

WCOMP =WPRE + ∆W + ∆ WCOMP = WPRE + k · M ⊙ ∆W +

1 − p · (1 − M) ⊙ ∆W, ∆ WCOMP =

[Figure 1]

(k − 1)(M − p) 1 − p ⊙ ∆W,

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

 p · (1 − k)

 

[Figure 6]

[Figure 7]

C−1

1 C

∆Wij · ∇Lcij + (k − 1) ·

∆Wij · ∇Lcij

∆L ≈

1 − p ·

c=0

Mij=0

Mij=1

C−1

d

k

p · (1 − k) 1 − p · (1 − p) + (k − 1) · p ·

1 C

∆Wij · ∇Lcij = 0.

≈

c=0

i=1

j=1

|𝑘 = −1.0 𝑘 = −0.5 𝑘 = 0.0 𝑘 = 0.5 𝑘 = 0.8 Pre-train Post-train|
|---|

55

80

[Figure 8]

70

5-shotAccuracy(%)

0-shotAccuracy(%)

60 50

- 52 51

54

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

[Figure 9]

(c) Performance on HumanEval.

Pass@1(%)

40 30 20

60 50

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

[Figure 10]

- 53

40 30 20

45 44 43

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

(a) Performance on GSM8K.

(b) Performance on TruthfulQA.

- Figure 1: The performance of LLaMA3-8B-Instruct on the GSM8K, TruthfulQA, and HumanEval datasets under varying p and k.

It has been verified that ∆L is approximately 0, which indicates the validity of the proposed format. Note that in DARE, the drop operation can be realized by setting k to 0. Thus, our format is an extension of DARE with broader settings of k.

We conducted validation experiments for the extension of DARE across a wide range of models and tasks. The representative results for LLaMA3-8B-Instruct and ViT-B-32 are shown in 1 and Figure 2, while the complete results for all models and tasks are presented in the Appendix B.1. Specifically, on threee representative text datasets—GSM8K, TruthfulQA, and HumanEval, when both the rescale rate k and sign change rate kp are small (e.g., less than 0.5), the performance of our adjusted model is very close to that of the original post-trained model and significantly outperforms the pre-trained model. Regarding the weight scalar k introduced in our extension, we observed that, compared to the setting where k = 0 (which reverts to the original DARE configuration), using

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

|𝑘 = −1.0 𝑘 = −0.5 𝑘 = 0.0 𝑘 = 0.5 𝑘 = 0.8 Pre-train Post-train|
|---|

100

[Figure 18]

[Figure 19]

[Figure 20]

98

82

100

Accuracy(%)

Accuracy(%)

Accuracy(%)

96

80

53

94 92

78 76

96 46

34 32 30

46 44 42

44 42

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

(c) Performance on GTSRB.

(a) Performance on DTD.

(b) Performance on EuroSAT.

- Figure 2: The performance of ViT-B-32 on the DTD, EuroSAT, and GTSRB datasets under varying p and k.

k ̸= 0 generally yields competitive performance across different datasets. This demonstrates the effectiveness of our extension. For the ViT model, the results on the DTD, EuroSAT, and GTSRB datasets are more consistent with our expectations. Regardless of the rescale and sign change rates, the performance of the adjusted model is almost identical to that of the original post-trained model.

Interestingly, when k < 0, indicating that the delta parameters are flipped in sign, the constructed model still achieves competitive performance to the post-trained model. This challenges the prior assumption that the sign of delta parameters is critical for performance (Yadav et al., 2023; Liu et al., 2024), suggesting that what truly matters during post-training is not the specific directional adjustments of individual parameters, but rather a more collective behavior of the entire delta parameters.

- 4.3 FURTHER DISCUSSIONS ON DARE

Yu et al. (2024) and Deep et al. (2024) claim that DARE and DELLA-Merging are effective because the random drop of delta parameters ensures an approximation of the original embeddings, thereby preserving model performance. However, according to equation 5, we argue that random drop of delta parameters is a sufficient but not necessary condition for maintaining model performance. Furthermore, we contend that ensuring randomness in the element-wise product of delta parameters and approximation term is the necessary and sufficient condition.

k Random Biased ∆W Biased ∆W · ∇L

To verify the above analysis, we conduct two experiments on GSM8K dataset. First, we disrupt the randomness of the delta parameter drop operation by multiplying all negative delta parameters by k and all positive delta parameters by (1 − k · p)/(1 − p). The results are shown in the middle column of Table 1, illustrating that the model performance remains intact. This validates that the randomness of the delta parameter dropout operation is a sufficient but not necessary condition for maintaining model performance. Furthermore, we disrupt the randomness of the dropout operation on the approximation term by multiplying all negative products by k and all positive products by (1−

- 0.5 76.35 74.15 0.0

- 0.7 75.89 75.36 0.0

- 0.9 76.19 76.04 26.76
- 1.1 75.89 75.59 0.15

- 1.3 75.36 74.91 0.0

- 1.5 75.59 74.83 0.0

Table 1: Validation of the discussion on DARE. The leftmost column shows the random drop in DARE. The middle column illustrates the approach of multiplying all negative delta parameters by k and all positive delta parameters by

1−k·p

1−p . The rightmost column demonstrates the method of first calculating the product of delta parameters and gradients, and then multiplying all negative products by k and all

positive products by 11−−kp·p.

k · p)/(1 − p). The results, as depicted in the rightmost of Table 1, show a significant decline in model performance. This validates that the randomness of the dropout operation on the product of delta parameters and approximation term is a necessary and sufficient condition for maintaining model performance.

|DARE Ties Drop-only BitDelta Twin<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]|
|---|

[Figure 26]

[Figure 27]

[Figure 28]

0.1 0.3 0.5 0.7 0.9 Drop Rate 𝑝

BitDelta Twin

- Figure 3: Validation of our theoretical derivation of DARE, BitDelta, Twin-Merge(sparsity rate=0.9), and Ties-Merge.

- 5 UNIFYING EDITING OPERATIONS WITH DECREASED PERFORMANCE

This section discusses three delta parameter editing operations that incur reduced results, including quantization, low-rank approximation, and pruning. We respectively choose BitDelta (Liu et al., 2024), Twin-Merging (Lu et al., 2024), and TIES-Merging (Yadav et al., 2023) as typical works.

- 5.1 EXPRESS BITDELTA WITH APPROXIMATION TERM

BitDelta quantizes delta parameters down to 1 bit, utilizing the sign bit matrix and a high-precision scalar, where the latter is initially computed by the average magnitude of delta parameters. Specifically, BitDelta can be represented by

WBitDelta = WPOST + ∆ WBitDelta = WPRE + ∆W + ∆ WBitDelta

k

d

1 d · k

|∆Wij| · Sign(∆W) = WPRE + AVG(|∆W|) · Sign(∆W),

= WPRE+

j=1

i=1

(7)

where | · | denotes the operation of taking magnitudes. AVG(|∆W|) represents the average magnitude of ∆W. Since ∆W = |∆W| ⊙ Sign(∆W), based on Equation (7), we can further obtain

∆ WBitDelta = (AVG(|∆W|) − |∆W|) ⊙ Sign(∆W). (8) Based on Equation (1), we get

∆LBitDelta ≈

C−1

1 C

c=0

d

k

(AVG(|∆W|) − |∆Wij|) · Sign(∆Wij) · ∇Lcij. (9)

i=1

j=1

d

k

d

k

|∆Wij| = 0, it is hard

((AVG(|∆W|) − |∆Wij|) = d · k · AVG(|∆W|) −

Though

j=1

i=1

i=1

j=1

to conclude that Equation (9) equals 0 due to the multiplication of Sign(∆Wij) · ∇Lcij. Based on the approximation of the loss in Figure 3, it can be observed that the loss of BitDelta on GSM8K

is greater than 0, which is consistent with its performance degradation on GSM8K compared to post-trained model.

- 5.2 EXPRESS TWIN-MERGING AND TIES-MERGING WITH APPROXIMATION TERM

Twin-Merging employs singular value decomposition on delta parameters to derive task-specific knowledge. TIES-Merging preserves delta parameters with the highest magnitudes to minimize redundancy. Their computation processes are

WTwin = WPOST + ∆ WTwin = WPRE + ∆W + ∆ WTwin = WPRE + UrΣrVrT, WTIES = WPOST + ∆ WTIES = WPRE + ∆W + ∆ WTIES = WPRE + M ⊙ ∆W,

(10)

where rank r ≤ min(d,k) denotes the number of linearly independent columns (or rows) in ∆W = UΣV T. Ur ∈ Rd×r consists of the first r columns of U (whose columns are the left singular vectors of ∆W). Σr is the r × r diagonal matrix containing the top r singular values. Vr ∈ Rk×r includes the first r columns of V (whose columns are the right singular vectors of ∆W). M ∈ Rd×k is a binary mask matrix where an entry of 1 indicates that the corresponding delta parameter is among the top-n percent in magnitude. n is the proportion of delta parameters to be retained. According to Equation (10), we derive

∆WTwin = UrΣrVrT − ∆W, ∆ WTIES = M ⊙ ∆W − ∆W = −¬M ⊙ ∆W,

(11)

where ¬M is the element-wise NOT operation. Based on Equation (1), we get

∆LTwin ≈

1 C

C−1

c=0

d

i=1

k

j=1

(UrΣrVrTij − ∆Wij) · ∇Lcij,

∆LTIES ≈ −

1 C

C−1

c=0

d

i=1

k

j=1

¬Mij · ∆Wij · ∇Lcij.

(12)

We exploit the value of the approximation term through experiments. Models were constructed using LLaMA3-8B-Instruct, and the approximation term was calculated on the GSM8K dataset. As shown in Figure 3, for TIES-Merging, when the drop rate is relatively low, the approximation term is also lower. However, as the drop rate increases to a certain level (e.g., 0.9), the performance begins to degrade compared to DARE, which is consistent with the observations in DARE. For Twin-Merging, the approximation loss is greater than zero, which aligns with the observed performance degradation on the GSM8K dataset.

- 5.3 EXTENSION OF BITDELTA

We also extend the applicability of BitDelta by offering a more general form. Firstly, in addition to selecting the signs of delta parameters, we hypothesize that the effectiveness of BitDelta may stem from its choice of a holistic statistic that reflects the properties of the delta parameters. Specifically, BitDelta utilizes the average magnitude of delta parameters to achieve the best approximation error in the L2 norm. To validate this, we conduct an experiment where we alter the holistic statistic selected by BitDelta, introducing varying degrees of noise to the average value. As illustrated in the ”Degenerate” line of Figure 5, using the true average magnitude of the delta parameters yields nearly optimal performance on GSM8K, TruthfulQA, and HumanEval. The performance changes along the degenerate line are quite steep, and slight modifications to this average value may result in a degradation of model performance.

Secondly, instead of using a single value, we sample delta parameter magnitude matrices from both standard normal and uniform distributions, with the average magnitude serving as the mean. The experimental results, as depicted in Figure 5, demonstrate that even when these parameters are randomly sampled from distributions, the model performance remains on par with a statistic value used in BitDelta. This further underscores the significance of selecting an appropriate holistic statistic for the delta parameters.

Finally, while preserving the relative magnitude relationships of delta parameters, we enhance the effectiveness of BitDelta by employing multiple bits. Specifically, we divide the delta parameters into M blocks based on their magnitude, from smallest to largest. Each block is then represented by the average value of the delta parameters within that block. When M = 1, this approach corresponds

[Figure 29]

[Figure 30]

|LLaMA-3-8B-Instruct Mistral-7B-Instruct-v0.3|
|---|

[Figure 31]

[Figure 32]

80 76

58

5-shotAccuracy(%)

zero-shotAccuracy(%)

56 54

72

52 48

51.5

51.0

44 40

50.5

0 1 2 3 4 5 Magnitude Bits

0 1 2 3 4 5 Magnitude Bits

(a) Performance on GSM8K.

(b) Performance on TruthfulQA.

- Figure 4: Effectiveness of increasing the number of bits in BitDelta. The left subplot shows the performance of LLaMA3-8B-Instruct and Mistral-7B-Instruct-v0.3 on the GSM8K dataset as the number of bits increases. The right subplot shows the performance on the TruthfulQA dataset. In each subplot, we use the dashed line to represent the performance of the original post-trained model.

to BitDelta, and when M equals the total number of parameters in the model, it degenerates to the original post-trained model. The number of bits used is given by log2 M. As shown in Figure 5, increasing the number of bits significantly improves the model performance. When the number of bits is 4, the performance already surpasses that of the original post-trained model. This again highlights the redundancy in the delta parameters and demonstrates the potential for further advancements by expanding the bit representation in BitDelta.

6 UNIFYING EDITING OPERATIONS WITH IMPROVED PERFORMANCE

|Degenerate Normal Uniform Pre-train Post-train|
|---|

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

(a) Performance on GSM8K.

5-shotAccuracy(%)

20 10

0

60

40 30

50

0.5 1.0 1.5 2.0 2.5 3.0

[Figure 38]

(b) Performance on TruthfulQA.

0-shotAccuracy(%)

44

54 52

48 46

50

[Figure 39]

(c) Performance on HumanEval.

Pass@1(%)

40 30 20

60 50

[Figure 40]

0.5 1.0 1.5 2.0 2.5 3.0 0.5 1.0 1.5 2.0 2.5 3.0

70

10 0

Multiple of Original Average Magnitude Multiple of Original Average Magnitude Multiple of Original Average Magnitude

- Figure 5: Validation of the extension of BitDelta. The degenerate curve at 1.0 represents the original BitDelta. The full results on 8 datasets are shown in Figure 12.

EXPO (Zheng et al., 2024) is a recent method to extrapolate delta parameters, which can boost LLMs’ alignment. This section chooses EXPO as the representative approach for illustration.

- 6.1 EXPRESS EXPO WITH APPROXIMATION TERM

Technically, EXPO first computes delta parameters between an aligned model and its initial finetuning checkpoints, and then extrapolates delta parameters with a suitable scaling factor for obtaining a better-aligned model. The calculation procedure is

WEXPO = WPOST + ∆ WEXPO = WPRE + ∆W + ∆ WEXPO = WPRE + ∆W + α∆W, (13) where α controls the extrapolation length. Based on Equation (13), we derive

∆ WEXPO = α∆W. (14)

Referring to Equation (1), we obtain

C−1

α C ·

∆LEXPO ≈

c=0

d

k

∆Wij · ∇Lcij. (15)

i=1

j=1

An intuitive explanation for the improvements that EXPO achieves is that the DPO/RLHF training process of these models is suboptimal, which leads to the direction of loss reduction (the negative gradient) still aligning with the direction of the delta parameters, causing Equation (15) to be negative. Consequently, the loss of the edited model on alignment dataset is lower than that of the original posttraining model, resulting in enhanced performance on alignment benchmarks.

[Figure 41]

0.0002

0.0001

0.0000

- -0.0001
- -0.0002

Δℒ

- -0.0003
- -0.0004

We validated the aforementioned hypothesis by conducting experiments on Zephyr7B-DPO-Full (trained by EXPO). We calculated the gradient of the models using DPO loss on UltraFeedback (Cui et al., 2024). As shown in Figure 6, when α is relatively small, the value of the loss approximation term gradually decreases, reflecting that the model is indeed suboptimal. Moving further in this direction decreases the loss and improves performance accordingly. However, as α increases, the loss term gradually increases until it exceeds zero, which is consistent with the observation in EXPO that there is an optimal value for α.

0.0 0.1 0.2 0.3 0.4 0.5 0.6

Figure 6: Validation of our theoretical analysis of EXPO. we can observe that the approximation term first decreases and then increases as alpha changes, indicating that optimal performance is achieved at the trough.

- 6.2 FUTHER DISCUSSIONS ON EXPO

[Figure 42]

[Figure 43]

EXPO claims that extrapolating delta parameters leads to better models. However, based on the derivation in Equation (15), we believe that whether to use extrapolation or interpolation primarily depends on the direction of the gradient, which is influenced by the specific data. Specifically, for LLaMA3-8B-Instruct, we uniformly selected α in the range of 1.0 to 1.0 at intervals of 0.1, performing both interpolation and extrapolation of the model’s delta parameters. As show in Figure 7, on most datasets, interpolation outperformed extrapolation, except for the IFEval dataset, where extrapolation significantly improved performance. This confirms that whether to interpolate or extrapolate is not a fixed formula but depends on the specific data.We also conducted experiments on Qwen2-7B and Mistral-

|Interpolation Extrapolation|
|---|

[Figure 44]

15

10

PerformanceGap

5

0

-5

ARCChallengeGSM8KHellaSwagMMLUTruthfulQAWinograndeIFEvalHumanEval

Figure 7: Comparison of Extrapolation and Interpolation Performance on LLaMA3-8B-Instruct. The performance gap represents the difference between the model’s performance after extrapolation or interpolation and the original performance.

- 7B(shown in Section B.3), and the results indicate that even for the same task, whether extrapolation or interpolation is required can vary across different models.

- 7 CONCLUSION AND DISCUSSIONS

Post-training is a core step in the training of large models. In recent years, significant efforts have been directed towards editing the delta parameters of post-training to achieve improvements in either

performance or efficiency. However, while previous work has shown some effectiveness, the complexity of large model parameters has led to a fragmented understanding of delta parameter editing, with different studies focusing on different aspects of its effectiveness, lacking a unified perspective.

In this paper, we provide a unified perspective on the previous work related to post-training delta parameter editing using Riemann sum approximation. We find that the changes in model capability after altering the delta parameters can be analyzed through the loss differences approximated using Riemann sums. By analyzing this approximation term, we can infer the reasons why the existing delta parameter editing methods lead to maintained, improved, or reduced model performance.

Our work offers a concise, unified, and powerful explanation for many previous work in the field of post-training delta parameter editing. We validate our hypothesis through numerical experiments. From our conclusions, several potential applications emerge for future work in this direction: (1) Model Quantization: By finding an edit that sets the approximation term to zero while using lower precision, we can achieve nearly lossless compression of the model. (2) Model Enhancement: By analyzing the approximation term, we might be able to find ways to enhance the model’s capabilities without additional training data. (3) Post-training Mechanism Analysis: Since the model’s capability remains almost unchanged when the approximation term is zero, we can construct more concise post-training delta parameters. This simplifies the parameter changes during the post-training phase, enabling a more effective analysis of the parameter mechanisms in this stage.

Additionally, our work highlights a critical observation: the analysis of parameter changes during the post-training phase should not be limited to specific parameters, such as knowledge neurons, but should consider the overall distribution of parameters. This is because the key constraint of the approximation term being zero does not depend on the changes in a specific parameter during post-training but requires a comprehensive consideration of all parameter deltas. This suggests that trying to infer the impact on the global model parameters from changes in a single or a few local parameters is likely futile.

REFERENCES

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Shoufa Chen, Chongjian Ge, Zhan Tong, Jiangliu Wang, Yibing Song, Jue Wang, and Ping Luo. Adaptformer: Adapting vision transformers for scalable visual recognition. In Advances in Neural Information Processing Systems 35, 2022.

Gong Cheng, Junwei Han, and Xiaoqiang Lu. Remote sensing image scene classification: Benchmark and state of the art. Proceedings of the IEEE, 105(10):1865–1883, 2017.

Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3606–3613, 2014.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, et al. Ultrafeedback: Boosting language models with scaled ai feedback. In Forty-first International Conference on Machine Learning, 2024.

Pala Tej Deep, Rishabh Bhardwaj, and Soujanya Poria. Della-merging: Reducing interference in model merging through magnitude-based sampling. CoRR, abs/2406.11617, 2024.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 4171–4186. Association for Computational Linguistics, 2019.

Jesse Dodge, Gabriel Ilharco, Roy Schwartz, Ali Farhadi, Hannaneh Hajishirzi, and Noah A. Smith. Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping. CoRR, abs/2002.06305, 2020.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations. OpenReview.net, 2021.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aur´elien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozi`ere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Gr´egoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, and et al. The llama 3 herd of models. CoRR, abs/2407.21783, 2024.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. KTO: model alignment as prospect theoretic optimization. In International Conference on Machine Learning. PMLR, 2024.

Zeyu Han, Chao Gao, Jinyang Liu, Jeff Zhang, and Sai Qian Zhang. Parameter-efficient fine-tuning for large models: A comprehensive survey. CoRR, abs/2403.14608, 2024.

Xuehai He, Chunyuan Li, Pengchuan Zhang, Jianwei Yang, and Xin Eric Wang. Parameter-efficient model adaptation for vision transformers. In Thirty-Seventh AAAI Conference on Artificial Intelligence, pp. 817–825. AAAI Press, 2023.

Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7):2217–2226, 2019.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for NLP. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 2790–2799. PMLR, 2019.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations. OpenReview.net, 2022.

Gabriel Ilharco, Marco T´ulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In The Eleventh International Conference on Learning Representations. OpenReview.net, 2023.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L´elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timoth´ee Lacroix, and William El Sayed. Mistral 7b. CoRR, abs/2310.06825, 2023.

Virginia Klema and Alan Laub. The singular value decomposition: Its computation and some applications. IEEE Transactions on automatic control, 25(2):164–176, 1980.

Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In Proceedings of the IEEE international conference on computer vision workshops, pp. 554–561, 2013.

Yann LeCun, Corinna Cortes, and CJ Burges. Mnist handwritten digit database. ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist, 2, 2010.

Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, pp. 4582–4597. Association for Computational Linguistics, 2021.

Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. arXiv preprint arXiv:2109.07958, 2021.

James Liu, Guangxuan Xiao, Kai Li, Jason D. Lee, Song Han, Tri Dao, and Tianle Cai. Bitdelta: Your fine-tune may only be worth one bit. CoRR, abs/2402.10193, 2024.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In 2021 IEEE/CVF International Conference on Computer Vision, pp. 9992–10002. IEEE, 2021.

Zhenyi Lu, Chenghao Fan, Wei Wei, Xiaoye Qu, Dangyang Chen, and Yu Cheng. Twin-merging: Dynamic integration of modular expertise in model merging. CoRR, abs/2406.15479, 2024.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. CoRR, abs/2308.09583, 2023.

Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Baolin Wu, Andrew Y Ng, et al. Reading digits in natural images with unsupervised feature learning. In NIPS workshop on deep learning and unsupervised feature learning, volume 2011, pp. 4. Granada, 2011.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems 36, 2023.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.

Mark Sandler, Andrey Zhmoginov, Max Vladymyrov, and Andrew Jackson. Fine-tuning image transformers using learnable memory. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pp. 12145–12154. IEEE, 2022.

Johannes Stallkamp, Marc Schlipsing, Jan Salmen, and Christian Igel. The german traffic sign recognition benchmark: a multi-class classification competition. In The 2011 international joint conference on neural networks, pp. 1453–1460. IEEE, 2011.

Yuxuan Tong, Xiwen Zhang, Rui Wang, Ruidong Wu, and Junxian He. Dart-math: Difficulty-aware rejection tuning for mathematical problem-solving. CoRR, abs/2407.13690, 2024.

Jianxiong Xiao, Krista A Ehinger, James Hays, Antonio Torralba, and Aude Oliva. Sun database: Exploring a large collection of scene categories. International Journal of Computer Vision, 119: 3–22, 2016.

Yi Xin, Siqi Luo, Haodi Zhou, Junlong Du, Xiaohong Liu, Yue Fan, Qing Li, and Yuntao Du. Parameter-efficient fine-tuning for pre-trained vision models: A survey. CoRR, abs/2402.02242, 2024.

Prateek Yadav, Derek Tam, Leshem Choshen, Colin A. Raffel, and Mohit Bansal. Ties-merging: Resolving interference when merging models. In Advances in Neural Information Processing Systems 36, 2023.

Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. Language models are super mario: Absorbing abilities from homologous models as a free lunch. In International Conference on Machine Learning. PMLR, 2024.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. A survey of large language models. CoRR, abs/2303.18223, 2023.

Chujie Zheng, Ziqi Wang, Heng Ji, Minlie Huang, and Nanyun Peng. Weak-to-strong extrapolation expedites alignment. CoRR, abs/2404.16792, 2024.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models, 2023. URL https: //arxiv.org/abs/2311.07911.

- A EXPERIMENTAL DETAILS

For the loss estimation experiments, we set the constant C = 5 to calculate the approximation term. We interpolate the ∆ W parameter at values of 0.2, 0.4, 0.6, and 0.8 to generate different models. For all large language model evaluations in Chapters 4 and 5, we employ the lm-eval framework for assessment. Further details on the tested models and datasets can be found in the original paper.

- B FULL EXPERIMENTAL RESULTS

- B.1 EXTENSION OF DARE

We conduct a thorough experimental validation on the extension of DARE. The results of LLaMA38B-Instruct, Mistral-7B-Instruct-v0.3, and ViT-B-32 across eight benchmarks are presented in Figure 8, Figure 9, Figure 10, and Figure 11, respectively.

|𝑘 = −1.0 𝑘 = −0.5 𝑘 = 0.0 𝑘 = 0.5 𝑘 = 0.8 Pre-train Post-train|
|---|

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

(a) Performance on GSM8K.

5-shotAccuracy(%)

40 30 20

80

60 50

70

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

[Figure 52]

(d) Performance on ARC Challenge.

25-shotAccuracy(%)

54 52 50

62

58 56

60

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

[Figure 53]

(e) Performance on Winogrande.

5-shotAccuracy(%)

72 70 68

78

74

76

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

[Figure 54]

(f) Performance on IFEval.

0-shotAccuracy(%)

30 25 20

50

40 35

45

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

[Figure 55]

(h) Performance on HellaSwag.

10-shotAccuracy(%)

74 72 70

82

78 76

80

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

[Figure 56]

(i) Performance on MMLU.

5-shotAccuracy(%)

56 54 52

66

60 58

62

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

[Figure 57]

(b) Performance on TruthfulQA.

0-shotAccuracy(%)

45 44 43

55

52 51

54

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

[Figure 58]

(c) Performance on HumanEval.

Pass@1(%)

40 30 20

60 50

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

[Figure 59]

15 50

64

- Figure 8: The performance of LLaMA3-8B-Instruct on the all benchmarks under varying p and k.

|𝑘 = −1.0 𝑘 = −0.5 𝑘 = 0.0 𝑘 = 0.5 𝑘 = 0.8 Pre-train Post-train|
|---|

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

(a) Performance on GSM8K.

5-shotAccuracy(%)

40 38

50 48

44 42

46

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

[Figure 67]

[Figure 68]

(d) Performance on ARC Challenge.

25-shotAccuracy(%)

60.5

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

(e) Performance on Winogrande.

5-shotAccuracy(%)

70

78 76

72

74

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

[Figure 69]

(f) Performance on IFEval.

0-shotAccuracy(%)

30 25

50

40 35

45

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

[Figure 70]

(h) Performance on HellaSwag.

10-shotAccuracy(%)

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

[Figure 71]

83.2

(i) Performance on MMLU.

5-shotAccuracy(%)

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

[Figure 72]

[Figure 73]

(b) Performance on TruthfulQA.

0-shotAccuracy(%)

45.0 42.5

60.0

50.0 47.5

55.0

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

(c) Performance on HumanEval.

Pass@1(%)

27.5 25

35

30

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

[Figure 74]

59.0

52.5

57.5

22.5

32.5

37.5

40

61.0

62.0

64.0 63.5 63.0 62.5

61.5

64.5

80 84.4

84.0

84.2

83.8 83.6 83.4

84.6

59.5

62.0 61.5 61.0 60.5 60.0

82

- Figure 9: The performance of Mistral-7B-Instruct-v0.3 on the all benchmarks under varying p and k.

- B.2 EXTENSION OF BITDELTA The results of LLaMA3-8B-Instruct across eight benchmarks are presented in Figure 12.

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

|𝑘 = −1.0 𝑘 = −0.5 𝑘 = 0.0 𝑘 = 0.5 𝑘 = 0.8 Pre-train Post-train|
|---|

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

80

50 40

70

25-shotAccuracy(%)

60

5-shotAccuracy(%)

0-shotAccuracy(%)

60

Pass@1(%)

55

50

30 20 10

40 30

50 45 40

20 10 0

- 52 51

57 56

- 54

53

- 55

0

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

(c) Performance on HumanEval.

(a) Performance on GSM8K.

(d) Performance on ARC Challenge.

(b) Performance on TruthfulQA.

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

70

50

80

80

10-shotAccuracy(%)

5-shotAccuracy(%)

5-shotAccuracy(%)

0-shotAccuracy(%)

45

65

75 70

40 35

75

60

[Figure 90]

70

[Figure 91]

65 60 55

[Figure 92]

30 25 20

55

[Figure 93]

[Figure 94]

65 60

[Figure 95]

50

[Figure 96]

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

(h) Performance on HellaSwag.

(e) Performance on Winogrande.

(i) Performance on MMLU.

(f) Performance on IFEval.

### Figure 10: The performance of Qwen2-7B-Instruct on the all benchmarks under varying p and k.

|𝑘 = −1.0 𝑘 = −0.5 𝑘 = 0.0 𝑘 = 0.5 𝑘 = 0.8 Pre-train Post-train|
|---|

79

100

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

98

82

77

100

Accuracy(%)

Accuracy(%)

Accuracy(%)

Accuracy(%)

96

80

75

98

94 92

78 76

73 71

96 46

34 32 30

46 44 42

62 60 58

44 42

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

(c) Performance on GTSRB.

(a) Performance on DTD.

(d) Performance on Cars.

(b) Performance on EuroSAT.

100

78

100

100

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

98

76

98

98

96

Accuracy(%)

Accuracy(%)

Accuracy(%)

Accuracy(%)

74

96

96

94

92

72 70

94 92

94 92

50 48

[Figure 105]

[Figure 106]

66 64 62

34 32 30

62 60 58

[Figure 107]

[Figure 108]

[Figure 109]

46

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Drop Rate 𝑝

(h) Performance on SUN397.

(e) Performance on MNIST.

(i) Performance on SVHN.

(f) Performance on RESISC45.

### Figure 11: The performance of ViT-B-32 on the all benchmarks under varying p and k.

|Degenerate Normal Uniform Pre-train Post-train|
|---|

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

60 50

54 52

70

60

25-shotAccuracy(%)

5-shotAccuracy(%)

0-shotAccuracy(%)

60

55

50

Pass@1(%)

40 30 20

50

50

40 30

45

48 46

40 35

20 10

10 0

30 25

0

44

0.5 1.0 1.5 2.0 2.5 3.0 (a) Performance on GSM8K.

0.5 1.0 1.5 2.0 2.5 3.0 0.5 1.0 1.5 2.0 2.5 3.0

0.5 1.0 1.5 2.0 2.5 3.0 Multiple of Original Average Magnitude

Multiple of Original Average Magnitude

Multiple of Original Average Magnitude

Multiple of Original Average Magnitude

(c) Performance on HumanEval.

(d) Performance on ARC Challenge.

(b) Performance on TruthfulQA.

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

45

80

75

10-shotAccuracy(%)

60

5-shotAccuracy(%)

5-shotAccuracy(%)

0-shotAccuracy(%)

40 35

70

70

50

60 50 40

65 60 55

30 25 20

40

30

15 0.5 1.0 1.5 2.0 2.5 3.0 0.5 1.0 1.5 2.0 2.5 3.0 0.5 1.0 1.5 2.0 2.5 3.0 0.5 1.0 1.5 2.0 2.5 3.0

Multiple of Original Average Magnitude

Multiple of Original Average Magnitude

Multiple of Original Average Magnitude

Multiple of Original Average Magnitude

(h) Performance on HellaSwag.

(e) Performance on Winogrande.

(i) Performance on MMLU.

(f) Performance on IFEval.

### Figure 12: Validation of the extension of BitDelta on LLaMA3-8B-Instruct.

- B.3 DISCUSSION ON EXPO In Figure 13, we present the comparision of interpolation and extrapolation.

[Figure 118]

[Figure 119]

|Interpolation Extrapolation|
|---|

[Figure 120]

[Figure 121]

6

PerformanceGap

PerformanceGap

4

2

0

0

- -5

10

5

15

- -10 ARCChallengeGSM8KHellaSwagMMLUTruthfulQAWinograndeIFEvalHumanEval

-2

ARCChallengeGSM8KHellaSwagMMLUTruthfulQAWinograndeIFEvalHumanEval

(a) Mistral-7B-Instruct-v0.3. (b) Qwen2-7B-Instruct

### Figure 13: Comparison of Extrapolation and Interpolation Performance.

