## TinyFusion: Diffusion Transformers Learned Shallow

# arXiv:2412.01199v1[cs.CV]2Dec2024

Gongfan Fang*, Kunjun Li*, Xinyin Ma, Xinchao Wang† National University of Singapore

{gongfan, kunjun, maxinyin}@u.nus.edu, xinchao@nus.edu.sg

#### Abstract

𝐦𝐢𝐧𝖒,𝚫𝚽 𝓛(𝒙, 𝚽 + 𝚫𝚽,𝖒)

[Figure 1]

[Figure 2]

[Figure 3]

𝚫𝚽

[Figure 4]

| | | | |
|---|---|---|---|
|[Figure 5]<br><br>[Figure 6]<br><br>𝝓𝟒 T| |[Figure 7]<br><br>ransform|er Layer|
| | | | |
|𝝓𝟑|T|[Figure 8]<br><br>[Figure 9]<br><br>ransform|er Layer|
| | | | |
|[Figure 10]<br><br>[Figure 11]<br><br>𝝓𝟐 T| |ransform|er Layer|
| | | | |
|𝝓𝟏 T| |ransform|[Figure 12]<br><br>[Figure 13]<br><br>er Layer|
| | | | |

(LoRA/Full)

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Transformer Layer

Transformer Layer

0

Diffusion Transformers have demonstrated remarkable capabilities in image generation but often come with excessive parameterization, resulting in considerable inference overhead in real-world applications. In this work, we present TinyFusion, a depth pruning method designed to remove redundant layers from diffusion transformers via endto-end learning. The core principle of our approach is to create a pruned model with high recoverability, allowing it to regain strong performance after fine-tuning. To accomplish this, we introduce a differentiable sampling technique to make pruning learnable, paired with a co-optimized parameter to simulate future fine-tuning. While prior works focus on minimizing loss or error after pruning, our method explicitly models and optimizes the post-fine-tuning performance of pruned models. Experimental results indicate that this learnable paradigm offers substantial benefits for layer pruning of diffusion transformers, surpassing existing importance-based and error-based methods. Additionally, TinyFusion exhibits strong generalization across diverse architectures, such as DiTs, MARs, and SiTs. Experiments with DiT-XL show that TinyFusion can craft a shallow diffusion transformer at less than 7% of the pretraining cost, achieving a 2× speedup with an FID score of 2.86, outperforming competitors with comparable efficiency. Code is available at https://github.com/ VainF/TinyFusion

[Figure 26]

[Figure 27]

[Figure 28]

Joint

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Transformer Layer

[Figure 36]

Opt.

- 0

[Figure 37]

[Figure 38]

- 1

Transformer Layer

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Transformer Layer

Transformer Layer

1

|Differentiable Sampling of Layer Mask|𝖒|Recoverability Estimation with 𝚫𝚽|
|---|---|---|

Local Block

Figure 1. This work presents a learnable approach for pruning the depth of pre-trained diffusion transformers. Our method simultaneously optimizes a differentiable sampling process of layer masks and a weight update to identify a highly recoverable solution, ensuring that the pruned model maintains competitive performance after fine-tuning.

siderable inference costs due to the huge parameter scale, which poses significant challenges for deployment. To resolve this problem, there has been growing interest from both the research community and industry in developing lightweight models [12, 23, 32, 58].

The efficiency of diffusion models is typically influenced by various factors, including the number of sampling steps [33, 43, 45, 46], operator design [7, 48, 52], computational precision [19, 30, 44], network width [3, 12] and depth [6, 23, 36]. In this work, we focus on model compression through depth pruning [36, 54], which removes entire layers from the network to reduce the latency. Depth pruning offers a significant advantage in practice: it can achieve a linear acceleration ratio relative to the compression rate on both parallel and non-parallel devices. For example, as will be demonstrated in this work, while 50% width pruning [12] only yields a 1.6× speedup, pruning 50% of the layers results in a 2× speedup. This makes depth pruning a flexible and practical method for model compression.

#### 1. Introduction

Diffusion Transformers have emerged as a cornerstone architecture for generative tasks, achieving notable success in areas such as image [11, 26, 40] and video synthesis [25, 59]. This success has also led to the widespread availability of high-quality pre-trained models on the Internet, greatly accelerating and supporting the development of various downstream applications [5, 16, 53, 55]. However, pre-trained diffusion transformers usually come with con-

This work follows a standard depth pruning framework: unimportant layers are first removed, and the pruned model is then fine-tuned for performance recovery. In the literature, depth pruning techniques designed for diffusion transformers or general transformers primarily focus on heuristic approaches, such as carefully designed importance scores [6, 36] or manually configured pruning

*Equal contribution †Corresponding author

schemes [23, 54]. These methods adhere to a loss minimization principle [18, 37], aiming to identify solutions that maintain low loss or error after pruning. This paper investigates the effectiveness of this widely used principle in the context of depth compression. Through experiments, we examined the relationship between calibration loss observed post-pruning and the performance after fine-tuning. This is achieved by extensively sampling 100,000 models via random pruning, exhibiting different levels of calibration loss in the searching space. Based on this, we analyzed the effectiveness of existing pruning algorithms, such as the feature similarity [6, 36] and sensitivity analysis [18], which indeed achieve low calibration losses in the solution space. However, the performance of all these models after finetuning often falls short of expectations. This indicates that the loss minimization principle may not be well-suited for diffusion transformers.

Building on these insights, we reassessed the underlying principles for effective layer pruning in diffusion transformers. Fine-tuning diffusion transformers is an extremely time-consuming process. Instead of searching for a model that minimizes loss immediately after pruning, we propose identifying candidate models with strong recoverability, enabling superior post-fine-tuning performance. Achieving this goal is particularly challenging, as it requires the integration of two distinct processes, pruning and fine-tuning, which involve non-differentiable operations and cannot be directly optimized via gradient descent.

To this end, we propose a learnable depth pruning method that effectively integrates pruning and fine-tuning. As shown in Figure 1, we model the pruning and finetuning of a diffusion transformer as a differentiable sampling process of layer masks [13, 17, 22], combined with a co-optimized weight update to simulate future fine-tuning. Our objective is to iteratively refine this distribution so that networks with higher recoverability are more likely to be sampled. This is achieved through a straightforward strategy: if a sampled pruning decision results in strong recoverability, similar pruning patterns will have an increased probability of being sampled. This approach promotes the exploration of potentially valuable solutions while disregarding less effective ones. Additionally, the proposed method is highly efficient, and we demonstrate that a suitable solution can emerge within a few training steps.

To evaluate the effectiveness of the proposed method, we conduct extensive experiments on various transformerbased diffusion models, including DiTs [40], MARs [29], SiTs [34]. The learnable approach is highly efficient. It is able to identify redundant layers in diffusion transformers with 1-epoch training on the dataset, which effectively crafts shallow diffusion transformers from pre-trained models with high recoverability. For instance, while the models pruned by TinyFusion initially exhibit relatively high cal-

ibration loss after removing 50% of layers, they recover quickly through fine-tuning, achieving a significantly more competitive FID score (5.73 vs. 22.28) compared to baseline methods that only minimize immediate loss, using just 1% of the pre-training cost. Additionally, we also explore the role of knowledge distillation in enhancing recoverability [20, 23] by introducing a MaskedKD variant. MaskedKD mitigates the negative impact of the massive or outlier activations [47] in hidden states, which can significantly affect the performance and reliability of fine-tuning. With MaskedKD, the FID score improves from 5.73 to 3.73 with only 1% of pre-training cost. Extending the training to 7% of the pre-training cost further reduces the FID to 2.86, just 0.4 higher than the original model with doubled depth.

Therefore, the main contribution of this work lies in a learnable method to craft shallow diffusion transformers from pre-trained ones, which explicitly optimizes the recoverability of pruned models. The method is general for various architectures, including DiTs, MARs and SiTs.

#### 2. Related Works

Network Pruning and Depth Reduction. Network pruning is a widely used approach for compressing pre-trained diffusion models by eliminating redundant parameters [3, 12, 31, 51]. Diff-Pruning [12] introduces a gradientbased technique to streamline the width of UNet, followed by a simple fine-tuning to recover the performance. SparseDM [51] applies sparsity to pre-trained diffusion models via the Straight-Through Estimator (STE) [2], achieving a 50% reduction in MACs with only a 1.22 increase in FID on average. While width pruning and sparsity help reduce memory overhead, they often offer limited speed improvements, especially on parallel devices like GPUs. Consequently, depth reduction has gained significant attention in the past few years, as removing entire layers enables better speedup proportional to the pruning ratio [24, 27, 28, 36, 54, 56, 58]. Adaptive depth reduction techniques, such as MoD [41] and depth-aware transformers [10], have also been proposed. Despite these advances, most existing methods are still based on empirical or heuristic strategies, such as carefully designed importance criteria [36, 54], sensitivity analyses [18] or manually designed schemes [23], which often do not yield strong performance guarantee after fine-tuning.

Efficient Diffusion Transformers. Developing efficient diffusion transformers has become an appealing focus within the community, where significant efforts have been made to enhance efficiency from various perspectives, including linear attention mechanisms [15, 48, 52], compact architectures [50], non-autoregressive transformers [4, 14, 38, 49], pruning [12, 23], quantization [19, 30, 44], feature

|𝐦𝐢𝐧𝖒,𝚫𝚽 𝓛(𝒙,𝚽 + 𝚫𝚽,𝖒)|
|---|

Learnable Distribution

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

1:2 Local Blocks

[Figure 67]

[Figure 68]

[Figure 69]

|Δ𝜙4 ⋅ 𝔪4|
|---|

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Weight Update

[Figure 74]

[Figure 75]

- 𝔪1

- 𝔪2

- 𝔪3

- 𝔪4

𝝓𝟒

- 0

[Figure 76]

[Figure 77]

- 1 0

Transformer Layer

Diff. Sampling

[Figure 78]

⊕

[Figure 79]

[Figure 80]

|Δ𝜙3 ⋅ 𝔪3|
|---|

[Figure 81]

[Figure 82]

|∼|
|---|

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

Weight Update

[Figure 88]

[Figure 89]

[Figure 90]

𝝓𝟑

1

Transformer Layer

Retained Layer

###### Mixed Sampling ⇒ Exploration still in Progress

[Figure 91]

1:2 Local Blocks

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

|Δ𝜙2 ⋅ 𝔪2|
|---|

[Figure 99]

[Figure 100]

[Figure 101]

𝝓𝟐

[Figure 102]

Weight Update

Transformer Layer

- 0

[Figure 103]

[Figure 104]

- 1 0

Diff. Sampling

[Figure 105]

Retained Layer

[Figure 106]

⊕

|∼|
|---|

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

|Δ𝜙1 ⋅ 𝔪1|
|---|

𝝓𝟏

[Figure 117]

Weight Update

1

Transformer Layer

Confident Sampling ⇒ Good solution identified

Figure 2. The proposed TinyFusion method learns to perform a differentiable sampling of candidate solutions, jointly optimized with a weight update to estimate recoverability. This approach aims to increase the likelihood of favorable solutions that ensure strong post-finetuning performance. After training, local structures with the highest sampling probabilities are retained.

caching [35, 57], etc. In this work, we focus on compressing the depth of pre-trained diffusion transformers and introduce a learnable method that directly optimizes recoverability, which is able to achieve satisfactory results with low re-training costs.

where ∆Φ = {∆ϕ1,∆ϕ2,··· ,∆ϕM} represents appropriate update from fine-tuning. The objective formulated by Equation 2 poses two challenges: 1) The non-differentiable nature of layer selection prevents direct optimization using gradient descent; 2) The inner optimization over the retained layers makes it computationally intractable to explore the entire search space, as this process necessitates selecting a candidate model and fine-tuning it for evaluation. To address this, we propose TinyFusion that makes both the pruning and recoverability optimizable.

#### 3. Method

##### 3.1. Shallow Generative Transformers by Pruning

This work aims to derive a shallow diffusion transformer by pruning a pre-trained model. For simplicity, all vectors in this paper are column vectors. Consider a L-layer transformer, parameterized by ΦL×D = [ϕ1,ϕ2,··· ,ϕL]⊺, where each element ϕi encompasses all learnable parameters of a transformer layer as a D-dim column vector, which includes the weights of both attention layers and MLPs. Depth pruning seeks to find a binary layer mask mL×1 = [m1,m2,··· ,mL]⊺, that removes a layer by:

##### 3.2. TinyFusion: Learnable Depth Pruning

A Probabilistic Perspective. This work models Equation 2 from a probabilistic standpoint. We hypothesize that the mask m produced by “ideal” pruning methods (might be not unique) should follow a certain distribution. To model this, it is intuitive to associate every possible mask m with a probability value p(m), thus forming a categorical distribution. Without any prior knowledge, the assessment of pruning masks begins with a uniform distribution. However, directly sampling from this initial distribution is highly inefficient due to the vast search space. For instance, pruning a 28-layer model by 50% involves evaluating 2814 = 40,116,600 possible solutions. To overcome this challenge, this work introduces an advanced and learnable algorithm capable of using evaluation results as feedback to iteratively refine the mask distribution. The basic idea is that if certain masks exhibit positive results, then other masks with similar pattern may also be potential solutions and thus should have a higher likelihood of sampling in subsequent evaluations, allowing for a more focused search on promising solutions. However, the definition of “similarity pattern” is still unclear so far.

ϕi(xi), if mi = 1, xi, otherwise,

xi+1 = miϕi(xi) + (1 − mi)xi =

(1) where the xi and ϕi(xi) refers to the input and output of layer ϕi. To obtain the mask, a common paradigm in prior work is to minimize the loss L after pruning, which can be formulated as minm Ex [L(x,Φ,m)]. However, as we will show in the experiments, this objective – though widely adopted in discriminative tasks – may not be well-suited to pruning diffusion transformers. Instead, we are more interested in the recoverability of pruned models. To achieve this, we incorporate an additional weight update into the optimization problem and extend the objective by:

, (2)

Ex [L(x,Φ + ∆Φ,m)]

min

min

m

∆Φ

Recoverability: Post-Fine-Tuning Performance

Sampling Local Structures. In this work, we demonstrate that local structures, as illustrated in Figure 2, can serve as effective anchors for modeling the relationships between different masks. If a pruning mask leads to certain local structures and yields competitive results after finetuning, then other masks yielding the same local patterns are also likely to be positive solutions. This can be achieved by dividing the original model into K non-overlapping blocks, represented as Φ = [Φ1,Φ2,··· ,ΦK]⊺. For simplicity, we assume each block Φk = [ϕk1,ϕk2,··· ,ϕkM]⊺ contains exactly M layers, although they can have varied lengths. Instead of performing global layer pruning, we propose an N:M scheme for local layer pruning, where, for each block Φk with M layers, N layers are retained. This results in a set of local binary masks m = [m1,m2,...,mK]⊺. Similarly, the distribution of a local mask mk is modeled using a categorical distribution p(mk). We perform independent sampling of local binary masks and combine them for pruning, which presents the joint distribution:

[Figure 118]

[Figure 119]

𝑥𝑖+1 𝑚𝑖 ⨂ + (1−𝑚𝑖) ⨂

[Figure 120]

| |[Figure 121]|[Figure 122]<br><br>Pretrained 𝑊|
|---|---|---|
|𝑁 ×| | |
| | | |

|[Figure 123]<br><br>[Figure 124]<br><br>Identity f(x)=x|
|---|

B

r

[Figure 125]

A

| |𝑥𝑖|[Figure 126]|
|---|---|---|
| | | |
| | | |

Figure 3. An example of forward propagation with differentiable pruning mask mi and LoRA for recoverability estimation.

Notably, when τ → 0, the STE gradients will approximate the true gradients, yet with a higher variance which is negative for training [22]. Thus, a scheduler is typically employed to initiate training with a high temperature, gradually reducing it over time.

Joint Optimization with Recoverability. With differentiable sampling, we are able to update the underlying probability using gradient descent. The training objective in this work is to maximize the recoverability of sampled masks. We reformulate the objective in Equation 2 by incorporating the learnable distribution:

p(m) = p(m1) · p(m2)···p(mK) (3)

If some local distributions p(mk) exhibit high confidence in the corresponding blocks, the system will tend to sample those positive patterns frequently and keep active explorations in other local blocks. Based on this concept, we introduce differential sampling to make the above process learnable.

, (6)

Ex,{m

k∼p(mk)} [L(x,Φ + ∆Φ,{mk}]

min

min

∆Φ

{p(mk)}

Recoverability: Post-Fine-Tuning Performance

where {p(mk)} = {p(m1),··· ,p(mK)} refer to the categorical distributions for different local blocks. Based on this formulation, we further investigate how to incorporate the fine-tuning information into the training. We propose a joint optimization of the distribution and a weight update ∆Φ. Our key idea is to introduce a co-optimized update ∆Φ for joint training. A straightforward way to craft the update is to directly optimize the original network. However, the parameter scale in a diffusion transformer is usually huge, and a full optimization may make the training process costly and not that efficient. To this end, we show that ParameterEfficient Fine-Tuning methods such as LoRA [21] can be a good choice to obtain the required ∆Φ. For a single linear matrix W in Φ, we simulate the fine-tuned weights as:

Differentiable Sampling. Considering the sampling process of a local mask mk, which corresponds a local block Φk and is modeled by a categorical distribution p(mk). With the N:M scheme, there are MN possible masks. We construct a special matrix mˆ N:M to enumerate all possible masks. For example, 2:3 layer pruning will lead to the candidate matrix mˆ 2:3 = [[1,1,0],[1,0,1],[0,1,1]]. In this case, each block will have three probabilities p(mk) = [pk1,pk2,pk3]. For simplicity, we omit mk and k and use pi to represent the probability of sampling i-th element in mˆ N:M. A popular method to make a sampling process differentiable is Gumbel-Softmax [13, 17, 22]:

exp((gi + log pi)/τ) j exp((gj + log pj)/τ)

. (4)

y = one-hot

###### Wfine-tuned = W + α∆W = W + αBA, (7)

where α is a scalar hyperparameter that scales the contribution of ∆W. Using LoRA significantly reduces the number of parameters, facilitating efficient exploration of different pruning decisions. As shown in Figure 3, we leverage the sampled binary mask value mi as the gate and forward the network with Equation 1, which suppresses the layer outputs if the sampled mask is 0 for the current layer. In addition, the previously mentioned STE will still provide non-zero gradients to the pruned layer, allowing it to be further updated. This is helpful in practice, since some layers

where gi is random noise drawn from the Gumbel distribution Gumbel(0,1) and τ refers to the temperature term. The output y is the index of the sampled mask. Here a StraightThrough Estimator [2] is applied to the one-hot operation, where the onehot operation is enabled during forward and is treated as an identity function during backward. Leveraging the one-hot index y and the candidate set mˆ N:M, we can draw a mask m ∼ p(m) through a simple index operation:

m = y⊺mˆ (5)

###### Method Depth #Param Iters IS ↑ FID ↓ sFID ↓ Prec. ↑ Recall ↑ Sampling it/s ↑

DiT-XL/2 [40] 28 675 M 7,000 K 278.24 2.27 4.60 0.83 0.57 6.91 DiT-XL/2 [40] 28 675 M 2,000 K 240.22 2.73 4.46 0.83 0.55 6.91 DiT-XL/2 [40] 28 675 M 1,000 K 157.83 5.53 4.60 0.80 0.53 6.91 U-ViT-H/2 [1] 29 501 M 500 K 265.30 2.30 5.60 0.82 0.58 8.21 ShortGPT [36] 28⇒19 459 M 100 K 132.79 7.93 5.25 0.76 0.53 10.07 TinyDiT-D19 (KD) 28⇒19 459 M 100 K 242.29 2.90 4.63 0.84 0.54 10.07 TinyDiT-D19 (KD) 28⇒19 459 M 500 K 251.02 2.55 4.57 0.83 0.55 10.07

DiT-L/2 [40] 24 458 M 1,000 K 196.26 3.73 4.62 0.82 0.54 9.73 U-ViT-L [1] 21 287 M 300 K 221.29 3.44 6.58 0.83 0.52 13.48 U-DiT-L [50] 22 204 M 400 K 246.03 3.37 4.49 0.86 0.50 Diff-Pruning-50% [12] 28 338 M 100 K 186.02 3.85 4.92 0.82 0.54 10.43 Diff-Pruning-75% [12] 28 169 M 100 K 83.78 14.58 6.28 0.72 0.53 13.59 ShortGPT [36] 28⇒14 340 M 100 K 66.10 22.28 6.20 0.63 0.56 13.54 Flux-Lite [6] 28⇒14 340 M 100 K 54.54 25.92 5.98 0.62 0.55 13.54 Sensitivity Analysis [18] 28⇒14 340 M 100 K 70.36 21.15 6.22 0.63 0.57 13.54 Oracle (BK-SDM) [23] 28⇒14 340 M 100 K 141.18 7.43 6.09 0.75 0.55 13.54 TinyDiT-D14 28⇒14 340 M 100 K 151.88 5.73 4.91 0.80 0.55 13.54 TinyDiT-D14 28⇒14 340 M 500 K 198.85 3.92 5.69 0.78 0.58 13.54 TinyDiT-D14 (KD) 28⇒14 340 M 100 K 207.27 3.73 5.04 0.81 0.54 13.54 TinyDiT-D14 (KD) 28⇒14 340 M 500 K 234.50 2.86 4.75 0.82 0.55 13.54

DiT-B/2 [40] 12 130 M 1,000 K 119.63 10.12 5.39 0.73 0.55 28.30 U-DiT-B [50] 22 - 400 K 85.15 16.64 6.33 0.64 0.63 TinyDiT-D7 (KD) 14⇒7 173 M 500 K 166.91 5.87 5.43 0.78 0.53 26.81

- Table 1. Layer pruning results for pre-trained DiT-XL/2. We focus on two settings: fast training with 100K optimization steps and sufficient fine-tuning with 500K steps. Both fine-tuning and Masked Knowledge Distillation (a variant of KD, see Sec. 4.4) are used for recovery.

might not be competitive at the beginning, but may emerge as competitive candidates with sufficient fine-tuning.

Pruning Decision. After training, we retain those local structures with the highest probability and discard the additional update ∆Φ. Then, standard fine-tuning techniques can be applied for recovery.

#### 4. Experiments

##### 4.1. Experimental Settings

Our experiments were mainly conducted on Diffusion Transformers [40] for class-conditional image generation on ImageNet 256 × 256 [8]. For evaluation, we follow [9, 40] and report the Fr´echet inception distance (FID), Sliding Fr´echet Inception Distance (sFID), Inception Scores (IS), Precision and Recall using the official reference images [9]. Additionally, we also extend our methods to other models, including MARs [29] and SiTs [34]. Experimental details can be found in the following sections and appendix.

##### 4.2. Results on Diffusion Transformers

DiT. This work focuses on the compression of DiTs [40]. We consider two primary strategies as baselines: the first

Depth Pruning Width Pruning Linear Speedup

12

11.60

| |
|---|

10

SpeedUp

8

6.45

| |
|---|

6

4.46

| |
|---|

| |
|---|

4.39

4

3.41

| |
|---|

2.76

| |
|---|

3.36

2.30

| |
|---|

| |
|---|

2.71

1.08 1.17 1.27 1.40 1.55 1.74 1.99

| |
|---|

2

| |
|---|

2.20

1.04 1.26 1.36 1.64 1.91

| | |
|---|---|
| | |

| |
|---|

| |
|---|

| |
|---|

0 20 40 60 80

Compression Ratio (%)

Figure 4. Depth pruning closely aligns with the theoretical linear speed-up relative to the compression ratio.

involves using manually crafted patterns to eliminate layers. For instance, BK-SDM [23] employs heuristic assumptions to determine the significance of specific layers, such as the initial or final layers. The second strategy is based on systematically designed criteria to evaluate layer importance, such as analyzing the similarity between block inputs and outputs to determine redundancy [6, 36]; this approach typically aims to minimize performance degradation after pruning. Table 1 presents representatives from both strategies, including ShortGPT [36], Flux-Lite [6], DiffPruning [12], Sensitivity Analysis [18] and BK-SDM [23], which serve as baselines for comparison. Additionally,

Method Depth Params Epochs FID IS

MAR-Large 32 479 M 400 1.78 296.0 MAR-Base 24 208 M 400 2.31 281.7 TinyMAR-D16 32⇒16 277 M 40 2.28 283.4

SiT-XL/2 28 675 M 1,400 2.06 277.5 TinySiT-D14 28⇒14 340 M 100 3.02 220.1

- Table 2. Depth pruning results on MARs [29] and SiTs [34].

we evaluate our method against innovative architectural designs, such as UViT [1], U-DiT [50], and DTR [39], which have demonstrated improved training efficiency over conventional DiTs.

Table 1 presents our findings on compressing a pretrained DiT-XL/2 [40]. This model contains 28 transformer layers structured with alternating Attention and MLP layers. The proposed method seeks to identify shallow transformers with {7,14,19} sub-layers from these 28 layers, to maximize the post-fine-tuning performance. With only 7% of the original training cost (500K steps compared to 7M steps), TinyDiT achieves competitive performance relative to both pruning-based methods and novel architectures. For instance, a DiT-L model trained from scratch for 1M steps achieves an FID score of 3.73 with 458M parameters. In contrast, the compressed TinyDiT-D14 model, with only 340M parameters and a faster sampling speed (13.54 it/s vs. 9.73 it/s), yields a significantly improved FID of 2.86. On parallel devices like GPUs, the primary bottleneck in transformers arises from sequential operations within each layer, which becomes more pronounced as the number of layers increases. Depth pruning mitigates this bottleneck by removing entire transformer layers, thereby reducing computational depth and optimizing the workload. By comparison, width pruning only reduces the number of neurons within each layer, limiting its speed-up potential. As shown in Figure 4, depth pruning closely matches the theoretical linear speed-up as the compression ratio increases, outperforming width pruning methods such as Diff-Pruning [12].

MAR & SiT. Masked Autoregressive (MAR) [29] models employ a diffusion loss-based autoregressive framework in a continuous-valued space, achieving high-quality image generation without the need for discrete tokenization. The MAR-Large model, with 32 transformer blocks, serves as the baseline for comparison. Applying our pruning method, we reduced MAR to a 16-block variant, TinyMAR-D16, achieving an FID of 2.28 and surpassing the performance of the 24-block MAR-Base model with only 10% of the original training cost (40 epochs vs. 400 epochs). Our approach also generalizes to Scalable Interpolant Transformers (SiT) [34], an extension of the DiT architecture that employs a flow-based interpolant framework to bridge data

2.00

Min:0.195

Std:1.300

Max:37.694

1.75

1.50

Density

1.25

1.00

0.75

0.50

###### ShortGPT Learnable

0.25

0.00

100 101

Calibration Loss

Sensitivity Flux-Lite

Oracle

Figure 5. Distribution of calibration loss through random sampling of candidate models. The proposed learnable method achieves the best post-fine-tuning FID yet has a relatively high initial loss compared to other baselines.

Strategy Loss IS FID Prec. Recall

Max. Loss 37.69 NaN NaN NaN NaN Med. Loss 0.99 149.51 6.45 0.78 0.53 Min. Loss 0.20 73.10 20.69 0.63 0.58 Sensitivity 0.21 70.36 21.15 0.63 0.57 ShortGPT [36] 0.20 66.10 22.28 0.63 0.56 Flux-Lite [6] 0.85 54.54 25.92 0.62 0.55 Oracle (BK-SDM) 1.28 141.18 7.43 0.75 0.55 Learnable 0.98 151.88 5.73 0.80 0.55

Table 3. Directly minimizing the calibration loss may lead to non-optimal solutions. All pruned models are fine-tuned without knowledge distillation (KD) for 100K steps. We evaluate the following baselines: (1) Loss – We randomly prune a DiT-XL model to generate 100,000 models and select models with different calibration losses for fine-tuning; (2) Metric-based Methods – such as Sensitivity Analysis and ShortGPT; (3) Oracle – We retain the first and last layers while uniformly pruning the intermediate layers following [23]; (4) Learnable – The proposed learnable method.

and noise distributions. The SiT-XL/2 model, comprising 28 transformer blocks, was pruned by 50%, creating the TinySiT-D14 model. This pruned model retains competitive performance at only 7% of the original training cost (100 epochs vs. 1400 epochs). As shown in Table 2, these results demonstrate that our pruning method is adaptable across different diffusion transformer variants, effectively reducing the model size and training time while maintaining strong performance.

##### 4.3. Analytical Experiments

Is Calibration Loss the Primary Determinant? An essential question in depth pruning is how to identify redundant layers in pre-trained diffusion transformers. A common approach involves minimizing the calibration loss, based on the assumption that a model with lower calibration loss after pruning will exhibit superior performance. However, we demonstrate in this section that this hypothesis may not hold for diffusion transformers. We begin by examining the solution space through random depth pruning at a 50% ratio, generating 100,000 candidate models with

###### Pattern ∆W IS ↑ FID ↓ sFID ↓ Prec. ↑ Recall ↑

- 1:2 LoRA 54.75 33.39 29.56 0.56 0.62

- 2:4 LoRA 53.07 34.21 27.61 0.55 0.63 7:14 LoRA 34.97 49.41 28.48 0.46 0.56

- 1:2 Full 53.11 35.77 32.68 0.54 0.61

- 2:4 Full 53.63 34.41 29.93 0.55 0.62 7:14 Full 45.03 38.76 31.31 0.52 0.62

- 1:2 Frozen 45.08 39.56 31.13 0.52 0.60

- 2:4 Frozen 48.09 37.82 31.91 0.53 0.62 7:14 Frozen 34.09 49.75 31.06 0.46 0.56

Table 4. Performance comparison of TinyDiT-D14 models compressed using various pruning schemes and recoverability estimation strategies. All models are fine-tuned for 10,000 steps, and FID scores are computed on 10,000 sampled images with 64 timesteps.

calibration losses ranging from 0.195 to 37.694 (see Figure 5). From these candidates, we select models with the highest and lowest calibration losses for fine-tuning. Notably, both models result in unfavorable outcomes, such as unstable training (NaN) or suboptimal FID scores (20.69), as shown in Table 3. Additionally, we conduct a sensitivity analysis [18], a commonly used technique to identify crucial layers by measuring loss disturbance upon layer removal, which produces a model with a low calibration loss of 0.21. However, this model’s FID score is similar to that of the model with the lowest calibration loss. Approaches like ShortGPT [36] and a recent approach for compressing the Flux model [6], which estimate similarity or minimize mean squared error (MSE) between input and output states, reveal a similar trend. In contrast, methods with moderate calibration losses, such as Oracle (often considered less competitive) and one of the randomly pruned models, achieve FID scores of 7.43 and 6.45, respectively, demonstrating significantly better performance than models with minimal calibration loss. These findings suggest that, while calibration loss may influence post-fine-tuning performance to some extent, it is not the primary determinant for diffusion transformers. Instead, the model’s capacity for performance recovery during fine-tuning, termed “recoverability,” appears to be more critical. Notably, assessing recoverability using traditional metrics is challenging, as it requires a learning process across the entire dataset. This observation also explains why the proposed method achieves superior results (5.73) compared to baseline methods.

Learnable Modeling of Recoverability. To overcome the limitations of traditional metric-based methods, this study introduces a learnable approach to jointly optimize pruning and model recoverability. Table 3 illustrates different configurations of the learnable method, including the local pruning scheme and update strategies for recoverability estimation. For a 28-layer DiT-XL/2 with a fixed 50%

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 13

- 14

- 15

- 16

- 17

- 18

- 19

- 20

- 21

- 22

- 23

- 24

- 25

- 26

- 27

LayerIndexinDiT-XL

0 2000 4000 6000 8000 10000

Train iterations

Figure 6. Visualization of the 2:4 decisions in the learnable pruning, with the confidence level of each decision highlighted through varying degrees of transparency. More visualization results for 1:2 and 7:14 schemes are available in the appendix.

layer pruning rate, we examine three splitting schemes: 1:2, 2:4, and 7:14. In the 1:2 scheme, for example, every two transformer layers form a local block, with one layer pruned. Larger blocks introduce greater diversity but significantly expand the search space. For instance, the 7:14 scheme divides the model into two segments, each retaining 7 layers, resulting in 147 × 2 = 6,864 possible solutions. Conversely, smaller blocks significantly reduce optimization difficulty and offer greater flexibility. When the distribution of one block converges, the learning on other blocks can still progress. As shown in Table 3, the 1:2 configuration achieves the optimal performance after 10K finetuning iterations. Additionally, our empirical findings underscore the effectiveness of recoverability estimation using LoRA or full fine-tuning. Both methods yield positive postfine-tuning outcomes, with LoRA achieving superior results (FID = 33.39) compared to full fine-tuning (FID = 35.77) under the 1:2 scheme, as LoRA has fewer trainable parameters (0.9% relative to full parameter training) and can adapt more efficiently to the randomness of sampling.

Visualization of Learnable Decisions. To gain deeper insights into the role of the learnable method in pruning, we visualize the learning process in Figure 6. From bottom to top, the i-th curve represents the i-th layer of the pruned model, displaying its layer index in the original DiT-XL/2. This visualization illustrates the dynamics of pruning decisions over training iterations, where the transparency of each data point indicates the probability of being sampled. The learnable method shows its capacity to explore and handle various layer combinations. Pruning decisions for certain layers, such as the 7-th and 8-th in the compressed model, are determined quickly and remain stable throughout the process. In contrast, other layers, like the 0-th layer, require additional fine-tuning to estimate their recoverability. Notably, some decisions may change in the later stages

|[Figure 127]|
|---|
| |
|[Figure 128]|
|[Figure 129]|

|[Figure 130]|
|---|
| |
|[Figure 131]|
|[Figure 132]|

|[Figure 133]|
|---|
| |
|[Figure 134]|
|[Figure 135]|

|[Figure 136]|[Figure 137]|
|---|---|
| | |
|[Figure 138]|[Figure 139]|
|[Figure 140]|[Figure 141]|

|[Figure 142]|
|---|
| |
|[Figure 143]|
|[Figure 144]|

|[Figure 145]|[Figure 146]|
|---|---|
| | |
|[Figure 147]|[Figure 148]|
|[Figure 149]|[Figure 150]|

|[Figure 151]|
|---|
| |
|[Figure 152]|
|[Figure 153]|

|[Figure 154]|
|---|
| |
|[Figure 155]|
| |
|[Figure 156]|

Figure 7. Images generated by TinyDiT-D14 on ImageNet 224×224, pruned and distilled from a DiT-XL/2.

| | | | | | | |
|---|---|---|---|---|---|---|
| | |-429.01| | |191.20| |
| | |Activation:| | |Activation:| |
| | |Min| | |Max| |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | |-526.62| |53.77| |
| | |Activation:| |Activation:| |
| | |Min| |Max| |

fine-tuning Strategy Init. Distill. Loss FID @ 100K

0.5

0.4

MinActivation:-429.01

MaxActivation:191.20

fine-tuning - 5.79 Logits KD - 4.66

0.4

MinActivation:-526.62

MaxActivation:53.77

0.3

Density

Density

0.3

+Std:12.54

-Std:-12.54

+Std:14.15

-Std:-14.15

0.2

0.2

RepKD 2840.1 NaN Masked KD (0.1σ) 15.4 NaN Masked KD (2σ) 387.1 3.73 Masked KD (4σ) 391.4 3.75

0.1

0.1

0.0

0.0

102 101 100 0 100 101 102

102 101 100 0 100 101

Activation Value (log)

Activation Value (log)

(a) DiT-XL/2 (Teacher)

(b) TinyDiT-D14 (Student)

Figure 8. Visualization of massive activations [47] in DiTs. Both teacher and student models display large activation values in their hidden states. Directly distilling these massive activations may result in excessively large losses and unstable training.

Table 5. Evaluation of different fine-tuning strategies for recovery. Masked RepKD ignores those massive activations (|x| > kσx) in both teacher and student, which enables effective knowledge transfer between diffusion transformers.

once these layers have been sufficiently optimized. The training process ultimately concludes with high sampling probabilities, suggesting a converged learning process with distributions approaching a one-hot configuration. After training, we select the layers with the highest probabilities for subsequent fine-tuning.

##### 4.4. Knowledge Distillation for Recovery

In this work, we also explore Knowledge Distillation (KD) as an enhanced fine-tuning method. As demonstrated in Table 5, we apply the vanilla knowledge distillation approach proposed by Hinton [20] to fine-tune a TinyDiT-D14, using the outputs of the pre-trained DiT-XL/2 as a teacher model for supervision. We employ a Mean Square Error (MSE) loss to align the outputs between the shallow student model and the deeper teacher model, which effectively reduces the FID at 100K steps from 5.79 to 4.66.

Masked Knowledge Distillation. Additionally, we evaluate representation distillation (RepKD) [23, 42] to transfer hidden states from the teacher to the student. It is important to note that depth pruning does not alter the hidden dimension of diffusion transformers, allowing for direct alignment

of intermediate hidden states. For practical implementation, we use the block defined in Section 3.2 as the basic unit, ensuring that the pruned local structure in the pruned DiT aligns with the output of the original structure in the teacher model. However, we encountered significant training difficulties with this straightforward RepKD approach due to massive activations in the hidden states, where both teacher and student models occasionally exhibit large activation values, as shown in Figure 8. Directly distilling these extreme activations can result in excessively high loss values, impairing the performance of the student model. This issue has also been observed in other transformer-based generative models, such as certain LLMs [47]. To address this, we propose a Masked RepKD variant that selectively excludes these massive activations during knowledge transfer. We employ a simple thresholding method, |x − µx| < kσx, which ignores the loss associated with these extreme activations. As shown in Table 5, the Masked RepKD approach with moderate thresholds of 2σ and 4σ achieves satisfactory results, demonstrating the robustness of our method.

Generated Images. In Figure 7, We visualize the generated images of the learned TinyDiT-D14, distilled from an

off-the-shelf DiT-XL/2 model. More visualization results for SiTs and MARs can be found in the appendix.

#### 5. Conclusions

This work introduces TinyFusion, a learnable method for accelerating diffusion transformers by removing redundant layers. It models the recoverability of pruned models as an optimizable objective and incorporates differentiable sampling for end-to-end training. Our method generalizes to various architectures like DiTs, MARs and SiTs.

#### References

- [1] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22669–22679, 2023.
- [2] Yoshua Bengio, Nicholas L´eonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.
- [3] Thibault Castells, Hyoung-Kyu Song, Bo-Kyeong Kim, and Shinkook Choi. Ld-pruner: Efficient pruning of latent diffusion models using task-agnostic insights. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 821–830, 2024.
- [4] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Conference on Computer Vision and Pattern Recognition, pages 11315–11325, 2022.
- [5] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis, 2023.
- [6] Javier Mart´ın Daniel Verd´u. Flux.1 lite: Distilling flux1.dev for efficient text-to-image generation. 2024.
- [7] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022.
- [8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.
- [9] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.
- [10] Maha Elbayad, Jiatao Gu, Edouard Grave, and Michael Auli. Depth-adaptive transformer. arXiv preprint arXiv:1910.10073, 2019.
- [11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis.

- In Forty-first International Conference on Machine Learning, 2024.
- [12] Gongfan Fang, Xinyin Ma, and Xinchao Wang. Structural pruning for diffusion models. In Advances in Neural Information Processing Systems, 2023.
- [13] Gongfan Fang, Hongxu Yin, Saurav Muralidharan, Greg Heinrich, Jeff Pool, Jan Kautz, Pavlo Molchanov, and Xinchao Wang. Maskllm: Learnable semi-structured sparsity for large language models. arXiv preprint arXiv:2409.17481, 2024.
- [14] Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, and Junshi Huang. Scaling diffusion transformers to 16 billion parameters. arXiv preprint arXiv:2407.11633, 2024.
- [15] Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, Youqiang Zhang, and Junshi Huang. Dimba: Transformermamba diffusion models. arXiv preprint arXiv:2406.01159, 2024.
- [16] Shanghua Gao, Zhijie Lin, Xingyu Xie, Pan Zhou, MingMing Cheng, and Shuicheng Yan. Editanything: Empowering unparalleled flexibility in image editing and generation. In Proceedings of the 31st ACM International Conference on Multimedia, Demo track, 2023.
- [17] Emil Julius Gumbel. Statistical theory of extreme values and some practical applications: a series of lectures. US Government Printing Office, 1954.
- [18] Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. Advances in neural information processing systems, 28, 2015.
- [19] Yefei He, Luping Liu, Jing Liu, Weijia Wu, Hong Zhou, and Bohan Zhuang. Ptqd: Accurate post-training quantization for diffusion models. Advances in Neural Information Processing Systems, 36, 2024.
- [20] Geoffrey Hinton, Oriol Vinyals, Jeff Dean, et al. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2(7), 2015.
- [21] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.
- [22] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144, 2016.
- [23] Bo-Kyeong Kim, Hyoung-Kyu Song, Thibault Castells, and Shinkook Choi. Bk-sdm: Architecturally compressed stable diffusion for efficient text-to-image generation. In Workshop on Efficient Systems for Foundation Models@ ICML2023, 2023.
- [24] Bo-Kyeong Kim, Geonmin Kim, Tae-Ho Kim, Thibault Castells, Shinkook Choi, Junho Shin, and Hyoung-Kyu Song. Shortened llama: A simple depth pruning for large language models. arXiv preprint arXiv:2402.02834, 11, 2024.
- [25] PKU-Yuan Lab and Tuzhan AI etc. Open-sora-plan, 2024.
- [26] Black Forest Labs. FLUX, 2024.
- [27] Youngwan Lee, Yong-Ju Lee, and Sung Ju Hwang. Ditpruner: Pruning diffusion transformer models for text-toimage synthesis using human preference scores.

- [28] Youngwan Lee, Kwanyong Park, Yoorhim Cho, Yong-Ju Lee, and Sung Ju Hwang. Koala: self-attention matters in knowledge distillation of latent diffusion models for memory-efficient and fast image synthesis. arXiv e-prints, pages arXiv–2312, 2023.
- [29] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. arXiv preprint arXiv:2406.11838, 2024.
- [30] Xiuyu Li, Yijiang Liu, Long Lian, Huanrui Yang, Zhen Dong, Daniel Kang, Shanghang Zhang, and Kurt Keutzer. Q-diffusion: Quantizing diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17535–17545, 2023.
- [31] Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snapfusion: Text-to-image diffusion model on mobile devices within two seconds. Advances in Neural Information Processing Systems, 36, 2024.
- [32] Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxllightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024.
- [33] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787, 2022.
- [34] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740, 2024.
- [35] Xinyin Ma, Gongfan Fang, Michael Bi Mi, and Xinchao Wang. Learning-to-cache: Accelerating diffusion transformer via layer caching, 2024.
- [36] Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng Chen. Shortgpt: Layers in large language models are more redundant than you expect. arXiv preprint arXiv:2403.03853, 2024.
- [37] Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila, and Jan Kautz. Pruning convolutional neural networks for resource efficient inference. arXiv preprint arXiv:1611.06440, 2016.
- [38] Zanlin Ni, Yulin Wang, Renping Zhou, Jiayi Guo, Jinyi Hu, Zhiyuan Liu, Shiji Song, Yuan Yao, and Gao Huang. Revisiting non-autoregressive transformers for efficient image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7007– 7016, 2024.
- [39] Byeongjun Park, Sangmin Woo, Hyojun Go, Jin-Young Kim, and Changick Kim. Denoising task routing for diffusion models. arXiv preprint arXiv:2310.07138, 2023.
- [40] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.

- [41] David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, and Adam Santoro. Mixture-of-depths: Dynamically allocating compute in transformer-based language models. arXiv preprint arXiv:2404.02258, 2024.
- [42] Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua Bengio. Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550, 2014.
- [43] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.
- [44] Yuzhang Shang, Zhihang Yuan, Bin Xie, Bingzhe Wu, and Yan Yan. Post-training quantization on diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1972–1981, 2023.
- [45] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.
- [46] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023.
- [47] Mingjie Sun, Xinlei Chen, J Zico Kolter, and Zhuang Liu. Massive activations in large language models. arXiv preprint arXiv:2402.17762, 2024.
- [48] Yao Teng, Yue Wu, Han Shi, Xuefei Ning, Guohao Dai, Yu Wang, Zhenguo Li, and Xihui Liu. Dim: Diffusion mamba for efficient high-resolution image synthesis. arXiv preprint arXiv:2405.14224, 2024.
- [49] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. 2024.
- [50] Yuchuan Tian, Zhijun Tu, Hanting Chen, Jie Hu, Chao Xu, and Yunhe Wang. U-dits: Downsample tokens in u-shaped diffusion transformers. arXiv preprint arXiv:2405.02730, 2024.
- [51] Kafeng Wang, Jianfei Chen, He Li, Zhenpeng Mi, and Jun Zhu. Sparsedm: Toward sparse efficient diffusion models. arXiv preprint arXiv:2404.10445, 2024.
- [52] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Yujun Lin, Zhekai Zhang, Muyang Li, Yao Lu, and Song Han. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024.
- [53] Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and MingHsuan Yang. Diffusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 56(4): 1–39, 2023.
- [54] Fang Yu, Kun Huang, Meng Wang, Yuan Cheng, Wei Chu, and Li Cui. Width & depth pruning for vision transformers. In Conference on Artificial Intelligence (AAAI), 2022.
- [55] Tao Yu, Runseng Feng, Ruoyu Feng, Jinming Liu, Xin Jin, Wenjun Zeng, and Zhibo Chen. Inpaint anything: Segment anything meets image inpainting. arXiv preprint arXiv:2304.06790, 2023.
- [56] Dingkun Zhang, Sijia Li, Chen Chen, Qingsong Xie, and Haonan Lu. Laptop-diff: Layer pruning and normalized dis-

- tillation for compressing diffusion models. arXiv preprint arXiv:2404.11098, 2024.
- [57] Xuanlei Zhao, Xiaolong Jin, Kai Wang, and Yang You. Real-time video generation with pyramid attention broadcast. arXiv preprint arXiv:2408.12588, 2024.
- [58] Yang Zhao, Yanwu Xu, Zhisheng Xiao, and Tingbo Hou. Mobilediffusion: Subsecond text-to-image generation on mobile devices. arXiv preprint arXiv:2311.16567, 2023.
- [59] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, 2024.

## TinyFusion: Diffusion Transformers Learned Shallow Supplementary Material

#### 6. Experimental Details

Models. Our experiments evaluate the effectiveness of three models: DiT-XL, MAR-Large, and SiT-XL. Diffusion Transformers (DiTs), inspired by Vision Transformer (ViT) principles, process spatial inputs as sequences of patches. The DiT-XL model features 28 transformer layers, a hidden size of 1152, 16 attention heads, and a 2 × 2 patch size. It employs adaptive layer normalization (AdaLN) to improve training stability, comprising 675 million parameters and trained for 1400 epochs. Masked Autoregressive models (MARs) are diffusion transformer variants tailored for autoregressive image generation. They utilize a continuousvalued diffusion loss framework to generate high-quality outputs without discrete tokenization. The MAR-Large model includes 32 transformer layers, a hidden size of 1024, 16 attention heads, and bidirectional attention. Like DiT, it incorporates AdaLN for stable training and effective token modeling, with 479 million parameters trained over 400 epochs. Finally, Scalable Interpolant Transformers (SiTs) extend the DiT framework by introducing a flow-based interpolant methodology, enabling more flexible bridging between data and noise distributions. While architecturally identical to DiT-XL, the SiT-XL model leverages this interpolant approach to facilitate modular experimentation with interpolant selection and sampling dynamics.

Datasets. We prepared the ImageNet 256 × 256 dataset by applying center cropping and adaptive resizing to maintain the original aspect ratio and minimize distortion. The images were then normalized to a mean of 0.5 and a standard deviation of 0.5. To augment the dataset, we applied random horizontal flipping with a probability of 0.5. To accelerate training without using Variational Autoencoder (VAE), we pre-extracted features from the images using a pre-trained VAE. The images were mapped to their latent representations, normalized, and the resulting feature arrays were saved for direct use during training.

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 13

- 14

- 15

- 16

- 17

- 18

- 19

- 20

- 21

- 22

- 23

- 24

- 25

- 26

- 27

LayerIndexinDiT-XL

0 2000 4000 6000 8000 10000

Train iterations

Figure 9. 1:2 Pruning Decisions

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 13

- 14

- 15

- 16

- 17

- 18

- 19

- 20

- 21

- 22

- 23

- 24

- 25

- 26

- 27

LayerIndexinDiT-XL

0 2000 4000 6000 8000 10000

Train iterations

Figure 10. 2:4 Pruning Decisions

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 13

- 14

- 15

- 16

- 17

- 18

- 19

- 20

- 21

- 22

- 23

- 24

- 25

- 26

- 27

LayerIndexinDiT-XL

0 2000 4000 6000 8000 10000

Train iterations

Figure 11. 7:14 Pruning Decisions

Training Details The training process began with obtaining pruned models using the proposed learnable pruning method as illustrated in Figure 12. Pruning decisions were made by a joint optimization of pruning and weight updates through LoRA with a block size. In practice, the block size is 2 for simplicity and the models were trained for 100 epochs, except for MAR, which was trained for 40 epochs. To enhance post-pruning performance, the Masked Knowledge Distillation (RepKD) method was employed during the recovery phase to transfer knowledge from teacher mod-

els to pruned student models. The RepKD approach aligns the output predictions and intermediate hidden states of the pruned and teacher models, with further details provided in the following section. Additionally, as Exponential Moving Averages (EMA) are updated and used during image generation, an excessively small learning rate can weaken EMA’s effect, leading to suboptimal outcomes. To address this, a progressive learning rate scheduler was implemented to gradually halve the learning rate throughout training. The

[Figure 157]

[Figure 158]

[Figure 159]

Update Categorical Distribution

[Figure 160]

| |[Figure 161]<br><br>[Figure 162]<br><br>𝐦𝐢𝐧 𝓛(𝚽 + 𝚫𝚽)<br><br>[Figure 163]| | | |
|---|---|---|---|---|
|Joint<br><br>Opt.<br><br>[Figure 164]<br><br>[Figure 165]<br><br>𝐒𝐡𝐚𝐫𝐞𝐝 𝚫𝚽 (LoRA/Full)<br><br>Update<br><br>[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]<br><br>| | | | |
| |[Figure 170]<br><br>[Figure 171]<br><br>Transfo<br><br>[Figure 172]<br><br>[Figure 173]<br><br>[Figure 174]<br><br>Transform<br><br>[Figure 175]<br><br>ransformer<br><br>ransformer| |rm<br><br>e<br><br>L<br><br>L|er Layer<br><br>r Layer<br><br>ayer<br><br>ayer|
|TransfoTrans<br><br>[Figure 176]<br><br>[Figure 177]<br><br>| |[Figure 178]<br><br>[Figure 179]<br><br>rmerformerLayerLayer<br><br>[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]<br><br>[Figure 184]<br><br>[Figure 185]<br><br>| | |

[Figure 186]

|𝚽|
|---|

[Figure 187]

[Figure 188]

[Figure 189]

| | |
|---|---|
|Transform|[Figure 190]<br><br>[Figure 191]<br><br>er Layer|
| | |
|Transform|[Figure 192]<br><br>[Figure 193]<br><br>er Layer|
| | |
|Transform|[Figure 194]<br><br>[Figure 195]<br><br>[Figure 196]<br><br>er Layer|
| | |
|Transform|[Figure 197]<br><br>[Figure 198]<br><br>er Layer|
| | |

[Figure 199]

| | |
|---|---|
| |[Figure 200]|
| | |
|[Figure 201]<br><br>[Figure 202]<br><br>Transform|er Layer|
| | |
|[Figure 203]<br><br>[Figure 204]| |
| | |
|[Figure 205]<br><br>[Figure 206]<br><br>Transform|er Layer|
| | |

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

Diff.

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

Sampling

### ~

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

Local Block Recoverability Estimation

Differentiable Sampling of Candidate Solutions

Winner Decision

Figure 12. Learnable depth pruning on a local block

corresponds to the masked distillation loss applied to the hidden states, as illustrated in Figure 13, which encourages alignment between the intermediate representations of the pruned model and the original model. The corresponding hyperparameters αKD, αDiff and αRep can be found in Table 6.

Hidden States Hidden States

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

Masked

mask mask

Distillation

|Massive Activation ( 𝑥 > 𝑘 ⋅ 𝜎𝑥)<br><br>|
|---|

| | |
|---|---|
|[Figure 269]<br><br>[Figure 270]<br><br>Transform|er Block|
| | |
|Transform|[Figure 271]<br><br>[Figure 272]<br><br>[Figure 273]<br><br>er Block|
| | |
|[Figure 274]<br><br>[Figure 275]<br><br>Transform|er Block|
| | |
|[Figure 276]<br><br>[Figure 277]<br><br>Transform|er Block|
| | |

| | |
|---|---|
| |[Figure 278]<br><br>[Figure 279]|
| | |
|[Figure 280]<br><br>[Figure 281]<br><br>Transform|er Block|
| | |
| |[Figure 282]|
| | |
|Transform|[Figure 283]<br><br>[Figure 284]<br><br>er Block|
| | |

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

Hidden State Alignment. The masked distillation loss LRep is critical for aligning the intermediate representations of the student and teacher models. During the recovery phase, each layer of the student model is designed to replicate the output hidden states of a corresponding two-layer local block from the teacher model. Depth pruning does not alter the internal dimensions of the layers, enabling direct alignment without additional projection layers. For models such as SiTs, where hidden state losses are more pronounced due to their unique interpolant-based architecture, a smaller coefficient β is applied to LRep to mitigate potential training instability. The gradual decrease in β throughout training further reduces the risk of negative impacts on convergence.

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

DiT TinyDiT

Learning the optimal sub-layers

Figure 13. Masked knowledge distillation with 2:4 blocks.

details of each hyperparameter are provided in Table 6.

#### 7. Visualization of Pruning Decisions

Figures 9, 10 and 11 visualize the dynamics of pruning decisions during training for the 1:2, 2:4, and 7:14 pruning schemes. Different divisions lead to varying search spaces, which in turn result in various solutions. For both the 1:2 and 2:4 schemes, good decisions can be learned in only one epoch, while the 7:14 scheme encounters optimization difficulty. This is due to the 147 =3,432 candidates, which is too huge and thus cannot be adequately sampled within a single epoch. Therefore, in practical applications, we use the 1:2 or 2:4 schemes for learnable layer pruning.

Iterative Pruning and Distillation. Table 7 assesses the effectiveness of iterative pruning and teacher selection strategies. To obtain a TinyDiT-D7, we can either directly prune a DiT-XL with 28 layers or craft a TinyDiT-D14 first and then iteratively produce the small models. To investigate the impact of teacher choice and the method for obtaining the initial weights of the student model, we derived the initial weights of TinyDiT-D7 by pruning both a pre-trained model and a crafted intermediate model. Subsequently, we used both the trained and crafted models as teachers for the pruned student models. Across four experimental settings, pruning and distilling using the crafted intermediate model yielded the best performance. Notably, models pruned from the crafted model outperformed those pruned from the pre-trained model regardless of the teacher model employed in the distillation process. We attribute this su-

#### 8. Details of Masked Knowledge Distillation

Training Loss. This work deploys a standard knowledge distillation to learn a good student model by mimicking the pre-trained teacher. The loss function is formalized as:

L = αKD · LKD + αDiff · LDiff + β · LRep (8)

Here, LKD denotes the Mean Squared Error between the outputs of the student and teacher models. LDiff represents the original pre-training loss function. Finally, LRep

###### Model Optimizer Cosine Sched. Teacher αKD αGT β Grad. Clip Pruning Configs

DiT-D19 AdamW(lr=2e-4, wd=0.0) ηmin = 1e-4 DiT-XL 0.9 0.1 1e-2 → 0 1.0 LoRA-1:2 DiT-D14 AdamW(lr=2e-4, wd=0.0 ηmin = 1e-4 DiT-XL 0.9 0.1 1e-2 → 0 1.0 LoRA-1:2 DiT-D7 AdamW(lr=2e-4, wd=0.0) ηmin = 1e-4 DiT-D14 0.9 0.1 1e-2 → 0 1.0 LoRA-1:2 SiT-D14 AdamW(lr=2e-4, wd=0.0) ηmin = 1e-4 SiT-XL 0.9 0.1 2e-4 → 0 1.0 LoRA-1:2 MAR-D16 AdamW(lr=2e-4, wd=0.0) ηmin = 1e-4 MAR-Large 0.9 0.1 1e-2 → 0 1.0 LoRA-1:2

Table 6. Training details and hyper-parameters for mask training

Teacher Model Pruned From IS FID sFID Prec. Recall

DiT-XL/2 DiT-XL/2 29.46 56.18 26.03 0.43 0.51 DiT-XL/2 TinyDiT-D14 51.96 36.69 28.28 0.53 0.59 TinyDiT-D14 DiT-XL/2 28.30 58.73 29.53 0.41 0.50 TinyDiT-D14 TinyDiT-D14 57.97 32.47 26.05 0.55 0.60

Table 7. TinyDiT-D7 is pruned and distilled with different teacher models for 10k, sample steps is 64, original weights are used for sampling rather than EMA.

Masked KD

5.5

Finetune

5.0

DiT-L/2 Scratch

4.5

FID

4.0

3.5

3.0

100 200 300 400 500

Steps

Figure 14. FID and training steps.

perior performance to two factors: first, the crafted model’s structure is better adapted to knowledge distillation since it was trained using a distillation method; second, the reduced search space facilitates finding a more favorable initial state for the student model.

#### 9. Analytical Experiments

Training Strategies Figure 14 illustrates the effectiveness of standard fine-tuning and knowledge distillation (KD), where we prune DiT-XL to 14 layers and then apply various fine-tuning methods. Figure 3 presents the FID scores across 100K to 500K steps. It is evident that the standard fine-tuning method allows TinyDiT-D14 to achieve performance comparable to DiT-L while offering faster inference. Additionally, we confirm the significant effectiveness of distillation, which enables the model to surpass DiTL at just 100K steps and achieve better FID scores than the 500K standard fine-tuned TinyDiT-D14. This is because the distillation of hidden layers provides stronger supervision. Further increasing the training steps to 500K leads to significantly better results.

Learning Rate IS FID sFID Prec. Recall lr=2e-4 207.27 3.73 5.04 0.8127 0.5401 lr=1e-4 194.31 4.10 5.01 0.8053 0.5413 lr=5e-5 161.40 6.63 6.69 0.7419 0.5705

Table 8. The effect of Learning rato for TinyDiT-D14 finetuning w/o knowledge distillation

Learning Rate. We also search on some key hyperparameters such as learning rates in Table 8. We identify the effectiveness of lr=2e-4 and apply it to all models and experiments.

#### 10. Visulization

Figure 15 and 16 showcase the generated images from TinySiT-D14 and TinyMAR-D16, which were compressed from the official checkpoints. These models were trained using only 7% and 10% of the original pre-training costs, respectively, and were distilled using the proposed masked knowledge distillation method. Despite compression, the models are capable of generating plausible results with only 50% of depth.

#### 11. Limitations

In this work, we explore a learnable depth pruning method to accelerate diffusion transformer models for conditional image generation. As Diffusion Transformers have shown significant advancements in text-to-image generation, it is valuable to conduct a systematic analysis of the impact of layer removal within the text-to-image tasks. Additionally, there exist other interesting depth pruning strategies that need to be studied, such as more fine-grained pruning strategies that remove attention layers and MLP layers independently instead of removing entire transformer blocks. We leave these investigations for future work.

[Figure 309]

###### Figure 15. Generated images from TinySiT-D14

[Figure 310]

###### Figure 16. Generated images from TinyMAR-D16

