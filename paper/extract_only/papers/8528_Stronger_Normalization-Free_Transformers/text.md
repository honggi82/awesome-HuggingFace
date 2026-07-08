# arXiv:2512.10938v2[cs.LG]31Mar2026

## Stronger Normalization-Free Transformers

###### Mingzhi Chen1 Taiming Lu1 Jiachen Zhu2 Mingjie Sun3 Zhuang Liu1 1Princeton University 2New York University 3Carnegie Mellon University

[Figure 1]

Point-wise functions

[Figure 2]

ViT ImageNet acc

82.8% 82.5% 82.4% 82.3% 82.2% 82.2% 81.6% 81.6%

Attention / FFN

0

replace with

Norm Layer

- -0.5

- 0.5

-1

- 1

point-wise functions

...

...

-2 -1 0 1 2

a) We search point-wise functions of different shapes as norm layer replacement.

[Figure 3]

[Figure 4]

[Figure 5]

method ViT acc (↑) DiT FID (↓) DNA acc (↑)

LN 82.3% 45.91 86.9% DyT 82.5% 45.66 86.9% Derf 82.8% 43.94 87.3%

LayerNorm: √𝑥𝜎−2+𝜇𝜖 DyT: tanh(𝛼𝑥) Derf: erf(𝛼𝑥 + 𝑠)

c) Performance across domains.

b) Formulation of LayerNorm (LN), DyT, and Derf (ours).

Figure 1 We introduce Dynamic erf (Derf), a point-wise function to replace normalization layers, which outperforms other normalization-based and -free designs across domains. (a) We identify the feasible function shape for replacing the normalization layer and propose a large set of point-wise functions within this space. Evaluating all candidates, we identify and introduce Derf as the strongest choice. (b) LayerNorm, DyT (Zhu et al., 2025), and Derf operate in fundamentally different ways: with channels 𝐶 and tokens 𝑇, LayerNorm normalizes each token across the channel axis, whereas DyT and Derf apply independent scalar mappings to each element. (c) Across ImagenNet-1K classification and generation, and DNA modeling, Derf consistently outperforms LayerNorm and DyT. Derf demonstrates that a point-wise function can not only replace normalization but also surpass it.

##### Abstract

Although normalization layers have long been viewed as indispensable components of deep learning architectures, the recent introduction of Dynamic Tanh (DyT) (Zhu et al., 2025) has demonstrated that alternatives are possible. The point-wise function DyT constrains extreme values for stable convergence and reaches normalization-level performance; this work seeks further for function designs that can surpass it. We first study how the intrinsic properties of point-wise functions influence training and performance. Building on these findings, we conduct a large-scale search for a more effective function design. Through this exploration, we introduce Derf(𝑥) = erf(𝛼𝑥+𝑠), where erf(𝑥) is the rescaled Gaussian cumulative distribution function, and identify it as the most performant design. Derf outperforms LayerNorm, RMSNorm, and DyT across a wide range of domains, including visual recognition and generation, speech representation, and DNA sequence modeling. Our analysis also suggests that the performance gains of Derf largely stem from its improved generalization rather than stronger fitting capacity. Its simplicity and stronger performance make Derf a practical choice for normalization-free Transformer architectures. Our code is available at this link.

1

#### 1 Introduction

Normalization layers have become a critical component in modern deep neural networks. Since the invention of Batch Normalization (Ioffe and Szegedy, 2015), more and more variants have been developed to adapt normalization to various architectures and model modalities (Ba et al., 2016; Salimans and Kingma, 2016; Ulyanov et al., 2016; Wu and He, 2018; Zhang and Sennrich, 2019). By regulating the distribution of intermediate activations, normalization layers have long demonstrated their strong capability in stabilizing training and accelerating model convergence (Santurkar et al., 2018; Bjorck et al., 2018).

Due to the inherent formulation of normalization layers, they heavily rely on activation statistics during training. This introduces additional memory access and synchronization overhead (Zhang and Sennrich, 2019; Chen et al., 2020; Yang et al., 2022). Moreover, some normalization methods are highly sensitive to batch size, and inappropriate batch settings can lead to unstable training (Wu and He, 2018; Lian and Liu, 2019; Singh and Krishnan, 2020). These issues motivate recent efforts to develop normalization-free methods. Among these attempts, Dynamic Tanh (Zhu et al., 2025), an S-shaped point-wise function, has emerged as a simple yet effective drop-in replacement for normalization layers. This work has established the foundation for point-wise functions that match the performance of normalization layers, yet functions that can surpass them remain unexplored. In this work, we aim to discover point-wise functions that outperform normalization layers to push toward stronger Transformer architectures (Vaswani et al., 2017; Dosovitskiy, 2021).

We first systematically study how the intrinsic properties of point-wise functions affect the training dynamics and final performance. Specifically, we focus on four fundamental and representative function properties: zero-centeredness, boundedness, center sensitivity, and monotonicity. Each property is independently examined through controlled experiments on a diverse set of point-wise functions to assess its impact on the training result. This analysis isolates a subset of point-wise functions as effective normalization replacements and yields a concrete design principle for normalization-free Transformers.

Guided by these principles, we identify a set of promising point-wise functions that have the potential to surpass the performance of normalization layers. Within this set, we empirically search for the optimal designs, among which Dynamic erf (Derf) emerges as a simple yet the most performant function (Figure 1a). Derf augments erf(𝑥) with learnable parameters, where the error function erf(𝑥) is an S-shaped, rescaled cumulative distribution of a standard Gaussian around zero.

We evaluate Derf spanning multiple modalities (vision, language, speech, and DNA sequences); covering various tasks (classification, generation, and sequence modeling), under different training paradigms (supervised and self-supervised). Across all these settings, Derf consistently surpasses LayerNorm, and Dynamic Tanh (Figure 1b,1c). To pinpoint the source of these gains, we measure the training loss in evaluation mode after optimization. Derf exhibits higher training loss than normalization-based models, indicating that its superior performance stems from stronger generalization rather than enhanced fitting capacity. Overall, our work demonstrates that well-designed point-wise functions can outperform normalization layers.

#### 2 Background

Normalization layers. Normalization layers have become pivotal components of modern neural networks. Among the various normalization techniques, Batch Normalization (BN) (Ioffe and Szegedy, 2015), Layer Normalization (LN) (Ba et al., 2016), and Root Mean Square Normalization (RMSNorm) (Zhang and Sennrich, 2019) are the three most widely used in deep learning models.

𝑥 − 𝜇 √𝜎2 + 𝜖

+ 𝛽 (1)

𝑦 = 𝛾 *

All normalization methods adhere to a unified paradigm, formalized in Equation 1, where activations within each group are centered and scaled by their mean 𝜇 and standard deviation 𝜎 (with 𝜖 for numerical stability) to maintain consistent scale and stable gradient flow. The main distinction among different normalization methods lies in how the activations are grouped when computing 𝜇 and 𝜎. For example, LN computes the statistics along the channel dimension for each token independently. Given a token representation 𝑥 ∈ R𝐶, the mean and variance are computed as Equation 2, where 𝐶 denotes the number of hidden features (channels).

###### Zero-centered

###### Bounded

###### Center Sensitive

###### Monotonic

Centered

Bounded

Sensitive

Monotonic

Not centered

Not bounded

Not sensitive

Not monotonic

- Figure 2 Key properties of point-wise function. The four properties: zero-centeredness, boundedness, center sensitivity, and monotonicity collectively characterize functional behavior on activations and influence training dynamics. Blue curves represent functions that satisfy each property, while red curves violate them.

Due to its per-token normalization, LN is particularly well-suited for Transformer architectures, where activations across tokens exhibit diverse statistics.

𝜇 =

1 𝐶

1 𝐶

∑︁𝐶

𝑥𝑘, 𝜎2 =

𝑘=1

∑︁𝐶

(𝑥𝑘 − 𝜇)2, (2)

𝑘=1

Point-wise functions. The strong reliance of normalization layers on activation statistics has motivated further exploration of statistics-free methods (He and Hofmann, 2024; Heimersheim, 2024; Jha and Reagen, 2024; Zhu et al., 2025). Among these approaches, point-wise functions (Zhu et al., 2025) have emerged as simple yet effective alternatives to traditional normalization methods. Unlike normalization, a point-wise function applies the same parametric mapping 𝑓(𝑥;𝜃) to each activation independently. The parameters 𝜃 are fixed or learned, rather than being computed from batch-, token-, or channel-level statistics. A recent study (Zhu et al., 2025) introduces the Dynamic Tanh (DyT) function (Equation 3), where 𝛼 is a learnable parameter. This design is motivated by the observation that Layer Normalization often produces an S-shaped input-output mapping in practice. The saturating nature of the tanh function squashes extreme activations, thereby fulfilling a role analogous to the re-centering and re-scaling effects of normalization layers.

DyT(𝑥) = 𝛾 * tanh(𝛼𝑥) + 𝛽 (3)

While DyT has shown similar empirical performance to normalization layers across various Transformer architectures, a comprehensive analysis of the design space for these statistics-free operators remains missing. In this work, we target at the optimal form of the point-wise function as normalization replacement. We identify the function properties crucial for convergence and performance, and then we introduce Derf, a point-wise function consistently surpassing normalization layers rather than merely matching their performance.

#### 3 Function Property Analysis

Training Transformers without normalization requires understanding the factors that make a point-wise function stable and effective as a replacement. In this section, we examine four essential properties: zerocenteredness, boundedness, center sensitivity, and monotonicity (see Figure 2). These properties collectively characterize the fundamental shape of point-wise functions and their behavior on activations. By isolating the impact of each property, we explore its influence on optimization and final performance.

To investigate these properties, we replace each normalization layer with a point-wise function of the form: 𝑦 = 𝛾 · 𝑓(𝛼𝑥) + 𝛽, (4)

where 𝑓(·) denotes the chosen base function with learnable 𝛼 rescaling the input. 𝛾 and 𝛽 are affine parameters, similar to those in normalization layers. We begin with three base functions: tanh(𝑥), erf(𝑥), and arctan(𝑥). In subsequent experiments, we modify these functions with controlled transformations to examine the impact of each property. All experiments are conducted with ViT-Base (Dosovitskiy, 2021), and top-1 accuracy on ImageNet-1K (Deng et al., 2009) is reported. In Appendix A, we provide more detailed training results.

###### 3.1 Zero-centeredness

Zero-centeredness means that the function’s outputs are balanced around zero, with positive and negative values of similar magnitude and symmetry. Because normalization layers inherently recenter activations to the origin for stabilizing gradients, maintaining this property could reduce internal covariate shifts and promote smoother gradient flow during training.

Setup. Under the ViT setup, we manipulate the centering of the functions. For each base function, we consider two types of shifts: horizontal and vertical, defined in Equation 5. In this form, 𝜆horiz and 𝜆vert respectively denote the magnitudes of horizontal and vertical shifts. For both types of shifts, we vary 𝜆 over {±12,±1,±2} to examine how increasing deviation from zero-centeredness affects the function’s behavior. All other training settings remain unchanged.

𝑓horiz(𝑥) = 𝑓(𝑥 + 𝜆horiz), 𝑓vert(𝑥) = 𝑓(𝑥) + 𝜆vert, (5)

Results. As shown in Table 1, the results are consistent across different base functions: for horizontal shifts, performance remains largely comparable to the zero-centered base function when |𝜆horiz| ≤ 0.5. However, as |𝜆horiz| increases, performance gradually degrades, and training diverges when |𝜆horiz| ≥ 2. Similarly, vertical shifts consistently lead to a decline in performance as |𝜆vert| grows with training failure once |𝜆vert| ≥ 2. These results show that zero-centeredness is a requirement for stable convergence and effective training.

function shift type -2 -1 -0.5 -0.1 𝜆 = 0 +0.1 +0.5 +1 +2

horizontal × 82.0% 82.5% 82.6% 82.6% 82.7% 82.5% 82.1% × vertical × 81.8% 82.3% 82.4% 82.6% 82.5% 82.3% 81.6% ×

erf(𝑥)

horizontal × 82.1% 82.5% 82.6% 82.5% 82.6% 82.4% 82.2% × vertical × 81.5% 81.9% 82.4% 82.5% 82.3% 81.9% 81.4% ×

tanh(𝑥)

horizontal × 81.9% 82.3% 82.3% 82.3% 82.4% 82.2% 82.0% × vertical × 81.4% 81.9% 82.2% 82.3% 82.3% 82.0% 81.2% ×

arctan(𝑥)

Table 1 Results of zero-centeredness on ViT-Base. Horizontal shift corresponds to modifying the input as 𝑓(𝛼𝑥 ± 𝜆), while vertical shift adds or subtracts a constant to the output as 𝑓(𝛼𝑥) ± 𝜆. “×” indicates training failure.

###### 3.2 Boundedness

Boundedness refers to the property of a function whose output is constrained within a finite range. Formally, a function 𝑓(·) is bounded if there exist constants 𝑎,𝑏 ∈ R such that 𝑎 ≤ 𝑓(𝑥) ≤ 𝑏 for all 𝑥 in its domain. This ensures that activations remain finite and do not accumulate variance across layers. Unbounded functions, in contrast, may induce signal explosion and gradient instability.

Setup. Under the same ViT setup, we study the role of boundedness with two methods. Firstly, we select three inherently unbounded S-shaped functions (e.g., arcsinh(𝑥)) and compare them with their clamped versions shown in Equation 6, where 𝑓𝑢(𝑥) denotes the unbounded point-wise function, and 𝜆 is a chosen value specifying the clipping range.

𝑦 = clip(𝑓𝑢(𝑥),−𝜆𝑢,𝜆𝑢), (6)

Secondly, we gradually transition bounded functions (e.g., erf(𝑥)) toward unbounded linear form, defined in Equation 7, where 𝑓𝑏 denotes a bounded point-wise function, and 𝜆 controls how quickly the function becomes unbounded. We vary 𝜆𝑢 over {0.5,0.8,1.0,2.0,3.0,5.0} in the first method and 𝜆𝑏 over {0.01,0.1,0.5} for the second. The original unmodified function is also included as a baseline.

𝑦 = (1 − 𝜆)𝑓𝑏(𝑥) + 𝜆𝑏𝑥, 𝜆𝑏 ∈ (0,1). (7)

Results. For the first method, among the three unbounded functions in Table 2, only arcsinh(𝑥) and logsign(𝑥) converge effectively, while linear(𝑥) does not. For the convergent functions, their clipped versions consistently

18

linear(x)

power23(x) logquad(x)

12

arcsinh(x) logsign(x)

6

0

3 6 9 12 15 18

- Figure 3 Visualization of several unbounded point-wise functions on the positive half-axis, illustrating their different growth rates. 𝑎𝑟𝑐𝑠𝑖𝑛ℎ(𝑥) refers to its standard analytical form. The remaining functions are defined as linear(𝑥) = 𝑥, power23(𝑥) = sign(𝑥)·𝑥23 , logsign(𝑥) = sign(𝑥) ln(|𝑥|+1),

smoothsign(𝑥) = 1+𝑥|𝑥|, and logquad(𝑥) = sign(𝑥) ln(𝑥2+1). Among them, logquad(𝑥) shows the fastest growth that still ensures stable convergence.

𝜆𝑢 arcsinh(𝑥) logsign(𝑥) linear(𝑥) − 82.2% 82.2% × 0.5 82.3% 82.4% 82.1%

- 0.8 82.3% 82.4% 82.2%
- 1.0 82.4% 82.4% 82.2%
- 2.0 82.4% 82.4% 82.1%
- 3.0 82.4% 82.3% 82.1% 5.0 82.3% 82.3% 82.0%

Table 2 Results of clamping for boundedness on ViT-Base. Clipped version of unbounded functions consistently achieves better performance than unbounded baselines. “−” denotes the original unmodified function. “×” indicates training failure.

outperform the unbounded baselines across all tested 𝜆 values. These results indicate that incorporating boundedness can improve optimization and result in better performance. For the second, as shown in Table 3, the results are consistent with clipping the intrinsic unbounded functions: the unbounded variant yields slightly lower accuracy than the bounded baseline.

𝜆𝑏 erf(𝑥) tanh(𝑥) arctan(𝑥) isru(𝑥) − 82.6% 82.5% 82.3% 82.2% 0.01 82.4% 82.4% 82.1% 82.2% 0.1 82.3% 82.3% 82.1% 82.1% 0.5 × × × ×

- Table 3 Results of removing boundedness on ViT-Base. Performance decreases as the function is less bounded. “−” denotes the original function without modification and “×” donotes training failure.

Limitation of growth rate. From Table 2 and Table 3, we observe that there is an upper limit on their acceptable growth rate. Large growth rates often lead to training failure. To determine this limit, we evaluate a family of inherently unbounded functions with varying growth rates, as illustrated in Figure 3. Among them, logquad(𝑥) exhibits the fastest growth that still allows training convergence (see Table 4). Functions with faster growth, such as linear(𝑥) and power23(𝑥), tend to cause optimization divergence in the early stages of training. This failure occurs because rapidly growing functions fail to suppress variance effectively, leading to large gradient norms at the start of optimization.

logsign(𝑥) arcsinh(𝑥) logquad(𝑥) power23(𝑥) linear(𝑥) 82.2% 82.2% 82.1% × ×

- Table 4 Results of unbounded functions with different growth rates on ViT-Base. Point-wise functions have a growth rate upper bound, with logquad(𝑥) being the fastest function that still converges. “×” indicates training failure.

###### 3.3 Center Sensitivity

We use center sensitivity to characterize how quickly a point-wise function becomes responsive to input variations around zero. Without center sensitivity, a function is locally flat around the origin, returning zero or near-zero over a finite interval. The region around zero is particularly important, as most activations tend to concentrate near the origin during training. Consequently, the responsiveness of a function in this area directly influences how effectively small signals can propagate through the network.

Setup. Since center sensitivity is difficult to isolate independently, we approximate it using a controllable near-zero inactive region. Under the same ViT setup, we modify each base function to incorporate a symmetric flat region around the origin with a sensitivity scale 𝜆 > 0 to control the extent of this region. Specifically, for inputs in the range 𝑥 ∈ [−𝜆,𝜆], we enforce 𝑓(𝑥) = 0 and smoothly shift the positive and negative parts outward for |𝑥| > 𝜆 to ensure continuity at the boundaries. A smaller 𝜆 results in a narrower flat region and higher sensitivity near zero, while a larger 𝜆 leads to lower sensitivity. We vary 𝜆 over {0.1,0.5,1.0,2.0,3.0} across three base functions.

- Results. As shown in Table 5, the best performance is achieved at 𝜆 = 0. As 𝜆 increases, the performance consistently degrades. This trend is not very clear when 𝜆 ≤ 0.5, but once 𝜆 exceeds 1.0, the degradation becomes much more obvious. Finally, when 𝜆 ≥ 3.0, the training process diverges at an early stage.

function 𝜆 = 0 0.1 0.5 1.0 2.0 3.0

erf(𝑥) 82.6% 82.5% 82.5% 82.1% 81.3% × tanh(𝑥) 82.5% 82.5% 82.4% 82.1% 81.1% × arctan(𝑥) 82.3% 82.3% 82.1% 81.8% 80.9% ×

Table 5 Results of center sensitivity (𝜆) on ViT-Base. “×” indicates training failure. The best performance is achieved when no flat region is given, showing the importance of center sensitivity.

3.4 Monotonicity

Monotonicity ensures a function’s output consistently increases (or decreases) as the input increases, preserving the relative order of inputs throughout the transformation. Non-monotonic functions may disrupt the relative ordering of activations. Furthermore, since a non-monotonic function necessarily has regions where its derivative changes sign, it may also produce flipped gradient signals during training.

Setup. Each base function selected can serve as the monotonically increasing case, while its negated counterpart is defined as 𝑓neg(𝑥) = −𝑓(𝑥), representing the monotonically decreasing variant. As nonmonotonic comparisons, we include hump-shaped functions and oscillatory functions (see Figure 4) to examine how violations of monotonicity influence the training performance. To control potential confounding factors, we rescale each function so that its output range matches that of the monotonic functions. After rescaling, all functions are aligned in terms of zero-centeredness, boundedness, and center sensitivity.

- -0.5

1.0

0.5

0.0

- -1.0

-8 -4 0 4 8

[Figure 6]

Figure 4 Visualization of point-wise functions with different monotonicity behaviors. erf(𝑥) and sin(𝑥) refer to their standard form. The remaining functions are defined as negerf(𝑥) = −erf(𝑥), dampx(𝑥) = 1+2𝑥𝑥2 , dampexp(𝑥) = 2.72𝑥 · 𝑒−|𝑥|

function 𝑓(𝑥) 𝑓neg(𝑥)

erf(𝑥) 82.6% 82.5% tanh(𝑥) 82.5% 82.5% arctan(𝑥) 82.3% 82.2%

(a) Monotonic

function 𝑓(𝑥)

sin(𝑥) 81.6% dampx(𝑥) 80.7% dampexp(𝑥) 81.2%

(b) Non-monotonic

Table 6 Results of monotonicity on ViT-Base. Monotonic functions consistently achieve better performance than their negated versions and other non-monotonic functions, whether humpshaped or oscillatory. This identifies monotonicity as a key property for effective learning.

- Results. As shown in Table 6, both increasing and decreasing monotonic functions train stably and achieve high accuracy. In contrast, non-monotonic functions, whether hump-shaped or oscillatory, consistently perform worse than monotonic functions and lead to a clear drop in final accuracy. These results highlight monotonicity as a key property for point-wise functions to ensure effective learning.

top-1 acc ↑ FID ↓ ViT-Base DiT-B/4 DiT-L/4

function alias

– LayerNorm 82.3% 64.93 45.91 2𝜋−1/2

∫︀ 𝑥

0 𝑒−𝑡2 𝑑𝑡 erf(𝑥) 82.8% 63.23 43.94 (𝑒𝑥 − 𝑒−𝑥)(𝑒𝑥 + 𝑒−𝑥)−1 tanh(𝑥) 82.6% 63.71 45.48 sin(clip(𝑥, −𝜋2 , 𝜋2 )) satursin(𝑥) 82.6% 63.90 44.83 clip (ln(𝑥 + √

𝑥2 + 1), −1, 1) arcsinhclip(𝑥) 82.5% 64.72 45.48 𝑥(𝑥2 + 1)−1/2 isru(𝑥) 82.3% 65.72 45.93 sign(𝑥) ((1 − 𝑒−

√

|𝑥|)) exproot(𝑥) 82.4% 65.20 46.91 clip(𝑥, −1, 1) linearclip(𝑥) 82.3% 66.08 45.49 − sign(𝑥) (𝑒−|𝑥| − 1) expsign(𝑥) 82.2% 64.85 45.82 clip (sign(𝑥) ln(|𝑥| + 1), −1, 1) logsignclip(𝑥) 82.4% 65.59 46.34 𝑥(√

𝑥2 + 1 + 1)−1 relsign(𝑥) 82.3% 68.42 48.33 arctan(𝑥) arctan(𝑥) 82.4% 67.07 46.62 𝑥(1 + |𝑥|)−1 smoothsign(𝑥) 82.4% 68.84 47.29 clip (sign(𝑥) ln(𝑥2 + 1), −1, 1) logquadclip(𝑥) 82.2% 65.92 47.12 clip (sign(𝑥) 𝑥2/3, −1, 1) power23clip(𝑥) 82.1% 66.11 46.47 sign(𝑥) ln(|𝑥| + 1) (ln(|𝑥| + 1) + 1)−1 saturlog(𝑥) 81.8% 68.23 47.44 𝑥3(|𝑥|3 + 1)−1 cubsign(𝑥) 81.4% 70.22 49.16

- Table 7 Top-1 accuracy on ViT-Base and image generation quality (FID) on DiT-B/4 and DiT-L/4. Different functions show noticeable differences in performance. Among all the point-wise functions and LayerNorm, erf(𝑥) shows the best performance in both top-1 accuracy and FID. Visualization of each function is included in Appendix B.

#### 4 Function Search

From the previous section, we observe that functions that are near zero-centered, bounded, center-sensitive (responsive to input variations around zero), and monotonic (increasing or decreasing) tend to yield better optimization performance. Building upon these insights, we start to construct our function set from widely used scalar functions and cumulative distribution functions (CDFs), including polynomial, rational, exponential, logarithmic, and trigonometric forms. We then generate variants via simple transformations such as translation, scaling, mirroring, rotation, and clipping. Functions that satisfy our four function properties after these transformations are retained as the candidate subset used in the search. For example, we transform the unbounded function arcsinh(𝑥) by clipping it to the range [−1,1], limiting it to a finite range and conforming to all four principles. In Appendix B, we provide further details about how we obtain these candidate functions. Within this set, we evaluate their performance, and Derf emerges as the most effective function.

Setup. We conduct an empirical search on two representative vision architectures: Vision Transformer (ViT-Base) (Dosovitskiy, 2021) and Diffusion Transformer (DiT-B/4 and DiT-L/4) (Peebles and Xie, 2023). Models are trained on ImageNet-1K (Deng et al., 2009) under their default training settings. For ViT, model performance is measured using top-1 accuracy on the ImageNet-1K validation set. For DiT, we follow the standard ImageNet reference batch evaluation and report the Fréchet Inception Distance (FID) as the metric.

Formulation. We quantitatively evaluate a set of functions under the constraint of our function properties, as illustrated in Figure 5. Each point-wise function is instantiated in a unified form in Equation 8, where 𝑓(·) denotes a candidate point-wise function, with learnable parameter 𝑠 and 𝛼 recentering and rescaling the input. The parameters 𝛾 and 𝛽 follow the same role as in standard normalization layers. We introduce a learnable shift parameter 𝑠, as it improves the final performance to varying degrees across different functions. Detailed ablation results on the effect of 𝑠 are provided in Section 7.1.

𝑦 = 𝛾 * 𝑓(𝛼𝑥 + 𝑠) + 𝛽, (8)

- 0.5

0.75

- 1

- 0.5

0.75

- 1

power23clip(x)

logsignclip(x)

satursin(x)

expsign(x)

linearclip(x)

cubsign(x) exproot(x) saturlog(x)

erf(x)

arctan(x)

0.25

0.25

arcsinhclip(x)

tanh(x)

smoothsign(x)

logquadclip(x)

isru(x)

relsign(x)

0

0

0.5 1 1.5 2 2.5 3

0.5 1 1.5 2 2.5 3

- Figure 5 Visualization of candidate point-wise functions on the positive half-axis. All functions are self-symmetric with respect to the origin.

Quantitative evaluation. As shown in Table 7, even though these S-shaped functions appear highly similar in form, their empirical training results show noticeable differences in final performance. Among all the point-wise functions, erf(𝑥) with the introduced transformations stands out as the best-performing function, consistently surpassing all other candidates and the baseline normalization layers.

#### 5 Dynamic erf (Derf)

From the search, we identify erf(𝑥) as the most performant point-wise function. The error function erf(·) is closely related to the cumulative distribution function (CDF) of a standard Gaussian distribution. Specifically, erf(𝑥) can be defined by Equation 9. In our setup, erf(𝑥) is in the form augmented with learnable parameters, which we introduce as Derf, Dynamic erf. Given an input tensor 𝑥, a Derf layer is defined in Equation 10, where both the shift 𝑠 and the scale 𝛼 are learnable scalars. 𝛾 and 𝛽 are learnable per-channel vectors. To integrate Derf into a transformer-based architecture, we replace each normalization layer with a corresponding Derf layer. In particular, the pre-attention, the pre-FFN, and the final normalization layers are all substituted in a one-to-one manner, ensuring consistent incorporation of Derf across the entire model.

∫︁ 𝑥

2 √𝜋

2

erf(𝑥) =

𝑑𝑡 (9)

𝑒−𝑡

0

Derf(𝑥) = 𝛾 * erf(𝛼𝑥 + 𝑠) + 𝛽 (10)

Parameter initialization. We initialize 𝛾 to an all-one vector and 𝛽 to an all-zero vector following the same strategy as in standard normalization layers. For the additional scalar parameters introduced by Derf, the scaling parameter 𝛼 is initialized to 0.5, while the shift parameter 𝑠 is initialized to 0. Unless otherwise specified, these initialization settings are adopted throughout all experiments.

#### 6 Experiments

We evaluate the effectiveness of Derf across various transformer-based and a few other modern architectures. For each model, we replace the original normalization layers with DyT and Derf, following the standard training and evaluation protocols, as detailed in Appendix C. Across all tested architectures, Derf consistently achieves stronger performance over the baseline normalization methods and DyT. Besides each model’s default normalization, we also report results with other common normalization methods in Appendix D.

Vision Transformers. We train ViT-Base and ViT-Large models (Dosovitskiy, 2021) on ImageNet-1K (Deng et al., 2009) using LayerNorm (LN), DyT, and Derf for comparison. Table 8 reports the top-1 classification accuracy. Compared to LN and DyT, Derf achieves clearly higher top-1 accuracy.

model LN DyT Derf ΔLN ΔDyT

ViT-B 82.3% 82.5% 82.8% ↑ 0.5% ↑ 0.3% ViT-L 83.1% 83.6% 83.8% ↑ 0.7% ↑ 0.2%

- Table 8 Supervised classification accuracy on ImageNet-1K. Derf achieves higher top-1 accuracy than both LN and DyT on different model sizes, demonstrating its effectiveness in vision transformer architectures.

Diffusion Transformers. We train three Diffusion Transformer (DiT) (Peebles and Xie, 2023) models on ImageNet-1K (Deng et al., 2009). Consistent with the original DiT setup, the affine parameters in the normalization layers are retained for class conditioning across LN, DyT, and Derf. After training, we evaluate the FID scores using the standard ImageNet “reference batch” to measure image generation quality, as reported in Table 9. Derf achieves a clear improvement in FID compared to both LayerNorm and DyT.

model LN DyT Derf ΔLN ΔDyT DiT-B/4 64.93 63.94 63.23 ↓ 1.70 ↓ 0.71 DiT-L/4 45.91 45.66 43.94 ↓ 1.97 ↓ 1.72 DiT-XL/2 19.94 20.83 18.92 ↓ 1.02 ↓ 1.91

- Table 9 Image generation quality (FID) on ImageNet. Lower FID indicates better image generation quality. Derf achieves lower FID scores than both LN and DyT across all DiT model sizes.

Speech models. We train two wav2vec 2.0 Transformer models (Baevski et al., 2020) on the LibriSpeech dataset (Panayotov et al., 2015) for speech representation learning. We report the final validation loss in

- Table 10. Compared to LayerNorm and DyT, Derf yields lower validation loss on different model sizes.

model LN DyT Derf ΔLN ΔDyT wav2vec 2.0 Base 1.95 1.95 1.93 ↓ 0.02 ↓ 0.02 wav2vec 2.0 Large 1.92 1.91 1.90 ↓ 0.02 ↓ 0.01

- Table 10 Speech pretraining validation loss on the LibriSpeech dataset. Derf achieves lower validation loss than both LN and DyT across two wav2vec 2.0 models, indicating its better representation quality.

DNA models. For the long-range DNA sequence modeling task, we pretrain the HyenaDNA model (Nguyen et al., 2023) and the Caduceus model (Schiff et al., 2024) using the human reference genome from (GRCh38, 2013). Model evaluation is conducted on the GenomicBenchmarks dataset (Grešová et al., 2023). We report the averaged accuracy over all subtasks. As shown in Table 11, Derf surpasses both normalization layers and DyT in performance, demonstrating its robustness in genomic sequence modeling.

model Norm DyT Derf ΔNorm ΔDyT

Hyena 85.2% 85.2% 85.7% ↑ 0.5% ↑ 0.5% Caduceus 86.9% 86.9% 87.3% ↑ 0.4% ↑ 0.4%

- Table 11 DNA classification accuracy on the GenomicBenchmarks dataset, averaged over each subtask. Each model is evaluated with its default normalization layer (LN for Heyna, RMSNorm for Caduceus). Derf consistently achieves higher accuracy than both normalization layers and DyT, indicating its effectiveness in DNA model.

Language models. We pretrain a GPT-2 (124M) model on the OpenWebText dataset and report the validation loss in Table 12. For DyT and Derf, we additionally finetune the initialization of the learnable parameter 𝛼. We observe that Derf achieves comparable performance to LN, while clearly outperforming DyT.

model LN DyT Derf ΔLN ΔDyT GPT-2 2.94 2.97 2.94 0.00 ↓ 0.03

- Table 12 GPT-2 validation loss on the OpenWebText dataset. Derf matches the performance of LN while achieving lower validation loss than DyT.

- 6.1 Stronger Generalization or Better Fitting?

Given Derf’s superior performance, we aim to determine whether the gains arise from improved fitting capacity or stronger generalization. To this end, we compare the training loss of models respectively trained with normalization layers, DyT, and Derf. Since lower training loss indicates stronger fitting ability, this comparison helps us assess whether Derf improves optimization or enhances generalization.

Setup. We compute training losses across diverse architectures and scales. To measure fitting capacity fairly, we do not use the loss during optimization, which is confounded by stochastic regularization (e.g., stochastic depth (Huang et al., 2016)) and train-time augmentations. Instead, after training, we switch to evaluation mode, disable stochastic depth (when present), adopt the test-time preprocessing pipeline, and compute the loss on the training set. This yields a fair estimate of each model’s fitting capacity. In Appendix E, we provide the detailed procedure for computing the evaluation-mode training loss for each model.

Results. Across all architectures and scales, both Derf and DyT result in higher training loss than normalizationbased models, with Derf generally yielding slightly lower training loss than DyT, as shown in Table 13. This consistent pattern indicates that neither Derf nor DyT improves fitting capacity over normalization layers.

model Norm Derf DyT

ViT-B 0.2623 0.2681 0.2714 ViT-L 0.2034 0.2066 0.2083 DiT-B 0.1531 0.1533 0.1535 DiT-L 0.1501 0.1510 0.1518 DiT-XL 0.1432 0.1436 0.1440 wav2vec 2.0 B 1.8509 1.8821 1.8946 wav2vec 2.0 L 1.8241 1.8563 1.8641 Hyena 1.1297 1.1526 1.1631 Caduceus 0.8917 0.9129 0.9203 GPT-2 2.9478 2.9702 2.9822

- Table 13 Evaluation-mode training loss of normalization layers (Norm), Derf, and DyT after optimization. Bolded indicates the lowest loss, and underlined means the second-lowest loss. Across all model architectures, the training loss follows the relation: Norm < Derf < DyT. Both DyT and Derf exhibit higher training loss than normalization layers, while Derf achieves slightly lower loss than DyT.

Discussion. Despite the reduced fitting capacity, Derf delivers consistent performance gains across all evaluated tasks. We hypothesize that these gains arise primarily from both better generalization than normalization layers and stronger fitting capacity than DyT.

Firstly, point-wise functions promote stronger generalization. Although Derf yields higher training loss, it achieves superior downstream performance, indicating that its benefits stem not from improved fitting but from enhanced generalization. This difference likely originates from the contrasting operational principles between normalization layers and point-wise functions. Normalization layers adapt their transformation

based on training statistics, allowing them to dynamically fit activation distributions throughout training. In contrast, point-wise functions are controlled by only a small set of learnable scalar parameters (e.g., 𝛼 for DyT and 𝛼, 𝑠 for Derf) that do not adapt to activation statistics after training. They apply the same transformation regardless of activation distribution. This limited adaptability constrains overfitting and effectively serves as an implicit regularizer, leading to improved generalization.

Secondly, Derf exhibits stronger fitting power than DyT. It achieves lower training loss while retaining the implicit regularization of point-wise functions, combining higher fitting capacity with strong generalization to outperform both DyT and normalization-based models.

#### 7 Analysis

In this section, we begin with two ablation studies examining the influence of the learnable shift parameter 𝑠 on the training results, followed by an analysis of an approximation of Derf.

###### 7.1 Effect of s

Removing 𝑠. We investigate the effect of the learnable scalar parameter 𝑠 by removing it from the point-wise function. As shown in Table 14, introducing this learnable shift consistently improves the overall training performance, and the degree of improvement varies across different functions. The stronger results of erf(𝑥) over tanh(𝑥) indicate that Derf surpasses DyT not only because of the shift 𝑠.

top-1 acc ↑ FID ↓ function without 𝑠 with 𝑠 without 𝑠 with 𝑠

erf(𝑥) 82.6% 82.8% 63.39 63.23 tanh(𝑥) 82.5% 82.6% 63.94 63.71 satursin(𝑥) 82.4% 82.6% 65.28 63.90 isru(𝑥) 82.2% 82.3% 66.14 65.72 arctan(𝑥) 82.3% 82.4% 67.41 67.07 arcsinhclip(𝑥) 82.4% 82.5% 65.19 64.72

- Table 14 Ablation study of 𝑠. Top-1 accuracy on ViT-Base and FID score on DiT-B/4, comparing models with and without 𝑠. 𝑠 improves the overall training performance, while its effect varies across different point-wise functions.

Scalar vs. vector 𝑠. We further examine whether using a per-channel vector parameter instead of a scalar 𝑠 leads to any performance improvement. As shown in Table 15, across all three point-wise functions, the choice between a scalar and a per-channel vector shows no significant impact on the final performance. Therefore, we adopt the scalar form of 𝑠 for efficiency and simplicity during training.

function vector scalar erf(𝑥) 82.8% 82.8% arctan(𝑥) 82.5% 82.4% arcsinhclip(𝑥) 82.5% 82.5%

- Table 15 Top-1 accuracy of scalar vs. vector 𝑠 on ViT-Base. Using either a scalar or a per-channel vector for the parameter 𝑠 yields nearly identical performance.

function ViT-B ViT-L DiT-B DiT-L

tanh(𝑥) 82.6% 83.6% 63.71 45.48 tanh(𝜀𝑥) 82.7% 83.7% 63.88 45.13 erf(𝑥) 82.8% 83.8% 63.23 43.94

Table 16 Top-1 accuracy of tanh(𝜀𝑥) on ViT and DiT. tanh(𝜀𝑥) yields a comparable or slightly improved performance over tanh(𝑥) but still remains below erf(𝑥).

hu

###### 7.2 Approximating Derf

Given the superior performance of erf(𝑥) over tanh(𝑥), we approximate erf(𝑥) by scaling tanh(𝑥) and examine whether this modification can lead to performance improvement. We introduce a fixed coefficient 𝜀 and use tanh(𝜀𝑥), where 𝜀 is obtained by minimizing the following objective:

∫︁ +∞

⃒tanh(𝜀𝑥) − erf(𝑥)⃒𝑑𝑥. (11)

min

𝜀

−∞

The optimal value is found to be 𝜀 ≈ 1.205. As shown in Table 16, tanh(𝜀𝑥) achieves a comparable or slightly improved performance over the original tanh(𝑥), while still performing worse than erf(𝑥). This indicates that simply scaling tanh(𝑥) is insufficient to match the behavior or performance of erf(𝑥).

#### 8 Related Work

Normalization layers. Since the introduction of Batch Normalization (BN) (Ioffe and Szegedy, 2015), various normalization methods have been proposed to better stabilize training. To address BN’s limitations with small batches, several alternatives (Salimans and Kingma, 2016; Wu and He, 2018; Yan et al., 2020; Shen et al., 2020; Singh and Krishnan, 2020) have been explored. In parallel, LayerNorm (Ba et al., 2016; Nguyen and Salazar, 2019; Xu et al., 2019; Xiong et al., 2020) and RMSNorm (Zhang and Sennrich, 2019) were designed for RNN (Hochreiter and Schmidhuber, 1997) and Transformer architectures (Vaswani et al., 2017). Task-specific variants (Ulyanov et al., 2016; Wu and He, 2018; Shen et al., 2020) further adapt normalization to applications such as object detection and style transfer.

Mechanisms of normalization. A series of studies has investigated how normalization layers contribute to model convergence. From an optimization perspective, normalization stabilizes gradient flow (Balduzzi et al., 2017; Daneshmand et al., 2020; Lubana et al., 2021), reduces sensitivity to initialization (Zhang et al., 2019; De and Smith, 2020; Shao et al., 2020), and implicitly tunes learning rates (Arora et al., 2019; Tanaka and Kunin, 2021). It has also been shown to smooth the loss landscape (Santurkar et al., 2018; Bjorck et al., 2018; Karakida et al., 2019) and reduce sharpness (Lyu et al., 2022; Dai et al., 2023; Mueller et al., 2023), promoting more stable optimization dynamics. Understanding these underlying functionalities provides valuable guidance for designing normalization-free training methods.

Normalization-free methods. Building on this understanding of normalization, recent work explores how to achieve stable convergence without normalization. One line of work operates at the parameter and optimization level, using tailored initialization schemes (Zhang et al., 2019; De and Smith, 2020; Bachlechner et al., 2021), self-normalizing activations (Klambauer et al., 2017), weight normalization (Salimans and Kingma, 2016; Brock et al., 2021a), or adaptive gradient clipping (Brock et al., 2021b) to maintain stable gradient propagation. Another line of work modifies the architecture through structural simplifications (He and Hofmann, 2024), Softmax-only formulations (Jha and Reagen, 2024), and bounded convolutional operators (Liu et al., 2017, 2018). More recently, point-wise functions such as Dynamic Tanh (Zhu et al., 2025) have been proposed, with theoretical analyses revealing their similarity to normalization operations (Stollenwerk, 2025). Unlike previous methods that aim to match the performance of normalization layers, Derf consistently delivers stronger performance across diverse models.

#### 9 Conclusion

In this work, we demonstrate that well-designed point-wise functions do not merely match the performance of normalization layers, but can surpass them. By revisiting the design space of point-wise functions, we identify zero-centeredness, boundedness, center sensitivity, and monotonicity as four key properties that enable strong performance in Transformer-based models. Among the functions satisfying these properties, Derf stands out as the most effective design: it consistently outperforms normalization-based methods and another notable point-wise function, DyT, across a wide range of modalities and tasks. Its simplicity and strong empirical performance make Derf a compelling replacement for normalization layers in many Transformer architectures.

###### Acknowledgments

We gratefully acknowledge the use of the Neuronic GPU computing cluster maintained by the Department of Computer Science at Princeton University. This work was substantially performed using Princeton Research Computing resources, a consortium led by the Princeton Institute for Computational Science and Engineering (PICSciE) and Research Computing at Princeton University. This work is also supported by the computational resources generously provided by Google’s TPU Research Cloud program.

#### References

Sanjeev Arora, Zhiyuan Li, and Kaifeng Lyu. Theoretical analysis of auto rate-tuning by batch normalization. ICLR, 2019.

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

Thomas Bachlechner, Bodhisattwa Prasad Majumder, Henry Mao, Gary Cottrell, and Julian McAuley. Rezero is all you need: Fast convergence at large depth. In UAI, 2021.

Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. In NeurIPS, 2020.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

David Balduzzi, Marcus Frean, Lennox Leary, JP Lewis, Kurt Wan-Duo Ma, and Brian McWilliams. The shattered gradients problem: If resnets are the answer, then what is the question? In ICML, 2017.

Nils Bjorck, Carla P Gomes, Bart Selman, and Kilian Q Weinberger. Understanding batch normalization. In NeurIPS, 2018.

Andrew Brock, Soham De, and Samuel L Smith. Characterizing signal propagation to close the performance gap in unnormalized resnets. ICLR, 2021a.

Andrew Brock, Soham De, Samuel L Smith, and Karen Simonyan. High-performance large-scale image recognition without normalization. In ICML, 2021b.

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In ECCV, 2020.

Zhaodong Chen, Lei Deng, Guoqi Li, Jiawei Sun, Xing Hu, Ling Liang, Yufei Ding, and Yuan Xie. Effective and efficient batch normalization using a few uncorrelated data for statistics estimation. IEEE Transactions on Neural Networks and Learning Systems, 2020.

Yan Dai, Kwangjun Ahn, and Suvrit Sra. The crucial role of normalization in sharpness-aware minimization. In NeurIPS, 2023.

Hadi Daneshmand, Jonas Kohler, Francis Bach, Thomas Hofmann, and Aurelien Lucchi. Batch normalization provably avoids ranks collapse for randomly initialised deep networks. In NeurIPS, 2020.

Soham De and Sam Smith. Batch normalization biases residual blocks towards the identity function in deep networks. In NeurIPS, 2020.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.

Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021. Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,

Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Ensembl GRCh38. p13 (genome reference consortium human build 38), insdc assembly, 2013. Katarína Grešová, Vlastimil Martinek, David Čechák, Petr Šimeček, and Panagiotis Alexiou. Genomic

benchmarks: a collection of datasets for genomic sequence classification. BMC Genomic Data, 2023.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Bobby He and Thomas Hofmann. Simplifying transformer blocks. ICLR, 2024. Stefan Heimersheim. You can remove gpt2’s layernorm by fine-tuning. arXiv preprint arXiv:2409.13710, 2024. Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural Computation, 1997. Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic

depth. In ECCV, 2016. Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, 2015. Nandan Kumar Jha and Brandon Reagen. Aero: Softmax-only llms for efficient private inference. arXiv preprint arXiv:2410.13060, 2024. Ryo Karakida, Shotaro Akaho, and Shun-ichi Amari. The normalization method for alleviating pathological sharpness in wide neural networks. In NeurIPS, 2019. Günter Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter. Self-normalizing neural networks. In NeurIPS, 2017. Xiangru Lian and Ji Liu. Revisit batch normalization: New understanding and refinement via composition optimization. In AISTATS, 2019.

Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434, 2024.

Weiyang Liu, Yan-Ming Zhang, Xingguo Li, Zhiding Yu, Bo Dai, Tuo Zhao, and Le Song. Deep hyperspherical learning. NeurIPS, 2017.

Weiyang Liu, Zhen Liu, Zhiding Yu, Bo Dai, Rongmei Lin, Yisen Wang, James M Rehg, and Le Song. Decoupled networks. In CVPR, 2018.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, 2021.

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In CVPR, 2022.

Ekdeep S Lubana, Robert Dick, and Hidenori Tanaka. Beyond batchnorm: Towards a unified understanding of normalization in deep learning. In NeurIPS, 2021.

Kaifeng Lyu, Zhiyuan Li, and Sanjeev Arora. Understanding the generalization benefit of normalization layers: Sharpness reduction. In NeurIPS, 2022.

Maximilian Mueller, Tiffany Vlaar, David Rolnick, and Matthias Hein. Normalization layers are all that sharpness-aware minimization needs. In NeurIPS, 2023.

Eric Nguyen, Michael Poli, Marjan Faizi, Armin Thomas, Michael Wornow, Callum Birch-Sykes, Stefano Massaroli, Aman Patel, Clayton Rabideau, Yoshua Bengio, et al. Hyenadna: Long-range genomic sequence modeling at single nucleotide resolution. In NeurIPS, 2023.

Toan Q Nguyen and Julian Salazar. Transformers without tears: Improving the normalization of self-attention. IWSLT, 2019.

Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: an asr corpus based on

public domain audio books. In ICASSP, 2015. William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou,

Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020.

Tim Salimans and Durk P Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In NeurIPS, 2016.

Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, and Aleksander Madry. How does batch normalization help optimization? In NeurIPS, 2018.

Yair Schiff, Chia-Hsiang Kao, Aaron Gokaslan, Tri Dao, Albert Gu, and Volodymyr Kuleshov. Caduceus: Bi-directional equivariant long-range dna sequence modeling. In ICML, 2024.

Jie Shao, Kai Hu, Changhu Wang, Xiangyang Xue, and Bhiksha Raj. Is normalization indispensable for training deep neural network? In NeurIPS, 2020.

Sheng Shen, Zhewei Yao, Amir Gholami, Michael Mahoney, and Kurt Keutzer. Powernorm: Rethinking batch normalization in transformers. In ICML, 2020.

Saurabh Singh and Shankar Krishnan. Filter response normalization layer: Eliminating batch dependence in the training of deep neural networks. In CVPR, 2020.

Felix Stollenwerk. The mathematical relationship between layer normalization and dynamic activation functions. arXiv preprint arXiv:2503.21708, 2025.

Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In CVPR, 2016.

Hidenori Tanaka and Daniel Kunin. Noether’s learning dynamics: Role of symmetry breaking in neural networks. In NeurIPS, 2021.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.

Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Instance normalization: The missing ingredient for fast stylization. arXiv preprint arXiv:1607.08022, 2016.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser,

and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. Yuxin Wu and Kaiming He. Group normalization. In ECCV, 2018. Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan,

Liwei Wang, and Tieyan Liu. On layer normalization in the transformer architecture. In ICML, 2020. Jingjing Xu, Xu Sun, Zhiyuan Zhang, Guangxiang Zhao, and Junyang Lin. Understanding and improving layer normalization. In NeurIPS, 2019.

Junjie Yan, Ruosi Wan, Xiangyu Zhang, Wei Zhang, Yichen Wei, and Jian Sun. Towards stabilizing batch statistics in backward propagation of batch normalization. ICLR, 2020.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.

Qiming Yang, Kai Zhang, Chaoxiang Lan, Zhi Yang, Zheyang Li, Wenming Tan, Jun Xiao, and Shiliang Pu. Unified normalization for accelerating and stabilizing transformers. In ACM MM, 2022.

Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix:

Regularization strategy to train strong classifiers with localizable features. In ICCV, 2019. Biao Zhang and Rico Sennrich. Root mean square layer normalization. In NeurIPS, 2019. Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk

minimization. 2018. Hongyi Zhang, Yann N Dauphin, and Tengyu Ma. Fixup initialization: Residual learning without normalization. ICLR, 2019. Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and Yi Yang. Random erasing data augmentation. In AAAI, 2020. Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, and Zhuang Liu. Transformers without normalization. In CVPR, 2025.

### Appendix

#### A Property Analysis Details

In this section, we provided detailed explanation and visualization on how different function properties affect model training.

###### A.1 Zero-centeredness

We plot the training curves for 𝜆horiz and 𝜆vert with values {0,0.1,1} in Figure 6. The trends are consistent with those observed in top-1 accuracy on ImageNet-1K. For horizontal shifts, the training loss with 𝜆horiz = 0.1 nearly overlaps with that of 𝜆horiz = 0, and even reaches a slightly lower loss. In contrast, vertical shifts exhibit a monotonic pattern: increasing 𝜆vert consistently raises the training loss, suggesting reduced fitting capacity under larger vertical shift.

- 3

- 4

- 5

- 6

- 7

λ = 0

- λ = 0.1

- λ = 1

loss

0 50 100 150 200 250 300

epoch

(a) Horizontal shift

- 3

- 4

- 5

- 6

- 7

λ = 0

- λ = 0.1

- λ = 1

loss

0 50 100 150 200 250 300

epoch

(b) Vertical shift

- Figure 6 Training loss curve for horizontal and vertical shifts on the base point-wise function erf(𝑥). The trends are consistent with the patterns observed in top-1 accuracy on ImageNet-1K.

###### A.2 Center Sensitivity

We visualize the training losses obtained as 𝜆 varies over {0, 0.1, 0.5, 1.0, 2.0} on the base point-wise function erf(𝑥). As shown in Figure 7, training loss shows a clear monotonic trend: larger 𝜆 consistently leads to higher loss, indicating that the width of the flat zone directly limits the model’s fitting capacity.

- 3

- 4

- 5

- 6

- 7

[Figure 7]

λ = 0

λ = 0.1

- λ = 0.5

- λ = 1.0

- λ = 2.0

loss

[Figure 8]

0 50 100 150 200 250 300

epoch

Figure 7 Training loss curve for different center sensitivity (controlled by 𝜆). A larger 𝜆 leads to higher training loss and poorer fitting ability.

- 3

- 4

- 5

- 6

- 7

erf(x) sin(x) negerf(x)

dampx(x)

loss

0 50 100 150 200 250 300

epoch

Figure 8 Training loss curve for different monotonicity. Monotonic functions consistently achieve lower training loss than non-monotonic functions.

###### A.3 Monotonicity

We plot the training losses of four functions with distinct monotonicity patterns: the monotonically increasing erf(𝑥), the monotonically decreasing negerf(𝑥), the hump-shaped dampx(𝑥), and the oscillatory sin(𝑥). As shown in Figure 8, both increasing and decreasing monotonic functions achieve clearly lower training loss, indicating stronger fitting capacity. In contrast, the non-monotonic functions exhibit higher training loss. This behavior aligns closely with the top-1 accuracy trends observed on ImageNet-1K.

#### B Function Search Details

In function search, a wide variety of common functional forms are systematically explored under the constraint of our function properties. The candidates range from polynomial and rational functions to the trigonometric and hyperbolic families, as well as various cumulative distribution functions. Beyond these common functional forms, we also experiment with their variants through translation, scaling, concatenation, and clipping.

We categorize all candidate functions (see Table 7) into four groups: natural functions, transformed basic functions, clipped unbounded functions, and canonical ratio functions, and present detailed descriptions and visualizations of how each group is constructed.

Natural functions. This category consists of three functions: erf(𝑥), tanh(𝑥), and arctan(𝑥). As shown in

- Figure 9, these functions naturally satisfy all the function properties, including zero-centeredness, boundedness, center sensitivity, and monotonicity. Among them, only arctan(𝑥) is rescaled so that all three functions have their ranges unified to [−1,1].

Transformed basic functions. This category consists of six functions: satursin(𝑥), expsign(𝑥), exproot(𝑥), relsign(𝑥), isru(𝑥), and cubsign(𝑥). These functions are constructed by starting from simple and commonly used primitives, such as power functions and polynomial forms. Through transformations including translation, scaling, and rotation, we reshape their original structures so that they satisfy all four function properties while preserving the qualitative behavior of the underlying base functions, as shown in Figure 10.

Clipped unbounded functions. This category consists of five functions: logsign(𝑥), logquad(𝑥), arcsinh(𝑥), power23(𝑥), and linear(𝑥). These functions inherently satisfy zero-centeredness and center sensitivity. For logsignclip(𝑥), logquad(𝑥), and power23clip(𝑥), either due to domain asymmetry or because the original form is not monotonic, we construct the negative branch by mirroring the positive side around the origin to ensure monotonicity, as shown in Figure 11. To additionally enforce boundedness, we clip their outputs to the interval [−1,1], which leads to improved performance in practice.

- 0.5

- 1

0

erf(x)

tanh(x)

- -3 -2 -1 0 1 2 3

- -1

- -0.5

arctan(x)

###### Figure 9 Visualization of natural functions.

Original Function

After Transformation

1.0

1.0

0.5

0.5

0.0

0.0

0.5

0.5

sin(x)

satursin(x)

1.0

1.0

4 2 0 2 4

4 2 0 2 4

- 0

- 1

- 2

- 3

- 4

- 5

1.0

0.5

0.0

e−x

expsign(x)

0.5

1.0

2 0 2 4

4 2 0 2 4

- 0 2 4 6 8

0.00

0.25

0.50

0.75

1.00

e− x

4 2 0 2 4

1.0

0.5

0.0

0.5

1.0

exproot(x)

4 2 0 2 4

- 1

- 0

- 1

- 2

- 3

- 4

1.0

0.5

0.0

1 + x2

relsign(x)

0.5

1.0

4 2 0 2 4

1.5

1.0

0.5

1.0

0.0

0.5

(1 + x2)−1

isru(x)

0.5

1.0

0.0

0 1 2 3 4

4 2 0 2 4

20

1.0

10

0.5

0

0.0

x−3

cubsign(x)

0.5

10

1.0

20

4 2 0 2 4

4 2 0 2 4

- Figure 10 Visualization of transformed basic functions.

- 0

- 1

- 2

- 3

Original Function

ln(x + 1)

4 2 0 2 4

1.0

0.5

0.0

0.5

1.0

After Transformation

logsignclip(x)

- 0 1 2 3 4

- 0

- 1

- 2

- 3

ln(x2 + 1)

4 2 0 2 4

1.0

0.5

0.0

0.5

1.0

logquadclip(x)

- 4 2 0 2 4

2

- 1

- 0

- 1

- 2

arcsinh(x)

4 2 0 2 4

1.0

0.5

0.0

0.5

1.0

arcsinhclip(x)

0 1 2 3 4

- 0

- 1

- 2

- 3

x2/3

4 2 0 2 4

1.0

0.5

0.0

0.5

1.0

power23clip(x)

- 4 2 0 2 4

4

- 2

0 1 2 3 4

4

1.0

2

0.5

0

0.0

0.5

linearclip(x)

x

1.0

4 2 0 2 4

###### Figure 11 Visualization of clipped unbounded functions.

- 0

2

4

Original Function

x

4 2 0 2 4

1.0

0.5

0.0

0.5

1.0

After Transformation

smoothsign(x)

8 4 0 4 8

2

- 1

2

4

4 2 0 2 4

1.0

- 0

- 1

- 2

0.5

0.0

0.5

sign(x) · ln(|x| + 1)

saturlog(x)

1.0

4 2 0 2 4

###### Figure 12 Visualization of canonical ratio functions.

Canonical ratio functions. This category consists of two functions: saturlog(𝑥) and smoothsign(𝑥). Both functions are constructed using the canonical ratio form |𝑓(𝑓𝑥()𝑥|)+1, which naturally enforces boundedness and monotonicity. By selecting 𝑓(𝑥) to be an odd, zero-centered base function, the resulting ratio form automatically satisfies zero-centeredness and center sensitivity as well. As shown in Figure 12, this construction yields smooth saturating behaviors that remain stable across a wide input range.

#### C Experimental Settings

Vision Transformers. For all supervised classification experiments on ImageNet-1K, we adopt the training configurations summarized in Table 17. ViT-B and ViT-L share the same hyperparameters, except that ViT-L employs a modified AdamW momentum setting with (𝛽1=0.9, 𝛽2=0.95) and a higher stochastic depth rate of 0.5.

|config<br><br>|value|
|---|---|
|optimizer base learning rate weight decay optimizer momentum effective batch size learning rate schedule warmup epochs training epochs augmentation label smoothing (Szegedy et al., 2016) mixup (Zhang et al., 2018) cutmix (Yun et al., 2019) random erase (Zhong et al., 2020) drop path (Huang et al., 2016) exp. moving average (EMA)<br><br>|AdamW 4e-3 0.05 𝛽1=0.9, 𝛽2=0.999 (𝐵), 0.95 (𝐿) 4096 cosine decay 20 300 rand-m9-mstd0.5-inc1 0.1 0.8 1.0 0.25 0.15 (B), 0.5 (L) 0.9999|

Table 17 Training Configurations of ViT.

Diffusion Transformers. We use the official implementation (Peebles and Xie, 2023) to train all DiT model sizes as shown in Table 18. We observe that the default learning rate is suboptimal for the models in this work. For both the search function experiments and the final evaluation of Derf, we go through three learning rates, 1 × 10−4, 2 × 10−4, and 4 × 10−4, for all models, whether they use LayerNorm or a point-wise function, and report the best result. We also observe that the zero initialization negatively affects the performance of Derf models and other point-wise function models. Therefore, we retain the zero initialization for LN models but remove it for the other models.

|config|value<br><br>|
|---|---|
|optimizer base learning rate weight decay optimizer momentum effective batch size learning rate schedule training epochs exp. moving average (EMA)|AdamW {1e-4, 2e-4, 4e-4} 0 𝛽1=0.9, 𝛽2=0.999 256 constant 80 0.9999<br><br>|

Table 18 Training Configurations of DiT.

Speech models. For both wav2vec 2.0 models, we retain the first GroupNorm layer and the LayerNorm located after the convolutional feature extractor, since both primarily serve as data normalization to handle the unnormalized input data. We use the official implementation (Baevski et al., 2020) for both the Base and Large models, keeping all hyperparameters identical to the original setup, as shown in Table 19. The only change we make is running all models—whether normalization-based or point-wise-function-based—in fp32 precision instead of the default bf16. We report the final validation loss.

|config<br><br>|value|
|---|---|
|optimizer learning rate weight decay optimizer momentum max tokens learning rate schedule warmup updates max updates dropout (input to encoder) dropout (target features) dropout (transformer) layer dropout feature grad mult latent temp max sample size<br><br>|Adam 5e-4 (B), 3e-4 (L) 0.01 𝛽1=0.9, 𝛽2=0.98 1400000 (B), 1200000 (L) polynomial decay 32000 (B), 20000 (L) 400000 (B), 250000 (L) 0.1 0.1 0.0 (B), 0.1 (L) 0.05 (B), 0.2 (L) 0.1 [2,0.5,0.999995] (B), [2.0,0.1,0.999995] (L) 250000 (B), 320000 (L)|

Table 19 Training Configurations of wav2vec 2.0.

DNA models. For both the HyenaDNA model (Nguyen et al., 2023) and the Caduceus model (Schiff et al., 2024), we directly follow their official implementations without modifying hyperparameters, as shown in Table 20. In particular, Hyena uses LayerNorm and Caduceus uses RMSNorm. For our evaluation, we replace each model’s original normalization layer with Derf and report the average accuracy across all tasks.

|config|value<br><br>|
|---|---|
|optimizer learning rate sequence length effective batch size training steps RC augmentation MLM probability bidirectional<br><br>|AdamW 6e-4 (H), 8e-3 (C) 1024 (H), 131072 (C) 1024 (H), 8 (C) 10000 (H), 50000 (C) true (H), false (C) 0.0 (H), 0.15 (C) false (H), true (C)|

Table 20 Training Configurations of HyenaDNA and Caduceus. H denotes HyenaDNA, C denotes Caduceus.

Language models. For the GPT-2 (124M) model, we follow the hyperparameters as shown in Table 21. For Derf and DyT, we configure the 𝛼 initialization separately for the point-wise function layer following the attention layer and for the other point-wise function layers. We try multiple combinations of these initialization settings and report the best validation loss.

|config<br><br>|value|
|---|---|
|optimizer base learning rate 𝛼 initialization attention 𝛼 initialization other weight decay optimizer momentum gradient clipping block size gradient accumulation steps effective batch size learning rate schedule warmup iterations training iterations dropout mixed precision|AdamW 6e-4 {0.5, 1.0, 2.0, 4.0} {0.1, 0.3, 0.5, 1.0} 0.1 𝛽1=0.9, 𝛽2=0.95 1.0 1024 40 491,520 cosine decay 2,000 300,000 0.0 bf16<br><br>|

Table 21 Training Configurations of GPT-2 (124M).

#### D Additional Results

Beyond evaluating each model with its default normalization layer (typically LN), we additionally test RMSNorm and GroupNorm (GN) to enable a more complete comparison. RMSNorm is widely used in modern large language models, including T5 (Raffel et al., 2020), LLaMA (Touvron et al., 2023a,b; Dubey et al., 2024), Qwen (Bai et al., 2023; Yang et al., 2024), and DeepSeek (Liu et al., 2024; Guo et al., 2025), while GN is employed in several vision architectures, including ConvNeXt (Liu et al., 2022), DETR (Carion et al., 2020), and Swin Transformer (Liu et al., 2021).

All evaluations follow the same experimental settings described in the previous section. These additional results show that Derf not only surpasses the default choices used in each model, but also outperforms the other normalization alternatives we evaluate.

Vision Transformers. For both ViT-Base and ViT-Large (Dosovitskiy, 2021), the default normalization layer is LayerNorm. To complement the results, we also evaluate RMSNorm (Zhang and Sennrich, 2019) and GN (Wu and He, 2018) as additional replacements in Table 22. Compared to all other methods, Derf achieves clearly higher top-1 accuracy, demonstrating its effectiveness in vision transformer architectures.

model LN DyT Derf RMSNorm GN

ViT-B 82.3% 82.5% 82.8% 82.4% 82.5% ViT-L 83.1% 83.6% 83.8% 83.0% 83.1%

- Table 22 Supervised classification accuracy on ImageNet-1K. Derf achieves higher top-1 accuracy than all other methods on different model sizes.

Diffusion Transformers. For DiT models (Peebles and Xie, 2023), we additionally evaluate RMSNorm (Zhang and Sennrich, 2019) as an alternative normalization layer and compare its performance with LN, DyT, and Derf. As shown in Table 23, Derf achieves a clear improvement in FID compared to all other methods.

model LN DyT Derf RMSNorm

DiT-B/4 64.93 63.94 63.23 65.08 DiT-L/4 45.91 45.66 43.94 45.02 DiT-XL/2 19.94 20.83 18.92 20.76

- Table 23 Image generation quality (FID) on ImageNet. Lower FID indicates better image generation quality. Derf achieves lower FID scores than all other methods across different DiT models.

Speech models. For two wav2vec 2.0 Transformer models (Baevski et al., 2020), we additionally evaluate RMSNorm (Zhang and Sennrich, 2019) as an alternative normalization layer and compare its performance with LN, DyT, and Derf in Table 24. Compared to other methods, Derf yields lower validation loss on different model sizes

model LN DyT Derf RMSNorm

wav2vec 2.0 Base 1.95 1.95 1.93 1.95 wav2vec 2.0 Large 1.92 1.91 1.90 1.93

- Table 24 Speech pretraining validation loss on the LibriSpeech dataset. Derf achieves lower validation loss than all other methods across two wav2vec 2.0 models.

DNA models. For the HyenaDNA model (Nguyen et al., 2023) and the Caduceus model (Schiff et al., 2024), we additionally evaluate both LayerNorm and RMSNorm for each architecture, regardless of their default choices, and compare their performance with DyT and Derf in Table 25.

Language models. For the GPT-2 (124M) model, we additionally evaluate RMSNorm (Zhang and Sennrich,

model LN DyT Derf RMSNorm

Hyena 85.2% 85.2% 85.7% 85.2% Caduceus 87.0% 86.9% 87.3% 86.9%

- Table 25 DNA classification accuracy on the GenomicBenchmarks dataset, averaged over each subtask. Derf consistently outperforms other methods across two different DNA models.

2019) for a more complete comparison of normalization choices. As shown in Table 26, Derf achieves comparable performance to both LN and RMSNorm, while clearly outperforming DyT.

model LN DyT Derf RMSNorm GPT-2 2.94 2.97 2.94 2.95

- Table 26 GPT-2 validation loss on the OpenWebText dataset. Derf matches the performances of both LayerNorm and RMSNorm while achieving lower validation loss than DyT.

#### E Loss Calculation Details

Vision Transformers. For ViT models, we measure fitting capacity under a deterministic evaluation setup. We switch the model to evaluation mode, disable drop-path, mixup, cutmix, label smoothing, and all data augmentations, and apply only the standard test-time preprocessing (center crop and normalize). The cross-entropy loss is then computed on the training set and averaged over all samples.

Diffusion Transformers. For DiT models, we evaluate fitting capacity by switching the model to evaluation mode. We apply the standard test-time preprocessing (center crop, random horizontal flip, and normalize). Since DiT does not employ drop-path, no stochastic regularization needs to be disabled. We then compute the diffusion MSE loss over the first 100 training batches and report the average.

Other models. For all other models, wav2vec 2.0, HyenaDNA, Caduceus, and GPT2, we simply apply the same procedure: use the standard test-time preprocessing, disable drop-path or dropout when present, and compute the training loss over the full training set, reporting the average.

#### F Computation Cost

We implement DyT, Derf, and LayerNorm using custom Triton kernels and measured forward and backward runtimes of a single norm layer operation under a unified setup. All operators use comparable fusion levels, memory access patterns, and precision, and are benchmarked using the same batch size and sequence length, across a range of hidden dimensions. As shown in Figure 13 and Table 27, Derf and DyT exhibit nearly identical runtime in both forward and backward passes across all practical hidden dimensions. In the forward pass, Derf and LayerNorm have comparable runtime across dimensions. In the backward pass, Derf can be slightly slower than LayerNorm at small hidden dimensions, but becomes consistently faster as the dimension increases, which aligns with the growing reduction and synchronization overhead in norm layers.

↓

↓

Figure 13 Forward and backward wall-clock runtime comparison for DyT, Derf, and LayerNorm.

Forward Backward

Derf 0.42 0.73 DyT 0.42 0.73 LN 0.42 0.74

Table 27 Time (ms) for running 15k hidden dimension.

