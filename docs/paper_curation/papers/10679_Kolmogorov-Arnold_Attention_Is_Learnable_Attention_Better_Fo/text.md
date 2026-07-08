# arXiv:2503.10632v3[cs.LG]21Oct2025

## KOLMOGOROV-ARNOLD ATTENTION: IS LEARNABLE ATTENTION BETTER FOR VISION TRANSFORMERS?

Subhajit Maity1 Killian Hitsman2 Xin Li2 Aritra Dutta2,1 1 Department of Computer Science 2 Department of Mathematics University of Central Florida, United States {Subhajit, killian.hitsman, xin.li, aritra.dutta}@ucf.edu

### ABSTRACT

Kolmogorov-Arnold networks (KANs) are a remarkable innovation that consists of learnable activation functions, with the potential to capture more complex relationships from data. Presently, KANs are deployed by replacing multilayer perceptrons (MLPs) in deep networks, including advanced architectures such as vision Transformers (ViTs). Given the success of replacing MLP with KAN, this work asks whether KAN could learn token interactions. In this paper, we design the first learnable attention called Kolmogorov-Arnold Attention (KArAt) for ViTs that can operate on any basis, ranging from Fourier, Wavelets, Splines, to Rational Functions. However, learnable activations in the attention cause a memory explosion. To remedy this, we propose a modular version of KArAt that uses a low-rank approximation. By adopting the Fourier basis into this, Fourier-KArAt and its variants, in some cases, outperform their traditional softmax counterparts, or show comparable performance on CIFAR-10, CIFAR-100, and ImageNet-1K datasets. We also deploy Fourier KArAt to ConViT and Swin-Transformer, and use it in detection and segmentation with ViT-Det. We dissect the performance of these architectures on the classification task by analyzing their loss landscapes, weight distributions, optimizer paths, attention visualizations, and transferability to other datasets, and contrast them with vanilla ViTs. KArAt’s learnable activation yields a better attention score across all ViTs, indicating improved token-to-token interactions and contributing to enhanced inference. Still, its generalizability does not scale with larger ViTs. However, many factors, including the present computing interface, affect the relative performance of parameter- and memory-heavy KArAts. We note that the goal of this paper is not to produce efficient attention or challenge the traditional activations; by designing KArAt, we are the first to show that attention can be learned and encourage researchers to explore KArAt in conjunction with more advanced architectures that require a careful understanding of learnable activations. Our open-source code and implementation details are available on: subhajitmaity.me/KArAt

Keywords Multi-Head Self-Attention · Vision Transformers · Kolmogorov-Arnold Network · Kolmogorov-Arnold Transformers · Kolmogorov-Arnold Attention

### 1 Introduction

Artificial general intelligence has become a rapidly growing research direction, and Kolmogorov-Arnold Network (KAN) [4] marks a remarkable innovation in that. KANs with learnable activation functions can potentially capture more complex relationships and facilitate meaningful interaction between the model and human intuition. After training a KAN on a specific problem, researchers can extract the learned univariate functions that the model uses to approximate complex multivariable functions. By studying these learned functions, researchers can gain insights into the underlying relationships from the data and refine the model. KANs exhibit state-of-the-art performance in finding symbolic function representations [5], continual learning of one-dimensional functions [6].

KANs were integrated with neural network architectures, the primary ones being conventional MLPs or convolution neural networks (CNNs). E.g., [7, 8, 9, 10, 11] combine KAN with CNNs, [12] combine KAN with U-Net, etc. Interestingly, [5] claimed to make the first fairer comparison between KANs and MLPs on multiple ML tasks on

No. of parameters vs. Accuracy on Imagenet-1k

82.5

80.0

77.5

Tiny

DEiT

75.0

Acc@1(%)

Small

ViT+KAN

72.5

Base

DEiT+KAN

ViT

KAT

70.0

67.5

65.0

62.5

0 20 40 60 80 100 120 # Parameters (M)

(a)

| | |
|---|---|
| | |

scaled dot-product

dimension dimension

| | |
|---|---|
| | |

| | |
|---|---|
| | |

- (iii)
- (iv)

| | |
|---|---|
| | |

|dimension|
|---|

(i)

| | |
|---|---|
| | |

scaled dot-product

dimension

dimension

| | |
|---|---|
| | |

|dimension| |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

|dimension|
|---|

(ii)

(b)

- Figure 1: (a) Model parameters vs. Top-1 accuracy in ImegeNet-1K training of vanilla ViTs [1], Vision KAN (DeiT+KAN) by [2], ViT+KAN and Kolmogorov-Arnold Transformer (KAT) by [3].(b-i) The traditional softmax attention. (b-ii) The KolmogorovArnold Attention (KArAt) replaces the softmax with a learnable operator, Φi,j. (b-iii) Regular KArAt uses an operator matrix, Φi,j with N2 learnable units acting on each row of Ai,j, and is prohibitively expensive. (b-iv) Modular KArAt uses an operator Φi,j ∈ RN×r with r ≪ N, followed by a learnable linear projector W ∈ Rr×N.

small-scale datasets. By setting the same parameter counts in both networks, [5] concluded that KANs with BSpline basis significantly underperform compared to their MLP counterparts in all ML tasks except symbolic formula representation; [13] reinstates a similar claim for vision. However, KAN’s exploration of more advanced architectures, such as Transformers, remains limited. [2] replace the MLP layers inside the encoder blocks of data-efficient image Transformers (DeiTs) [14] with KANs and proposed DeiT+KAN, [3] proposed two variants: ViT+KAN that replaces the MLP layers inside ViT’s encoder blocks, and Kolmogorov-Arnold Transformer (KAT), albeit similar to ViT+KAN but with a refined group-KAN strategy.

While KANs were claimed to be not superior to simple MLPs in vision tasks [5, 13], we were curious to find out how ViTs can see images through the lens of KANs. In that effort, Figure 1a demonstrates the performance of vanilla ViTs (-Tiny, -Small [15, 16], and -Base [1]) and their KAN counterparts on ImageNet-1K [17] training. While different variants of DeiT+KAN [2] and ViT+KAN [3] show a 5%–18% drop in the Top-1 accuracy compared to vanilla ViTs and DeiTs in ImageNet-1K, the KATs [3] show approximately 1%–2.5% gain in Top-1 accuracy compared to vanilla ViTs and DeiTs by keeping about the same number of parameters. These results show KANs require further investigation in more advanced architectures such as Transformers. Nevertheless, first, we want to understand why there is a discrepancy in these two seemingly equivalent implementations.

Searching for an answer, we realized that while DeiT+KAN and ViT+KAN replace the MLP layer with KAN in the Transformer’s encoder block, KAT implements a sophisticated group-KAN strategy that reuses a learnable function among a group of units in the same layer and chooses different bases for different encoder blocks. This strategy improves accuracy by keeping almost the same parameter counts. Therefore, simply replacing MLPs with KANs might not guarantee better performance, but a properly designed KAN could. But, can we only learn attention?

Irrespective of the tasks or datasets at hand, traditional multihead attention, the heart of the Transformers, employs a softmax non-linearity as a one-size-fits-all mechanism for token interactions without the necessary justification for its use. We hypothesize that adopting learnable KAN operators could be a better fit for approximating the underlying non-linear relationship between image patches, providing the flexibility for modeling token interactions in ViTs. Therefore, in the rise of next-generation Transformers, such as Show-o [18], RL-VLM-F [19], Google’s TITAN [20], and SAKANA AI’s Transformer2 [21], that mimic the human brain, we ask: How can we deploy learnable multi-head self-attention to the (vision) Transformers?

We note that there is a line of research that proposes efficient attention mechanisms in terms of sparse attention [22, 23, 24, 25, 26, 27, 28] or linear/kernelized attention [29, 30, 31, 32, 33, 34], to remedy computational and memory complexities of attention calculation. In contrast, this paper tries to understand how learnable multi-head self-attention (MHSA) modules can perform over regular self-attention used in ViTs, a defining technology in vision in the last six years. Taken together, our contributions are:

Designing a Learnable Attention Module for ViTs. This pilot study is on the self-attention in ViTs. In §3, we propose a general learnable Kolmogorov-Arnold multi-head self-attention, or KArAt, for ViTs that can operate on any basis.

Due to the memory explosion, any full-rank learnable MHSA operator in ideal KArAt causes a training bottleneck. We propose a low-rank approximation of the operator to reduce memory requirements to a feasible level, but leave the possibility of exploring alternative techniques. By adopting the Fourier basis into this modular version, we design Fourier KArAt; see §4 for details. We cannot perform tasks with varying sequences and cross-attention without major modifications to KArAt. The most prominent modalities other than vision, such as audio, speech, video, and text, have varying sequence length, involve modeling with cross-attention-based encoder-decoder architectures, for which token interactions would be substantially different from self-attention. These are non-trivial technical challenges for implementing KArAt.

Benchmarking, Evaluation, and Analysis (§5-§6). We benchmark Fourier KArAt on CIFAR-10, CIFAR-100, [35] and ImageNet-1K [17] datasets and evaluate their performance against vanilla ViTs (ViT-Tiny, -Small [15, 16], and -Base [1]), and other popular ViTs (ConViT [36] and Swin-Transformer [37]). We use small-scale datasets, SVHN [38], Oxford Flowers 102 [39], and STL-10 [40] for understanding the transfer learning capability of the models. Additionally, we embed KArAt with ViT-Det [41] for the object detection and segmentation on MS COCO [42]. We dissect KArAts’ performance and generalization capacity by analyzing their loss landscapes, weight distributions, optimizer path, and attention visualization, and compare them with traditional softmax attention in vanilla ViTs. In addition, we investigate KArAts’ explanability for visual understanding; see §5.3.

### 2 Background

Multi-layer Perceptrons (MLPs) consist of L layers. By convention, a[0] = x[0] denotes the input data and the lth-layer is defined for input x[l−1] ∈ Rd

l−1 as a[l] = Φ[l](Wlx[l−1] + bl), Wl ∈ Rd

l×dl−1,bl ∈ Rd

.

l

In MLP, the nonlinear activation functions, Φ(·), are fixed [43]. For supervised tasks, given a training dataset D with N elements of the form (input, ground-truth) pairs, {(xi,yi⋆)}Ni=1, the loss, L(X,Y ⋆|W) = dY (a[L],Y ⋆) (metric induced by the space Rd

L) is calculated in the forward pass. The layerwise weights, {(Wl,bl)}l∈[L] are learned by minimizing L(X,Y ⋆|W).

Kolmogorov-Arnold Network (KAN) [6] is a neural network involving learnable activations parametrized by a chosen set of basis functions defined on a set of grid points or knots. The idea for this network stems from the Kolmogorov-Arnold Representation Theorem.

Theorem 1 (Kolmogorov-Arnold Representation Theorem). [44] For any multivariate continuous function, f : [0,1]n → R, there exists a finite composition of continuous single-variable functions, ϕq,p : [0,1] → R,Φq : R → R

such that f(x) = f(x1,x2,··· ,xn) = 2q=1n+1 Φq np=1 ϕq,p(xp) .

- Theorem 1 describes an exact finite expression using 2 layers, but [6] generalized this representation to multi-layers via KAN; it is analogous to the Universal Approximation Theorem [45]. For input x[0], an L-layer KAN is given as

KAN(x[0]) = ΦL ◦···◦Φl ◦···◦Φ1(x[0]). In each KAN layer, the activation function Φ is learnable, and Φ[l] = [Φ[ijl]] operates on x[l−1] ∈ Rd

l−1 to produce x[l] ∈ Rd

, such that:

l

dl−1

Φ[ijl](x[jl−1]).

x[il] = Φ[il:](x[l−1]) =

j=1

Originally, [6] chose the B-spline basis functions. That is, the activation functions were defined to be ϕ(x) = w(b(x) + spline(x)), where b(x) = SiLU(x) = 1+xe−x and spline(x) =

ciBi(x), with Bi(·) being one of the

i

k-th degree B-spline basis functions parametrized by the G grid points on a uniform grid, [−I,I]. In this case, the representation for each function is given as:

G+k−1

Φ[ijl](x[jl−1]) = wij[l](SiLU(x[jl−1]) +

c[ijml] Bijm(x[jl−1])).

m=1

The weights {([c[ijml] ],[wij[l]])}Ll=1 are learned by minimizing the loss for a supervised learning task. The bases for KANs’ activations could be Fourier [46, 47], wavelet [48], fractals [3], etc.

Multi-head self-attention (MHSA) in ViTs [1]. The encoder-only vanilla ViT architecture is inspired by the Transformer proposed in [49]. For simplicity, we do not mention the details of the layer normalization and other technicalities. Our central focus is the MHSA architecture. For limited space, we move this discussion to §A.

### 3 How Can We Design Learnable Attention?

Last year, we witnessed a surge in embedding KANs in different DNN architectures [50, 7, 12]. To our knowledge, the work closely related to ours is KAT, where [3] replaces the MLP layers in the ViTs with KAN layers; also, see [2]. We note that, for attentive graph neural networks (GNNs), [51] unifies the scoring functions of attentive GNNs and names it as Kolmogorov-Arnold Attention (KAA). However, it is orthogonal to our MHSA design; deploying learnable MHSA is complicated. In this Section, we design learnable multi-head self-attention (MHSA) module for ViTs, with improved interpretability, adaptability, and expressiveness that operate inside its encoder blocks.

Let Ai,j ∈ RN×N be the attention matrix for ith head in the jth encoder block. Instead of using the softmax function row-wise, we can use a learnable activation function, σ˜ on the row vectors of each attention head Ai,j. With any choice

of the basis functions (e.g., B-Spline, Fourier, Wavelets, etc.), the activated attention row vector, σ˜(Ai,jk,:) for k ∈ [N] can be written as

  

  





- Ai,jk,1
- Ai,jk,2

ϕ11(·) ϕ12(·) ... ϕ1N(·) ϕ21(·) ϕ22(·) ... ϕ2N(·)

⊤

σ˜ (Ai,jk,:) =

= Φi,j (Ai,jk,:)⊤

, (1)

... .

. .

. Ai,jk,N

 

 

ϕN1(·) ϕN2(·) ... ϕNN(·)

where Φi,j = [ϕi,jpq] ∈ RN×N and each matrix entry, ϕpq, is referred to as a learnable unit, represented using a set of basis functions {ψmi,j}m=1. The coefficients associated with the basis functions are the learnable parameters. In our convention, Ai,jk,: ∈ R1×N is a row vector. We apply the transpose operation to each row vector to adopt the convention used in KAN layers. Hence Φi,j (Ai,jk,:)⊤ ∈ RN×1, and we transpose it to obtain learnable attention row vectors σ˜ (Ai,jk,:) ; see Figures 1b(ii)-(iii).

Projection onto the probability simplex. The softmax function acts on each row of A to create a probability distribution; the learnable attention does not guarantee that. To ensure that each row vector of Φ(A) lies on a probability simplex, we project them onto the ℓ1-unit ball. That is, for each attention matrix, Ai,j ∈ RN×N, and for each k ∈ [N],

we want to have ∥σ˜ (Ai,jk,:) ∥1 = 1 and σ˜ (Ai,jk,l) ≥ 0 for l ∈ [N]. We cast a sparse approximation problem, whose variants have been well-studied in the past decade and arise frequently in signal processing [52, 53, 54] and matrix approximation [55, 56]; see §3. Algorithm 1 provides an ℓ1-projection pseudocode; for each k ∈ [N], setting

y = σ˜ (Ai,jk,:) and m = N in Algorithm 1, we obtain the projected attention vector.

### 4 Our Architecture

Different from learning activations for MLP, as in KAN and KAT, implementing the generic learnable attention incurs impractical computational and memory costs on present computing hardware and DL toolkits. For learnable activation,

σ˜ : RN → RN, a full-rank operator, Φi,j ∈ RN×N acts on each row Ai,jk,: of Ai,j, and produces σ˜(Ai,jk,:). These operations are extremely compute-heavy and have large memory footprints. E.g., ViT-Tiny [1, 15, 16] has 5.53M parameters and requires nearly 0.9 GB of GPU memory to train. Implementing the learnable attention equation 1 in ViT-Tiny training with B-splines of order 5 and grid size 10 and Fourier basis with grid size 10 increases the parameter count to 30.68M and 39.06M, respectively. We could not train the version with B-splines for its humongous memory requirements, and B-spline computations are non-parallelizable. The one with the Fourier basis is parallelizable, but it takes approximately 60 GB of GPU memory when computing with a batch of one image. This computational bottleneck is agnostic of the basis functions.

How can we remedy this? Deep neural networks exhibit low-rank structure [57, 58]. Recently, [59] showed that across various networks, the training largely occurs within a low-dimensional subspace. Therefore, we postulate that the attention matrices also have an underlying low-rank structure.

To validate this, we perform spectral analysis of the attention matrices before and after the softmax activation; see Figures 5a-5b. The scree test [60] shows that for traditional attention matrices without softmax, there are 8 significant singular values; after softmax activation, this count increases to 16. The above observation verifies that with or without the nonlinear activation, the attention matrix Ai,j possesses an underlying low-rank structure. This motivates us to use a lower-dimensional operator, Φ for the learnable attention calculation with r = 12. In §C.4.2-C.4.3, we perform an ablation to find the best r.

th head

th head th head th head

th head

th head

Encoder Block

Encoder Block Encoder Block Encoder Block

Encoder Block

Encoder Block

(a) (b)

- Figure 2: Different configurations to update Φ:(a) Blockwise configuration, where Φi,1 ̸= Φi,2 ̸= · · · ̸= Φi,L for all i = 1, 2, ..., h; (b) universal configuration, where Φi,1 = Φi,2 = · · · = Φi,L = Φi for all i = 1, 2, ..., h.

Instead of Φi,j ∈ RN×N, we use an operator, Φi,j ∈ Rr×N such that r ≪ N, and the new learned activation is rdimensional for k ∈ [N]. This process significantly reduces the computational overhead. Next, we post-multiply another learnable weight matrix, Wi,j ∈ RN×r to project them back to their original dimension. For each k ∈ [N],

⊤

this operation results in computing σ(Ai,jk,:) = Wi,j Φi,j (Ai,jk,:)⊤

(Figure 1b(iv)); see other variants in §C.5. Our approximation is not unique or the best; it is one of the feasible solutions given the present computing interface.

Fourier Kolmogorov-Arnold Attention. Although the default basis for KANs is B-Splines in [6], the number of MHSA parameters is much more than that of an MLP. Specifically, for L encoder blocks, each with h attention heads, the parameter complexity is O(N2Lh(G + k)), for Φ ∈ RN×N and O(NrLh(G + k)), for Φ ∈ Rr×N, if the model uses B-Splines of degree k.

Moreover, as [3] mentioned, B-splines are localized functions, not standard CUDA functions. Although efficient CUDA implementations for cubic B-Splines exist [61, 62], their overall implementation results in slower, sparse, non-scalable, and complicated GPU execution; also, see [47, 63]. In §C.4.1, our extensive experiments with B-spline basis in KAN for image classification tasks verify their poor generalizability on medium-scale datasets (e.g., CIFAR-10 and CIFAR-100).

So, what could be an attractive basis? A fundamental question in function approximation is whether the function converges pointwise almost everywhere. Carleson in 1966 proved the following fundamental result for the Fourier approximation of Lp periodic functions.

- Theorem 2. [64] Let f be an Lp periodic function for p ∈ (1,∞], with Fourier coefficients fˆ(n). Then limN→∞ |n|≤N fˆ(n)einx = f(x), for almost every x.

In particular, if a function is continuously differentiable, its Fourier series converges to it everywhere. In the past, [47, 46] used the Fourier basis in KAN, [65] proposed a Fourier analysis network (FAN) to capture periodic phenomena. Motivated by these, we use the Fourier basis to approximate the effect of the smooth softmax function. We employ the Fourier basis, {sin(·),cos(·)} with gridsize G to design learnable attention. Algorithm 1 can be used

optionally. For Φi,j = [ ϕi,jpq] ∈ Rr×N, we have Φi,j : RN → Rr. Hence, each row Ai,jk,:, for k ∈ [N], transformed into Φi,jp (Ai,jk,:)⊤ = Nq=1 ϕi,jpq(Ai,jk,q), via the Fourier bases such that

ϕi,jpq(Ai,jk,q) = Gm=1 apqm cos(mAi,jk,q) + bpqm sin(mAi,jk,q), (2)

where m refers to the mth grid point on a uniform grid of size G. The weights, [{[apqm],[bpqm]}Gm=1,Wi,j] are updated in the backpropagation. If we use a Fourier basis in Φ ∈ Rr×N, the parameter complexity of MHSA becomes O(2NrhGL). Regardless of the basis used, the linear projection takes O(N2rhL) FLOPs. For any Transformer, L and h are fixed. So, one may omit them from the complexity results. See FLOPs and memory requirement in Table 13.

Blockwise and Universal operator configuration. We consider two configurations for updating the operator Φi,j. (a) Blockwise: In this configuration, each encoder block learns the attention through h distinct operators Φ for each of the h heads, totaling hL operators; see Figure 2 (a). Like the MHSA in vanilla ViTs, the blockwise configuration is designed to learn as many different data representations as possible. (b) Universal: This is motivated from the KAT [3]. In KAT, the MLP head is replaced with different variations of a KAN head—KAN and Group-KAN. In Group-KAN, the KAN layer shares rational base functions and their coefficients among the edges. Inspired by this, in our update configuration, all L encoder blocks share the same h operators; Φi,j = Φi for j = 1,2,...,L; see Figure 2-(b). Rather than learning attention through hL operators, this configuration only uses h operators. Here, we share all learnable units and their parameters across L blocks in each head. We postulate that the blockwise mode with more operators captures

Table 1: Performance of the best-performing Fourier KArAt models compared to the conventional vanilla ViT baselines. The best and the second-best Top-1 accuracies are given in red and blue, respectively. The ↓ and ↑ arrows indicate the relative loss and gain, respectively, compared to the base models.

CIFAR-10 CIFAR-100 ImageNet-1K

Model

Parameters Acc.@1 Acc.@5 Acc.@1 Acc.@5 Acc.@1 Acc.@5

ViT-Base 83.45 99.19 58.07 83.70 72.90 90.56 85.81M

+ G1B 81.81 (1.97%↓) 99.01 55.92(3.70%↓) 82.04 68.03(6.68%↓) 86.41 87.51M (1.98% ↑) + G1U 80.75(3.24%↓) 98.76 57.36 (1.22% ↓) 82.89 68.83 (5.58%↓) 87.69 85.95M (0.16% ↑)

ViT-Small 81.08 99.02 53.47 82.52 70.50 89.34 22.05M

+ G3B 79.78 (1.60%↓) 98.70 54.11 (1.20%↑) 81.02 67.77 (3.87%↓) 87.51 23.58M (6.94%↑) + G3U 79.52(1.92%↓) 98.85 53.86(0.73%↑) 81.45 67.76(3.89%↓) 87.60 22.18M (0.56%↑)

ViT-Tiny 72.76 98.14 43.53 75.00 59.15 82.07 5.53M

+ G3B 76.69(5.40%↑) 98.57 46.29(6.34%↑) 77.02 59.11 (0.07%↓) 82.01 6.29M (13.74%↑)

+ G3U 75.56 (3.85%↑) 98.48 46.75(7.40%↑) 76.81 57.97(1.99%↓) 81.03 5.59M (1.08%↑)

Table 2: Performance of Fourier KArAt fine-tuned on small datasets from ImageNet-1K pre-trained weights.

Acc.@1 SVHN Flowers 102 STL-10

Model

ViT-Base 97.74 92.24 97.26 + G1B 96.83 89.66 95.30 + G1U 97.21 89.43 95.78

ViT-Small 97.48 91.46 96.09 + G3B 97.04 89.67 95.26 + G3U 97.11 90.08 95.45

ViT-Tiny 96.69 84.21 93.20 + G3B 96.37 83.67 93.09 + G3U 96.39 83.70 92.93

more nuances from the data. In contrast, the universal mode is suitable for learning simpler decision boundaries from the data. We also note that these two modes can be used with equation 1 as shown in Figure 1b. Finally, Algorithm 2 in the Appendix provides the pseudocode for applying Fourier-KARAT in the ViT encoder block; see more nuanced discussion about them in §C.4.6.

### 5 Empirical Study

Implementing KArAt. We provide the implementation details in §C.1; see §C.1.1 for baselines and datasets. We integrate 5 wavelet bases, rational basis, Fourier basis, and 3 different base activation functions in our learnable MHSA design; see §C.1.3. Fourier KArAt performs the best, and we report it in the main paper. It has 1 hyperparameter, the grid size, G, and 2 configurations, blockwise or universal. We vary grid sizes from the set G = {1,3,5,7,9,11}. For each value, we perform universal and blockwise updates; see Figure 2. With this formalization, GnB and GnU denote Fourier-KArAt with grid size n and weights updated in blockwise and universal mode, respectively. See the ablation study of these variants in §C.5. We dispense the ℓ1 projection, as projecting the learned attention vectors to a probability simplex degrades the performance; see §C.4.5.

##### 5.1 Trained Model Quality, Transferability, and Generalizability

- (i) Image Classification. Table 1 presents the performance of the best-performing Fourier KArAt variants; see Figure 6 for the training loss and test accuracy curves. Fourier KArAt is easily optimized in the initial training phase, gaining at par or better loss value and accuracy than the vanilla ViTs. However, in the later training phase, except for the fewer parameter variants (ViT-Tiny+Fourier KArAt), the loss curve flattens faster; we infer that for most of the Fourier-KArAt scenarios. We also notice that larger models like ViT-Base are more susceptible to slight changes in the parameters, making the training process challenging to manage, which we try to investigate in the rest of the paper using several analytical studies. Overall, ViT-Tiny+Fourier KArAt variants outperform ViT-Tiny on CIFAR-10 and CIFAR-100 by 5.40% and 7.40%, respectively, and on ImageNet-1K, it achieves a similar accuracy. ViT-Small and -Base models with Fourier KArAt variants can barely outperform the vanilla models, and the accuracy differences increase as we tackle larger base models.
- (ii) Performance on other ViTs for Image Classification. We fused KArAt with modern ViT-based architectures, ConViT [36] and Swin Transformer [37]; §C.1.4 provides their implementation details. Table 3 shows that Fourier KArAt outperforms ConViT on CIFAR-10, experiences a modest drop on ImageNet-1K, and shows lower performance for hierarchical models, such as Swin, which requires a careful investigation.
- (iii) Transferability: Image Classification. Unlike ViTs, which were pre-trained on large datasets and then transferred to various mid-sized or small image recognition benchmarks [1], which helped them achieve state-of-the-art results, KArAts were never pre-trained with such datasets to showcase their full potential. Due to limited computing resources, we could not perform large-scale training. We investigate the KArAts’ transfer capability by fine-tuning on smaller datasets like SVHN [38], Oxford Flowers 102 [39], and STL-10 [40] from their ImageNet-1K pre-trained weights; see dataset details in §C.1.1. Table 2 shows KArAts transfer well across all datasets and have comparable performance with vanilla ViTs, even when their ImageNet-1K performance was not equivalent.
- (iv) Generalizability: Detection & Segmentation. To understand the robustness of the Fourier KArAt, we employ ViT

Table 3: Performance comparison of Fourier-KArAt and softmax attention on various vision transformer architectures.

CIFAR-10 ImageNet-1K Acc.@1 Acc.@5 Acc.@1 Acc.@5

Model

ConViT-Tiny 71.36 97.86 57.91 81.79 + G3B 75.57 98.61 56.57 80.75 + G3U 74.51 98.63 56.51 80.93

Swin-Tiny 84.83 99.43 76.14 92.81 + G3B 79.34 98.81 73.19 90.97

Table 4: Attention Transfer from ViT+Fourier-KArAt to vanilla ViTs, following [66].

Attention CIFAR-10 CIFAR-100 Transfered From Acc.@1 Acc.@5 Acc.@1 Acc.@5

Model

ViT-Tiny None 72.76 98.14 43.53 75.00 ViT-Tiny+G3B None 76.69 98.57 46.29 77.02 ViT-Tiny ViT-Tiny+G3B 77.94 98.71 48.91 78.42

ViT-Base None 83.45 99.19 58.07 83.70 ViT-Base+G1B None 81.81 99.01 55.92 82.04 ViT-Base ViT-Base+G1B 81.34 98.70 55.33 80.80 ViT-Base ViT-Tiny+G3B 80.39 99.02 56.11 82.44

backbones for detection and segmentation; see results and discussion in §C.2.

Key Takeaways. (a) The Top-1 accuracies from Tables 1 and 2 show Blockwise configuration is more desirable over Universal for image classification. (b) From Table 2, we can infer that the Fourier KArAt transfers well in fine-tuning tasks. However, the transferability of hyperparameters, e.g., grid size G, across datasets remains an open question. (c) KArAt generalizes well. For random initialization in detection, the vanilla ViT-Det has an advantage over KArAt, and thus, we see a small performance gap. The best KArAt hyperparameters for this task are yet to be found. Overall, KArAt shows significant potential if the incompatibilities are properly addressed.

- 5.2 Performance Analysis We dissect the performance of Fourier KArAt using the following tools:

- (ii) Loss Landscape. We visualize the loss landscape of KArAts to understand why they converge slowly in the later training phase, and why their generalizability does not scale with larger models. Following [67], we plot the loss surface along the directions in which the gradients converge; see §C.1.6 for implementation. Figure 3 shows that Fourier KArAt in ViT architectures significantly impacts the smoothness of the loss surfaces. ViT-Tiny, with the fewest parameters, has the smoothest loss landscape and modest generalizability. In contrast, ViT-Tiny+Fourier KArAt’s loss landscape is spiky; it indicates the model is full of small-volume minima [68]. However, the model is modest in the number of parameters, so the gradient descent optimizer can still find an optimized model with better generalizability than the vanilla ViT-Tiny; hence, it gains a better test accuracy; see Table 1. ViT-Base, however, has more parameters than Tiny, and its loss surface is much spikier than ViT-Tiny. Finally, the loss surface of ViT-Base+Fourier KArAt is most spiky, making it a narrow margin model with sharp minima in which small perturbations in the parameter space lead to high misclassification due to their exponentially larger volume in high-dimensional spaces; see Figure 3(d). Moreover,

ViT-Base+G3B has 14 times more parameters than ViT-Tiny+G1B. With learnable activations, gradient descent optimizers fail to find the best-optimized model, as small differences in the local minima translate to exponentially large disparities. The increasing number of local minima in the later phases of the training causes the slow convergence.

- (iii) Attention Visualization. MHSA in vanilla ViTs captures region-to-region interaction in images. DINO [69] provides an innovative way to explain the dominant regions in an image that contribute towards the inference decision. It maps the self-attention values of the CLS token at the last encoder layer, which is directly responsible for capturing the information related to the class label. While the computer vision community predominantly uses this attention visualization, it is unsuitable for our case. Unlike softmax, Fourier KArAt does not inherently ensure learned attention

values in [0,1], as we dispense the ℓ1 projection. Therefore, we cannot interpret its performance similarly. Nonetheless, pre-softmax or pre-KArAt values in the attention matrix Ai,j are supposed to capture token-to-token interactions, and we adapt the attention maps to ignore the negative values for the sake of visualization. We use the ImageNet-1K trained models of ViT-Tiny (with traditional MHSA and Fourier KArAt) and plot the attention maps of the last layers in Figures 4 and 11.

- 5.3 KArAt’s Flexibility over Traditional Attention in ViT

Stemming from Kolmogorov-Arnold Representation Theorem (Theorem 1) [44], KANs [6] with learnable activation functions can approximate complex mathematical functions and symbolic representations, and provide an interpretable alternative to MLPs [6, 4, 3]. Precisely, if the target function admits a structure of the composition of L-KAN layers with smooth activation functions, then one can use an L-layer KAN with B-Spline basis to approximate it well [Theorem 2.1 in [6]]. This allows them to model complex relations that lie in the data distribution; KANs can discover the need for a new function whose numerical behavior suggests it may be a Bessel function (Figure 23(d) in [6]), and the authors also show that KANs can discover unknown equations (Figure 4(e)).

[Figure 1]

- Figure 3: 3D-visualization of Loss landscape for ViT-Tiny and ViT-Base along the two largest principal component directions of the successive change of model parameters. KArAt’s loss landscapes are significantly less smooth than those of traditional attention; spiky loss landscapes are undesirable for optimization stability and the generalizability of the resulting model. See Figure 10 for the loss contours and the optimizer trajectory.

[Figure 2]

- Figure 4: Vit-Tiny Attention map visualization. Original images for inference (the left), the attention score (middle), and image regions of the dominant head (Top row: Fourier KArAt, bottom row: traditional MHSA).

By using a similar argument, we hypothesize that KANs could be a better fit for approximating the underlying non-linear relationship between image patches and can be used for modeling token interactions with learnable components. Hence, KArAt can be viewed as an upgrade to vanilla self-attention, offering adaptability and more freedom for modeling token interactions in an interpretable way. KArAt provides the freedom to choose from a multitude of simple, easy-to-interpret, and tunable functions whose linear combination can produce a large class of unknown functions, rather than a fixed and known function, softmax. The traditional softmax attention adds a non-linearity for modeling such interactions in a one-size-fits-all mechanism without any justification. This interpretability is directly reflected in KArAt’s attention maps; see Figures 4 and 11.

Why KArAt’s attention maps are more explainable? DINO [69], being self-supervised, attends to the entire object and obtains richer features and deep visual representations [70]. To attend to the entire object like self-supervision, [71] in their supervised training, optimizes the attention map with additional loss objectives, which improves model interpretability and explanability. KArAt’s attention maps in Figures 4 and 11 capture entire objects without any extra loss, unlike some contours in the traditional softmax attention. It provides improved explanability and visual understanding, compared to spurious cues in supervised ViTs.

To show that KArAt’s attention maps are not just visually better, but the learnable attention leads to effective token interactions modeling, we follow the study by [66]. To understand how the attention modulates the performance of ViTs, [66] considered attention matrices from a masked auto-encoder [72], trained with self-supervision, having superior performance, and used them in a vanilla ViT, ignoring the query and key, and training the rest of the network. This study shows that attention is the most important component in a transformer and can bring improved performance over the vanilla supervised ViT. In the same spirit, we use the precomputed attention matrices from the KArAt models to train the value branch of a vanilla ViT from scratch; key and query branches remain frozen. See the implementation details in §C.1 and the results in Table 4.

ViT-Tiny, trained with attention maps of ViT-Tiny+G3B, outperforms the vanilla ViT-Tiny and surpasses the performance of the ViT-Tiny+G3B for CIFAR-10 and CIFAR-100 datasets. This aligns with [66] as the superior attention maps of KArAt improve the performance, and the model outperforms the KArAt variants as it does not suffer from overparameterization and a spiky loss landscape; see Figure 3. In contrast, ViT-Base+G1B shows inferior performance to vanilla ViT-Base due to overparameterization and a spiky loss landscape, and hence produces a subpar model with poor attention maps. Thus, the ViT-Base trained with attention matrices from ViT-Base+G1B fails to surpass the performance of the vanilla ViT-Base or its KArAt variant. For vanilla ViT-Base, trained with the superior attention

maps of ViT-Tiny+G3B, it could not improve the performance due to the redundancy of attention maps and the lack of attention head diversity [73].

### 6 Discussion and Conclusion

The learnable MHSA design raises many interesting questions regarding KArAt’s computing scalability and the resulting models’ generalizability. Below, we discuss their potential positive (+) and negative (−) implications and encourage researchers to elaborate on them.

Designing KArAt self-attention is not parameter efficient and they are memory hungry (−). Having learnable activations in the MHSA causes a memory explosion on the present GPUs. Hence, calculations with the full-rank operator, Φ ∈ RN×N, are prohibitively expensive, regardless of the basis. Present GPUs and CUDA functions are not optimized for KAN implementation as noted by many before us; see [3, 61, 62, 47, 63]. Due to this, although parameter increase is not always significant (Table 1), optimized Fourier KArAt variants utilize 2.5 − 3× more GPU memory compared to traditional softmax attention. We propose a low-rank approximation to reduce memory requirements to a feasible level (Table 13), but researchers should explore alternative techniques. Interestingly, while there is a significant training time discrepancy between vanilla ViTs and Fourier-KArAts, their inference speeds are comparable on the present compute interface and hardware configuration; see Figure 7d in §C.3. Hence, we hope the future holds more efficient system-level optimization recipes for training KArAt. Additionally, we witnessed KArAts’ inconsistent training behavior with parameter changes in §C.4.4. Each ViT model with particular Fourier-KArAt variants has a typical G value that brings out its best performance; there is no universal G to follow.

KArAts show a better attention score (+) and transfer learning capability (+) for all ViTs, but their generalizability does not scale with larger ViTs (−). Fourier KArAt concentrates high interaction scores on the entire object compared to spurious cues [71] like vanilla ViTs [1, 14]. They exhibit decent transfer learning and generalizability. The distribution of weights guarantees the stability of the training. Hence, by [74], we can postulate that first-order optimization algorithms (e.g., ADAM) can optimize KArAt’s loss landscape. But [74] cannot explain why KArAt’s generalizability does not scale with larger models due to high parameterization. We encourage researchers to investigate these models’ local or global optimality and generalizability through theoretical and empirical studies.

Overparameterized KArAts’ weights lie in a much lower-dimensional subspace (+). Spiky loss landscape with local minima is probably the major impediment towards KArAt’s scalability; see Figure 3. Modern transformers are highly over-parameterized, and their weight matrices show low-rank structures [59]; spectral analysis (§C.7) of KArAt shows its attention matrices possess a better low-rankness than traditional attention (Figure 13). Considering this, properly guided parameter reduction in KArAt can result in an optimized model that generalizes well at scale.

KArAts can provide better interpretability for smaller models (+). KANs are claimed to be more interpretable and accurate than MLPs due to their flexible choice of univariate functions [6]. It is evident from KArAts’ attention maps, which are directly responsible for performance improvement; see §5.3. ViT-Tiny+KArAts outperforms vanilla ViT-Tiny on small-scale datasets, and performs at par on ImageNet-1K (Table 1) and transfer learning (Table 2). In the future, KArAt’s learnable activation makes a case for investigating it to add more explanability in vision tasks and can find more interpretability in smaller models trained in a limited data scenario.

Conclusion. Across our benchmarking, KArAt outperforms traditional MHSA in smaller ViTs; results on larger models are mixed. Additionally, we note that the computing requirement impedes KArAt’s performance—training and fine-tuning these models are expensive. However, learnable activations can make a substantial difference in a model’s interpretability, and these limitations are not KArAt’s weaknesses. At its early stage, KArAt’s decent performance on diverse tasks shows its potential. In the future, one can extend this study beyond ViTs and check the resilience of KArAt in language processing and on large multimodal models [75, 76, 77, 78, 79, 80, 81, 82].

### References

- [1] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, G Heigold, S Gelly, et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In ICLR, 2020.
- [2] Ziwen Chen, Gundavarapu, and WU DI. Vision-KAN: Exploring the Possibility of KAN Replacing MLP in Vision Transformer. https://github.com/chenziwenhaoshuai/Vision-KAN.git, 2024.
- [3] Xingyi Yang and Xinchao Wang. Kolmogorov-Arnold Transformer. In ICLR, 2025.

- [4] Ziming Liu, Pingchuan Ma, Yixuan Wang, Wojciech Matusik, and Max Tegmark. KAN 2.0: Kolmogorov-Arnold Networks Meet Science. arXiv preprint arXiv:2408.10205, 2024.
- [5] Runpeng Yu, Weihao Yu, and Xinchao Wang. KAN or MLP: A Fairer Comparison. arXiv preprint arXiv:2407.16674, 2024.
- [6] Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljaˇci´c, Thomas Y Hou, and Max Tegmark. KAN: Kolmogorov-Arnold Networks. In ICLR, 2025.
- [7] Md Meftahul Ferdaus, Mahdi Abdelguerfi, Elias Ioup, David Dobson, Kendall N Niles, Ken Pathak, and Steven Sloan. KANICE: Kolmogorov-Arnold Networks with Interactive Convolutional Elements. In International Conference on AI-ML Systems, pages 1–10, 2024.
- [8] Alexander Dylan Bodner, Antonio Santiago Tepsich, Jack Natan Spolski, and Santiago Pourteau. Convolutional Kolmogorov-Arnold Networks. arXiv preprint arXiv:2406.13155, 2024.
- [9] Ivan Drokin. Kolmogorov-Arnold Convolutions: Design Principles and Empirical Studies. arXiv preprint arXiv:2407.01092, 2024.
- [10] Diab W Abueidda, Panos Pantidis, and Mostafa E Mobasher. DeepOKAN: Deep operator network based on Kolmogorov Arnold networks for mechanics problems. Computer Methods in Applied Mechanics and Engineering, 436:117699, 2025.
- [11] Zhen Wang, Anazida Zainal, Maheyzah Md Siraj, Fuad A Ghaleb, Xue Hao, and Shaoyong Han. An intrusion detection model based on Convolutional Kolmogorov-Arnold Networks. Scientific Reports, 15(1):1917, 2025.
- [12] Chenxin Li, Xinyu Liu, Wuyang Li, Cheng Wang, Hengyu Liu, Yifan Liu, Zhen Chen, and Yixuan Yuan. U-KAN Makes Strong Backbone for Medical Image Segmentation and Generation. In AAAI, volume 39, pages 4652–4660, 2025.
- [13] Basim Azam and Naveed Akhtar. Suitability of KANs for Computer Vision: A preliminary investigation. arXiv preprint arXiv:2406.09087, 2024.
- [14] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In ICML, pages 10347–10357, 2021.
- [15] Kan Wu, Jinnian Zhang, Houwen Peng, Mengchen Liu, Bin Xiao, Jianlong Fu, and Lu Yuan. TinyViT: Fast Pretraining Distillation for Small Vision Transformers. In ECCV, pages 68–85, 2022.
- [16] Andreas Peter Steiner, Alexander Kolesnikov, Xiaohua Zhai, Ross Wightman, Jakob Uszkoreit, and Lucas Beyer. How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers. TMLR, 2022.
- [17] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A Large-Scale Hierarchical Image Database. In CVPR, pages 248–255, 2009.
- [18] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. In ICLR, 2025.
- [19] Yufei Wang, Zhanyi Sun, Jesse Zhang, Zhou Xian, Erdem Biyik, David Held, and Zackory Erickson. Rl-vlm-f: Reinforcement learning from vision language foundation model feedback. In ICML, 2024.
- [20] Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to Memorize at Test Time. arXiv preprint arXiv:2501.00663, 2024.
- [21] Qi Sun, Edoardo Cetin, and Yujin Tang. Transformer2: Self-Adaptive LLMs. In ICLR, 2025.
- [22] Ali Khaleghi Rahimian, Manish Kumar Govind, Subhajit Maity, Dominick Reilly, Christian Kümmerle, Srijan Das, and Aritra Dutta. Fibottention: Inceptive Visual Representation Learning with Diverse Attention Across Heads. arXiv preprint arXiv:2406.19391, 2024.
- [23] Chulhee Yun, Yin-Wen Chang, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank Reddi, and Sanjiv Kumar. O(n) Connections are Expressive Enough: Universal Approximability of Sparse Transformers. In NeurIPS, volume 33, pages 13783–13794, 2020.
- [24] Han Shi, Jiahui Gao, Xiaozhe Ren, Hang Xu, Xiaodan Liang, Zhenguo Li, and James Tin-Yau Kwok. SparseBERT: Rethinking the Importance Analysis in Self-attention. In ICML, pages 9547–9557, 2021.
- [25] Olga Kovaleva, Alexey Romanov, Anna Rogers, and Anna Rumshisky. Revealing the Dark Secrets of BERT. In EMNLP-IJCNLP, 2019.
- [26] Pengchuan Zhang, Xiyang Dai, Jianwei Yang, Bin Xiao, Lu Yuan, Lei Zhang, and Jianfeng Gao. Multi-Scale Vision Longformer: A New Vision Transformer for High-Resolution Image Encoding. In ICCV, pages 2998–3008, 2021.

- [27] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big Bird: Transformers for Longer Sequences. In NeurIPS, volume 33, pages 17283–17297, 2020.
- [28] Qipeng Guo, Xipeng Qiu, Pengfei Liu, Yunfan Shao, Xiangyang Xue, and Zheng Zhang. Star-Transformer. In NAACL, pages 1315–1325, 2019.
- [29] Dongchen Han, Yifan Pu, Zhuofan Xia, Yizeng Han, Xuran Pan, Xiu Li, Jiwen Lu, Shiji Song, and Gao Huang. Bridging the Divide: Reconsidering Softmax and Linear Attention. In NeurIPS, volume 37, pages 79221–79245, 2025.
- [30] Tan M Nguyen, Tam Nguyen, Long Bui, Hai Do, Duy Khuong Nguyen, Dung D Le, Hung Tran-The, Nhat Ho, Stan J Osher, and Richard G Baraniuk. A Probabilistic Framework for Pruning Transformers Via a Finite Admixture of Keys. In ICASSP, pages 1–5, 2023.
- [31] Tan Nguyen, Vai Suliafu, Stanley Osher, Long Chen, and Bao Wang. FMMformer: Efficient and Flexible Transformer via Decomposed Near-field and Far-field attention. In NeurIPS, volume 34, pages 29449–29463, 2021.
- [32] Tan Nguyen, Minh Pham, Tam Nguyen, Khai Nguyen, Stanley Osher, and Nhat Ho. FourierFormer: Transformer Meets Generalized Fourier Integral Theorem. In NeurIPS, volume 35, pages 29319–29335, 2022.
- [33] Jiachen Lu, Jinghan Yao, Junge Zhang, Xiatian Zhu, Hang Xu, Weiguo Gao, Chunjing Xu, Tao Xiang, and Li Zhang. SOFT: Softmax-free Transformer with Linear Complexity. In NeurIPS, volume 34, pages 21297–21309, 2021.
- [34] Tan Nguyen, Tam Nguyen, Hai Do, Khai Nguyen, Vishwanath Saragadam, Minh Pham, Khuong Duy Nguyen, Nhat Ho, and Stanley Osher. Improving Transformer with an Admixture of Attention Heads. In NeurIPS, volume 35, pages 27937–27952, 2022.
- [35] Alex Krizhevsky, Geoffrey Hinton, et al. Learning Multiple Layers of Features from Tiny Images. 2009.
- [36] Stéphane D’Ascoli, Hugo Touvron, Matthew L Leavitt, Ari S Morcos, Giulio Biroli, and Levent Sagun. ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases. In ICML, volume 139, pages 2286–2296, 2021.
- [37] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows. In ICCV, pages 10012–10022, 2021.
- [38] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Baolin Wu, Andrew Y Ng, et al. Reading Digits in Natural Images with Unsupervised Feature Learning. In NeurIPS Workshop on Deep Learning and Unsupervised Feature Learning, volume 2011, page 4, 2011.
- [39] Maria-Elena Nilsback and Andrew Zisserman. Automated Flower Classification over a Large Number of Classes. In Indian Conference on Computer Vision, Graphics & Image Processing, pages 722–729. IEEE, 2008.
- [40] Adam Coates, Andrew Ng, and Honglak Lee. An Analysis of Single-Layer Networks in Unsupervised Feature Learning. In AISTATS, pages 215–223, 2011.
- [41] Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He. Exploring Plain Vision Transformer Backbones for Object Detection. In ECCV, pages 280–296, 2022.
- [42] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO: Common Objects in Context. In ECCV, pages 740–755, 2014.
- [43] Yoshua Bengio, Ian Goodfellow, Aaron Courville, et al. Deep Learning, volume 1. MIT press Cambridge, MA, USA, 2017.
- [44] Andrey N. Kolmogorov. On the representation of continuous functions of several variables as superpositions of continuous functions of a smaller number of variables. Doklady Akademii Nauk SSSR, 1956.
- [45] Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer Feedforward Networks are Universal Approximators. Neural networks, 2(5):359–366, 1989.
- [46] Ali Mehrabian, Parsa Mojarad Adi, Moein Heidari, and Ilker Hacihaliloglu. Implicit Neural Representations with Fourier Kolmogorov-Arnold Networks. arXiv preprint arXiv:2409.09323, 2024.
- [47] Jinfeng Xu, Zheyu Chen, Jinze Li, Shuo Yang, Wei Wang, Xiping Hu, and Edith C-H Ngai. FourierKAN-GCF: Fourier Kolmogorov-Arnold Network–An Effective and Efficient Feature Transformation for Graph Collaborative Filtering. arXiv preprint arXiv:2406.01034, 2024.
- [48] Zavareh Bozorgasl and Hao Chen. Wav-KAN: Wavelet Kolmogorov-Arnold Networks. arXiv preprint arXiv:2405.12832, 2024.

- [49] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention Is All You Need. In NeurIPS, volume 30, 2017.
- [50] Remi Genet and Hugo Inzirillo. TKAN: Temporal Kolmogorov-Arnold Networks. arXiv preprint arXiv:2405.07344, 2024.
- [51] Taoran Fang, Tianhong Gao, Chunping Wang, YihaoShang, Wei Chow, Lei Chen, and Yang Yang. KAA: Kolmogorov-Arnold Attention for Enhancing Attentive Graph Neural Networks. In ICLR, 2025.
- [52] Kurt Bryan and Tanya Leise. Making Do with Less: An Introduction to Compressed Sensing. SIAM Review, 55(3):547–566, 2013.
- [53] David L Donoho. Compressed sensing. IEEE Transactions on information theory, 52(4):1289–1306, 2006.
- [54] Aritra Dutta. Weighted Low-Rank Approximation of Matrices: Some Analytical and Numerical Aspects. PhD thesis, University of Central Florida, 2016.
- [55] Toby Boas, Aritra Dutta, Xin Li, Katie Mercier, and Eric Niderman. Shrinkage Function And Its Applications In Matrix Approximation. The Electronic Journal of Linear Algebra, 32:163–171, 2017.
- [56] Gilbert W Stewart. On the Early History of the Singular Value Decomposition. SIAM Review, 35(4):551–566, 1993.
- [57] Erkki Oja. Simplified Neuron Model as a Principal Component Analyzer. Journal of Mathematical Biology, 15:267–273, 1982.
- [58] Prateek Jain, Chi Jin, Sham M Kakade, Praneeth Netrapalli, and Aaron Sidford. Streaming PCA: Matching Matrix Bernstein and Near-Optimal Finite Sample Guarantees for Oja’s Algorithm. In COLT, pages 1147–1164, 2016.
- [59] Soo Min Kwon, Zekai Zhang, Dogyoon Song, Laura Balzano, and Qing Qu. Efficient Low-Dimensional Compression of Overparameterized Models. In AISTATS, pages 1009–1017, 2024.
- [60] Raymond B Cattell. The Scree Test For The Number Of Factors. Multivariate behavioral research, 1(2):245–276, 1966.
- [61] Daniel Ruijters and Philippe Thévenaz. GPU Prefilter for Accurate Cubic B-spline Interpolation. The Computer Journal, 55(1):15–20, 2012.
- [62] Daniel Ruijters, Bart M ter Haar Romeny, and Paul Suetens. Efficient GPU-Based Texture Interpolation using Uniform B-splines. Journal of Graphics Tools, 13(4):61–69, 2008.
- [63] Avik Pal and Dipankar Das. Understanding the Limitations of B-Spline KANs: Convergence Dynamics and Computational Efficiency. In NeurIPS Workshop on Scientific Methods for Understanding Deep Learning, 2024.
- [64] Lennart Carleson. On convergence and growth of partial sums of Fourier series. Matematika, 11(4):113–132, 1967.
- [65] Yihong Dong, Ge Li, Yongding Tao, Xue Jiang, Kechi Zhang, Jia Li, Jing Su, Jun Zhang, and Jingjing Xu. FAN: Fourier Analysis Networks. arXiv preprint arXiv:2410.02675, 2024.
- [66] Alexander C Li, Yuandong Tian, Beidi Chen, Deepak Pathak, and Xinlei Chen. On the Surprising Effectiveness of Attention Transfer for Vision Transformers. In NeurIPS, volume 37, pages 113963–113990, 2024.
- [67] Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein. Visualizing the Loss Landscape of Neural Nets. In NeurIPS, volume 31, 2018.
- [68] Ronny Huang, Zeyad Emam, Micah Goldblum, Liam Fowl, Justin K. Terry, Furong Huang, and Tom Goldstein. Understanding Generalization Through Visualizations. In Proceedings on "I Can’t Believe It’s Not Better!" at NeurIPS Workshops, volume 137, pages 87–97, 2020.
- [69] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging Properties in Self-Supervised Vision Transformers. In ICCV, pages 9650–9660, 2021.
- [70] Linus Ericsson, Henry Gouk, and Timothy M Hospedales. How Well Do Self-Supervised Models Transfer? In CVPR, pages 5414–5423, 2021.
- [71] Hila Chefer, Idan Schwartz, and Lior Wolf. Optimizing Relevance Maps of Vision Transformers Improves Robustness. In NeurIPS, volume 35, pages 33618–33632, 2022.
- [72] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked Autoencoders Are Scalable Vision Learners. In CVPR, pages 16000–16009, 2022.
- [73] Tianlong Chen, Zhenyu Zhang, Yu Cheng, Ahmed Awadallah, and Zhangyang Wang. The Principle of Diversity: Training Stronger Vision Transformers Calls for Reducing All Levels of Redundancy. In CVPR, pages 12020– 12030, 2022.

- [74] Jingzhao Zhang, Haochuan Li, Suvrit Sra, and Ali Jadbabaie. Neural Network Weights Do Not Converge to Stationary Points: An Invariant Measure Perspective. In ICML, pages 26330–26346, 2022.
- [75] Meta AI. Introducing Meta Llama 3: The most capable openly available LLM to date. https://ai.meta.com/blog/meta-llama-3/.
- [76] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 Technical Report. arXiv preprint arXiv:2412.15115, 2024.
- [77] Omkar Thawakar, Ashmal Vayani, Salman Khan, Hisham Cholakal, Rao M Anwer, Michael Felsberg, Tim Baldwin, Eric P Xing, and Fahad Shahbaz Khan. MobiLlama: Towards Accurate and Lightweight Fully Transparent GPT. In ICLR Workshop on Sparse LLMs, 2025.
- [78] Ron Campos, Ashmal Vayani, Parth Parag Kulkarni, Rohit Gupta, Aritra Dutta, and Mubarak Shah. GAEA: A Geolocation Aware Conversational Model. In WACV, 2026.
- [79] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language Models are Few-Shot Learners. In NeurIPS, volume 33, pages 1877–1901, 2020.
- [80] Meta AI. Llama 3.2: Vision and edge models. Meta AI Blog, Nov 2024.
- [81] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: A Family of Highly Capable Multimodal Models. arXiv preprint arXiv:2312.11805, 2023.
- [82] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a Visual Language Model for Few-Shot Learning. In NeurIPS, volume 35, pages 23716–23736, 2022.
- [83] Laurent Condat. Fast Projection onto the Simplex and the ℓ1 Ball. Mathematical Programming, 158(1):575–585, 2016.
- [84] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An Imperative Style, High-Performance Deep Learning Library. In NeurIPS, volume 32, 2019.
- [85] Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. In ICLR, 2015.
- [86] Yann LeCun, Corinna Cortes, and CJ Burges. MNIST handwritten Digit Database. ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist, 2, 2010.
- [87] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask R-CNN. In ICCV, pages 2961–2969, 2017.
- [88] Ross Girshick. Fast R-CNN. In ICCV, pages 1440–1448, 2015.
- [89] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. IEEE TPAMI, 39(6):1137–1149, 2016.
- [90] Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature Pyramid Networks for Object Detection. In CVPR, pages 2117–2125, 2017.
- [91] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.
- [92] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv preprint arXiv:1708.07747, 2017.

### Contents

- 1 Introduction 1
- 2 Background 3
- 3 How Can We Design Learnable Attention? 4
- 4 Our Architecture 4
- 5 Empirical Study 6

- 5.1 Trained Model Quality, Transferability, and Generalizability . . . . . . . . . . . . . . . . . . . . . . 6
- 5.2 Performance Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 5.3 KArAt’s Flexibility over Traditional Attention in ViT . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 6 Discussion and Conclusion 9

- A Multihead Self Attention—Continued 15
- B A Sparse Approximation Problem and its Solution 15
- C Addendum to Empirical Study 15

- C.1 Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- C.1.1 Baselines and Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- C.1.2 Small-Scale Datasets used . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- C.1.3 Implementing Different Basis and Base Activation Functions in KArAt . . . . . . . . . . . . 17
- C.1.4 Deploying Fourier KArAt to other ViTs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- C.1.5 Implementing Fourier KArAt for Object Detection and Semantic Segmentation . . . . . . . . 20
- C.1.6 Plotting Loss Landscapes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

- C.2 Generalizability: Detection & Segmentation—Continued . . . . . . . . . . . . . . . . . . . . . . . . 20
- C.3 Computation Time, FLOPS, Memory Requirement, and Throughput of Fourier KArAt . . . . . . . . 20
- C.4 Ablation Study with KAN and Fourier KArAt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- C.4.1 Experiments with KANs Operated on B-Spline Basis . . . . . . . . . . . . . . . . . . . . . . 22
- C.4.2 Ablation with the Hidden Dimension, r in Fourier KArAt . . . . . . . . . . . . . . . . . . . 23
- C.4.3 Enforcing an Extreme Low Rank Structure in Fourier KArAt . . . . . . . . . . . . . . . . . . 23
- C.4.4 Ablation on the Impact of grid size, G on Fourier KArAt . . . . . . . . . . . . . . . . . . . . 25
- C.4.5 Does Fourier KArAt Require ℓ1-Projection? . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- C.4.6 More Discussion on Blockwise and Universal Configuration . . . . . . . . . . . . . . . . . . 28

- C.5 Alternate Variants of Fourier KArAt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29

- C.5.1 Alternative Approaches to the Lower Rank Attention Structures . . . . . . . . . . . . . . . . 29
- C.5.2 Fourier KArAt + softmax Attention —A Hybrid version . . . . . . . . . . . . . . . . . . . . 30
- C.5.3 More Combinations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30

- C.6 Distribution of the Weights . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31

- C.7 Spectral Analysis of Attention . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31

### Appendix

Organization. We organized the Appendix as follows: §B contains the solution to problem equation 3. We also provide the pseudocode to implement Fourier-KArAt in ViT’s encoder block; see Algorithm 2. §C discusses additional numerical results, including implementation details, hyperparameter tuning, computation time analysis, diverse ablation studies, and different variants of Fourier KArAt. Finally, we concluded §C with the spectral analysis of the attention matrices for traditional and learnable attention.

### A Multihead Self Attention—Continued

Let X ∈ RH×W×C be an input image of resolution H × W. ViT uses a p × p patch and generates N = HWp2 input tokens. Each token is vectorized along C channels to produce XT ∈ RN×p

2C, where each row, XTi,:, represents a flattened patch across channels. A learnable embedding matrix, E ∈ Rp

2C×d projects XTi,: to an embedding vector in Rd such that after appending the learnable class token, xG and adding the positional encoding matrix, PE ∈ R(N+1)×d of compatible size, the final input, XI = [xG;XE] + PE. The matrix XI is layer normalized and further projected on the row spaces of three learnable weight matrices, WQ,WK,WV ∈ Rd×d to generate the query, key, and value matrices, Q,K, and V , respectively, via Q = XIWQ,K = XIWK, and V = XIWV . These matrices are further divided into h partitions (also, called heads), Qi,Ki,V i ∈ RN×

d h , such that each partition generates a self-attention

√iKi⊤ d/h

matrix for the ith head, Ai = Q

. Finally, we project V i onto the column space of σ(Ai) as σ(Ai)V i, where σ is the softmax operator applied row-wise. The outputs of different heads are further concatenated into a large matrix

σ(A1)V 1 σ(A2)V 2 ··· σ(Ah)V h WO, post-multiplied by the learnable weight, WO ∈ Rd×d. This process is sequentially performed in L encoder blocks; see Figure 1b(i).

### B A Sparse Approximation Problem and its Solution

Let y ∈ Rm be a given vector. If we want to approximate y with a vector, x ∈ Rm, with positive components and ∥x∥1 = 1, we can write the constrained optimization problem as:

x⋆ = arg minx∈Rm 1

2∥x − y∥2

(3)

subject to xi ≥ 0 for i ∈ [m],and ∥x∥1 = 1.

First, we rewrite the constrained problem equation 3 as an unconstrained problem using the Lagrange multipliers as:

m

- 1

- 2∥x − y∥2 + λ

xi − 1 − µTx, (4)

L(x,λ,µ) =

i=1

where λ,µ are Lagrange multipliers. Using the Karush-Kuhn-Tucker (KKT) stationarity condition on equation 4, we find 0 ∈ ∂L(x,λ,µ), which implies, xi − yi + λ = µi. The complementary slackness gives, µixi = 0 for i ∈ [m].

m

xi = 1, and µi ≥ 0, respectively.

Further, for i ∈ [m], the primal and dual feasibility conditions are xi ≥ 0,

i=1

The stationarity and complementary slackness conditions give, xi − yi + λ ≥ 0, (xi − yi + λ)xi = 0. If λ ≥ yi, then xi = 0. Otherwise, we have that

m

m

m

(yi − λ) = 1 implying λ = m1 (

yi − 1).

xi =

i=1

i=1

i=1

Note that, in Algorithm 1, one can replace y with Φi,j[(Ai,j)⊤k,:] so that the constraint ∥ σ (Ai,jk,:) ∥1 = 1 and σ (Ai,jk,l) ≥ 0 is satisfied for the low-rank attention matrices in Fourier-KArAt. This algorithm is also kept as an optional sub-process for Algorithm 2.

#### C Addendum to Empirical Study This section complements our empirical results in §5.

###### Algorithm 1 Projection on ℓ1 ball [83] Input: y ∈ Rm Sort: y as y(1) ≥ y(2) ≥ ... ≥ y(m)

- i
- j=1

Calculate: ρ ≜ max i ∈ [m]| y(i) − 1i

y(j) − 1

ρ

Set: λ = ρ1

y(i) − 1 Output: x⋆ = (y − λ)+ ≜ max (y − λ, 0)

i=1

- Table 5: Hyper-parameter settings for image classification and recognition experiments conducted in this work for vanilla ViTs and ViT+Fourier KArAts.

Input Size 224 × 224 Crop Ratio 0.9 Batch Size 128 for ImageNet-1K and 32 for CIFAR-10 & CIFAR-100

Optimizer AdamW Optimizer Epsilon 1 × 10−6 Momentum 0.9 Weight Decay 0.05 Gradient Clip 1.0

Learning Rate Schedule Cosine Learning Rate 5 × 10−4 × Batch512Size Warmup LR 1 × 10−6 Min LR 1 × 10−5 Epochs 100 Decay Epochs 1 Warmup Epochs 5 Decay Rate 0.988 Exponential Moving Average (EMA) True EMA Decay 0.99992 Random Resize & Crop Scale and Ratio (0.08,1.0),(0.67,1.5) Random Flip Horizontal 0.5; Vertical 0.0 Color Jittering 0.4 Auto-augmentation rand-m15-n2-mstd1.0-inc1 Mixup True Cutmix True Mixup, Cutmix Probability 0.5, 0.5 Mixup Mode Batch Label Smoothing 0.1 Patch Size 16

- Algorithm 2 Fourier-KArAt in jth Vision Transformer block

Input: X ∈ RN×d, Fourier basis {sin (·), cos (·)} Output: O ∈ RN×d Parameters: WQi , WKi , WVi ∈ Rd×dh, dh = hd, G, [{[apqm], [bpqm]}Gm=1, Wi,j], WO,j Hyperparameters: Blockwise, universal, Algorithm 1 (ℓ1 projection), L encoder blocks for each head i ∈ {1, . . . , h} do:

Qi,j ← XWQi,j Ki,j ← XWKi,j V i,j ← XWVi,j Ai,j = Q

i,j√(Ki,j)⊤ dh

###### end for for each head i ∈ {1, . . . , h} do:

for each row k ∈ {1, . . . , N} do:

Φi,j[(Ai,j)⊤k,:] ← Nq=1 ϕi,jpq (Ai,jk,q) = Gm=1 apqm cos (mAi,jk,q) + bpqm sin (mAi,jk,q) if ℓ1 projection then

Execute Algorithm 1 else

pass

⊤

σ(Ai,jk,:) ← Wi,j Φi,j[(Ai,jk,:)⊤]

end for end for O ← σ(A1,j)V 1,j σ(A2,j)V 2,j · · · σ(AN,j)V N,j WO,j if Blockwise Mode then

Return O

else if universal Mode then pass learnable units and parameters to (j + 1)th encoder block Return O

##### C.1 Implementation Details

We implement the framework in Python using PyTorch [84] and use the training strategy of DEiT [14]. We train all models for 100 epochs with the ADAM optimizer [85] with a base learning rate of 3.125 × 10−5 and batch-size 32, except for the experiments on ImageNet-1K that use a learning rate of 1.25×10−4 with batch-size 128. All experiments use a warm-up of 5 epochs and a cosine scheduler with a weight decay of 5 × 10−2. The experiments were performed on two 80 GB NVIDIA H100 GPUs. The hyperparameter settings for experiments with ViTs are given in Table 5.

##### C.1.1 Baselines and Datasets

For our image classification benchmarking, we chose 5 popular vision Transformers—ViT-Tiny, ViT-Small [15, 16], ViT-Base [1], ConViT-Tiny [36], and Swin-Transformer-Tiny [37]. We incorporate Fourier-KArAt in them by replacing their softmax function with a learnable activation. We perform our benchmarking on CIFAR-10, CIFAR-100 [35] and ImageNet-1K [17] datasets. We use small-scale datasets, SVHN [38], Oxford Flowers-102 [39], and STL-10 [40] for understanding the transfer learning capability of the models. Additionally, we use ViT-Det [41] on the MS COCO [42] dataset for the object detection and segmentation task.

##### C.1.2 Small-Scale Datasets used

We use 3 small-scale datasets, SVHN [38], Oxford Flowers 102 [39], and STL-10 [40] for transfer learning task. SVHN is similar to MNIST [86], a 10-class digit classification dataset with small images like CIFAR-10 [35], divided into 73,257 training and 26,032 test images. The challenge in this dataset comes from the distracting elements surrounding the concerned class. The OxfordFlowers-102 [39] is a small yet challenging dataset with unbalanced training images (40-254 images) for the 102 classes. The STL-10 [40] is a CIFAR-like 10-class dataset with a larger image size, but only 500 training and 800 testing images per class.

##### C.1.3 Implementing Different Basis and Base Activation Functions in KArAt

Theorem 2 uses the Fourier basis to span any Lp periodic function, where p ∈ (1,∞]. With this view, we aim to represent the softmax function using any chosen set of basis functions. Although our KArAt design spans each

Singular Values before Softmax

175

150

125

SingularValue

100

75

50

25

0

0 25 50 75 100 125 150 175 200 Index

Singular Values after Softmax

1.75

1.50

1.25

SingularValue

1.00

0.75

0.50

0.25

0.00

0 25 50 75 100 125 150 175 200 Index

(b) Attention matrix σ(Ai,j) after softmax activation.

(a) Attention matrix Ai,j before softmax activation.

- Figure 5: Spectral analysis of the attention matrices before and after softmax shows that they are low-rank. For this experiment, we use all 3 heads in the last encoder block of ViT-Tiny on 5 randomly sampled images from the CIFAR-10 validation set. We plot all 15 singular vectors (each of 197 dimensions) where the singular values are arranged in non-increasing order.

attention unit as a linear combination of Fourier basis, the originally designed KAN layer [6] includes a residual activation base function, b(x), which was set to SiLU; any other activation function can be used. We refer to b(x) as the base activation function. To see how these base functions influence the performance of Fourier KArAt, we run both blockwise and universal modes on ViT-Tiny+KArAt with Identity1, SiLU, and GELU2 base functions. Similar to equation 2, the attention unit then has the following form:

###### ϕi,jpq(Ai,jk,q) = b(Ai,jk,q) + Gm=1 apqm cos(mAi,jk,q) + bpqm sin(mAi,jk,q). (5)

For the experiments in the main paper, in the general form, equation 5, we use b(x) = 0 for all x ∈ R, grid size, G = 3, and the row dimension of the Φi,j is set to r = 12, or otherwise noted. We encourage the reader to explore other configurations that differ from those previously outlined.

- (i) Different Basis Functions Used and Their Results. We incorporate a variety of widely used KAN activation functions into KArAt. By design, KArAt can utilize any basis function for activating the attention units. Therefore, in addition to the Fourier basis, we embed 5 different wavelet bases, and the Rational Function basis into KArAt. The function representations for these choices of bases are outlined in Table 6.

We use the safe Padé approximation unit (PAU) of order (m,n) for rational function basis; KAT also uses the same basis function [3]. In contrast to PAUs, safe PAUs ensure stability and the prevention of poles, making it a clear choice for rational functions. We only consider safe PAUs of order (5,4) in this paper; other order combinations can be used.

Being an integral part in signal analyses and image reconstruction, wavelet functions serve as a candid choice for approximating signals and feature extraction [48]. Having zero mean and finite energy drives wavelets to make fair and meaningful representations of signals in local regimes. Similar to the Fourier transform, the inverse continuous wavelet transform (CWT) resolves the original signal. We consider mother wavelets, mentioned in Wav-KAN [48], as basis functions. Unique to this choice, KArAt learns scale, s, translation, τ, as measured in the CWT, and a learnable weight coefficient w. For any choice of wavelet base function, ϕ, the input is first shifted by the scale and translation parameters and then activated by the mother wavelet.

Embedding different bases in other ViT variants with KArAt is straightforward. Although our original design in Algorithm 2 represents each matrix entry ϕpq using the Fourier basis, any functions outlined in Table 6 can be used. All other components in Algorithm 2 stay the same.

1Identity(t) = t. 2GELU(t) = tψ(t), where ψ(·) denotes the cumulative distribution function for the Gaussian Distribution.

- Table 6: Different basis functions and their representations for b(x) = 0. * For our experiments, the morlet central frequency hyperparameter is ω0 = 5, but other nonnegative values can be used. ** Meyer is defined to be (m ◦ ν)(˜x) = m(ν(˜x)), where m(t) = I −∞, 12 (t) + I 12,1 (t)cos π2ν(2t − 1) and ν(x) = x4(35 − 84x + 70x2 − 20x3)I[0,1](x).

Basis Function Representation, ϕ(x) Initialized Specifications Fourier

G

k=1

ak cos (kx) + bk sin (kx) ak, bk ∼ N(0, 1), G denotes the grid size

Rational (m, n) a1+0+|ab1x+···+amxm

1x+···+bnxn| ai, bj ∼ N(0, 1), i = 0, . . . , m, j = 1, . . . , n Mexican Hat π1/24w√3 x ˜2 − 1 e−x˜

2

2 , x˜ = x−sτ

w ∼ N(0, 1), (Translation) τ = 0, (Scale) s = 1

Morlet* w cos (ω0x˜)e−x˜

2

2 , x˜ = x−sτ DOG −wdxd e−x˜

2

2 , x˜ = x−sτ

Meyer** w sin (π|x˜|) (m ◦ ν) (˜x), x˜ = x−sτ Shannon wsinc πx ˜ ω(˜x), x˜ = x−sτ ω(˜x) is the symmetric hamming window

- Table 7: Blockwise and universal ViT-Tiny+KArAt and their Top-1 accuracies on CIFAR-10 using different base activation functions (SiLU, Identity, GELU).

Table 8: Blockwise configuration ViT+KArAt and their Top-1 accuracies using different base functions on CIFAR10 and CIFAR-100.

Acc.@1 on CIFAR-10

Acc.@1 CIFAR-10 CIFAR-100

Model Base Activation

Model Basis Function

Identity 75.90 SiLU 76.25 GELU 76.48

Rational (5,4) 69.22 37.93 Mexican Hat 69.62 38.92

ViT-Tiny+G3B

Morlet 70.84 37.88 DOG 73.89 44.23

ViT-Tiny+KArAt

Identity 75.21 SiLU 75.25 GELU 75.28

Meyer 71.13 41.65 Shannon 71.41 39.40

ViT-Tiny+G3U

Rational (5,4) 72.59 41.52 Mexican Hat 72.58 44.44

Identity 74.87 SiLU 75.53 GELU 75.18

ViT-Tiny+G1B

Morlet 24.19 6.22

ViT-Base+KArAt

DOG 72.28 48.69 Meyer 28.83 6.67 Shannon 69.84 39.98

Identity 74.44 SiLU 75.12 GELU 74.82

ViT-Tiny+G1U

We implement rational and wavelet basis activation functions from Table 6 in ViT-Tiny+KArAt and ViT-Base+KARAT using the blockwise configuration and show their performance on the image classification task on CIFAR-10 and CIFAR-100; see the results in Table 8. Performance across model variants is not consistent with the choice of wavelet basis function used. ViT-Tiny+KArAt achieves its highest Top-1 accuracy with DOG (derivative of Gaussian) on CIFAR-10 and -100, whereas Rational (5,4) breaks this trend with ViT-Base+KArAt. Another observation shows that ViT-Base+KArAt performs poorly with both Meyer and Morlet functions, contrary to ViT-Tiny+KArAt. Hence, we cannot make a proper remark regarding the choice of basis.

- (ii) Base Activation Functions Used and Their Results. We implement different base functions in ViT-Tiny+KArAt using the blockwise and universal configurations and show their performance on the image classification task on CIFAR-10; see the results in Table 7. For the majority of our experiments, including those in the main paper, we define and use ZeroModule and set the base activation function to be identically zero, or b(x) = 0. Slightly lower computations within a single KAN layer help motivate us to use this base activation function. Overall, choosing between

the activations SiLU and GELU for ViT-Tiny+KArAt with G3B,G3U modes results in a slightly lower Top-1 accuracy compared to the results using the ZeroModule in the main paper (see Table 1).

##### C.1.4 Deploying Fourier KArAt to other ViTs

We consider three modern ViT-based architectures, including ConViT [36] and Swin Transformer [37]. Incorporating the Fourier KArAt variants into these models is not straightforward. Below, we outline their implementation details.

- (i) ConViT [36] uses two different attention mechanisms, the gated positional self-attention (GPSA) and MHSA. All of the variants of ConViT vary the number of attention heads per encoder while keeping the number of encoders the same. The purpose of the ConViT is to combine a CNN and a ViT in one encoder block. With the first ten layers using GPSA and the last two layers using MHSA, the model learns a balanced structure between CNNs and ViTs. Our variant of ConViT replaces the MHSAs with Fourier KArAts. We implement the universal and blockwise attention modes on this model. We give the ConViT’s hyperparameter configuration details in Table 9.
- (ii) Swin Transformers [37] has a variable number of tokens. The number of tokens considered in the attention operation varies based on the window size, and thus, we do not implement a universal attention mode due to incompatibility. We replace regular and shifted window MHSAs, W-MHSA and SW-MHSA, with regular and shifted Fourier KArAts. We give the Swin-Transformer’s hyperparameter configuration details in Table 10.

##### C.1.5 Implementing Fourier KArAt for Object Detection and Semantic Segmentation

Implementing Fourier KArAt for detection and segmentation is an involved task. Firstly, RCNN [87, 88, 89] frameworks employ an augmentation strategy to train on dynamic image sizes, which also dynamically change the number of tokens and thus, token interactions in the attention. However, KArAt needs to ensure fixed-length input and output; hence, the dynamic resize becomes incompatible. Secondly, vanilla ViT-Det [41] involves window partitioning in the self-attention, making the number of tokens variable in the attention matrices throughout the blocks of encoders. Thirdly, as this window mechanism changes the number of token interactions in the attention matrices, the universal mode of KArAt also becomes incompatible. To solve all these, we restrict the input image size to 224 × 224, and use a window size of 14 and a patch size of 16. This inherently fixes the number of tokens to 196, and the window size of 14 ensures 196 tokens in the windowed attention blocks. Thus, the number of tokens throughout the model remains fixed. We note that the restriction on augmentation and limiting the input size impacts the baseline performance of ViT-Det. The performance analysis is detailed in §5.1 and inference visualization is provided in Figure 8.

##### C.1.6 Plotting Loss Landscapes

Following [67], we perform principal component analysis (PCA) of the parameter change over training progression to understand the major directions of parameter convergence. Considering the fully trained model as the minimum in the loss hyperplane and the two principal component directions as X and Y axes, we plot the loss values over the validation set of CIFAR-10 along the Z axis for ViT-Tiny and -Base for the traditional MHSA and KArAt (G3B and G1B, respectively).

##### C.2 Generalizability: Detection & Segmentation—Continued

We choose Mask RCNN [87]-based framework for this purpose and employ a feature pyramid network [90] based on ViT-Det [41]; it is non-trivial to implement Fourier KArAt, see §C.1.5. We trained ViT-Det [41] using ViT-Base backbone on the MS COCO [42] dataset for 50 epochs with two settings: (i) fine-tuning on the ImageNet-1K pre-trained weights on traditional softmax attention, and (ii) random initialization. Table 12 shows that for the fine-tuning on ImageNet pre-trained weights, Fourier-KArAt shows ∼ 4 − 13% gap in average precision (AP) from its vanilla variant; their qualitative performance in Figure 8 is similar. Overall, for ViT-Det, Fourier-KArAt performs inferiorly to its conventional counterpart in detection and segmentation. Interestingly, Figure 9 shows, vanilla ViT-Det using softmax activation reaches its peak performance quickly (within 12 epochs), Fourier-KArAt delays in achieving it (within 30 and 45 epochs for G1B and G1U, respectively), proving the incompatibility of softmax-based weights initialization. It is, however, still a good initialization for KArAt as the performance is better than random initialization.

##### C.3 Computation Time, FLOPS, Memory Requirement, and Throughput of Fourier KArAt

The overall computation for Fourier KArAt variants is higher than their conventional softmax MHSA, and we have delineated it in Figure 7. Primarily, the Fourier KArAt variants have a longer training time. We show the training time comparison between the traditional MHSA and its Fourier KArAt versions for 100 epochs on CIFAR-10, CIFAR-100, and ImageNet-1K datasets for all the models (ViT-Tiny, -Small, and -Base) in Figures 7a– 7c, respectively. We only compare the best-performing Fourier KArAt models (G1B and G1U for ViT-Base, and G3B and G3U for ViT-Tiny & ViT-Small) with their traditional softmax MHSA counterparts. We also observe that universal mode GnU training times are consistently slightly less than the blockwise modes GnB.

During the training, we monitored the GPU memory requirements, and as expected, Fourier KArAt variants utilize significantly more memory than traditional MHSA. In particular, the GPU memory requirements scale by 2.5 − 3×, compared to the traditional softmax MHSA.

[Figure 3]

- (a) ViT-Tiny

[Figure 4]

- (b) ViT-Small

[Figure 5]

- (c) ViT-Base

- Figure 6: Training loss and test accuracy of vanilla ViTs and their Fourier KArAt versions on CIFAR-10, CIFAR-100, and ImageNet-1K datasets (left to right).

- Table 9: Hyper-parameter settings for image classification conducted in this work for vanilla ConViT-Tiny and ConViT-Tiny+Fourier KArAt.

Input Size 224 × 224 Batch Size 128 for Imagenet-1K & 32 for CIFAR-10

Optimizer AdamW Optimizer Epsilon 1 × 10−8 Momentum 0.9 Weight Decay 0.05 Gradient Clip 1.0

Learning Rate Schedule Cosine Learning Rate 5 × 10−4 × Batch512Size Warmup LR 1 × 10−6 Min LR 1 × 10−5 Epochs 100 Decay Epochs 30 Warmup Epochs 5 Decay Rate 0.1 Exponential Moving Average (EMA) False EMA Decay 0.99996 Locality up to Layer 10 Locality Strength 1 Random Resize & Crop Scale and Ratio (0.08,1.0),(0.67,1.5) Random Flip Horizontal 0.5; Vertical 0.0 Color Jittering 0.4 Auto-augmentation rand-m9-mstd0.5-inc1 Mixup True Cutmix True Mixup, Cutmix Probability 0.5, 0.5 Mixup Mode Batch Label Smoothing 0.1

We also compare the throughput during inference in Figure 7d and see slightly faster inference in universal mode than blockwise, except for ViT-Base. While there is a massive training time discrepancy between vanilla ViTs and Fourier KArAt ViTs, the inference speeds for Fourier KArAt variants are comparable to their vanilla counterparts. Although there is a minor difference in throughput between universal and blockwise modes during inference, theoretically, both variants for any model with the same grid size should have the same number of FLOPs.

##### C.4 Ablation Study with KAN and Fourier KArAt

In this section, we perform a detailed ablation study with different hyperparameter settings. Our first set of experiments shows why B-Splines are not a good basis for KAN for image classification tasks. The experiments with B-Splines solidify our argument for why we primarily use the Fourier basis in KArAt. After that, we perform rigorous ablation studies of Fourier KArAt on the hidden dimension, grid size effects, and many others.

##### C.4.1 Experiments with KANs Operated on B-Spline Basis

We test the performance of the B-spline KANs in the classification task on the CIFAR-10 [35], CIFAR-100 [35], MNIST [91, 86], and Fashion-MNIST [92] datasets. We performed extensive experiments to see if there are any benefits to choosing B-Splines for the basis functions in KAN layers. Table 14 shows the Top-1 accuracies from different variants of a Deep KAN. In this set of experiments, we closely followed earlier works [5, 6]. Hyperparameters involved in these experiments include the order of the B-spline k, grid size G, grid range [−I,I], layers, and width. Although the KANs with B-Spline basis yield high accuracies in the small-scale MNIST and Fashion-MNIST datasets, they fail to generalize over larger datasets (CIFAR-10 and -100).

- Table 10: Hyper-parameter settings for image classification conducted in this work for vanilla Swin-Transformer-Tiny and Swin-Transformer-Tiny+Fourier KArAt.

Input Size 224 × 224 Crop Ratio 0.9 Batch Size 128 for Imagenet-1K & 32 for CIFAR-10

Optimizer AdamW Optimizer Epsilon 1 × 10−8 Momentum 0.9 Weight Decay 0.05 Gradient Clip 1.0

Learning Rate Schedule Cosine Learning Rate 5 × 10−4 × Batch512Size Warmup LR 1 × 10−6 Min LR 1 × 10−5 Epochs 100 Decay Epochs 30 Patience Epochs 10 Warmup Epochs 5 Decay Rate 0.1 Exponential Moving Average (EMA) False EMA Decay 0.99996 Fused Layer Norm False Fused Window Process False Window Size 7 Random Resize & Crop Scale and Ratio (0.08,1.0),(0.67,1.5) Random Flip Horizontal 0.5; Vertical 0.0 Color Jittering 0.3 Auto-augmentation rand-m9-mstd0.5-inc1 Mixup True Cutmix True Mixup, Cutmix Probability 0.5, 0.5 Mixup Mode Batch Label Smoothing 0.1

##### C.4.2 Ablation with the Hidden Dimension, r in Fourier KArAt

While avoiding the computational overhead for computing Φi,j ∈ RN×N, we make use of the low-rank structure that the attention heads show (see Figure 5) by comparing different values of r. Particularly, we consider the values, r = 8,12,24, on the Fourier KArAt variant of ViT-Base model, Φi,j ∈ Rr×N; see Table 16 for results on CIFAR-10. In this ablation, we observe that changing the hidden dimension has a negligible impact on the model’s performance. This can be explained by the sudden drop in the singular values, as shown in Figure 13. As long as r remains greater than the sudden drop index, the model should not be impacted by the changing of r except for changes in computational requirement; a higher r would incur a higher compute time as the size of the operator Φ scales with r.

##### C.4.3 Enforcing an Extreme Low Rank Structure in Fourier KArAt

While our modular design of Fourier KArAt avoids the computational requirement of Φi,j by using a low hidden dimension r, it also enforces a low-rank structure in the attention matrix Ai,j. In this context, we attempt to find the best possible value for r; see §C.4.2. However, we want to investigate how low the hidden dimension r can be used without substantially compromising the performance of learnable Fourier KArAt. The significance of this question lies in the information bottleneck created by an extremely low hidden dimension r that helps to understand the tradeoff between computing requirements and final trained model quality. To this end, we experiment with extreme low-rankness in the hidden dimension, r = 2,4, and report our findings in Table 18. Although r = 2,4 are not optimal, and we observe an insignificant drop in the performance across both datasets (CIFAR-10 and CIFAR-100), it comes at a reduced computing requirement, especially required VRAM; see GPU memory in Table 18. For instance, ViT-Tiny+G3B with r = 2

Table 11: Hyper-parameter settings for object detection and instance segmentation experiments.

Input Size 224 × 224 Batch Size 32

Optimizer AdamW Optimizer Epsilon 1 × 10−6 Momentum 0.9,0.999 Weight Decay 0.1

Learning Rate Schedule Warmup Scheduler Learning Rate 1 × 10−4 Iterations 184375 Decay Rate 0.988

Random Flip Horizontal 0.5; Vertical 0.0 Patch Size 16 Attention Window Size 14 Window Attention Block Indices 0,1,3,4,6,7,9,10 Encoder Output Layer Index 11 Pyramid Scale Factors 4.0,2.0,1.0,0.5 Output Channels 256 Proposal Generator Input Layers (corresponding to feature pyramid) p2,p3,p4,p5,p6 Proposal Generator Input Sizes 32,64,128,256,512 Proposal Generator Training Pre-NMS Top-K 12000 Proposal Generator Evaluation Pre-NMS Top-K 6000 Proposal Generator Training Post-NMS Top-K 2000 Proposal Generator Evaluation Post-NMS Top-K 1000 Proposal Generator NMS Threshold 0.7 Region-of-Interest Heads IOU Threshold 0.5 Region-of-Interest Score Threshold 0.05 Region-of-Interest NMS Threshold 0.5

- Table 12: Fourier KArAt on object detection and instance segmentation tasks on the MS COCO [42] dataset. The header Box and Mask refer to detection and segmentation tasks, respectively.

Box Mask AP AP50 AP75 AP AP50 AP75 ViT-Det-Base

Model Initialization

32.34 49.16 34.43 28.35 46.04 29.48 + G1B 22.48 36.82 23.13 20.11 34.09 20.46 + G1U 26.68 43.25 27.88 24.01 40.21 24.66 ViT-Det-Base

ViT-Base on ImageNet-1K

15.28 26.88 15.22 13.39 24.34 12.86

+ G1B 10.32 18.77 10.05 8.94 16.51 8.50 + G1U 10.99 20.04 10.82 9.69 17.99 9.25

Random Initialization

Training Time by Models for 100 epochs on CIFAR-10 Vanilla

G3B (Tiny, Small) & G1B (Base) G3U (Tiny, Small) & G1U (Base)

60

50

TrainingTime(hours)

40

30

20

10

0

ViT-Tiny ViT-Small ViT-Base

(a) Training time on CIFAR-10 dataset.

Training Time by Models for 100 epochs on CIFAR-100

40

TrainingTime(hours)

30

Vanilla

G3B (Tiny, Small) & G1B (Base) G3U (Tiny, Small) & G1U (Base)

20

10

- 0

50

100

150

200

250

TrainingTime(hours)

Training Time by Models for 100 epochs on ImageNet-1K

Vanilla

G3B (Tiny, Small) & G1B (Base) G3U (Tiny, Small) & G1U (Base)

(c) Training time on ImageNet-

- 1K dataset.

0

ViT-Tiny ViT-Small ViT-Base

ViT-Tiny ViT-Small ViT-Base

(b) Training time on CIFAR100 dataset.

Throughput by Models

175

Throughput(images/second)

150

125

Vanilla

100

G3B (Tiny, Small) & G1B (Base) G3U (Tiny, Small) & G1U (Base)

75

50

25

0

ViT-Tiny ViT-Small ViT-Base

(d) Throughput comparison during inference.

- Figure 7: A detailed comparison of computing requirements.We compare the training times for 100 epochs with the hyperparameter settings given in Table 5 for all the datasets CIFAR-10, CIFAR-100, and ImageNet-1K. We also compare the throughputs of different models on ImageNet-1K; the throughput results will be similar for other datasets, as the input size is 224 × 224.

- Table 13: Parameter, computation, and memory requirement for Fourier-KArAt (with hidden dimension, r = 12) compared to the traditional softmax attention. This Table particularly shows the individual computation required for the attention activation. The memory requirement shown is approximate and is based on averages of batches of 32 images of resolution 224 × 224. We note that changing r will affect the performance and memory requirements. In our main paper, all the experiments were performed with r = 12.

Parameters GFLOPs

Model

GPU Memory Attention Activation Total Attention Activation Total

ViT-Base 0 85.81M 0.016 17.595 7.44 GB

+ G1B 1.70M 87.51M 0.268 17.847 17.36 GB + G1U 0.14M 85.95M 0.268 17.847 16.97 GB

ViT-Small 0 22.05M 0.008 4.614 4.15 GB

+ G3B 1.53M 23.58M 0.335 4.941 11.73 GB + G3U 0.13M 22.18M 0.335 4.941 11.21 GB

ViT-Tiny 0 5.53M 0.005 1.262 2.94 GB + G3B 0.76M 6.29M 0.168 1.425 7.48 GB + G3U 0.06M 5.59M 0.168 1.425 7.29 GB

outperforms the vanilla variant, only has a modest 0.03M parameters increment from the vanilla ViT-Tiny, albeit similar GFLOPs, but an extra 1.07GB of GPU memory usage. Compared to the r = 12 variant, the memory usage is 3.47 GB lower, a total of 0.73M parameters less, but the relative decrease in accuracy is only 2.96% lower on the CIFAR-10 dataset. In ViT-Base backbones, the performance remains comparable, and even in the case of ViT-Base+G1B on CIFAR-100, the variant with r = 2 surpasses the original variant with r = 12. Although there is no concrete strategy, this observation supports our claim on a probable research direction involving parameter reduction (discussed in §6) to improve scalability in such over-parameterized models. This also indicates that for over-parameterized larger models, the scope of parameter reduction may find a model with learnable attention that can outperform its traditional counterpart across multiple datasets.

##### C.4.4 Ablation on the Impact of grid size, G on Fourier KArAt

KANs are highly dependent on certain hyperparameters, and Fourier-KArAt has only one hyperparameter to tune the performance — grid size G. Thus, we perform extensive experiments involving grid size G and present them in Table 15. We observe that each of the particular ViT models, in conjunction with particular Fourier-KArAt variants, has a typical G value that brings out its best performance, and there is no universal value of G to follow. When performing validation with ViT-Base+ variants on CIFAR-10 and CIFAR-100, the accuracy drops as the grid size passes a size after

###### 5. However, this behavior is not persistent with the ViT-Small/Tiny+ variants; see Table 153.

3We did not perform gridsize 5, 7, 9, and 11 experiments with ViT-Tiny+KArAt as it was already outperforming the base ViT-Tiny with gridsize 1 and 3, and the experiments are extensively resource intensive.

[Figure 6]

(a) ViT-Det with softmax attention

[Figure 7]

- (b) ViT-Det Fourier KArAt G1B

[Figure 8]

- (c) ViT-Det Fourier KArAt G1U

- Figure 8: Detection and segmentation tasks inference visualization using ViT-Det with ViT-Base as backbone for traditional MHSA and Fourier KArAt. For each sample, the ground-truth is given on the right side and the inference is on the left.

ViT-Det Bounding Box mAP vs. Iterations

30

25

20

mAP

15

10

Vanilla

5

Fourier KArAt G1B Fourier KArAt G1U

0

0K 25K 50K 75K 100K 125K 150K 175K Iterations

(a) Performance on Object Detection

ViT-Det Segmentation Mask mAP vs. Iterations

25

20

15

mAP

10

Vanilla

5

Fourier KArAt G1B Fourier KArAt G1U

0

0K 25K 50K 75K 100K 125K 150K 175K Iterations

(b) Performance on Instance Segmentation

- Figure 9: Training Curves of ViT-Det and Fourier KArAt in object detection and instance segmentation tasks while training from ImageNet-1K pretrained ViT-Base with traditional MHSA weights initialization. In this particular task, the universal configuration (G1U), performs strictly better than the blockwise configuration (G1B).

- Table 14: Experimental results on multi-layer KANs organized similarly to an MLP. These experiments involve B-Splines as the basis functions, as mentioned in Section 2.

(a) Experiments on CIFAR-10 and CIFAR-100.

(b) Experiments on MNIST and Fashion-MNIST

Acc.@1 CIFAR-10 CIFAR-100

Spline Order Grids Grid Range Layers

10 (-4, 4) (10, 10) 22 1

(-4, 4) (10, 10) 21 4 (-6, 6) (10, 10) 22 4 (-8, 8) (10, 10) 22 4

20

1

(-4, 4) (10, 10) 22 1 (-6, 6) (10, 10) 22 1 (-8, 8) (10, 10) 22 1

40

- 3

10 (-4, 4) (10, 10) 22 1

20

(-4, 4) (10, 10) 22 4 (-6, 6) (10, 10) 22 4 (-8, 8) (10, 10) 39 4

40

(-4, 4) (10, 10) 22 1 (-6, 6) (10, 10) 29 1 (-8, 8) (10, 10) 31 1

- 4 10 (-4, 4) (10, 10) 21 1

- 5

- (-1, 1) (10, 10) 10 4
- (-2, 2) (10, 10) 10 1 (-4, 4) (10, 10) 22 4 (-4, 4) (20, 20) 40 4 (-4, 4) (40, 40) 45 9 (-6, 6) (10, 10) 39 7 (-8, 8) (10, 10) 42 9

10

20 (-4, 4) (10, 10) 10 1 40 (-4, 4) (10, 10) 21 1

Acc.@1 MNIST FMNIST 1 10 (-4, 4) (10, 10) 92 85

Spline Order Grids Grid Range Layers

- 3 10 (-4, 4) (10, 10) 94 86

- 4 10 (-4, 4) (10, 10) 94 86

- 5

- (-1, 1) (10, 10) 89 83
- (-2, 2) (10, 10) 90 95 (-4, 4) (10, 10) 95 86 (-4, 4) (20, 40) 96 87 (-4, 4) (40, 40) 96 88 (-6, 6) (10, 10) 95 86 (-8, 8) (10, 10) 95 86

10

20 (-4, 4) (10, 10) 93 85 40 (-4, 4) (10, 10) 91 83

##### C.4.5 Does Fourier KArAt Require ℓ1-Projection?

To ensure that each row vector of the learned attention matrix lies on a probability simplex, we project it onto the ℓ1-unit ball. We use Algorithm 1 to project learned attention vectors to the probability simplex and compare Top-1 accuracies

to the baseline model. We note that using ℓ1 projection does not substantially increase the training time. However, from

- Table 15: Ablation on grid size G for ViT-Tiny, ViTSmall and ViT-Base. We find a particular grid size suitable for each of the models.

Acc.@1 CIFAR-10 CIFAR-100

Model

ViT-Base 83.45 58.07 + G1B 81.81 55.92 + G1U 80.75 57.36 + G3B 80.09 56.01 + G3U 81.00 57.15 + G5B 79.80 54.83 + G5U 81.17 56.38 + G11B 50.47 42.02 + G11U 40.74 39.85 ViT-Small 81.08 53.47 + G1B 79.00 53.07 + G1U 66.18 53.86 + G3B 79.78 54.11 + G3U 79.52 53.86 + G5B 78.64 53.42 + G5U 78.75 54.21 + G11B 77.39 52.62 + G11U 78.57 53.35 ViT-Tiny 72.76 43.53 + G1B 75.75 45.77 + G1U 74.94 46.00 + G3B 76.69 46.29 + G3U 75.56 46.75

+ G5B 75.85 – + G5U 74.71 – + G7B 75.11 – + G7U 74.45 – + G9B 74.85 – + G9U 73.97 – + G11B 74.52 – + G11U 73.58 –

- Table 16: Ablation on hidden dimension r on CIFAR10 with ViT-Base. Here, we compare the values of r ∈ {8,12,24}.

Model r Acc.@1 on CIFAR-10 ViT-Base – 58.07 + G3B 24 80.54 + G3U 24 80.81 + G5B 24 77.99 + G5U 24 80.52 + G3B 12 80.09 + G3U 12 81.00 + G5B 12 79.80 + G5U 12 81.17 + G3B 8 80.76 + G3U 8 80.40 + G5B 8 79.79 + G5U 8 80.83

- Table 17: Comparing performance of Fourier KArAt for

ViT-Tiny and ViT-Base with or without using ℓ1 projection in Algorithm 1. ViT-Tiny and ViT-Base use tradi-

tional softmax and do not require ℓ1 projection.

Acc.@1

Model ℓ1 Projection

on CIFAR-10 ViT- Tiny ✗ 72.76 + G3B ✓ 41.99 + G3U ✓ 40.85 + G3B ✗ 76.69 + G3U ✗ 75.56 ViT-Base ✗ 83.45 + G3B ✓ 47.44 + G3U ✓ 46.11 + G3B ✗ 80.09 + G3U ✗ 81.00

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

(a) ViT-Tiny (b) ViT-Tiny Fourier KARAT (c) ViT-Base (d) ViT-Base Fourier KARAT

- Figure 10: Optimization path for ViT-Tiny and ViT-Base (the smallest and the largest model) along the two largest principal component directions of the successive change of model parameters. We show the loss contours along with the trajectory of the optimizers. Perturbed contours indicate corresponding non-smooth, spiky loss surfaces.

Table 17, we observe that incorporating this algorithm in the Fourier-KArAt does not improve its performance; instead, the performance significantly degrades. Algorithm 1 can be used with the choice of any basis function. However, we cannot comment on the role of ℓ1-projection in KArAt’s performance with basis functions other than Fourier.

##### C.4.6 More Discussion on Blockwise and Universal Configuration

Although the primary inspiration behind the universal configuration comes from the shared basis functions in KAT [3], it aligns with the fact that softmax is a fixed function conventionally being used across all heads of all encoder layers

Table 18: Complete compute requirement of ViT-Tiny+Fourier KArAt with extremely low hidden dimension training.

Acc.@1

Model r

Parameters GFLOPs GPU Memory CIFAR-10 CIFAR-100

ViT-Tiny 72.76 43.53 5.53M 1.262 2.94GB

12 76.69 46.29 6.29M 1.425 7.48GB 4 75.90 45.76 5.80M 1.313 4.68GB 2 74.39 44.36 5.56M 1.285 4.01GB

+ G3B

12 75.56 46.75 5.59M 1.425 7.29GB 4 73.13 44.98 5.57M 1.313 5.08GB 2 71.56 42.43 5.55M 1.285 4.41GB

+ G3U

ViT-Base 83.45 58.07 85.81M 17.595 7.44GB

12 81.81 55.92 87.51M 17.847 17.36GB 4 81.20 55.29 86.41M 17.668 12.05GB 2 81.55 56.49 86.13M 17.623 9.48GB

+ G1B

12 80.75 57.36 85.95M 17.847 16.97GB 4 79.69 55.70 85.89M 17.668 12.11GB 2 80.36 54.04 85.87M 17.623 9.92GB

+ G1U

[Figure 13]

- Figure 11: Vit-Tiny attention map characterization. Original image for inference (the center), the attention maps (top row), and

contributing image regions (bottom row) for all three heads in ViT-Tiny: traditional MHSA (left) and Fourier KArAt G3B (right). The traditional MHSA sporadically focuses on fine-grained features of the primary object in the image. In contrast, the learnable attention in Fourier KArAt identifies the primary object features present significantly across all heads.

in the ViT architectures. This raises the question of whether the token-to-token interaction in the self-attention can be modeled using a single universal function, or each instance of attention operations needs dynamic modeling using learnable functions. To investigate further, we compared the learned coefficients {apqm, bpqm} from Equation 2 and the weight matrix W of the linear projector following the learnable basis, for each head across all the encoder layers in ViT-Tiny+Fourier KArAt. Thus, we obtain 36 sets (3 heads for each of the 12 layers) of such coefficients and weights from the model with blockwise configuration G3B, and 3 sets from the model with universal configuration G3U. Although mostly centered at zero, the distribution of each of these sets of parameters varies significantly from the others. The difference is mostly visible in the presence of long tails and the length of the tails. This characteristic is present not only in the blockwise configuration, but the three heads of the universal configuration also show a similar pattern.

- C.5 Alternate Variants of Fourier KArAt In this section, we experiment with different attention variants on Fourier KArAt.

##### C.5.1 Alternative Approaches to the Lower Rank Attention Structures

In the main paper, we approximated the effect of the operator, Φi,j ∈ RN×N casting it as the product of two rank-r operators—one operator with learnable activation, Φ, and the other is the learnable the linear transformation matrix, W such that Φ = ΦW. Here, we abuse the notations for simplicity. However, a natural question is how many different configurations are possible with Φ and W such that they can approximate the effect of Φ. Specifically, we

Table 19: Ablation study on the operator variants in Fourier KArAt. Note that Φ(WA) and W Φ(A) refer to order of using linear projector W and operator Φ, and Φ[2]( Φ[1](A)) refers to two consecutive learnable operators Φ[1] and Φ[2] applied to A.

Acc.@1 on CIFAR-10

Model

Φ(WA) W Φ(A) Φ[2]( Φ[1](A)) ViT-Base

+ G11B 41.02 50.47 30.14 + G11U – 47.74 –

ViT-Tiny

+ G11B 67.04 75.41 33.09 + G11U 70.98 73.67 36.98

Table 20: Performance using softmax and variants of Fourier KArAt in separate heads.

Low Rank Configurations No. of Distinct Heads Acc.@1 on

Model

(for Fourier KArAt) softmax Fourier KArAt CIFAR-10 ViT-Base 12 0 83.45

σˆ = Φ(WA) 0 12 41.02 σˆ = Φ(WA) 6 6 47.60 σˆ = W Φ(A) 0 12 50.47

ViT-Base + G11B

σˆ = Φ[2]( Φ[1](A)) 0 12 30.14 σˆ = Φ[2]( Φ[1](A)) 6 6 30.24

ViT-Tiny 3 0 72.76

- σˆ = Φ[1]( Φ[2](A) 0 3 33.09
- σˆ = Φ[2]( Φ[1](A)) 1 2 32.51 σˆ = Φ[2]( Φ[1](A)) 2 1 34.73

ViT-Tiny

+ G11B

- σˆ = Φ[2]( Φ[1](A)) 0 3 36.98
- σˆ = Φ[2]( Φ[1](A)) 1 2 33.12
- σˆ = Φ[2]( Φ[1](A)) 2 1 70.57

ViT-Tiny + G11U

use the following configurations — (i)4 W Φ(A), where Φ ∈ Rr×N and W ∈ RN×r, (ii) Φ(WA), where W ∈ Rr×N and Φ ∈ RN×r, and (iii) Φ[2]( Φ[1](A)), where Φ[1] ∈ Rr×N and Φ[2] ∈ RN×r are low-rank operators for learning activation functions. In the first configuration, Φ(·) acts on each row of A to produce an r-dimensional vector, and then W projects it back to an N-dimensional subspace. In the second configuration, W, first projects each row of A to produce an r-dimensional vector, and then Φ with learnable activation produces N-dimensional vectors. In the third configuration, Φ[1] first produces r-dimensional vector from rows of A by learning activations, and Φ[2] obtains N-dimensional vectors by learning a second level of activations.

Primarily, we started with the full-rank operator Φ ∈ RN×N and found its computations to be prohibitively expensive, regardless of the choice of the basis. Next, we conduct an ablation study to see which operator configuration works better; see Table 19. Our experiments show that the operator configuration (ii) demonstrates an inferior performance. We postulate that by down-projecting the N-dimensional attention row vector to a smaller dimension, r loses adequate token-to-token interaction, and this fails to capture dependencies from significant attention units within a row. After that, from this limited information, learnable activation cannot significantly help the model’s performance. On the other hand, (iii) also fails to perform due to its inability to model the attention well, despite having refined information, from the intermediate r-dimensional subspace. Overall, apart from configuration (i), the performances of the other configurations were inconsistent over multiple experiments. Also, they lag in training stability, particularly configuration (iii). Considering these observations, we proceeded to find the best possible grid size G only with the configuration (i).

##### C.5.2 Fourier KArAt + softmax Attention —A Hybrid version

We consider another variant where we mix the learnable and pre-defined activation in each head. E.g., it is possible to have 2 of the 3 attention heads in each encoder block in ViT-Tiny activated by softmax and the third activated by the Fourier KArAt. With the idea of incorporating KAN to replace softmax activation, we are curious to see if the ViT model performs better with a hybrid mode of activation; see Table 20 for results.

##### C.5.3 More Combinations

We have also experimented with more configurations of Fourier KArAt in various permutations of the strategies mentioned above and have not found any significant combination in terms of performance. We also carefully design a particular training strategy where the attention operators ( Φ, W, and Φ wherever applicable) of Fourier KArAt are trained with a separate learning rate to alleviate the problem of the mismatch of the data passing through these learnable layers, as they consider each row of an attention matrix A as an input. The models have a gradient and loss explosion during training.

4This configuration was used in the main paper.

50K

Epoch 0

40K

Frequency

Frequency

30K

20K

10K

0K

-0.1 -0.05 0.0 0.05 0.1

250K

Epoch 0

200K

Frequency

Frequency

150K

100K

50K

0K

-0.1 -0.05 0.0 0.05 0.1

50K

Epoch 50

40K

Frequency

30K

20K

10K

0K

-0.1 -0.05 0.0 0.05 0.1

(a) ViT-Tiny

250K

Epoch 50

200K

Frequency

150K

100K

50K

0K

-0.1 -0.05 0.0 0.05 0.1

(c) ViT-Base

50K

Epoch 100

40K

30K

20K

10K

0K

-0.1 -0.05 0.0 0.05 0.1

250K

Epoch 100

200K

150K

100K

50K

0K

-0.1 -0.05 0.0 0.05 0.1

50K

50K

Epoch 0

Epoch 50

40K

40K

Frequency

Frequency

Frequency

30K

30K

20K

20K

10K

10K

0K

0K

-0.1 -0.05 0.0 0.05 0.1

-0.1 -0.05 0.0 0.05 0.1

(b) ViT-Tiny+Fourier KArAt

250K

250K

Epoch 0

Epoch 50

200K

200K

Frequency

Frequency

Frequency

150K

150K

100K

100K

50K

50K

0K

0K

-0.1 -0.05 0.0 0.05 0.1

-0.1 -0.05 0.0 0.05 0.1

(d) ViT-Base+Fourier KArAt

50K

Epoch 100

40K

30K

20K

10K

0K

-0.1 -0.05 0.0 0.05 0.1

250K

Epoch 100

200K

150K

100K

50K

0K

-0.1 -0.05 0.0 0.05 0.1

- Figure 12: Weight distribution of ViT-Tiny and -Base with traditional MHSA and Fourier KArAt. The columns (left to right) represent weights at initialization, epoch 50, and epoch 100.

##### C.6 Distribution of the Weights

[74] considered an invariant measure perspective to describe the training loss stabilization of the neural networks. We adopt this idea to study the distribution of the weights of the smallest and largest models of the ViTs, Tiny and Base, and their Fourier KArAt variants. Figure 12 shows the distribution of the weights of these models during different training phases—The evolution of the weights’ distributions for respective models remains invariant. Based on this observation, from [74], we can guarantee the convergence of loss values of all models; also, see Figure 6. However, as mentioned in [74], this perspective cannot comment on the generalization capacity and structural differences between different neural networks.

##### C.7 Spectral Analysis of Attention

Although we cannot comment on the generalizability by studying the distribution of the weights, from Table 1, we realize all the KArAt variants have more parameters than their vanilla counterparts. Therefore, it would be interesting to see how their attention matrices behave. As discussed in §4, the attention matrices in traditional MHSA have a low-rank structure. Following that study, we verified the apparent low-rank structure that Fourier KArAt’s learned attention matrices possess.

We use all 3 heads in the last encoder block of ViT-Tiny on 5 randomly sampled images from the CIFAR-10 validation set. There are a total of 15 singular vectors (each of 197 dimensions) for any attention matrix of shape 197 × 197, where the singular values are arranged in non-increasing order. Let σi be the ith singular value across all heads and samples (it is permutation invariant). For each i ∈ [197], we plot [ln(σimin),ln( ¯σi),ln(σimax)], where σ¯i is the average of ith-indexed singular value across all samples and heads.

To investigate the inherent low-rank structure of attention, we plot the natural logarithm of singular values of attention matrices, σi, before and after attention activation in Figure 13. From Figure 13a, we observe that the traditional softmax attention and our learnable Fourier KArAt have almost similar low-rank structures. However, Figure 13b shows that the traditional MHSA has significantly larger singular values than its KArAt variant, G3B. It can also be noticed that before the activation, both traditional MHSA and Fourier KArAt feature a sharp drop in singular values between the 50th and 75th indices. This sharp drop in singular values vanishes in softmax attention, indicating a normalization. However, due to the hidden dimension r, Fourier KArAt enforces a much lower rank than the traditional MHSA.

Additionally, we observe from the weight distributions in Figure 12 that Fourier KArAt variants have more entries close to zero than their ViT counterparts. This, along with the low-rankness, can inspire low-rank + sparse training strategies of Fourier KArAT.

103

Before softmax attention

Before Fourier KArAt

101

logSingularValuelni

10 1

10 3

10 5

10 7

10 9

0 25 50 75 100 125 150 175 200 Index

(a) Before attention activation

102

After softmax attention

After Fourier KArAt

100

logSingularValuelni

10 2

10 4

10 6

10 8

10 10

0 25 50 75 100 125 150 175 200 Index

(b) After attention activation

- Figure 13: Spectral decomposition of the attention matrix for ViT-Tiny on CIFAR-10 dataset with traditional softmax attention and our learnable Fourier KArAt. The traditional softmax attention and our learnable Fourier KArAt have almost similar low-rank structure, before activation functions are used.

