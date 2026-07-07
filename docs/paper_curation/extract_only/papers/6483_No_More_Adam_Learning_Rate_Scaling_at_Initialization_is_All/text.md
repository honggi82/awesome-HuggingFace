## No More Adam: Learning Rate Scaling at Initialization is All You Need

Minghao Xu*1 Lichuan Xiang*12 Xu Cai2 Hongkai Wen1

# arXiv:2412.11768v2[cs.LG]17Dec2024

### Abstract

In this work, we question the necessity of adaptive gradient methods for training deep neural networks. SGD-SaI is a simple yet effective enhancement to stochastic gradient descent with momentum (SGDM). SGD-SaI performs learning rate Scaling at Initialization (SaI) to distinct parameter groups, guided by their respective gradient signal-to-noise ratios (g-SNR). By adjusting learning rates without relying on adaptive second-order momentum, SGD-SaI helps prevent training imbalances from the very first iteration and cuts the optimizer’s memory usage by half compared to AdamW. Despite its simplicity and efficiency, SGD-SaI consistently matches or outperforms AdamW in training a variety of Transformer-based tasks, effectively overcoming a long-standing challenge of using SGD for training Transformers. SGD-SaI excels in ImageNet1K classification with Vision Transformers(ViT) and GPT-2 pretraining for large language models (LLMs, transformer decoder-only), demonstrating robustness to hyperparameter variations and practicality for diverse applications. We further tested its robustness on tasks like LoRA fine-tuning for LLMs and diffusion models, where it consistently outperforms state-of-the-art optimizers. From a memory efficiency perspective, SGD-SaI achieves substantial memory savings for optimizer states, reducing memory usage by 5.93 GB for GPT-2 (1.5B parameters) and 25.15 GB for Llama2-7B compared to AdamW in full-precision training settings. 1

*Equal contribution 1Department of Computer Science, University of Warwick, Coventry, UK 2Collov Labs. Correspondence to: Hongkai Wen <hongkai.wen@warwick.ac.uk>.

1The PyTorch implementation is available at https:// github.com/AnonymousAlethiometer/SGD_SaI/

+73.81GB

###### StateTensorMemory(GB)

100

AdamW Prodigy Adam-mini

80

60

SGD-SaI

+24.33GB

40

+2.06GB

+17.79GB

20

+5.93GB

+7.28GB

+0.59GB

+2.44GB

+0.00GB+0.12GB

+0.25GB

+0.09GB

0

600

###### OptimizerStepTime(ms)

+335ms

AdamW Prodigy Adam-mini

+241ms

+292ms

400

+143ms

SGD-SaI

+206ms

+166ms+155ms

200

+70ms+70ms

+72ms

+66ms

+33ms

0

0 1 2 3 4 5 6 7 Model Size (Billions of Parameters)

Figure 1. The chart illustrates how memory usage and optimizer step time (in wall-clock time) increase with larger model sizes. It highlights the substantial memory overhead of storing optimizer states as model sizes grow. SGD-SaI exhibits significantly lower memory usage than AdamW and has the shortest optimization step runtime. This runtime refers to the wall clock time required for the optimizer step function. All statistics were measured on a single NVIDIA A100-80GB.

### 1. Introduction

Stochastic gradient-based optimization methods, such as Stochastic Gradient Descent (SGD), are fundamental to modern machine learning, enabling the successful training of models across a wide range of scientific and engineering applications. However, training objectives and data are often noisy in practice, and gradients may become sparse due to the inherent characteristics of regularization or specific architectural designs. Moreover, architectural differences can introduce imbalances in the learning dynamics across different parameters. To address these challenges, adaptive gradient methods (Ghorbani et al., 2022) have been developed to handle better non-stationary objectives, noisy data, and sparse gradients. Among these methods, Adam (Kingma & Ba, 2014) and AdamW (Loshchilov & Hutter, 2019) have become indispensable for training Transformer-based models,

###### Step 0 Step t

###### Training Process

|Gradient g0<br><br>|
|---|
|Model<br><br>Block.0.QKV Block.1.QKV Block.2.QKV<br><br>Block.0.QK Block.0.V Block.1.QK Block.1.V Block.2.QK Block.2.V<br><br>We take the ViT as an example.<br><br>Adam-mini Partition<br><br>PyTorch Default Partition<br><br>ViT Blocks Example<br><br>Adam-mini Partition Rules<br><br>|whole gradient of the block|
|---|
<br><br>|gradient element|
|---|
<br><br>|

|Local Gain α0<br><br>|
|---|
||Adam-mini|Partition-wise Adaptive Local Gain|
|---|---|
<br><br>Adam-mini<br><br>|Adam(W)|Adaptive Local Gain|
|---|---|
<br><br>Adam(W)<br><br>|SGD|Fixed Local Gain|
|---|---|
<br><br>SGD<br><br>|SGD-SaI|Partition-wise Pre-conditioned Local Gain|
|---|---|
<br><br>SGD-SaI<br><br>Local gains are calculated based on Adam-mini partition.<br><br>Local gains are calculated based on each element of PyTorch defaults.<br><br>Local gains are fixed (pre-defined) based on PyTorch defaults.<br><br>Local gains are calculated (once) based on PyTorch defaults.<br><br>|

|Update Direction 0<br><br>|
|---|

|Local Gain αt<br><br>|
|---|
||Adam(W)|Adaptive Local Gain|
|---|---|
<br><br>Adam(W)<br><br>|Adam-mini|Block-wise Adaptive Local Gain|
|---|---|
<br><br>Adam-mini<br><br>|SGD-SaI|Block-wise Pre-conditioned Local Gain|
|---|---|
<br><br>|SGD|Fixed Local Gain|
|---|---|
<br><br>SGD<br><br>Different local gains throughout training, and vary across partitions.<br><br>Fixed local gain throughout training, and vary across partitions.|

Adam(W)SGDAdam-mini

SGDAdam(W)SGD-SaIAdam-mini

Adaptive Gradient Methods

Local gains are continuously re-calculated at each step.

Non-Adaptive Gradient Methods

Local gains are fixed at each step. (pre-defined or calculated once).

- Figure 2. This graph illustrates the differences in local gain behaviours exhibited by four optimizers throughout the training process. We present two popular adaptive gradient methods: Adam(W) and the memory-efficient Adam-mini. The local gains for these methods are recalculated continuously at each step based on the gradients. In contrast, SGD and SGD-SaI are both non-adaptive methods, meaning their local gains remain fixed throughout the training.

including Large Language Models (LLMs) (Radford et al., 2019; Team et al., 2023) and Diffusion Models (DMs) (Ho et al., 2020; Rombach et al., 2022). Their popularity stems from their relative robustness and efficiency in optimizing high-dimensional parameter spaces. The core mechanism of Adam’s adaptability lies in its second-order momentum term, v, which acts as a local gain (Hinton et al., 2012), dynamically adjusting the learning rate for each parameter. This mechanism enables Adam to perform effectively even in noisy or sparse gradients, addressing imbalances in the learning process across different parameters.

However, this adaptability comes with significant costs when the model size scales up. Specifically, Adam requires storing and updating each parameter’s first-order (mean) and second-order (variance) momentum terms. This increases memory usage by at least 3x compared to the parameter size alone. For instance, training a 7-billion parameter model in FP32 using Adam requires approximately 50 GB of memory for the state tensors, a significant challenge even with high-end hardware like NVIDIA A100-80G GPUs. Compared to SGD, the memory demand of Adam can be at least double (Zhang et al., 2024b), posing a severe limitation on the scalability of deep learning research.

Numerous previous works have sought to reduce memory usage by simplifying optimizer states while preserving the adaptive gradient term to address the memory bottleneck

while maintaining the effectiveness of adaptive methods. Approaches such as 8-bit Adam (Dettmers et al., 2021), Adafactor (Shazeer & Stern, 2018), and sign-based methods (Bernstein et al., 2018; Kunstner et al., 2023) focus on quantizing or sparsifying the optimizer states. Meanwhile, Adam-mini (Zhang et al., 2024b) introduces parameter block grouping to share adaptive learning rates, leveraging Hessian structure insights (Zhang et al., 2024a) to reduce memory usage. However, these methods often come with trade-offs. Many risk a performance downgrade compared to AdamW. From an efficiency standpoint, these approaches also introduce additional update complexity. Simplified state tensors still require computations based on full gradients for each parameter at each time step, increasing the overall computational burden. Adam-mini, in particular, necessitates fine-grained parameter partitioning (Zhang et al., 2024b), further complicating its implementation. As a result, these limitations lead to longer optimizer step times, ultimately slowing down the training process.

In this work, we challenge the necessity of adaptive gradient methods for model training and propose a memory- and computation-efficient alternative. We begin by revisiting the foundational motivation behind Adam’s use of secondorder momentum. Inspired by the concept of the gradient Signal-to-Noise Ratio (g-SNR) (Xiang et al., 2023), which quantifies the relationship between a gradient’s norm and variance, we leverage this metric to analyze and measure

gradient distribution differences across parameters. Through empirical analysis, we investigate the temporal consistency for g-SNR during training and explain why this value can be determined at first training iterations. Furthermore, we analysed the g-SNR distribution across different ViT parameters and explored the g-SNR value correlated with varying parameters of type and its architecture characteristics. Building on this, we argue that g-SNR can be leveraged to adjust learning rate scales, balancing the learning progress based on the distribution of gradients. Incorporating a preconditioned learning rate scale computed during the first training iteration, called Scaled at Initialization(SaI), facilitates stable training progress without incurring the memory and computational overhead associated with adaptive gradient terms. We call our method SGD-SaI, a novel optimization approach that eliminates the need for adaptive gradient methods, treating them as simple yet effective updates compared to SGD. In summary, our contributions are as follows:

- • We challenge the necessity of adaptive gradient methods, specifically identified the existing challenges on Adam-like methods and proposed to use constant gSNR value to replace the second-order momentum to reduce both the memory and computation cost, called Scaled at Initialization(SaI).
- • We empirically analysed the statistics of g-SNR on parameters during training and identified its characteristics over time and distribution over parameters.
- • We formula our insight into proposed methods, SGDSaI, solved the long-stand challenge that SGD can not successfully train tasks with transformer architectures and observed outstanding performance in ViT and decoder-only transformer (LLMs).
- • We extend our empirical analysis to other popular and practical task training, such as LoRA training on LLMs and Diffusion Models(DMs) and traditional CNN tasks. We observed consistent improvement compared to existing SOTA optimizers.

### 2. Related Work

Adaptive Gradient Methods:Stochastic gradient descent (SGD) is an efficient optimization method commonly used in deep learning, but it struggles with tasks that have nonstationary objectives or involve very noisy and/or sparse gradients (Kingma & Ba, 2014), often requiring extensive hyperparameter tuning. To improve upon these limitations, adaptive gradient methods were developed to continuously and dynamically adjust learning rates for individual parameters throughout the training process (Duchi et al., 2011; Graves, 2014; Zeiler, 2012), with the Adam optimizer becoming particularly popular. Adam combines features from

AdaGrad (Ward et al., 2020), which effectively manages sparse gradients, and RMSProp (Hinton et al., 2012), which is suitable for online and non-stationary tasks, allowing it to outperform SGD in many cases with less tuning effort. However, Adam has its own challenges, leading to the creation of enhancements such as AdamW (Loshchilov & Hutter, 2019), which introduces decoupled weight decay for better generalization, and adaptations (Dozat, 2016) that incorporate Nesterov momentum for faster convergence. To address early training noise, warm-up phases and Rectified Adam (Liu et al., 2021) have been proposed. Additionally, Adaptive Weight Decay (Ghiasi et al., 2023) further improves convergence, while (Mishchenko & Defazio, 2023) introduced a dynamic component for automatic learning rate adjustments within the Adam framework.

Adam in Transformer Realm: Transformers (Vaswani, 2017) have become essential in modern deep learning, particularly in natural language processing. While the Adam (Kingma & Ba, 2014) optimizer generally outperforms Stochastic Gradient Descent (SGD) in training Transformer architectures (Xiao et al., 2021), it has a significant downside: as model sizes grow, Adam’s memory requirements, which are twice that of SGD due to first and second-order momentum storage (Kingma & Ba, 2014), become a concern. To mitigate this overhead, researchers have explored methods like sign-based optimization (Bernstein et al., 2018; Kunstner et al., 2023) and low-precision quantization (Li et al., 2023a; Dettmers et al., 2021; 2022; Dettmers & Zettlemoyer, 2023), although these can compromise performance. Studies have shown that Adam’s adaptive learning rates based on gradient norm history contribute to its performance advantage (Zhang et al., 2024a), whereas SGD lacks this capability. However, finding the right learning rate scale for SGD to match Adam’s performance remains unresolved. Adam’s insights, rooted in RMSprop (Hinton et al., 2012), suggest that a global learning rate should be adjusted according to local gains. Researchers have developed block-wise dynamic learning rates that perform comparably to Adam with reduced memory use (Zhang et al., 2024b). Similar trends are seen in parameter-efficient fine-tuning, emphasizing the importance of local gains for learning rate adjustments (Zhang & Pilanci, 2024). Furthermore, theoretical analyses have raised doubts about the necessity of adaptive gradient methods. While Adam offers practical benefits, research (Li et al., 2024) indicates that the convergence rates of Adam and SGD are not significantly different.

Gradient at Initialization: Recent research has highlighted the importance of gradient patterns at initialization, demonstrating a strong correlation between these early signals and a model’s eventual performance. Pruning at Initialization (PaI) methods, inspired by the lottery ticket hypothesis (Frankle & Carbin, 2018), leverage this principle by iden-

tifying high-potential subnetworks before training begins. These techniques typically remove parameters associated with the lowest gradients or the weakest early learning responses (Tanaka et al., 2020; Frankle et al., 2020; Lee et al., 2018), emphasizing how initial gradient-based criteria can guide the formation of effective, sparse architectures.

From a gradient sparsity perspective, PaI methods effectively preserve the essential characteristics of the full network’s gradient distribution. The resulting subnetworks maintain similar gradient variance and overall gradient magnitude by masking out parameters tied to minimal gradient or learning response. This careful selection ensures that the pruned models exhibit performance levels on par with their unpruned counterparts despite operating with significantly fewer parameters.

A similar observation has also been revealed in Zero-Cost NAS studies (Abdelfattah et al., 2021; Li et al., 2023b; Xiang et al., 2023), which aim to predict the performance of untrained networks by analyzing gradient patterns, finding that gradient score rankings—such as the gradient sum—correlate more strongly with architectural structures than with data batches or initialization parameters. Research by (Bhardwaj et al., 2021) highlights that gradient flow patterns are inherently linked to a network’s architecture. Additionally, studies (Li et al., 2023b; Xiang et al., 2023) show that gradient sparsity, measured by mean and variance, is closely related to convergence rates and generalization ability. They emphasize calculating gradient sparsity blockwise due to the diverse distributions of gradients across parameter blocks. Moreover, (Lei et al., 2023) suggests that a balanced training procedure with low-variance gradients enhances sparse training.

### 3. Problem Setting

Notations. A neural network is defined based on a set of trainable parameters in specific architectures. We denote the neural network’s parameters as θ ∈ Rd, where d is the total number of parameters. The training loss function L(θ) defines the objective to be minimized. The parameter space is partitioned into B blocks based on the definition of network architectures, denoted as θ(i) ∈ Rd

i for i ∈

{1,2,...,B}, where d = Bi=1 di. Each parameter θj(i) within block i for j ∈ [di] is associated with its own gradient

gj(i) = ∇θ(i)

L(θ).

j

Key notations used throughout the paper are as follows: We denote t ∈ N as the index for the training step, η > 0 as the global learning rate, λ ≥ 0 as the weight decay

coefficient, µ as the momentum coefficient, gt(i) ∈ Rd

i

as the gradient of the loss w.r.t. θ(i) at step t. [di] is the index set {1,2,3,...,di} corresponding to the parameters in block i. And O(∗) means the complexity, here we use it

to measure the storage.

Stochastic Gradient-based Optimization: Given the loss function L(θ), the goal of a general optimization process is to update θ in the following form iteratively:

θt+1 = θt − ηtDt, (1)

where Dt denotes the update direction at step t. The choice of Dt defines the specific optimization algorithm. For SGD, the update direction is defined as the negative gradient of the loss with respect to θt:

Dt = gt, where gt = ∇L(θt) (2)

The first-order momentum term was introduced to SGD to enhance the optimisation process, and it is called SGD with momentum(SGDM) (Nesterov, 1983). The momentum m can be defined as:

mt = β1mt−1 + (1 − β1)gt (3) The update becomes:

Dt = mt (4)

This addition helps accelerate convergence by incorporating information from previous gradients to smooth out the update steps. Specifically, it reduces oscillations in the optimization trajectory, particularly in scenarios with steep or narrow ravines in the loss landscape. By maintaining a running average of past gradients, the momentum term allows SGD to move more consistently in directions that lead to faster convergence, addressing challenges like slow progress on flat regions of the loss surface.

Adaptive Gradient Methods: Adaptive gradient methods like Adam adopted first-order momentum mt as we mentioned above while introducing the second-order momentum vt, which tracks squared gradients to adjust the learning rate for each parameter, the vt defined as:

vt = β2vt−1 + (1 − β2)gt2. (5) The update direction for Adam is as follows:

1 √vˆt + ϵ

Dt = αtmt, where αt =

. (6)

vˆt term is the vt with bias correction. Notably, αt is the local learning rate gain (aka. adaptive learning rate). The key computational challenge is the storage and updating of vt, which requires O(d) additional memory.

Memory Efficient Adam: As Scaling Law (Kaplan et al., 2020) introduced, Transformer model sizes in recent days have significantly increased compared to the model size when Adam was introduced. Consequently, the memory

overhead of the Adam optimizer has become a significant issue, as it requires at least 3x times of memory compared to parameter size. Several approaches have been proposed to reduce the memory overhead of the second-order momentum vt, including (a) Adafactor (Shazeer & Stern, 2018) shares the v across dimensions, reducing storage from O(d) to O(

√

d). However, Adafactor trades off memory savings for lower update precision. (b) Low-bit optimisers quantize (Dettmers et al., 2021) the storage of vt to low-precision formats (e.g., 8-bit) to save memory. While effective, quantization introduces additional implementation complexity. (c) Adam-mini (Zhang et al., 2024b) partitions the block and uses the moving average of the estimated vt for each block, thereby reducing storage from O(d) to O(B). However, Adam-mini not only introduces additional computational costs compared to the Adam update process, but its complex partition policy is also incompatible with the default PyTorch partitioning; for example, in default PyTorch partitioning, the attention QKV considered the same group of parameters, while Adam-mini requires further partitioning them with based on heads or neurons. Furthermore, it adaptively calculates and updates αt over time, highlighting its intensive computational complexity. While these approaches reduce memory usage, they all retain the secondmomentum term vt with a trade-off to either performance or update speed.

Problem Statements. We aim to eliminate the need for the explicit second-order momentum vt entirely while maintaining effective learning rate adaptation. Instead of using αt(i) = η

, we aim to design a new rescaling factor

vˆt(i)+ϵ

αt(i) that adapts to the loss landscape without requiring the computation or storage of vt(i). Given a block-wise parameter θ(i), we seek a new function F such that:

αt(i) = F(g1(i),g2(i),...,gt(i)). (7) where F determines the local learning rate gain for each parameter block using the gradient history [g1(i),g2(i),...,gt(i)].

### 4. Methods

Considering the substantial memory overhead introduced by the second-order momentum in the Adam optimizer, this section explores strategies to reduce this cost by revisiting the foundational motivations for adaptive gradient methods.

In the following subsections, we design a memory-efficient learning rate local gain, termed g-SNR, to replace the second-order momentum. We analyse the distribution of g-SNR across different parameter groups throughout the network. This aligns with the motivation of parallel works (Zhang et al., 2024b) that focus on partitioning parameter

groups for learning rate adjustment. Furthermore, we investigate the behaviour of g-SNR during training, demonstrating how dynamic local gains can be replaced with constant preconditioned values calculated in the initial iterations.

Finally, we introduce our proposed method, SGD-SaI, detailing its design and implementation. This method builds on the insights derived from g-SNR analysis, offering a memory-efficient alternative to second-order momentum while maintaining competitive performance.

#### 4.1. Memory Efficient Local Gain: g-SNR

Adam builds upon RMSprop, designed to find a local gain for the learning rate, enabling parameter-specific adjustments within deep neural networks (Hinton et al., 2012; Kingma & Ba, 2014). By incorporating second-order momentum, Adam improves upon SGD by better handling problems with non-stationary objectives and tasks characterized by noise or sparse gradients (Kingma & Ba, 2014). This mechanism allows Adam to dynamically rescale gradients, effectively adjusting the learning pace across parameter blocks with distinct gradient patterns. Consequently, Adam outperforms SGD when training architectures with heterogeneity problems in the Hessian matrix, such as Transformers (Zhang et al., 2024a;b). Another key insight arises from the warm-up mechanism: even with second-order momentum, Adam still requires a warm-up phase to reduce the learning rate at the beginning of training, aiming to mitigate gradient variance (Liu et al., 2021). During this phase, gradients are known to be sparse and noisy. Reducing the learning rate directly during the warm-up phase effectively lowers gradient variance, stabilizing the training process straightforwardly and efficiently.

Intuitively, adaptive gradient methods dynamically adjust the learning rate for each parameter during training. This mechanism encourages parameters with less learning history to learn more while slowing down the learning pace for parameters progressing too quickly. Essentially, it acts as a compensatory approach to address learning imbalances across parameters after they arise. However, if we can predict and pre-empt these imbalances before they occur, we could potentially eliminate the need for second-order momentum, which relies on learning history to evaluate and correct them.

Considering the root cause of why learning imbalance occurred across different parameters, we discussed them in two main parts. Firstly, as inherited from the architecture characteristics, the parameter in different layers or with different architectures will receive distinct gradient pattern (Tanaka et al., 2020; Li et al., 2023b; Xiang et al., 2023), thus bringing the optimal learning rate for different parameters are distinct and need to be re-adjust with local gain (Hinton et al., 2012), Secondly, within the parameter groups, the

g-SNR (Bias Only) Distribution Across Parameter Blocks

g-SNR (Weights Only) Distribution Across Parameter Blocks

g-SNR (Mixed) Distribution Across Parameter Blocks

25

50

norm1 norm2

norm1 norm2

100

norm1 norm2

20

attn.qkv attn.proj

attn.qkv attn.proj

40

attn.qkv attn.proj

80

mlp.fc1 mlp.fc2 patch_embed.proj

mlp.fc1 mlp.fc2 patch_embed.proj

Frequency

mlp.fc1 mlp.fc2 cls_token

15

Frequency

Frequency

30

60

10

20

head

head

40

pos_embed

fc_norm

fc_norm

patch_embed.proj

head

5

10

20

fc_norm

0

0

0

19.6 19.7 19.8 19.9 20.0 20.1 20.2 g-SNR

0 100 200 300 400 500 600 700 800 g-SNR

0 100 200 300 400 500 600 700 800 g-SNR

- Figure 3. We observe that the g-SNR varies across different parameter blocks. However, for most weights, the parameter blocks that share the same structure across different transformer layers (blocks) tend to have similar g-SNR values. Additionally, the g-SNR values for the bias parameters are consistently low magnitude. Our method can be viewed as partitioning all parameter blocks based on their structure.

- Figure 4. We plot the g-SNR distribution over time for three different transformer blocks: shallow (block 0), middle (block 5), and deep (block 11). Additionally, we analyze some distinct types of parameter blocks. Our observations indicate that while the g-SNR values vary across different parameter blocks, they tend to remain relatively constant over time.

block is then given by

Gnorm(i) G(vari) + ϵ

Gsnr(i) =

,

where ϵ is a small constant added for numerical stability. To ensure consistent scaling across all blocks, we normalize the g-SNR of each block by the maximum g-SNR among all blocks:

Gsnr(i) maxk G(snrk)

G˜(snri) =

.

This normalization confines the g-SNR values between 0 and 1, facilitating a fair comparison and adjustment of learning rates across different parameter blocks. Thus, we establish the local gain by replacing vt with the following expression:

##### αt(i) = F(gt(i)) = G˜(snri)

where αt(i) represents the local gain at step t guided by the temporal value G˜snr(i) which determines the update direction Dt. This approach reduces the memory overhead of vt from O(d) to O(B).

Adapting the learning rate according to the normalized gradient signal-to-noise ratio significantly influences gradient variance during training. When the high gradient noise or sparsity in block i occurs, G˜(snri) tend to have relatively lower value, the learning rate η is scaled down by a factor α(i) = G˜(snri), resulting in a reduced learning rate η(i) = α(i)η. This adjustment decreases the magnitude of parameter updates for that block:

gradient can be noisy or sparse based on the objective and data, that will introduce imbalance update to parameters.

We propose using the gradient signal-to-noise ratio (g-SNR) introduced by (Xiang et al., 2023) to adjust the learning rate block-wisely, as it measures the norm and variance of gradients of the parameter block, which reflects overall update magnitude and variance of gradient between paramters. Specifically for each block i, the gradient norm (ℓ2-norm) and variance are calculated as

θt(+1i) = θt(i) − η(i)∇θ(i)L(θt). Lowering the learning rate mitigates the amplification of gradient noise, thereby reducing gradient variance within each training step, leading to smoother convergence and enhanced robustness (Liu et al., 2019). If G˜(snri) remains low across multiple batches, the continued reduction of η(i) further stabilizes training by preventing large, erratic updates.

di

di

2

2

1 di

gj(i)

gj(i) − g¯(i)

G(normi) =

, G(vari) =

,

j=1

j=1

di j=1 gj(i), and di is the number of param-

where g¯(i) = d1

i

eters in block i. The gradient signal-to-noise ratio for each

###### Experiments on Different Hyperparameters

(a) CNN in CIFAR-10 (b) ViT in ImageNet-1k

- Figure 5. Comparison of top-1 test accuracy distributions for CNNs on CIFAR-10 (Left) and ViTs on ImageNet-1k (Right) across different hyperparameter combinations. Each method demonstrates distinct performance trends, including Adam, AdamW, SGD, and SGD-SaI. Adam-Mini is only compared in the ViT case as its modification target on transformer training. SGD-SaI consistently shows enhanced robustness and performance under varying hyperparameter settings.

#### 4.2. Statistics Analysis for g-SNR

Building on the insights above, we implemented the g-SNR mechanism using PyTorch’s Default Partition (Zhang et al., 2024b), which computes g-SNR within each parameter block and dynamically re-scales the learning rate accordingly. To assess its effectiveness, we conducted experiments on Vision Transformer (ViT) pre-training tasks using ImageNet-1K, selecting ViT/S-16 for comprehensive tracing and analysis of gradient patterns throughout the training process.

Our analysis revealed that g-SNR remains relatively stable over time while exhibiting distinct patterns across different parameter classes, as shown in Fig. 4. Specifically, we examined transformer blocks from shallow, middle, and deep layers within the network and parameters outside the transformer blocks, such as positional embeddings.

Given the g-SNR definition we provide in the previous subsection, we analyze its behaviour as follows: As modern initialization schemes (e.g., Xavier (Kumar, 2017), Kaiming (He et al., 2015b)) ensure that at t = 0:

##### G(normi) (0) and G(vari)(0)

are well-controlled. This implies that G(snri)(0) starts from a stable, architecture-driven ratio. During the training process, parameters are updated and controlled by the step size η is the learning rate. Assuming η is sufficiently small to

stabilize training process, we have θt(+1i) ≈ θt(i). Thus, the change in parameters per iteration is small.Consider the gradient at iteration t + 1:

gt(+1i) = ∇θ(i)L(θt+1).

Consider a first-order Taylor expansion of the gradient around θ(i)(t):

gt(+1i) ≈ gt(i) + Jt(i)∆θt(i),

where Jt(i) is the Jacobian (or a first-order sensitivity matrix) of g(i) w.r.t. θ(i), and ∆θt(i) = θt(+1i) −θt(i). Since ∥∆θt(i)∥ is small, the change in the gradient vector is also small. Hence,

gj(i()t+1) ≈ gj(i()t), ∀j.

Because each component gj(i()t+1) differs only slightly from gj(i()t), their average and variance remain stable:

g¯t(+1i) ≈ g¯t(i), G(vari)(t+1) ≈ G(vari)(t).

Similarly, for gradient norm,

G(normi) (t+1) =

di

gj(i()t+1) 2 ≈ G(normi) (t).

j=1

Since both G(normi) (t) and G(vari)(t) remain nearly unchanged,

G(normi) (t+1) G(vari)(t+1) + ϵ

G(snri)(t+1) =

≈

G(normi) (t) G(vari)(t) + ϵ

= G(snri)(t).

Thus, Gsnr(t) remains effectively constant over iterations. Even though parameters change, the ”shape” or statistical

profile of the gradient distribution does not drastically alter. The g-SNR measures a dimensionless ratio that characterizes this shape. Minor parameter shifts do not significantly affect this ratio; hence, it remains nearly constant.

This finding aligns with the observation by (Xiang et al., 2023) that g-SNR strongly correlates with architecture. Leveraging this insight, we replaced the dynamic calculation of g-SNR with constant values determined during initialization, significantly reducing computational costs during each training step.

When calculating g-SNR using PyTorch’s Default Partition, we observed that the g-SNR values vary significantly across partitions. By leveraging constant g-SNR values, this approach effectively assigns a pre-conditioned learning rate scale to each partition. (Zhang et al., 2024b) highlights a key limitation of PyTorch’s default parameter partitioning: its lack of granularity for optimizers like Adam-mini. While PyTorch groups parameters such as attention QKV together, Adam-mini requires finer partitions, such as by attention heads or neurons, to perform effectively, especially in Transformer-based architectures. This limitation stems from the default partitioning’s failure to align with Hessian sub-block structures critical for optimization.

This observation does not hold true in our case. Our empirical results, shown in Fig. 5 and discussed further in Sec. 5, demonstrate that our method works effectively with PyTorch’s Default Partition and does not require any additional fine-grained partitioning strategies. The distribution of g-SNR across different partitions is detailed in Fig. 3, where we observe that, for most weights, parameter blocks sharing the same structure across different Transformer layers exhibit similar g-SNR values. Additionally, the g-SNR values for bias parameters remain consistently low, reflecting their uniform magnitude. A notable exception is the norm1 weights from blocks.0, which connect to the input from embedded patches, whereas all other norm1 weights connect to the output of the previous block. This observation highlights that our g-SNR values can effectively identify distinct characteristics among different parameter groups and the network’s topological impacts. Moreover, it indicates that gradient sparsity and noise levels vary across parameter groups, confirming the necessity of using a local gain mechanism to balance learning rates across partitions.

Notably, our approach, compatible with PyTorch’s Default

Partition, enables simultaneous updates of each coarsegrained parameter block and eliminates the need for dynamic learning rate calculations. This efficiency resulted in a threefold speedup in the optimizer update step compared to Adam-mini when training the GPT2-small model. Moreover, it reduces the implementation complexity associated with the exhaustive Hessian calculations required for fine-grained parameter partitioning (Zhang et al., 2024b).

In summary, instead of relying on second-order momentum to compute gradient history and adjust learning rates to address imbalanced updates after they occur, our g-SNR approach determines the gradient sparsity level at the first iteration of training. This enables assigning appropriate pre-conditioned learning rate scales to different parameter partitions, simplifying the update process, improving memory efficiency, and significantly speeding up optimization.

#### 4.3. Proposed Methods Detail: SGD-SaI

We propose a new method called SGD-SaI that removes adaptive gradient components by rescaling the learning rates of each parameter block using the g-SNR calculated from the initial batch. The algorithm details are presented in Algorithm 6. By leveraging the initial g-SNR, we capture the inherent gradient characteristics of different parameter blocks, allowing for a constant scaling factor that addresses the variations in gradient magnitudes across blocks.

As our method eliminates the dynamic terms associated with adaptive gradient algorithms, it only introduces a few computations at the first iteration compared to naive Stochastic Gradient Descent with Momentum (SGDM). Specifically, the additional computation involves calculating the g-SNR for each parameter block during the initial batch. After this initial computation, the training proceeds similarly to standard SGDM, making our method computationally efficient and comparable in complexity to traditional SGD.

To update the g-SNR based on the actual gradient sparsity without affecting the gradient computation, we adopt decoupled weight decay as proposed by Loshchilov and Hutter (Loshchilov & Hutter, 2019). Decoupled weight decay applies regularization directly to the parameters rather than incorporating it into the gradient computation. This approach is equivalent to regularization in SGD and allows us to accurately compute the gradient statistics needed for the g-SNR without the weight decay term distorting the gradient values. By doing so, we ensure that the g-SNR reflects the gradients’ true sparsity and noise characteristics.

Our implementation remains extremely straightforward, as we adopt the simplest approach that requires only minimal modifications to the existing SGD optimizer. This simplicity ensures that existing tricks and frameworks that support SGD can seamlessly integrate with and support our method.

thermore, we explore Parameter-Efficient Fine-Tuning (PEFT) tasks for GPT-2 LoRA(Hu et al., 2021) finetuning on the E2E (Novikova et al., 2017) dataset and Diffusion Model fine-tuning to capture visual concepts. For image classification, we report the top-1 validation accuracy, while for Large Language Model (LLM) finetuning tasks, to evaluate the results of the fine-tuning, we report metrics such as BLEU (Papineni et al., 2002), NIST (Doddington, 2002), MET (Banerjee & Lavie, 2005), ROUGE-L (Lin, 2004), and CIDEr (Vedantam et al., 2015). For all these metrics, higher scores indicate better performance. Additionally, we perform qualitative evaluations for the Diffusion Model (DM) fine-tuning task.

#### Algorithm 1 SGD-SaI

Require: T (total steps), η (learning rate), θi (i-th parameter block), L(θ) (loss function), λ (weight decay), µ (momentum), ϵ (small constant), maximize

- 1: for t ← 1 to T do
- 2: Compute gradient: gti ← ∇θiL(θt−1)
- 3: if maximize then
- 4: gti ← −gti
- 5: end if
- 6: /* Apply momentum */
- 7: if t > 1 then
- 8: mit ← µmit−1 + (1 − µ)gti
- 9: else
- 10: mit ← gti
- 11: /* Compute g-SNR */
- 12: Gisnr ←

Ginorm Givar + ϵ

- 13: /* Normalize g-SNR */
- 14: G˜isnr ←

Gisnr maxk Gksnr

- 15: end if
- 16: /* Apply weight decay */
- 17: θti ← θti−1 − ληθti−1
- 18: /* Update parameters with scaled learning rate */
- 19: θti ← θti − ηG˜isnrmit
- 20: end for

• Convolutional Neural Networks (CNNs). We study ResNet-18 (11M parameters) on the CIFAR-10 dataset, and architectures from NATS-Bench (Dong et al., 2021) on CIFAR-10, CIFAR-100, and ImageNet16120 (Krizhevsky & Hinton, 2009; Chrabaszcz et al., 2017). All these are image classification tasks, and we report the top-1 test accuracy as the evaluation metric.

#### 5.1. LLM Pre-train

Setups. We pre-train GPT-2-Small (125M) (Radford et al., 2019) on OpenWebText (Gokaslan & Cohen, 2019). We compare SGD-SaI with AdamW (Loshchilov & Hutter, 2019) and Adam-mini (Zhang et al., 2024b). We follow the same settings as described in the previous study (Zhang et al., 2024b). We analyse the loss metrics for each optimizer.

- Figure 6. Our Algorithm. we introduce a simple parameter-blockwise scaling using the normalized g-SNR to rescale the learning step size. This allows SGD to perform block-wise effective learning, unlocking its potential to work well on networks with block heterogeneity problems (Zhang et al., 2024a).

### 5. Experiments

This section evaluates our method through several tasks, including pre-training for Large Language Model (LLM) and Vision Transformer (ViT), Parameter-Efficient FineTuning (PEFT) tasks on LLM and Diffusion Model (DM), and traditional Convolutional Neural Network (CNN) tasks. The specific tasks are outlined as follows:

- • Large Language Model(Transformer Decode Only) We pre-train GPT-2 (Radford et al., 2019) on OpenWebText (Gokaslan & Cohen, 2019). We profile the optimizer state tensors’ memory usage and optimizer step time for GPT-2-XL(1.5B) and LLM2-7B.
- • Vision Transformer We investigate the Vision Transformer (ViT/S-16) (Dosovitskiy et al., 2021) on the ImageNet-1k dataset (Deng et al., 2009) for image classification tasks. We profile the optimizer state tensors’ memory usage and optimizer step time for ViT-H/14.
- • Parameter-Efficient Fine-Tuning (PEFT) LoRA Fur-

For large-scale LLMs, we provide profiling results focusing on memory usage and wall-clock time during the optimizer step for GPT-2-XL (1.5B parameters) and Llama-2 (7B parameters). Due to resource constraints, these results are limited to the optimizer step time and do not encompass full training runs. We compare SGD-SaI with SGDM, AdamW, Adam (Kingma & Ba, 2014), Adam-mini, and Prodigy (Mishchenko & Defazio, 2023). The reported metrics include memory usage of the state tensors and the time costs associated with the optimizer steps. All results were obtained using a single NVIDIA A100 (80GB).

Results. Figure 8 compares optimizers (AdamW, Adammini, and SGD-SaI) during pre-training of GPT-2-Small across multiple metrics. While SGD-SaI demonstrates a slightly slower initial convergence speed compared to the Adam family optimizers due to its design, it achieves superior final convergence with a lower training loss (outperforming Adam-mini by 0.13). Similarly, validation loss shows a marginal improvement, with SGD-SaI reducing it by 0.03 compared to Adam-mini.

Efficiency. For the GPT-2-Small pre-training task. Re-

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

- Figure 7. This figure displays the training and evaluation loss and accuracy of the ViT on ImageNet1k). Although our method has a slower convergence speed, we can still achieve comparable performance by the end of the training process. Additionally, our approach is designed to have a lower memory footprint and a faster optimization speed.

| | |
|---|---|
| |-0.13|
| | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | |2-nd momentum| | | | |
| | |-0.1| | | | |
| | | | | | | |
| | | | | | | |

Optimizers

-0.03

3x faster

- Figure 8. Metrics comparison of optimizers (AdamW, Adam-mini, and SGD-SaI) during pre-training of GPT-2 Small. The figure includes four subplots: (a) Train Loss shows that SGD-SaI achieves the lowest train loss, outperforming Adam-mini by 0.13. (b) Validation Loss illustrates a slight improvement in SGD-SaI with a reduction of 0.03 compared to Adam-mini. (c) Update Speed highlights that SGD-SaI is three times faster than Adam-mini, with AdamW showing moderate performance. (d) Memory Usage indicates that AdamW consumes 100% memory, while both Adammini and SGD-SaI utilize approximately half, demonstrating better efficiency. Annotated values provide clarity on performance metrics, with red highlights emphasizing improvements from Adammini.

the default PyTorch partitioning, highlighting its simplicity and efficiency. For the untrained models. By design, the state tensors for Adam and AdamW are approximately twice the size of the gradient, while Prodigy’s state tensors are roughly four times larger. In contrast, SGD-SaI has state tensors of the same size as standard SGDM. This effectively reduces memory usage by up to 75% compared to Prodigy and by 50% compared to Adam(W). The detailed discussion can be found in the Appendix B. As shown in Table 1, SGDSaI maintains a manageable memory footprint, enabling it to work with large models like Llama-2 (7B) without running into out-of-memory (OOM) errors. In contrast, other optimizers, such as AdamW and Prodigy, exceed available memory limits at this model size, highlighting the scalability challenges posed by memory-intensive optimizers when dealing with long context lengths in LLMs. Adam-Mini requires a partitioning strategy for different parameter groups while adjusting the learning rate adaptively at each time step. This increases memory usage and computational cost as different groups can not update simultaneously. For models larger than 1 billion parameters, the performance gains from Adam-Mini decrease by approximately 45%, while the reduction achieved with SGD-SaI remains around 50%.

#### 5.2. ViT Pre-train

Setups. We pre-train ViT-S/16 (Dosovitskiy et al., 2021) on the ImageNet1k dataset (Deng et al., 2009) for the image classification task. We compare SGD-SaI with AdamW (Loshchilov & Hutter, 2019) as well as popular optimizers including SGDM, Adam (Kingma & Ba, 2014), Adam-mini (Zhang et al., 2024b) and Prodigy (Mishchenko & Defazio, 2023). After conducting a grid search within the same hyperparameter range, we compare the optimiser results. We report the peak and mean of the top-1 validation accuracy to evaluate their generalisation ability and sensitivity to hyperparameter changes. Detailed hyperparameters are in Appendix A.1.

garding update speed, SGD-SaI demonstrates a significant advantage, being three times faster than Adam-mini in parameter updates and outperforming AdamW. Furthermore, the memory efficiency of SGD-SaI is noteworthy—it consumes only half the memory required by AdamW while maintaining performance comparable to Adam-mini, which employs intricate partitioning strategies. Unlike Adam-mini, which requires complex parameter partitioning (eg. users need to manually transform the Pytorch default partitions like the combined QKV block into separate Q, K and V blocks.), SGD-SaI achieves similar or better results using

Due to the intensive computational power requirements for the ViT variants during the grid search, we cannot provide

|Model|Method<br><br>|State Mem (GB)|Wall Time (ms)|
|---|---|---|---|
|GPT2-1.5B|SGDM<br><br>AdamW<br><br>Adam<br><br>Prodigy<br><br>Adam-Mini<br><br>SGD-SaI(ours)<br><br>|5.93 11.86 11.86 23.72 6.52 5.93<br><br>|41.0 ± 12.0 138.0 ± 6.0 145.0 ± 7.0 360.0 ± 45.0 223.0 ± 2.0 68.0 ± 21.0|
|Llama2-7B|SGDM<br><br>AdamW<br><br>Adam<br><br>Prodigy<br><br>Adam-Mini<br><br>SGD-SaI(ours)<br><br>|25.15 49.48 49.48 98.96 27.21 25.15<br><br>|100.0 ± 20.0 OOM OOM OOM 421.0 ± 22.0 180.0 ± 30.0|

hyperparameters are modified; thus, we exclude it from this part of the comparison for fairness.

Optimizer Peak@top1 (%) Avg@top1 (%) SGDM 63.80 ± 0.35 14.33 ± 19.38 Adam 61.56 ± 0.93 20.93 ± 22.05 Adam-Mini 72.29 ± 0.43 36.65 ± 35.39 AdamW 73.04 ± 0.31 37.21 ± 35.43 Prodigy 73.24 ± 0.21 N/A SGD-SaI(Ours) 72.92 ± 0.07 57.55 ± 18.46

Table 2. Comparison of peak and average top-1 validation accuracy on ImageNet-1k for ViT-S/16 trained from scratch. Each optimizer’s performance is evaluated over a hyperparameter search space, reporting the highest accuracy (Peak@top1) and the average accuracy (Avg@top1) across all trials. Results are averaged over three seeds, with standard deviations for statistical analysis. Our method achieves significantly higher robustness to hyperparameter variations, maintaining a high average performance (57.55%) and outperforming other optimizers by at least 20%.

Table 1. The efficiency metrics of various models with different optimizers were evaluated using an A100-80GB GPU. The table above summarizes the results, which include the tensor memory usage and wall-clock time (optimization step time measured in milliseconds) for each model-optimizer configuration. For the large language models (LLMs), experiments were conducted with a context length of 1024 and a batch size of 1. All models were profiled in full (FP32) precision.

the complete training results. Instead, we follow the same procedure outlined in Section 5.1 and present only the profiling results regarding memory usage and wall-clock time during the optimizer step for ViT-S/16 and ViT-H/14. Additionally, we compare SGD-SaI with SGDM, AdamW, Adam, Adam-mini, and Prodigy. All results were obtained using a single NVIDIA A100 with 80GB of memory.

Results. We report peak performance under the best hyperparameters, averaging results over three random seeds, and present the mean and standard deviation (see Table 2). Our simple re-scaling strategy significantly boosts SGDM’s performance from 63.80 to 72.92, nearly matching AdamW’s 73.04. Meanwhile, the recent SOTA optimizer Prodigy achieves a slightly higher peak at 73.24, though it requires additional one-time memory usage. We will discuss these results further in later sections. Notably, our approach achieves the lowest standard deviation (0.07) across three random seeds, compared to Prodigy’s second-lowest at 0.21, highlighting the stability of our method during training.

In addition, we examine average performance across the hyperparameter search grid using a rest setting that deviates from the best hyperparameters as a tweaked version. Under these conditions, we observe that most previous methods, including AdamW, struggle significantly, leading to dramatic drops in average performance. For example, AdamW, despite being an update over SGD intended to improve robustness to hyperparameters, achieves only 37.21 with a standard deviation of 35.43. In contrast, our method maintains overall performance, achieving an average of 57.55 with a much lower variance (standard deviation of 18.46). Prodigy, a parameter-free optimizer not designed to adjust learning rate and weight decay, fails to converge when these

Our method demonstrates superior robustness and effectiveness in training ViT-S/16 models from scratch on ImageNet1k, outperforming previous optimizers across peak and average performance metrics. While alternative optimizers like AdamW and Prodigy achieve high peak accuracy, their performance drops significantly under hyperparameter variations, highlighting their sensitivity. In contrast, our approach maintains a stable and high average accuracy across diverse hyperparameter settings with minimal standard deviation. It underscores its resilience to hyperparameter tuning and potential for more efficient and reliable model training in real-world applications. This stability makes our method particularly suitable for scenarios where hyperparameter tuning is constrained, offering a consistent and robust solution for training large models.

Although our method does not utilize adaptive gradient adjustments, it achieves a stable and steady learning pace, ultimately reaching a comparable performance, as shown in Figure 7. Empirically, this stability arises from our preconditioned learning rate, which ensures each step is wellcontrolled and converges reliably with sufficient training steps. In contrast, Adam-family optimization methods often achieve faster convergence but are prone to being trapped in suboptimal minima due to their aggressive adaptivity.

Efficiency. As shown in Table 3, our method achieves a wall-clock time for optimizer steps comparable to SGDM, while being significantly faster than Adam-mini, Adam(W), and Prodigy. We must note that we present the wall clock time for each optimizer step rather than the total runtime. We did not rely on the grid search results to report the total runtime because all grid search experiments were conducted on a cluster. Due to complex factors, such as the cluster’s I/O bottleneck and network congestion, distributed train-

ing can be considerably slowed down. We chose to maintain the same settings and device while profiling the LLM. Regarding memory usage, similar trends were observed during the pre-training of GPT-2 Small. For example, the ViT-H/14-0.66B model uses only 2.42 GB of memory with SGD-SaI, compared to 4.86 GB with AdamW and 9.70 GB with Prodigy. Our method reduces memory consumption by 50% compared to Adam(W) and by 75% compared to Prodigy. Further empirical analysis can be found in App. B.

|Model<br><br>|Method<br><br>|State Mem (GB)|Wall Time (ms)|
|---|---|---|---|
|ViT-S/16(0.0229B)<br><br>|SGDM AdamW Adam Prodigy Adam-Mini SGD-SaI(ours)<br><br>|0.08 0.17 0.17 0.33 0.08 0.08|7.9 ± 0.3<br><br>45.0 ± 8.0 50.0 ± 1.5 78.0 ± 0.0 84.0 ± 5.0 12.4 ± 0.2<br><br>|
|ViT-H/14(0.66B)|SGDM<br><br>AdamW<br><br>Adam<br><br>Prodigy<br><br>Adam-Mini<br><br>SGD-SaI(ours)<br><br>|2.42 4.86 4.86 9.70 2.54 2.42|40.0 ± 1.0<br><br>124.0 ± 4.0 127.0 ± 2.0 260.0 ± 3.0 220.0 ± 20.0<br><br>54.0 ± 13.0|

Table 3. We maintain the same settings as in Table 1. We ensure a comparable memory footprint to SGDM, while keeping the optimizer step time controlled, resulting in a performance that is 4-6 x faster than Adam-mini.

#### 5.3. Parameter Efficient Fine Tuning: PEFT

We primarily consider PEFT tasks on LLM fine-tuning and the Diffusion Model fine-tuning.

- 5.3.1. LLMS PARAMETER EFFICIENT FINE-TUNING

Setup. We fine-tune the GPT-2 model using the E2E dataset (Novikova et al., 2017). The current state-of-the-art (SOTA) methods include scaled-SGD and scaled-AdamW from (Zhang & Pilanci, 2024), which adjust the learning rates for parameters A and B with a Riemannian preconditioner. Our primary comparison is between SGD-SaI and these methods, along with Adam-mini. To evaluate the results of the fine-tuning, we report metrics such as BLEU (Papineni et al., 2002), NIST (Doddington, 2002), MET (Banerjee & Lavie, 2005), ROUGE-L (Lin, 2004), and CIDEr (Vedantam et al., 2015). For all these metrics, higher scores indicate better performance.

Results. We adopted the same experiment setting to investigate whether our methods suit LoRA Training (Hu et al., 2021). We set the default learning rate to 1e − 3 and weight decay to 1e − 2. Empirically, we observe that SGD-SaI outperforms previous state-of-the-art (SOTA) scaled optimizers and unscaled ones. Table 4 presents surprising results regarding the final scores for LoRA fine-tuning of the GPT-2 medium model with a rank of 4 on the E2E natural language generation tasks. With this simple precondition on SGDM, our method performs significantly better than the previous

SOTA strategy using rescaled SGD. Furthermore, our approach exhibits a substantial improvement over AdamW in fine-tuning the GPT-2 architecture, even without meticulous tuning and searching for hyperparameters.

|Method<br><br>|BLEU<br><br>|NIST<br><br>|MET|ROUGE-L<br><br>|CIDEr|
|---|---|---|---|---|---|
|SGDr = 4 scaled SGDr = 4 AdamWr = 4 Adam-Minir = 4 scaled AdamW r = 4<br><br>|66.6 69.2 68.9 68.7 69.6<br><br>|8.54 8.71 8.69 8.66 8.77|44.2 46.3 46.5 46.3 46.6<br><br>|68.2 70.9 71.3 71.1 71.8|2.32 2.48 2.51 2.50 2.52<br><br>|
|SGD-SaI (ours)r = 4|69.9<br><br>|8.81|46.7|72.1|2.53|

Table 4. This table presents scores for LoRA fine-tuning of GPT-2 medium model on E2E Natural Language Generation (NLG) challenge with different optimizers. SGD-SaI outperforms all scaled and unscaled optimizers on all evaluation metrics. In particular, our method closes the performance gap between SGD and AdamW and reveals its effectiveness in performing block-wise scaling.

- 5.3.2. DMS PARAMETER EFFICIENT FINE-TUNING

Setup. Using the diffusion model, we extend our experiments to include LoRA fine-tuning on image generation tasks. Specifically, we utilize the ChilloutMix model to address real-world concepts, following the same approach outlined in Mix-of-show (Gu et al., 2024; Zhang & Pilanci, 2024). Additionally, we compare our method with the state-of-the-art (SOTA) optimized approach using scaledAdamW (Zhang & Pilanci, 2024). To evaluate the images generated by the diffusion model, we conduct a qualitative assessment to determine which method captures visual concepts more effectively.

Results. Face generation is a challenging task; the model should understand the visual concept of a specific person’s face based on its prompt text. Here, we set the learning rate as default 0.1, a large enough default value. We observed as Fig. 9, even without carefully tuning the learning rate, our scaled methods have shown a significantly better ability to capture the visual concept of potter than the previous SOTA scaled approach scaled-AdamW (Zhang & Pilanci, 2024). It should be verified that our optimizer has better parameters robustness on training and leads to better convergence in final performance; this should be an essential benefit for the practical use of the optimizer.

- 5.4. Convolutional Neural Network(CNN)

Setup. We follow a similar approach to Section 5.2 for evaluating CNN models. A grid search is performed on ResNet18 (He et al., 2015a) using the CIFAR-10 dataset, and across various architectures from NATS-Bench (Dong et al., 2021) on CIFAR-10, CIFAR-100, and ImageNet16120 (Krizhevsky & Hinton, 2009; Chrabaszcz et al., 2017). All tasks involve image classification. We compare SGDSaI with traditional optimizers (SGD and Adam-family)

| |
|---|

| |
|---|

[Figure 1]

Scaled AdamW

[Figure 2]

Ours

- Figure 9. Generation results for prompt “a pencil sketch of 〈Vpotter〉” by Mix-of-Show model with scaledAdamW optimizers(up) and our optimizer(down). Our method generates photos that better capture the prompt and align with visual concepts from training samples; At the same time, previous SOTA-scaled AdamW has some significant bad cases that do not follow the prompt, we marked them with a red bounding box.

10 20 30 40 50 60 70 80 90 Top1 Accuracy (%)

0

20

40

60

80

100

120

Count

cifar10 Acc@1 Distribution

optimizer

Adam

SGD

SGD(nesterov)

AdamW

SGD-SaI

0 10 20 30 40 50 60 Top1 Accuracy (%)

0

10

20

30

40

50

60

70

80

Count

cifar100 Acc@1 Distribution

optimizer Adam

| |
|---|

SGD SGD(nesterov)

| |
|---|

AdamW SGD-SaI

| |
|---|

0 5 10 15 20 25 30 35 40 Top1 Accuracy (%)

0

20

40

60

80

100

Count

ImageNet16-120 Acc@1 Distribution

optimizer

Adam

SGD

SGD(nesterov)

AdamW

SGD-SaI

- Figure 10. These figures show the accuracy distributions of eleven architectures trained on different optimizers using the same hyperparameter candidates. This row presents the top-1 evaluation accuracy distributions on CIFAR-10, CIFAR-100 and ImageNet16-120. The curves in those histograms are the results of kernel density estimation (KDE).

and report top-1 test accuracy. Details of the grid search experiments are provided in Appendix A.2.

Results. Figure 5 (left graph) presents the performance of ResNet18. Our method achieves a peak accuracy of 95.36%, which not only surpasses that of Adam(W) and SGD but also shows greater stability. In addition, we evaluated a range of search spaces, including datasets such as CIFAR10, CIFAR-100, and ImageNet16-120, as well as architectures of varying sizes. We conducted a grid search across eleven architectures, testing three learning rates and four weight decay values. The distribution of top-1 accuracies is illustrated in Fig. 10, which demonstrates the stability of our method across different architectures and hyperparameter settings. Our approach results in models with lower standard deviations and higher mean accuracies, indicating enhanced stability and generalization. These findings highlight the robustness of our method across various CNN architectures.

### 6. Conclusion

In summary, our results demonstrate that simply applying selective learning rate scaling at initialization (SGDSaI) can unlock performance comparable to—if not better than—leading adaptive gradient methods like AdamW, all while retaining the simplicity and efficiency of SGDM. By leveraging g-SNR to guide parameter group scaling, SGDSaI not only mitigates early training imbalances but also

substantially reduces optimizer memory overhead, enabling more resource-efficient model training. Its robustness across a wide range of Transformer-based tasks, including ImageNet classification with ViT, GPT-2 pretraining, LoRA fine-tuning, and diffusion modelling, underscores its versatility and practicality.

### 7. Limitation

While SGD-SaI demonstrates promising results across various Transformer-based tasks, our study is constrained by limited computational resources, preventing us from conducting large-scale pre-training on more extensive models such as Llama-2-7B. This remains an avenue for future research. However, to address the efficiency challenges of training larger models, we have performed detailed profiling of GPU memory usage and optimizer step speed on these architectures. These preliminary analyses indicate the potential scalability of SGD-SaI, but comprehensive evaluations on larger-scale models are necessary to establish its effectiveness and efficiency in such settings fully. Moreover, our methods ensure a steady and stable update during training, allowing the model to converge better in a given task with sufficient training steps. Thus, we might observe that the convergence speed is relatively lower than Adam’s in the early stage of training; as our primary focus is to investigate the effectiveness of the SaI approach, we left the acceleration of convergence speed in future work.

### References

Abdelfattah, M. S., Mehrotra, A., Dudziak, Ł., and Lane, N. D. Zero-cost proxies for lightweight nas. In International Conference on Learning Representations, 2021.

Banerjee, S. and Lavie, A. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, pp. 65–72, 2005.

Bernstein, J., Wang, Y.-X., Azizzadenesheli, K., and Anandkumar, A. signsgd: Compressed optimisation for nonconvex problems, 2018. URL https://arxiv.org/ abs/1802.04434.

Beyer, L., Zhai, X., and Kolesnikov, A. Better plain vit baselines for imagenet-1k, 2022a. URL https: //arxiv.org/abs/2205.01580.

Beyer, L., Zhai, X., and Kolesnikov, A. Big vision. https://github.com/google-research/ big_vision, 2022b.

Bhardwaj, K., Li, G., and Marculescu, R. How does topology influence gradient propagation and model performance of deep networks with densenet-type skip connections?, 2021. URL https://arxiv.org/abs/ 1910.00780.

Chrabaszcz, P., Loshchilov, I., and Hutter, F. A downsampled variant of imagenet as an alternative to the cifar datasets, 2017.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Dettmers, T. and Zettlemoyer, L. The case for 4-bit precision: k-bit inference scaling laws. In International Conference on Machine Learning, pp. 7750–7774. PMLR, 2023.

Dettmers, T., Lewis, M., Shleifer, S., and Zettlemoyer, L. 8-bit optimizers via block-wise quantization. CoRR, abs/2110.02861, 2021. URL https://arxiv.org/ abs/2110.02861.

Dettmers, T., Lewis, M., Belkada, Y., and Zettlemoyer, L. Llm. int8 (): 8-bit matrix multiplication for transformers at scale. arXiv preprint arXiv:2208.07339, 2022.

Doddington, G. Automatic evaluation of machine translation quality using n-gram co-occurrence statistics. In Proceedings of the second international conference on Human Language Technology Research, pp. 138–145, 2002.

Dong, X., Liu, L., Musial, K., and Gabrys, B. NATSBench: Benchmarking nas algorithms for architecture topology and size. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2021. doi: 10.1109/ TPAMI.2021.3054824. doi:10.1109/TPAMI.2021.

3054824.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer,

- M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby,
- N. An image is worth 16x16 words: Transformers for image recognition at scale, 2021. URL https: //arxiv.org/abs/2010.11929.

Dozat, T. Incorporating Nesterov Momentum into Adam. In Proceedings of the 4th International Conference on Learning Representations, pp. 1–4, 2016.

Duchi, J., Hazan, E., and Singer, Y. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12 (61):2121–2159, 2011. URL http://jmlr.org/ papers/v12/duchi11a.html.

Frankle, J. and Carbin, M. The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv preprint arXiv:1803.03635, 2018.

Frankle, J., Dziugaite, G. K., Roy, D. M., and Carbin, M. Pruning neural networks at initialization: Why are we missing the mark? arXiv preprint arXiv:2009.08576, 2020.

Ghiasi, A., Shafahi, A., and Ardekani, R. Improving robustness with adaptive weight decay, 2023. URL https://arxiv.org/abs/2210.00094.

Ghorbani, B., Suo, D., Cardoze, D., Dahl, G., Cohen, J., Gilmer, J., Agarwal, N., Krishnan, S., Medapati, S., and Nado, Z. Adaptive gradient methods at the edge of stability. 2022. URL https://arxiv.org/abs/2207.

14484.

Gokaslan, A. and Cohen, V. Openwebtext corpus. http://Skylion007.github.io/ OpenWebTextCorpus, 2019.

Graves, A. Generating sequences with recurrent neural networks, 2014. URL https://arxiv.org/abs/ 1308.0850.

Gu, Y., Wang, X., Wu, J. Z., Shi, Y., Chen, Y., Fan, Z., Xiao, W., Zhao, R., Chang, S., Wu, W., et al. Mixof-show: Decentralized low-rank adaptation for multiconcept customization of diffusion models. Advances in Neural Information Processing Systems, 36, 2024.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition, 2015a. URL https: //arxiv.org/abs/1512.03385.

He, K., Zhang, X., Ren, S., and Sun, J. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision, pp. 1026–1034, 2015b.

Hinton, G., Srivastava, N., and Swersky, K. Neural networks for machine learning lecture 6a overview of mini-batch gradient descent. Cited on, 14(8):2, 2012.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Krizhevsky, A. and Hinton, G. Learning multiple layers of features from tiny images. Technical Report 0, University of Toronto, Toronto, Ontario, 2009. URL https://www.cs.toronto.edu/ ˜kriz/learning-features-2009-TR.pdf.

Kumar, S. K. On weight initialization in deep neural networks. arXiv preprint arXiv:1704.08863, 2017.

Kunstner, F., Chen, J., Lavington, J. W., and Schmidt, M. Noise is not the main factor behind the gap between sgd and adam on transformers, but sign descent might be. arXiv preprint arXiv:2304.13960, 2023.

Lee, N., Ajanthan, T., and Torr, P. H. Snip: Single-shot network pruning based on connection sensitivity. arXiv preprint arXiv:1810.02340, 2018.

Lei, B., Xu, D., Zhang, R., He, S., and Mallick, B. K. Balance is essence: Accelerating sparse training via adaptive gradient correction, 2023. URL https://arxiv. org/abs/2301.03573.

Li, B., Chen, J., and Zhu, J. Memory efficient optimizers with 4-bit states, 2023a. URL https://arxiv.org/ abs/2309.01507.

- Li, G., Yang, Y., Bhardwaj, K., and Marculescu, R. Zico: Zero-shot nas via inverse coefficient of variation on gradients. In The Eleventh International Conference on Learning Representations, 2023b.
- Li, H., Rakhlin, A., and Jadbabaie, A. Convergence of adam under relaxed assumptions. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA,

2024. Curran Associates Inc.

Lin, C.-Y. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pp. 74–81, 2004.

Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., and Han, J. On the variance of the adaptive learning rate and beyond. arXiv preprint arXiv:1908.03265, 2019.

Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., and Han, J. On the variance of the adaptive learning rate and beyond, 2021.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization, 2019.

Mishchenko, K. and Defazio, A. Prodigy: An expeditiously adaptive parameter-free learner. arXiv preprint arXiv:2306.06101, 2023.

Nesterov, Y. E. A method of solving a convex programming problem with convergence rate o(1/k2). Doklady Akademii Nauk SSSR, 269(3):543–547, 1983. URL http://mi.mathnet.ru/dan46009. MathNet: http://mi.mathnet.ru/dan46009, MathSciNet: http://mathscinet.ams.org/ mathscinet-getitem?mr=0701288, zbMATH: https://zbmath.org/?q=an:0535.90071.

Novikova, J., Duˇsek, O., and Rieser, V. The e2e dataset: New challenges for end-to-end generation. arXiv preprint arXiv:1706.09254, 2017.

Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pp. 311–318, 2002.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models, 2022. URL https://arxiv.org/ abs/2112.10752.

Shazeer, N. and Stern, M. Adafactor: Adaptive learning rates with sublinear memory cost. In International Conference on Machine Learning, pp. 4596–4604. PMLR, 2018.

Steiner, A., Kolesnikov, A., Zhai, X., Wightman, R., Uszkoreit, J., and Beyer, L. How to train your vit? data, augmentation, and regularization in vision transformers, 2022. URL https://arxiv.org/abs/2106.10270.

Tanaka, H., Kunin, D., Yamins, D. L., and Ganguli, S. Pruning neural networks without any data by iteratively conserving synaptic flow. Advances in neural information processing systems, 33:6377–6389, 2020.

Team, G., Anil, R., Borgeaud, S., Wu, Y., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Vaswani, A. Attention is all you need. arXiv preprint arXiv:1706.03762, 2017.

Vedantam, R., Lawrence Zitnick, C., and Parikh, D. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4566–4575, 2015.

Ward, R., Wu, X., and Bottou, L. Adagrad stepsizes: Sharp convergence over nonconvex landscapes. Journal of Machine Learning Research, 21(219):1–30, 2020.

Xiang, L., Hunter, R., Xu, M., Dudziak, Ł., and Wen, H. Exploiting network compressibility and topology in zerocost nas. In AutoML Conference 2023, 2023.

Xiao, T., Singh, M., Mintun, E., Darrell, T., Doll´ar, P., and Girshick, R. Early convolutions help transformers see better, 2021. URL https://arxiv.org/abs/ 2106.14881.

Zeiler, M. D. Adadelta: An adaptive learning rate method, 2012. URL https://arxiv.org/abs/ 1212.5701.

Zhang, F. and Pilanci, M. Riemannian preconditioned lora for fine-tuning foundation models. arXiv preprint arXiv:2402.02347, 2024.

Zhang, Y., Chen, C., Ding, T., Li, Z., Sun, R., and Luo, Z.Q. Why transformers need adam: A hessian perspective. arXiv preprint arXiv:2402.16788, 2024a.

Zhang, Y., Chen, C., Li, Z., Ding, T., Wu, C., Ye, Y., Luo, Z.-Q., and Sun, R. Adam-mini: Use fewer learning rates to gain more, 2024b. URL https://arxiv.org/ abs/2406.16793.

### A. More Experiments Details

#### A.1. Details for ViT Experiments

In this section, we will list list the settings of the experiment regarding to Section 5.2.

Hyperparameter Settings: We start by following the settings in (Beyer et al., 2022a; Steiner et al., 2022; Beyer et al., 2022b); Specifically, we include NesterovSGD as a baseline, offering better performance than naive SGD. All optimizers are tested using a grid search within the same hyperparameter ranges: learning rate lr ∈ {0.1,0.01,0.001,0.0001} and weight decay wd ∈ {0.01,0.001,0.0001}.

#### A.2. Details for CNN Experiments

In this section, we will list the settings of the experiment regarding to Section 5.4.

Models and Datasets: We follow the same settings and train some CNN-based architectures proposed in NATSBenchmark (Dong et al., 2021). We test the optimizers on CIFAR-10/CIFAR-100 (Krizhevsky & Hinton, 2009) and ImageNet16-120 (Chrabaszcz et al., 2017). Based on the NATS-Benchmark work, we test different sizes of architectures. Here, we select ten architectures with top-10 validation accuracy and one architecture with bottom-1 validation accuracy in terms of different datasets and training epochs to present.

Hyperparameter Settings: The optimal learning rate and weight decay are chosen by performing the grid search. The learning rate and weight decay are selected from η ∈ {0.1,0.01,0.001} and λ ∈ {0.5,0.05,0.005,0.0005}, respectively. We use the same cosine annealing scheduler on three datasets without learning rate warmup. We use the same data augmentation methods and set the batch size to 256 for all datasets. The experiments are designed to run for full training without early stopping. There is no linear scaling on the initial weight decay either since we are doing the grid search within a feasible range. The seed is only 777 which is the same seed reported by NATS-Benchmark on Size Search Space. The original NATS-Benchmark were produced using SGD with a fixed learning rate 0.1 and weight decay 0.0005 and the default setting of the Nesterov Momentum. For fair comparison, we apply the same grid search policy for SGD, as the baseline with or without Nesterov Momentum.

Results: The performance of various architectures has been represented as histograms showing top-1 accuracy on different datasets. Fig. 10 demonstrates that our method outperforms other optimizers in terms of evaluation accuracies within the same hyperparameter search space.

#### A.3. Extra Results for ResNet18 on CIFAR10

As a classic model of the CNNs, we also conduct the grid search on ResNet18 as an extended experiment.

Models and Datasets: We follow the similar setting in the Section 5.4. We particularly choose the CIFAR10 (Krizhevsky & Hinton, 2009) as the dataset we test on. We test on the classic ResNet18 model.

Hyperparameter Settings: Since we are focusing on a single model with one dataset—unlike the NAST-Benchmark CNN experiments discussed in Section 5.4—we are scaling up our search by exploring a wider range of learning rates and weight decays. The learning rates are chosen from the set η ∈ {0.1,0.01,0.001,0.0001} and the weight decays from λ ∈ {0.5,0.05,0.005,0.0005,0.00005}. We will repeat our grid search three times using three different random seeds. The random seeds used for the experiments are {42, 888, 999}. We opted for a step learning rate scheduler rather than a cosine annealing scheduler to test our method’s resilience to different learning rate scheduling policies. The learning rate will decrease by a factor of 10 every 80 epochs, with a total of 200 epochs for training. Our data augmentation methods remain consistent, with a batch size set to 128. These experiments will run for the entire training duration without early stopping, and there will be no linear scaling applied to the initial weight decay, as we are conducting a grid search within a reasonable range. The distribution of accuracies averaged over the three seeds for each hyperparameter combination is depicted in Fig.5. The best performance of each optimizer, along with the optimal learning rate and weight decay, is annotated with red numbers on the graph. For simplicity, we are only testing the Stochastic Gradient Descent with Momentum (SGDM) as the baseline. The momentum is set to the default value of 0.9, consistent with both the Adam(W) optimizer and our method to ensure fair comparison.

Results: The grid search results are shown in the Fig. 5. Not only does our method converge better (ours 95.36% v.s. SGDM 95.26%), but it also demonstrates greater resilience to changes in hyperparameters. This means the performance is less likely to downgrade compared to SGDM.

All the experiments in this paper were conducted using various types of GPUs, including NVIDIA GeForce RTX 3090, NVIDIA A100 PCIe 40GB, and NVIDIA A100 80GB. To ensure consistent experimental conditions, each experiment was conducted using only one GPU type.

### B. Optimizer Analysis

This section provides a supplementary analysis for Section 5. We will detail the optimizers and empirically estimate the lower boundary of the state tensors in memory.

#### Algorithm 2 SGD

Algorithm 3 Adam Require: t (step), η (lr), θi (i-th params), L(θ) (loss function), λ (weight decay), β1,β2 (betas), maximize

Require: t (step), η (lr), θi (i-th params), L(θ) (loss function), λ

(weight decay), µ (momentum), τ (dampening), maximize

- 1: repeat
- 2: for t ← 1 do
- 3: gti ← ∇L(θti−1)
- 4:
- 5: /* do weight decay */
- 6: gti ← gti + λθti−1
- 7:
- 8: /* do momentum */
- 9: if µ ̸= 0 and t > 1 then
- 10: mit ← µmit−1 + (1 − τ)gti
- 11: else
- 12: mit ← gti
- 13: end if
- 14:
- 15: if maximize then
- 16: θti ← θti−1 + ηmit
- 17: else
- 18: θti ← θti−1 − ηmit
- 19: end if
- 20: end for
- 21: until epochs end

- 1: repeat
- 2: for t ← 1 do
- 3: if maximize then
- 4: gti ← −∇L(θti−1)
- 5: else
- 6: gti ← ∇L(θti−1)
- 7: end if
- 8:
- 9: /* do weight decay */
- 10: gti ← gti + λθti−1
- 11:
- 12: /* do momentum */
- 13: if t > 1 then
- 14: mit ← β1mit−1 + (1 − β1)gti
- 15: vti ← β2vti−1 + (1 − β2)(gti)2
- 16: else
- 17: mit ← (1 − β1)gti; vti ← (1 − β2)(gti)2
- 18: end if
- 19: mˆit ← m

i t

1−β1t ;vˆti ← v

i t

1−β2t

- 20:
- 21: θti ← θti−1 − ˆη vti+ϵ

mit

- 22: end for
- 23: until epochs end

#### B.1. Break Down SGD

Stochastic Gradient Descent (SGD) is faster than adaptive gradient methods primarily due to its simplicity. The key difference between SGD and these adaptive methods is that SGD uses a fixed learning rate, while adaptive methods adjust the learning rate dynamically for each parameter at each step. This adjustment can be done at the element level, as seen in Adam(W), or at the block level, like in Adammini.

14 for AdamW), the first-order moment m and the secondorder moment v are stored in GPU memory throughout the entire training process.

The advantages of SGD can be summarized as follows: a. Runtime efficiency: It offers a fast and efficient iteration time for each optimization step. b. Memory efficiency: When using momentum, SGD requires only one instance of the gradients 2 and incurs no additional memory overhead when momentum is not applied.

#### B.2. Break Down Adam(W)

The main difference between Adam and AdamW lies in how they apply weight decay. AdamW applies a direct penalty to the weights themselves, which is known as Decoupled Weight Decay, whereas Adam applies the penalty to the gradients at the outset, utilizing L2 Regularization. Both algorithms enhance adaptability by scaling the learning rate for each parameter individually. In every optimization step, the scaling ratios are recalculated by dividing the first-order moment by the square root of the second-order moment, both of which are maintained and updated in state tensors. As illustrated in Alog. 3 (line 17 for Adam) and Alog. 4 (line

Generally, the estimated minimum memory requirement for the state tensors in both Adam and AdamW is approximately twice the size of the gradient tensors. This is because m and v share the same shape as the corresponding gradient tensors.

#### B.3. Break Down Adam-mini

Adam-mini is a variant of the Adam optimizer. As shown in the Alog. 5, Adam-mini redesigns the adaptive update rules by using the mean of the squared gradients instead of the original squared gradients in most layers except for the embedding layer. This version of Adam reduces the number of learning rates to the number of blocks in each layer while keeping the update rules unchanged in the embedding layers. Therefore, the reduction in memory usage is influenced by the proportion of non-embedding parameters in the model. This limitation not only restricts the potential for memory savings but also incurs additional computational costs due to the extra operations (see Alog. 5 Line 15, 16) needed

Algorithm 4 AdamW Require: t (step), η (lr), θi (i-th params), L(θ) (loss function), λ (weight decay), β1,β2 (betas), maximize

Algorithm 5 Adam-mini Require: t (step), η (lr), θi (i-th params), L(θ) (loss function), λ (weight decay), β1,β2 (betas), maximize

- 1: repeat
- 2: for t ← 1 do
- 3: if maximize then
- 4: gti ← −∇L(θti−1)
- 5: else
- 6: gti ← ∇L(θti−1)
- 7: end if
- 8: /* do momentum */
- 9: if t > 1 then
- 10: mit ← β1mit−1 + (1 − β1)gti
- 11: vti ← β2vti−1 + (1 − β2)(gti)2
- 12: else
- 13: mit ← (1 − β1)gti; vti ← (1 − β2)(gti)2
- 14: end if
- 15: mˆit ← m

i t

1−β1t ;vˆti ← v

i t

1−β2t

- 16:
- 17: /* do weight decay */
- 18: θti ← θti−1 − ληθti−1
- 19:
- 20: θti ← θti−1 − ˆη vti+ϵ

mit

- 21: end for
- 22: until epochs end

- 1: repeat
- 2: for t ← 1 do
- 3: if maximize then
- 4: gti ← −∇L(θti−1)
- 5: else
- 6: gti ← ∇L(θti−1)
- 7: end if
- 8:
- 9: /* do momentum */
- 10: if t > 1 then
- 11: mit ← β1mit−1 + (1 − β1)gti
- 12: if θti−1 ∈ embedding layer then

- 13: vti ← β2vti−1 + (1 − β2)(gti)2
- 14: else
- 15: Divide θti−1 into Q,K heads if needed.
- 16: vti ← β2vti−1 + (1 − β2)Mean((gti)2)
- 17: end if
- 18: else
- 19: mit ← (1 − β1)gti; vti ← (1 − β2)(gti)2
- 20: end if
- 21: mˆit ← m

i t

1−β1t ;vˆti ← v

i t

1−β2t

- 22:
- 23: /* do weight decay */
- 24: θti ← θti−1 − ληθti−1
- 25:
- 26: θti ← θti−1 − ˆη vti+ϵ

mit

- 27: end for
- 28: until epochs end

when calculating the new v compared to the original Adam algorithm.

Regarding the lower boundary of the state tensor memory, as mentioned in (Zhang et al., 2024b), Adam-mini can reduce the memory used for the Adam optimizer’s v by at least 90%. This results in a memory cost savings of approximately 45% to 50% compared to the original Adam.

the memory for the tensor state is occupied by four tensors: m,v,x0,s (Algo. 6 Line 11, 12, 17, 18). Consequently, Prodigy can be very memory-intensive when applied to large models with billions of parameters.

#### B.4. Break Down Prodigy

Prodigy is one of the most popular variants of the Adam optimizer, offering a new approach to calculating the step size. It alleviates the need for extensive learning rate tuning. While most of Adam’s update rules remain unchanged, Prodigy introduces a new scaling ratio, denoted as d (for D-Adaption), which adaptively adjusts the learning rate. To update the scaling ratio d for each optimization step, Prodigy requires the maintenance of two additional tensors: the initial weight value x0 and the denominator s, both of which share the same shape as the gradient.

As a result, the lower boundary for estimating the memory required by Prodigy’s state tensor is approximately four times the size of the gradient by default. The majority of

### C. Profiling Results on ”g-SNR Calculation” Stage

As discussed in Section B, our method requires calculating the scale ratio during the initial step of the optimization process, which we refer to as the ”g-SNR Calculation” stage. This procedure may take extra time because it involves calculating both the gradient norm and its standard deviation. As shown in Table 5 and Table 6, although square root operations are generally considered more computationally intensive than standard float addition and multiplication, the time taken for the ”g-SNR Calculation” is still relatively small. Since this calculation is performed only once during

|Model|Method<br><br>|Iter Times (ms)|g-SNR Calc (ms)<br><br>|
|---|---|---|---|
|ViT-S/16<br><br>|SGDM SGD-SaI (ours)|12.2 ± 2.9<br><br>13.7 ± 3.8<br><br><br>|0 14.5|
|ViT-H/14|SGDM<br><br>SGD-SaI (ours)|48.6 ± 7.8<br><br>65.5 ± 10.0<br><br>|0 43.3|
|GPT2-1.5B<br><br>|SGDM SGD-SaI (ours)|287.4 ± 0.9 340.1 ± 1.1<br><br>|0 267.6|

Algorithm 6 Prodigy Require: t (step), η (lr, default 1 with cosine annealing),

θi (i-th params), L(θ) (loss function), λ (weight decay), β1,β2 (betas), maximize, d0 > 0 (default 1e−6), x0

- 1: repeat
- 2: for t ← 1 do
- 3: if maximize then
- 4: gti ← −∇L(θti−1)
- 5: else
- 6: gti ← ∇L(θti−1)
- 7: end if
- 8:
- 9: /* do momentum */
- 10: if t > 1 then
- 11: mit ← β1mit−1 + (1 − β1)dtgti
- 12: vti ← β2vti−1 + (1 − β2)d2t(gti)2
- 13: else
- 14: mit ← (1 − β1)dtgti; vti ← (1−β2)d2t(gti)2
- 15: rt−1 = 0; st−1 = 0
- 16: end if
- 17: rt = √β2rt−1 + (1 −

√β2)ηd2t⟨gti,x0 − xt⟩

- 18: st = √β2st−1 + (1 −

√β2)ηd2tgti

- 19: dtˆ+1 = r

k

∥st∥1

- 20: dt+1 = max(dk,dtˆ+1)
- 21:
- 22: /* do weight decay */
- 23: θti ← θti−1 − ληθti−1
- 24:
- 25: θti ← θti−1 − ηd

t

√

vti+dtϵ

mit

- 26: end for
- 27: until epochs end

- Table 5. RTX 3090 Profile Results. The results here are based on a single NVIDIA GeForce RTX 3090 GPU. The trials were conducted over 20 iterations, recording the time taken for each optimization step, which is referred to as the ”iteration time” column. We compared the time taken for the g-SNR calculation stage and found that it takes an equal amount of time or less than an optimization step. However, since this calculation is only performed once, it is considered tolerable.

|Model<br><br>|Method|Iter Times (ms)|g-SNR Calc (ms)<br><br>|
|---|---|---|---|
|ViT-S/16<br><br>|SGDM SGD-SaI (ours)|7.1 ± 0.0<br><br>13.0 ± 0.1<br><br>|0 13.7|
|ViT-H/14<br><br>|SGDM SGD-SaI (ours)|51.9 ± 10.7<br><br>63.1 ± 7.8<br><br>|0 106.0|
|GPT2-1.5B<br><br>|SGDM SGD-SaI (ours)|353.2 ± 0.9 392.8 ± 0.2<br><br>|0<br><br>353.8|

- Table 6. A100 PCIe 40GB Profile Results. It follows the same setting as Table 5, expect for the GPU type.

compared to Adam(W) is primarily due to the ability to reduce memory usage, allowing for larger batch sizes and enabling more data to be processed in parallel. In contrast, we not only reduce the memory footprint but also eliminate the entire adaptive local gain calculation, achieving a significant breakthrough in both memory and computational efficiency.

each training procedure by design, its duration is negligible compared to the overall training iterations. Therefore, our method remains efficient for optimization in the long run.

### D. Algorithm Overview

In this section, we will provide an overview of how SGDSaI operates. As a non-adaptive gradient method, SGD-SaI calculates the preconditioned scaling factor based on the g-SNR values before applying the first batch of data in the optimization step. These scaling factors vary across different partitions, as they are closely linked to the architectures being utilized. However, once established, they remain constant throughout the entire training process.

While Adam-mini is memory-efficient, its complex partition rules and repetitive local gain recalculation result in significant computational costs. The improvement in throughput

