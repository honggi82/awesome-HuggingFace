# arXiv:2403.07487v4[cs.CV]3Aug2024

## Motion Mamba: Efficient and Long Sequence Motion Generation

Zeyu Zhang12∗† , Akide Liu1∗ , Ian Reid3 , Richard Hartley2 , Bohan Zhuang1 , and Hao Tang4

1 Monash University 2 The Australian National University 3 Mohamed bin Zayed University of Artificial Intelligence 4 National Key Laboratory for Multimedia Information Processing, School of Computer Science, Peking University

###### https://steve-zeyu-zhang.github.io/MotionMamba

The character is running straight then slow down a bit. The character is walking in a "S" shape trajectory.

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

The character is walking straight then turn right.

The character is dancing ballet.

Fig. 1: Motion Mamba has achieved significantly superior performance on long squence modeling and motion generation efficiency compared with other well-designed state-of-the-art methods such as MLD [6], MotionDiffuse [55], and MDM [50].

Abstract. Human motion generation stands as a significant pursuit in generative computer vision, while achieving long-sequence and efficient motion generation remains challenging. Recent advancements in state space models (SSMs), notably Mamba, have showcased considerable promise in long sequence modeling with an efficient hardware-aware design, which appears to be a significant direction upon building motion generation model. Nevertheless, adapting SSMs to motion generation faces hurdles since the lack of a specialized design architecture to model motion sequence. To address these challenges, we propose Motion Mamba, a simple yet efficient approach that presents the pioneering motion generation model utilized SSMs. Specifically, we design a Hierarchical Temporal Mamba (HTM) block to process temporal data by ensemble varying numbers of isolated SSM modules across a symmetric U-Net architecture aimed at preserving motion consistency between frames. We also design a Bidirectional Spatial Mamba (BSM) block

∗Equal contribution. †Work done while being a research assistant at Monash University.

Corresponding author: bjdxtanghao@gmail.com

to bidirectionally process latent poses, to enhance accurate motion generation within a temporal frame. Our proposed method achieves up to 50% FID improvement and up to 4 times faster on the HumanML3D and KIT-ML datasets compared to the previous best diffusion-based method, which demonstrates strong capabilities of high-quality long sequence motion modeling and real-time human motion generation.

Keywords: Human Motion Generation · Selective State Space Models · Latent Diffusion Models

#### 1 Introduction

Human motion generation involves creating high-quality 3D human motion based on given conditions, such as a text prompt, which generates Euler angles for each joint in a human skeleton. To emulate human motion effectively, virtual characters must respond to the conditional context, exhibit natural movement, and perform motion accurately. Recent motion generation models are categorized into four main approaches: autoencoder-based [1,10,17,37,49,59], utilizing transformers for latent space compression and motion synthesis; GAN-based [4,20,31], using discriminators to enhance the realism of generated motions; autoregressive models [25], treating motion sequences as languages with specialized codebooks; and diffusion-based [6,50,55], employing denoising steps for motion generation. Challenges vary across methods, with autoencoder models struggling to generate accurate motions from detailed descriptions due to textual information compression, GAN-based models facing training difficulties, especially in conditional tasks, and diffusion-based models relying on complex transformer-based architectures results in inefficient of motion prediction.

Although diffusion-based models excel at generating motion with robust performance and often exhibit superior diversity, they encounter two limitations.

- 1) Convolutional or transformer-based diffusion methods exhibit limitations in generating long-range motion sequences. Previous transformer-based methodologies [6,50,55] have focused on modeling long-range dependencies and acquiring comprehensive global context information. Despite these advances, they are frequently associated with a substantial increase in computational requirements. Furthermore, transformer architectures are not intrinsically designed for temporal sequential modeling, which poses an inherent limitation. 2) The efficiency of inference in transformer-based diffusion methods is constrained. Although prior research has attempted to leverage the Variational Autoencoder for denoising operations in the latent space [6], the inference speed remains adversely affected by the attention mechanism’s quadratic scaling, leading to inefficient motion generation. Consequently, exploring a new architectural paradigm that accommodates long-range dependencies and maintains a linear computational complexity is crucial for sustaining motion generation tasks.

Recent advances have sparked renewed interest in state space models (SSMs) [14,15], a field that originated from the foundational classic state space model

[26]. Modern versions of SSMs stand out due to their ability to effectively capture long-range dependencies, a capability greatly improved by the introduction of parallel training techniques. This evolution has led to various methodologies based on SSM, notably the linear state space layers (LSSL) [15], the structured state-space sequence model (S4) [14], the diagonal state space (DSS) [19], and S4D [13]. These methods have been carefully designed to handle sequential data across various tasks and modalities, paying special attention to modeling longrange dependencies. Their efficacy in managing long sequences is attributed to the implementation of convolutional computations [15] and near-linear computational strategies, such as mamba [12], marking a significant stride in sequentially oriented tasks, including large language model decoding and motion sequence generation.

Adapting selective state space modules for motion generation tasks presents notable challenges, primarily due to the lack of specialized design in SSMs for capturing the sensitive motion details required for temporal representation and the complexities involved in aggregating latent space. In response to these challenges, we have meticulously developed a motion generation architecture, specifically tailored to address the intricacies of long-term sequence generation, while optimizing for computational efficiency with near-linear-time complexity. This innovation is embodied in the Motion Mamba model, a simple yet potent approach to motion generation. The Motion Mamba framework pioneers a diffusion-based generative system, incorporating two key components oriented toward SSM as shown in Figure. 2: (1) a Hierarchical Temporal Mamba (HTM) block: This component is ingeniously crafted to arrange motion frames in sequential order, using hierarchically adjusted scanning. It is adept at identifying temporal dependencies at various depths, thereby facilitating a thorough comprehension of the dynamics inherent in motion sequences. (2) a Bidirectional Spatial Mamba (BSM) block: This block is designed to unravel the structured latent skeleton by evaluating data from both forward and reverse directions. Its primary goal is to safeguard the continuity of information flow, significantly bolstering the model’s capacity for precise motion generation through the retention of dense informational exchange.

The Motion Mamba introduces a new approach to motion generation that strikes an exceptional trade-off between accuracy and efficiency, shown in Fig. 1. Our experimental results underscore the significant improvements brought about by Motion Mamba, showcasing a remarkable improvement in the Fréchet Inception Distance (FID), with a reduction of up to 50% from the prior state-of-the-art metric of 0.473 to an impressive 0.281 on the HumanML3D dataset [17]. Furthermore, we emphasize our framework’s unparalleled inference speed, which is four times faster than previous methods, achieving an average inference time of only 0.058 seconds per sequence compared to the 0.217 seconds required by the MLD [6] method per sequence. These outcomes unequivocally establish Motion Mamba’s state-of-the-art performance, concurrently ensuring fast inference speeds for conditional human motion generation tasks.

Our contributions to the field of motion generation can be summarized as:

- 1. We introduce a simple yet effective framework, named Motion Mamba, which is a pioneering method integrates a selective scanning mechanism into motion generation tasks.
- 2. Motion Mamba is comprised of two modules: Hierarchical Temporal Mamba (HTM) and Bidirectional Spatial Mamba (BSM), which are designed for temporal and spatial modeling, respectively. HTM blocks are tasked with processing temporal motion data, aiming to enhance motion consistency across frames. BSM blocks are engineered to bidirectionally capture the channel-wise flow of hidden information within the latent pose representations.
- 3. Motion Mamba framework demonstrated exceptional performance on text-tomotion generation task, through experimental validation on the HumanML3D [17] and KIT-ML [39] datasets. Our methodology achieved state-of-the-art generation quality and significantly improved long-squence modeling, meanwhile optimizing inference speed.

#### 2 Related Works

Human Motion Generation. Generating human motion is a significant application of computer vision, essential for various applications like 3D modelling and robot manipulation. Recently, the predominant method of achieving human motion generation, known as the Text-to-Motion task, involves learning a common latent space for both language and motion.

DVGAN [31] creates the GAN [11] discriminator by densely validating at each time-scale and perturbing the discriminator input for translation invariance, enabling motion generation and completion. ERD-QV [20] enhances latent representations through two additive modifiers: a time-to-arrival embedding applied universally and an additive scheduled target noise vector used during extended transitions. It further improves transition quality by incorporating a GAN framework with two discriminators operating at different timescales. HP-GAN [4], trained with a modified version of the improved WGAN-GP [16], utilizes a custom loss function designed for human motion prediction. It learns a probability density function of future human poses conditioned on previous poses.

Autoencoders [28,44] are notable generative models known for their ability to represent data robustly by compressing high-dimensional data into a latent space, which is widely adopted in human motion generation [57, 58]. JL2P [1] uses RNN-based autoencoders [23] to learn a combined representation of language and pose. It restricts the direct mapping from text to motion to a one-toone relationship. MotionCLIP [49] uses Transformer-based Autoencoders [52] to reconstruct motion while ensuring alignment with the corresponding text label in the CLIP [41] space. This alignment effectively integrates the semantic knowledge from CLIP into the human motion manifold. TEMOS [37] and T2M [17] combine a Transformer-based VAE [27] with a text encoder to generate distribution parameters that work within the VAE latent space. AttT2M [59] and TM2D [10] incorporate a body-part spatio-temporal encoder into VQ-VAE [51] for enhanced learning of a discrete latent space with increased expressiveness.

Diffusion models [7, 22, 42, 48] have recently surpassed GANs and VAEs in generating 2D images. Developing a motion generation model based on diffusion models is obviously an attractive direction. MotionDiffuse [55] introduces the inaugural framework for text-driven motion generation based on diffusion models. It showcases several desirable properties, including probabilistic mapping, realistic synthesis, and multi-level manipulation. MDM [50] utilizes a classifierfree Transformer-based diffusion model for the human motion domain to predict sample rather than noise in each diffusion step. MLD [6] performs a diffusion process in latent motion space, rather than using a diffusion model to establish connections between raw motion sequences and conditional inputs.

State Space Models. Recently, state space sequence models (SSMs) [14,15], drawing inspiration from classical state-space models [26], have emerged as a promising architecture for sequence modeling [24]. Mamba [12] introduces a selective SSM architecture, integrating time-varying parameters into the SSM framework, and proposes a hardware-aware algorithm to facilitate highly efficient training and inference processes. Some research works leverage SSM in computer vision to process 2D data. The 2D SSM [3] introduces an SSM block at the beginning of each transformer block [8,52]. This approach aims to achieve efficient parameterization, accelerated computation, and a suitable normalization scheme. SGConvNeXt [30] presents a structured global convolution method inspired by ConvNeXt [33], incorporating multi-scale sub-kernels to achieve both parameterization efficiency and effective long sequence modeling. ConvSSM [47] integrates the tensor modeling principles of ConvLSTM [46] with SSMs, elucidating the utilization of parallel scans in convolutional recurrences. This approach enables subquadratic parallelization and rapid autoregressive generation. Vim [60] introduces a bidirectional SSM block [53] for efficient and versatile visual representation learning, achieving performance comparable to established ViT [8] methods. VMamba [32] introduces a Cross-Scan Module (CSM) designed to traverse the spatial domain and transform any non-causal visual image into ordered patch sequences. This approach achieves linear complexity while preserving global receptive fields. There have also been attempts to utilize SSMs to handle higher-dimensional data. Mamba-ND [29] explores various combinations of SSM and different scan directions within the SSM block to adapt Mamba [12] to higher-dimensional tasks. Recent efforts have sought to replace the traditional transformer-based U-Net within the diffusion denoiser with the SSM block, with the aim of enhancing image generation efficiency. DiffuSSM [54] adeptly manages higher resolutions without relying on global compression, thus maintaining detailed image representation throughout the diffusion process.

#### 3 The Proposed Method

In this section, we delineate the architecture and operational principles of the Motion Mamba framework, designed for generating human motion over long ranges efficiently from textual descriptions. Initially, we discuss the foundational concepts underpinning our approach, including the Mamba Model [12] and the

latent diffusion model [6]. Following this, we detail our uniquely crafted architecture that leverages the Mamba model to enhance motion generation efficiency. This architecture comprises two principal components: the Hierarchical Temporal Mamba (HTM) block, which addresses temporal aspects, and the Bidirectional Spatial Mamba (BSM) block, focusing on spatial dynamics.

##### 3.1 Preliminaries

Selective Structured State Space Sequence Model. SSMs particularly through the contributions of structured state space sequence models (S4) and Mamba, have demonstrated exceptional proficiency in handling long sequences. These models operationalize the mapping of a 1-D function or sequence, x(t) ∈ R  → y(t) ∈ R, through a hidden state h(t) ∈ RN, employing A ∈ RN×N as the evolution parameters, B ∈ RN×1 and C ∈ R1×N as the projection parameters, respectively.

The discretized system can then be expressed as follows, incorporating a step size ∆:

ht = Aht−1 + Bxt, yt = Cht.

(1)

This adaptation facilitates the computation of output through global convolution, leveraging a structured convolutional kernel K, which encompasses the entire length M of the input sequence x:

K = (CB,CAB,...,CAM−1B), y = x ∗ K.

(2)

Selective models like Mamba introduce time-varying parameters, deviating from the linear time invariance (LTI) assumption and complicating parallel computation. However, hardware-aware optimizations, such as associative scans, have been developed to address these computational challenges, highlighting the ongoing evolution and application of SSMs in modeling complex temporal dynamics. Latent Motion Diffusion Model. Diffusion probabilistic models offer a significant advancement in motion generation by gradually reducing noise from a Gaussian distribution to a target data distribution p(x) through a T-length learned Markov process [7,22,42,45,48,50,55] , giving {xt}Tt=1. In the motion generation, we define our trainable diffusion models with a denoiser ϵθ (xt,t) which anneal the random noise to motion sequence {xˆ1:t N}Tt=1 iteratively. To address the inefficiencies of applying diffusion models directly to raw motion sequences, we employ a low-dimensional motion latent space for the diffusion process. Given an input condition c, such as a descriptive sentence w1:N = {wi}Ni=1, an action label a from a predefined set A, or an empty condition c = ∅, and the motion representation that combines 3D joint rotations, positions, velocities, and foot contact as proposed in [17]. The frozen CLIP [41] text encoder τθw has been employed to obtain projected text embedding τθw(w1:N) ∈ R1×d, thereby conditional denoiser comprised in term of ϵθ(zt,t,τθ(c)). The latent diffusion model ϵθ (xt,t) aimed

###### ×N

###### T

|Linear|
|---|

### ...

Conv1D SSM(A1, x)

Conv1D SSM(A2, x)

Conv1D SSM(Ak, x)

... ...

k = 2n-1

k = 2n-1 k=5

k=5

Linear Rearrange

k=3

k=3 k=1

k=1

Linear

M

###### S+ S-

Linear

Linear

Conv1D SSM(+)

Conv1D SSM(-)

Temporal Input

Denoiser Block Mixer Block

HTM Block

T S+ S-

BSM Block

Forward Spatial

Rearrange

Concatenate Feature

Skip Connection

Backward Spatial

Fig. 2: This figure illustrates the architecture of the proposed Motion Mamba model. Each of encoder and decoder blocks consists of a Hierarchical Temporal Mamba block (HTM) and a Bidirectional Spatial Mamba (BSM) block, which possess hierarchical scan and bidirectional scan within SSM layers respectively. This symmetric distribution of scans ensure a balanced and coherence framework across the encoder-decoder architecture.

to generate the human motion sequence in terms of xˆ1:L = {xˆi}Li=1, where L denotes the sequence length or number of frames [35,36,38,56]. Afterthat we reused the motion Variational AutoEncoder (VAE) V = {E,D} proposed in MLD [6] to manipulate the motion sequence in latent space z = E(x1:L), and decompress the intermediate representation to motion sequence by xˆ1:L = D(z) = DE(x1:L) [27, 43, 52]. Finally, our latent diffusion model is trained with an objective focusing on minimization of MSE between true and predicted noise in the latent space, facilitating efficient and high-quality motion generation [2,22].

##### 3.2 Motion Mamba

The architecture of the proposed Motion Mamba framework is illustrated in Figure. 2. At its core, Motion Mamba utilizes a denoising U-Net architecture, which is distinguished for its effectiveness in modeling the continuous, temporal sequences of motion frames. This effectiveness is attributed to the inherent longsequence modeling capacity of the Mamba model. The denoiser, denoted by ϵθ, comprises N blocks including encoder E1..N and decoder D1..N . Additionally, the architecture is enhanced with a transformer-based attention mixer block M, designed to augment the model’s ability to capture complex temporal dynamics.

ϵθ(x) ≡ {E1...N,M,D1..N}. (3)

The encoder blocks are represented as E1..N, arranged sequentially, and the decoder blocks as D1..N, configured in reverse order to facilitate effective bottomup and top-down information flow. Given that selective operations have signif-

- Algorithm 1 Hierarchical Temporal Mamba (HTM) Block.

Require: compressed latent representations z : (T, B, C) Ensure: transformed representations zHTM : (T, B, E)

- 1: /* linear projection layer */
- 2: x, z : (T, B, E) ← Linear(z)
- 3: /* set of scans and memory matrices */
- 4: K = {S2Nn−1, S2Nn−1−1, . . . , S1}
- 5: Memory matrices: {A1, . . . , Ak}
- 6: for each scan Si in K do
- 7: x′o : (T, B, E) ← Conv1D(x)
- 8: Bo, Co, ∆o ← Linear(x′o)
- 9: Transform Ao and Bo using ∆o

- 10: Oi ← SSMAi,x(x′o)
- 11: end for
- 12: /* aggregation of outputs */
- 13: zHTM : (T, B, E) ← Linear(Aggregate({O1, . . . , Ok}))
- 14: Return: zHTM

icantly lower computational complexity compared to attention-based methods, we have increased the number of scans to achieve higher quality generations. Concurrently, it is imperative to maintain a balance between the model’s parameters and its efficiency. Thereby, a novel aspect of our model is the introduction of a hierarchical scan strategy, characterized by a sequence of scan numbers as,

K = {S2N−1,S2(N−1)−1,...,S1}. (4)

This sequence specifies the number of scans allocated to each layer, in descending order of complexity. For instance, the uppermost encoder layer, E1, and the lowermost decoder layer, DN, are allocated S2N−1 scans, indicating the highest scanning complexity. Conversely, the lowest encoder layer, EN, and the uppermost decoder layer, D1, are assigned S1 scans, reflecting the lowest level of scanning complexity.

 

S2N−1 for i = 1 S2(N−i)−1 for i = 2,...,N − 1 S1 for i = N

(5)

Ei(S) =



 

S2N−1 for j = N S2(N−j)−1 for j = N − 1,...,2 S1 for j = 1

(6)

Dj(S) =



This hierarchical scanning approach ensures that processing capabilities are evenly distributed throughout the encoder-decoder architecture., facilitating a detailed and nuanced analysis of temporal sequences. Within this structured framework, each denoiser is equipped with a specialized Hierarchical Temporal Mamba (HTM) block, which serves to augment the model’s ability to process

- Algorithm 2 Bidirectional Spatial Mamba (BSM) Block.

Require: compressed latent representations z : (T, B, C) Ensure: transformed representations zBSM : (C, B, E)

- 1: /* dimension rearrangement */
- 2: z′ : (C, B, T) ← Rearrange(z)
- 3: /* linear projection after normalization */
- 4: z′ : (C, B, T) ← Norm(z)
- 5: x, z : (C, B, E) ← Linear(z′)
- 6: for o in {forward, backward} do
- 7: x′o : (C, B, E) ← Conv1D(x)
- 8: Bo, Co, ∆o ← Linear(x′o)
- 9: Transform Ao and Bo using ∆o

- 10: yo ← SSM(Ao, Bo, Co)

- 11: end for
- 12: /* gating and summing outputs */
- 13: zBSM : (C, B, T) ← GateAndSum(yforward, ybackward, z)
- 14: zBSM′ : (T, B, C) ← Rearrange(z)
- 15: Return: zBSM′

temporal information effectively. Additionally, the proposed Motion Mamba incorporates an attention-based mixer block denoted as M, strategically integrated to enhance conditional fusion.

Hierarchical Temporal Mamba (HTM) block processes compressed latent representations, denoted as z, with the dimensions (T,B,C), of which procedure shown in Algorithm 1. Here, T signifies the temporal dimension, as specified in the Variational AutoEncoder (VAE) framework. Initially, the input z is subjected to a linear projection layer, producing transformed representations x and z with dimension E. Our analysis revealed an increased density of motion within the lower-level feature spaces. Consequently, we developed a hierarchical scanning methodology that is executed at various depths of the network. This approach not only accommodates the diverse motion densities, but also significantly reduces computational overhead. This step utilizes a hierarchically structured set of scans, K = {S2N

n−1−1,...,S1}, in conjunction with a corresponding series of memory matrices {A1,...,Ak}. Each sub-SSM scan first applies a 1-D convolution to x, resulting in x′o. x′o is then linearly projected to derive Bo, Co, and ∆o. These projections Bo, Co use ∆o to effect transformations in Ao and Bo, respectively. After executing a sequence of SSM scans {SSMA

n−1,S2N

k,x}, a set of outputs {O1,...,Ok} is compiled. This collection is subsequently aggregated via a linear projection to obtain the final output of the HTM block.

1,x,SSMA

2,x,...,SSMA

Bidirectional Spatial Mamba (BSM) block focuses on enhancing latent representation learning through a novel approach of dimension rearrangement and bidirectional scanning, of which the process is shown in Algorithm 2. Initially, it alters the original input dimensions from (T,B,C) to (C,B,T), effectively swap-

ping the temporal and channel dimensions. After this rearrangement, the input, now denoted z′, undergoes a linear projection after normalization, resulting in dimensions x and z of size E. The process involves bidirectional scanning of the latent channel dimension, where x is subjected to a 1-D convolution, yielding x′o for both forward and backward directions. Each x′o is then linearly projected to obtain Bo, Co, and ∆o, which are utilized to transform Ao and Bo, respectively. The final output token sequence, zl, is computed by gating and summing the forward yforward and backward ybackward output with z. This component is engineered to decode the structured latent skeleton by analyzing data from both forward and reverse viewpoints. Its main objective is to ensure the seamless continuity of information flow, thereby substantially enhancing the model’s ability to generate accurate motion. This is achieved through the maintenance of a dense informational exchange, which is critical for the model’s performance.

#### 4 Experiments

##### 4.1 Datasets

We evaluate our proposed Motion Mamba on two prominent Text-to-Motion synthesis benchmarks as follows: HumanML3D. The HumanML3D [17] dataset aggregates 14,616 motions sourced

from the AMASS [34] and HumanAct12 [18] datasets, with each motion accompanied by three textual descriptions, culminating in 44,970 scripts. This dataset spans a wide range of actions such as exercising, dancing, and acrobatics, presenting a rich motion-language corpus.

KIT-ML. The KIT-ML dataset [39] is comprised of 3,911 motions paired with 6,278 textual descriptions, serving as a compact yet effective benchmark for evaluation. For both datasets, the pose representation adopted is derived from T2M [17], ensuring consistency in motion representation across evaluations.

##### 4.2 Evaluation Metrics

We adapt the standard evaluation metrics on following aspects throughout our experiments, including: Generation Quality. We implement a Fréchet inception distance (FID) [21] to quantify the realism and diversity of motion generated by models. Moreover, we use multi-modal distance (MM Dist) to measure the distance between motions and texts and assess motion-text alignment. Diversity. We use the diversity metric to measure motion diversity, which calculates variance in features extracted from the motions. Additionally, we employ multimodality (MModality) to assess diversity within generated motions sharing the same text description.

##### 4.3 Comparative Studies

We evaluate our method against the state-of-the-art methods on the HumanML3D [17] and KIT-ML [39] datasets. We train our Motion Mamba with HTM arrangement strategy MM ({S2N

n−1,...,S1}), BSM bidirectional block strategy on the

- Table 1: Comparison of text-conditional motion synthesis on HumanML3D [17]. These metrics are evaluated by the motion encoder from [17]. Empty MModality indicates the non-diverse generation methods. We employ real motion as a reference and sort all methods by descending FIDs. The right arrow → means that the closer to the real motion, the better. Bold and underline indicate the best and second best result.

Method

R Precision ↑ FID↓ MM Dist↓ Diversity→ MModality↑

Top 1 Top 2 Top 3 Real 0.511±.003 0.703±.003 0.797±.002 0.002±.000 2.974±.008 9.503±.065 Seq2Seq [40] 0.180±.002 0.300±.002 0.396±.002 11.75±.035 5.529±.007 6.223±.061 LJ2P [1] 0.246±.001 0.387±.002 0.486±.002 11.02±.046 5.296±.008 7.676±.058 T2G [5] 0.165±.001 0.267±.002 0.345±.002 7.664±.030 6.030±.008 6.409±.071 Hier [9] 0.301±.002 0.425±.002 0.552±.004 6.532±.024 5.012±.018 8.332±.042 TEMOS [38] 0.424±.002 0.612±.002 0.722±.002 3.734±.028 3.703±.008 8.973±.071 0.368±.018 T2M [17] 0.457±.002 0.639±.003 0.740±.003 1.067±.002 3.340±.008 9.188±.002 2.090±.083 MDM [50] 0.320±.005 0.498±.004 0.611±.007 0.544±.044 5.566±.027 9.559±.086 2.799±.072 MotionDiffuse [55] 0.491±.001 0.681±.001 0.782±.001 0.630±.001 3.113±.001 9.410±.049 1.553±.042 MLD [6] 0.481±.003 0.673±.003 0.772±.002 0.473±.013 3.196±.010 9.724±.082 2.413±.079

Motion Mamba (Ours) 0.502±.003 0.693±.002 0.792±.002 0.281±.009 3.060±.058 9.871±.084 2.294±.058

- Table 2: We involve KIT-ML [39] dataset and evaluate the SOTA methods on the text-to-motion task.

R Precision ↑ FID↓ MM Dist↓ Diversity→ MModality↑

Method

Top 1 Top 2 Top 3 Real 0.424±.005 0.649±.006 0.779±.006 0.031±.004 2.788±.012 11.08±.097 Seq2Seq [40] 0.103±.003 0.178±.005 0.241±.006 24.86±.348 7.960±.031 6.744±.106 T2G [5] 0.156±.004 0.255±.004 0.338±.005 12.12±.183 6.964±.029 9.334±.079 LJ2P [1] 0.221±.005 0.373±.004 0.483±.005 6.545±.072 5.147±.030 9.073±.100 Hier [9] 0.255±.006 0.432±.007 0.531±.007 5.203±.107 4.986±.027 9.563±.072 2.090±.083 TEMOS [38] 0.353±.006 0.561±.007 0.687±.005 3.717±.051 3.417±.019 10.84±.100 0.532±.034 T2M [17] 0.370±.005 0.569±.007 0.693±.007 2.770±.109 3.401±.008 10.91±.119 1.482±.065 MDM [50] 0.164±.004 0.291±.004 0.396±.004 0.497±.021 9.191±.022 10.85±.109 1.907±.214 MotionDiffuse [55] 0.417±.004 0.621±.004 0.739±.004 1.954±.062 2.958±.005 11.10±.143 0.730±.013 MLD [6] 0.390±.008 0.609±.008 0.734±.007 0.404±.027 3.204±.027 10.80±.117 2.192±.071

Motion Mamba (Ours) 0.419±.006 0.645±.005 0.765±.006 0.307±.041 3.021±.025 11.02±.098 1.678±.064

latent dimension = 2 with 11 layers. We evaluate our model and previous works with suggested metrics in HumanML3D [17] and calculate 95% confidence interval by repeat evaluation 20 times. The results for the HumanML3D dataset are presented in Table 1. Our model outperforms other methods significantly across various evaluation metrics, including FID, R precision, multi-modal distance, and diversity. For instance, our Motion Mamba outperforms previous best diffusion based motion generation MLD by 40.5% in terms of FID, and up to 10% improvement on R Precision, we aslo obatined best MModality by 3.060. The results for the KIT-ML dataset are presented in Table 2. We have also outperformed other well-established methods in FID and multi-modal distance.

##### 4.4 Ablation Studies

We concluded the ablation studies including long sequence evaluation, hierarchical design with HTM, bidirectional design in the BSM, number of latent dimensions, and number of layers of our proposed motion mamba in Table 4.

- Table 3: In order to evaluate the models’ capability in long sequence motion generation, we compared our method with an existing approach on the recently introduced HumanML3D-LS dataset. This dataset comprises motion sequences longer than 190 frames from the original evaluation set. Our model demonstrates superior performance compared to other methods.

R Precision ↑ FID↓ MM Dist↓ Diversity→ MModality↑

Method

Top 1 Top 2 Top 3 Real 0.437±.003 0.622±.004 0.721±.004 0.004±.000 3.343±.015 8.423±.090 MDM [50] 0.368±.005 0.553±.006 0.672±.005 0.802±.044 3.860±.025 8.817±.068 MotionDiffuse [55] 0.367±.004 0.521±.004 0.623±.004 2.460±.062 3.789±.005 8.707±.143 1.602±.013 MLD [6] 0.403±.005 0.584±.005 0.690±.005 0.952±.020 3.580±.016 9.050±.085 2.711±.104

Motion Mamba (Ours) 0.417±.003 0.606±.003 0.713±.004 0.668±.019 3.435±.015 9.021±.070 2.373±.084

Long Sequence Motion Generation. The HumanML3D [17] dataset exhibits a longtailed and right-skewed distribution with a significant proportion of long-sequence human motions, as shown in Figure 3. We suggest previous studies overlooked the challenges in the long-sequence generation problem. Thus, we introduce a new dataset variant, HumanML3D-LS, comprising motion sequences longer than 190 frames extracted from the original test set. This addition allows us to showcase our capability in generating long-sequence motions. Subsequently, we evaluate the performance of our method on HumanML3D-LS and compare it with other diffusion-based motion generation approaches. The comparative results are presented in Table 3. Motion Mamba by leverage the benefits on long-range dependency modeling make it well suitable for long sequence motion generation. Hierarchical Design with HTM. In our ablation studies, we observed a slight improvement upon reversing the scan order from a lower to a higher level, specifically transitioning from MM {S1,...,SN} to MM {SN,...,S1}. This enhancement suggests a correlation with the increase in temporal motion density within the lower-level feature spaces. Furthermore, to achieve the optimal result, we introduce the hierarchical design to arrange the scanning frequency, resulting in the sequence MM {S2N

| | | |
|---|---|---|
| | | |
| | | |
| | | |

Fig. 3: The figure shows a long tail distribution of the HumanML3D [17], which has a significant proportion of long-sequence human motions.

n−1,...,S1}. This expansion in the number of scans led to a performance increase. We attribute this enhancement to the observation that individual selective scan operations significantly reduce the parameter count, especially when compared to the parameter-intensive constructs of self-attention and feedforward network blocks prevalent in transformer architectures.

Bidirectional Design in BSM. We developed three distinct variations of latent scanning mechanisms, differentiated by their scanning directions. In the context of motion generation tasks, we posit that the flow of hidden information within the structured latent skeleton holds significance, an aspect previously underexplored. Our ablation study reveals that a single scan across the la-

- Table 4: Evaluation of text-based motion synthesis on HumanML3D [17]: we use metrics in Table 1 and provides real reference, we evaluate the various HTM and BSM design choices, the dimension of the latent input, the different number of layer of Motion Mamba model.

R Precision

Models

Top 3↑ FID↓ MM Dist.↓ Diversity→ MModality↑ Real 0.797±.002 0.002±.000 2.974±.008 9.503±.065 -

MM ({S1, . . . , SN}) 0.673±.003 1.278±.012 3.802±.041 8.678±.096 3.127±.024 MM ({SN, . . . , S1}) 0.738±.002 0.962±.011 3.433±.003 9.180±.071 2.723±.033 MM ({S1, . . . , S2Nn−1}) 0.698±.002 0.856±.008 3.624±.037 9.229±.067 2.826±.017 MM ({S2Nn−1, . . . , S1}) 0.792±.002 0.281±.009 3.060±.058 9.871±.084 2.294±.058 MM (SingleScan) 0.736±.003 1.063±.010 3.443±.026 9.180±.067 2.676±.041 MM (BiScan, layer) 0.735±.004 0.789±.007 3.408±.034 9.374±.059 2.591±.046 MM (BiScan, block) 0.792±.002 0.281±.009 3.060±.058 9.871±.084 2.294±.058

- MM (Dim, 1) 0.706±.003 0.652±.011 3.541±.072 9.141±.082 2.612±.055
- MM (Dim, 2) 0.792±.002 0.281±.009 3.060±.058 9.871±.084 2.294±.058 MM (Dim, 5) 0.741±.008 0.728±.009 3.307±.027 9.427±.099 2.314±.062 MM (Dim, 7) 0.738±.004 0.599±.007 3.359±.068 9.166±.075 2.488±.037 MM (Dim, 10) 0.715±.003 0.628±.008 3.548±.043 9.200±.075 2.884±.096

MM (9 layers) 0.755±.002 1.080±.012 3.309±.057 9.721±.081 2.974±.039 MM (11 layers) 0.792±.002 0.281±.009 3.060±.058 9.871±.084 2.294±.058 MM (27 layers) 0.750±.003 0.975±.008 3.336±.096 9.249±.071 2.821±.063 MM (37 layers) 0.754±.005 0.809±.010 3.338±.061 9.355±.062 2.741±.077

tent dimension yields minimal improvement. Subsequently, we investigated both layer-based and block-based bidirectional scans. Our findings indicate that the block-based bidirectional scan achieves optimal performance. This suggests that spatial information flows are encoded within the latent spaces and that bidirectional scanning facilitates the exchange of this information, thereby enhancing the efficacy of motion generation tasks.

Architecture Design for Motion Mamba. The proposed Motion Mamba which is grounded in a standardized motion latent diffusion system. We delved into the interplay between dimensional aspects and the module’s capacity (measured by the number of layers) to ascertain their impact on system performance. Experimental results demonstrate that the Motion Mamba achieves superior performance at a latent dimension of 2, diverging from prior works where the optimal dimension was identified as 1. We attribute this discrepancy to our HTM, which necessitates multiple scans correlating with the sequence length, thus implicating dimensionality as a pivotal factor. A marginal increase in dimensionality enabled us to attain peak performance, simultaneously enhancing efficiency compared to models with a dimensionality of 10. Furthermore, we conducted experiments to determine the optimal layer count for Motion Mamba, inspired by the design of its selective scanning mechanism. Notably, a single Mamba layer achieves a parameter reduction of approximately 75% compared to a conventional transformer encoder block. By increasing the number of layers, we aim to uncover the relationship between model capacity and its performance. Our findings reveal that, through the integration of our specially designed HTM and BSM (Bidirectional Scanning Module) blocks, the Motion Mamba reaches its optimal performance with 11 layers. This represents a slight increase over the MLD [6] baseline. How-

ever, due to the reduced parameter count in each layer, Motion Mamba exhibits significantly greater efficiency than previous methodologies.

##### 4.5 Inference Time

Inference time remains a significant challenge for diffusion-based methods. To address this, we enhance the inference speed by incorporating the efficient Mamba block within a lightweight architecture. Compared to the previous strong baseline, such as the MLD model cited in [6], which reports an average inference time of 0.217 seconds, our Motion Mamba model achieves a notable reduction in computational overhead, as shown in Figure

- 4. Specifically, it requires four times less computational effort, thereby facilitating faster and real-time inference speeds.
- 5 Discussion and Conclusion

4.0

TEMOS (AIT: 0.017s, FID: 3.734)

T2M (AIT: 0.038s, FID: 1.067)

TEMOS

3.5

MotionDiffuse (AIT: 14.74s, FID: 0.63)

MDM (AIT: 24.74s, FID: 0.544)

3.0

MLD (AIT: 0.217s, FID: 0.473)

MotionMamba(Ours) (AIT: 0.058s, FID: 0.281)

2.5

2.0

FID

1.5

T2M

1.0

MotionDiffuse MLD

0.5

MotionMamba(Ours)

MDM

0.0

10 1 100 101 Average Inference Time per Sentence (AIT) in seconds

Fig. 4: The figure shows the average inference time per sentence (AIT) vs FID, our proposed motion mamba obtained 0.058s AIT and 0.281 FID overall outperform previous methods. We evaluate all methods on a single V100 GPU.

In this study, we introduced Motion Mamba, a novel framework designed for efficient and extended sequence motion generation. Our approach represents the inaugural integration of the Mamba model within the domain of motion generation, featuring significant advancements including the implementation of Hierarchical Temporal Mamba (HTM) blocks. These blocks are specifically engineered to enhance temporal alignment through hierarchically organized selective scanning. Furthermore, Bidirectional Spatial Mamba (BSM) blocks have been developed to amplify the exchange of information flow within latent spaces, thereby augmenting the model’s ability to bidirectionally capture skeleton-level density features with greater precision. Compared to previous diffusion-based motion generation methodologies that predominantly utilize transformer blocks, our Motion Mamba framework achieves SOTA performance, evidencing an improvement of up to 50% in FID scores and a quadrupled improvement in inference speed. Through comprehensive experimentation across a variety of human motion generation tasks, the effectiveness and efficiency of our proposed Motion Mamba model have been robustly demonstrated, marking a significant leap forward in the field of human motion generation.

#### Acknowledgements

This work is partially supported by the Fundamental Research Funds for the Central Universities, Peking University.

## Motion Mamba Supplementary

#### 1 Implementation Details

Motion Mamba operates within the latent spaces, leveraging the capabilities of the Motion Variational AutoEncoder (VAE) V = {E,D}, as proposed in the seminal work by Chen et al. [6]. For the configuration of the Motion Mamba denoiser ϵθ, we have opted for an architecture comprising 11 layers (N = 11), with the latent dimensionality set to z ∈ R2,d. The Hierarchical Temporal Mamab (HTM) modules are arranged in a scan pattern of {S2N

n−1,...,S1}, while the Bidirectional Spatial (BSH) modules incorporate a block-level bidirectional scan policy. Additionally, we utilize a pretrained CLIP-VIT-L-14 model in a frozen state to derive text embeddings τθw(w1:N) ∈ R1×d.

All models under the Motion Mamba framework are meticulously trained using the AdamW Optimizer, with the learning rate steadfastly maintained at 10−4. We have standardized our global batch size at 512, which is judiciously distributed across 4 GPUs to facilitate data-parallel training. The training regime is extended over 2,000 epochs to ensure convergence to an optimal set of parameters. For the diffusion sampling process, we maintain the number of steps at 1,000 and 50 during the training and inference phases, respectively. The entire training procedure is executed on a single-node GPU server, outfitted with 4 NVIDIA A100 GPUs, spanning approximately 4 hours. Inference speed evaluations of our Motion Mamba models are conducted on a single NVIDIA V100 GPU for fair comparison, while module development and additional inference tasks are performed on a single NVIDIA GeForce RTX 3090/4090 GPU.

MM vs MDM

MM vs MDM

79%

70%

MM vs T2M

MM vs T2M

74%

67%

MM vs MLD

MM vs MLD

62%

59%

MM vs GT

MM vs GT

47%

46%

0 10 20 30 40 50 60 70 80 Which of the two motions corresponds better to the text prompt?

0 10 20 30 40 50 60 70 80 Which of the two motions is more realistic?

(a) Text-Motion Correspondence User Study

(b) Quality User Study

- Fig. 1: User Study in two aspects including text-motion correspondence and quality, we compare Motion Mamba(MM) with previous methods including MDM [50] , T2M [17] , MLD [6] and ground truth.

#### 2 User Study

In this work, we undertake a comprehensive evaluation of Motion Mamba’s performance, encompassing both qualitative analyses across various datasets and a user study to assess its real-world applicability. A diverse collection of 20 motion sequence sets, prompted randomly and extracted from the HumanML3D [17] test set, were generated utilizing three distinct methodologies—MDM [50], T2M [17], MLD [6]—alongside Motion Mamba and a baseline of ground truth motions. Subsequently, 50 participants were randomly selected to evaluate the motion sequences generated by these methods.

The user study was administered through a Google Forms interface, as depicted in Fig. 4, ensuring that motion sequences were presented anonymously without revealing their generative model origins. Our analysis focused on two critical dimensions: the fidelity of text-to-motion correspondence and the overall quality of the generated motions.

Empirical results, illustrated in Fig. 1a and Fig. 1b, unequivocally demonstrate Motion Mamba’s superior performance relative to the benchmark methods in terms of both text-motion alignment and motion quality. Specifically, Motion Mamba achieved significant margins over MDM [50], T2M [17], and MLD [6] by 79%, 74%, and 62% in text-motion correspondence, respectively, as highlighted in Fig. 1a. When juxtaposed with ground truth data—meticulously captured with state-of-the-art, noise-free devices—Motion Mamba’s generated sequences exhibited a remarkably close adherence to the intended text descriptions, underscoring its proficiency in aligning textual prompts with motion sequences.

Further reinforcing these findings, Motion Mamba’s generated motions were also found to surpass the aforementioned methods by substantial margins of 70%, 67%, and 59%, respectively, in terms of quality, as reported in Fig. 1b . This underscores Motion Mamba’s ability to not only closely match the textmotion correspondence of high-fidelity ground truth data but also to produce high-quality motion sequences that resonate well with real user experiences.

#### 3 Visualization

Our study delves into the visualization of motion generation by capturing intricate motion sequences, utilizing prompts and their variations derived from HumanML3D [17]. We meticulously compare our proposed Motion Mamba methodology with established state-of-the-art techniques, namely MotionDiffuse [55], MDM [50], and MLD [6]. Presenting three distinct motion sequences, we meticulously analyze and visualize each, offering a comprehensive assessment of our approach’s efficacy.

The person is walking in a semi-circle and in clockwise direction.

The person walks up the corner stairs.

The person first walks forward then walks backward.

Method

[Figure 5]

[Figure 6]

[Figure 7]

###### Ours

[Figure 8]

[Figure 9]

[Figure 10]

MLD

[Figure 11]

[Figure 12]

[Figure 13]

MDM

[Figure 14]

[Figure 15]

[Figure 16]

MD

###### Fig. 2: We compared the proposed Motion Mamba with well-established state-of-theart methods such as MotionDiffuse [55], MDM [50], and MLD [6]. We presented three distinct motion prompts and visualized them in the form of motion sequence. The results demonstrated our superior performance compared to existing methods.

a person bends forward at the waist while both hands are tucked inside their armpits and elbows move up and down.

a man walks forward then turns right. a person does a squat and raises both arms over its head.

someone is playing the violin.

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

a man is pretending to be a chicken. constantly pecking at the ground and waving his arms like a chicken.

a person walks forward on an angle to the left.

person walks up and squats slightly to pose a position.

a person squats and raises both arms and then stand backup.

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

a man stumbles to his right. the motion seems surprised so he was probably pushed.

a person standing loses balance falling to the right and recovers standing.

a man performs a squat while lifting his arms to shoulder height and hands above his head.

a man runs to the right then runs to the left then back to the middle.

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

a man walks up and down from either stairs, rocks, or some unlevel terrain requiring a step.

a man pats his left hand with his right hand.

a man waves his right hand.

a man kicks with his left leg.

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

###### Fig. 3: We have included extra examples to showcase the proposed Motion Mamba model. These examples feature randomly selected prompts sourced from HumanML3D [17], providing additional visualizations of the model’s capabilities.

[Figure 33]

- Fig. 4: This figure presents the User Interface (UI) deployed for our User Study, wherein participants are presented with two videos, labeled as Video A and Video B, respectively. These videos are selected randomly from a pool consisting of outputs generated by three distinct methods, in addition to the Ground Truth (GT) for comparison. Participants are posed with two types of evaluative questions to gauge the effectiveness of the generated motions. The first question, "Which of the two motions is more realistic?", aims to assess the overall quality and realism of the motion capture. The second question, "Which of the two motions corresponds more accurately to the text prompt?", is designed to evaluate the congruence between the generated motion and the provided text prompt. This dual-question approach facilitates a comprehensive assessment of both the quality of the motion generation and its fidelity to the specified text prompts.

#### References

- 1. Ahuja, C., Morency, L.P.: Language2pose: Natural language grounded pose forecasting. In: 2019 International Conference on 3D Vision (3DV). pp. 719–728. IEEE

(2019)

- 2. Bao, F., Li, C., Cao, Y., Zhu, J.: All are worth words: a vit backbone for score-based diffusion models. arXiv preprint arXiv:2209.12152 (2022)
- 3. Baron, E., Zimerman, I., Wolf, L.: 2-d ssm: A general spatial layer for visual transformers. arXiv preprint arXiv:2306.06635 (2023)
- 4. Barsoum, E., Kender, J., Liu, Z.: Hp-gan: Probabilistic 3d human motion prediction via gan. In: Proceedings of the IEEE conference on computer vision and pattern recognition workshops. pp. 1418–1427 (2018)
- 5. Bhattacharya, U., Rewkowski, N., Banerjee, A., Guhan, P., Bera, A., Manocha, D.: Text2gestures: A transformer-based network for generating emotive body gestures for virtual agents. In: 2021 IEEE virtual reality and 3D user interfaces (VR). pp. 1–10. IEEE (2021)
- 6. Chen, X., Jiang, B., Liu, W., Huang, Z., Fu, B., Chen, T., Yu, G.: Executing your commands via motion diffusion in latent space. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18000–18010 (2023)
- 7. Dhariwal, P., Nichol, A.: Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems 34, 8780–8794 (2021)
- 8. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (2020)
- 9. Ghosh, A., Cheema, N., Oguz, C., Theobalt, C., Slusallek, P.: Synthesis of compositional animations from textual descriptions. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1396–1406 (2021)
- 10. Gong, K., Lian, D., Chang, H., Guo, C., Jiang, Z., Zuo, X., Mi, M.B., Wang, X.: Tm2d: Bimodality driven 3d dance generation via music-text integration. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9942–9952 (2023)
- 11. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. Advances in neural information processing systems 27 (2014)
- 12. Gu, A., Dao, T.: Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752 (2023)
- 13. Gu, A., Goel, K., Gupta, A., Ré, C.: On the parameterization and initialization of diagonal state space models. Advances in Neural Information Processing Systems 35, 35971–35983 (2022)
- 14. Gu, A., Goel, K., Ré, C.: Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396 (2021)
- 15. Gu, A., Johnson, I., Goel, K., Saab, K., Dao, T., Rudra, A., Ré, C.: Combining recurrent, convolutional, and continuous-time models with linear state space layers. Advances in neural information processing systems 34, 572–585 (2021)
- 16. Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., Courville, A.C.: Improved training of wasserstein gans. Advances in neural information processing systems 30 (2017)
- 17. Guo, C., Zou, S., Zuo, X., Wang, S., Ji, W., Li, X., Cheng, L.: Generating diverse and natural 3d human motions from text. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5152–5161 (2022)

- 18. Guo, C., Zuo, X., Wang, S., Zou, S., Sun, Q., Deng, A., Gong, M., Cheng, L.: Action2motion: Conditioned generation of 3d human motions. In: Proceedings of the 28th ACM International Conference on Multimedia. pp. 2021–2029 (2020)
- 19. Gupta, A., Gu, A., Berant, J.: Diagonal state spaces are as effective as structured state spaces. Advances in Neural Information Processing Systems 35, 22982–22994

(2022)

- 20. Harvey, F.G., Yurick, M., Nowrouzezahrai, D., Pal, C.: Robust motion inbetweening. ACM Transactions on Graphics (TOG) 39(4), 60–1 (2020)
- 21. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017)
- 22. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems 33, 6840–6851 (2020)
- 23. Hopfield, J.J.: Neural networks and physical systems with emergent collective computational abilities. Proceedings of the national academy of sciences 79(8), 2554– 2558 (1982)
- 24. Hu, V.T., Baumann, S.A., Gui, M., Grebenkova, O., Ma, P., Fischer, J., Ommer, B.: Zigma: A dit-style zigzag mamba diffusion model. In: Arxiv (2024)
- 25. Jiang, B., Chen, X., Liu, W., Yu, J., Yu, G., Chen, T.: Motiongpt: Human motion as a foreign language. Advances in Neural Information Processing Systems 36

(2024)

- 26. Kalman, R.E.: A New Approach to Linear Filtering and Prediction Problems. Journal of Basic Engineering 82(1), 35–45 (1960)
- 27. Kingma, D.P., Welling, M.: Auto-encoding variational bayes. stat 1050, 1 (2014)
- 28. Kramer, M.A.: Nonlinear principal component analysis using autoassociative neural networks. AIChE journal 37(2), 233–243 (1991)
- 29. Li, S., Singh, H., Grover, A.: Mamba-nd: Selective state space modeling for multidimensional data. arXiv preprint arXiv:2402.05892 (2024)
- 30. Li, Y., Cai, T., Zhang, Y., Chen, D., Dey, D.: What makes convolutional models great on long sequence modeling? In: The Eleventh International Conference on Learning Representations (2022)
- 31. Lin, X., Amer, M.R.: Human motion modeling using dvgans. arXiv preprint arXiv:1804.10652 (2018)
- 32. Liu, Y., Tian, Y., Zhao, Y., Yu, H., Xie, L., Wang, Y., Ye, Q., Liu, Y.: Vmamba: Visual state space model. arXiv preprint arXiv:2401.10166 (2024)
- 33. Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S.: A convnet for the 2020s. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 11976–11986 (2022)
- 34. Mahmood, N., Ghorbani, N., Troje, N.F., Pons-Moll, G., Black, M.J.: Amass: Archive of motion capture as surface shapes. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5442–5451 (2019)
- 35. Pavlakos, G., Choutas, V., Ghorbani, N., Bolkart, T., Osman, A.A.A., Tzionas, D., Black, M.J.: Expressive body capture: 3d hands, face, and body from a single image. In: Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR) (2019)
- 36. Petrovich, M., Black, M.J., Varol, G.: Action-conditioned 3D human motion synthesis with transformer VAE. In: International Conference on Computer Vision (ICCV) (2021)
- 37. Petrovich, M., Black, M.J., Varol, G.: Temos: Generating diverse human motions from textual descriptions. In: European Conference on Computer Vision. pp. 480–

497. Springer (2022)

- 38. Petrovich, M., Black, M.J., Varol, G.: TEMOS: Generating diverse human motions from textual descriptions. In: European Conference on Computer Vision (ECCV)

(2022)

- 39. Plappert, M., Mandery, C., Asfour, T.: The kit motion-language dataset. Big data 4(4), 236–252 (2016)
- 40. Plappert, M., Mandery, C., Asfour, T.: Learning a bidirectional mapping between human whole-body motion and natural language using deep recurrent neural networks. Robotics and Autonomous Systems 109, 13–26 (2018)
- 41. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PMLR (2021)
- 42. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)
- 43. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical image computing and computer-assisted intervention. pp. 234–241. Springer (2015)
- 44. Rumelhart, D.E., Hinton, G.E., Williams, R.J.: Learning internal representations by error propagation. Parallel Distributed Processing pp. 318–362 (1986)
- 45. Saharia, C., Ho, J., Chan, W., Salimans, T., Fleet, D.J., Norouzi, M.: Image superresolution via iterative refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence (2022)
- 46. Shi, X., Chen, Z., Wang, H., Yeung, D.Y., Wong, W.K., Woo, W.c.: Convolutional lstm network: A machine learning approach for precipitation nowcasting. Advances in neural information processing systems 28 (2015)
- 47. Smith, J., De Mello, S., Kautz, J., Linderman, S., Byeon, W.: Convolutional state space models for long-range spatiotemporal modeling. Advances in Neural Information Processing Systems 36 (2024)
- 48. Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., Ganguli, S.: Deep unsupervised learning using nonequilibrium thermodynamics. In: International conference on machine learning. pp. 2256–2265. PMLR (2015)
- 49. Tevet, G., Gordon, B., Hertz, A., Bermano, A.H., Cohen-Or, D.: Motionclip: Exposing human motion generation to clip space. In: European Conference on Computer Vision. pp. 358–374. Springer (2022)
- 50. Tevet, G., Raab, S., Gordon, B., Shafir, Y., Cohen-or, D., Bermano, A.H.: Human motion diffusion model. In: The Eleventh International Conference on Learning Representations (2022)
- 51. Van Den Oord, A., Vinyals, O., et al.: Neural discrete representation learning. Advances in neural information processing systems 30 (2017)
- 52. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017)
- 53. Wang, J., Yan, J.N., Gu, A., Rush, A.M.: Pretraining without attention. arXiv preprint arXiv:2212.10544 (2022)
- 54. Yan, J.N., Gu, J., Rush, A.M.: Diffusion models without attention. arXiv preprint arXiv:2311.18257 (2023)
- 55. Zhang, M., Cai, Z., Pan, L., Hong, F., Guo, X., Yang, L., Liu, Z.: Motiondiffuse: Text-driven human motion generation with diffusion model. IEEE Transactions on Pattern Analysis and Machine Intelligence (2024)

- 56. Zhang, Y., Black, M.J., Tang, S.: We are more than our joints: Predicting how 3d bodies move. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3372–3382 (2021)
- 57. Zhang, Z., Liu, A., Chen, Q., Chen, F., Reid, I., Hartley, R., Zhuang, B., Tang, H.: Infinimotion: Mamba boosts memory in transformer for arbitrary long motion generation. arXiv preprint arXiv:2407.10061 (2024)
- 58. Zhang, Z., Wang, Y., Wu, B., Chen, S., Zhang, Z., Huang, S., Zhang, W., Fang, M., Chen, L., Zhao, Y.: Motion avatar: Generate human and animal avatars with arbitrary motion. arXiv preprint arXiv:2405.11286 (2024)
- 59. Zhong, C., Hu, L., Zhang, Z., Xia, S.: Attt2m: Text-driven human motion generation with multi-perspective attention mechanism. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 509–519 (2023)
- 60. Zhu, L., Liao, B., Zhang, Q., Wang, X., Liu, W., Wang, X.: Vision mamba: Efficient visual representation learning with bidirectional state space model. arXiv preprint arXiv:2401.09417 (2024)

