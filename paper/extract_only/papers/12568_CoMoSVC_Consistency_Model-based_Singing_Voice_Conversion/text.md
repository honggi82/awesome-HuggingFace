# arXiv:2401.01792v1[eess.AS]3Jan2024

## CoMoSVC: Consistency Model-based Singing Voice Conversion

#### Yiwen Lu1, Zhen Ye1, Wei Xue1†, Xu Tan2, Qifeng Liu1, Yike Guo1†

1 Hong Kong University of Science and Technology 2 Microsoft Research Asia ∗

### Abstract

The diffusion-based Singing Voice Conversion (SVC) methods have achieved remarkable performances, producing natural audios with high similarity to the target timbre. However, the iterative sampling process results in slow inference speed, and acceleration thus becomes crucial. In this paper, we propose CoMoSVC, a consistency model-based SVC method, which aims to achieve both high-quality generation and high-speed sampling. A diffusion-based teacher model is first specially designed for SVC, and a student model is further distilled under self-consistency properties to achieve one-step sampling. Experiments on a single NVIDIA GTX4090 GPU reveal that although CoMoSVC has a significantly faster inference speed than the state-of-the-art (SOTA) diffusion-based SVC system, it still achieves comparable or superior conversion performance based on both subjective and objective metrics. Audio samples and codes are available at https://comosvc.github.io/.

Keywords: Singing Voice Conversion, Diffusion Model, Consistency Model

### 1 Introduction

Singing Voice Conversion(SVC) aims to convert one singer’s voice to another one’s, while preserving the content and melody. It has wide applications in music entertainment, singing voice beautification, and art creation Zhang et al. [2023].

Statistical methods Kobayashi et al. [2014, 2015b,a] are applied to the SVC tasks with parallel training data from both the source and target singers, which is usually infeasible, and thus the non-parallel SVC methods have become the mainstream. Two-stage methods are generally used for SVC, the first stage disentangles and encodes singer-independent and singer-dependent features from the audio. Then the second decoding stage generates the converted audio by replacing the singer-dependent feature with the target one. Since the substantial impact of the second stage on the quality of the converted audios, it has become crucial to design and optimize this stage. Therefore, many generative models have been used for the SVC decoding, including the autoregressive (AR) models, generative adversarial network (GAN), Normalizing Flow, and diffusion models. AR models are firstly used to develop USVC Nachmani and Wolf [2019], and PitchNet Deng et al. [2020] further improves USVC by adding a pitch adversarial network to learn the joint phonetic and pitch representation. However, AR models are slow due to the recursive nature, then non-AR GAN-based UCD-SVC Polyak et al. [2020] and FastSVC Liu et al. [2021a] are later proposed. Since the unstable training of GAN, a flow-based end-to-end SVC system, named SoVITS-SVC SVC-Develop-Team [2023] received widespread attention for its excellent converted results in fast speed. Recently, it has been shown that the conversion performance can be substantially improved by the diffusion-based SVC methods such as DiffSVC Liu et al. [2021b] and the diffusion version of SoVITS-SVC.

∗†Corresponding authors: Wei Xue {weixue@ust.hk}, Yike Guo {yikeguo@ust.hk}

Under Review.

However, the iterative sampling process results to the slow inference of the diffusion-based SVC methods. A new generative model named consistency model Song et al. [2023] has been proposed to realize one-step generation. Subsequently for speech synthesis, CoMoSpeech Ye et al. [2023] exploits the consistency model to achieve both high-quality synthesis and fast inference speed. Inspired by this, a consistency model-based SVC method, named CoMoSVC, is further developed in this paper to achieve high-quality, high-similarity and high-speed SVC. Based on the structure of EDM Karras et al. [2022], a diffusion-based teacher model with outstanding generative capability is firstly designed, and a student model is further distilled from it to achieve one-step sampling. Experiments reveal that while the sampling speed of CoMoSVC is approximately 500 and 50 times faster than that of the diffusion-based SoVITS-SVC and DiffSVC respectively, the comparable performance is still retained and some improvements can even be achieved in both quality and similarity.

### 2 Background

The diffusion model generates samples by first adding noise to data during the forward process and then reconstructing the data structure in the reverse process. We assume that the original data distribution is pdata(x), and the forward process can be represented by a stochastic differential equation (SDE) Song et al. [2021], Karras et al. [2022]:

dxt = f(xt,t)dt + g(t)dwt, (1) where wt is the standard wiener process, f(xt,t) and g(t) are drift and diffusion coefficients respectively. With setting f(xt,t) = 0 and g(t) = √2t, the same as the choice in Karras et al. [2022], the SDE can be defined by:

√

2tdwt. (2) The reverse process can also be expressed by a reverse-time SDE Song et al. [2021]:

dxt =

√

2tdw¯t, (3)

dxt = −2t∇log pt(xt)dt +

where pt(xt) is the distribution of xt, ∇log pt(xt) is the score function, and w¯t is the reverse-time standard wiener process. Song et al. [2021] found that there exists a probability flow (PF) ordinary

differential equation (ODE), whose solution trajectories’ distribution at time t is the same as pt(xt). The PF ODE with such property can be represented by

xt − Dϕ (xt,t) t

dxt dt

, (4)

= −t∇log pt (xt) =

where Dϕ is the neural network with ϕ as parameters to approximate the denoiser function. Then for sampling, the PF ODE is solved by initiating from xT, as

0

xt − Dϕ (xt,t) t

dt. (5)

x0 = xT +

T

However, the diffusion model generally needs a large number of iterations to solve the PF ODE, making the sampling slow. A consistency model Song et al. [2023] is proposed for one-step sampling based on the self-consistency property, making any point from the same PF ODE trajectory be mapped to the same initial point. The self-consistency properties have two constraints: firstly, any pair of points xt

and xt

will be mapped to the same point, which can be represented by:

m

n

,tn). (6) Secondly, the initial point should also be mapped to itself and this constraint is called the boundary condition. To avoid numerical instability, it can be given by

Dϕ(xt

,tm) = Dϕ(xt

m

n

Dϕ(xϵ,tϵ) = xϵ, (7) where ϵ is a fixed small positive number and set as 0.002. , all the singers have their identification number, which will be encoded as a singer embedding.

### 3 Proposed Method

CoMoSVC is a two-stage model, where the first stage encodes the extracted features and the singer identity into embeddings. These embeddings are concatenated and serve as the conditional input for the second stage to generate mel-spectrogram, which can be further rendered to audio by using a pre-trained vocoder. The training process depicted in Fig. 1 takes the waveform and its singer identity as the input to reconstruct the mel-spectrogram, while the inference process illustrated in Fig. 2 replaces the singer identity with the target one to generate the converted mel-spectrogram.

[Figure 1]

###### Waveform

###### Singer ID

EncodingDecoding

Singer Embedding

Loudness

Content Pitch

cond

###### Teacher Model CoMoSVC

###### Teacher Model

[Figure 2]

[Figure 3]

[Figure 4]

ln(𝑡)~𝑁(𝑃 ,𝑃 ) 𝐱 = 𝐱 +𝑁(0,𝑡 𝐼)

𝐱 = 𝐱 + t ∗ 𝑁(0,I)

𝐱 = 𝐱 + t ∗ 𝑁(0, I)

[Figure 5]

[Figure 6]

𝐱 ∅ estimated from 𝐱

𝐷∅(𝐱 ,𝑡,𝑐𝑜𝑛𝑑)

Reconstruction Loss

[Figure 7]

[Figure 8]

[Figure 9]

Consistency Loss

𝐷 ( 𝐱 ∅ ,𝑡 , 𝑐𝑜𝑛𝑑) 𝐷 (𝐱 ,𝑡 , 𝑐𝑜𝑛𝑑)

Ground Truth 𝐱

Training of Teacher Model Consistency Distillation

Figure 1: The Training Process.

#### 3.1 Encoding

This section encodes both singer-independent and singer-dependent features, which can be shown in the upper part of both Fig. 1 and Fig. 2. We extract content, pitch, and loudness features to capture singer-independent information in audio, while the singer ID is used as the singer-dependent information. The content features are extracted by using the pre-trained acoustic model ContentVec Qian et al. [2022] and the large dimensionality of these features allows for enhancing the clarity of lyrics in the converted audio. To represent pitch information, we use the widely-used and classical F0 estimator DIO Morise et al. [2009]. The squared magnitude of the audio signal is calculated as the loudness feature. After feature extraction, we applied a linear layer to all the embeddings to unify the dimensions and concatenate them to form the conditional input for the decoding stage.

#### 3.2 Decoding

This stage is the key component of CoMoSVC, during which the mel-spectrograms can be generated from the conditional input. A teacher model is first trained and then a student model is distilled from it, which will be introduced in section. 3.2.1 and section. 3.2.2 respectively. The sampling process of both the teacher model and student model will be explained in section. 3.2.3.

[Figure 10]

###### Waveform

###### Target Singer ID

EncodingDecoding

Target Singer Embedding

Loudness

Content Pitch

cond

Teacher Model

CoMoSVC

[Figure 11]

[Figure 12]

𝐱 = t ∗ 𝑁(0,I)

𝐱 = t ∗ 𝑁(0, I)

… Nsteps Onestep

[Figure 13]

[Figure 14]

Sampling of Teacher Model Sampling of Student Model

Figure 2: The inference process.

#### 3.2.1 Teacher Model

We use the architecture of EDM Karras et al. [2022] as the teacher model to train the denoiser function Dϕ due to its high generative ability. Moreover, the structure of Dϕ used here is the noncausal Wavenet Rethage et al. [2018]. We use x0 ∼ pdata(x) and cond to denote the ground truth mel-spectrogram and the conditional input. According to (4), the empirical ODE is

xt − Dϕ (xt,t,cond) t

dxt dt

, (8)

=

where xt = x0 + t ∗ N(0,I), represents the result after adding noise. Similar to Karras et al. [2022], we use a different network Fϕ instead of directly approximating the denoiser function by Dϕ. The network is preconditioned with a skip connection to make the estimation more flexible, it can be given by

Dϕ(xt,t,cond) = cskip (t)xt + cout(t)Fϕ (cin (t)xt,t,cnoise (t)). (9)

cskip(t) modulates the skip connection, cin (t) and cout (t) scale the magnitudes of xt and Fϕ respectively, and cnoise (t) maps noise level t into a conditioning input for Fϕ. To satisfy the boundary condition mentioned in (7) and ensure cskip (t) and cout (t) differential, we choose

σdata2 (t − ϵ)2 + σdata2

σdata (t − ϵ) σdata2 + t2

, (10)

cskip (t) =

, cout (t) =

- Algorithm 1 Training procedure

Input: The denoiser function Dϕ of the teacher model; the conditional input cond; the original data distribution pdata and µ

- 1: repeat
- 2: Sample n ∼ U(1,N − 1) and x0 ∼ pdata
- 3: Sample xt

n+1 ∼ N(x0,(tn+1)2 ∗ I)

- 4: xˆϕt

n

← t

n

tn+1 xt

n+1

+ t

n+1−tn

tn+1 Dϕ xt

n+1

,tn+1,cond

- 5: Lθ ← d Dθ xt

n+1

,tn+1,cond ,Dθ− x ˆϕt

n

,tn,cond

- 6: θ ← θ − η∇θL(θ,θ−;ϕ)
- 7: θ− ← stopgrad(µθ− + (1 − µ)θ)
- 8: until convergence

- Algorithm 2 Sampling procedure

Input: The denoiser function Dθ of the consistency model; the conditional input cond ; a set of time points ti∈{0,...,N}

- 1: Sample xN ∼ N(0,σ(tN)2 ∗ I)
- 2: x ← Dθ(xN,tN,cond)
- 3: if one-step synthesis
- 4: Output: x
- 5: else multi-step synthesis
- 6: for i = N − 1 to 1 do
- 7: Sample z ∼ N(0,I)
- 8: xi ← x + t2i − ϵ2z

- 9: x ← Dθ(xi,ti,cond)
- 10: end for Output: x

where σdata is the standard deviation of pdata(x). The loss function Lϕ used to train the Dϕ can be designed by

Lϕ = E[λ(t)∥Dϕ (xt,t,cond) − x0∥2], (11)

where λ(t) = (t2 + σdata2 )/(t · σdata )2, denotes the weight corresponding to different noise level t. The entire procedure is depicted in the lower left section in Fig. 1.

#### 3.2.2 Consistency Distillation

A student model can be further distilled from the pre-trained denoiser function Dϕ to ultimately achieve one-step sampling, the process is illustrated in Algorithm. 1 and the lower right section of

Fig. 1. First, we randomly sample n from the uniform distribution U(1,N − 1) and obtain xt

n+1

by adding tn+1 ∗ N(0,I) to x0, then we use the Dϕ to get the one-step estimation xˆϕt

. According to (4), since first-order Euler Solver is used here, it can be given by

from xt

n+1

n

tn+1 − tn tn+1

tn tn+1

xˆϕt

,tn+1,cond . (12)

#### xt

+

Dϕ xt

=

n+1

n+1

n

The structure of the student model is inherited from the teacher model’s denoiser function Dϕ, resulting in Dθ and Dθ−. The parameters θ and θ− are initialized with ϕ, θ− is a running average of

the past values of θ. Afterwards, we use Dθ− x ˆϕt

,tn+1,cond to obtain different outputs of the pair of adjacent points xˆϕt

,tn,cond and Dθ xt

n+1

n

. The consistency distillation is trained by minimizing the L2 distance between the two outputs:

and xt

n

n

,tn+1,cond ,Dθ− x ˆϕt

d Dθ xt

,tn,cond

n+1

n

(13)

,tn+1,cond − Dθ− x ˆϕt

,tn,cond ∥2.

= ∥Dθ xt

n+1

n

The parameter θ is updated by:

θ ← θ − η∇θL(θ,θ−;ϕ). (14)

To stabilize the training, the exponential moving average (EMA) update and stop grad are adopted to θ−, as:

θ− ← stopgrad µθ− + (1 − µ)θ , (15) where µ is a momentum coefficient, empirically set as 0.95.

- 3.2.3 Sampling Process

The sampling processes of both the two models are depicted in the lower part of Fig. 2. The teacher model takes a number of iterations for sampling, while the student model can achieve one-step sampling as summarized in Algorithm. 2. We first sample the noise that has the same shape as the mel-spectrogram by xt

N

= tN ∗N(0,I), and the output of Dθ(xt

N

,tN,cond) is the sampling result. The proposed CoMoSVC also supports multi-step sampling by chaining the outputs at multiple time steps. However, there will be a trade-off between the number of iterations and sampling quality.

- 4 Experiments

- 4.1 Experimental Setup

We conduct our experiment on two open-source datasets, which are M4Singer Zhang et al. [2022] and OpenSinger Huang et al. [2021], respectively. The former dataset has 29.77 hours of singing voice and 20 singers, and the latter one contains 50 hours and 66 singers. All the audios are resampled to 24kHz and normalized. Then we calculate the volume feature, extract the F0 curve along with the voiced/unvoiced flag for each frame by using DIO Morise et al. [2009] and the 768-dimensional content feature from the 12th layer by utilizing ContentVec Qian et al. [2022]. All these features are projected to 256 dimensions and then concatenated as the conditional input for the decoding stage. We use the vocoder2 pre-trained with singing voice from M4singer Zhang et al. [2022], and the mel-spectrograms are computed with 512-point fast Fourier transform (FFT), 512-point window size and 128-point hop size with 80 frequency bins.

All the models are trained for 1 million iterations on a single NVIDIA GTX4090 GPU with a batch size of 48, with learning rates as 1e-4 and 5e-5 respectively and the optimizer is AdamW.

We first conduct the reconstruction experiment to evaluate the capabilities of different decoding stages in the autoencoding settings. Then two sets of experiments are conducted for any-to-many SVC task, a) train on the OpenSinger dataset for the target singer and use M4singer as the source for conversion ; b) train on the M4singer dataset for the target singer and use OpenSinger as the source for conversion. Moreover, we increase the sampling steps of CoMoSVC and conduct the conversion experiments to evaluate the effect of sampling steps.

- 4.2 Baselines We compare the proposed CoMoSVC with the SOTA SVC methods, including:

- • SoVITS-Flow: The flow version of SoVITS-SVC3.
- • SoVITS-Diff: The diffusion version of SoVITS-SVC4. The number of diffusion steps is 1000.
- • DiffSVC Liu et al. [2021b]: The first SVC method based on the diffusion model, and the number of steps is 100.

The same feature embeddings and training details as described in Sec. 4.1 are used for the baseline methods for fair comparison.

- 4.3 Evaluation

We evaluate the reconstruction ability of different methods by objective metrics and the conversion ability by both subjective and objective metrics. For the subjective test, we invited 12 volunteers to

- 2https://github.com/M4Singer/M4Singer/tree/master/code
- 3https://github.com/svc-develop-team/so-vits-svc?tab=readme-ov-file#sovits-model
- 4https://github.com/svc-develop-team/so-vits-svc?tab=readme-ov-file#diffusion-model-optional

Table 1: Objective Evaluations for Reconstruction METHOD NFE(↓) FPC(↑) PESQ(↑) CER(↓) SIM(↑)

SoVITS-Flow 1 0.935 2.486 25.03 0.948 SoVITS-Diff 1000 0.938 2.826 26.04 0.970 DiffSVC 100 0.941 2.917 25.47 0.972

Teacher 50 0.945 2.967 20.96 0.982 CoMoSVC 1 0.943 2.948 24.69 0.970

- Table 2: Objective Evaluations for SVC, where “M” and “O” in the SVC setting row stands for M4Singer and OpenSinger, respectively.

SVC SETTING M→O O→M METHOD NFE(↓) SIM(↑) CER(↓) SIM(↑) CER(↓)

SoVITS-Flow 1 0.784 21.76 0.585 22.62 SoVITS-Diff 1000 0.804 20.40 0.598 25.50 DiffSVC 100 0.801 21.27 0.598 22.58

Teacher 50 0.794 20.52 0.614 19.57 CoMoSVC 1 0.801 21.49 0.585 19.76

give Mean Opinion Score (MOS) on naturalness and similarity on the converted audios. Real-Time Factor (RTF), Character Error Rate (CER) obtained by Whisper Radford et al. [2023] and speaker SIMilarity (SIM) calculated by the cosine distance between the speaker embeddings 5 are used as the objective metrics for SVC evaluation. Since the flow version SoVITS-SVC is end-to-end and the other methods are two-stage, we use the time ratio of transforming embeddings into latent representations (Flow) / mel-spectrograms (Others) to the duration of audio to represent RTF for clear comparison. For reconstruction, F0 Pearson correlation coefficient (FPC) 6 and PESQ [ITU-T] are used additionally for evaluation.

#### 4.3.1 Reconstruction

As illustrated in Table. 1, the teacher model outperforms all the models in all the metrics. Similarly, CoMoSVC outperforms all the baselines in all metrics except for similarity, where it achieves comparable results. This indicates the outstanding generative ability of the decoding stage of CoMoSVC with only one step, which is hundreds or thousands of times fewer than all the baselines.

#### 4.3.2 SVC Performances

We conduct two sets of SVC experiments as described in section. 4.1 and the source audios are all unseen during training. As illustrated by CER and SIM in Table. 2, CoMoSVC performs comparably to all the baselines. The subjective evaluations in Table. 3 reveal that CoMoSVC achieves comparable naturalness to the diffusion-based SVC methods. Furthermore, the similarity of CoMoSVC exceeds that of all the baselines in both experiments, demonstrating an improvement of at least 0.05 to diffusion-based SVC methods. Moreover, both the naturalness and similarity of CoMoSVC show an increment of approximately 1 compared to the flow version SoVITS-SVC. As to the inference speed, the RTF of CoMoSVC is 0.002 smaller than that of the flow version SoVITS-SVC. In comparison with the diffusion-based SVC methods, CoMoSVC is more than 45 times faster than DiffSVC and almost 500 times faster than the diffusion-version SoVITS-SVC.

- 5https://github.com/Jungjee/RawNet
- 6https://github.com/open-mmlab/Amphion/blob/main/evaluation/metrics/f0/

- Table 3: Subjective Evaluations for SVC, where, as an example, M4Singer→OpenSinger means converting the singing voices from the M4Singer to target timbres in the OpenSinger.

SVC SETTING M4Singer→OpenSinger OpenSinger→M4Singer METHOD NFE(↓) RTF(↓) MOS/N(↑) MOS/S(↑) MOS/N(↑) MOS/S(↑)

SoVITS-Flow 1 0.008 3.27 ± 0.21 3.03 ± 0.23 3.10 ± 0.22 2.90 ± 0.23 SoVITS-Diff 1000 2.978 4.43 ± 0.14 3.90 ± 0.19 4.32 ± 0.15 3.99 ± 0.19 DiffSVC 100 0.278 4.44 ± 0.14 3.91 ± 0.19 4.23 ± 0.19 3.95 ± 0.21

Teacher 50 0.148 4.43 ± 0.15 3.92 ± 0.19 4.47 ± 0.14 4.05 ± 0.18 CoMoSVC 1 0.006 4.42 ± 0.13 3.96 ± 0.19 4.27 ± 0.16 4.00 ± 0.19

- Table 4: Evaluations for Effect of Sampling Steps, where the number in the method name represents the number of sampling steps.

SVC SETTING M→O METHOD MOS/N(↑) MOS/S(↑)

CoMoSVC-4 4.46 ± 0.14 3.95 ± 0.19 CoMoSVC-2 4.36 ± 0.14 3.88 ± 0.19

- CoMoSVC-1 4.42 ± 0.13 3.96 ± 0.19 SVC SETTING O→M METHOD MOS/N(↑) MOS/S(↑) CoMoSVC-4 4.38 ± 0.15 4.05 ± 0.19

- CoMoSVC-2 4.34 ± 0.15 4.01 ± 0.19 CoMoSVC-1 4.27 ± 0.16 4.00 ± 0.19

- 4.3.3 Effect of Sampling Steps

In general, as the number of sampling steps increases, there is a small increment in the metrics presented in Table 4. The slight improvement and minor fluctuation indicates that CoMoSVC already achieves accurate score estimation through only one-step discretization yielding high-quality results.

- 5 Conclusion

In this paper, we propose the CoMoSVC, which is based on the consistency model to achieve high-quality, high-similarity and high-speed SVC. The proposed CoMoSVC is a two-stage model where the first stage encodes the features from the waveform, then the second stage utilizes a student model distilled from a pre-trained teacher model to generate converted audios. The comprehensive subjective and objective evaluations demonstrate the effectiveness of CoMoSVC.

### Acknowledgments

The research was supported by the Theme-based Research Scheme (T45-205/21-N) and Early Career Scheme (ECS-HKUST22201322), Research Grants Council of Hong Kong.

### References

Chengqi Deng, Chengzhu Yu, Heng Lu, Chao Weng, and Dong Yu. Pitchnet: Unsupervised singing voice conversion with pitch adversarial network. In Proc. Intl. Conf. on Acoustics, Speech, and Signal Processing (ICASSP), 2020.

Rongjie Huang, Feiyang Chen, Yi Ren, Jinglin Liu, Chenye Cui, and Zhou Zhao. Multi-singer: Fast multi-singer singing voice vocoder with a large-scale corpus. In Proc. ACM Int. Conf. on Multimedia (ACM MM), 2021.

Intl. Telecommunications Union (ITU-T). Perceptual evaluation of speech quality (PESQ), an objective method for end-to-end speech quality assessment of narrowband telephone networks and speech codecs. Recommendation P.862, Intl. Telecommunications Union (ITU-T), February 2001.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. In Proc. Conf. on Neural Information Processing Systems (NeurIPS), 2022.

Kazuhiro Kobayashi, Tomoki Toda, Graham Neubig, Sakriani Sakti, and Satoshi Nakamura. Statistical singing voice conversion with direct waveform modification based on the spectrum differential. In Proc. InterSpeech, 2014.

Kazuhiro Kobayashi, Tomoki Toda, and Satoshi Nakamura. Statistical singing voice conversion based on direct waveform modification and its parameter generation algorithms. IEICE Tech. Rep., 115(253):7–12, 2015a.

Kazuhiro Kobayashi, Tomoki Toda, Graham Neubig, Sakriani Sakti, and Satoshi Nakamura. Statistical singing voice conversion based on direct waveform modification with global variance. In Proc. InterSpeech, 2015b.

Songxiang Liu, Yuewen Cao, Na Hu, Dan Su, and Helen Meng. FastSVC: Fast cross-domain singing voice conversion with feature-wise linear modulation. In Proc. Intl. Conf. Multimedia and Expo (ICME), 2021a.

Songxiang Liu, Yuewen Cao, Dan Su, and Helen Meng. Diffsvc: A diffusion probabilistic model for singing voice conversion. In Proc. IEEE Workshop on Automatic Speech Recognition and Understanding (ASRU), pages 741–748, 2021b.

Masanori Morise, Hideki Kawahara, and Haruhiro Katayose. Fast and reliable f0 estimation method based on the period extraction of vocal fold vibration of singing voice and speech. In Audio Engineering Society Conference: 35th International Conference: Audio for Games, 2009.

Eliya Nachmani and Lior Wolf. Unsupervised singing voice conversion. In Proc. InterSpeech, 2019. Adam Polyak, Lior Wolf, Yossi Adi, and Yaniv Taigman. Unsupervised cross-domain singing voice

conversion. In Proc. InterSpeech, 2020.

Kaizhi Qian, Yang Zhang, Heting Gao, Junrui Ni, Cheng-I Lai, David Cox, Mark Hasegawa-Johnson, and Shiyu Chang. ContentVec: An improved self-supervised speech representation by disentangling speakers. In Proc. Intl. Conf. Machine Learning (ICML), 2022.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In Proc. Intl. Conf. Machine Learning (ICML), 2023.

Dario Rethage, Jordi Pons, and Xavier Serra. A wavenet for speech denoising. In Proc. Intl. Conf. on Acoustics, Speech, and Signal Processing (ICASSP), 2018.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In Proc. Intl. Conf. on Learning Representations (ICLR), 2021.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In Proc. Intl. Conf. Machine Learning (ICML), 2023.

SVC-Develop-Team. Softvc vits singing voice conversion. https://github.com/

##### svc-develop-team/so-vits-svc, 2023.

Zhen Ye, Wei Xue, Xu Tan, Jie Chen, Qifeng Liu, and Yike Guo. Comospeech: One-step speech and singing voice synthesis via consistency model. In Proc. ACM Int. Conf. on Multimedia (ACM MM), 2023.

Lichao Zhang, Ruiqi Li, Shoutong Wang, Liqun Deng, Jinglin Liu, Yi Ren, Jinzheng He, Rongjie Huang, Jieming Zhu, Xiao Chen, et al. M4singer: A multi-style, multi-singer and musical score provided mandarin singing corpus. In Proc. Conf. on Neural Information Processing Systems (NeurIPS), 2022.

Xueyao Zhang, Yicheng Gu, Haopeng Chen, Zihao Fang, Lexiao Zou, Liumeng Xue, and Zhizheng Wu. Leveraging content-based features from multiple acoustic models for singing voice conversion. arXiv preprint arXiv:2310.11160, 2023.

