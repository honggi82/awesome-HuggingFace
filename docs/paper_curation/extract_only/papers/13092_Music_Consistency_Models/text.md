## Music Consistency Models

Zhengcong Fei, Mingyuan Fan, Junshi Huang Kunlun Inc. {feizhengcong}@gmail.com

# arXiv:2404.13358v1[cs.SD]20Apr2024

Abstract

Consistency models have exhibited remarkable capabilities in facilitating efficient image/video generation, enabling synthesis with minimal sampling steps. It has proven to be advantageous in mitigating the computational burdens associated with diffusion models. Nevertheless, the application of consistency models in music generation remains largely unexplored. To address this gap, we present Music Consistency Models (MusicCM), which leverages the concept of consistency models to efficiently synthesize mel-spectrogram for music clips, maintaining high quality while minimizing the number of sampling steps. Building upon existing text-to-music diffusion models, the MusicCM model incorporates consistency distillation and adversarial discriminator training. Moreover, we find it beneficial to generate extended coherent music by incorporating multiple diffusion processes with shared constraints. Experimental results reveal the effectiveness of our model in terms of computational efficiency, fidelity, and naturalness. Notable, MusicCM achieves seamless music synthesis with a mere four sampling steps, e.g., only one second per minute of the music clip, showcasing the potential for real-time application.

### 1 Introduction

In recent years, the field of text-to-music generation has witnessed tremendous progress in synthesizing natural and coherent music clips, primarily driven by the development of diffusion models. Several diffusion-based methods such as Noise2Music [Huang et al., 2023b], Mousai [Schneider et al., 2023], MusicLDM [Chen et al., 2023], Make-An-Audio [Huang et al., 2023c], and AudioLDM [Liu et al., 2023c; Liu et al., 2023a; Zhu et al., 2023], have achieved notable performance by integrating additional audio mel-spectrogram into existing image diffusion models [Rombach et al., 2022a; Ramesh et al., 2022] to effectively handle the temporal and spectral characteristics. However, these diffusion-based approaches inherently necessitate a considerable number of sampling steps during the music synthesis process in inference, e.g., 50-step DDIM sampling [Song et al., 2020]. This

limitation poses an obstacle to the efficient and expeditious generation of high-quality musical compositions.

To address the issue of high sampling cost in diffusion models, the concept of consistency models has been introduced within the fields of image/video generation [Song et al., 2023; Song and Dhariwal, 2023; Luo et al., 2023a; Luo et al., 2023b; Xiao et al., 2023; Wang et al., 2023a]. These methods are designed to generate samples in a single step while retaining the crucial advantages of diffusion models. These advantages include the ability to trade computational resources for sample quality through multi-step sampling and the capability of performing zero-shot data editing [Yang et al., 2023b]. By enabling efficient synthesis with a minimal number of steps, such models have achieved remarkable progress, reducing the required steps from 50 to as few as 4. Despite the notable success achieved in image generation, the application of consistency models in the realm of music synthesis remains largely unexplored.

Following this premise, we present the Music Consistency Models, denoted as MusicCM, a diffusion-based consistency model designed for music creation according to text prompts. Our approach is centered around the utilization of existing diffusion models within the domain of music generation, while also incorporating the concepts of consistency distillation. Specifically, by implementing the MusicCM framework, our objective is to mitigate the necessity for extensive sampling procedures while concurrently upholding the production of high-quality synthesized music. Through the utilization of adversarial discrimination [Sauer et al., 2023b; Fei, 2020], the model is compelled to generate samples that reside directly on the manifold of authentic musical compositions during each forward pass, thereby circumventing issues such as blurriness and other artifacts often encountered in alternative distillation methods. In light of the works [BarTal et al., 2023; Wang et al., 2023b; Fei, 2021a], we further introduce an enhanced generation process that combines several reference diffusion generation processes bound together with a set of shared constraints. In between, MusicCM is applied to different regions in the generated mel-spectrogram, serving as a reference, predicting a denoising sampling step for each. Subsequently, a global denoising sampling step is taken that harmonizes these individual steps through a least squares optimal solution, ultimately resulting in a unified and coherent synthesis.

Experimentally, we have obtained quantitative and qualitative findings that unequivocally validate the efficacy of our proposed approach. Notably, by integrating the aforementioned techniques, our method attains a remarkable level of fidelity in music synthesis, employing a mere 4 to 6 sampling steps, equivalent to approximately one second for each minute of music clips. This outcome serves as a testament to the potential of our method for facilitating rapid and real-time synthesis processes.

Contributions. To sum up, our contributions are as follows: (i) We present MusicCM, a novel framework that aims to bridge the gap between diffusion models and consistency models in the domain of music generation. Our approach incorporates consistency distillation and an adversarial discriminator to enable the efficient synthesis of high-quality music clips. (ii) To address the challenges of maintaining long music consistency while ensuring memory efficiency, we introduce multiple diffusion processes with shared constraints during the inference stage. (iii) We provide both quantitative and qualitative results to demonstrate the effectiveness of

- our MusicCM approach. To improve reproducibility, we will publicly release the source code and trained models of all experiments. Finally, by exploring the potential of consistency models in music generation, we aim to contribute to the field of fast music synthesis and provide a simplified and effective baseline for future research.

2 Related Works

Text-to-image generation. Remarkable advancements have been achieved in text-to-image synthesis, primarily attributed to the emergence of generative models [Goodfellow et al., 2014; Ho and Salimans, 2022; Kang et al., 2023; Shen and Zhou, 2021; Fei et al.,

- 2022a]. Especially diffusion models [Kawar et al., 2023; Mou et al., 2023; Nichol et al., 2021; Rombach et al., 2022a; Ruiz et al., 2023] play a crucial role. Vari-

ous methodologies, such as [Ramesh et al., 2022; Gafni et al., 2022], propose a two-stage approach where the input text undergoes conversion into image embeddings using a prior model, which is then utilized for image synthesis. Stable Diffusion [Rombach et al., 2022a] introduces a VAE-based approach in the latent space to decrease computational demand and optimizes the model with large-scale datasets [Schuhmann et al., 2022]. [Sauer et al., 2023b] use score distillation from a teacher models combination with an adversarial loss to maintain high quality with few-step settings. Subsequent methods [Zhang et al., 2023; Fei et al., 2023b; Fei et al., 2024b; Fei et al., 2024a; Huang et al., 2023a] have incorporated additional conditional inputs, such as depth maps or sketches, for spatially controllable image synthesis.

Consistency model. Inference cost is an important factors for multimodal generation [Fei, 2019; Fei, 2021b; Yan et al., 2021; Fei et al., 2022b]. Consistency model [Song et al.,

- 2023] has been developed to address the limitation of numerous inference steps in diffusion models. By leveraging the probability flow ordinary differential equation, consistency models aim to learn the mapping of any point at any

given time step to the initial point of the trajectory, which corresponds to the original clean image. The introduction of the consistency model allows for efficient one-step image generation without compromising the benefits of multi-step iterative sampling. Consequently, this approach also facilitates the production of high-quality results through multistep inference. Moreover, LCM [Luo et al., 2023a] explores consistency models in the latent space to save memory consumption and improve inference efficiency. Subsequently, several methods [Luo et al., 2023b; Sauer et al., 2023b; Xiao et al., 2023; Wang et al., 2023a] have also delved into efficient generation techniques, building upon the foundation set by the consistency model, and achieved remarkable outcomes. Inspired by the success of the consistency model in the realm of image/video generation, we propose extending its application to the domain of music generation.

Conditional music generation. Numerous studies have been conducted on the audio generation guided by text, including Diffsound [Yang et al., 2023a], AudioGen [Kreuk

- et al., 2022], AudioLDM [Liu et al., 2023b], and Makean-Audio [Huang et al., 2023d] showing impressive results. In the domain of music, there exist text-to-music models such as retrieval-based MuBERT [MubertAI, ], languagemodel-based MusicLM [Agostinelli et al., 2023], diffusionbased Riffusion [Forsgren and Martiros, 2022] Noise2Music [Huang et al., 2023b] and so on [Li et al., 2023; Melechovsky
- et al., 2023]. However, music models often necessitate substantial quantities of privately owned music data that are inaccessible to the public, impeding the reproducibility and further development of research efforts. Among these aforementioned models, MusicLDM is based on open-source Stable Diffusion [Rombach et al., 2022b], CLAP [Wu et al., 2023; Fei et al., 2023a], and HiFi-GAN [Kong et al., 2020a] architectures. Therefore, we base our approach on these, to create MusicCM for our experiments.

### 3 Methodology

The proposed MusicCM framework extends the principles of consistency models. Specifically, we begin by providing a concise overview of consistency models. Subsequently, we delve into the intricate details of the proposed MusicCM framework. A presentation of the overall structure is depicted in Figure 1. Lastly, we devise a restriction mechanism to address the challenges associated with lengthy and coherent music creation.

#### 3.1 Preliminaries: Consistency Models

In order to expedite the image generation, [Song et al., 2023] brings into the conception of the consistency model. It endeavors to enhance a learning framework that can efficiently map any given point in time to the initial point of the PFODE trajectory. Formally, the self-consistency property can be formulated as:

fθ(xt,t) = fθ(xt′,t′),∀t,t′ ∈ [ϵ,T], (1)

where ϵ is a time step, T is the overall denoising step, and xt denotes the noised input.

|[Figure 1]| |
|---|---|
| | |

|[Figure 2]| |
|---|---|
| | |

|[Figure 3]| |
|---|---|
| | |

🔥 Tunable ❄ Frozen

🔥 ❄ ❄

EMA

Forward Diffusion

Real / Fake

DDIM

Student Student*

Teacher

Discriminator

| |[Figure 4]| |
|---|---|---|
| | | |

| |[Figure 5]|
|---|---|
| | |

[Figure 6]

| |[Figure 7]|
|---|---|
| | |

Adversarial Loss

Distillation Loss

Source Music

- Figure 1: Overview of music consistency models. Given a source music mel-spectrogram x0, a forward diffusion operation is first performed to add noise to the music. Then, the noised xn+k is entered into the student and teacher model to predict music clips. xˆn is estimated by the teacher model and fed into the EMA student model. To learn self-consistency, a distillation loss is imposed to constrain the output of the two student models to be consistent, and an adversarial loss is used to fool a discriminator which is trained to distinguish the generated samples

xpred0 from real music x0. The whole consistency distillation is conducted in the latent space, and conditional guidance is omitted for ease of presentation. The teacher model is a music diffusion model, and the student shares the same network structure as the teacher model and is initialized with the parameters of the teacher model.

To accelerate the training and extract the strong prior knowledge of the established diffusion models [Rombach et al., 2022a], consistency distillation is proposed as:

,tn))] (2)

Ldistil(θ,θ∗;Φ) = E[d(fθ(xt

,tn+1),fθ∗(ˆxt

n+1

n

where Φ means the applied ODE solver and the model parameters θ∗ are obtained from the exponential moving average (EMA) of θ. xˆt

according to: xˆt

is the estimation of xt

n

n

,tn+1) (3)

n ← xt

+ (tn − tn−1)Φ(xt

n+1

n+1

LCM [Luo et al., 2023a] conducts the above consistency optimization in the latent space and applies classifier guidance [Ho and Salimans, 2022] in Equation 3 to inject control signals, such as textual prompts. For more details, please refer to the original works [Song et al., 2020].

#### 3.2 Application to Music

The proposed MusicCM is also established within the latent space to mitigate the computational burden, in accordance with LCM. To harness the substantial knowledge encapsulated in pre-trained music diffusion models and speed up the training process, we employ consistency distillation and adversarial discrimination strategies. It is pertinent to observe that the pre-trained diffusion models under consideration may encompass diverse typologies, e.g., MusicLDM and Noise2Music.

In MusicCM, we apply DDIM [Song et al., 2020] as the basic ODE solver Ψ to estimate xˆt

as: xˆt

n

,tn+1,tn,c) (4)

n ≈ xt

+ Ψ(xt

n+1

n+1

where c means the conditional inputs, which can be textual prompts in a text-to-music generation.

Classifier-free guidance. Given the pivotal role of classifier-free guidance in the synthesis of high-quality content, we extend its application to the consistency distillation stage, introducing a parameter denoted as w to regulate the scale of guidance:

,tn+1,tn,c) (5) − wΨ(xt

n ≈xt

xˆt

+ (1 + w)Ψ(xt

n+1

n+1

,tn+1,tn,ϕ) (6) To maintain alignment of initial parameters and design of the consistency model consistent with the teacher diffusion model, we train the consistency model with a fixed value of w, such as 2.0. It is noteworthy that classifier-free guidance is exclusively implemented in the training phase for the teacher diffusion model and is not deemed essential during the inference process of the consistency model. During the inference phase, we can sample 4∼6 LCM steps to produce plausible results on text-to-music generation.

n+1

Adversarial discriminator. We follow the discriminator design in [Sauer et al., 2023a], where a frozen pre-trained feature network F and a set of trainable lightweight discriminator head Dφ,k, which are applied on features at different layers of F. Similar to [Sauer et al., 2023b], the discriminator is conditioned on text embedding and mel-spectrum embedding. We then utilize a hinge loss [Lim and Ye, 2017] as the adversarial objective function:

LGadv(ˆxt

))] (7)

,φ) = −E[

Dφ,k(Fk(ˆxt

n

n

k

|[Figure 8]|
|---|
|[Figure 9]|

|[Figure 10]|
|---|
|[Figure 11]|

- Figure 2: Comparison of long music generation through independent paths vs. shared restricted paths. Input text prompt: Bright, cheerful and groovy piano, classical. As expected, there is no coherency between clips in independent; Starting from the same noise, our shared restriction process steers these initial diffusion paths into consistent and high quality music clips.

whereas the discriminator is trained to minimize as:

LDadv(ˆxt

;φ)

n

= E[

k

max(0,1 − Dφ,k(Fk(x0))) + γR1(φ)]

(8)

+ E[

max(0,1 + Dφ,k(Fk(ˆxt

)))]

n

k

Where R1 denotes the R1 gradient penalty [Mescheder et al., 2018]. Rather than computing the gradient penalty with respect to the pixel values, we compute it on the input of each discriminator head Dψ,k. Finally, based on the above analysis, the overall objective is:

L = LGadv + λLdistil (9) where λ is the balancing factor.

#### 3.3 Shared Restricted Process for Long Music

In light of the memory constraints inherent in the music diffusion model when generating prolonged and coherent musical sequences, we introduce a parallel restriction process, as outlined in [Bar-Tal et al., 2023]. Specifically, at each denoising generation step, we amalgamate the denoising directions, supplied by the reference MusicCM model, from all the crops, and strive to follow them all as closely as possible, constrained by the fact that nearby crops share common value in mel-spectrogram. Intuitively, we encourage each crop to be a real sample from the reference MusicCM model. It is crucial to note that despite potential variations in denoising directions among individual crops, our framework yields a cohesive denoising step, resulting in the production of highquality and seamlessly connected long-form music.

Algorithm 1 Shared restricted generation for long music Input MusicCM model Ψ, resolution of the desired music H′,W′, mappings defining crops from the music {gi}ni=1, conditioned text c;

- 1: ▷ Noise initialization
- 2: XTl ∼ N(0,I),XTl ∈ RH

′×W′

- 3: for t = T,..., 1 do
- 4: ▷ Take crops from mel-spectrograms
- 5: Xti,s ← gi(Xtl)
- 6: ▷ Per-crop diffusion updates
- 7: Xti,s−1 ← Ψ(Xti,s,z)
- 8: Xtl−1 ← π(Xtl,z)
- 9: end for

Formally, define a long mel-spectrograms space Xl with H′ ≥ H, W′ ≥ H directly from a trained consistency music model Ψ working in space Xs. Let gi(Xs) ∈ Xs is an i-th H × W crop of image Xl, and z serve as condition space mapping from corresponding set. We consider n such crops that cover the original images Xl and get as:

n

||gi(Xl) − Ψ(gi(Xl), z)||2 (10)

π(Xtl, z) = argminXl∈Xl

i=1

that is a least-squares problem, the solution of which can be calculated analytically. Mapping function gi is defined as fixed-size crops from the full mel-spectrogram. The entire process is listed in Algorithm 1. Our maps gi,...,gn provide crops with a sliding window of size step in the latent space. In particular, n = H

′−H

′−W

step . Note that we can compute the per-crop diffusion updates in parallel (i.e., in a batch),

step · W

- Table 1: The evaluation of generation quality among MusicLDMs and other text-to-music baselines. All experiments are performed on a single NVIDIA A100 GPU. The inference overhead of generating 10s music clips at a time is reported as average for each music clip.

Model Step FDpann ↓ FDvgg ↓ Inception Score ↑ KL Div. ↓ Latency Riffusion [Forsgren and Martiros, 2022] 50 68.95 10.77 1.34 5.00 MuBERT [MubertAI, ] - 31.70 19.04 1.51 4.69 AudioLDM [Liu et al., 2023b] 50 38.92 3.08 1.67 3.65 MusicLDM [Chen et al., 2023] 50 26.67 2.40 1.81 3.80 2.19s MusicCM 4 27.13 2.48 1.78 3.88 0.37s MusicCM 1 32.12 3.31 1.60 4.12 0.23s

resulting in a total of T·n

b calls to the reference MusicCM Ψ, where b denotes the batch size. Note that while each denoising steps Ψ(gi(Xl)) may pull to a different direction, our process fuses these inconsistent directions into a global denoising step Ψ(xl), resulting in a high-quality seamless music clips.

Taking a more intuitive step, we illustrate the property in Figure 2, where we consider a long music of H × 8W. We also show results when independently applying Ψ on eight non-overlapping crops. As expected, there is no coherency between crops since this amounts to eight random samples. Starting from the same initial noise, our generation process, allows us to fuse these initially-unrelated diffusion paths, and steer the generation into high-quality, coherent music.

### 4 Experiments

In this section, we commence by elucidating the particulars of the experimental configurations. Subsequently, an assessment of MusicCM ensues, focusing on text-music relevance, novelty, and generation temporal efficiency through established metrics. Lastly, a subjective listening test is undertaken to provide a supplementary evaluation.

#### 4.1 Experimental Setup

Datasets. Our music consistency model undergoes training utilizing two extensively employed datasets, namely Audiostock [Chen et al., 2023] and MagnaTagATune [Bogdanov et al., 2019], in conjunction with the frozen VAE and Hifi-GAN. The Audiostock dataset encompasses 9,000 music tracks for the training phase, contributing to a cumulative duration of 455.6 hours. Each track within this dataset is accompanied by an accurate textual description. To augment the diversity of musical styles and facilitate emotional perception, an additional 5,000 music clips are incorporated, with an average sampling approach from MagnaTagATune based on the statistical analysis of the top 50 tags.

Evaluation metrics. Referring from evaluation methodologies established in prior research on audio generation [Liu et al., 2023b], our study employs frechet distance (FD), inception score (IS), and kullback-leibler (KL) divergence to assess the quality of generated musical audio outputs. (i) FD metric gauges audio quality by leveraging an audio embedding model to quantify the similarity between the embedding spaces of the generated and target audio. In this investigation, we employ two standard audio embedding models: VGGish

[Hershey et al., 2017] and PANN [Kong et al., 2020b], denoting the resulting distance as FDvgg and FDpann, respectively. (ii) IS serves to measure the diversity and the quality of the full set of audio outputs. (iii) KL divergence is measured on individual pairs of generated and ground-truth audio samples and averaged. We use the audioldm eval library1 to execute the evaluation of all aforementioned metrics. The comparative assessment involves juxtaposing the ground truth audio sourced from the Audiostock 1000-track test set with the 1000 tracks of music generated by each system, grounded in their corresponding textual descriptions.

Implementation details. For convenience, we opt to directly utilize the existing pre-trained music diffusion models MusicLDM2 as the teacher model while maintaining fixed network parameters throughout the consistency distillation process. The model structure of the consistency model is the same as the teacher diffusion model and is initialized with the teacher’s model parameters. AdamW optimizer with a learning rate of 1e-5 is adopted to train MusicCM. The EMA rate used in our experiments is 0.95. We trained all MusicLDM modules with music clips of 10.24 seconds at a 16khz sampling rate. In both the VAE and diffusion model, music clips are represented as mel-spectrograms with T = 1024 frames and F = 128 mel-bins. VAE utilizes a downsampling rate of P = 8 and a latent dimension of C = 16. The architecture of MusicCM’s latent diffusion model follows that of MusicLDM. We use the ViT-B AudioMAE from repository3 as a discriminator feature network by default. The distillation process incorporates a weighting factor of λ = 3 consistently across all experiments according to prior experiments.

#### 4.2 Time Efficiency

We quantify the inference time entailed in the process of textto-music synthesis using our proposed MusicCM, conducting a comparative analysis against both the baseline method and other open-source models. The outcomes of this comparative evaluation are delineated in Table 1. The findings reveal a noteworthy efficiency advantage for our proposed approach, as it necessitates a mere 1-4 steps for inference, rendering it considerably swifter compared to the baseline method requiring 50 DDIM steps. It is imperative to emphasize that the inference cost encompasses not only iterative denoising but also other components such as text feature encoding and latent

- 1https://github.com/haoheliu/audioldm eval

- 2https://github.com/RetroCirce/MusicLDM
- 3https://github.com/facebookresearch/AudioMAE

##### Table 2: Evaluation results for measuring the text-audio relevance and novelty (plagiarism). The results of MusicCM are calculated for four inference steps.

Relevance Novelty and Plagiarism Risk TA Similarity↑ SIMAA@90 ↓ SIMAA@95 ↓

Test Set 0.325 — Retrieval Max 0.423 — —

MuBERT 0.131 0.107 0 MusicLDM 0.281 0.430 0.047 MusiCM 0.275 0.330 0.021

code decoding. More encouragingly, our method’s efficacy is further underscored by its capacity to synthesize longer music sequences with diminished memory consumption and temporal overhead. To illustrate, on a single NVIDIA A100 GPU, our approach can generate a one-minute music clip in merely 1.2 seconds. This comparative analysis underscores the pronounced efficiency of our proposed approach.

#### 4.3 Quantitative Comparison to State-of-the-Art

Generation quality. We present the FD, IS, and KL results in comparison with baseline models, as summarized in Table 1. The results of Riffusion and MuBERT are reported from the paper [Chen et al., 2023]. Upon comparative analysis, our findings indicate that MusicCM (1-step) achieves a level of performance on par with Riffusion and MuBERT. This parity is attributed to our contributions in optimizing distillation datasets and employing a robust baseline model. Furthermore, MusicCM (4-steps) exhibits superior performance compared to the benchmark AudioLDM across nearly all metrics, with only marginal performance degradation compared to MuiscLDM. We attribute these advantages to the efficacy of score distillation incorporated in our methodology.

Text-audio relevance, novelty, and plagiarism. We also conduct an evaluation using two metric groups, as stipulated by [Chen et al., 2023]. The assessment aims to gauge textaudio relevance, novelty, and potential plagiarism risks inherent in diverse models. (i) Text-audio similarity quantifies the relevance between the text and the audio in a common embedding space. (ii) To ascertain the degree to which models exhibit direct replication of samples from the training set, we initially compute the dot products between the audio embedding of each generated music output and all audio embeddings within the Audiostock training set. Subsequently, we identify the maximum similarity among the nearest neighbors in the training set. The nearest-neighbor audio similarity ratio is then calculated as the fraction of generated outputs for which the similarity of the nearest-neighbors exceeds a specified threshold. This ratio is denoted as SIMAA@90 when the threshold is set to 0.9 and SIMAA@95 with a threshold of 0.95. Table 2 delineates the mean values of text-audio similarity and nearest-neighbor audio similarity ratios, employing two distinct thresholds, for the 1000 tracks within the Audiostock test set and the generated music outputs from MuBERT, MusicLDM, and MusicCM. Two reference points are incorporated for the assessment of text-audio similarity: “Test Set” and “Retrieval Max”. The outcomes reveal that the distilled

- Table 3: Ablation study. We report FID, Is, KL, and text-audio similarity metrics in different settings. The results are calculated for four inference steps.

Discriminator FDpann ↓ IS ↑ KL ↓ TA similarity ↑

MusicCM 27.13 1.78 3.88 0.275 w.o. Ladv 28.65 1.73 3.95 0.271 w. CLAP 27.55 1.74 3.92 0.295

- Table 4: Subjective listening test to evaluate different aspects including quality, relevance, and musicality for model outputs.

##### Time Quality ↑ Relevance ↑ Musicality ↑

MusicLDM 10s 0.341 0.283 0.541 MusicCM 10s 0.358 0.267 0.517 MusicLDM 60s 0.292 0.275 0.516 MusicCM 60s 0.350 0.291 0.525

MusicCM exhibits a text-music alignment capability akin to the baseline while notably enhancing novelty and mitigating the risk of plagiarism.

#### 4.4 Model Analysis

We conduct an ablation study on several choices in Table 3. We will discuss each part in the following.

Adversarial loss term. In accordance with the findings presented in line 2, it becomes evident that adversarial losses play a crucial role. While the distillation loss in isolation demonstrates a certain level of effectiveness, its amalgamation with the adversarial loss yields a discernible enhancement in the obtained results. Notably, in our preceding human testing endeavors, a more intuitive comprehension is achieved. Particularly, when generating content with elaborate and extensive prompts, the incorporation of adversarial loss yields a perceptibly clearer and higher-quality output of musical clips.

Discriminator feature networks. Recent insights suggest that ViT trained with CLIP or MAE exhibits particular aptitude in assessing the performance of generative models. Concurrently, these models also demonstrate efficacy when employed as discriminator feature networks. As delineated in line 3, we subsequently substitute AudioMAE with CLAP, resulting in an optimization in the aspect of text-audio similarity but a concomitant loss in preserving essential audio characteristics. Upon holistic consideration of the overall outcomes, AudioMAE emerges as the preferred choice. Nonetheless, it is acknowledged that the pursuit of more advanced feature networks for music perception remains a direction for future exploration.

#### 4.5 Subjective Listening Test

We additionally undertake a subjective listening test involving MusicLDM and MusicCM (4-steps) across varying durations of music clips to conduct a more nuanced assessment of the perceived auditory experience elicited by the generated music, as illustrated in 4. In this specific evaluation, we enlist the participation of 8 subjects who are instructed

pop rock, happy, upbeat mood, with a simple, repetitive drumbeat

techno music with a strong, upbeat tempo and high melodic riffs

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

50-step (Baseline)

|[Figure 16]|
|---|

|[Figure 17]|
|---|

|[Figure 18]|
|---|

|[Figure 19]|
|---|

- 1-step
- 2-step

|[Figure 20]|
|---|

|[Figure 21]|
|---|

|[Figure 22]|
|---|

|[Figure 23]|
|---|

|[Figure 24]|
|---|

|[Figure 25]|
|---|

|[Figure 26]|
|---|

|[Figure 27]|
|---|

4-step

|[Figure 28]|
|---|

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

8-step

- Figure 3: Qualitative visualization results under different inference steps. Larger steps generally yield better quality and time continuity of music. Importantly, our MusicCM can produce plausible results with fewer steps or even only one step.

to listen to two distinct groups of generations randomly selected from the test set. Subsequently, they are tasked with determining whether the music is generated by the model or created by a human. Each group comprises 15 generations from both models, accompanied by their respective text descriptions. The subjects are then prompted to rate the music based on three criteria: (i) Quality, pertaining to the sound quality of the generated samples; (ii) Relevance, assessing the alignment of the music with the provided text; and (iii) Musicality, gauging the overall aesthetic appeal of the music. Our observations indicate the following: (i) For 10-second music generation, the samples generated by the MusicCM strategy exhibit superior music quality compared to those of the original MusicLDM, while remaining comparable in terms of relevance, thereby reinforcing the earlier analysis; (ii) In the generation of longer music sequences, our MusicCM approach demonstrates a significant superiority in terms of music quality. This enhancement is attributed to the incorporation of a shared restricted diffusion process.

#### 4.6 Qualitative Results

In addition to our quantitative analyses outlined above, we present qualitative results pertaining to varying inference steps in Figure 3. The findings elucidate that when the sampling step is restricted, as exemplified by step = 1, the generated mel-spectrogram exhibits perceptible blurriness, manifesting inaccuracies in detailing and an inability to preserve the temporal structure of objects. With an increment in the number of iteration steps, there is a discernible enhancement in the quality of the mel-spectrograms, concomitant with an improved preservation of temporal-frequency structures. For instance, with 4-6 steps, outcomes comparable to DDIM’s 50 steps can be attained, thereby substantively reducing sampling steps and concurrently enhancing generation speed.

The outcomes obtained in the domain of text-to-music generation underscore the efficacy of the proposed MusicCM, demonstrating a commendable equilibrium between quality and speed.

### 5 Limitations

In this section, we outline the recognized limitations of our study, serving as a roadmap for future improvements. Firstly, our MusicCM relies on a strong teacher model as the distillation target. It is hard to combine teachers from different domains or different music sampling ratios. Secondly, the consistency distillation process requires fine-tuning the model. While consistency distillation only requires a small number of training steps, it may lead to unsatisfactory results when the training data for the teacher model is unavailable. Finally, we also intend to investigate the optimization of different discriminator feature networks during adversarial training.

### 6 Conclusion

This study introduces music consistency models, presenting a universal approach for distilling a pre-trained diffusion model into a swift, few-step music generation model. The amalgamation of an adversarial and a distillation objective is employed for the distillation of robust baseline models. Empirical results substantiate the efficacy of our approach, illustrating the attainment of high-fidelity music synthesis within merely four steps. This outcome underscores its potential for real-time synthesis, a notable advancement over preceding methods necessitating approximately 50 DDIM steps. Our model facilitates the generation of high-quality music in a single step, thereby opening new avenues for real-time generation utilizing foundational models.

### References

[Agostinelli et al., 2023] Andrea Agostinelli, Timo I Denk, Zal´an Borsos, Jesse Engel, Mauro Verzetti, Antoine Caillon, Qingqing Huang, Aren Jansen, Adam Roberts, Marco Tagliasacchi, et al. MusicLM: Generating music from text. arXiv preprint:2301.11325, 2023.

[Bar-Tal et al., 2023] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. 2023.

[Bogdanov et al., 2019] Dmitry Bogdanov, Minz Won, Philip Tovstogan, Alastair Porter, and Xavier Serra. The mtg-jamendo dataset for automatic music tagging. In Machine Learning for Music Discovery Workshop, International Conference on Machine Learning (ICML 2019), Long Beach, CA, United States, 2019.

[Chen et al., 2023] Ke Chen, Yusong Wu, Haohe Liu, Marianna Nezhurina, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. Musicldm: Enhancing novelty in text-to-music generation using beat-synchronous mixup strategies. arXiv preprint arXiv:2308.01546, 2023.

- [Fei et al., 2022a] Zhengcong Fei, Mingyuan Fan, Li Zhu, and Junshi Huang. Progressive text-to-image generation. arXiv preprint arXiv:2210.02291, 2022.
- [Fei et al., 2022b] Zhengcong Fei, Xu Yan, Shuhui Wang, and Qi Tian. Deecap: Dynamic early exiting for efficient image captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12216–12226, 2022.

- [Fei et al., 2023a] Zhengcong Fei, Mingyuan Fan, and Junshi Huang. A-jepa: Joint-embedding predictive architecture can listen. arXiv preprint arXiv:2311.15830, 2023.
- [Fei et al., 2023b] Zhengcong Fei, Mingyuan Fan, and Junshi Huang. Gradient-free textual inversion. In Proceedings of the 31st ACM International Conference on Multimedia, pages 1364–1373, 2023.

- [Fei et al., 2024a] Zhengcong Fei, Mingyuan Fan, Changqian Yu, and Junshi Huang. Scalable diffusion models with state space backbone. arXiv preprint arXiv:2402.05608, 2024.
- [Fei et al., 2024b] Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, and Junshi Huang. Diffusionrwkv: Scaling rwkv-like architectures for diffusion models. arXiv preprint arXiv:2404.04478, 2024.

- [Fei, 2019] Zheng-cong Fei. Fast image caption generation with position alignment. arXiv preprint arXiv:1912.06365, 2019.
- [Fei, 2020] Zhengcong Fei. Actor-critic sequence generation for relative difference captioning. In Proceedings of the 2020 International Conference on Multimedia Retrieval, pages 100–107, 2020.
- [Fei, 2021a] Zhengcong Fei. Memory-augmented image captioning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 1317–1324, 2021.

[Fei, 2021b] Zhengcong Fei. Partially non-autoregressive image captioning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 1309–1316, 2021.

[Forsgren and Martiros, 2022] Seth Forsgren and Hayk Martiros. Riffusion - Stable diffusion for real-time music generation. 2022.

[Gafni et al., 2022] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Makea-scene: Scene-based text-to-image generation with human priors. In European Conference on Computer Vision, pages 89–106. Springer, 2022.

[Goodfellow et al., 2014] Ian Goodfellow, Jean PougetAbadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.

[Hershey et al., 2017] Shawn Hershey, Sourish Chaudhuri, Daniel PW Ellis, Jort F Gemmeke, Aren Jansen, R Channing Moore, Manoj Plakal, Devin Platt, Rif A Saurous, Bryan Seybold, et al. Cnn architectures for large-scale audio classification. pages 131–135. IEEE, 2017.

[Ho and Salimans, 2022] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

- [Huang et al., 2023a] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. arXiv preprint arXiv:2302.09778, 2023.
- [Huang et al., 2023b] Qingqing Huang, Daniel S Park, Tao Wang, Timo I Denk, Andy Ly, Nanxin Chen, Zhengdong Zhang, Zhishuai Zhang, Jiahui Yu, Christian Frank, et al. Noise2music: Text-conditioned music generation with diffusion models. arXiv preprint:2302.03917, 2023.
- [Huang et al., 2023c] Rongjie Huang, Jiawei Huang, Dongchao Yang, Yi Ren, Luping Liu, Mingze Li, Zhenhui Ye, Jinglin Liu, Xiang Yin, and Zhou Zhao. Make-anaudio: Text-to-audio generation with prompt-enhanced diffusion models. arXiv preprint arXiv:2301.12661, 2023.
- [Huang et al., 2023d] Rongjie Huang, Jiawei Huang, Dongchao Yang, Yi Ren, Luping Liu, Mingze Li, Zhenhui Ye, Jinglin Liu, Xiang Yin, and Zhou Zhao. Make-anaudio: Text-to-audio generation with prompt-enhanced diffusion models. arXiv preprint:2301.12661, 2023.

[Kang et al., 2023] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10124– 10134, 2023.

[Kawar et al., 2023] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023.

- [Kong et al., 2020a] Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae. HifiGAN: Generative adversarial networks for efficient and high fidelity speech synthesis. 33:17022– 17033, 2020.
- [Kong et al., 2020b] Qiuqiang Kong, Yin Cao, and Turab Iqbal et al. Panns: Large-scale pretrained audio neural networks for audio pattern recognition. 2020.

[Kreuk et al., 2022] Felix Kreuk, Gabriel Synnaeve, Adam Polyak, Uriel Singer, Alexandre D´efossez, Jade Copet, Devi Parikh, Yaniv Taigman, and Yossi Adi. AudioGen: Textually guided audio generation. 2022.

[Li et al., 2023] Peike Li, Boyu Chen, Yao Yao, Yikai Wang, Allen Wang, and Alex Wang. Jen-1: Text-guided universal music generation with omnidirectional diffusion models. arXiv preprint arXiv:2308.04729, 2023.

[Lim and Ye, 2017] Jae Hyun Lim and Jong Chul Ye. Geometric gan. arXiv preprint arXiv:1705.02894, 2017.

- [Liu et al., 2023a] Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, and Mark D Plumbley. Audioldm: Text-to-audio generation with latent diffusion models. arXiv preprint arXiv:2301.12503, 2023.
- [Liu et al., 2023b] Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, and Mark D Plumbley. AudioLDM: Text-to-audio generation with latent diffusion models. 2023.
- [Liu et al., 2023c] Haohe Liu, Qiao Tian, Yi Yuan, Xubo Liu, Xinhao Mei, Qiuqiang Kong, Yuping Wang, Wenwu Wang, Yuxuan Wang, and Mark D Plumbley. Audioldm 2: Learning holistic audio generation with self-supervised pretraining. arXiv preprint arXiv:2308.05734, 2023.

- [Luo et al., 2023a] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.
- [Luo et al., 2023b] Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolin´ario Passos, Longbo Huang, Jian Li, and Hang Zhao. Lcm-lora: A universal stable-diffusion acceleration module. arXiv preprint arXiv:2311.05556, 2023.

[Melechovsky et al., 2023] Jan Melechovsky, Zixun Guo, Deepanway Ghosal, Navonil Majumder, Dorien Herremans, and Soujanya Poria. Mustango: Toward controllable text-to-music generation. arXiv preprint arXiv:2311.08355, 2023.

[Mescheder et al., 2018] Lars Mescheder, Andreas Geiger, and Sebastian Nowozin. Which training methods for gans do actually converge? In International conference on machine learning, pages 3481–3490. PMLR, 2018.

[Mou et al., 2023] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023.

[MubertAI, ] MubertAI. Mubert: A simple notebook demonstrating prompt-based music generation.

[Nichol et al., 2021] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.

[Ramesh et al., 2022] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

- [Rombach et al., 2022a] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [Rombach et al., 2022b] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. pages 10684–10695, 2022.

[Ruiz et al., 2023] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500–22510, 2023.

- [Sauer et al., 2023a] Axel Sauer, Tero Karras, Samuli Laine, Andreas Geiger, and Timo Aila. Stylegan-t: Unlocking the power of gans for fast large-scale text-to-image synthesis. arXiv preprint arXiv:2301.09515, 2023.
- [Sauer et al., 2023b] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042, 2023.

[Schneider et al., 2023] Flavio Schneider, Zhijing Jin, and Bernhard Sch¨olkopf. Mo\ˆ usai: Text-to-music generation with long-context latent diffusion. arXiv preprint arXiv:2301.11757, 2023.

[Schuhmann et al., 2022] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022.

[Shen and Zhou, 2021] Yujun Shen and Bolei Zhou. Closedform factorization of latent semantics in gans. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1532–1540, 2021.

[Song and Dhariwal, 2023] Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. arXiv preprint arXiv:2310.14189, 2023.

[Song et al., 2020] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

[Song et al., 2023] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023.

- [Wang et al., 2023a] Xiang Wang, Shiwei Zhang, Han Zhang, Yu Liu, Yingya Zhang, Changxin Gao, and Nong Sang. Videolcm: Video latent consistency model. arXiv preprint arXiv:2312.09109, 2023.
- [Wang et al., 2023b] Yinhuai Wang, Jiwen Yu, Runyi Yu, and Jian Zhang. Unlimited-size diffusion restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1160–1167, 2023.

[Wu et al., 2023] Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. pages 1–5. IEEE, 2023.

[Xiao et al., 2023] Jie Xiao, Kai Zhu, Han Zhang, Zhiheng Liu, Yujun Shen, Yu Liu, Xueyang Fu, and Zheng-Jun Zha. Ccm: Adding conditional controls to text-to-image consistency models. arXiv preprint arXiv:2312.06971, 2023.

[Yan et al., 2021] Xu Yan, Zhengcong Fei, Zekang Li, Shuhui Wang, Qingming Huang, and Qi Tian. Semiautoregressive image captioning. In Proceedings of the 29th ACM International Conference on Multimedia, pages 2708–2716, 2021.

- [Yang et al., 2023a] Dongchao Yang, Jianwei Yu, Helin Wang, Wen Wang, Chao Weng, Yuexian Zou, and Dong Yu. DiffSound: Discrete diffusion model for text-to-sound generation. 2023.
- [Yang et al., 2023b] Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. Diffusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 56(4):1–39, 2023.

[Zhang et al., 2023] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.

[Zhu et al., 2023] Yueyue Zhu, Jared Baca, Banafsheh Rekabdar, and Reza Rawassizadeh. A survey of ai music generation tools and models. arXiv preprint arXiv:2308.12982, 2023.

