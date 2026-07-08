OneFlow: Concurrent Mixed-Modal and Interleaved Generation with Edit Flows

John Nguyen1, Marton Havasi1, Tariq Berrada1,2, Luke Zettlemoyer1, Ricky T. Q. Chen1 1FAIR at Meta, 2Univ. Grenoble Alpes, Inria, CNRS, Grenoble INP, LJK, France

We present OneFlow, the first non-autoregressive multimodal model that enables variable-length and concurrent mixed-modal generation. Unlike autoregressive models that enforce rigid causal ordering between text and image generation, OneFlow combines an insertion-based Edit Flow for discrete text tokens with Flow Matching for image latents. OneFlow enables concurrent text-image synthesis with hierarchical sampling that prioritizes content over grammar. Through controlled experiments across model sizes from 1B to 8B, we demonstrate that OneFlow outperforms autoregressive baselines on both generation and understanding tasks while using up to 50% fewer training FLOPs. OneFlow surpasses both autoregressive and diffusion-based approaches while unlocking new capabilities for concurrent generation, iterative refinement, and natural reasoning-like generation.

# arXiv:2510.03506v3[cs.AI]10Dec2025

Website: https://oneflow.framer.ai/

Prompt tokens Generated tokens Generated images

- t=0

Show me cute cats Here is a goofy cat and a sleepy one

|[Figure 1]<br><br>[Figure 2]|
|---|

|[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]|
|---|

[Figure 6]

- t=1

- t=2

Show me cute cats

Show me cute cats

sleepy

|[Figure 7]|
|---|

|[Figure 8]|
|---|

Show me cute cats

sleepy one

[Figure 9]

|[Figure 10]<br><br>[Figure 11]|
|---|

|[Figure 12]<br><br>[Figure 13]|
|---|

Show me cute cats

sleepy one

a

[Figure 14]

|[Figure 15]<br><br>[Figure 16]|
|---|

|[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]|
|---|

sleepy one

Show me cute cats

Here

cat

and

a

[Figure 20]

|[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]|
|---|

|[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]|
|---|

a

Show me cute cats Here is

goofy cat

and

a sleepy one

Figure 1 OneFlow is a variable-length non-autoregressive model that can concurrently generate interleaved text and variable number of images using insertions as a primitive operation.

1 Introduction

Native Multimodal Models — models capable of handling both multimodal understanding and generation within a single backbone — have advanced considerably in visual understanding and generation. These models typically employ a unified transformer architecture with next-token prediction to handle both discrete and continuous generation (Team, 2024; Wu et al., 2025a; Ma et al., 2025a; Deng et al., 2025b; Zhou et al., 2025). Recent work like Transfusion (Zhou et al., 2025) and Bagel (Deng et al., 2025b) demonstrates that leveraging modality-specific training objectives within shared architectures can significantly improve performance, particularly on continuous modalities such as vision.

However, both autoregressive (AR) and diffusion-based multimodal approaches face fundamental architectural constraints. Autoregressive models can handle interleaved data but require strict sequential generation each image must be fully completed before text generation can continue, preventing simultaneous cross-

|[Figure 28]|[Figure 29]|[Figure 30]|[Figure 31]|[Figure 32]|
|---|---|---|---|---|
| | |[Figure 33]|[Figure 34]|[Figure 35]|
|[Figure 36]|[Figure 37]| | | |
| | |[Figure 38]|[Figure 39]|[Figure 40]|

###### Figure 2 Text-to-image generation. Generated images at 512×512 resolution from OneFlow. Prompts are in Figure 14.

[Figure 41]

###### Figure 3 Visual question answering. OneFlow generates by simply inserting tokens based on confidence, resulting in a natural hierarchical sampling and implicit reasoning where the most difficult answer tokens are generated later.

Generated tokens: t = 0 t = 1

[Figure 42]

I want to paint flowers but I need a photo for inspiration. Can you show me two pictures of flowers and describe them?

In this first image, a beautiful arrangement of pink flowers is on display, with a total of six flowers spread out in a visually appealing manner.<|image|>The roses are delicate and have a soft, pink hue, which adds to the overall charm of the scene. The bouquet seems to be the main focus of the image, with the flowers captured in a close-up shot, showcasing their intricate details. The display creates an elegant and serene atmosphere, evoking a sense of romance and appreciation for the beauty of nature. This image features a close-up view of a beautiful pink flower sitting in a glass vase on a table.<|image|>The flower appears to be a gerbera, with a vibrant pink color and a prominent center, possibly a bud. The glass vase is positioned at the base of the flower, enhancing the visual appeal of the scene. The table beneath the vase provides a simple yet elegant backdrop, allowing the flower to be the focal point of the image.

[Figure 43]

###### Figure 4 Concurrent interleaved text & image generation. OneFlow can insert variable number of images in the generated sequence, which are concurrently denoised alongside the text. This allows the text and images to depend on each other during the generation process.

modal refinement. Conversely, diffusion-based multimodal models such as MMaDA (Yang et al., 2025), FUDOKI (Wang et al., 2025), and Unidisc (Swerdlow et al., 2025) enable simultaneous mixed-modal generation but only for predetermined single text-image pairs where modality assignments must be known a priori and rely on independent time schedules for each modality. Neither paradigm supports the simultaneous generation of variable-length interleaved sequences.

We present OneFlow, the first model to achieve simultaneous generation of interleaved data. Unlike autoregressive models that enforce sequential completion of each modality, and unlike diffusion models restricted to fixed length generation, OneFlow combines an insertion-based discrete text generation using Edit Flows with Flow Matching for image generation. This enables concurrent refinement of both text and images with per-image time schedules, using a novel interleaved time schedule.

Through controlled experiments across various model sizes and compute regimes, we demonstrate that OneFlow outperforms both autoregressive (AR) and diffusion baselines on generation and understanding tasks while requiring 50% fewer training FLOPS. Moreover, we find that concurrent mixed modal pretraining yields 4% relative improvement on VQA and 1.5% on image generation over sequential pretraining. We summarize our contributions below.

Contributions:

- 1. We introduce OneFlow, a non-autoregressive multimodal model that unifies image and text generation under a simultanous Edit Flow and Flow Matching framework.
- 2. OneFlow enables new capabilities such as concurrent mixed-modal generation, which helps boost performance over uni-modal generation on a wide range of benchmarks.
- 3. Through controlled experiences from 1B to 8B, we find that OneFlow scales better than autoregressive multimodal models, more so with mixed-modal training.
- 4. OneFlow outperforms or is competitive with existing AR and diffusion-based models, on a diverse range of image generation and image understanding benchmarks.

### 2 OneFlow: Mixed-Modal Generation through Flow Matching

OneFlow handles multiple modalities through a sequence model, where elements in the sequence can be discrete tokens or continuous embeddings, e.g., of images. Concretely, let T denote the space of a single element of the sequence, which can take either a discrete value, up to some fixed vocabulary size M, or a continuous value, i.e., T = [M] ∪ R. Then our state space is defined as the set of all possible sequences up to

some maximum length N, i.e., X = Nn=0 T n.

During generation, our model transforms noisy sequences into clean sequences. We do this by combining discrete-valued and continuous-valued generative processes. Specifically, we make use of the Edit Flows (Havasi et al., 2025) framework which enables variable-length sequence generation through the use of edit operations. It starts with a noisy sequence and iteratively applies edits until it is denoised into a generation. We focus on the insertion capabilities of Edit Flows, which is conceptually simple yet extremely powerful, as it allows inserting arbitrary number of tokens—and images—into the generated sequence. When images are inserted, we initialize them with noise and then use Flow Matching (Lipman et al., 2024) to generate the image. Since the same model predicts both the text edits and the image denoising, OneFlow achieves variable-length, non-autoregressive joint image and text generation. In the following, we state equations with only intuitive justifications and explanations. Full mathematical details and derivations can be found in Appendix D.

- 2.1 Discrete Text Generation via Edit Flows

Edit Flows uses a continuous-time Markov chain (CTMC) to iteratively refine variable-length discrete sequences. We start with an empty sequence X0 = ∅ at time t = 0, and transform the sequence through insertion operations. Let ins(x,i,a), x ∈ X,i ∈ {1,...,n},a ∈ T , be the sequence resulting from inserting the token value a to the right of position i of the sequence x, resulting in

ins(x,i,a) = (x1,...,xi,a,xi+1,...,xn). (1)

This forms the primitive operation that we use during generation.

During training, we take a data sequence X1 and randomly delete tokens with equal probability to obtain Xt. This defines the process X[0,1] that we will fit to. The probability of each token being deleted is set by a monotonic scheduler κt with κ0 = 0,κ1 = 1.

P(xi in Xt) = κt, for each xi ∈ X1. (2)

In preliminary experiments, we tested different κt but found that the linear schedule κt = t works most consistently across our diverse benchmarks. Deleted tokens are removed from the sequence. Noting that on average we retain Et[κt] fraction of the original tokens, with the linear schedule we retain 50% of the data sequence. This can lead to significant FLOPS savings during training, and tuning the scheduler can save even more if desired.

- 2.1.1 Parameterization

The parameterization of an Edit Flow model for insertions naturally decomposes into two predictions: (i) how many tokens are missing at the right of position i, and (ii) which tokens are missing. Thus, at each position i of the sequence, our model outputs two quantities

- • λi : X → R+ is a scalar that predicts the number of missing tokens between i and i + 1.
- • Qi : X → [M] is a normalized distribution that predicts what tokens are missing.

These two predictions form the CTMC rate, so for sampling we would use (up to o(h) error), P(Xt+h = ins(Xt,i,a) | Xt) = h κ˙

t

1−κt λi(Xt)Qi(a|Xt). (3) The ratio κ˙

t

1−κt dictates the distribution of insertion times according to the schedule κt imposed during training

(2), where κ˙t = dκ

t

dt . Note that unlike prior work (Havasi et al., 2025), we factor out this ratio κ˙

t

1−κt from the rate predictions and use a simplified model that is independent of t. In our practical implementation, we do not feed time values into the network for predicting insertions. While not theoretically justified, we found this t-independence assumption to work better in practice, likely because Xt already contains sufficient information for predicting the insertions.

- 2.1.2 Insertion prediction λi

The main component that determines whether insertions occur is the prediction head λi, which is trained by regressing onto the number of missing tokens. Each position i of the noisy sequence Xt has a corresponding number of missing tokens ki, which is the number of deleted tokens between Xti and Xti+1. The original Edit Flows loss was constructed through a choice of Bregman divergence (Holderrieth et al., 2024) which results in

ℓPoisson(λi) = i λi(Xt) − ki log λi(Xt). (4)

Alternatively, (4) can be interpreted as the negative log-likelihood of a Poisson distribution, so λi is trained to fit a Poisson distribution to model missing token counts ki. However, the distribution of k has a very high concentration for ki = 0. Furthermore, during sampling, the key prediction is whether the missing token count is either zero or nonzero. As such, we explicitly model the probability of inserting zero tokens.

P(k = 0) = π, and P(k) = (1 − π)Pois(k;λnonzero | k > 0) for k > 0 (5)

where π ∈ (0,1) is the probability of observing zero, and λnonzero > 0 is the rate parameter but restricted to only modeling the distribution of nonzero counts. We train π by using a binary cross entropy (BCE) loss to detect if the missing count is zero, and we train λnonzero using the original loss ((4)) on nonzero counts. For sampling (3), we can use the expectation λi(Xt) = (1 − πi(Xt))λinonzero(Xt). However, we found that a consistently better sampling strategy is to first sample whether there are zero insertions using π, then simply use the rate λinonzero(Xt) if there are nonzero insertions. We provide sampling pseudocode in Algorithm 2.

- 2.1.3 Bag-of-tokens prediction Qi

To determine what token to insert at each position, we make use of the output head Q which is a softmax over the discrete vocabulary [M]. We use the same Edit Flows loss, which is a sum of cross-entropy loss. Let Ai denote the set of deleted tokens between Xti and Xti+1, then for each position i, the loss is

ℓtokens(Qi) = − a∈Ai log Qi(a|Xt). (6)

- 2.1.4 Combined loss

###### At each training iteration, we randomly delete tokens from the data sequence, and learn to predict the set of missing tokens at each position, resulting in the total insertion loss:

Ltext = Et,X

t|X1

n

1 n

ℓtokens(Qi) + ℓPoisson(λinonzero)1[k

i>0] + ℓBCE(πi) (7)

i=1

where n is the length of the noisy sequence Xt. Note that this differs from the original training objective in Edit Flows (Havasi et al., 2025) which additionally weights the loss by the factor κ˙

1−κt , not affecting the optimal solution. We found that not using this factor produces better results.

t

- 2.2 Continuous Image Generation via Flow Matching

Following standard practice, we generate images starting from a Gaussian noise of fixed dimension Nimg, applying a deterministic generation procedure that follows an ordinary differential equation. Let Yt ∈ RN

img

denote the noisy image, then the generative process is

d dtYt = v(Yt,t), Y0 ∼ N(0,I), (8)

where v : RN

img

× R → RN

img is a velocity field that determines the direction to transform Yt into a clean sample by t = 1. During training, we sample a noise Y0 and obtain Yt with a linear schedule Yt = tY1+(1−t)Y0. The Flow Matching loss can then be written as

Limage = Et,Y

0,Y1 ∥v(Yt,t) − (Y1 − Y0)∥2 . (9)

In OneFlow, we use a pretrained autoencoder to map images into latent space. We then design the velocity network v(·) to use a shared Transformer backbone as text but with additional U-Nets to downsample and upsample between the backbone and autoencoder embedding spaces, making use of the same architectural design as Transfusion (Zhou et al., 2025). See illustration in Figure 23.

- 2.3 Concurrent Mixed-modal Generation

To generate multiple modalities, we simply concatenate them into a single sequence. We now present two multimodal time schedules, an independent schedule that can be used when the number of images is known, and an interleaved schedule that needs to be used when the number of images is arbitrary. OneFlow is designed to work with variable-length text and variable number of images.

- 2.3.1 Independent mixed-modal generation

We can consider the simple case with a fixed number of images—typically one. In such case, we can generate both the text and image simultaneously by using two time values ttext and timg, where ttext determines the state of the insertion generation process and timg determines the image generation process. Following prior work, we simply set independent time schedules, one for the text and one for each image. This allows the modalities to be concurrently generated and be dependent on each other during the generation process. However, this naïve process does not allow us to insert images.

- 2.3.2 Interleaved mixed-modal generation A much more complicated setting arises when the number of images is variable and images are being inserted

- as part of the generation process. Similar to the text-only setting, we start generating from the empty sequence. We then model image insertion as a special token value <|image|>, which is added to the token prediction output Q. During generation, when the model predicts an image insertion, we insert noise embeddings of dimension Nimg into the sequence to represent an inserted image initialized at timg = 0.

ins(x,i,<|image|>) = x1,...,xi,y1,...,yN

img

,xi+1,...,xn , yi ∼ N(0,I). (10)

Subsequent steps during generation would then simultaneously generate the image embeddings while also performing more insertions into the sequence. However, since the image is generated at a later time, this implies there is a delay between the image time and the text time, i.e. timg ≤ ttext, which needs to be taken into account during training.

During training, we need to ensure that the text and image noise levels are consistent with with the ones seen during generation. Based on the schedule in (2), the time at which an insertion happens is a random variable that has κ as its cumulative density function, so the time difference between the inserted image time timg and the initial text time ttext is given by

timg = ttext − κ−1(u), where u ∼ Unif(0, 1). (11)

We call this the interleaved time schedule, which imposes a distributional dependency between the time values timg for each image and the text time ttext. In order for the model to learn to fully generate all images, during training we sample from an extended time interval, τtext from [0,2], since the <|image|>token can be inserted

- at τtext = 1.0 at the latest, and fully denoised by τtext = 2.0. The probability for each token being in Xt is then determined by κ(min{1,τtext}) in place of (2). We also sample for each image an extended time value τimg = τtext − κ−1(u). Finally, we determine if an image is deleted from the sequence by checking τimg < 0, and if so, the insertion loss ((7)) will include the <|image|> token which the model would learn to insert.

Otherwise if τimg ≥ 0, the image is in the sequence and we set timg = min{1,τimg}; using the Flow Matching loss (9) to train the velocity. A detailed derivation and in-depth explanation can be found in Appendix D.1.

- 3 Findings and Unlocking New Capabilities in Unified Multimodal

To study the performance of OneFlow against AR, we design a series of controlled experiments. We establish strong baselines by comparing both our approach and AR against other unified multimodal models in the literature. Finally, we explore the novel capabilities that OneFlow enables beyond existing methods. We present our experimental results through five research questions:

- §3.1 How does OneFlow scale compared to AR?
- §3.2 What is the impact of mixed-modal vs. sequential pretraining?
- §3.3 What emergent behaviors does OneFlow exhibit during generation?
- §3.4 How does OneFlow compare to other unified multimodal models?
- §3.5 What new capabilities does OneFlow enable beyond existing methods?

- 3.0.1 Training stages

Our training consists of two main stages: multimodal pretraining and instruction finetuning. During the pretraining stage, we use a mixture of image understanding and image generation data to learn representations for both image and text. We trained with a sequence length of 512 and a global batch size of 4096. We set the mixed generation probability (the likelihood of concurrently generating clean text and the image) to be either 0 or 0.2.

For finetuning, we use a mixture of VQA, text, and interleaved data to give the model the ability to respond to visual question answering problems. We also fine-tune on image generation data at a higher resolution of 512×512 to improve the model’s image generation capabilities. We study the model’s behavior at the 1B scale for our ablations and controlled experiments, and the scaling trend up to 8B is detailed in Section 3.1.

- 3.0.2 Datasets

For multimodal pretraining, we use image-text pairs from a filtered version of the Conceptual Captions dataset (CC12M (Sharma et al., 2018)), the YFCC dataset (Thomee et al., 2016), and licensed data, for a total of 400M examples. During instruction finetuning, we use a filtered image portion of the PerceptionLM dataset (Cho et al., 2025), interleaving data from Chameleon (Team, 2024), and a filtered portion of the Cambrian-7M (Tong et al., 2024a) dataset, totaling around 40M examples.

- 3.0.3 Baselines

To evaluate our model’s performance against existing methods, we compare against two baselines: (1) an autoregressive (AR) + Flow Matching (FM) multimodal model based on Transfusion (Zhou et al., 2025), where text tokens are generated autoregressively and image tokens via FM, and (2) a masked diffusion model based on LLaDA (Nie et al., 2025). For the masked diffusion baseline, we tested two sampling variants: low-confidence and random remasking, with random remasking performing better across all experiments. Unlike Transfusion, we follow Janus-Flow (Ma et al., 2025a) and adopt a dual-encoder setup. For image encoders, we use a pretrained SigLIP2 ViT-SO400M-16@512 (Tschannen et al., 2025) for understanding and an SD3 VAE (Esser et al., 2024) for generation. Following Transfusion, we use U-Net adapters. Figure 23 illustrates how OneFlow consumes a mixed-modal sequence. To improve performance of the AR model, we follow Chameleon (Team, 2024) and sweep over several generation lengths for optimizing CIDEr score. For image generation, we use a first order Euler solver with entropy rectifying guidance (Ifriqi et al., 2025), we set the guidance scale to 5.0 across 50 sampling steps.

- 3.0.4 Model Configurations

To investigate scaling trends, we train models at three different sizes – 1B, 3B, and 8B parameters, and token counts. For each benchmark, we plot all results over log-FLOPS curve and regress linear trendlines. Following Transfusion, we also estimate relative compute efficiency by measuring the parity FLOPS ratio: the ratio between the number of FLOPS required by our AR and OneFlow to reach the same level of performance.

- 3.0.5 FLOPS Estimation

To estimate the FLOPS during training, we track the number of tokens throughout training for all models. During multimodal pretraining, we do not delete images and OneFlow uses on average 50% fewer generated text tokens compared to AR. We use Flash Attention and follow their FLOPS estimation (Dao et al., 2022).

- 3.0.6 Evaluation setup

Following Cambrian (Tong et al., 2024a) and PLM (Cho et al., 2025), we group VQA tasks into five groups: General, Knowledge, OCR & Chart, Hard Perception, and Hallucination. We evaluate image generation quality using the FID metric (Heusel et al., 2017) on the COCO-2014 (Lin et al., 2014) validation set, using our base model at 256×256 resolution. To assess prompt alignment, we report CLIPScore (Hessel et al., 2021) and DPG-Bench (Hu et al., 2024). Additionally, we include WISE (Niu et al., 2025) cultural to better understand knowledge-based generation.

- 3.1 OneFlow Scales Better than AR

We first discuss controlled experiments, where we study the performance across different frameworks when trained on the same data sets using the same backbone architecture. Specifically, we study the performance of OneFlow and AR in controlled settings at various model sizes and token counts. To ensure OneFlow has no advantage in data-constrained settings, we trained both models on 2B image-text pairs over 500k iterations using a batch size of 4096. Both models were initialized from Llama 3.2 1B (AI@Meta, 2024). For AR, the number of tokens predicted during training equals the sequence length, whereas for OneFlow, the number of predicted tokens corresponds to the number of deleted tokens, which on average is 50% of the data sequence. We measure compute efficiency using the parity FLOP ratio (Table 1), which quantifies the relative FLOPS required by OneFlow to match AR’s performance. Our results show that OneFlow achieves

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

80.0

9.25

1.45

0.610

77.5

9.00

0.605

8.75

75.0

1.40

0.600

8.50

DPGScore

72.5

###### ROUGE-L

###### CIDEr

0.595

###### FID

8.25

1.35

70.0

8.00

0.590

67.5

1.30

7.75

0.585

65.0

###### AR + FM OneFlow

7.50

0.580

62.5

1.25

7.25

60.0

1014 1015 1016

1014 1015 1016

1014 1015 1016

1014 1015 1016

FLOPs

FLOPs

FLOPs

FLOPs

- Figure 5 Performance of OneFlow vs. AR +FM baseline models at different model scales, data and compute. For text-to-image generation, we report DPG-Bench and FID. For image-to-text caption quality, we report CIDEr and ROUGE. In every benchmark, OneFlow consistently exhibits better scaling laws than AR + FM. Model sizes include 1B, 3B, and 8B.

DPG ↑ FID ↓ CIDEr ↑ ROUGE ↑ Parity FLOPS Ratio 0.49 0.97 0.32 0.42

Table 1 Parity FLOPS ratio in a controlled experiment. Both models were trained on 2B image-text pairs. Parity FLOPS Ratio represents the relative amount of OneFlow FLOPS needed to match the final AR + FM performance.

the largest compute savings on dense prompt alignment and captioning tasks, requiring less than half the FLOPS of AR to reach the same level.

We find that OneFlow scales better than AR on every benchmark. This scaling advantage is especially pronounced on DPG Bench, where OneFlow scales significantly better than AR. For image captioning as well, OneFlow shows a notable performance gap relative to AR. Figure 5 visualizes the scaling trend, and the final metrics are shown in Tables 2 and 3, along with a comparison against other state-of-the-art models. A detailed breakdown of the controlled comparisons among VQA task categories can be found in Table 19.

- 3.2 Mixed Modal Pre-training Enables Better Generation and Understanding

[Figure 44]

Figure 7 Mixed modal concurrent training.

In this section, we study the impact of mixed-modal pretraining by comparing concurrent and sequential training approaches. Specifically, we train two 1B variants: one using sequential pretraining (either text-to-image or image-to-text), and another using the same data but with 20% of examples incorporating concurrent generation. Our ablation experiment reveals that mixed-modal training significantly improves downstream visual question answering performance (Figure 6) and provides a minor boost to image generation. These results demonstrate the benefits of concurrent mixed-modal pretraining. Given these consistent improvements, we use the mixed-modal pretrained model for all subsequent finetuning experiments.

| | |relative| | |
|---|---|---|---|---|
| | |+4.0%<br><br>+4.0%| | |
| | | | | |
| | | | | |
| | | | | |

48 50 52 54

OneFlow Mixed

OneFlow

AR + FM

AVG VQA

70 75 80 85

- 0

- 1

- 2

+1.5%

relative

+9.3%

DPG

0.58 0.60 0.62 0.64

- 0

- 1

- 2

+1.6%

relative

+3.3%

WISE

- Figure 6 Mixed modal vs Sequential pretraining. Mixed modal pretraining vs sequential pretraining. Mixed pretraining achieves 4% relative improvement on VQA tasks and slight improvements on image generation as well.

Image Generation Captioning

Text Image

Model Size

FID↓ CLIP↑ DPG↑ WISE (c.) ↑ CIDEr↑ ROUGE↑ BLEU4↑ Unified MLM

MetaMorph (Tong et al., 2024b) 7B AR AR 11.8 26.6 – – – – – LMFusion (Shi et al., 2024b) 7B AR Diff 14.0 24.4 – – 38.4 – – Transfusion (Zhou et al., 2025) 7B AR Diff 16.0 26.5 77.8 – 33.7 – – Janus-Pro (Chen et al., 2025b) 1.5B AR AR 15.2† 26.0† 82.0† 0.20 – – – Janus-Flow (Ma et al., 2025a) 1.5B AR FM 12.4† 26.1† 80.1† 0.13 – – – Bagel (Deng et al., 2025b) 7B AR FM 27.7† 26.2† 84.7† 0.44 – – –

Multimodal Diffusion

UniDisc (Swerdlow et al., 2025) 1.4B Mask Mask 23.9 – – – – – – D-DiT (Li et al., 2025) 2B Mask Diff – – – – 56.2 – – Muddit (Shi et al., 2025) 1B Mask Mask – – – – 59.7 – – MMaDA (Yang et al., 2025) 8B Mask Mask 33.2† 25.1† 74.2† 0.67 – – – FUDOKI (Wang et al., 2025) 1.5B DFM DFM – – 83.6 – – – –

Controlled Comparisons

AR + FM Ablation 1B AR FM 12.2 26.5 73.4 0.61 123.9 57.2 0.39 Mask + FM Ablation 1B Mask FM 11.3 26.5 75.5 0.64 128.4 58.6 0.39 OneFlow 1B EF FM 12.1 26.6 79.1 0.62 138.1 60.8 0.41 OneFlow Mixed 1B EF FM 9.7 26.6 80.3 0.63 139.8 60.9 0.42

OneFlow 8B EF FM 10.7 26.7 79.3 0.65 141.1 61.1 0.42 OneFlow Mixed 8B EF FM 9.5 26.6 80.4 0.68 142.1 61.1 0.43

- Table 2 Image generation and captioning benchmarks after multimodal pretraining. We find that mixed-modal training consistently improves performance. †Evaluated using official open-source model weights. Highlighting denotes best results across all models. 3.2.1 Mixed Generation Probability

FID

###### CIDEr

###### ROUGE-L

###### DPG Score

1.39

0.72

0.72

15.0

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.71

0.71

1.38

14.5

0.70

0.70

1.37

0.69

0.69

14.0

1.36

0.68

0.68

13.5

1.35

0.67

0.67

1.34

0.66

0.66

13.0

0.0 0.1 0.2 0.4 0.8 1.0

0.0 0.1 0.2 0.4 0.8 1.0

0.0 0.1 0.2 0.4 0.8 1.0

0.0 0.1 0.2 0.4 0.8 1.0

Mixed Gen Prob

Figure8 Effectofmixed-generationprobabilityonperformance. Higher mixed-generation probabilities consistently improve image understanding (CIDEr, ROUGE-L) and compositional generation (DPG Score), while image generation quality (FID) remains stable. This demonstrates that mixed training enhances understanding and compositional capabilities without compromising generation performance.

In this section, we study the impact of different mixed-generation probabilities over 200k training steps. Results show a clear monotonic trend: higher mixed-generation probability consistently improves image understanding across all metrics. CIDEr scores increase from 135.3 (0% mixed) to 138.3 (100% mixed), ROUGE-L improves from 0.67 to 0.71, and DPG Score rises from 0.67 to 0.71. Importantly, image generation quality (FID, DPG Score) remains stable across all probabilities, demonstrating that mixed training strengthens understanding capabilities without degrading generation performance.

- 3.3 Hierarchical Generation Enables Emergent Reasoning

In Figure 3, we present OneFlow’s sampling process when prompted with a visual question. In response to prompts such as "{question} Explain why.", OneFlow generates a reasoning chain before arriving at the final answer, without any Chain-of-Thought (CoT) (Wei et al., 2022) prompting or RL post-training. For example, when asked "Is there a snowboard in the image? Explain why.", the model first implicitly performs visual search by examining the image and searching through likely locations for the snowboard. Similarly, for the math puzzle in (Figure 3 bottom), the model first identifies objects in the image that match the prompt description—the green sphere and the large shiny cylinder—before arriving at the final answer.

General Knowledge OCR & Chart Vision Halluc.

RealWorld

TextVQA

ChartQA

DocVQA

MMMU

VQAv2

Params

POPE

MMB

MME

AI2D

GQA

Model

Multimodal LM Show-O (Xie et al., 2024) 1.3B – – 61.0 1232.9 27.4 – – – – – 84.5 MetaMorph (Tong et al., 2024b) 7B 75.2 – – – 41.8 – – 37.1 60.5 58.3 – Janus-Flow (Ma et al., 2025a) 1.5B 74.9 79.8 60.3 1333.1 29.3 – 64.6 55.5 – – 88.0 Janus-Pro† (Chen et al., 2025b) 1.5B 73.4 67.9 59.3 1443.0 33.4 62.8 21.2 35.8 53.9 53.5 84.8 Janus-Pro† (Chen et al., 2025b) 7B 76.9 74.1 62.0 1531.0 38.2 68.1 24.3 – 57.2 56.4 85.2

Mask Diffusion Muddit (Shi et al., 2025) 1B – 67.7 57.1 1104.6 – – – – – – – D-DiT (Li et al., 2025) 2B – 60.1 59.2 1124.7 – – – – – – 84.0 MMaDA (Yang et al., 2025) 8B 68.5 76.7 61.3 – 30.2 – – – – – 86.1

Discrete Flow FUDOKI (Wang et al., 2025) 1.5B 73.9 – 57.6 1485.4 34.3 – – – – – 86.1

Controlled Comparisons

AR + FM Ablation 1B 64.7 66.0 55.0 1394.8 28.9 59.1 22.7 35.5 48.3 50.5 85.6 Mask + FM Ablation 1B 66.0 64.4 55.6 1462.2 28.4 55.1 18.7 34.7 44.6 50.6 84.6 OneFlow 1B 69.0 67.7 57.8 1497.1 29.8 58.5 23.8 35.0 50.4 50.6 85.7

OneFlow 8B 72.5 73.7 61.9 1542.5 33.1 63.4 37.1 42.1 58.6 54.0 86.3

- Table 3 VQA performance comparison. OneFlow outperforms AR and Mask models across all benchmarks in controlled experiments using identical finetuning data. Highlighting shows best results in the 1B controlled comparisons. Our results are also competitive with existing autoregressive and discrete diffusion models. †Evaluated using official open source weights.

Our results align with findings in Physics of LLMs (Ye et al., 2024) and MetaMorph (Tong et al., 2024b), where the authors suggest that LLMs precompute reasoning graphs before generating tokens. However, our findings demonstrate that the model can perform the same reasoning chain without autoregressive decoding. This suggests that reasoning capabilities can emerge in non-autoregressive architectures and transfer effectively to OneFlow. We show more example VQA generations compared to the AR baseline in Figure 10.

- 3.4 Comparison with State-of-The-Art Unified Models

In Tables 2 and 3, we benchmark OneFlow against other autoregressive and diffusion multimodal models. Our models deliver competitive performance across all tasks, rivaling state-of-the-art results despite significant differences in training procedures.

Understanding Performance. On visual question answering, our 1B model nearly matches MMaDA 8B, while our 8B model surpasses MMaDA 8B on most benchmarks (Table 3). This is noteworthy because models like MMaDA received extensive post-training with chain-of-thought data and RL finetuning, whereas OneFlow did not. As discussed in Section 3.3, we demonstrate that reasoning naturally emerges during generation. Similarly, while FUDOKI leveraged a pretrained multimodal model, OneFlow delivers competitive performance without this advantage. We observe that initializing from a pretrained text-only autoregressive model consistently enhances both prompt alignment for image generation and VQA performance, yielding similar gains as the AR+FM model (Table 5).

Generation Performance. For image generation, we measured FID using open-source weights on COCO2014 (Lin et al., 2014) validation set. Since MMaDA and Bagel received extensive aesthetic finetuning that produces images stylistically distant from real images (yielding higher FID scores). For fair comparison, we benchmark against MMaDA Base and Bagel without thinking mode. We show qualitative comparisons in Figure 9.

|[Figure 45]|
|---|

Bagel

|[Figure 46]|
|---|

MMaDA

|[Figure 47]|
|---|

OneFlow

|[Figure 48]|
|---|

|[Figure 49]|
|---|

|[Figure 50]|
|---|

|[Figure 51]|
|---|

|[Figure 52]|
|---|

|[Figure 53]|
|---|

|[Figure 54]|
|---|

|[Figure 55]|
|---|

|[Figure 56]|
|---|

A polar bear balancing on a blue barrel.

A table with some oranges and some cups.

A very cute cat sitting on a piece of luggage.

A building with a door sitting behind a sign.

- Figure 9 Qualitative comparison of OneFlow and SOTA models. We notice that OneFlow gets the details of the prompt correctly, for instance the polar bear is ‘balancing on a blue barrel’. The visual details of our generation are also better compared to MMaDA possibly due to using continuous image tokens rather than discrete. In the last column, the figure shows OneFlow handles common semantic challenges more effectively, as it was able to generate a building with

‘a door sitting behind a sign’.

[Figure 57]

Question: Subtract all large rubber spheres. Subtract all big shiny cylinders. How many objects are left? Explain your answer:

In the image, there is one object that matches the description of the large sphere: the green sphere, and the only large shiny cylinder, which is the yellow cylinder. Therefore, the total number of objects left in the image is four.

There are five objects left in the image. To find their count, we look at the positions of the spheres relative to the cylinders and the cube. From left to right, the first sphere is the green, the second is the purple, the third is the blue, and the fourth is the brown. The fifth object, which is the yellow cylinder, is not counted since it is not listed in the options. Therefore, the total count of objects that are left with the categories (green, purple, blue, brown) is three.

Counting

Ground Truth: 4

AR OneFlow

[Figure 58]

Question: In which direction is the front wheel of the car on the right side facing? You must explain your answer. Explain your answer:

The plastic stool in the image is located on the right side of the canal, near the side of the canal. It is blue and stands out against the natural colors of the scene, contrasting with the red colors of the buildings and the canal.

There is no plastic stool visible in the image. The scene includes a canal, boats, trees, and traditional buildings, but none of these items appear to be present within the frame. The stool in question may be part of the rustic outdoor setting or an unrelated object not in the frame.

OneFlow

Visual Search

Ground Truth: Right

AR

- Figure 10 (left) OneFlow first locates the object in the prompt before calculation. (right) OneFlow locates the target (stool), then analyzes color contrast. A red arrow is added for clarity and is not in the original image.

5.7

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8

- 1

- 2

- 3

- 4

- 5

- 6

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| |OneFlow AR+FM<br><br>| |
|---|
| | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

5.6

###### CHAIRScoreS

CHAIRScoreI

###### GPT4oScore

5.5

5.4

| |
|---|

5.3

0.0 0.5 1.0 1.5 2.0

0.0 0.5 1.0 1.5 2.0

0.0 0.5 1.0 1.5 2.0

CFG Scale

- Figure 11 OneFlow with classifier-free guidance produces longer and more detailed answers OneFlow matches the AR baseline in GPT-4o as judge (Cheng et al., 2025), while higher guidance scales trade off between detailedness and hallucination. CHAIRS and CHAIRI measure the rate of hallucinated objects at the sentence and instance levels, respectively (lower is better).

[Figure 59]

Input image

- [CFG0.0:] A dog sitting in the driver’s seat of a truck.
- [CFG1.0:] A dog sitting in the driver’s seat of a red and gray truck.

- [CFG2.0:] A red and grey suv truck with a dog sitting in the driver’s seat. The truck is parked on a grassy field with a tree on the left side and a clear blue sky in the background.

- Figure 12 Edit Flows with classifier-free guidance produces longer and more detailed answers, examples of classifier-free guidance effects on text generation are shown in Figure 18.

- 3.5 OneFlow Unlocks New Capabilities

- 3.5.1 Classifier-free guidance improves text detailedness

The use of continuous-time Markov chains allows us to apply classifier-free guidance (CFG) to our model’s insertion rates. Specifically, given an unconditional prediction λ(Xt)Q(Xt) and a conditional prediction λ(Xt|c)Q(Xt|c), where c is the prompt and w is the guidance weight, the modified insertion rate is:

λcfg(Xt|c) = λ(Xt|c)wλ(Xt)1−w and Qcfg(Xt|c) ∝ Q(Xt|c)wQ(Xt)1−w. (12)

As shown in Figures 12 and 18, higher CFG values consistently increase the length and detail of generated text. We quantitatively evaluated caption quality and hallucination using CapArena (Cheng et al., 2025) by prompting GPT4-o. Figure 11 shows that increasing CFG leads to more detailed captions, with OneFlow matching AR’s level of detail at a guidance scale of 1. However, this increased detail comes at the expense of hallucinations at very high CFG values.

- 3.5.2 Simultaneous generation of interleaved text and images

When autoregressive multi-modal models insert an image, they append it at the end of the current generation, fully denoise it, then continue the generation process. However, OneFlow is able to simultaneously denoise images and the text. When the model deems it necessary, it is able to insert an new image in the existing text and denoise it along with the text, as proposed in Section 2.3.

To train this model, we took OneFlow 1B Mixed and finetuned it on the interleaved subset of the Chameleon dataset (Team, 2024) for 20000 steps. This subset contains 17000 examples that interleave both text and image data. Figure 4 shows the generation order of the tokens where two images were generated as part of the answer, with more detailed examples in Appendix A and animated versions in the supplementary material.

#### Text Generation

≈0.5x sequence length

1x sequence length

1x sequence length

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

inputs

token

| | |
|---|---|
| | |

[Figure 66]

mask token

blank token

targets

Autoregressive

Mask diffusion

Edit Flow

#### Interleaved Text and Image Generation

###### Autoregressive (+diffusion)

OneFlow

≈0.5x sequence length

up to 2x sequence length (clean & noisy)

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

inputs

clean image

[Figure 73]

| | |
|---|---|
| | |

noisy image

[Figure 74]

[Figure 75]

[Figure 76]

image

insert-image token

image

targets

Transfusion (autoreg. + diffusion)

OneFlow

- Figure 13 Illustration of the model input and targets during training for (top) text generation and (bottom) interleaved generation. On text generation, Edit Flow simply deletes tokens instead of replacing them with a special mask token, resulting in lower FLOPS but with the same information as the mask diffusion framework. On interleaved generation, to train autoregressive with diffusion denoising, the images are typically duplicated so that both the clean and the noisy images are in the sequence. On the other hand, OneFlow deletes tokens and images during training which reduces the sequence length.

Generative Model Framework Text AR AR Mask Diffusion Discrete FM Edit Flow Image AR Diffusion / FM Mask Diffusion Discrete FM FM

Training Properties Attention Causal Block Causal Bidirectional Bidirectional Bidirectional Tokens / Iter Seq Len up to 2x Seq Len Seq Len Seq Len ∼50% Seq Len

Generation Capabilities

Understanding ✓ ✓ ✓ ✓ ✓ Single image generation ✓ ✓ ✓ ✓ ✓ Variable length ✓ ✓ × × ✓ Concurrent mixed-modal × × ✓ ✓ ✓ Interleaved generation × ✓ × × ✓

Chameleon (Team, 2024) JanusPro (Chen et al., 2025b)

Transfusion (Zhou et al., 2025) Bagel (Deng et al., 2025b)

MMaDA (Yang et al., 2025) FUDOKI (Wang et al., 2025) OneFlow

Examples

###### Table 4 High-level comparison between different frameworks for building unified models of text and image generation.

- 4 Related Work

Unified Multimodal Models. Early unified models like Flamingo (Alayrac et al., 2022) pioneered the integration of vision and language via cross-attention. Emu2 (Sun et al., 2023) advanced this paradigm by employing a unified autoregressive objective. Chameleon (Team, 2024) adopted an early-fusion, AR architecture that unifies images and text as discrete tokens. Subsequent work spans three architectural paradigms: diffusion (Yang et al., 2025; Li et al., 2025; Swerdlow et al., 2025; Shi et al., 2025; Wang et al., 2025), autoregressive (Alayrac et al., 2022; Team, 2024; Sun et al., 2024b; Chen et al., 2025b; Wu et al., 2025a; Yu et al., 2023), and hybrids (Zhao et al., 2024; Zhou et al., 2025; Xie et al., 2024; Tong et al., 2024b; Ma et al., 2025a; Shi et al., 2024b). Transfusion (Zhou et al., 2025) improves upon Chameleon by leveraging AR for text and diffusion for images. The Janus series collectively introduces decoupled visual pathways for understanding and generation (Wu et al., 2024; Ma et al., 2025b; Chen et al., 2025a). BAGEL scales multimodal pretraining using a Mixture-of-Transformers (MoT) (Liang et al., 2024; Deng et al., 2025a). In parallel, diffusion-based models have emerged as promising alternatives: DDiT applies discrete diffusion to text and continuous diffusion to images (Li et al., 2025); UniDisc uses fully discrete diffusion with Llama tokenizer for text and MAGVIT-v2 (Yu et al., 2024) for images (Swerdlow et al., 2025); FUDOKI uses discrete flow matching instead of masked diffusion, with a SigLIP (Zhai et al., 2023) encoder for understanding and a LlamaGen (Sun et al., 2024a) tokenizer for generation (Wang et al., 2025). MMaDA scales masked diffusion with mixed chain-of-thought fine-tuning and UniGRPO (Yang et al., 2025). MuDDiT combines a pretrained T2I (Meissonic (Bai et al.,

- 2024)) with a lightweight text decoder for MaskGIT-style discrete diffusion (Chang et al., 2022; Shi et al.,
- 2025). While these models are limited by a fixed generation order or fixed-length outputs, our approach fundamentally differs by simultaneously generating interleaved content and a variable number of images. We summarize the differences between each framework in Table 4.

Discrete Diffusion and Discrete Flow Matching. Iterative refinement models, including diffusion (SohlDickstein et al., 2015; Ho et al., 2020) and flow models (Liu et al., 2022; Albergo et al., 2023; Lipman et al., 2024), have been adapted for discrete token spaces. Discrete diffusion models typically learn to reverse a corruption process (Austin et al., 2021; Lou et al., 2024), while discrete flow models transport between two distributions with an interpolating scheme (Campbell et al., 2024b; Gat et al., 2024). These frameworks offer a large design space for discrete generative modeling (Shaul et al., 2024; Wang et al., 2025), and can further be generalized to a large family of continuous-time Markov processes with a unified training recipe (Holderrieth et al., 2024). However, recent works on discrete diffusion have predominantly focused on a simplified mask construction (Sahoo et al., 2024; Shi et al., 2024a; Ou et al., 2024; Zheng et al., 2024). This masking framework, however, cannot be easily applied to variable-length and especially simultaneous interleaved generation.

Edit-based Non-autoregressive Language Models. Early non-autoregressive models for variable-length generation (Gu et al., 2019a,b; Stern et al., 2019; Reid et al., 2022) often relied on multiple models and evaluations to handle edit operations, e.g. by inserting placeholder tokens first and then replacing them with the actual tokens. While later work like Edit Flows (Havasi et al., 2025) improved on this by using a continuous-time framework and removes the need for any placeholder tokens. In order to finetune from existing mask diffusion models, Wu et al. (2025b) and Kim et al. (2025) have also explored inserting only mask tokens, breaking down insertions back into two steps. In contrast, we simply made use of autoregressive models as initialization, modifying only model outputs for insertion predictions with no special tokens used in the model inputs, as this allows us to save on training FLOPS and inference costs. Related to combining discrete and continuous generative processes, Campbell et al. (2024a) also proposed modeling insertions with a diffusion model for denoising, but only considered singular insertions and required an additional generative model due to not using an interleaved time schedule. In contrast, our approach considers mixed-modal sequence data, denoising images concurrently with per-image time values, and allows for parallel token insertions.

- 5 Conclusion and Limitations

We introduced OneFlow, a novel non-autoregressive multimodal model that overcomes the fixed-length generation limitations of diffusion models and has better scaling than autoregressive multimodal models. We introduced mixed-modal generation approaches, which through extensive controlled experiments, improve

on benchmarks for both image understanding and image generation. We also propose a novel approach to interleaved generation that simultaneously denoises images and inserts text tokens, with promising qualitative results. Interleaved generation is still in its infancy and we expect to see more incoming research efforts in constructing large-scale data sets (Awadalla et al., 2024; Laurençon et al., 2023; Zhu et al., 2023) and designing comprehensive benchmarks.

A limitation of requiring bidirectional attention is the lack of key-value caching, which increases inference cost. We do find that OneFlow can already obtain good captioning performance with very few model evaluations—outperforming AR with only 6 sampling steps (Figure 21). Therefore, further reducing inference costs, with semi-autoregressive models (Arriola et al., 2025; Gat et al., 2025) or more sophisticated methods, would be an exciting research direction.

Acknowledgements

We thank Maha Elbayad, Emily Dinan, Xiaochuang Han, Lili Yu, Chunting Zhou, and Melissa Hall for building the Transfusion code base which we built our code upon. We thank Peter Tong and David Fan for pointers on VQA and feedback on the paper. We thank Koustuv Sinha, Sharut Gupta, Marjan Ghazvininejad, Jakob Verbeek, and Amir Bar for feedback on the paper draft and fruitful discussions throughout the project.

References

AI@Meta. Llama 3 model card. 2024. https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md.

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikołaj Bińkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karén Simonyan. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems, volume 35, pages 23716–23736, 2022.

Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797, 2023.

Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. arXiv preprint arXiv:2503.09573, 2025.

Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021.

Anas Awadalla, Le Xue, Oscar Lo, Manli Shu, Hannah Lee, Etash Guha, Sheng Shen, Mohamed Awadalla, Silvio Savarese, Caiming Xiong, et al. Mint-1t: Scaling open-source multimodal data by 10x: A multimodal dataset with one trillion tokens. Advances in Neural Information Processing Systems, 37:36805–36828, 2024.

Jinbin Bai, Tian Ye, Wei Chow, Enxin Song, Qing-Guo Chen, Xiangtai Li, Zhen Dong, Lei Zhu, and Shuicheng Yan. Meissonic: Revitalizing masked generative transformers for efficient high-resolution text-to-image synthesis. arXiv preprint arXiv:2410.08261, 2024.

Andrew Campbell, William Harvey, Christian Weilbach, Valentin De Bortoli, Thomas Rainforth, and Arnaud Doucet. Trans-dimensional generative modeling via jump diffusion models. Advances in Neural Information Processing Systems, 36, 2024a.

Andrew Campbell, Jason Yim, Regina Barzilay, Tom Rainforth, and Tommi Jaakkola. Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design. arXiv preprint arXiv:2402.04997, 2024b.

Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11315–11325, 2022.

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811,

- 2025a.

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811,

- 2025b.

Kanzhi Cheng, Wenpo Song, Jiaxin Fan, Zheng Ma, Qiushi Sun, Fangzhi Xu, Chenyang Yan, Nuo Chen, Jianbing Zhang, and Jiajun Chen. Caparena: Benchmarking and analyzing detailed image captioning in the llm era. arXiv preprint arXiv:2503.12329, 2025.

Jang Hyun Cho, Andrea Madotto, Effrosyni Mavroudi, Triantafyllos Afouras, Tushar Nagarajan, Muhammad Maaz, Yale Song, Tengyu Ma, Shuming Hu, Suyog Jain, et al. Perceptionlm: Open-access data and models for detailed visual understanding. arXiv preprint arXiv:2504.13180, 2025.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025a.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025b.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky T. Q. Chen, Gabriel Synnaeve, Yossi Adi, and Yaron Lipman. Discrete flow matching. Advances in Neural Information Processing Systems, 37:133345–133385, 2024.

Itai Gat, Heli Ben-Hamu, Marton Havasi, Daniel Haziza, Jeremy Reizenstein, Gabriel Synnaeve, David Lopez-Paz, Brian Karrer, and Yaron Lipman. Set block decoding is a language model inference accelerator. arXiv preprint arXiv:2509.04185, 2025.

Jiatao Gu, Qi Liu, and Kyunghyun Cho. Insertion-based decoding with automatically inferred generation order. Transactions of the Association for Computational Linguistics, 7:661–676, 2019a.

Jiatao Gu, Changhan Wang, and Junbo Zhao. Levenshtein transformer. Advances in neural information processing systems, 32, 2019b.

Marton Havasi, Brian Karrer, Itai Gat, and Ricky T. Q. Chen. Edit flows: Flow matching with edit operations. arXiv preprint arXiv:2506.09018, 2025.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7514–7528, 2021.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Peter Holderrieth, Marton Havasi, Jason Yim, Neta Shaul, Itai Gat, Tommi Jaakkola, Brian Karrer, Ricky T. Q. Chen, and Yaron Lipman. Generator matching: Generative modeling with arbitrary markov processes. arXiv preprint arXiv:2410.20587, 2024.

Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

Tariq Berrada Ifriqi, Adriana Romero-Soriano, Michal Drozdzal, Jakob Verbeek, and Karteek Alahari. Entropy rectifying guidance for diffusion and flow models, 2025. https://arxiv.org/abs/2504.13987.

Jaeyeon Kim, Lee Cheuk-Kit, Carles Domingo-Enrich, Yilun Du, Sham Kakade, Timothy Ngotiaoco, Sitan Chen, and Michael Albergo. Any-order flexible length masked diffusion. arXiv preprint arXiv:2509.01025, 2025.

Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. Advances in Neural Information Processing Systems, 36:71683–71702, 2023.

Zijie Li, Henry Li, Yichun Shi, Amir Barati Farimani, Yuval Kluger, Linjie Yang, and Peng Wang. Dual diffusion for unified image generation and understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2779–2790, 2025.

Weixin Liang, Lili Yu, Liang Luo, Srinivasan Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen-tau Yih, Luke Zettlemoyer, et al. Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models. arXiv preprint arXiv:2411.04996, 2024.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO: Common objects in context. 2014.

Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer, Ricky T. Q. Chen, David

Lopez-Paz, Heli Ben-Hamu, and Itai Gat. Flow matching guide and code. arXiv preprint arXiv:2412.06264, 2024. Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with

rectified flow. arXiv preprint arXiv:2209.03003, 2022. Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In Proceedings of the 41st International Conference on Machine Learning, pages 32819–32848, 2024.

Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai Yu, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7739–7751, 2025a.

Yiyang Ma, Xingchao Liu, Xiaokang Chen, Zhiyu Wu, Wen Liu, Zhenda Xie, Zizheng Pan, Xingkai Yu, Liang Zhao, Yuxuan Wang, Jie Liu, and Chong Ruan. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025b.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, JUN ZHOU, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. In ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy, 2025.

Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.

Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. arXiv preprint arXiv:2406.03736, 2024.

Machel Reid, Vincent J Hellendoorn, and Graham Neubig. Diffuser: Discrete diffusion via edit-based reconstruction. arXiv preprint arXiv:2210.16886, 2022.

Subham Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 37:130136–130184, 2024.

Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of ACL, 2018.

Neta Shaul, Itai Gat, Marton Havasi, Daniel Severo, Anuroop Sriram, Peter Holderrieth, Brian Karrer, Yaron Lipman, and Ricky T. Q. Chen. Flow matching with general discrete paths: A kinetic-optimal perspective. arXiv preprint arXiv:2412.03487, 2024.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and generalized masked diffusion for discrete data. Advances in neural information processing systems, 37:103131–103167, 2024a.

Qingyu Shi, Jinbin Bai, Zhuoran Zhao, Wenhao Chai, Kaidong Yu, Jianzong Wu, Shuangyong Song, Yunhai Tong, Xiangtai Li, Xuelong Li, et al. Muddit: Liberating generation beyond text-to-image with a unified discrete diffusion model. arXiv preprint arXiv:2505.23606, 2025.

Weijia Shi, Xiaochuang Han, Chunting Zhou, Weixin Liang, Xi Victoria Lin, Luke Zettlemoyer, and Lili Yu. Lmfusion: Adapting pretrained language models for multimodal generation. arXiv preprint arXiv:2412.15188, 2024b.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015.

Mitchell Stern, William Chan, Jamie Kiros, and Jakob Uszkoreit. Insertion transformer: Flexible sequence generation via insertion operations. In International Conference on Machine Learning, pages 5976–5985. PMLR, 2019.

Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024a.

Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Emu: Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222, 2023.

Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12491–12501, 2024b.

Alexander Swerdlow, Mihir Prabhudesai, Siddharth Gandhi, Deepak Pathak, and Katerina Fragkiadaki. Unified multimodal discrete diffusion. arXiv preprint arXiv:2503.20853, 2025.

Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and

Li-Jia Li. Yfcc100m: The new data in multimedia research. Communications of the ACM, 59(2):64–73, 2016.

Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2024a.

Shengbang Tong, David Fan, Jiachen Zhu, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. Metamorph: Multimodal understanding and generation via instruction tuning. arXiv preprint arXiv:2412.14164, 2024b.

Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

Jin Wang, Yao Lai, Aoxue Li, Shifeng Zhang, Jiacheng Sun, Ning Kang, Chengyue Wu, Zhenguo Li, and Ping Luo. Fudoki: Discrete flow-based unified understanding and generation via kinetic-optimal velocities. arXiv preprint arXiv:2505.20147, 2025.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Zhenda Xie, Chao Wu, and Chong Ruan. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848, 2024.

Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12966–12977, 2025a.

Zirui Wu, Lin Zheng, Zhihui Xie, Jiacheng Ye, Jiahui Gao, Yansong Feng, Zhenguo Li, Victoria W., Guorui Zhou, and Lingpeng Kong. Dreamon: Diffusion language models for code infilling beyond fixed-size canvas, 2025b. https://hkunlp.github.io/blog/2025/dreamon.

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.

Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large diffusion language models. arXiv preprint arXiv:2505.15809, 2025.

Tian Ye, Zicheng Xu, Yuanzhi Li, and Zeyuan Allen-Zhu. Physics of language models: Part 2.2, how to learn from mistakes on grade-school math problems. arXiv preprint arXiv:2408.16293, 2024.

Lijun Yu, José Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, Alexander G. Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A. Ross, and Lu Jiang. Language model beats diffusion – tokenizer is key to visual generation, 2024. https://arxiv.org/abs/2310.05737.

Lili Yu, Bowen Shi, Ramakanth Pasunuru, Benjamin Muller, Olga Golovneva, Tianlu Wang, Arun Babu, Binh Tang, Brian Karrer, Shelly Sheynin, Candace Ross, Adam Polyak, Russell Howes, Vasu Sharma, Puxin Xu, Hovhannes Tamoyan, Oron Ashual, Uriel Singer, Shang-Wen Li, Susan Zhang, Richard James, Gargi Ghosh, Yaniv Taigman, Maryam Fazel-Zarandi, Asli Celikyilmaz, Luke Zettlemoyer, and Armen Aghajanyan. Scaling autoregressive multi-modal models: Pretraining and instruction tuning, 2023. https://arxiv.org/abs/2309.02591.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.

Chuyang Zhao, Yuxing Song, Wenhao Wang, Haocheng Feng, Errui Ding, Yifan Sun, Xinyan Xiao, and Jingdong Wang. Monoformer: One transformer for both diffusion and autoregression. arXiv preprint arXiv:2409.16280, 2024.

Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling. arXiv preprint arXiv:2409.02908, 2024.

Chunting Zhou, LILI YU, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. In The Thirteenth International Conference on Learning Representations, 2025. https://openreview.net/ forum?id=SI2hI0frk6.

Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi. Multimodal c4: An open, billion-scale corpus of images interleaved with text. Advances in Neural Information Processing Systems, 36:8958–8974, 2023.

A Additional Generation Examples

|[Figure 77]|
|---|

A man standing in the

middle of a nebula filled with stars.

|[Figure 78]|
|---|

- A shake next to a cake.

|[Figure 79]|
|---|

Space scene. Magician standing on landscape silhouette with tree and

fractal colorful nebula.

|[Figure 80]|
|---|

A dog wearing a hat and a scarf.

|[Figure 81]|
|---|

A blue jay standing on a large basket of rainbow macarons.

|[Figure 82]|
|---|

Fantasy portal in forest landscape surrounded by tree and magic. Green nature with magical architecture concept art.

|[Figure 83]|
|---|

A cute animal holding an ice cream cone.

|[Figure 84]|
|---|

An angel in the middle of the road.

|[Figure 85]|
|---|

Portrait of a hipster wearing sunglasses. Stylized drawing of a fashionista on a colored background.

|[Figure 86]|
|---|

People walking with

umbrellas with balloons in the sky

|[Figure 87]|
|---|

A painting of a city street at night.

|[Figure 88]|
|---|

A blue cat.

- Figure 14 Generated images from the OneFlow model at 512×512 resolution.

|[Figure 89]|
|---|

Lovely adventurer and

explorer in tropical forest with mountains.

|[Figure 90]|
|---|

Cute and fluffy Labrador puppy in hat. Watercolor painting.

|[Figure 91]|
|---|

A majestic castle

illuminated by lights. The ground is covered in a blanket of snow, with a body of water in the foreground.

|[Figure 92]|
|---|

A robot made out of wood.

Generated Text Generated Images Prompt: I’m looking for a new hobby that’s relaxing and will allow me to be outdoors to get some fresh air. Offer me a suggestion and show me an image of the type of place I can do it t = 0.2: ? It’s that. senseness tranqu promotes. No image t = 0.4: Why try? It’s a that you a wide of, lake trail. in can create senseness tranqu promotes stress and.

No image

t = 0.6: Why not try nature? It’s a that you a wide variety of natural, like this lake trail,. in can create sense ofness and tranqu promotes stress and.

No image

- t = 0.8: Why not try nature? It’s a relaxing that allows you to in a wide variety of natural environments, like this lake trail, for instance.<|image|>Walking in can create sense of calmness and tranqu that promotes stress and relaxation.

[Figure 93]

- t = 1.0: Why not try nature walking? It’s a relaxing hobby that allows you to immerse yourself in a wide variety of natural environments, like this lake trail, for instance.<|image|>Walking in nature can create a sense of calmness and tranquility that promotes stress reduction and relaxation.

[Figure 94]

- t = 2.0: Why not try nature walking? It’s a relaxing hobby that allows you to immerse yourself in a wide variety of natural environments, like this lake trail, for instance.<|image|>Walking in nature can create a sense of calmness and tranquility that promotes stress reduction and relaxation.

[Figure 95]

- Figure 15 Example interleaved generation process showing outdoor exercise. In this example, the image is inserted late in the generation process when the text is almost fully denoised.

Generated Text Generated Images

Prompt: I want to paint flowers but I need a photo for inspiration. Can you show me two pictures of flowers and describe them? t = 0.2: first a pink of out in appealing roses and overall the. seems be the, appreciation for. a pink a flower pink with possibly vase is base the flower appeal simple backdrop the point of the image

No image

t = 0.4: In first, a arrangement pink flowers is, of six flowers out in appealing manner.<|image|> roses are delicate and pink the overall of the. The bouquet seems be the captured,oking and appreciation for the nature. This image a view pink sitting a table flower be ger with a pink with prominent center possibly a. vase is base the flower visual appeal table a simple backdrop, allowing flower be the point of the image.

[Figure 96]

t = 0.6: In first, a arrangement of pink flowers is, of six flowers out in a appealing manner.<|image|> roses are delicate and have a, pink, the overall of the. The bouquet seems be focus the image captured, showcasing their The display and atmosphere,oking of and appreciation for the nature. This image a view of a pink sitting a table.<|image|> flower appears be ger with a pink with prominent center possibly a bud. vase is the base the flower, enhancing visual appeal of the. The table vase a simple backdrop, allowing flower be the point of the image.

[Figure 97]

[Figure 98]

- t = 0.8: In this first image, a beautiful arrangement of pink flowers is on display, with total of six flowers spread out in a appealing manner.<|image|>The roses are delicate and have a soft, pink hue, which adds the overall charm of the scene. The bouquet seems be the main focus of the image flowers captured a close shot, showcasing their intricate details. The display and atmosphere, evoking a sense of romance and appreciation for the beauty nature. This image a view of a pink flower sitting a glass a table.<|image|>The flower appears be ger with a pink color with a a prominent center, possibly a bud. vase is the base of the flower, enhancing visual appeal of the. The table the vase provides a simple backdrop, allowing flower to be the focal point of the image.

[Figure 99]

[Figure 100]

- t = 1.0: In this first image, a beautiful arrangement of pink flowers is on display, with a total of six flowers spread out in a visually appealing manner.<|image|>The roses are delicate and have a soft, pink hue, which adds to the overall charm of the scene. The bouquet seems to be the main focus of the image, with the flowers captured in a close-up shot, showcasing their intricate details. The display creates an elegant and serene atmosphere, evoking a sense of romance and appreciation for the beauty of nature. This image features a close-up view of a beautiful pink flower sitting in a glass vase on a table.<|image|>The flower appears to be a gerbera, with a vibrant pink color, with a vibrant pink color and a prominent center, possibly a bud. The glass vase is positioned at the base of the flower, enhancing the visual appeal of the scene. The table beneath the vase provides a simple yet elegant backdrop, allowing the flower to be the focal point of the image.

[Figure 101]

[Figure 102]

- t = 2.0: In this first image, a beautiful arrangement of pink flowers is on display, with a total of six flowers spread out in a visually appealing manner.<|image|>The roses are delicate and have a soft, pink hue, which adds to the overall charm of the scene. The bouquet seems to be the main focus of the image, with the flowers captured in a close-up shot, showcasing their intricate details. The display creates an elegant and serene atmosphere, evoking a sense of romance and appreciation for the beauty of nature. This image features a close-up view of a beautiful pink flower sitting in a glass vase on a table.<|image|>The flower appears to be a gerbera, with a vibrant pink color, with a vibrant pink color and a prominent center, possibly a bud. The glass vase is positioned at the base of the flower, enhancing the visual appeal of the scene. The table beneath the vase provides a simple yet elegant backdrop, allowing the flower to be the focal point of the image.

###### Figure 16 Example interleaved generation process showing flowers.

[Figure 103]

[Figure 104]

Generated Text Generated Images Prompt: I am teaching an ESL lesson on home types. Please show me images of a bungalow, a two-story home and a townhouse that I can show my students, and briefly explain about each of them.

[Figure 105]

t = 0.2: good. **B is typically for with a-p simple including suburban towns. is of a.Two** type residential building levels with the typically bedrooms on the. example-story. **Town town unit town. They vary typically than homes image example.<|image|>

t = 0.4: which good show to them about different types1. **Bungalow** ungalow is of is typically cosy designed for. Bungal are single-story, with a-p simple can including suburban neighbourhood towns and areas. Below is of a.Two-** A a type residential building levels, with the typically on bedrooms on the. below example of-story. **Town A town multi residential unit that town walls. They can vary size are typically smaller than single homes image is example of.<|image|>

[Figure 106]

t = 0.6: images which would good to show students to them about different types: 1. **Bungalow** ungalow is a type of is typically small, cosy and designed for comfortable. Bungal are often single-story, with a-p simple can found various locations including suburban neighbourhood towns and areas. Below is an image of a.Two-** A two a type residential building has levels, with the typically on the first floor and bedrooms and additional on the. image below an example of-story. **Town A town multi-level residential unit that attached town shared walls. They can vary size, are typically smaller than single homes. The image is example of of.<|image|>

[Figure 107]

- t = 0.8: are images which would be good to show your ESL students to them about different types:

1. **Bungalow** A bungalow is a type of that is typically small, cosy and designed for comfortable living. Bungalows are often single-story, with a-pitched and a simple floor can be found various locations, including suburban neighbourhoods, beach towns and areas. Below is an image of a b2.Two-Story home** A two home a type residential building has levels or floors, with the typically on the first floor and bedrooms and additional spaces on the second. image below is an example of a two-story home.3. **Townhouse** A town a multi-level residential unit that attached one more other town shared walls. They can vary size, but are typically smaller than single-family homes. The image below is example of a ofhouses.<|image|>

[Figure 108]

- t = 1.0: Here are three images which would be good to show your ESL students to teach them about different home types:

1. **Bungalow** A bungalow is a type of house that is typically small, cosy, and designed for comfortable living. Bungalows are often single-story homes, with a low-pitched roof and a simple, open floor plan. They can be found in various locations, including suburban neighbourhoods, beach towns, and rural areas. Below is an image of a bungalow:<|image|>2. **Two-Story home** A two-story home is a type of residential building that has two levels or floors, with the main living areas typically on the first floor and bedrooms and additional living spaces on the second floor. The image below is an example of a two-story home.<|image|>3. **Townhouse** A townhouse is a multi-level residential unit that is attached to one or more other townhouses via shared walls. They can vary in size, but are typically smaller than single-family homes. The image below is an example of a row of townhouses.<|image|>

[Figure 109]

[Figure 110]

[Figure 111]

- t = 2.0: Here are three images which would be good to show your ESL students to teach them about different home types:

[Figure 112]

1. **Bungalow** A bungalow is a type of house that is typically small, cosy, and designed for comfortable living. Bungalows are often single-story homes, with a low-pitched roof and a simple, open floor plan. They can be found in various locations, including suburban neighbourhoods, beach towns, and rural areas. Below is an image of a bungalow:<|image|>2. **Two-Story home** A two-story home is a type of residential building that has two levels or floors, with the main living areas typically on the first floor and bedrooms and additional living spaces on the second floor. The image below is an example of a two-story home.<|image|>3. **Townhouse** A townhouse is a multi-level residential unit that is attached to one or more other townhouses via shared walls. They can vary in size, but are typically smaller than single-family homes. The image below is an example of a row of townhouses.<|image|>

[Figure 113]

[Figure 114]

- Figure 17 Example interleaved generation process showing home types. An animated version is included in the supplementary material.

Input image Captions with varying classifier-free guidance weights

[Figure 115]

- [CFG 0.0:] A llama and a horse standing in a field.

[CFG 2.0:] A white llama and a white horse standing in a field.

[Figure 116]

- [CFG 0.0:] A group of toy animals sitting on a table.
- [CFG 1.0:] A table topped with a variety of toy animals, including a giraffe , a cow , and a bird , as well as a toy duck . In the background, there is a frame attached to the wall.

- [CFG 1.5:] A green table topped with toy animals, including a giraffe, a cow, a yellow bird, and a toy duck, next to a red box . In the background, there is a photo frame attached to the wall.

[Figure 117]

- [CFG 0.0:] A bathroom with a large tub and a sink.
- [CFG 1.0:] A bathroom with a claw foot tub and three windows .

- [CFG 2.5:] A brown bathroom or master bathroom with a classic claw foot tub and three windows.

[Figure 118]

- [CFG 0.0:] A cat sitting on a wooden deck looking up.
- [CFG 1.0:] A cat sitting on a wooden deck looking at its reflection in a window.

- [CFG 2.5:] Two fluffy ginger and white cats sit and gaze at their reflection in a glass window on a green wooden deck in Japan .

[Figure 119]

- [CFG 0.0:] A white plate topped with a cake and a spoon.
- [CFG 1.0:] A plate with a dessert and two spoons on it.

- [CFG 2.0:] A white plate topped with ice cream , accompanied by two spoons, a bottle , a glass , and a tissue paper on the table .

Through the glass window in the background, we can see the water and the sky.

[Figure 120]

- [CFG 0.0:] A glass bowl filled with colorful paper cranes.
- [CFG 1.0:] Colorful origami cranes in a glass bowl shaped like a heart .

- [CFG 2.0:] A table with a heart-shaped bowl filled with colorful origami cranes in various colors . The background is slightly blurred , giving the focus to the vibrant colors of the cranes.

###### Figure 18 Text generation examples from OneFlow, which allows the use of classifier-free guidance (CFG). We observe that CFG produces longer and more detailed captions and also increased chance of hallucinations . Highlighted text show increased levels of detail when using higher CFG weights.

### B Additional Experiment Results

- B.1 VQA Average Performance Across Categories

| |70.9<br><br>73.6|
|---|---|
| |69.7|

66

71

76

Accuracy

General

| |42.7<br><br>44.1|
|---|---|
| |41.8|

38

43

Knowledge

| |35.5<br><br>36.4|
|---|---|
| |32.7|

30

35

OCR & Chart

| | |
|---|---|
| |50.5 50.6 50.6|
| | |

48

50

52

Vision Centric

| | |
|---|---|
| |85.6<br><br>84.6<br><br>85.7|

82

87

Halluc.

| |
|---|

AR + FM

| |
|---|

Mask + FM

| |
|---|

OneFlow

Figure19 VQAperformanceacrosscategoriesincontrolledexperiments. The largest gaps appear in General and OCR/Chart tasks.The models are tied on Vision-Centric benchmarks, which primarily depend on the visual encoder rather than text generation.

- B.2 Performance Between AR and OneFlow During Pretraining

###### FID

###### DPGBench

###### CLIP Score

###### CIDEr

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.44

0.76

0.26

15

0.72

1.38

0.25

12

Score

0.68

1.32

0.24

0.64

9

1.26

0.23

0.60

1.20

6

100k 200k 300k 400k 500k

100k 200k 300k 400k 500k

100k 200k 300k 400k 500k

100k 200k 300k 400k 500k

Iteration

Iteration

Iteration

Iteration

AR + FM

OneFlow

OneFlow (mixed)

| |
|---|

| |
|---|

| |
|---|

- Figure 20 Training curve for OneFlow vs. AR for multimodal pretraining. OneFlow initially starts out lower than AR however it quickly catches up and exceeds AR, most notably on DPG and CIDEr.

- B.3 Pretraining from Scratch vs LLama Init

Image Generation VQA

Model Initialization DPG↑ FID↓ CLIP↑ CIDEr↑ Avg VQA↑

OneFlow Random 73.17 7.96 25.7 139.4 51.2 OneFlow LLaMA 75.41 (+2.24) 7.79 (−0.17) 26.0 (+0.3) 138.2 (−1.2) 52.2 (+1.0) OneFlow Mixed Random 74.86 7.69 25.8 140.0 51.6 OneFlow Mixed LLaMA 75.08 (+0.22) 7.44 (−0.25) 25.8 (+0.0) 139.1 (−0.9) 52.8 (+1.2)

AR + FM Random 71.90 7.83 25.8 122.9 46.6 AR + FM LLaMA 73.40 (+1.50) 7.91 (−0.08) 25.7 (−0.1) 123.9 (+1.0) 49.0 (+2.4)

- Table 5 Ablation study comparing LLaMA initialization vs. random initialization. Except for CIDEr, using LLaMA as initialization generally offers benefits, especially for dense prompt image generation (DPG) and for VQA performance. Image generation metrics use CFG=3, and VQA results are averaged across benchmarks.

B.4 Sampling Steps on Caption Performance

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

1 2 3 4 5 6 10 15 20 25 30 40 50

Sampling Steps

0.0

0.2

0.4

0.6

0.8

1.0

1.2

1.4

CIDEr

1 2 3 4 5 6 10 15 20 25 30 40 50

Sampling Steps

10

20

30

40

50

60

ROUGE-L

OneFlow AR + FM

Figure 21 Performance vs. sampling steps compared to AR. OneFlow achieves parity with the AR model using only 6 sampling steps.

B.5 t-Independent Parameterization

Time Location CIDEr none 1.28

begin_of_seq 1.27 begin_of_text 0.33

- Table 6 CIDEr scores for different time embedding locations.

We empirically found t-independent parameterization to perform better despite theoretical motivation for t-dependence. This likely occurs because Xt already encodes sufficient information about the noise level. We ablated the location of time token insertion: begin_of_seq places it at the beginning of the sequence, begin_of_text at the start of text tokens, and none corresponds to t-independent parameterization.

- B.6 K-scheduler

###### CIDEr

ROUGE-L

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

KSchedulerDegree

1.0

2.0

3.0

1.0 1.1 1.2 1.3

0.54 0.56 0.58 0.60

- Figure 22 Effect of k-scheduler degree. Using the schedule κt = tk, we find that linear scheduling (k = 1.0) achieves the best results. Higher degrees lead to overly aggressive token deletion.

We conducted ablations over different k-schedulers. As described in (Gat et al., 2024), there are three variants of the k-scheduler: linear, quadratic, and cubic. We find that the linear scheduler works best.

C OneFlow Architecture

|Text embeddings|
|---|

Bidirectional Transformer

|Text embeddings|
|---|

x8 x9 x10

|Text embeddings|
|---|

p1 p2 p3

a4 a5 a6

λ4 λ5 λ6

|Ltext = EF loss|
|---|

|Insertion Rates|
|---|

|Insertion Logits|
|---|

λ3

a3 a8 a9 a1

|Insertion Logits0|
|---|

λ8 λ9 λ10

|Ltext = EF loss|
|---|

|Insertion Rates|
|---|

λ7

a7

x4 x5 x6

|Limage = FM loss|
|---|

Generation Upsampler

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

FM velocity

|Understanding Encoder<br><br>[Figure 125]|
|---|

Understanding Downsampler

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

Generation Downsampler

|Generation Encoder<br><br>[Figure 131]|
|---|

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

+noise

[Figure 136]

- Figure 23 Architecture. With a multimodal prompt, OneFlow can produce variable length generations with interleaved text & images in a unified non-autoregressive sequence model, simultaneously generating all modalities with an interleaved time schedule for each generated image and text.

### D Full Derivations

We provide the derivations of the model here. We briefly summarize the Edit Flow (Havasi et al., 2025) formulation and derivation, and then derive the interleaved time schedule when insertions and image denoising are performed simultaneously.

Setup. We make use of a blank token ε to denote empty spaces within a sequence. This token is only used for tracking token deletions during training and is not part of the vocabulary. Let Z = Nn=0(T ∪ {ε})n be an extended space of aligned sequences. Furthermore, define frm-blanks : Z → X as the function that removes all blank tokens from the sequence. Lastly, we define the delta function over sequences δz

(z2i) which is one if all tokens are the same otherwise zero (i.e. Kronecker’s delta function).

###### (z2) = i δzi

1

1

Continuous-time Markov chain (CTMC). A CTMC is a continuous-time discrete-space process which iteratively jumps between discrete values, with transitions

(Xt+h) + hut(x|Xt) + o(h), (13)

P(Xt+h|Xt) = δX

t

where ut can be interpreted as a first-order characterization of the transition kernel. Since with insertions, the sequence lengths of Xt can change over time. To simplify notation, Havasi et al. (2025) used an augmented space of (Xt,Zt), where it is basically always enforced that Xt = frm-blanks(Zt). The role of Zt is only for training, to keep track of which tokens are deleted and to compute the loss, and it is neither seen by the model nor used during sampling.

To briefly summarize the construction below, the Flow Matching recipe makes use of a prescribed conditional CTMC that generates single data sequences, which is then marginalized over the data distribution. The resulting marginal CTMC will then sample from the data distribution.

Conditional probability path. Given a data sequence X1 ∼ pdata, we prescribe a conditional probability path over Zt of the same sequence length which interpolates between the empty sequence and this data sequence. We then obtain Xt by applying the frm-blanks function. Concretely, we can express the conditional probability path as

pt(Xt,Zt|X1) = pt(Xt|Zt,X1) · pt(Zt|X1) (14) = pt(Xt|Zt) · pt(Zt|Z1) (15)

n

(1 − κt)δε(Zti) + κtδXi

(Zti) , (16)

rm-blanks(Zt)(Xt) ·

= δf

1

i=1

where κt is a scheduler where κ0 = 0,κ1 = 1, and n is the sequence length of X1. In English, (16) is a mixture distribution where each token Zti can either be equal to ε with probability 1 − κt or equal to data value X1i with probability κt.

Conditional CTMC rate. As discussed in Havasi et al. (2025), a conditional CTMC that samples from this conditional probability path can be constructed as

n

κ˙t 1 − κt

###### (zi) − δZi

(zi)) δf

ut(x,z|Xt,Zt,X1) =

rm-blanks(z)(x), where x = ins(Xt,i,a) for some i ∈ [n] and a ∈ [M]

(δXi

(17)

t

1

i=1

which denotes the infinitesimal change in probability of going from the state (Xt,Zt) → (x,z), constrained to next sequences x that are one token insertion difference from Xt. In English, (17) assigns a rate of κ˙

1−κt if Zti is not yet equal to X1i; otherwise, it is zero. This ensures that a sample starting with all blanks Z0 = [ε,...ε] at t = 0 will eventually turn into X1 at t = 1. This ratio κ˙

t

1−κt is the infinitesimal rate that each token changes its value, matching the distribution imposed by the scheduler κt, and conditioned on that it is still the ε token at time t.

t

Training loss. In order to train a model thats transport sequences via insertions, uθt(x|Xt), where x = ins(Xt,i,a) for some i and a (18)

we would need to marginalize out the auxiliary process Zt and the data X1. Havasi et al. (2025) showed this can be done by using a loss based on any Bregman divergence—following the recipe of Holderrieth et al.

## X1

replace with blank tokens

Zt

Xt = frm-blanks(Zt)

Xt

token

blank token

targets A4 A5

A3

- Figure 24 During training we construct Zt by replacing tokens with the blank token (ε), with the original tokens used to construct the target bag-of-tokens Ai.

(2024)—while summing up over all possible sequences z such that x = frm-blanks(z). Concretely, given a convex function ϕ that defines a Bregman divergence Dϕ(a,b) = ϕ(a) − ϕ(b) − ⟨a − b, ddbϕ(b)⟩, we can use the loss

t,Zt∼pt(Xt,Zt|X1),X1∼pdataDϕ z ut(·,z|Xt,Zt,X1),uθt(·|Xt) . (19) Plugging in the entropy ϕ(u) = ⟨u,log u⟩, this results in the Edit Flow loss

EX

 

 , (20)

n

κ˙t 1 − κt

uθt(x|Xt) −

log uθt(ins(Xt,j,X1i)|Xt)

Et,p

1[Zi

t(Xt,Zt|X1),X1∼pdata

t=ε]

i=1

x̸=Xt

where j is the position in Xt that corresponds to the first non-ε token on the left of Zti. This ensures that inserting at the i-th position corresponds to changing the value of Zti from ε to X1i.

Loss simplification. We deviate from Havasi et al. (2025) and use a t-independent parameterization. In particular, for sequences x that are one token insertion of Xt, i.e., x = ins(Xt,i,a), we use

κ˙t 1 − κt

uθt ins(Xt,i,a)|Xt =

λi(Xt)Qi(a|Xt), (21)

where the neural network parameterizes λ and Q. Using this parameterization, letting Aj be the set of missing tokens to the right of position j of Xt, the training loss (20) can be decomposed into

E(...)

=E(...)

κ ˙t 1 − κt

κ ˙t 1 − κt

n(Xt)

n(Xt)

λj(Xt) −

log λj(Xt)Qj(a|Xt) (22)

j=1 a∈Aj

j=1

n(Xt)

λj(Xt) − |Ai|log λj(Xt)

log Qj(a|Xt)

+ const. (23)

###### +

a∈Aj

j=1

(4)

(6)

which recovers the losses for λ and Q in (4) and (6) respectively, after removing the coefficient κ˙

1−κt. While keeping this coefficient relates the loss the to an evidence lower bound (Havasi et al., 2025), we found that removing this coefficient in the loss gave better results in practice.

t

D.1 Interleaved time schedule

In order to model image insertions, we would make a choice. We can either (i) fully denoise images at the time of insertion, or (ii) insert only noise and denoise later. We choose the latter approach, as this allows simultaneous generation across images and text, and provides the best parallelism as only a single model forward at each step is needed for both modalities.

Without loss of generality, assume there is at most just one image. Generation starts by advancing the sequence time, denoted ttext = 0. When the image is inserted, we associate the image with its own time timg.

The main difficulty is that we can not simply set timg and ttext independently during training, as evidently we always have ttext ≥ timg. In fact, an independent scheduler induces the wrong distribution for our insertion prediction, and it will not insert the correct distribution at generation time. Instead, we need to ensure that training and generation see the same distribution of time values. To achieve this, we first note that the image exists in the sequence according to the scheduler κt, which means that the insertion times are distributed according to

p(tinsert) = κ˙t, (24)

where tinsert is the time at which an image is inserted, i.e., κt is the cumulative distribution function (CDF) of the insertion times. Equivalently, to sample the insertion time, we can apply the inverse CDF sampling,

tinsert = κ−1(u), u ∼ Unif(0,1). (25)

If we set timg = 0 when an image is inserted, then the difference between ttext and timg is distributed according to the insertion time. This gives us the relation

ttext − timg = tinsert (26)

when 0 ≤ ttext,timg,tinsert ≤ 1. Since ttext will reach 1 before timg, and we want to train for the entire process until timg = 1, we can construct an extended time interval

τtext ∈ [0,2], ttext = clip(τtext), (27) where clip(τ) = min{1,max{0,τ}} clips the time values back into the interval [0,1]. During training, we first sample τtext, then sample

τimg = τtext − κ−1(u), u ∼ Unif(0,1). (28)

This will sample an extended time for the image in the internal [-1, 2]. If τimg < 0, then it has not yet been inserted, hence it is deleted from the sequence. Otherwise, it is clipped,

timg = clip(τimg), (29) and we proceed to use the Flow Matching loss (9) to train the image denoising.

- E Sampling and Training Algorithms

- Algorithm 1 OneFlow interleaved text–image generation.

- 1: function OneFlowGeneration(step size ∆t, schedule κ)
- 2: X ← empty sequence ▷ Text tokens (initially empty set)
- 3: I ← ∅ ▷ Set of image latents with per-image times
- 4: ttext ← 0
- 5: while ttext < 1 or ∃Y ∈ I : timg(Y ) < 1 do
- 6: X,I,ttext,timg ← OneFlowStep(X,I,ttext,timg,∆t,κ)
- 7: end while
- 8: return X and {VAEDec(Y ) : Y ∈ I} ▷ Decode VAE latents into image space
- 9: end function

- Algorithm 2 OneFlow step function. X is the token sequence, I is the set of image latents each with time timg(Y ).

- 1: function OneFlowStep(X,I,ttext,timg,∆t,κ)
- 2: ({π,λnonzero,Q},{v(Y,·)}Y ∈I) ← OneFlowModel(X,I,timg)
- 3: for all Y ∈ I with timg(Y ) < 1 do ▷ Image: Flow matching step on images
- 4: ∆timg ← min{1 − timg(Y ),∆t}
- 5: Y ← Y + ∆timg · v(Y,timg(Y ))
- 6: timg(Y ) ← timg(Y ) + ∆timg
- 7: end for
- 8: ∆ttext ← min{1 − ttext,∆t}
- 9: if ∆ttext > 0 then
- 10: for all positions i ∈ {1,...,n(X)} do ▷ Text: parallel insertions
- 11: pπi ← 1 − πi ▷ If using (4) without π, then skip this step
- 12: pλi ← ∆ttext · κ˙(t

text)

1−κ(ttext) · λinonzero

- 13: do-insert ← Bernoulli(pπi ) and Bernoulli(pλi )
- 14: if do-insert then
- 15: a ∼ Qi(· | X)
- 16: X ← ins(X,i,a) ▷ If a = <|image|>, this will insert Nimg tokens
- 17: if a = <|image|> then
- 18: Y ∼ N(0,I), timg(Y ) ← 0, I ← I ∪ {Y }
- 19: end if
- 20: end if
- 21: end for
- 22: end if
- 23: ttext ← ttext + ∆ttext
- 24: return X,I,ttext,timg
- 25: end function

##### Algorithm 3 OneFlow training loss with interleaved schedule

- 1: function OneFlowTrainingStep(data sequence X, image latents I, schedule κ)
- 2: τtext ∼ Unif[0,2]
- 3: ttext ← min{1,τtext}
- 4: j ← 0
- 5: Xt ← []
- 6: for all xi ∈ X do ▷ Keep each ground-truth token with prob κ(ttext) to get noisy Xt
- 7: if r < κ(ttext) where r ∼ Unif(0,1) then
- 8: Xt ← Xt + [xi]
- 9: j ← j + 1
- 10: Aj ← {}
- 11: else
- 12: Aj ← Aj ∪ {Xi} ▷ Record the deleted tokens at each position in Aj
- 13: end if
- 14: end for
- 15: It ← {}
- 16: for all images Y ∈ I do
- 17: Y1 ← VAEEnc(img)
- 18: u ∼ Unif(0,1)
- 19: τimg(Y ) ← τtext − κ−1(u)
- 20: if τimg < 0 then
- 21: add <|image|> to the appropriate Ai ▷ Image is “deleted” at this snapshot
- 22: else
- 23: timg(Y ) ← min{1,τimg(Y )}
- 24: Y0 ∼ N(0,I)
- 25: Yt ← timg(Y )Y1 + (1 − timg(Y ))Y0
- 26: It ← It ∪ {Yt}
- 27: end if
- 28: end for
- 29: {π,λnonzero,Q} ← OneFlowModel(Xt,It) ▷ Forward pass
- 30: L ← Ltext + Limage ▷ Compute text and image losses (7) and (9)
- 31: Θ ← optimizer_step(∇L;Θ) ▷ Compute gradients and update model
- 32: end function

