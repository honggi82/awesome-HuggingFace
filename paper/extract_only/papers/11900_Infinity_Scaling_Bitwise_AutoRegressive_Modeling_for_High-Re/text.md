arXiv:2412.04431v2[cs.CV]17Jun2025

# Infinity∞: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis

### Jian Han∗, Jinlai Liu∗, Yi Jiang∗, Bin Yan Yuqi Zhang, Zehuan Yuan†, Bingyue Peng, Xiaobing Liu ByteDance

{hanjian.thu123,liujinlai.licio,jiangyi.enjoy,bin.yan}@bytedance.com, {zhangyuqi.hi,yuanzehuan,bingyue.peng,will.liu}@bytedance.com,

#### Codes and models: https://github.com/FoundationVision/Infinity

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

- Figure 1: High-resolution image synthesis results from Infinity, showcasing its capabilities in precise prompt following, spatial reasoning, text rendering, and aesthetics across different styles and aspect ratios.

## Abstract

We present Infinity, a Bitwise Visual AutoRegressive Modeling capable of generating high-resolution, photorealistic images following language instruction. Infinity redefines visual autoregressive model under a bitwise token prediction framework with an infinite-vocabulary tokenizer & classifier and bitwise self-correction mechanism, remarkably improving the generation capacity and details. By theoretically scaling the tokenizer vocabulary size to infinity and concurrently scaling the transformer size, our method significantly unleashes powerful scaling capabilities compared to vanilla VAR. Infinity sets a new record for autoregressive text-to-image models, outperforming top-tier diffusion models like SD3-Medium and SDXL. Notably, Infinity surpasses SD3-Medium by improving the GenEval benchmark score from 0.62 to 0.73 and the ImageReward benchmark score from 0.87 to 0.96, achieving a win rate of 66%. Without extra optimization, Infinity generates a high-quality 1024×1024 image in 0.8 seconds, making it 2.6× faster than SD3-Medium and establishing it as the fastest text-to-image model. Models and codes will be released to promote further exploration of Infinity for visual generation and unified tokenizer modeling.

∗Equal contribution. †Corresponding author: yuanzehuan@bytedance.com

Preprint. Under review.

perturb

Continuous Feature

0.01 -8 -4.2 5.23

- -0.1 -8 -4.2 5.23

- -1 -1 -1 +1

slightly different Quantized Feature

Index

+1 -1 -1 +1 9 1

totally different

∈ ℎ×2 ∈ ℎ×2×

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16

+1 -1 +1 -1 +1 -1 +1 -1

Quantized -1 -1 +1 Feature

Index

9 Loss

+1

Loss Loss Loss Loss

- Figure 2: Visual tokenizer quantizes continuous features and then gets index labels. Conventional classifier (left) predicts 2d indices. Infinite-Vocabulary Classifier (right) predicts d bits instead. Slight perturbations to near-zero values in continuous features cause a complete change of index labels. Bit labels (i.e. quantized features) change subtly and still provide steady supervision. Besides, parameters of conventional classifiers grow exponentially as d increases, while IVC grows linearly. If d = 32 and h = 2048, the conventional classifier requires 8.8 trillion parameters, exceeding current compute limits. By contrast, IVC only requires 0.13M parameters.

## 1 Introduction

The visual generation[27, 52, 20, 48, 42] has recently witnessed rapid advancements, enabling highquality, high-resolution images and video synthesis[8, 21]. Text-to-image generation[50, 46, 45, 7, 43, 21] is one of the most challenging tasks due to its need for complex language adherence and intricate scene creation. Currently, visual generation is primarily divided into two main approaches: Diffusion models and AutoRegressive models.

Diffusion models[27, 52, 20, 43, 42, 21], trained to invert the forward paths of data towards random noise, effectively generate images through a continuous denoising process. AutoRegressive

- models[15, 22, 75, 61], on the other hand, harness the scalability and generalizability of language
- models[16, 2, 28, 62, 63, 70, 57, 3, 60] by employing a visual tokenizer[64, 47, 74] to convert images into discrete tokens and optimize these tokens causally, allowing image generation through next-token prediction or next-scale prediction. AutoRegressive models encounter significant challenges in highresolution text-to-image synthesis[75, 67]. They exhibit inferior reconstruction quality when utilizing discrete tokens as opposed to continuous tokens. Additionally, their generated visual contents are less detailed than those by diffusion models. The inefficiency and latency in visual generation, stemming from the raster-scan method of next-token prediction, further exacerbates these issues.

Recently, Visual AutoRegressive modeling (VAR)[61] redefined autoregressive learning on images as coarse-to-fine “next-scale prediction”. It demonstrates superior generalization and scaling capabilities compared to diffusion transformers while requiring fewer steps. VAR leverages the powerful scaling properties of LLMs [31, 25] and can simultaneously refine previous scale steps, benefiting from the strengths of diffusion models as well. However, the index-wise discrete tokenizer[64, 22, 79, 61, 37, 66, 69, 79] employed in AutoRegressive or Visual AutoRegressive models faces significant quantization errors with a limited vocabulary size resulting in difficulties in reconstructing finegrained details especially in high-resolution images. In the generation stage, index-wise tokens suffer from fuzzy supervision leading to visual detail loss and local distortions. Moreover, train-test discrepancies from teacher-forcing training, inherent to LLMs, amplify cumulative errors in visual details. These challenges make index-wise tokens a significant bottleneck for AutoRegressive models.

We propose a novel approach called bitwise modeling, which substitutes index-wise tokens with bitwise tokens throughout the process. Our bitwise modeling framework consists of three primary modules: bitwise visual tokenizer, bitwise infinite-vocabulary classifier, and bitwise self-correction. Inspired by the success and widespread adoption of binary vector quantization[76, 81], we scaled up the tokenizer vocabulary to 264, significantly surpassing all previous AutoRegressive model vocabularies[77, 55]. This expansion allows for reconstruction quality that far exceeds previous

discrete tokenizers, achieving results comparable to continuous VAEs[48], with scores improving from 0.87 to 0.33 on ImageNet-256 benchmark[19]. In Fig.2, we transform the conventional token prediction from a large integer into binary bits in a bitwise infinite-vocabulary classifier to address optimization and computation challenges, enabling the learning of massive vocabularies in Visual AutoRegressive models. Additionally, we incorporated bitwise self-correction during training by randomly flipping some bits to simulate prediction mistakes and re-quantizing the residual features, thus endowing the system with self-correcting capabilities. Our method, Infinity: Bitwise Visual AutoRegressive Modeling, maintains the scaling and speed advantages of Visual AutoRegressive modeling while achieving detailed reconstruction and generation quality comparable to that of continuous Diffusion models.

Infinity sets a new record for AutoRegressive models, and also surpasses leading diffusion models including SDXL[43], PixArt-Sigma[12],DALL-E3[7] and Stable-Diffusion 3[21] on several challenging text-to-image benchmarks. Notably, Infinity surpasses SD3 by improving the GenEval benchmark score from 0.62 to 0.73, ImageReward benchmark score from 0.87 to 0.96, HPSv2.1 benchmark score from 30.9 to 32.3, achieving a win rate of 66% for human evaluation and a 2.6× reduction in inference latency with the same model size. Specifically, Infinity shows powerful scaling laws for image generation capabilities by scaling up the image tokenizer vocabulary size and the corresponding transformer size. As the image tokenizer and transformer sizes increase, the content and details of high-quality image generation show significant improvement.

In summary, the contributions of our work are as follows:

- 1. We propose Infinity, an autoregressive model with Bitwise Modeling, which significantly improves the scaling and visual detail representation capabilities of discrete generative models. We believe this framework opens up new possibilities of ‘infinity’ for the discrete generation community.
- 2. Infinity demonstrates the potential of scaling tokenizers and transformers by achieving near-continuous tokenizer performance with its image tokenizer and surpassing diffusion models in high-quality text-to-image generation.
- 3. Infinity enables a discrete autoregressive text-to-image model to achieve exceptionally strong prompt adherence and superior image generation quality, while also delivering the fastest inference speed.

## 2 Related Work

### 2.1 AutoRegressive Models

AutoRegressive models, leveraging the powerful scaling capabilities of LLMs[44, 9, 16, 62, 63], use discrete image tokenizers[64, 47, 22] in conjunction with transformers to generate images based on next-token prediction. VQ-based methods [64, 47, 22, 35, 55] employ vector quantization to convert image patches into index-wise tokens and use a decoder-only transformer to predict the next token index. However, these methods are limited by the lack of scaled-up transformers and the quantization error inherent in VQ-VAE[64], preventing them from achieving performance on par with diffusion models. Parti [75], Emu3 [67], chameleon[59], loong[68] and VideoPoet[32] scaled up autoregressive models in text-to-image or video synthesis. Inspired by the global structure of visual information, Visual AutoRegressive modeling(VAR) redefines the autoregressive modeling on images as a next-scale prediction framework, significantly improving generation quality and sampling speed. HART[58] adopted hybrid tokenizers based on VAR. Fluid[23] proposed random-order models and employed a continuous tokenizer rather than a discrete tokenizer.

### 2.2 Diffusion Models.

Diffusion models have seen rapid advancements in various directions. Denoising learning mechanisms [27, 41] and sampling efficiency [53, 52, 38, 39, 4] have been continuously optimized to generate high-quality images. Latent diffusion models [48] is the first to propose modeling in the latent space rather than the pixel space for diffusion[50]. Recently, latent diffusion models[18, 21] have also adopted scaling up VAE to improve the representation in the latent space. DiT [42] and U-Vit[5] employ a more scalable transformer to model diffusion, achieving superior results. Consequently, mainstream text-to-image diffusion models[21, 7, 14] have adopted the DiT architecture. DiT also inspire the text-to-video diffusion models[6, 8]

[Figure 13]

- -2 +3 +1.6 -0.1 -4.7

- -1 +1 +1 -1 -1

- -1 +1 -1 -1 -1

- -1 +2 +2.6 +0.9 -3.7

- -1 +1 +1 +1 -1

Bitwise Self-Correction

Quant

...

1 ... 8 9 1 ... 8 9

1

[Figure 14]

###### GT

1

1 3 4 1 ... 3 4

1

Random Flip bits

IVC

풇풍  

###### Pred

1

Residual

[Figure 15]

− 풇풍  

Causal VAR Transformer

Quant

1 ... 3 4 1 ... 8 9

Text

S

1

Random Flip bits

풇풍  

+1 +1 +1 -1 -1

###### ...

Bitwise Self-Correction

Bit Label Predict Token

+1.6 Continuous Feature

+1 -1

1

i i

Token

[Figure 16]

[Figure 17]

- -1-

[Figure 18]

[Figure 19]

- -1

[Figure 20]

-1

[Figure 21]

- -1

[Figure 22]

-1

[Figure 23]

- -1

[Figure 24]

-1

[Figure 25]

- -1

[Figure 26]

-1

[Figure 27]

- -1

[Figure 28]

-1

[Figure 29]

- -1 -1

[Figure 30]

[Figure 31]

-1

[Figure 32]

- -1 -1

[Figure 33]

[Figure 34]

-1

[Figure 35]

-1

[Figure 36]

- -1

[Figure 37]

-1

[Figure 38]

- -1 -1

[Figure 39]

[Figure 40]

-1

[Figure 41]

- -1 -1

[Figure 42]

[Figure 43]

-1

[Figure 44]

-1

[Figure 45]

- -1

[Figure 46]

+1

[Figure 47]

- -1 -1

[Figure 48]

[Figure 49]

-1 +1 +1 -1

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

-1

[Figure 54]

+1

[Figure 55]

-1

[Figure 56]

+1

[Figure 57]

- -1 -1

###### i = +1 -1 ... +1

next-scale prediction with bit labels

- Figure 3: Framework of Infinity. Infinity introduces bitwise modeling, which incorporates a bitwise multi-scale visual tokenizer, Infinite-Vocabulary Classifier (IVC), and Bitwise Self-Correction. When

predicting Rk, the sequence (R1,R2,...,Rk−1) serves as the prefixed context and the text condition guides the prediction through a cross attention mechanism. Different from VAR, Infinity performs next-scale prediction with bit labels.

### 2.3 Scaling models

Scaling laws in autoregressive language models reveal a power-law relationship between model size, dataset size, and compute with test set cross-entropy loss [31, 25, 1]. These laws help predict larger model performance, leading to efficient resource allocation and ongoing improvements without saturation [9, 62, 63, 80, 70, 28]. This has inspired research into scaling in visual generation [56, 78, 61, 21, 8]

## 3 Infinity Architecture

### 3.1 Visual AutoRegressive Modeling

Infinity incorporates a visual tokenizer and a transformer for image synthesis. During the training stage, a sample consists of a text prompt t and a ground truth image I. The proposed visual tokenizer first encodes the image I into a feature map F ∈ Rh×w×d with stride s and then quantize the feature map F into K multi-scale residual maps (R1,R2,...,RK). The resolution of Rk is hk × wk and it grows larger gradually from k = 1 → K. Based on this sequence of residuals, we can gradually approximate the continuous feature F as in Eq.1

k

up(Ri,(h,w)) (1)

Fk =

i=1

Here up(·) means bilinear upsampling and Fk is the cumulative sum of the upsampled R≤k.

Subsequently, transformer learns to predict residuals R of the next scale conditioned on previous predictions and the text input in an autoregressive manner. Formally, the autoregressive likelihood can be formulated as:

p(R1,...,RK) =

K

p(Rk | R1,...,Rk−1,Ψ(t)), (2)

k=1

where Ψ(t) is the text embeddings from Flan-T5 [17] model. (R1,...,Rk−1,Ψ(t)) serves as the prefixed context When predicting Rk. Besides, the text embeddings Ψ(t) further guide the prediction through a cross attention mechanism. In particular, as shown in Fig. 3, the text embeddings

Ψ(t) ∈ RL×C is projected into a ⟨SOS⟩ ∈ R1×1×h as the input of the first scale, where h is the hidden dimension of transformer. The transformer is required to predict R1 based on ⟨SOS⟩ in the first scale. In the latter k-th scale, to match the spatial size of the input and the output label Rk, we take the downsampled feature Fk−1 from the last scale k − 1 as the input to predict Rk in parallel.

Fk−1 = down(Fk−1,(hk,wk)), (3)

where down(·) is bilinear downsampling and the spatial size of both Fk−1 and Rk is (hk,wk). Refer to Alg.1 for detailed procedure to obtain binary quantization results and transformer’s inputs. In

previous index-wise [61] representations, the shape of prediction is (hk,wk,Vd). Vd is the vocabulary size of the visual tokenizer. For binary quantization [76, 81] with code embedding dimension d,

Vd = 2d. When d is large, the required computational resource grows unaffordable.

The transformer consists of a stack of repeated blocks, where each block includes RoPE2d [26], SelfAttention, Cross Attention, and FFN layers. The text embeddings Ψ(t) provide effective guidance for image synthesis in each cross-attention layer. During the training stage, we exploit a block-wise causal attention mask to ensure that the transformer can only attend to its prefixed context, i.e.,

(⟨SOS⟩, F1,..., Fk−1), when predicting Fk. During the inference stage, we perform KV-Caching to speed up inference and there’s no need for masking.

### 3.2 Visual Tokenizer

Increasing the vocabulary size has significant potential for improving reconstruction and generation quality. However, directly enlarging the vocabulary in existing tokenizers[77, 61] leads to a substantial increase in memory consumption and computational burden. To address these challenges and fully exploit the potential of discrete tokenizers, this paper proposes a new bitwise multi-scale residual quantizer, which significantly reduces memory usage, enabling the training of extremely large vocabulary, e.g. 264.

Bitwise Multi-scale Residual Quantizer. We replace the original vector quantizer of VAR [61] with a dimension-independent bitwise quantizer. In this paper, we consider two candidates, LFQ [77] and BSQ[81]. Given K scales in the multi-scale quantizer, on the k-th scale, the input continuous residual vector zk ∈ Rd are quantized into binary output qk as shown below.

sign(zk) if LFQ √1

(4)

qk = Q(zk) =

dsign( z

|zk|) if BSQ

k

To encourage codebook utilization, an entropy penalty Lentropy = E[H(q(z))] − H[E(q(z))] [30] is adopted, where H(·) represents the entropy operation. To obtain the distribution of q(z), we need to compute the similarities between the input z and the whole codebook when using LFQ. However, this leads to unaffordable space and time complexity of O(2d). When the codebook dimension d becomes large, e.g. 20, an out-of-memory (OOM) issue is faced as shown in Tab. 3. By contrast, since both input and output in BSQ are unit vectors, BSQ[81] proposes an approximation formula for the entropy penalty, reducing the computational complexity to O(d). As shown in Tab 3, there is no obvious increase in memory consumption for BSQ even when codebook size is 264. Unless otherwise stated, we adopt BSQ by default.

### 3.3 Infinite-Vocabulary Classifier

The visual tokenizer obtains discrete labels by quantizing residual features. Consequently, the transformer predicts next-scale residual features’ labels yk ∈ [0,Vd)h

k×wk and optimizes the target through the cross-entropy loss. Previous works [61, 76] directly predict these index labels using a classifier of Vd classes. However, it suffers from two drawbacks, huge computational costs and fuzzy supervision.

As illustrated in Section 3.2, we exploit a bitwise VQ-VAE as the visual tokenizer, where the vocabulary size Vd is extremely large. For example, if Vd = 232 and h = 2048, a conventional classifier would require a weight matrix W ∈ Rh×V

d of 8.8 trillion parameters, which exceeds the limits of current computational resources.

Moreover, VQ-VAE exploits the sign function during quantization as in Eq.4. After that, the positive elements are multiplied with the corresponding base and summed to get the index label yk(m,n) as in Eq.5, where m ∈ [0,hk) and n ∈ [0,wk).

d−1

yk(m,n) =

p=0

Rk(m,n,p)>0 · 2p (5)

Owing to the merits of the quantization method, slight perturbations to those near-zero features cause a significant change in the label. As a result, the conventional index-wise classifier [61, 11, 77] is difficult to optimize.

To address the problems in computation and optimization, we propose Infinite-Vocabulary Classifier (IVC). As shown in Fig.2, instead of using a conventional classifier with Vd classes, we use d binary classifiers in parallel to predict if the next-scale residual Rk(m,n,p) is positive or negative, where d = log2(Vd). The proposed Infinite-Vocabulary Classifier is much more efficient in memory and parameters compared to the conventional classifier. When Vd = 216 and h = 2048, it saves 99.95% parameters and GPU memory. Besides, when there exist near-zero values that confuse the model in some dimensions, the supervision in other dimensions is still clear. Therefore, compared with conventional index-wise classifiers, the proposed Infinite-Vocabulary Classifier is easier to optimize.

### 3.4 Bitwise Self-Correction

Weakness of teacher-forcing training. VAR [61] inherits the teacher-forcing training from LLMs. However, next-scale prediction in vision is quite different from next-token prediction in language. Specifically, we cannot decode the complete image until residuals Rk from all scales are obtained. We find that the teacher-forcing training brings about severe train-test discrepancy for visual generation. In particular, the teacher-forcing training makes the transformer only refine features in each scale without the ability to recognize and correct mistakes. Mistakes made in former scales will be propagated and amplified in latter scales, finally messing up generated images (left images in Fig.12).

In this work, we propose Bitwise Self-Correction (BSC) to address this issue. In particular, we obtain Rkflip via randomly flipping the bits in Rk with a probability uniformly sampled from [0,p], imitating different strengths of errors made in the prediction of the k-th scale.

Here comes the key component of bitwise self-correction. Rkflip contains errors while Rk doesn’t. After replacing Rk with Rkflip as predictions on the k-th scale, we recompute the transformer input Fk. Besides, re-quantization is performed to get a new target Rk+1. The whole process of bitwise self-correction is illustrated in Alg.2. We also provide a simplified illustration in Fig.3 (right) for better understanding. Notably, BSC is accomplished by revising the inputs and labels of the transformer. It neither adds extra computational cost nor disrupts the original parallel training characteristics.

Each scale undergoes the same process of random-flipping and re-quantization. The transformer takes partially randomly flipped features as inputs, taking the prediction errors into consideration. The re-quantized bit labels enable the transformer to autocorrect errors made in former predictions. In such way, we address the train-test discrepancy caused by teacher-forcing training and empower Infinity to have the self-correction ability.

Algorithm 1 Visual Tokenizer Encoding Input: raw feature F, scale schedule

Algorithm 2 Encoding with BSC Input: raw feature F, random flip ratio p, scale

schedule {(hr1,w1r),...,(hrK,wKr )}, Rqueue = [], Fqueue = [] for k = 1,2,··· ,K do

{(hr1,w1r),...,(hrK,wKr )}

Rqueue = [] ▷ multi-scale bit labels Fqueue = [] ▷ inputs for transformer for k = 1,2,··· ,K do

Rk=Q(down(F − Fkflip−1,(hk,wk))) Queue_Push(Rqueue,Rk)

Rk = Q(down(F − Fk−1,(hk,wk)) Queue_Push(Rqueue,Rk)

Rkflip = Random_Flip(Rk,p) Fkflip = ki=1 up(Riflip,(h,w)) Fk = down(Fkflip,(hk+1,wk+1)) Queue_Push( Fqueue, Fk)

Fk = ki=1 up(Ri,(h,w)) Fk = down(Fk,(hk+1,wk+1)) Queue_Push( Fqueue, Fk)

end for Output: Rqueue, Fqueue

end for Output: Rqueue, Fqueue

- Table 1: Evaluation on the GenEval [24] and DPG [29] benchmark. † result is with prompt rewriting.

GenEval↑ DPG↑ Two Obj. Position Color Attri. Overall Global Relation Overall

Methods # Params

Diffusion Models LDM [49] 1.4B 0.29 0.02 0.05 0.37 - - SDv1.5 [49] 0.9B 0.38 0.04 0.06 0.43 74.63 73.49 63.18 PixArt-alpha [13] 0.6B 0.50 0.08 0.07 0.48 74.97 82.57 71.11 SDv2.1 [49] 0.9B 0.51 0.07 0.17 0.50 77.67 80.72 68.09

- DALL-E 2 [45] 6.5B 0.66 0.10 0.19 0.52 - - -
- DALL-E 3 [7] - - - - 0.67† 90.97 90.58 83.50 SDXL [43] 2.6B 0.74 0.15 0.23 0.55 83.27 86.76 74.65 PixArt-Sigma [12] 0.6B 0.62 0.14 0.27 0.55 86.89 86.59 80.54 SD3 (d=24) [21] 2B 0.74 0.34 0.36 0.62 - - 84.08 SD3 (d=38) [21] 8B 0.89 0.34 0.47 0.71 - - AutoRegressive Models

LlamaGen [55] 0.8B 0.34 0.07 0.04 0.32 65.16 Chameleon [59] 7B - - - 0.39 - - HART [58] 732M - - - 0.56 - - 80.89 Show-o [72] 1.3B 0.80 0.31 0.50 0.68 - - 67.48 Emu3 [67] 8.5B 0.81† 0.49† 0.45† 0.66† - - 81.60 Infinity 2B 0.85† 0.49† 0.57† 0.73† 93.11 90.76 83.46

3.5 Dynamic Aspect Ratios and Position Encoding

Infinity can generate photo-realistic images with various aspect ratios, which is significantly different from VAR [61] that can only generate square images. The main obstacles of generating various aspect ratio images lie in two folds. The first is to define the height hk and width wk of Rk based on varying aspect ratios. In the supplementary material, we pre-define a list of scales, also called scale schedule, as {(hr1,w1r),...,(hrK,wKr )} for each aspect ratio. We ensure that the aspect ratio of each tuple (hrk,wkr) is approximately equal to r, especially in the latter prediction scales. Additionally, for different aspect ratios at the same scale k, we keep the area of hrk × wkr to be roughly equal, ensuring that the training sequence lengths are roughly the same.

Secondly, we need to carefully design a resolution-aware positional encoding method to handle features of various scales and aspect ratios. This issue poses a significant challenge, as the existing solutions [65, 61, 54, 26, 40] exhibit substantial limitations under such conditions. In this paper, we apply RoPE2d [26] on features of each scale to preserve the intrinsic 2D structure of images. Additionally, we exploit learnable scale embeddings to avoid confusion between features of different scales. Compared to learnable APE element-wisely applied on features, learnable embeddings applied on scales bring fewer parameters, can adapt to varying sequence lengths, and are easier to optimize.

- 4 Experiment

### 4.1 Dataset

Data Curation. We curated a large-scale dataset from open-source academic data and high-quality internally collected data. The pre-training dataset is constructed by collecting and cleaning opensource academic datasets such as LAION [51], COYO [10], OpenImages [33]. We exploit an OCR model and a watermark detection model to filter undesired images with too many texts or watermarks. Additionally, we employ Aesthetic-V2 to filter out images with low aesthetic scores.

### 4.2 Implementation

Infinity redefines text-to-image as a coarse-to-fine, next-scale prediction task. In line with its architecture, we propose to train Infinity in a progressive strategy. Specifically, we first train Infinity of 2B parameters on the pre-training dataset with 256 resolution for 150k iterations using a batch size of 4096 and a learning rate of 6e-5. Then we switch to 512 resolution and train 110k iterations using the same hyper-parameters. Next, we fine-tune Infinity at 1024 resolution with a smaller, high-quality dataset. In this stage, we train Infinity for 60k iterations using a batch size of 2048 and a learning rate of 2e-5. All training stages use images with varying aspect ratios.

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

house: white: pink tinted windows: surrounded by flowers: cute: scenic: garden: fairy like: epic: photography: photorealistic: insanely detailed and intricate: textures: grain: ultra realistic

Insect made from vintage 1960s electronic components. capacitors. resistors. transistors. wires. diodes. solder, circuitboard

hyperrealistic black and white photography of cats fashion show in style of helmut newton

close-up shot of a diecast toy car, diarama, night, lights from windows, bokeh, snow

Denis Villeneuve's extreme macro cinematographic closeup in water

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

two superheroes called spacefrog (a dashing green cartoon-like frog with a red cape) rand astroduck (a yellow fuzzy duck, part-robot, with blue/grey armor), near a garden pond, next to their spaceship, a classic flying saucer, called the Tadpole 3000. photorealistic

A creative 3D image to be placed at the bottom of a mobile application's homepage, depicting a miniature school and children carrying backpacks

Create an image with "Explore More" in an adventurous font over a picturesque hiking trail.

An enchanted miniature village bustling with activity, featuring tiny houses, markets, and residents.

[Figure 67]

[Figure 68]

A close-up photograph of a Corgi dog. The dog is wearing a black hat and round, dark sunglasses. The Corgi has a joyful expression, with its mouth open and tongue sticking out, giving an impression of happiness or excitement.

a robot holding a huge eggplant, sunny nature background

[Figure 69]

[Figure 70]

The image presents a picturesque mountainous landscape under a cloudy sky. The mountains, blanketed in lush greenery, rise majestically, their slopes dotted with clusters of trees and shrubs. The sky above is a canvas of blue, adorned with fluffy white clouds that add a sense of tranquility to the scene. In the foreground, a valley unfolds, nestled between the towering mountains. It appears to be a rural area, with a few buildings and structures visible, suggesting the presence of a small settlement. The buildings are scattered, blending harmoniously with the natural surroundings. The image is captured from a high vantage point, providing a sweeping view of the valley and the mountains.

Product photography, a perfume placed on a white marble table with pineapple, coconut, limenext to it as decoration, white curtains, full of intricate details, realistic, minimalist, layered gestures in a bright and concise atmosphere, minimalist style

##### Figure 4: Qualitative results from Infinity.

As for evaluation, we report results on popular text-to-image benchmarks like GenEval [24] and DPG [29]. We also measure our method on two human preference evaluation benchmarks, i.e., ImageReward [73] and HPSv2.1 [71]. These two benchmarks have trained models to predict human preference scores by learning from abundant human-ranked text-image pairs. We also build a validation set consisting of 40K text-image pairs to measure FID.

### 4.3 Text-to-Image Generation

Overall Results. Fig.1 and Fig.4 present generated images from our Infinity-2B model, showcasing Infinity’s strong capabilities in generating high-fidelity images from various categories following user prompts. Qualitative comparison results among Infinity and other top-tier models can be found in the appendix.

Prompt-Following. Fig.6 presents three examples demonstrating the superior prompt-following ability of Infinity. As highlighted in red, Infinity consistently adheres to user prompts, whether they are short or extremely long texts. We attribute these improvements to the bitwise token prediction and scaling autoregressive modeling.

Text Rendering. As illustrated in Fig.7, Infinity can render text according to user prompts across diverse categories. Despite diverse backgrounds and subjects, Infinity accurately renders corresponding texts according to user requirements, such as fonts, styles, colors, and more.

Benchmark. As in Tab 1, on GenEval[24], our model with a re-writer achieves the best overall score of 0.73. Besides, Infinity also reaches the highest position reasoning score of 0.49. On DPG [29]. Our model reaches an overall score of 83.46, surpassing SDXL [43], Playground v2.5 [36], and DALLE 3 [7]. What’s more, Infinity achieves the best relation score of 90.76 among all open-source T2I models, demonstrating its stronger ability to generate spatially consistent images based on user prompts.

[Figure 71]

- Figure 5: Human Preference Evaluation. We ask users to select the better one in a side-by-side comparison in terms of Overall Quality, Prompt Following, and Visual Aesthetics. Infinity is more preferred by humans compared to other open-source models.

Human Preference Evaluation. We conduct human preference evaluation in both human studies and benchmarks. As in Fig.5, the generation results of Infinity are more frequently selected by humans in terms of overall quality, prompt following, and visual aesthetics in contrast to other open-sourced T2I models. Please refer to the appendix for more details. Tab.2 lists the results of two human preference benchmarks, i.e., ImageReward [73] and HPSv2.1 [71]. Infinity reaches the highest ImageReward and HPSv2.1, indicating our method could generate images that are more appealing to humans.

Infinity Flux Schnell SD3-Medium PixArt Sigma HART

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

a green cube on the top of a yellow sphere, the behind is a red triangle. A cat is on the right and a dog is on the left

Infinity Flux Schnell SD3-Medium PixArt Sigma HART

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

a giant, towering cloud in the shape of a man looms over the earth, with arms outstretched like a mythical deity. The cloud man, composed of swirling, gray and white vapors, shoots lightning bolts down to the earth, piercing through the dark sky. The bolts illuminate the scene with a dramatic, otherworldly light, casting sharp shadows on the ground below.

Infinity Flux Schnell SD3-Medium PixArt Sigma HART

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Create a mesmerizing image of three intricately designed potions displayed on an ornate, antique wooden table within a charming old apothecary. The first potion is a captivating cobalt blue, housed in a stunning pentagon-shaped glass bottle that sparkles with its many facets; its label, meticulously crafted with delicate silver filigree and botanical illustrations of ethereal flowers, prominently features the letters “I” in an ornate, swirling script, while a silver ribbon interwoven with tiny sapphire beads wraps around the neck, adorned with a charm in the shape of a crescent moon. The second potion is a rich crimson red, contained in a flat, oval-shaped glass bottle adorned with intricate engravings of mystical symbols, including runes and ancient scripts; its label displays the letters “N” in embossed gold leaf, framed by elaborate floral designs, and is topped with a cork stopper embellished with a miniature brass key and tiny ruby gemstones. The third potion is a vivid emerald green, held in a sleek square glass bottle featuring enchanting etchings of mythical creatures like dragons and phoenixes; its scroll-like label, crafted from aged parchment, prominently features the letter “F” intertwined with ancient alchemical symbols and delicate vine patterns. All three bottles are approximately the same height, creating a harmonious display against a backdrop filled with shelves overflowing with dried herbs, colorful glass jars, and ancient scrolls, all illuminated by soft, warm light filtering through a stained-glass window, enhancing the magical atmosphere of the apothecary.

##### Figure 6: Prompt-following qualitative comparison. We highlight text in red that Infinity-2B consistently adheres to while the other four models fail to follow. Zoom in for better comparison.

[Figure 87]

[Figure 88]

[Figure 89]

A portrait photo of a kangaroo wearing an orange hoodie and blue sunglasses standing on the grass in front of the Sydney Opera House holding a sign on the chest with the text 'Welcome Friends!'

Render the text 'Welcome to the Future' with a vibrant neon effect, set against a backdrop of a futuristic cityscape at night, filled with towering skyscrapers, cars, and holographic advertisements.

Create an image with the text 'Stay Positive' in an uplifting style, featuring bright and cheerful colors, swirling floral patterns, and a radiant sun in the background.

[Figure 90]

[Figure 91]

[Figure 92]

An image of a futuristic robot at work in a high-tech laboratory, with advanced machinery and holographic displays in the background, the text 'Innovate' in sleek, modern font hovering above.

The text 'Peace and Love' placed elegantly on a serene ocean horizon during sunset, with calm waves and a sky painted in soft pink and orange hues

A snowy village scene with cozy cottages covered in snow, twinkling lights in the windows, smoke rising from chimneys, snow-covered trees, and children playing in the snow. The text 'Winter Wonderland' is written prominently in the snow in front of the village, as if carefully crafted by a child.

##### Figure 7: Text rendering results from our Infinity-2B model. Infinity-2B could generate text-consistent images following user prompts across diverse categories.

Inference Latency. As shown in Tab. 2, Infinity demonstrates a significant advantage in generation speed compared to diffusion models at around 2 billion parameters. Furthermore, our tests reveal that the speed advantage of Infinity becomes more substantial as the model size increases. Infinity achieves 7× faster inference latency compared to SD3.5 [21] at the same 8 billion parameters.

- Table 2: Human Preference Metrics and Inference Latency. We compared our method with SoTA open-source models. Infinity achieved the best human preference results with the fastest speed.

Methods # Params

ImageReward↑ HPSv2.1↑ Latency↓ Rank Score Rank Score Rank Time

SD-XL [43] 2.6B 4 0.600 4 30.06 4 2.7s SD3-Medium [21] 2B 3 0.871 3 30.91 3 2.1s PixArt Sigma [12] 630M 2 0.872 2 31.47 2 1.1s Infinity 2B 1 0.962 1 32.25 1 0.8s

- Table 3: Comparison of memory consumption (GB) between different quantizers during training. As codebook dimension d increases, MSR-BSQ shows significant advantages over MSR-LFQ, enabling nearly infinite vocabulary size of 264.

Quantizer d = 16 d = 18 d = 20 d = 32 d = 64

LFQ 37.6 53.7 OOM OOM OOM BSQ 32.4 32.4 32.4 32.4 32.4

- Table 4: By scaling up visual tokenizer’s vocabulary, discrete tokenizer surpasses continuous VAE of SD [48] on ImageNet-rFID.

VAE (stride=16) TYPE IN-256 rFID↓ IN-512 rFID↓

Vd = 216 Discrete 1.22 0.31 Vd = 224 Discrete 0.75 0.30 Vd = 232 Discrete 0.61 0.23 Vd = 264 Discrete 0.33 0.15 SD VAE [49] Contiguous 0.87 N/A

- Table 5: IVC saves 99.95% params and gets better performance to conventional classifier (Vd = 216) Classifier # Params vRAM Recons. Loss↓ FID↓ ImageReward↑ HPSv2.1↑ Convention 124M 2GB 0.184 4.49 0.79 31.95 IVC 0.65M 10MB 0.180 3.83 0.91 32.31

- Table 6: Model architectures for scaling visual autoregressive modeling. Note that GFLOPs are rough values since they are affected by the length of the text prompt.

# Params GFLOPs Hidden Dimension Heads Layers

125M 30 768 8 12 361M 440 1152 12 16 940M 780 1536 16 24 2.2B 1500 2080 20 32 4.7B 2600 2688 24 40

### 4.4 Scaling Visual Tokenizer’s Vocabulary

Scaling Up the Vocabulary Benefits Reconstruction. Restricted by the vocabulary size, discrete VQ-VAEs have always lagged behind continuous ones, hindering the performance of AR-based T2I models. In this work, we successfully train a discrete VQ-VAE matching its continuous counterparts by scaling up the vocabulary size. As in Tab. 4, we observe consistent rFID improvements as scaling up the vocabulary size from 216 to 264. It’s noteworthy that our discrete tokenizer achieves a rFID of 0.61 on ImageNet 256×256 when Vd = 232, outperforming the continuous VAE of SD [49].

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

- Figure 8: Impact of Infinite-Vocabulary Classifier. Predicting bitwise labels with the InfiniteVocabulary Classifier (Right) generates images with richer details compared to predicting index-wise labels using a conventional classifier (Left).

[Figure 101]

- Figure 9: Effects of Scaling Up the Vocabulary. We analyze the impact of scaling the vocabulary

size under consistent training hyperparameters throughout. Vocabulary size Vd = 216 converges faster and achieves better results for small models (125M and 361M parameters). As we scale up

the model size to 2.2B, Infinity with a vocabulary size Vd = 232 beats that one with Vd = 216. Experiment with 5M high-quality image-text pair data under 256 × 256 resolution.

[Figure 102]

- Figure 10: Effects of Scaling Visual AutoRegressive Modeling. We analyze the impact of scaling model size under consistent training hyperparameters throughout (Experiment with 10M pre-training data and 256 × 256 resolution). Validation loss smoothly decreases as a function of the model size and training iterations. Besides, Validation loss is a strong predictor of overall model performance. There is a strong correlation between validation loss and holistic image evaluation metrics.

Infinite Vocabulary Classifier Benefits Generation. We compare predicting bit labels with IVC to predicting index labels using a conventional classifier under the vocabulary size of 216, since a larger vocabulary causes OOM for the conventional classifier. We use the reconstruction loss on Rk, FID on the validation set and ImageReward for comprehensive evaluation. As shown in Tab.5, IVC achieves lower reconstruction loss and FID, suggesting IVC has better fitting capabilities. Beyond the quantitative results, training Infinity with IVC yields images with richer details as in Fig.8, which is consistent with a higher ImageReward.

### 4.5 Scaling Bitwise AutoRegressive Modeling

Scaling Up the Vocabulary Benefits Generation. We then scale up the vocabulary size to 232 during training the T2I model, which exceeds the range of the Int32 data type and can be considered infinitely large. In Fig.9, we illustrate the effect of scaling up the vocabulary from 216 to 232 for image generation. For small models (125M and 361M), the vocabulary size of 216 converges faster and achieves better results. However, as we scaled up the transformer to 2.2B, the vocabulary size of 232 beats 216 after 40K iterations. Therefore, it’s worthwhile to scale up the vocabulary along with scaling up the transformer. As illustrated in Tab.1,2, with infinite vocabulary and IVC, Infinity achieves superior performance among various benchmarks, elevating the ceiling of AR visual generation.

Scaling Up Transformer Benefits Generation. In Fig.10, we depict the validation loss against the total training iterations and computational FLOPs for various model sizes of Infinity. The detailed model architectures for different sizes can be found in Tab.6. We consistently notice a reduction in validation loss with an increase in training steps and computational FLOPs. Nevertheless, the advantages gained from training smaller models for extended periods lag behind those obtained from training larger models for shorter durations. This trend aligns with findings in language models, emphasizing the promising outlook for increasing model sizes with appropriate training.

In Fig.10, we plot GenEval, ImageReward, and HPSv2 scores against validation loss for different model sizes ranging from 125M to 4.7B. We observe a strong correlation between validation loss and evaluation metrics. To further quantify their correlation, we calculate the Pearson correlation coefficients through linear regression. The correlation coefficients for GenEval, ImageReward, and HPSv2 are -0.983, -0.981, and -0.979, respectively. These results demonstrate a nearly linear correlation between validation loss and the evaluation metrics when scaling up model sizes from

Scaling up training compute

Scalingupmodelsize

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

- Figure 11: Semantics and visual quality improve consistently with scaling up model size and training compute. Zoom in for better comparison.

125M to 4.7B. This promising phenomenon encourages us to scale up Infinity to achieve better performance.

Visualization of Scaling Effects. To delve deeper into the scaling effect of Infinity, we compare a set of generated 256×256 images of three model sizes (125M, 940M, 4.7B) across three distinct training schedules (10K, 40K, 90K iterations) as illustrated in Fig.11. The semantics and visual quality of generated images improve steadily when scaling up model size and training compute, which is consistent with the scaling behaviors of Infinity.

### 4.6 Bitwise Self-Correction

In Tab.7 and Fig.12, we list the evaluation metrics and present images generated by models trained using teacher-forcing and bitwise self-correction methods. Substantial advantages are observed after applying bitwise self-correction. Furthermore, we prove that the significant advantages are primarily driven by the self-correction mechanism rather than applying flipping. As shown in Tab.7, simply random flipping Rk doesn’t bring improvements. Self-Correction imitates prediction errors and applies re-quantification to correct them. We emphasize that Self-Correction is essential for AR-based T2I models since it empowers models to correct errors automatically, significantly mitigating the train-test discrepancy.

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

- Figure 12: Impact of Self-Correction. Teacher-forcing training introduces great train-test discrepancy which degrades performance during inference (left). Bitwise Self-Correction auto-corrects mistakes and thus generates better results (right). Decoding with τ = 1 and cfg = 3.

Table 7: Bitwise Self-Correction makes significant improvements. Experiment with 5M high-quality data and 512 × 512 resolution. FID is measured on the validation set with 40K images. Decoding with τ = 1 and cfg = 3.

Method FID↓ ImageReward↑ HPSv2.1↑

Baseline 9.76 0.52 29.53 Baseline + Random Flip 9.69 0.52 29.20 Baseline + Bitwise Self-Correction 3.48 0.76 30.71

4.7 Ablation Studies

Optimal Strength for Bitwise Self-Correction. Bitwise Self-Correction mitigates the train-test discrepancy caused by teacher-forcing training. Here we delve into the optimal strength for applying bitwise self-correction in Tab.8. We empirically find that mistake imitation that is too weak (10% and 20%) fails to fully leverage the potential of Bitwise Self-Correction. Random flipping 30% bits yields the best results.

Table 8: Comparison between different strengths of Bitwise Self-Correction. Experiment with 5M high-quality data and 512 × 512 resolution. Decoding with τ = 1 and cfg = 3.

Method FID↓ ImageReward↑ HPSv2.1↑ w/o Bitwise Self-Correction 9.76 0.515 29.53 Bitwise Self-Correction (p = 10%) 3.45 0.751 30.47 Bitwise Self-Correction (p = 20%) 3.48 0.763 30.71 Bitwise Self-Correction (p = 30%) 3.33 0.775 31.05

[Figure 118]

- Figure 13: Comparison between learnable APE and our positional embeddings. Our method, i.e., applying RoPE2d along with learnable scale embeddings on features of each scale, converges faster and reaches higher training accuracy.

Positional Embedding. Learnable APE adopted in VAR [61] brings too many parameters and gets confused when the sequence length varies. However, the sequence length changes frequently when training with various aspect ratios. Simply applying RoPE2d [26] or normalized RoPE2d [40] can

not distinguish features from different resolutions. In this work, we apply RoPE2d and learnable scale embeddings on features of each scale. RoPE2d preserves the intrinsic 2D structure of images. Learnable scale embeddings avoid confusion between features of different scales. To verify the effectiveness, we compare it with the learnable APE in Fig.13. It’s obvious that applying RoPE2d along with learnable scale embeddings on features of each scale converges faster and reaches higher training accuracy.

Decoding. Decoding is crucial for improving generation quality. VAR adopts the pyramid ClassiferFree Guidance (CFG) on predicted logits. That is, the strength of CFG increases linearly as the scale goes from 1 to K. Such a pyramid scheme is used to tackle the issue of the model collapsing frequently when applying large CFG at early scales. We found that Infinity supports large CFG values even in very early scales equipped with Bitwise Self-Correction. Since Infinity is more robust to sampling, we revisit different decoding methods and find the best as illustrated in Tab.9. We visualize the comparison results of different decoding methods in Fig.14. We achieve the best generation results.

Table 9: Comparison between different decoding methods.

Method Param FID↓ ImageReward↑ HPSv2.1↑

Greedy Sampling τ = 0.01,cfg = 1 9.97 0.397 30.98 Normal Sampling τ = 1.00,cfg = 1 4.84 0.706 31.59 Pyramid CFG τ = 1.00,cfg = 1 → 3 3.48 0.872 32.48 Pyramid CFG τ = 1.00,cfg = 1 → 5 2.98 0.929 32.32 CFG on features τ = 1.00,cfg = 3 3.00 0.953 32.13 CFG on logits τ = 1.00,cfg = 3 2.91 0.952 32.31 CFG on logits (Ours) τ = 1.00,cfg = 4 2.82 0.962 32.25

Prompt: three gold cosmetic jars immersed in white skincare cream, side view, realistic texture, full background cream, natural light

[Figure 119]

Greedy Sample

Normal Sample Pyramid CFG Ours

Prompt: generate the words 'Welcome Home' in a cozy and warm font on a wooden door background.

[Figure 120]

Greedy Sample

Normal Sample Pyramid CFG Ours

Prompt: a couple under an umbrella in the rain, candid moment, in the style of romantic film stills, moody lighting, intimate and tender

[Figure 121]

Greedy Sample

Normal Sample Pyramid CFG Ours

- Figure 14: Comparison of different sampling methods. In contrast to Greedy Sample, Normal Sample and Pyramid Sample, our method could generate images with richer details and higher text-image alignments.

## 5 Conclusion

We introduce Infinity, a bitwise visual autoregressive model to perform Text-to-Image generation. Infinity is a pioneering framework for bitwise token modeling with the IVC and self-correction innovation. Extensive qualitative and quantitative results demonstrate Infinity significantly raised the upper limit for Autogressive Text-To-Image generative models, matching or surpassing leading diffusion models. We believe our framework, Infinity, will substantially promote the development of autoregressive visual modeling and inspire the community for faster and more realistic generation models.

## 6 Acknowledges

Many colleagues from ByteDance supported this work. We are grateful to Guanyang Deng for his efforts in data processing. We also thank Chongxi Wang and Taekmin Kim for their contributions to model deployment. Special thanks to Xiaoxiao Qin for her work in human preference evaluation. Additionally, we are thankful to Hui Wu, Fu Li, Xing Wang, Hongxiang Hao, and Chuan Li for their contributions to infrastructure.

## References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.
- [3] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [4] Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analytic-dpm: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. arXiv preprint arXiv:2201.06503, 2022.
- [5] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22669–22679, 2023.
- [6] Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233, 2024.
- [7] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.
- [8] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. OpenAI, 2024.
- [9] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [10] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/coyo-dataset, 2022.
- [11] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11315–11325, 2022.
- [12] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024.
- [13] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.
- [14] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.
- [15] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In International conference on machine learning, pages 1691–1703. PMLR, 2020.
- [16] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023.
- [17] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53, 2024.
- [18] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023.
- [19] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.
- [20] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

- [21] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.
- [22] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.
- [23] Lijie Fan, Tianhong Li, Siyang Qin, Yuanzhen Li, Chen Sun, Michael Rubinstein, Deqing Sun, Kaiming He, and Yonglong Tian. Fluid: Scaling autoregressive text-to-image generative models with continuous tokens, 2024.
- [24] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36, 2024.
- [25] Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B Brown, Prafulla Dhariwal, Scott Gray, et al. Scaling laws for autoregressive generative modeling. arXiv preprint arXiv:2010.14701, 2020.
- [26] Byeongho Heo, Song Park, Dongyoon Han, and Sangdoo Yun. Rotary position embedding for vision transformer. In European Conference on Computer Vision, pages 289–305. Springer, 2025.
- [27] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [28] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.
- [29] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.
- [30] Aren Jansen, Daniel PW Ellis, Shawn Hershey, R Channing Moore, Manoj Plakal, Ashok C Popat, and Rif A Saurous. Coincidence, categorization, and consolidation: Learning to recognize sounds with minimal supervision. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 121–125. IEEE, 2020.
- [31] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.
- [32] Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, Krishna Somandepalli, Hassan Akbari, Yair Alon, Yong Cheng, Josh Dillon, Agrim Gupta, Meera Hahn, Anja Hauth, David Hendon, Alonso Martinez, David Minnen, Mikhail Sirotenko, Kihyuk Sohn, Xuan Yang, Hartwig Adam, Ming-Hsuan Yang, Irfan Essa, Huisheng Wang, David A. Ross, Bryan Seybold, and Lu Jiang. Videopoet: A large language model for zero-shot video generation, 2024.
- [33] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International journal of computer vision, 128(7):1956–1981, 2020.
- [34] Black Forest Labs. Flux. https://blackforestlabs.ai/announcing-black-forest-labs/, 2024.
- [35] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11523–11532, 2022.
- [36] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2.5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024.
- [37] Xiang Li, Hao Chen, Kai Qiu, Jason Kuen, Jiuxiang Gu, Bhiksha Raj, and Zhe Lin. Imagefolder: Autoregressive image generation with folded tokens. arXiv preprint arXiv:2410.01756, 2024.
- [38] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787, 2022.
- [39] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022.
- [40] Xiaoxiao Ma, Mohan Zhou, Tao Liang, Yalong Bai, Tiejun Zhao, Huaian Chen, and Yi Jin. Star: Scale-wise text-to-image generation via auto-regressive representations. arXiv preprint arXiv:2406.10797, 2024.
- [41] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International conference on machine learning, pages 8162–8171. PMLR, 2021.

- [42] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.
- [43] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [44] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
- [45] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.
- [46] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. PMLR, 2021.
- [47] Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems, 32, 2019.
- [48] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [49] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [50] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-toimage diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022.
- [51] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021.
- [52] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.
- [53] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.
- [54] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [55] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.
- [56] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222, 2023.
- [57] Yu Sun, Shuohuan Wang, Shikun Feng, Siyu Ding, Chao Pang, Junyuan Shang, Jiaxiang Liu, Xuyi Chen, Yanbin Zhao, Yuxiang Lu, et al. Ernie 3.0: Large-scale knowledge enhanced pre-training for language understanding and generation. arXiv preprint arXiv:2107.02137, 2021.
- [58] Haotian Tang, Yecheng Wu, Shang Yang, Enze Xie, Junsong Chen, Junyu Chen, Zhuoyang Zhang, Han Cai, Yao Lu, and Song Han. Hart: Efficient visual generation with hybrid autoregressive transformer. arXiv preprint arXiv:2410.10812, 2024.
- [59] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.
- [60] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [61] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv preprint arXiv:2404.02905, 2024.
- [62] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [63] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

- [64] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.
- [65] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [66] Junke Wang, Yi Jiang, Zehuan Yuan, Binyue Peng, Zuxuan Wu, and Yu-Gang Jiang. Omnitokenizer: A joint image-video tokenizer for visual generation. arXiv preprint arXiv:2406.09399, 2024.
- [67] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint

- arXiv:2409.18869, 2024.

[68] Yuqing Wang, Tianwei Xiong, Daquan Zhou, Zhijie Lin, Yang Zhao, Bingyi Kang, Jiashi Feng, and Xihui Liu. Loong: Generating minute-level long videos with autoregressive language models. arXiv preprint

- arXiv:2410.02757, 2024.

- [69] Mark Weber, Lijun Yu, Qihang Yu, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. Maskbit: Embedding-free image generation via bit tokens. arXiv preprint arXiv:2409.16211, 2024.
- [70] BigScience Workshop, Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili´c, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, et al. Bloom: A 176b-parameter open-access multilingual language model. arXiv preprint arXiv:2211.05100, 2022.
- [71] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.
- [72] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.
- [73] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36, 2024.
- [74] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627, 2021.
- [75] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022.
- [76] Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.
- [77] Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.
- [78] Lili Yu, Bowen Shi, Ramakanth Pasunuru, Benjamin Muller, Olga Golovneva, Tianlu Wang, Arun Babu, Binh Tang, Brian Karrer, Shelly Sheynin, et al. Scaling autoregressive multi-modal models: Pretraining and instruction tuning. arXiv preprint arXiv:2309.02591, 2(3), 2023.
- [79] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. arXiv preprint arXiv:2406.07550, 2024.
- [80] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.
- [81] Yue Zhao, Yuanjun Xiong, and Philipp Krähenbühl. Image and video tokenization with binary spherical quantization. arXiv preprint arXiv:2406.07548, 2024.

## A Predefined Scale Schedules

As listed in Tab.10, for each aspect ratio r, we predefine a specific scale schedule {(hr1,w1r),...,(hrK,wKr )}. We ensure that the aspect ratio of each tuple (hrk,wkr) is approximately equal to r, especially in the latter scales. Additionally, for different aspect ratios at the same scale k, we keep the area of hrk × wkr to be roughly equal, ensuring that the training sequence lengths are roughly the same. We adopt buckets to support training various aspect ratios at the same time. The consistent sequence lengths of different aspect ratios improve training efficiency. During the inference stage, Infinity could generate photo-realistic images covering common aspect ratios (1:1, 16:9, 4:3, etc.) as well as special aspect ratios (1:3, 3:1, etc.) following the predefined scale schedules.

Table 10: Predefined scale schedules {(hr1,w1r),...,(hrK,wKr )} for different aspect ratios. Following the text guided next-scale prediction scheme, Infinity takes K=13 scales to generate a 1024 × 1024

(or other aspect ratio) image.

Aspect Ratio Resolution Scale Schedule 1.000 (1:1) 1024×1024 (1,1) (2,2) (4,4) (6,6) (8,8) (12,12) (16,16) (20,20) (24,24) (32,32) (40,40) (48,48) (64,64)

- 0.800 (4:5) 896×1120 (1,1) (2,2) (3,3) (4,5) (8,10) (12,15) (16,20) (20,25) (24,30) (28,35) (36,45) (44,55) (56,70)
- 1.250 (5:4) 1120×896 (1,1) (2,2) (3,3) (5,4) (10,8) (15,12) (20,16) (25,20) (30,24) (35,28) (45,36) (55,44) (70,56)

- 0.750 (3:4) 864×1152 (1,1) (2,2) (3,4) (6,8) (9,12) (12,16) (15,20) (18,24) (21,28) (27,36) (36,48) (45,60) (54,72)
- 1.333 (4:3) 1152×864 (1,1) (2,2) (4,3) (8,6) (12,9) (16,12) (20,15) (24,18) (28,21) (36,27) (48,36) (60,45) (72,54)

- 0.666 (2:3) 832×1248 (1,1) (2,2) (2,3) (4,6) (6,9) (10,15) (14,21) (18,27) (22,33) (26,39) (32,48) (42,63) (52,78)
- 1.500 (3:2) 1248×832 (1,1) (2,2) (3,2) (6,4) (9,6) (15,10) (21,14) (27,18) (33,22) (39,26) (48,32) (63,42) (78,52)

- 0.571 (4:7) 768×1344 (1,1) (2,2) (3,3) (4,7) (6,11) (8,14) (12,21) (16,28) (20,35) (24,42) (32,56) (40,70) (48,84)
- 1.750 (7:4) 1344×768 (1,1) (2,2) (3,3) (7,4) (11,6) (14,8) (21,12) (28,16) (35,20) (42,24) (56,32) (70,40) (84,48) 0.500 (1:2) 720×1440 (1,1) (2,2) (2,4) (3,6) (5,10) (8,16) (11,22) (15,30) (19,38) (23,46) (30,60) (37,74) (45,90)
- 2.000 (2:1) 1440×720 (1,1) (2,2) (4,2) (6,3) (10,5) (16,8) (22,11) (30,15) (38,19) (46,23) (60,30) (74,37) (90,45) 0.400 (2:5) 640×1600 (1,1) (2,2) (2,5) (4,10) (6,15) (8,20) (10,25) (12,30) (16,40) (20,50) (26,65) (32,80) (40,100)

- 2.500 (5:2) 1600×640 (1,1) (2,2) (5,2) (10,4) (15,6) (20,8) (25,10) (30,12) (40,16) (50,20) (65,26) (80,32) (100,40) 0.333 (1:3) 592×1776 (1,1) (2,2) (2,6) (3,9) (5,15) (7,21) (9,27) (12,36) (15,45) (18,54) (24,72) (30,90) (37,111)
- 3.000 (3:1) 1776×592 (1,1) (2,2) (6,2) (9,3) (15,5) (21,7) (27,9) (36,12) (45,15) (54,18) (72,24) (90,30) (111,37)

[Figure 122]

- Figure 15: Distribution of Prompt Categories

## B Human Preference Evaluation

In order to measure the overall performance, we have conducted a human preference evaluation. We build a website and recruit volunteers to rank the generated images from different T2I models.

Prompts. We have collected 360 prompts in total, including prompts randomly sampled from Parti [75] and other human-written prompts. As illustrated in Fig.15, these prompts are divided into nine categories, such as human (28%), animal (15%), products/artifacts (12%), landscape (9%),

[Figure 123]

- Figure 16: Distribution of Prompts Challenges

foods, indoor scene, architecture, plants, and text rendering. It is worth noting that we incorporate a variety of human-related prompts, such as faces, bodies, and movements, in the human category as a supplement to the Parti prompts. In Fig.16, we also list the challenges of these prompts, which includes simple prompts, complex prompts, quantity, positioning & perspective, painting style, detail, semantic understanding, color, and imagination. These statistics demonstrate that the prompts used for evaluation are balanced, covering various categories and challenges well.

Generated Images. We compare Infinity with four open-source models: PixArt-Sigma [12], SD3Medium [21], SDXL [43], and HART [58]. The images of other models are generated by running their official inference code. No cherry-picking for any models.

Human Evaluation. For the human evaluation process, we build a website which presents two images from two anonymous models at the same time. There is one image generated by Infinity while the other is from other four models. Volunteers are required to pick a better one from two images in terms of overall quality, prompt following, and visual aesthetics, respectively. Besides the aforementioned criterion, we make sure each side-by-side comparison is evaluated by at least two volunteers to reduce human bias. We filter out pairs with opposite results evaluated by two volunteers. These contradictory pairs are sent to a third volunteer to assess. Then we take the consensus from three as the final results. Note that the whole process of human evaluation is completely double-blind. That is, a volunteer doesn’t know which model it is, as well as other volunteers’ results when performing a side-by-side comparison.

Results. As in Fig.6 of the submitted manuscript, we observe a remarkable human preference for Infinity over the other four open-source models. Especially for the comparison with HART [58] (another SOTA AR-based model), Infinity earns 90.0%, 83.9%, and 93.2% win rate in terms of overall quality, prompt following, and visual aesthetics, respectively. As for the diffusion family, Infinity earns 76.0%, 79.0%, 66.0% win rate to PixArt-Sigma, SDXL and SD3-Medium, respectively. What’s more, Infinity reaches 71.1% win rate towards SD3-Medium regarding visual aesthetics. These results reveal that Infinity is more capable of generating visually appealing images. We attribute these great advantages to the proposed bitwise modeling, which has lifted the upper limits of AR models by large margins.

## C More Qualitative Results

Fig.17 shows the qualitative comparison results among Infinity and other top-tier models. The images of other models are obtained either by querying their open-source demo website (HART [58]) or running their official inference code locally (Flux-Schnell [34], SD3-Medium [21], and PixArt Sigma [12]). Whether a thumbnail or a zoom-in image, we observe significant differences among the generated images from different models. In particular, the AR model like HART generates images

Infinity Flux Schnell SD3-Medium PixArt Sigma HART

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

photorealistic closeup video of two pirate ships battling each other as they sail inside a cup of coffee

Infinity Flux Schnell SD3-Medium PixArt Sigma HART

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

a tiny astronaut hatching from an egg on the moon

Infinity Flux Schnell SD3-Medium PixArt Sigma HART

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

a young woman with glasses reading a thick book at a mahogany desk

Infinity Flux Schnell SD3-Medium PixArt Sigma HART

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

an elderly person practicing gentle yoga, seated position, calm expression

Infinity Flux Schnell SD3-Medium PixArt Sigma HART

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

an oil painting of a house

Figure 17: T2I qualitative comparison among our Infinity-2B model and the other four open-source models. Here we select three diffusion models (Flux Schnell, SD3-Medium and PixArt Sigma), one AR model (HART) for comparison. Zoom in for better comparsion.

with fewer details, blurred human faces and texture-less background compared to diffusion models. In contrast, Infinity overcomes those shortcomings of AR models and generates comparable or better images when compared to diffusion models like Flux-Schnell, SD3-Medium, and PixArt Sigma. For the first and second examples, Infinity adheres to the text prompts better than SD3-Medium, HART, and PixArt-Sigma. For the third and fourth examples, Infinity performs better in human hands and legs. For the last example, Infinity and PixArt Sigma have successfully generated images in an oil painting style while the other three failed. Flux Schnell performs worst in this example.

