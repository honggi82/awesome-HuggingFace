## UNCAGE: Contrastive Attention Guidance for Masked Generative Transformers in Text-to-Image Generation

### Wonjun Kang1,2 Byeongkeun Ahn3 Minjae Lee2 Kevin Galim2 Seunghyuk Oh2 Hyung Il Koo2,4 Nam Ik Cho1

1Seoul National University 2FuriosaAI 3Independent Researcher 4Ajou University

{kangwj1995, minjae.lee, kevin.galim, seunghyukoh, hikoo}@furiosa.ai, byeongkeunahn0@gmail.com, nicho@snu.ac.kr

# arXiv:2508.05399v1[cs.CV]7Aug2025

Meissonic [top left, ICLR’25]

Halton Scheduler [top right, ICLR’25]

UNCAGE (ours) [bottom left & right]

###### Object Mixture Missing Objects Attribute Leakage

“a turtle and a pink apple” “a rabbit and a yellow car” “a green bench and a red apple”

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

Figure 1: Our training-free, Unmasking with Contrastive Attention Guidance (UNCAGE) enhances the performance of Masked Generative Transformers in compositional T2I generation, with negligible inference overhead.

###### Abstract

Text-to-image (T2I) generation has been actively studied using Diffusion Models and Autoregressive Models. Recently, Masked Generative Transformers have gained attention as an alternative to Autoregressive Models to overcome the inherent limitations of causal attention and autoregressive decoding through bidirectional attention and parallel decoding, enabling efficient and high-quality image generation. However, compositional T2I generation remains challenging, as even state-of-the-art Diffusion Models often fail to accurately bind attributes and achieve proper text-image alignment. While Diffusion Models have been extensively studied for this issue, Masked Generative Transformers exhibit similar limitations but have not been explored in this context. To address this, we propose Unmasking with Contrastive Attention Guidance (UNCAGE), a novel training-free method that improves compositional fidelity by leveraging attention maps to prioritize the unmasking of tokens that clearly represent individual objects. UNCAGE consistently improves performance in both quantitative and qualitative evaluations across multiple benchmarks and metrics, with negligible inference overhead. Our code is available at https://github.com/furiosa-ai/uncage.

### Introduction

Text-to-image (T2I) generation (Saharia et al. 2022; Ramesh

- et al. 2021) has emerged as a crucial task in computer vision, offering vast potential for creative applications. To achieve high-quality generation, various generative models, including GANs (Kang et al. 2023), Diffusion Models (Rombach
- et al. 2022), and Autoregressive Models (Sun et al. 2024; Tian et al. 2024), have been extensively studied and rapidly advanced. Diffusion Models (Ho, Jain, and Abbeel 2020; Song, Meng, and Ermon 2021) have significant advantages over GANs (Goodfellow et al. 2014) in terms of stable training and sample diversity. As a result, Diffusion Models have become the most widely adopted approach for visual generation tasks (Blattmann et al. 2023; Voleti et al. 2024), including T2I generation. However, with the recent success of autoregressive modeling in the NLP domain, exemplified by models like GPT (Brown et al. 2020; Achiam et al.
- 2023) and LLaMA (Touvron et al. 2023a,b), there has been growing interest in applying Autoregressive Models to image generation as well (Sun et al. 2024; Liu et al. 2024).

Autoregressive Models generate image tokens sequentially, producing only one token per step, which makes inference computationally expensive. Moreover, the atten-

tion mechanism in these models is applied causally rather than bidirectionally, limiting the model’s ability to fully leverage contextual information during image generation. Masked Generative Transformers (MGTs) address these limitations (Chang et al. 2022, 2023; Bai et al. 2025), offering an alternative generative paradigm that enhances efficiency through parallel decoding and leverages information more effectively through bidirectional attention.

As T2I generation advances, there is growing focus on compositional T2I generation, where the goal is to correctly render multiple objects and their attributes based on the prompt. However, most T2I generation models struggle with misaligned attribute binding, resulting in discrepancies between the prompt and the generated image. For example, as shown in Figure 1, given the prompt “a turtle and a pink apple,” the model may generate a turtle with a pink appleshaped shell, instead of depicting both objects separately.

Attend-and-Excite (Chefer et al. 2023) is the first study to address this issue in Diffusion Models. Chefer et al. (2023) observe that attention maps in Diffusion Models correlate strongly with object regions, but attribute binding problems occur when these maps fail to accurately localize individual objects and overlap with each other. To mitigate this, they introduce an attention-based loss and refine the output at each denoising step by computing gradients and updating the generation process accordingly. Subsequent studies (Rassin et al. 2023; Sueyoshi and Matsubara 2024) build on Attend-and-Excite to further improve attribute binding in compositional T2I generation. However, these methods rely on gradient-based iterative refinement during inference, resulting in substantial inference overhead.

MGTs, like most Diffusion Models, utilize attentionbased architectures, which can lead to misaligned attribute binding due to inaccurate attention maps. However, no prior work has addressed this issue for MGTs. Moreover, Diffusion Models and MGTs generate outputs in fundamentally different ways depending on the timestep. Diffusion Models progressively refine arbitrary regions of the image through iterative denoising, whereas MGTs predict all token positions in parallel at each timestep, and only a subset of these tokens is unmasked, which then remains fixed throughout the process. Due to this fundamental difference, directly applying refinement strategies (Chefer et al. 2023; Rassin et al. 2023; Sueyoshi and Matsubara 2024) developed for Diffusion Models to MGTs is non-trivial. This highlights the need for methods specifically designed for MGTs.

In MGTs, the overall structure is largely determined during the early generation steps, as shown in the examples on the right of Figure 2. In the upper right example, the model failed to generate “a pink apple and a car” and instead produced only a car. We hypothesize that if the apple-related token predicted at t = 3 had been unmasked and retained, it could have served as a foundation for generating the apple in subsequent steps (the lower right example), thereby improving alignment. This observation indicates that unmasking order strategies, particularly during early steps, are crucial for improved text-image alignment.

Based on these insights, we propose Unmasking with Contrastive Attention Guidance (UNCAGE) to address mis-

aligned attribute binding in compositional T2I generation by leveraging attention maps to prioritize the unmasking of tokens that clearly represent individual objects. This process is training-free and incurs negligible additional computational cost, as it relies solely on attention maps that have already been computed to guide the unmasking order. In summary, our main contributions are:

- • We are the first to address inaccurate attribute binding in compositional T2I generation with Masked Generative Transformers, improving text-image alignment.
- • We propose a novel training-free method, Unmasking with Contrastive Attention Guidance (UNCAGE), which mitigates this issue while incurring negligible additional cost.
- • We demonstrate the effectiveness of UNCAGE through quantitative and qualitative experiments across multiple benchmarks and evaluation metrics.

### Related Works

##### Masked Generative Transformers (MGTs)

While the success of GPT (Brown et al. 2020; Achiam et al. 2023) has inspired autoregressive approaches for image generation (Chen et al. 2020), MaskGIT (Chang et al. 2022) takes a different approach, drawing inspiration from BERT (Devlin et al. 2019). By leveraging a bidirectional Transformer and masked token prediction, MaskGIT enables parallel decoding, significantly accelerating inference while preserving high image quality. Its iterative refinement mechanism also makes it highly effective for tasks such as inpainting, outpainting, and local editing. Muse (Chang et al. 2023) extends the masked generative approach of MaskGIT to T2I generation, leveraging a pre-trained language model for text conditioning. This method enables highly efficient parallel decoding while maintaining stateof-the-art image quality, outperforming both Autoregressive and Diffusion models in terms of speed and fidelity. Recently, Meissonic (Bai et al. 2025) has incorporated novel architectural designs, enhanced positional encoding methods, and refined sampling techniques to achieve T2I generation performance comparable to state-of-the-art Diffusion Models like Stable Diffusion XL (Podell et al. 2024).

Unmasking Order of MGTs MGTs typically employ random order or confidence-based order for token unmasking (Chang et al. 2022; Bai et al. 2025). There are a few studies on the advanced unmasking (sampling) order of MGTs, but no research has yet focused on attribute binding in compositional T2I generation. Lee et al. (2023) propose Text-Conditioned Token Selection (TCTS) and Frequency Adaptive Sampling (FAS), which refine the sampling process to improve image quality. Lezama et al. (2022) introduce Token-Critic, an auxiliary Transformer that enhances MaskGIT’s sampling by identifying implausible tokens. MaskSketch (Bashkirova et al. 2023) tackles the sketch-toimage synthesis task by extracting attention maps from pretrained masked image generation models when a sketch input is provided, and using them to prioritize the sampling of tokens with high structural similarity. Besnier et al. (2025)

AttentionMaps “a pink apple and a car”

Attention to

Attention to

Attention to

Unmasking w/o Contrastive Attention Guidance

“pink”

“apple”

“car”

Pretrained Model

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

|[Figure 16]|
|---|

|[Figure 17]|
|---|

[Figure 18]

[Figure 19]

|[Figure 20]|
|---|

Transformer Blocks

−

min

…

,

Unmasking w/ Contrastive Attention Guidance

Transformer Blocks

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

anchor (“apple”) positive (“pink”) negative (“car”)

Contrastive Attention Score Map of

Transformer Blocks

“apple”

Guides token unmasking here at t = 3 to generate “apple”

t = 0 t = 1 t = 3 t = 15 t = 63

text image

|[Figure 27]|
|---|

- Figure 2: Overview of Unmasking with Contrastive Attention Guidance (UNCAGE). Without UNCAGE, as shown at t = 3, the model fails to select (unmask) the apple-related token despite reasonable candidate, resulting in an image that contains only the car. In contrast, with UNCAGE, it effectively id that clearly represent each object, allowing the model to select the apple-related token at t = 3 and successfully the car and the apple in the final image.

|[Figure 28]<br><br>generating a<br><br>identifies tokens generate both|
|---|

introduce the Halton Scheduler, which improves token selection by leveraging a low-discrepancy sequence to enhance the image diversity and quality while ensuring a more uniform distribution of information across sampling steps.

cross-attention maps to better align with linguistic bindings, outperforming Attend-and-Excite. In addition, Predicated Diffusion (Sueyoshi and Matsubara 2024) introduces a predicate logic-based attention guidance framework, leveraging fuzzy logic to enforce text-image alignment.

[Figure 29]

While these methods improve image quality, most of them require additional training, and none of them address compositional T2I generation or the challenge of attribute binding. Unlike prior works, we focus on attribute binding in compositional T2I generation and propose an effective training-free method with negligible additional cost.

While these methods improve compositional T2I generation, they are all designed for Diffusion Models, and this problem has not been explored for MGTs. In addition, these methods rely on iterative refinement via gradient computation and incur substantial inference overhead, while we propose a training-free method with negligible inference cost.

##### Text-to-Image (T2I) Generation

### Preliminaries

During the era of VAEs (Kingma, Welling et al. 2013) and GANs (Goodfellow et al. 2014), research focused mainly on simpler image generation tasks, such as unconditional or class-to-image generation. However, with the success of Diffusion Models (Ho, Jain, and Abbeel 2020), which enable more stable and diverse generation, research on T2I generation has significantly advanced (Rombach et al. 2022; Esser et al. 2024). Meanwhile, inspired by the success of Autoregressive Models like GPT (Brown et al. 2020) in language modeling, there has been a growing interest in applying them to T2I generation (Sun et al. 2024), demonstrating that autoregressive modeling is also effective in this domain. Furthermore, research has recently begun to explore T2I generation by MGTs (Bai et al. 2025), which aims to address the limitations of autoregressive modeling.

Unmasking Order of MGTs Unlike Autoregressive Models, MGTs predict all tokens in parallel at each timestep. However, only a subset of these predictions is selected (unmasked) for all subsequent timesteps, while the rest are discarded (re-masked) for future prediction. For MGTs in image generation, unmasking tokens in a purely random order likely leads to artifacts in the final image. To address this issue, most MGTs, including MaskGIT (Chang et al. 2022) and Meissonic (Bai et al. 2025), use the logits as confidence scores to guide the unmasking order. Additionally, random noise is added for maintaining stochasticity. In practice, after sampling each token, the logit value of each token is used as confidence scores Fc(t), where Fc(t)[i,j] denotes the logit value of the token at position (i,j) at timestep t. Stochasticity is implemented via adding Gumbel noises Fg(t), where the temperature of the Gumbel noise is linearly decreased from 1.0 to 0.01 across timesteps (Bai et al. 2025). Finally, the unmasking order scores F(t) are computed as the sum of Fc(t) and Fg(t). The top-k tokens are unmasked based on their scores, where k is determined by a cosine schedule:

Compositional T2I Generation As T2I generation has advanced, focus has shifted to compositional T2I generation that renders multiple objects and their attributes based on the prompt. In this setting, ensuring alignment between the prompt and the image is crucial, particularly because attributes of different objects can become entangled during compositional generation. Attend-and-Excite (Chefer et al. 2023) incorporates an attention map-based loss, refining the output at each denoising step by computing gradients and adjusting the generation process accordingly. By leveraging syntactic structure, SynGen (Rassin et al. 2023) adjusts

F(t) = Fc(t) + Fg(t). (1)

Recently, Besnier et al. (2025) proposed Halton Scheduler that utilizes input-independent quasi-random Halton sequences for pre-determined unmasking patterns, reducing

the correlation between tokens unmasked in the same step and maximizing information gain.

Challenges in Compositional T2I Generation As shown in Figure 1, due to misaligned attribute binding, existing unmasking order methods in MGTs, including Halton Scheduler, often fail at compositional T2I generation. Similar issues occur in Diffusion Models, where inaccurate attention maps have been identified as a key cause of attribute mismatch, and addressed through attention-based losses and gradient-based refinement. However, these methods are not directly applicable to MGTs. Unlike in Diffusion Models, once a token is unmasked during the generation in MGTs, it becomes fixed and cannot be refined further, as it is directly used in the final image. This highlights the importance of the unmasking strategy for compositional T2I generation.

### Method

In this section, we propose Unmasking with Contrastive Attention Guidance (UNCAGE), a novel training-free unmasking order method for compositional T2I generation, which is outlined in Figure 2 and Algorithm 1.

Notation From a given text prompt P (e.g., “an Attribute1 Object1 and an Attribute2 Object2”), we extract a set of subject tokens S. This set is composed of object tokens (forming the set O) and attribute tokens (forming the set A), such that S = O ∪ A. For each subject token s ∈ S, we define Mts as the attention map of image tokens attending to token s at timestep t. For each object token o ∈ O (e.g., Object1), we define No as the set of subject tokens forming negative pairs with o (e.g., Attribute2 and Object2), and Po as the set of subject tokens forming positive pairs with o (e.g., Attribute1 and Object1), where Po also includes o itself.

##### Motivation

Suppose the image token at position [i,j] is unmasked at timestep t. We want this token to clearly represent the individual object o (e.g., Object1) with its corresponding attributes. For this, we identify two key constraints: i) for positive pairs (po ∈ Po), and ii) for negative pairs (no ∈ No).

Positive Pair Constraint For position [i,j] to distinctively represent object o (e.g., Object1), not only object o itself but also the other subjects that form positive pairs with object o (po ∈ Po, e.g., Attribute1) should have high attention scores at the same position [i,j].

For example, if Mto[i,j] is low, the token may not contribute meaningfully to object o (e.g., Object1) at timestep t, potentially resulting in a missing object in the final image. If Mto[i,j] is high but Mp

t [i,j] is low, while object o (e.g., Object1) may be generated successfully, its corresponding attributes (e.g., Attribute1) may fail to be properly represented, resulting in attribute leakage. Therefore, the value of minp

o

o∈Po(Mp

t [i,j]) should be large enough.

o

Negative Pair Constraint In contrast, for position [i,j] to distinctively represent object o (e.g., Object1), the attention scores of subjects that form negative pairs with object o (no ∈ No, e.g., Object2) should be low at position [i,j].

Algorithm 1: Unmasking with Contrastive Attention Guidance (UNCAGE) Input: prompt P, previously unmasked tokens prev tokens, current timestep t, Masked Generative Transformer MGT, a set of subject token indices S

Parameter: guidance weight wa Output: unmasked tokens at timestep t

- 1: pred tokens, Mt ← MGT(P,prev tokens,t)

- 2: for s ∈ S do
- 3: Mts ← Mt[:,:,s]
- 4: Mts ← Gaussian(Mts)
- 5: end for
- 6: for each position [i,j] in the attention map do
- 7: Fa(t)[i,j] ← maxo∈O(minp

o∈Po(Mp

o

t [i,j]) − maxn

o∈No(Mn

o

t [i,j]))

- 8: end for
- 9: F(t) ← Fc(t) + Fg(t) + waFa(t)
- 10: unmask pos ← arg topk(F(t),k = cos sched(t))

- 11: return unmask(unmask pos,pred tokens)

For example, if both Mto[i,j] and Mn

t [i,j] are high, the image token is likely to encode attributes from both objects, increasing the risk of object mixture when it is selected and preserved in the final image. Therefore, the value of maxn

o

o∈No(Mn

t [i,j]) should be small enough.

o

##### Proposed Method: UNCAGE

Building on this insight, we propose Unmasking with Contrastive Attention Guidance (UNCAGE) to unmask tokens that clearly represent individual objects. Figure 2 shows the overview of UNCAGE.

Contrastive Attention Guidance For each object o, we construct two guidance signals: i) minp

o∈Po(Mp

t [i,j]) for positive pairs, and ii) −maxn

o

o∈No(Mn

t [i,j]) for negative pairs. Combining these, we compute the contrastive attention score Fao(t)[i,j] to measure how distinctively position [i,j] represents object o as below:

o

positive pair

negative pair

contrastive

(Mp

(Mn

Fao(t)[i,j] =

t [i,j]) (2)

t [i,j])−

o

min

max

o

po∈Po

no∈No

Then, the final contrastive attention score of each position [i,j] corresponds to the most distinctive object’s score, obtained by taking the maximum across all objects:

Fao(t)[i,j]. (3)

Fa(t)[i,j] = max o∈O

A high Fa(t)[i,j] indicates that a token at position [i,j] clearly represents an object with its corresponding attributes. Note that Equation (3) reduces to absolute difference as

Fa(t)[i,j] = |MtObject1[i,j] − MtObject2[i,j]| in the simple case, “an Object1 and an Object2.” The method can also be generalized for use with more than two objects, with results shown in the experiment section.

Implementation of UNCAGE We first obtain attention maps Mt at the given timestep t (Algorithm 1, line 1),

similar to Attend-and-Excite (Chefer et al. 2023). Specifically, the attention maps from the single-modal Transformer blocks are averaged over all blocks and all attention heads, and then rescaled. We use the attention weights after softmax, with image tokens as queries and text tokens as keys. The intuition is that the image tokens will be semantically related to the text tokens in proportion to the amount of attention they pay to the text tokens. After extracting these attention maps, we smooth them using Gaussian blur with a standard deviation of 2.0 (Algorithm 1, lines 2–5).

Next, we compute the contrastive attention guidance Fa using Eq.2 and Eq.3 at each spatial location [i,j] (Algorithm 1, lines 6–8). Then, we add Fa(t) to the baseline unmasking score Fc(t) + Fg(t) to obtain the final unmasking score F(t) (Algorithm 1, line 9):

F(t) = Fc(t) + Fg(t) + waFa(t). (4)

The scores can be combined using weights, with wa denoting the scale of guidance. The original unmasking scheme

corresponds to wa = 0.0. Finally, we select the top-k tokens with the highest F(t) values for unmasking, and proceed to the next timestep (Algorithm 1, lines 10-11).

### Experiments

##### Experimental Setup

For the pretrained MGT, we use Meissonic (Bai et al. 2025), a state-of-the-art MGT for T2I generation. The model generates 1024×1024 images using parallel decoding with 64 timesteps and a cosine schedule. For existing unmasking methods, we test i) random unmasking (Fg), ii) confidencebased unmasking (Fc), iii) Meissonic’s approach (Fc + Fg), and iv) Halton Scheduler (Besnier et al. 2025), a state-ofthe-art unmasking method for image generation.

For UNCAGE, we test: i) using only negative pair guidance, ii) using only positive pair guidance, and iii) using both (our full contrastive guidance). We use wa = 3 as the default hyperparameter setting, and we also test the effect of varying wa in an ablation study. As the overall structure of the final output in MGTs is largely determined during the early timesteps, we apply UNCAGE only during the first 16 steps and use the baseline (Fc + Fg) for the remaining 48 steps as our default setting. We also test applying UNCAGE for 64 (all) timesteps, denoted as UNCAGE†. A detailed ablation study on guidance timesteps is in the Appendix.

Benchmark Datasets We use two benchmark datasets: the Attend-and-Excite dataset (Chefer et al. 2023) and the SSD dataset (Weimin, Jieke, and Meng 2025). The Attendand-Excite dataset is categorized into three main groups: Animal-Animal, Animal-Object, and Object-Object. For instance, the Animal-Object category includes prompts such as “a mouse and a red car.” This category is constructed using combinations of 12 animals and 12 objects, where each object is assigned one of 11 different colors. Each set contains 66, 144, and 66 prompts, respectively. The Similar Subjects dataset (SSD) is designed to evaluate compositional T2I generation in challenging settings. It consists of 31 prompts with two objects and 22 prompts with three objects involving semantically similar subjects, such as “A

Table 1: Quantitative results on the Attend-and-Excite and SSD datasets. UNCAGE is applied for the first 16 steps; † indicates all 64 steps. Bold and underlined denote best and second-best results. UNCAGE consistently outperforms existing methods across benchmarks and evaluation metrics.

(a) CLIP text-image similarities.

Attend-and-Excite SSD

Method / Dataset

Avg. A.-A. A.-O. O.-O. Two Three

Random (Fg) 26.50 31.81 30.97 27.90 28.14 29.06 Confidence (Fc) 28.37 35.33 36.69 29.34 29.64 31.87 Meissonic (Fc + Fg) 29.82 36.15 37.65 29.73 30.24 32.72 Halton Scheduler 30.81 36.15 37.52 30.01 30.46 32.99 UNCAGE (n. only) 30.43 36.11 37.43 30.08 30.52 32.91 UNCAGE (p. only) 30.22 36.05 37.45 30.11 30.53 32.87 UNCAGE (ours) 30.43 36.28 37.65 30.08 30.52 32.99 UNCAGE† (ours) 30.50 36.31 37.54 30.17 30.61 33.03

(b) CLIP text-text similarities.

Attend-and-Excite SSD

Method / Dataset

Avg. A.-A. A.-O. O.-O. Two Three

Random (Fg) 66.70 76.98 69.76 64.03 65.76 68.65 Confidence (Fc) 73.12 85.14 84.88 67.71 71.04 76.38 Meissonic (Fc + Fg) 73.98 87.05 86.34 69.21 72.24 77.76 Halton Scheduler 76.47 87.27 86.10 69.85 72.97 78.53 UNCAGE (n. only) 77.05 87.00 86.33 69.94 72.66 78.60 UNCAGE (p. only) 76.36 86.53 85.55 70.12 72.58 78.23 UNCAGE (ours) 77.05 87.16 86.77 69.94 72.66 78.72 UNCAGE† (ours) 77.21 86.92 85.88 70.44 73.50 78.79

(c) GPT-based evaluation.

Attend-and-Excite SSD

Method / Dataset

Avg. A.-A. A.-O. O.-O. Two Three

Random (Fg) 5.94 8.52 6.57 5.65 5.59 6.45 Confidence (Fc) 5.78 9.19 8.13 5.18 4.57 6.57 Meissonic (Fc + Fg) 6.18 9.65 8.52 5.59 5.03 6.99 Halton Scheduler 6.65 9.67 8.88 5.79 5.35 7.27 UNCAGE (n. only) 6.80 9.65 8.66 6.16 5.37 7.33 UNCAGE (p. only) 6.95 9.56 8.66 6.20 5.40 7.35 UNCAGE (ours) 6.80 9.62 8.74 6.16 5.37 7.34 UNCAGE† (ours) 6.86 9.61 8.81 6.27 5.36 7.38

(d) User study results.

Attend-and-Excite SSD

Method / Dataset

Avg. Animal-Animal Two Objects

Tie 18.2% 31.7% 25.0% Meissonic (baseline) 30.2% 23.0% 26.6% UNCAGE (ours) 51.6% 45.3% 48.4%

leopard and a tiger.” For evaluation, we generate 4 images per prompt for the Attend-and-Excite dataset and 16 images per prompt for the SSD dataset using different random seeds. In total, 1,968 images are generated and used for evaluation.

Evaluation Metrics We measure CLIP (Radford et al. 2021) text-image similarity and CLIP text-text similarity, following Chefer et al. (2023). CLIP text-image similarity is computed via cosine similarity between the CLIP embeddings of the generated image and the prompt. Text-text similarity uses BLIP (Li et al. 2022) to convert the generated images to text captions, which are then compared with

“a bird and a horse” “a cat and a frog” “a turtle and a pink balloon” “a red bench and a yellow clock”

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

UNCAGE(ours)Meissonic(baseline)HaltonScheduler

[Figure 38]

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

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

- Figure 3: Qualitative results on Attend-and-Excite dataset. The same seeds are applied to each prompt across all methods.

the prompt using CLIP. However, as CLIP-based metrics are insufficient for accurately evaluating compositional T2I generation (Chefer et al. 2023; Weimin, Jieke, and Meng 2025), we perform an additional GPT-based evaluation to thoroughly validate the effectiveness of UNCAGE. Similar to prior works (Ku et al. 2024; Kang, Galim, and Koo 2024; Kang et al. 2025), we conduct GPT-based evaluation using the template in the Appendix. Finally, we conduct a user study to compare the Meissonic (baseline) and UNCAGE. Ten respondents compared both methods for each prompt in the Attend-and-Excite (Animal-Animal) and SSD (Two Objects) datasets, evaluating 4 and 16 images per prompt, respectively. In total, each respondent evaluated 760 images and selected the better result for each prompt, or chose a tie.

##### Quantitative Results

- Table 1 presents the quantitative results. CLIP text-image similarity, CLIP text-text similarity, and GPT-based evaluation exhibit consistent trends, showing general alignment across different settings.

First, Meissonic (baseline, Fc + Fg) performs poorly

on the Animal-Animal and the SSD dataset, but excels on the Animal-Object and Object-Object categories. This is likely because, in the Animal-Object and Object-Object categories, the objects are semantically distinct, making object mixture less likely. In contrast, in the Animal-Animal category and the SSD dataset, object mixture is more likely due to higher semantic similarity between entities.

For UNCAGE, using only negative guidance (n. only) or positive guidance (p. only) already outperforms Meissonic across all metrics in terms of average scores. Moreover, when using both (contrastive), UNCAGE overall outperforms Halton Scheduler. This demonstrates that both our proposed negative pair guidance and positive pair guidance work effectively, and combining them into contrastive guidance makes the method even more powerful. The gain is especially significant on the Animal-Animal dataset and the SSD dataset, where the baseline performs poorly. In contrast, for the Animal-Object and Object-Object datasets, where the baseline already performs well, the benefit of UNCAGE is relatively modest. In summary, UNCAGE achieves state-of-the-art performance across multiple bench-

similarityscore

CLIP (txt-img)

30.5

30.0

0 1 2 3 4 5

guidance weight wa

CLIP (txt-txt)

78.0

76.0

74.0

0 1 2 3 4 5

guidance weight wa

- Figure 4: Ablation results for contrastive attention guidance weight on the Attend-and-Excite (Animal-Animal) dataset.

Meissonic (baseline)

UNCAGE (ours)

“a lion and a frog” “a dog and a black apple”

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Figure 5: Cases without improvement.

marks and metrics compared to Meissonic and Halton Scheduler, with especially greater gain on challenging tasks.

Notably, applying UNCAGE for all 64 timesteps (UNCAGE†) yields even greater gains and the best average scores across all metrics. Nevertheless, as UNCAGE† exhibits dataset-dependent variance in its effectiveness, we default to applying UNCAGE for the first 16 steps to ensure more uniform performance gains across all datasets.

User Study Results As shown in Table 1d, UNCAGE was preferred nearly twice as often as the baseline. This suggests that UNCAGE is effective not only in improving quantitative metrics but also in better aligning with human preferences.

##### Qualitative Results

- Figure 3 presents the qualitative results on the Attendand-Excite dataset, demonstrating that UNCAGE generally produces better results compared to Meissonic and Halton Scheduler. In the case of “a bird and a horse,” Meissonic and Halton Scheduler fail to generate both animals independently. Instead, they either generate only a horse or produce a fused entity where the bird and the horse are not clearly distinguishable. In contrast, UNCAGE effectively separates and accurately generates both the bird and the horse. For “a turtle and a pink balloon,” Meissonic, Halton Scheduler, and UNCAGE all successfully generate the turtle and the balloon as separate objects. However, while the generated balloon by UNCAGE is correctly pink in all cases, other methods often produce a balloon in a different color instead of pink.

##### Ablation Study

Additional ablation studies, including the number of guidance timesteps, Gaussian smoothing of the attention map, Fc and Fg terms, are provided in the Appendix.

Ablation on Guidance Weight wa To study the effect of wa, we conduct an ablation study on the Animal-Animal dataset by varying wa from 0 to 5 in increments of 0.5. As shown in Figure 4, increasing wa from 0 gradually improves both CLIP text-image and text-text similarity scores, until around wa = 3, after which performance begins to saturate or slightly decline. These upward-concave trends suggest that UNCAGE is effective and functioning as intended, a behavior commonly observed in other guidance methods such as classifier-free guidance (Ho and Salimans 2021).

##### Inference Overhead Analysis

To demonstrate that UNCAGE introduces negligible inference overhead, we measure the runtime on an A100 GPU. Meissonic takes 16 seconds to generate an image over 64 steps. UNCAGE adds 0.0013 seconds per step for the first 16 steps, resulting in 0.0208 seconds of additional latency, which corresponds to 0.13% of the total runtime. In contrast, Chefer et al. (2023) report that Attend-and-Excite increases Stable Diffusion’s inference time on an A100 from 5.6 seconds to 9.7–15.4 seconds, roughly doubling the runtime.

### Limitations and Future Work

First, while UNCAGE improves compositional T2I generation, it does not always yield better results. Since UNCAGE is training-free and modifies only the unmasking order without changing the pretrained model, its effectiveness can be limited. For example, in Figure 5, the prompt “a dog and a black apple” results in “a black dog and an apple” instead. This likely reflects pretrained bias, where black dogs are common and apples are typically red.

Second, while UNCAGE outperforms existing methods in MGTs with negligible inference overhead, the gain is relatively modest compared to guidance methods in Diffusion Models (Chefer et al. 2023). This is primarily due to the fundamental differences between UNCAGE and the methods commonly used in Diffusion Models. In Diffusion Models, guidance is explicitly provided through iterative refinement via gradient computation, which leads to larger gains at the cost of significantly increased inference time. As future work, it would be interesting to explore methods that achieve greater gains in MGTs, even at higher cost.

### Conclusion

While interest in T2I generation using Masked Generative Transformers (MGTs) has recently increased, these often struggle to accurately bind attributes and achieve proper text-image alignment in compositional T2I generation. To address this, we have proposed Unmasking with Contrastive Attention Guidance (UNCAGE), which is, to our knowledge, the first method specifically designed for compositional T2I generation with MGTs. By leveraging attention maps to prioritize tokens that clearly represent individual objects and guide the unmasking order accordingly, UNCAGE improves compositional fidelity in a training-free manner. UNCAGE demonstrates consistent improvements via quantitative and qualitative experiments across multiple benchmarks and metrics, with negligible inference overhead.

### References

Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Bai, J.; Ye, T.; Chow, W.; Song, E.; Chen, Q.-G.; Li, X.; Dong, Z.; Zhu, L.; and YAN, S. 2025. Meissonic: Revitalizing Masked Generative Transformers for Efficient HighResolution Text-to-Image Synthesis. In The Thirteenth International Conference on Learning Representations.

Bashkirova, D.; Lezama, J.; Sohn, K.; Saenko, K.; and Essa,

- I. 2023. Masksketch: Unpaired structure-guided masked image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1879–1889. Besnier, V.; Chen, M.; Hurych, D.; Valle, E.; and Cord, M. 2025. Halton Scheduler for Masked Generative Image Transformer. In The Thirteenth International Conference on Learning Representations. Blattmann, A.; Dockhorn, T.; Kulal, S.; Mendelevitch, D.; Kilian, M.; Lorenz, D.; Levi, Y.; English, Z.; Voleti, V.; Letts, A.; et al. 2023. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127. Brown, T.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J. D.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell, A.; Agarwal, S.; Herbert-Voss, A.; Krueger, G.; Henighan, T.; Child, R.; Ramesh, A.; Ziegler, D.; Wu, J.; Winter, C.; Hesse, C.; Chen, M.; Sigler, E.; Litwin, M.; Gray, S.; Chess, B.; Clark, J.; Berner, C.; McCandlish, S.; Radford, A.; Sutskever, I.; and Amodei, D. 2020. Language Models are Few-Shot Learners. In Larochelle, H.; Ranzato, M.; Hadsell, R.; Balcan, M.; and Lin, H., eds., Advances in Neural Information Processing Systems, volume 33, 1877–1901. Curran Associates, Inc. Chang, H.; Zhang, H.; Barber, J.; Maschinot, A.; Lezama,
- J.; Jiang, L.; Yang, M.-H.; Murphy, K.; Freeman, W. T.; Rubinstein, M.; et al. 2023. Muse: Text-to-image generation via masked generative transformers. In Proceedings of the 40th International Conference on Machine Learning, 4055– 4075. Chang, H.; Zhang, H.; Jiang, L.; Liu, C.; and Freeman, W. T.

2022. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11315–11325.

Chefer, H.; Alaluf, Y.; Vinker, Y.; Wolf, L.; and Cohen-Or,

- D. 2023. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG), 42(4): 1–10. Chen, M.; Radford, A.; Child, R.; Wu, J.; Jun, H.; Luan, D.; and Sutskever, I. 2020. Generative Pretraining From Pixels. In III, H. D.; and Singh, A., eds., Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, 1691–1703. PMLR. Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Burstein, J.; Doran, C.; and

Solorio, T., eds., Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 4171–4186. Minneapolis, Minnesota: Association for Computational Linguistics.

Esser, P.; Kulal, S.; Blattmann, A.; Entezari, R.; M¨uller, J.; Saini, H.; Levi, Y.; Lorenz, D.; Sauer, A.; Boesel, F.; Podell, D.; Dockhorn, T.; English, Z.; and Rombach, R. 2024. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. In Forty-first International Conference on Machine Learning.

Goodfellow, I.; Pouget-Abadie, J.; Mirza, M.; Xu, B.; Warde-Farley, D.; Ozair, S.; Courville, A.; and Bengio, Y. 2014. Generative Adversarial Nets. In Ghahramani, Z.; Welling, M.; Cortes, C.; Lawrence, N.; and Weinberger, K., eds., Advances in Neural Information Processing Systems, volume 27. Curran Associates, Inc.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising Diffusion Probabilistic Models. In Larochelle, H.; Ranzato, M.; Hadsell, R.; Balcan, M.; and Lin, H., eds., Advances in Neural Information Processing Systems, volume 33, 6840–6851. Curran Associates, Inc.

Ho, J.; and Salimans, T. 2021. Classifier-Free Diffusion Guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications.

Kang, M.; Zhu, J.-Y.; Zhang, R.; Park, J.; Shechtman, E.; Paris, S.; and Park, T. 2023. Scaling up gans for text-toimage synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 10124– 10134.

Kang, W.; Galim, K.; and Koo, H. I. 2024. Eta inversion: Designing an optimal eta function for diffusion-based real image editing. In European Conference on Computer Vision, 90–106. Springer.

Kang, W.; Galim, K.; Koo, H. I.; and Cho, N. I. 2025. Counting guidance for high fidelity text-to-image synthesis. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 899–908. IEEE.

Kingma, D. P.; Welling, M.; et al. 2013. Auto-encoding variational bayes.

Ku, M.; Jiang, D.; Wei, C.; Yue, X.; and Chen, W. 2024. VIEScore: Towards Explainable Metrics for Conditional Image Synthesis Evaluation. In Ku, L.-W.; Martins, A.; and Srikumar, V., eds., Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 12268–12290. Bangkok, Thailand: Association for Computational Linguistics.

Lee, J.; Jang, S.; Jo, J.; Yoon, J.; Kim, Y.; Kim, J.-H.; Ha, J.-W.; and Hwang, S. J. 2023. Text-conditioned sampling framework for text-to-image generation with masked generative models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 23252–23262.

Lezama, J.; Chang, H.; Jiang, L.; and Essa, I. 2022. Improved masked image generation with token-critic. In European Conference on Computer Vision, 70–86. Springer.

Li, J.; Li, D.; Xiong, C.; and Hoi, S. 2022. BLIP: Bootstrapping Language-Image Pre-training for Unified VisionLanguage Understanding and Generation. In Chaudhuri, K.; Jegelka, S.; Song, L.; Szepesvari, C.; Niu, G.; and Sabato,

- S., eds., Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, 12888–12900. PMLR.

Liu, D.; Zhao, S.; Zhuo, L.; Lin, W.; Qiao, Y.; Li, H.; and Gao, P. 2024. Lumina-mgpt: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining. arXiv preprint arXiv:2408.02657.

Podell, D.; English, Z.; Lacey, K.; Blattmann, A.; Dockhorn,

- T.; M¨uller, J.; Penna, J.; and Rombach, R. 2024. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. In The Twelfth International Conference on Learning Representations.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; Krueger, G.; and Sutskever, I. 2021. Learning Transferable Visual Models From Natural Language Supervision. In Meila, M.; and Zhang, T., eds., Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, 8748–8763. PMLR.

Ramesh, A.; Pavlov, M.; Goh, G.; Gray, S.; Voss, C.; Radford, A.; Chen, M.; and Sutskever, I. 2021. Zero-Shot Textto-Image Generation. In Meila, M.; and Zhang, T., eds., Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, 8821–8831. PMLR.

Rassin, R.; Hirsch, E.; Glickman, D.; Ravfogel, S.; Goldberg, Y.; and Chechik, G. 2023. Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment. In Thirty-seventh Conference on Neural Information Processing Systems.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 10684– 10695.

Saharia, C.; Chan, W.; Saxena, S.; Li, L.; Whang, J.; Denton,

- E.; Ghasemipour, S. K. S.; Gontijo-Lopes, R.; Ayan, B. K.; Salimans, T.; Ho, J.; Fleet, D. J.; and Norouzi, M. 2022. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. In Oh, A. H.; Agarwal, A.; Belgrave, D.; and Cho, K., eds., Advances in Neural Information Processing Systems.

Song, J.; Meng, C.; and Ermon, S. 2021. Denoising Diffusion Implicit Models. In International Conference on Learning Representations.

Sueyoshi, K.; and Matsubara, T. 2024. Predicated Diffusion: Predicate Logic-Based Attention Guidance for Text-toImage Diffusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8651–8660.

Sun, P.; Jiang, Y.; Chen, S.; Zhang, S.; Peng, B.; Luo, P.; and Yuan, Z. 2024. Autoregressive model beats diffu-

sion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525.

Tian, K.; Jiang, Y.; Yuan, Z.; PENG, B.; and Wang, L. 2024. Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.-A.; Lacroix, T.; Rozi`ere, B.; Goyal, N.; Hambro, E.; Azhar, F.; et al. 2023a. LLaMA: Open and Efficient Foundation Language Models. arXiv preprint arXiv:2302.13971. Touvron, H.; Martin, L.; Stone, K.; Albert, P.; Almahairi, A.; Babaei, Y.; Bashlykov, N.; Batra, S.; Bhargava, P.; Bhosale, S.; et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Voleti, V.; Yao, C.-H.; Boss, M.; Letts, A.; Pankratz, D.; Tochilkin, D.; Laforte, C.; Rombach, R.; and Jampani, V. 2024. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. In European Conference on Computer Vision, 439–457. Springer.

Weimin, Q.; Jieke, W.; and Meng, T. 2025. Self-Cross Diffusion Guidance for Text-to-Image Synthesis of Similar Subjects. In CVPR.

### Details of GPT-based Evaluation

While CLIP-based metrics are straightforward and widely used, they have limitations in accurately evaluating compositional T2I generation (Chefer et al. 2023; Weimin, Jieke, and Meng 2025). Therefore, we conduct an additional GPTbased evaluation to obtain more accurate and fine-grained assessments, and to further validate the effectiveness of UNCAGE. In this section, we describe the evaluation template used and provide sample responses generated by GPT during the evaluation.

##### GPT-based Evaluation Template

We use the template in Listing 1 for GPT-based evaluation. It is adapted from the prompt proposed by Ku et al. (2024), with modifications tailored to our task. We use the model gpt-4o-2024-11-20 provided by the OpenAI API.

Listing 1: GPT-based evaluation template.

|You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image (s) based on the given rules. You will have to give your output in this way ( Keep your reasoning concise and short.) : { \"score\" : [...], \"reasoning\" : \"...\" } RULES: Prompt and image will be provided: The objective is to evaluate how the image is well aligned to the prompt. On scale of 0 to 10: A score from 0 to 10 will be given based on the success of the alignment. (0 indicates that the image does not follow the prompt at all. 10 indicates that the image in the image follows the<br><br>prompt perfectly.) Put the score in a list such that output score = [score]<br><br>|
|---|

##### Example GPT-based Evaluation Outputs

To demonstrate that our GPT-based evaluation provides reliable assessments, we include sample responses generated by GPT. Specifically, we test the prompt “a monkey and a frog” from the Animal-Animal category of the Attend-and-Excite dataset. The outputs generated by Meissonic and UNCAGE are shown in Figure 6.

As shown in Figure 6a, Meissonic fails to generate a distinct monkey and frog, instead producing a green creature that appears to be a mixture of both. In contrast, as shown in Figure 6b, UNCAGE successfully generates a clearly separated and recognizable monkey and frog.

GPT assigns a score of 5 to the baseline output, accurately describing it as “a creature that appears to be a hybrid of a monkey and a frog.” In comparison, it gives a score of 10

to our output, noting that it “depicts both a monkey and a frog clearly and accurately.” These examples illustrate that GPT’s assessments closely align with human judgment, supporting the reliability of our GPT-based evaluation.

##### “a monkey and a frog”

[Figure 86]

(a) Output from Meissonic (baseline), failing to generate distinct objects.

[Figure 87]

(b) Output from UNCAGE (ours), generating a clearly separated monkey and frog.

Figure 6: Example images used for GPT-based evaluation.

- Listing 2: GPT evaluation output for the image generated by Meissonic (baseline).

|{ \"score\" : [5], \"reasoning\" : \"The image depicts a creature that appears to be a hybrid of<br><br>a monkey and a frog, rather than<br><br>showing a distinct monkey and a frog as separate entities. While creative, it does not fully align with the prompt's request for both animals to be present .\" }<br><br>|
|---|

- Listing 3: GPT evaluation output for the image generated by UNCAGE (ours).

|{ \"score\" : [10], \"reasoning\" : \"The image perfectly aligns with the prompt, depicting both a monkey and a frog clearly and accurately. The composition is wellexecuted, and the subjects are easily identifiable.\" }<br><br>|
|---|

### Comparison with Stable Diffusion

To better understand the effectiveness of UNCAGE in Masked Generative Transformers, we compare it with the effectiveness of Attend-and-Excite (Chefer et al. 2023) applied to Diffusion Models. In Table 2, we report the performance of Attend-and-Excite applied to Stable Diffusion 2.1.

First, we observe that the Stable Diffusion 2.1 baseline performs consistently well across all datasets, with relatively small performance differences between them. This

contrasts with Meissonic (Bai et al. 2025), which performs poorly on the Animal-Animal and SSD datasets, but achieves strong performance on the Animal-Object and Object-Object datasets.

Overall, we observe that the performance gains from Attend-and-Excite on Stable Diffusion 2.1 are larger than those from UNCAGE on Meissonic. However, this difference stems from the fundamental nature of the two approaches. Attend-and-Excite relies on gradient-based iterative refinement, which introduces significant inference overhead compared to its baseline. In contrast, our method is training-free and guides the sampling order with negligible inference cost.

As discussed in the experiment section, UNCAGE incurs only 0.13% additional runtime, whereas Attend-andExcite roughly doubles the inference time. Considering this inference overhead, UNCAGE demonstrates effective performance gains at minimal cost.

- Table 2: Comparison with improvements in Diffusion Models. We report the performance of Attend-and-Excite (Chefer et al. 2023) applied to Stable Diffusion 2.1 (Rombach et al. 2022).

(a) CLIP text-image similarities.

Attend-and-Excite SSD

Method / Dataset

Avg. A.-A. A.-O. O.-O. Two Three

Meissonic 29.82 36.15 37.65 29.73 30.24 32.72 UNCAGE (ours) 30.43 36.28 37.65 30.08 30.52 32.99 UNCAGE† (ours) 30.50 36.31 37.54 30.17 30.61 33.03

Stable Diffusion 2.1 33.12 35.39 34.27 32.07 31.51 33.27 Attend-and-Excite 34.24 36.03 36.74 32.45 32.15 34.32

(b) CLIP text-text similarities.

Attend-and-Excite SSD

Method / Dataset

Avg. A.-A. A.-O. O.-O. Two Three

Meissonic 73.98 87.05 86.34 69.21 72.24 77.76 UNCAGE (ours) 77.05 87.16 86.77 69.94 72.66 78.72 UNCAGE† (ours) 77.21 86.92 85.88 70.44 73.50 78.79

Stable Diffusion 2.1 81.52 82.67 78.04 72.29 73.00 77.50 Attend-and-Excite 85.13 85.15 83.33 73.25 74.80 80.33

(c) GPT-based evaluation.

Attend-and-Excite SSD

Method / Dataset

Avg. A.-A. A.-O. O.-O. Two Three

Meissonic 6.18 9.65 8.52 5.59 5.03 6.99 UNCAGE (ours) 6.80 9.62 8.74 6.16 5.37 7.34 UNCAGE† (ours) 6.86 9.61 8.81 6.27 5.36 7.38

Stable Diffusion 2.1 7.62 8.33 6.41 7.48 6.87 7.34 Attend-and-Excite 8.85 9.36 8.14 8.67 7.89 8.58

### Additional Ablation Study

In this section, we conduct additional ablation studies to further analyze UNCAGE. In the ablation study, we set wa = 3.

##### Ablation on the Number of Guidance Timesteps

In our default setting, UNCAGE is applied for the first 16 timesteps, with the remaining 48 following the baseline. To further investigate its impact, we conduct an ablation with 0 (Meissonic, baseline), 1, 2, 4, 8, 16 (ours), 32, and all 64 timesteps (full), and report the results in Table 3.

First, we observe a consistent trend across all metrics where the average score increases as the number of guidance timesteps grows. This indicates that UNCAGE effectively provides guidance in a direction that improves performance.

Notably, applying UNCAGE during the first 16 timesteps yields results comparable to full guidance, suggesting that the overall structure of the final output is largely determined during the early stages of generation. Furthermore, even applying UNCAGE only during the first 4 timesteps, or even just the first timestep, yields noticeable improvements over the Meissonic (baseline). This highlights the importance of providing guidance at early timesteps and the effectiveness of UNCAGE.

Table 3: Ablation on the number of guidance timesteps.

(a) CLIP text-image similarities.

Attend-and-Excite SSD

Method / Dataset

Avg. A.-A. A.-O. O.-O. Two Three

Meissonic 29.82 36.15 37.65 29.73 30.24 32.72 Halton Scheduler 30.81 36.15 37.52 30.01 30.46 32.99

- UNCAGE (1) 29.84 36.27 37.63 29.87 30.40 32.80

- UNCAGE (2) 30.20 36.25 37.82 29.97 30.44 32.94 UNCAGE (4) 30.27 36.27 37.74 30.02 30.63 32.99 UNCAGE (8) 30.44 36.30 37.62 30.03 30.60 33.00 UNCAGE (16) 30.43 36.28 37.65 30.08 30.52 32.99 UNCAGE (32) 30.47 36.36 37.53 30.16 30.59 33.02 UNCAGE (64) 30.50 36.31 37.54 30.17 30.61 33.03

(b) CLIP text-text similarities.

Attend-and-Excite SSD

Method / Dataset

Avg. A.-A. A.-O. O.-O. Two Three

Meissonic 73.98 87.05 86.34 69.21 72.24 77.76 Halton Scheduler 76.47 87.27 86.10 69.85 72.97 78.53

- UNCAGE (1) 74.55 87.51 86.33 70.01 72.94 78.27

- UNCAGE (2) 75.55 87.48 86.89 69.52 72.57 78.40 UNCAGE (4) 75.75 87.03 87.02 70.08 73.17 78.61 UNCAGE (8) 76.71 86.90 86.92 70.15 73.43 78.82 UNCAGE (16) 77.05 87.16 86.77 69.94 72.66 78.72 UNCAGE (32) 76.98 86.99 86.34 70.19 73.30 78.76 UNCAGE (64) 77.21 86.92 85.88 70.44 73.50 78.79

(c) GPT-based evaluation.

Attend-and-Excite SSD

Method / Dataset

Avg. A.-A. A.-O. O.-O. Two Three

Meissonic 6.18 9.65 8.52 5.59 5.03 6.99 Halton Scheduler 6.65 9.67 8.88 5.79 5.35 7.27

- UNCAGE (1) 6.25 9.67 8.59 5.70 5.21 7.08

- UNCAGE (2) 6.54 9.59 8.74 5.82 5.08 7.15 UNCAGE (4) 6.64 9.64 8.87 6.05 5.18 7.28 UNCAGE (8) 6.90 9.62 8.83 6.07 5.19 7.32 UNCAGE (16) 6.80 9.62 8.74 6.16 5.37 7.34 UNCAGE (32) 6.84 9.64 8.78 6.26 5.39 7.38 UNCAGE (64) 6.86 9.61 8.81 6.27 5.36 7.38

##### Ablation on Gaussian Smoothing of the Attention Map

Following Attend-and-Excite (Chefer et al. 2023), we apply Gaussian smoothing to the attention map when obtaining it for contrastive attention guidance computation in our default setting. We investigate how this smoothing affects the performance of UNCAGE.

As shown in Table 4, UNCAGE without Gaussian smoothing already outperforms Meissonic (baseline) across all metrics in terms of average scores, and in some cases even outperforms UNCAGE (ours). However, our method yields consistently stronger and more stable performance overall. This suggests that Gaussian smoothing helps stabilize the effectiveness of UNCAGE, which is consistent with findings from Attend-and-Excite.

Table 4: Ablation results for Gaussian smoothing of the attention map.

(a) CLIP text-image similarities.

Attend-and-Excite SSD

Method / Dataset

Avg. A.-A. A.-O. O.-O. Two Three

Meissonic 29.82 36.15 37.65 29.73 30.24 32.72 UNCAGE (ours) 30.43 36.28 37.65 30.08 30.52 32.99 UNCAGE w/o blur 30.21 36.13 37.62 29.97 30.39 32.86

(b) CLIP text-text similarities.

Attend-and-Excite SSD

Method / Dataset

Avg. A.-A. A.-O. O.-O. Two Three

Meissonic 73.98 87.05 86.34 69.21 72.24 77.76 UNCAGE (ours) 77.05 87.16 86.77 69.94 72.66 78.72 UNCAGE w/o blur 76.07 86.41 87.07 69.55 73.57 78.53

(c) GPT-based evaluation.

Attend-and-Excite SSD

Method / Dataset

Avg. A.-A. A.-O. O.-O. Two Three

Meissonic 6.18 9.65 8.52 5.59 5.03 6.99 UNCAGE (ours) 6.80 9.62 8.74 6.16 5.37 7.34 UNCAGE w/o blur 6.50 9.56 8.77 6.09 5.24 7.23

##### Ablation on Confidence and Randomness Terms

We use both the confidence (Fc) and randomness (Fg) terms by default when evaluating the effectiveness of UNCAGE. To further investigate the role of contrastive attention guidance (Fa) and its interaction with these terms, we conduct an ablation study with three additional settings:

- (1) UNCAGE w/o Fg term (F(t) = Fc(t) + waFa(t)),
- (2) UNCAGE w/o Fc term (F(t) = Fg(t) + waFa(t)),
- (3) UNCAGE w/o Fc,Fg terms (F(t) = waFa(t)).

We evaluate how each combination affects performance and analyze the contribution of each term to the overall effectiveness of Fa. The results are shown in Table 5.

Notably, both ablations (1) and (2) show clear performance improvements over the setting without UNCAGE.

Table 5: Ablation results for confidence and randomness terms.

(a) CLIP text-image similarities.

Attend-and-Excite SSD

Method / Dataset

Avg. A.-A. A.-O. O.-O. Two Three

Random (Fg) 26.50 31.81 30.97 27.90 28.14 29.06 Confidence (Fc) 28.37 35.33 36.69 29.34 29.64 31.87 Meissonic (Fc + Fg) 29.82 36.15 37.65 29.73 30.24 32.72 UNCAGE (ours) 30.43 36.28 37.65 30.08 30.52 32.99 UNCAGE w/o Fg 29.36 35.96 37.53 29.78 30.01 32.53 UNCAGE w/o Fc 28.39 35.31 36.33 29.10 29.59 31.74 UNCAGE w/o Fc,Fg 28.79 34.88 36.26 29.16 29.57 31.73

(b) CLIP text-text similarities.

Attend-and-Excite SSD

Method / Dataset

Avg. A.-A. A.-O. O.-O. Two Three

Random (Fg) 66.70 76.98 69.76 64.03 65.76 68.65 Confidence (Fc) 73.12 85.14 84.88 67.71 71.04 76.38 Meissonic (Fc + Fg) 73.98 87.05 86.34 69.21 72.24 77.76 UNCAGE (ours) 77.05 87.16 86.77 69.94 72.66 78.72 UNCAGE w/o Fg 76.19 86.02 85.59 69.94 71.32 77.81 UNCAGE w/o Fc 73.20 85.28 83.36 68.43 71.71 76.40 UNCAGE w/o Fc,Fg 74.05 84.01 82.67 69.67 70.93 76.27

(c) GPT-based evaluation.

Attend-and-Excite SSD

Method / Dataset

Avg. A.-A. A.-O. O.-O. Two Three

Random (Fg) 5.94 8.52 6.57 5.65 5.59 6.45 Confidence (Fc) 5.78 9.19 8.13 5.18 4.57 6.57 Meissonic (Fc + Fg) 6.18 9.65 8.52 5.59 5.03 6.99 UNCAGE (ours) 6.80 9.62 8.74 6.16 5.37 7.34 UNCAGE w/o Fg 6.61 9.50 8.66 6.15 5.42 7.27 UNCAGE w/o Fc 6.14 9.20 8.68 5.58 5.14 6.95 UNCAGE w/o Fc,Fg 5.35 8.54 8.30 5.16 4.94 6.46

This demonstrates the effectiveness of UNCAGE itself, independent of the presence of the confidence or randomness terms. Furthermore, using only Fa (ablation 3) outperforms using only Fg (randomness term) and achieves performance comparable to using only Fc (confidence term). This suggests that UNCAGE not only serves as a guidance mechanism but is also as effective as the confidence term itself.

### Additional Qualitative Results

In this section, we provide additional qualitative results on the Attend-and-Excite (Chefer et al. 2023) dataset and the SSD (Weimin, Jieke, and Meng 2025) dataset.

##### Attend-and-Excite Dataset

Additional qualitative results on the Attend-and-Excite dataset are provided in Figures 7 and 8. The same seeds are applied to each prompt across all methods.

##### SSD Dataset

Additional qualitative results on the SSD dataset are provided in Figures 9 to 11. The same seeds are applied to each prompt across all methods.

”a bear and a frog” ”a cat and a lion” ”a turtle and a pink apple” ”a gray crown and a purple apple”

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

UNCAGE(ours)Meissonic(baseline)

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

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

[Figure 117]

[Figure 118]

[Figure 119]

”a dog and a rabbit” ”a rabbit and a mouse” ”a rabbit and a yellow car” ”a green bench and a red apple”

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

UNCAGE(ours)Meissonic(baseline)

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

- Figure 7: Additional qualitative results on Attend-and-Excite (Chefer et al. 2023) dataset (1). The same seeds are applied to

”a bear and a mouse” ”a dog and a lion” ”a monkey and a green bowl” ”a white car and a black bowl”

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

UNCAGE(ours)Meissonic(baseline)

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

”a horse and a monkey” ”a bear and a rabbit” ”a mouse and a red bench” ”a brown bowl and a green clock”

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

UNCAGE(ours)Meissonic(baseline)

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

- Figure 8: Additional qualitative results on Attend-and-Excite (Chefer et al. 2023) dataset (2). The same seeds are applied to

#### UNCAGE(ours)Meissonic(baseline)

“a beagle and a collie” “a eagle and a condor”

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

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

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

- Figure 9: Additional qualitative results on SSD (Weimin, Jieke, and Meng 2025) dataset (1). The same seeds are applied to each prompt across all methods.

#### UNCAGE(ours)Meissonic(baseline)

“a eagle and a owl” “a seal and a manatee”

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

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

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

- Figure 10: Additional qualitative results on SSD (Weimin, Jieke, and Meng 2025) dataset (2). The same seeds are applied to

#### UNCAGE(ours)Meissonic(baseline)

“a parrot and a pigeon” “a duck and a penguin”

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

- Figure 11: Additional qualitative results on SSD (Weimin, Jieke, and Meng 2025) dataset (3). The same seeds are applied to

