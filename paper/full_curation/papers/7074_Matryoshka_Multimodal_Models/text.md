## arXiv:2405.17430v2[cs.CV]29Jul2024

### Matryoshka Multimodal Models

Mu Cai1 Jianwei Yang2 Jianfeng Gao2 Yong Jae Lee1 1University of Wisconsin-Madison 2Microsoft Research, Redmond https://matryoshka-mm.github.io/

##### Abstract

Large Multimodal Models (LMMs) such as LLaVA have shown strong performance in visual-linguistic reasoning. These models first embed images into a fixed large number of visual tokens and then feed them into a Large Language Model (LLM). However, this design causes an excessive number of tokens for dense visual scenarios such as high-resolution images and videos, leading to great inefficiency. While token pruning and merging methods exist, they produce a single-length output for each image and cannot afford flexibility in trading off information density v.s. efficiency. Inspired by the concept of Matryoshka Dolls, we propose M3: Matryoshka Multimodal Models, which learns to represent visual content as nested sets of visual tokens that capture information across multiple coarse-to-fine granularities. Our approach offers several unique benefits for LMMs: (1) One can explicitly control the visual granularity per test instance during inference, e.g., adjusting the number of tokens used to represent an image based on the anticipated complexity or simplicity of the content; (2) M3 provides a framework for analyzing the granularity needed for existing datasets, where we find that COCO-style benchmarks only need around 9 visual tokens to obtain an accuracy similar to that of using all 576 tokens; (3) Our approach provides a foundation to explore the best trade-off between performance and visual token length at the sample level, where our investigation reveals that a large gap exists between the oracle upper bound and current fixed-scale representations.

##### 1 Introduction

Large Multimodal models (LMMs) [1, 2, 3, 4, 5, 6, 7] have shown strong performance in visuallinguistic understanding and reasoning. Models such as LLaVA [2, 5, 4] first embed the input image with a fixed number of visual tokens, and then feed them as prefix tokens to a Large Language Model (LLM) [8, 9] to reason about the input image. Similar model designs are borrowed in video LMMs [10, 11], where each frame contributes a fixed number of tokens to form the final video representation.

In reality, the number of visual tokens can be prohibitively large in the case of high-resolution images, and even more so for long videos. Existing works [10, 4, 12, 13] mainly tackle this issue by increasing the input context length and consequently, feeding a large number e.g., 3-8k of visual tokens into the LLM. This approach has a couple of significant drawbacks: (1) the extremely long context makes both training and inference inefficient; (2) an excessive number of visual tokens can actually harm the LMM’s performance, distracting it from attending to the relevant information, as we show in Sec. 4.3. Several recent works [14, 15, 16] use heuristics to prune and merge visual tokens to reduce the sequence length. However, they produce a single-length output and do not afford control over the final sequence length, which could be useful to trade information density versus efficiency while accounting for resource constraints in the deployment phase.

Preprint. Under review.

[Figure 1]

[Figure 2]

# M3

[Figure 3]

Describe this image for me.

[Figure 4]

In the heart of a bustling restaurant, a young girl finds solace at a table…

[Figure 5]

X

In the heart of a bustling restaurant, a young girl with vibrant hair is seated at a wooden table, her attention captivated by the camera…

X

[Figure 6]

…

[Figure 7]

In the heart of a bustling restaurant, a young girl with long, dark hair is the center of attention. She's dressed in a blue and white striped sweater,. … The table is adorned with a white paper bag, perhaps holding her meal. A blue Pepsi cup rests on the table …

X

- Figure 1: Matryoshka Multimodal Models. We enforce the coarser set of visual tokens XS

i−1

to be derived from the finer level of visual tokens XS

i

. As a result, the granularity of Matryoshka visual tokens gradually changes in a controllable manner. The image is from MSCOCO [17] validation set.

Images and videos naturally exhibit a hierarchical structure from coarse to fine details, and our human visual system has evolved to recognize visual information in this coarse to fine manner, as shown by biologists and psychologists decades ago [18, 19]. Can we create a similar structure for LMMs, where within one suite of model weights, the visual content tokens are organized into different scales of granularities? Conceptually, our goal is to learn the visual tokens to have a nested structure, similar to the Matryoshka Doll [20]. Matryoshka Representation Learning (MRL) [20] builds the Matryoshka mechanism over a neural network’s representation vector, where each of the segments with various feature dimensions is capable of handling tasks like classification or retrieval. However, for LMMs, the inefficiency mainly comes from the number of tokens. Thus, inspired by, but different from MRL, our work is motivated to build Matryoshka Multimodal Models upon the token length dimension, so that we can flexibly adjust it.

[Figure 8]

- Figure 2: MMBench evaluation results under M3, oracle under LLaVA-1.5-M3, LLaVA-1.5 with average pooling at inference time, LLaVA-1.5 separately trained for each specific scale, and other methods. M3 shows as least as good performance as LLaVA trained for each specific scale. A large gap exists between the oracle upperbound and model’s actual performance on a specific scale.

Specifically, we propose M3: Matryoshka Multimodal Models, which enforces an LMM to learn a hierarchy of visual representation granularities at the token sequence level, instead of the feature dimension level as in MRL [20]. With this representation, at inference time, the visual granularity can be flexibly controlled based on specific requirements, e.g., to account for the input image’s information density and efficiency constraints. Our training process is simple and straightforward. During training, we encode the image into M sets of visual tokens from coarse to fine, XS

,

i

i = 1,··· ,M, where the number of visual tokens gradually increases, i.e., |XS

i|. And importantly, the visual tokens in a coarser level are derived from the visual tokens in a finer level, i.e., XS

i−1| < |XS

, ∀i. In this way, the visual information in [XS

i−1 ⊂ XS

i

] gradually includes more fine-grained details. For example, given a natural image as shown in Figure 1, XS

,··· ,XS

###### ,XS

1

2

M

includes high-level semantics such as the restaurant and girl, while XS

1

includes more details such as the Pepsi cup and white paper bag. All other training settings, such as

M

the loss function and model architecture, are kept the same as LLaVA [2, 5, 4].

Our approach, M3, introduces several novel properties and benefits for LMMs. First, our approach can adaptively and efficiently represent visual content. Under one suite of weights, it generates

multiple nested sets of visual tokens with different granualarities in information density. This enables flexibility in the number of visual tokens used for any image during inference, enabling control over the best tradeoff between cost and performance based on the image or video content. For example, one can use all visual tokens for images with dense details and use just a few tokens for simpler images. This flexibility can be particularly significant when handling very long visual sequences, such as videos. For instance, given a fixed budget of 2880 visual tokens, a user could represent a video of 2880 frames each with one token or represent the same video by sampling 5 frames each with 576 tokens.

Second, our approach can be used as a general framework to evaluate the visual complexity of vision-language datasets or benchmarks, i.e., which level of granularity is needed in order to perform the given task correctly. Surprisingly, we find that most benchmarks, especially those mainly crafted from natural scenes (such as COCO) [21, 22, 23], can be handled well with only ∼ 9 tokens per image. In contrast, dense visual perception tasks such as document understanding or OCR [24, 25] require a greater amount of tokens (144−576 tokens) per image to handle the task well. The detailed findings are presented in Sec. 4.2.

Finally, our approach provides a foundation to tackle a critical task in LMMs: How to use the least amount of visual tokens while answering the visual questions correctly?. Based on the model’s predictions on the test set, we find that compared to full visual tokens, the oracle can use far fewer tokens while performing much better. For example, under six common LMM benchmarks used in LLaVA-NeXT [4], the oracle with the trained M3 model can use as few as 8.9 visual tokens on average to achieve performance that is 8% points better than LLaVA-NeXT which uses 576 tokens per image grid. This indicates that there is a large room for improvement compared to the oracle upperbound, as we show in Sec. 4.2.

To enable further research on adaptive LMMs that learn diverse information granularities, we publicly release our code and models.

##### 2 Related Work

Large Multimodal Models. Large Language Models (LLMs) like ChatGPT [26], GPT-4 [27], and LLaMA [28] have demonstrated impressive reasoning and generalization capabilities for text. The landscape of LLMs has been significantly transformed by the recent introduction of models that also incorporate visual information, such as GPT-4V(ision)[1]. Building upon open-source LLMs [28, 8], a plethora of multimodal models have made significant strides, spearheaded by models like LLaVA [2, 5] and MiniGPT-4 [3], which combine LLaMA’s [28] language capabilities with a CLIP [29] based image encoder. Recently, LMMs on more tasks and modalities have emerged, such as region level LMMs [30, 31, 32, 33, 34], 3D LMMs [35], and video LMMs [10, 11, 12]. However, existing LMMs typically represent the visual content with a large and fixed number of tokens, which makes it challenging to scale to very long visual sequences such as high-resolution images or long-form videos. In this work, we propose to adaptively and efficiently represent the visual content by learning multiple nested sets of visual tokens, providing flexibility in the number of visual tokens used for any image during inference.

Matryoshka Representation Learning. Matryoshka Representation Learning (MRL) [20] addresses the need for flexible representations that can adapt to multiple downstream tasks with varying computational resources. This approach, inspired by the nested nature of Matryoshka dolls, encodes information at different granularities within the same high-dimensional feature vector produced by a neural network. The adaptability of MRL extends across different modalities, including vision (ResNet [36], ViT [37]), vision + language (ALIGN [38]), and language (BERT [39]), demonstrating its versatility and efficiency. Recent work [40] extends MRL to both the text embedding space and the Transformer layers space. Our approach is inspired by MRL, but instead of learning multiple nested embeddings for a high-dimensional feature vector, we learn nested visual tokens along the token length dimension for the visual input. We are the first to show that the idea of Matryosha learning can enable explicit control over the visual granularity of the visual content that an LMM processes.

Token Reduction. One of the main causes of inefficiency in recent LMMs is their large number of prefix visual tokens that are fed into the LLM [2, 3]. The quadratic complexity in Transformers [41] is the key issue in scaling the input sequence length for Transformers. Token reduction serves as an effective technique to reduce computational costs in Transformers. Sparse attention methods such as Linformer [42] and ReFormer [43] conduct attention operations within local windows rather than the

[Figure 9]

[Figure 10]

#### Matryoshka Multimodal Models

[Figure 11]

Text Prompt : Describe the scene for me.

[Figure 12]

X

[Figure 13]

[Figure 14]

X

| | | |
|---|---|---|

CLIP Image Encoder

| | | | | | |
|---|---|---|---|---|---|

X

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|

| | | | | | | | |
|---|---|---|---|---|---|---|---|

|Large Language Model|
|---|

Granularity Controller

[Figure 15]

[Figure 16]

[Figure 17]

: There are a group of people standing in the ski facility, some of them are holding a green flag while other are …

- Figure 3: Architecture of our proposed Matryoshka Multimodal Models. The visual features from CLIP are represented as several groups of coarse-to-fine visual tokens. At test time, users can explicitly control the granularity of the visual features.

full context, thereby reducing the quadratic complexity of the vanilla attention operation. Another notable method is Token Merging (ToMe) [14], which utilizes full attention but gradually reduces the number of tokens in each transformer block by selecting the most representative tokens through bipartite matching for the Vision Transformer (ViT). A recent work [44] further studies different families of token reduction methods for ViT. However, prior approaches produce a single length output per input image and do not offer multiple granularities over the reduced token sequence. Our M3 approach instead learns a multi-granularity, coarse-to-fine token representation within the same model architecture and weights, enabling it to easily be adjusted to various computational or memory constraints.

A concurrent work [45] shares a similar spirit with our approach, representing an image with varying numbers of visual tokens using a single set of model weights. While their method reformats the visual tokens into a sequential list via transformation layers, we use average pooling to preserve the spatial structure of the visual tokens, demonstrating effectiveness in our experiments.

##### 3 M3: Matryoshka Multimodal Models

Our goal is to learn a Large Multimodal Model (LMM) that represents visual content as nested sets of visual tokens capturing information across multiple coarse-to-fine granularities, so that one can explicitly control the visual granularity per test instance during inference. Here we introduce how we learn a Matryoshka doll-like token sequence.

LMMs such as LLaVA [2] typically input a sequence of visual tokens as prefix tokens to the LLM for visual-linguistic reasoning. The visual encoder from pretrained vision-language models, such as CLIP [29] and SigLIP [46], is typically utilized to project the images into the set of visual tokens. In particular, the CLIP visual encoder represents an input image I as an H × W grid of visual tokens XH×W, where each Xi ∈ RC is a C dimensional feature vector. Our goal is to learn nested sets of visual tokens [XS

] which encode the visual information in a coarse-to-fine manner. To this end, we enforce XS

,··· ,XS

###### ,XS

1

2

M

,∀i. Importantly, we do not introduce any new learnable parameters to the LMM. We instead optimize the CLIP visual encoder to learn the nested visual representation directly, and train the ensuing LLM to adapt to the learned nested set of tokens.

i ⊂ XS

i+1

For ease of exposition, we consider CLIP-ViT-L-336 [29] as the visual encoder, where an image is encoded as 24×24 visual tokens (576 total). We create M sets of tokens e.g., |Si| ∈ {1,9,36,144,576}, in which the visual tokens at the coarser level are derived directly from those at the finer level. Specifically, given the initial 24 × 24 visual tokens, We sequentially apply 2 × 2 pooling with a stride 2, resulting in 12 × 12,6 × 6, and 3 × 3 visual tokens. Finally, we apply 3 × 3 pooling and get the most condensed single visual token. In this way, the sets of Matryoshka visual tokens can gradually preserve the spatial information in the original tokens while simultaneously forming a coarse-to-fine nested representation.

We train M3 by averaging the autoregressive next token prediction loss for each scale Si for each image Ii. Specifically, given a Matryoshka visual representation XS

for scale Si, we maximize the likelihood of the predicted tokens matching the ground-truth answer Xa:

i

P(Xa | XS

,Xq) =

i

L

,Xq,Xa,<j), (3.1)

Pθ(xj | XS

i

j=1

where θ is the trainable parameters of the model, which includes both the CLIP visual encoder and the ensuing LLM. Xq denotes the question in text format, L denotes the token length of the ground truth answer Xa, and Xa,<j denotes all the ground truth answer tokens before the current prediction token xj, where j denotes the token index during text token generation. We omit system messages for clarity, though they are part of the conditioning. Figure 3 shows our model architecture.

The final objective averages over all M visual token scales:

M

1 M

,Xq). (3.2)

−log P(Xa | XS

min

i

θ

i=1

With this objective function, M3 learns nested sets of visual tokens that gradually include more details with increasing scale. For example, in Figure 1, the smaller set of visual tokens describes the whole scene at a high level while the larger set of visual tokens includes more details such as the Pepsi cup. Our training objective affords our model to conduct visual question answering under any granularity during inference. This can be particularly useful in resource constrained applications; e.g., the visual granularity can be flexibly adjusted based on the anticipated simplicity or complexity of the visual content while taking into account compute and memory constraints.

##### 4 Experiments

In this section, we first detail the experiment settings in Sec 4.1. Then we show the performance of M3 on both image-level benchmarks 4.2 and video-level benchmarks 4.3. Finally, we analyze the behavior of Matryoshka Multimodal Models and provide ablations in Sec 4.4 and 4.5.

###### 4.1 Experiment Settings

Model We use LLaVA-1.5 [5] and LLaVA-NeXT [4] as the base LMMs, both with Vicuna 7B as the language model backbone. We finetune the whole model using the exact visual instruction data from LLaVA-1.5 and LLaVA-NeXT, respectively. The learning rate of LLM is 2 × 10−5 and

- 1 × 10−5, respectively for LLaVA-1.5 and LLaVA-NeXT. The learning rate for the visual encoder is
- 2 × 10−5 for both models. We train both models for 1 epoch using 8 NVIDIA H100 GPUs.

Instead of training the language model from scratch, we initialize the language model weights from pre-trained LLaVA-1.5 and LLaVA-NeXT, which we empirically works better. We name our Matryoshka Multimodal Models LLaVA-1.5-M3 and LLaVA-NeXT-M3.

Visual Token Scales We design 5 scales for the visual tokens. LLaVA-1.5 [5] and LLaVA-NeXT [4] both leverage CLIP-ViT-L-336 [29] as the visual encoder, where an image is embedded into 24 × 24 visual tokens. We gradually apply 2 × 2 pooling with stride 2, resulting in 12 × 12,6 × 6, and

- 3 × 3 visual tokens, where we finally apply a 3 × 3 pooling to get the final single visual token. Therefore, the size of Matryoshka visual token sets are S ∈ {1,9,36,144,576}, following a nested manner. The efficiency anlaysis on the system level is shown in Appendix B, where M3 boosts the speed of the LMM prefill process through diminished floating-point operations (FLOPs) and lessens computational memory requirements.

Evaluations. For image understanding, we evaluate LLaVA-1.5 and LLaVA-NeXT on (a) diverse multimodal benchmarks: POPE [22], GQA [47], MMBench [23], VizWiz [48], SEEDBench [49], ScienceQA [50], MMMU [51], and (b) document understanding/Optical character recognition (OCR) benchmarks: DocVQA [52], ChartQA [25], AI2D [53] and TextVQA [24].

For video understanding, we use both (a) open ended video question answering benchmarks evaluated by GPT-3.5: MSVD-QA [54], MSRVTT-QA [54] and ActivityNet-QA [55]; and (b) multichoice video question answering benchmarks: NExT-QA [56], IntentQA [57], and EgoSchema [58].

- Table 1: Comparison between LLaVA-1.5-M3 across various benchmarks under image understanding benchmarks. LLaVA-1.5-M3 maintains the performance of LLaVA-1.5 while outperforming QwenVL and InstructBLIP with fewer tokens.

Approach # Tokens MMBench GQA POPE VizWiz SEEDBench Qwen-VL [7] 256 38.2 59.3 - 35.2 56.3

Qwen-VL-Chat [7] 256 60.6 57.5 - 38.9 58.2 InstructBLIP-7B [59] 32 36.0 49.2 - 34.5 53.4

InstructBLIP-13B [59] 32 - 49.5 78.9 33.4 LLaVA-1.5-7B [5] 576 64.8 62.0 85.9 54.4 60.5

LLaVA-1.5-M3

576 65.9 61.9 87.4 54.9 60.6 144 66.4 61.3 87.0 53.1 59.7

36 64.8 60.3 85.5 52.8 58.0 9 63.1 58.0 83.4 51.9 55.4 1 59.5 52.6 78.4 49.4 50.1

- Table 2: Comparison of approaches with the SS baseline and M3 across various benchmarks under LLaVA-NeXT [4]. Here # Tokens denotes the number of visual tokens per image grid in LLaVANeXT. SS denotes the baseline model trained with a Specific Scale of visual tokens. M3 is at least as good as SS, while performing better on tasks such as TextVQA, ChartQA, and MMBench. Oracle denotes the case where the best tradeoff between visual tokens and performance is picked.

# Tokens Per Grid

Approach TextVQA AI2D ChartQA DocVQA MMBench POPE ScienceQA MMMU

SS 64.53 64.83 59.28 75.40 66.58 87.02 72.29 34.3

576

M3 63.13 66.71 58.96 72.61 67.96 87.20 72.46 34.0 144

SS 62.16 65.77 55.28 67.69 67.78 87.66 72.15 36.4

M3 62.61 68.07 57.04 66.48 69.50 87.67 72.32 36.1 36

SS 58.15 65.90 45.40 56.89 67.01 86.75 71.87 36.2

M3 58.71 67.36 50.24 55.94 68.56 87.29 72.11 36.8 9

SS 50.95 65.06 37.76 44.21 65.29 85.62 72.37 36.8

M3 51.97 66.77 42.00 43.52 67.35 86.17 71.85 35.2 1

SS 38.39 63.76 28.96 33.11 61.43 82.83 72.32 35.3

M3 38.92 64.57 31.04 31.63 62.97 83.38 71.19 34.8 Oracle

# Tokens 31.39 11.54 41.78 64.09 8.90 6.08 7.43 22.85 Performance 70.51 76.36 70.76 81.73 74.35 94.29 76.07 50.44

###### 4.2 Image Understanding

LLaVA-1.5-M3 We evaluate LLaVA-1.5-M3 on the common multimodal understanding and reasoning benchmarks. Results are shown in Table 1. LLaVA-1.5-M3 with full tokens maintains the performance of LLaVA-1.5 across diverse benchmarks. More importantly, our approach shows strong performance even with 1 or 9 tokens. Specifically, in MMBench, a comprehensive multimodal understanding benchmark, LLaVA-1.5-M3 with 9 tokens surpasses Qwen-VL-Chat with 256 tokens, and achieves similar performance as Qwen-VL-Chat with even 1 token. Compared with InstructBLIP [59], LLaVA-1.5M3 with 9 tokens surpasses InstructBLIP-7B and InstructBLIP-13B across all benchmarks. This demonstrates that our model has both flexibility and strong empirical performance under diverse number of visual tokens.

LLaVA-NeXT-M3 We use the proposed Matryoshka Multimodal Models to finetune LLaVA-NeXT, and compare LLaVA-NeXT-M3 with SS, which denotes the setting where the LLaVA-NeXT is trained under a Specific Scale of visual tokens also for 1 epoch. We also include the oracle upperbound performance. Specifically, ‘Oracle’ denotes the case where the best tradeoff between visual tokens and performance is picked for each test instance. Specifically, for each test instance, we select the the scale with the fewest amount of tokens but can answer the question correctly. Results are shown in Table 2. Our approach, M3, is at least as good as SS, while performing better on tasks such as document understanding (TextVQA and ChartQA) and common benchmarks such as MMBench [23].

Our results also show that dataset level biases towards the visual token scales do exist. For example, ScienceQA maintains consistent performance across all visual token scales. AI2D and MMBench only encounter a small performance drop for even as few as 9 to 1 tokens. On the other hand, dense

- Table 3: Overall accuracy of LLaVA-NeXT-M3 and recent video LMMs on various video understanding benchmarks. Here # Tokens denotes the overall number of visual tokens across all frames.

Approach # Tokens MSVD MSRVTT ActivityNet NextQA IntentQA EgoSchema

Video-LLaMA [61] - 51.6 29.6 12.4 - - LLaMA-Adapter [62] - 54.9 43.8 34.2 - - Video-ChatGPT [63] - 64.9 49.3 35.2 - - -

Video-LLaVA [64] 2048 70.7 59.2 45.3 - - -

InternVideo [65] - - - - 59.1 - 32.1 LLaVA-NeXT-7B [4] 2880 78.8 63.7 54.3 63.1 60.3 35.8

2880 78.2 64.5 53.9 63.1 58.8 36.8 720 79.0 64.5 55.0 62.6 59.6 37.2 180 77.9 63.7 55.0 61.4 59.3 37.6

LLaVA-NeXT-7B-M3

45 75.8 63.0 53.2 59.5 58.7 38.8 5 73.5 62.7 50.8 56.5 56.7 36.2

visual perception tasks such as TextVQA and DocVQA show a significant performance drop with fewer tokens. This analysis shows that M3 could serve as a framework to analyze the granularity that a benchmark needs.

Furthermore, there is a large gap between the model’s actual performance under full tokens and the upper-bound oracle. This indicates that using full tokens cannot always result in the optimal performance for all samples; i.e., there is a large room of improvement towards the oracle point.

###### 4.3 Video Understanding

Following IG-VLM [60], we directly conduct zero-shot inference on diverse video benchmarks using LLaVA-NeXT-M3. Specifically, 6 frames are uniformly sampled over the entire video, then arranged as a collage, which is fed into LLaVA-NeXT along with the question to get the response. Results under LLaVA-NeXT-M3 and recent video LMMs are show in Table 3.

LLaVA-NeXT-M3 with full visual tokens again shows comparable performance with LLaVA-NeXT. More interestingly, results indicate that full visual tokens usually do not lead to the best performance in video understanding tasks. Specifically, on 4 out of 6 benchmarks, full visual tokens show less desirable performance compared to 720 or 180 visual tokens. We suspect that very long visual context could bring distraction (e.g., too much focus on potentially irrelevant background) to the model’s prediction, where a compact representation of the video focusing on the more relevant information may be more advantageous.

Finally, for most video understanding tasks such as ActivityNet, IntentQA and EgoSchema, with 9 tokens per image grid (45 tokens in total), the accuracy difference compared to full tokens (2880 in total) is less than 1%. This demonstrates that the video questions in these benchmarks usually require very sparse visual information, as the source of such video understanding benchmarks mostly comes from natural scenes, which matches our observation in image understanding benchmarks.

###### 4.4 In-depth Analysis

M3 shows much stronger performance compared to heuristics based sampling at test time. A simple way to reduce the number of visual tokens via a training-free way is to conduct heuristic token merging or reduction. In Table 4, we compare M3 with three training-free approaches: average pooling, spatial sampling, and sequential sampling. M3 is much more resilient when the number of tokens decreases, while the heuristic based sampling approaches show dramatic performance drop. A visualization of the spatial and sequential sampling is shown in Figure 5.

M3 serves as a good metric for image complexity. We extract the response from LLaVA-NeXT-M3 in the TextVQA benchmark, and show the samples where using visual tokens across different scales can answer the question correctly and incorrectly. Shown in Figure 4, the OCR performance aligns with the complexity of the images, which indicates that M3 can be utilized as a metric towards sample level complexity.

Large gap between oracle and actual performance. As shown in Table 2, the oracle upper-bound can use very few (6 ∼ 64) tokens yet achieve at least 10% better performance compared to full visual

- Table 4: Comparison between M3, and heuristics based sampling baselines—average pooling, spatial sampling, and sequential sampling—at inference time on MMBench with the LLaVA-NeXT architecture.

# Tokens M3 Average Pooling Spatial Sampling Sequential Sampling

576 67.96 67.18 67.18 67.18 144 69.50 61.68 65.81 60.14

36 68.56 50.77 60.05 44.76 9 67.35 45.45 45.45 31.96 1 62.97 19.33 26.29 22.42

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Q: how much is a polos crazy bike?

Q: what directive is the sign giving?

Q: what number is on the black and white sign?

Q: what brand is the apricot brandy?

Q: what beer company is a sponsor on the score board?

Q: what is the telephone number of andrew yates?

# Tokens 577 144 36 9 1 Correct? ✔ ✔ ✔ ✔ ✔

# Tokens 577 144 36 9 1 Correct? ✔ ✔ ✔ ✘ ✘

# Tokens 577 144 36 9 1 Correct? ✘ ✘ ✘ ✘ ✘

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

- Figure 4: TextVQA test samples with correct and incorrect predictions upon different scales. Answers vary with different number of visual tokens. In addition, M3 can serve as a framework to evaluate the complexity of images.

tokens. This suggests that a visual token scale predictor, where the model learns to automatically select the best visual token scale given the input images or both input images and questions, has potential to achieve a better tradeoff. This would be interesting future work.

Zero-shot generalization to longer visual sequences. Here we extend the length of the visual tokens at inference time to study the model’s zero-shot generalization behavior. Results under LLaVANeXT are shown in Table 5. Here LLaVA-NeXT-M3 is trained on 2 × 2 image grids but evaluated on 3 × 3 grids. We set the number of visual tokens to be 144 in each image during evaluation. The model obtains a significant improvement in document understanding by 2.12, 1.80, and 4.11 on TextVQA, ChartQA, and DocVQA, respectively, while maintaining the same performance on benchmarks mainly composed of natural scene images. 3 × 3 image grids with 144 tokens per grid own 1440 tokens, yet achieve similar performance with the default LLaVA-NeXT 2 × 2 image grids with 2880 total tokens (576 tokens per grid). This indicates it is promising to feed more subimages while making the number of visual tokens within each subimage much smaller.

- 4.5 Ablation Studies

[Figure 32]

[Figure 33]

(a) Sequential (b) Spatial

- Figure 5: Visualization of sequential and spatial sampling. Given 24×24 girds, the visualized cells denote the sampled tokens.

We ablate the key designs in M3, including the sampling method of Matryoshka visual tokens, and training strategy.

Matryoshka visual token sampling. Here we compare three different ways to select the visual tokens for Matryoshka Multimodal Models, including average pooling, spatial sampling, and sequential sampling, which is illustrated in Figure 5. Shown in Table 6, averaging pooling shows better performance than the two alternatives across diverse benchmarks. In general, sequential sampling performs the worst. We hypothesize that this is due to the visual tokens having spatial information, while sequential sampling does not naturally align with the spatial distribution of the visual tokens.

Training the entire LMM vs only training CLIP. Since the nested behavior of Matryoshka visual tokens is learned within the CLIP visual encoder, we next evaluate whether it is necessary to also

- Table 5: Performance comparison of different image grid configurations with LLaVA-NeXT-M3. # Grids # Tokens per grid Overall # Tokens TextVQA AI2D ChartQA DocVQA MMBench POPE ScienceQA

- 2 × 2 144 720 62.61 68.07 57.04 66.48 69.50 87.67 72.32

- 3 × 3 144 1440 64.73 67.75 58.84 70.59 69.50 87.67 72.22 2 × 2 576 2880 63.13 66.71 58.96 72.61 67.96 87.20 72.46

- Table 6: Ablation on Matryoshka visual token sampling including average pooling, sequential sampling, and spatial sampling.

TextVQA MMBench AI2D Num of Vis Tokens Avg Pooling Sequential Spatial Avg Pooling Sequential Spatial Avg Pooling Sequential Spatial

576 63.13 59.37 60.45 67.96 64.60 64.43 66.71 65.61 64.96 144 62.61 55.80 58.33 69.50 64.18 64.52 68.07 64.90 64.96

36 58.71 52.79 52.39 68.56 63.92 64.69 67.36 64.51 64.02 9 51.97 44.05 44.19 67.35 63.14 62.11 66.77 63.70 63.92 1 38.92 28.03 29.91 62.97 59.36 57.47 64.57 63.21 63.08

finetune the LLM. Shown in Table 7, training the whole LLM achieves better performance. This demonstrates that by also training the LLM, the model can better adapt to the patterns of the visual tokens distributed in the Matryoshka manner.

As explained in Sec. 3 and 4.1, we (a) initialize the LLM weights from LLaVA and (b) minimize the loss averaged upon all visual token scales for each sample during training. An alternative choice is to randomly sample a visual token scale. Shown in Table 8, initializing the LLM weights from LLaVA and minimizing the losses over all scales shows consistent performance boost compared to using the vanilla text-only pre-trained LLM weights [8] and randomly selecting a visual token scale. Initializing the LLM weights from LLaVA makes the training process of M3 more stable. By learning all scales at once, the model is forced to learn the nested behavior for each sample, which leads to better performance.

##### 5 Conclusion and Future Work

We introduced M3: Matryoshka Multimodal Models, which learns to represent visual content as nested sets of visual tokens, capturing information across multiple coarse-to-fine granularities. LMMs equipped with M3 afford explicit control over the visual granularity per test instance during inference. We also showed that M3 can serve as an analysis framework to investigate the visual granularity needed for existing datasets, where we discovered that a large number of multimodal benchmarks only need as few as 9 visual tokens to obtain accuracy similar to that of using all visual tokens, especially for video understanding. Furthermore, we disclosed a large performance-efficiency gap between the oracle upper-bound and the model’s performance.

Our work can be naturally extended to other domains. For example, the long context in a text-only LLM or vision tokens in dense vision tasks can also be represented as nested sets of tokens in a Matryoshka manner. One limitation of our current approach is that we are lacking an effective visual token predictor that can bridge the gap between the oracle and LMM’s actual performance at a specific scale. We believe this would be an exciting next direction of research in this space.

##### Acknowledgement

This work was supported in part by NSF CAREER IIS2150012, and Institute of Information & communications Technology Planning & Evaluation(IITP) grants funded by the Korea government(MSIT) (No. 2022-0-00871, Development of AI Autonomy and Knowledge Enhancement for AI Agent Collaboration) and (No. RS2022-00187238, Development of Large Korean Language Model Technology for Efficient Pre-training), and Microsoft Accelerate Foundation Models Research Program.

##### References

[1] OpenAI. Gpt-4v(ision) system card. https://cdn.openai.com/papers/GPTV_System_

Card.pdf, 2023.

- Table 7: Performance comparison of training LLaVA-NeXT-M3 with and without training the LLM across diverse benchmarks. We see a clear drop when freezing the LLM.

Num of Vis Tokens TextVQA MMBench AI2D DocVQA w/ LLM w/o LLM w/ LLM w/o LLM w/ LLM w/o LLM w/ LLM w/o LLM

576 63.13 61.16 67.96 63.66 66.71 63.92 72.61 69.15 144 62.61 57.79 69.50 65.21 68.07 63.73 66.48 59.77

36 58.71 49.75 68.56 63.92 67.36 62.89 55.94 44.08 9 51.97 36.15 67.35 61.08 66.77 62.05 43.52 28.36 1 38.92 19.72 62.97 51.80 64.57 60.59 31.63 17.37

- Table 8: Impact of (a) initializing the LLM weights from LLaVA, and (b) averaging the loss from all scales vs randomly selecting a scale for each sample during training.

Technique TextVQA AI2D Init LLM weights from LLaVA ✓ ✓ ✓ ✓

Average losses over all scales ✓ ✓ ✓ ✓

576 60.36 62.25 61.01 63.13 62.40 65.06 65.84 66.71 144 59.61 61.02 59.80 62.61 63.67 65.61 65.77 68.07

36 54.86 55.91 55.32 58.71 63.67 65.32 66.68 67.36 9 46.84 47.04 48.80 51.97 63.02 64.83 65.38 66.77 1 33.78 33.68 36.05 38.92 61.53 63.21 63.37 64.57

- [2] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 2023.
- [3] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. ICLR, 2024.
- [4] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024.
- [5] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2024.
- [6] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. Cogvlm: Visual expert for pretrained language models, 2023.
- [7] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.
- [8] Vicuna. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. https: //vicuna.lmsys.org/, 2023.
- [9] Meta. Llama-3. https://ai.meta.com/blog/meta-llama-3/, 2024.
- [10] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023.
- [11] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023.
- [12] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model, April 2024.
- [13] Gemini Team. Gemini: A family of highly capable multimodal models, 2024.
- [14] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your ViT but faster. In International Conference on Learning Representations, 2023.

- [15] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. arXiv preprint arXiv:2403.06764, 2024.
- [16] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. arXiv preprint arXiv:2403.15388, 2024.
- [17] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014.
- [18] Mike G Harris and Christos D Giachritsis. Coarse-grained information dominates fine-grained information in judgments of time-to-contact from retinal flow. Vision research, 40(6):601–611, 2000.
- [19] Jay Hegd´e. Time course of visual perception: coarse-to-fine processing and beyond. Progress in neurobiology, 84(4):405–439, 2008.
- [20] Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, et al. Matryoshka representation learning. Advances in Neural Information Processing Systems, 35:30233–30249, 2022.
- [21] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017.
- [22] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023.
- [23] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023.
- [24] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019.
- [25] Ahmed Masry, Do Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2263–2279, Dublin, Ireland, May

2022. Association for Computational Linguistics.

- [26] OpenAI. Chatgpt. https://openai.com/blog/chatgpt/, 2023.
- [27] OpenAI. Gpt-4 technical report. 2023.
- [28] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [30] Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P. Meyer, Yuning Chai, Dennis Park, and Yong Jae Lee. Making large multimodal models understand arbitrary visual prompts. In IEEE Conference on Computer Vision and Pattern Recognition, 2024.
- [31] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. arXiv preprint arXiv:2307.03601, 2023.

- [32] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023.
- [33] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023.
- [34] Hao Zhang, Hongyang Li, Feng Li, Tianhe Ren, Xueyan Zou, Shilong Liu, Shijia Huang, Jianfeng Gao, Lei Zhang, Chunyuan Li, and Jianwei Yang. Llava-grounding: Grounded visual chat with large multimodal models, 2023.
- [35] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. NeurIPS, 2023.
- [36] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.
- [37] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021.
- [38] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 4904–4916. PMLR, 2021.
- [39] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.
- [40] Xianming Li, Zongxi Li, Jing Li, Haoran Xie, and Qing Li. 2d matryoshka sentence embeddings. arXiv preprint arXiv:2402.14776, 2024.
- [41] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, pages 5998–6008, 2017.
- [42] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity, 2020.
- [43] Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In International Conference on Learning Representations, 2020.
- [44] Joakim Bruslund Haurum, Sergio Escalera, Graham W. Taylor, and Thomas B. Moeslund. Which tokens to use? investigating token reduction in vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, October 2023.
- [45] Wenbo Hu, Zi-Yi Dou, Liunian Harold Li, Amita Kamath, Nanyun Peng, and Kai-Wei Chang. Matryoshka query transformer for large vision-language models. arXiv preprint arXiv:2405.19315, 2024.
- [46] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023.
- [47] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019.
- [48] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3608–3617, 2018.

- [49] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seedbench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023.
- [50] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 2022.
- [51] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of CVPR, 2024.
- [52] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021.
- [53] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Bastian Leibe, Jiri Matas, Nicu Sebe, and Max Welling, editors, Computer Vision – ECCV 2016, pages 235–251, Cham, 2016. Springer International Publishing.
- [54] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In Proceedings of the 25th ACM international conference on Multimedia, pages 1645–1653, 2017.
- [55] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In AAAI, volume 33, pages 9127–9134, 2019.
- [56] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of questionanswering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786, 2021.
- [57] Jiapeng Li, Ping Wei, Wenjuan Han, and Lifeng Fan. Intentqa: Context-aware video intent reasoning. In Int. Conf. Comput. Vis., pages 11963–11974, 2023.
- [58] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. In Adv. Neural Inform. Process. Syst., 2024.
- [59] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023.
- [60] Wonkyun Kim, Changin Choi, Wonseok Lee, and Wonjong Rhee. An image grid can be worth a video: Zero-shot video question answering using a vlm. arXiv preprint arXiv:2403.18406, 2024.
- [61] Hang Zhang, Xin Li, and Lidong Bing. Video-LLaMA: An instruction-tuned audio-visual language model for video understanding. In Conf. Empirical Methods in Natural Language Processing, pages 543–553, 2023.
- [62] Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199, 2023.
- [63] Muhammad Maaz, Hanoona Abdul Rasheed, Salman Khan, and Fahad Shahbaz Khan. Videochatgpt: Towards detailed video understanding via large vision and language models. ArXiv abs/2306.05424, 2023.

- [64] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. ArXiv abs/2311.10122, 2023.
- [65] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, Sen Xing, Guo Chen, Junting Pan, Jiashuo Yu, Yali Wang, Limin Wang, and Yu Qiao. Internvideo: General video foundation models via generative and discriminative learning. ArXiv abs/2212.03191, 2022.
- [66] Zhihang Yuan, Yuzhang Shang, Yang Zhou, Zhen Dong, Chenhao Xue, Bingzhe Wu, Zhikai Li, Qingyi Gu, Yong Jae Lee, Yan Yan, et al. Llm inference unveiled: Survey and roofline model insights. arXiv preprint arXiv:2402.16363, 2024.
- [67] Zhihang Yuan, Yuzhang Shang, Yue Song, Qiang Wu, Yan Yan, and Guangyu Sun. Asvd: Activation-aware singular value decomposition for compressing large language models. arXiv preprint arXiv:2312.05821, 2023.

##### A Broader Impact

The broader impact of M3, a framework with nested visual representations, has potential benefits and risks associated with its deployment and release. Our model is trained using the exact same architecture and data of LLaVA-1.5 [5] and LLaVA-NeXT [4]. All the concerns are same as LLaVA. Specifically, as one example, LLaVA conducts instruction tuning using GPT-4 and GPT-4V generated data. The bias from GPT-4 and GPT-4V would still exist in LLaVA.

##### B Efficiency Analysis

To illuminate the computational benefits conferred by M3, we employ the roofline-based LLMViewer analysis as detailed in [66]. Our analysis is set within a hypothetical context designed to emphasize the effects of M3 on processing efficiency in LMMs. We study the LLaVA-1.5 case where a 336 × 336 resolution image is processed using a CLIP-ViT image encoder, resulting in 576 visual tokens. Accompanied by a text prompt with an assumed number of 30 tokens, the nested visual tokens in M3 substantially lowers the visual token count. The consequences of this reduction are substantial as outlined in Table 9, detailing the computational costs involved in the LMM prefill process. Notably, M3 not only boosts the speed of the LMM prefill process through diminished floating-point operations (FLOPs) but also lessens computational memory requirements.

It is crucial to highlight that the advantages of M3 are not limited to just efficiency improvements. The token reduction approach of M3 can also enhance other LMM acceleration methods, such as quantization and factorization, as referenced in [67]. This complementary relationship accentuates the broad potential of M3 to contribute to a wider array of efficiency-boosting strategies.

Table 9: Computation Cost Analysis. The development device is Tesla V100 GPU, and time estimated by the roofline model represents the theoretical performance that the hardware can achieve.

# Tokens FLOPs (TB) Prefill Time (ms) Total Memory (GB) Storing Activation (GB)

576 8.0 58.1 21.6 3.8 144 2.2 19.5 15.0 0.7

36 0.9 18.0 13.8 0.3 9 0.5 17.7 13.6 0.2 1 0.4 17.6 13.5 0.1

##### C More Visualizations on Nested Visual Representation

Shown in Figure 6, with more visual tokens, LMMs can discover more details, such as furniture and human attributes. Besides, LMMs can generate higher quality descriptions with more visual tokens, as demonstrated by the OCR capability in Figure 6 (b).

[Figure 34]

[Figure 35]

# M3

[Figure 36]

Describe this image for me.

[Figure 37]

The image shows an interior space that appears to be a living room or a combined living and dining area ...

X

[Figure 38]

The image shows an interior space that appears to be a living room or a lobby. The room has a warm color scheme with beige walls and a darker brown floor. There is a large, L-shaped sofa…

X

[Figure 39]

…

[Figure 40]

The image shows an interior space that appears to be a living room or a combined living and dining area…There is a large, L-shaped sofa with a light-colored upholstery, positioned in the center of the room. In front of the sofa, there is a glass-top coffee table with various…

X

- (a)
- (b)

[Figure 41]

[Figure 42]

# M3

[Figure 43]

Describe this image for me.

[Figure 44]

This is a black and white photograph capturing a moment from a baseball game. In the foreground, there are three individuals…

X

[Figure 45]

This is a black and white photograph capturing a moment from a baseball game. In the foreground, three baseball players are standing on a field. The player on the left is wearing a baseball uniform with the name "KIMBLE" on the front, a cap, and a glove..

X

[Figure 46]

…

[Figure 47]

This is a black and white photograph capturing a moment from a baseball game. In the left section, we see a player from the Kimberly team. He is dressed in a white baseball uniform with the word "KIMBERLY" emblazoned across the chest. He is holding a baseball glove, ready for action.

X

- Figure 6: More visualization examples. With more visual tokens, LMMs can discover more details, and generate higher quality descriptions. The images are from MSCOCO [17] validation set.

