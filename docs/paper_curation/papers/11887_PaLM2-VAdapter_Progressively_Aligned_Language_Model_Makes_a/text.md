# arXiv:2402.10896v2[cs.CV]1Jun2024

## PaLM2-VAdapter: Progressively Aligned Language Model Makes a Strong Vision-language Adapter

Junfei Xiao1,2, Zheng Xu2, Alan Yuille1, Shen Yan2†, Boyu Wang2† 1Johns Hopkins University 2Google Research

### Abstract

This paper demonstrates that a progressively aligned language model can effectively bridge frozen vision encoders and large language models (LLMs). While the fundamental architecture and pre-training methods of vision encoders and LLMs have been extensively studied, the architecture and training strategy of visionlanguage adapters vary significantly across recent works. Our research undertakes a thorough exploration of the state-of-the-art perceiver resampler architecture and builds a strong baseline. However, we observe that the vision-language alignment with perceiver resampler exhibits slow convergence and limited scalability with a lack of direct supervision. To address this issue, we propose PaLM2-VAdapter, employing a progressively aligned language model as the vision-language adapter. Compared to the strong baseline with perceiver resampler, our method empirically shows faster convergence, higher performance and stronger scalability. Extensive experiments on various Visual Question Answering (VQA) and captioning tasks for both images and videos demonstrate that our model exhibits state-of-the-art visual understanding and multi-modal reasoning capabilities. Notably, our method achieves these advancements with 30∼70% fewer parameters than the state-of-theart large vision-language models, marking a significant efficiency improvement.

### 1 Introduction

With the notable successes of large language model (LLM) [6, 31, 2], coupled with advancements in vision-language pretraining [27, 14, 17, 42], researchers are now well-equipped to construct sophisticated Large Vision-Language Models (LVLMs). This is achieved by integrating robust unimodal models, namely vision encoders and LLMs, thereby circumventing the need to develop these models from scratch [1, 16, 21, 7]. These LVLMs have demonstrated exceptional performance across a variety of multi-modal benchmarks, showcasing their impressive capabilities in understanding, reasoning, and generalizing across different contexts [1, 16, 25].

Contrasting with traditional full-model finetuning approaches, recent research has shifted towards freezing both vision encoder and LLM during LVLM training [1, 16, 25]. There are two main reasons for this. Firstly, vision encoders and LLMs have learned very strong feature extraction ability and reasoning ability through the large-scale pretraining on high-quality data, and finetuning could lead to catastrophic forgetting. Secondly, as these base models are getting bigger, freezing them saves training costs. Therefore, the focus is on training an adapter that connects the vision encoder and the LLM for cross-modality alignment.

To build strong LVLMs using pre-trained and frozen vision encoders and LLMs, the keys lie in the design and training strategy of the adapter. Existing methods like Flamingo and AnyMAL [1, 25]

Preprint. Under review. †Equal Advising

[Figure 1]

[Figure 2]

|[Figure 3]|[Figure 4]|
|---|---|

Stronger Scalability

PaLM2-VAdapter

Higher Performance

| | |
|---|---|
| |Faster Convergence|

SOTA Adapter

- Figure 1: Faster, higher, and stronger. Our progressively aligned language model demonstrates faster convergence, higher performance and stronger scalability as an adapter for vision-language alignment.

employ the perceiver resampler as their adapter architecture, resulting an effective way for crossmodality alignment. On the other hand, BLIP-2 [16] tackles the adapter pre-training issue by introducing Q-Former, which takes an additional pretraining stage with multi-task learning on imagetext pairs. Although these methods demonstrate impressive performance, questions regarding the optimal adapter architecture and the necessity of adapter pretraining still remain open for exploration.

To address the open questions in the design and training of adapters for LVLMs, we conduct an in-depth study into the latest cross-attention based adapter architectures, particularly focusing on the perceiver resampler and make a strong baseline. However, we observed that the perceiver resampler adapter exhibits slow convergence and limited scalability, especially when scaling up the vision encoder. To overcome these challenges, we propose PaLM2-VAdapter, which employs a progressive alignment strategy for bridging frozen vision encoders and LLM decoders. Specifically, the classic alignment framework is used in a progressive way with two stages and a tiny PaLM-2 model is trained as different roles (stage 1: LM decoder, stage 2: adapter). Compared to the baseline models using state-of-the-art adapters, PaLM2-VAdapter demonstrates faster convergence, higher performance and stronger scalability, as detailed in Figure 1.

We evaluate our models on various vision-language benchmarks in both image-based and video-based captioning and QA tasks. Our models consistently show state-of-the-art or comparable performance, while only requiring 30∼80% fewer parameters than previous models. This efficiency underscores the effectiveness of our proposed PaLM2-VAdapter in advancing the field of LVLMs.

To sum up, our contributions lie in three folds:

- 1. We conduct a comprehensive study of the state-of-the-art adapter architecture (i.e., perceiver resampler) and build a strong baseline with it.
- 2. We propose PaLM2-VAdapter, a progressive alignment strategy to train a tiny PaLM2 language model as the vision-language adapter, making solid improvement on convergence, performance and scalability.
- 3. Our models achieve state-of-the-art performance on various visual captioning and QA benchmarks but use 30∼70% less parameters than other models.

### 2 Related Work

#### 2.1 Vision-language Pre-training

Vision-language pre-training aims to learn universal multimodal representations through a set of pretraining objectives, including image-text matching [18, 4, 10], image-text contrastive learning [27, 14, 40, 11], and also auto-regressive image captioning [17, 42, 32, 34]. However, models pretrained on image-text pairs often lack the complex reasoning and few-shot learning abilities of Large Language Models (LLMs), primarily due to their focus on image captions [20, 27, 14, 28, 30]. To overcome

this, recent efforts have shifted towards integrating pretrained vision encoders and LLMs into larger vision-language models. This strategy extends their capabilities to more advanced tasks such as image captioning and Visual Question Answering (VQA), leveraging LLMs for improved performance.

#### 2.2 Large Language Models (LLMs)

Arming with scaled-up data and models, Large Language Models (LLMs) have demonstrated emergent capabilities like zero-shot generalization and in-context learning ability. This has sparked a surge in research and development, leading to significant advancements in models like FlanT5 [9], PaLM

- 2 [2], GPT-4 [26], LLaMA [31] and etc. Given the complex reasoning and remarkable understanding ability, LLMs are utilized as a "head". In this paper, we aims to bridge strong vision encoders with the PaLM 2 series of LLMs, extending its capability to understand and do reasoning with visual embeddings. To avoid the PaLM 2 model losing any knowledge or its strong language reasoning ability, our method keeps the large PaLM 2 model frozen all the time.

2.3 Large Vision-language Models (LVLMs)

Large Vision-language Models (LVLMs) connect vision and language together and extend the reasoning ability of LLMs to process with multi modal input. Numerous works have been proposed in this direction, including Flamingo [1], OpenFlamingo [3], BLIP-2 [16], InstructBLIP [22], MiniGPT4 [44], LLaVA [21]. Flamingo is the first work in this line, which uses the perceiver resampler as an adapter to feed visual tokens into language models. However, the number of trainable parameters in Flamingo is still more than billions, making the alignment with limited efficiency. BLIP-2 proposes a lightweight Q-Former as the adapter. However, the Q-Former needs a complex training process, including a two-stage training with three training objectives (vision-lanauge contrastive loss, matching loss and generation loss). InstructBLIP and MiniGPT-4 are extensions of BLIP-2 by using instruction tuning data or additional projection layer. LLaVA uses a simple projection layer to convert vision representations into the same dimension as the language. In this paper, we propose a progressive alignment strategy to use a pre-trained language model as the adapter, which shows faster convergence, higher performance and stronger scalability than the state-of-the-art perceiver resampler.

- 3 Method

Our study is based on a classic visual-language alignment pipeline which keeps the visual encoder and large language model (LLM) frozen all the time. An adapter is inserted between the vision encoder and LLM to project the encoded visual embeddings to the language representation space. This section firstly provides a preliminary overview of vision-language adapter architectures (§3.1) and then explains the model framework of visual-language alignment with adapter (§3.2). Lastly, we present our method using progressive vision-language alignment strategy for training a tiny language model as adapter (§3.3).

#### 3.1 Preliminary

Existing large vision-language models adopt various kinds of adapter architectures for cross-modality alignment. In this paper, we present an in-depth exploration of the state-of-the-art cross-attention based adapters and propose to progressively aligned self-attention based language model.

Cross-attention based adapter. The adapters in this style adopt the cross-attention mechanism for visual feature alignment. Specifically, the visual features extracted by the vision encoder are served as the keys and values which are cross-attentioned to a set of learnable queries, shown in Figure 2a. We conduct a comprehensive study of the state-of-the-art perceiver resampler architecture and establish a very strong baseline model using 6-layer perceiver resampler as the adapter (detailed in §4.2).

Self-attention based adapter. Self-attention layers can also be introduced in adapters to improve representation quality. Notably, self-attention based adapters could use pretrained language models for initialization to get better convergence and improve the performance.

#### 3.2 Visual-language Alignment with Adapter

As shown in Figure 2a, the vision-language model has three major parts: vision encoder, visual adapter and LLM. The target is to align the visual features with the LLM representation space. The visual encoder and the LLM are both frozen all the time. This setup greatly reduces training cost

Image & Video Captions

- ( i )
- ( ii )

Image & video Captions

[Figure 5]

Tiny PaLM-2

[Figure 6]

LLM Decoder (PaLM-2)

[Figure 7]

###### Vision Encoder

Cross-Attention

…

Describe the following: <visual tokens>

Image & video Captions

[Figure 8]

[Figure 9]

[Figure 10]

Adapter

Vision Encoder

Large PaLM-2

[Figure 11]

…

…

[Figure 12]

[Figure 13]

[Figure 14]

Tiny PaLM-2

[Figure 15]

Images (duplicated)

Learnable Queries

Perceiver Resampler

… Vision Encoder

[Figure 16]

[Figure 17]

[Figure 18]

…

Video frames

(a) Classic visual-language alignment

(b) Progressive alignment with PaLM2-VAdapter

- Figure 2: Method overview. (a): The classic model framework for visual-language alignment, consisting of three major parts: a vision encoder, an adapter and a LLM decoder. (b): Our progressive alignment strategy of our PaLM2-VAdapter. (i) A tiny PaLM2 language model (∼108M) is trained as the LM decoder in the first stage and (ii) then trained as the vision-language adapter (with an addition 1-layer perceiver resampler) for aligning the same vision encoder and a large PaLM2 decoder.

and preserves their strong visual feature extraction and reasoning ability learned from large-scale pre-training. The CoCa [42] vision encoder is pre-trained with image-text pairs and is used to convert images and video frames into a set of feature tokens. These feature tokens are projected by a lightweight visual adapter to the LLM representation space. We adopt PaLM 2 [2] series models as the LLM decoder and the training task is to generate captions based on the visual embedded prefix.

- 3.3 Progressive Visual-language Alignment

As language models emerge strong representation ability through the generative pre-training task and usually shows great scalability, we propose to introduce a tiny PaLM2 language model, using a progressive vision-language alignment strategy to make strong vision-language adapters. Specifically, our method uses a tiny PaLM2 language model (TLM) as the adapter and trains it in a progressive way, which consists of two stages:

- Stage 1 - TLM trained as the decoder: In the first stage, the language model starts from a pretrained tiny PaLM2 model (∼108M) and is finetuned with the classic vision-language alignment task ( shown in Figure 2b(i)).
- Stage 2 - TLM trained as the adapter: In the second stage, given this pre-aligned tiny PaLM2 model, an additional 1-layer perceiver resampler is added before the aligned tiny PaLM2 model to bridge the same vision encoder and a larger PaLM2 model (shown in Figure 2b(ii)).

Compared to our strongest model with state-of-the-art adapter (i.e., perceiver resampler), our method is proven to have faster convergence, higher performance and stronger scalability (detailed in §4.3). In addition to the effective architecture, the proposed progressive alignment strategy greatly advance PaLM2-VAdapter, making remarkable improvements for vision-language alignment (detailed in §4.4). Notably, the additional perceiver resampler is very crucial for efficient cross-modality fusion based on our empirical observation (detailed in §4.5).

- 4 Experiments

- 4.1 Implementation Details

Model. We adopt CoCa [42] pretrained ViTs as our vision encoders. The input resolution is 288 and the patch size is 18x18. We adopt PaLM 2 [2] pretrained models as the LLM. Perceiver resampler [1] is used as the baseline adapter architecture, with 256 learnable queries. Our proposed adapter consists of a 1-layer perceiver resampler and a tiny transformer-based language model (∼110M). Two fully-connected layers are applied before and after adapter for dimension matching.

Data. Our models are trained on image-text paired data of WebLI [7] dataset and video-text paired data of VTP [1] and SMIT [24] datasets. The ablations in Table 1 are solely trained on WebLI.

|Query & Key Final<br><br>|COCO Cap. VQAv2 (Val)|
|---|---|
|✗ ✓<br><br>Shared ✗ Separate ✗ Separate ✓<br><br>|38.4 32.2 44.0 46.7 46.8 52.5 36.2 37.6|

(a) LayerNorm options.

|FFN Time embedding<br><br>|COCO Cap. VQAv2 (Val)|
|---|---|
|✓ ✗ ✗ ✓ ✓ ✓<br><br>|34 38.3 33.8 45.1 46.8 52.5|

(b) Feed-forward network & time embedding.

|Query Dim<br><br>|COCO Cap. VQAv2|
|---|---|
|384 768 1536|40.9 45.4 46.8 52.5 38.3 45.0<br><br>|

|Hidden Dim<br><br>|COCO Cap. VQAv2|
|---|---|
|384 768 1536<br><br>|40.6 46.7 46.8 52.5 38.5 32.1|

|#Layers|COCO Cap. VQAv2|
|---|---|
|1 3 6<br><br>|37.7 37.5 40.8 47.6 46.8 52.5|

(c) Query dimension.

(d) Hidden dimension.

(e) Number of layers.

Table 1: In-depth analysis with key components of perceiver resampler. Results on COCO captioning benchmark (CIDEr score) and VQAv2 validation set (accuracy) are reported. Models are trained on WebLI (image-text paired dataset).

Training. The images and videos are duplicated or sampled to 8 frames [37] as the visual inputs. The base learning rate is 5e-4 and is scheduled with a warm-up and linear decay. The training batch size is 2048. By default, experiments are trained with 250K steps. We use a prompt template of "Describe the following: <visual tokens>" for training. For detailed information, please refer to Appendix A.

Evaluation. All the input resolution is the same as training (i.e., 288) with a patch size of 18. We evaluate our method on captioning tasks and Visual Question Answering (VQA) tasks for both images and videos. Specifically, COCO [8], VQAv2 [12], TextVQA [29], VizWiz [5], OKVQA [23] are used for image-based evaluation. MSRVTT [36], VATEX [33], MSVD-QA [35], and iVQA [38] are used for video-based evaluation. We use different prompts for the LLM decoder on different tasks. For detailed prompts information, please refer to Appendix A&B.

#### 4.2 A Strong Baseline with Perceiver Resampler

To figure out the effectiveness of different model components of cross-attention based adapters , we conduct a comprehensive ablation study based on perceiver resampler, which is the state-of-the-art adapter architecture. As shown in Table 1, our study covers different choices to apply LayerNorm, important modules (i.e., Feed-Forward Network FFN and time embedding), dimension of queries and cross-attention layers and also the number of perceiver resampler layers.

Based on the empirical results, we get several design rules for perceiver resampler based adapter: 1) LayerNorms are important and should be separately applied to the queries and the cross-modality inputs (as keys and values). 2) Feed-Forward Network (FFN) and time embedding make the adapter training stable and effective and can greatly improve the performance. 3) The dimension of the learnable queries and the cross-attention layer should be set moderate. Following this rules, we build a very strong baseline achieving 81.4 CIDEr on COCO captioning, 38.2 CIDEr on MSRVTT captioning and 53.1 accuracy on VQAv2.

#### 4.3 Faster, Higher, and Stronger

Although the baseline shows reasonable performance, we observe that it has limited scalability and slow convergence (shown in Figure 1). To address these issues, we propose to introduce a tiny language model as an adapter and train it progressively (shown in Figure 2b). Compared to the strong baseline based on state-of-the-art architecture (shown in Table 2), our PaLM2-VAdapter shows:

|Method Vision Enc.<br><br>|Converg. COCO MSRVTT Steps (K) CIDEr CIDEr|
|---|---|
|Perceiver Res. ViT-B PaLM2-VAdapter ViT-B|250 81.4 38.2 60 (-76%) 83.0 (+1.6) 42.1 (+3.9)<br><br>|
|Perceiver Res. ViT-L PaLM2-VAdapter ViT-L<br><br>|250 82.4 38.2 60 (-76%) 89.6 (+7.2) 42.7 (+4.5)|

Table 2: Faster, higher and stronger. Compared to the perceiver resampler baseline, PaLM2-VAdapter shows faster convergence, higher performance and stronger scalability. PaLM2-1B is used as the LLM decoder.

Faster convergence. While the perceiver resampler baselines take 250K steps to converge, our PaLM2-VAdapter only need 60K steps to converge which is ∼3×faster.

Higher performance. PaLM2-VAdapter achieves much higher performance than the baseline perceiver resampler models (ViT-B: 83.0 vs. 81.4, ViT-L: 89.6 vs. 82.4) when aligning the same vision encoder and LLM decoder pairs.

Stronger scalability. Perceiver resampler shows marginal improvement when the vision encoder is scaled from ViT-B to ViT-L. However, our PaLM2-VAdapter makes much larger improvement (COCO: 6.6 vs 1.0, MSRVTT: 0.6 vs 0.0) , showing stronger scalability.

#### 4.4 Progressive Training Does Help

We conduct a comparison regarding different pre-training strategies using the same adapter architecture (1-layer perceiver resampler + PaLM2-108M), detailed in Table 3. The ablation compares three training strategies for the adapter: a) randomly initialized; b) Generative pre-trained on language data (PaLM2 pretraining) , initialized from a PaLM2 checkpoint; c) Pretrained with the proposed progressive training strategy. The tiny PaLM2 model is first initialized from the PaLM2 checkpoint and then fine-tuned with vision-language generative pre-training (stage 1, the tiny PaLM2 model is trained as the LM decoder). The results prove the effectiveness of the progressive training strategy applied to the adapter including language-only generative pre-training [2] and vision-language generative pre-training (stage 1, shown in Figure 2b(i)).

|Language Only Vision-language (LM pretraining) (stage-1)<br><br>|COCO VQAv2 CIDEr Accuracy|
|---|---|
|✗ ✗ ✓ ✗ ✓ ✓|79.2 50.8 81.3 52.1 83.0 53.8<br><br>|

Table 3: Comparison of different adapter pretraining settings. Both language-only generative pre-training (PaLM2) and vision-language generative pre-training (stage-1, language model as decoder) can improve the final aligned large vision-language model’s performance.

|Cross-attention<br><br># Layers Module Type|COCO VQAv2<br><br>CIDEr Accuracy|
|---|---|
|Attentional Pooler 1 Perceiver Resampler 1 Perceiver Resampler 6|81.1 53.5 85.6 55.1 70.3 49.7<br><br>|

Table 4: Comparison of using different types of cross-attention modules. A lightweight perceiver resampler cross-attention module is the best cross-modality fusion choice for PaLM2VAdapter.

#### 4.5 Perceiver Resampler is Still Needed

In our first vision-language alignment stage (shown in Figure 2b(i)), we follow CoCa [42] to use an attentional pooler as the cross-attention module. This attentional pooler consists of a simple cross-attention layer and a LayerNorm layer for the final queried features. Based on our observation of our in-depth empirical study with the perceiver resampler architecture (detailed in Section 4.2), we replace the attentional pooler with a 1-layer perceiver resampler to improve cross-modal alignment and achieve better performance, shown in Table 4. On the other hand, we observe that adding more layers of perceiver resampler does not lead to better performance with our adapter design which is contrary to the observation with vanilla perceiver resampler adaper. The empirical results show that a 1-layer perceiver resampler seems to be the best choice for cross-modality fusion in our proposed PaLM2-VAdapter.

#### 4.6 Visual Captioning

Image captioning. As detailed in Table 5, we evaluate the zero-shot image captioning performance on the COCO dataset [8]. Compared to the state-of-the-art AnyMAL model, our method shows comparable image captioning capability, but only requires 70% parameters (10.8B vs. 15B), proving the effectiveness of our progressive alignment strategy. Additionally, the scalability of our PaLM2-VAdapter is evidenced through the vision encoder scaling experiment (from ViT-B to ViT-g), indicating that a more powerful vision encoder correlates with enhanced image captioning performance. Qualitative examples are provided in Figure 3 and Appendix C.

Video captioning. As detailed in Table 6, we evaluate the zero-shot video captioning performance on the MSRVTT and VATEX datasets [36, 33]. Compared to the state-of-the-art Flamingo models,

Method # Total # Trainable COCO Params Params CIDEr

CM3Leon[43] 7B 7B 61.6 Flamingo-3B[1] 3.2B 1.2B 73.0 Flamingo-9B[1] 9.3B 1.6B 79.4 Flamingo-80B[1] 80B 10.2B 84.3 IDEFICS-9B[15] 9B 1.5B 46.0 IDEFICS-80B[15] 80B 14B 91.8 AnyMAL-15B[25] 15B 100M∗ 99.5 PaLM2-VAdapter 1B (ViT-B) 1.8B 120M 83.0 PaLM2-VAdapter 1B (ViT-L) 2.0B 120M 89.6 PaLM2-VAdapter 1B (ViT-g) 2.8B 130M 97.5 PaLM2-VAdapter 8B (ViT-g) 10.8B 130M 95.2

Table 5: Zero-shot Image Captioning. The best result is bolded and the second-best result is underlined. Our model demonstrates comparable zero-shot visual understanding ability. *: Estimated by given information.

|Method # Total # Trainable<br><br>Params Params<br><br>|MSRVTTVATEX<br><br>CIDEr CIDEr|
|---|---|
|VideoCoCa[37] 2.1B 2.1B DeCap[19] 140M 50M Flam.-3B[1] 3.2B 1.2B Flam.-9B[1] 9.3B 1.6B Flam.-80B[1] 80B 14B<br><br>|27.1 22.8 34.8 18.7<br><br>- 40.1<br>- 39.5<br>- 46.7<br>|
|PaLM2-VA.1B(ViT-B) 1.8B 120M PaLM2-VA. 1B(ViT-L) 2.0B 120M PaLM2-VA. 1B(ViT-g) 2.8B 130M PaLM2-VA. 8B(ViT-g)10.8B 130M|42.1 38.3 42.7 45.5 45.6 51.2 47.7 53.0<br><br>|

Table 6: Zero-shot Video Captioning. The best result is bolded and the second-best result is underlined. Our model demonstrates the stateof-the-art zero-shot visual understanding ability on videos.

|[Figure 19]|[Figure 20]|
|---|---|

|[Figure 21]|[Figure 22]|[Figure 23]|[Figure 24]|
|---|---|---|---|
|[Figure 25]|[Figure 26]|[Figure 27]|[Figure 28]|

###### Caption:

###### Caption:

###### Caption:

###### Caption:

A motorcycle is parked in front of a house

A cow is licking a metal pipe

A video of a Minecraft game where a person is building a car.

Cartoon characters are hanging from a rope

|[Figure 29]|[Figure 30]|
|---|---|

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

###### Caption:

###### Caption:

###### Caption:

###### Caption:

A dog is sleeping on a pillow with a teddy bear

A person is drawing a picture of a cartoon character

A band is playing on stage in front of a large crowd

A surfer is riding a wave in the ocean

- Figure 3: Qualitative examples of Visual Captioning. Left: Image captioning on the COCO dataset. Right: Video captioning on the MSRVTT dataset. PaLM2-VAdapter demonstrates strong visual understanding ability.

our method makes solid improvement on the VATEX benchmark but only requires 14% parameters (10.8B vs. 80B). Similar to image captioning, PaLM2-VAdapter still shows strong scalability when the vision encoder is scaled up. Moreover, scaling up language model also improves video captioning performance, indicating that a larger language model lead to stronger ability to understand sequential visual information of videos. Qualitative examples are provided in Figure 3 and Appendix C.

#### 4.7 Visual Question Answering

Image question answering. As detailed in Table 7, we evaluate the zero-shot image question answering performance on the VQAv2, TextVQA, VizWiz, and OKVQA datasets [12, 29, 5, 23]. Compared to the state-of-the-art IDEFICS models, our method shows comparable image question answering ability but only requires 14% parameters (10.8B vs. 80B), proving the effectiveness of our progressive alignment strategy. PaLM2-VAdapter shows very strong scalability - always achieving

|Method<br><br># Total # Trainable Params Params|VQAv2 TextVQA VizWiz OKVQA Accuracy Accuracy Accuracy Accuracy<br><br>|
|---|---|
|Flamingo-3B [1] 3.2B 1.2B Flamingo-9B [1] 9.3B 1.6B Flamingo-80B [1] 80B 10.2B BLIP-2 (FlanT5xxL) [16] 12.1B 108M InstructBLIP (V-13B) [22] 14.1B 108M<br><br>IDEFICS-9B [15] 9B 1.5B IDEFICS-80B [15] 80B 14B<br><br>AnyMAL 13B (ViT-G) [25] 15B 100M<br><br>|49.2 30.1 28.9 41.2 51.8 31.8 28.8 44.7 56.3 35.0 31.6 50.6<br><br>65.0† 44.1∗ 29.4 45.9 - 50.7†∗ 33.4 -<br><br>50.9 25.9 35.5 38.4 60.0 30.9 36.0 45.2 59.6 24.7 24.4 33.1<br><br><br>|
|PaLM2-VAdapter 1B (ViT-B) 1.8B 120M PaLM2-VAdapter 1B (ViT-L) 2.0B 120M PaLM2-VAdapter 1B (ViT-g) 2.8B 130M PaLM2-VAdapter 8B (ViT-g) 10.8B 130M|53.8 18.7 28.6 31.0 55.0 22.2 37.2 31.7 57.9 23.7 44.1 33.6 60.6 24.8 43.7 40.9<br><br>|

- Table 7: Zero-shot Image Question Answering. The best result is bolded and the second-best result is underlined. Our model demonstrates strong zero-shot vision-language reasoning ability on the four classic benchmarks, comparable to the state-of-the-art methods. *: with additional OCR inputs. † : in-domain images were used.

|Method<br><br># Total # Trainable Params Params|MSRVTT-QA MSVD-QA iVQA<br><br>(Top-1 Acc.) (Top-1 Acc.) (iVQA Acc.)|
|---|---|
|Just Ask [38] 600M 600M HiTeA [41] 297M 297M<br><br>FrozenBiLM [39] 890M 30M Flamingo-3B [1] 3.2B 1.2B Flamingo-9B [1] 9.3B 1.6B<br><br>Flamingo-80B [1] 80B 14B|5.6 13.5 13.3<br><br>8.6 18.2 -<br><br>16.9 33.7 26.2<br><br>11.0 27.5 32.7 13.7 30.2 35.2<br><br>17.4 35.6 40.7<br><br><br>|
|PaLM2-VAdapter 1B (ViT-B) 1.8B 120M PaLM2-VAdapter 1B (ViT-L) 2.0B 120M PaLM2-VAdapter 1B (ViT-g) 2.8B 130M PaLM2-VAdapter 8B (ViT-g) 10.8B 130M|12.7 26.2 25.8<br><br>14.0 18.6 28.3<br><br>15.9 27.7 26.1<br><br><br>19.6 40.5 36.7<br><br>|

- Table 8: Zero-shot Video Question Answering. The best result is bolded and second-best result is underlined. Our model demonstrates state-of-the-art zero-shot multi-modal reasoning ability on videos.

better performance when the vision encoder and LLM decoder are scaled up. Qualitative examples are provided in Figure 4 and Appendix C.

Video question answering. As detailed in Table 8, we evaluate the zero-shot video question answering performance on the MSRVTT-QA, MSVD-QA and iVQA datasets [36, 35, 38]. Compared to the state-of-the-art Flamingo models, our method shows state-of-the-art video question answering ability but only requires 14% parameters (10.8B vs. 80B), proving the remarkable effectiveness of our method. The results also justify the strong scalability of PaLM2-VAdapter. Qualitative examples are provided in Figure 4 and Appendix C.

### 5 Limitation & Discussion

Our PaLM2-VAdapter makes a significant improvement in efficiency, operating with substantially fewer parameters and much less training cost. However, its alignment process encounters challenges as the LLM decoder scales, just like other large vision-language models. The key of this challenge lies in ensuring visual embeddings seamlessly transition into the scaled-up LLMs’ input representation space. A potential solution involves the direct quantization of visual embeddings into language tokens, leveraging the shared LLM codebook across models of varying sizes for zero-shot transferability. So, here comes the question: Can the visual embeddings be “translated" to words?

|[Figure 39]|[Figure 40]|
|---|---|

|[Figure 41]|[Figure 42]|[Figure 43]|[Figure 44]|
|---|---|---|---|
|[Figure 45]|[Figure 46]|[Figure 47]|[Figure 48]|

Q: What did someone cut a

Q: Are there ducks on the shoreline? A: Yes

Q: How many zebras are standing up? A: 3

Q: What is taking its clothes off? A: Monkey

rack of? A: Ribs

|[Figure 49]|[Figure 50]|[Figure 51]|[Figure 52]|
|---|---|---|---|
|[Figure 53]|[Figure 54]|[Figure 55]|[Figure 56]|

|[Figure 57]|[Figure 58]|
|---|---|

Q: Is this a vegetarian pizza? A: No

Q: Is the cat watching TV? A: Yes

Q: Who makes a goal? A: Soccer player

Q: What is the man doing? A: Riding a motorcycle

- Figure 4: Qualitative examples of Visual Question Answering. Left: Image question answering on the VQAv2 dataset. Right: video question answering on the MSVD-QA dataset.

|Setting Softmax Temp. Temp. Decay<br><br>|COCO CIDEr|
|---|---|
|Baseline - -|44.1<br><br>|
|Gumbel-Softmax 1.0 -<br>Gumbel-Softmax 2.0 Gumbel-Softmax 2.0 Exponential∗<br>|0 13.1 15.3<br><br>|

To answer this question, we conduct a study to see if the visual embeddings output by the adapter can easily be “translated" to a sequence of words and then used as the prefix for the LLM decoder. We introduce a fully-connected layer (FC layer) after the adapter and use the gumel-softmax operation [13] to quantize the visual embeddings.

Table 9: Quantize the visual embeddings to words. The baseline is aligned with image-text pairs. ∗: the gumbel-softmax temperature is exponential decayed.

The output logits from the FC layer correspond to the words of the LLM codebook and the word with highest logit will be assigned to the corresponding visual token. As shown in Table 9, the gumbel-softmax operation is very hard to train. We explore a lot of hyper-parameters to make the training stable, however, the best result we got is just 15.3 CIDEr score on the COCO captioning dataset (shown in the last line), with the softmax temperature set to 2.0 and exponentially decayed. Compared to the baseline whose visual embeddings is not quantized, there is a huge performance drop when the visual embeddings are quantized to the words of LLM codebook.

This implies that the visual embeddings might share the same representation space of LLM codebook but cannot be “translated" to words with simple matching. We believe this is an interesting direction for future exploration: make the encoder and adapter have zero-shot scalability to larger LLMs.

### 6 Conclusion

In this paper, we propose PaLM2-VAdapter, which uses a tiny language model with progressive training strategy to effectively align vision encoders and LLMs. Demonstrating exceptional zero-shot generalization capabilities across diverse vision-language tasks, PaLM2-VAdapter marks a significant stride in efficiency, operating with substantially fewer parameters than existing models.

Our contributions extend beyond mere technical enhancements in Large Vision-Language Models (LVLMs). We establish a simple but effective framework for future research in vision-language alignment, fostering advancements in multi-modal integration. Morevover, the PaLM2-VAdapter’s success in combining vision and language modality paves the way for further explorations, potentially revolutionizing various applications incorporating more modalities (e.g., audio, pose, ...). Our findings highlight the critical role and vast potential of adapter training strategy in the rapidly evolving domain of multi-modal alignment.

### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In NeurIPS, 2022. 1, 3, 4, 7, 8, 13
- [2] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. PaLM 2 technical report. arXiv preprint arXiv:2305.10403, 2023. 1, 3, 4, 6
- [3] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. OpenFlamingo: An opensource framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390, 2023. 3
- [4] Hangbo Bao, Wenhui Wang, Li Dong, Qiang Liu, Owais Khan Mohammed, Kriti Aggarwal, Subhojit Som, Songhao Piao, and Furu Wei. Vlmo: Unified vision-language pre-training with mixture-of-modality-experts. NeurIPS, 2022. 2
- [5] Jeffrey P Bigham, Chandrika Jayant, Hanjie Ji, Greg Little, Andrew Miller, Robert C Miller, Robin Miller, Aubrey Tatarowicz, Brandyn White, Samual White, et al. Vizwiz: nearly real-time answers to visual questions. In Proceedings of the 23nd annual ACM symposium on User interface software and technology, pages 333–342, 2010. 5, 7
- [6] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In NeurIPS, 2020. 1
- [7] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Nan Ding, Keran Rong, Hassan Akbari, Gaurav Mishra, Linting Xue, Ashish V Thapliyal, James Bradbury, Weicheng Kuo, Mojtaba Seyedhosseini, Chao Jia, Burcu Karagol Ayan, Carlos Riquelme Ruiz, Andreas Peter Steiner, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and Radu Soricut. PaLI: A jointly-scaled multilingual language-image model. In ICLR, 2023. 1, 4
- [8] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015. 5, 6
- [9] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416, 2022. 3
- [10] Zi-Yi Dou, Aishwarya Kamath, Zhe Gan, Pengchuan Zhang, Jianfeng Wang, Linjie Li, Zicheng Liu, Ce Liu, Yann LeCun, Nanyun Peng, et al. Coarse-to-fine vision-language pre-training with fusion in the backbone. NeurIPS, 2022. 2
- [11] Jiali Duan, Liqun Chen, Son Tran, Jinyu Yang, Yi Xu, Belinda Zeng, and Trishul Chilimbi. Multi-modal alignment using representation codebook. In CVPR, 2022. 2
- [12] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering. In CVPR, 2017. 5, 7
- [13] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. In ICLR, 2017. 9
- [14] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In ICML, 2021. 1, 2
- [15] Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M Rush, Douwe Kiela, et al.

- OBELISC: An open web-scale filtered dataset of interleaved image-text documents. arXiv preprint arXiv:2306.16527, 2023. 7, 8
- [16] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2023. 1, 2, 3, 8
- [17] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, 2022. 1, 2
- [18] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. NeurIPS, 2021. 2
- [19] Wei Li, Linchao Zhu, Longyin Wen, and Yi Yang. Decap: Decoding CLIP latents for zeroshot captioning via text-only training. In The Eleventh International Conference on Learning Representations, 2023. 7
- [20] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO: Common objects in context. In ECCV,

2014. 2

- [21] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023. 1, 3, 15
- [22] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023. 3, 8
- [23] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge. In CVPR, 2019. 5, 7
- [24] Mathew Monfort, SouYoung Jin, Alexander Liu, David Harwath, Rogerio Feris, James Glass, and Aude Oliva. Spoken moments: Learning joint audio-visual representations from video descriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14871–14881, June 2021. 4
- [25] Seungwhan Moon, Andrea Madotto, Zhaojiang Lin, Tushar Nagarajan, Matt Smith, Shashank Jain, Chun-Fu Yeh, Prakash Murugesan, Peyman Heidari, Yue Liu, Kavya Srinet, Babak Damavandi, and Anuj Kumar. AnyMAL: An efficient and scalable any-modality augmented language model. arXiv preprint arXiv:2309.16058, 2023. 1, 7, 8
- [26] OpenAI. GPT-4 Technical Report. arXiv preprint arXiv:2303.08774, 2023. 3
- [27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML,

2021. 1, 2, 15

- [28] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 2
- [29] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 5, 7
- [30] Krishna Srinivasan, Karthik Raman, Jiecao Chen, Michael Bendersky, and Marc Najork. Wit: Wikipedia-based image text dataset for multimodal multilingual machine learning. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2443–2449, 2021. 2
- [31] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 1, 3
- [32] Jianfeng Wang, Xiaowei Hu, Zhe Gan, Zhengyuan Yang, Xiyang Dai, Zicheng Liu, Yumao Lu, and Lijuan Wang. Ufo: A unified transformer for vision-language representation learning. arXiv preprint arXiv:2111.10023, 2021. 2
- [33] Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, high-quality multilingual dataset for video-and-language research. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4581–4591, 2019. 5, 6

- [34] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. Simvlm: Simple visual language model pretraining with weak supervision. ICLR, 2021. 2
- [35] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In ACM Multimedia, 2017. 5, 8
- [36] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5288–5296, 2016. 5, 6, 8
- [37] Shen Yan, Tao Zhu, Zirui Wang, Yuan Cao, Mi Zhang, Soham Ghosh, Yonghui Wu, and Jiahui Yu. Video-text modeling with zero-shot transfer from contrastive captioners. arXiv preprint arXiv:2212.04979, 2022. 5, 7
- [38] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Just ask: Learning to answer questions from millions of narrated videos. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1686–1697, 2021. 5, 8
- [39] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Zero-shot video question answering via frozen bidirectional language models. Advances in Neural Information Processing Systems, 35:124–141, 2022. 8
- [40] Jinyu Yang, Jiali Duan, Son Tran, Yi Xu, Sampath Chanda, Liqun Chen, Belinda Zeng, Trishul Chilimbi, and Junzhou Huang. Vision-language pre-training with triple contrastive learning. In CVPR, 2022. 2
- [41] Qinghao Ye, Guohai Xu, Ming Yan, Haiyang Xu, Qi Qian, Ji Zhang, and Fei Huang. Hitea: Hierarchical temporal-aware video-language pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15405–15416, 2023. 8
- [42] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. TMLR, 2022. 1, 2, 4, 6, 15
- [43] Lili Yu, Bowen Shi, Ramakanth Pasunuru, Benjamin Muller, Olga Golovneva, Tianlu Wang, Arun Babu, Binh Tang, Brian Karrer, Shelly Sheynin, et al. Scaling autoregressive multi-modal models: Pretraining and instruction tuning. arXiv preprint arXiv:2309.02591, 2023. 7
- [44] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 3

### A Implementation Details.

- Table 10 provides the detailed training recipe regarding to the hyper-parameters. We use 512 TPU v3 for training all the experiments. The stage-1 alignment takes 8 hours and stage-2 alignment takes around 20 hours.

|Hyperparameter<br><br>|Setting|
|---|---|
|Warmup steps Learning rate Batch size AdamW β Weight decay Image resolution Patch size Prompt LLM decode mode|1000 5e-4 2048 (0.9,0.999) 0.0001 288 18 “Describe the following: <visual tokens> :” Greedy<br><br>|

Table 10: Training settings for vision-language alignment.

|Benchmark<br><br>|Task Type|Prompt Template<br><br>|
|---|---|---|
|COCO MSRVTT VATEX<br><br>|Image & Video Captioning<br><br>|Describe the following <visual tokens> :|
|VQAv2 TextVQA Vizwiz OKVQA|Image Question Answering<br><br>|Answer the question given the images. Given <visual tokens>. Question : <question> ? Answer:|
|MSRVTT-QA MSVD-QA iVQA<br><br>|Video Question Answering<br><br>|Answer the question given the images. Given <visual tokens>. Question : <question> ? Answer in exactly one word:|

- Table 11: Prompt templates used for the Visual Captioning and Question Answering benchmarks.

### B Zero-Shot Generalization to VQA tasks

Following Flamingo [1], we use several pseudo samples from the downstream VQA tasks as the prompt prefix context (4 examples are used in our experiments). All the pseudo examples in the prompt are randomly selected from the training set and just include questions and answers (without image). An example zero-shot VQA prompt with two pseudo samples is:

Answer the question given the images. Given Question: Is the river clear? Answer: yes Given Question: Is this arctic or desert? Answer: desert Given <visual embeddings > Image question: Where is he cooking? Answer:

### C Additional Qualitative Examples

Figure 5 shows additional qualitative examples regarding to image and video tasks (i.e., captioning and QA). Our models demonstrate strong vision-language understanding ability and reasoning ability.

|[Figure 59]|[Figure 60]|
|---|---|

|[Figure 61]|[Figure 62]|[Figure 63]|[Figure 64]|
|---|---|---|---|
|[Figure 65]|[Figure 66]|[Figure 67]|[Figure 68]|

Caption:

Caption:

Caption:

Caption:

Two women are walking down the street in the rain

A baseball game is being played on a field

A person is cutting an onion on a cutting board

A video of a man speaking in a crowd

|[Figure 69]|[Figure 70]|
|---|---|

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Caption:

Caption: A time lapse of a city at night

Caption: A man is playing the violin

Caption:

A baby elephant is walking through the forest

A man is sitting in a car and he is talking about the car

|[Figure 79]|[Figure 80]|
|---|---|

|[Figure 81]|[Figure 82]|[Figure 83]|[Figure 84]|
|---|---|---|---|
|[Figure 85]|[Figure 86]|[Figure 87]|[Figure 88]|

Q: How many wheels do you

Q: What is sitting atop the

Q: What does a person jump off? A: Cliff

Q: What is a woman coating a pork chop in? A: Flour

see? A: 4

chairs? A: Dog

|[Figure 89]|[Figure 90]|[Figure 91]|[Figure 92]|
|---|---|---|---|
|[Figure 93]|[Figure 94]|[Figure 95]|[Figure 96]|

|[Figure 97]|[Figure 98]|
|---|---|

Q: What color is the bird? A: Yellow

Q: What color is the train

Q: What is the woman pencilling on? A: Eyebrow

Q: What is a woman holding? A: Cat

engine? A: Red

Figure 5: Additional qualitative examples. Top left: Image captioning on the COCO dataset. Top right: Video captioning on the MSRVTT dataset. Bottom left: Image question answering on the VQAv2 dataset. Bottom right: Video question answering on the MSVD-QA dataset.

### D Compatibility to CLIP Vision Encoder

We conduct an additional experiment with a vision encoder pretrained with contrastive loss (same as CLIP) on the same image-text paired data used in CoCa. The table below clearly shows that our method leads to consistent improvements, whether we use ViT pre-trained with CLIP or ViT pre-trained with CoCa. This is in comparison to the state-of-the-art adapter design, known as the perceiver resampler.

Vision Enc. ViT Pretrain LLM Adapter COCO Cap. MSRVTT Cap. VQAv2

ViT-B CLIP [27] PaLM2-1B Perceiver Resampler 75.6 37.6 43.2 ViT-B CLIP [27] PaLM2-1B PaLM2-VAdapter 81.7 41.1 48.9 ViT-B CoCa [42] PaLM2-1B Perceiver Resampler 81.4 38.2 52.5 ViT-B CoCa [42] PaLM2-1B PaLM2-VAdapter 83.0 42.1 53.8

- Table 12: Generalization to other encoders. Our PaLM2-VAdapter can also generalize to other vision encoders like CLIP [27].

E Comparison to MLP adapter

We also experiment with another baseline using the MLP architecture in the style of LLaVA v1.5 [21] (all the same experiment settings except using a different adapter), detailed in Table 13. As shown in the following table, the MLP baseline performs much lower performance than our PaLM2-VAdapter.

Vision Enc. LLM Adapter COCO Cap. MSRVTT Cap. VQAv2 CoCa-B PaLM2-1B MLP 80.3 41.2 43.4 CoCa-B PaLM2-1B Perceiver Resampler 81.4 38.2 52.5 CoCa-B PaLM2-1B PaLM2-VAdapter 83.0 42.1 53.8

- Table 13: Comparision to MLP Adapter. Our PaLM2-VAdapter beats the commonly used MLP adapter (e.g., LLaVA [21]) by a large margin.

### F Broader Impact

This work presents a method to build vision language adapters effectively and efficiently. It fits in the broader context of large vision language models and share many of the benefits and issues of such models. The advancements in vision language models enable many useful applications across various fields. However, it is crucial to acknowledge potential biases and ethical implications in the models, especially because the models utilizes pre-trained checkpoints and datasets and thus inherits such issues. Research directions including mitigating biases in training data, improving algorithmic fairness and privacy-preserving techniques are becoming extremely vital to explore in order to address these harmful issues and benefit the broader community.

