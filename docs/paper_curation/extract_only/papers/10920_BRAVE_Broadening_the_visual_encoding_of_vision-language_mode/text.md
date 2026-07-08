[Figure 1]

## BRAVE : Broadening the visual encoding of vision-language models

# arXiv:2404.07204v1[cs.CV]10Apr2024

Oğuzhan Fatih Kar1,2 Alessio Tonioni1 Petra Poklukar1 Achin Kulshrestha1 Amir Zamir2 Federico Tombari1

1Google 2Swiss Federal Institute of Technology Lausanne (EPFL) https://brave-vlms.epfl.ch

###### Improving visual capabilities of VLMs with BRAVE SoTA Performance in Captioning & VQA

[Figure 2]

Are the butterfly’s feet visible?

Is the door of the truck open?

Is there a hand using the mouse in this image?

|[Figure 3]|
|---|

|[Figure 4]|
|---|

|[Figure 5]|
|---|

|[Figure 6]|
|---|

|[Figure 7]|
|---|

|[Figure 8]|
|---|

(a) Yes (b) No (a) Yes (b) No (a) Yes (b) No

InstructBLIP (a) (b) (a) (a) (a) (a) (EVA Encoder)

✅

❌ ❌

LLaVA-1.5 (b) (b) (a) (b) (a) (a) (CLIP Encoder)

✅

❌ ❌

BLIP-2 InstructBLIP LLaVA-1.5 BRAVE

BRAVE (a) (b) (a) (b) (a) (b) (Multiple Encoders)

✅ ✅ ✅

Fig. 1: We propose BRAVE to broaden the visual capabilities of vision-language models (VLMs). Left: In contrast to existing methods, e.g. InstructBLIP [23] or LLaVA1.5 [61], that use a single vision encoder [56,79], BRAVE combines diverse features from multiple vision encoders into a more versatile and compact representation. The examples are taken from [79] and assess the VLM’s ability to differentiate images with visual differences. Right: BRAVE leads to state-of-the-art performance on a wide range of captioning and visual question answering tasks. Furthermore, it significantly improves the performance on benchmarks, e.g. MMVP, where commonly employed vision encoders, e.g. CLIP, fail.

Abstract. Vision-language models (VLMs) are typically composed of a vision encoder, e.g. CLIP, and a language model (LM) that interprets the encoded features to solve downstream tasks. Despite remarkable progress, VLMs are subject to several shortcomings due to the limited capabilities of vision encoders, e.g. “blindness” to certain image features, visual hallucination, etc. To address these issues, we study broadening the visual encoding capabilities of VLMs. We first comprehensively benchmark several vision encoders with different inductive biases for solving VLM tasks. We observe that there is no single encoding configuration that consistently achieves top performance across different tasks, and encoders with different biases can perform surprisingly similarly. Motivated by this, we introduce a method, named BRAVE, that consolidates features from multiple frozen encoders into a more versatile representation that can be directly fed as the input to a frozen LM. BRAVE achieves stateof-the-art performance on a broad range of captioning and VQA benchmarks and significantly reduces the aforementioned issues of VLMs, while

requiring a smaller number of trainable parameters than existing methods and having a more compressed representation. Our results highlight the potential of incorporating different visual biases for a more broad and contextualized visual understanding of VLMs.

### 1 Introduction

Vision-language models (VLMs) have recently seen significant improvements on solving tasks requiring both visual and text understanding capabilities such as captioning, visual question answering (VQA), and instruction following. These advancements are fueled by the progress in uni-modal vision encoders [28, 70] and language models (LMs) [20, 22], which are then combined with different “bridging” techniques [3,53,62] to create VLMs. Improved training recipes [23,61] and scalability efforts in terms of training data and model size [5,17,72,82,94] further boost the performance.

Despite the progress, VLMs are subject to several shortcomings: on the language side, the LMs are known to be susceptible to hallucinations and logical faults [7,35,74,76], while on the vision side, they are limited by the capabilities of the vision encoder. For example, Tong et al. [79] observed that commonly employed vision encoders such as CLIP [70] can exhibit “blindness”, i.e. fail to differentiate images with clear visual differences (See Fig. 1, left). Similarly, Li et al. [56] showed that VLMs are prone to visual hallucinations where the models imagine incorrect details about a given image. These limitations create a bottleneck for developing performant and visually grounded VLMs.

To address these, we study broadening the visual encoding capabilities of VLMs. It is known in machine learning that different representations of an input can yield different generalization properties, and each can play a role when ensembled to create a more complete representation [31]. Inspired by this, we first perform a comprehensive evaluation of VLMs which differ only in their vision encoders, i.e. visual biases. For this, we consider encoders with different biases, e.g. different objectives, training data, and model sizes. Our results in Table 2 show that different encoders lead to varying performance across vision-language tasks and there is no single encoder achieving a consistent top performance across tasks. Furthermore, we find that encoders with different biases can perform surprisingly similarly, suggesting that there could be multiple cues [33,42] that can be leveraged to solve the vision-language tasks.

Motivated by these findings, we propose to employ various vision encoders for VLMs and introduce a method to learn how to combine them efficiently. We denote the method as BRAVE, which stands for broadening the visual encoding of VLMs. As shown in Fig. 2, BRAVE combines features from an arbitrary number of vision encoders into a compressed fixed-length visual representation which is then fed as a soft visual prompt to a frozen LM. We achieve the bridging of encoders and the LM by introducing a lightweight multi-encoder querying transformer (MEQ-Former), which is a multi-encoder generalization of the QFormer proposed in the single encoder BLIP-2 framework [53]. In particular, MEQ-Former accepts a textual prompt and a set of learnable queries as inputs

and jointly cross-attends to the features from different vision encoders (Fig. 2). We provide a simple yet effective recipe in Sec. 3 for training the MEQ-Former in a single stage that bypasses the two-stage pre-training paradigm of [53]. Notably, this is achieved with a smaller number of trainable parameters during pre-training than the existing methods [3,5,17,23,58,61,82] (Sec. 4).

Our extensive evaluation on a wide range of captioning and VQA tasks shows that BRAVE effectively consolidates diverse visual signals into a broad and contextual representation, leading to consistently better performance over the state-ofthe-art and improved robustness against out-of-distribution inputs, as shown in

- Fig. 1 and Sec. 4. Furthermore, in contrast to previous works that benefit from scaling the language axis [3,17,23,61] by using larger LMs, our work demonstrates that scaling along the vision axis also has a significant potential for VLMs.

Our contributions can be summarized as follows:

- – We conduct a systematic analysis of several vision encoders with different inductive biases, e.g. training data, objective, model size, on solving visionlanguage tasks under a unified training and evaluation framework.
- – We introduce BRAVE, a method that efficiently consolidates features from any number of vision encoders into a compressed and contextual representation, leading to 1) state-of-the-art performance on several captioning and VQA tasks and 2) significantly improved robustness against visual hallucinations and out-of-distribution inputs.
- – We perform a comprehensive ablation study of BRAVE, highlighting the impact of the design choices, which we hope provide useful insights to the researchers for further advancements in VLMs.

### 2 Impact of vision encoders for vision-language models

To quantify the impact of visual biases on the performance of VLMs, we compare VLMs with different vision encoders on commonly evaluated VQA and captioning tasks. For this, we develop a pre-training, fine-tuning and evaluation setup, as explained next. To the best of our knowledge, this is the first unified and comprehensive evaluation of different vision encoders for vision-language understanding and generation tasks.

VLM architecture. We use a frozen vision encoder which is connected to a frozen language model by a bridge network with trainable parameters. For the bridge, we adopt the Q-Former module from [53]. This choice is based on the following reasons: 1) the Q-Former resamples visual features to a fixed-length output before feeding them to the LM which enables a fair comparison of vision encoders with different output dimensionalities, 2) the Q-Former is efficient to train and evaluate as it is compatible with frozen pre-trained vision encoders and LMs. For the vision encoder, we consider a diverse set of options as described in the next paragraph. For the LM, we follow [53] and use FlanT5-XL [22]. The Q-Former has 110M parameters consisting of a series of cross- and self-attention

- Table 1: Overview of vision encoders we benchmarked. All encoders use a ViT [25] backbone, yet they differ in terms of objective, training data, and model size. ITC: Image-text contrastive learning [70], MIM: Masked image modelling [39], LGC: Local-to-global correspondence learning [10]. See Sec. 2 for details.

Vision encoder Parameters Training data Objective CLIP-L/14 [70] 0.3B OpenAI WIT ITC

OpenCLIP-G/14 [19] 1.8B LAION-2B ITC EVA-CLIP-g [28] 1B LAION-400M MIM (CLIP features) SIGLIP-G/14 [95] 1.8B WebLI ITC (Sigmoid loss)

SILC-G/16 [68] 1.8B WebLI ITC + LGC

ViT-e [17] 3.8B JFT-3B Classification ViT-G [94] 1.8B JFT-3B Classification DINOv2-L/14 [69] 0.3B LVD-142M LGC + iBOT [100]

layers and a fixed number of learnable queries. Unless specified otherwise, we use 32 queries of dimension 768 which is also the hidden dimension of the Q-Former. The queries interact with visual features through cross-attention layers, enabling a significant reduction in their dimensionality. For example, CLIP-L/14 [70] visual features are reduced from 257×1024 to 32×768. The output of the Q-Former is then linearly projected to the textual embedding space of the LM to create a soft visual prompt, and then prepended to the textual prompt that specifies the task. Both visual and textual prompts are fed as inputs to the frozen LM. For captioning, we use the textual prompt “A photo of ” and for VQA, we directly use the question as the prompt. Similar to InstructBLIP [23], we give the textual prompt as additional input to the Q-Former together with the learnable queries to extract more task-aligned visual prompts.

Vision encoders. As summarized in Table 1, we consider eight recently introduced vision encoders: CLIP [70], OpenCLIP [19], EVA-CLIP [28], SIGLIP [95], SILC [68], ViT-e [17], ViT-G [94], and DINOv2 [69]. While they all use ViT-based backbones [25], they differ in terms of 1) training data, e.g. LAION [72], WebLI [17], LVD-142M [69], etc., 2) training objective, e.g. image-text contrastive learning, masked image modelling, classification, etc., and 3) model size, e.g. 300M to 4B parameters. Due to this diversity, they incorporate different biases and thus potentially capture different aspects of the underlying depicted scene.

Pre-training data and objectives. We pre-train the Q-Former using WebLI [17] dataset at 224 × 224 resolution. We use the filtered and de-duplicated English-only subset with 100 million image-text pairs [17,68]. We keep both the vision encoder and the LM frozen and only train the parameters of the Q-Former. The model is trained with captioning objective using the alt-text in WebLI as target, as it has been shown to be effective [87]. Similar to [53], we remove the last Transformer layer of the encoder. We refer the reader to the supplementary material for additional training details and hyperparameters.

- Table 2: Benchmarking vision encoders on different vision-language tasks. For COCO captioning, we report CIDEr [81] score. For VQA tasks, we report top1 accuracy. For MMVP, we report average pair accuracy [79]. Top-3 best results are shown with Blue , Green , and Orange , respectively. All VLMs use FlanT5-XL [22] as the language model and a Q-Former [53] to bridge vision and language modalities. See Sec. 2 for details.

COCO cap. ↑ Karpathy val

VQAv2 ↑ Karpathy val

OKVQA ↑ val

GQA ↑ test-dev

MMVP ↑ test

Vision encoder

CLIP-L/14 133.0 74.4 61.0 48.7 15.3 OpenCLIP-G 128.3 73.3 60.6 48.0 22.0 EVA-CLIP-g 140.9 77.0 63.0 50.1 27.3 SIGLIP-G/14 133.0 74.7 62.5 48.6 24.0

SILC-G/16 141.1 77.0 63.4 49.7 24.0 ViT-e 137.8 75.6 61.9 49.1 25.3 ViT-G 133.8 74.2 61.2 48.3 20.7

DINOv2-L/14 127.6 71.3 59.0 48.0 22.0

Evaluation tasks. We evaluate the obtained VLMs on standard captioning and VQA tasks (See Fig. 3 for an overview). For captioning, we use the COCO captioning benchmark [18] and fine-tune the pre-trained VLM on the commonly employed Karpathy training split [50]. Similar to the pre-training, only the QFormer parameters are updated. For VQA, we again follow the standard practices [17,23,53,61] and fine-tune on a mixture of VQA data that enforces the Q-Former to extract more task-aligned visual features. Our mixture includes the training sets of VQAv2 [34] and OKVQA [66] as well as the synthetically generated VQA data from VQ2A [11], resulting in total 17M training examples. We then evaluate the fine-tuned models on VQAv2 and OKVQA validation sets. Similar to [23, 53], we also evaluate the VLMs’ zero-shot VQA capabilities on GQA [45]. Finally, we evaluate the zero-shot performance on MMVP [79] that includes images with semantic differences that are challenging for the current VLMs to handle. Unlike captioning, we fine-tune both Q-Former and the LM parameters while keeping the vision encoder frozen, as we find this to improve the performance, similar to [61]. For both captioning and VQA tasks, we use 224 × 224 image resolution. See the supplementary for more details and results.

Results. We report the findings in Table 2 and make the following observations:

- – Encoders with different biases can obtain similar performance. While we observe some variation in VLMs’ performance across different encoders, the gap remains small, especially among the top-performing ones. For example, EVA-CLIP-g, SILC-G/16, and ViT-e achieve the best results for VQAv2, GQA and COCO captioning despite their differences in training objective and data, as shown in Table 1. Similar observations can be made for OpenCLIP and DINOv2 encoders that have a small performance gap.

- – MMVP stays challenging for all encoders. While it is originally curated with “CLIP-blind” pairs in [79], i.e. images that CLIP perceives as similar despite their visual differences (See Fig. 1 for examples), most of the encoders stay below the random guess accuracy of 25%. Exceptions are EVA-CLIP and ViT-e which slightly surpass the random guess level (27.3% and 25.3%).
- – For tasks requiring compositional reasoning and open-world knowledge, the performance decreases and the gap among VLMs diminishes, as shown by the results on GQA and OKVQA. More precisely, the standard deviation across all the VLMs’ performance on GQA and OKVQA is 0.8 and 1.36, respectively, but 1.74 on VQAv2 and 4.91 on COCO.
- – Scaling the size of vision encoder can improve the performance as seen by the consistent improvements obtained by ViT-e over ViT-G when scaling from 1.8B to 3.8B parameters.
- – Pre-training data distribution can play a critical role. OpenCLIPG/14 model is larger than CLIP-L/14 and uses more training data, yet it noticeably underperforms on most of the evaluated VQA and captioning tasks except the MMVP benchmark, where OpenCLIP significantly outperforms the CLIP model. Since both models are trained using the same training objective, this indicates that the impact of the training objective and dataset to the performance of the VLM can be disentangled, and both are important.
- – Performance correlates between VQA and captioning tasks. Encoders with better performance on VQA tasks generally also perform better on captioning, and vice versa. Spearman’s rank correlation between VQA tasks and COCO captioning is 0.90, 0.80, 0.79, 0.56 for VQAv2, OK-VQA, GQA, and MMVP, respectively. Thus, improving the visual component in VLMs is expected to improve performance for a broad range of tasks.

The above results highlight the importance of using the right vision encoder for VLMs, motivating us to focus on the following question: Can we broaden the vision capabilities of VLMs through combining vision encoders with different visual biases? We answer positively in the next section.

### 3 Combining vision encoders with BRAVE

Motivated by the findings in Sec. 2, we propose using multiple vision encoders with different, and potentially complementary, visual biases to create more capable VLMs. To do this, we introduce BRAVE, a method that combines the strengths of different vision encoders while staying efficient in terms of number of trainable parameters. Below, we provide details of BRAVE.

Overview. As depicted in Fig. 2, we introduce the multi-encoder querying transformer, or MEQ-Former, that combines visual features from an arbitrary set of encoders and outputs a fixed-length visual representation that is given as a soft prompt to a frozen LM. It takes as an input a sequence comprised of a fixed number of learnable queries and embedded text tokens which describe the task

###### BRAVE Framework

###### MEQ-Former Architecture

Output: Two giraffes walking next to each other.

Output features

VE #1

FCFCFC

.........

Input image

[Figure 9]

LM

|Feed Forward|
|---|

|Feed Forward|
|---|

[Figure 10]

[Figure 11]

FC

| | |
|---|---|
|Cross Attention| |

|FC| |
|---|---|
| | |

x N

VE #2

[Figure 12]

| | |
|---|---|
|Self Attention| |

MEQ-Former

...

...

...

...

...

. . . . . . . . .

VE #K

Text prompt

Learnable queries

Concatenated visual features

Text prompt: A photo of

Learnable queries

[Figure 13]

[Figure 14]

Frozen

applied every other block

- Fig. 2: Overview of BRAVE. Left: We keep all the vision encoders (VEs) and the language model (LM) frozen. The linear projection layers are used to concatenate features from K different VEs, e.g. K = 5, sequence-wise. These are then resampled by the MEQ-Former which accepts a set of learnable queries and a text prompt describing the task as inputs. The output of MEQ-Former is projected to the input space of the LM using fully-connected (FC) layers. The total number of trainable parameters is 116M (≈ 1% of the total parameters). Right: Architecture of the MEQ-Former with N = 12 transformer layers. It interacts with the concatenated visual features through cross-attention layers and produces a fixed-length output to be fed as soft visual prompt to the frozen LM.

– for captioning, they are the textual prompt “A photo of”, while for VQA, they are the question prompt for a given image (See supplementary for examples). MEQ-Former interacts with visual features through cross-attention layers. The visual features from different encoders are first linearly projected to the same dimension and then concatenated sequence-wise. The obtained concatenated feature is given as a key and value pair to the cross-attention layers in MEQ-Former, and is cross-attended to by the MEQ-Former’s query sequence (Fig. 2). This resampling enables to efficiently processing a large number of visual features since it bypasses the quadratic complexity of self-attention. It also acts as a “bottleneck” that keeps the total number of VLM’s parameters low compared to a naive ensembling of VLMs that is prohibitively expensive (See a comparison in Sec. 4.4). Moreover, the visual features are not “marked” with any encoderspecific embedding, thus MEQ-Former is not biased to differentiate between encoders, allowing for a simpler design.

Differences with previous work. Our work differs from other popular approaches like LLaVa [61] and PaLI [17] that require significantly more trainable parameters (≈10B) and do not have a resampling mechanism. Resampling visual features from a single encoder has been successfully demonstrated in previous work for single- [23,53] and multiple-frames [3,52,92]. To the best of our knowledge, our work is the first one demonstrating an effective resampling mechanism to consolidate visual features from several arbitrarily different encoders while

keeping the number of trainable parameters small (our MEQ-Former having 116M parameters vs BLIP-2’s Q-Former is 188M). See Sec. 5 for more comparisons.

Pre-training details. We use the same data and objective as in Sec. 2 and only train the MEQ-Former while keeping the vision encoders and the LM frozen. During pre-training, we randomly mask features from each encoder with 20% probability as we found that this can act as a regularizer and enforces MEQ-Former to avoid local minima by attending only to the features of a single encoder (Ablated in Sec. 4.4). After pre-training, we fine-tune the MEQ-Former and (optionally) the LM on the downstream tasks, as explained in Sec. 4.

Architecture. For the main results, we combine five vision encoders, namely EVA-CLIP-g [28], CLIP-L/14 [70], SILC-G/16 [68], ViT-e [17], and DINOv2L/14 [69] to cover all training datasets and objectives from Table 1. Please see Sec. 4.4 and the supplementary for analysis on the contributions of encoders. For the LM, we employ FlanT5-XL [22] as before. We set the visual feature dimension after linear projection as 1408 for concatenation. For MEQ-Former, we use 32×5 = 160 learnable queries to proportionally scale the model capacity to the number of vision encoders and set the hidden dimension as 768. The MEQ-Former resampling results in 14× reduction in total feature size (from 1223 × 1408 to 160 × 768). The total number of trainable parameters during pre-training is 116M which is about 1% of the total number of parameters in the VLM (10.3B).

Implementation. We use FLAX [40] with the Scenic [24] library to implement our models and training pipeline. For evaluations, we use both Scenic and EvalAI API [88]. We pre-train with batch size of 1024 on 64 TPUv5 chips. Please see the supplementary for more details.

### 4 Experiments

#### 4.1 Overview

We perform evaluations on a broad range of captioning (Sec. 4.2) and VQA tasks (Sec. 4.3) to show the effectiveness of BRAVE. Please see Fig. 1 right for a summary of the results and Fig. 3 for an overview of the tasks. We then provide a comprehensive analysis of BRAVE in Sec. 4.4.

For captioning, we evaluate on the popular COCO [18] and NoCaps [2] benchmarks. The latter includes out-of-distribution samples with novel classes that do not exist in the COCO benchmark, hence it serves as a challenging testbed to assess robustness. The results (Table 3) show that BRAVE pushes the state-of-theart for both benchmarks while staying efficient in terms of trainable parameters.

For VQA, we evaluate on the popular VQAv2 [34], OKVQA [66], GQA [45], and VizWiz-QA [37] benchmarks. To assess the robustness of VLMs trained with BRAVE, we additionally include 1) the POPE [56] benchmark that measures

NoCaps

VQAv2

###### POPE

###### COCO

Novel Object Captioning

General VQA

Visual Hallucinations

General Captioning

|[Figure 15]|
|---|

|[Figure 16]|
|---|

|[Figure 17]|
|---|

|[Figure 18]|
|---|

|[Figure 19]|
|---|

Q: Is there a bottle in the image? A: No.

Caption: A crab cake sandwich on a hamburger bun.

Q: What color is the hydrant? A: Black and Yellow.

Q: Is there a surfboard in the image? A: No.

Caption: A large bus sitting next to a very tall building.

OKVQA

###### GQA

VizWiz-QA

MMVP

Outside Knowledge

Spatial Reasoning

Unanswerable Questions

Confusing Pairs

|[Figure 20]|
|---|

|[Figure 21]|
|---|

[Figure 22]

|[Figure 23]|
|---|

| |
|---|

| |
|---|

Q: What company makes this sneakers? A: Converse

Q: On which side of the image is the man? A: Right

Q: Who is this mail for? A: Unanswerable.

Q: Are there cookies stacked on top of other cookies? A: Yes (left) — No (right).

- Fig. 3: Overview of the evaluation tasks. They evaluate different capabilities of VLMs, which is important to understand their strengths and weaknesses. The visualizations are obtained from the corresponding publications [2,18,34,37,45,56,66,79], respectively.

object hallucinations in VLMs, and 2) the MMVP [79] benchmark which exposes the shortcomings of CLIP-based vision encoders that are widely used in VLMs. Our results (Table 4) show that BRAVE achieves consistent improvements over recent methods for all the tasks by leveraging diverse visual features.

#### 4.2 Image captioning

Fine-tuning details. Similar to Sec. 2, we only fine-tune the MEQ-Former on COCO captioning and keep the vision encoders and the LM frozen. During fine-tuning, we set a higher input resolution as 336 × 336 to further boost the performance, similar to previous works [17,53,61] (See Sec. 4.4 for an ablation). Following [17, 23, 53, 82, 91], we also zero-shot evaluated the COCO fine-tuned model on NoCaps [2]. Please see the supplementary for more training details.

Results. We summarize the captioning evaluations in Table 3. BRAVE uses the least number of trainable parameters (116M) yet achieves strong results for both COCO and NoCaps benchmarks. For NoCaps, BRAVE is the best performing method with significant gains over recent methods. This is especially the case for out-domain samples with novel classes, demonstrating the usefulness of diversity in visual features for robustness. For COCO, BRAVE stays competitive with the best performing model, PaLI-17B [17]. Notably, BRAVE achieves this while using 150× fewer trainable parameters (116M vs 16.9B), 16× less pre-training data (100M vs 1.6B) and 3× fewer image pixels (336 × 336 vs 588 × 588) than PaLI-17B, suggesting that having different visual biases is effective for generalization, while keeping the sample complexity low.

- Table 3: Captioning. We compare BRAVE with state-of-the-art captioning methods on COCO and NoCaps (zero-shot). CIDEr [81] score is reported (Best in bold, second best is underlined). Numbers of other methods are taken from the corresponding publications. Dashed line means no result was reported. BRAVE achieves the best results in NoCaps evaluation sets (both out-domain and overall in validation & test) and second best for COCO, while having much fewer trainable parameters than other methods. For example, compared to PaLI-17B [17], BRAVE uses significantly fewer trainable parameters, training data, and image pixels. Yet, it outperforms the baselines on NoCaps and stays competitive on COCO, demonstrating the effectiveness of having different visual biases for generalization. See Sec. 4.2 for more details.

# params COCO (fine-tuned) NoCaps (zero-shot, val) NoCaps (zero-shot, test) Method Trainable Total Karpathy test out-domain overall out-domain overall

Flamingo [3] 10.6B 80B 138.1 - - - SimVLM [86] 632M 632M 143.3 113.7 112.2 - 110.3 Qwen-VL [5] 9.6B 9.6B - - 121.4 - -

BLIP-2 [53] 1.1B 4.1B 144.5 124.8 121.6 - InstructBLIP [23] 188M 14.2B - - 121.9 - -

CoCa [91] 2.1B 2.1B 143.6 - 122.4 - 120.6 GiT2 [82] 5.1B 5.1B 145.0 130.6 126.9 122.3 124.8

PaLI-17B [17] 16.9B 16.9B 149.1 - 127.0 126.7 124.4 BRAVE 116M 10.3B 148.0 133.3 127.6 127.1 125.6

#### 4.3 Visual question answering

Fine-tuning details. We use the same VQA mixture as in Sec. 2 and fine-tune both the MEQ-Former and the LM at 224 × 224 resolution. Vision encoders are kept frozen. This is followed by a high-resolution fine-tuning stage on VQAv2 and OKVQA training sets at 336×336 resolution, similar to previous work [17,53,61]. The resulting model is evaluated on VQAv2 and OKVQA benchmarks as well as zero-shot evaluated on GQA [45], VizWiz-QA [37], MMVP [79] and POPE [56], following [23,53,61,79]. To be able to compare to the methods that fine-tune on GQA training set, e.g. [5,58,61], we also perform an additional training stage on top of the mixture-finetuned model, and evaluate the model on the GQA evaluation set (test-dev). Please see the supplementary for the details.

Results. We summarize the VQA evaluations in Table 4. BRAVE achieves strong results for all the benchmarks in both fine-tuned and zero-shot evaluation scenarios. For OKVQA, GQA, VizWiz-QA, MMVP, and POPE benchmarks, BRAVE achieves the best results, and for VQAv2, it stays competitive with PaLI-17B:

- – For MMVP, BRAVE significantly improves the performance over previous approaches, making it less susceptible to failure modes of CLIP-based encoders. This can be further seen from the qualitatives in Figures 1 and 4.
- – For VizWiz-QA, BRAVE achieves the best results without using any scene text understanding or OCR data during pre-training, as opposed to [5,13,61].
- – For GQA and OKVQA that requires different capabilities, i.e. spatial reasoning and open-world knowledge, BRAVE noticeably improves the performance.

- Table 4: Visual question answering. We compare BRAVE with state-of-the-art VQA methods on VQAv2, OKVQA, GQA, VizWiz-QA, MMVP, and POPE. Top-1 accuracies are reported (Best in bold, second best is underlined). For MMVP, we report average pair accuracy [79]. Numbers of other methods are taken from the corresponding publications. Dashed line means no result was reported. BRAVE achieves the best results for six out of seven benchmarks, while staying efficient in terms of trainable parameters. Similar to the COCO captioning results in Table 3, BRAVE stays competitive with PaLI-17B [17] for VQAv2, while using significantly less pre-training data, image resolution, and model parameters (both trainable and total). See Sec. 4.3 for details.

# params Fine-tuned Zero-shot Method Trainable Total

VQAv2 test-dev

OKVQA val

GQA test-dev

VizWiz-QA test-dev

GQA test-dev

MMVP test

POPE test

SimVLM [86] 632M 632M 80.0 - - - - - Flamingo [3] 10.2B 80B 82.0 57.8 - 31.6 - - -

MiniGPT-v2 [13] 7B 8B - 57.8 60.1 53.6 - - GiT2 [82] 5.1B 5.1B 81.7 - - - - - -

Qwen-VL [5] 9.6B 9.6B 79.5 58.6 59.3 35.2 - - SPHINX-2k [58] 13B 16.5B 80.7 62.6 63.1 44.9 - - 87.2 PaLI-17B [17] 16.9B 16.9B 84.3 64.5 - - - - -

BLIP-2 [53] 1.2B 12.1B 81.6 54.7 - 29.4 44.7 - 85.3 InstructBLIP [23] 188M 14.2B - 55.5 - 33.4 49.5 16.7 78.9

LLaVa1.5 [61] 13B 13.4B 80.0 - 63.3 53.6 - 24.7 85.9 LLaVA1.5 (I-MoF) [79] 13B 13.6B 79.3 - - - - 31.3 86.7

###### BRAVE 3B 10.3B 82.5 66.0 66.3 54.2 52.7 42.0 87.6

- – For POPE, BRAVE further reduces the visual hallucinations over dual encoder methods [58,79], confirming that using multiple encoders can be effective for visual grounding (Also see Sec. 5 for a discussion).
- – For VQAv2, BRAVE is the second best after PaLI-17B, while staying more efficient in terms of training data, model size, and input resolution, similar to the COCO captioning result in Sec. 4.2. We expect BRAVE to benefit from such scaling efforts as well, which is a future direction to investigate.

#### 4.4 Analysis of BRAVE

We study different aspects of BRAVE including an ablation study, contribution of vision encoders to the final performance, comparing MEQ-Former to an ensembling strategy, and the role of pre-training data. Please see the supplementary for more qualitative and quantitative results.

Ablation study. We perform a comprehensive study in Table 5 where we ablate different design choices, e.g. training data and choice of language model. The ablations are labeled from A1 to A8, each showing a particular deviation from the original pre-training setup A0 explained in Sec. 3. We summarize the key findings:

Can you see the key "Z" in the image?

Is the peacock's head visible in the image?

Are the butterfly's feet visible?

Can you see a door in this image?

Is the person in the picture on the grass or on the gravel path?

|[Figure 24]|
|---|

|[Figure 25]|
|---|

|[Figure 26]|
|---|

|[Figure 27]|
|---|

|[Figure 28]|
|---|

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

|[Figure 32]|
|---|

|[Figure 33]|
|---|

(a) Yes (b) No (a) Yes (b) No (a) Yes (b) No (a) Yes (b) No (a) Grass (b) Gravel path

EVA (b) (b) ❌ (a) (b) ✅ (b) (b) ❌ (b) (b) ❌ (a) (b) ✅ CLIP (a) (a) ❌ (b) (b) ❌ (b) (b) ❌ (b) (b) ❌ (b) (b) ❌ SILC (a) (b) ✅ (b) (b) ❌ (a) (a) ❌ (b) (b) ❌ (b) (b) ❌ DINOv2 (a) (a) ❌ (b) (b) ❌ (a) (b) ✅ (b) (b) ❌ (b) (b) ❌ ViT-e (b) (b) ❌ (a) (b) ✅ (b) (b) ❌ (a) (b) ✅ (b) (b) ❌ BRAVE (a) (b) ✅ (a) (b) ✅ (a) (b) ✅ (a) (b) ✅ (a) (b) ✅

Can you see the dorsal fin of the animal?

Is the ground wet?

Are there cookies stacked on top of other cookies?

Is the very top of the cake strawberries or cream?

Do you see any window in this image?

|[Figure 34]|
|---|

|[Figure 35]|
|---|

|[Figure 36]|
|---|

|[Figure 37]|
|---|

|[Figure 38]|
|---|

|[Figure 39]|
|---|

|[Figure 40]|
|---|

|[Figure 41]|
|---|

|[Figure 42]|
|---|

|[Figure 43]|
|---|

(a) Yes (b) No (a) Yes (b) No (a) Strawberries (b) Cream (a) Yes (b) No (a) Yes (b) No

EVA (a) (a) ❌ (a) (a) ❌ (a) (b) ✅ (b) (b) ❌ (b) (b) ❌ CLIP (a) (a) ❌ (a) (a) ❌ (b) (b) ❌ (a) (b) ✅ (a) (b) ✅ SILC (a) (a) ❌ (a) (a) ❌ (a) (b) ✅ (a) (b) ✅ (a) (b) ✅ DINOv2 (a) (a) ❌ (a) (a) ❌ (a) (b) ✅ (a) (b) ✅ (b) (b) ❌ ViT-e (a) (a) ❌ (a) (a) ❌ (a) (b) ✅ (b) (b) ❌ (b) (b) ❌ BRAVE (a) (b) ✅ (a) (b) ✅ (a) (b) ✅ (a) (b) ✅ (a) (b) ✅

- Fig. 4: Qualitative results. We compare predictions of BRAVE and the VLMs with different vision encoders, e.g. CLIP, on samples from the MMVP benchmark. Following [79], a model is considered correct only if it answers to both images in a pair correctly, i.e. if it can successfully differentiate between images with semantic differences. Note that the images in a pair are seen independently, i.e. neither of the images is provided as context for the other one. All encoders output some correct predictions, yet none of them performs consistently well on a broad range of inputs. BRAVE alleviates this by combining diverse visual features, leading to a more consistent performance. The quantitative difference is indeed stark: 42% for BRAVE vs 27.3% for best single encoder (Tables 2 and 4). See supplementary for more qualitative results.

- – (A1): LM fine-tuning for VQA significantly boosts the performance compared to frozen LM, similar to the observation in [61]. To minimize the performance loss, we perform a LoRA [43] fine-tuning by inserting LoRA layers to the LM. With a rank of 128, we compensated for 70% of the performance gap while using 10× fewer trainable parameters than full fine-tuning (324M vs 3B). This underscores the development of parameter-efficient fine-tuning methods as an important direction for more efficient VLMs.
- – (A2): Using synthetic VQA data [11] noticeably enhances our performance, while being much cheaper to collect than human annotated data. This can potentially be further improved by using more diverse templates [36,78].
- – (A3,A4,A5): Encoder dropout during training improves the performance for COCO captioning and VQAv2 but slightly degrades for OKVQA. Using text prompt as an additional input to MEQ-Former helps with extracting better task-aligned features. Similarly, high-resolution fine-tuning stage gives a boost to the performance at the expense of processing more visual tokens.
- – (A6,A7): These ablations combine the impact of previously ablated design choices, further demonstrating their contribution to the final performance.

- Table 5: Ablation study of BRAVE. We start from the base pre-training setup explained in Sec. 3, denoted as A0. Each row corresponds to a particular deviation from A0, while keeping the rest fixed. For example, A5 evaluates the performance when high-resolution fine-tuning is disabled. N/A means the ablation is not applicable for that evaluation. See Sec. 4.4 for the discussions and supplementary for more results.

COCO Captioning (Karpathy val)

VQAv2 (Karpathy val)

OKVQA (val)

Ablation ID Study Subject Original Change

- A0 Base - - 147.0 81.8 65.7

- A1

LM fine-tuning for VQA

Full fine-tuning

No fine-tuning LoRA (r=64) LoRA (r=128)

N/A N/A N/A

78.6 80.7 81.0

57.5 62.8 62.9

- A2 Synthetic VQA data [11] ✓ ✗ N/A 81.1 64.0

- A3 Encoder dropout ✓ ✗ 145.3 81.3 66.0

- A4 MEQ-Former text input ✓ ✗ 145.9 81.4 64.9

- A5 High-res fine-tuning ✓ ✗ 145.2 79.6 65.0

- A6 A3 + A5 ✓ ✗ 144.0 79.0 64.5

- A7 A4 + A5 ✓ ✗ 145.1 78.3 63.4

- A8 Language model FlanT5-XL FlanT5-L 142.5 79.9 65.5

– (A8): A stronger LM significantly helps for all the tasks, underscoring that scaling the VLM along the vision and language axes simultaneously have complementary benefits, and both are important.

Contribution of vision encoders. To understand the impact of vision encoders to the final performance of BRAVE, we perform additional evaluations:

- – In Fig. 5-left, we perform a robustness analysis by removing a subset of vision encoders during inference. This is performed for all the combinations and we report the average drop in performance for COCO captioning and VQAv2 tasks (Full results in supplementary). The results show that the performance of BRAVE degrades gracefully up to two encoder removals for both tasks, indicating some redundancy among the encoders. Because of this, even if some encoders are missing, BRAVE can make use of the remaining set of encoders to compensate. Beyond the two encoder removal, the drop in the performance is more severe, suggesting that some features are more unique and consequently harder to replace using the remaining set.
- – In Fig. 5-right, we compute attention scores of features from different encoders that are cross-attended by the MEQ-Former queries. These are averaged for randomly sampled 1k samples from COCO and VQAv2 validation sets. The results show that the MEQ-Former can cross-attend, or “pay attention to”, some encoders more than the others depending on the downstream task adaptively, further confirming the usefulness of having different biases.

###### Robustness to missing encoders Attention scores of different encoders

[Figure 44]

- Fig. 5: Contribution of vision encoders to BRAVE. Left: We analyze the robustness of BRAVE when a subset of encoders are removed during evaluation. We report the average drop in CIDEr for COCO and accuracy for VQAv2. Right: We compute average attention scores for different vision encoders cross-attended by the MEQ-Former for COCO and VQAv2. See Sec. 4.4 for the discussions.

- Table 6: MEQ-Former vs Ensembling. We compare resampling vision encoder features by using an ensemble of Q-Formers and the proposed MEQ-Former. The latter uses significantly fewer trainable parameters and better captures the strengths of different vision encoders, leading to consistently better performance. All evaluations are performed at 224 × 224 resolution. See Sec. 4.4 for details.

Bridge # of parameters COCO Cap. VQAv2 OKVQA GQA Q-Former Ensemble 605M 140.9 78.5 64.3 50.6

MEQ-Former 116M 145.2 79.6 65.0 51.5

MEQ-Former vs Ensembling. We show that resampling the features from all vision encoders with MEQ-Former lead to strong performance while being efficient in terms of the number of trainable parameters. In Table 6, we compare MEQ-Former to an ensemble of Q-Formers [53]. Each Q-Former is first fully pretrained with its corresponding single vision encoder. This is followed by a joint pre-training and a fine-tuning stage by ensembling all vision encoders and their pre-trained Q-Formers. The outputs of all Q-Formers are fed as input to the same LM (FlanT5-XL [22]). We use the same five encoders as before, hence the only difference is using an ensemble of Q-Formers instead of MEQ-Former for resampling, resulting in 5× the number of trainable parameters. Our results in Table 6 show that MEQ-Former resamples visual features more effectively for several tasks while using fewer trainable parameters.

Role of pre-training data. In Table 7 we evaluate the impact of pre-training data by comparing the BRAVE models pre-trained on WebLI [17] and CC3M [12] datasets. The latter has about 30× fewer samples than the former, and pretraining with it leads to a noticeable degradation in performance, suggesting that more work is needed to reduce the sample complexity of VLMs, e.g. as studied in [48].

- Table 7: Role of pre-training data. We compare VLMs pre-trained on WebLI [17] and CC3M [12] datasets. All evaluations are performed at 224 × 224 resolution. The latter has significantly less image-text pairs which leads to a degradation in the performance, suggesting more studies are needed to reduce the sample complexity of VLM training. See Sec. 4.4 for details.

Pre-training Dataset COCO Cap. VQAv2 OKVQA GQA

CC3M 138.3 76.9 63.4 50.0 WebLI 145.2 79.6 65.0 51.5

### 5 Related work

Several works have focused on training VLMs to solve tasks requiring both vision and text understanding capabilities. Most of these works make use of pre-trained vision encoders [8,25,29,46,70,94] and LMs [5,20,22,41,71,80,98]. Architecturewise, different approaches were developed to project visual features to the textual embedding space, e.g. Q-Former [23,53,101], P-Former [48], cross-attention [3, 5,17,90], linear projector [13,14,62] and MLP [61,84]. Different objectives were also considered depending on the downstream tasks, e.g. image-text contrastive learning [47, 68, 70, 89, 96], auto-regressive text generation [15, 17, 21, 82, 83, 86] or both [53–55,91]. Please also see [97] for a recent survey. In contrast to ours, these works consider a single vision encoder which has inherent limitations [44, 56,77,79] (as shown in Sec. 2).

Concurrent to our work, LLaVa-MoF [79] proposes combining CLIP [70] and DINOv2 [69] encoders using separate adapters. SPHINX [58] mixes CLIP, DINOv2, and Q-Former outputs with linear projection layers. Both of the methods input a concatenation of the extracted visual embeddings to the LM, which is not scalable to multiple encoders. We differ from these works by proposing a flexible and unified resampling mechanism that combines features from several vision encoders while staying efficient and improving the performance (Tables 3, 4).

Our work can also be viewed as scaling the VLMs along the vision axis, unlike other works that explored scaling homogenously model size across the visual and/or language component [1,3,4,16,17,26,32,75,85], training data [5,9,12, 46,47,57,73,82,84] and number of modalities [6,38,63–65,67,93]. BRAVE can be combined with these efforts to further improve performance. Moreover, several studies investigated the visual aspect of VLMs by evaluating them against hallucinations and other robustness issues [30,44,56,59,60,77,99]. Our work improves the robustness of the VLMs to those issues by broadening visual capabilities of the VLMs.

### 6 Conclusions and Limitations

In this paper, we focus on broadening the visual encoding aspect of VLMs. We first perform a comprehensive evaluation of several vision encoders on solving VLM tasks. Our findings revealed that there is no encoder achieving a consistent

top performance across tasks and encoders with different biases can perform similarly. Motivated by this, we introduce BRAVE that empowers VLMs by combining diverse features from several vision encoders using a small number of trainable parameters. Extensive evaluations on a broad range of tasks show that BRAVE achieves state-of-the-art performance across the board and improves robustness of VLMs against out-of-distribution inputs and visual hallucinations. Below we discuss some limitations of our work and possible future directions:

- – Adaptive mechanisms: While MEQ-Former can resample from the useful features and discard the irrelevant ones, it still needs forward passes from all the encoders. A future direction could be to explore adaptive mechanisms to pre-select which encoders to resample from, reducing the inference cost.
- – Improving sample efficiency: BRAVE uses around 100M image-text pairs during pre-training which is an order of magnitude less than recent methods [5,17,82]. Further lowering the sample complexity is an important direction that will help reducing the development costs, e.g. by using a strategy similar to P-Former [48].
- – More broad vision encoders: The set of encoders we investigated do not completely cover all types of visual biases, e.g. inclusion of those with strong 3D [27,49] or semantic [51] priors could further improve the performance. Our results show that the current set is a good starting point to see the benefits of consolidation. Another interesting direction is to strengthen vision encoders by distilling the MEQ-Former-combined features back into a single encoder.
- – Different modalities and multiple frames. While we focus on image and text modalities, BRAVE is a general method that could be potentially used to fuse more modalities, e.g. audio or 3D, or extended to incorporate multiple frames, e.g. for video understanding or in-context learning.
- – Biases of text-generative models. As we employ pre-trained LMs in our setup, the resulting VLMs are susceptible to the longstanding bias and fairness issues of these models, thus they should be used with caution especially for safety-critical applications.

### 7 Acknowledgements

We thank Diego Martin Arroyo, Ferjad Naeem, Xingyi Zhou, Yannick Strümpler and Yongqin Xian for their help with the project.

### References

- 1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)
- 2. Agrawal, H., Desai, K., Wang, Y., Chen, X., Jain, R., Johnson, M., Batra, D., Parikh, D., Lee, S., Anderson, P.: Nocaps: Novel object captioning at scale. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 8948–8957 (2019)

- 3. Alayrac, J.B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al.: Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems 35, 23716–23736 (2022)
- 4. Awadalla, A., Gao, I., Gardner, J., Hessel, J., Hanafy, Y., Zhu, W., Marathe, K., Bitton, Y., Gadre, S., Sagawa, S., et al.: Openflamingo: An open-source framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390 (2023)
- 5. Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., et al.: Qwen technical report. arXiv preprint arXiv:2309.16609 (2023)
- 6. Bai, Y., Geng, X., Mangalam, K., Bar, A., Yuille, A., Darrell, T., Malik, J., Efros, A.A.: Sequential modeling enables scalable learning for large vision models. arXiv preprint arXiv:2312.00785 (2023)
- 7. Bang, Y., Cahyawijaya, S., Lee, N., Dai, W., Su, D., Wilie, B., Lovenia, H., Ji, Z., Yu, T., Chung, W., et al.: A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity. arXiv preprint arXiv:2302.04023 (2023)
- 8. Brock, A., De, S., Smith, S.L., Simonyan, K.: High-performance large-scale image recognition without normalization. In: International Conference on Machine Learning. pp. 1059–1071. PMLR (2021)
- 9. Byeon, M., Park, B., Kim, H., Lee, S., Baek, W., Kim, S.: Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/coyo-dataset (2022)
- 10. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9650–9660

(2021)

- 11. Changpinyo, S., Kukliansky, D., Szpektor, I., Chen, X., Ding, N., Soricut, R.: All you may need for vqa are image captions. arXiv preprint arXiv:2205.01883 (2022)
- 12. Changpinyo, S., Sharma, P.K., Ding, N., Soricut, R.: Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 3557–3567 (2021)
- 13. Chen, J., Zhu, D., Shen, X., Li, X., Liu, Z., Zhang, P., Krishnamoorthi, R., Chandra, V., Xiong, Y., Elhoseiny, M.: Minigpt-v2: large language model as a unified interface for vision-language multi-task learning. arXiv preprint arXiv:2310.09478

(2023)

- 14. Chen, K., Zhang, Z., Zeng, W., Zhang, R., Zhu, F., Zhao, R.: Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195

(2023)

- 15. Chen, T., Saxena, S., Li, L., Fleet, D.J., Hinton, G.: Pix2seq: A language modeling framework for object detection. In: International Conference on Learning Representations (2022)
- 16. Chen, X., Djolonga, J., Padlewski, P., Mustafa, B., Changpinyo, S., Wu, J., Ruiz, C.R., Goodman, S., Wang, X., Tay, Y., et al.: Pali-x: On scaling up a multilingual vision and language model. arXiv preprint arXiv:2305.18565 (2023)
- 17. Chen, X., Wang, X., Changpinyo, S., Piergiovanni, A., Padlewski, P., Salz, D., Goodman, S., Grycner, A., Mustafa, B., Beyer, L., et al.: Pali: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794 (2022)
- 18. Chen, X., Fang, H., Lin, T.Y., Vedantam, R., Gupta, S., Dollár, P., Zitnick, C.L.: Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325 (2015)

- 19. Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., Jitsev, J.: Reproducible scaling laws for contrastive language-image learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2818–2829 (2023)
- 20. Chiang, W.L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J.E., et al.: Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April

2023) (2023)

- 21. Cho, J., Lei, J., Tan, H., Bansal, M.: Unifying vision-and-language tasks via text generation. In: International Conference on Machine Learning. pp. 1931–1942. PMLR (2021)
- 22. Chung, H.W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., et al.: Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416 (2022)
- 23. Dai, W., Li, J., Li, D., Tiong, A., Zhao, J., Wang, W., Li, B., Fung, P., Hoi, S.: InstructBLIP: Towards general-purpose vision-language models with instruction tuning. In: Thirty-seventh Conference on Neural Information Processing Systems

(2023), https://openreview.net/forum?id=vvoWPYqZJA

- 24. Dehghani, M., Gritsenko, A., Arnab, A., Minderer, M., Tay, Y.: Scenic: A jax library for computer vision research and beyond. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21393–21398 (2022)
- 25. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (2021)
- 26. Driess, D., Xia, F., Sajjadi, M.S., Lynch, C., Chowdhery, A., Ichter, B., Wahid, A., Tompson, J., Vuong, Q., Yu, T., et al.: Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378 (2023)
- 27. Eftekhar, A., Sax, A., Bachmann, R., Malik, J., Zamir, A.R.: Omnidata: A scalable pipeline for making multi-task mid-level vision datasets from 3d scans. 2021 IEEE/CVF International Conference on Computer Vision (ICCV) pp. 10766– 10776 (2021)
- 28. Fang, Y., Wang, W., Xie, B., Sun, Q., Wu, L., Wang, X., Huang, T., Wang, X., Cao, Y.: Eva: Exploring the limits of masked visual representation learning at scale. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19358–19369 (2023)
- 29. Fang, Y., Wang, W., Xie, B., Sun, Q.S., Wu, L.Y., Wang, X., Huang, T., Wang, X., Cao, Y.: EVA: Exploring the limits of masked visual representation learning at scale. ArXiv abs/2211.07636 (2022)
- 30. Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., et al.: Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394 (2023)
- 31. Geman, S., Bienenstock, E., Doursat, R.: Neural networks and the bias/variance dilemma. Neural computation 4(1), 1–58 (1992)
- 32. Gong, T., Lyu, C., Zhang, S., Wang, Y., Zheng, M., Zhao, Q., Liu, K., Zhang, W., Luo, P., Chen, K.: Multimodal-gpt: A vision and language model for dialogue with humans. arXiv preprint arXiv:2305.04790 (2023)
- 33. Goodwin, C.: Seeing in depth. Social studies of science 25(2), 237–274 (1995)
- 34. Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., Parikh, D.: Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In:

- Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 6904–6913 (2017)
- 35. Guo, B., Zhang, X., Wang, Z., Jiang, M., Nie, J., Ding, Y., Yue, J., Wu, Y.: How close is chatgpt to human experts? comparison corpus, evaluation, and detection. arXiv preprint arXiv:2301.07597 (2023)
- 36. Guo, J., Li, J., Li, D., Tiong, A.M.H., Li, B., Tao, D., Hoi, S.: From images to textual prompts: Zero-shot visual question answering with frozen large language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10867–10877 (2023)
- 37. Gurari, D., Li, Q., Stangl, A.J., Guo, A., Lin, C., Grauman, K., Luo, J., Bigham, J.P.: Vizwiz grand challenge: Answering visual questions from blind people. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 3608–3617 (2018)
- 38. Han, J., Zhang, R., Shao, W., Gao, P., Xu, P., Xiao, H., Zhang, K., Liu, C., Wen, S., Guo, Z., et al.: Imagebind-llm: Multi-modality instruction tuning. arXiv preprint arXiv:2309.03905 (2023)
- 39. He, K., Chen, X., Xie, S., Li, Y., Doll’ar, P., Girshick, R.B.: Masked autoencoders are scalable vision learners. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 15979–15988 (2021)
- 40. Heek, J., Levskaya, A., Oliver, A., Ritter, M., Rondepierre, B., Steiner, A., van Zee, M.: Flax: A neural network library and ecosystem for jax, 2020. URL http://github. com/google/flax 1 (2020)
- 41. Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D.d.L., Hendricks, L.A., Welbl, J., Clark, A., et al.: Training computeoptimal large language models. arXiv preprint arXiv:2203.15556 (2022)
- 42. Howard, I.P., Rogers, B.J.: Binocular vision and stereopsis. Oxford University Press, USA (1995)
- 43. Hu, J.E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Chen, W.: LoRA: Low-rank adaptation of large language models. ArXiv abs/2106.09685 (2021), https://api.semanticscholar.org/CorpusID:235458009
- 44. Huang, W., Liu, H., Guo, M., Gong, N.Z.: Visual hallucinations of multi-modal large language models. arXiv preprint arXiv:2402.14683 (2024)
- 45. Hudson, D.A., Manning, C.D.: Gqa: A new dataset for real-world visual reasoning and compositional question answering. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6700–6709 (2019)
- 46. Ilharco, G., Wortsman, M., Wightman, R., Gordon, C., Carlini, N., Taori, R., Dave, A., Shankar, V., Namkoong, H., Miller, J., Hajishirzi, H., Farhadi, A., Schmidt, L.: Openclip (Jul 2021). https://doi.org/10.5281/zenodo.5143773, https://doi.org/10.5281/zenodo.5143773, if you use this software, please cite it as below.
- 47. Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H., Li, Z., Duerig, T.: Scaling up visual and vision-language representation learning with noisy text supervision. In: International conference on machine learning. pp. 4904–4916. PMLR (2021)
- 48. Jian, Y., Gao, C., Vosoughi, S.: Bootstrapping vision-language learning with decoupled language pre-training. Advances in Neural Information Processing Systems 36 (2024)
- 49. Kar, O.F., Yeo, T., Atanov, A., Zamir, A.: 3d common corruptions and data augmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18963–18974 (2022)

- 50. Karpathy, A., Fei-Fei, L.: Deep visual-semantic alignments for generating image descriptions. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 3128–3137 (2015)
- 51. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. arXiv preprint arXiv:2304.02643 (2023)
- 52. Korbar, B., Xian, Y., Tonioni, A., Zisserman, A., Tombari, F.: Text-conditioned resampler for long form video understanding. arXiv preprint arXiv:2312.11897

(2023)

- 53. Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597 (2023)
- 54. Li, J., Li, D., Xiong, C., Hoi, S.C.H.: BLIP: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In: International Conference on Machine Learning (2022)
- 55. Li, J., Selvaraju, R., Gotmare, A., Joty, S., Xiong, C., Hoi, S.C.H.: Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems 34, 9694–9705 (2021)
- 56. Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W.X., Wen, J.R.: Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355

(2023)

- 57. Lin, J., Yin, H., Ping, W., Lu, Y., Molchanov, P., Tao, A., Mao, H., Kautz, J., Shoeybi, M., Han, S.: Vila: On pre-training for visual language models. arXiv preprint arXiv:2312.07533 (2023)
- 58. Lin, Z., Liu, C., Zhang, R., Gao, P., Qiu, L., Xiao, H., Qiu, H., Lin, C., Shao, W., Chen, K., et al.: Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575 (2023)
- 59. Liu, F., Lin, K., Li, L., Wang, J., Yacoob, Y., Wang, L.: Mitigating hallucination in large multi-modal models via robust instruction tuning. In: The Twelfth International Conference on Learning Representations (2023)
- 60. Liu, H., Xue, W., Chen, Y., Chen, D., Zhao, X., Wang, K., Hou, L., Li, R., Peng, W.: A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253 (2024)
- 61. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744 (2023)
- 62. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. arXiv preprint arXiv:2304.08485 (2023)
- 63. Liu, S., Fan, L.J., Johns, E., Yu, Z., Xiao, C., Anandkumar, A.: Prismer: A visionlanguage model with an ensemble of experts. ArXiv abs/2303.02506 (2023)
- 64. Lu, J., Clark, C., Lee, S., Zhang, Z., Khosla, S., Marten, R., Hoiem, D., Kembhavi, A.: Unified-io 2: Scaling autoregressive multimodal models with vision, language, audio, and action. arXiv preprint arXiv:2312.17172 (2023)
- 65. Lu, J., Clark, C., Zellers, R., Mottaghi, R., Kembhavi, A.: Unified-IO: A unified model for vision, language, and multi-modal tasks. In: The Eleventh International Conference on Learning Representations (2023)
- 66. Marino, K., Rastegari, M., Farhadi, A., Mottaghi, R.: Ok-vqa: A visual question answering benchmark requiring external knowledge. In: Proceedings of the IEEE/cvf conference on computer vision and pattern recognition. pp. 3195–3204

(2019)

- 67. Mizrahi, D., Bachmann, R., Kar, O.F., Yeo, T., Gao, M., Dehghan, A., Zamir, A.: 4M: Massively multimodal masked modeling. In: Advances in Neural Information Processing Systems (2023)
- 68. Naeem, M.F., Xian, Y., Zhai, X., Hoyer, L., Van Gool, L., Tombari, F.: Silc: Improving vision language pretraining with self-distillation. arXiv preprint arXiv:2310.13355 (2023)
- 69. Oquab, M., Darcet, T., Moutakanni, T., Vo, H.Q., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.Y.B., Li, S.W., Misra, I., Rabbat, M.G., Sharma, V., Synnaeve, G., Xu, H., Jégou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: DINOv2: Learning robust visual features without supervision. ArXiv abs/2304.07193 (2023)
- 70. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning

(2021)

- 71. Raffel, C., Shazeer, N.M., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the limits of transfer learning with a unified textto-text transformer. The Journal of Machine Learning Research 21(1), 5485–5551

(2020)

- 72. Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., Schramowski, P., Kundurthy, S., Crowson, K., Schmidt, L., Kaczmarczyk, R., Jitsev, J.: LAION-5B: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems 35, 25278–25294 (2022)
- 73. Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al.: Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems 35, 25278–25294 (2022)
- 74. Shen, Y., Heacock, L., Elias, J., Hentel, K.D., Reig, B., Shih, G., Moy, L.: Chatgpt and other large language models are double-edged swords (2023)
- 75. Team, G., Anil, R., Borgeaud, S., Wu, Y., Alayrac, J.B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A.M., Hauth, A., et al.: Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 (2023)
- 76. Thorp, H.H.: Chatgpt is fun, but not an author (2023)
- 77. Thrush, T., Jiang, R., Bartolo, M., Singh, A., Williams, A., Kiela, D., Ross, C.: Winoground: Probing vision and language models for visio-linguistic compositionality. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5238–5248 (2022)
- 78. Tiong, A.M.H., Li, J., Li, B., Savarese, S., Hoi, S.C.: Plug-and-play vqa: Zeroshot vqa by conjoining large pretrained models with zero training. arXiv preprint arXiv:2210.08773 (2022)
- 79. Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., Xie, S.: Eyes wide shut? exploring the visual shortcomings of multimodal llms. arXiv preprint arXiv:2401.06209

(2024)

- 80. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al.: Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023)
- 81. Vedantam, R., Lawrence Zitnick, C., Parikh, D.: Cider: Consensus-based image description evaluation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4566–4575 (2015)

- 82. Wang, J., Yang, Z., Hu, X., Li, L., Lin, K., Gan, Z., Liu, Z., Liu, C., Wang, L.: Git: A generative image-to-text transformer for vision and language. arXiv preprint arXiv:2205.14100 (2022)
- 83. Wang, P., Yang, A., Men, R., Lin, J., Bai, S., Li, Z., Ma, J., Zhou, C., Zhou, J., Yang, H.: Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In: International Conference on Machine Learning. pp. 23318–23340. PMLR (2022)
- 84. Wang, W., Lv, Q., Yu, W., Hong, W., Qi, J., Wang, Y., Ji, J., Yang, Z., Zhao, L., Song, X., et al.: Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079 (2023)
- 85. Wang, W., Bao, H., Dong, L., Bjorck, J., Peng, Z., Liu, Q., Aggarwal, K., Mohammed, O.K., Singhal, S., Som, S., Wei, F.: Image as a foreign language: BEiT pretraining for all vision and vision-language tasks. ArXiv abs/2208.10442

(2022)

- 86. Wang, Z., Yu, J., Yu, A.W., Dai, Z., Tsvetkov, Y., Cao, Y.: Simvlm: Simple visual language model pretraining with weak supervision. arXiv preprint arXiv:2108.10904 (2021)
- 87. Xu, J., Zhou, X., Yan, S., Gu, X., Arnab, A., Sun, C., Wang, X., Schmid, C.: Pixel Aligned Language Models. arXiv preprint arXiv: 2312.09237 (2023)
- 88. Yadav, D., Jain, R., Agrawal, H., Chattopadhyay, P., Singh, T., Jain, A., Singh, S.B., Lee, S., Batra, D.: Evalai: Towards better evaluation systems for ai agents

(2019)

- 89. Yao, L., Huang, R., Hou, L., Lu, G., Niu, M., Xu, H., Liang, X., Li, Z., Jiang, X., Xu, C.: Filip: Fine-grained interactive language-image pre-training. arXiv preprint arXiv:2111.07783 (2021)
- 90. Ye, Q., Xu, H., Xu, G., Ye, J., Yan, M., Zhou, Y., Wang, J., Hu, A., Shi, P., Shi, Y., et al.: mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178 (2023)
- 91. Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini, M., Wu, Y.: Coca: Contrastive captioners are image-text foundation models. Transactions on Machine Learning Research (2022)
- 92. Yu, S., Cho, J., Yadav, P., Bansal, M.: Self-chained image-language model for video localization and question answering. arXiv preprint arXiv:2305.06988 (2023)
- 93. Yuan, L., Chen, D., Chen, Y.L., Codella, N., Dai, X., Gao, J., Hu, H., Huang, X., Li, B., Li, C., et al.: Florence: A new foundation model for computer vision. arXiv preprint arXiv:2111.11432 (2021)
- 94. Zhai, X., Kolesnikov, A., Houlsby, N., Beyer, L.: Scaling vision transformers. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12104–12113 (2022)
- 95. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pre-training. arXiv preprint arXiv:2303.15343 (2023)
- 96. Zhai, X., Wang, X., Mustafa, B., Steiner, A., Keysers, D., Kolesnikov, A., Beyer, L.: Lit: Zero-shot transfer with locked-image text tuning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18123– 18133 (2022)
- 97. Zhang, D., Yu, Y., Li, C., Dong, J., Su, D., Chu, C., Yu, D.: Mm-llms: Recent advances in multimodal large language models. arXiv preprint arXiv:2401.13601

(2024)

- 98. Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X., Lin, X.V., et al.: Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068 (2022)

- 99. Zhao, Y., Pang, T., Du, C., Yang, X., Li, C., Cheung, N.M.M., Lin, M.: On evaluating adversarial robustness of large vision-language models. Advances in Neural Information Processing Systems 36 (2024)
- 100. Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., Kong, T.: iBoT: Image BERT pre-training with online tokenizer. In: International Conference on Learning Representations (2022)
- 101. Zhu, D., Chen, J., Shen, X., Li, X., Elhoseiny, M.: Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. arXiv preprint arXiv:2304.10592 (2023)

