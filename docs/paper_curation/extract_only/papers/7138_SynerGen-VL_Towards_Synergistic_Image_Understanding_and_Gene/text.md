# arXiv:2412.09604v1[cs.CV]12Dec2024

## SynerGen-VL: Towards Synergistic Image Understanding and Generation with Vision Experts and Token Folding

Hao Li1,2∗†, Changyao Tian2,1∗†, Jie Shao3,1∗†, Xizhou Zhu4,5∗, Zhaokai Wang6,1†, Jinguo Zhu1,

Wenhan Dou4,5, Xiaogang Wang2, Hongsheng Li2, Lewei Lu5, Jifeng Dai4,1,7 1OpenGVLab, Shanghai AI Laboratory 2MMLab, The Chinese University of Hong Kong

3Nanjing University 4Tsinghua University 5SenseTime Research 6Shanghai Jiao Tong University 7Beijing National Research Center for Information Science and Technology

### Abstract

The remarkable success of Large Language Models (LLMs) has extended to the multimodal domain, achieving outstanding performance in image understanding and generation. Recent efforts to develop unified Multimodal Large Language Models (MLLMs) that integrate these capabilities have shown promising results. However, existing approaches often involve complex designs in model architecture or training pipeline, increasing the difficulty of model training and scaling. In this paper, we propose SynerGen-VL, a simple yet powerful encoder-free MLLM capable of both image understanding and generation. To address challenges identified in existing encoder-free unified MLLMs, we introduce the token folding mechanism and the vision-expert-based progressive alignment pretraining strategy, which effectively support high-resolution image understanding while reducing training complexity. After being trained on large-scale mixed image-text data with a unified next-token prediction objective, SynerGen-VL achieves or surpasses the performance of existing encoder-free unified MLLMs with comparable or smaller parameter sizes, and narrows the gap with task-specific state-of-the-art models, highlighting a promising path toward future unified MLLMs. Our code and models shall be released.

### 1. Introduction

The remarkable success of Large Language Models (LLMs) [7, 59, 84] has been extended to the multimodal domain, achieving impressive performance in image understanding [11, 44, 80, 99] and image generation [75, 83, 95]. Recent research has aimed to develop unified Multimodal

* Equal contribution. † Interns at Shanghai AI Laboratory. Corresponding to Jifeng Dai <daijifeng@tsinghua.edu.cn>.

Large Language Models (MLLMs) with synergistic image understanding and generation capabilities [8, 20, 81, 92, 94, 107]. Although they have demonstrated competitive performance in both tasks, they often involve complex designs as illustrated in Fig. 1(a)∼(d), such as (a) relying on external diffusion models for image generation [20, 24, 81, 108], (b) using different training objectives (i.e. diffusion and autoregression) for the two tasks [96, 107], (c) employing distinct image encoders for each task [92], and (d) require additional semantic pretraining for image tokenizers [94]. These complexities disrupt the simplicity of the next token prediction paradigm of LLMs, increasing systematic difficulty and limiting scalability.

To address these complexities, some studies have tried to develop unified MLLMs with simple architectures, eliminating dependencies on external models, distinct taskspecific models, and additional semantic pretraining [8, 45, 91]. As shown in Fig. 1(e), these approaches adopt a similar tokenization strategy for both images and text, and model both image understanding and generation tasks within a unified next token prediction framework. The image tokenizers [22] are pretrained for reconstruction on pure image data, without requiring human annotations or text supervision, which allows for a broad data distribution and strong scalability. These concise and scalable designs have demonstrated a promising path toward synergistic image understanding and generation.

Nevertheless, these methods still face some key challenges in practical use. Specifically, (1) since both image understanding and generation rely entirely on MLLMs, substantial training is required to incorporate vision capabilities into MLLMs. However, this may interfere with the pretrained knowledge of LLMs, resulting in reduced general perception and generalization capabilities. Although existing methods try to avoid this by training MLLM from scratch using mixed text and multimodal data, they face considerable challenges in optimizing stability, data qual-

Image Text

Image Text

Image Text

Image Text

Image Text

Semantic Encoder Tokenizer

Semantic Tokenizer

Semantic Encoder

Tokenizer

Tokenizer

Tokenizer

Tokenizer

Tokenizer

Tokenizer

Tokenizer

Pretrained Diffusion Model

Diffusion

MLLM (AR)

MLLM (AR)

MLLM (AR)

MLLM (AR)

MLLM (AR)

Text Response

Generated Image

Text Response

Generated Image

Text Response

Generated Image

Text Response

Generated Image

Text Response

Generated Image

e.g., DreamLLM, SEED-X, VL-GPT, MM-Interleaved

e.g., Transfusion, Show-o e.g., Janus e.g., VILA-U e.g., LWM, Chameleon, Emu3, Ours

(a) (b) (c) (d) (e)

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

No externalized models No additional semantic pretraining Unified training objective & inference for und. and gen.

No externalized models No additional semantic pretraining

No externalized models No additional semantic pretraining

No externalized models No additional semantic pretraining

No externalized models No additional semantic pretraining Unified training objective & inference for und. and gen.

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Unified training objective & inference for und. and gen.

Unified training objective & inference for und. and gen.

Unified training objective & inference for und. and gen.

[Figure 14]

[Figure 15]

Figure 1. Comparison among exemplary unified MLLMs for synergizing image understanding and generation tasks. Compared with methods (a)∼(d) that incorporate complicated designs of model architectures, training methods, and the use of external pretrained diffusion models, (e) encoder-free unified MLLMs adopt a simple design that uses the simple next token prediction framework for both images understanding and generation tasks, allowing for broader data distribution and better scalability.

ity, and training cost [8, 91]; (2) current visual tokenizers require low feature downsample ratios to ensure reconstruction with fine details [22]. This results in long visual token sequences for high-resolution images, which is unsuitable to LLMs and limits the use of high-resolution images, thus affecting performance, especially for image understanding.

4W

[Figure 16]

W

4H

[Figure 17]

[Figure 18]

[Figure 19]

H

4096 tokens

###### Token Fold (16x)

256 tokens

256 tokens

In this paper, we aim to build a simple yet powerful unified MLLM that addresses the aforementioned challenges. Specifically, 1) inspired by image understanding models with Multimodal Mixture-of-Experts (MMoE) [37, 51, 89] structure, we introduce vision experts with additional parameters dedicated to image representation. Aligning the vision experts to the frozen LLM helps integrate vision capabilities while minimizing disruption to the LLM’s pretrained knowledge; 2) to effectively support high-resolution images, the input visual token sequence can be compressed to reduce its length, while an additional decoder would be employed during image generation to reconstruct detailed image sequences from the compressed representations.

###### …

… …

###### …

| |
|---|

| |
|---|

MLLM

Vision Expert

MLLM

###### … …

###### … …

256 tokens

256 tokens

Token Unfold (16x)

4096 tokens

Chameleon, Emu3, … Ours

Figure 2. Comparision between SynerGen-VL and previous encoder-free unified MLLMs. SynerGen-VL adopts a token folding and unfolding mechanism and vision experts to build a strong and simple unified MLLM. With the same image context length, SynerGen-VL can support images of much higher resolutions, ensuring the performance of both high-resolution image understanding and generation.

Following this perspective, we propose SynerGen-VL, a high-performance unified MLLM with synergistic image understanding and generation capabilities, using nonsemantic discrete image tokens to represent images. As shown in Fig. 2, compared with previous encoder-free unified MLLMs, SynerGen-VL employs additional vision experts, i.e. image-specific Feed-Forward Networks (FFNs), to incorporate vision capabilities into pretrained LLMs. Meanwhile, SynerGen-VL uses a hierarchical architecture to increase the feature downsampling ratio within the MLLM. Specifically, the input image token sequences are downsampled by token folding to reduce their lengths. To generate high-quality images, the generated token sequences are unfolded by a shallow autoregressive Transformer head. To preserve the LLM’s pretrained knowledge,

we perform two-stage alignment pretraining with mixed image understanding and generation data: (1) only imagespecific FFNs are trained with noisy web data to achieve basic semantic understanding and image generation aligned with the representation space of LLM; (2) image-specific FFNs and self-attention layers are trained with high-quality image understanding and generation data to further integrate multimodal features into the pretrained LLM. After alignment pretraining, SynerGen-VL supports image understanding and generation tasks simultaneously through su-

pervised instruction fine-tuning.

We train SynerGen-VL on large-scale mixed image-text data and evaluate it on a range of image understanding and generation benchmarks. Experimental results demonstrate that, with its simple design, SynerGen-VL achieves or surpasses the performance of existing encoder-free unified MLLMs with comparable or smaller parameter sizes, and narrows the gap with task-specific state-of-the-art (SoTA) models. In particular, with only 2.4B activated parameters, SynerGen-VL achieves image understanding and generation performance on par with Emu3 [91], which has 8B parameters, highlighting its strong potential as a promising path towards next-generation unified MLLM. Our contributions are summarized as follows:

- • We propose SynerGen-VL, a Multimodal Large Language Model (MLLM) with simple architecture and training process, capable of handling both image understanding and generation through a unified next token prediction paradigm.
- • We introduce the token folding mechanism and the vision-expert-based progressive alignment pretraining to unified MLLMs, which effectively support highresolution image understanding and reduce training difficulty.
- • Experiments demonstrate that SynerGen-VL achieves competitive performance in a range of image understanding and generation benchmarks, revealing a promising path towards future unified MLLM.

### 2. Related Work

Unified MLLMs for Synergistic Image Understanding and Generation. Unifying image understanding and generation in a single MLLM has attracted wide academic attention. Early efforts primarily integrate an external diffusion decoder for image generation [24, 38, 76, 77, 93]. Inspired by the success of next-token prediction in LLMs, some studies explore using discrete visual tokens to represent and generate images in a fully autoregressive paradigm [8, 30, 45, 91, 94, 101]. To achieve high performance for both image understanding and generation, some recent methods have decoupled image understanding and generation. Transfusion [107] and Show-o [96] integrate textual autoregressive modeling for image understanding and visual diffusion modeling for image generation. Janus [92] uses two different image representations, respectively for understanding and generation, to address the varying levels of information granularity required by the two tasks.

However, previous methods either involve complex designs or face challenges such as computational cost and optimization stability. To address the issues, our method leverages Multimodal Mixture-of-Experts and a token folding strategy to construct a fully autoregressive MLLM, en-

abling synergistic high-resolution image understanding and generation. Experiments show that SynerGen-VL achieves state-of-the-art performance on various benchmarks.

Encoder-free MLLMs. Most existing MLLMs adopt an encoder-based framework that integrates a separate image encoder like CLIP [62] into a pretrained LLM [1, 7, 12]. Meanwhile, some recent attempts have also begun to develop encoder-free MLLMs architecture due to their simplicity. Some works [8, 91, 96, 107] adopt VQ tokenizers [22] to represent images as discrete tokens. Others [10, 19, 51] use simple linear projection (i.e., patch embedding layer) to embed the images. In this paper, we build an encoder-free MLLM using discrete image representation through VQ tokenizers, which has stronger reconstruction ability to support both understanding and generation.

Token Folding and Unfolding. In language processing, early attempts like Funnel Transformer [17] and DataMUX [57] propose the downsample-upsample paradigm, i.e. compress the token length in intermediate Transformer layers, to process long sequences efficiently. MegaByte [102] segments sequences into patches, and then uses a local sub-model within patches and a global model between patches. HRED [56] uses a lower-frequency model to process input sub-sequences without global context, and decodes outputs at the original data frequency. Block Transformer [28] introduces a global-to-local structure to optimize the inference efficiency of autoregressive LLMs. In this paper, we adopt the token folding and unfolding mechanism to support high-resolution image understanding and generation. Since current visual tokenizers generate very long visual token sequences for high-resolution images, which is unsuitable to LLMs, we fold the visual token sequences before LLM modeling, and decode them back into the original local token sequences for image generation.

### 3. SynerGen-VL

##### 3.1. Architecture

SynerGen-VL is a unified MLLM with synergistic image understanding and generation capabilities. Fig. 3 shows an overview of SynerGen-VL. Similar to previous work [8, 45, 91], SynerGen-VL requires no externalized image generation models or additionally pretrained semantic encoders. It uses a single LLM with the unified next-token prediction objective for both tasks. Specifically, the input images and text are represented as discrete tokens by their corresponding tokenizers. The input multimodal token sequence consists of both image and text tokens, which always starts with a special token <s> and ends with another special token </s>. Special tokens <boi> and <eoi> are inserted before and after each image to indicate the beginning and end

[Figure 20]

123 234 456 567 987 876 765 543 112 223 334 445 211 322 433 544

[Figure 21]

VQ Tokenizer

Token Fold

<user> Describe the image <assistant> A cat that ...

VQ Tokenizer Token Fold MLP

[Figure 22]

Discrete Image Tokens

Token Embeddings

Text Tokenizer

MLP

Text Tokenizer

…

<BOI>

Multimodal Large Language Model

…

Multimodal Large Language Model

Text Head

Multi-head Self-attention

(b) Image Understanding (Autoregressive Text Prediction)

N×

A cat that ...

Text FFN Vision FFN

A cat with a yellow flower on its head and a golden necklace

Token Fold Text Tokenizer MLP

<BOI>

Token Unfold

Multimodal Large Language Model

Text Head

Token Unfold

[Figure 23]

123 234 987 876 456 567 765 543 112 223 211 322 334 445 433 544

VQ Decoder

(c) Image Generation (Autoregressive Image Prediction)

(a) Next Token Prediction Training

Figure 3. Overview of the proposed SynerGen-VL. The image and text are represented as discrete tokens, and modeled with a single LLM and unified next-token prediction paradigm. Text and vision expert FFNs are introduced to incorporate visual capabilities into the pretrained LLM. To support processing high-resolution images, the input image token sequence is folded to reduce its length, and unfolded by a shallow autoregressive Transformer head to generate images.

of the image, respectively. The multimodal token sequence is processed with a causal Transformer [86] initialized from a pretrained LLM. The image and text output tokens are predicted autoregressively, then the image output tokens can be decoded into pixels with the pretrained VQ decoder.

Input Embedding with Visual Token Folding. Existing discrete VQ-based image tokenizers require low feature downsample ratios to ensure reconstruction with fine details. This leads to long visual token sequences, limiting the use of high-resolution images in LLMs for detail-rich image understanding such as OCR-related tasks. To address this issue, we employ Token Folding to increase the feature downsampling ratio within the MLLM. Specifically, given an image I ∈ RH×W×3 (e.g., H = W = 512), an off-theshelf pretrained discrete image tokenizer is used to encode the image into a 2D grid of discrete tokens with shape h×w, where h = H/p and w = W/p. Here, p is the tokenizer’s downsampling ratio (e.g., p = 8). The visual token embeddings are then obtained from a learnable look-up table, and a learnable positional embedding PE is added to each token embedding to preserve spatial prior. Similar to Pixel Shuffle, token embeddings are folded by concatenating every m × n token patch into one single visual token (m = 2 and n = 8 by default). Here, each folded token patch can

be rectangular, following the latest practice of MLLMs for perception [88]. As shown in Fig. 3 (a), this results in an additional downsampling ratio of m × n, greatly compressing the token sequence for MLLM. For example, for an image of size 512×512, the original Emu3 [91] tokenizer produces visual 4096 tokens, while SynerGen-VL uses only 256 tokens to represent it in MLLM with a token folding ratio of 2 × 8.

After Token Folding, an MLP is applied to each folded image patch embedding to align its feature dimension with the LLM’s input dimension, yielding the final visual input features xV ∈ R(mh··wn)×d. The whole image embedding process can be formulated as:

###### xV = MLP(TokenFold(TokenEmbed(I) + PE)). (1)

For text input, we employ the built-in word tokenizer and the text token embedding look-up table of the pretrained LLM to encode it into text embeddings xT.

The visual token embeddings xV are concatenated with the text token embeddings xT and learnable special token embeddings (i.e., <s>, </s>, <boi>, <eoi>), according to the input order to form the final multimodal inputs into the MLLM.

###### Incorporating Visual Capabilities with Multimodal

Mixture-of-Experts (MMoEs). To avoid substantial tuning of the pretrained LLM while incorporating visual capabilities into it, we introduce additional parameters to each LLM’s Feed-Forward Network (FFN) layer as vision experts dedicated to image representation. Specifically, the FFN output of the i-th token is altered to

FFN-MMoE(xi) =

FFNV (xi), if xi is visual, FFNT(xi), if xi is textual,

(2)

where FFNT denotes the original FFN in the pretrained LLM for text tokens, and FFNV denotes the vision expert FFN, which shares the same architecture as FFNT and is initialized from the corresponding pretrained text FFN.

Instead of tuning the entire pretrained LLM, we perform a two-stage alignment pretraining on the vision expert FFNs with mixed image understanding and generation data. By aligning the visual representations with the representation space of the pretrained LLM, we minimize the impact of the LLM’s pretrained knowledge, ensuring the general perception and generalization capabilities. We introduce the two-stage alignment pretraining in Sec. 3.2.

Image Generation with Visual Token Unfolding. As the token-folding operation reduces the number of visual tokens at the input side of MLLM, the visual tokens at the output must be unfolded to generate images. We leverage a small causal transformer as the image generation head. Such head shares the same micro-architecture as the LLM but with fewer layers (e.g., 4) and has its own image token embedding and position embedding look-up tables, accompanied by an output classifier for VQ tokens.

For the i-th folded image patch, hi is the corresponding output embedding generated by the MLLM. To predict the j-th discrete token id vij in this image patch, its probability distribution is formulated autoregressively as:

p(vij|vi<j,hi) = Softmax(fθ(vi<j,hi)), (3)

where fθ represents the causal Transformer head with parameters θ, vi<j denotes all generated VQ token ids before the i-th visual tokens in this image patch. After generating VQ token ids for all image patches, we concatenate them to obtain the complete sequence of VQ token ids with the shape of h × w for image pixel decoding.

##### 3.2. Training

Training Objective. The overall training objective of SynerGen-VL consists of two main components: text token prediction and image token prediction. Both modalities employ the same next-token prediction objective, formulated as

###### L = −

i∈T

log p(ˆxiT = xiT|x<i) − λ

i∈V

log p(ˆxiV = xiV |x<i),

(4)

Task #Sam. Datasets Gen. 667M

LAION-Aesthetics [67], Megalith [52], SAM [33],

- S.1 Objects365 [69], ImageNet-1k [18], Und. 667M Laion-En [67], COYO [6], SAM [33]

Gen. 170M

LAION-Aesthetics [67], Megalith [52], Objects365 [69],

Unsplash [85], Dalle-3-HQ [3], JourneyDB [74],

Internal Dataset Captioning: Laion-En [67], Laion-Zh [67], COYO [6], GRIT [60], COCO [40], TextCaps [71] Detection: Objects365 [69], GRIT [60], All-Seeing [90] OCR (large): Wukong-OCR [26], LaionCOCO-OCR [68], Common Crawl PDF

OCR (small): MMC-Inst [41], LSVT [79], ST-VQA [5], RCTW-17 [70], ReCTs [106], ArT [13], SynthDoG [32], ChartQA [53], CTW [104], DocVQA [15], TextOCR [73],

- S.2 Und. 170M

COCO-Text [87], PlotQA [55], InfoVQA [54]

Table 1. Summary of datasets used in Visual Alignment Pretraining. “S.1” and “S.2” denote the first and second stage. “Gen.” and “Und.” denote the image generation and understanding task. “#Sam.” denotes the number of total samples seen during training of each task at each stage. Note that all data used for image understanding in the second stage is also used in InternVL-1.5 [11].

where T ,V are the index sets indicating all text and image tokens in the multimodal sequence, respectively, xˆiT and xˆiV denote the predicted text and image token at position i. The final loss objective is the weighted sum of the text and image losses, with a hyperparameter λ to balance the relative loss weight between image understanding and image generation.

Visual Alignment Pretraining. To preserve the LLM’s pretrained knowledge, we conduct a progressive two-stage alignment pretraining strategy, with both stages utilizing a mixture of data for image understanding and generation. The detailed dataset composition is shown in Tab. 1.

The first stage aims to bridge visual elements with concepts in the representation space of the pretrained LLM, thereby obtaining basic semantic understanding and image generation abilities. To avoid interfering with the LLM’s pretrained knowledge, we freeze the parameters of the LLM components and only train the image-specific parameters (i.e., the visual token embedding and projection layers, the vision experts in MLLM, and the visual token unfolding head). For image understanding, we use the large-scale noisy image-text pair data LAION-En [67] and Coyo-700M [6] for basic concept learning, while incorporating a portion of synthesized captions from samples in [6, 33, 67] generated by InternVL-7B to achieve better semantic alignments. For image generation, apart from the large-scale noisy LAION-Aesthetics data [67], we follow [9, 92] to accelerate the pixel dependency learning with [18] and improve the learning of object concepts and relations through [33, 52, 69]. To distinguish the image understanding and generation tasks, we use the prompt of “Provide a one-sentence caption for the image” for understanding data, while adding “Generate an image of:” before the text prompts of the generation data.

In the second stage, we further integrate visual capabili-

Model #A-Param POPE MMB MMVet MMMU MME MME-P MathVista SEED-I OCRBench Understanding Only Encoder-based LLaVA-1.5 [43] 7B 85.9 64.3 31.1 35.4 - 1512 - 58.6 Mini-Gemini-2B [38] 3.5B - 59.8 31.1 31.7 1653 - 29.4 - DeepSeek-VL-1.3B [48] 2B 87.6 64.6 34.8 32.2 1532 - 31.1 66.7 409 PaliGemma-3B [4] 2.9B 87.0 71.0 33.1 34.9 1686 - 28.7 69.6 614 MiniCPM-V2 [100] 2.8B - 69.1 41.0 38.2 1809 - 38.7 67.1 605 InternVL-1.5 [11] 2B - 70.9 39.3 34.6 1902 - 41.1 69.8 654 Qwen2-VL [88] 2B - 74.9 49.5 41.1 1872 - 43.0 - 809 Encoder-free Fuyu-8B (HD) [2] 8B - 10.7 21.4 - - - - - EVE-7B [19] 7B 83.6 49.5 25.6 32.3 1483 - 25.2 61.3 327 Mono-InternVL [51] 1.8B - 65.5 40.1 33.7 1875 - 45.7 67.4 767 Understanding & Generation Encoder-based

Emu [78] 14B - - 36.3 - - - - - Emu2 [76] 37B - 63.6 48.5 34.1 - 1345 - 62.8 SEED-X [24] 17B 84.2 75.4 - 35.6 - 1436 - - LWM [45] 7B 75.2 - 9.6 - - - - - DreamLLM [20] 7B - 58.2 36.6 - - - - - Janus [92] 1.3B 87.0 69.4 34.3 30.5 - 1338 - 63.7 Encoder-free

Chameleon [8] 7B - - 8.3 22.4 - - - - Show-o [96] 1.3B 84.5 - - 27.4 - 1233 - - VILA-U [94] 7B 85.8 - 33.5 - - 1402 - 59.0 Emu3-Chat [91] 8B 85.2 58.5 37.2 31.6 - - - 68.2 687 SynerGen-VL (Ours) 2.4B 85.3 53.7 34.5 34.2 1837 1381 42.7 62.0 721

- Table 2. Results on general MLLM benchmarks. Our model with 2.4B parameters achieves competitive image understanding performance compared with significantly larger encoder-free unified MLLMs such as Emu3-Chat-8B [91].

ties into the pretrained LLM by training the image-specific parameters and the self-attention layers with high-quality mixed data. Specifically, for image understanding, we follow [51] to sample from the high-quality pretraining data of InternVL-1.5 [11], resulting in 170 million samples with task-related prompts. For image generation, we select the data with high aesthetic scores and caption quality, resulting in 20 million samples from [3, 52, 69, 74, 85] as well as 5 million high-quality internal data.

Throughout both stages, SynerGen-VL is trained simultaneously for image understanding and generation. To enhance the image understanding capability and take advantage of SynerGen-VL’s ability to process high-resolution images, we implement a dynamic resolution strategy for understanding tasks following InternVL-1.5 [11] in the second stage and set the maximum number of image tiles to 6.

Joint Instruction Tuning. During the instruction tuning stage, we unfreeze all the model parameters. For image understanding, we adopt the dataset from InternVL-1.5, including around 5M bilingual instructions for supervised learning, covering various tasks such as visual question answering, multimodal dialogue, mathematics, knowledge, etc. We also increase the maximum number of image tiles to

12 to handle high-resolution images. For image generation, we solely use the 10M internal dataset to further enhance the image generation quality.

### 4. Experiments

##### 4.1. Implementation Details

SynerGen-VL is built upon InternLM2-1.8B [7], using the same text tokenizer and conversation format. The discrete image tokenizer originates from Emu3 [91], characterized by a codebook size of 32,768 and a spatial downsampling rate of 8. The input image is resized to 512 × 512. For image generation data, the short edge of the image is resized to 512 and the long edge is cropped to 512. The total number of model parameters is 3.6B, of which the number of activation parameters is 2.4B. In the pretraining phase, the global batch size for image understanding and generation tasks is 6988 for stage 1 and 5090 for stage 2, respectively. The loss weight hyperparameter λ is set to 2. The instruction tuning phase is trained for 3 epochs in total. Following previous works [75, 92], classifier-free guidance (CFG) strategy is also implemented for image generation. During training, we randomly replace the original user caption prompt with “Here is a random image <UNCOND>:” with a probability

Method #A-Param TextVQA SQA-I GQA DocVQA AI2D ChartQA InfoVQA Understanding Only Encoder-based

MobileVLM-V2 [14] 1.7B 52.1 66.7 59.3 - - - Mini-Gemini-2B [39] 3.5B 56.2 - - 34.2 - - PaliGemma-3B [4] 2.9B 68.1 - - - 68.3 MiniCPM-V2 [100] 2.8B 74.1 - - 71.9 - - InternVL-1.5 [11] 2B 70.5 84.9 61.6 85.0 69.8 74.8 55.4

Encoder-free EVE-7B [19] 7B 51.9 63.0 60.8 - - - Mono-InternVL [51] 1.8B 72.6 93.6 59.5 80.0 68.6 73.7 43.0

Understanding & Generation Encoder-based

Emu2 [76] 37B 66.6 - 65.1 - - - LWM [45] 7B 18.8 47.7 44.8 - - - DreamLLM [20] 7B 41.8 - - - - - MM-Interleaved [82] 13B 61.0 - 60.5 - - - Janus [92] 1.3B - - 59.1 - - - Encoder-free

Chameleon⋄ [8] 7B 4.8 47.2 - 1.5 46.0 2.9 5.0 Show-o [96] 1.3B - - 61.0 - - - VILA-U [94] 7B 60.8 - 60.8 - - - Emu3-Chat [91] 8B 64.7 89.2 60.3 76.3 70.0 68.6 43.8 SynerGen-VL (Ours) 2.4B 67.5 92.6 59.7 76.6 60.8 73.4 37.5

- Table 3. Comparison with existing MLLMs on visual question answering benchmarks. #A-Params denotes the number of activated parameters during inference. ⋄Some results of Chameleon are sourced from [51].

of 10%, where <UNCOND> is a learnable special token embedding. During inference, the logit of each unfolded image token is calculated as: lg = lu + s(lc − lu), where lc,lu are the conditional and unconditional logits, respectively. s is the CFG-scale with default number of 7.5.

Due to the limited space, please refer to the supplementary material for more detailed training configurations.

- 4.2. Image Understanding

Results. Evaluation results are shown in Tab. 2 and Tab. 3. Compared with existing encoder-free unified MLLMs, our SynerGen-VL with 2.4B parameters surpasses previous methods (especially for encoder-free unified MLLMs) with comparable parameter sizes while achieving comparable performance to models with significantly larger parameter sizes, showcasting its competitive image understanding capability. Notably, on image understanding benchmarks requiring high-resolution detailed image comprehension, such as OCRBench, TextVQA, DocVQA, and ChartQA, our SynerGen-VL achieves results superior to much larger encoder-free MLLMs such as Emu3-Chat-8B [91], highlighting its advantages with high-resolution image processing capabilities. Moreover, as a encoder-free unified MLLM, SynerGen-VL also obtains image understanding performance competitive to encoder-based understandingonly MLLMs such as LLaVA-1.5 [42], while surpassing larger encoder-free task-specific MLLMs such as EVE7B [19] and Fuyu-8B (HD) [2], demonstrating its great potential of unifying image understanding and generation.

Evaluation Benchmarks. To evaluate the general multimodal understanding capabilities of SynerGen-VL, we compare with image understanding models as well as unified image understanding and generation models on 8 comprehensive multimodal benchmarks including MMBenchEN test [46], MMVet [103], MMMU val [105], MME [23], MathVista test-mini [50], POPE [36], SEED-Image [34], and OCRBench [47]. These general benchmarks covers assessment of various capabilities for visual question answering, document and chart interpretation, and other complex visual scenarios. We further evaluate model’s VQA performances on 7 widely-adopted benchmarks including TextVQA val [72], ScienceQA test [49], GQA test-dev [29], DocVQA test [15], AI2D test [31], ChartQA test [53], and InfographicsVQA test [54]. Part of the results are evaluated using VLMEvalKit [21] or sourced from the OpenCompass leaderboard [16].

##### 4.3. Image Generation

Evaluation Benchmarks. We use the MSCOCO-30K [40], MJHQ-30K [35], and GenEval [25] benchmarks to evaluate our model’s image generation capabilities. For MSCOCO30K and MJHQ-30K, we generate 30k images and compare

Method # A-Param Single Obj. Two Obj. Counting Colors Position Color Attri. Overall↑ Generation Only

LlamaGen [75] 0.8B 0.71 0.34 0.21 0.58 0.07 0.04 0.32 LDM [65] 1.4B 0.92 0.29 0.23 0.70 0.02 0.05 0.37 SDv1.5 [65] 0.9B 0.97 0.38 0.35 0.76 0.04 0.06 0.43 SDXL [61] 2.6B 0.98 0.74 0.39 0.85 0.15 0.23 0.55 PixArt-α [9] 0.6B 0.98 0.50 0.44 0.80 0.08 0.07 0.48 DALL-E 2 [64] 6.5B 0.94 0.66 0.49 0.77 0.10 0.19 0.52

###### Understanding & Generation

SEED-X† [24] 17B 0.97 0.58 0.26 0.80 0.19 0.14 0.49 Show-o [96] 1.3B 0.95 0.52 0.49 0.82 0.11 0.28 0.53 LWM [45] 7B 0.93 0.41 0.46 0.79 0.09 0.15 0.47 Chameleon [8] 34B - - - - - - 0.39 Emu3-Gen [91] 8B 0.98 0.71 0.34 0.81 0.17 0.21 0.54 Janus [92] 1.3B 0.97 0.68 0.30 0.84 0.46 0.42 0.61 SynerGen-VL (Ours) 2.4B 0.99 0.71 0.34 0.87 0.37 0.37 0.61

- Table 4. Evaluation of text-to-image generation on GenEval [25] benchmark. #A-Params denotes the number of activated parameters during inference. † indicates models with external pretrained diffusion model. Obj.: Object. Attri.: Attribution.

Model #A-Param MS-COCO↓ MJHQ↓ Generation Only

DALL-E [63] 12B 27.50 LDM [65] 1.4B 12.64 GLIDE [58] 5B 12.24 DALL-E 2 [64] 6.5B 10.39 RAPHAEL [97] 3B 6.61 Imagen [66] 34B 7.27 SDv1.5 [65] 0.9B 9.62 SDXL [61] 0.9B 7.38 8.76 PixArt-α [9] 0.6B 7.32 6.14

Understanding & Generation

NExT-GPT [93] 13B 11.18 SEED-X [24] 17B 14.99 Show-o [96] 1.3B 9.24 15.18 LWM [45] 7B 12.68 17.77 VILA-U [94] 7B - 7.69 Emu3-Gen [91] 8B 19.3 Janus [92] 1.3B 8.53 10.10 SynerGen-VL (Ours) 2.4B 7.65 6.10

- Table 5. Image generation results on MSCOCO-30K [40] and MJHQ-30K [35] datasets. FID [27] is reported. #A-Param denotes the number of activated parameters during inference.

scores. Compared with previous unified MLLMs for both image understanding and generation, SynerGen-VL can achieve competitive performance without using an external diffusion model. In particular, compared with Emu3 [91] that use the same tokenizer, our method has a significant improvement in FID scores with less model parameters. We believe this is because the usage of vision experts simplifies the training difficulty. We also evaluate our model’s ability to generate high-quality aesthetic images on MJHQ [35], as shown in the Tab. 5. Compared with previous generationonly methods, SynerGen-VL achieves competitive generation performance. These results validate that our method applies to both natural images and synthetic aesthetic images.

Results on GenEval. Following previous studies, we evaluate our model’s text-to-image generation capabilities on the GenEval benchmark [25] from six dimensions: “single object”, “two objects”, “number”, “color”, “position”, and “color attribution”. Our model achieve competitive overall scores with previous generation-only models of similar sizes. SynerGen-VL performs comparably to Janus [92], which uses independent encoders for perception and generation, demonstrating the effectiveness of using vision experts in our approach. Compared to Emu3 [91], our model achieves better overall performance with fewer parameters.

them with the reference images and use Fr´echet Inception Distance (FID) [27] to assess the overall generation quality. For GenEval, we generate four images for each prompt and utilize its official framework to assess our model’s objectlevel image-text alignment.

### 5. Ablation Study

In this section, we ablate the effectiveness of the two important techniques of SynerGen-VL, i.e., token folding and progressive alignment pre-training with MMoEs. In this ablation study, we use Qwen2-0.5B-Instruct [98] as the initialized LLM and image size 256 unless otherwise specified.

Results on MSCOCO and MJHQ. Tab. 5 shows the zeroshot FID of our model on MSCOCO 30K [40]. Compared with previous generation-only models such as GLIDE [58] and DALL-E 2[64], our method can achieve better FID

##### 5.1. Effectiveness of Token Folding

To verify the effectiveness of token folding on highresolution image understanding, we compare SynerGen-VL with the baseline version without token folding and the dynamic resolution strategy on image understanding tasks. Specifically, the baseline model directly use the tokenized sequence as the input image sequence without token folding, where the input image size is 256 × 256 and the tokenized sequence length is 1024. Meanwhile, for the model with token folding, we follow InternVL-1.5 [11] to implement the dynamic resolution strategy to provide highresolution input images. For fair comparison, we use a token folding ratio of 2×4 and control the maximum number of dynamic image patches so that the average length of image token sequence after token folding is also 1024.

We train the models with a subset of stage 2 (S.2) understanding data, and evaluate the pre-trained models on VQA benchmarks. Results are shown in Tab. 6. On datasets requiring precise understanding of detailed image information such as TextVQA, DocVQA, ChartVQA, and InfographicVQA, the model with token folding achieves significantly better results, demonstrating its advantages of highresolution image understanding.

Model TextVQA GQA DocVQA AI2D ChartQA InfoVQA

w/o token folding 18.7 45.3 14.7 42.0 20.9 18.7 w/ token folding 35.0 45.1 36.7 42.1 49.7 21.1

- Table 6. Comparison between models with and without tokenfolding on VQA benchmarks. The model with token folding demonstrates significant performance improvements with the same image token sequence length.

##### 5.2. Effectiveness of the Progressive Alignment Pretraining with MMoEs

We ablate our proposed visual alignment pre-training strategy on various benchmarks, including visual question answering (VQA), natural language processing (NLP) and text-to-image (T2I) generation, as shown in Tab. 7. To ensure fair comparison, neither token folding nor dynamic resolution strategies are employed. For experimental efficiency, only 1/6 of the training data is used for both stages.

The results show that our progressive strategy matches or exceeds the fully parameter-trained strategy on VQA benchmarks and significantly outperforms it on text-toimage generation benchmarks. Meanwhile, on NLP benchmarks, our model with progressive alignment pretraining delivers results much closer to the pre-trained LLM (Qwen2-0.5B-Instruct) compared with the fully parametertrained model. This validates that our approach effectively preserves the original knowledge in the pre-trained LLM while learning robust visual representations. Furthermore, the two-stage training strategy outperforms training solely

with stage 1 or stage 2, particularly on VQA and text-toimage generation benchmarks. This underscores the importance of learning basic visual concepts and pixel dependencies from large-scale noisy data, as well as enhancing image-text alignment and image aesthetics with highquality data.

##### 5.3. Analysis of Relationship Between Image Generation and Understanding

We provide visualization and analysis to understand the relationship between image generation and understanding tasks, i.e. how the two tasks might be related in terms of their processing or feature utilization.

Similarity between Gen and Und features

1.0

0.8

AverageSimilarity

0.6

0.4

0.2

0.0

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24

Layer

Figure 4. Cosine similarity of visual features between generation and understanding tasks across different layers. The representations of the image understanding and generation tasks are similar in shallow layers but disentagle in deeper layers.

Image Feature Similarity. We first analyze whether the two tasks share similar representations. We use the same input image paired with text instructions of generation or understanding, and compute the cosine similarity between visual features of the two tasks at each layer. As shown in Fig. 4, the two features are nearly identical (0.999) at shallower layers, but the similarity decreases as layers deepen. It finally reaches a near-zero value (0.035) at the last layer, suggesting that the two representations are disentangled. This observation implies that while image generation and understanding may share foundational visual representations in the early stages, they develop task-specific representations based on different instructions of image generation and understanding at deeper layers.

Attention Map Visualization. In Fig. 5, we further investigate whether the two tasks have similar attention map patterns. We discover that in both tasks, locality is present at early layers, where visual tokens only attend to its nearby tokens (i.e. near the diagonal). Text tokens and images have few interactions with each other. As layers deepen, longer dependency is observed, and finally global interactions are achieved at the last layer. Text and image also interact more often than at shallower layers. The attention weight also dis-

VQA Benchmarks ↑ NLP Benchmarks ↑ T2I Benchmark ↓

Stage Strategy

TextVQA GQA DocVQA AI2D ChartQA InfoVQA MMLU CMMLU AGIEVAL MATH MSCOCO

Baseline (Qwen2-0.5B) - - - - - - 42.3 51.4 29.3 12.1 S.1 + S.2 Full 14.3 42.9 11.3 24.7 12.4 12.6 23.1 23.0 8.1 0.9 30.7

- S.1 only Progressive 0.1 13.0 0.2 0.3 0.0 0.0 42.3 51.4 29.3 12.1 28.3

- S.2 only Progressive 8.7 36.9 8.6 40.9 11.7 16.2 37.6 45.3 28.9 7.2 34.9 S.1 + S.2 Progressive 13.2 41.2 11.4 41.9 12.8 17.0 39.3 48.2 26.2 8.9 20.2

- Table 7. Zero-shot performance of different pre-training strategies. “S.1” and “S.2” denote the first and second pre-training stage. “Full” and “Progressive” denote the full parameter tuning and our progressive tuning strategy with MMoEs, respectively. FID [27] is reported for text-to-image generation (T2I) on MSCOCO [40].

plays a periodicity nature, such as in Layer 4. Visualization in the input image suggests that the period is the number of tokens in each row, validating the locality. When comparing the attention maps in the two tasks, we observe that locality is more obvious in generation than in understanding at the same layer. This can be explained that local details are required to generate a spatially consistent and semantically coherent image, while understanding the whole image requires global context.

### 6. Conclusion

In this paper, we introduce SynerGen-VL, an encoder-free MLLM that effectively unifies image understanding and generation within a simplified framework. By leveraging token folding and vision experts, SynerGen-VL addresses the complexities of high-resolution image processing while maintaining the integrity of pretrained language model knowledge. Our approach eliminates dependencies on external diffusion models or additional semantic encoder pretraining, achieving competitive performance across various benchmarks with a relatively small parameter size. The experiment results underscore SynerGen-VL’s potential as a scalable and efficient solution for future unified MLLMs.

Acknowledgments This work is supported by the National Key R&D Program of China (NO. 2022ZD0161300), by the National Natural Science Foundation of China (62376134).

Layer 4 Layer 12 Layer 20 Layer 24

Understanding

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

46

Text: <system> ... <user> <BOI><IMG><EOI> Provide a one-sentence caption for the image: <assistant> A crowd of people get ready to board a bus in the city.

256

29

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Generation

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

66

Text: <system> ... <user> Generate an image of “A crowd of people get ready to board a bus in the city”. <assistant> <BOI><IMG><EOI>

256

2

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Figure 5. Attention map visualization of understanding and generation tasks. In the second and fourth rows, we visualize a query token (red) and its attended tokens (blue) in the input image. Each token corresponds to a horizontal rectangular area in the original image due to the 2 × 4 token folding. Darker blue indicates larger attention weights.

### References

- [1] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 3
- [2] Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sa˘gnak Ta¸sırlar. Introducing our multimodal models, 2023. 6, 7
- [3] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, Wesam Manassra, Prafulla Dhariwal, Casey Chu, Yunxin Jiao, and Aditya Ramesh. Improving image generation with better captions. 2023. 5, 6
- [4] Lucas Beyer, Andreas Steiner, Andr´e Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024. 6, 7
- [5] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Marc¸al Rusinol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas. Scene text visual question answering. In ICCV, pages 4291–4301, 2019. 5
- [6] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/ kakaobrain/coyo-dataset, 2022. 5
- [7] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al. Internlm2 technical report. arXiv preprint arXiv:2403.17297, 2024. 1, 3, 6
- [8] ChameleonTeam. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818,

2024. 1, 2, 3, 6, 7, 8

- [9] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 5, 8
- [10] Yangyi Chen, Xingyao Wang, Hao Peng, and Heng Ji. A single transformer for scalable vision-language modeling. arXiv preprint arXiv:2407.06438, 2024. 3
- [11] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with opensource suites. arXiv:2404.16821, 2024. 1, 5, 6, 7, 9
- [12] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang,

- Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, 2023. 3
- [13] Chee Kheng Chng, Yuliang Liu, Yipeng Sun, Chun Chet Ng, Canjie Luo, Zihan Ni, ChuanMing Fang, Shuaitao Zhang, Junyu Han, Errui Ding, et al. Icdar2019 robust reading challenge on arbitrary-shaped text-rrc-art. In ICDAR, pages 1571–1576, 2019. 5
- [14] Xiangxiang Chu, Limeng Qiao, Xinyu Zhang, Shuang Xu, Fei Wei, Yang Yang, Xiaofei Sun, Yiming Hu, Xinyang Lin, Bo Zhang, et al. Mobilevlm v2: Faster and stronger baseline for vision language model. arXiv preprint arXiv:2402.03766, 2024. 7
- [15] Christopher Clark and Matt Gardner. Simple and effective multi-paragraph reading comprehension. In ACL, pages 845–855, 2018. 5, 7
- [16] Contributors. Opencompass: A universal evaluation platform for foundation models. https://github.com/ open-compass/opencompass, 2023. 7
- [17] Zihang Dai, Guokun Lai, Yiming Yang, and Quoc Le. Funnel-transformer: Filtering out sequential redundancy for efficient language processing. Advances in neural information processing systems, 33:4271–4282, 2020. 3
- [18] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, pages 248–255, 2009. 5
- [19] Haiwen Diao, Yufeng Cui, Xiaotong Li, Yueze Wang, Huchuan Lu, and Xinlong Wang. Unveiling encoder-free vision-language models. arXiv preprint arXiv:2406.11832,

2024. 3, 6, 7

- [20] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. Dreamllm: Synergistic multimodal comprehension and creation. In ICLR, 2024. 1, 6, 7
- [21] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, Dahua Lin, and Kai Chen. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models, 2024. 7
- [22] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 1, 2, 3
- [23] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. MME: A comprehensive evaluation benchmark for multimodal large language models. arXiv: 2306.13394, 2023. 7
- [24] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396,

2024. 1, 3, 6, 8

- [25] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. Advances in Neural Information Processing Systems, 36, 2024. 7, 8

- [26] Jiaxi Gu, Xiaojun Meng, Guansong Lu, Lu Hou, Niu Minzhe, Xiaodan Liang, Lewei Yao, Runhui Huang, Wei Zhang, Xin Jiang, et al. Wukong: A 100 million large-scale chinese cross-modal pre-training benchmark. NeurIPS, 35: 26418–26431, 2022. 5
- [27] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 8, 10
- [28] Namgyu Ho, Sangmin Bae, Taehyeon Kim, Hyunjik Jo, Yireun Kim, Tal Schuster, Adam Fisch, James Thorne, and Se-Young Yun. Block transformer: Global-to-local language modeling for fast inference. arXiv preprint arXiv:2406.02657, 2024. 3
- [29] Drew A. Hudson and Christopher D. Manning. GQA: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, pages 6700–6709,

2019. 7

- [30] Yang Jin, Kun Xu, Kun Xu, Liwei Chen, Chao Liao, Jianchao Tan, Yadong Mu, et al. Unified language-vision pretraining in llm with dynamic discrete visual tokenization. In International Conference on Learning Representations,

2024. 3

- [31] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV, pages 235–251, 2016. 7
- [32] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocrfree document understanding transformer. In ECCV, 2022. 5
- [33] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chlo´e Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross B. Girshick. Segment anything. arXiv: 2304.02643, 2023. 5
- [34] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv: 2307.16125, 2023. 7
- [35] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024. 7, 8
- [36] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In EMNLP, pages 292–305,

2023. 7

- [37] Yunxin Li, Shenyuan Jiang, Baotian Hu, Longyue Wang, Wanqi Zhong, Wenhan Luo, Lin Ma, and Min Zhang. Unimoe: Scaling unified multimodal llms with mixture of experts. arXiv preprint arXiv:2405.11273, 2024. 2
- [38] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv: 2403.18814, 2024. 3, 6

- [39] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814,

2024. 7

- [40] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. In ECCV, pages 740–755, 2014. 5, 7, 8, 10
- [41] Fuxiao Liu, Xiaoyang Wang, Wenlin Yao, Jianshu Chen, Kaiqiang Song, Sangwoo Cho, Yaser Yacoob, and Dong Yu. Mmc: Advancing multimodal chart understanding with large-scale instruction tuning. arXiv preprint arXiv:2311.10774, 2023. 5
- [42] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv: 2310.03744, 2023. 7
- [43] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023. 6
- [44] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 1
- [45] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with ringattention. arXiv preprint arXiv:2402.08268, 2024. 1, 3, 6, 7, 8
- [46] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? arXiv: 2307.06281, 2023. 7
- [47] Yuliang Liu, Zhang Li, Hongliang Li, Wenwen Yu, Mingxin Huang, Dezhi Peng, Mingyu Liu, Mingrui Chen, Chunyuan Li, Lianwen Jin, et al. On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895, 2023. 7
- [48] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, et al. Deepseek-vl: Towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024. 6
- [49] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In NeurIPS, 2022. 7
- [50] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv: 2310.02255, 2023. 7
- [51] Gen Luo, Xue Yang, Wenhan Dou, Zhaokai Wang, Jifeng Dai, Yu Qiao, and Xizhou Zhu. Mono-internvl: Pushing the boundaries of monolithic multimodal large language models with endogenous visual pre-training. arXiv preprint arXiv:2410.08202, 2024. 2, 3, 6, 7

- [52] madebyollin. Megalith-huggingface. https : //huggingface.co/datasets/madebyollin/ megalith-10m, 2024. 5, 6
- [53] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In ACL, pages 2263–2279, 2022. 5, 7
- [54] Minesh Mathew, Viraj Bagal, Rub`en Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In WACV, pages 1697–1706, 2022. 5, 7
- [55] Nitesh Methani, Pritha Ganguly, Mitesh M Khapra, and Pratyush Kumar. Plotqa: Reasoning over scientific plots. In WACV, pages 1527–1536, 2020. 5
- [56] Asier Mujika. Hierarchical attention encoder decoder. arXiv preprint arXiv:2306.01070, 2023. 3
- [57] Vishvak Murahari, Carlos Jimenez, Runzhe Yang, and Karthik Narasimhan. Datamux: Data multiplexing for neural networks. Advances in Neural Information Processing Systems, 35:17515–17527, 2022. 3
- [58] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 8
- [59] OpenAI. GPT-4 technical report. arXiv: 2303.08774, 2023. 1
- [60] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023. 5
- [61] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 8
- [62] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021. 3
- [63] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr, 2021. 8
- [64] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022. 8
- [65] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. 2022 ieee. In CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 8
- [66] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour,

- Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 8
- [67] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. NeurIPS, 35: 25278–25294, 2022. 5
- [68] Christoph Schuhmann, Andreas K¨opf, Richard Vencu, Theo Coombes, and Romain Beaumont. Laion coco: 600m synthetic captions from laion2b-en. https://laion.ai/blog/laion-coco/, 2022. 5
- [69] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In ICCV, pages 8430–8439, 2019. 5, 6
- [70] Baoguang Shi, Cong Yao, Minghui Liao, Mingkun Yang, Pei Xu, Linyan Cui, Serge Belongie, Shijian Lu, and Xiang Bai. Icdar2017 competition on reading chinese text in the wild (rctw-17). In ICDAR, pages 1429–1434, 2017. 5
- [71] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: A dataset for image captioning with reading comprehension. In ECCV, pages 742–758,

2020. 5

- [72] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA models that can read. In CVPR,

2019. 7

- [73] Amanpreet Singh, Guan Pang, Mandy Toh, Jing Huang, Wojciech Galuba, and Tal Hassner. Textocr: Towards largescale end-to-end reasoning for arbitrary-shaped scene text. In CVPR, pages 8802–8812, 2021. 5
- [74] Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, et al. Journeydb: A benchmark for generative image understanding. Advances in Neural Information Processing Systems, 36, 2024. 5, 6
- [75] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 1, 6, 8
- [76] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. arXiv: 2312.13286, 2023. 3, 6, 7
- [77] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. arXiv: 2307.05222, 2023. 3
- [78] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. In ICLR, 2024. 6
- [79] Yipeng Sun, Zihan Ni, Chee-Kheng Chng, Yuliang Liu, Canjie Luo, Chun Chet Ng, Junyu Han, Errui Ding, Jingtuo

- Liu, Dimosthenis Karatzas, et al. Icdar 2019 competition on large-scale street view text with partial labeling-rrc-lsvt. In ICDAR, pages 1557–1562, 2019. 5
- [80] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv: 2312.11805, 2023. 1
- [81] Changyao Tian, Xizhou Zhu, Yuwen Xiong, Weiyun Wang, Zhe Chen, Wenhai Wang, Yuntao Chen, Lewei Lu, Tong Lu, Jie Zhou, et al. Mm-interleaved: Interleaved image-text generative modeling via multi-modal feature synchronizer. arXiv:2401.10208, 2024. 1
- [82] Changyao Tian, Xizhou Zhu, Yuwen Xiong, Weiyun Wang, Zhe Chen, Wenhai Wang, Yuntao Chen, Lewei Lu, Tong Lu, Jie Zhou, et al. Mm-interleaved: Interleaved image-text generative modeling via multi-modal feature synchronizer. arXiv preprint arXiv:2401.10208, 2024. 7
- [83] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv preprint arXiv:2404.02905, 2024. 1
- [84] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1
- [85] Unsplash. Unsplash Dataset. https://unsplash. com/data, 2020. Online; accessed March-2020. 5, 6
- [86] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, pages 5998–6008, 2017. 4
- [87] Andreas Veit, Tomas Matera, Lukas Neumann, Jiri Matas, and Serge Belongie. Coco-text: Dataset and benchmark for text detection and recognition in natural images. arXiv preprint arXiv:1601.07140, 2016. 5
- [88] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 4, 6
- [89] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023. 2
- [90] Weiyun Wang, Min Shi, Qingyun Li, Wenhai Wang, Zhenhang Huang, Linjie Xing, Zhe Chen, Hao Li, Xizhou Zhu, Zhiguo Cao, et al. The all-seeing project: Towards panoptic visual recognition and understanding of the open world. In ICLR, 2024. 5
- [91] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, Yingli Zhao, Yulong Ao, Xuebin Min, Tao Li, Boya Wu, Bo Zhao, Bowen Zhang, Liangdong Wang, Guang Liu, Zheqi He, Xi Yang, Jingjing Liu, Yonghua Lin, Tiejun Huang, and Zhongyuan Wang. Emu3:

Next-token prediction is all you need. arXiv: 2409.18869,

2024. 1, 2, 3, 4, 6, 7, 8

- [92] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848, 2024. 1, 3, 5, 6, 7, 8
- [93] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and TatSeng Chua. Next-gpt: Any-to-any multimodal llm. arXiv: 2309.05519, 2023. 3, 8
- [94] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024. 1, 3, 6, 7, 8
- [95] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024. 1
- [96] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528,

2024. 1, 3, 6, 7, 8

- [97] Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, and Ping Luo. Raphael: Text-toimage generation via large mixture of diffusion paths. Advances in Neural Information Processing Systems, 2024. 8
- [98] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 8
- [99] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). arXiv: 2309.17421, 9, 2023. 1
- [100] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 6, 7
- [101] Lili Yu, Bowen Shi, Ramakanth Pasunuru, Benjamin Muller, Olga Golovneva, Tianlu Wang, Arun Babu, Binh Tang, Brian Karrer, Shelly Sheynin, et al. Scaling autoregressive multi-modal models: Pretraining and instruction tuning. arXiv preprint arXiv:2309.02591, 2(3), 2023. 3
- [102] Lili Yu, D´aniel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer, and Mike Lewis. Megabyte: Predicting million-byte sequences with multiscale transformers. Advances in Neural Information Processing Systems, 36: 78808–78823, 2023. 3
- [103] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv: 2308.02490, 2023. 7
- [104] Tai-Ling Yuan, Zhe Zhu, Kun Xu, Cheng-Jun Li, Tai-Jiang Mu, and Shi-Min Hu. A large chinese text dataset in the

- wild. Journal of Computer Science and Technology, 34: 509–521, 2019. 5
- [105] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multidiscipline multimodal understanding and reasoning benchmark for expert agi. arXiv: 2311.16502, 2023. 7
- [106] Rui Zhang, Yongsheng Zhou, Qianyi Jiang, Qi Song, Nan Li, Kai Zhou, Lei Wang, Dong Wang, Minghui Liao, Mingkun Yang, et al. Icdar 2019 robust reading challenge on reading chinese text on signboard. In ICDAR, pages 1577–1581, 2019. 5
- [107] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024. 1, 3
- [108] Jinguo Zhu, Xiaohan Ding, Yixiao Ge, Yuying Ge, Sijie Zhao, Hengshuang Zhao, Xiaohua Wang, and Ying Shan. Vl-gpt: A generative pre-trained transformer for vision and language understanding and generation. arXiv preprint arXiv:2312.09251, 2023. 1

#### A. Detailed Training Configurations More detailed hyper-parameters used in the training stages are listed in Tab. 8.

Alignment Pre-training Instruction S.1 S.2 Tuning

Configuration

Maximum number of image tiles 1 6 12 LLM sequence length 4, 096 8, 192 16, 384 Use thumbnail ✗ ✓ ✓ Global batch size (per-task) 6, 988 5, 090 1, 760 Peak learning rate 1e−4 5e−5 5e−5 Learning rate schedule constant with warm-up cosine decay cosine decay Weight decay 0.05 0.05 0.01 Training steps 95k 35k 12k Warm-up steps 200 Optimizer AdamW Optimizer hyperparameters β1 = 0.9, β2 = 0.95, eps = 1e−8 Gradient accumulation 1 Numerical precision bfloat16

Table 8. Hyper-parameters used in the alignment pre-training and instruction tuning stages.

#### B. Visualization For qualitative evaluation, we visualize examples for image understanding and image generation as follows.

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

A sprawling urban landscape with numerous skyscrapers, highlighting the dense architecture of the city. Tall buildings dominate the skyline, surrounded by smaller structures and patches of greenery.

A stunning river meandering through a valley, framed by a majestic mountain range, combining vibrant yellows and oranges with precisionist lines, blending villagecore charm and east-west artistic fusion, creating a hyper-realistic yet dreamlike naturecore aesthetic.

An impressionist manga art style, blending influences from Paul Hedler and Makoto Shinkai. It features vibrant, warm colors and dynamic brushstrokes, capturing a lively urban scene with a focus on lighting and atmosphere.

A beautiful woman dressed in a colorful floral top, in the style of victor enrich, patchwork patterns, daria endresen, bold color choices, asymmetric designs, sandro botticelli, 32k uhd.

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

A 3D-style cute girl rendered in a Disneylike cartoon style, high detail and quality, with a clean background.

The sun is setting over the sea and mountains. A row of potted plants lined up in front of a stone wall.

A grassy field with mountains in the background.

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Realistic photograph, portrait of a beautiful robin, sunset in the background, cinematic lighting.

A butterfly sitting on top of some yellow flowers.

A close up of some pink flowers on a branch.

Black woman, li-core, greebles, watercolor, futuristic, mechanical, fire, white background.

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Illustration paintings roaring tiger of successful unique wall paintings.

An empty road with trees in the background.

A small bird sitting on a branch of a tree.

Impressionist painting, Beijing, vibrant colors, brushstrokes, bustling streets, traditional architecture, hazy atmosphere, soft light, Claude Monet.

###### Figure 6. Qualitative results of image generation. The images are of size 512 × 512.

###### Image Captioning

[Figure 57]

Prompt: Describe this image in detail.

Response: The image depicts a panoramic view of a bustling urban cityscape, likely a major metropolitan area. The foreground is dominated by numerous high-rise buildings, showcasing a dense concentration of skyscrapers. These buildings vary in height and architectural style, with some featuring modern glass facades and others displaying more traditional designs. The city appears to be a hub of economic activity, given the density and variety of the structures.

In the midground, the city extends into a large body of water, likely a harbor or bay, with numerous boats and ships. This waterway is likely a significant transportation route, facilitating the movement of goods and people. The presence of numerous boats suggests a thriving maritime industry.

The background features a vast expanse of water, likely a bay or a coastal area, with more high-rise buildings and urban development extending into the distance. The horizon is lined with a range of mountains or large hills, adding a natural backdrop to the otherwise man-made environment. The sky above is mostly clear with a few scattered clouds, indicating good weather conditions.

The overall scene is one of economic prosperity and urban development, with the combination of modern skyscrapers and the expansive waterway suggesting a city that is both economically vibrant and strategically located for maritime trade. The presence of numerous boats and the extensive waterway indicate a well-developed infrastructure, likely supporting a significant population and economy.

Prompt: Where is this most likely to be?

Response: This image is most likely of Hong Kong. The presence of the distinctive twin-towered skyscrapers, the hilly terrain, the bay with numerous boats, and the urban skyline with numerous high-rise buildings are characteristic features of Hong Kong’s skyline.

Prompt: Tell me about this logo.

[Figure 58]

Response: This is a logo for the CVPR conference, standing for the Conference on Computer Vision and Pattern Recognition, scheduled from June 17 to June 21, 2024, in Seattle, WA. The logo features a stylized blue cityscape of Seattle, prominently displaying notable landmarks such as the Space Needle. The text “CVPR” is displayed in large, bold letters to the right of the cityscape, with the dates and location information below it. The design is simple yet recognizable, effectively communicating the event’s location and timing.

Prompt: What is the content in this image?

Response: The image appears to be a screenshot from a video game. The scene is set in a dimly lit, stone-walled environment that resembles a castle or fortress. The walls are made of large stone blocks, and there is a staircase leading up to a doorway on the right side of the image. The doorway is open, revealing a dark interior.

[Figure 59]

In the foreground, there is a character with brown hair, wearing a red and white outfit. The character is facing away from the camera, looking up towards the stairs.

In the background, another character is standing on the stairs, facing the camera. This character has a white cap, a red and white outfit, and is carrying a backpack. The character is looking down the stairs towards the open doorway.

At the bottom of the image, there is a dialogue box with Chinese text. The text reads: “就一起回未白镇去吧。”.

###### OCR

[Figure 60]

Prompt: Describe this image from @albumreceipt.

Response: This image appears to be a tracklist for a music release titled “Curtain Call: The Hits” by EMINEM, in a receipt-like style. The tracklist is as follows:

- 1. Intro - 0:33
- 2. Fack - 3:25
- 3. The Way I Am - 4:21
- 4. My Name Is - 4:28
- 5. Stan - 6:44
- 6. Love Yourself - 5:26
- 7. Shake That - 4:34
- 8. Sing For The Moment - 5:40
- 9. Without Me - 4:51
- 10. Like Toy Soldiers - 4:55
- 11. The Real Slim Shady - 4:44
- 12. Mockingbird - 4:11
- 13. Guilty Conscience - 3:20
- 14. Cleanin’ Out My Closet - 4:58
- 15. Just Lose It - 4:08
- 16. When I’m Gone - 4:41
- 17. Stan (live, bonus track) - 6:20

The total tracklist includes 17 songs. The total length of the songs in this tracklist is 73:59. The release date is December 6, 2005. The label is “Aftermath Entertainment”.

###### Document Understanding

[Figure 61]

Prompt: Which platform will the train depart from? Response: The train will depart from King’s Cross Station in London, specifically Platform 9 3/4. Prompt: Where should Mr. Potter buy the items needed for school?

Response: Based on the image, Mr. Potter should buy the items needed for school from the Diagon Alley, which is mentioned as a requirement for the school trip.

###### Humor Understanding

[Figure 62]

Prompt: What is the punchline?

Response: The punchline to the joke in the image is the unexpected and humorous twist on the common phrase “finding the square root of a number.” Instead of finding the square root, the image humorously shows someone finding a tree root in the square shape. The root structure is exaggerated to the point where it appears to be an infinite loop, adding a layer of absurdity and humor to the joke.

###### Image to Latex

[Figure 63]

Prompt: Convert the formula in the figure to latex format. Response: Certainly, I’ll convert the formula in the figure to LaTeX format for you.

\begin{equation} \mathcal{L} = - \sum_{i \in \mathcal{T}} \log p(\hat{x}ˆi_T = xˆi_T | xˆ{<i})

- \lambda \sum_{i \in \mathcal{V}} \log p(\hat{x}ˆi_V = xˆi_V | xˆ{<i}), \end{equation}

###### Math

Prompt: What is the perimeter of the rectangle?

[Figure 64]

Response: To calculate the perimeter of the rectangle, we need to add up the lengths of all four sides. In this case, the rectangle has two sides that are each 6 feet long, and two sides that are each 3 feet long.

The perimeter of a rectangle is calculated by adding the lengths of all four sides. So, the formula is: Perimeter = 2(length + width)

For this rectangle, the length is 6 feet and the width is 3 feet. Plugging these values into the formula, we get: Perimeter = 2(6 + 3) = 2(9) = 18 Therefore, the perimeter of the rectangle is 18 feet.

Prompt: Lena has $0.13. How much money will Lena have left if she buys a shapes stencil and a green bouncy ball? (Unit: $)

[Figure 65]

Response: To find out how much money Lena will have left after buying a shapes stencil and a green bouncy ball, we need to subtract the total cost of these items from the amount she has.

The total cost of the shapes stencil and the green bouncy ball is $0.08 + $0.04 = $0.12. To find out how much money Lena will have left, we subtract the total cost from the amount she has: $0.13 - $0.12 = $0.01 Therefore, Lena will have $0.01 left after buying a shapes stencil and a green bouncy ball.

