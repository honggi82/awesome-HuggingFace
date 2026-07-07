# arXiv:2404.06212v1[cs.CV]9Apr2024

## OMNIFUSION TECHNICAL REPORT

Elizaveta Goncharova Anton Razzhigaev Matvey Mikhalchuk Maxim Kurkin

Irina Abdullaeva Matvey Skripkin Ivan Oseledets Denis Dimitrov Andrey Kuznetsov AIRI, Sber AI, Skoltech

ABSTRACT

Last year, multimodal architectures served up a revolution in AI-based approaches and solutions, extending the capabilities of large language models (LLM). We propose an OmniFusion model based on a pretrained LLM and adapters for visual modality. We evaluated and compared several architecture design principles for better text and visual data coupling: MLP and transformer adapters, various CLIP ViT-based encoders (SigLIP, InternVIT, etc.), and their fusing approach, image encoding method (whole image or tiles encoding) and two 7B LLMs (the proprietary one and open-source Mistral). Experiments on 8 visual-language benchmarks show the top score for the best OmniFusion setup in terms of different VQA tasks in comparison with open-source LLaVA-like solutions: VizWiz, Pope, MM-Vet, ScienceQA, MMBench, TextVQA, VQAv2, MMMU. We also propose a variety of situations, where OmniFusion provides highly-detailed answers in different domains: housekeeping, sightseeing, culture, medicine, handwritten and scanned equations recognition, etc. Mistral-based OmniFusion model is an open-source solution with weights, training and inference scripts available at https://github.com/AIRI-Institute/OmniFusion.

[Figure 1]

Figure 1: Comparison of OmniFusion performance on the benchmarks and generation examples.

[Figure 2]

[Figure 3]

Figure 2: OmniFusion VQA examples.

### 1 Introduction

In recent years, multimodal architectures emerged as a powerful paradigm for enhancing artificial intelligence (AI) systems, enabling them to process and understand multiple types of data simultaneously [1, 2, 3]. The integration of different data modalities, such as text and images, has significantly improved the capabilities of large language models (LLMs) in various tasks, ranging from visual question answering (VQA) [4] to complex decision-making processes [5, 6]. However, the challenge of effectively coupling various data types remains a significant obstacle in the development of truly integrative AI models. Furthermore, such multimodal multitask architectures are interpreted as the first steps towards the development of the artificial general intelligence (AGI), expanding the number of challenges in world cognition.

This work introduces the OmniFusion model, a novel multimodal architecture that leverages the strengths of pretrained LLMs and introduces specialized adapters for processing visual information. We evaluated multiple architectural designs to fuse text and visual data, such as MLP and transformer adapters. At the same time we concentrated on the comparison of various image encoders like CLIP-ViT [7] and SigLIP [8] and on visual encoders fusion techniques. The evaluated approaches gave us a broad scope of improving the visual context for better visual information extraction and retrieval.

One of the key innovations of OmniFusion is its flexible approach to image encoding, exploring both the whole image and the tiled image encoding strategies, which allows for a more nuanced understanding of visual content in relation to textual data [9]. This adaptability is critical in addressing the diverse requirements of visual-language benchmarks, where OmniFusion has demonstrated superior performance across a range of VQA tasks, outperforming existing open-source solutions.

Our extensive evaluations on eight visual-language benchmarks, including VizWiz [10], POPE [11], MM-Vet [12], ScienceQA [13], MMBench [14], TextVQA [15], VQAv2 [16], and MMMU [17], confirm the effectiveness of the OmniFusion model. The model not only achieves competitive performance with the leading approaches in various VQA tasks but also excels in providing detailed answers across multiple domains, such as housekeeping, sightseeing, culture, and medicine, showcasing its versatility and broad applicability.

AI: Wisdom teeth removal and treating cavities on the second molars.

AI: Wisdom teeth removal and treating cavities on the second molars.

— Text (token) embedding

— Text (token) embedding

— Image embedding

— Image embedding

BPE

BPE

— Trainable embedding — Predicted embedding

— Trainable embedding — Predicted embedding

| | |
|---|---|
|[912, 112, ... , 121, 47]| |

| | |
|---|---|
|[912, 112, ... , 121, 47]| |

Mistral-7B

Mistral-7B

| | |
|---|---|
| | |

| | | | |
|---|---|---|---|
|[124, 912, ... , 121, 47]| | | |

| | | | |
|---|---|---|---|
|[124, 912, ... , 121, 47]| | | |

| | |
|---|---|
|[54002]| |

| | |
|---|---|
|[54002]| |

| | |
|---|---|
|[54001]| |

| | |
|---|---|
|[54001]| |

Adapter

FeatureMerging

| | |
|---|---|
| | |
| | |
| | |

CLIP-ViT-L

DINO-v2

BPE

###### BPE

BPE

BPE

###### BPE

BPE

CLIP-ViT-L

This is a dialog with AI assistant. User: What recommendation

This is a dialog with AI assistant. User: What recommendation

<IMG_START>

<IMG_END>

<IMG_START>

<IMG_END>

| | |
|---|---|
| | |

should be given to the patient?

should be given to the patient?

[Figure 4]

[Figure 5]

- Figure 3: OmniFusion architecture with feature merging (left) and with single adapter (right): MLP or transformer layer.

### 2 OmniFusion

#### 2.1 Model Architecture

The OmniFusion model integrates a pretrained LLM with special adapters for image embeddings, facilitating the fusion of the text and visual modalities. This approach has already proved its applicability in various VLM developments because the adapter approach is undemanding of computational resources in comparison with end-to-end training pipelines. Moreover, it does not need large interleaved text-image datasets to be trained, while for end-to-end approaches such data is extremely required. Overall, we have two major points that were investigated and covered further in this report: adapter technique selection and encoding visual data strategy determination.

We used special tokens with trainable embeddings to signify the beginning and the end of a token sequence from the non-text modality. This approach enhances the model’s ability to distinguish between textual and visual data streams. The visual encoder’s output embeddings are processed either by a transformer adapter, consisting of a single transformer layer with four heads, or through a two-layer MLP.

Our experiments showed the most effective performance with a merging strategy that combines features of two encoders: CLIP-ViT-L and DINO-v2. The merging adapter is essentially a two-layer MLP, where the first layer is independent for each visual token, and the output layer is shared across all tokens. The merging strategy implies summing up the outputs after the first layer, optimizing the integration of textual and visual information (Figure 3).

#### 2.2 Training pipeline

When the overall architecture setup is fixed, we continue with overall training, which includes determination of the stages and the used datasets. The OmniFusion model undergoes a two-stage training process designed to harness its multimodal capabilities efficiently.

- Stage 1: Pretraining. In the first stage, adapters and special tokens undergo pretraining on a vast dataset of image-text pairs. This phase aims to fine-tune the adapters for transforming visual embeddings and training special tokens to mark the transition between text and image data.

During this stage, we leverage image-text pairs tailored specifically for image captioning purposes. The imagecaptioning data we employ is sourced from ShareGPT4V-PT, which is supported by existing captioning datasets such as COCO and TextCaps. Additionally, we incorporate proprietary datasets containing document descriptions obtained through optical character recognition (OCR) in the pretraining set.

The instructions provided to the model during pretraining, include variations of prompts such as "Give a brief description of the image," "Describe the image in detail," and "Provide a short description of the image."

It’s worth noting that we currently do not employ interleaved data during pretraining, which could potentially enhance the few-shot learning capabilities of multimodal models. However, we plan to explore this avenue in our future work.

We use the following sources of image-text captions: ShareGPT4V-PT (695K pairs), LAION-CC-SBU with BLIP captions (558K pairs) [18]. Overall, we utilize 1.2M image captions during the pretraining phase for the adapter alignment.

- Stage 2: Fine-tuning. The second stage entails fine-tuning the multimodal model using a set of instructional dialogues. This process aims to enhance the model’s ability to comprehend and respond to complex queries that require an integrated analysis of textual and visual information. Moreover, it is structured to facilitate multi-task fine-tuning in accordance with the natural instructions.

Recent research [19] has highlighted the potential pitfalls associated with the synthetic nature of data used for visual instruction tuning, as well as the lack of task diversity, which can lead to significant hallucinations in the output of multimodal models. To mitigate these challenges, we construct the supervised fine-tuning dataset by combining academic benchmarks with corresponding instructions and synthetic conversational data. This approach ensures a more diverse and robust training environment for the model. Table 1 provides the datasets utilized in the fine-tuning procedure.

Table 1: Distribution of the datasets for supervised fine-tuning.

|Task|Dataset Source<br><br>|Samples|
|---|---|---|
|Caption VQA WebQA OCRQA Conversation DocVQA<br><br>|ShareGPT4V [20] COCO [21], SAM-9K [22] WebData [23] TextVQA [24], OCRVQA [25] LLaVA-v1.5-665K [18] Proprietary data (ru)<br><br>|100K 20K, 9K 1.5K 120K 665K 20K|
|Text-only SFT|Proprietary data (ru), Alpaca (en)|10K|

Training hyperparameters. During the training procedure, we utilize a standard approach with a learning rate of 1e-3 and a batch size of 256 for the pretraining phase. For the SFT phase, we adjust the learning rate to 2e-5 and the batch size to 128. We train the model using the bf16 precision, employing the AdamW optimizer with a weight decay set to 0.

The sequence length is initially set to 2048 for the base experiment, and we extend it to 4096 for experiments involving grid splitting. All experiments are conducted using 8 Nvidia A100-80Gb GPUs.

Table 2: Training Procedure Parameters

|Parameter|Pretraining Phase|SFT Phase|
|---|---|---|
|Learning Rate|1e-3|2e-5<br><br>|
|Batch Size<br><br>|256|128|
|Precision<br><br>|bf16| |
|Optimizer<br><br>|AdamW| |
|Weight Decay<br><br>|0| |
|Sequence Length<br><br>|2048 (Base), 4096 (Grid Splitting)| |
|Hardware<br><br>|8 Nvidia A100-80Gb GPUs| |

- 3 Experiments

#### 3.1 Experimental setup

Vision encoders. The influence of the image encoders is inevitable for multimodal systems. Basically, vision encoders are kept frozen via image-text alignment, so, its initial capabilities are crucial for further performance of LMMs. In our experiments, we evaluated several encoders varying in size, training data distribution, and image resolution. Overall, the visual encoders that we tested in the experiments, are listed below:

1. CLIP ViT-bigG/141. A CLIP ViT-bigG/14 model was trained with the LAION-2B English subset of LAION-5B [26] following the openclip architecture [27].

1https://github.com/mlfoundations/open_clip

- 2. CLIP ViT-L/14 [7] from openai/clip-vit-large-patch14. This is the vision encoder that was retrieved from the CLIP model developed by researchers at OpenAI. The model is the standard version of vision encoders providing effective representation of the image features. The clip-L is used in such multimodal models as LLaVA-v1.5, LLaVA-Next, InternLM-XComposer2-VL [28] and others.
- 3. SigLIP/16-5122. The SigLIP model was proposed in [8] by Google Research. SigLIP proposes to replace the loss function used in CLIP with a simple pairwise sigmoid loss. The model outperforms most of the existing models so far and provides state-of-the-art results in the classification tasks.
- 4. InternViT-6B3. InternViT-6B-448px-V1-2 [29] is the largest open-sourced vision/vision-language foundation model to date, achieving 32 state-of-the-art performances on a wide range of tasks such as visual perception, cross-modal retrieval, multimodal dialogue, etc. The model was proposed by the OpenGVLab team. Pretrain Dataset: LAION-en, LAION-COCO, COYO, CC12M, CC3M, SBU, Wukong, LAION-multi, OCR data. The VLMs that utilize such encoders are as follows: InternVL-Chat-V1.2, InternVL-Chat-V1.2-Plus.

Table 3 provides the overall description of the observed vision encoders that can be used for vision encoding in multimodal models.

Table 3: Vision encoders characteristics.

| |patch resolution|fine-tuning resolution<br><br>|num hidden layers|image tokens num<br><br>|emb size|emb layer|
|---|---|---|---|---|---|---|
|CLIP ViT-bigG/14<br><br>|14x14<br><br>|224x224|48<br><br>|256|1664<br><br>|-2|
|CLIP VIT-large/14|14x14<br><br>|336x336<br><br>|24|576<br><br>|1024<br><br>|-2|
|SigLIP-base/16-512<br><br>|16x16|512x512|12|1024<br><br>|768<br><br>|-2|
|InternViT-6B-448px-V1-2<br><br>|14x14|448x448<br><br>|45|1024<br><br>|3200|-1|

To evaluate the performance of different visual encoders on the multimodal model’s visual understanding abilities, we trained the final model using the same data samples but with various vision backbones. We then assessed the model’s quality on multimodal benchmarks to compare the effectiveness of different visual encoders. We present the evaluation results of OmniFusion trained with different vision backbones in Table 4.

Table 4: Comparison of OmniFusion performance with different vision backbones on multimodal benchmarks.

| |Resolution|ScienceQA-full<br><br>|ScienceQA-img|text-VQA<br><br>|VQAv2<br><br>|VizWiz<br><br>|POPE<br><br>|MMBench|MMMU|MM-Vet|
|---|---|---|---|---|---|---|---|---|---|---|
|CLIP ViT-bigG/14|336<br><br>|68.71|66.68<br><br>|47.58<br><br>|73.28|49.92<br><br>|82.80|61.73<br><br>|30.60<br><br>|30.10|
|CLIP VIT-large/14<br><br>|336|71.21<br><br>|68.17<br><br>|57.20|78.42<br><br>|54.11|84.50|65.72<br><br>|30.90|33.60|
|SigLIP-base/16-512<br><br>|512<br><br>|71.92|68.62<br><br>|57.05<br><br>|78.88|51.84<br><br>|86.5<br><br>|66.80|31.60<br><br>|33.70|
|InternViT-6B-448px-V1-2<br><br>|448|73.38<br><br>|69.71|61.57<br><br>|80.08|52.78<br><br>|84.47<br><br>|67.10|35.40<br><br>|35.10|

Notably, the best performance is achieved by OmniFusion trained with the largest visual encoder, InternViT-6B-448pxV1-2. Despite the higher resolution for SigLIP, CLIP VIT-large/14 and SigLIP-base/16-512 achieve comparable results on most tasks. However, both of them significantly outperform CLIP ViT-bigG/14 with the smallest resolution.

Mix of image encoders. Rather than relying solely on a single image encoder, one approach to enhance image representation is to merge embeddings from multiple encoders. In this section, we considered various image encoders and adapters for mixing encoder features. We trained a language model and adapter using the features from multiple image encoders simultaneously. OpenAI CLIP-ViT-L/14 (336 × 336) and ViT-DINO-v2 models were used as image encoders. Mixing of encoder features was performed in the adapter module. The following 4 different adapter module architectures were investigated.

The first architecture (baseline) corresponds to the encoder and adapter used in the previous experiments. It uses only one encoder (OpenAI CLIP-ViT-L/14) and an adapter with a transformer decoder layer. More specifically, the adapter unit has the following architecture. The model first uses the transformer encoder layer and then does a linear transformation to change the dimension of the embedding vectors.

The second architecture (embedding concatenation) uses two encoders, first, the vectors of the feature sequences from the penultimate layers of encoders are independently linearly mapped. Two different linear transformations are used for the features from the first and the second encoders. Afterward, the embedding sequences are then concatenated and fed to the encoder layer of the transformer and mapped linearly (to change the dimension of the embeddings) as in the baseline.

- 2https://github.com/merveenoyan/siglip
- 3https://github.com/OpenGVLab/InternVL-MMDetSeg

The third architecture, inspired by COMM [30], also uses 2 encoders. An important difference from the previous version is the use of the features from all layers of image encoders. First, layer normalization is applied to the feature from all layers of each encoder, then linear transformation is applied (different mappings are used for the features from the different layers), after which, a linear combination of the features from different layers is calculated (the coefficients of the linear combination are trainable parameters), the resulting features for two encoders are summed (sequences of embeddings from two encoders have different lengths, so the shorter sequence is concatenated with zero vectors before the addition). Finally, the GELU activation function and the final linear transformation are applied.

The last architecture, the fourth one, is similar to the third, but after aggregating features from different layers of encoders (counting linear combinations of features from different layers of encoders), it uses a different method of mixing features from two encoders. GELU is applied to the embedding sequences from two encoders, then the embedding sequences are concatenated and fed to the multi-head attention layer as a query, and a trained parameter matrix is used for the key and value. In the end, the GELU transform and linear mapping are applied. This approach is interesting because by changing the size of the trained parameter matrix, it is possible to change the number of vectors in the output embedding sequence. We used a matrix of 576 rows so that the length of the visual embedding sequence is the same as in the baseline.

- Table 5 presents the results of the tests on various benchmark models that use different methods of mixing features from two visual encoders.

Table 5: Comparison of OmniFusion performance with different CLIP VIT-L/14 and DinoV2 encoders mix.

| |ScienceQA-full|ScienceQA-img<br><br>|Text-VQA|VQAv2<br><br>|VizWiz|POPE<br><br>|MMBench<br><br>|MMMU|MM-Vet|
|---|---|---|---|---|---|---|---|---|---|
|Baseline (one encoder)|71.21|68.17<br><br>|57.20|78.42|54.11<br><br>|84.50<br><br>|65.72|32.30<br><br>|33.60|
|Embeddings concatenation<br><br>|72.25<br><br>|68.62|54.02|75.95<br><br>|51.33<br><br>|85.40<br><br>|67.87|34.10<br><br>|30.60|
|Projection & summation|71.09<br><br>|67.18<br><br>|60.87<br><br>|78.77<br><br>|57.09|85.73<br><br>|69.24<br><br>|36.50|34.60|
|Projection & multi-head attention<br><br>|72.01<br><br>|68.17|47.75<br><br>|70.84<br><br>|50.82|84.93|57.47<br><br>|29.40<br><br>|21.90|

The best method turned out to be one that uses features from all the layers of encoders and sums them to mix the features from two encoders. We also noticed that the last approach with a multi-headed attention layer turned out to be the worst in terms of the inference speed.

In the initial experiments, we also compared two adapter techniques: linear projection, implemented as a two-layer MLP with GeLU activation, and a simple transformer-based Encoder block with one layer and four heads. We found that the best results were obtained with the simple linear projection. While the transformer-based adapter performs comparably to MLP on some benchmarks (such as VizWiz or ScienceQA), we observed a drop of up to 4 points on OCR-based tasks.

We hypothesize that this difference can be explained by the fact that originally, Vision Transformer (VIT) models trained in the CLIP pipeline are naturally aligned with text, making linear transformation suitable for this task. Additionally, we speculate that larger trainable adapters require more data for successful alignment.

Scaling Images to HD. Following the methodology of LLaVA-NeXT [9], we investigate whether image resolution plays a significant role in capturing fine-grained pixel scale information.

Our approach involved resizing images adaptively to the input size and dividing them into non-overlapping sections based on the resolution of the image encoder. This strategy ensured that we retained all the specific details from the original images, thus enhancing the model’s performance on tasks that rely on such details.

To encode the image, we employ CLIP-ViT-L/14 (336 × 336), treating each patch as a separate entity. Initially, we resized and padded the image to the most suitable dimensions. Then, we divided it into a grid of 336 × 336 patches, encoding each patch individually. As has been observed in previous research, encoding the entire image into a sequence of image tokens is crucial for preserving the overall context of the scene, so, we also kept this image representation in our final experiment.

To assess the model’s capability to encode images with higher resolution. we trained three versions of the same model. The baseline model is OmniFusion-7B with the frozen OpenAI Clip-L vision encoder. The second model follows the same architecture, but during fine-tuning, the image is split into patches and encoded by the Clip-L encoder. Finally, we augmented the training data with 20K proprietary samples from the DocVQA domain to evaluate whether this additional dataset, combined with image splitting, enhances the model’s performance in the document domain.

- Table 6 presents the results of OmniFusion trained with the strongest vision backbone in standard resolution (448×448), and in high resolution with CLIP-ViT-L/14 (336 × 336), and grid split on standard multimodal benchmarks, while
- Table 7 shows the results for specific benchmarks related to document and infographics analysis.

Table 6: Comparison of OmniFusion performance w and w/o grid splitting of the input image.

| |ScienceQA full<br><br>|ScienceQA img|text-VQA w OCR<br><br>|Text-VQA w/o OCR<br><br>|VQAv2<br><br>|POPE|MMBench|MMMU|MM-Vet<br><br>|GQA|
|---|---|---|---|---|---|---|---|---|---|---|
|OmniFusion InternViT-6B-448px-V1-2|73.38|69.71<br><br>|61.57<br><br>|56.48|80.08<br><br>|84.47<br><br>|67.10<br><br>|35.40<br><br>|35.10|63.44|
|OmniFusion grid split|71.23<br><br>|67.97<br><br>|63.31<br><br>|60.06<br><br>|80.86|86.72<br><br>|69.00<br><br>|34.10|36.60<br><br>|64.18|
|OmniFusion grid split + ruDocVQA<br><br>|73.21|69.16<br><br>|65.53<br><br>|63.50|80.94|87.21<br><br>|67.61|36.60<br><br>|39.40|64.57|

Table 7: Comparison of OmniFusion w and w/o grid splitting for document and infographics analysis.

| |InfoVQA<br><br>|ChartQA<br><br>|DocVQA|multiDocVQA|
|---|---|---|---|---|
|OmniFusion InternViT-6B-448px-V1-2|29.24|19.92<br><br>|38.44|22.5/11.43|
|OmniFusion grid split|30.2|20.2|46.76<br><br>|27.09/14.5|
|OmniFusion grid split + rudocVQA|36.99|28.96|62.82<br><br>|36.38/20.55|

Based on the observed results, we make the following observations:

- 1. First, high image resolution is particularly relevant in domains such as OCR. Additionally, the evaluation of hallucination (POPE) is also influenced by image resolution.
- 2. Another interesting finding concerns document-based benchmarks. Adding 20K samples of proprietary document-level data in Russian significantly boosted the model’s performance on document analysis (refer to Table 7, line 3).

Tuning with synthesized TeX formulas. In this section, we examined SFT for the task of generating LaTeX code from a formula image. The datasets used for training included CROHME [31] (containing pairs of handwritten formulas and LaTeX code) and the im2latex dataset (containing pairs of formulas rendered in different fonts and their corresponding LaTeX code). Both of these preprocessed datasets were obtained from the open-source repository LaTeX-OCR4. Overall, the character sequence length in the training dataset ranges from 1 to 4296 symbols.

- As image encoders we used CLIP VIT-large/14 and Donut [32] vision encoder. The last was fine-tuned for the text and LaTeX formula recognition in the open-source repository texify5 and provided a good latent representation for this specific case of math equation recognition. SFT stage consisted of two steps:

- 1. Freezing the image encoder and LLM, fine-tuning the adapter and the special tokens.
- 2. Freezing the image encoder, fine-tuning the adapter, and the special tokens, and fine-tuning LLM using LoRA.

- As a result of the study, when using CLIP VIT-large/14 as the image encoder, the model performs poorly in long formula recognition but is capable of perceiving the handwritten ones, whereas the Texify vision encoder shows the opposite behavior. The comparison of the two encoders for TeX formula recognition is presented in Table 8.

The normalized edit distance (NED) was utilized as a metric to evaluate the quality of LaTeX code generation across 30,600 samples from the im2latex test dataset, encompassing formulas ranging in length from 4 to 2226 symbols.

Table 8: Comparison of two image encoders for OmniFusion for TeX formulas recognition.

|Image Encoders<br><br>|NED ↓|
|---|---|
|CLIP VIT-large/14<br><br>|0.32|
|Texify vision encoder|0.19|

- Figure 4 provides examples of LaTeX formula understanding by OmniFusion with the Texify vision encoder.

- 4https://github.com/lukas-blecher/LaTeX-OCR
- 5https://github.com/VikParuchuri/texify

|[Figure 6]|
|---|

\sum _ { j = 1 } ^ { n } \left| \mathbf { a } _ { j } \right| ^ { 2 } = \sum _{j = 1} ^{ n } \sum _{i = 1}^{n } \left( \mathbf {a } _ { \mathbf {j } } \cdot \mathbf {v } _ { 1 } \right) ^ {2} = \sum_{i=1}^ {n} \sum_{j=1}^{n} ( \mathbf{a } _ {\mathbf{j} } \cdot { \mathbf{v } } _ {1 } ) ^ {n} = \left( i = 1 \right) = \left| A \mathbf{ v } _ {{ 1 }} \right| ^{2} = { \sum_{ i =1 } ^{n} } \sigma _ { i } ^{2} ( A ) .

[Figure 7]

|[Figure 8]|
|---|

q ^ { \star } \triangleq \operatorname { a r g } \operatorname * { m i n } _ { q } \mathbb { E } \Bigl [ \bigl \| q ( z _ { \theta } ) - z _ {\xi } ^ { \prime } \bigr \| ^ { 2 } \Bigr ] , \quad \mathrm { w h e r e } \quad q ^ { * } ( z _ {\theta } ) = \operatorname { E } [ z _ {\zeta } | z _ {\vartheta } ] ,

[Figure 9]

Figure 4: An example of LaTeX formula understanding by OmniFusion fine-tuned with the Texify vision encoder is depicted below. The upper image displays the input image, while the lower image showcases the compilation of LaTeX code generated by the model.

#### 3.2 Evaluation on benchmarks

In this section, we present the evaluation results of the introduced model. Table 9 presents a comparison of existing leading approaches to training multimodal models with our best options of OmniFusion. We provide a comparison of the models with standard and high resolution across several benchmarks in zero-shot settings. For the evaluation of OmniFusion and LLaVA, we utilize the recent open-source library lmms-eval [33].

Main results. The main results are as follows: at standard resolution, OmniFusion using the largest vision encoder InternViT-6B-448px-V1-2 achieves the best scores on most of the benchmarks, being on an equal basis with models based on larger LLMs (such as Vicuna-13B [37]).

Moreover, in the subgroup of models that operate on images with the same resolution, the proposed approach of encoder mixing significantly improves results in Text-VQA and GQA datasets, adding one point, on average, to the MMMU benchmark as well.

For high resolution, the version of OmniFusion with grid splitting provides comparative results with state-of-the-art open-source multimodal models.

### 4 Related Work

Taking advantage of LMMs powerful capability to follow instructions, active work is underway to extend their ability to perceive other types of data, particularly visual data. Early work in the field, including FROMAGe [38], LLaVA [39], MiniGPT4 [40], and BLIP-2 [41], encoded visual information into one or more visual tokens by training only a lightweight projection that efficiently matched images to the language model. This approach has enabled tasks such as image captioning and visual question answering to be partly solved. Several subsequent studies have focused on enhancing and adjusting the projection architecture [42, 1, 43, 44], and determining the need for pretraining LLM on image text data [45, 2, 34, 46]. Another promising approach was introduced in [47, 48], the authors showed that being pre-trained from scratch on large amount of interleaved data multimodal models obtain external abilities to understand

Table 9: Comparison with recent multimodal models on zero-shot benchmarks. The best results across model subgroups are in bold; the best performance for models with LLMs of the same size is underlined. As we lack information on how TextVQA is assessed, for some models we include additional evaluation with supported OCR tokens in brackets. We do not provide information on whether the train set of the benchmark was included.

ScienceQA full

ScienceQA img

Model LLM Encoder Res.

Text-VQA VQAv2 POPE GQA MMMU MM-Vet Standard resolution

|LLaVA-v1.5 13B [18]<br><br>|Vicuna-13B|CLIP VIT-large/14<br><br>|336<br><br>|74.96|72.88|48.73 (60.75)<br><br>|79.52<br><br>|85.92<br><br>|63.24<br><br>|34.80|36.70|
|---|---|---|---|---|---|---|---|---|---|---|---|
|LLaVA-v1.5 7B [18]|Vicuna-7B|CLIP VIT-large/14<br><br>|336<br><br>|70.41<br><br>|70.43<br><br>|46.07 (58.20)|78.50|85.87<br><br>|61.97|35.30<br><br>|30.55|
|Qwen-VL-Chat [34]<br><br>|Qwen-7B|CLIP VIT-bigG/14|448<br><br>|-|68.2|63.80<br><br>|-|88.10<br><br>|57.50<br><br>|37.00<br><br>|47.30|
|Mini-Gemini [35]|Vicuna-7B|CLIP VIT-large/14 ConvNeXt-L<br><br>|336|-|-<br><br>|65.20<br><br>|-<br><br>|88.10<br><br>|-|36.10<br><br>|40.80|
|OmniFusion|GigaChat-7B<br><br>|CLIP VIT-large/14<br><br>|336|71.21|68.17<br><br>|51.87 (57.20)|78.42|84.50<br><br>|65.72|32.30<br><br>|33.60|
|OmniFusion<br><br>|GigaChat-7B|InternViT-6B-448px V1-2|448<br><br>|74.13<br><br>|71.29<br><br>|56.48 (61.57)<br><br>|80.08<br><br>|84.14<br><br>|63.44<br><br>|35.40|35.10|

Encoder Merge

|OmniFusion<br><br>|GigaChat-7B|CLIP VIT-large/14 +DinoV2|336<br><br>|71.09<br><br>|67.18|49.75 (60.87)<br><br>|78.77|85.73|64.59<br><br>|36.50|34.60<br><br>|
|---|---|---|---|---|---|---|---|---|---|---|---|
|OmniFusion|Mistral-7B<br><br>|CLIP VIT-large/14 + DinoV2|336|70.30<br><br>|67.55|48.93 (59.15)<br><br>|-|82.18|64.00<br><br>|35.90|33.10|
|OmniFusion|Vicuna-7B<br><br>|CLIP VIT-large/14 + DinoV2|336|71.26<br><br>|69.96|49.58 (61.04)|-<br><br>|85.81<br><br>|62.60|36.90<br><br>|33.60|

HR resolution

|LLaVA-Next 7B|Vicuna-7B<br><br>|CLIP VIT-large/14<br><br>|672<br><br>|73.21<br><br>|70.15<br><br>|64.85<br><br>|80.06<br><br>|86.40|64.23<br><br>|35.10|44.08|
|---|---|---|---|---|---|---|---|---|---|---|---|
|LLaVA-Next 13B<br><br>|Vicuna-13B|CLIP VIT-large/14<br><br>|672|75.85<br><br>|73.57<br><br>|66.92|80.92<br><br>|86.26|65.36<br><br>|35.90<br><br>|49.12|
|DeepSeek-VL [36]|DeepSeek-7B<br><br>|SigLIP-L SAM-B|384 1024<br><br>|-|-|-|-<br><br>|88.10<br><br>|-|36.6<br><br>|41.50|
|Mini-Gemini-HD [35]|Vicuna-7B|CLIP VIT-large/14 ConvNeXt-L<br><br>|672<br><br>|-<br><br>|-|68.40<br><br>|-|-<br><br>|-<br><br>|36.8<br><br>|41.30|
|OmniFusion + grid-split|GigaChat-7B<br><br>|CLIP VIT-large/14|672|73.21<br><br>|69.16|63.5 (65.53)<br><br>|80.94<br><br>|87.21<br><br>|64.57<br><br>|36.60<br><br>|39.40|

Proprietary models

|GPT-4V|Unk|Unk|Unk<br><br>|-<br><br>|-<br><br>|78.00|-|-|-<br><br>|56.80|49.90|
|---|---|---|---|---|---|---|---|---|---|---|---|
|Gemini Pro<br><br>|Unk<br><br>|Unk|Unk<br><br>|-<br><br>|-<br><br>|74.60<br><br>|-|-|-<br><br>|47.90|64.30<br><br>|
|Qwen-VL-Plus<br><br>|Unk<br><br>|Unk|Unk<br><br>|-<br><br>|-<br><br>|78.90<br><br>|-|-<br><br>|-|45.20<br><br>|55.70|
|Qwen-VL-Max<br><br>|Unk|Unk|Unk<br><br>|-<br><br>|-|79.50|-<br><br>|-|-<br><br>|51.40|61.80|

long context and support few-shot learning. However, since training these models requires a lot of computational resources there’s greater interest in lighter-weight approaches. These involve partially freezing specific parts of the backbones within the multimodal pipeline [49, 39, 34].

Significant improvement in visual-textual perceptual quality was achieved by curating more synthetic fine-tuning data, as introduced in LLaVA-1.5 [18]. This model has a relatively simple architecture and uses only publicly available datasets, highlighting the importance of data and multimodal task diversity for VLM training. A number of subsequent papers have improved this approach by enhancing the visual grounding capability in LLaVA-G [50], adapting to specific domains as LLaVA-Med [5]; instructively learning to use tools and invoking other task-specific models in LLaVA-PLUS [4], improving efficiency through MoE-based sparse modification in MoE-LLaVA [51], and reducing the underlying language model to Phi-2.7B in the LLaVA-Phi paper [52]. The Video-LLaVA work [3] continues to expand the horizons of multimodality to the video comprehension.

- At the same time, state-of-the-art results have been achieved through careful investigation of the most beneficial process of visual-textual pre-learning and the effective incorporation of the visual modality into VLLM. The VILA paper [53] argues that unfreezing the LLM allows for stronger in-context learning and multi-image reasoning capabilities. In addition, these results also further support the conclusions of the Monkey paper that the image resolution is more important than the number of visual tokens. Such an approach to embedding visual modality may help to provide greater scalability and better quality on fine-grained visual-language tasks. COMM [30] revised the effectiveness of already existing visual models in MLLM and proposed a simple yet effective multi-level features merging strategy for different image encoders to enhance visual capabilities. The follow-up VistaLLM work [54] proposed an instruction-guided image tokenizer that filters global embeddings using task descriptions to extract compressed and refined features from images into a unified general-purpose model.

Despite these advancements, there are still open challenges in extracting better visual features from images, indicating the need for further development of multimodal architectures in this area. Our work takes mutual advantages of both previously mentioned research directions, i.e., using the LLaVA-based VLM architecture as a baseline and effectively combining multiple visual encoders into a single visual branch providing a good opportunity to enhance latent visual representation for general and specific domains.

### 5 Conclusion

In conclusion, our OmniFusion model represents significant progress in the field of multimodal learning, integrating textual and visual data within a single framework. By evaluating various architecture design principles, and image encoding methods, and integrating multiple vision encoders, we have demonstrated the model’s superior performance across a wide array of visual-language benchmarks. Our findings underscore the effectiveness of OmniFusion in handling complex VQA tasks, outperforming existing solutions, and providing detailed, domain-specific responses. With the release of our open-source Mistral-based OmniFusion model, including weights and scripts for training and inference, we aim to contribute to the broader AI research community, facilitating further developments in multimodal AI systems.

Further development of our research will shed light on the new approaches of efficient image embedding extraction including not only encoders research but also some hierarchical image representation methods. In terms of overall architecture design, we plan to go deeper into long image context processing, especially in video modality integration.

- At the same time, providing fast and reliable image output generation using our latent diffusion model Kandinsky [55] is also one of the major research tasks, which will lead to the new OmniFusion features like local image editing by text prompt, image generation during multimodal dialog context, etc. References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning. ArXiv, abs/2204.14198, 2022.
- [2] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, W. Zhang, Pan Lu, Conghui He, Xiangyu Yue, Hongsheng Li, and Yu Jiao Qiao. Llama-adapter v2: Parameter-efficient visual instruction model. ArXiv, abs/2304.15010, 2023.
- [3] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. ArXiv, abs/2311.10122, 2023.
- [4] Shilong Liu, Hao Cheng, Haotian Liu, Hao Zhang, Feng Li, Tianhe Ren, Xueyan Zou, Jianwei Yang, Hang Su, Jun-Juan Zhu, Lei Zhang, Jianfeng Gao, and Chun yue Li. Llava-plus: Learning to use tools for creating multimodal agents. ArXiv, abs/2311.05437, 2023.
- [5] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. ArXiv, abs/2306.00890, 2023.
- [6] Chaoyou Fu, Renrui Zhang, Haojia Lin, Zihan Wang, Timin Gao, Yongdong Luo, Yubo Huang, Zhengye Zhang, Longtian Qiu, Gaoxiang Ye, et al. A challenger to gpt-4v? early explorations of gemini in visual expertise. arXiv preprint arXiv:2312.12436, 2023.
- [7] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021.
- [8] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training, 2023.
- [9] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024.
- [10] Danna Gurari, Qing Li, Abigale J. Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P. Bigham. Vizwiz grand challenge: Answering visual questions from blind people, 2018.
- [11] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models, 2023.
- [12] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities, 2023.
- [13] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering,

- 2022.

- [14] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player?,

- 2023.

- [15] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read, 2019.
- [16] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 6325–6334, 2017.
- [17] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi, 2023.
- [18] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. ArXiv, abs/2310.03744, 2023.
- [19] Zhiyang Xu, Chao Feng, Rulin Shao, Trevor Ashby, Ying Shen, Di Jin, Yu Cheng, Qifan Wang, and Lifu Huang. Vision-flan: Scaling human-labeled tasks in visual instruction tuning, 2024.
- [20] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions, 2023.
- [21] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollar, and C. Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server, 2015.
- [22] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything, 2023.
- [23] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, Xiaoyi Dong, Haodong Duan, Qi Fan, Zhaoye Fei, Yang Gao, Jiaye Ge, Chenya Gu, Yuzhe Gu, Tao Gui, Aijia Guo, Qipeng Guo, Conghui He, Yingfan Hu, Ting Huang, Tao Jiang, Penglong Jiao, Zhenjiang Jin, Zhikai Lei, Jiaxing Li, Jingwen Li, Linyang Li, Shuaibin Li, Wei Li, Yining Li, Hongwei Liu, Jiangning Liu, Jiawei Hong, Kaiwen Liu, Kuikun Liu, Xiaoran Liu, Chengqi Lv, Haijun Lv, Kai Lv, Li Ma, Runyuan Ma, Zerun Ma, Wenchang Ning, Linke Ouyang, Jiantao Qiu, Yuan Qu, Fukai Shang, Yunfan Shao, Demin Song, Zifan Song, Zhihao Sui, Peng Sun, Yu Sun, Huanze Tang, Bin Wang, Guoteng Wang, Jiaqi Wang, Jiayu Wang, Rui Wang, Yudong Wang, Ziyi Wang, Xingjian Wei, Qizhen Weng, Fan Wu, Yingtong Xiong, Chao Xu, Ruiliang Xu, Hang Yan, Yirong Yan, Xiaogui Yang, Haochen Ye, Huaiyuan Ying, Jia Yu, Jing Yu, Yuhang Zang, Chuyu Zhang, Li Zhang, Pan Zhang, Peng Zhang, Ruijie Zhang, Shuo Zhang, Songyang Zhang, Wenjian Zhang, Wenwei Zhang, Xingcheng Zhang, Xinyue Zhang, Hui Zhao, Qian Zhao, Xiaomeng Zhao, Fengzhe Zhou, Zaida Zhou, Jingming Zhuo, Yicheng Zou, Xipeng Qiu, Yu Qiao, and Dahua Lin. Internlm2 technical report, 2024.
- [24] Amanpreet Singh, Vivek Natarjan, Meet Shah, Yu Jiang, Xinlei Chen, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8317–8326, 2019.
- [25] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In ICDAR, 2019.
- [26] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. Laion-5b: An open large-scale dataset for training next generation image-text models, 2022.
- [27] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, July 2021. If you use this software, please cite it as below.
- [28] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model, 2024.
- [29] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023.

- [30] Dongsheng Jiang, Yuchen Liu, Songlin Liu, Jin’e Zhao, Hao Zhang, Zhen Gao, Xiaopeng Zhang, Jin Li, and Hongkai Xiong. From clip to dino: Visual encoders shout in multi-modal large language models, 2024.
- [31] Harold Mouchère, Christian Viard-Gaudin, Richard Zanibbi, and U. Garain. Icfhr2016 crohme: Competition on recognition of online handwritten mathematical expressions. In 2016 15th International Conference on Frontiers in Handwriting Recognition (ICFHR), pages 607–612, 2016.
- [32] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In European Conference on Computer Vision, pages 498–517. Springer, 2022.
- [33] Bo Li*, Peiyuan Zhang*, Kaichen Zhang*, Fanyi Pu*, Xinrun Du, Yuhao Dong, Haotian Liu, Yuanhan Zhang, Ge Zhang, Chunyuan Li, and Ziwei Liu. Lmms-eval: Accelerating the development of large multimoal models, March 2024.
- [34] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. 2023.
- [35] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models, 2024.
- [36] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, and Chong Ruan. Deepseek-vl: Towards real-world vision-language understanding, 2024.
- [37] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023.
- [38] Jing Yu Koh, Ruslan Salakhutdinov, and Daniel Fried. Grounding language models to images for multimodal generation. ArXiv, abs/2301.13823, 2023.
- [39] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. ArXiv, abs/2304.08485, 2023.
- [40] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. ArXiv, abs/2304.10592, 2023.
- [41] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning, 2023.
- [42] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Albert Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. ArXiv, abs/2305.06500, 2023.
- [43] Yiren Jian, Chongyang Gao, and Soroush Vosoughi. Bootstrapping vision-language learning with decoupled language pre-training. ArXiv, abs/2307.07063, 2023.
- [44] Junyu Lu, Ruyi Gan, Di Zhang, Xiaojun Wu, Ziwei Wu, Renliang Sun, Jiaxing Zhang, Pingjian Zhang, and Yan Song. Lyrics: Boosting fine-grained language-vision alignment and comprehension via semantic-aware visual objects. ArXiv, abs/2312.05278, 2023.
- [45] Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Jiao Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. ArXiv, abs/2303.16199, 2023.
- [46] Qiang Zhou, Zhibin Wang, Wei Chu, Yinghui Xu, Hao Li, and Yuan Qi. Infmllm: A unified framework for visual-language tasks. ArXiv, abs/2311.06791, 2023.
- [47] Alex Jinpeng Wang, Linjie Li, Kevin Qinghong Lin, Jianfeng Wang, Kevin Lin, Zhengyuan Yang, Lijuan Wang, and Mike Zheng Shou. Cosmo: Contrastive streamlined multimodal model with interleaved pre-training, 2024.
- [48] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world, 2023.
- [49] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Nan Ding, Keran Rong, Hassan Akbari, Gaurav Mishra, Linting Xue, Ashish Thapliyal, James Bradbury, Weicheng Kuo, Mojtaba Seyedhosseini, Chao Jia, Burcu Karagol Ayan, Carlos Riquelme, Andreas Steiner, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and Radu Soricut. Pali: A jointly-scaled multilingual language-image model, 2023.

- [50] Hao Zhang, Hongyang Li, Feng Li, Tianhe Ren, Xueyan Zou, Shilong Liu, Shijia Huang, Jianfeng Gao, Lei Zhang, Chun yue Li, and Jianwei Yang. Llava-grounding: Grounded visual chat with large multimodal models. ArXiv, abs/2312.02949, 2023.
- [51] Bin Lin, Zhenyu Tang, Yang Ye, Jiaxi Cui, Bin Zhu, Peng Jin, Junwu Zhang, Munan Ning, and Li Yuan. Moe-llava: Mixture of experts for large vision-language models. ArXiv, abs/2401.15947, 2024.
- [52] Yichen Zhu, Minjie Zhu, Ning Liu, Zhicai Ou, Xiaofeng Mou, and Jian Tang. Llava-phi: Efficient multi-modal assistant with small language model. ArXiv, abs/2401.02330, 2024.
- [53] Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. ArXiv, abs/2312.07533, 2023.
- [54] Shraman Pramanick, Guangxing Han, Rui Hou, Sayan Nag, Ser nam Lim, Nicolas Ballas, Qifan Wang, Rama Chellappa, and Amjad Almahairi. Jack of all tasks, master of many: Designing general-purpose coarse-to-fine vision-language model. ArXiv, abs/2312.12423, 2023.
- [55] Anton Razzhigaev, Arseniy Shakhmatov, Anastasia Maltseva, Vladimir Arkhipkin, Igor Pavlov, Ilya Ryabov, Angelina Kuts, Alexander Panchenko, Andrey Kuznetsov, and Denis Dimitrov. Kandinsky: An improved text-toimage synthesis with image prior and latent diffusion. In Yansong Feng and Els Lefever, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 286–295, Singapore, December 2023. Association for Computational Linguistics.

### A. Acknowledgements

The authors would also like to extend their deepest appreciation to SberAI and Sber Devices teams for providing help with multimodal data engineering, GigaChat large language models checkpoints, and invaluable advice in OmniFusion research (G. Novikov, S. Markov, F. Minkin, et al.).

### B. OmniFusion Examples

Below there are few examples of OmniFusion generations in English and Russian.

[Figure 10]

What kind of plant is shown in the background?

In the background of the image, there is a plant with large green leaves and white flowers. This plant is most likely a peace lily (Spathiphyllum wallisii), which is a popular indoor plant known for its ability to thrive in low-light environments.

What is the algorithm for caring for him?

The algorithm for caring for a peace lily is as follows:

- 1. Place the plant in an area with bright, indirect light.
- 2. Water the plant when the top inch of soil feels dry to the touch. 3. Fertilize the plant once a month during the growing season (spring and summer) with a balanced fertilizer. 4. Prune the plant as needed to maintain its desired shape and size. 5. Keep the plant away from cold drafts and direct sunlight, as these can damage the leaves. 6. Clean the leaves regularly to remove dust and debris, which can block the plant's ability to photosynthesize. 7. Repot the plant every 2-3 years as it grows, using a well-draining potting mix and a pot that is slightly larger than the current one.

[Figure 11]

Explain this meme.

The meme features a cartoon image of SpongeBob SquarePants holding a piece of paper with code written on it. The text above the image reads, "When you can't understand your code," and the text below the image says, "So you decide to move on." This meme is a humorous way of expressing the frustration and difficulty that many programmers face when trying to understand and debug their code. The decision to "move on" is often a necessary step in the process of problem-solving, as it allows the programmer to take a step back and approach the problem from a different angle or with a fresh perspective.

[Figure 12]

Tell me the breed of this dog

The dog in the image is a beagle.

How should one take care of this dog?

To take care of a beagle, it is important to provide them with regular exercise, mental stimulation, and a healthy diet. Beagles are energetic and playful dogs, so they need plenty of opportunities to run, play, and explore. Regular walks, trips to the park, and games of fetch can help satisfy their need for physical activity. Additionally, beagles are intelligent and curious dogs, so providing them with interactive toys, puzzles, and training sessions can help keep them mentally stimulated and prevent boredom. Lastly, a healthy diet is essential for a beagle's overall well-being. Beagles should be fed high-quality dog food that is appropriate for their age, size, and activity level. It is also important to avoid overfeeding, as beagles are prone to weight gain and obesity.

What is the origin of this breed?

The beagle is a breed of dog that originated in England. They were originally bred for hunting small game, such as rabbits and hares, and were known for their keen sense of smell and ability to follow a scent trail. Over time, the beagle has become a popular companion dog, known for their friendly and affectionate nature.

[Figure 13]

Describe what you see in the image in detail

The image presents a circle made up of 12 wine glasses, arranged in a circular pattern. The glasses are of varying sizes and shapes, creating a visually interesting composition. The background of the image is a solid blue color, providing a stark contrast to the yellow wine glasses. The arrangement of the glasses suggests a sense of unity and harmony, as if they are all part of a larger whole.

[Figure 14]

What should I cook from this for family dinner?

Based on the image, you could cook a variety of dishes for your family dinner using the items in the refrigerator. Some suggestions include: 1. Roast turkey: You have a whole turkey in the fridge, so you could roast it for a traditional Thanksgiving or holiday meal. 2. Salad: The lettuce, tomatoes, cucumbers, and peppers can be used to make a fresh and healthy salad. 3. Stir-fry: You can use the chicken, peppers, and onions to make a delicious stir-fry dish. 4. Soup: The chicken, vegetables, and broth can be combined to make a hearty and nutritious soup. 5. Sandwiches: You could use the turkey, cheese, and bread to make sandwiches for a quick and easy meal. These are just a few ideas, and you can mix and match the ingredients to create your own unique dishes. It's important to consider your family's preferences and dietary restrictions when planning the meal.

Sounds nice, but I don't have cheese

Sounds nice, but I don't have cheese If you don't have cheese, you can still make a variety of dishes using the items in the refrigerator. Some suggestions include: 1. Roast turkey: You have a whole turkey in the fridge, so you could roast it for a traditional Thanksgiving or holiday meal. 2. Salad: The lettuce, tomatoes, cucumbers, and peppers can be used to make a fresh and healthy salad. 3. Stir-fry: You can use the chicken, peppers, and onions to make a delicious stir-fry dish. 4. Soup: The chicken, vegetables, and broth can be combined to make a hearty and nutritious soup. 5. Sandwiches: You could use the turkey and bread to make sandwiches for a quick and easy meal. These are just a few ideas, and you can mix and match the ingredients to create your own unique dishes.

[Figure 15]

Как ты думаешь, к чему приурочен этот постер?

Я думаю, что этот постер приурочен к автомобильной гонке в Японии.

Какого числа состоится гонка?

5 апреля 2021 года.

Болид чьей команды изображен на постере?

Я думаю, что на постере изображена команда Ferrari.

Молодец!

Спасибо!

[Figure 16]

Что можно приготовить из этих продуктов?

Что можно приготовить из этих продуктов? На картинке представлены различные продукты питания, включая яйца, муку, сахар, масло, молоко, соль и другие. Их можно использовать для приготовления разнообразных блюд, таких как омлеты, блины, пироги, торты, печенье, соусы и многое другое.

Напиши рецепт блинов.

Рецепт блинов

Ингредиенты: 1 стакан муки; 1 стакан молока; 1 яйца;

- 1 ст. л. сахара; 1/2 ч. л. соды или разрыхлителя;
- 2 ст. л. подсолнечного масла.

Приготовление:

- 1. В миске смешайте муку, сахар и соду.
- 2. Добавьте яйцо и молоко, хорошо перемешайте.
- 3. Добавьте масло и снова перемешайте.
- 4. Разогрейте сковороду на среднем огне и смажьте её маслом.
- 5. Налейте небольшое количество теста на сковороду и равномерно распределите его по всей поверхности сковороды.
- 6. Жарьте блины до золотистого цвета с обеих сторон.
- 7. Готовые блины выложите на тарелку и подавайте на стол.

Приятного аппетита!

[Figure 17]

Из какой игры эти герои?

Эти герои являются персонажами из игры Minecraft.

[Figure 18]

Который час?

На картинке изображены часы, на которых отчетливо видны цифры 21:42.

[Figure 19]

Посмотри на изображение и ответь в следующей форме: 1. Название предмета 2. Алгоритм его сборки 3. Примерное время сборки в часах

- 1. Название предмета: велосипед.
- 2. Алгоритм его сборки: сначала нужно собрать раму, затем установить переднее и заднее колеса, после этого следует установить руль и тормоза.
- 3. Примерное время сборки в часах: 2-3 часа.

