# arXiv:2406.20076v5[cs.CV]10Mar2025

## EVF-SAM: Early Vision-Language Fusion for Text-Prompted Segment Anything Model

Yuxuan Zhang1* Tianheng Cheng1∗ Lianghui Zhu1 Rui Hu1 Lei Liu2 Heng Liu2 Longjin Ran2 Xiaoxin Chen2 Wenyu Liu1 Xinggang Wang1† 1 School of EIC, Huazhong University of Science & Technology 2 vivo AI Lab

|widespread<br><br>capabilifor SAM re-<br><br>SAM for observe that<br><br>mechanism<br><br>factors. usion (EVF)<br><br>, which<br><br>and effion Large<br><br>RefCOCO/val<br><br>RefCOCO/testA<br><br>RefCOCO/testB<br><br>RefCOCO+/val<br><br>RefCOCO+/testA<br><br>RefCOCO+/testB<br><br>RefCOCOg/val<br><br>RefCOCOg/test<br><br>LISA (avg: 67.9) PixelLM (avg: 69.2) PixelLLM (avg: 72.3) UniRef++-L (avg: 73.4) GLaMM (avg: 75.6) U-LLaVA (avg: 75.9)<br><br>UNINEXT-H (avg: 76.6) PSALM (avg: 77.1) EVF-SAM (avg: 79.0)<br><br>|
|---|

##### Abstract

Segment Anything Model (SAM) has attracted attention for its superior interactive segmentation ties. Despite this, the potential of text prompts f mains largely unexplored. In this paper, we investigate what text prompt encoders are beneficial for adapting Referring Expression Segmentation (RES) and o

(1) multimodal prompts and (2) the early-fusion incorporating text-to-image attention are crucial We introduce the Early Vision-Language Fus framework and further formulate the EVF-SAM is simple but demonstrates superior effectiveness ciency compared to mainstream methods based Language Models (LLM). We also successfully integrate segmentation data from diverse tasks into an unified hybrid dataset to conduct joint training. The well-designed data strategies resolve issues such as semantic conflict and ambiguity. The proposed EVF-SAM based on BEIT-3 obtains state-of-the-art performance on RES tasks (79.0 in average on RefCOCO/+/g), and extends RES capability to various granularities, e.g., semantic-level RES and part-level RES. EVF-SAM maintains the model size with only 1.32B parameters, considerably fewer than the mainstream LLM-based methods of over 7B parameters. Code and models will be made publicly available.

Figure 1. EVF-SAM achieves competitive performance among various benchmarks for referring expression segmentation.

abilities like point-prompted and box-prompted segmentation, it is a pity that the text-prompted segmentation ability remains conceptual. We retrospect this task to Referring Expression Segmentation (RES). RES focuses on the solution to predicting the segmentation mask according to the text description given by users, which enjoys several explorations by some traditional models [4, 9, 13, 14, 27, 34, 38, 39, 52, 60, 61, 67, 69, 71], and is broadened by some Large Language Models (LLM) [25, 44, 47, 50, 62, 65, 68, 75].

The key challenge lies in empowering SAM with language understanding ability for segmentation according to text prompts, e.g., referring expression segmentation. Fig. 3 summarizes previous end-to-end works which explore the text-prompted abilities of SAM: (a) SAM with text encoder: leverage an off-the-shelf text encoder(e.g., CLIP [45]) to extract text features and feed into SAM. However, text encoders fail to extract positional information from unimodal input, leading to poor performance. (b) SAM with LLM: fine-tune a Large Language Model (LLM) to auto-regressively generate embeddings for the target object. These methods are under the late-fusion approach, where the image features extracted by the image encoder are further integrated within the LLM. However, the text-

##### 1. Introduction

Segment Anything Model (SAM) [24] brings interactive segmentation paradigm to public view. Well-trained on the SA-1B dataset, SAM achieves stunning performance and quickly becomes popular as a vision foundation model for object localization and beyond. Various SAM variants [22, 64, 74, 76] have been explored, achieving better efficiency or higher precision. Despite SAM’s surprising

*Equal contribution. †Corresponding author (xgwang@hust.edu.cn).

###### 1/32 layer 16/32 layer 32/32 layer

###### Attention visualization

[Figure 1]

[Figure 2]

[Figure 3]

| |
|---|

[Figure 4]

img

img

img

LISA (late-fusion)

text

text

text

img text

img text

img text

'vase on clock'

###### 1/24 layer 12/24 layer 24/24 layer Attention visualization

[Figure 5]

[Figure 6]

[Figure 7]

| |
|---|

[Figure 8]

img

img

img

Ours (early-fusion)

text

text

text

img text

img text

img text

'vase on clock'

- Figure 2. Comparison between late-fusion and early-fusion. We visualize the attention map of representative late-fusion method (i.e., LISA [25]) and early-fusion methods (i.e., EVF-SAM) among different layers. The Y axis represents the ‘Query’, the X axis represents the ‘Key’. We use red, blue and orange boxes to highlight the ‘image to text attention’, ‘text-to-image attention’, and targeted segmentation object. The figures demonstrate that the ‘text-to-image attention’ is more crucial for Referring Expression Segmentation task, while latefusion methods ignore it. To compensate for this, we propose the early-fusion method for high-quality text-guided visual features.

to-image attention, which is proved essential (in Fig. 2) for Referring Expression Segmentation task, is neglected in LLMs. Additionally, these LLM-based models are often computationally expensive and hard to train.

To overcome shortcomings of existing technical approaches, we propose: (1) multimodal prompts that incorporate both text and image outperform text-only prompts; (2) the early-fusion architectures, which incorporate textto-image attention, exhibit significant advantages over textonly encoders and Large Language Models. Motivated by the above observations, we realize the text-promptd SAM using an Early Vision-Language Fusion encoder (EVF), and propose EVF-SAM, as shown in Fig. 3 (c). The proposed EVF-SAM is a simple yet effective framework built upon the off-the-shelf foundation models, comprising an well-pretrained Vision-Language Model (VLM) (i.e., BEIT-3 [59]), to generate prompt embeddings for SAM. Besides, EVF-SAM does not include elaborate designs or modules and is easy for scaling to larger models. Moreover, EVF-SAM reduces huge amounts of parameters (e.g., -82% parameters compared to LISA [25]), with better performance outperforming previous attempts with Large Language Models [25, 47, 68], as shown in Fig. 1.

Besides the simple architecture, we highlight the importance of well-managed training data. We construct a hybrid dataset with publicly available datasets by including RES data (e.g., RefCOCO/+/g [21, 41, 42, 72]), semantic-level data (e.g., ADE20K [77, 78]), instance-level data (e.g., Objects365 [51]), and part-level data (e.g., Pascal-Part [5]). To effectively train the model on the diverse collection of hybrid dataset, we carefully design several data strategies for our EVF-SAM. To be specific, we propose to use a special token ‘[semantic]’ to solve the semantic con-

flict problem, and use some merging/filtering principles to solve the ambiguity problem. These additional data not only boost EVF-SAM at Referring Expression Segmentation (RES) tasks (78.0 cIoU to 79.0 cIoU in average on RefCOCO/+/g [21, 41, 42, 72]), but also extend EVF-SAM’s capabilities at various granularity (e.g., semantic-level RES, part-level RES).

Our main contributions can be summarized as follows:

- • We propose a novel Referring Expression Segmentation (RES) framework by introducing an Early VisionLanguage Fusion encoder (EVF). The EVF framework demonstrates superior effectiveness and efficiency compared to mainstream late-fusion frameworks, e.g., RES via Large Language Models (LLM). The EVF framework emphasizes the multimodal prompt and the earlyfusion mechanism, which are crucial for the RES task, as highlighted by extensive experiments and visualizations. Building upon the EVF framework, we propose EVF-SAM.
- • We jointly train EVF-SAM on a hybrid dataset, extending the model’s capability to various granularities of RES tasks, e.g., semantic-level RES, part-level RES. Our proposed data strategies successfully map various segmentation datasets to a unified representation, resolving distribution problems such as semantic conflict and ambiguity.
- • EVF-SAM achieves state-of-the-art performance on the RES tasks (i.e., 79.0 cIoU in average on RefCOCO/+/g). Meanwhile, EVF-SAM reduces the parameter count to 1.3B, a substantial decrease compared to the LLM-based models of more than 7B parameters.

[Figure 9]

[Figure 10]

[Figure 11]

SAM

SAM SAM

LLM

Text Enc.

EVF

Image Enc.

SAM

[Figure 12]

[Figure 13]

[Figure 14]

“zebra top left” “zebra top left” “Can you segment the zebra top left in

Late Fusion

V

the picture?”

V L

(b) Late Fusion

(a) SAM with Text Encoder

(b) SAM with Large Language Model (c) SAM with Early V-L Fusion

- Figure 3. Comparisons of different Text-prompted SAM. (a) A natural idea to support text prompts is to use an off-the-shelf text encoder to generate text embeddings for SAM [24, 30]. (b) Several works [25, 47, 68] adopt Large Language Models (LLM) to generate prompt embeddings for SAM in an auto-regressive manner. (c) Our proposed EVF-SAM exploits an effective early vision-language fusion encoder for text-prompted SAM with higher performance and fewer parameters.

###### V L

currently lacks language understanding abilities and it’s infeasible to directly use text prompts for Referring Expression Segmentation (RES). Several works [30, 49, 76] employ multi-stage architectures (e.g., employing grounded detectors) to incorporate SAM with language interpretation capability. However, multi-stage models suffer from noise accumulation and difficulty in optimization, and is out of our concern. Under end-to-end manners, vanilla SAM proposes employing a CLIP text encoder to cast linguistic feature into SAM, as shown in Fig. 4 (a). Some advanced works propose using late-fusion manner, by employing offthe-shelf unimodal encoders to pre-extract features independently, followed by a fusion structure, as shown in Fig. 4 (b). RefSAM [30] employs a lightweight cross-modal MLP to project the text embeddings into SAM’s prompt encoder LISA [25, 68] employs a LLM (LLaVA [36]) to extract multimodal embeddings for SAM through the auto-regressive decoder. The aforementioned methods either suffer from poor performance or are computationally expensive. In this work, our EVF-SAM is in ‘early-fusion’ manner, which is both effective and efficient, as shown in Fig. 4 (c).

SAM

SAM

SAM

Late Fusion

Early Fusion

###### V L

L

V L

(a) Vanilla (b) Late Fusion (c) Early Fusion

- Figure 4. Architectural explorations for text-prompted SAM. ‘L’ and ‘V’ denote the text encoder and vision encoder. We mainly explore three schemes: (a) vanilla baseline with a simple text encoder, (b) multimodal inputs with a late-fusion, i.e., concatenation, and (c) multimodal inputs with early-fusion

##### 2. Related Work

###### 2.1. Text-Prompted Segment Anything Models

Segment Anything Model. SAM [24] is an interactive segmentation model capable of predicting masks based on various types of prompts (points, boxes, and masks). Trained on a large-scale dataset, SAM demonstrates strong generalization capability for segmenting diverse common objects. Advanced works address the massive computation cost of SAM or broaden SAM’s capability. EfficientSAM [64] distills the image encoder of SAM, achieving comparable performance with significantly fewer parameters. Fast-SAM [76], leveraging the YOLOv8 [18] architecture, achieves a 50× speedup for inference. SAMHQ [22] utilizes low-level features from the image encoder to address the segmentation quality of SAM. MatteAnything [70] make SAM capable of outputting detailed matting masks. Inpaint-Anything [73] integrate SAM with AIGC models to produce objects-removing tools. VRPSAM [56] enables visual prompt for referring segmentation. Edit Everything [63] allows users to edit images using simple text instructions.

###### 2.2. Referring Expression Segmentation

Referring Expression Segmentation (RES) is a multimodal segmentation task requiring accurate pixel-wise segmentation and fine-grained language understanding.

RES via Text Encoders. Prevalent methods tend to leverage transformer-based text encoders (e.g., BERT [7], CLIP [45]) to encode texts into embeddings as guidance for segmentation. RefTr [26] uses a vision-language encoder to fuse multimodal features and regresses the box and mask with a carefully designed query processor. LAVT [69] leverages a hierarchical ViT [10] to perform languageaware visual encoding. CRIS [60] designs a vision-

Text-prompted SAM. Although SAM excels in visualbased segmentation tasks with box/point/mask prompts, it

###### SAM

[Figure 15]

[Figure 16]

Mask Decoder

[Figure 17]

!

#### ❄

[Figure 18]

SAM Image Encoder

Prompt Encoder

[Figure 19]

!

Prompt Embeddings

EVF Resize to 224x224

[Figure 20]

!

Trainable Modules

Multimodal Encoder

! !

[Figure 21]

Projector

[Figure 22]

“ zebra top left ”

[Figure 23]

❄ Frozen Modules

[CLS] Token

- Figure 5. The overall architecture of EVF-SAM. The proposed EVF-SAM maintains the original architecture of SAM and keeps the weights of the SAM Image Encoder frozen. EVF-SAM exploits the Multimodal Encoder with Early Vision-Language Fusion (EVF) to encode both text prompts and the low-resolution input image (which is resized to 224 × 224). Then the output [CLS] token is projected as prompt embeddings and fed into the prompt encoder of SAM for generating the referring segmentation results.

language decoder to merge CLIP features, propagating finegrained semantic information from textual representations to each pixel-level activation. PolyFormer [38] follows the encoder-decoder structure, employing a transformer decoder to generate regression results. Novel methods pay attention to being compatible with multiple tasks to formulate a uniform model. UNINEXT [67], UniRef++ [61] and UniLSeg [39] employ similar frameworks but focus on utilizing datasets from different fields to empower their generalization capability. Although these traditional models are usually lightweight and achieve fine performance, they fail to integrate with large-scale foundation models(e.g., SAM[24], LLaVA[36]), thereby struggling to keep pace with the trend of increasingly extensive pre-training.

perform better for encode text prompts for referring image segmentation.

###### 2.3. Late-fusion and Early-fusion

Late-fusion models. Late-fusion denotes applying off-theshelf unimodal encoders(e.g., CLIP [45]) to pre-extract features independently, followed by a fusion structure. This leverages the strong prior knowledge gained during the encoders’ pre-training, requiring only minimal adjustment (e.g., simple projection layers) to align the multimodal feature space. This is particularly advantageous for models with complex mechanisms (e.g., causal attention), easing the convergence and gain the performance. Representative late-fusion methods are Multimodal Large Language Models (MLLM). LLaVA [36]) employs the off-the-shelf CLIP image encoder (CLIP [45]) to pre-extract visual features. The visual features is projected to a Large Language Model (i.e., LLaMA [57]) using a simple linear layer. Furthermore, Qwen-VL [1] utilizes finer-designed training recipe and carefully managed data, building a robust MLLM. Mipha [79] explores the lightweight architecture, following LLaVA’s recipe to finetune phi [29]. There are also attempts attempts to decouple causal models from late fusion, e.g., Fuyu [2], EVE [8]. But they exhibit significantly slower convergence and poorer performance compared to late-fusion MLLMs.

RES via Large Language Models. In the context of the rapid development of Large Language Models [1, 36, 54, 55] (LLM), a number of works [25, 44, 50, 65, 68, 75] have leveraged these models to encode expression texts for referring expression segmentation tasks. LISA [25, 68] finetunes LLaVA [36] to make it answer questions related to segmentation with a fixed template like ‘It is [SEG].’, where the hidden embeddings at the place of special token [SEG] will be seen as multimodal features. PixelLM [50] extends LISA by building a segmentation codebook to enable multiobject segmentation. PixelLLM [66] empowers the visionlanguage model to take locations (e.g., a set of points or boxes) as either inputs or outputs. PerceptionGPT [44] proposes an end-to-end architecture by designing a perceptionenhanced vision language model, which improves inference efficiency. u-LLaVA [65] supports multi-task, e.g., object grounding. PSALM [75] imports mask tokens to LLM input for better performance. However, LLMs neglect the text-to-image attention, which is proved essential for RES in our later experiments, leading to their sub-optimal performance. Additionally, they tend to adopt heavy architectures, especially the LLMs, leading to a heavy computation burden for downstream applications. In contrast, we find that lightweight early vision-language fusion models

Early-fusion models. Early-fusion methods proposes using shallow unimodal embedding layers to project features, and then using deep fusion layers to conduct multimodal interaction. The early integration of fusion make both modalities attach to dense information of the other one, so the extraction of multimodal feature is advantageously effective and efficient. This is crucial for tasks strongly relied on the enhance of all included modalities. Representative early-fusion methods are Vision-Language Models (VLM) with bidirectional attention. ViLT [23] conducts detailed comparisons to prove the effectiveness and efficiency of early-fusion methods over late-fusion meth-

- Table 1. Comparison of cIoU on different benchmarks between our proposed EVF-SAM and previous state-of-the-art methods. Bold: the best results. Underline: the second-best results. AVG represents the average metric across the eight RefCOCO-series benchmarks. We abbreviate the datasets: COCO (C) [33], RefCOCO (RC) [21, 41, 42, 72], Objects365 (O) [51], Video segmentation datasets (V), ADE20K (A) [77, 78], COCO-Stuff (CS) [3], PACO-LVIS (PL) [46], PASCAL-Part (PP) [5], GranD (G) [47], PASCAL VOC2010 (PV) [11], MUSE (M) [50], gRefCOCO (gRC) [35], COCO-Interactive (CI) [75], FSS-1000 (F) [28], SA-1B (SA) [24], PartImageNet (PIN) [12], HumanParsing (HP) [31, 32], GoldG (GG) [20].

Method

Text Prompt Encoder

SAM? Training Data

RefCOCO RefCOCO+ RefCOCOg

AVG

val testA testB val testA testB val test LAVT [69] BERT-B (104M) ✗ RC, gRC 72.7 75.8 68.8 62.1 68.4 55.1 - - PolyFormer-L [38] BERT-B (104M) ✗ RC, gRC 76.9 78.5 74.8 72.2 75.7 66.7 71.2 71.2 73.4 UNINEXT-H [67] BERT-B (104M) ✗ O, C, RC, V 82.2 83.4 81.3 72.5 76.4 66.2 74.4 76.4 76.6 UniLSeg-100 [39] CLIP-B (63M) ✗ SA, RC, gRC 81.7 83.2 79.9 73.2 78.3 68.2 - - UniRef++-L [61] BERT-B (104M) ✗ RC, F, V 79.1 82.1 77.5 68.4 74.0 61.5 71.4 72.8 73.4 LISA [25] Vicuna (7B) ✓ A, CS, RC, PL, PP 74.1 76.5 71.1 62.4 67.4 56.5 66.4 68.5 67.9 PixelLM [50] LLaMA2 (13B) ✗ A, CS, RC, PL, M 73.0 76.5 68.2 66.3 71.7 58.3 69.3 70.5 69.2 PixelLLM [66] T5-XL (3B) ✓ RC, GG 76.9 78.5 74.4 69.2 72.1 64.5 70.7 72.4 72.3 GLaMM [47] Vicuna (7B) ✓ G, RC 79.5 83.2 76.9 72.6 78.7 64.6 74.2 74.9 75.6 u-LLaVA [65] Vicuna (7B) ✓ A, CS, RC, PL, PV 80.4 82.7 77.8 72.2 76.6 66.8 74.8 75.6 75.9 PSALM [75] Phi-1.5 (1.3B) ✗ C, RC, CI 83.6 84.7 81.6 72.9 75.5 70.1 73.8 74.4 77.1 EVF-SAM BEIT-3 (673M) ✓ RC 82.1 83.7 80.0 75.2 78.3 70.1 76.8 77.4 78.0 EVF-SAM BEIT-3 (673M) ✓ RC, O, A, PP, PIN, HP 82.4 84.2 80.2 76.5 80.0 71.9 78.2 78.3 79.0

- Table 2. Motivation analysis. Both CLIP and BEIT-3 are of Large scale, with comparable numbers of parameters. Specifically, CLIP has a total parameter count of 428M, while BEIT-3 totals 673M parameters. ‘L’ and ‘V’ denote the linguistic input and the vision input. The reported metrics are evaluated on RefCOCO/testA, except that the metric of LLaVA [36] is borrowed from LISA-7B [25].

CLIP (L)

CLIP (L+V)

BEIT-3 (L)

BEIT-3 (L+V)

LLaVA (L+V) cIoU 63.4 67.9 65.1 83.7 79.1

ods (e.g., Pixel-BERT [15]) in terms of multimodal feature extraction. BEIT-3 [59] project visual signal to language space and performs masked language modeling pretraining, building a powerful backbone. ONE-PEACE[58] extends this approach to incorporate audio modalities, along with utilizing a larger volume of pre-training data and increasing the model’s parameter size.

In this work, we proposed that ‘early-fusion’ (i.e., EVFSAM) is prioritized over ‘late-fusion’ (i.e., ‘RES via text encoder’, ‘RES via LLMs’, introduced in Sec. 2.2) in the Referring Expression Segmentation task.

- 3. Method

coder, which converts points, boxes, and masks into embeddings that incorporate positional information. As a result, when we are dealing with the Referring Expression Segmentation (RES) task based on SAM, the key challenge is to get the positional information from the input referring prompt. SAM [24] explored text-prompted segmentation with a CLIP text encoder, as illustrated in Fig. 3 (a). However, it is infeasible for the unimodal text encoder to extract positional information. Our reproduced metrics in Tab. 2 shows that CLIP-prompted SAM achieves 63.4 cIoU, which is inferior to well-defined baselines. Moreover, we observe performance improvements after using multimodal prompts, i.e., 63.4 v.s. 67.9 for CLIP and 65.1 v.s. 83.7 for BEIT-3, proving that multimodal referring information is effective for text-prompted SAM.

Early-fusion outperforms late-fusion. Existing works employ pre-encoded text feature (e.g., RES via text encoder) or pre-encoded image feature (e.g., RES via LLM), as illustrated in Sec. 2.2, demonstrating the ‘late-fusion’ manner. We visualize the attention map within multimodal fusion layers in Fig. 2, and find shortcomings within ‘latefusion’ methods. The attention map W is calculated by W = clamp10(2 × sigmoid(QKT/

√

d)), where Q denotes query, K denotes Key, and clamp(·) is a clipping function for a more distinct comparison We firstly observe enhanced image-to-text attention at the first few layers, indicating that ‘RES via text encoder’ methods, which lack image-to-text attention, may struggle with accurately interpreting text descriptions. Moreover, the ‘text-to-image attention’ takes the lead at later layers, showing the inefficiency of ‘RES via LLM’ methods, which results in a limited multimodal perception of visual features. The attention visualization example in Fig. 2 further illustrates the imprecise object lo-

###### 3.1. Motivation: Early Vision-Language Fusion.

We investigate how to encode text prompts for SAM in this section. We start by using a text encoder following SAM’s trial, and extend to late-fusion and early-fusion, as shown in Fig. 3. We conduct preliminary experiments on RefCOCO (testA), shown in Tab. 2.

Multimodal referring information is crucial. The interactive capability of SAM [24] hinges on its prompt en-

calization capabilities of late-fusion methods. In contrast, our proposed ‘early-fusion’ methods which utilize shallow embedding layers and deep fusion layers (e.g., [23, 59]) is effective and efficient at both ‘image-to-text attention’ and ‘text-to-image attention’, overcoming the shortcomings of ‘late-fusion’. Experiment metrics in Tab. 3.1 further support our motivation. SAM with CLIP-text and CLIP-image, which implements a straightforward ‘late-fusion’ approach through concatenation, achieves a cIoU of 67.9. Furthermore, SAM with LLaVA, representing a more complex form of ‘late-fusion’, reaches 79.1 cIoU but requires 7B parameters. Conversely, SAM with BEIT-3, exemplifying our ”early-fusion” approach, achieves a cIoU of 83.7, surpassing all late-fusion methods while utilizing just 1.3 billion parameters.

###### 3.2. Architecture

Fig. 5 illustrates the overview of EVF-SAM, which is a simple yet effective framework with three modules: Multimodal Encoder, Projector, and Segment Anything Model (SAM).

Multimodal Encoder. The Early Vision-Language Fused encoder adopts the input image and text and outputs fused multimodal embeddings. In EVF-SAM, we mainly adopt BEIT-3 [59] as the Multimodal Encoder, which formulates a multi-way transformer. The text is tokenized by XLMRobertaTokenizer [6] while the image is resized to 2242 and patched by a 1/16 convolution layer. Within each block of the encoder, the image and text tokens will be fused in the attention block and then fed into separate Feed-Forward Networks (FFN). We follow ViT [10] to retrieve the [CLS] token as the output multimodal embeddings.

Adapted prompt encoder for SAM. In EVF-SAM, we maintain the architecture of the image encoder and mask decoder of SAM, while extending the prompt encoder to further gather the embeddings from the Multimodal Encoder. Specifically, the original prompt encoder encodes point or box prompts to sparse embeddings of RB×N×D, where B is the batch size, N is the number of points/boxes, and D is the embedding dimension. In EVF-SAM, the projected multimodal embeddings of RB×1×D from the Multimodal Encoder will be concatenated to sparse embeddings (of RB×0×D, as there is no points/boxes inputs) and then fed into the mask decoder.

###### 3.3. Data strategy

Since publicly available open-source RES data is limited, a simple thought is that we use datasets of other segmentation sub-tasks. To be specific, we include Instance Segmentation (Objects365 [51]), Semantic Segmentation (ADE20K [77, 78]) and Part Segmentation (PascalPart [5], HumanParsing [32], PartImageNet [12]) to boost the performance of our EVF-SAM, as well as to extend the

Table 3. Comparison of the model efficiency of EVF-SAM with LLM-based RES methods. Training time means the total time cost of training on RefCOCO [21, 72]. Evaluation time means the average time cost of forward during evaluation on refcoco/unc/testA.

Methods Training time Evaluation time cIoU LISA-7B-LORA 54.7h 0.772s 76.5 EVF-SAM 19.7h (-64%) 0.283s (-63%) 84.2 (+10%) Effi-EVF-SAM-S 8.2h (-85%) 0.129s (-83%) 83.5 (+9%)

model’s capability granularity.

However, segmentation data from different tasks exhibit varying distributions, making it challenging to effectively integrate them for joint training. We carefully analyze the semantic conflict between semantic segmentation data and RES data, and the ambiguity problem between instance segmentation data and RES data in Sec. A.1. Furthermore, we detailedly explain our data strategies in Sec. A.2, including applying a special token ‘[semantic]’ for solving semantic conflict, and using filtering/merging principles to solve the ambiguity problem.

Powered by our data strategies, we successfully integrate segmentation data from diverse tasks into an unified hybrid dataset. Upon joint training with the hybrid dataset, we observe the performance gain of EVF-SAM from average 78.0 cIoU to 79.0 cIoU, as shown in Tab. 1. Furthermore, EVFSAM can handle various granularities of text-prompted segmentation such as semantic-level (e.g., stuff, multi-object) and part-level (e.g., body-part) segmentation. Detailed definition, metrics and visualizations are shown in Sec. A.3 and Sec. A.4, separately.

##### 4. Experiments

###### 4.1. Datasets and Metrics

Datasets. We mainly conduct the experiments on RefCLEF [21], RefCOCO, RefCOCO+ [21, 72], and RefCOCOg [41, 42]. Specifically, RefCOCO+ excludes geometric expression (e.g., ‘on the left’), while RefCOCOg contains longer expressions (8 words per prompt on average). Among different splits of testing datasets, ‘testA’ is human-centric, while ‘testB’ aims for common objects.

To produce our best result, we employ multi-task unified training, as illustrated in Sec. 3.3. Objects365 [51], ADE20K [77, 78], Pascal-Part [5], HumanParsing [32] and PartImageNet [12] are additionally included.

Metrics. The gIoU and the cIoU are the most commonly calculated metrics on referring expression segmentation benchmarks. The gIoU is the average intersection-overunions (IoU) among all images in the test datasets, while the cIoU is the cumulative intersection over the cumulative union. If not specifically declared, we follow previous works and report the cIoU as the main metric.

- Table 4. Ablation on fusion methods. We evaluate the performance of using different pre-trained Multimodal Encoders in EVF-SAM, e.g., CLIP from OpenAI [45] or OpenCLIP [17]. Li denotes the i-th layer in the BEIT-3 model (totally 24 layers for BEIT-3-Large). Half of the layers are activated to assess the impact of the modality fusion stage on model performance. †: pre-trained models provided by OpenAI. ‡: pre-trained models provided by OpenCLIP.

RefCOCO RefCOCO+ RefCOCOg

Encoder Params Text Image Modality Fusion

AVG val testA testB val testA testB val test

CLIP variants.

CLIP-Large† 123M ✓ - 61.0 63.4 59.9 43.1 45.9 40.6 48.9 49.6 51.6 CLIP-Large† 428M ✓ ✓ Late (Concat) 67.4 68.9 64.4 50.5 54.6 46.7 55.1 56.2 58.0 CLIP-Large‡ 123M ✓ - 60.8 63.2 59.0 42.9 46.4 39.2 49.2 50.5 51.4 CLIP-Large‡ 428M ✓ ✓ Late (Concat) 66.1 67.8 63.1 49.8 51.9 44.1 54.1 55.0 56.5 CLIP-Huge‡ 302M ✓ - 61.7 64.2 60.1 44.2 47.8 40.2 49.6 50.9 52.3 CLIP-Huge‡ 986M ✓ ✓ Late (Concat) 66.3 68.2 64.3 49.8 53.5 45.1 55.4 56.7 57.4

Early-fused vision-language models.

ViLT 133M ✓ - 61.0 63.0 60.0 42.5 45.4 39.5 49.3 49.5 51.3 ViLT 136M ✓ ✓ Late (Concat) 61.4 64.0 59.6 42.8 46.4 40.1 49.5 50.0 51.7 ViLT 136M ✓ ✓ Early 73.9 75.3 70.9 61.1 64.4 55.2 65.1 66.8 66.6 BEIT-3-Large 370M ✓ - 61.6 65.1 59.4 44.0 47.6 40.6 49.5 50.8 52.3 BEIT-3-Large 673M ✓ ✓ Late (Concat) 67.7 70.2 65.4 51.1 55.0 46.9 57.2 57.0 58.8 BEIT-3-Large 673M ✓ ✓ Early (L1 ∼ L12) 80.6 82.2 78.8 72.4 75.7 66.7 73.7 75.0 75.6 BEIT-3-Large 673M ✓ ✓ Early (L1 ∼ L24) 82.1 83.7 80.0 75.2 78.3 70.1 76.8 77.4 78.0

###### 4.2. Implementation Details

cific datasets but performed poorly on others, our approach achieves consistently high scores across multiple datasets. This demonstrates that our method does not rely on specific linguistic expressions (e.g., geometric words) for prediction but rather possesses strong generalization capabilities.

Unless specified, we initialize the proposed EVF-SAM with the public weights of SAM-ViT-Huge 1 [24] and BEIT-3Large 2 [59]. All models are trained on 4 NVIDIA L40s GPUs with mixed precision. We adopt DeepSpeed [53] with ZeRO-2 for model parallel to optimize memory consumption. During training, the batch size of each GPU is 16 and we use gradient accumulation for 2 steps, therefore, the total batch size per iteration is 128. We adopt AdamW [40] optimizer and set the initial learning rate to 1e-4 with a linear-decay schedule. We train all models for 15k iterations and use the binary cross-entropy loss (BCE) and dice loss (the weight of both losses is 1.0).

Besides metrics on RefCOCO/+/g, we also evaluate the model’s performance on semantic-level referring expression segmentation, part-level referring expression segmentation and reasoning segmentation. Metrics are shown in Sec. A.3, Sec. A.4 and Sec. B.1.

###### 4.4. Efficiency analysis

To prove the efficiency of our EVF-SAM over previous LLM-based models, we present comparisons of training and evaluation time in Tab. 3. We take LISA [25], which employs LLaVA [36] as the multimodal extractor, for comparison. We follow LISA’s reported setting to conduct 10k steps of training with a global batch of 160. Considering that different datasets contain descriptions of various lengths (e.g.,

###### 4.3. Main Results

We mainly report the cIoU metric of RefCOCO/+/g benchmarks and compare our proposed EVF-SAM with recent state-of-the-art methods in Tab. 1. The upper part of Tab. 1 presents traditional methods based on text encoders. Despite their advantages in terms of fewer parameters or taskspecific architectures, these methods either achieve less competitive results or require vast amounts of data and computational resources due to their lack of integration with foundation models. The methods listed in the lower portion of Tab. 1 are based on Large Language Models (LLMs), achieving more competitive performance but at the expense of significantly increased parameter size. Our EVF-SAM achieves the highest average cIoU score across all RES benchmarks, using only readily available data, manageable computation resources, and moderate parameter size. Notably, unlike previous methods that excelled on spe-

- 3.6 words per sentence for RefCOCO, 8.4 words per sentence for RefCOCOg), we conduct experiments only on RefCOCO to avoid sample bias. Our EVF-SAM can reduce 64% to 85% training time cost compared to LLM-based models (i.e., LISA). We also record the evaluation time on RefCOCO/testA (750 images with 5556 referring instances) per image. Our EVF-SAM can reduce 63% to 83% inference time cost, too.
- 4.5. Ablation Study

In this section, we conduct experiments to investigate the vision-language models for text-prompted SAM and study the effects of the designs of the proposed EVF-SAM. A few ablations are listed in the supplementary. Unless specified, we mainly report the cIoU on testA of RefCOCO.

- 1SAM: https : / / github . com / facebookresearch / segment-anything
- 2BEIT-3: https://github.com/microsoft/unilm

###### Multimodal Encoder and fusion methods. In Tab. 4,

- Table 5. Comparison of effects of different foundation models. The main bottleneck of EVF-SAM lies in the the Multimodal Encoder. Our proposed architecture can perform well on various SAM-like methods.

Multimodal Encoder SAM Params

RefCOCO RefCOCO+ RefCOCOg

AVG val testA testB val testA testB val test

CLIP-Large SAM-ViT-H 1.08B 61.0 63.4 59.9 43.1 45.9 40.6 48.9 49.6 51.6 ViLT SAM-ViT-H 783M 73.9 75.3 70.9 61.1 64.4 55.2 65.1 66.8 66.6 BEIT-3-Base SAM-ViT-H 863M 78.9 80.6 75.3 69.8 74.2 63.0 71.6 72.9 73.3 BEIT-3-Large Efficient-SAM-S 700M 82.5 83.5 80.4 75.4 77.9 70.2 76.1 77.1 77.9 BEIT-3-Large SAM-ViT-H 1.32B 82.1 83.7 80.0 75.2 78.3 70.1 76.8 77.4 78.0 BEIT-3-Large SAM-2-L 898M 82.7 84.1 80.0 76.3 80.1 71.8 77.0 78.4 78.8

- Table 6. Ablation on trainable modules. We mainly evaluate the effects of fine-tuning or freezing the Multimodal Encoder, the prompt encoder and mask decoder of SAM. ‘✓’ denotes trainable, while ‘∗’ denotes frozen.

fine-tuning the Multimodal Encoder is crucial and it adapts the Multimodal Encoder to encode text and image inputs to multimodal representation for referring image segmentation. Notably, EVF-SAM can achieve competitive results with all modules of SAM kept frozen, and it can be seamlessly regarded as a strong extension for the original SAM, which simultaneously supports text prompts, box prompts and point prompts. Tab. 6 shows fine-tuning the prompt encoder and mask decoder of SAM brings significant improvements.

Multimodal Enc. Prompt Enc. Mask Dec. cIoU

∗ ∗ ✓ 21.2 ✓ ∗ ∗ 82.9 ✓ ∗ ✓ 83.3 ✓ ✓ ✓ 83.7

we explore the effects of different Multimodal Encoders, e.g., CLIP, ViLT [23], and BEIT-3, and fusion methods, e.g., late fusion or early fusion. As shown in Tab. 4, using a text-only encoder in EVF-SAM obtains limited segmentation performance on RefCOCO. Using Multimodal Encoders with both image and text inputs remarkably improves 4.5 cIoU, 4.6 cIoU, 4.0 cIoU, 1.0 cIoU, and 4.5 cIoU for CLIP-Large† (OpenAI 3), CLIP-Large‡ (OpenCLIP 4), CLIP-Huge‡ (OpenCLIP), ViLT, and BEIT-3, respectively. It demonstrates the superiority of using multimodal prompts (text and input image) and showcases that the image embeddings will also provide useful guidance for SAM to segment objects accurately. We further evaluate the effects of early fusion on ViLT and BEIT-3, which adopts modality fusions in all self-attention layers. Specifically, we adopt two settings for BEIT-3 to analyze, e.g., fusions among former 12 layers (L1 ∼ L12), and fusions among all layers (L1 ∼ L24). Tab. 4 indicates that BEIT-3 with early fusion (fusing former 12 layers or fusing all 24 layers) significantly improves compared to late fusion or using text only. In addition, ViLT with early fusion also achieves 11.1 cIoU improvements compared to the baseline with textonly prompts, showing the effectiveness of early fusion and multimodal inputs for prompting SAM. Therefore, Tab. 4 demonstrates that (1) Multimodal Encoder with the input image and text and (2) early fusions between the image and text encoder are much effective for text-prompted SAM.

Effects of Different Foundation Models. In Tab. 5, we explore the effects of using different foundation models in EVF-SAM. For the Multimodal Encoder, we adopt CLIP-Large (only text encoder), ViLT, BEIT-3-Large, and BEIT-3-Base. We also modify EVF-SAM with EfficientSAM [64] and SAM-2 [48] to formulate a lighter or better version. As shown in Tab. 5, EVF-SAM with BEIT-3-Base brings a severe performance drop which indicates a better Multimodal Encoder leads to better prompts for SAM. Remarkably, Tab. 5 shows a negligible difference between Efficient-SAM-S and SAM-H in EVF-SAM, which indicates that EVF-SAM performs well for different SAM variants. In addition, it also provides insights about designing text-prompted SAMs for future research, e.g., developing a larger and better Multimodal Encoder is more important to empower SAM with text-prompted abilities.

##### 5. Conclusion

In this paper, we explore the effective ways to prompt SAM with texts and demonstrate the importance of using the Multimodal Encoder with early-fusion and multimodal inputs. To this end, we propose EVF-SAM, which establishes a new and simple path for extending SAMs’ textprompted segmentation abilities with the off-the-shelf foundation models. We conduct experiments on the Referring Expression Segmentation (RES) tasks with various benchmarks to evaluate the performance of text-prompted SAM. Experimental results showcase that our EVF-SAM achieves state-of-the-art performance for segmenting objects with referring texts on RefCOCO/+/g benchmarks, outperforming recent approaches based on Large Language Models with huge numbers of parameters. Moreover, we propose carefully-designed data strategies for balancing conflicts

Ablation on trainable modules. In Tab. 6, we evaluated the effects of fine-tuning (✓) or freezing (∗) modules in the proposed EVF-SAM, i.e., the Multimodal Encoder, the prompt encoder, and the mask decoder. The image encoder of SAM is kept frozen during training. As Tab. 6 shows,

- 3OpenAI: https://github.com/openai/CLIP
- 4OpenCLIP: https://github.com/mlfoundations/open_

clip

from data of different distributions. We construct a harmonious hybrid dataset, leading to an uni-model that performs well on various segmentation tasks. We hope this study can bring new ideas or insights to inspire future research on prompting SAM with texts.

##### References

- [1] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023. 4
- [2] Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sa˘gnak Ta¸sırlar. Introducing our multimodal models, 2023. 4
- [3] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Cocostuff: Thing and stuff classes in context. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1209–1218, 2018. 5, 4
- [4] Ding-Jie Chen, Songhao Jia, Yi-Chen Lo, Hwann-Tzong Chen, and Tyng-Luh Liu. See-through-text grouping for referring image segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7454–7463, 2019. 1
- [5] Xianjie Chen, Roozbeh Mottaghi, Xiaobai Liu, Sanja Fidler, Raquel Urtasun, and Alan Yuille. Detect what you can: Detecting and representing objects using holistic models and body parts. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1971–1978,

2014. 2, 5, 6, 1, 3

- [6] Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzm´an, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised cross-lingual representation learning at scale. arXiv preprint arXiv:1911.02116, 2019. 6
- [7] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 3
- [8] Haiwen Diao, Yufeng Cui, Xiaotong Li, Yueze Wang, Huchuan Lu, and Xinlong Wang. Unveiling encoder-free vision-language models. Advances in Neural Information Processing Systems, 37:52545–52567, 2025. 4
- [9] Henghui Ding, Chang Liu, Suchen Wang, and Xudong Jiang. Vision-language transformer and query generation for referring segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16321–16330,

2021. 1

- [10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 3, 6
- [11] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object

- classes (voc) challenge. International journal of computer vision, 88:303–338, 2010. 5, 2
- [12] Ju He, Shuo Yang, Shaokang Yang, Adam Kortylewski, Xiaoding Yuan, Jie-Neng Chen, Shuai Liu, Cheng Yang, Qihang Yu, and Alan Yuille. Partimagenet: A large, highquality dataset of parts. In European Conference on Computer Vision, pages 128–145. Springer, 2022. 5, 6, 1
- [13] Ronghang Hu, Marcus Rohrbach, and Trevor Darrell. Segmentation from natural language expressions. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part I 14, pages 108–124. Springer, 2016. 1
- [14] Zhiwei Hu, Guang Feng, Jiayu Sun, Lihe Zhang, and Huchuan Lu. Bi-directional relationship inferring network for referring image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4424–4433, 2020. 1
- [15] Zhicheng Huang, Zhaoyang Zeng, Bei Liu, Dongmei Fu, and Jianlong Fu. Pixel-bert: Aligning image pixels with text by deep multi-modal transformers. arXiv preprint arXiv:2004.00849, 2020. 5
- [16] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019. 3
- [17] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, 2021. If you use this software, please cite it as below. 7
- [18] Glenn Jocher, Ayush Chaurasia, and Jing Qiu. Ultralytics YOLO, 2023. 3
- [19] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910, 2017. 3
- [20] Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. Mdetrmodulated detection for end-to-end multi-modal understanding. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1780–1790, 2021. 5
- [21] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787–798, 2014. 2, 5, 6
- [22] Lei Ke, Mingqiao Ye, Martin Danelljan, Yu-Wing Tai, ChiKeung Tang, Fisher Yu, et al. Segment anything in high quality. Advances in Neural Information Processing Systems, 36,

2024. 1, 3

- [23] Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Visionand-language transformer without convolution or region supervision. In International conference on machine learning, pages 5583–5594. PMLR, 2021. 4, 6, 8

- [24] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023. 1, 3, 4, 5, 7
- [25] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. arXiv preprint arXiv:2308.00692,

2023. 1, 2, 3, 4, 5, 7

- [26] Muchen Li and Leonid Sigal. Referring transformer: A one-step approach to multi-task visual grounding. Advances in neural information processing systems, 34:19652–19664,

2021. 3

- [27] Muchen Li and Leonid Sigal. Referring transformer: A one-step approach to multi-task visual grounding. Advances in neural information processing systems, 34:19652–19664,

2021. 1

- [28] Xiang Li, Tianhan Wei, Yau Pun Chen, Yu-Wing Tai, and Chi-Keung Tang. Fss-1000: A 1000-class dataset for fewshot segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2869–2878, 2020. 5
- [29] Yuanzhi Li, S´ebastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463, 2023. 4
- [30] Yonglin Li, Jing Zhang, Xiao Teng, and Long Lan. Refsam: Efficiently adapting segmenting anything model for referring video object segmentation. arXiv preprint arXiv:2307.00997, 2023. 3
- [31] Xiaodan Liang, Si Liu, Xiaohui Shen, Jianchao Yang, Luoqi Liu, Jian Dong, Liang Lin, and Shuicheng Yan. Deep human parsing with active template regression. IEEE transactions on pattern analysis and machine intelligence, 37(12):2402– 2414, 2015. 5, 1
- [32] Xiaodan Liang, Chunyan Xu, Xiaohui Shen, Jianchao Yang, Si Liu, Jinhui Tang, Liang Lin, and Shuicheng Yan. Human parsing with contextualized convolutional neural network. In Proceedings of the IEEE international conference on computer vision, pages 1386–1394, 2015. 5, 6, 1
- [33] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 5
- [34] Chenxi Liu, Zhe Lin, Xiaohui Shen, Jimei Yang, Xin Lu, and Alan Yuille. Recurrent multimodal interaction for referring image segmentation. In Proceedings of the IEEE international conference on computer vision, pages 1271–1280,

2017. 1

- [35] Chang Liu, Henghui Ding, and Xudong Jiang. Gres: Generalized referring expression segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 23592–23601, 2023. 5
- [36] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. 3, 4, 5, 7

- [37] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 3
- [38] Jiang Liu, Hui Ding, Zhaowei Cai, Yuting Zhang, Ravi Kumar Satzoda, Vijay Mahadevan, and R Manmatha. Polyformer: Referring image segmentation as sequential polygon generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18653– 18663, 2023. 1, 4, 5
- [39] Yong Liu, Cairong Zhang, Yitong Wang, Jiahao Wang, Yujiu Yang, and Yansong Tang. Universal segmentation at arbitrary granularity with language instruction. arXiv preprint arXiv:2312.01623, 2023. 1, 4, 5, 2
- [40] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 7
- [41] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 11–20, 2016. 2, 5, 6
- [42] Varun K Nagaraja, Vlad I Morariu, and Larry S Davis. Modeling context between objects for referring expression understanding. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 792–807. Springer,

2016. 2, 5, 6

- [43] Gerhard Neuhold, Tobias Ollmann, Samuel Rota Bulo, and Peter Kontschieder. The mapillary vistas dataset for semantic understanding of street scenes. In Proceedings of the IEEE international conference on computer vision, pages 4990– 4999, 2017. 4
- [44] Renjie Pi, Lewei Yao, Jiahui Gao, Jipeng Zhang, and Tong Zhang. Perceptiongpt: Effectively fusing visual perception into llm. arXiv preprint arXiv:2311.06612, 2023. 1, 4
- [45] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 1, 3, 4, 7
- [46] Vignesh Ramanathan, Anmol Kalia, Vladan Petrovic, Yi Wen, Baixue Zheng, Baishan Guo, Rui Wang, Aaron Marquez, Rama Kovvuri, Abhishek Kadian, et al. Paco: Parts and attributes of common objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7141–7151, 2023. 5
- [47] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Erix Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: Pixel grounding large multimodal model. arXiv preprint arXiv:2311.03356, 2023. 1, 2, 3, 5
- [48] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feicht-

- enhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 8, 1
- [49] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang Li, Qing Jiang, and Lei Zhang. Grounded sam: Assembling open-world models for diverse visual tasks,

2024. 3

- [50] Zhongwei Ren, Zhicheng Huang, Yunchao Wei, Yao Zhao, Dongmei Fu, Jiashi Feng, and Xiaojie Jin. Pixellm: Pixel reasoning with large multimodal model. arXiv preprint arXiv:2312.02228, 2023. 1, 4, 5
- [51] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8430–8439, 2019. 2, 5, 6, 1
- [52] Hengcan Shi, Hongliang Li, Fanman Meng, and Qingbo Wu. Key-word-aware network for referring expression image segmentation. In Proceedings of the European Conference on Computer Vision (ECCV), pages 38–54, 2018. 1
- [53] Shuaiwen Leon Song, Bonnie Kruft, Minjia Zhang, Conglong Li, Shiyang Chen, Chengming Zhang, Masahiro Tanaka, Xiaoxia Wu, Jeff Rasley, Ammar Ahmad Awan, et al. Deepspeed4science initiative: Enabling large-scale scientific discovery through sophisticated ai system technologies. arXiv preprint arXiv:2310.04610, 2023. 7
- [54] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, et al. Generative multimodal models are in-context learners. arXiv preprint arXiv:2312.13286, 2023. 4
- [55] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222, 2023. 4
- [56] Yanpeng Sun, Jiahui Chen, Shan Zhang, Xinyu Zhang, Qiang Chen, Gang Zhang, Errui Ding, Jingdong Wang, and Zechao Li. Vrp-sam: Sam with visual reference prompt. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23565–23574, 2024. 3
- [57] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 4, 5
- [58] Peng Wang, Shijie Wang, Junyang Lin, Shuai Bai, Xiaohuan Zhou, Jingren Zhou, Xinggang Wang, and Chang Zhou. One-peace: Exploring one general representation model toward unlimited modalities. arXiv preprint arXiv:2305.11172, 2023. 5
- [59] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for all vision and visionlanguage tasks. arXiv preprint arXiv:2208.10442, 2022. 2, 5, 6, 7, 3

- [60] Zhaoqing Wang, Yu Lu, Qiang Li, Xunqiang Tao, Yandong Guo, Mingming Gong, and Tongliang Liu. Cris: Clipdriven referring image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11686–11695, 2022. 1, 3
- [61] Jiannan Wu, Yi Jiang, Bin Yan, Huchuan Lu, Zehuan Yuan, and Ping Luo. Uniref++: Segment every reference object in spatial and temporal spaces. arXiv preprint arXiv:2312.15715, 2023. 1, 4, 5
- [62] Zhuofan Xia, Dongchen Han, Yizeng Han, Xuran Pan, Shiji Song, and Gao Huang. Gsva: Generalized segmentation via multimodal large language models. arXiv preprint arXiv:2312.10103, 2023. 1
- [63] Defeng Xie, Ruichen Wang, Jian Ma, Chen Chen, Haonan Lu, Dong Yang, Fobo Shi, and Xiaodong Lin. Edit everything: A text-guided generative system for images editing. arXiv preprint arXiv:2304.14006, 2023. 3
- [64] Yunyang Xiong, Bala Varadarajan, Lemeng Wu, Xiaoyu Xiang, Fanyi Xiao, Chenchen Zhu, Xiaoliang Dai, Dilin Wang, Fei Sun, Forrest Iandola, et al. Efficientsam: Leveraged masked image pretraining for efficient segment anything. arXiv preprint arXiv:2312.00863, 2023. 1, 3, 8
- [65] Jinjin Xu, Liwu Xu, Yuzhe Yang, Xiang Li, Yanchun Xie, Yi-Jie Huang, and Yaqian Li. u-llava: Unifying multimodal tasks via large language model. arXiv preprint arXiv:2311.05348, 2023. 1, 4, 5, 2
- [66] Jiarui Xu, Xingyi Zhou, Shen Yan, Xiuye Gu, Anurag Arnab, Chen Sun, Xiaolong Wang, and Cordelia Schmid. Pixelaligned language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13030–13039, 2024. 4, 5
- [67] Bin Yan, Yi Jiang, Jiannan Wu, Dong Wang, Zehuan Yuan, Ping Luo, and Huchuan Lu. Universal instance perception as object discovery and retrieval. In CVPR, 2023. 1, 4, 5
- [68] Senqiao Yang, Tianyuan Qu, Xin Lai, Zhuotao Tian, Bohao Peng, Shu Liu, and Jiaya Jia. An improved baseline for reasoning segmentation with large language model. arXiv preprint arXiv:2312.17240, 2023. 1, 2, 3, 4
- [69] Zhao Yang, Jiaqi Wang, Yansong Tang, Kai Chen, Hengshuang Zhao, and Philip HS Torr. Lavt: Language-aware vision transformer for referring image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18155–18165, 2022. 1, 3, 5
- [70] Jingfeng Yao, Xinggang Wang, Lang Ye, and Wenyu Liu. Matte anything: Interactive natural image matting with segment anything model. Image and Vision Computing, 147: 105067, 2024. 3
- [71] Linwei Ye, Mrigank Rochan, Zhi Liu, and Yang Wang. Cross-modal self-attention network for referring image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10502– 10511, 2019. 1
- [72] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 69–85. Springer, 2016. 2, 5, 6

- [73] Tao Yu, Runseng Feng, Ruoyu Feng, Jinming Liu, Xin Jin, Wenjun Zeng, and Zhibo Chen. Inpaint anything: Segment anything meets image inpainting. arXiv preprint arXiv:2304.06790, 2023. 3
- [74] Chaoning Zhang, Dongshen Han, Yu Qiao, Jung Uk Kim, Sung-Ho Bae, Seungkyu Lee, and Choong Seon Hong. Faster segment anything: Towards lightweight sam for mobile applications. arXiv preprint arXiv:2306.14289, 2023. 1
- [75] Zheng Zhang, Yeyao Ma, Enming Zhang, and Xiang Bai. Psalm: Pixelwise segmentation with large multi-modal model. arXiv preprint arXiv:2403.14598, 2024. 1, 4, 5
- [76] Xu Zhao, Wenchao Ding, Yongqi An, Yinglong Du, Tao Yu, Min Li, Ming Tang, and Jinqiao Wang. Fast segment anything, 2023. 1, 3
- [77] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 633–641,

2017. 2, 5, 6, 1, 4

- [78] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. International Journal of Computer Vision, 127:302–321, 2019. 2, 5, 6, 1, 4
- [79] Yichen Zhu, Minjie Zhu, Ning Liu, Zhiyuan Xu, and Yaxin Peng. Llava-phi: Efficient multi-modal assistant with small language model. In Proceedings of the 1st International Workshop on Efficient Multimedia Computing under Limited, pages 18–22, 2024. 4

## EVF-SAM: Early Vision-Language Fusion for Text-Prompted Segment Anything Model

### Supplementary Material

##### A. Language-guided Multi-task Segmentation.

Due to the limitation of Referring Expression Segmentation (RES) data in aspects of both quantity (80k) and quality (mask accuracy, caption diversity), employing data from other sub-tasks of segmentation (e.g., semantic segmentation) become popular among recent works (e.g., LISA [25], u-LLaVA [65]). If properly implemented, additional data can not only improve RES performance, but also extend the segmentation granularities of RES (i.e., semantic-level segmentation, part-level segmentation, introduced in Sec. A.3 and Sec. A.4). However, existing works usually apply a rough manner, leading to weak or even negative influence on models’ RES performance. Moreover, they are weak at other granularites of RES.

In this section, we firstly analyze the challenges of employing data from other sub-tasks of segmentation (e.g., semantic segmentation), then we present our detailed recipe of our data strategies to building the hybrid dataset. Finally we introduce two tasks called ”Semantic-level Referring Expression Segmentation” and ”Part-level Referring Expression Segmentation” to evaluate the models’ performance at different granularites of RES.

###### A.1. Challenges.

Existing segmentation data can be generally separated to the semantic-level data and the instance-level data. When applying these data for Referring Expression Segmentation (RES), Semantic conflict will arise in the semantic-data and ambiguity problem will arise in the instance-level data. Semantic conflict: The semantic conflict, as discussed in UniLSeg [39], mainly exhibits between semantic segmentation datasets and RES datasets. Tab. 7 presents several representative cases. The semantic segmentation datasets only provide the category name as annotation, letting go of the information regarding to instance number. However, in RES tasks, the description should explicitly align with the related region. As can be seen in the first case in Tab. 7, ‘cat’ does not align with all three cats in the image. Some works (i.e., [39]) propose using fixed template like ‘all the category’ to avoid the semantic conflict, while introducing additional misalignment. The over-description problem can be seen in the second case in Tab. 7. ‘All the cats’ is used to describe the single cat in the image. Furthermore, difficulty of handling plural forms can be seen in the third case in Tab. 7. The fixed template would fail at categories like ‘bread’, ‘man’. The experiment in Sec. B.2 shows how semantic conflict affects performance.

Ambiguity problem: The ambiguity problem mainly exhibits between instance-level dataset and RES datasets. In instance-level datasets, multiple objects may align to the same category. However, in RES datasets, the correspondence is one-to-one. Directly using instance segmentation data to train an RES model will result in inconsistent target assignments, leading to unstable training.

###### A.2. Unified training with multi-task datasets.

To solve the problems mentioned above, we propose data strategies for various datasets. Specifically, we propose filter/merging principles for instance-level data and introduce special token ‘[semantic]’ to semantic-level data. We will open-source related codes on our project page.

Instance-level data: We include Objects365 [51] to extend RES data. Specifically, (a) for each image, we exclude categories with more than one instance to avoid ambiguity problems. (b) we employ SAM-2 [48] to automatically annotate masks according to the selected ground-truth bounding boxes. Despite filtering, a substantial number of annotations remain, owing to the high density of annotations in Objects365 [51]. From the original 600K images and 10M annotations, we retain 524K images and 1.8M annotations. The mask quality from automatic annotation is high due to the accurate ground-truth from Objects365 [51] and the powerful segmentation capability of SAM-2 [48]. Notably, the remaining annotations are valuable for addressing long-tail problems because those excluded annotations predominantly represent head categories.

Semantic-level data: We introduce ADE20K [77, 78] to broaden RES models’ capability of various granularities. We introduce a special token ‘[semantic]’ and input ‘[semantic] {category}’ to RES models. The special token would not be limited to common grammar so it is helpful to avoid semantic conflict. As shown in Tab. 7, the reconstructed descriptions with our special token have no conflict with the mask annotation.

Part-level data: To enable the model to segment parts of objects, we introduce PartImageNet [12], HumanParsing [31, 32] and PASCAL-Part [5] to train our model. For datasets annotated in semantic-level, i.e., HumanParsing, we implement the same strategy as ADE20K. Exceptionally, we align the definition of ‘left’ and ‘right’ with RES datasets (e.g., RefCOCO). For datasets annotated in instance-level, i.e., PartImageNet and PASCAL-Part, we merge instance masks of the same category to convert the dataset to semantic-level. Then, the same strategy as

Table 7. Explanation to semantic conflict. The top row contains the source image and annotated mask in the ADE20K [77, 78] dataset. The middle row presents the comparison of templates used by different works to apply semantic data into Referring Expression Segmentation, where red fonts represent the wrong description, while the green fonts represent the right description. The bottom row contains ideal caption. Previous works either prompt with the raw category name without modifications or apply a fixed template, leading to unaligned descriptions in some cases. We use special token ‘[semantic]’ as the task identifier, overcoming the misalignment problem.

[Figure 24]

[Figure 25]

[Figure 26]

Image & mask Raw [25] cat cat man Fixed template [39] all the cats all the cats all the mans Special token (ours) [semantic] cat [semantic] cat [semantic] man Ideal caption all the cats/cats/... the cat/cat/... all the men/...

ADE20K is implemented.

By combining those datasets, we observe a significant performance gain of 1.0 cIoU in average on RefCOCO/+/g [21, 41, 42, 72], as shown in Tab. 1. Moreover, our model is empowered the capability of RES of various granularities (i.e., semantic-level segmentation, part-level segmentation, introduced in Sec. A.3 and Sec. A.4).

###### A.3. Semantic-level RES

In this section, we will make a detailed explanation of our proposed new task ‘Semantic-level Referring Expression Segmentation’, present the metric comparison, and provide some visualizations.

Definition. Basic RES tasks only focus on the instancelevel segmentation. A special prompt only refers to a single object in the image. Trained only on RefCOCO/+/g [21, 41, 42, 72] datasets, the model cannot correctly respond to more complex prompts (e.g., multiple objects, stuff categories).

To this end, we propose ‘Semantic-level RES’, aiming at improving RES models’ capability of segmenting semanticlevel objects. We make perfect use of existing semantic segmentation datasets. The annotated category is managed as text prompt of the RES model, while the semantic mask corresponding to the category is managed as prediction target. Notably, we include stuff categories during training and evaluation.

‘Semantic-level RES’ differs from semantic segmentation in that we manage object category as model input, not the output. ‘Semantic-level RES’ differs from open vocabulary segmentation in that this is an in-domain task.

Metrics on Semantic-level RES. Being a sub-task of Referring Segmentation, we take cIoU and gIoU as metrics following the RES setting. This metric examines the com-

Table 8. Semantic-level Referring Segmentation results. * denotes zero-shot results. EVF-SAM outperforms previous methods that include ADE20K [77, 78] during training, indicating the effectiveness of our proposed special token ‘[semantic]’.

ADE20K [77, 78] Pascal VOC [11] gIou cIoU gIou cIoU

Method

LISA-7B 49.0 67.4 53.6* 46.7* LISA-7B-explanatory 55.4 70.8 56.2* 47.0* u-LLaVA 42.1 75.3 57.5 44.8 EVF-SAM 64.5 76.8 78.2* 81.8*

petence of a model to segment multi-objects and stuff categories.

We compare our model with LISA [25] and uLLaVA [65], both of which also include ADE20K [77, 78] data during training. As shown in Tab. 8, our model demonstrates a significant advantage over previous methods in semantic-level RES. This superior performance is attributed to our novel approach to resolving the semantic conflict (i.e., using the special token ‘[semantic]’). The solution of semantic conflict greatly improves model’s capability of segmenting background classes (e.g., ‘sky’, ‘sea’, or simply ‘background’) and segmenting multi-objects (e.g., ‘people’, ‘flowers’).

Visualizations on ADE20K. Fig. 6 shows the visualization of ‘Semantic-level Referring Segmentation’ task on ADE20K val. Given an semantic annotated category as text prompt for RES models, the models are to predict semantic masks containing all pixels corresponding to the category. Visualization results are provided by our EVF-SAM. Trained on Semantic-level Referring Segmentation tasks, EVF-SAM is able to segment multiple objects (e.g., drawings, chairs) or background categories (e.g., floor, ground).

[Figure 27]

[Figure 28]

|[Figure 29]|
|---|

| |
|---|

| |
|---|

“[semantic] painting”

“[semantic] person”

“[semantic] tree”

| |
|---|

[Figure 30]

[Figure 31]

[Figure 32]

| |
|---|

| |
|---|

“[semantic] ceiling” “[semantic] floor”

“[semantic] grass”

- Figure 6. Visualization Results on ADE20K Val. We add a special token ‘[semantic]’ in front of category name to construct the referring prompt, as written below each image. We use red boxes to highlight the targeted object and use blue masks to visualize the segmentation result. Upper three images showcase EVF-SAM’s capability at multi-object RES. Below three images showcase EVF-SAM’s capability at stuff-object RES.

- Table 9. Part-level Referring Expression Segmentation results. We present evaluation metrics on Pascal-Part and compare the results with LISA, which also includes Pascal-Part during training. Our EVF-SAM outperforms LISA by a large margin.

Method gIoU cIoU

LISA 21.2 27.0 EVF-SAM 53.7 71.4

###### A.4. Part-level RES

In this section, we will make a detailed explanation of our proposed new task ‘Part-level RES’, present the metric comparison, and provide some visualizations.

Definition Similar to Semantic-level RES, we further propose ‘Part-level RES’, aiming at improving RES models’ capability of segmenting part of objects. We make perfect use of existing part segmentation datasets. The annotated category is managed as text prompt of the RES model, while the semantic mask corresponding to the category is managed as prediction target.

Metrics on Part-level RES We evaluate part-level RES on Pascal-Part dataset. It is worth noting that we merge the instance-level annotations to semantic-level annotations to avoid ambiguity problems, as illustrated in Sec. 3.3. We still following the RES settings to evaluate cIoU and gIoU

- as validation metrics. The metrics are shown in Tab. 9. We compare our EVF-SAM with LISA [25], which also includes Pascal-Part [5] dataset during training. The experiment results show that our EVF-SAM outperforms LISA
- at part-level RES task by a huge margin. Visualizations on Pascal-Part. Fig. 7 shows the visualization of ‘Part-level Referring Expression Segmentation’ on Pascal-Part val.

##### B. Additional Experiments

###### B.1. Reasoning segmentation.

ReasonSeg [25] is a Referring Expression Segmentation dataset targeting long and indirect descriptions. It comprises 239 training images, 200 validation images and 779 test images. Text annotations of ReasonSeg are posed in an instructional format, such as ‘What is the food with the most Vitamin C in this image?’. Considering the limited amount of data and the risk of over-fitting, we process evaluation using both the zero-shot setting and the finetuning setting. The zero-shot performance on ReasonSeg of our EVF-SAM surpasses LLaVA-free methods (e.g., OVSeg, SEEM), but falls behind methods with LLaVA1.1 [36]. When we finetune our EVF-SAM with 239 ReasonSeg training samples, we observe significant improvement in reasoning performance, outperforming LISA-7B-LLaVA1.1 [25] However, finetuned EVF-SAM still falls behind LISA with LLaVA1.5 [37].

We analyze LISA’s superior performance on ReasonSeg compared to EVF-SAM, attributing it to several key factors:

- • LISA’s reasoning abilities significantly improved when transitioning from LLaVA 1.1 to LLaVA 1.5. This enhancement stems from LLaVA 1.5’s incorporation of reasoning-focused Visual Question Answering (VQA) datasets (e.g., CLEVR [19], GQA [16]), which effectively transfer reasoning knowledge to ReasonSeg. However, EVF-SAM’s foundation model (i.e., BEIT-3 [59]) is only pre-trained using Masked Language Modeling (MLM) recipe.
- • LLM-free methods (e.g., EVF-SAM) are absent of instruction tuning, hindering their ability to accurately interpret the QA formatted data.

| |
|---|

| |
|---|

[Figure 33]

[Figure 34]

[Figure 35]

| |
|---|

“[semantic] motorbike wheel”

“[semantic] dog eye” “[semantic] cat ear”

[Figure 36]

[Figure 37]

[Figure 38]

| |
|---|

| |
|---|

| |
|---|

“[semantic] car door” “[semantic] aeroplane body”

“[semantic] bird head”

- Figure 7. Visualization Results on Pascal-Part Val. We add a special token ‘[semantic]’ in front of category name to construct the referring prompt, as written below each image. We use red boxes to highlight the targeted object and use blue masks to visualize the segmentation result. Upper three images showcase EVF-SAM’s capability at multi-part RES. Below three images showcase EVF-SAM’s capability at broad-part RES.

- Table 10. Comparision of segmentation capability on ReasonSeg [25] dataset. Our EVF-SAM outperforms previous methods on both cIoU and gIoU metrics.

val test gIoU cIoU gIoU cIoU

Method Finetune?

OVSeg ✗ 28.5 18.6 26.1 20.8 X-Decoder ✗ 22.6 17.9 21.7 16.3 SEEM ✗ 25.5 21.2 24.3 18.7 LISA-7B-LLaVA1.1 ✗ 44.4 46.0 36.8 34.1 LISA-7B-LLaVA1.1 ✓ 52.9 54.0 36.8 34.1 LISA-7B-LLaVA1.5 ✗ 53.6 52.3 48.7 48.8 LISA-7B-LLaVA1.5 ✓ 61.3 62.9 55.6 56.9 LISA-13B-LLaVA1.5 ✓ 65.0 72.9 61.3 62.2 EVF-SAM ✗ 38.3 30.0 38.9 32.6 EVF-SAM ✓ 58.7 55.7 55.1 50.5

###### B.2. Experiments of simply adding extra data.

We tried to introduce some extra semantic segmentation datasets (ADE20K [77, 78], Mapillary [43]) to proceed with joint training. We do not include COCO-Stuff [3] to avoid data leakage with RefCOCO/+/g. We report comparison results in Tab. 11. From the experimental results: (a) We find that several metrics gain when introducing extra semantic segmentation data (e.g., RefCOCO+/testB). This may results from the analogous data composition between RefCOCO+ and ADE20K, indicating the feasibility of introducing data of other segmentation sub-task to boost RES model. (b) We find that the Semantic-level Referring Segmentation capability improves a lot when including extra semantic segmentation data, indicating the necessity of building multi-task training. (c) However, We observe performance degradation on various metrics (e.g., RefCOCO/val, RefCOCO/testA). This is due to the difference distribution between datasets from RES and other segmen-

tation sub-tasks.

###### B.3. Multimodal feature representation.

In Tab. 12, we explore the effects of using different multimodal features representations as prompts for SAM. Specifically, we adopt different outputs of the Multimodal Encoder: (a) the image [CLS] token, (b) the AvgPool over image tokens, and (c) the text [CLS] token. Tab. 12 shows that using image [CLS] token is more effective while combining image and text tokens through concatenation leads to a performance drop.

To unveil how the multimodal encoder contributes to prompting SAM with texts, we visualize the attention maps between the [CLS] token (prompt embeddings) and the image tokens from the last layer of BEIT-3. As shown in

- Fig. 8, the attention maps focus on the target objects and are consistent with the input text prompts. The deep fusion of text and image embeddings leads to accurate regiontext alignment. Consequently, the prompt embeddings contain abundant object-related information, including semantics and spatial localization, which is conducive to SAM achieving precise object segmentation.

C. Qualitative Results

In this section, we mainly visualize the qualitative results on RefCOCO val and RefCOCOg val datasets, as shown in

- Fig. 9 and Fig. 10, respectively. Moreover, we compare the qualitative results of different ways to prompt SAM with texts: (1) our proposed EVF-SAM, (2) SAM with LLM (LISA [25]), and (3) SAM with a CLIP text encoder implemented in this paper (suggested by [24], which are based on the same SAM-Huge model. The qualitative results can

[Figure 39]

[Figure 40]

Segmentation

[Figure 41]

- Table 11. Results of adding extra semantic data. ∗ means zero-shot results. The reported ADE20K results are evaluated on Semanticlevel Referring Segmentation tasks proposed in main text.

“ the elephant with its trunk in the water ”

RefCOCO RefCOCO+ RefCOCOg ADE20K val testA testB val testA testB val test val

ADE20K Mapillary

82.1 83.7 80.0 75.2 78.3 70.1 76.8 77.4 54.2∗

✓ 81.7 83.6 80.3 75.4 78.4 71.3 75.5 77.6 75.9

✓ 81.9 83.5 80.3 75.1 78.0 70.8 75.3 77.4 59.6∗

✓ ✓ 81.8 83.4 79.7 75.6 78.0 70.7 75.8 76.9 76.1

###### a giraffe snacking on the tree ”

Attention Maps Segmentation Mask Attention Maps Segmentation Mask Attention Maps Segmentation Mask

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

“ loaded hot dog with bite taken out ”

“ sheep standing close to and behind another sheep ” “ a laptop on a bed ”

- Figure 8. Visualizations of Attention Maps in Multimodal Encoder. To unveil the effects of the Multimodal Encoder, we visualize the attention maps between the [CLS] token and image tokens in the last layer of BEIT-3-Large. Specifically, we sum up the attention maps from all heads.

Table 12. Ablations on multimodal feature representation. BEIT-3 contains two [CLS] tokens for visual and textual modalities. We also explore the effects of AvgPool and late fusion between two modalities.

[CLS]Text [CLS]Image AvgPoolImage Fusion cIoU ✓ - 83.5 ✓ - 83.7 ✓ - 83.5 ✓ ✓ Concat 83.2

demonstrate the superiority of the proposed EVF-SAM.

Visualizations on RefCOCO. Fig. 9 shows the qualitative comparisons on the RefCOCO val, which contains simple descriptive expression texts. The proposed EVF-SAM can follow the expressions and segment more accurately with clear boundaries.

Visualizations on RefCOCOg. Fig. 10 illustrates the qualitative comparisons on the RefCOCOg val, which aims to segment objects with long expression texts. The SAM with a vanilla CLIP text encoder produces inferior segmentation results given the long-expression texts. However, the proposed EVF-SAM outperforms LISA when using long expressions, even though LISA adopts LLaMA-7B [57] to understand the instructions and generate prompt embeddings, showcasing that the lightweight vision-language models can understand complex expressions. In addition, the proposed EVF-SAM can also understand the texts or expressions towards spatial locations, such as ‘the umbrella closest to the camera’.

EVF-SAM LISA CLIP-SAM EVF-SAM LISA CLIP

[Figure 48]

[Figure 49]

[Figure 50]

“ woman in blue ”

[Figure 51]

[Figure 52]

[Figure 53]

“ right taco thing ”

[Figure 54]

[Figure 55]

[Figure 56]

“ left laptop ”

[Figure 60]

[Figure 61]

[Figure 62]

“ sleeping woman ”

- Figure 9. Visualization Results on RefCOCO val. We compared the qualitative results on RefCOCO which contains simple descriptive expressions.

EVF-SAM LISA CLIP-SAM

[Figure 63]

[Figure 64]

[Figure 65]

“ the plate holding the fruit ”

[Figure 66]

[Figure 67]

[Figure 68]

EVF-SAM LISA

“woman in blue”

“ the umbrella closest to the camera ”

[Figure 75]

[Figure 76]

[Figure 77]

“black cat”

“ a wooden chair leg in the background of the photo ”

[Figure 78]

[Figure 79]

[Figure 80]

“ white dog laying on the couch ”

- Figure 10. Visualization Results on RefCOCOg val. Considering that RefCOCOg contains longer expressions and we provide qualitative results to show the capability of our EVF-SAM for understanding long expressions.

