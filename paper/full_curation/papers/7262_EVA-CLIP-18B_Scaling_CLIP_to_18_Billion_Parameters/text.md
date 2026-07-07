## EVA-CLIP-18B: Scaling CLIP to 18 Billion Parameters

# arXiv:2402.04252v1[cs.CV]6Feb2024

Quan Sun1* Jinsheng Wang1∗ Qiying Yu1,2∗ Yufeng Cui1 Fan Zhang1 Xiaosong Zhang1 Xinlong Wang1

1 Beijing Academy of Artificial Intelligence 2 Tsinghua University ∗equal contribution

code & models: baaivision/EVA/EVA-CLIP-18B

### Abstract

Scaling up contrastive language-image pretraining (CLIP) is critical for empowering both vision and multimodal models. We present EVA-CLIP-18B, the largest and most powerful open-source CLIP model to date, with 18-billion parameters. With only 6-billion training samples seen, EVA-CLIP-18B achieves an exceptional 80.7% zero-shot top-1 accuracy averaged across 27 widely recognized image classification benchmarks, outperforming its forerunner EVA-CLIP (5-billion parameters) and other open-source CLIP models by a large margin. Remarkably, we observe a consistent performance improvement with the model size scaling of EVA-CLIP, despite maintaining a constant training dataset of 2-billion image-text pairs from LAION-2B and COYO-700M. This dataset is openly available and much smaller than the in-house datasets (e.g., DFN5B, WebLI-10B) employed in other state-of-the-art CLIP models. EVA-CLIP-18B demonstrates the potential of EVAstyle [30, 29, 63] weak-to-strong visual model scaling. With our model weights made publicly available, we hope to facilitate future research in vision and multimodal foundation models.

### 1. Introduction

Recent years witnessed the rapid growth of Large Multimodal Models (LMMs) [3, 64, 62, 69, 5, 46], with CLIP models [53, 19, 63, 43, 75, 28, 17] serving as a foundational vision encoder to deliver robust and transferable visual representations, and Large Language Models (LLMs) [65, 54] serving as a general interface for reasoning across different modalities. However, as LLMs have scaled up to around 100B parameters or higher [11, 20, 65], the adopted vision foundation models continue to operate at a much smaller scale, lagging far behind the LLMs.

*Correspondence to wangxinlong@baai.ac.cn

Figure 1: Scaling behavior of EVA-CLIP with zero-shot classification performance averaged across 27 image classification benchmarks, compared with the current state-of-the-art and largest CLIP models (224px). The diameter of each circle demonstrates the forward GFLOPs × the number of training samples seen. The performance of EVA-CLIP consistently improves as scaling up.

This paper introduces EVA-CLIP-18B, the largest opensource CLIP model with 18-billion parameters to narrow this gap. EVA-CLIP [63] open-sources a series of effective and efficient CLIP models, which have been leveraged as the vision foundation by numerous impactful works across 2D / 3D vision and multimodal modeling [42, 78, 77, 50, 69, 64]. We further scale up EVA-CLIP to this significant parameter size building upon the scaling philosophy of EVA [30, 29] and EVA-CLIP [63]. With merely 6-billion training samples seen and trained on publicly available datasets, EVA-CLIP18B achieves the exceptional 80.7% average zero-shot top1 accuracy on 27 widely recognized image classification benchmarks, significantly surpassing its forerunner EVA-02CLIP-E/14+ (5-billion parameters) and other open-source CLIP models. Besides, the models have not exhibited any

| |total image text|samples|image batch|image cls.|video cls.|retrieval|
|---|---|---|---|---|---|---|
|model+|#param. #param. #param.|data seen|size size gpus for training|avg. acc.|avg. acc.|MR|
|EVA-01-CLIP-g/14+ [63] EVA-01-CLIP-g/14+ [63] OpenCLIP-G/14+ [2] InternVL-C+ [17] DFN5B-CLIP-H/14+ [28]|1.1B 1.0B 124M<br><br>1.3B 1.0B 354M<br>2.5B 1.8B 695M 14.0B 6.0B 8.0B<br><br><br>1.0B 632M 354M|LAION-400M [59] 11B Merged-2B [63] 11B LAION-2B [58] 39B<br><br>custom [17] 29B DFN-5B [28] 39B|2242 41k 256×A100 (40GB) 2242 114k 112×A100 (40GB) 2242 160k 512×A100 (80GB) 2242 164k 640×A100 (80GB) 2242 79k TPUv4|72.2 75.1 76.2 78.0 78.3|66.2 68.8 68.7 73.7 67.0|80.9 85.3 85.7 86.6 86.6|
|EVA-02-CLIP-E/14+ [63] DFN5B-CLIP-H/14+ [28]|5.0B 4.4B 695M 1.0B 632M 354M|LAION-2B [58] 9B DFN-5B [28] 5B|2242 144k 144×A100 (80GB) 3782 79k TPUv4|78.7 79.2|72.1 68.4|85.7 87.2|
|EVA-CLIP-8B + EVA-CLIP-18B +|8.1B 7.5B 695M 18.1B 17.5B 695M|Merged-2B [63] 9B Merged-2B+ 6B|2242 178k 384×A100 (40GB) 2242 108k 360×A100 (40GB)|79.4 80.7|73.6 75.0|86.2 87.8|

- Table 1: CLIP model configurations and zero-shot performance on 33 benchmarks including 27 image classification, 4 video classification and 2 image-text retrieval datasets. DFN-5B [28] are 5B images filtered from a pool of 43B uncurated image-text pairs consisting of 12.8B image-text pairs from CommonPool-12.8B [32] and 30B additional public image-text pairs. The dataset used for training InternVL-C [17] is custom mixtures, see detail in [17].

|method| |[<br><br>67<br><br>[<br><br>[<br><br>]<br><br>A<br><br>21<br><br>to<br><br>[<br><br>51<br><br>[<br><br>[<br><br>Flowers-1<br><br>RESISC4<br><br>Food-101<br><br>FER2013<br><br>Rendered<br><br>ObjectNe<br><br>EuroSAT<br><br>CIFAR-1<br><br>CIFAR-1<br><br>ImageNe<br><br>ImageNe<br><br>ImageNe<br><br>ImageNe<br><br>ImageNe<br><br>VOC200<br><br>Caltech1<br><br>Country-<br><br>Birdsnap<br><br>SUN397<br><br>Stanford<br><br>GTSRB<br><br>MNIST<br><br>STL-10<br><br>FGVC<br><br>avg.<br><br>PCam<br><br>DTD<br><br>Pets<br><br>.|4<br><br>a<br><br>31<br><br>]<br><br>]<br><br>4<br><br>[<br><br>18<br><br>]<br><br>[<br><br>]<br><br>40<br><br>]<br><br>27<br><br>[<br><br>]<br><br>[<br><br>10<br><br>SST<br><br>33<br><br>8<br><br>]<br><br>]<br><br>]<br><br>[<br><br>34<br><br>t-Adv<br><br>t-Ren<br><br>ircraf<br><br>t-Ske<br><br>[<br><br>Cars<br><br>72<br><br>t-1K<br><br>t-V2<br><br>[<br><br>[<br><br>9<br><br>p-1<br><br>211<br><br>61 [<br><br>102 [<br><br>00<br><br>]<br><br>23<br><br>41<br><br>5<br><br>[<br><br>01<br><br>[<br><br>0 t<br><br>[<br><br>7<br><br>]<br><br>7 [|
|---|---|---|---|
|EVA-01-CLIP-g/14+|78.5 71.5 73.6 92.5 67.6 72.3|98.3 88.7 62.6 87.7 74.2 32.4 28.9 91.7 65.8 61.7 73.8 52.2 74.5 93.5 49.3 49.9 94.2 58.4 70.3 98.9 85.7|72.2|
|EVA-01-CLIP-g/14+|79.3 72.5 74.1 92.7 68.4 75.3|99.1 90.1 72.0 89.5 74.7 39.9 31.8 90.7 70.2 67.8 73.2 56.0 79.7 93.7 66.5 62.4 94.9 58.6 71.4 99.5 84.7|75.1|
|OpenCLIP-G/14+|80.4 73.6 69.3 92.8 69.9 73.0|98.3 87.5 71.6 89.4 75.0 53.6 34.9 94.9 73.0 69.1 71.1 59.6 81.5 93.1 62.7 63.6 95.3 65.3 72.6 98.5 87.4|76.2|
|InternVL-C +|83.2 77.3 83.8 95.7 74.3 80.6|99.4 93.1 80.6 89.5 76.3 53.3 35.1 94.4 69.2 70.8 79.4 56.2 85.8 95.3 65.5 48.7 96.3 68.4 74.4 99.4 80.0|78.0|
|DFN5B-CLIP-H/14 +|83.5 77.4 71.7 92.9 72.8 76.7|98.8 90.5 85.8 89.5 77.0 71.4 34.4 95.8 77.4 70.7 65.2 54.7 92.5 95.8 67.7 65.2 96.5 54.8 76.1 98.9 81.5|78.3|
|EVA-02-CLIP-E/14+|82.1 75.7 82.1 94.7 72.2 79.6|99.3 93.2 74.7 90.5 75.3 58.7 37.0 94.7 77.6 68.2 75.9 59.0 84.5 94.9 67.7 64.4 96.0 62.6 75.7 99.3 87.9|78.7|
|DFN5B-CLIP-H/14+|84.3 78.3 79.6 93.6 73.3 79.6|98.8 90.5 83.6 88.9 77.4 72.5 37.9 96.0 80.5 70.9 61.1 56.1 91.6 96.2 67.9 69.6 96.8 55.5 75.9 99.1 81.9|79.2|
|EVA-CLIP-8B +|83.5 77.7 85.2 95.3 74.3 81.2|99.3 92.3 84.8 89.6 76.2 60.5 41.7 94.8 79.0 71.0 68.9 56.1 86.4 95.5 70.9 58.1 96.4 66.2 75.3 99.3 85.1|79.4|
|EVA-CLIP-18B +|83.8 77.9 87.3 95.7 74.7 82.2|99.4 93.8 83.0 89.8 77.7 59.7 43.1 94.9 79.9 72.1 79.8 59.3 86.0 95.8 72.4 65.2 96.1 67.5 76.9 99.6 85.8|80.7|

[26]

[57]

.[36]

.[35]

.[68]

]

0]

]

t[47]

53]

[39]

9]

2[53]

]

]

cc.

- Table 2: EVA-CLIP zero-shot image classification performance on 27 datasets. We report top-1 accuracy on all datasets. The best results are in bold and the second best are underlined.

signal of performance saturation, shedding light on further scaling of vision models. An intuitive demonstration is shown in Figure 1.

The successful training of EVA-CLIP-18B exemplifies the potential of the EVA-style visual model scaling philosophy. We keep open-sourcing the training code and weights of our models to encourage further research and empower the development of vision and multimodal foundation models.

### 2. Weak-to-Strong Vision Scaling

Our scaling-up procedure is guided by the principles of EVA [30] and EVA-CLIP [63]. The EVA philosophy for scaling visual models follows a weak-to-strong paradigm, designed to improve visual models through a strategic progression. This process begins with a large EVA vision model distilling knowledge from a small EVA-CLIP model, which in turn serves as the vision encoder initialization to stabilize and accelerate the training of a larger EVA-CLIP. After that, the closed-loop scaling-up cycle continues and a larger EVA is distilled out. Throughout our model scaling cycle, the training dataset remains largely fixed to demonstrate the

effectiveness of our model-scale specific scaling philosophy, although scaling up datasets can further unleash the scaling power of our method.

Specifically, in this work, we pre-train a large EVA model named EVA-18B using a small EVA-CLIP (EVA-02-CLIPE/14+) [63] as the teacher, which is trained to reconstruct masked image-text aligned vision features from EVA-02CLIP-E/14+. EVA-18B omits bias terms of QKV projections and uses RMSNorm [76] instead of LayerNorm [4] following LLaMA [65]. Subsequently, we leverage the EVA model as the vision encoder initialization for EVA-CLIP pre-training with the image-text contrastive learning objective. Besides, we also introduce a smaller counterpart, EVACLIP-8B, which undergoes similar pre-training methodologies. Notably, our experiments demonstrate sustained performance improvement with the progressive weak-teach-strong scaling up of EVA-CLIP.

### 3. Experiments

Settings. Following EVA-CLIP [63], we initialized the model with pre-trained vision and text encoders. Specifi-

zero-shot text retrieval zero-shot image retrieval

Flickr30K COCO Flickr30K COCO

method+ R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 MR EVA-01-CLIP-g/14 + 87.9 98.0 99.5 61.7 83.2 89.9 72.5 91.5 95.4 44.5 69.1 77.7 80.9

- EVA-01-CLIP-g/14+ 93.3 99.5 99.9 69.4 88.3 93.2 79.2 95.2 97.3 51.1 74.7 82.5 85.3 OpenCLIP-G/14+ 93.5 99.3 99.7 69.0 87.8 93.1 80.9 95.1 97.2 52.6 76.1 83.6 85.7

- EVA-02-CLIP-E/14+ 94.3 99.6 99.8 69.4 88.6 93.3 79.7 94.9 97.3 52.5 75.9 83.4 85.7 EVA-CLIP-8B + 95.6 99.6 99.9 70.3 89.3 93.9 80.8 95.5 97.6 53.0 76.0 83.4 86.2

- DFN5B-CLIP-H/14 + 92.9 99.3 99.9 72.3 90.2 94.6 80.1 95.2 97.3 53.9 78.0 85.6 86.6 InternVL-C + 93.8 99.7 100.0 70.3 89.2 93.8 82.1 96.0 98.1 54.1 77.1 84.8 86.6

- DFN5B-CLIP-H/14 + 93.6 99.3 99.6 71.8 90.4 94.9 82.1 96.0 97.9 55.6 79.2 86.3 87.2 EVA-CLIP-18B + 96.7 99.7 100.0 73.6 90.9 95.0 83.3 96.3 98.3 56.2 78.5 85.6 87.8

Table 3: Zero-shot retrieval performance on Flickr30K [74] and COCO [45].

cally, we employ a pre-trained EVA-18B [30, 29] as the vision encoder and EVA-02-CLIP-E/14+ [63] for the text encoder. We adopt the LAMB optimizer [73] with β1 = 0.9, β2=0.95, and a weight decay of 0. We apply different learning rates and layer decay rates to the vision encoder and text encoder to ensure optimal training. We set the peak learning rate as 4e-4 and 4e-5 for the vision encoder and the text encoder respectively, with 2000 warm-up steps. Afterwards, the learning rates decay to 0 with a cosine schedule. The learning rate layer decay rates are configured as 0.9 and 0.75 for the vision and text encoders. The temperature parameter remains constant at 0.01. Further, we use the DeepSpeed optimization library [56] with ZeRO stage-3 partition [55], gradient checkpointing [16] and flash attention [24] to optimize the training cost.

Dataset. Our Merged-2B dataset consists of 1.6 billion samples from LAION-2B [58] and 0.4 billion samples from COYO-700M [12]. Note that the use of a subset from LAION-2B is not the result of deliberate filtering, but rather due to image downloading failures. The use of 0.4 billion COYO-700M samples aims to complement the number of training samples to nearly the same as LAION-2B. Merged2B+ consists of all samples from Merged-2B, along with additional 20 million samples from LAION-COCO [1] and 23 million samples from Merged-video including VideoCC [48], InternVid [70] and WebVid-10M [6]. Merged-video is included at the end of the training process.

EVA-CLIP-18B pre-trains with 5.4 billion samples from Merged-2B seen with 50% of patch dropout ratio [44], 0.6 billion samples from Merged-2B and 20 million samples from LAION-COCO without patch dropout, and 24 million samples from Merged-video with 50% of patch dropout ratio.

Evaluation. We evaluate on 33 widely used datasets across image, video classification and image-text retrieval. All datasets used to evaluate EVA-CLIP-18B are reported in Table 11. We utilize the specified prompt templates following [53, 38].

|method<br><br>|image encoder layers width heads<br><br>|text encoder layers width heads<br><br>|# params image text total|
|---|---|---|---|
|EVA-CLIP-8B + EVA-CLIP-18B +<br><br>|32 4096 32 48 5120 40|32 1280 20 32 1280 20<br><br>|7.5B 695M 8.1B 17.5B 695M 18.1B|

Table 4: Architecture configurations.

Zero-Shot Image Classification. We show the exceptional performance of EVA-CLIP on all 27 zero-shot image classification benchmarks in Table 2. EVA-CLIP-18B achieves 80.7% top-1 accuracy averaged across all 27 benchmarks. These results significantly outperform the previous best opensourced DFN5B-CLIP-H/14+ [28] by +1.5%, and the largest existing CLIP model, InternVL-C [17], by +2.7%. For Birdsnap dataset, the download was limited to 2195 test images due to broken links.

|method+|#Frames<br><br>|UCF-101 K-400 K-600 K-700|avg.|
|---|---|---|---|
|EVA-01-CLIP-g/14+ DFN5B-CLIP-H/14+ DFN5B-CLIP-H/14+ OpenCLIP-G/14+<br><br>EVA-01-CLIP-g/14+<br><br>EVA-02-CLIP-E/14+ EVA-CLIP-8B + InternVL-C + EVA-CLIP-18B +<br><br><br>|1 1 1 1 1 1 1 1 1<br><br>|76.0 65.4 64.5 58.8<br><br>78.2 65.2 65.5 59.2<br><br>79.2 66.7 67.0 60.7<br><br>80.5 67.1 66.9 60.3 78.9 67.3 67.3 61.5 83.1 70.7 70.0 64.4 85.7 71.3 71.2 66.1<br><br><br>85.2 71.8 71.7 66.4<br><br>86.0 72.9 72.9 68.2<br><br><br>|66.2 67.0 68.4 68.7 68.8 72.1 73.6 73.7 75.0<br><br>|
|EVA-CLIP-18B + EVA-CLIP-18B +<br><br>|8 16<br><br>|88.2 79.3 79.2 72.1 88.4 79.4 79.4 72.2<br><br>|79.7 79.8<br><br>|

Table 5: EVA-CLIP zero-shot video classification performance. We report top1 accuracy for UCF-101 [60], average of top1 and top5 accuracy for Kinetics-400 [15], Kinetics-600 [13] and Kinetics-700 [14].

Zero-Shot Video Classification. We report the top-1 accuracy for UCF-101 [60] and the mean of top-1 and top-5 accuracy for Kinetics-400 [15], Kinetics-600 [13] and Kinetics700 [14]. In Table 5 we demonstrate that EVA-CLIP-18B also outperforms other CLIP models on zero-shot video clas-

|method+<br><br>DFN5B-CLIP-H/14+|IN-1K IN-A IN-R IN-V2 IN-Sketch ObjectNet 83.5 71.7 92.9 77.4 72.8 76.7|∆↓ 4.4|avg. acc.<br><br>79.2|
|---|---|---|---|
|OpenCLIP-G/14+|80.4 69.3 92.8 73.6 69.9 73.0|3.9|76.5|
|SigLIP-SO [75] (reported)|82.0 71.9 95.1 76.1 74.0 70.6|3.7|78.3|
|DFN5B-CLIP-H/14+|84.3 79.6 93.6 78.3 73.3 79.6|2.8|81.5|
|EVA-01-CLIP-g/14 +|78.5 73.6 92.5 71.5 67.6 72.3|2.5|76.0|
|EVA-01-CLIP-g/14+|79.3 74.1 92.7 72.5 68.4 75.3|2.2|77.1|
|BASIC-L [52] (reported)|85.7 85.6 95.7 80.6 76.1 82.3|1.4|84.3|
|SigLIP-SO+ [75] (reported)|83.0 82.5 95.8 77.2 74.5 77.0|1.3|81.7|
|EVA-02-CLIP-E/14+|82.1 82.1 94.7 75.7 72.2 79.6|1.0|81.1|
|InternVL-C +|83.2 83.8 95.7 77.3 74.3 80.6|0.7|82.5|
|EVA-CLIP-8B +|83.5 85.2 95.3 77.7 74.3 81.2|0.6|82.9|
|EVA-CLIP-18B +|83.8 87.3 95.7 77.9 74.7 82.2|0.2|83.6|

- (a) Zero-shot performance on ImageNet variants and ObjectNet. “avg. acc.”: the averaged top-1 accuracy on different ImageNet variants (i.e., IN-{1K, V2, ReaL, Adv., Ren., Ske.}), and ObjectNet. “∆↓”: The gap between the averaged top-1 accuracy and the ImageNet-1K top-1 accuracy (the lower the better). EVA-CLIP suffers from the smallest performance drop (only 0.2% top-1 accuracy gap for EVA-CLIP-18B) while EVA-CLIP-18B achieves 83.6% top-1 accuracy averaged on all 6 benchmarks.

|method<br><br>|ImageNet-1K[26]<br><br>ImageNet-V2[57]<br><br>ImageNet-Adv.[36]<br><br>ImageNet-Ren.[35]<br><br>ImageNet-Ske.[68]<br><br>ObjectNet[8]|CIFAR-10[40]<br><br>CIFAR-100[40]<br><br>MNIST[41]<br><br>SUN397[72]<br><br>Birdsnap[9]<br><br>DTD[21]<br><br>EuroSAT[34]<br><br>Food-101[10]<br><br>PCam[67]<br><br>RESISC45[18]<br><br>STL-10[23]<br><br>|avg.top-1acc..|
|---|---|---|---|
|BASIC-L [52] (reported) EVA-CLIP-18B +<br><br>|85.7 80.6 85.6 95.7 76.1 82.3 83.8 77.9 87.3 95.7 74.7 82.2 -1.9 -2.7 +1.7 +0.0 -1.4 -0.1<br><br>|97.5 82.3 40.3 76.2 59.2 64.6 51.0 95.1 59.6 72.7 99.6 99.4 93.8 83.0 77.7 79.9 72.1 79.8 95.8 65.2 76.9 99.6 +1.9 +11.5 +42.7 +1.5 +20.7 +7.5 +28.8 +0.7 +5.6 +4.2 +0.0<br><br>|76.7 (77.8) 84.9 (84.1) +8.2 (+6.3)<br><br>|

- (b) Comparison EVA-CLIP-18B’s zero-shot image classification performance with BASIC-L [52] on 17 datasets. Our report includes the top-1 accuracy for all datasets, considering that BASIC-L only provided top-1 accuracy for these specific 17 datasets. ( ) is the average top-1 accuracy removing Birdsnap due to the different test size between EVA-CLIP-18B and BASIC-L. EVA-CLIP-18B outperforms BASIC-L with a notable margin of +8.2 (+6.3) in average top-1 accuracy, despite exhibiting lower performance on ImageNet variants.

Table 6: Robustness evaluation of CLIP models and comparison with BASIC-L [52] on 17 Benchmarks.

sification benchmarks by a large margin. When sampling a single center frame per video, EVA-CLIP-18B achieves accuracies of 86.0%, 72.9%, 72.9%, and 68.2% across the four evaluated benchmarks. Further, when uniformly sample 8 or 16 frames per video, we observe an improvement of +4.7% / +4.8% averaged across four benchmarks compared to the single-frame setting.

Zero-Shot Image-Text Retrieval. In Table 3, we report the zero-shot image and text retrieval results on Flickr30K [74] and COCO [45]. EVA-CLIP-18B achieves an average recall of 87.8% across all retrieval benchmarks, significantly outperforming competitors.

Robustness. In Table 6, we demonstrate that scaling up EVA-CLIP significantly enhances the robustness of visual representations. EVA-CLIP suffers from the smallest performance drop (∆↓) between ImageNet-1K and ImageNet variants including adversarial ones, with merely 0.2% top-1 accuracy gap for EVA-CLIP-18B.

For a more robust and comprehensive evaluation of robustness and zero-shot performance, it is advisable to include more benchmarks covering more image distributions. However, we want to note that higher ImageNet top-1 accuracy

does not necessarily lead to better overall performance, as evidenced in Table 6b, where BASIC-L [52] exhibits higher ImageNet-related top-1 accuracy but considerably lower overall average top-1 accuracy compared to EVA-CLIP18B across a broader range of datasets and distributions, showing a difference of -8.2%.

Linear Probing on ImageNet-1K. In Table 7, we present the results of linear probing on ImageNet-1K [26]. EVACLIP-18B achieves an average top-1 accuracy of 88.9%, surpassing InternVL-C [17] by 0.7%.

|method<br><br>|#param|top1 acc.<br><br>|
|---|---|---|
|OpenCLIP-G/14 (reported)<br><br>EVA-01-CLIP-g/14 +<br><br>EVA-02-CLIP-E/14+ InternVL-C (reported) EVA-CLIP-8B + EVA-CLIP-18B +<br><br><br>|1.8B 1.0B<br><br>4.4B<br>5.9B 7.5B<br><br><br>17.5B|86.2 86.5 88.1 88.2 88.5 88.9<br><br>|

Table 7: Linear Probing on ImageNet-1K [26]. The top-1 accuracy shows a continuous improvement with the scaling up of EVA-CLIP.

3D Representation. We adopt the Uni3D [77] setting to explore the effectiveness of scaling up teachers. With the scaling up of EVA-CLIP in Table 8, we observe consistent improvements in 3D representation learning capabilities. Further, Uni3D-base equipped with EVA-CLIP-18B sets new records on ModelNet [71] and ScanObjectNN [66] benchmarks.

|teacher|data<br><br>|O-LVIS MNet40 ScanObjNN|
|---|---|---|
|OpenCLIP-G/14 + EVA-02-CLIP-E/14+ EVA-CLIP-8B + EVA-CLIP-18B +<br><br>|w/o LVIS w/o LVIS w/o LVIS w/o LVIS<br><br>|44.5 85.8 58.9<br><br>45.8 86.1 61.7<br><br>46.2 87.3 62.7<br><br>47.0 87.6 65.3<br><br><br>|
|EVA-02-CLIP-E/14+ EVA-CLIP-18B +<br><br>|Ensembled Ensembled<br><br>|51.7 86.3 63.8 53.2(+1.5) 88.6(+2.3) 67.8(+4.0)<br><br>|

- Table 8: EVA-CLIP-18B enhances zero-shot 3d classification performance. We use Uni3D-base [77] as the baseline and scale the teacher from 5B to 18B. We report top-1 accuracy on ObjaverseLVIS [25], ModelNet40 [71] and ScanObjectNN [66].

4. Ablation Studies

Video Data. In Table 9, we conduct ablations on EVACLIP-18B’s zero-shot performance, comparing results when trained with and without Merged-Video. The training objective for the video data aligns with that of images, encompassing the extraction of features from video where 8 frames are uniformly sampled. The mean of all [CLS] embeddings serves as a representation for the video. The outcomes reveal substantial performance improvements associated with training using Merged-Video. The zero-shot performance, averaged across UCF-101 [60] and Kinetics-400 [15] / 600 [13] / 700 [14], indicates a gain of +0.7 for evaluation with one middle frame and +0.8 for evaluation with 8 frames.

| |classification image video (#F 1) video (#F 8)|retrieval avg. recall<br><br>|
|---|---|---|
|w/o video data w/ video data<br><br>|80.7 74.3 78.9<br>80.7 75.0 (+0.7) 79.7 (+0.8)<br>|87.9 87.8 (-0.1)<br><br>|

- Table 9: Video data enhances zero-shot video classification performance. We respectively report performances averaged on 27 image classification benchmarks, 4 video benchmarks and 2 imagetext retrieval benchmarks.

Image Resolution. In Table 10, we investigate the impact of larger image resolutions on zero-shot performance. Notably, there is an average top-1 accuracy gain of +0.9 when the resolution increases from 2242 to 4482 for EVA-CLIP-8B. Similarly, an increase from 2242 to 3362 results in a gain of +0.5, even when trained with low global batch sizes of 24k for EVA-CLIP-8B + and 23k for EVA-CLIP-18B+.

|method+<br><br>|resolution<br><br>|IN-1K IN-A IN-R IN-V2 IN-Ske. ObjectNet|avg.|
|---|---|---|---|
|EVA-CLIP-8B + EVA-CLIP-8B +<br><br>|224×224 448×448<br><br>|83.5 85.2 95.3 77.7 74.3 81.2 83.8 88.7 95.4 77.7 74.1 82.9 +0.3 +3.5 +0.1 +0.0 -0.2 +1.7<br><br>|82.9 83.8 +0.9<br><br>|

EVA-CLIP-18B + 224×224 83.8 87.3 95.7 77.9 74.7 82.2 83.6 EVA-CLIP-18B+ 336×336 83.9 88.9 95.6 78.2 74.3 83.6 84.1

###### +0.1 +1.6 -0.1 +0.3 -0.4 +1.4 +0.5

Table 10: Increasing resolution. We report zero-shot performance on ImageNet variants and ObjectNet.

### 5. Conclusion

We present EVA-CLIP-18B, the currently largest and most performant open-sourced CLIP model with 18-billion parameters. We show that following EVA’s weak-to-strong vision scaling principle, we can further scale up CLIP models to a new record and advance SOTA on multiple prevalent benchmarks across image, video and 3D domains. Importantly, we demonstrate that scaling up the size of EVA-CLIP models consistently boosts performance with no sign of saturation, shedding light on future vision model scaling.

### References

- [1] Laion coco: 600m synthetic captions from laion2b-en. https: //laion.ai/blog/laion-coco/. 3
- [2] Reaching 80 zero-shot accuracy with openclip: Vit-g/14 trained on laion-2b. https://laion.ai/blog/giant-openclip/. 2
- [3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. arXiv preprint arXiv:2204.14198, 2022. 1
- [4] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016. 2
- [5] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023. 1
- [6] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1728–1738, 2021. 3
- [7] Hangbo Bao, Li Dong, and Furu Wei. Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254, 2021. 9
- [8] Andrei Barbu, David Mayo, Julian Alverio, William Luo, Christopher Wang, Dan Gutfreund, Josh Tenenbaum, and Boris Katz. Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. In NeurIPS, 2019. 2, 4, 9, 10
- [9] Thomas Berg, Jiongxin Liu, Seung Woo Lee, Michelle L Alexander, David W Jacobs, and Peter N Belhumeur. Birdsnap: Large-scale fine-grained visual categorization of birds. In CVPR, 2014. 2, 4, 9, 10
- [10] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101– mining discriminative components with random forests. In ECCV,

2014. 2, 4, 9, 10

- [11] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. arXiv preprint arXiv:2005.14165,

2020. 1

- [12] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Imagetext pair dataset. https://github.com/kakaobrain/ coyo-dataset, 2022. 3
- [13] Joao Carreira, Eric Noland, Andras Banki-Horvath, Chloe Hillier, and Andrew Zisserman. A short note about kinetics-600. arXiv preprint arXiv:1808.01340, 2018. 3, 5, 9
- [14] Joao Carreira, Eric Noland, Chloe Hillier, and Andrew Zisserman. A short note on the kinetics-700 human action dataset. arXiv preprint arXiv:1907.06987, 2019. 3, 5, 9
- [15] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In CVPR, 2017. 3, 5, 9
- [16] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost, 2016. 3
- [17] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023. 1, 2, 3, 4
- [18] Gong Cheng, Junwei Han, and Xiaoqiang Lu. Remote sensing image scene classification: Benchmark and state of the art. Proceedings of the IEEE, 2017. 2, 4, 9, 10
- [19] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning, 2022. 1
- [20] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311,

2022. 1

- [21] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, , and A. Vedaldi. Describing textures in the wild. In CVPR, 2014. 2, 4, 9, 10
- [22] Kevin Clark, Minh-Thang Luong, Quoc V Le, and Christopher D Manning. ELECTRA: Pre-training text encoders as discriminators rather than generators. arXiv preprint arXiv:2003.10555, 2020. 9

- [23] Adam Coates, Andrew Ng, and Honglak Lee. An analysis of singlelayer networks in unsupervised feature learning. In AISTAT, 2011. 2, 4, 9, 10
- [24] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memory-efficient exact attention with io-awareness, 2022. 3
- [25] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13142–13153, 2023. 5
- [26] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 2, 4, 9, 10
- [27] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. ”the PASCAL Visual Object Classes Challenge 2007 (VOC2007) Results. ”http://www.pascalnetwork.org/challenges/VOC/voc2007/workshop/index.html”, 2007. 2, 9, 10
- [28] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. arXiv preprint arXiv:2309.17425, 2023. 1, 2, 3
- [29] Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva-02: A visual representation for neon genesis. arXiv preprint arXiv:2303.11331, 2023. 1, 3
- [30] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. arXiv preprint arXiv:2211.07636, 2022. 1, 2, 3
- [31] Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In CVPRW, 2004. 2, 9, 10
- [32] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, Eyal Orgad, Rahim Entezari, Giannis Daras, Sarah Pratt, Vivek Ramanujan, Yonatan Bitton, Kalyani Marathe, Stephen Mussmann, Richard Vencu, Mehdi Cherti, Ranjay Krishna, Pang Wei Koh, Olga Saukh, Alexander Ratner, Shuran Song, Hannaneh Hajishirzi, Ali Farhadi, Romain Beaumont, Sewoong Oh, Alex Dimakis, Jenia Jitsev, Yair Carmon, Vaishaal Shankar, and Ludwig Schmidt. Datacomp: In search of the next generation of multimodal datasets. arXiv preprint arXiv:2304.14108,

2023. 2

- [33] Ian J Goodfellow, Dumitru Erhan, Pierre Luc Carrier, Aaron Courville, Mehdi Mirza, Ben Hamner, Will Cukierski, Yichuan Tang, David Thaler, Dong-Hyun Lee, et al. Challenges in representation learning: A report on three machine learning contests. In ICONIP, 2013. 2, 9, 10
- [34] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens., 2019. 2, 4, 9, 10
- [35] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In CVPR, 2021. 2, 4,

- 9, 10

[36] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In CVPR, 2021. 2, 4, 9,

- 10

- [37] Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In ECCV, 2016. 9
- [38] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip. https://github. com/mlfoundations/open_clip, 2021. 3
- [39] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In ICCVW, 2013. 2, 9, 10
- [40] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. 2, 4, 9, 10
- [41] Yann LeCun, L´eon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 1998. 2, 4, 9, 10
- [42] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597,

2023. 1

- [43] Xianhang Li, Zeyu Wang, and Cihang Xie. Clipa-v2: Scaling clip training with 81.1arXiv preprint arXiv:2306.15658, 2023. 1
- [44] Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, and Kaiming He. Scaling language-image pre-training via masking,

2022. 3, 9

- [45] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 3, 4, 9
- [46] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023. 1
- [47] Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013. 2, 9, 10
- [48] Arsha Nagrani, Paul Hongsuck Seo, Bryan Seybold, Anja Hauth, Santiago Manen, Chen Sun, and Cordelia Schmid. Learning audiovideo modalities from image captions, 2022. 3
- [49] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In ICVGIP, 2008. 2, 9, 10
- [50] Ting Pan, Lulu Tang, Xinlong Wang, and Shiguang Shan. Tokenize anything via prompting. arXiv preprint arXiv:2312.09128, 2023. 1
- [51] Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. Cats and dogs. In CVPR, 2012. 2, 9, 10
- [52] Hieu Pham, Zihang Dai, Golnaz Ghiasi, Hanxiao Liu, Adams Wei Yu, Minh-Thang Luong, Mingxing Tan, and Quoc V Le. Combined scaling for zero-shot transfer learning. arXiv preprint arXiv:2111.10050, 2021. 4
- [53] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 1, 2, 3, 9, 10
- [54] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J Liu, et al. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020. 1
- [55] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20, 2020. 3, 9

- [56] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In KDD, 2020. 3, 9
- [57] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do imagenet classifiers generalize to imagenet?, 2019. 2, 4, 9, 10
- [58] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. arXiv preprint arXiv:2210.08402, 2022. 2, 3
- [59] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 2
- [60] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012. 3, 5, 9
- [61] Johannes Stallkamp, Marc Schlipsing, Jan Salmen, and Christian Igel. Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. Neural networks, 2012. 2, 9, 10
- [62] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. arXiv preprint arXiv:2312.13286, 2023. 1
- [63] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023. 1, 2, 3
- [64] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222, 2023. 1
- [65] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1, 2
- [66] Mikaela Angelina Uy, Quang-Hieu Pham, Binh-Son Hua, Thanh Nguyen, and Sai-Kit Yeung. Revisiting point cloud classification: A new benchmark dataset and classification model on real-world data. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1588–1597, 2019. 5
- [67] Bastiaan S Veeling, Jasper Linmans, Jim Winkens, Taco Cohen, and Max Welling. Rotation equivariant cnns for digital pathology. In MICCAI, 2018. 2, 4, 9, 10
- [68] Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. Learning robust global representations by penalizing local predictive power. NeurIPS, 2019. 2, 4, 9, 10
- [69] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023. 1
- [70] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, Conghui He, Ping Luo, Ziwei Liu, Yali Wang, Limin Wang, and Yu Qiao. Internvid: A large-scale video-text dataset for multimodal understanding and generation, 2024. 3

- [71] Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, and Jianxiong Xiao. 3d shapenets: A deep representation for volumetric shapes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1912– 1920, 2015. 5
- [72] Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In CVPR, 2010. 2, 4, 9, 10
- [73] Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. Large batch optimization for deep learning: Training bert in 76 minutes, 2019. 3, 9
- [74] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. TACL, 2014. 3, 4, 9
- [75] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. arXiv preprint arXiv:2303.15343, 2023. 1, 4
- [76] Biao Zhang and Rico Sennrich. Root mean square layer normalization. arXiv preprint arXiv:1910.07467, 2019. 2
- [77] Junsheng Zhou, Jinsheng Wang, Baorui Ma, Yu-Shen Liu, Tiejun Huang, and Xinlong Wang. Uni3d: Exploring unified 3d representation at scale. arXiv preprint arXiv:2310.06773, 2023. 1, 5
- [78] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592,

2023. 1

## EVA-CLIP-18B: Scaling CLIP to 18 Billion Parameters Supplementary Material

Dataset Classes Test size Evaluation Metric ImageNet-1K [26] 1000 50,000 accuracy ImageNet-V2 [57] 1000 10,000 accuracy ImageNet-Adversarial [36] 1000 7,500 accuracy ImageNet-R(endition) [35] 1000 30,000 accuracy ImageNet-Sketch [68] 1000 50,899 accuracy ObjectNet [8] 1000 50,273 accuracy CIFAR-10 [40] 10 10,000 accuracy CIFAR-100 [40] 100 10,000 accuracy MNIST [41] 10 10,000 accuracy Caltech101 [31] 101 9144 accuracy SUN397 [72] 397 108,754 accuracy FGVC Aircraft [47] 100 3,333 accuracy Country-211 [53] 211 21,100 accuracy Stanford Cars [39] 196 8,041 accuracy Birdsnap [9] 500 2,195 accuracy Describable Textures [21] 47 1,880 accuracy EuroSAT[34] 10 27,000 accuracy Facial Emotion Recognition 2013 [33] 8 3,574 accuracy Oxford Flowers 102 [49] 102 6,149 accuracy Food-101 [10] 102 25,250 accuracy GTSRB [61] 43 12,630 accuracy PatchCamelyon [67] 2 32,768 accuracy Oxford-IIIT Pets [51] 37 3,669 accuracy Rendered SST2 [53] 2 1,821 accuracy RESISC45 [18] 45 31,500 accuracy STL-10 [23] 10 8000 accuracy Pascal VOC 2007 Classification [27] 20 4,952 accuracy UC-F101 [60] 101 11,213 accuracy Kinetics-400 [15] 400 19,240 mean(top1, top5) Kinetics-600 [13] 600 29,788 mean(top1, top5) Kinetics-700 [14] 700 33,966 mean(top1, top5) Flickr30K [74] - 1000 recall COCO [45] - 5000 recall

Table 11: Datasets used to evaluate EVA-CLIP models.

|config<br><br>|EVA-CLIP-{8B, 8B+}|
|---|---|
|image enc. weight init. text enc. weight init. image-text data image enc. peak learning rate image enc. layer-wise lr decay [22, 7] text enc. peak learning rate text enc. layer-wise lr decay [22, 7] learning rate schedule optimizer optimizer hyper-parameters weight decay input resolution patch size batch size samples seen drop image patch [44] drop path [37] random resized crop numerical precision ZeRO optimizer [55]<br><br>|EVA-8B / EVA-CLIP-8B EVA-02-CLIP-E/14+ / EVA-CLIP-8B Merged-2B<br><br>4e-4 / 2e-4 0.9 / 0.85<br>4e-5 / 2e-5 0.75<br><br><br>cosine decay LAMB [73] β1, β2, ϵ = 0.9, 0.95, 1e-6 0 2242 / 4482 142 178k / 24k 9B / 800M 0.5 / 0.0 0.0 (0.9, 1) DeepSpeed bf16 [56] stage 3|

##### Table 12: EVA-CLIP-8B and EVA-CLIP-8B+ training settings.

|config|EVA-CLIP-{18B,18B+}|
|---|---|
|image enc. weight init. text enc. weight init. image-text data image enc. peak learning rate image enc. layer-wise lr decay [22, 7] text enc. peak learning rate text enc. layer-wise lr decay [22, 7] learning rate schedule optimizer optimizer hyper-parameters weight decay input resolution patch size batch size samples seen drop image patch [44] drop path [37] random resized crop numerical precision ZeRO optimizer [55]<br><br>|EVA-18B / EVA-CLIP-18B EVA-02-CLIP-E/14+ / EVA-CLIP-18B Merged-2B+<br><br>4e-4 / 2e-4 0.9 / 0.85<br>4e-5 / 2e-5 0.75<br><br><br>cosine decay LAMB [73] β1, β2, ϵ = 0.9, 0.95, 1e-6 0 2242 / 3362 142 108k / 23k 6B / 400M 0.5 / 0.0 0.0 (0.9, 1) DeepSpeed bf16 [56] stage 3|

### 6. Training Settings

We present detailed training settings of EVA-CLIP-8B and EVA-CLIP-18B in Tabs. 12 and 13.

### 7. Image Transformations for Evaluation

Two prevalent image transformations utilized in zero-shot evaluation are: 1) direct resizing of images to a fixed size, such as 224×224, and 2) resizing images based on the shortest side, followed by center cropping to achieve a fixed size. In Table 14, our study systematically investigates the impact

##### Table 13: EVA-CLIP-18B and EVA-CLIP-18B+ training settings.

of these two image transformations in zero-shot evaluations. Notably, there exists a significant performance gap between the two transformations, observed particularly in zero-shot image classification on ObjectNet [8] and VOC2007 [27], and zero-shot retrieval on Flickr30K [74] and COCO [45]. EVA-CLIP-18B shows robustness with almost the same average accuracy across different image transformations in zero-shot image/video classification.

For zero-shot image classification and video classification,

RenderedSST2[53]

|method|ImageNet-1K[26]<br><br>ImageNet-V2[57]<br><br>ImageNet-Adv.[36]<br><br>ImageNet-Ren.[35]<br><br>ImageNet-Ske.[68]<br><br>ObjectNet[8]<br><br>|CIFAR-10[40]<br><br>CIFAR-100[40]<br><br>MNIST[41]<br><br>Caltech101[31]<br><br>SUN397[72]<br><br>FGVCAircraft[47]<br><br>Country-211[53]<br><br>StanfordCars[39]<br><br>Birdsnap[9]<br><br>DTD[21]<br><br>EuroSAT[34]<br><br>FER2013[33]<br><br>Flowers-102[49]<br><br>Food-101[10]<br><br>GTSRB[61]<br><br>PCam[67]<br><br>Pets[51]<br><br>RenderedSST2[53]<br><br>RESISC45[18]<br><br>STL-10[23]<br><br>VOC2007[27]|avg.top-1acc..|
|---|---|---|---|
|DFN5B-CLIP-H/14+* DFN5B-CLIP-H/14+†<br><br>|84.0 77.8 79.6 92.9 72.4 79.6 84.3 78.3 79.3 93.6 73.3 73.5<br><br>|98.8 90.5 83.6 88.7 77.0 64.9 36.1 95.7 80.5 70.9 61.1 56.1 91.6 96.1 67.8 69.6 96.7 55.5 75.9 99.1 78.2 98.8 90.5 83.6 88.9 77.4 72.5 37.9 96.0 80.3 70.9 61.1 56.1 91.4 96.2 67.9 69.6 96.8 55.5 75.9 99.1 81.9|78.5 78.9<br><br>|

ImageNet-Adv.[36]

ImageNet-Ren.[35]

FGVCAircraft[47]

ImageNet-Ske.[68]

StanfordCars[39]

ImageNet-1K[26]

ImageNet-V2[57]

###### avg.top-1acc..

Country-211[53]

Flowers-102[49]

CIFAR-100[40]

Caltech101[31]

RESISC45[18]

CIFAR-10[40]

VOC2007[27]

Food-101[10]

FER2013[33]

EuroSAT[34]

ObjectNet[8]

SUN397[72]

GTSRB[61]

MNIST[41]

STL-10[23]

Birdsnap[9]

PCam[67]

DTD[21]

Pets[51]

EVA-CLIP-18B * 83.7 77.9 87.3 95.6 74.4 82.2 99.4 93.8 83.0 89.4 77.5 58.4 41.8 94.9 79.9 71.9 79.8 59.3 85.9 95.8 72.4 65.2 96.0 67.5 76.8 99.6 82.4 80.4 EVA-CLIP-18B † 83.8 77.7 86.2 95.7 74.7 76.2 99.4 93.8 83.0 89.8 77.7 59.7 43.1 94.9 78.4 72.1 79.8 59.3 86.0 95.7 72.3 65.2 96.1 67.5 76.9 99.6 85.8 80.4

(a) Impact of image transformations on zero-shot image classification performance. Different transformations can significantly influence zero-shot image classification performance, particularly for ObjectNet [8]. EVA-CLIP-18B shows robustness with the same average top-1 accuracy across different image transformations.

|method+|#Frames|UCF-101 K-400 K-600 K-700<br><br>|avg. acc.|
|---|---|---|---|
|DFN5B-CLIP-H/14+* DFN5B-CLIP-H/14+†<br><br>|1 1|78.5 65.2 66.0 59.2<br><br>79.2 66.7 67.0 60.7<br><br><br>|67.2 68.4|

EVA-CLIP-18B * 1 86.0 72.2 72.6 67.4 74.6 EVA-CLIP-18B † 1 85.6 72.9 72.9 68.2 74.9

EVA-CLIP-18B * 8 88.2 79.3 79.2 72.0 79.7 EVA-CLIP-18B † 8 87.9 79.2 79.1 72.1 79.6

(b) Impact of image transforms on zero-shot video classification performance.

zero-shot text retrieval zero-shot image retrieval

Flickr30K COCO Flickr30K COCO

method+ R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 MR DFN5B-CLIP-H/14 +* 92.3 99.1 99.7 70.6 89.6 94.4 80.7 95.5 97.7 54.1 78.0 85.4 86.4 DFN5B-CLIP-H/14 +† 93.6 99.3 99.6 71.8 90.4 94.9 82.1 96.0 97.9 55.6 79.2 86.3 87.2

EVA-CLIP-18B * 95.4 99.5 99.8 72.8 89.7 94.3 83.2 95.9 97.8 55.6 77.9 85.3 86.7 EVA-CLIP-18B † 96.7 99.7 100.0 73.6 90.9 95.0 83.3 96.3 98.3 56.2 78.5 85.6 87.8

(c) Impact of image transforms on zero-shot retrieval performance.

Table 14: Impact of image transformations on zero-shot evaluation. † denotes the direct resizing of images to a fixed size, while * indicates resizing images based on the shortest side and subsequently center cropping to achieve a fixed size.

we present results obtained by selecting the best-performing transformation between the two. In the case of zero-shot retrieval tasks, we specifically choose the transformation that involves direct resizing of images to a fixed size.

