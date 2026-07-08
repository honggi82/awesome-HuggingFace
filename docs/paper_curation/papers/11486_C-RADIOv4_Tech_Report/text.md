# arXiv:2601.17237v1[cs.CV]24Jan2026

[Figure 1]_images/imageFile1.png>)

2026-1-27

## C-RADIOv4 (Tech Report)

Mike Ranzinger* Greg Heinrich* Collin McCarthy Jan Kautz Andrew Tao Bryan Catanzaro Pavlo Molchanov

By leveraging multi-teacher distillation, agglomerative vision backbones provide a unified student model that retains and improves the distinct capabilities of multiple teachers. In this tech report, we describe the most recent release of the C-RADIO family of models, C-RADIOv4, which builds upon AM-RADIO/RADIOv2.5 in design, offering strong improvements on key downstream tasks at the same computational complexity. We release -SO400M (412M params), and -H (631M) model variants, both trained with an updated set of teachers: SigLIP2, DINOv3, and SAM3. In addition to improvements on core metrics and new capabilities from imitating SAM3, the C-RADIOv4 model family further improves any-resolution support, brings back the ViTDet option for drastically enhanced efficiency at high-resolution, and comes with a permissive license.

Links: Code (on GitHub) | Models (on Hugging Face)

### 1. Description

AM-RADIO [17] introduced the concept of agglomerative foundation models, which is a method of creating a new foundation model by distilling the feature representations from other heterogeneous models. In the original formulation, we used DFN CLIP [11], DINOv2 [8], and SAM [13] as the core teacher set. We also introduced multi-resolution training in the sense that DFN CLIP and DINOv2 were distilled at one resolution, and SAM was distilled at a higher resolution. We identified that there was a phenomenon we called “mode switching” where the student model learned to change its representations based on the resolution in order to minimize the training loss, and it caused very inconsistent behavior during inference depending on the resolution used. In PHI-S [16] we identified teacher distribution balancing as an important component to training strong agglomerative models, in RADIOv2.5 [12] we found that training against all teachers at both resolutions was sufficient to overcome the mode switching issue, and in FeatSharp [18] we proposed an upsampling method which is suitable for certain fixed-resolution models such as DFN CLIP and SigLIP [24, 21], which is preferable compared to bilinear resampling. One of the key points made in the original AM-RADIO was that improved teachers tend to yield improved students, and this trend continues to hold. Since the previous works, SigLIP2 [21] has become the frontier text-image foundation encoder, DINOv3 [19] has pushed the boundaries of self-supervised learning (SSL) and dense representations, yielding an incredibly strong dense perception model, and SAM has upgraded to SAM3 [6] making large inroads toward solving computer vision. In keeping with our initial premise, we have upgraded

∗Equal Contribution

the core set of teachers to [SigLIP2, DINOv3, SAM3], to upgrade our agglomerative model. We inherit DINOv3’s improved semantic segmentation capability, and SigLIP2’s enhanced text alignment. SAM3 as a teacher doesn’t show improved metrics on our selected benchmarks, but including it as a teacher allows for replacing the vision encoder backbone of SAM3, along with creative use cases which exploit agglomerative models, such as that in RADSeg [3]. Beyond simply updating our teacher set, we further our commitment to versatility by improving the ability of our model to operate at any resolution, and bring back “ViTDet mode”, allowing most of the transformer blocks to operate in windowed mode, which has a dramatic effect on inference speed on high-resolution images (see figure 9).

### 2. Method Updates

Agglomerative models rely on distillation from multiple vision foundation models. For the RADIO family of models, the distillation operates over both the dense feature space and summarization tokens. In this section, we describe the novel techniques employed to develop the improved C-RADIOv4 model over previous variants in the form of what has changed since we fully described the method in RADIOv2.5 [12].

2.1. Updated Teachers For C-RADIOv4, we update the teacher set to:

- • SigLIP2-g-384 [21]
- • DINOv3-7B [19]
- • SAM3 [6]

In order to reduce the computational demand, we

© 2026 NVIDIA. All rights reserved.

|Model Name Variant Params<br><br>|Summary Zero Shot kNN|Segmentation ADE20k VOC|Probe3d NAVI SPair|
|---|---|---|---|
|SigLIP2<br><br>SO400M 412M g 1,164M|83.88 85.76<br><br>84.75 86.39<br><br><br>|44.0 42.7 72.7|48.8 38.7<br>49.40 42.60<br><br><br>|
|DINOv3<br><br>H+ 841M 7B 6,716M<br><br>|- 85.77<br>- 85.42<br>|54.8<br>55.9 86.6<br>|63.3 56.3<br>64.4 58.7<br>|
|RADIOv2.5 H 631M C-RADIOv3 H 631M|82.51 85.81 82.65 86.23|51.58 85.97<br>52.75 86.41<br>|60.89 56.24 62.10 58.54<br><br>|
|C-RADIOv4<br><br>SO400M 412M H 631M|82.01 85.76<br>83.09 86.59<br>|55.14 87.22 55.20 87.24<br><br>|62.44 60.01<br>63.44 60.57<br><br><br>|

- Table 1 | Comparison between different vision encoders. C-RADIOv4 is competitive with DINOv3 on dense tasks at a fraction of the parameters.

[Figure 2]_images/imageFile2.png>)

- Figure 1 | PCA feature visualization, comparing C-RADIOv3-H vs C-RADIOv4-H. Object boundaries are now significantly cleaner.

ADE20k 512px 1024px 1536px DINOv3-7B 55.9 57.3† 57.8†

Model

C-RADIOv4-H 55.20 57.02 57.72

- Table 2 | Comparison of resolution scaling for ADE20k [25] linear probe between DINOv3-7B and C-RADIOv4-H. Both models exhibit strong resolution scaling properties. †The metrics for DINOv3-7B come from figure 11 in [19], where values past the decimal are approximate.

dropped support for DFN CLIP [11] in this release, in favor of the more ubiquitous SigLIP2 [21], as the latter is being deployed in more places (e.g. Qwen3 VL [4]), and because both models have similar representations and application domains.

with 𝑓 being the input-dependent semantics, 𝑔 being a data-invariant bias, and ℎ the entangled residual. We also identified these same noise patterns in the FeatSharp [18] work, notably with the SigLIP2 models, as they have these “holes” along the border of the output feature maps. SAM [13] has strong artifacts at the borders of the ViTDet [14] windows. DINOv3-H+ happens to have somewhat frequent large magnitude noise patches. In all of these cases, left unmitigated, the model will learn to mimic this noise in the MLP adapter, and it will even leak down into the backbone features. We combat this with two different forms of shift equivariance in the loss formulation, both of which make it impossible for the student to know the exact positions of the patches being matched between student and teacher. We visualize these artifacts in figure 2, where high-energy representations appear, often in otherwise mostly uniform image regions.

#### 2.2. Stochastic Resolutions

Instead of training at 2 different resolutions as in RADIOv2.5 [12], we sample from {128,192,224,256,384,432} in the low-resolution partition, and {512,768,1024,1152} in the highresolution partition, enabling an even smoother resolution scaling curve, and also gaining substantial quality at low resolutions. For SigLIP2 [21], we use FeatSharp [18] to do 3× upsampling from 384px to 1152px in our high-resolution training partition. The low-resolution partition uses the raw outputs. Finally, for SAM3, we use the mosaic augmentation as proposed in RADIOv2.5 [12], since it only supports inputs of size 1152 × 1152. Figures 3 and 4 show how multi-resolution support has evolved over time, with figure 4 showing how RADIO models compare against DINOv2/3, as they also support multi-res. In table 2 we show that C-RADIOv4 is strongly robust to increasing the resolution for semantic segmentation, including at resolutions higher than trained. Considering that C-RADIOv4-H has an order of magnitude fewer parameters than DINOv3-7B, it performs very competitively.

#### 2.3. Shift Equivariance

Since PHI-S [16], the formulation of the feature loss has been 𝑍1

∑︀

𝑖(𝑥 − 𝑦^)2 with 𝑦^ being the teacher features normalized by PHI-S, and 𝑥 being the student prediction, and 𝑍 the normalizing constant. A drawback of this approach is that the student is not only learning the useful features of the teacher, but also the fixed pattern noise. In DVT [23], the authors find that all of these vision foundation models have this noise, and they learn to disentangle the models outputs into 𝑉 𝑖𝑇(𝑥) ≈ 𝑓(𝑥) + 𝑔(𝐸𝑝𝑜𝑠) + ℎ(𝑥,𝐸𝑝𝑜𝑠),

#### 2.3.1. Shift Equivariant Loss

To prevent the student from matching dense teacher features which are not explicitly input-dependent semantics [23], for a particular image, we randomly shift the crop seen by the student, and also each teacher. The teachers get independent shifts relative to each other, and the student. We track a mapping ℱ𝑆→𝑇 which transforms the features produced by the student to be spatially aligned with the teacher. The sampled shifts are in increments of the patch size, which eliminates interpolation effects where possible. Thus, we adopt a new loss formulation as follows:

1 |Ω|

𝐿spatial(x,y^) =

∑︁

(ℱ𝑆→𝑇[x]𝑢 − y^𝑢)2 . (1)

𝑢∈Ω

With Ω being the set of common spatial positions seen between student and teacher, x the student output, and y^ the PHI-S [16] normalized teacher output.

#### 2.3.2. Shift Equivariant MESA

MESA [9] is an indispensable tool for training classification models, which attempts to converge the weights to flat regions where perturbations to the input don’t cause chaotic changes in the output and improves generalization. To further combat fixed-pattern noise from emerging in our student model, we apply the MESA formulation of matching the exponential moving average (EMA) of the student model, but with the added twist of introducing different crops for the student and its EMA, and relating them via the ℱ𝑆→𝑆˜ transform, and then use the following formulation:

[Figure 3]_images/imageFile3.png>)

[Figure 4]_images/imageFile4.png>)

[Figure 5]_images/imageFile5.png>)

- Figure 2 | Visualization of DINOv3 and C-RADIOv4 (adapter) predictions. Notice the out-of-place speckles produced by DINOv3.

- Column 1: Input Image
- Column 2: DINOv3 PCA visualization.
- Column 3: C-RADIOv4 adapter PCA visualization.
- Column 4: Error heatmap between the student adapter prediction and the DINOv3 teacher.

1 |Ω|

∑︁

(ℱ𝑆→𝑆˜ [𝐿𝑁(x)]𝑢 − 𝐿𝑁(˜x)𝑢)2

𝐿𝑚𝑒𝑠𝑎(x,x˜) =

𝑢∈Ω

(2)

with 𝐿𝑁 being layer norm, without the learnable affine projection, and 𝑆,˜ x˜ the respective crop and output of the EMA student model.

#### 2.4. DAMP

To further encourage the robustness of our model, we employ DAMP [20], which applies multiplicative noise to the weights of our model during training.

#### 2.5. Balanced Summary Loss

In PHI-S [16], the entire study revolved around normalizing the distributions of the spatial features of each teacher, so that large activations wouldn’t dominate the loss. However, no attention was paid to the summary features. The reason for this was because the summary features were matched using cosine similarity, which is inherently normalized. However, what we have since noticed is that while the magnitudes of the summary features are normalized onto the unit hypersphere, their directional variance is not. If teachers produced features with a uniform distribution over the unit hypersphere, this would not be important, but instead what we found is that features tend to fall into a cone, and the radius of this cone is different for each teacher. The effect of this is that cones with larger radius will produce larger losses than those

#### Model Disp(Θy)

SigLIP2-g-384 0.694 DINOv3-H+ 2.120 DINOv3-7B 2.186

- Table 3 | The angular dispersion of the summary token for key teachers.

with small radius. Further, it’s not so much the angle between student and teacher that matters, but rather that angle relative to all other embeddings. To mitigate this, we no longer use cosine distance as our summary loss, and instead adopt the following:

x⊺y ‖x‖‖y‖

cos(x,y) =

(3) Θ(x,y) = arccos(cos(x,y)) (4)

E[y] ‖E[y]‖

𝜇y =

(5)

Disp(Θy) = E[︁Θ(y,𝜇y)2]︁ (6)

Θ(x,y)2 Disp(Θy)

𝐿𝑎𝑛𝑔𝑙𝑒(x,y) =

(7)

With x the student predictions, y the teacher prediction, 𝜇y being the expected direction of y, and Disp(Θy) being the angular dispersion of y, which we use to normalize the cone radius between teachers, allowing the student to focus on the relative angles, and to prevent one teacher from dominating another. We report the angular dispersions we found in Table 3, showing that there is a significant difference between SigLIP2 and DINOv3. Left unmitigated, DINOv3 would dominate the loss term, biasing the student toward matching it, at the expense of matching SigLIP2.

### 3. Results

#### 3.1. Metrics

In figure 3, we show the zero-shot accuracy of RADIOv2.5 and various versions of C-RADIO. We note that RADIOv2.5 and C-RADIOv2 don’t have a SigLIP2-aligned head, so we report the accuracy using their DFN CLIP head. In general, DFN CLIP is easier for the student model to match, and thus, up until C-RADIOv4, we were unable to match the zero-shot accuracy of our previous RADIOv2.5 model. All models shown exhibit strong resolution scaling, however, C-RADIOv4 strongly improves classification at low resolution relative to previous model generations. We achieve maximum zero-shot score at 1024px resolution, using aspect-preserving resizing.

ImageNet-1k Zero-shot Accuracy vs Resolution

- 76

- 77

- 78

- 79

- 80

- 81

- 82

- 83

Top-1Accuracy

Model

RADIOv2.5-H_clip C-RADIOv2-H_clip

C-RADIOv3-H C-RADIOv4-H

200 400 600 800 1000 Resolution

- Figure 3 | ImageNet-1K zero-shot accuracy as a function of input resolution.

200 400 600 800 1000 Resolution

- 83

- 84

- 85

- 86

Top-1Accuracy

ImageNet-1k kNN Accuracy vs Resolution

Model

RADIOv2.5-H C-RADIOv3-H C-RADIOv4-H DINOv2-g-reg

DINOv3-H+

DINOv3-7B

- Figure 4 | ImageNet-1K kNN accuracy as a function of input resolution.

We’re able to directly compare against DINOv2/3 models using k-NN (k-nearest neighbors) classification, as used by DINOv2 and prior RADIO works. We show these results in figure 4, where we can see DINOv3 has made large improvements on k-NN as compared to DINOv2, however, the ability to scale with resolution is concentrated around 192-256px, and higher resolutions degrade. For RADIO, the CRADIO models drastically improve on RADIOv2.5, with C-RADIOv4 generally being a bit better than C-RADIOv3. This suggests that C-RADIOv3’s issues in zero-shot more likely stem from issues matching SigLIP2 specifically, and not from lacking good holistic representations. Nonetheless, C-RADIOv4 is still an improvement for both tasks. Curiously, DINOv3’s H+ model is better at kNN classification than the 7B model. Starting at 256px, C-RADIOv4-H is able to match or surpass DINOv3.

|Model|Depth<br><br>Surface<br><br>NAVI SPair Normals<br><br>|
|---|---|
|RADIOv2.5-H|85.69 62.46 60.89 56.24<br><br>|
|C-RADIOv2-H C-RADIOv3-H<br><br>|85.02 60.10 59.82 53.98<br>86.18 62.52 62.10 58.54<br>|
|C-RADIOv4-SO440M C-RADIOv4-H|85.29 61.91 62.44 60.01 85.55 61.70 63.44 60.57|

- Table 4 | Full Probe3d [10] evals for various RADIO models. Higher is better for all metrics.

- 3.2. SAM3

Older versions of RADIO had support for the SAM [13] adaptor, which allowed RADIO to replace SAM’s vision encoder, while using their decoding and memory stack to perform the segmentation. This ability has found clever use in works such as RADSeg [3], which uses the SAM head to refine segmentation masks and set a new state of the art for open vocabulary semantic segmentation. CRADIOv4 upgrades our SAM support to SAM3 [6], again allowing the core vision model to be replaced. We have a fork of the SAM3 codebase at https://github.com/mranzinger/sam3-radio demonstrating how to do this replacement. In figures 6 and 7, we demonstrate RADIO’s ability to replace the vision encoder, first with the image demo provided by SAM3, and second on our own image, with different queries. Qualitatively, RADIO has no issues acting in place of SAM3’s Perception Encoder [5] backbone. In table

- 5 we show the benchmark results on SA-Co/Gold [6] instance segmentation. C-RADIOv4 becomes the second-best model, albeit, inheriting an uneven ability to replace SAM3’s vision encoder. On “metaclip_nps” and “sa1b_nps”, which are dominated by natural images, RADIO works quite well, with a smaller gap to SAM3, however, the gap becomes much larger on “fg_sports_equipment” and “wiki_common”. Improving on our ability to better match SAM3 is an open research direction.

C-RADIOv4 supports "ViTDet-mode" [14], which allows the model to operate in either full global attention (e.g. ViTDet-mode disabled), or with mostly windowed attention, and a few global attention layers throughout. This can be controlled with the "vitdet_window_size" flag when constructing the model. SAM3 uses a ViT-L+ architecture, with 4 global layers, and the rest with windows of 24 × 24 tokens. For increased efficiency, C-RADIOv4 supports windows anywhere between 6 × 6 to 32 × 32 tokens, the only limitation being that the window size multiplied by the patch size needs to evenly divide the input image resolution. Smaller windows have the potential to be faster, as it reduces the quadratic penalty of self-attention [22], however, hardware may reduce/e-

###### Latency vs Resolution

SO400M - No VitDet

5000

SO400M - VitDet-8

SO400M - VitDet-16

Huge - No VitDet

4000

Latency(ms/image)

Huge - VitDet-8

Huge - VitDet-16

3000

2000

1000

0

500 1000 1500 2000 2500 3000 3500 4000

Resolution (pixels)

Figure 5 | Latency analysis on A100 for both versions of C-RADIOv4, with and without ViTDet of a specified window size. The latency difference between ViTDet mode with window size 8 and 16 is negligible.

liminate the gap between similar window sizes, and smaller windows may come with a slight degradation in quality. We show the inference times for singleimage inference on an A100 GPU in figure 9. For the SO400M model, a window size ≤ 12 is faster than SAM3’s encoder. A window size of 8 for the ViT-H RADIO is nearly as fast. We also show the latencies for both C-RADIOv4 models for resolutions between 256px and 4096px, with and without ViTDet, in figure 5. While ViTDet doesn’t change the complexity of the model away from 𝑂(𝑇2) with 𝑇 being the number of tokens, due to the fact that 4 layers still employ global attention, it does substantially reduce the growth factor.

The curious case of “person” On the SAM3 github (as of 1/14/2026), there is the github issue 253 that points out that the github example for SAM3 doesn’t work properly with the “person” query. We are able to replicate this behavior. However, with C-RADIOv4 acting as the vision backbone, the query works as expected. We show this behavior in figure 8. The RADIO variant works in global attention mode, as well as with ViTDet with window size 8 (others would probably work too, these are the only two we tested). This seems to suggest that the slight difference in representations between the two vision encoders is having a thresholding effect with this particular query.

### 4. Conclusion

Owing to improvements in base foundation models, particularly DINOv3 [19], as well as improvements to our distillation algorithm, C-RADIOv4 enjoys large improvements over its predecessors. Different from previous releases, we include an SO400M [1]

Text Prompt: "shoe"

SAM3 SAM3 with RADIO

[Figure 6]_images/imageFile6.png>)

[Figure 7]_images/imageFile7.png>)

Box Prompt

Input Prompt SAM3 SAM3 with RADIO

[Figure 8]_images/imageFile8.png>)

[Figure 9]_images/imageFile9.png>)

[Figure 10]_images/imageFile10.png>)

- Figure 6 | Mask results from the SAM3 image demo, either with regular SAM3, or with the vision encoder replaced by RADIO. RADIO is able to replicate the SAM3 results.

SA-Co/Gold Instance Segmentation (cgF1) metaclip_nps sa1b_nps crowded fg_food fg_sports_equipment attributes wiki_common Avg

Model

Human 72.8 OWLv2 24.6

DINO-X 21.3 Gemini 2.5 13.0

SAM3 47.3 53.7 61.1 53.4 65.5 54.9 42.5 54.1

SO400M-VDT8 43.0 44.5 54.9 38.4 38.4 40.3 22.2 40.3 SO400M-G 43.8 45.7 55.9 40.1 39.8 41.6 23.1 41.4 H-VDT8 45.2 48.1 56.6 40.3 45.3 44.0 26.2 43.7 H-VDT12 45.6 48.4 57.3 40.2 46.1 45.2 26.7 44.2 H-G 45.9 48.8 57.4 40.9 46.5 45.9 27.3 44.7

C-RADIOv4

Table 5 | Results on SA-Co/Gold [6] instance segmentation. “VDT” refers to ViTDet window size 𝑊, while “G” refers to global attention throughout.

Text Prompt: "shoe" Text Prompt: "helmet"

[Figure 11]_images/imageFile11.png>)

[Figure 12]_images/imageFile12.png>)

Text Prompt: "bike" Text Prompt: "spectator"

[Figure 13]_images/imageFile13.png>)

[Figure 14]_images/imageFile14.png>)

##### Figure 7 | Text query results with RADIO replacing the SAM3 vision encoder, while keeping the decoder unchanged.

Text Prompt: "person"

SAM3 SAM3 with RADIO

[Figure 15]_images/imageFile15.png>)

[Figure 16]_images/imageFile16.png>)

##### Figure 8 | The SAM3 demo (github issue) is unable to mask with the “person” query, while C-RADIOv4[SO400M,H] works great.

###### Vision Encoder Benchmark: SAM3 vs RADIO

320.0

SAM3

C-RADIOv4-SO400M

- 2 × 102

- 3 × 102

C-RADIOv4-H

259.3

###### InferenceTime(ms)

127.4

121.0

114.5

109.1

107.3

104.2

99.5 92.9

98.8

102

8 12 18 24 72

VitDet Window Size

- Figure 9 | A100 single-image benchmarking results for the vision encoder of SAM3 vs RADIO, for both SO400M and Huge sizes. SAM3 uses a ViT-L+ architecture with window size 24. A window size of 72 is equivalent to full global attention.

version, which is often able to be competitive with ViT-H, while being cheaper. We demonstrate how C-RADIOv4 may be used to replace the vision encoder in SAM3 [6], and with the SO400M variant with ViTDet window size ≤ 12, it’s even faster than the ViT-L+ Perception Encoder [5] that SAM3 uses. Past RADIO models have seen widespread usage, including in the Nemotron Nano V2 VL [15] vision-language model, autonomous vehicles, robotics, OCR document parsing [7], open vocabulary semantic segmentation [3, 2], and many more. Given the commercially permissive license, we hope that both the academic and industrial community will be able to leverage this foundational model to build great things.

### References

- [1] Ibrahim Alabdulmohsin, Xiaohua Zhai, Alexander Kolesnikov, and Lucas Beyer. Getting vit in shape: Scaling laws for compute-optimal model design. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.
- [2] Omar Alama, Avigyan Bhattacharya, Haoyang He, Seungchan Kim, Yuheng Qiu, Wenshan Wang, Cherie Ho, Nikhil Keetha, and Sebastian Scherer. Rayfronts: Open-set semantic ray frontiers for online scene understanding and exploration. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 5930–5937, 2025.

- [3] Omar Alama, Darshil Jariwala, Avigyan Bhattacharya, Seungchan Kim, Wenshan Wang, and Sebastian Scherer. Radseg: Unleashing parameter and compute efficient zero-shot open-vocabulary segmentation using agglomerative models, 2025.
- [4] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report, 2025.
- [5] Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Abdul Rasheed, Junke Wang, Marco Monteiro, Hu Xu, Shiyu Dong, Nikhila Ravi, Shang-Wen Li, Piotr Dollar, and Christoph Feichtenhofer. Perception encoder: The best visual embeddings are not at the output of

- the network. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.
- [6] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, TsungHan Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichtenhofer. Sam 3: Segment anything with concepts, 2025.
- [7] Kateryna Chumachenko, Amala Sanjay Deshmukh, Jarno Seppanen, Ilia Karmanov, Chia-Chih Chen, Lukas Voegtle, Philipp Fischer, Marek Wawrzos, Saeid Motiian, Roman Ageev, Kedi Wu, Alexandre Milesi, Maryam Moosaei, Krzysztof Pawelec, Padmavathy Subramanian, Mehrzad Samadi, Xin Yu, Celina Dear, Sarah Stoddard, Jenna Diamond, Jesse Oliver, Leanna Chraghchian, Patrick Skelly, Tom Balough, Yao Xu, Jane Polak Scowcroft, Daniel Korzekwa, Darragh Hanley, Sandip Bhaskar, Timo Roman, Karan Sapra, Andrew Tao, and Bryan Catanzaro. Nvidia nemotron parse 1.1, 2025.
- [8] Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. In Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024.
- [9] Jiawei Du, Zhou Daquan, Jiashi Feng, Vincent Tan, and Joey Tianyi Zhou. Sharpness-aware training for free. In Advances in Neural Information Processing Systems, 2022.
- [10] Mohamed El Banani, Amit Raj, Kevis-Kokitsi Maninis, Abhishek Kar, Yuanzhen Li, Michael Rubinstein, Deqing Sun, Leonidas Guibas, Justin Johnson, and Varun Jampani. Probing the 3D Awareness of Visual Foundation Models. In CVPR, 2024.
- [11] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks, 2023.
- [12] Greg Heinrich, Mike Ranzinger, Hongxu Yin, Yao Lu, Jan Kautz, Andrew Tao, Bryan Catanzaro, and Pavlo Molchanov. Radiov2.5: Improved baselines for agglomerative vision foundation models. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 22487–22497, 2025.
- [13] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloé Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross B. Girshick. Segment anything. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 3992–4003, 2023.

- [14] Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He. Exploring plain vision transformer backbones for object detection. In Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part IX, page 280–296, Berlin, Heidelberg, 2022. Springer-Verlag.
- [15] NVIDIA, :, Amala Sanjay Deshmukh, Kateryna Chumachenko, Tuomas Rintamaki, Matthieu Le, Tyler Poon, Danial Mohseni Taheri, Ilia Karmanov, Guilin Liu, Jarno Seppanen, Guo Chen, Karan Sapra, Zhiding Yu, Adi Renduchintala, Charles Wang, Peter Jin, Arushi Goel, Mike Ranzinger, Lukas Voegtle, Philipp Fischer, Timo Roman, Wei Ping, Boxin Wang, Zhuolin Yang, Nayeon Lee, Shaokun Zhang, Fuxiao Liu, Zhiqi Li, Di Zhang, Greg Heinrich, Hongxu Yin, Song Han, Pavlo Molchanov, Parth Mannan, Yao Xu, Jane Polak Scowcroft, Tom Balough, Subhashree Radhakrishnan, Paris Zhang, Sean Cha, Ratnesh Kumar, Zaid Pervaiz Bhat, Jian Zhang, Darragh Hanley, Pritam Biswas, Jesse Oliver, Kevin Vasques, Roger Waleffe, Duncan Riach, Oluwatobi Olabiyi, Ameya Sunil Mahabaleshwarkar, Bilal Kartal, Pritam Gundecha, Khanh Nguyen, Alexandre Milesi, Eugene Khvedchenia, Ran Zilberstein, Ofri Masad, Natan Bagrov, Nave Assaf, Tomer Asida, Daniel Afrimi, Amit Zuker, Netanel Haber, Zhiyu Cheng, Jingyu Xin, Di Wu, Nik Spirin, Maryam Moosaei, Roman Ageev, Vanshil Atul Shah, Yuting Wu, Daniel Korzekwa, Unnikrishnan Kizhakkemadam Sreekumar, Wanli Jiang, Padmavathy Subramanian, Alejandra Rico, Sandip Bhaskar, Saeid Motiian, Kedi Wu, Annie Surla, Chia-Chih Chen, Hayden Wolff, Matthew Feinberg, Melissa Corpuz, Marek Wawrzos, Eileen Long, Aastha Jhunjhunwala, Paul Hendricks, Farzan Memarian, Benika Hall, Xin-Yu Wang, David Mosallanezhad, Soumye Singhal, Luis Vega, Katherine Cheung, Krzysztof Pawelec, Michael Evans, Katherine Luna, Jie Lou, Erick Galinkin, Akshay Hazare, Kaustubh Purandare, Ann Guan, Anna Warno, Chen Cui, Yoshi Suhara, Shibani Likhite, Seph Mard, Meredith Price, Laya Sleiman, Saori Kaji, Udi Karpas, Kari Briski, Joey Conway, Michael Lightstone, Jan Kautz, Mohammad Shoeybi, Mostofa Patwary, Jonathen Cohen, Oleksii Kuchaiev, Andrew Tao, and Bryan Catanzaro. Nvidia nemotron nano v2 vl, 2025.
- [16] Mike Ranzinger, Jon Barker, Greg Heinrich, Pavlo Molchanov, Bryan Catanzaro, and Andrew Tao. Phis: Distribution balancing for label-free multi-teacher distillation, 2024.
- [17] Mike Ranzinger, Greg Heinrich, Jan Kautz, and Pavlo Molchanov. Am-radio: Agglomerative vision foundation model reduce all domains into one. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12490–12500, 2024.
- [18] Mike Ranzinger, Greg Heinrich, Pavlo Molchanov, Bryan Catanzaro, and Andrew Tao. Featsharp: Your vision model features, sharper. In Forty-second International Conference on Machine Learning, 2025.

- [19] Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, and Piotr Bojanowski. DINOv3, 2025.
- [20] Trung Trinh, Markus Heinonen, Luigi Acerbi, and Samuel Kaski. Improving robustness to corruptions with multiplicative weight perturbations. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.
- [21] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features, 2025.
- [22] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems, page 6000–6010, Red Hook, NY, USA, 2017. Curran Associates Inc.
- [23] Jiawei Yang, Katie Z Luo, Jiefeng Li, Congyue Deng, Leonidas J. Guibas, Dilip Krishnan, Kilian Q Weinberger, Yonglong Tian, and Yue Wang. Dvt: Denoising vision transformers. 2024.
- [24] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pretraining. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 11941–11952, 2023.
- [25] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. International Journal of Computer Vision, 127:302 – 321, 2016.

