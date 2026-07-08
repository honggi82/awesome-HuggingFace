# arXiv:2602.18996v2[cs.CV]25Mar2026

## Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction

Shannan Yan1,2 Leqi Zheng1 Keyu Lv1 Jingchen Ni1 Hongyang Wei1 Jiajun Zhang3 Guangting Wang2 Jing LYU2 Chun Yuan1† Fengyun Rao2†

1Tsinghua University 2WeChat Vision, Tencent Inc. 3USTC

ysn24@mails.tsinghua.edu.cn, yuanc@sz.tsinghua.edu.cn, fengyunrao@tencent.com

#### Abstract

We study the task of establishing object-level visual correspondence across different viewpoints in videos, focusing on the challenging egocentric-to-exocentric and exocentricto-egocentric scenarios. We propose a simple yet effective framework based on conditional binary segmentation, where an object query mask is encoded into a latent representation to guide the localization of the corresponding object in a target video. To encourage robust, view-invariant representations, we introduce a cycle-consistency training objective: the predicted mask in the target view is projected back to the source view to reconstruct the original query mask. This bidirectional constraint provides a strong self-supervisory signal without requiring ground-truth annotations and enables test-time training (TTT) at inference. Experiments on the Ego-Exo4D and HANDAL-X benchmarks demonstrate the effectiveness of our optimization objective and TTT strategy, achieving state-of-the-art performance. The code is available at https://github. com/shannany0606/CCMP.

#### 1. Introduction

Understanding visual correspondence across different viewpoints is a core capability for embodied agents operating in complex environments. In applications such as humanrobot interaction [1, 4], autonomous navigation [45, 57], and assistive robotics [20, 34], an agent must consistently identify and reason about the same object or scene across drastically different visual perspectives. For instance, a service robot may need to interpret instructions from an egocentric wearable camera and then locate the referred object from its own third-person viewpoint. Achieving such crossview correspondence is crucial for grounding language, enabling coordination, and executing goal-directed actions.

†Corresponding author. This work was done when Shannan Yan was an intern at Tencent Inc.

[Figure 1]

Figure 1. Cycle-Consistent Visual Correspondence with TestTime Training. Our framework learns object-level correspondences by enforcing cycle consistency: the object mask is transferred from source to target view and projected back to reconstruct the original query. This self-supervised constraint enables robust cross-view alignment and supports test-time training to further improve performance during inference.

However, the substantial viewpoint disparity between egocentric (first-person) and exocentric (third-person) cameras introduces major challenges such as appearance variation, occlusions, and disjoint spatial references, which traditional correspondence models, often trained on co-visible or static scenes, fail to handle effectively.

The task is challenging due to several compounding factors. First, the visual appearance of objects can vary drastically across views because of changes in camera angle, lighting, occlusion, and resolution. Egocentric views are often shaky, cluttered, and subject to motion blur, while exocentric views are more stable but may lack fine-grained detail. Second, the spatial layout and context around an object can differ significantly across viewpoints, making it difficult to rely on background cues for matching. Third, establishing correspondence requires reasoning not just over spatial features but also over temporal dynamics, as objects may move or deform differently across camera views. These challenges make traditional appearance-based or trackingbased methods insufficient and call for models that can learn robust, view-invariant object representations while being sensitive to fine-grained semantic and temporal cues.

To address the challenge of establishing cross-view visual correspondence, we propose a simple yet effective framework based on conditional binary segmentation. Our

approach leverages the powerful vision foundation model DINOv3 [33] as backbone, and introduces a single conditioning token (CDT) to inject source image information into the vision transformer. The modified visual tokens are then used to predict a binary mask in the target view. This compact design enables cross-view alignment with minimal architectural changes and maintains full compatibility with pretrained backbones.

To enhance training supervision, we introduce a cycleconsistency objective that enforces consistency between the original query mask and its round-trip projection through the target view. As illustrated in Figure 1, the predicted mask originates in the source view, is generated in the target view, and is expected to reconstruct the initial query mask. This bidirectional constraint encourages the model to learn view-invariant representations and provides selfsupervision in the absence of target-view annotations. Crucially, it also enables test-time training (TTT) at inference time, further improving correspondence quality under domain shift or distributional variation.

We validate our approach on the challenging EgoExo4D [15] dataset, which contains diverse egocentric and exocentric video pairs with rich object-level annotations. In the Exo Query setup, our method outperforms all prior baselines by over 3.10%, demonstrating strong cross-view correspondence capabilities. In the Ego Query setting, we achieve an IoU of 41.95%, closely approaching the previous state-of-the-art (SOTA) method O-MaMa [30], which scores 42.57%. We further evaluate on HANDALX [12], where our method outperforms ObjectRelator by 36.0% on zero-shot segmentation. Incorporating the cycleconsistency constraint during training and leveraging TTT at inference consistently yield improvements across both setups. Despite its conceptual simplicity, our framework effectively captures fine-grained correspondences under extreme viewpoint changes.

In summary, our contributions are:

- • We propose a simple, modular, and end-to-end framework for cross-view visual object correspondence that leverages vision foundation models with minimal architectural modifications.
- • We introduce a novel cycle-consistency objective that enforces semantic alignment between source and target views. This self-supervisory signal enables effective test-time training during inference.
- • We validate our approach on the Ego-Exo4D and HANDAL-X benchmarks, achieving SOTA performance and demonstrating its strong effectiveness.

#### 2. Related Works

Cross-view Video Understanding. Bridging ego- and exo-centric perspectives can enrich video understanding, yet most work targets one view. Egocentric research spans

classification [25, 48], question answering [2, 54], and captioning [22, 51], but lagged behind exocentric methods due to scarce data. The advent of large-scale benchmarks like Ego4D [14], EPIC-KITCHENS-100 [10], and richer modeling [18, 24] has closed this gap. However, few works connect both views [23, 27], until Ego-Exo4D’s time-aligned annotations [15]. Recently, Baade et al. [3] synthesize paired masks from raw segmentations using predictive cycle consistency and iterative pseudo-labeling. ObjectRelator [12] fine-tunes PSALM [58] with auxiliary modules that enforce view-invariant embeddings through self-supervised alignment. O-MaMa [30] reformulates cross-view segmentation as a mask-matching problem by integrating FastSAM [59] to generate candidate masks in advance. In this paper, we present an end-to-end baseline that requires no extra data and does not rely on auxiliary modules.

Vision Foundation Models. With the rapid advancement of deep learning and large language models [19, 28, 29, 31, 32, 39, 47, 49, 50, 52, 56, 60–62], recent progress in vision foundation models has significantly advanced representation learning by leveraging transformer architectures and self-supervised techniques. Vision Transformer (ViT) [11] introduced a transformer-based approach for image recognition, followed by DeiT [42] and DeiT3 [43], which improved training efficiency and performance on limited data. MoCo v3 [7] extended contrastive learning to ViTs, while MAE [17] proposed masked image reconstruction as a pretext task. DINO [6], DINOv2 [33] and DINOv3 [38] employed self-distillation to learn rich semantic features. CLIP [36] aligned vision and language via contrastive learning on image-text pairs, inspiring extensions like SigLIP [53] and SigLIP2 [44], which replaced contrastive loss with sigmoid-based objectives for better crossmodal alignment. These models form the foundation of modern vision systems, demonstrating strong generalization and scalability across diverse tasks. However, none are explicitly designed for ego-exo correspondence, which remains particularly challenging.

Test-Time Training. Test-time training (TTT) has evolved from a self-supervised adaptation method for distribution shifts into a versatile framework across modalities, including images, videos, and language. Early work [40] proposed optimizing a model on each unlabeled test sample via a self-supervised task to enhance robustness under covariate shifts. Subsequent studies [9, 13, 16, 41, 46] expanded TTT to masked autoencoding, streaming video adaptation, retrieval-augmented language modeling, and generative video Transformers, demonstrating its broad applicability and growing impact. To the best of our knowledge, our method is the first to successfully apply TTT to this task and achieve a clear performance improvement.

#### 3. Approach

In this section, we present our approach to visual correspondence. Given a source image Is, a target image It, and an object mask Ms in Is, the objective is to accurately segment the corresponding object Mt in the target image. To ensure clarity and consistency, we define Ego2Exo as the task where the ego-centric view (circular field of view) serves as the query and the exo-centric view as the target, and Exo2Ego as the reverse setting.

The section is organized as follows: Section 3.1 introduces our simple yet effective pipeline, Section 3.2.1 details the objective function, and the implementation details are provided in Section 3.3.

##### 3.1. Pipeline Overview

We propose a transformer-based framework that leverages conditional features and a mask-guided attention mechanism to establish robust visual correspondences.

The overall pipeline is illustrated in Figure 2. Our model comprises three main components: Source Feature Extractor, Transformer Encoder, and Multi-task Decoder.

Source Feature Extractor. The objective of this module is to extract an object-specific feature representation from a source image Is using its corresponding mask Ms. First, we obtain the feature map Fs ∈ RC×H×W from Is using a backbone network Fsfe(·): Fs = Fsfe(Is).

The mask Ms is resized (if necessary) to match the spatial dimensions of Fs and is then normalized such that its elements sum to one. Specifically, we compute the normalized mask M˜s as: M˜s = M

i,j Ms[i,j]+τ , where τ is a small hyperparameter (typically 1×10−6) used to prevent numerical errors. This yields a compact, stable, and scale-invariant object-focused representation.

s

Next, we compute the masked object feature zs ∈ RC as a weighted average of the feature map Fs over the spatial locations, using M˜s as the weights:zs =

H i=1

W j=1

M˜s[i,j] · Fs[:,i,j].

This produces a compact representation that highlights the regions specified by the mask, which is then projected onto the condition token CDT in the transformer encoder.

Transformer Encoder. The target image It is divided into patches, which are then linearly projected to form n visual tokens [x1,x2,...,xn] for the transformer encoder. Along with the condition token CDT and the class token CLS, the final input to the transformer encoder is

###### xinput = [CLS,CDT,x1,x2,...,xn].

These tokens are fed into a standard transformer encoder, enabling the CDT to condition the transformer fea-

tures through cross-token attention, which facilitates objectaware representation learning in the target image.

Multi-task Decoder. The output tokens from the transformer encoder are processed by two parallel heads: the Mask Head, which generates the feature for each visual token yi; and the CLS Head, which, with an additional classification token CLS, predicts whether the object in Is is visible in It (i.e., performs binary visibility classification).

The final segmentation mask Mˆt is generated employing a lightweight prediction head consisting of two convolutional layers applied solely on the visual tokens.

The overall design allows the model to jointly reason about spatial alignment, object visibility, and semantic consistency, resulting in improved correspondence performance under complex appearance and viewpoint variations.

##### 3.2. Learning Visual Correspondences

###### 3.2.1. Objective Function

We employ multiple objective functions to effectively learn visual correspondences: the mask loss Lmask, the auxiliary loss Laux, and the cycle-consistency loss Lcycle. The total training objective is defined as

Ltotal = Lmask + λauxLaux + λcycleLcycle, (1)

where λaux and λcycle are hyperparameters that balance the contributions of the auxiliary and cycle-consistency losses, respectively.

Mask Loss. Since our task is binary segmentation, we adopt a combination of Binary Cross-Entropy (BCE) loss and Dice loss to supervise the predicted target mask Mˆt, which is similar to [15]. Precisely, our mask loss is composed of Lbce as well as Ldice.

Lmask(Mt,Mˆt) = Lbce(Mt,Mˆt) + λdiceLdice(Mt,Mˆt),

(2) where Lbce(Mt,Mˆt) and Ldice(Mt,Mˆt) are defined as

Lbce(Mt,Mˆt) = −

1 N i

Mti log(Mˆti) + (1 − Mti)log(1 − Mˆti) ,

2 i MtiMˆti + ϵ i Mti + i Mˆti + ϵ

Ldice(Mt,Mˆt) = 1 −

,

(3) where ϵ is a small constant (e.g., 10−6 to prevent division by zero).

[Figure 2]

- Figure 2. Model overview. CLS denotes class tokens, and CDT denotes condition tokens. The CLS head determines whether the object in the target image corresponding to a given object mask in the source image is visible. The bottom-left image shows the source image with the object mask, while the bottom-right image shows the target image.

Auxiliary Loss. To facilitate training and improve gradient flow, we introduce an auxiliary supervision signal by applying the same mask loss to intermediate predictions from the last few transformer encoder layers. Specifically, the auxiliary loss is computed between the ground-truth target mask Mt and the intermediate predicted masks, using the same formulation as in Eq. 2. The total auxiliary loss is averaged across selected layers. This deep supervision encourages the model to learn meaningful representations at different levels of the network.

Cycle-consistency Loss. To improve the robustness of learned visual correspondences, we introduce a cycleconsistency constraint. The key idea is to map the source mask Ms to the predicted target mask Mˆt, and then map it back to the reconstructed source mask Mˆs, ensuring that Mˆs closely matches the original Ms. Formally, the cycleconsistency loss is defined as:

Lcycle = Lbce(Ms,Mˆs), (4)

where Lbce is the part of binary cross entropy loss used in Eq. 2. This objective encourages the model to learn more consistent and reliable correspondences by enforcing a closed-loop mapping between source and target domains.

Importantly, the cycle-consistency loss Lcycle does not rely on ground-truth target masks, making it applicable during inference for test-time training (TTT). We do not explic-

itly handle the corner case of invisible objects in the cycle, as such instances are rare in Ego-Exo4D.

###### 3.2.2. Visibility Prediction and Test-Time Training

Visibility Prediction. Since visibility prediction is inherently an instance-level task that determines whether an object is visible as a whole rather than a pixel-level segmentation task, we treat it separately from mask prediction. To this end, we introduce a lightweight post-training step specifically for visibility classification. After training the main model, we freeze the entire network and fine-tune only the CLS Head, which serves as a binary classifier applied to the CLS token. This design allows us to leverage the instance-level semantic representation encoded in the CLS token without altering the learned correspondence or segmentation capabilities of the backbone.

Test-Time Training. Since the cycle-consistency loss Lcycle does not require ground-truth target masks, it can be leveraged at inference time to further refine the model. Specifically, we apply test-time training (TTT) for image pairs. During TTT, we fine-tune only the last K transformer encoder layers of the model for each test pair, using T gradient update steps with a learning rate of lrttt. This allows the model to adapt to the specific test pair and improve correspondence quality.

- 3.3. Implementation Details

Architecture. We adopt the ConvNeXt-based pretrained DINOv3-L model [38] as the source feature extractor for its efficiency. For the transformer encoder, we employ the ViTbased pretrained DINOv3-L model, which provides rich visual representations. The compact feature representation obtained from the source image is linearly projected to form the CDT token, aligning the feature dimension with the transformer input.

Training Details. The training process on Ego-Exo4D consists of two stages. In the first stage (linear probing), we freeze the two DINOv3 backbones and train the remaining modules for 64K iterations. In the second stage, all parameters are unfrozen and optimized for 640K iterations. To address GPU memory limitations (40GB), we adopt gradient accumulation with a step size of 16, resulting in an effective number of parameter updates of 704K / 16 = 44K. The training process takes approximately 72 hours on 8 NVIDIA RTX A800 GPUs. For visibility prediction, we fine-tune only the CLS Head for 96K iterations, which takes approximately 1 hour on the same hardware. More training details are provided in the supplementary material.

Hyperparameter Settings. In the total loss formulation, we set the loss weights as follows: λdice = 5, λaux = 1, and λcycle = 10. The auxiliary loss is applied to the second-tolast layer of the transformer encoder. For TTT, we update the last K=4 layers for T=2 steps in Ego2Exo and the last K=11 layers for T=6 steps in Exo2Ego, using a learning rate of 5 × 10−6.

- 4. Experiments

- 4.1. Experimental Setting

Datasets. To evaluate the effectiveness of our approach, we adopt the Ego-Exo4D correspondence dataset [15]. This dataset contains 1.8 million annotated object masks sampled at 1 fps from 1,335 video takes, covering a wide range of domains such as soccer, basketball, music, cooking, bike repair, and healthcare, all recorded in natural, unscripted environments. Due to privacy restrictions, 66 training takes (out of 821) were redacted from the released dataset, limiting our training data and making direct comparison with prior works less favorable. We thus use 755 takes for training, 201 for validation, and 295 for testing.

To further evaluate the generalization ability of different methods, we conduct experiments on HANDAL-X, a crossview object segmentation benchmark introduced by ObjectRelator [12]. HANDAL-X contains multi-view image pairs that capture objects from complete 360° viewpoints with corresponding object-centric masks. The dataset comprises 44,102 training pairs and 14,074 test pairs. Unlike the

egocentric–exocentric Ego-Exo4D, HANDAL-X provides a complementary setting for assessing cross-view correspondence under broader viewpoint variations.

Data Preprocessing. To enhance the diversity of training data, we adopt three complementary preprocessing strategies. First, we unify the Ego2Exo and Exo2Ego tasks into a single cross-view mapping framework, enabling the model to jointly learn from both directional data and perform bidirectional correspondence within a unified formulation. This design also enables the application of Lcycle. Second, we introduce same-view exemplar synthesis by asynchronously sampling Ego2Ego and Exo2Exo pairs, which increases data diversity and strengthens intra-view consistency. Third, for cross-view pairs, we relax temporal alignment by pairing query frames with temporally offset target frames, improving the model’s resilience to timing discrepancies across views.

Evaluation Metrics. We adopt the following evaluation metrics following the Ego-Exo4D correspondence benchmark: 1) Visibility Accuracy (VA) [5]: Evaluates the model’s ability to predict object visibility in the target view, considering occlusions and out-of-frame cases. The CLS head is specifically designed to optimize this metric. 2) Intersection over Union (IoU): Measures the overlap between the predicted and ground-truth masks. 3) Location Error (LE): Defined as the normalized distance between the centroids of the predicted and ground-truth masks. 4) Contour Accuracy (CA) [35]: Evaluates mask shape similarity after aligning the centroids of the predicted and ground-truth masks.

According to the evaluation criteria of Ego-Exo4D Correspondence Challenge, the primary evaluation metric is the mean Intersection-over-Union of both Ego2Exo and Exo2Ego task settings. We refer to this metric as mIoU.

##### 4.2. Comparison to competitive approaches

Baselines. We evaluate against the following 7 opensource models: 1) XSegTx: A Transformer-based spatial model adapted from SegSwap [37] that independently estimates correspondences at each time step. 2) XViewXMem: A spatio-temporal model adapted from XMem [8] that leverages temporal context to generalize object tracking across views using ground-truth masks from one view per frame. 3) SEEM: A universal segmentation framework that interprets visual prompts in reference to the input image. 4) PSALM: A model extends Phi-1.5 1.3B model [21] with a mask decoder and a flexible prompting schema to perform diverse pixel-wise segmentation tasks within a single unified framework. 5) CMX: A transformer-based segmentation model that integrates information from two modalities. 6) ObjectRelator: A cross-view correspondence model that fuses visual and textual cues and aligns representations

Table 1. Evaluation results on the Ego-Exo4D correspondence benchmark v2 (test set). The best performance is highlighted in bold, and the second-best is underlined.

Ego Query Exo Query Method VA↑ IoU↑ LE↓ CA↑ VA↑ IoU↑ LE↓ CA↑ mIoU↑

XSegTx (random weights) [15] 50.00 0.48 0.118 0.014 50.00 1.08 0.203 0.024 0.78 XSegTx [15] 66.31 18.99 0.070 0.386 82.01 27.14 0.104 0.358 23.07 XMem [8, 15] 64.39 19.28 0.151 0.262 60.35 16.56 0.160 0.204 17.92 XView-XMem [15] 61.24 14.84 0.115 0.242 61.72 21.37 0.139 0.269 18.11 XView-XMem (+ XSegTx) [15] 66.79 34.90 0.038 0.559 59.71 25.00 0.117 0.327 29.95 SEEM [63] ∅ 1.53 0.258 0.041 ∅ 4.34 0.289 0.062 2.94 PSALM [58] ∅ 7.40 0.266 0.121 ∅ 2.10 0.294 0.058 4.75 CMX [55] ∅ 6.80 0.110 0.137 ∅ 12.00 0.166 0.177 9.40 ObjectRelator [12] 95.95 35.27 0.036 0.540 97.36 40.31 0.068 0.500 37.79 O-MaMa [30] 50.00 42.57 0.033 0.590 50.05 44.08 0.082 0.524 43.32 Ours 98.92 41.95 0.038 0.669 99.86 47.18 0.081 0.591 44.57

between egocentric and exocentric views with specialized cross-view modules. 7) O-MaMa: An object mask matching approach that selects the best mask candidate from FastSAM [59] proposals between images.

Results on Ego-Exo4D. The quantitative results on the Ego-Exo4D correspondence benchmark are summarized in Table 1. Despite limited training data, it achieves an mIoU of 44.57%, representing a +2.9% relative improvement over the previous SOTA method, O-MaMa. Our method further surpasses all prior baselines in the Exo Query setup by a +7.0% relative improvement in IoU and attains competitive performance in the Ego Query setup. The improvement is also consistent across other metrics, with a relative gain of up to +13.4% in CA under the Ego Query setting. Overall, these results highlight the strong effectiveness and robustness of our approach.

We observe that most methods perform worse under the Ego Query setting. This observation is consistent with intuition, as exo-view objects are generally smaller and appear within more cluttered environments, making segmentation more challenging. The judgement is quantitatively validated in Section D and Figure 4 (b). Meanwhile, although SEEM and PSALM demonstrate strong performance on other out-of-domain tasks, they struggle to generalize effectively in this setting. This suggests that training on crossview datasets is crucial for achieving robust performance, as knowledge learned from other domains does not readily transfer to cross-view correspondence.

Results on HANDAL-X. Since ObjectRelator additionally reports results on HANDAL-X, we also include evaluations on this dataset without test-time training. As shown in Table 2, our approach exhibits superior cross-view generalization, achieving an IoU of 78.8% on HANDAL-X

Table 2. Evaluation results on the HANDAL-X benchmark.

###### Method Fine-tuning Datasets IoU↑

XSegTx [15] ∅ 1.5 SEEM [63] ∅ 2.5 PSALM [58] ∅ 14.2 PSALM [58] Ego-Exo4D 39.9 ObjectRelator [12] Ego-Exo4D 42.8 Ours Ego-Exo4D 78.8

PSALM [58] Ego-Exo4D, HANDAL-X 83.4 ObjectRelator [12] Ego-Exo4D, HANDAL-X 84.7 Ours Ego-Exo4D, HANDAL-X 85.0

without any training on this dataset, surpassing all baselines by a relative improvement of 84.1%. After fine-tuning on HANDAL-X, our method continues to consistently outperform all competing approaches. These results can be attributed to our compact design, along with the effective training data augmentation strategy.

Notably, all methods exhibit performance gains on HANDAL-X after being trained on Ego-Exo4D, indicating that exposure to egocentric–exocentric data effectively enhances generalization to unseen cross-view configurations, even under distinct scenes and viewpoints.

##### 4.3. Ablation Study

Loss Components. As described in Section 3.2.1, we enhance the baseline loss Lmask with a cycle consistency loss Lcycle and an auxiliary loss Laux. As shown in Table 3, removing either component causes a performance degradation, confirming their effectiveness. Test-time tuning (TTT) further brings consistent improvements. Notably, removing Lcycle leads to a noticeable drop in mIoU, highlighting its importance in providing the self-supervised signal es-

###### Table 3. Ablation on optimization components.

###### Configuration Lcycle Laux TTT Ego-IoU↑ Exo-IoU↑ mIoU↑

Ours (full) ✓ ✓ ✓ 41.95 47.18 44.57 w/o Lcycle ✗ ✓ ✓ 40.28 45.82 43.05 w/o Laux ✓ ✗ ✓ 40.64 43.81 42.90 w/o TTT ✓ ✓ ✗ 41.79 44.18 42.99

[Figure 3]

- Figure 3. Visualization illustrating the contribution of test-time training.

Table 4. Ablation on data augmentation. ”RTA” refers to the relaxed temporal alignment strategy used for both tasks.

Training data Ego-IoU↑ Exo-IoU↑ mIoU↑

Ours 41.95 47.18 44.57 w/o Ego2Ego & Exo2Exo 40.88 45.50 43.19 w/o RTA 40.60 45.45 43.03

Table 5. Disentangling feature quality from method design. CBS denotes the proposed conditional binary segmentation.

Framework Backbone CBS Lcycle TTT Ego-IoU↑ Exo-IoU↑ mIoU↑

XSegTx DINOv3 ✗ ✗ ✗ 26.52 34.36 30.44 Ours DINOv3 ✓ ✓ ✓ 41.95 47.18 44.57 Ours DINOv2 ✓ ✓ ✓ 41.48 44.50 42.99

ing results are reported in Table 5. Notably, even with slightly weaker DINOv2 features, our method still outperforms “baseline + DINOv3,” indicating that the improvements stem primarily from our architectural components rather than the choice of pre-trained features.

Dice Loss in Lcycle and TTT. It is intuitive to incorporate Dice loss into the self-supervised learning objective, i.e., setting Lcycle = Lbce(Ms,Mˆs) + λ′diceLdice(Ms,Mˆs)(λ′dice = 5). To maintain a balanced optimization, we additionally explore a configuration with λcycle = 1. As reported in Table 6, the results indicate that introducing Dice loss into the optimization objective hinders the model’s ability to learn effectively during TTT.

Table 6. Ablation on dice loss in Lcycle and TTT.

λ′dice λcycle Ego-IoU↑ Exo-IoU↑ mIoU↑ 0 10 41.95 47.18 44.57 5 10 38.25 42.14 40.20 5 1 40.45 44.57 42.51

sential for effective TTT. As illustrated in Figure 3, after applying TTT, the model not only focuses more accurately on the true target object while suppressing visually similar distractors but also produces predicted masks that more completely cover the ground-truth regions.

Data Augmentation. To quantify the individual contributions of each data preprocessing strategy, we conduct a comprehensive ablation study. Since ObjectRelator has already validated the effectiveness of jointly learning from both Ego2Exo and Exo2Ego tasks, we omit this experiment. As shown in Table 4, removing any single strategy consistently degrades both per-view IoU and overall mIoU, confirming that each component contributes to more robust and generalizable segmentation.

Contribution beyond DINOv3 features. To isolate and highlight the contribution of our framework beyond the use of DINOv3 features, we adopt XSegTx [15] from EgoExo4D as a baseline and replace its original backbone with DINOv3 for controlled comparison. The correspond-

Linear Probing Stage. We conduct an ablation on the first training stage, as shown in Table 7. The results demonstrate that this stage is essential for achieving accurate object segmentation, as it stabilizes cross-view feature alignment and provides a strong initialization for subsequent end-to-end fine-tuning.

Table 7. Ablation on the linear probing stage.

Setting Ego-IoU↑ Exo-IoU↑ mIoU↑ Ours 41.95 47.18 44.57

w/o Linear Probing Stage 37.97 43.69 40.83

Results across Different Scenarios. The test set of the Ego-Exo4D Correspondence Benchmark includes 6 distinct scenarios. As shown in Figure 4 (a), cooking, health, and bike repair present greater challenges due to smaller object sizes and more complex environments. Nevertheless, our method consistently achieves an IoU exceeding 40% across all scenarios, highlighting its robustness in handling diverse objects and environments.

[Figure 4]

- Figure 4. (a) Performance per activity scenario; (b) Performance across different object sizes in the target view.

Results across Different Target Object Sizes. We evaluate our method across different object sizes in the target view, as shown in Figure 4 (b). Our method performs well on objects occupying more than 0.1% of the target-view image area, while smaller objects remain challenging.

It is worth noting that although the IoU of the Ego2Exo task is consistently higher than that of the Exo2Ego task within each object-size bin, the overall IoU of Ego2Exo (41.95%) remains lower than that of Exo2Ego (47.18%). This discrepancy suggests that the two tasks differ in their underlying object-size distributions. Ego2Exo likely contains a larger proportion of smaller target objects, which causes most methods including ours to yield lower IoU and also limits the effectiveness of TTT, as reflected in Table 1 and Table 3.

##### 4.4. Qualitative Results

To better demonstrate the effectiveness of our pipeline, we present representative qualitative results on Ego-Exo4D in Figure 5, covering both Ego2Exo and Exo2Ego scenarios, and on HANDAL-X in Figure 6. These visual examples highlight the challenging nature of the task, with large variations in scale, perspective, and object appearance due to deformation or occlusion. Despite these difficulties, our method consistently and accurately segments the queried object across diverse activities and viewpoints. The results demonstrate not only the robustness of our approach in handling extreme viewpoint shifts and motion but also support the effectiveness of using a cycle-consistent loss, which encourages reliable cross-view alignment by enforcing mutual consistency between Ego and Exo predictions. More visual examples and failure cases are provided in the supplementary material.

[Figure 5]

- Figure 5. Qualitative results on the Ego-Exo4D correspondence benchmark. Each row corresponds to one sample. From top to bottom, the first and second rows show samples of Ego2Exo, while the third and fourth rows show samples of Exo2Ego.

[Figure 6]

- Figure 6. Qualitative results on the HANDAL-X benchmark. Each column corresponds to one sample.

#### 5. Conclusion

We present a simple yet effective approach for object correspondence between egocentric and exocentric views. By unifying cross-view tasks, enriching training with sameview and temporally relaxed pairs, and leveraging a cycleconsistent loss with test-time training, our method achieves strong performance on the Ego-Exo4D and HANDAL-X benchmarks. Extensive experiments demonstrate its robustness to viewpoint shift, occlusion, and motion. Our findings suggest that careful training design and self-supervision can enhance correspondence without requiring complex pipelines or large temporal context. We hope our work offers meaningful insights and a solid foundation that facilitates future research on this underexplored task.

#### Acknowledgment

This work was supported by the National Key R&D Program of China (2022YFB4701400/4701402), SSTIC Grant(KJZD20230923115106012, KJZD20230923114916032, GJHZ20240218113604008).

#### References

- [1] Ashraf Alam. Social robots in education for long-term human-robot interaction: socially supportive behaviour of robotic tutor for creating robo-tangible learning environment in a guided discovery learning interaction. ECS Transactions, 2022. 1
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. NeurIPS, 2022. 2
- [3] Alan Baade and Changan Chen. Self-supervised cross-view correspondence with predictive cycle consistency. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16753–16763, 2025. 2
- [4] Christoph Bartneck, Tony Belpaeme, Friederike Eyssel, Takayuki Kanda, Merel Keijsers, and Selma Sabanovi´ˇ c. Human-robot interaction: An introduction. Cambridge University Press, 2024. 1
- [5] Kay Henning Brodersen, Cheng Soon Ong, Klaas Enno Stephan, and Joachim M Buhmann. The balanced accuracy and its posterior distribution. In 2010 20th international conference on pattern recognition, 2010. 5
- [6] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 2
- [7] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In ICCV, 2021. 2
- [8] Ho Kei Cheng and Alexander G Schwing. Xmem: Longterm video object segmentation with an atkinson-shiffrin memory model. In ECCV, 2022. 5, 6
- [9] Karan Dalal, Daniel Koceja, Jiarui Xu, Yue Zhao, Shihao Han, Ka Chun Cheung, Jan Kautz, Yejin Choi, Yu Sun, and Xiaolong Wang. One-minute video generation with test-time training. In Proceedings of the Computer Vision and Pattern Recognition Conference, 2025. 2
- [10] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Evangelos Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. Rescaling egocentric vision: Collection, pipeline and challenges for epic-kitchens-100. IJCV, 2022. 2
- [11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 2

- [12] Yuqian Fu, Runze Wang, Yanwei Fu, Danda Pani Paudel, Xuanjing Huang, and Luc Van Gool. Objectrelator: Enabling cross-view object relation understanding in ego-centric and exo-centric videos. ICCV, 2025. 2, 5, 6, 1
- [13] Yossi Gandelsman, Yu Sun, Xinlei Chen, and Alexei Efros. Test-time training with masked autoencoders. NeurIPS,

2022. 2

- [14] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In CVPR, 2022. 2
- [15] Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, Siddhant Bansal, Bikram Boote, et al. Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives. In CVPR, 2024. 2, 3, 5, 6, 7, 1
- [16] Moritz Hardt and Yu Sun. Test-time training on nearest neighbors for large language models. In ICLR, 2024. 2
- [17] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. 2
- [18] Baoxiong Jia, Ting Lei, Song-Chun Zhu, and Siyuan Huang. Egotaskqa: Understanding human tasks in egocentric videos. NeurIPS, 2022. 2
- [19] Zhaolu Kang, Junhao Gong, Jiaxu Yan, Wanke Xia, Yian Wang, Ziwen Wang, Huaxuan Ding, Zhuo Cheng, Wenhao Cao, Zhiyuan Feng, et al. Hssbench: Benchmarking humanities and social sciences ability for multimodal large language models. arXiv preprint arXiv:2506.03922, 2025. 2
- [20] Maria Kyrarini, Fotios Lygerakis, Akilesh Rajavenkatanarayanan, Christos Sevastopoulos, Harish Ram Nambiappan, Kodur Krishna Chaitanya, Ashwin Ramesh Babu, Joanne Mathew, and Fillia Makedon. A survey of robots in healthcare. Technologies, 2021. 1
- [21] Yuanzhi Li, S´ebastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463, 2023. 5
- [22] Kevin Lin, Linjie Li, Chung-Ching Lin, Faisal Ahmed, Zhe Gan, Zicheng Liu, Yumao Lu, and Lijuan Wang. Swinbert: End-to-end transformers with sparse attention for video captioning. In CVPR, 2022. 2
- [23] Jia-Wei Liu, Weijia Mao, Zhongcong Xu, Jussi Keppo, and Mike Zheng Shou. Exocentric-to-egocentric video generation. NeurIPS, 2024. 2
- [24] Shaowei Liu, Subarna Tripathi, Somdeb Majumdar, and Xiaolong Wang. Joint hand motion and interaction hotspots prediction from egocentric videos. In CVPR, 2022. 2
- [25] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In CVPR,

2022. 2

- [26] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2017. 1

- [27] Mi Luo, Zihui Xue, Alex Dimakis, and Kristen Grauman. Put myself in your shoes: Lifting the egocentric perspective from exocentric videos. In European Conference on Computer Vision. Springer, 2024. 2
- [28] Tianci Luo, Jinpeng Wang, Shiyu Qin, Niu Lian, Yan Feng, Bin Chen, Chun Yuan, and Shu-Tao Xia. Prompthub: Enhancing multi-prompt visual in-context learning with locality-aware fusion, concentration and alignment. In The Fourteenth International Conference on Learning Representations, 2026. 2
- [29] Keyu Lv, Manyi Zhang, Xiaobo Xia, Jingchen Ni, Shannan Yan, Xianzhi Yu, Lu Hou, Chun Yuan, and Haoli Bai. What makes low-bit quantization-aware training work for reasoning llms? a systematic study. arXiv preprint arXiv:2601.14888, 2026. 2
- [30] Lorenzo Mur-Labadia, Maria Santos-Villafranca, Jesus Bermudez-Cameo, Alejandro Perez-Yus, Ruben MartinezCantin, and Jose J Guerrero. O-mama: Learning object mask matching between egocentric and exocentric views. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025. 2, 6
- [31] Jingchen Ni, Keyu Lyu, Yu Guo, and Chun Yuan. Semantic alignment and hard sample retraining for visible-infrared person re-identification. In 2025 IEEE International Conference on Multimedia and Expo (ICME). IEEE, 2025. 2
- [32] Jingchen Ni, Quan Zhang, Dan Jiang, Keyu Lv, Ke Zhang, and Chun Yuan. Fcl-cod: Weakly supervised camouflaged object detection with frequency-aware and contrastive learning. arXiv preprint arXiv:2603.22969, 2026. 2
- [33] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 2
- [34] Yaokun Pang, Xianchen Xu, Shoue Chen, Yuhui Fang, Xiaodong Shi, Yiming Deng, Zhong-Lin Wang, and Changyong Cao. Skin-inspired textile-based tactile sensors enable multifunctional sensing of wearables and soft robots. Nano Energy, 2022. 1
- [35] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In CVPR, 2016. 5
- [36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. 2021. 2
- [37] Xi Shen, Alexei A Efros, Armand Joulin, and Mathieu Aubry. Learning co-segmentation by segment swapping for retrieval and discovery. In CVPRW, 2022. 5
- [38] Oriane Sim´eoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025. 2, 5, 1
- [39] Zihan Su, Hongyang Wei, Kangrui Cen, Yong Wang, Guanhua Chen, Chun Yuan, and Xiangxiang Chu. Genera-

- tion enhances understanding in unified multimodal models via multi-representation generation. arXiv preprint arXiv:2601.21406, 2026. 2
- [40] Yu Sun, Xiaolong Wang, Liu Zhuang, John Miller, Moritz Hardt, and Alexei A. Efros. Test-time training with selfsupervision for generalization under distribution shifts. In ICML, 2020. 2
- [41] Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, et al. Learning to (learn at test time): Rnns with expressive hidden states. arXiv preprint arXiv:2407.04620, 2024. 2
- [42] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herv´e J´egou. Training data-efficient image transformers & distillation through attention. 2021. 2
- [43] Hugo Touvron, Matthieu Cord, and Herv´e J´egou. Deit iii: Revenge of the vit. In ECCV, 2022. 2
- [44] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025. 2
- [45] Erdem Turan, Stefano Speretta, and Eberhard Gill. Autonomous navigation for deep space small satellites: Scientific and technological advances. Acta Astronautica, 2022. 1
- [46] Renhao Wang, Yu Sun, Arnuv Tandon, Yossi Gandelsman, Xinlei Chen, Alexei A Efros, and Xiaolong Wang. Test-time training on video streams. Journal of Machine Learning Research, 2025. 2
- [47] Yuji Wang, Jingchen Ni, Yong Liu, Chun Yuan, and Yansong Tang. Iterprime: Zero-shot referring image segmentation with iterative grad-cam refinement and primary word emphasis. In Proceedings of the AAAI Conference on Artificial Intelligence, 2025. 2
- [48] Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, and Saining Xie. Convnext v2: Co-designing and scaling convnets with masked autoencoders. In CVPR, 2023. 2
- [49] Shannan Yan. Redualsvg: Refined scalable vector graphics generation. In International Conference on Artificial Neural Networks. Springer, 2023. 2
- [50] Shannan Yan, Jingchen Ni, Leqi Zheng, Jiajun Zhang, Peixi Wu, Dacheng Yin, Jing Lyu, Chun Yuan, and Fengyun Rao. Adamem: Adaptive user-centric memory for long-horizon dialogue agents. arXiv preprint arXiv:2603.16496, 2026. 2
- [51] Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, Antoine Miech, Jordi Pont-Tuset, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Vid2seq: Large-scale pretraining of a visual language model for dense video captioning. In CVPR,

2023. 2

- [52] Hao Yu, Tangyu Jiang, Shuning Jia, Shannan Yan, Shunning Liu, Haolong Qian, Guanghao Li, Shuting Dong, and Chun Yuan. Comrope: Scalable and robust rotary position embedding parameterized by trainable commuting angle matrices.

- In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 4508–4517, 2025. 2
- [53] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, 2023. 2, 1
- [54] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 2023. 2
- [55] Jiaming Zhang, Huayao Liu, Kailun Yang, Xinxin Hu, Ruiping Liu, and Rainer Stiefelhagen. Cmx: Cross-modal fusion for rgb-x semantic segmentation with transformers. IEEE Transactions on intelligent transportation systems, 2023. 6
- [56] Jiajun Zhang, Zeyu Cui, Jiaxi Yang, Lei Zhang, Yuheng Jing, Zeyao Ma, Tianyi Bai, Zilei Wang, Qiang Liu, Liang Wang, et al. From completion to editing: Unlocking contextaware code infilling via search-and-replace instruction tuning. arXiv preprint arXiv:2601.13384, 2026. 2
- [57] Sitong Zhang, Yibing Li, and Qianhui Dong. Autonomous navigation of uav in multi-obstacle environments based on a deep reinforcement learning approach. Applied Soft Computing, 2022. 1
- [58] Zheng Zhang, Yeyao Ma, Enming Zhang, and Xiang Bai. Psalm: Pixelwise segmentation with large multi-modal model. In ECCV, 2024. 2, 6, 1
- [59] Xu Zhao, Wenchao Ding, Yongqi An, Yinglong Du, Tao Yu, Min Li, Ming Tang, and Jinqiao Wang. Fast segment anything. arXiv preprint arXiv:2306.12156, 2023. 2, 6
- [60] Leqi Zheng, Chaokun Wang, Canzhi Chen, Jiajun Zhang, Cheng Wu, Zixin Song, Shannan Yan, Ziyang Liu, and Hongwei Li. Lagcl4rec: When llms activate interactions potential in graph contrastive learning for recommendation. In The 2025 Conference on Empirical Methods in Natural Language Processing, 2025. 2
- [61] Leqi Zheng, Chaokun Wang, Zixin Song, Cheng Wu, Shannan Yan, Jiajun Zhang, and Ziyang Liu. Negative feedback really matters: Signed dual-channel graph contrastive learning framework for recommendation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.
- [62] Leqi Zheng, Jiajun Zhang, Canzhi Chen, Chaokun Wang, Hongwei Li, Yuying Li, Yaoxin Mao, Shannan Yan, Zixin Song, Zhiyuan Feng, et al. What should i cite? a rag benchmark for academic citation prediction. arXiv preprint arXiv:2601.14949, 2026. 2
- [63] Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Wang, Lijuan Wang, Jianfeng Gao, and Yong Jae Lee. Segment everything everywhere all at once. NeurIPS,

2023. 6, 1

## Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction

### Supplementary Material

- A. Organization This document contains the following sections:

- • More training details are provided in Section B.
- • Test-time training on HANDAL-X is provided in Section C.
- • More ablation study is provided in Section D.
- • Efficiency analysis is provided in Section E
- • More qualitative results are provided in Section F.
- • Limitation and future work are provided in Section G.

All blue-highlighted rows in the tables denote the default configurations of our method. We refer to test-time training as TTT throughout the paper.

- B. More Training Details

We train our model using the AdamW optimizer [26] with a cosine learning rate schedule and linear warm-up. We use a batch size of 16. The image size is 512 × 512. The training process on Ego-Exo4D [15] consists of two stages. In the first stage (linear probing), we freeze the two DINOv3 [38] backbones and train the remaining modules for 64K iterations. The learning rate decays from the maximum value of 1 × 10−3 to a minimum of 1 × 10−4. In the second stage, all parameters are unfrozen and optimized for 640K iterations. The learning rate decays from the maximum value of 1 × 10−5 to a minimum of 1 × 10−6. To address GPU memory limitations (40GB), we adopt gradient accumulation with a step size of 16, resulting in an effective number of parameter updates of 704K / 16 = 44K. The training process takes approximately 72 hours on 8 NVIDIA RTX A800 GPUs. We maintain an exponential moving average (EMA) of the model parameters throughout training, and use the EMA model as the final model for evaluation. For visibility prediction, we fine-tune only the CLS Head for 96K iterations, which takes approximately 1 hour on the same hardware, using the same training setup as the main binary segmentation task. In the TTT stage, the adaptation takes approximately 3 hours for Ego2Exo and 12 hours for Exo2Ego.

On HANDAL-X, we train for 10 epochs with the learning rate decaying from the maximum value of 2 × 10−4 to a minimum of 2 × 10−6. The offline training stage requires approximately 2 hours, and the TTT stage requires an additional 1 hour.

Table 8. Evaluation results on the HANDAL-X benchmark.

###### Method Fine-tuning Datasets IoU↑

XSegTx [15] ∅ 1.5 SEEM [63] ∅ 2.5 PSALM [58] ∅ 14.2 PSALM [58] Ego-Exo4D 39.9 ObjectRelator [12] Ego-Exo4D 42.8 Ours (w/o TTT) Ego-Exo4D 78.8 Ours Ego-Exo4D 80.6

PSALM [58] Ego-Exo4D, HANDAL-X 83.4 ObjectRelator [12] Ego-Exo4D, HANDAL-X 84.7 Ours (w/o TTT) Ego-Exo4D, HANDAL-X 85.0 Ours Ego-Exo4D, HANDAL-X 85.3

#### C. Test-time Training on HANDAL-X

Since our results on the HANDAL-X benchmark [12] without TTT already surpass all baselines, we omit the TTT results from the main text. Table 8 presents the quantitative performance with TTT on HANDAL-X, further demonstrating its effectiveness and generalization across benchmarks. We observe that when the baseline IoU is already very high, TTT yields only marginal improvements. The corresponding qualitative results are provided in Figure 10.

#### D. More Ablation Study

Mask Prediction Method. To enable the model to adaptively predict segmentation masks conditioned on given object features, we further explore an alternative implementation named Cosine Prediction for mask generation. The final segmentation mask Mˆt is predicted using both visual tokens and the updated condition token ycdt. Specifically, for the i-th visual token yi, the prediction is computed as:

Mˆti = Sigmoid(τ · Cos(ycdt,yi) − β), (5)

where Cos(·,·) denotes cosine similarity, and τ and β are learnable temperature and bias parameters as in [53]. They are initialized to 10 and 5, respectively.

Table 9 presents an ablation study comparing the proposed variant with our original method. The results demonstrate that direct mask prediction yields better performance than predicting masks conditioned on object features.

Table 9. Ablation on the mask prediction method.

###### Method Ego-IoU↑ Exo-IoU↑ mIoU↑

Ours 41.95 47.18 44.57 Cosine Prediction 40.29 46.75 43.52

Dice weight. We investigate the influence of the Dice Loss weight λdice in our mask supervision objective Lmask. The Dice Loss Ldice plays an essential role, particularly in scenarios where the target occupies only a small region of the spatial mask, as it effectively addresses class imbalance and encourages better alignment of predicted and groundtruth masks. Table 10 presents a detailed ablation of performance across different values of λdice. We find that setting λdice = 5 yields the best overall performance, outperforming all other configurations. Notably, when λdice = 0, which effectively removes the Dice Loss, the performance drops significantly across all metrics, confirming the importance of including Ldice. These results highlight the importance of balancing the Dice component within the mask loss.

Table 10. Ablation of dice weight.

###### λdice Ego-IoU↑ Exo-IoU↑ mIoU↑

- 0 28.22 32.76 30.49

0.5 41.11 46.24 43.68

- 1 41.84 47.17 44.51
- 2 41.33 46.70 44.02 5 41.95 47.18 44.57

10 41.54 46.72 44.13

Gradian Update Steps of TTT. It is notable that for TTT on Ego-Exo4D, we update for T=2 steps in Ego2Exo but T=6 steps in Exo2Ego. To justify this choice, we conduct an ablation study, and the results are presented in Table 11 and Table 12. We observe that Ego2Exo achieves its best performance with only 2 update steps, after which further updates cause slight degradation. In contrast, Exo2Ego continues to benefit from additional updates and reaches its peak performance at 7 or more steps. Considering the tradeoff between efficiency and performance, we adopt T=6 steps as our default setting. These findings highlight the importance of tuning the number of update steps for each direction individually, as the two tasks differ in their underlying object-size distributions and consequently in their adaptation behavior.

Table 11. Ablation of TTT steps (Ego2Exo).

###### Steps Ego-IoU↑

- 1 41.91

- 2 41.95

- 3 41.90
- 4 41.88
- 5 41.84

Table 13. Ablation of finetuning layers (Ego2Exo).

###### Layers Ego-IoU↑

- 3 41.90

- 4 41.95

- 5 41.94
- 6 41.93
- 7 41.87

Table 12. Ablation of TTT steps (Exo2Ego).

###### Steps Exo-IoU↑

- 3 46.81
- 4 46.98
- 5 47.09

- 6 47.18

- 7 47.23

Table 14. Ablation of finetuning layers (Exo2Ego).

###### Layers Exo-IoU↑

- 8 47.06
- 9 47.14
- 10 47.17

- 11 47.18

- 12 47.14

Fine-tuning Layes of TTT. It is notable that for TTT on Ego-Exo4D, we update the last K=4 layers in Ego2Exo but K=11 layers in Exo2Ego. To justify this choice, we conduct an ablation study, and the results are reported in Table 13 and Table 14. We observe that Ego2Exo achieves its best performance when only a small number of layers are adapted, while deeper adaptation yields diminishing returns or slight degradation. In contrast, Exo2Ego benefits from updating a substantially larger portion of the network, with performance peaking at 11 layers. These findings suggest that Ego2Exo requires only lightweight adjustments for effective adaptation, whereas Exo2Ego demands broader model capacity to accommodate the larger cross-view domain gap.

#### E. Effeciency Analysis

We agree that performance–latency trade-offs better reflect practical deployment than a single inference-time number. Accordingly, we provide an efficiency analysis in Figure 7, reporting mIoU as a function of inference time by varying the number of test-time optimization steps (from 0 to 1, 2, and beyond). As shown, most of the performance gain is achieved with only 2 gradient updates, while further updates bring diminishing returns. This indicates that our method can achieve improvement with limited additional latency.

#### F. More Qualitative Results

We provide additional qualitative results on the Ego-Exo4D correspondence benchmark in Figure 8 and Figure 9. We present diverse examples that cover all six scenarios: cook-

[Figure 7]

- Figure 7. Performance–latency trade-off under test-time training.

ing, health, bike repair, music, basketball, and soccer. The cooking scenario occupies three rows in each figure because it constitutes the largest portion of the benchmark. Across all scenarios, our method consistently produces masks that closely match the ground truth annotations, demonstrating strong robustness to variations in scene context, object category, occlusion, and viewpoint. These results also illustrate the effectiveness of TTT, which enables the model to focus more accurately on the target object while suppressing visually similar distractors and to generate masks that more completely cover the ground truth regions.

We further present qualitative results on HANDAL-X in Figure 10, illustrating six examples spanning diverse hand–object interaction categories and highlighting the effectiveness of TTT. Our method successfully recovers the target masks in most cases.

#### G. Limitation and Future Work

All qualitative results are presented without excluding failure cases, providing a comprehensive view of the model’s potential errors. We summarize the common failure patterns, from frequent to rare:

- • Incomplete coverage of the ground-truth regions.
- • Attraction to objects visually similar to the target object in the scene.
- • Complete failure to detect the target object. We observe that TTT partially mitigates these errors,

though some failures persist, leaving room for further improvement.

For future work, we plan to incorporate temporal cues to better capture object dynamics and further reduce the failure patterns identified above.

[Figure 8]

###### Figure 8. More qualitative results on the Ego-Exo4D correspondence benchmark (Ego2Exo). Each row shows a representative sample. Rows 1–3 correspond to cooking, row 4 to health, row 5 to bike repair, row 6 to music performance, row 7 to basketball, and row 8 to soccer.

[Figure 9]

###### Figure 9. More qualitative results on the Ego-Exo4D correspondence benchmark (Exo2Ego). Each row shows a representative sample. Rows 1–3 correspond to cooking, row 4 to health, row 5 to bike repair, row 6 to music performance, row 7 to basketball, and row 8 to soccer.

[Figure 10]

###### Figure 10. More qualitative results on the HANDAL-X benchmark. Each row shows a representative sample.

