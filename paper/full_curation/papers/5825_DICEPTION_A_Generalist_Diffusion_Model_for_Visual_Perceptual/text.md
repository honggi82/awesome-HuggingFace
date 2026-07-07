# arXiv:2502.17157v3[cs.CV]9Oct2025

## DICEPTION: A Generalist Diffusion Model for Visual Perceptual Tasks

Canyu Zhao1 Yanlong Sun2 Mingyu Liu1 Huanyi Zheng1 Muzhi Zhu1 Zhiyue Zhao1 Hao Chen1 Tong He1,3 Chunhua Shen1,4 1 Zhejiang University 2 Tsinghua University

3 Shanghai AI Laboratory 4 Zhejiang University of Technology

#### Abstract

This paper’s primary objective is to develop a robust generalist perception model capable of addressing multiple tasks under constraints of computational resources and limited training data. We leverage text-to-image diffusion models pre-trained on billions of images and successfully introduce our DICEPTION, a visual generalist model. Exhaustive evaluations demonstrate that DICEPTION effectively tackles diverse perception tasks, even achieving performance comparable to SOTA single-task specialist models. Specifically, we achieve results on par with SAMvit-h using only 0.06% of their data (e.g., 600K vs. 1B pixel-level annotated images). We designed comprehensive experiments on architectures and input paradigms, demonstrating that the key to successfully re-purposing a single diffusion model for multiple perception tasks lies in maximizing the preservation of the pre-trained model’s prior knowledge. Consequently, DICEPTION can be trained with substantially lower computational costs than conventional models requiring training from scratch. Furthermore, adapting DICEPTION to novel tasks is highly efficient, necessitating fine-tuning on as few as 50 images and approximately 1% of its parameters. Finally, we demonstrate that a subtle application of classifier-free guidance can improve the model’s performance on depth and normal estimation. We also show that pixel-aligned training, as is characteristic of perception tasks, significantly enhances the model’s ability to preserve fine details. DICEPTION offers valuable insights and presents a promising direction for the development of advanced diffusion-based visual generalist models. Code and model: Homepage.

#### 1 Introduction

Foundation models [51, 90, 123, 124, 121, 11, 7, 86, 78, 94, 6, 40], typically requiring extensive training on billions of data samples, play a pivotal role in their respective domains. In natural language processing (NLP), current foundation models [9, 103, 104, 27] have already demonstrated the potential to serve as versatile solutions, solving diverse fundamental tasks and with minimal fine-tuning needed for new tasks. This success can be attributed to the relatively small representational differences among various language tasks. However, in the domain of computer vision, task representations can differ substantially, and up to date, we still lack an effective approach to unify these distinct tasks. Consequently, existing vision foundation models usually excel at one single specific task, such as image segmentation [51, 90] or monocular depth estimation [123, 124, 121], because they are trained on data tailored exclusively to that task. Owing to the pronounced disparity in visual representations across tasks, coupled with the single-task specialization that characterizes current vision foundation models, fine-tuning these models for new tasks remains a formidable challenge. Although some efforts [12, 78, 40, 91] have been made to learn universal visual representations for more generalized vision foundation models, their performance still falls noticeably short compared to specialized models.

Preprint. Under review.

[Figure 1]

- Figure 1: With one single model, DICEPTION solves 6 perception tasks without relying on any taskspecific modules. The red dots in the figure indicate the input points used for interactive segmentation. DICEPTION can quickly adapt to new tasks by fine-tuning less than 1% of its parameters on as few as 50 images. For additional visualizations, please refer to Figures S8, S11, S10, S15, S16, S17, S18, S19, S20, S21, S22 in the Appendix. We select Person as the instance segmentation example for the purpose of consistent visualization, which does not mean our method is limited to only human instances.

Recent studies [113, 71, 70, 75, 2, 117] on visual generalist models are predominantly trained from scratch, often requiring substantial computational resources and large datasets to achieve good results. Unfortunately, the price of collecting a sufficiently large and high-quality multi-task dataset is substantial. Here, inspired by the success of diffusion models, we propose the hypothesis that leveraging their powerful priors can help mitigate the significant computational and data overhead for training powerful generalist models. While some existing works [49, 118, 39, 127, 96] have demonstrated that this is feasible in single-task scenarios, the potential of diffusion model priors in multi-task settings remains largely under-explored.

In this paper, we successfully leverage the priors of diffusion models to achieve results on par with the state-of-the-art models on various tasks with only minimal training data. We name our powerful visual generalist model DICEPTION. For each task, we require substantially less data than specialized foundation models. For instance, compared to SAM segmentation trained on 1 billion pixel-level annotated samples, DICEPTION achieves comparable performance using a significantly smaller dataset of 600K samples, without any training data cherry-picking.

More significantly, DICEPTION highlights that the generative image priors lead to surprisingly more efficient and effective pathways to generalist image understanding models. We analyze a series of design choices for transferring one single modern diffusion model to multiple perception tasks, and identify that the key to successful transfer lies in preserving as much of the pretrained prior as possible, eliminating the need to design any complex module or training recipe. Even more notably, DICEPTION is capable of quickly adapting to new tasks using as few as 50 training images and fine-tuning less than 1% of its parameters. We also demonstrate that pixel-level aligned training for perception tasks significantly enhances the model’s ability to preserve fine details and mitigates generated artifacts, which is of high significance for downstream applications. We believe DICEPTION provides valuable insights for the design and training of strong diffusion-based visual generalist models.

In summary, our main contributions are as follows.

- • We introduce DICEPTION, to the best of our knowledge, the first unified multi-task perception model with fully shared parameters that achieves quantitative performance comparable

- to specialized models while requiring significantly less data. E.g., we achieve competitive results with SAM-vit-h with only 0.06% of its data. We are capable of addressing six visual perception tasks within one single model.
- • This work offers a comprehensive experimental analysis elucidating the critical designs for effectively re-purposing diffusion models towards perception tasks, including architecture, input injection strategies and sampling timestep selection. Our findings establish that the preservation of the pretrained generative prior is paramount for achieving rapid adaptation and robust multi-task performance. Notably, DiT architectures are shown to be particularly conducive to this objective.
- • The proposed unified multi-task paradigm yields compelling advantages. For instance, DICEPTION rapidly adapts to novel tasks in few-shot settings, demonstrating strong performance with as few as 50 training images and fine-tuning only 1% of parameters. Training on pixel alignment tasks significantly mitigates the artifacts often observed in other generative models for low-level image processing tasks such as image highlighting. Furthermore, the unified prediction space enables interactive segmentation to achieve matting-level accuracy.

#### 2 Related Work

##### 2.1 Vision Foundation Models

Vision foundation models are models that are trained on large-scale datasets and demonstrate excellent performance within their trained domains. Vision foundation models now exist for a broad range of vision tasks, including monocular depth estimation [123, 124, 121, 7], object detection [11], segmentation [51, 90], multimodal tasks [86, 66], image and video generation [94, 29, 6], and more recently, emerging 3D models [109, 73]. While many works [115, 50, 60, 87, 138, 141] have sought to leverage the prior knowledge embedded in these models to tackle other tasks, such efforts often require complex network designs and intricate training strategies, typically transferring only to a limited number of tasks. Some foundation models [91, 40, 78, 12] emphasize representation learning, aiming to solve diverse downstream tasks by relying on generalized features. However, the results of these methods often fall short when compared with specialized foundation models. In contrast, our approach ensures consistent accuracy across multiple tasks while also enabling swift adaptation to new downstream tasks.

##### 2.2 Diffusion Models

Diffusion models [29, 94, 6] have achieved remarkable success in image and video generation in recent years. The idea is to gradually add noise to the data and train a model to reverse this process, denoising step by step to generate the result. Recent diffusion models [29] utilize flow matching [65, 1, 68] and the DiT architecture [80], making them more scalable and efficient. Diffusion models have enabled a wide range of notable applications, including conditional image generation [135, 63, 128, 76, 85], image editing [8, 48, 120], story generation [111, 140], video generation [42, 36, 137, 126, 6, 52, 110], and video editing [13, 67, 14]. These successes underscore the substantial prior knowledge embedded in diffusion models.

Building on this insight, many studies [118, 39, 127, 49, 141] leverage diffusion models for downstream image understanding tasks. However, these approaches typically require separate fine-tuning for each individual task. Recently, we find several concurrent works [116, 55] also use diffusion models for multitask learning. Yet, these methods often involve complex network architectures and training procedures, and their evaluations tend to focus only on a very limited subset of image understanding results. In contrast, our DICEPTION offers a simpler solution. We not only conduct detailed evaluations of our method across a variety of tasks but also demonstrate that the simplicity, paired with the inherent strengths of diffusion models, can be sufficient to deliver strong results without relying on overly complicated setups.

##### 2.3 Multi-task Generalist Models

Recently, there has been a surge of interest in exploring visual multitask learning. Some approaches [113, 114] draw inspiration from in-context learning in NLP, adapting it for the visual domain. Others [71, 70, 75, 2] have advocated for sequence modeling methods, utilizing a transformer

encoder-decoder architecture. In these approaches, different encoders map various tasks into a shared representation space, and distinct decoders are employed to transform tokens into the outputs specific to each task. However, these methods face notable limitations: they need to train a separate encoder and decoder for every individual task and they usually rely on substantial amounts of data to attain optimal performance.

The recent success of high-quality Vision Language Models (VLMs) [66] has also encouraged researchers to leverage them for building multitask models. Yet, these VLM-based methods [4, 108, 17, 69, 92, 61] typically focus on multimodal understanding tasks, such as image captioning, rather than general visual perception tasks. Meanwhile, some approaches [101, 137, 79] combine diffusion models with autoregressive models, focusing primarily on instruction-following image generation or editing tasks, rather than addressing image perception tasks. Although certain studies [54, 47, 18, 35] have tried to apply VLMs to more advanced semantic perception tasks, they struggle to establish a unified generalist visual model.

##### 2.4 Compared with One Diffusion

The concurrent work, One Diffusion [55], addresses multi-task image generation, whereas our approach focuses on multi-task image understanding. We excel at performing a broader range of image understanding tasks with higher quality. While One Diffusion’s strategy of treating different images as different views benefits generation tasks, their failure to distinguish between conditions and images introduces harmful degrees of freedom for perception tasks, as illustrated in the redhighlighted regions of Figure S14. Specifically, when performing perception tasks, One Diffusion tends to generate an image similar to the original input, rather than the desired perceptual results.

Although One Diffusion suggests that more detailed text prompts can lead to better results, we argue that performance in perception tasks should not overly depend on the quality of text prompts. In contrast, our method uses only simple task prompts to distinguish between different tasks, rather than allowing the text prompts to dominate the results.

Crucially, while One Diffusion requires a massive amount of data (75 million samples) and computational resources for from-scratch training, we leverage the priors of pretrained models and demonstrate that, with significantly less data (1.8 million samples), we achieve performance on par with state-of-the-art results. In the image understanding tasks shared by both approaches, we consistently produce more stable and higher-quality results than One Diffusion.

#### 3 Method

##### 3.1 Overview

Our methodology builds upon pre-trained text-to-image diffusion models [29], steering perception tasks using text prompts. As shown in Figure 2, we concatenate the input image tokens, the noisy tokens, task prompt embeddings, and point embeddings for interactive segmentation along the token dimension. Training employs a flow matching loss [29], exclusively computed on the noisy tokens. In inference, each denoising step refines only these noisy tokens, leaving all other conditioning tokens unchanged throughout the iterative denoising process.

##### 3.2 Unifying Task Representation into RGB Space

The decision to unify representations of diverse tasks in RGB space was motivated by two key factors: (1) It maximally leverages the priors in text-to-image models, which have been extensively trained within the RGB domain. (2) RGB serves as a foundational representation in computer vision, providing a common visual framework through which a wide variety of tasks can be coherently and intuitively visualized.

We focus on several of the most fundamental tasks in computer vision: monocular depth estimation, normal estimation, human keypoint estimation and segmentation. Segmentation, in particular, encompasses interactive segmentation, entity segmentation, and instance segmentation. Our instance segmentation segments target instances with category name as input. All these tasks can be unified within an RGB space, with the difference being the number of channels. For single-channel representations, such as depth maps and segmentation masks, we align them with RGB by repeating the

###### 2. Pipeline

###### 1. Unified Different Tasks into RGB

Input Image Noise Task Prompt Point Embedding

Image2Depth Image2Normal Image2Entity

|Valid Embeds|
|---|

|Image2Seg|
|---|

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

|Interactive Seg|
|---|

[Figure 6]

[Figure 7]

[Figure 8]

(u,v)

Image2Seg Image2Pose Image2Instance - Person

|Invalid Embeds|
|---|

|Image2Depth|
|---|

|Other Tasks|
|---|

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

VAE+Patchify VAE+Patchify Text Encoder Point Embedding Processing

| | | | | |
|---|---|---|---|---|

| | | | | |
|---|---|---|---|---|

| | | | | |
|---|---|---|---|---|

| | | | | |
|---|---|---|---|---|

- 3. Point Embedding Processing Interactive Seg (5 Points at Most)

Token-wise Concatenation

| |
|---|

Invalid Embeds Valid Embeds

| | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

2-Layer MLP

Positional Encoding

(u,v)

| |
|---|

| |
|---|

+

Fully Parameter-Sharing DiT

- If 2 Points:
- If 3 Points:

### =

| | | | | |
|---|---|---|---|---|

| | | | | |
|---|---|---|---|---|

| | | | | |
|---|---|---|---|---|

| | | | | |
|---|---|---|---|---|

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|

| | | | | |
|---|---|---|---|---|

Other Tasks

| | | | | |
|---|---|---|---|---|

### =

| | | | | |
|---|---|---|---|---|

| | | | | |
|---|---|---|---|---|

Flow Matching Loss

- Figure 2: We propose a generalist diffusion model solving multiple perception tasks, DICEPTION. We select Person as the instance segmentation example for the purpose of consistent visualization, which does not mean our method is limited to only human instances. At each denoising step, the point embedding, input image latent, and task embedding remain fixed, while only the noise latent is updated.

channel three times. For inherently three-channel representations, such as normal maps, we treat them directly as RGB images.

Entity segmentation is to segment every instance in an image but with no category. We assign each mask within an image a random color and merge them into a three-channel RGB mask. Painter [113] found that assigning color randomly makes the model hard to optimize. However, we find this approach has no adverse impact on the training and enables the model to effectively learn to distinguish different instances by painting them with different colors. Each instance’s mask can be extracted from the RGB mask using clustering algorithms during post-processing without significant performance degradation. We also apply the random color assignment in instance segmentation. Our method is capable of segmenting instances of the same semantic category. By default, we use KMeans for mask extraction.

Let xr denote the pre-unified raw representation for each task, and x represents the unified RGB-like output representation. We formalize this process as: x = Ψ(xr).

##### 3.3 DICEPTION: A Unified Framework

Architecture. Our model adopts the same architecture as SD3 [29]. We aim to keep the architecture as unchanged as possible, fully leveraging the pre-trained prior knowledge. To do so, we concatenate the input image tokens, noisy tokens, task embeddings, and point embeddings along the token dimension as input to the model. During training, the loss is computed only on the noisy tokens. Similarly, during inference, at each timestep, only the noisy tokens are updated, while the other tokens remain unchanged. We use simple task prompts to direct the model to perform various tasks, such as "image to depth", "image to normal", and "image to segmentation". An additional category name is provided in instance segmentation, such as "image to instance - cat".

Introduction of Point Embeddings For point-prompted interactive segmentation, a naive approach is directly painting points on the image. But this strategy is highly sensitive to the size of the points. If the painted points are too large, they can obscure small regions, causing segmentation to fail. Conversely, if the painted points are too small, the model may lose relevant point information after VAE downsampling and patchification. To address this, we introduce a minimal straightforward two-layer MLP Φ(·) that enables the model to understand the point prompt.

Inspired by SAM [51], we apply sin-cos positional encoding to the point coordinates p, then pass them into the MLP Φ(·) to produce point embeddings that match the dimension of the input hidden states. We use two learnable embeddings to indicate whether the embedding is valid or not: ξp for

valid point embeddings and ξnp for invalid point embeddings. The processed point embedding is summed with ξp. For other tasks, we simply use ξnp as the point embedding. During training, we randomly select 1–5 points to guide the segmentation. When the number of selected points is fewer than 5, we pad the point embeddings to a length of 5 with ξnp. When performing tasks that do not require point input, the point embedding is simply a length-5 sequence, where each element is ξnp. By denoting the final point embedding as ξ, this process is formulated as:

Concat(Φ(PE(p)) + ξp,ξnp) if interactive segmentation ξnp else

(1)

ξ =

Input Formulation and Loss. DICEPTION introduces two additional inputs based on SD3: the input image x′ and point embedding ξ. For the input image, we first apply VAE to down-sample it by a factor of 8, after which it is 2 × 2 patchified into sequences. We denote this pre-processing as τ. Subsequently, the task prompt token e, point embedding ξ, noisy token zt, and input image token z′ are concatenated along the token dimension to form the complete input. We follow the flow matching [65, 1, 68] loss in training SD3 [29], which minimizes the discrepancy between the model’s predicted velocity v and the ground-truth velocity u. During training, the loss is applied solely to the noisy tokens:

z0 = τ(x),z′ = τ(x′) Loss = Ez

(2)

0,t∥vθ(zt,z′,t,e,ξ) − u(zt)∥22.

##### 3.4 Adapting to New Tasks

Practical applications often require models to adapt quickly to new tasks with limited training data. Traditional foundation models, however, are often domain-specific and require extensive data and architectural modifications for adaptation. Powerful diffusion models also struggle with efficient adaptation to downstream tasks via few-parameter fine-tuning on limited data.

DICEPTION effectively addresses this limitation. We conducted experiments on lung segmentation, tumor segmentation, and image highlighting, which represent tasks with varying degrees of overlap with the model’s original domain. We train fewer than 1% of the model’s parameters using LoRA [44] without any complex architectural modifications. Notably, despite the limited availability of training samples (50 per task), DICEPTION consistently delivered successful and high-quality performance across all target tasks. These results provide compelling evidence for the potential of DICEPTION as a unified foundation model.

#### 4 Experiments

##### 4.1 Implementation Details

Data. We randomly select 500k images from the OpenImages [53] dataset and use DepthPro [7] and StableNormal [127] to generate depth and normal annotations. For interactive segmentation, we randomly select 400k images from the SA-1B [51] dataset, as well as 200k images with fine-grained hair masks synthesized from the AM2k [58], AIM500 [59], and P3M-10k [57]. Entity segmentation data is from EntityV2 [84], while instance segmentation data comes from the COCO-Rem [97], and human pose data is sourced from COCO [64]. For few-shot fine-tuning, we select 50 samples from the Chest X-Ray dataset [112], LOL-v2 [125], and Kaggle’s Brain Tumor dataset as training samples. More details can be found in Appendix A.

Training. Our training lasts for 24 days using 4 NVIDIA H800 GPUs. We employ the AdamW optimizer with a constant learning rate of 2e−5 and a batch size of 28 per GPU. We found that the training process is highly stable. However, the convergence speed for segmentation tasks was slower compared to depth and normal tasks. Therefore, we increased the proportion of segmentation data in each batch. Specifically, in each batch, depth and normal each account for 15%, interactive segmentation, entity segmentation, and instance segmentation each account for 20%, and pose estimation accounts for 20%. We observe that, by the end of training, despite the loss no longer significantly decreasing, the model’s performance on segmentation tasks continues to improve.

During few-shot fine-tuning, we apply a rank-128 LoRA to all attention Q, K, and V layers in the network, which accounts for less than 1% of the total network parameters. The task prompts for

different tasks are “image-to-segmentation lung," “image-to-segmentation tumor," and “image-tohighlight." LoRA training is conducted on a single NVIDIA H100 GPU, with a constant learning rate of 2e−5 and a batch size of 8. Please refer to Appendix D for more few-shot fine-tuning visualizations.

Inference. We perform 28 steps of denoising during inference which follows the settings of the pre-trained model SD3 [29]. The inference can be run on a GPU of 24GB memory with a batch size of 4. The classifier-free-guidance value is by default set to 2, more analysis in Appendix B.

##### 4.2 Comparisons with Existing Methods

- Table 1: Quantitative comparison of depth estimation with both specialized models and multi-task models on zero-shot datasets. Our visual generalist model can perform on par with SOTA models. We use the same evaluation protocol (†) as Genpercept [118].

|Method<br><br>|Training Samples|KITTI [33] AbsRel|NYUv2 [77] AbsRel|ScanNet [24] AbsRel<br><br>|DIODE [106] AbsRel|ETH3D [95] AbsRel|
|---|---|---|---|---|---|---|
| | |↓ δ1↑<br><br>|↓ δ1↑<br><br>|↓ δ1↑<br><br>|↓ δ1↑|↓ δ1↑|
|MiDaS [89] Omnidata [28] DPT-large [88]<br><br>DepthAnything† [123] DepthAnything v2† [124]<br><br>Depth Pro† [7] Metric3D v2† [45]<br><br>DiverseDepth [129] LeReS [130] HDN [132] GeoWizard [32]<br><br>DepthFM [34] Marigold† [49]<br><br>DMP Official† [56] GeoWizard† [32] DepthFM† [34] Genpercept† [118]<br><br>|2M 12.2M 1.4M<br><br>63.5M 62.6M -<br><br>16M 320K 354K 300K 280K 63K 74K 280K 63K 90K|0.236 0.630 0.149 0.835 0.100 0.901 0.080 0.946 0.080 0.943 0.055 0.974 0.052 0.979 0.190 0.704 0.149 0.784 0.115 0.867 0.097 0.921 0.083 0.934 0.099 0.916 0.240 0.622 0.129 0.851 0.174 0.718 0.094 0.923<br><br>|0.111 0.885 0.074 0.945 0.098 0.903 0.043 0.980 0.043 0.979 0.042 0.977 0.039 0.979 0.117 0.875<br><br>0.090 0.916<br><br>0.069 0.948 0.052 0.966 0.065 0.956 0.055 0.964 0.109 0.891 0.059 0.959 0.082 0.932<br><br>0.091 0.932<br><br><br>|0.121 0.846 0.075 0.936 0.082 0.934 0.043 0.981 0.042 0.979 0.041 0.978 0.023 0.989 0.109 0.882 0.091 0.917 0.080 0.939 0.061 0.953 - 0.064 0.951 0.146 0.814 0.066 0.953 0.095 0.903 0.056 0.965<br><br>|0.332 0.715 0.339 0.742 0.182 0.758 0.261 0.759 0.321 0.758 0.217 0.764 0.147 0.892 0.376 0.631 0.271 0.766 0.246 0.780 0.297 0.792 0.225 0.800 0.308 0.773 0.361 0.706 0.328 0.753 0.334 0.729 0.302 0.767|0.184 0.752 0.166 0.778 0.078 0.946 0.058 0.984 0.066 0.983 0.043 0.974 0.040 0.983 0.228 0.694 0.171 0.777 0.121 0.833<br><br>0.064 0.961<br><br>- -<br><br>0.065 0.960<br><br>0.128 0.857 0.077 0.940 0.101 0.902<br><br>0.066 0.957<br>|
|Painter† [113] Unified-IO† [71]<br><br>4M-XL† [75] OneDiffusion† [55]|24K 48K 759M 500K<br><br>|0.324 0.393 0.188 0.699 0.105 0.896 0.101 0.908<br><br>|0.046 0.979 0.059 0.970 0.068 0.951 0.087 0.924|0.083 0.927 0.063 0.965 0.065 0.955 0.094 0.906<br><br>|0.342 0.534 0.369 0.708 0.331 0.734 0.399 0.661<br><br>|0.203 0.644 0.103 0.906 0.070 0.953 0.072 0.949|
|Ours-single† Ours†<br><br>|500K 500K|0.064 0.952 0.069 0.949<br><br>|0.066 0.953 0.061 0.960|0.077 0.942 0.072 0.944<br><br>|0.283 0.717 0.289 0.722<br><br>|0.052 0.971 0.050 0.975|

We compare the performance of specialized models, existing multi-task models, and our DICEPTION across various tasks. Specifically, we evaluate depth using the same protocol as Genpercept [118], normal estimation using the same method as StableNormal [127], interactive segmentation using the same approach as SAM [90], and human keypoints using the same method as Painter [113]. We also assess instance segmentation and entity segmentation on the MS COCO dataset. For entity segmentation, we assigned all predicted categories to the same label.

Table 3: Evaluation of human keypoints estimation on MS COCO.

| |HRNet[100] HRFormer[133] ViTPose[119]<br><br>|Painter[113] Ours|
|---|---|---|
|AP↑<br><br>|76.3 77.2 78.3|72.5 57.8|

As in Tables 1 and 2, our DICEPTION outperforms existing multi-task models and achieves performance on par with state-of-the-art specialized models or demonstrates only an acceptable performance decrease. Although some multitask methods achieve marginally better performance on certain datasets, such as Painter [113] and Unified-IO [70], they exhibit considerably poorer results on others such as outdoor settings (KITTI) and NYUv2 normal map benchmark. This further underscores the robust generalization capabilities of our approach. We contend that focusing on a model’s performance across diverse datasets is more meaningful, as it better reflects the model’s generalization ability and real-world applicability.

Table 4: Evaluation of text-based instance segmentation on the MS COCO.

Method SparK [102] OneFormer [46] Mask2Former [19] Ours AP↑ 45.1 49.2 50.1 33.2

For interactive segmentation, as shown in Figure 3, we achieve results on par with SAM-vit-h using only 0.06% of their data. SAM shows a clear advantage only on certain out-of-distribution

- Table 2: Quantitative comparison of surface normal estimation with both specialized models and multi-task models. All methods are evaluated with the same method of StableNormal [127].

|Method|Training Samples<br><br>|NYUv2 [77] mean med| |ScanNet [24] mean med<br><br>| |DIODE-indoor [106] mean med| |
|---|---|---|---|---|---|---|---|
| | |↓ ↓<br><br>|11.25◦↑ 22.5◦↑ 30◦↑<br><br>|↓ ↓|11.25◦↑ 22.5◦↑ 30◦↑<br><br>|↓ ↓|11.25◦↑ 22.5◦↑ 30◦↑<br><br>|
|DINSE [3] Geowizard [32]<br><br>GenPercept [118] Marigold [49] StableNormal [127]|160K 280K 90K 90K 250K<br><br>|18.572 10.845<br><br>20.363 11.898 20.896 11.516 20.864 11.134<br><br>19.707 10.527<br><br><br>|54.732 74.146 80.256 46.954 73.787 80.804 50.712 73.037 79.216 50.457 73.003 79.332 53.042 75.889 81.723|18.610 9.885<br><br>19.748 9.702<br><br><br>18.600 8.293 18.463 8.442 17.248 8.057<br><br>|56.132 76.944 82.606 58.427 77.616 81.575 64.697 79.329 82.978 64.727 79.559 83.199 66.655 81.134 84.632|18.453 13.871<br><br>19.371 15.408<br><br><br>18.348 13.367 16.671 12.084 13.701 9.460<br><br>|36.274 77.527 86.976 30.551 75.426 86.357 39.178 79.819 88.551 45.776 82.076 89.879 63.447 86.309 92.107|
|Unified-IO [70] 4M-XL [75]|210K 759M<br><br>|28.547 14.637 37.278 13.661<br><br>|39.907 63.912 71.240 44.660 60.553 65.327|17.955 10.269 30.700 11.614<br><br>|54.120 77.617 83.728 48.743 68.867 73.623|31.576 16.615 18.189 12.979<br><br>|27.855 64.973 73.445 36.622 81.844 87.050<br><br>|
|Ours-single Ours|500K 500K<br><br>|18.292 10.145 18.338 10.106<br><br>|52.693 76.966 83.041 52.850 77.079 82.903<br><br>|18.807 10.327 18.842 10.266<br><br>|52.919 75.152 82.968<br>53.610 74.895 82.864<br>|16.229 11.012 16.297 11.117<br><br>|50.137 83.573 88.972 50.548 83.325 88.774<br><br>|

Ours

+18.8

DRAM

Ours-single

+17.1

NDISPark

SAM-vit-h

+9.1

BBBC038v1

+7.8

DOORS

+7.1

Hypersim

+5.4

CityScapes

+4.1

OVIS

- +2.9
- +3.1

NDD20

ADE20k

+2.1

PPDLS

+0.1

iShape

- -1.5
- -1.4

STREETS

Total mIoU Comparison

GTEA

60

- -4.3
- -3.9

LVIS

46.93 47.10 48.90

50

ZeroWaste

- -10.2
- -10.0

EgoHOS

40

VISOR

-11.7

TimberSeg

30

-12.4

WoodScape

20

-14.0

PIDRay

- -15.3
- -14.9

Plittersdoff

10

TrasCan

-16.7

IBD

0

Ours-single Ours SAM-vit-h

20 15 10 5 0 5 10 15

IoU delta at 1 point

- Figure 3: Comparisons of mIoU with SAM-vit-h. We achieve results on par with SAM using only 0.06% of their data (600K vs. 1B). The performance of SAM is clearly better only on some datasets that are out-of-distribution for us, such as the Woodscape [131] Fisheye dataset.

datasets that are outside the scope of our model’s training, such as WoodScape fisheye dataset. Notably, while most specialized models require extensive data or complex data pipelines, our method achieves excellent results with significantly less data and no training data cherry-picking. Evaluation across diverse datasets highlights the strong in-the-wild generalization capability of our model, demonstrating that it does not overfit to the biases inherent in specific datasets.

We observe that, although our model generates high-quality visualizations for human pose and instance segmentation, the corresponding evaluation metrics remain relatively low. This is also observed on the evaluation of small objects in entity segmentation. We found that this is due to the errors introduced by the post-processing rather than our model’s performance. In Appendix C, we provide a comprehensive explanation of the post-processing procedure and analyze the underlying causes of metrics degradation.

##### 4.3 Ablations and Analysis

Model designs, classifier-free guidance and pixel-aligned training. Our crucial analyses covering the elucidation of critical designs for effectively re-purposing diffusion models for perception tasks, as well as significant findings and insights, are detailed in the Appendix due to space limit. Specifically, the analysis of different architectures and input paradigms is presented in Appendix B.1, B.2 and B.3. The effectiveness of modest classifier-free guidance in improving results is discussed in Appendix B.4. The inherent few-step capability of flow-matching on perception tasks is analyzed in Appendix B.5. The benefits of pixel-aligned training are detailed in Appendix B.6 and B.7.

Table 6: Average recall (AR) of entity segmentation on the MS COCO validation set.

Method AR-small↑ AR-medium↑ AR-large↑ EntityV2 [84] 0.313 0.551 0.683

Ours-single 0.123 0.424 0.648 Ours 0.121 0.439 0.637

Comparisons with Our Single-task Models. For the training of single-task models, we ensure that the network architecture remains the same and the total amount of training data seen for each specific task is the same as that for the multi-task model. For example, if the multi-task model is trained for 100 iterations with 4 depth data samples per batch, the single-task model will also be trained for 100 iterations with 4 data samples per batch. In our current data setting (approximately 1.8 million samples), we have not observed a significant gap between the multi-task and single-task models, nor have we seen a trend of mutual promotion between different tasks, as shown by “Ours-single" in Tables 1, 2, 6 and Figure 3.

We believe that it is more appropriate to explore with larger datasets in order to draw more solid conclusions. We leave this as future work.

Multi-point Prompted Segmentation. Ambiguity is a significant issue in interactive segmentation. For example, if a point is placed on a person’s clothing, the model may segment the clothing, but the desired result is the person. Therefore, more points are needed to resolve this ambiguity. As illustrated in Table 5, additional points help the model better segment the desired results.

One-step Training and One-step Inference. Genpercept [118] demonstrates that diffusion model trained with one-step denoising significantly enhances both the speed and accuracy of perceptual tasks. However, our experimental results reveal a notable increase of failure cases when applying one-step diffusion in a multi-task setting, as illustrated in Figure 4. We believe that this is due to the potential overlap of denoising trajectories for different tasks. These overlapping trajectories can interfere with each other, resulting in failure cases with one-step inference. In contrast, in a single-task setting, since the denoising trajectories pertain to a single task, one-step is more effective and stable. However, we observe that our model, trained with multi-step denoising, can be applied directly to few-step inference with minimal degradation in performance. We provide the results and a more detailed analysis in Appendix B.5 due to space limitations.

Table 5: Comparisons between 1-point and 5-point as input. 5 points are selected randomly.

Method 1-point 5-point mIoU↑ 47.1 57.2

#### 5 Conclusion

We have introduced DICEPTION, a multi-task visual generalist model based on the diffusion model. Our approach unifies different tasks in the RGB space, leveraging the prior knowledge of pre-trained image generation model to achieve results that are on par with specialized foundation models. We achieve good performance without carefully cherry-picking extremely high-quality data or by using an exceptionally large amount of data. In few-shot finetuning, we are able to achieve high-quality results with minimal data and minimal trainable parameters.

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Figure 4: The model trained with 1-step denoising tends to produce more failure cases in multi-task scenarios.

Furthermore, we provide in-depth experimental analyses of strategies for transferring diffusion models to perception tasks. We also discuss the contributions of classifier-free guidance in enhancing model performance, demonstrate that there is no performance gap between our single-task and multi-task model, and highlight the improved detail preservation achieved through pixel-aligned perception training. We believe that DICEPTION sheds light on how to effectively use priors of diffusion model to build a strong visual generalist.

#### References

- [1] Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571, 2022.
- [2] Roman Bachmann, O˘guzhan Fatih Kar, David Mizrahi, Ali Garjani, Mingfei Gao, David Griffiths, Jiaming Hu, Afshin Dehghan, and Amir Zamir. 4m-21: An any-to-any vision model for tens of tasks and modalities. arXiv preprint arXiv:2406.09406, 2024.
- [3] Gwangbin Bae and Andrew J Davison. Rethinking inductive biases for surface normal estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9535–9545, 2024.
- [4] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.
- [5] Dina Bashkirova, Mohamed Abdelfattah, Ziliang Zhu, James Akl, Fadi Alladkani, Ping Hu, Vitaly Ablavsky, Berk Calli, Sarah Adel Bargal, and Kate Saenko. Zerowaste dataset: Towards deformable object segmentation in cluttered scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21147–21157, 2022.
- [6] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [7] Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R Richter, and Vladlen Koltun. Depth pro: Sharp monocular metric depth in less than a second. arXiv preprint arXiv:2410.02073, 2024.
- [8] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023.
- [9] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [10] Juan C Caicedo, Allen Goodman, Kyle W Karhohs, Beth A Cimini, Jeanelle Ackerman, Marzieh Haghighi, CherKeng Heng, Tim Becker, Minh Doan, Claire McQuin, et al. Nucleus segmentation across imaging experiments: the 2018 data science bowl. Nature methods, 16(12):1247–1253, 2019.
- [11] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020.
- [12] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.
- [13] Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23206–23217, 2023.
- [14] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stablevideo: Text-driven consistency-aware diffusion video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23040–23050, 2023.
- [15] Jiazhou Chen, Yanghui Xu, Shufang Lu, Ronghua Liang, and Liangliang Nan. 3-d instance segmentation of mvs buildings. IEEE Transactions on Geoscience and Remote Sensing, 60:1–14, 2022.
- [16] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.
- [17] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024.

- [18] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision language model. arXiv preprint arXiv:2406.01584, 2024.
- [19] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Maskedattention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1290–1299, 2022.
- [20] Luca Ciampi, Carlos Santiago, Joao Costeira, Claudio Gennaro, and Giuseppe Amato. Night and Day Instance Segmented Park (NDISPark) Dataset: a Collection of Images taken by Day and by Night for Vehicle Detection, Segmentation and Counting in Parking Areas, May 2022.
- [21] Luca Ciampi, Carlos Santiago, Joao Paulo Costeira, Claudio Gennaro, and Giuseppe Amato. Domain adaptation for traffic density estimation. In VISIGRAPP (5: VISAPP), pages 185–195, 2021.
- [22] Nadav Cohen, Yael Newman, and Ariel Shamir. Semantic segmentation in art paintings. In Computer graphics forum, volume 41, pages 261–275. Wiley Online Library, 2022.
- [23] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016.
- [24] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017.
- [25] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Evangelos Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. Rescaling egocentric vision: Collection, pipeline and challenges for epic-kitchens-100. International Journal of Computer Vision, pages 1–23, 2022.
- [26] Ahmad Darkhalil, Dandan Shan, Bin Zhu, Jian Ma, Amlan Kar, Richard Higgins, Sanja Fidler, David Fouhey, and Dima Damen. Epic-kitchens visor benchmark: Video segmentations and object relations. Advances in Neural Information Processing Systems, 35:13745–13758, 2022.
- [27] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [28] Ainaz Eftekhar, Alexander Sax, Jitendra Malik, and Amir Zamir. Omnidata: A scalable pipeline for making multi-task mid-level vision datasets from 3d scans. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10786–10796, 2021.
- [29] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.
- [30] Alireza Fathi, Xiaofeng Ren, and James M Rehg. Learning to recognize objects in egocentric activities. In CVPR 2011, pages 3281–3288. IEEE, 2011.
- [31] Jean-Michel Fortin, Olivier Gamache, Vincent Grondin, François Pomerleau, and Philippe Giguère. Instance segmentation for autonomous log grasping in forestry operations. In 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 6064–6071. IEEE, 2022.
- [32] Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma, Ping Tan, Shaojie Shen, Dahua Lin, and Xiaoxiao Long. Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image. In European Conference on Computer Vision, pages 241–258. Springer, 2024.
- [33] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. The International Journal of Robotics Research, 32(11):1231–1237, 2013.
- [34] Ming Gui, Johannes Schusterbauer, Ulrich Prestel, Pingchuan Ma, Dmytro Kotovenko, Olga Grebenkova, Stefan Andreas Baumann, Vincent Tao Hu, and Björn Ommer. Depthfm: Fast monocular depth estimation with flow matching. arXiv preprint arXiv:2403.13788, 2024.
- [35] Qiushan Guo, Shalini De Mello, Hongxu Yin, Wonmin Byeon, Ka Chun Cheung, Yizhou Yu, Ping Luo, and Sifei Liu. Regiongpt: Towards region understanding vision language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13796–13806, 2024.

- [36] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.
- [37] Agrim Gupta, Piotr Dollar, and Ross Girshick. Lvis: A dataset for large vocabulary instance segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5356– 5364, 2019.
- [38] Timm Haucke, Hjalmar S Kühl, and Volker Steinhage. Socrates: Introducing depth in visual wildlife monitoring using stereo vision. Sensors, 22(23):9082, 2022.
- [39] Jing He, Haodong Li, Wei Yin, Yixun Liang, Leheng Li, Kaiqiang Zhou, Hongbo Zhang, Bingbing Liu, and Ying-Cong Chen. Lotus: Diffusion-based visual foundation model for high-quality dense prediction. arXiv preprint arXiv:2409.18124, 2024.
- [40] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000–16009, 2022.
- [41] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance, 2022.
- [42] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022.
- [43] Jungseok Hong, Michael Fulton, and Junaed Sattar. Trashcan: A semantically-segmented dataset towards visual detection of marine debris. arXiv preprint arXiv:2007.08097, 2020.
- [44] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.
- [45] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. arXiv preprint arXiv:2404.15506, 2024.
- [46] Jitesh Jain, Jiachen Li, Mang Tik Chiu, Ali Hassani, Nikita Orlov, and Humphrey Shi. Oneformer: One transformer to rule universal image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2989–2998, 2023.
- [47] Qing Jiang, Yuqin Yang, Yuda Xiong, Yihao Chen, Zhaoyang Zeng, Tianhe Ren, Lei Zhang, et al. Chatrex: Taming multimodal llm for joint perception and understanding. arXiv preprint arXiv:2411.18363, 2024.
- [48] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023.
- [49] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9492–9502, 2024.
- [50] Samar Khanna, Medhanie Irgau, David B Lobell, and Stefano Ermon. Explora: Parameter-efficient extended pre-training to adapt vision transformers under domain shifts. arXiv preprint arXiv:2406.10973, 2024.
- [51] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023.
- [52] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [53] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International journal of computer vision, 128(7):1956–1981, 2020.
- [54] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9579–9589, 2024.

- [55] Duong H. Le, Tuan Pham, Sangho Lee, Christopher Clark, Aniruddha Kembhavi, Stephan Mandt, Ranjay Krishna, and Jiasen Lu. One diffusion to generate them all, 2024.
- [56] Hsin-Ying Lee, Hung-Yu Tseng, and Ming-Hsuan Yang. Exploiting diffusion prior for generalizable dense prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7861–7871, 2024.
- [57] Jizhizi Li, Sihan Ma, Jing Zhang, and Dacheng Tao. Privacy-preserving portrait matting. In Proceedings of the 29th ACM international conference on multimedia, pages 3501–3509, 2021.
- [58] Jizhizi Li, Jing Zhang, Stephen J Maybank, and Dacheng Tao. Bridging composite and real: towards end-to-end deep image matting. International Journal of Computer Vision, 130(2):246–266, 2022.
- [59] Jizhizi Li, Jing Zhang, and Dacheng Tao. Deep automatic natural image matting. arXiv preprint arXiv:2107.07235, 2021.
- [60] Siyuan Li, Lei Ke, Martin Danelljan, Luigi Piccinelli, Mattia Segu, Luc Van Gool, and Fisher Yu. Matching anything by segmenting anything. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18963–18973, 2024.
- [61] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision, pages 323–340. Springer, 2025.
- [62] Yin Li, Zhefan Ye, and James M Rehg. Delving into egocentric actions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 287–295, 2015.
- [63] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-Ming Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8640–8650, 2024.
- [64] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. Microsoft coco: Common objects in context, 2015.
- [65] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [66] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024.
- [67] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with cross-attention control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8599–8608, 2024.
- [68] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.
- [69] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, et al. Deepseek-vl: towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024.
- [70] Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26439–26455, 2024.
- [71] Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi. Unified-io: A unified model for vision, language, and multi-modal tasks. In The Eleventh International Conference on Learning Representations, 2022.
- [72] Xiaoqian Lv, Shengping Zhang, Qinglin Liu, Haozhe Xie, Bineng Zhong, and Huiyu Zhou. Backlitnet: A dataset and network for backlit image enhancement. Computer Vision and Image Understanding, 218:103403, 2022.
- [73] Baorui Ma, Huachen Gao, Haoge Deng, Zhengxiong Luo, Tiejun Huang, Lulu Tang, and Xinlong Wang. You see it, you got it: Learning 3d creation on pose-free videos at scale. arXiv preprint arXiv:2412.06699, 2024.

- [74] Massimo Minervini, Andreas Fischbach, Hanno Scharr, and Sotirios A Tsaftaris. Finely-grained annotated datasets for image-based plant phenotyping. Pattern recognition letters, 81:80–89, 2016.
- [75] David Mizrahi, Roman Bachmann, Oguzhan Kar, Teresa Yeo, Mingfei Gao, Afshin Dehghan, and Amir Zamir. 4m: Massively multimodal masked modeling. Advances in Neural Information Processing Systems, 36:58363–58408, 2023.
- [76] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2iadapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 4296–4304, 2024.
- [77] Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012.
- [78] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [79] Xichen Pan, Li Dong, Shaohan Huang, Zhiliang Peng, Wenhu Chen, and Furu Wei. Kosmos-g: Generating images in context with multimodal large language models. arXiv preprint arXiv:2310.02992, 2023.
- [80] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.
- [81] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [82] Mattia Pugliatti and Francesco Topputo. Doors: Dataset for boulders segmentation. statistical properties and blender setup, 2022.
- [83] Jiyang Qi, Yan Gao, Yao Hu, Xinggang Wang, Xiaoyu Liu, Xiang Bai, Serge Belongie, Alan Yuille, Philip HS Torr, and Song Bai. Occluded video instance segmentation: A benchmark. International Journal of Computer Vision, 130(8):2022–2039, 2022.
- [84] Lu Qi, Jason Kuen, Weidong Guo, Tiancheng Shen, Jiuxiang Gu, Jiaya Jia, Zhe Lin, and Ming-Hsuan Yang. High-quality entity segmentation. arXiv preprint arXiv:2211.05776, 2022.
- [85] Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, et al. Unicontrol: A unified diffusion model for controllable visual generation in the wild. arXiv preprint arXiv:2305.11147, 2023.
- [86] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [87] Frano Rajiˇc, Lei Ke, Yu-Wing Tai, Chi-Keung Tang, Martin Danelljan, and Fisher Yu. Segment anything meets point tracking. arXiv preprint arXiv:2307.01197, 2023.
- [88] René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179–12188, 2021.
- [89] René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3):1623–1637, 2020.
- [90] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.
- [91] Tianhe Ren, Yihao Chen, Qing Jiang, Zhaoyang Zeng, Yuda Xiong, Wenlong Liu, Zhengyu Ma, Junyi Shen, Yuan Gao, Xiaoke Jiang, et al. Dino-x: A unified vision model for open-world object detection and understanding. arXiv preprint arXiv:2411.14347, 2024.
- [92] Zhongwei Ren, Zhicheng Huang, Yunchao Wei, Yao Zhao, Dongmei Fu, Jiashi Feng, and Xiaojie Jin. Pixellm: Pixel reasoning with large multimodal model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26374–26383, 2024.

- [93] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10912–10922, 2021.
- [94] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [95] Thomas Schops, Johannes L Schonberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with high-resolution images and multicamera videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3260–3269, 2017.
- [96] Jiahao Shao, Yuanbo Yang, Hongyu Zhou, Youmin Zhang, Yujun Shen, Matteo Poggi, and Yiyi Liao. Learning temporally consistent video depth from video diffusion priors. arXiv preprint arXiv:2406.01493, 2024.
- [97] Shweta Singh, Aayan Yadav, Jitesh Jain, Humphrey Shi, Justin Johnson, and Karan Desai. Benchmarking object detectors with coco: A new path forward, 2024.
- [98] Corey Snyder and Minh Do. Streets: A novel camera network dataset for traffic flow. Advances in Neural Information Processing Systems, 32, 2019.
- [99] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2023.
- [100] Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. Deep high-resolution representation learning for human pose estimation, 2019.
- [101] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14398– 14409, 2024.
- [102] Keyu Tian, Yi Jiang, Qishuai Diao, Chen Lin, Liwei Wang, and Zehuan Yuan. Designing bert for convolutional networks: Sparse and hierarchical masked modeling. arXiv preprint arXiv:2301.03580, 2023.
- [103] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [104] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [105] Cameron Trotter, Georgia Atkinson, Matt Sharpe, Kirsten Richardson, A Stephen McGough, Nick Wright, Ben Burville, and Per Berggren. Ndd20: A large-scale few-shot dolphin dataset for coarse and fine-grained categorisation. arXiv preprint arXiv:2005.13359, 2020.
- [106] Igor Vasiljevic, Nick Kolkin, Shanyi Zhang, Ruotian Luo, Haochen Wang, Falcon Z Dai, Andrea F Daniele, Mohammadreza Mostajabi, Steven Basart, Matthew R Walter, et al. Diode: A dense indoor and outdoor depth dataset. arXiv preprint arXiv:1908.00463, 2019.
- [107] Boying Wang, Libo Zhang, Longyin Wen, Xianglong Liu, and Yanjun Wu. Towards real-world prohibited item detection: A large-scale x-ray benchmark. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5412–5421, 2021.
- [108] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [109] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697–20709, 2024.

- [110] Wen Wang, Qiuyu Wang, Kecheng Zheng, Hao Ouyang, Zhekai Chen, Biao Gong, Hao Chen, Yujun Shen, and Chunhua Shen. Framer: Interactive frame interpolation. arXiv preprint arXiv:2410.18978, 2024.
- [111] Wen Wang, Canyu Zhao, Hao Chen, Zhekai Chen, Kecheng Zheng, and Chunhua Shen. Autostory: Generating diverse storytelling images with minimal human efforts. International Journal of Computer Vision, pages 1–22, 2024.
- [112] Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, and R Summers. Hospitalscale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In IEEE CVPR, volume 7, page 46. sn, 2017.
- [113] Xinlong Wang, Wen Wang, Yue Cao, Chunhua Shen, and Tiejun Huang. Images speak in images: A generalist painter for in-context visual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6830–6839, 2023.
- [114] Xinlong Wang, Xiaosong Zhang, Yue Cao, Wen Wang, Chunhua Shen, and Tiejun Huang. Seggpt: Segmenting everything in context. arXiv preprint arXiv:2304.03284, 2023.
- [115] Xuehao Wang, Feiyang Ye, and Yu Zhang. Task-aware low-rank adaptation of segment anything model. arXiv preprint arXiv:2403.10971, 2024.
- [116] Zhaoqing Wang, Xiaobo Xia, Runnan Chen, Dongdong Yu, Changhu Wang, Mingming Gong, and Tongliang Liu. Lavin-dit: Large vision diffusion transformer. arXiv preprint arXiv:2411.11505, 2024.
- [117] Yuling Xi, Hao Chen, Ning Wang, Peng Wang, Yanning Zhang, Chunhua Shen, and Yifan Liu. A dynamic feature interaction framework for multi-task visual perception. International Journal of Computer Vision, 131(11):2977–2993, 2023.
- [118] Guangkai Xu, Yongtao Ge, Mingyu Liu, Chengxiang Fan, Kangyang Xie, Zhiyue Zhao, Hao Chen, and Chunhua Shen. Diffusion models trained with large data are transferable visual models. arXiv preprint arXiv:2403.06090, 2024.
- [119] Yufei Xu, Jing Zhang, Qiming Zhang, and Dacheng Tao. Vitpose: Simple vision transformer baselines for human pose estimation. Advances in Neural Information Processing Systems, 35:38571–38584, 2022.
- [120] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18381–18391, 2023.
- [121] Honghui Yang, Di Huang, Wei Yin, Chunhua Shen, Haifeng Liu, Xiaofei He, Binbin Lin, Wanli Ouyang, and Tong He. Depth any video with scalable synthetic data. arXiv preprint arXiv:2410.10815, 2024.
- [122] Lei Yang, Yan Zi Wei, Yisheng He, Wei Sun, Zhenhang Huang, Haibin Huang, and Haoqiang Fan. ishape: A first step towards irregular shape instance segmentation. arXiv preprint arXiv:2109.15068, 2021.
- [123] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10371–10381, 2024.
- [124] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. arXiv preprint arXiv:2406.09414, 2024.
- [125] Wenhan Yang, Shiqi Wang, Yuming Fang, Yue Wang, and Jiaying Liu. From fidelity to perceptual quality: A semi-supervised approach for low-light image enhancement. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3063–3072, 2020.
- [126] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [127] Chongjie Ye, Lingteng Qiu, Xiaodong Gu, Qi Zuo, Yushuang Wu, Zilong Dong, Liefeng Bo, Yuliang Xiu, and Xiaoguang Han. Stablenormal: Reducing diffusion variance for stable and sharp normal. ACM Transactions on Graphics (TOG), 43(6):1–18, 2024.
- [128] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

- [129] Wei Yin, Xinlong Wang, Chunhua Shen, Yifan Liu, Zhi Tian, Songcen Xu, Changming Sun, and Dou Renyin. Diversedepth: Affine-invariant depth prediction using diverse data. arXiv preprint arXiv:2002.00569, 2020.
- [130] Wei Yin, Jianming Zhang, Oliver Wang, Simon Niklaus, Long Mai, Simon Chen, and Chunhua Shen. Learning to recover 3d scene shape from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 204–213, 2021.
- [131] Senthil Yogamani, Ciarán Hughes, Jonathan Horgan, Ganesh Sistu, Padraig Varley, Derek O’Dea, Michal Uricár, Stefan Milz, Martin Simon, Karl Amende, et al. Woodscape: A multi-task, multi-camera fisheye dataset for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9308–9318, 2019.
- [132] Qian Yu, Xiaoqi Zhao, Youwei Pang, Lihe Zhang, and Huchuan Lu. Multi-view aggregation network for dichotomous image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3921–3930, 2024.
- [133] Yuhui Yuan, Rao Fu, Lang Huang, Weihong Lin, Chao Zhang, Xilin Chen, and Jingdong Wang. Hrformer: High-resolution transformer for dense prediction, 2021.
- [134] Lingzhi Zhang, Shenghao Zhou, Simon Stent, and Jianbo Shi. Fine-grained egocentric hand-object segmentation: Dataset, model, and applications. In European Conference on Computer Vision, pages 127–145. Springer, 2022.
- [135] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.
- [136] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Scaling in-the-wild training for diffusion-based illumination harmonization and editing by imposing consistent light transport. In The Thirteenth International Conference on Learning Representations, 2025.
- [137] Canyu Zhao, Mingyu Liu, Wen Wang, Weihua Chen, Fan Wang, Hao Chen, Bo Zhang, and Chunhua Shen. Moviedreamer: Hierarchical generation for coherent long visual sequence. arXiv preprint arXiv:2407.16655, 2024.
- [138] Zihan Zhong, Zhiqiang Tang, Tong He, Haoyang Fang, and Chun Yuan. Convolution meets lora: Parameter efficient finetuning for segment anything model. arXiv preprint arXiv:2401.17868, 2024.
- [139] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. International Journal of Computer Vision, 127:302–321, 2019.
- [140] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent self-attention for long-range image and video generation. arXiv preprint arXiv:2405.01434, 2024.
- [141] Muzhi Zhu, Yang Liu, Zekai Luo, Chenchen Jing, Hao Chen, Guangkai Xu, Xinlong Wang, and Chunhua Shen. Unleashing the potential of the diffusion model in few-shot semantic segmentation. arXiv preprint arXiv:2410.02369, 2024.

Appendix

- A Dataset

We summarize the datasets used in our work in Table S1. The depth and normal data samples are obtained by randomly selecting 500K images from OpenImages [53] and labeling them using Depth Pro [7] and StableNormal [127], respectively. The 400K point segmentation data samples are obtained by randomly selecting images from the SA-1B dataset [51]. For the synthesis of point segmentation data, we extract the foreground from P3M-10K [57], AIM500 [59] and AM2K [58], randomly applying transformations such as rotation, resizing, and flipping. These transformed foregrounds are then pasted onto different background images, resulting in 200K synthetic images with fine-grained hair segmentation.

Table S1: Dataset detail.

Training

|Task<br><br>|Data Samples<br><br>|Dataset|
|---|---|---|
|Depth Normal Point Segmentation Point Segmentation Human Pose Semantic Segmentation Entity Segmentation<br><br>|500K 500K 400K 200K 42K 120K 32K|OpenImages [53] + Depth Pro [7] OpenImages [53] + StableNormal [127] SA-1B [51] P3M-10K [57], AIM500 [59] and AM2K [58] MS COCO 2017 [64] COCO-Rem [97] EntityV2 [84]<br><br>|

Validation

|Task|Dataset<br><br>|
|---|---|
|Depth<br><br>|NYUv2 [77], KITTI [33], ScanNet [24], DIODE [106], ETH3D [95]|
|Normal<br><br>|NYUv2 [77], ScanNet [24], DIODE [106]|
|Point Segmentation<br><br>|PPDLS [74], DOORS [82], TimberSeg [31], NDD20 [105] STREETS [98], iShape [122], ADE20K [139], OVIS [83] Plittersdorf [38], EgoHOS [134], IBD [15], WoodScape [131] TrashCan [43], GTEA [30, 62], NDISPark [21, 20], VISOR [25, 26] LVIS [37], Hypersim [93], Cityscapes [23], DRAM [22] BBBC038v1 [10], ZeroWaste [5], PIDRay [107]|
|Entity Segmentation<br><br>|MS COCO 2017 [64]|
|Semantic Segmentation|MS COCO 2017 [64]|
|Human Keypoints<br><br>|MS COCO 2017 [64]|

For the validation set, we evaluate depth using the same evaluation protocol as Genpercept [118], conducting tests on the NYUv2 [77], KITTI [33], ScanNet [24], DIODE [106], ETH3D [95]. Similarly, for normal estimation, we follow the evaluation protocol of StableNormal [127] and perform evaluations on the NYUv2 [77], ScanNet [24], DIODE [106]. For interactive segmentation, we conduct extensive comparisons across 23 datasets. The remaining tasks, including Entity Segmentation, Instance Segmentation, and Human Keypoints, are evaluated on the MS COCO 2017 dataset [64]. We believe the comprehensive experiments on over 30 datasets in total provide solid evidence of the remarkable performance of our method.

- B Additional Analysis

- B.1 Token-wise Concat and Channel-wise Concat

We investigated two distinct methodologies for integrating an auxiliary input image into a Diffusion Transformer (DiT) architecture. The first approach involved concatenating the input image tokens with the noisy image tokens along the token dimension, subsequently feeding this combined sequence directly into the DiT model. The second strategy employed channel-wise concatenation of these inputs, followed by a shallow, two-layer Multi-Layer Perceptron (MLP) to align the channel dimensions with the DiT’s input.

Constrained by available computational resources, our analysis is conducted within 2 tasks: depth and surface normal estimation. The datasets utilized for depth and surface normal prediction in this ablation are identical to those specified in Table S1. All training hyperparameters remain consistent across both approaches, with the sole architectural divergence being the aforementioned two-layer MLP utilized for feature alignment in the channel-wise concatenation method.

Comparison of Training Loss Curves

2.00

Token-wise Concat

Channel-wise Concat

1.75

1.50

1.25

LossValue

1.00

0.75

0.50

0.25

0.00

0 1000 2000 3000 4000 5000 Training Step

Our findings indicate that the token-wise concatenation strategy is markedly more computationally efficient than its channel-wise counterpart. Specifically, the token-wise approach demonstrates substantially faster convergence speed, as illustrated by the training loss trajectories presented in Figure S1. Furthermore, as demonstrated in Figure S2, channel-wise concatenation is more prone to yielding suboptimal results. We believe that this enhanced efficiency and effectiveness stem from the token-wise concatenation method’s circumvention of additional network parameters. By avoiding the introduction of new trainable components, this strategy appears to more effectively leverage the inherent priors learned by the pre-trained diffusion model. Furthermore, for token-wise concatenation, we independently applied Rotary Position Embeddings (RoPE) [99] to both the input image tokens and the noisy tokens. This strategy ensures that corresponding tokens from these two sources share identical positional embeddings, facilitating the model’s rapid learning of their interrelations.

Figure S1: Loss curve of token-wise concatenation and channel-wise concatenation.

[Figure 19]

Figure S2: Depth and normal estimation multi-task visualizations comparing channel-wise concatenation, token-wise concatenation, and U-Net are shown. While channel-wise concatenation often leads to suboptimal performance and U-Net struggles with multi-task learning, DICEPTION effectively generates high-quality outputs for multiple tasks.

##### B.2 Architecture of Diffusion Model.

Before the advent of DiT [80], the UNet architecture was predominantly used in diffusion models. We also conduct multi-task experiments based on a UNet pre-trained model SDXL [81]. Specifically, we follow Marigold [49] by expanding the first convolution layer’s input channels from 4 to 8 to accommodate image inputs, and similarly use task prompts to guide the model in solving different tasks. However, as shown in Figure S2 and S3 , we find that this approach failed, even for a minimal multi-task scenario involving only depth and normal estimation.

Beyond the established UNet architecture, our research also encompasses an exploration of alternative DiT frameworks, notably PixArt-alpha [16], to ascertain the generalizability and efficacy of our proposed methodology when applied to different DiT models. We train DICEPTION-PixArt based on the PixArt-alpha-600M model using the same data for training DICEPTION and conduct a quantitative evaluation on depth and surface normal prediction, as illustrated in Table S2, S3 and S4.

It is pertinent to note that, with a parameter count of approximately 600M, the DICEPTION-PixArt variant, while not achieving the same performance benchmarks as our counterpart model trained on

the more extensive sd3 architecture, still exhibits a strong capacity for multi-task problem-solving. This multi-tasking proficiency is substantially superior to that of traditional UNet-based models. This result substantiates the versatility of our method and its compatibility with modern transformer-based diffusion models, even with smaller models.

Table S2: Quantitative comparison of depth estimation between ours and ours-PixArt.

|Method|Training Samples<br><br>|KITTI [33] AbsRel|NYUv2 [77] AbsRel<br><br>|ScanNet [24] AbsRel|DIODE [106] AbsRel<br><br>|ETH3D [95] AbsRel|
|---|---|---|---|---|---|---|
| | |↓ δ1↑|↓ δ1↑<br><br>|↓ δ1↑|↓ δ1↑|↓ δ1↑<br><br>|
|Ours Ours-PixArt|500K 500K<br><br>|0.069 0.949 0.093 0.905<br><br>|0.061 0.960 0.096 0.905|0.072 0.944 0.101 0.901<br><br>|0.289 0.722 0.282 0.709|0.050 0.975 0.071 0.944<br><br>|

O

Table S3: Quantitative comparison of surface normal estimation between ours and ours-PixArt.

|Method<br><br>|Training Samples|NYUv2 [77] mean med<br><br>| |ScanNet [24] mean med| |DIODE-indoor [106] mean med| |
|---|---|---|---|---|---|---|---|
| | |↓ ↓<br><br>|11.25◦↑ 22.5◦↑ 30◦↑|↓ ↓|11.25◦↑ 22.5◦↑ 30◦↑<br><br>|↓ ↓<br><br>|11.25◦↑ 22.5◦↑ 30◦↑|
|Ours Ours-PixArt<br><br>|500K 500K|18.338 10.106 20.487 12.393<br><br>|52.850 77.079 82.903 48.663 72.342 80.244|18.842 10.266 21.663 14.419<br><br>|53.610 74.895 82.864 37.043 70.781 79.786<br><br>|16.297 11.117<br><br>17.986 11.190<br><br><br>|50.548 83.325 88.774 50.276 79.316 85.248|

Regarding the challenges encountered with UNet-based architectures in multi-task learning paradigms, we posit that their limitations are fundamentally due to two key factors. Firstly, the approach of expanding the input convolution layer introduces additional parameters, thereby potentially disrupting the original model’s inherent prior knowledge. Secondly, the downsampling operations within the U-Net architecture result in a significant loss of information.

Table S4: Comparisons of 1-point interactive segmentation between ours and ours-PixArt.

Method Ours-PixArt Ours SAM-vit-h mIoU↑ 40.93 47.10 48.90

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Figure S3: The UNet-based model fails to perform multi-task.

##### B.3 ControlNet

ControlNet [135] has emerged as a popular approach for integrating novel image conditioning into diffusion models. However, our experiment shows that while ControlNet can learn the general output patterns associated with target tasks, its precision remains notably low, exhibiting limited performance even on single perception task. We train a ControlNet on top of a pre-trained SD3 model for human keypoint estimation. Following the setup of traditional setting [135], we introduce ControlNet into the first half of the SD3’s transformer blocks. As depicted in Figure S4, although the model successfully captures the overall visual style of human keypoint predictions, the accuracy of its estimations is significantly deficient.

[Figure 28]

- Figure S4: While ControlNet demonstrates the ability to learn the output modalities of perception tasks, its accuracy remains significantly low. Conversely, our proposed approach yields substantially improved accuracy.

##### B.4 Classifier-free Guidance

Classifier-free guidance (CFG) [41] is a technique used in conditional diffusion models to improve the quality of generated samples without additional training. It has become a cornerstone in existing textto-image models. During inference, it extrapolates from the model’s conditional and unconditional outputs to enhance the influence of the conditioning signal. Specifically, during denoising, the noise at each timestep is a fusion of conditional and unconditional noise:

nt = nt,uncond + CFG·(nt,cond − nt,uncond). (S1)

Typically, conditional noise nt,cond is the output predicted by the model when conditioned on the prompt embedding, while unconditional noise nt,uncond is the output predicted by the model when conditioned on the negative prompt embedding.

We evaluate the impact of varying CFG values on our multi-task performance. Specifically, our conditional noise nt,cond is the prediction of the model conditioned on the task prompt corresponding to each specific task, while the unconditional noise nt,uncond is the model’s prediction when conditioned on an empty string as the prompt. Our ablation study reveals that a modest application of CFG enhances the quality of depth and normal estimation, yielding perceptibly sharper results. However, this strategy basically has no influence on other tasks such as human keypoints estimation and segmentation, as shown in Figure S5..

Table S5: Interactive Segmentation mIoU of DICEPTION across different CFG. CFG has little influence on segmentation

CFG=1 CFG=2 CFG=3 CFG=4 CFG=5 mIoU of 23 Validation Datasets 47.10 47.12 47.08 46.91 46.57

We hypothesize that this is because tasks such as depth and normal estimation inherently demand high precision in the output pixel values to accurately represent continuous geometric surfaces, while other tasks such as human keypoints estimation and segmentation are less sensitive to subtle variations in pixel-level intensities. Additionally, it is also observed that a high CFG scale significantly degrades performance on depth and normal prediction, especially normal prediction. This degradation typically manifests as oversaturated results or the emergence of coarse, granular artifacts, as shown in Figure S5. To further validate our hypothesis, we evaluate the performance of our model across varying CFG values, as presented in the Table S6, S7 and S5. The results confirm that a mild CFG scale enhances prediction quality of depth and normal, whereas larger values adversely affect performance.

Table S6: Quantitative comparison of depth estimation with different CFG value.

|Method<br><br>|Training Samples<br><br>|KITTI [33] AbsRel|NYUv2 [77] AbsRel<br><br>|ScanNet [24] AbsRel|DIODE [106] AbsRel|ETH3D [95] AbsRel|
|---|---|---|---|---|---|---|
| | |↓ δ1↑<br><br>|↓ δ1↑|↓ δ1↑<br><br>|↓ δ1↑|↓ δ1↑|
|Ours-CFG=1<br>Ours-CFG=2<br>Ours-CFG=3<br>Ours-CFG=4<br>Ours-CFG=5<br>|500K 500K 500K 500K 500K<br><br>|0.075 0.945 0.069 0.949 0.092 0.910 0.105 0.876 0.124 0.831<br><br>|0.072 0.939 0.061 0.960 0.076 0.938 0.087 0.915 0.097 0.893|0.075 0.938 0.072 0.944 0.093 0.910 0.104 0.884 0.115 0.863<br><br>|0.243 0.741 0.289 0.722 0.343 0.679 0.362 0.654 0.383 0.609<br><br>|0.053 0.967 0.050 0.975 0.059 0.966 0.066 0.956 0.072 0.947|

O O O O

Table S7: Quantitative comparison of surface normal estimation with different CFG value.

|Method|Training Samples<br><br>|NYUv2 [77] mean med| |ScanNet [24] mean med<br><br>| |DIODE-indoor [106] mean med| |
|---|---|---|---|---|---|---|---|
| | |↓ ↓<br><br>|11.25◦↑ 22.5◦↑ 30◦↑<br><br>|↓ ↓|11.25◦↑ 22.5◦↑ 30◦↑<br><br>|↓ ↓|11.25◦↑ 22.5◦↑ 30◦↑|
|Ours-CFG=1<br><br>Ours-CFG=2<br><br>Ours-CFG=3<br><br>Ours-CFG=4<br><br>Ours-CFG=5<br><br><br>|500K 500K 500K 500K 500K|18.302 10.538<br><br>18.338 10.106<br><br>19.817 10.989<br><br><br>21.433 12.012 23.352 13.259<br><br>|52.533 75.977 82.573 52.850 77.079 82.903 51.312 72.509 79.497 47.543 69.175 77.003 43.016 65.727 73.443<br><br>|19.348 12.129 18.842 10.266 22.287 11.849 24.117 13.029 26.972 14.364<br><br>|46.410 74.805 82.176 53.610 74.895 82.864 49.110 70.075 77.376 41.334 65.865 73.278 35.419 57.822 68.776|17.946 8.686<br><br>16.297 11.117<br><br>18.546 12.475<br><br><br>22.886 14.784 27.046 19.286<br><br>|62.641 81.152 85.398 50.548 83.325 88.774 46.627 76.532 85.398 41.271 65.661 74.098 33.349 56.885 66.728|

[Figure 29]

- Figure S5: Results on different guidance scale. Depth and normal predictions are highly sensitive to the CFG value, whereas other tasks are barely affected. Based on both the visualization results and the evaluation metrics in Table S6, S7 and S5, we set the CFG value to 2 by default.

##### B.5 Flow-matching Inherently Support Few-step Inference In Perception

We conduct experiments and observe that our model inherently supports few-step inference for perception tasks without any additional techniques, including classifier free guidance, and shows very little performance degradation. The effectiveness of few-step acceleration varies across different tasks. For tasks such as depth and surface normal estimation, the number of inference steps can be reduced to as few as one with acceptable slight performance degradation. For more complex tasks such as interactive segmentation, the model is still able to achieve comparable results using significantly fewer steps while maintaining competitive performance, as demonstrated in Table S8, S9 and S10. To the best of our knowledge, this is the first time such a capability is demonstrated in diffusion model for multi-task perception. It strongly supports the advantage of flow-matching-based diffusion models in solving perception tasks.

- Table S8: Quantitative comparison of our few-step depth estimation results.

|Method<br><br>|Training Samples<br><br>|KITTI [33] AbsRel<br><br>|NYUv2 [77] AbsRel|ScanNet [24] AbsRel|DIODE [106] AbsRel<br><br>|ETH3D [95] AbsRel|
|---|---|---|---|---|---|---|
| | |↓ δ1↑<br><br>|↓ δ1↑<br><br>|↓ δ1↑|↓ δ1↑|↓ δ1↑|
|28-step 14-step<br><br>7-step 3-step 1-step<br><br>|500K 500K 500K 500K 500K<br><br>|0.069 0.949 0.077 0.942 0.081 0.939 0.083 0.938 0.086 0.936|0.061 0.960 0.063 0.958 0.065 0.953 0.069 0.953 0.072 0.945<br><br>|0.072 0.944 0.074 0.943 0.078 0.943 0.077 0.940 0.076 0.937<br><br>|0.289 0.722 0.272 0.718 0.286 0.714 0.294 0.707 0.305 0.702<br><br>|0.050 0.975 0.048 0.978 0.052 0.971 0.063 0.967 0.065 0.967|

- Table S9: Quantitative comparison of our few-step normal map results.

|Method|Training Samples<br><br>|NYUv2 [77] mean med<br><br>| |ScanNet [24] mean med| |DIODE-indoor [106] mean med| |
|---|---|---|---|---|---|---|---|
| | |↓ ↓|11.25◦↑ 22.5◦↑ 30◦↑|↓ ↓|11.25◦↑ 22.5◦↑ 30◦↑<br><br>|↓ ↓|11.25◦↑ 22.5◦↑ 30◦↑|
|28-step 14-step<br><br>7-step 3-step 1-step|500K 500K 500K 500K 500K<br><br>|18.338 10.106 18.631 10.463 18.335 10.492 18.067 10.417 18.094 10.382<br><br>|52.850 77.079 82.903 52.837 75.288 81.682<br><br>52.771 75.443 81.936<br><br>53.046 76.500 81.673<br><br><br>51.839 76.575 81.371<br><br>|18.842 10.266<br><br>18.337 10.579<br><br>19.008 10.363<br><br><br>19.337 10.329 19.386 10.395<br><br><br>|53.610 74.895 82.864 53.223 75.533 82.631 52.628 74.886 82.055 52.223 75.731 82.081 52.139 75.492 81.879|16.297 11.117<br><br>16.131 11.463<br><br>16.835 11.330<br><br>17.205 12.047<br><br><br>17.004 11.849<br>|50.548 83.325 88.774 50.849 83.391 88.829 50.039 82.443 88.218 50.046 83.010 87.531 49.808 82.972 87.582<br><br>|

Table S10: Interactive Segmentation mIoU of DICEPTION across different inference steps.

28-step 14-step 7-step 3-step 1-step mIoU of 23 Validation Datasets 47.10 47.01 46.89 45.18 42.53

We believe this is because flow matching explicitly imposes linear constraints at each intermediate denoising step—specifically, each noisy latent is constructed as a linear interpolation between the pure noise and the target signal. This process effectively straightens the denoising trajectory, allowing the model to follow an approximately linear path even when using only a few inference steps. In contrast, if the model is trained solely with one-step denoising, the intermediate steps are not constrained and lacks this linear constraint across the trajectory, thus producing poor results as we show in Section 4.3. In contrast, traditional ODE-based diffusion models do not impose such linear trajectory constraints, and therefore cannot support inference with few denoising steps (such as 4 steps) after being trained with multi-step denoising (such as 50 steps). Our additional experiment proves this. We further experiment with PixArt-alpha [16], which uses a DiT-style architecture but adopts a standard ODE-based scheduler. Its results significantly deteriorate when the number of inference steps is reduced, as shown in Table S11, further supporting our analysis.

In image generation tasks, simply reducing inference steps in a flow-matching-based text-to-image model also leads to noticeable quality degradation. This is due to the high complexity and variability introduced by diverse text prompts. In contrast, our perception tasks eliminate the influence of textual prompts, which we believe explains why prior works like One Diffusion [55] require 50 100 inference steps for denoising while ours works well with just a few steps. For comparisons on inference efficiency, we select One Diffusion as baseline and conduct a comparative study on our shared task, depth estimation, under varying numbers of inference steps, as demonstrated in Table S12. Unlike One Diffusion, which suffers from significant performance degradation during few-step inference and fails to produce reasonable results in the 1-step setting, our method is capable of generating high-quality outputs even with just a single inference step. The results demonstrate that our method significantly outperforms One Diffusion in both efficiency and output quality.

Table S11: Quantitative comparison of few-step depth estimation results using Pixart-alpha.

|Method<br><br>|Training Samples<br><br>|KITTI [33] AbsRel|NYUv2 [77] AbsRel<br><br>|ScanNet [24] AbsRel|DIODE [106] AbsRel<br><br>|ETH3D [95] AbsRel|
|---|---|---|---|---|---|---|
| | |↓ δ1↑<br><br>|↓ δ1↑<br><br>|↓ δ1↑|↓ δ1↑<br><br>|↓ δ1↑|
|20-step 10-step|500K 500K<br><br>|0.093 0.905 0.146 0.872<br><br>|0.096 0.905 0.153 0.861<br><br>|0.101 0.901 0.159 0.844|0.282 0.709 0.347 0.658<br><br>|0.071 0.944 0.119 0.895|

Table S12: Quantitative comparison of One Diffusion and DICEPTION in few-step depth estimation. We compared three experimental settings based on the number of steps: the default configuration, a quarter of the default steps, and one single step.

|KITTI [33]<br><br>|NYUv2 [77]<br><br>|ScanNet [24]|DIODE [106]<br><br>|ETH3D [95]|
|---|---|---|---|---|
|AbsRel↓ δ1↑<br><br>|AbsRel↓ δ1↑<br><br>|AbsRel↓ δ1↑|AbsRel↓ δ1↑<br><br>|AbsRel↓ δ1↑|

Method

Ours-28-step 0.069 0.949 0.061 0.960 0.072 0.944 0.289 0.722 0.050 0.975 Ours-7-step 0.081 0.939 0.065 0.953 0.078 0.943 0.286 0.714 0.052 0.971 Ours-1-step 0.086 0.936 0.072 0.945 0.076 0.937 0.305 0.702 0.065 0.967

OD-50-step 0.101 0.908 0.087 0.924 0.094 0.906 0.399 0.661 0.072 0.949 OD-12-step 0.142 0.867 0.114 0.871 0.128 0.853 0.411 0.659 0.092 0.910

OD-1-step FAIL FAIL FAIL FAIL FAIL FAIL FAIL FAIL FAIL FAIL

##### B.6 Few-shot Finetuning Comparisons on SD3 and Ours

We conduct a comparative evaluation of few-shot tuning performance between SD3 and our DICEPTION. All training data and settings are kept identical for both approaches to ensure a fair comparison. Our findings reveal that DICEPTION not only adapts to new tasks more rapidly but also achieves better performance post-convergence when compared to SD3. Specifically, Figure S6 (a) illustrates that after convergence, our method yields higher-quality results than SD3 on image highlighting. Furthermore, as depicted in Figure S6 (b), DICEPTION demonstrates faster convergence speed. These results collectively underscore the substantial potential of our model for efficient and effective adaptation to novel tasks.

Input Ours SD3 Input Ours SD3

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

(a) Better Results (b) Faster Convergence

- Figure S6: Image highlighting few-shot finetuning comparisons on SD3 and Ours. (a) Our DICEPTION achieves better performance. Pixel-level aligned training mitigates generated artifacts. (b) Results of Ours and SD3 in the same training iteration. Our DICEPTION is able to adapt to new tasks faster than SD3.

##### B.7 Pixel-Level Alignment Training Enhances Detail Preservation

We find that training on pixel-level aligned perception tasks endows the model with a strong ability to preserve fine-grained details. We argue that this capability holds significant practical value. For instance, while existing state-of-the-art method IC-Light [136] for image relighting can generate visually impressive results, it often suffers from noticeable detail loss such as inconsistency of the individuals’ appearance. In contrast, our approach demonstrates superior fidelity in preserving

fine-grained details, including nuances that may not be readily perceptible to the human eye. This is demonstrated in our qualitative comparisons in Figure S7.

It is important to emphasize that our goal is not to compare the lighting quality between methods, but rather to highlight our model’s ability to significantly reduce generative artifacts and retain structural details. We attribute this strength to the model’s exposure to pixel-level aligned tasks during training. Additional comparisons with SD3 [29] in Figure S6 further support this observation. We consider this finding highly promising and believe it holds substantial implications for detail-preserving generative modeling and downstream applications.

[Figure 38]

- Figure S7: Comparisons of detail preservation, rather than lighting quality. Pixel-level aligned training leads to improved preservation of fine-grained details. Better viewed with zoom-in. Input images are generated and from public available BAID dataset [72].

- C Post-processing
- Figure S8: Segmentation results on furry objects. Our interactive segmentation achieves mattinglevel accuracy.

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

- Algorithm 1 Keypoints Post-processing

Input: human pose RGB x, GT keypoints Kgt, RGB tolerance σ, distance threshold ξ Output: extracted keypoints Kpred

- 1: x′ = ExtractRedRegions(x, (255,0,0), σ)
- 2: xc = GetConnectedComponents(x′)
- 3: C = GetCircular(xc)
- 4: Kpred = ∅
- 5: for c ∈ C do
- 6: k′ = ComputeCenterCoordinates(c)
- 7: dmin = ∞
- 8: for k ∈ Kgt do
- 9: d = ComputeEuclideanDistance(k′,k)
- 10: if d < dmin then
- 11: dmin = d
- 12: t = GetKeypointType(k)
- 13: end if
- 14: end for
- 15: if dmin < ξ then
- 16: continue
- 17: end if
- 18: Kpred = Kpred ∪ {(k′,t)}
- 19: end for
- 20: return Kpred

##### C.1 Post-processing for Keypoints

For keypoints, since all keypoints were labeled in red during training, our first step in post-processing is to extract all red regions from the RGB output. Next, we identify all connected components within the extracted red regions. For each connected component, we further extract sub-regions that approximate a circular shape. This step is crucial because, in some cases, multiple predicted

- Algorithm 2 Segmentation Post-processing

Input: RGB segmentation mask m, RGB tolerance σ, area threshold ξ, kernel size k, connected

components number threshold η, duplicate mask threshold β Output: extracted masks Mpred

- 1: Get the number of peaks p of the histogram of m
- 2: Get the number of clusters n = Mean(p)
- 3: Get the clustered colors by C = KMeans(m,n)
- 4: Mpred = ∅
- 5: for c ∈ C do
- 6: if IsCloseToBlack(c,σ) then
- 7: continue
- 8: end if
- 9: m′ = GetMaskByRGB(m,c,σ)
- 10: m′ = BinaryFillHoles(m′)
- 11: m′ = RefineWithMorphology(m′,k)
- 12: a = GetArea(m′)
- 13: if a < ξ then
- 14: continue
- 15: end if
- 16: y = GetConnectedComponentsNumber(m′)
- 17: if y > η then
- 18: continue
- 19: end if
- 20: Mpred = Mpred ∪ {m′}
- 21: end for
- 22: Mpred = RemoveDuplicateMasks(Mpred,β)
- 23: return Mpred

keypoints may overlap, requiring us to separate them as much as possible. For example, when a person clasps his hands together, the keypoints for both hands may overlap.

Once the circular regions are identified, we compute their center points as the predicted keypoint coordinates. Since our model does not explicitly predict the type of each keypoint (e.g., hand, foot), we assign keypoint types by measuring the distance between the extracted keypoints and the ground truth (GT) keypoints. Each predicted keypoint is assigned the type of its nearest GT keypoint. To ensure robustness, we apply a distance threshold, considering only those predicted keypoints that are sufficiently close to a GT keypoint. Finally, all extracted keypoints that are successfully matched to a GT keypoint form our final predicted keypoint coordinates after post-processing. The algorithm is shown in Algorithm 1.

##### C.2 Post-processing for RGB Masks

For entity segmentation and instance segmentation RGB masks, we employ clustering algorithms to extract the object masks. Specifically, we first compute the histogram peaks for each of the three RGB channels and estimate the number of clusters by averaging the peak counts across the three channels. We then use KMeans clustering to group the colors and identify the clustered regions in the RGB mask. For each identified cluster, we extract regions with RGB values close to the cluster’s centroid. This step is followed by morphological operations to refine the extracted masks, such as filling holes and removing small, fragmented regions. We further filter the masks by computing their area, excluding any regions that are too small to be meaningful.

Table S13: When post-processing RGB masks, small regions and excessive numbers of objects significantly lead to performance degradation.

|Category<br><br>|AP ↑|
|---|---|
|Bear Dog Cat Person Bird Book<br><br>|76.3 68.9 71.7 18.6 10.4 10.8|

Additionally, we also consider the number of connected components within the extracted masks, discarding overly fragmented results that have too many connected components. Finally, we refine the extracted masks by calculating the Intersection over Union (IoU) between them, removing any duplicate or overlapping masks. The algorithm is shown in Algorithm 2.

##### C.3 Performance Degradation of Keypoints

For human keypoints, the Performance degradation is primarily due to two factors: Firstly, we utilize skeletal-form RGB images rather than heatmaps. While the former produces visually appealing results, the extraction of keypoints during post-processing introduces considerable errors. Secondly, our evaluation follows the 192×256 top-down human keypoints protocol. The original 192×256 images are resized to 768×768 before being input into the model, resulting in extremely blurred inputs that likely contribute to the diminished performance.

##### C.4 Performance Degradation of RGB Masks

We observe that while the quality of our instance segmentation visualizations is high, the average precision (AP) for certain categories remains unsatisfactory. For example, for the Person category, we conducted exhaustive experiments and achieved good visualization results (highlighted by the green rectangle in Figure S9), but AP is low (as in Table S13).

We trace the root cause of metrics degradation during post-processing and find that this is particularly due to small objects and an excessive number of objects. Specifically, during mask processing, we filter out small noise regions. The genesis of these artifacts is predominantly attributed to subtle colorimetric fluctuations or minor inconsistencies in pixel values within areas of a mask intended to be uniformly colored. However, this operation also removes some positive samples, such as the crowd and the bird highlighted in red in rows 3 to 5 in Figure S9. These samples are susceptible to being misidentified as noise due to their diminutive size. Despite this limitation, the filtering of these noise regions is maintained because their persistence would otherwise exert a more detrimental impact on the quality of the final results. In our setting, filtering noise regions results in better metrics compared to not filtering them. Additionally, when an image contains an excessive number of objects of the same category (as in row 6 of Figure S9), post-processing may erroneously group similarly colored but distinct objects into a single class, leading to lower metrics. Furthermore, as in Table S13, we examine categories with fewer small objects and instances of those categories, such as bear, dog, and cat, and observe higher AP scores. However, for categories with opposite characteristics, their AP scores tend to be lower. This phenomenon is also observed in entity segmentation, which further elucidates why our entity segmentation results exhibit lower scores on small objects.

Although we can optimize post-processing by adjusting hyperparameters for each image to achieve the best results, this approach becomes impractical for large-scale evaluation, as it requires significant manual effort. Consequently, the dependency on post-processing remains a limitation of our approach.

#### D Additional Results

##### D.1 Additional Visualizations

We present additional visualization results of our method across various tasks, as can be seen in Figures S8, S11, S10, S15, S16, S17, S18, S19, S20, S21, S22. For interactive segmentation, we compare our approach with SAM. These results strongly demonstrate the potential of DICEPTION. DICEPTION is capable of achieving high-quality results, even in challenging scenarios. Furthermore, the few-shot fine-tuning of DICEPTION, which requires minimal data and trainable parameters, strongly demonstrates the remarkable transferability of DICEPTION to tackle new tasks. Our DICEPTION is capable of further refining the segmentation of fine details, such as intricate hair structures, achieving matting-level performance.

##### D.2 Comparative Experiments with One Diffusion

Qualitative visual comparisons between our method and One Diffusion in Figure S14 highlight key distinctions. In segmentation, our approach excels by simultaneously segmenting objects by semantic

|[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]|
|---|

[Figure 49]

[Figure 50]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 51]

[Figure 52]

| |
|---|

| |
|---|

| |
|---|
| |

[Figure 53]

[Figure 54]

| |
|---|

| |
|---|

[Figure 55]

[Figure 56]

###### Figure S9: When post-processing RGB masks, small regions and excessive numbers of objects lead to significant metric degradation.

Input Pred GT Input Pred GT

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

- Figure S10: Additional few-shot fine-tuning results on lung segmentation and tumor segmentation.

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Figure S11: Additional few-shot fine-tuning results on image highlighting.

class and differentiating individual instances—a capability lacking in One Diffusion. Moreover, the segmentation quality of our method is superior to that of One Diffusion, especially in object-dense regions where the latter exhibits noticeable performance degradation.

A critical limitation of One Diffusion is its apparent inability to distinguish input images from conditioning signals, leading to a conflation of image understanding tasks with image generation. For example, when performing human keypoint estimation, One Diffusion may erroneously generate an image depicting a similar pose rather than predicting the actual keypoints. Conversely, our model, being fundamentally oriented towards image perception, not only consistently yields high-quality, accurate results without confusion, but also performs challenging perception tasks inaccessible to One Diffusion, such as interactive segmentation.

- E Discussions and Limitations Discussions Our method highlights the following key findings:

- • The inherent prior knowledge of diffusion models is highly effective for perception tasks. By leveraging this prior effectively, our approach enables a single model to address multiple tasks. Notably, it achieves performance comparable to existing single-task specialized models, even on challenging tasks such as interactive segmentation, and does so with limited data.
- • Our comprehensive experimental evaluation demonstrates that token-wise concatenation is the most efficient and effective strategy for leveraging the prior knowledge of transformerbased diffusion models. Furthermore, we provide evidence that the DiT architecture works better compared to U-Net. This is attributed to the fact that transferring U-Net to multiple perception tasks not only introduces additional parameters that can potentially disrupt the pre-trained model’s prior but also suffers from significant information loss due to its inherent downsampling operations.
- • A modest CFG value can yield performance improvements for pixel-sensitive tasks such as depth and normal estimation.
- • We find that flow-matching models, when trained in a multi-step denoising setting, naturally support few-step inference for perception tasks.
- • Our DICEPTION exhibits a faster and more effective adaptation capability to new downstream tasks.
- • The efficacy of our approach is demonstrated on a different DiT architecture and smaller model, indicating its robustness.
- • The model demonstrates strong capability of detail preservation after pixel-aligned training on perception tasks.

To the best of our knowledge, we are the first to successfully leverage diffusion priors to address multiple perception tasks with a single model without exceptionally large or cherry-picking high-quality data, achieving performance on par with specialized models, even on the challenging interactive segmentation compared with SAM. In our view, the capabilities of our method are far from being fully realized, and further training with larger, higher-quality datasets has the potential to yield even more compelling results. For instance, in high-level tasks such as referring segmentation shown in Figure S12, our model achieves results with finer details than the ground truth. This not only demonstrates the model’s ability to benefit from related tasks but also showcases its strong semantic understanding. Furthermore, we observe early signs of task composition in our model, albeit with a low success rate. For instance, the model can predict the depth or normal map of an object indicated by point inputs while generating a black mask for other regions, as illustrated in Figure S13, though the success rate is very low. In conclusion, we believe that our work not only presents a generalist model with a vast capacity for improvement, but also provides comprehensive experiments and analyses that can serve as a valuable foundation for future research.

Limitations Although our DICEPTION achieves great results across multiple tasks, our model, as a diffusion model, leads to relatively longer inference times. On one H800, it takes an average of 0.8 seconds to process a single image. On one 4090-GPU card, inference for one image takes approximately 2 seconds. We believe that this issue can be addressed through few-step diffusion techniques, which we leave for future works.

Input Ours GT Input Ours GT

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

- Figure S12: DICEPTION achieves finer results on referring segmentation, showing the potential of mutual improvement between related tasks.

[Figure 77]

- Figure S13: Example of task composition. Our model can isolate a point-specified object to generate its corresponding depth map, while correctly suppressing predictions for all other regions. Although the success rate is very low, this result still reveals a promising capability.

Furthermore, our evaluation on certain tasks such as human keypoints estimation and text-based instance segmentation necessitates post-processing, which can introduce substantial errors. However, unlike some contemporary diffusion-based works [55, 116] that often omit quantitative evaluation on the task such as human keypoints estimation, we take a step further by providing evaluation metrics. Our analysis demonstrates that lower scores on these tasks are not due to model performance but are significantly influenced by the post-processing step. Consequently, the dependence on post-processing for quantitative evaluation on certain tasks remains a limitation of our method. Despite the limitations, we believe that DICEPTION is a valuable exploration for diffusion-based generalist visual perception foundation models.

Input One Diffusion Ours

[Figure 78]

[Figure 79]

[Figure 80]

|[Figure 81]|
|---|

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

|[Figure 87]|[Figure 88]|
|---|---|

[Figure 89]

- Figure S14: Our segmentation not only separates semantically identical objects but also distinguishes different instances of the same category, achieving higher segmentation quality. Moreover, One Diffusion tends to generate an image similar to the input when performing image understanding tasks, as red-highlighted in the figure.

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

###### Figure S15: Additional depth estimation visualizations.

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

###### Figure S16: Additional normal visualizations.

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

[Figure 120]

[Figure 121]

###### Figure S17: Additional entity segmentation visualizations.

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

###### Figure S18: Additional interactive segmentation visualizations.

Input Ours SAM Input Ours SAM

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

- Figure S19: Comparison of the segmentation results between DICEPTION and SAM-vit-h with 1-point input.

Input Ours SAM Input Ours SAM

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

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

- Figure S20: Comparison of the segmentation results between DICEPTION and SAM-vit-h with 5-point input.

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

###### Figure S21: Additional pose estimation visualizations.

[Figure 180]

[Figure 181]

Wine-glass Orange

[Figure 182]

[Figure 183]

Banana

Bear

[Figure 184]

[Figure 185]

Sheep

Pizza

[Figure 186]

[Figure 187]

Person

Chair

[Figure 188]

[Figure 189]

Broccoli

Person

[Figure 190]

[Figure 191]

Tie Suitcase

Figure S22: Additional text-based instance segmentation visualizations.

