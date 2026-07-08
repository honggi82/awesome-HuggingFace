## ZeroNVS: Zero-Shot 360-Degree View Synthesis from a Single Image

Kyle Sargent1, Zizhang Li1, Tanmay Shah2, Charles Herrmann2, Hong-Xing Yu1, Yunzhi Zhang1, Eric Ryan Chan1, Dmitry Lagun2, Li Fei-Fei1, Deqing Sun2, Jiajun Wu1 1Stanford University, 2Google Research

# arXiv:2310.17994v2[cs.CV]24Apr2024

### Abstract

We introduce a 3D-aware diffusion model, ZeroNVS, for single-image novel view synthesis for in-the-wild scenes. While existing methods are designed for single objects with masked backgrounds, we propose new techniques to address challenges introduced by in-the-wild multi-object scenes with complex backgrounds. Specifically, we train a generative prior on a mixture of data sources that capture object-centric, indoor, and outdoor scenes. To address issues from data mixture such as depth-scale ambiguity, we propose a novel camera conditioning parameterization and normalization scheme. Further, we observe that Score Distillation Sampling (SDS) tends to truncate the distribution of complex backgrounds during distillation of 360-degree scenes, and propose “SDS anchoring” to improve the diversity of synthesized novel views. Our model sets a new state-of-the-art result in LPIPS on the DTU dataset in the zero-shot setting, even outperforming methods specifically trained on DTU. We further adapt the challenging MipNeRF 360 dataset as a new benchmark for single-image novel view synthesis, and demonstrate strong performance in this setting. Code and models are available at this url.

### 1. Introduction

Models for single-image, 360-degree novel view synthesis (NVS) should produce realistic and diverse results: the synthesized images should look natural and 3D-consistent to humans, and they should also capture the many possible explanations of unobservable regions. This challenging problem has typically been studied in the context of single objects without backgrounds, where the requirements on both realism and diversity are simplified. Recent progress relies on large 3D datasets like Objaverse-XL [9] which have enabled training conditional diffusion [19] models to perform photorealistic and 3D-consistent NVS via Score Distillation Sampling [SDS; 27]. Meanwhile, since image diversity mostly lies in the background, not the object, the ignorance of background significantly lowers the expectation of synthesizing diverse images–in fact, most object-centric methods do not consider diversity metrics [19, 22, 28].

Neither assumption holds for the more challenging problem of zero-shot, 360-degree novel view synthesis on realworld scenes. There is no single, large-scale dataset of scenes with ground-truth geometry, texture, and camera parameters, analogous to Objaverse-XL for objects. The background, which cannot be ignored anymore, also needs to be well modeled for synthesizing diverse results.

We address both issues with our new model, ZeroNVS. Inspired by previous object-centric methods [19, 22, 28], ZeroNVS also trains a 2D conditional diffusion model followed by 3D distillation. But unlike them, ZeroNVS works well on scenes due to two technical innovations: a new camera parametrization and normalization scheme for conditioning, which allows training the diffusion model on diverse scene datasets, and an “SDS anchoring” mechanism, improving the background diversity over standard SDS.

To overcome the key challenge of limited training data, we propose training the diffusion model on a massive mixed dataset comprised of all scenes from CO3D [31], RealEstate10K [45], and ACID [17], so that the model may potentially handle complex in-the-wild scenes. The mixed data of such scale and diversity are captured with a variety of camera settings and have several different types of 3D ground truth, e.g., computed with COLMAP [33] or ORBSLAM [24]. We show that while the camera conditioning representations from prior methods [19] are too ambiguous or inexpressive to model in-the-wild scenes, our new camera parametrization and normalization scheme allows exploiting such diverse data sources and leads to superior NVS on real-world scenes.

Building a 2D conditional diffusion model that works effectively for in-the-wild scenes enables us to then study the limitations of SDS in the scene setting. In particular, we observe limited diversity from SDS in the generated scene backgrounds when synthesizing long-range (e.g., 180-degree) novel views. We therefore propose “SDS anchoring” to ameliorate the issue. In SDS anchoring, we propose to first sample several “anchor” novel views using the standard Denoising Diffusion Implicit Model (DDIM) sampling [37]. This yields a collection of pseudo-ground-truth novel views with diverse contents, since DDIM is not prone

CO3D

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Input view ———————— Novel views ———————— Input view ———————— Novel views ———————RealEstate10K

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Input view ———————— Novel views ———————— Input view ———————— Novel views ———————DTU (Zero-shot)

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Input view ———————— Novel views ———————— Input view ———————— Novel views ———————Mip-NeRF 360 (Zero-shot)

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Input view ———————— Novel views ———————— Input view ———————— Novel views ———————Figure 1. Results for view synthesis from a single image. All NeRFs are predicted by the same model.

to mode collapse like SDS. Then, rather than using these views as RGB supervision, we sample from them randomly as conditions for SDS, which enforces diversity while still ensuring 3D-consistent view synthesis.

ZeroNVS achieves strong zero-shot generalization to unseen data. We set a new state-of-the-art LPIPS score on the challenging DTU benchmark, even outperforming methods that were directly fine-tuned on this dataset. Since the popular benchmark DTU consists of scenes captured by a forward-facing camera rig and cannot evaluate more challenging pose changes, we propose to use the Mip-NeRF 360 dataset [2] as a single-image novel view synthesis benchmark. ZeroNVS achieves the best LPIPS performance on this benchmark. Finally, we show the potential of SDS anchoring for addressing diversity issues in background generation via a user study.

To summarize, we make the following contributions:

- • We propose ZeroNVS, which enables full-scene NVS from real images. ZeroNVS first demonstrates that SDS distillation can be used to lift scenes that are not objectcentric and may have complex backgrounds to 3D.
- • We show that the formulations on handling cameras and scene scale in prior work are either inexpressive or ambiguous for in-the-wild scenes. We propose a new camera conditioning parameterization and a scene normalization scheme. These enable us to train a single model on a large collection of diverse training data consisting of CO3D, RealEstate10K and ACID, allowing strong zeroshot generalization for NVS on in-the-wild images.
- • We study the limitations of SDS distillation as applied to scenes. Similar to prior work, we identify a diversity issue, which manifests in this case as novel view predictions with monotone backgrounds. We propose SDS anchoring to ameliorate the issue.
- • We show state-of-the-art LPIPS results on DTU zeroshot, surpassing prior methods finetuned on this dataset. Furthermore, we introduce the Mip-NeRF 360 dataset as a scene-level single-image novel view synthesis benchmark and analyze the performances of our and other methods. Finally, we show that our proposed SDS anchoring is preferred via a user study.

### 2. Related Work

3D generation. DreamFusion [27] proposed Score Distillation Sampling (SDS) as a way of leveraging a diffusion model to extract a NeRF given a user-provided text prompt. After DreamFusion, follow-up works such as Magic3D [16], ATT3D [21], ProlificDreamer [39], and Fantasia3D [7] improved the quality, diversity, resolution, or run-time. Other types of 3D generative models include GAN-based 3D generative models, which are primarily restricted to single object categories [4, 5, 13, 25, 26, 34] or to synthetic data [12]. Recently, 3DGP [35] adapted the GAN approach

to train 3D generative models on ImageNet. VQ3D [32] and IVID [41] leveraged vector quantization and diffusion, respectively, to learn generative models on ImageNet. One critical critical challenge for scene-based 3D-aware methods 360-degree viewpoint change. Both VQ3D and 3DGP demonstrate only limited camera motion, while IVID generally focuses on small camera motion but can achieve 360degree views for a small subset of scenes.

Single-image novel view synthesis. PixelNeRF [44] and DietNeRF [15] learn to infer NeRFs from sparse views via training an image-based 3D feature extractor or semantic consistency losses, respectively. However, these approaches do not produce renderings resembling crisp natural images from a single image. Several recent diffusion-based approaches achieve high quality results by separating the problem into two stages. First, a (potentially 3D-aware) diffusion model is trained, and second, the diffusion model is used to distill 3D-consistent scene representations given an input image via techniques like score distillation sampling [27], score Jacobian chaining [38], textual inversion or semantic guidance leveraging the diffusion model [10, 22], or explicit 3D reconstruction from multiple sampled views of the diffusion model [18, 20]. Another diffusion-based work, GeNVS [6], achieves 360 camera motion but only for specific object categories such as fire hydrants. By contrast, ZeroNVS generates 360-degree camera motion by default for a variety of scene categories. This is enabled by innovations such as novel camera conditioning representations and SDS anchoring, which enable us to train on massive real scene datasets and then to perform scene-level NVS with up to 360-degree viewpoint change on diverse scene types.

### 3. Approach

We consider the problem of scene-level novel view synthesis from a single real image. Similar to prior work [19, 28], we first train a diffusion model pθ to perform novel view synthesis, and then leverage it to perform 3D SDS distillation. Unlike prior work, we focus on scenes rather than objects. Scenes present several unique challenges. First, prior works use representations for cameras and scale which are either ambiguous or insufficiently expressive for scenes. Second, the inference procedure of prior works is based on SDS, which has a known mode collapse issue and which manifests in scenes through greatly reduced background diversity in predicted views. We will attempt to address these challenges through improved representations and inference procedures for scenes compared with prior work [19, 28].

We shall begin by introducing some general notation. Let a scene S consist of a set of images X = {Xi}ni=1, depth maps D = {Di}ni=1, extrinsics E = {Ei}ni=1, and a shared field-of-view f. We note that an extrinsics matrix Ei can be identified with its rotation and translation components, defined by Ei = (EiR,EiT). We preprocess our data

[Figure 33]

3D scene and cameras

Camera B

[Figure 34]

[Figure 35]

[Figure 36]

Camera B after 3DoF projection

[Figure 37]

Camera A

Images taken by cameras

|Image A<br><br>[Figure 38]|
|---|

|Image B<br><br>[Figure 39]|
|---|

|3DoF projection leads to an incorrect image<br><br>[Figure 40]|
|---|

Figure 2. A 3DoF camera pose captures elevation, azimuth, and radius but is incapable of representing a camera’s roll (pictured) or cameras positioned and oriented arbitrarily in space.

to consist of square images and assume intrinsics are shared within a given scene, and that there is no skew, distortion, or off-center principal point.

We will focus on the design of the conditional information which is passed to the view synthesis diffusion model pθ in addition to the input image. This conditional information can be represented via a function, M(D,f,E,i,j), which computes a conditioning embedding given the depth maps and extrinsics for the scene, the field of view, and the indices i,j of the input and target view respectively. We learn a generative model over novel views pθ given by

Xj ∼ pθ(Xj|Xi,M(D,f,E,i,j)) .

The output of M and the input image Xi are the only information used by the model for NVS. Both Zero-1-to-3 (Section 3.1) and our model, as well as several intermediate models that we will study (Sections 3.2 and 3.3), can be regarded as different choices for M. As we illustrate in Figures 2, 3, 5 and 6, and verify in experiments, different choices for M can have drastic impacts on performance.

#### 3.1. Representing Objects for View Synthesis

Zero-1-to-3 [19] represents poses with 3 degrees of freedom, given by an elevation θ, azimuth ϕ, and radius z. Let P : SE(3) → R3 be the projection to (θ,ϕ,z), then

MZero−1−to−3(D,f,E,i,j) = P(Ei) − P(Ej)

is the camera conditioning representation used by Zero1-to-3. For object mesh datasets such as Objaverse [8] and Objaverse-XL [9], this representation is appropriate because the data is known to consist of single objects without

[Figure 41]

Possible Scale B

Possible Scale A

[Figure 42]

[Figure 43]

Input Camera

[Figure 44]

Novel Camera

Figure 3. To a monocular camera, a small object close to the camera (left) and a large object at a distance (right) appear identical, despite representing different scenes. Scale ambiguity in input view leads to multiple plausible novel views.

backgrounds, aligned and centered at the origin and imaged from training cameras generated with three degrees of freedom. However, such a parameterization limits the model’s ability to generalize to non-object-centric images, and to real-world data. In real-world data, poses can have six degrees of freedom, incorporating both rotation (pitch, roll, yaw) and 3D translation. An illustration of a failure of the 3DoF camera representation is shown in Figure 2.

#### 3.2. Representing Scenes for View Synthesis

For scenes, we should use a camera representation with six degrees of freedom that can capture all possible positions and orientations. One straightforward choice is the relative pose parameterization [40]. We propose to also include the field of view as an additional degree of freedom. We term this combined representation “6DoF+1”. This gives us

##### M6DoF+1(D,f,E,i,j) = [Ei−1Ej,f].

Importantly, M6DoF+1 is invariant to any rigid transformation E˜ of the scene, so that we have

M6DoF+1(D,f,E˜ · E,i,j) = [Ei−1Ej,f] .

This is useful given the arbitrary nature of the poses for our datasets which are determined by COLMAP or ORBSLAM. The poses discovered via these algorithms are not related to any meaningful alignment of the scene, such as a rigid transformation and scale transformation which align the scene to some canonical frame and unit of scale. Although we have seen that M6DoF+1 is invariant to rigid transformations of the scene, it is not invariant to scale. The scene scales determined by COLMAP and ORB-SLAM are also arbitrary and may vary significantly. One solution is to directly normalize the camera locations. Let R(E,λ) : SE(3)×R → SE(3) be a function that scales the translation

Standard SDS Our SDS anchoring

[Figure 45]

Generated view

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

SDS guidance DDIM sampling

[Figure 50]

Input view NeRF Input view NeRF

Generated view

- Figure 4. SDS-based NeRF distillation (left) uses the same guidance image for all 360 degrees of novel views. Our “SDS anchoring” (right) first samples novel views via DDIM [36], and then uses the nearest image (whether the input or a sampled novel view) for guidance.

InputviewGTtargetview

PredictedtargetviewVarianceheatmap

[Figure 51]

Less variance More variance

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

- Figure 5. Samples and variance heatmaps of the Sobel edges of

[Figure 60]

- Camera setup 1

[Figure 61]

Camera C Camera A Camera B

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Camera A Camera B

- Camera setup 2

Figure 6. Top: A scene with two cameras facing the object. Bottom: The same scene with a new camera added facing the ground. Addition of Camera C under M6DoF+1, agg. drastically changes the scene’s scale. M6DoF+1, viewer. avoids this.

in Figure 3. We therefore choose to introduce information about the scale of the visible content to our conditioning embedding function M. Rather than normalize by camera locations, Stereo Magnification [45] takes the 5-th quantile of each depth map of the scene, and then takes the 10-th quantile of this aggregated set of numbers, and declares this as the scene scale. Let Qk be a function which takes the k-th quantile of a set of numbers, then we define

multiple samples from ZeroNVS. M6DoF+1, viewer reduces randomness from scale ambiguity.

component of E by λ. Then we define

n

n

1 n

1 n

∥EiT −

EjT∥2 ,

s =

i=1

j=1

q =Q10({Q5(Di)}ni=1) , M6DoF+1, agg.(D,f,E,i,j) = R Ei−1Ej,

1 s

M6DoF+1, norm.(D,f,E,i,j) = R Ei−1Ej,

,f ,

1 q

,f ,

where s is the average norm of the camera locations when the mean of the camera locations is chosen as the origin. In M6DoF+1, norm., the camera locations are normalized via rescaling by 1s, in contrast to M6DoF+1 where the scales are arbitrary. This choice of M assures that scenes from our mixture of datasets will have similar scales.

where in M6DoF+1, agg., q is the scale applied to the translation component of the scene’s cameras before computing the relative pose. In this way M6DoF+1, agg. is different from M6DoF+1, norm. because the camera conditioning representation contains information about the scale of the visible content from the depth maps Di. Although conditioning with M6DoF+1, agg. improves performance, there are two issues. The first arises from aggregating the quantiles over all the images. In Figure 6, adding an additional Camera C to the scene changes the value of M6DoF+1, agg. despite nothing else having changed about the scene. This makes the view synthesis task from either Camera A or Camera B more ambiguous. To ensure this is impossible, we can simply eliminate the aggregation step over the quantiles of

#### 3.3. Addressing Scale Ambiguity with a New Normalization Scheme

The representation M6DoF+1, norm. achieves reasonable performance on real scenes by addressing issues in prior representations with limited degrees of freedom and handling of scale. However, a normalization scheme that better addresses scale ambiguity may lead to improved performance. Scene scale is ambiguous given a monocular input image [30, 43]. This complicates NVS, as we illustrate

###### NVS on DTU LPIPS ↓ PSNR ↑ SSIM ↑

DS-NeRF† 0.649 12.17 0.410 PixelNeRF 0.535 15.55 0.537 SinNeRF 0.525 16.52 0.560 DietNeRF 0.487 14.24 0.481 NeRDi 0.421 14.47 0.465

ZeroNVS (ours) 0.380 13.55 0.469

- Table 1. Comparison with the state of the art. We set a new state-of-the-art for LPIPS on DTU despite being the only method not fine-tuned on DTU. † =Performance reported in Xu et al. [42].

NVS LPIPS ↓ PSNR ↑ SSIM ↑ Mip-NeRF 360 Dataset

Zero-1-to-3 0.667 11.7 0.196 PixelNeRF 0.718 16.5 0.556 ZeroNVS (ours) 0.625 13.2 0.240

DTU Dataset

Zero-1-to-3 0.472 10.70 0.383 PixelNeRF 0.738 10.46 0.397 ZeroNVS (ours) 0.380 13.55 0.469

- Table 2. Zero-shot comparison. Comparison with baselines retrained on our mixture dataset. ZeroNVS outperforms Zero-1-to-3 even when Zero-1-to-3 is trained on the same scene data. Extensive video comparisons are in the supplementary.

all depth maps in the scene. The second issue arises from different depth statistics within the mixture of datasets we use for training. ORB-SLAM generally produces sparser depth maps than COLMAP, and therefore the value of Qk may have different meanings for each. We therefore use an off-the-shelf depth estimator [29] to fill holes in the depth maps. We denote the depth Di infilled in this way as D¯i. We then apply Qk to dense depth maps D¯i instead. We emphasize that the depth estimator is not used during inference or distillation. Its purpose is only for the model to learn a consistent definition of scale during training. These two fixes lead to our proposed normalization, which is fully viewer-centric. We define it as

qi =Q20(D¯i) ,

M6DoF+1, viewer(D,f,E,i,j) = R Ei−1Ej,

1 qi

,f ,

where in M6DoF+1, viewer, the scale qi applied to the cameras is dependent only on the depth map in the input view D¯i, different from M6DoF+1, agg. where the scale q computed by aggregating over all Di. At inference the value of qi can be chosen heuristically without compromising performance. Correcting for the scale ambiguities in this way improves metrics, which we show in Section 4.

- 3.4. Improving Diversity with SDS Anchoring

Diffusion models trained with the improved camera conditioning representation M6DoF+1, viewer achieve superior

NVS on DTU LPIPS ↓ PSNR ↑ SSIM ↑ All datasets 0.421 12.2 0.444

- -ACID 0.446 11.5 0.405
- -CO3D 0.456 10.7 0.407
- -RealEstate10K 0.435 12.0 0.429

Table 3. Ablation study on training data. Training on all datasets improves performance.

view synthesis results via 3D SDS distillation. However, for large viewpoint changes, novel view synthesis is also a generation problem, and it may be desirable to generate diverse and plausible contents rather than contents that are only optimal on average for metrics such as PSNR, SSIM, and LPIPS. However, Poole et al. [27] noted that even when the underlying generative model produces diverse images, SDS distillation of that model tends to seek a single mode. For novel view synthesis of scenes via SDS, we observe a unique manifestation of this diversity issue: lack of diversity is especially apparent in inferred backgrounds. Often, SDS distillation predicts a gray or monotone background for regions not observed by the input camera.

To remedy this, we propose “SDS anchoring” (Figure 4). With SDS anchoring, we first directly sample k novel views Xˆk = {Xˆj}kj=1 with Xˆj ∼ p(Xj|Xi,M(D,f,E,i,j)) from poses evenly spaced in azimuth for maximum scene coverage. We sample the novel views via DDIM [36], which does not have the mode collapse issues of SDS. Each novel view is generated conditional on the input view. Then, when optimizing the SDS objective, we condition the diffusion model not on the input view, but on the nearest view. As shown quantitatively in a user study in Section 4 and qualitatively in Figure 9, SDS anchoring produces more diverse background contents. We provide more details about the setup of SDS anchoring in the supplementary.

### 4. Experiments

#### 4.1. Setup

Datasets. Our models are trained on a mixture dataset consisting of CO3D [31], ACID [17], and RealEstate10K [45]. Each example is sampled uniformly at random from the three datasets. We train at 256 × 256 resolution, centercropping and adjusting the intrinsics for each image and scene as necessary. We train using our representation M6DoF+1, viewer unless otherwise specified. We provide more training details in the supplementary.

We evaluate our trained diffusion models on held-out subsets of CO3D, ACID, and RealEstate10K respectively, for 2D novel view synthesis. Our main evaluations are for zero-shot 3D consistent novel view synthesis, where we compare against other techniques on the DTU benchmark [1] and on the Mip-NeRF 360 dataset [2]. We evaluate at 256 × 256 resolution except for DTU, for which we use

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Input view GT novel view ZeroNVS (ours) PixelNeRF

PSNR=10.8, SSIM=0.22 PSNR=12.2, SSIM=0.30 Figure 7. Limitations of PSNR and SSIM for view synthesis evaluation. Misalignments can lead to worse PSNR and SSIM values for predictions that are more semantically sensible.

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

GT novel view Zero-1-to-3 NerDi ZeroNVS (ours) Figure 8. Qualitative comparison between baseline methods and our method.

400 × 300 resolution to be comparable to prior art.

Implementation details. Our diffusion model training code is written in PyTorch and based on the public code for Zero-1-to-3 [19]. We initialize from the pretrained Zero-1to-3-XL, swapping out the conditioning module to accommodate our novel parameterizations. Our distillation code is implemented in Threestudio [14]. We use a custom NeRF network combining features of Mip-NeRF 360 with InstantNGP [23]. The noise schedule is annealed following Wang et al. [39]. For details please see the supplementary.

#### 4.2. Main Results

We evaluate all methods using the standard set of novel view synthesis metrics: PSNR, SSIM, and LPIPS. We weigh LPIPS more heavily in the comparison due to the wellknown issues with PSNR and SSIM as discussed in [6, 10]. We confirm that PSNR and SSIM do not correlate well with performance in our setting, as illustrated in Figure 7.

The results are shown in Table 1. We first compare against baseline methods DS-NeRF [11], PixelNeRF [44],

SinNeRF [42], DietNeRF [15], and NeRDi [10] on DTU. All these methods are trained on DTU, but we achieve a state-of-the-art LPIPS despite being fully zero-shot. We show visual comparisons in Figure 8.

DTU scenes are limited to relatively simple forwardfacing scenes. Therefore, we introduce a more challenging benchmark dataset, the Mip-NeRF 360 dataset, to benchmark the task of 360-degree view synthesis from a single image. We use this benchmark as a zero-shot benchmark, and train three baseline models on our mixture dataset to compare zero-shot performance. Our method is the best on LPIPS for this dataset. On DTU, we exceed Zero-1-to-3 and the zero-shot PixelNeRF model on all metrics, not just LPIPS. Performance is shown in Table 2. All numbers for our method and Zero-1-to-3 are for NeRFs predicted from SDS distillation unless otherwise noted.

Limited diversity is a known issue with SDS-based methods, but the long run time of SDS-based methods makes typical generation-based metrics such as FID costprohibitive. Therefore, we quantify the improved diversity

[Figure 79]

[Figure 80]

Seed=0Seed=1Seed=2

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Novel views (standard SDS) Novel views (SDS anchoring)

Figure 9. Whereas standard SDS (left) tends to predict monotonous backgrounds, our SDS anchoring (right) generates more diverse background contents. Additionally, SDS anchoring generates noticeably different results depending on the seed.

2D novel view synthesis 3D NeRF distillation CO3D RealEstate10K ACID DTU

Conditioning PSNR SSIM LPIPS PSNR SSIM LPIPS PSNR SSIM LPIPS PSNR SSIM LPIPS MZero−1−to−3 12.0 .366 .590 11.7 .338 .534 15.5 .371 .431 10.3 .384 .477 M6DoF+1 12.2 .370 .575 12.5 .380 .483 15.2 .363 .445 9.5 .347 .472 M6DoF+1, norm. 12.9 .392 .542 12.9 .408 .450 16.5 .398 .398 11.5 .422 .421 M6DoF+1, agg. 13.2 .402 .527 13.5 .441 .417 16.9 .411 .378 12.2 .436 .420 M6DoF+1, viewer 13.4 .407 .515 13.5 .440 .414 17.1 .415 .368 12.2 .444 .421 Table 4. Ablation study on the conditioning representation M. Our M6DoF+1, viewer matches or outperforms other representations.

from SDS anchoring via a user study of 21 users on the MipNeRF 360 dataset. Users were asked to compare scenes predicted with and without SDS anchoring along three dimensions: Realism, Creativity, and Overall Preference. The preferences for SDS anchoring were: Realism (78%), Creativity (82%), and Overall Preference (80%). The supplementary provides more details about the setup of the study. Figure 9 includes qualitative examples that show the advantages of SDS anchoring, and the supplementary webpage contains the videos which were shown in the study.

We conduct multiple ablations to verify our contributions. We verify the benefits of each of our multiple multiview scene datasets in Table 3. Removing any of the three datasets on which ZeroNVS is trained reduces performance. In Table 4, we analyze the diffusion model’s performance on held-out subsets of our datasets, with the various parameterizations discussed in Section 3. We see that as the conditioning parameterization is further refined, the performance

continues to increase. Due to computational constraints, we train the ablation diffusion models for fewer steps than our main model, hence the slightly worse performance relative to Table 1. We provide more details in the supplementary.

### 5. Conclusion

We have introduced ZeroNVS, a system for 3D-consistent novel view synthesis from a single image for generic scenes. We showed its state-of-the-art performance on existing NVS benchmarks and proposed the Mip-NeRF 360 dataset as a more challenging benchmark for single-image NVS.

Acknowledgments. The work is in part supported by NSF CCRI #2120095, RI #2211258, ONR MURI N0001422-1-2740, and Google.

### References

- [1] Henrik Aanæs, Rasmus Ramsbøl Jensen, George Vogiatzis, Engin Tola, and Anders Bjorholm Dahl. Large-scale data for multiple-view stereopsis. International Journal of Computer Vision, pages 1–16, 2016. 6
- [2] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-NeRF 360: Unbounded anti-aliased neural radiance fields. In CVPR, 2022. 3, 6, 12
- [3] Thomas Breuel. Webdataset library, 2020. 11
- [4] Eric Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, and Gordon Wetzstein. pi-GAN: Periodic implicit generative adversarial networks for 3D-aware image synthesis. In CVPR,

2021. 3

- [5] Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas Guibas, Jonathan Tremblay, Sameh Khamis, Tero Karras, and Gordon Wetzstein. Efficient geometry-aware 3D generative adversarial networks. In CVPR, 2021. 3
- [6] Eric R. Chan, Koki Nagano, Matthew A. Chan, Alexander W. Bergman, Jeong Joon Park, Axel Levy, Miika Aittala, Shalini De Mello, Tero Karras, and Gordon Wetzstein. GeNVS: Generative novel view synthesis with 3D-aware diffusion models. In ICCV, 2023. 3, 7
- [7] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3D: Disentangling geometry and appearance for highquality text-to-3D content creation. In ICCV, 2023. 3
- [8] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3D objects. arXiv preprint arXiv:2212.08051, 2022. 4
- [9] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, and Ali Farhadi. ObjaverseXL: A universe of 10M+ 3D objects. arXiv preprint arXiv:2307.05663, 2023. 1, 4
- [10] Congyue Deng, Chiyu Jiang, Charles R Qi, Xinchen Yan, Yin Zhou, Leonidas Guibas, Dragomir Anguelov, et al. NeRDi: Single-view NeRF synthesis with language-guided diffusion as general image priors. In CVPR, 2022. 3, 7
- [11] Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ramanan. Depth-supervised NeRF: Fewer views and faster training for free. In CVPR, 2022. 7
- [12] Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen, Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic, and Sanja Fidler. GET3D: A generative model of high quality 3D textured shapes learned from images. In NeurIPS, 2022. 3
- [13] Jiatao Gu, Lingjie Liu, Peng Wang, and Christian Theobalt. StyleNeRF: A Style-based 3D-aware Generator for Highresolution Image Synthesis. In ICLR, 2022. 3
- [14] Yuan-Chen Guo, Ying-Tian Liu, Ruizhi Shao, Christian Laforte, Vikram Voleti, Guan Luo, Chia-Hao Chen, Zi-Xin Zou, Chen Wang, Yan-Pei Cao, and Song-Hai Zhang. threestudio: A unified framework for 3D content generation,

2023. 7

- [15] Ajay Jain, Matthew Tancik, and Pieter Abbeel. Putting NeRF on a diet: Semantically consistent few-shot view synthesis. In ICCV, 2021. 3, 7
- [16] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3D: High-resolution text-to-3D content creation. In CVPR, 2023. 3
- [17] Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Makadia, Noah Snavely, and Angjoo Kanazawa. Infinite nature: Perpetual view generation of natural scenes from a single image. In ICCV, 2021. 1, 6
- [18] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, and Hao Su. One-2-3-45: Any single image to 3D mesh in 45 seconds without pershape optimization. arXiv preprint arXiv:2306.16928, 2023. 3
- [19] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3D object. In CVPR, 2023. 1, 3, 4, 7, 11
- [20] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. SyncDreamer: Learning to generate multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453, 2023. 3
- [21] Jonathan Lorraine, Kevin Xie, Xiaohui Zeng, Chen-Hsuan Lin, Towaki Takikawa, Nicholas Sharp, Tsung-Yi Lin, MingYu Liu, Sanja Fidler, and James Lucas. ATT3D: Amortized text-to-3D object synthesis. In ICCV, 2023. 3
- [22] Luke Melas-Kyriazi, Christian Rupprecht, Iro Laina, and Andrea Vedaldi. RealFusion: 360° reconstruction of any object from a single image. In CVPR, 2023. 1, 3
- [23] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph., 41(4):102:1– 102:15, 2022. 7
- [24] Ra´ul Mur-Artal, J. M. M. Montiel, and Juan D. Tard´os. ORBSLAM: A versatile and accurate monocular SLAM system. IEEE Transactions on Robotics, 31(5):1147–1163, 2015. 1
- [25] Thu Nguyen-Phuoc, Chuan Li, Lucas Theis, Christian Richardt, and Yong-Liang Yang. HoloGAN: Unsupervised learning of 3D representations from natural images. In ICCV,

2019. 3

- [26] Michael Niemeyer and Andreas Geiger. GIRAFFE: Representing scenes as compositional generative neural feature fields. In CVPR, 2021. 3
- [27] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. DreamFusion: Text-to-3D using 2D diffusion. In ICLR,

2022. 1, 3, 6

- [28] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, and Bernard Ghanem. Magic123: One image to high-quality 3D object generation using both 2D and 3D diffusion priors. arXiv preprint arXiv:2306.17843, 2023. 1, 3, 11
- [29] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In ICCV, 2021. 6, 11

- [30] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 44(3), 2022. 5, 11
- [31] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3D: Large-scale learning and evaluation of real-life 3D category reconstruction. In ICCV, 2021. 1, 6
- [32] Kyle Sargent, Jing Yu Koh, Han Zhang, Huiwen Chang, Charles Herrmann, Pratul Srinivasan, Jiajun Wu, and Deqing Sun. VQ3D: Learning a 3D-aware generative model on ImageNet. In ICCV, 2023. 3
- [33] Johannes Lutz Sch¨onberger and Jan-Michael Frahm. Structure-from-motion revisited. In CVPR, 2016. 1
- [34] Ivan Skorokhodov, Sergey Tulyakov, Yiqun Wang, and Peter Wonka. EpiGRAF: Rethinking training of 3D GANs. In NeurIPS, 2022. 3
- [35] Ivan Skorokhodov, Aliaksandr Siarohin, Yinghao Xu, Jian Ren, Hsin-Ying Lee, Peter Wonka, and Sergey Tulyakov. 3D generation on ImageNet. In ICLR, 2023. 3
- [36] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv::2010.02502, 2020. 5, 6
- [37] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 1
- [38] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A. Yeh, and Greg Shakhnarovich. Score Jacobian chaining: Lifting pretrained 2D diffusion models for 3D generation. arXiv preprint arXiv:2212.00774, 2022. 3
- [39] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. ProlificDreamer: High-fidelity and diverse text-to-3D generation with variational score distillation. arXiv preprint arXiv:2305.16213, 2023. 3, 7
- [40] Daniel Watson, William Chan, Ricardo Martin-Brualla, Jonathan Ho, Andrea Tagliasacchi, and Mohammad Norouzi. Novel view synthesis with diffusion models. In ICLR, 2023. 4
- [41] Jianfeng Xiang, Jiaolong Yang, Binbin Huang, and Xin Tong. 3D-aware image generation using 2D diffusion models. In ICCV, 2023. 3
- [42] Dejia Xu, Yifan Jiang, Peihao Wang, Zhiwen Fan, Humphrey Shi, and Zhangyang Wang. SinNeRF: Training neural radiance fields on complex scenes from a single image. In ECCV,

2022. 6, 7

- [43] Wei Yin, Jianming Zhang, Oliver Wang, Simon Niklaus, Simon Chen, Yifan Liu, and Chunhua Shen. Towards accurate reconstruction of 3D scene shape from a single monocular image. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2022. 5
- [44] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelNeRF: Neural radiance fields from one or few images. In CVPR, 2021. 3, 7
- [45] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. ACM Trans. Graph. (Proc. SIGGRAPH), 37, 2018. 1, 5, 6

### A. Details: Diffusion model training

#### A.1. Model

We train diffusion models for various camera conditioning parameterizations: MZero−1−to−3, M6DoF+1, M6DoF+1, norm., M6DoF+1, agg., and M6DoF+1, viewer. Our runtime is identical to Zero-1-to-3 [19] as the camera conditioning novelties we introduce add negligible overhead and can be done mainly in the dataloader. We train our main model for 60,000 steps with batch size 1536. We find that performance tends to saturate after about 20,000 steps for all models, though it does not decrease. For inference of the 2D diffusion model, we use 500 DDIM steps and guidance scale 3.0.

Details for M6DoF+1: To embed the field of view f in radians, we use a 3-dimensional vector consisting of [f,sin(f),cos(f)]. When concatenated with the 4 × 4 = 16-dimensional relative pose matrix, this gives a 19dimensional conditioning vector.

Details for M6DoF+1, viewer: We use the DPT-SwinV2256 depth model [29] to infill depth maps from ORB-SLAM and COLMAP on the ACID, RealEstate10K, and CO3D datasets. We infill the invalid depth map regions only after aligning the disparity from the monodepth estimator to the ground-truth sparse depth map via the optimal scale and shift following Ranftl et al. [30]. We downsample the depth map 4× so that the quantile function is evaluated quickly.

At inference time, the value of Q20(D¯) may not be known since input depth map D is unknown. Therefore there is a question of how to compute the conditioning embedding at inference time. Values of Q20(D¯) between .7 − 1. work for most images and it can be chosen heuristically. For instance, for DTU we uniformly assume a value of .7, which seems to work well. Note that any value of Q20(D¯) is presumably possible; it is only when this value is incompatible with the desired SDS camera radius that distillation may fail, since the cameras may intersect the visible content.

#### A.2. Dataloader

One significant engineering component of our work is our design of a streaming dataloader for multiview data, built on top of WebDataset [3]. Each dataset is sharded and each shard consists of a sequential tar archive of scenes. The shards can be streamed in parallel via multiprocessing. As a shard is streamed, we yield random pairs of views from scenes according to a “rate” parameter that determines how densely to sample each scene. This parameter allows a trade-off between fully random sampling (lower rate) and biased sampling (higher rate) which can be tuned according to the available network bandwidth. Individual streams

from each dataset are then combined and sampled randomly to yield the mixture dataset. We will release the code together with our main code release.

### B. Details: NeRF prediction and distillation

#### B.1. SDS Anchoring

We propose SDS anchoring in order to increase the diversity of synthesized scenes. We sample 2 anchors at 120 and 240 degrees of azimuth relative to the input camera.

One potential issue with SDS anchoring is that if the samples are 3D-inconsistent, the resulting generations may look unusual. Furthermore, traditional SDS already performs quite well except if the criterion is diverse backgrounds. Therefore, to implement anchoring, we randomly choose with probability .5 either the input camera and view or the nearest sampled anchor camera and view as guidance. If the guidance is an anchor, we ”gate” the gradients flowing back from SDS according to the depth of the NeRF render, so that only depths above a certain threshold (1.0 in our experiments) receive guidance from the anchors. This seems to mostly mitigate artifacts from 3D-inconsistency of foreground content, while still allowing for rich backgrounds. We show video results for SDS anchoring on the webpage.

#### B.2. Hyperparameters

NeRF distillation via involves numerous hyperparameters such as for controlling lighting, shading, camera sampling, number of training steps, training at progressively increasing resolutions, loss weights, density blob initializations, optimizers, guidance weight, and more. We will share a few insights about choosing hyperparameters for scenes here, and release the full configs as part of our code release.

Noise scheduling: We found that ending training with very low maximum noise levels such as .025 seemed to benefit results, particularly perceptual metrics like LPIPS. We additionaly found a significant benefit on 360-degree scenes such as in the Mip-NeRF 360 dataset to scheduling the noise ”anisotropically;” that is, reducing the noise level more slowly on the opposite end from the input view. This seems to give the optimization more time to solve the challenging 180-degree views at higher noise levels before refining the predictions at low noise levels.

Miscellaneous: Progressive azimuth and elevation sampling following [28] was also found to be very important for training stability. Training resolution progresses stagewise, first with batch size 6 at 128x128 and then with batch size 1 at 256 × 256.

### C. Experimental setups

For our main results on DTU and Mip-NeRF 360, we train our model and Zero-1-to-3 for 60,000 steps. Performance for our method seems to saturate earlier than for Zero-1-to3, which trained for about 100,000 steps; this may be due to the larger dataset size. Objaverse, with 800,000 scenes, is much larger than the combination of RealEstate10K, ACID, and CO3D, which are only about 95,000 scenes in total.

For the retrained PixelNeRF baseline, we retrained it on our mixture dataset of CO3D, ACID, and RealEstate10K for about 560,000 steps.

#### C.1. Main results

For all single-image NeRF distillation results, we assume the camera elevation, field of view, and content scale are given. These parameters are identical for all DTU scenes but vary across the Mip-NeRF 360 dataset. For DTU, we use the standard input views and test split from from prior work. We select Mip-NeRF 360 input view indices manually based on two criteria. First, the views are wellapproximated by a 3DoF pose representation in the sense of geodesic distance between rotations. This is to ensure fair comparison with Zero-1-to-3, and for compatibility with Threestudio’s SDS sampling scheme, which also uses 3 degrees of freedom. Second, as much of the scene content as possible must be visible in the view. The exact values of the input view indices are given in Table 5.

The field of view is obtained via COLMAP. The camera elevation is set automatically via computing the angle between the forward axis of the camera and the world’s XY plane, after the cameras have been standardized via PCA following Barron et al. [2].

One challenge is that for both the Mip-NeRF 360 and DTU datasets, the scene scales are not known by the zero-shot methods, namely Zero-1-to-3, our method, and our retrained PixelNeRF. Therefore, for the zeroshot methods, we manually grid search for the optimal world scale in intervals of .1 to find the appropriate world scale for each scene in order to align the predictions to the generated scenes. Between five to nine samples within [.3,.4,.5,.6,.7,.8,.9,1.,1.1,1.2,1.3,1.4,1.5] generally suffices to find the appropriate scale. Even correcting for the scale misalignment issue in this way, the zero-shot methods generally do worse on pixel-aligned metrics like SSIM and PSNR compared with methods that have been fine-tuned on DTU.

#### C.2. User study

We conduct a user study on the seven Mip-NeRF 360 scenes, comparing our method with and without SDS anchoring. We received 21 respondents. For each scene, respondents were shown 360-degree novel view videos of the

Scene name Input view index Content scale

bicycle 98 .9 bonsai 204 .9 counter 95 .9 garden 63 .9 kitchen 65 .9

room 151 2. stump 34 .9

Table 5. Setup for the Mip-NeRF 360 dataset

.

scene inferred both with and without SDS anchoring. The videos were shown in a random order and respondents were unaware which video corresponded to the use of SDS anchoring. Respondents were asked:

- 1. Which scene seems more realistic?
- 2. Which scene seems more creative?
- 3. Which scene do you prefer? Respondents generally preferred the scenes produced by

SDS anchoring, especially with respect to “Which scene seems more creative?”

#### C.3. Ablation studies

We perform ablation studies on dataset selection and camera representations. For 2D novel view synthesis metrics, we compute metrics on a held-out subset of scenes from the respective datasets, randomly sampling pairs of input and target novel views from each scene. For 3D SDS distillation and novel view synthesis, our settings are identical to the NeRF distillation settings for our main results except that we use shorter-trained diffusion models. We train them for 25,000 steps as opposed to 60,000 steps for computational constraint reasons.

