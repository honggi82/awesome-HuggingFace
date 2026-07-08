arXiv:2506.18792v1[cs.CV]23Jun2025

# ViDAR: Video Diffusion-Aware 4D Reconstruction From Monocular Inputs

Michal Nazarczuk1∗ Sibi Catley-Chandar1,2∗ Thomas Tanay1 Zhensong Zhang1 Gregory Slabaugh2 Eduardo Pérez-Pellitero1

1 Huawei Noah’s Ark Lab 2 Queen Mary University of London

[Figure 1]

InputMoScaViDAR

Novel View Synthesis

[Figure 2]

MoSca ViDAR

[Figure 3]

- Figure 1: ViDAR provides a novel framework for Monocular Novel View Synthesis utilising a diffusion-aware reconstruction framework.

### Abstract

Dynamic Novel View Synthesis aims to generate photorealistic views of moving subjects from arbitrary viewpoints. This task is particularly challenging when relying on monocular video, where disentangling structure from motion is ill-posed and supervision is scarce. We introduce Video Diffusion-Aware Reconstruction (ViDAR), a novel 4D reconstruction framework that leverages personalised diffusion models to synthesise a pseudo multi-view supervision signal for training a Gaussian splatting representation. By conditioning on scene-specific features, ViDAR recovers fine-grained appearance details while mitigating artefacts introduced by monocular ambiguity. To address the spatio-temporal inconsistency of diffusion-based supervision, we propose a diffusion-aware loss function and a camera pose optimisation strategy that aligns synthetic views with the underlying scene geometry. Experiments on DyCheck, a challenging benchmark with extreme viewpoint variation, show that ViDAR outperforms all state-of-the-art baselines in visual quality and geometric consistency. We further highlight ViDAR’s strong improvement over baselines on dynamic regions and provide a new benchmark to compare performance in reconstructing motion-rich parts of the scene. Project page: https://vidar-4d.github.io/.

∗equal contribution

Preprint. Under review.

### 1 Introduction

4D reconstruction from monocular inputs is a challenging problem where the goal is to recover a 3D representation of a dynamic scene. It is increasingly important for modelling, comprehending, and interacting with the physical world and supports a wide range of downstream applications, ranging from augmented reality to generating data for training robust AI models [54].

Casually captured monocular videos are ubiquitous, however reconstructing 3D structure from them remains an inherently ill-posed problem. Static regions of the scene can typically be reconstructed well due to effective multi-view capture [4]. However for dynamic regions, depth information is not directly observable from a single viewpoint; in other words, it is difficult to disentangle the motion of the camera from motion within the scene. To mitigate this ambiguity many existing approaches impose strong regularisation [6; 28; 55] in the form of geometric assumptions, such as the object’s rigidity, that constrains the dynamics of the scene. Others [11; 23; 44; 62; 63] leverage learned priors, particularly those derived from large-scale models (e.g. monocular depth), to guide the reconstruction. While regularization based methods [11; 44] achieve geometrically compact scene representations, they often fall short in rendering high-quality, photorealistic appearances. Conversely recent generative approaches utilise powerful diffusion models to achieve higher visual quality in tasks such as single image to 3D [19; 21; 22; 50; 51; 64] and monocular reconstruction [48] but struggle to maintain spatio-temporal coherence, limiting their applicability in scenarios that demand accurate spatial reconstruction and temporal consistency, particularly in dynamic, real-world settings.

To tackle these challenges, we present Video Diffusion-Aware 4D Reconstruction (ViDAR), a monocular video reconstruction approach that leverages diffusion models as powerful appearance priors through a novel diffusion-aware reconstruction framework, which allows for improving visual fidelity without the loss of spatio-temporal consistency. We first train a monocular reconstruction baseline and generate a set of typically degraded multi-view images by sampling diverse camera poses and rendering the novel viewpoints. We then adopt a DreamBooth-style personalisation strategy [35], and tailor a pretrained diffusion model to the input video, which we use as a generative enhancer to inject rich visual information back into the degraded renders. This effectively generates a set of high-fidelity pseudo-multi-view observations for our scene, although due to the nature of the diffusion process, the resulting images are not necessarily spatially consistent. We observe that naively using these views as supervision leads to reconstructions degraded by artefacts and geometric inconsistencies. To mitigate this, we propose a method of diffusion-aware reconstruction, which selectively applies diffusion-based guidance to dynamic regions of the scene while jointly optimising the camera poses associated with the diffused views.

To the best of our knowledge, ViDAR is the first approach to incorporate a diffusion prior into monocular video reconstruction in a geometrically consistent manner. We demonstrate substantially improved qualitative and quantitative results compared to existing techniques (see Tabs.1, 2, Figs.1, 4), highlighting the effectiveness of diffusion-guided supervision when integrated with a reconstruction pipeline that accounts for geometric consistency.

We summarise our contributions as follows:

- 1. A personalised diffusion enhancement strategy that improves appearance quality by refining newly sampled renderings using a DreamBooth-adapted model.
- 2. A diffusion-aware reconstruction framework that combines dynamic-region-focused diffusion guidance with joint optimisation of the sampled camera poses for geometrically consistent reconstruction.
- 3. An extensive experimental evaluation, including both quantitative and qualitative comparisons with prior work, the introduction of a dynamic-region specific benchmark, as well as ablation studies isolating the impact of each component.

### 2 Related work

4D reconstruction Advances in novel view synthesis include the introduction of two seminal reconstruction paradigms, namely Neural Radiance Fields (NeRF)[27] and 3D Gaussian Splatting (3DGS)[9]. These developments in static scene reconstruction were quickly followed by several works on dynamic content. NeRF-based methods for video reconstruction include D-NeRF[33], StreamRF[13], HexPlane[2], K-Planes[3], Tensor4D[36], MixVoxels [43]. Similarly, Gaussian

###### Diffusion-Aware Reconstruction

Monocular Video

Personalised Diffusion Model

Sampled camera pose optimisation

[Figure 4]

Monocular 4D Reconstruction

Novel View Enhancement

[Figure 5]

Dynamic reconstruction optimisation

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Novel camera sampling

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

- Figure 2: A high-level overview of ViDAR. The input video is used to create a 4D reconstruction with a monocular approach. Further, novel camera views are sampled and enhanced with a personalised diffusion model for each scene. This constitutes a set of pseudo-multi-view supervision examples. Finally, our approach optimises the 4D representation with the use of original video and new multiview cues, in a diffusion-aware manner.

Splatting developments enabled research on dynamic novel view synthesis. Multi-view videos were reconstructed by: GaussianFlow[20], 4DGS[47], STG[16], SWinGS[37], Ex4DGS[10].

Monocular reconstruction The task of 4D monocular video reconstruction can be seen as a special case of 4D reconstruction under substantially more challenging conditions. This is due to the problem often being ill-posed: many of the target object surfaces may be seen only from one viewpoint throughout the video. Notably, among NeRF-based approaches, NSFF[14] proposes a time varying flow field, whereas Nerfies[29], HyperNeRF[30], DyCheck (T-NeRF)[4], DyBluRF[1], RoDynRF[24], CTNeRF[26] use a canonical representation with a time-dependent deformation. DynIBaR[15] uses Image Based Rendering for reconstruction. With Gaussian Splatting advancements, Dynamic 3D Gaussians[25] learn explicit motion of every Gaussian, whereas 4DGS[47], Deformable 3DGS[55], SC-GS [6] use a deformation field for transformation from canonical space. SplineGS [28] constrains the motion of Gaussians to splines to ensure temporal smoothness. DynPoint [62] and MotionGS[63] use an optical flow estimator for additional supervision. PGDVS[61] and BTimer[17] propose a transformer-based approach for generalisable reconstruction. Dynamic Gaussian Marbles [40] adopt a divide-and-conquer strategy to merge sets of Gaussians and create long trajectories, and restrict representation to isotropic Gaussians. MoDGS[23] improves the supervision from depth priors. D-NPC[7] proposes the use of neural implicit point cloud as the representation for monocular reconstruction. MoSca[11] and Shape of Motion[44] both utilise priors from pretrained foundational models (depth, optical flow, 2D tracking). Similarly, they both reconstruct static and dynamic content separately, and describe the motion of the Gaussians with lower dimensionality basis functions.

Diffusion enhanced reconstruction Several recent approaches explore the use of diffusion models to guide the reconstruction. ReconFusion[49] trains a diffusion model on a set of object images, and uses it to score the quality of sparse reconstruction, guiding it with RGB loss. DpDy[42] uses Score Distillation Sampling (SDS)[32] to supervise reconstruction with the use of image and depth diffusion model. CAT4D[48], concurrent to our work, uses a video-diffusion model to generate additional static cameras for the input video, followed by the reconstruction process. MVGD[5] proposes a direct rendering of novel views and depth as a conditional generative task. Other diffusion-based approaches include text or a single image to 3D generation [12; 18; 19; 21; 22; 38; 41; 46; 50; 51; 52; 56; 58; 59; 64]. Notably, our approach uses a monocular video as an input, and uses a personalised diffusion model along with our diffusion-aware reconstruction for accurate geometry modelling.

- 3 Method

Our method incorporates several stages which can be seen in Figure2. Firstly, we use a monocular reconstruction baseline to obtain a 4D representation of the scene (Sec.3.1), and generate a set of

degraded multi-view renders from sampled novel camera poses. Next, we personalise a diffusion model using the input video (Sec.3.2), which is used to enhance the degraded renders (Sec.3.2.1). Finally, we use the new set of enhanced pseudo-multi-view images to supervise and refine the 4D representation of the scene (Sec.3.3), in a diffusion-aware manner.

#### 3.1 Monocular Reconstruction

Given a casual monocular video of a dynamic scene with T frames I = [I1,I2,...IT], we perform initial reconstruction of the scene using an off-the-shelf 4D monocular reconstruction method, specifically, we use MoSca [11] in our implementation. The method reconstructs two sets of Gaussians for the given scene, namely static Gaussians Gs, and dynamic Gaussians Gd, that together create the scene representation: G = Gd ∪ Gs. MoSca leverages several priors in the reconstruction process: depth, optical flow, 2D tracking. Firstly, the optical flow is used to estimate the epipolar error and to determine the likelihood of image regions belonging to dynamic or static content. This is followed by the joint reconstruction of the static part of the scene Gs and fine-tuning of the input camera pose cinp. With that, a scaffold, a low dimensionality motion representation, is built through lifting 2D tracklets belonging to dynamic regions into 3D using depth information. Finally, a photometric reconstruction is performed to optimise the scene G, enabling rendering of novel views R of the scene.

- 3.1.1 Track Anything Gaussian Classification

We note that the epipolar error analysis introduced by MoSca for classification of dynamic parts of the image leads to occurrences of floater artefacts due to the inclusion of background among dynamic Gaussians. This may not be reflected heavily in quantitative performance, but leads to a decrease in the quality of generated pseudo-multi-view samples (Sec.3.2.1). To improve the constraint on dynamic Gaussians’ locations, we use dynamic masks Dt obtained from Track Anything[53] to reconstruct the static part of the scene Gs and generate motion scaffolds (as in MoSca[11]).

3.2 Diffusion Enhancement

We utilise a Stable Diffusion[34] model, specifically the pretrained Stable Diffusion XL (SDXL) [31] to improve the quality of rendered images and guide the reconstruction process. Following the observations of ReconFusion[49], we decide to use a multistep denoising process, in contrast to Score Distillation Sampling[32]. Conversely, given a sampled image Rm,t from camera cm at the time t, we follow a standard text-to-image[34] process and encode the image into latent space: x0 = E (Rm,t). Further, instead of generating the noisy latent for image generation, we introduce k steps of noise into the image-sourced latent x0 → xk using the original noise scheduler (here, Discrete Euler[8]). We then follow the denoising process for k steps to achieve a denoised latent xˆ0 which is then decoded to an enhanced version of the input image: Em,t = D (ˆx0).

Personalisation Similarly to some of the recent reconstruction approaches, e.g. Wang et al. [42], we apply the Dreambooth [35] fine-tuning approach to the SDXL model. To this end, we treat an input video I as a collection of images and fine-tune the diffusion model for a given scene such that a specific text token triggers the model to follow the appearance of the scene.

- 3.2.1 View Sampling and Rendering Enhancement

Given the scene-personalised diffusion model, we utilise the previously trained monocular reconstruction to generate a set of pseudo-multi-view ground truth images. Firstly, we sample M sets of images for each timestep t ∈ [0,T], effectively adding M new cameras with parameters cm where m ∈ [0,M] and cm ∈ Csample, where Csample constitutes a set of new camera trajectories. To this end, we select two existing views (as a camera position and rotation), a random one, and a challenging view (with the furthest distance from the mean) and sample a new view as their weighted linear combination. To introduce variety in the difficulty of the sampled views, we gradually increase the blending weight of the views towards the most challenging ones from the input trajectory. Simultaneously, we introduce noise of an increasing amplitude to the new cameras.

Thereafter, we use our trained monocular reconstruction to render a set of M new camera views Rm,t for each timestep. Further, we use our personalised diffusion model to enhance the rendered images

Rm,t → Em,t. This constitutes a new set of supervision images in a multi-view setting. We have chosen to generate a whole multi-view dataset in a single step instead of performing the enhancement on-the-fly. This enables the samples to be reused and reduces the computational demands (especially on GPU memory).

#### 3.3 Diffusion-Aware Reconstruction

We use our generated dataset {Em,t} as additional supervision to re-train our 4D monocular reconstruction method to predict a higher quality output Iˆm,t. However using these sampled views for training is challenging. The outputs Em,t of our personalised video diffusion models are high-fidelity and also preserve structure and coarse geometry, but due to the nature of the diffusion process and random noise schedule, they are not spatio-temporally consistent at the level of fine-grained detail and texture. This manifests as flickering and shifts of textures between consecutive frames. In some cases, coarse geometry may also be hallucinated, e.g. in novel viewpoints not seen during training. If we naively used these outputs to supervise monocular 4D reconstruction, the lack of spatio-temporal consistency in the training data would cause the model to either converge to a mean radiance value and cause blurry renderings, or to overfit to individual frames and learn a temporally inconsistent reconstruction. We propose the following mechanisms to overcome these challenges.

- 3.3.1 Dynamic Reconstruction

While dynamic regions of a scene are under-observed, static regions may be captured from multiple viewpoints across time, effectively creating multi-view supervision. Hence supervision is unnecessary in static regions and in fact could cause the quality to reduce, particularly if spatially inconsistent. We compute a mask of the dynamic regions of the scene Dm,t using Track Anything [53], and apply this mask to our data to mask out the static regions Em,tdyn = Em,t ⊙ Dm,t, where ⊙ denotes element-wise multiplication, and also to our predicted output Iˆm,tdyn = Iˆm,t ⊙ Dm,t. This ensures that only the dynamic regions of the scene are supervised by our generated data, which reduces the convergence to the mean effect in the static reconstruction and reduces floaters. For dynamic regions, we introduce a perceptual loss [60] to encourage our reconstruction to be texturally rich and reduce blur caused by training on spatially misaligned psuedo-GTs. During training we compute the loss as Ldyn = |Em,tdyn − Iˆm,tdyn|1 + λp|Em,tdyn − Iˆm,tdyn|vgg + λs|Em,tdyn − Iˆm,tdyn|ssim, where |·|1 is the L1 loss, |·|vgg is the perceptual loss using a pretrained VGG network [39], |·|ssim is the SSIM [45] loss and λp and λs are hyperparameters set to 0.1. The dynamic loss, Ldyn, is applied in addition to the default losses from the monocular reconstruction method, and is backpropagated to update G and Cinp.

- 3.3.2 Sampled Camera Pose Optimisation

Camera poses of casually captured monocular videos are typically noisy due to the difficulty of disentangling scene motion from camera motion, thus the need to optimise Cinp in many monocular reconstruction methods [11; 44]. Our sampled camera poses Csample are interpolated from Cinp and so are also noisy. As our psuedo-GTs corresponding to Csample are not always spatially consistent, it is even more difficult to disentangle scene motion from camera motion. To compensate for this, it is necessary to optimise our sampled camera poses during training to ensure the psuedo-GTs are aligned with the underlying scene geometry. However unlike dynamic reconstruction (Sec. 3.3.1) where we only use the dynamic masked region for supervision, we use the entire image Et,m as supervision for sampled camera pose optimisation. Despite fine-grained textural flickering, the coarse structure present in static regions provides a more consistent supervision signal for localisation than using only dynamic regions. We compute the loss as Lcam = |Em,t −Iˆm,t|1 +λp|Em,t −Iˆm,t|vgg + λs|Em,t − Iˆm,t|ssim. The camera loss, Lcam, is backpropagated separately to other losses and only updates Csample.

4 Results

- 4.1 Datasets

We evaluate the performance of ViDAR on the DyCheck dataset [4]. DyCheck was introduced as a real world benchmark for evaluating monocular to 4D methods and is extremely challenging: the

- Table 1: Quantitative results on co-visibility masked regions of scenes from the DyCheck (iPhone) dataset. Best, second and third results are highlighted in red, orange and yellow respectively. SoM-5 is full-res with wheel and space-out excluded.

Method PSNR-m↑ SSIM-m↑ LPIPS-m↓

T-NeRF[4] 16.96 0.5772 0.3789 NSFF[14] 15.46 0.5510 0.3960 Nerfies[29] 16.45 0.5699 0.3389 HyperNeRF[30] 16.81 0.5693 0.3319 4DGS[47] 13.64 - 0.4280 PGDVS[61] 15.88 0.5480 0.3400 DynPoint[62] 16.89 0.5730 DyBluRF[1] 17.37 0.5910 0.3730 D-NPC[7] 16.41 0.5820 0.3190 RoDynRF[24] 17.10 0.5340 0.5170 Gaussian Marbles[40] 16.02 0.5416 0.3398 SoM[44] 18.62 0.6820 0.2382 MoSca[11] 19.32 0.7060 0.2640 Ours 19.69 0.7126 0.2231

Half-Res

Gaussian Marbles[40] 15.84 0.5434 0.5681 SoM[44] 17.98 0.6422 0.3718 MoSca[11] 18.44 0.6560 0.4193 Ours 19.00 0.6672 0.3623

Full-Res

T-NeRF[4] 15.60 0.5500 0.5500 HyperNeRF[30] 15.99 0.5900 0.5100 DynIBaR[15] 13.41 0.4800 0.5500 Gaussian Marbles[40] 16.03 0.5425 0.5807 SoM[44] 16.72 0.6300 0.4500 CAT4D[48] 17.39 0.6070 0.3410 MoSca[11] 18.34 0.6636 0.4321 Ours 18.76 0.6751 0.3774

SoM-5

test views are far away from training views, camera poses are often inaccurate, depths are noisy and training views have issues such as overexposure and autofocus. The dataset consists of 14 casually captured scenes, 7 of which have no ground truth test views and are used for qualitative evaluation only and 7 with test views available. Due to the difficulty of obtaining accurate camera poses for all scenes, some methods choose to quantitatively evaluate on only 5 of the available 7 scenes and discard ‘space-out’ and ‘wheel’. To our knowledge, this is currently the only widely used benchmark which is appropriate for evaluating our method. As described in DyCheck [4], other datasets such as Nerfies [29] , HyperNeRF [30] and NSFF [14] suffer from teleporting cameras which makes them effectively multi-view. As described in MoSca [11], the NVIDIA dataset [57] is forward-facing with small-baseline static cameras and is significantly easier than DyCheck, thus our contributions which tackle highly ill-posed settings are less useful. We quantitatively and qualitatively evaluate our method and other state of the art baselines across all 14 scenes.

#### 4.2 Metrics

Following previous works [4; 11], we compute PSNR, SSIM and LPIPS on the co-visibility masked regions of the test views, which we denote with an -m addendum to each metric. We compute metrics at both half-resolution and full-resolution, and following [44], we also report results on a subset of 5 scenes which we label SoM-5.

#### 4.2.1 Limitations of Metrics and A New Benchmark

We note that the static regions of a scene are often observed from several viewpoints across different time steps in the captured monocular video. This effectively provides multi-view supervision for these regions, and although we are interested in reconstructing the entire scene which includes the static

- Table 2: Quantitative results on dynamic regions of scenes from the DyCheck (iPhone) dataset. Best, second and third results are highlighted in red, orange and yellow respectively. SoM-5 is full-res with wheel and space-out excluded.

Method PSNR-D↑ SSIM-D↑ LPIPS-D↓

Half-Res

T-NeRF[4] 13.86 0.8546 0.3491 Nerfies[29] 12.89 0.8425 0.3811 HyperNeRF[30] 13.27 0.8484 0.3558 Gaussian Marbles[40] 9.99 0.8175 0.3926 SoM[44] 14.80 0.8582 0.3008 MoSca[11] 15.63 0.8755 0.2904 Ours 16.46 0.8850 0.2793

Full-Res

Gaussian Marbles[40] 12.75 0.8607 0.5058 SoM[44] 14.82 0.8709 0.4347 MoSca[11] 15.39 0.8821 0.4413 Ours 16.32 0.8893 0.3921

SoM-5

Gaussian Marbles[40] 13.66 0.8658 0.4919 SoM[44] 12.50 0.8648 0.4890 MoSca[11] 15.83 0.8872 0.4404 Ours 16.69 0.8941 0.3778

- Table 3: Intersection of co-visibility mask with dynamic regions with respect to co-visibility mask area

[Figure 20]

[Figure 21]

Scene Dyn/Co-vis Intersection (%) apple 4.42 block 27.46 paper-windmill 3.58 space-out 20.63 spin 19.76 teddy 81.33 wheel 24.65 mean 25.97

Co-visibility mask Dynamic mask

Figure 3: An example of co-visibility and dynamic mask comparison.

regions, the dynamic regions are arguably the area of most interest and also the most under-observed. In order to better evaluate performance in the dynamic regions of the scene, we compute a set of dynamic masks for each scene using Track Anything [53]. We compute the intersection between the co-visibility masks and the dynamic regions of the scene and present results in Table 3. We find that on average only 26% of the co-visibility masked pixels correspond to the dynamic region. Some scenes such as apple and paper-windmill have an intersection as low as 4%. We show an example of this in Figure 3. The co-visibility masked metrics are heavily weighted towards the static regions of the scene. Although this is useful for evaluating overall reconstruction performance, it underweights the reconstruction performance of methods in the most difficult dynamic regions. We provide a complementary new benchmark for the evaluation of monocular to 4D reconstruction methods, where our computed dynamic masks can be used in place of the commonly used co-visibility masks. We use these masks to compute the PSNR, SSIM and LPIPS, which we denote with a -D addendum, for a range of baseline methods in Table 2.

#### 4.3 Evaluation

Baselines We compare against a wide range of baselines, including a number of recent state-of-theart methods such as MoSca [11], CAT4D [48], Shape Of Motion [44], Dynamic Gaussian Marbles [40] and 4DGS [47], which are based upon Gaussian Splatting [9]. We also compare against NeRFbased approaches T-NeRF [4], Nerfies [29], HyperNeRF [30], DyBluRF [1] and RoDynRF [24],

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

T-NeRF[4] Gaussian Marbles[40]

Shape Of Motion[44]

MoSca[11] Ours GT

Figure 4: Qualitative evaluation of our method against benchmark methods on the DyCheck test set.

Table 4: Quantitative results of an ablation study of the components of ViDAR. Method PSNR-m SSIM-m LPIPS-m Ours 19.00 0.6672 0.3623 W/o Tracking Based Gaussian Classification (TGS) 18.88 0.6651 0.3693 W/o Sampled Camera Optimisation (SO) 18.39 0.6514 0.4040 W/o Dynamic Reconstruction (DR) 18.93 0.6274 0.4497 W/o SO + DR + TGS 18.46 0.6075 0.4656

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

W/o SO/DR/TGS W/o DR W/o SO W/o TGS Ours GT

Figure 5: Qualitative evaluation of our ablation study with settings corresponding to Tab. 4.

neural point clouds approaches DynPoint [62] and D-NPC [7], generalized pre-trained transformer PGDVS [61], neural scene flow NSFF [14] and volumetric image-based rendering DynIBaR [15].

Quantitative and Qualitative Evaluation We present quantitative results of our method in Tables 1 and 2. Our method outperforms all state-of-the-art baselines in PSNR and SSIM and all but one in LPIPS, across all settings and resolutions. We typically improve PSNR by a large margin, achieving a minimum of 1dB improvement over all methods, except for MoSca where we average 0.94dB and 0.56dB higher in dynamic and co-visibility masked regions respectively. This indicates our method particularly improves dynamic region reconstruction. We note that CAT4D achieves a lower LPIPS score than our method, but the improved perceptual quality comes at the cost of reduced spatiotemporal consistency, which is reflected in the PSNR and SSIM scores, and also clearly shown in our supplementary video. We present a qualitative evaluation in Figure 4 where ViDAR demonstrates consistently superior visual quality and geometric consistency when compared to the best existing approaches. Although 2D image comparisons are indicative of performance, we encourage viewing our supplementary video results to appreciate the improvement in spatio-temporal consistency and visual quality over baselines.

Ablations We quantitatively evaluate each of our contributions in an ablation study presented in Table 4. The bottom row w/o SO + DR + TGS shows a naive approach of using the diffused novel views directly as supervision for our monocular baseline without diffusion-aware reconstruction. Due to the spatio-temporal inconsistencies of the diffused outputs, this leads to a poor quality reconstruction, as shown in Figure 5. We show that removing dynamic reconstruction leads to blurry reconstruction in static regions, while removing sampled camera optimization leads to geometric inconsistencies. We also show that using our tracking based Gaussian classification reduces floaters.

### 5 Conclusion

We present ViDAR, a novel method for 4D reconstruction of scenes from monocular inputs. ViDAR leverages video diffusion models by conditioning on scene-specific features to recover fine-grained appearance details of novel viewpoints. ViDAR overcomes the spatio-temporal inconsistency of diffusion-based supervision via a diffusion-aware loss function and a camera pose optimisation strategy. We show that ViDAR outperforms all state-of-the-art baselines on the challening DyCheck dataset, and we present a new benchmark to evaluate performance in dynamic regions.

Limitations: ViDAR limits the scope of diffusion to enhancing rendered images, which are limited by the initial accuracy of the 4D reconstruction, thus, cannot repair major geometrical artefacts.

### References

- [1] M.-Q. V. Bui, J. Park, J. Oh, and M. Kim. DyBluRF: Dynamic Deblurring Neural Radiance Fields for Blurry Monocular Video. arXiv preprint arXiv:2312.13528, 2023.
- [2] A. Cao and J. CV. HexPlane: A Fast Representation for Dynamic Scenes. In Computer Vision and Pattern Recognition Conference (CVPR), 2023.
- [3] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, and A. Kanazawa. K-Planes: Explicit Radiance Fields in Space, Time, and Appearance. In Computer Vision and Pattern Recognition Conference (CVPR), 2023.
- [4] H. Gao, R. Li, S. Tulsiani, B. Russell, and A. Kanazawa. Monocular Dynamic View Synthesis: A Reality Check. In Conference on Neural Information Processing Systems, 2022.
- [5] V. Guizilini, M. Z. Irshad, D. Chen, G. Shakhnarovich, and R. Ambrus. Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion. In Computer Vision and Pattern Recognition Conference (CVPR), 2025.
- [6] Y.-H. Huang, Y.-T. Sun, Z. Yang, X. Lyu, Y.-P. Cao, and X. Qi. SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes. In Computer Vision and Pattern Recognition Conference (CVPR), 2023.
- [7] M. Kappel, F. Hahlbohm, T. Scholz, S. Castillo, C. Theobalt, M. Eisemann, V. Golyanik, and M. Magnor. D-NPC: Dynamic neural point clouds for non-rigid view synthesis from monocular video. Proceedings of the Eurographics Conference (EG), 44, 2025.
- [8] T. Karras, M. Aittala, T. Aila, and S. Laine. Elucidating the design space of diffusion-based generative models. In Conference on Neural Information Processing Systems, 2022.
- [9] B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics, 42(4), July 2023.
- [10] J. Lee, C. Won, H. Jung, I. Bae, and H.-G. Jeon. Fully Explicit Dynamic Guassian Splatting. In Proceedings of the Neural Information Processing Systems, 2024.
- [11] J. Lei, Y. Weng, A. Harley, L. Guibas, and K. Daniilidis. MoSca: Dynamic gaussian fusion from casual videos via 4D motion scaffolds. Computer Vision and Pattern Recognition Conference (CVPR), 2025.
- [12] H. Li, H. Shi, W. Zhang, W. Wu, Y. Liao, L. Wang, L.-h. Lee, and P. Y. Zhou. Dreamscene: 3d gaussianbased text-to-3d scene generation via formation pattern sampling. In European Conference on Computer Vision (ECCV), 2024.
- [13] L. Li, Z. Shen, Z. Wang, L. Shen, and P. Tan. Streaming Radiance Fields for 3D Video Synthesis. In Conference on Neural Information Processing Systems, 2022.
- [14] Z. Li, S. Niklaus, N. Snavely, and O. Wang. Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes. In Computer Vision and Pattern Recognition Conference (CVPR), 2021.
- [15] Z. Li, Q. Wang, F. Cole, R. Tucker, and N. Snavely. DynIBaR: Neural Dynamic Image-Based Rendering. In Computer Vision and Pattern Recognition Conference (CVPR), 2023.
- [16] Z. Li, Z. Chen, Z. Li, and Y. Xu. Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis. In Computer Vision and Pattern Recognition Conference (CVPR), 2024.
- [17] H. Liang, J. Ren, A. Mirzaei, A. Torralba, Z. Liu, I. Gilitschenski, S. Fidler, C. Oztireli, H. Ling, Z. Gojcic, and J. Huang. Feed-Forward Bullet-Time Reconstruction of Dynamic Scenes from Monocular Videos. arXiv preprint arXiv:2412.03526, 2024.
- [18] Y. Liang, X. Yang, J. Lin, H. Li, X. Xu, and Y. Chen. Luciddreamer: Towards high-fidelity text-to-3d generation via interval score matching. In Computer Vision and Pattern Recognition Conference (CVPR), 2024.
- [19] C. Lin, P. Pan, B. Yang, Z. Li, and Y. Mu. DiffSplat: Repurposing Image Diffusion Models for Scalable 3D Gaussian Splat Generation. In International Conference on Learning Representations (ICLR), 2025.
- [20] Y. Lin, Z. Dai, S. Zhu, and Y. Yao. Gaussian-Flow: 4D Reconstruction with Dynamic 3D Gaussian Particle. In Computer Vision and Pattern Recognition Conference (CVPR), 2024.

- [21] M. Liu, R. Shi, L. Chen, Z. Zhang, C. Xu, X. Wei, H. Chen, C. Zeng, J. Gu, and H. Su. One-2-3-45++: Fast Single Image to 3D Objects with Consistent Multi-View Generation and 3D Diffusion. In Computer Vision and Pattern Recognition Conference (CVPR), 2024.
- [22] M. Liu, C. Zeng, X. Wei, R. Shi, L. Chen, C. Xu, M. Zhang, Z. Wang, X. Zhang, I. Liu, H. Wu, and H. Su. MeshFormer: High-Quality Mesh Generation with 3D-Guided Reconstruction Model. In Conference on Neural Information Processing Systems, 2024.
- [23] Q. Liu, Y. Liu, J. Wang, X. Lyu, P. Wang, W. Wang, and J. Hou. MoDGS: Dynamic Gaussian Splatting from Casually-captured Monocular Videos with Depth Priors. In International Conference on Learning Representations (ICLR), 2025.
- [24] Y.-L. Liu, C. Gao, A. Meuleman, H.-Y. Tseng, A. Saraf, C. Kim, Y.-Y. Chuang, J. Kopf, and J.-B. Huang. Robust dynamic radiance fields. In Computer Vision and Pattern Recognition Conference (CVPR), 2023.
- [25] J. Luiten, G. Kopanas, B. Leibe, and D. Ramanan. Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis. In International Conference on 3D Vision (3DV), 2024.
- [26] X. Miao, Y. Bai, H. Duan, F. Wan, Y. Huang, Y. Long, and Y. Zheng. CTNeRF: Cross-time Transformer for dynamic neural radiance field from monocular video. Pattern Recognition, 156:110729, 2024.
- [27] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. In European Conference on Computer Vision (ECCV), 2020.
- [28] J. Park, M.-Q. V. Bui, J. L. G. Bello, J. Moon, J. Oh, and M. Kim. SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video. In Computer Vision and Pattern Recognition Conference (CVPR), 2025.
- [29] K. Park, U. Sinha, J. T. Barron, S. Bouaziz, D. B. Goldman, S. M. Seitz, and R. Martin-Brualla. Nerfies: Deformable Neural Radiance Fields. International Conference on Computer Vision (ICCV), 2021.
- [30] K. Park, U. Sinha, P. Hedman, J. T. Barron, S. Bouaziz, D. B. Goldman, R. Martin-Brualla, and S. M. Seitz. HyperNeRF: a higher-dimensional representation for topologically varying neural radiance fields. ACM Transactions on Graphics, 40(6), 2021.
- [31] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. In International Conference on Learning Representations (ICLR), 2024.
- [32] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall. DreamFusion: Text-to-3D using 2D Diffusion. In International Conference on Learning Representations (ICLR), 2023.
- [33] A. Pumarola, E. Corona, G. Pons-Moll, and F. Moreno-Noguer. D-NeRF: Neural Radiance Fields for Dynamic Scenes. In Computer Vision and Pattern Recognition Conference (CVPR), 2021.
- [34] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Computer Vision and Pattern Recognition Conference (CVPR), 2021.
- [35] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman. DreamBooth: Fine Tuning Textto-image Diffusion Models for Subject-Driven Generation. In Computer Vision and Pattern Recognition Conference (CVPR), 2023.
- [36] R. Shao, Z. Zheng, H. Tu, B. Liu, H. Zhang, and Y. Liu. Tensor4D: Efficient Neural 4D Decomposition for High-fidelity Dynamic Reconstruction and Rendering. In Computer Vision and Pattern Recognition Conference (CVPR), 2023.
- [37] R. Shaw, M. Nazarczuk, J. Song, A. Moreau, S. Catley-Chandar, H. Dhamo, and E. Pérez-Pellitero. SWinGS: Sliding Windows for Dynamic 3D Gaussian Splatting. In European Conference on Computer Vision (ECCV), 2024.
- [38] J. Shriram, A. Trevithick, L. Liu, and R. Ramamoorthi. RealmDreamer: Text-Driven 3D Scene Generation with Inpainting and Depth Diffusion. In International Conference on 3D Vision (3DV), 2025.
- [39] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In International Conference on Learning Representations, 2015.
- [40] C. Stearns, A. W. Harley, M. Uy, F. Dubost, F. Tombari, G. Wetzstein, and L. Guibas. Dynamic Gaussian Marbles for Novel View Synthesis of Casual Monocular Videos. In SIGGRAPH Asia, 2024.

- [41] J. Tang, Z. Chen, X. Chen, T. Wang, G. Zeng, and Z. Liu. LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation. In European Conference on Computer Vision (ECCV), 2024.
- [42] C. Wang, P. Zhuang, A. Siarohin, J. Cao, G. Qian, H.-Y. Lee, and S. Tulyakov. Diffusion Priors for Dynamic View Synthesis from Monocular Videos. arXiv preprint arXiv:2401.05583, 2024.
- [43] F. Wang, S. Tan, X. Li, Z. Tian, and H. Liu. Mixed Neural Voxels for Fast Multi-view Video Synthesis. In International Conference on Computer Vision (ICCV), 2023.
- [44] Q. Wang, V. Ye, H. Gao, W. Zeng, J. Austin, Z. Li, and A. Kanazawa. Shape of Motion: 4D Reconstruction from a Single Video. In arXiv preprint arXiv:2407.13764, 2024.
- [45] Z. Wang, A. Bovik, H. Sheikh, and E. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE TIP, 13(4), 2004.
- [46] T. Wimmer, M. Oechsle, M. Niemeyer, and F. Tombari. Gaussians-to-Life: Text-Driven Animation of 3D Gaussian Splatting Scenes. In International Conference on 3D Vision (3DV), 2025.
- [47] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang. 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering. In Computer Vision and Pattern Recognition Conference (CVPR), 2024.
- [48] R. Wu, R. Gao, B. Poole, A. Trevithick, C. Zheng, J. T. Barron, and A. Holynski. CAT4D: Create Anything in 4D with Multi-View Video Diffusion Models. arXiv:2411.18613, 2024.
- [49] R. Wu, B. Mildenhall, P. Henzler, K. Park, R. Gao, D. Watson, P. P. Srinivasan, D. Verbin, J. T. Barron, B. Poole, and A. Holynski. ReconFusion: 3D Reconstruction with Diffusion Priors. In Computer Vision and Pattern Recognition Conference (CVPR), 2024.
- [50] J. Xu, W. Cheng, Y. Gao, X. Wang, S. Gao, and Y. Shan. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191, 2024.
- [51] Y. Xu, H. Tan, F. Luan, S. Bi, P. Wang, J. Li, Z. Shi, K. Sunkavalli, G. Wetzstein, Z. Xu, and K. Zhang. DMV3D: Denoising Multi-View Diffusion using 3D Large Reconstruction Model. In International Conference on Learning Representations (ICLR), 2024.
- [52] C. Yang, S. Li, J. Fang, R. Liang, L. Xie, X. Zhang, W. Shen, and Q. Tian. GaussianObject: High-Quality 3D Object Reconstruction from Four Views with Gaussian Splatting. In SIGGRAPH Asia, 2024.
- [53] J. Yang, M. Gao, Z. Li, S. Gao, F. Wang, and F. Zheng. Track Anything: Segment Anything Meets Videos. arXiv preprint arXiv:2304.11968, 2023.
- [54] S. Yang, W. Yu, J. Zeng, J. Lv, K. Ren, C. Lu, D. Lin, and J. Pang. Novel demonstration generation with gaussian splatting enables robust one-shot manipulation. arXiv preprint arXiv:2504.13175, 2025.
- [55] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin. Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction. In Computer Vision and Pattern Recognition Conference (CVPR), 2024.
- [56] T. Yi, J. Fang, J. Wang, G. Wu, L. Xie, X. Zhang, W. Liu, Q. Tian, and X. Wang. GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models. In Computer Vision and Pattern Recognition Conference (CVPR), 2024.
- [57] J. S. Yoon, K. Kim, O. Gallo, H. S. Park, and J. Kautz. Novel view synthesis of dynamic scenes with globally coherent depths from a monocular camera. In Computer Vision and Pattern Recognition Conference (CVPR), 2020.
- [58] H. Yu, C. Wang, P. Zhuang, W. Menapace, A. Siarohin, J. Cao, L. A. Jeni, S. Tulyakov, and H.-Y. Lee. 4Real: Towards Photorealistic 4D Scene Generation via Video Diffusion Models. In Conference on Neural Information Processing Systems, 2024.
- [59] Y. Zeng, Y. Jiang, S. Zhu, Y. Lu, Y. Lin, H. Zhu, W. Hu, X. Cao, and Y. Yao. STAG4D: Spatial-Temporal Anchored Generative 4D Gaussians. In European Conference on Computer Vision (ECCV), 2024.
- [60] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In Computer Vision and Pattern Recognition Conference (CVPR), 2018.
- [61] X. Zhao, A. Colburn, F. Ma, M. Ángel Bautista, J. M. Susskind, and A. G. Schwing. Pseudo-Generalized Dynamic View Synthesis from a Video. In International Conference on Learning Representations (ICLR), 2024.

- [62] K. Zhou, J.-X. Zhong, S. Shin, K. Lu, Y. Yang, A. Markham, and N. Trigoni. DynPoint: dynamic neural point for view synthesis. In Conference on Neural Information Processing Systems, 2023.
- [63] R. Zhu, Y. Liang, H. Chang, J. Deng, J. Lu, W. Yang, T. Zhang, and Y. Zhang. MotionGS: Exploring Explicit Motion Guidance for Deformable 3D Gaussian Splatting. In Conference on Neural Information Processing Systems, 2024.
- [64] Z.-X. Zou, Z. Yu, Y.-C. Guo, Y. Li, D. Liang, Y.-P. Cao, and S.-H. Zhang. Triplane Meets Gaussian Splatting: Fast and Generalizable Single-View 3D Reconstruction with Transformers. In Computer Vision and Pattern Recognition Conference (CVPR), 2024.

# ViDAR: Video Diffusion-Aware 4D Reconstruction From Monocular Inputs

## Supplementary Material

- A Additional Results In this section, we include additional qualitative and quantitative evaluation of ViDAR.

- A.1 Further Qualitative Evaluation

We present additional qualitative evaluation of ViDAR compared to MoSca and Shape Of Motion on the qualitative example scenes from the DyCheck dataset in Fig. 6. Our results show consistently greater geometric consistency and visual quality compared to the other approaches.

- A.2 Per-Scene Results

We provide a detailed quantitative evaluation for every scene of the DyCheck dataset in Tables5 and 6, in half and full resolution respectively. As in Tables1 and 2, we compute PSNR, SSIM and LPIPS on the co-visibility masked regions of the test views, which we denote with an -m addendum to each metric, as well as on the dynamic masked regions of the test views which we denote with a -D. With a few exceptions (e.g. Apple, co-visibility), ViDAR is consistently the best performing method.

- B Implementation Details In this section, we provide any implementation details not included in the main manuscript.

#### B.1 Monocular Reconstruction

We implement the monocular reconstruction step directly as MoSca[11], keeping the original hyperparameters intact. We substitute dynamic masks estimated from epipolar error by masks obtained from Track Anything[53].

#### B.2 Personalised Diffusion Model

We train our personalised diffusion model with a Dreambooth[35] approach implemented in the diffusers2 library as a LoRA fine-tuning process. We use the default implementation of the SDXL model with default parameters. We change the resolution to match our input resolution (720x960). Similarly, we change the number of training iterations from the default 500 to 5000, in response to the default model being suitable for personalisation with a smaller number of images (5-40), as opposed to our inputs (ranging above 400).

#### B.3 Camera Sampling

To obtain a set of varying samples for multi-view supervision, we propose a camera sampling strategy based on extreme poses within the input trajectory.

2https://huggingface.co/docs/diffusers

Preprint. Under review.

- Table 5: Per-scene quantitative evaluation of ViDAR against state-of-the-art methods on the DyCheck dataset at half resolution. Best, second and third results are highlighted in red, orange and yellow respectively.

Method PSNR-m↑ SSIM-m↑ LPIPS-m↓ PSNR-D↑ SSIM-D↑ LPIPS-D↓

T-NeRF[4] 17.43 0.7285 0.5081 13.63 0.9433 0.3108 Nerfies[29] 17.54 0.7505 0.4785 13.63 0.9437 0.3321 HyperNeRF[30] 17.64 0.7433 0.4775 13.36 0.9417 0.3362 Gaussian Marbles[40] 17.90 0.7328 0.4716 14.56 0.9370 0.3229 SoM[44] 18.95 0.8111 0.2917 14.88 0.9419 0.3016 MoSca[11] 19.40 0.8074 0.3392 16.97 0.9580 0.2176 Ours 19.18 0.8008 0.3149 17.75 0.9616 0.1893

Apple

T-NeRF[4] 17.52 0.6688 0.3460 14.07 0.8591 0.3474 Nerfies[29] 16.61 0.6393 0.3893 13.31 0.8433 0.4068 HyperNeRF[30] 17.54 0.6702 0.3312 13.73 0.8525 0.3483 Gaussian Marbles[40] 16.95 0.6509 0.3788 13.81 0.8480 0.3750 SoM[44] 17.99 0.6634 0.2608 15.23 0.8596 0.3378 MoSca[11] 18.11 0.6801 0.3416 15.32 0.8699 0.3499 Ours 18.91 0.6901 0.2168 15.93 0.8864 0.3052

Block

T-NeRF[4] 17.55 0.3672 0.2577 12.50 0.9730 0.3149 Nerfies[29] 17.34 0.3783 0.2111 10.84 0.9710 0.3929 HyperNeRF[30] 17.38 0.3819 0.2086 10.64 0.9706 0.4040 Gaussian Marbles[40] 16.62 0.3219 0.3517 11.68 0.9710 0.2680 SoM[44] 20.85 0.6725 0.1536 12.90 0.9738 0.2634 MoSca[11] 22.24 0.7450 0.1617 14.32 0.9789 0.2832 Ours 22.48 0.7516 0.1287 15.58 0.9807 0.3080

Paper

T-NeRF[4] 17.71 0.5914 0.3768 14.52 0.8620 0.3649 Nerfies[29] 17.79 0.6217 0.3032 13.85 0.8578 0.3407 HyperNeRF[30] 17.93 0.6054 0.3203 14.26 0.8597 0.3379 Gaussian Marbles[40] 15.32 0.5512 0.4235 11.15 0.8488 0.4278 SoM[44] 19.64 0.6313 0.2374 15.72 0.8805 0.2429 MoSca[11] 20.35 0.6582 0.2702 17.57 0.8985 0.2073 Ours 21.58 0.6890 0.2010 18.77 0.9091 0.2306

Space-out

T-NeRF[4] 19.16 0.5672 0.4427 15.65 0.9090 0.3394 Nerfies[29] 18.38 0.5846 0.3087 14.11 0.8899 0.3868 HyperNeRF[30] 19.20 0.5614 0.3254 15.65 0.9060 0.3230 Gaussian Marbles[40] 18.31 0.5461 0.3558 15.68 0.8963 0.3950 SoM[44] 21.05 0.7798 0.1698 18.44 0.9180 0.2815 MoSca[11] 21.06 0.7100 0.2084 18.85 0.9240 0.2688 Ours 21.28 0.7209 0.1853 19.37 0.9327 0.2566

Spin

T-NeRF[4] 13.71 0.5695 0.4286 13.49 0.6198 0.3730 Nerfies[29] 13.65 0.5572 0.3716 13.33 0.6050 0.3280 HyperNeRF[30] 13.97 0.5678 0.3498 13.65 0.6168 0.3028 Gaussian Marbles[40] 13.65 0.5432 0.4369 13.29 0.5972 0.3961 SoM[44] 14.00 0.5462 0.3399 13.68 0.5998 0.3335 MoSca[11] 15.09 0.6133 0.3587 14.84 0.6679 0.3146 Ours 15.97 0.6416 0.3096 15.62 0.6844 0.2901

Teddy

T-NeRF[4] 15.65 0.5481 0.2925 13.19 0.8162 0.3937 Nerfies[29] 13.82 0.4580 0.3097 11.15 0.7870 0.4805 HyperNeRF[30] 13.99 0.4550 0.3102 11.60 0.7913 0.4385 Gaussian Marbles[40] 16.02 0.5416 0.3398 9.99 0.8175 0.3926 SoM[44] 17.86 0.6695 0.2144 12.75 0.8338 0.3446 MoSca[11] 17.95 0.6852 0.2309 11.57 0.8310 0.3918 Ours 18.46 0.6943 0.2058 12.23 0.8402 0.3754

Wheel

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Input Shape Of Motion[44] MoSca[11] Ours

Figure 6: Qualitative evaluation of our method against benchmark methods on the DyCheck qualitative example set. 16

- Table 6: Per-scene quantitative evaluation of ViDAR against state-of-the-art methods on the DyCheck dataset at full resolution. Best, second and third results are highlighted in red, orange and yellow respectively.

Method PSNR-m↑ SSIM-m↑ LPIPS-m↓ PSNR-D↑ SSIM-D↑ LPIPS-D↓

Gaussian Marbles[40] 16.84 0.7022 0.6849 14.65 0.9495 0.4281 SoM[44] 17.74 0.7535 0.4946 14.88 0.9540 0.4036 MoSca[11] 18.19 0.7486 0.5651 17.21 0.9681 0.3073 Ours 18.02 0.7466 0.5359 18.07 0.9694 0.2559

Apple

Gaussian Marbles[40] 16.50 0.6492 0.5065 13.49 0.8660 0.4987 SoM[44] 17.42 0.6566 0.3879 14.60 0.8759 0.4667 MoSca[11] 17.56 0.6710 0.4658 14.97 0.8848 0.4808 Ours 18.43 0.6722 0.3932 15.54 0.8898 0.4100

Block

Gaussian Marbles[40] 15.96 0.2959 0.5778 11.30 0.9722 0.5094 SoM[44] 19.65 0.5518 0.2035 13.03 0.9737 0.4861 MoSca[11] 20.82 0.6289 0.2412 14.04 0.9770 0.5498 Ours 21.06 0.6477 0.1923 14.99 0.9777 0.5003

Paper

Gaussian Marbles[40] 15.19 0.5603 0.5434 11.07 0.8671 0.5334 SoM[44] 19.54 0.6178 0.3667 16.11 0.8939 0.3363 MoSca[11] 19.93 0.6280 0.4060 17.17 0.8988 0.3207 Ours 21.15 0.6443 0.3278 18.56 0.9079 0.3356

Space-out

Gaussian Marbles[40] 17.84 0.5154 0.4784 15.79 0.9143 0.4420 SoM[44] 20.57 0.7323 0.2663 18.65 0.9339 0.3535 MoSca[11] 20.61 0.6769 0.3108 18.50 0.9371 0.3657 Ours 20.69 0.6851 0.2868 19.37 0.9445 0.3094

Spin

Gaussian Marbles[40] 13.01 0.5496 0.6559 13.05 0.6269 0.5815 SoM[44] 13.58 0.5518 0.5377 13.40 0.6251 0.4928 MoSca[11] 14.52 0.5925 0.5777 14.41 0.6692 0.4986 Ours 15.63 0.6238 0.4786 15.46 0.6892 0.4133

Teddy

Gaussian Marbles[40] 15.55 0.5310 0.5295 9.89 0.8291 0.5473 SoM[44] 17.38 0.6318 0.3456 13.05 0.8393 0.5037 MoSca[11] 17.49 0.6460 0.3684 11.42 0.8395 0.5665 Ours 18.00 0.6505 0.3215 12.21 0.8469 0.5203

Wheel

Given the set of input camera poses (position and orientation), we calculate a mean camera pose. Then, we establish a sphere approximating the surface established by the input trajectory, assuming that a target dynamic object is being tracked by the recording. Finally, we select two views in the input trajectory that, when projected on the sphere, are characterised by the largest longitudinal displacement. These constitute the extreme camera poses.

Thereafter, for each time step spanning the whole time range of the input video, we sample the following new cameras:

- • Two random camera poses from the input trajectory are selected, and a new camera pose is calculated as their mean, and random noise is added. Total cameras: 4
- • For each of the two extreme views, a random camera pose from the input trajectory is selected, and a new camera pose is calculated as their weighted average, and random noise is added, with the weight increasing towards the extreme views. Total cameras: 12
- • The extreme camera views. Total cameras: 2

This constitutes our set of 18 new training cameras for each timestep of the input video cm ∈ Csample.

#### B.4 Multi-View Sample Enhancement

Having sampled a set of new trajectories, we render them with the previously trained monocular reconstruction model, in such way we obtain a set of degraded images {Rm,t}. To perform the

enhancement as described in Section 3.2, we utilise the Image2Image translation approach as implemented in diffusers.

#### B.5 Diffusion-Aware Reconstruction

We increase the total number of iterations from 8000 to 40000 in order to train on the additional generated data. During optimisation, we run two separate forward and backward passes, the first for sampled camera pose optimisation and the second for optimising the Gaussians and input camera poses. At each iteration, we randomly select two of the sampled camera poses which correspond to the same time step as the input camera. During the first pass, we render the images and compute the mean of the camera losses Lcam for both of the sampled cameras and update only the sampled camera poses Csample. During the second pass, we re-render the images using the updated camera poses and compute the dynamic loss Ldyn using the dynamic region masks. This loss is added to the existing monocular losses and is used to update the input camera poses Cinp and the Gaussians G.

