# arXiv:2510.07944v2[cs.CV]16Oct2025

## CVD-STORM: CROSS-VIEW VIDEO DIFFUSION WITH SPATIAL-TEMPORAL RECONSTRUCTION MODEL FOR AUTONOMOUS DRIVING

Tianrui Zhang2†∗, Yichen Liu1∗, Zilin Guo2†∗, Yuxin Guo1, Jingcheng Ni1, Chenjing Ding1, Dan Xu2, Lewei Lu1, Zehuan Wu1 {liuyichen,nijingcheng,guoyuxin,dingchenjing,luotto,wuzehuan}@sensetime.com {tzhangbu,zguobd}@connect.ust.hk, danxu@cse.ust.hk 1Sensetime Research, 2The Hong Kong University of Science and Technology

ABSTRACT

Generative models have been widely applied to world modeling for environment simulation and future state prediction. With advancements in autonomous driving, there is a growing demand not only for high-fidelity video generation under various controls, but also for producing diverse and meaningful information such as depth estimation. To address this, we propose CVD-STORM, a cross-view video diffusion model utilizing a spatial-temporal reconstruction Variational Autoencoder (VAE) that generates long-term, multi-view videos with 4D reconstruction capabilities under various control inputs. Our approach first fine-tunes the VAE with an auxiliary 4D reconstruction task, enhancing its ability to encode 3D structures and temporal dynamics. Subsequently, we integrate this VAE into the video diffusion process to significantly improve generation quality. Experimental results demonstrate that our model achieves substantial improvements in both FID and FVD metrics. Additionally, the jointly-trained Gaussian Splatting Decoder effectively reconstructs dynamic scenes, providing valuable geometric information for comprehensive scene understanding. Our project page is https://sensetimefvg.github.io/CVD-STORM.

[Figure 1]

- (a) Ground Truth

[Figure 2]

- (b) w/o STORM-VAE (1200 step)

[Figure 3]

- (c) w/ STORM-VAE (1200 step)

Figure 1: Early-Stage Generation Visualization. (a) shows the ground-truth sequence. (b) depicts the model’s output at training step 1,200 when using a standard VAE. (c) presents the corresponding output generated with our STORM-VAE at the same step. Notably, (c) exhibits significantly improved convergence and visual fidelity compared to (b), demonstrating the effectiveness of our approach even at early stage in training.

†This research is mainly done during their internship in SenseTime. ∗Equal contribution.

- 1 INTRODUCTION

Autonomous vehicles have emerged as a prominent research domain within artificial intelligence applications. The development of reliable self-driving systems necessitates both extensive data collection for training decision-making algorithms and sophisticated closed-loop simulations to verify planning outputs. These requirements present significant challenges, particularly the need for driving world models that accurately represent the environment and enables precise prediction of future scenarios. Concurrently, diffusion models have become the state-of-the-art approach for video generation, offering a promising solution for realistic simulation. Recent advances in this field have demonstrated the remarkable capability of these models to generate photorealistic videos Hong et al. (2022); Zheng et al. (2024); Peng et al. (2025), with successful applications extending to complex driving scenarios Kim et al. (2021); Zhao et al. (2025).

To serve as comprehensive driving world models, diffusion-based approaches must be capable of generating long-term, multi-view, and controllable videos. Early attempts such as Gao et al. (2023); Kim et al. (2021) struggled with generating extended sequences and following complex conditional inputs. Recent advancements, however, have significantly addressed these limitations by adopting architectures and methodologies from high-performing diffusion models. For example, Chen et al. (2024); Gao et al. (2024b); Ren et al. (2025) have all implemented spatial-temporal diffusion transformer (DiT) architectures and employed multi-stage training strategies, progressively enhancing generative fidelity and temporal consistency. Nevertheless, despite incorporating cross-view generation, these approaches lack explicit 3D information, which constrains their applicability as world models. To overcome this limitation, Gao et al. (2024a) directly applies an enhanced 3D Gaussian Splatting (3DGS) technique to diffusion outputs, though internal inconsistencies in the generated images remain inadequately resolved. UniScene Li et al. (2025) incorporates semantic occupancy as conditional guidance for LiDAR generation, but requires additional annotation during the training process. Other approaches Hassan et al. (2025); Liang et al. (2025) produce depth maps supervised by Depth Anything V2 Yang et al. (2024c), but these relative depth estimates cannot accurately represent real-world geometry. While Wu et al. (2024b) attempts to generate LiDAR data and video simultaneously, the LiDAR is not well aligned with the images.

To address these challenges, we propose CVD-STORM, a framework that generates long sequential multi-view driving videos while simultaneously decoding reconstructed scenes represented by dynamic 3D Gaussian Splatting (3DGS) Kerbl et al. (2023). First, we finetune an image VAE with an affiliated Gaussian decoder as described in STORM Yang et al. (2025), enabling the decoding of VAE latents into 3D Gaussians. This finetuned model, dubbed as STORM-VAE, serves as the latent encoder for training a cross-view video diffusion model with the same architecture as Chen et al. (2024). Recent research Yu et al. (2025); Leng et al. (2025); Fuest et al. (2024) has established that representation learning is crucial to diffusion model performance. Aligned with these findings, our experiments demonstrate that the latents encoded by STORM-VAE, which fuse information from LiDAR and across frames, significantly improve the generative quality and convergence rate. Figure 1 illustrates the impressive denoising ability of CVD-STORMat an early step, compared with the one without STORM-VAE. During inference, CVD-STORM can generate long-term six-view videos conditioned on text, bounding boxes (BBox), and high-definition maps (HDMap), while the Gaussian Splatting (GS) Decoder can directly reconstruct 4D scenes from the generated latents.

In summary, our main contributions are:

- • We introduce STORM-VAE, an extended VAE model incorporating a Gaussian Splatting decoder for 4D scene reconstruction. This auxiliary network integrates spatial and temporal information into the latent representation, moving beyond RGB-only encoding. Meanwhile, it can also achieve 4D reconstruction in the driving scenarios.
- • We propose CVD-STORM, a novel pipeline for driving world modeling that simultaneously generates multi-view videos and reconstructs 4D scenes. To manage the complexity of these tasks, we adopt a two-stage training strategy: first learning scene reconstruction, followed by training a conditional world model.
- • Our experiments demonstrate that CVD-STORM not only significantly improve the generative quality of the current world model by enhancing representation learning, but also addresses the challenges of 4D absolute depth estimation.

- 2 RELATED WORK

- 2.1 VIDEO DIFFUSION AND DRIVING WORLD MODEL

The diffusion approach has become the mainstream for generative tasks. With the advancements in 2D image diffusion models Rombach et al. (2022); Zhang et al. (2023); Labs et al. (2025); Li et al. (2024), this technique has rapidly extended to video generation Hong et al. (2022); Yang et al. (2024d); Gao et al. (2025); Zheng et al. (2024); Peng et al. (2025); Kong et al. (2024), yielding impressive visualizations and enabling precise control. In addition, related studies highlight its significant potential as a real-world simulator.

In the field of autonomous driving, research started to focus on constructing driving world models based on video generation to simulate realistic driving scenarios. For instance, GenAD Yang

- et al. (2024a) leverages large-scale web video datasets to enhance long-duration video generation capabilities, while Vista Gao et al. (2024c) incorporates action inputs to control vehicle trajectories. However, these approaches are limited to single-view generation and do not include other conditions to simulate road conditions. There still exists a significant gap between their capabilities and real-world driving requirements.

Therefore, generating multi-view video with precise control and long-term consistency has attracted significant research attention. Early approaches such as Gao et al. (2023); Zhao et al. (2025); Xie

- et al. (2025) achieved promising results for short-term videos but struggled to extend sequence length effectively. The emergenece of DiT Peebles & Xie (2023) substantially improved diffusion model scalability, prompting numerous researchers to incorporate transformer architectures into driving world models. UniMLVG Chen et al. (2024) enhancs Stable Diffusion 3.5 Esser et al.

- (2024) with temporal and multi-view modules, successfully unifying multiple datasets with heterogeneous structures during training. Similarly, MagicDriveV2 Gao et al. (2024b) also employs this design but encodes videos through 3D VAE to achieve greater data compression. This architecture has demonstrated exceptional performance when applied to larger-scale datasets Ren et al. (2025); Russell et al. (2025). Additionally, researchers also have successfully incorporated action control mechanisms to enable the generation of precisely controllable multi-view videos Ni et al. (2025b). Despite these advancements, current generative methods still fail to adequately capture important structural information, particularly depth data.

- 2.2 4D RECONSTRUCTION IN DRIVING SCENARIOS

Capturing 3D information is crucially important in driving scenarios and numerous studies has explored how to predict the depth or reconstruct the 4D scene in the front-view driving videos. Some research incorporates the structure prediction in the generative procedure. For instance, UniFuture Liang et al. (2025) directly unified the depth prediction into the video generation to attain highly aligned RGB-Depth correspondence. However, this work needs Depth Anything V2 Yang et al. (2024c) to generate pseudo supervision for depth. Additionally, this approach can only produce relative depth, which is insufficient for the autonomous driving application. Another unified framework GEM Hassan et al. (2025) mitigates problems with consistencies in long-range video generation, yet still preserves similar problem in depth estimation as UniFuture.

On the other hand, a considerable body of research has focused on incorporating reconstruction tasks into driving scenarios. MagicDrive3D Gao et al. (2024a) employs a two-stage pipeline that integrates Gaussian splatting for 3D reconstruction. Although presented as a unified framework, the secondstage reconstruction process exerts minimal influence on the generative model in the first stage, limiting true end-to-end interaction. More approaches concentrate primarily on pure reconstruction objectives. For instance, ReconDreamer Ni et al. (2025a) introduces a network trained to correct artifacts in novel views reconstructed from a pretrained 3D Gaussian representation. Similarly, OmniScene Wei et al. (2025) leverages forward Gaussian mapping to obtain a 3D scene representation in bird’s-eye view (BEV) format. Building upon this, STORM Yang et al. (2025) advances the paradigm by employing forward 4D Gaussian splatting to capture spatiotemporal dynamics through sequential scene modeling.

GS Decoder

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

ViT

[Figure 15]

[Figure 16]

VAE Encoder

[Figure 17]

[Figure 18]

[Figure 19]

Rendering

[Figure 20]

[Figure 21]

[Figure 22]

Gaussians

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

VAE Decoder

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Conditions

x N

MM-DiT Block

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Camera Pose

HD Map

Temporal Block

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Cross-view Block

Text Description

Bounding Box

Noise

[Figure 44]

[Figure 45]

[Figure 46]

CVD-STORM

Figure 2: Overall framework of the model. Our pipeline contains two models. The upper section illustrates STORM-VAE training, with the forward process indicated by blue arrows. STORM-VAE takes multi-view images from context timesteps and processes the image latents through two decoders: the VAE Decoder performs image reconstruction (updated by LVAE), while the GS Decoder performs scene reconstruction (updated by LSTORM). The lower section illustrates the inference pipeline of CVD-STORM, with the forward process shown by solid block arrows. The diffusion part can either use STORM-VAE latents as reference frames for prediction or generate from noise, while incorporating various conditioning inputs for guidance.

While both generative modeling and 3D reconstruction have been extensively studied in autonomous driving contexts, few works have explored the integration of these two tasks in a synergistic manner. The potential of jointly optimizing generation and reconstruction remains largely underexplored.

- 2.3 REPRESENTATION LEARNING IN DIFFUSION

Recent research has devoted considerable effort to exploring better latent representations for improving diffusion model performance Fuest et al. (2024). Yang et al. (2022); Tian et al. (2023); Deja

- et al. (2023) involves incorporating additional tasks during generation training, such as classification and segmentation. Other works focuses on aligning the latent space with that of foundation models. For example, Pernias et al. (2023) divides diffusion training into two stages, with the first stage dedicated to training an additional encoder that extracts image semantic features. REPA Yu et al.

- (2025) takes intermediate features from the diffusion model and projects them to align with features from pretrained models, while VA-VAE Yao et al. (2025) performs this alignment during variational auto-encoder (VAE) training. Building upon REPA, REPA-E Leng et al. (2025) finetunes the entire model end-to-end, allowing the alignment loss to update VAE parameters and thereby accelerating generation performance.

Inspired by these advances, we extend this representation learning approach to video diffusion models by introducing a reconstruction task during training and simultaneously tuning the VAE. This approach aims to enhance generation performance while achieving a significant additional capability — 4D reconstruction.

- 3 METHOD

- Figure 2 illustrates the overall pipeline of our proposed method. Our framework generates multiview driving videos conditioned on various inputs, including text prompts, bounding boxes, HD maps—with or reference frames, while simultaneously producing scene reconstructions represented as dynamic 3DGS. Our approach extends UniMLVG Chen et al. (2024) by enhancing its variational autoencoder (VAE) architecture and refining the training procedure. Specifically, we first finetune the pretrained image VAE to create STORM-VAE, which incorporates an additional reconstruction task adapted from STORM Yang et al. (2025). This modification introduces a Gaussian Decoder capable of reconstructing 3D Gaussians and their associated velocities. We then leverage STORMVAE to train a DiT-based diffusion model that employs three distinct transformer blocks operating

along different data dimensions, which improve both spatial coherence and temporal consistency in the generated outputs.

- 3.1 PRELIMINARY: STORM

Given a set of images {Itv ∈ RH×W×3} with corresponding camera poses from timestamps t ∈ Tc and viewpoints v ∈ V , STORM fuses image features through a Vision Transformer (ViT) and

generate pixel-level Gaussians {Gvt ∈ RH×W×12}. Each Gaussian is characterized by its center µ ∈ R3, orientation R ∈ SO(3), scale s ∈ R, opacity o ∈ R, and color c ∈ R3. The center µ is positioned along the ray cast from the camera center, allowing the Gaussian decoder to only output the depth value. Additionally, the model predicts the velocity of each Gaussian to model dynamic scene elements. To render target viewpoints at timestamp t′, the 3D Gaussian Splatting (3DGS) elements Gvt are transformed according to their predicted velocities into Gaussians at time t′, denoted as Gvt→t′. The target images are then rendered based on the union of all Gvt→t′. To enhance image quality, STORM incorporates auxiliary tokens to compose sky colors and adopts view-based exposure variations.

The training process is supervised by target views randomly sampled within a predefined sampling range. The image rendering loss Lrgb is formulated as:

Lrgb =

t′∈Tt,v∈V

∥D(F({Itv}),t′,v) − Itv′∥22, (1)

where F represents the ViT encoder, D denotes the decoder and rendering, including all image postprocessing operations, and Tt is the set of target timesteps. Additionally, the Gaussian rendering can also produce depth so we use the depth map obtained by projecting LiDAR on camera views to supervise the training. We define the overall loss as LSTORM and omit discussion of additional loss terms not directly relevant to this paper. For more detailed description of the methodology, please refer to Yang et al. (2025).

- 3.2 STORM-VAE

We introduce STORM-VAE, a novel variational autoencoder that incorporates STORM as an auxiliary network within the VAE framework. The upper part of Figure 2 illustrates the architecture of our proposed model. STORM-VAE builds upon a general VAE structure, specifically utilizing the pretrained VAE from Stable Diffusion 3.5 (SD3.5) Esser et al. (2024) in our setting. In the STORM-VAE pipeline, the VAE encoder E first encodes input images into latent representations, which are subsequently processed through two parallel branches. In the first branch, the latents are processed by the VAE decoder DV AE to ensure high-fidelity image reconstruction, supervised by the loss function LVAE. In the second branch, sampled context latents are fed into the Gaussian Splatting decoder (DGS), which shares architectural similarities with STORM. The key distinction is that STORM processes RGB images directly while the DGS operates on the VAE latent representations. Consequently, the new RGB rendering loss is formulated as:

Lrgb =

t′∈Tt,v∈V

DGS(E(Itv),t′,v) − It′v|22, (2)

where DGS is equivalent to D · F described in Section 3.1. The comprehensive training objective combines the VAE and STORM components as follows:

L = LVAE + λLSTORM. (3)

Here, LVAE comprises three components: the reconstruction loss LMSE, the perceptual loss LLPIPS, and the KL divergence loss LKL. We deliberately excluded the GAN loss from our implementation as our experiments indicated it compromised training stability. In our experiments, we set λ to 0.5.

- 3.3 CVD-STORM

The lower part of Figure 2 illustrates the architecture of CVD-STORM. Following UniMLVG Chen

- et al. (2024), we adopt SD3.5 as initialization and append a temporal block and a cross-view block after each Multi-Modality DiT (MM-DiT) block of SD3.5. The input latent of CVD-STORM is

Method Duration FID↓ FVD↓ mAPobj↑ mIoUr↑ mIoUv↑ DreamForge Mei et al. (2024) 20s 16 224.8 13.80 - UniScene Li et al. (2025) - 6.1 70.5 - - Glad Xie et al. (2025) - 11.2 118.0 - - DriveScape Wu et al. (2024a) - 8.3 76.4 - 64.43 28.86 MagicDrive2 Gao et al. (2024b) 5s 19.1 218.1 12.30 61.05 27.01 DriveSphere Yan et al. (2025) - - 103.4 21.45 - DiVE Jiang et al. (2024) 20s - 94.6 24.55 - UniMLVG Chen et al. (2024) 20s 5.8 36.1 22.50 70.81 29.12 CVD-STORM 20s 3.8 14.0 25.21 66.11 29.84

Table 1: Comparison of the generation quality and condition-following metrics on nuScenes validation set. The best results are in bold, while the second best results are in underlined. Since most of methods do not release their checkpoints, we list the results reported in their paper. − represents the values not mentioned in the corresponding papers. mIoUr and mIoUv are the short of the mean IoU of road and vehicle.

zt ∈ RT×V ×C×H×W, where T is the number of frames, V is the number of viewpoints, C is the latent dimension, and H,W are the latent spatial dimensions of a single image. The MM-DiT block performs attention only at the image level (i.e., across H × W dimensions), which requires reshaping the input to HW × TV × C before processing. Similarly, the temporal block operates on the sequence length dimension and the cross-view block operates on the view dimension, requiring to reshape the input to T × V HW × C and V × THW × C, respectively. We also incorporate the multiple conditioning approaches and multi-task framework from UniMLVG in our training. For details regarding these components, please refer to their paper. The training loss utilizes rectified flow Liu et al. (2022), formulated as:

#### LSD = Eϵ∼N(0,I) ∥ϵθ(zt,t,c) − (z0 − ϵ)∥2 , (4)

where ϵθ denotes the model, z represents the STORM-VAE latent, zt is the noisy latent, ϵ is the noise, t is the timestep, and c is the conditioning information.

Different from UniMLVG, we replace the SD3.5 VAE with our STORM-VAE, which provides enhanced latent representations and the capability to estimate absolute depth. Furthermore, rather than employing a multi-stage training process to progressively develop temporal and spatial generation capabilities, we jointly train the temporal blocks, spatial blocks, and MM-DiT blocks in a single stage. This integrated approach significantly simplifies the training procedure and reduces computational costs.

- 4 EXPERIMENTS

- 4.1 EXPERIMENT DETAILS

- 4.1.1 DATASETS

We adopt both single-view and multi-view datasets in our training: OpenDV-Youtube Yang et al. (2024b) for single-view data, and nuScenes Caesar et al. (2020), Waymo Sun et al. (2020), and Argoverse2 Wilson et al. (2023) for multi-view data. We set the sequence length of a single simple as 19. To enhance the extensibility and diversity of our model, we incorporate three different image resolutions: 144 × 256, 176 × 304, and 256 × 448, with sampling ratios of 0.1, 0.3, and 0.6, respectively. All the models are trained on H100 with batch size 32. For diffusion training, we leverage available dataset annotations, including 3D bounding boxes, HD maps, and camera parameters. For nuScenes specifically, we utilize 12 Hz interpolated annotations. Text descriptions for all frames and views are generated at 2 Hz (key frames for evaluation).

- 4.1.2 EVALUATION METRICS

To assess the effectiveness of our method in terms of realism, continuity, ad precise control, we selected four key metrics to compare against existing multi-view image and video generation meth-

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

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

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

- Figure 3: Qualitative results of Depth Estimation. This figure illustrates the depth of the videos generated by CVD-STORM at frame 0, 5, 10. Our GS decoder can successfully extract the depth information of dynamic and static objects.

ods. We use the widely recognized Fr´echet Inception Distance(FID) Heusel et al. (2017) for realism evalution and Fr´echet Video Distance (FVD) Unterthiner et al. (2018) for temporal coherence estimation. To evaluate controllablity, we evaluate two perception tasks: 3D object detection and BEV segmentation of road maps. These tasks serve as proxies for measuring the spatial accuracy and consistent geometry of our generated content. We adopt BEVFormer Li et al. (2022) and cross-view transformers Zhou & Kr¨ahenb¨uhl (2022) to evaluate the performance on these two tasks respectively.

- 4.1.3 IMPLEMENTATION DETAILS

For STORM-VAE training, we designate the 1st, 7th, 13th and 19th frames as the context frames while 3 timesteps are randomly sampled as targets. Since the Opendv-Youtube is a single-view dataset without LiDAR data, it is used exclusively to train the VAE image reconstruction branch. The other three datasets are utilized for both VAE and STORM training. To address viewpoint inconsistency across datasets, we standardize inputs to 6 views for all datasets and implement attention masking to avoid redundant data fusion.

For diffusion training, we freeze the encoder of STORM-VAE. As discussed in Section 3.3, we implement the single-stage training so we have to deal with the invariance across datasets. For OpenDV-Youtube, the cross-view block is omitted due to its single-view nature. For multi-view datasets, we randomly drop temporal and cross-view blocks to enhance the generative capability of each individual block, thereby improving the overall model stability and robustness. During inference, we use 3 frames as reference for autoregressive prediction. A cosine scheduler is used with initial learning rate of 6 × 10−5. The minimum learning rate is set to 1 × 10−7. The optimize is widely used AdamW. The inference steps are set to 50. All Experiments are conducted on H100 GPUs.

- 4.2 EXPERIMENT RESULTS 4.2.1 COMPARISON

Generation Tasks. Following the common evaluation protocols, we report quantitative metrics on the nuScenes validation set, shown in Table 1. Our model demonstrates exceptional performance compared to previous SOTA methods DiVE Jiang et al. (2024) and UniMLVG Chen et al.

Table 2: Ablation Study # Ref. frames FID FVD

VAE used FID FVD w/o STORM-VAE 9.36 52.85 w/ STORM-VAE 7.92 34.37

- 0 8.7 39.0

- 1 3.6 17.2 3 3.8 14.0

(a) Ablation study of the number of reference frames. The best results are in bold. The FID is about the same with reference frame, while FVD strictly decreases with larger reference frame count.

(b) Ablation study of the use of VAE. The best results are in bold. w/o STORM-VAE means using default vae of SD3.5. Both models are trained for 40k steps with Opendv, nuScenes, Waymo, Argoverse2. No pretarined weight is loaded for diffusion for fair comparison.

[Figure 83]

Reference frame 1

[Figure 84]

Generated frame 4

[Figure 85]

Generated frame 9

[Figure 86]

Generated frame 14

[Figure 87]

Generated frame 19

[Figure 88]

Generated frame 24

[Figure 89]

Generated frame 29

- Figure 4: Qualitative Results of Video Prediction. We produce this example using three reference frames. The first line is the first reference frame and the following lines are the predicted frames. Our method demonstrates strong temporal consistency in the video prediction task.

(2024). Specifically, our approach achieves significant improvements of 34.48% in Fr´echet Inception Distance (FID) and 61.21% in Fr´echet Video Distance (FVD) relative to the second-best method. Additionally, our model can generate high-quality videos with durations up to 20 seconds. Regarding condition consistency, our approach outperforms competing methods on mAP of object detection (mAPobj) and IoU of road (IoUr) of. It ranks second in IoU of vehicle (IoUv), performing marginally below UniMLVG in this particular metric.

STORM-VAE Results. We provide the visualization of the depth maps of the generative images in Figure 3. We put the more detailed evaluation and discussion in the Appendix.

- 4.3 ABLATION STUDY

Number of Reference Frames. The number of reference frames represents different types of tasks in the generative model. Without reference frames, the model conducts pure video generation, producing content based solely on conditional inputs. On the contrary, the model perform video prediction when the reference frames are given. We present qualitative results in Figures 4 and 5, with quantitative evaluations in Table 2a. As shown in the table, the FVD score is steadily improved as the number of reference frames increases, indicating that additional reference frames provide richer

temporal information from the ground truth, thereby improving temporal consistency. Conversely, when reference frames are provided, the model performs video prediction. For more results, please refer to the Appendix.

Effect of STORM-VAE. Table 2b demonstrates that our STORM-VAE significantly improves generation quality over the standard VAE baseline. Specifically, STORM-VAE yields a 15.38% reduction in Fr´echet Inception Distance (FID) and a 34.97% decrease in Fr´echet Video Distance (FVD), indicating substantial enhancements in both image and video generation quality. Furthermore, Figure 1 illustrates that our model accelerates convergence compared to the baseline. To ensure fair evaluation in this ablation study, we compare models trained for the same number of steps.

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

- Figure 5: Qualitative Results of Video Generation. We provide the examples generated with the conditons only, without any reference frame. For each scene, we list the 1st frame in the first line and the 10th frame in the second line. The bounding boxes and road maps are overlapping over the generative images. The object in the bounding boxes with the same color are should be of the same class. For example, cars should be generated in the blue 3D bounding boxes.

- 5 CONCLUSION

We introduce CVD-STORM, a novel framework that unifies long-sequence, multi-view video generation with dynamic 4D scene reconstruction. Our approach extends the traditional VAE architec-

ture by incorporating a Gaussian Splatting Decoder, namely STORM-VAE. This design not only enables high-quality 4D scene reconstruction but also substantially enhances representation learning, thereby improving the generative capabilities of our downstream diffusion model. Leveraging the pre-trained STORM-VAE, we train CVD-STORM using multiple datasets and support various conditioning types across diverse generative tasks. Experimental results demonstrate that CVDSTORM surpasses SOTA methods, particularly in image quality and temporal coherence. Furthermore, the Gaussian Splatting Decoder directly estimates absolute depth through neural rendering, providing richer 3D structural information than previous approaches.

REFERENCES

Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11621–11631, 2020.

Rui Chen, Zehuan Wu, Yichen Liu, Yuxin Guo, Jingcheng Ni, Haifeng Xia, and Siyu Xia. Unimlvg: Unified framework for multi-view long video generation with comprehensive control capabilities for autonomous driving. arXiv preprint arXiv:2412.04842, 2024.

Kamil Deja, Tomasz Trzci´nski, and Jakub M Tomczak. Learning data representations with joint diffusion models. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 543–559. Springer, 2023.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Michael Fuest, Pingchuan Ma, Ming Gui, Johannes Schusterbauer, Vincent Tao Hu, and Bjorn Ommer. Diffusion models and representation learning: A survey. arXiv preprint arXiv:2407.00783, 2024.

Ruiyuan Gao, Kai Chen, Enze Xie, Lanqing Hong, Zhenguo Li, Dit-Yan Yeung, and Qiang Xu. Magicdrive: Street view generation with diverse 3d geometry control. arXiv preprint arXiv:2310.02601, 2023.

Ruiyuan Gao, Kai Chen, Zhihao Li, Lanqing Hong, Zhenguo Li, and Qiang Xu. Magicdrive3d: Controllable 3d generation for any-view rendering in street scenes. arXiv preprint arXiv:2405.14475, 2024a.

Ruiyuan Gao, Kai Chen, Bo Xiao, Lanqing Hong, Zhenguo Li, and Qiang Xu. Magicdrive-v2: Highresolution long video generation for autonomous driving with adaptive control. arXiv preprint arXiv:2411.13807, 2024b.

Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. Advances in Neural Information Processing Systems, 37:91560–91596, 2024c.

Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.

Mariam Hassan, Sebastian Stapf, Ahmad Rahimi, Pedro Rezende, Yasaman Haghighi, David Br¨uggemann, Isinsu Katircioglu, Lin Zhang, Xiaoran Chen, Suman Saha, et al. Gem: A generalizable ego-vision multimodal world model for fine-grained ego-motion, object dynamics, and scene composition control. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 22404–22415, 2025.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.

Junpeng Jiang, Gangyi Hong, Lijun Zhou, Enhui Ma, Hengtong Hu, Xia Zhou, Jie Xiang, Fan Liu, Kaicheng Yu, Haiyang Sun, et al. Dive: Dit-based video generation with enhanced control. arXiv preprint arXiv:2409.01595, 2024.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.

Seung Wook Kim, Jonah Philion, Antonio Torralba, and Sanja Fidler. Drivegan: Towards a controllable high-quality neural simulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5820–5829, 2021.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, and Others. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025. URL https://arxiv.org/abs/2506.15742.

Xingjian Leng, Jaskirat Singh, Yunzhong Hou, Zhenchang Xing, Saining Xie, and Liang Zheng. Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers. arXiv preprint arXiv:2504.10483, 2025.

Bohan Li, Jiazhe Guo, Hongsi Liu, Yingshuang Zou, Yikang Ding, Xiwu Chen, Hu Zhu, Feiyang Tan, Chi Zhang, Tiancai Wang, et al. Uniscene: Unified occupancy-centric driving scene generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 11971– 11981, 2025.

Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, et al. Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748, 2024.

Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Yu Qiao, and Jifeng Dai. Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers. arXiv preprint arXiv:2203.17270, 2022.

Dingkang Liang, Dingyuan Zhang, Xin Zhou, Sifan Tu, Tianrui Feng, Xiaofan Li, Yumeng Zhang, Mingyang Du, Xiao Tan, and Xiang Bai. Seeing the future, perceiving the future: A unified driving world model for future generation and perception. arXiv preprint arXiv:2503.13587, 2025.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Jianbiao Mei, Tao Hu, Xuemeng Yang, Licheng Wen, Yu Yang, Tiantian Wei, Yukai Ma, Min Dou, Botian Shi, and Yong Liu. Dreamforge: Motion-aware autoregressive video generation for multiview driving scenes. arXiv preprint arXiv:2409.04003, 2024.

Chaojun Ni, Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Wenkang Qin, Guan Huang, Chen Liu, Yuyin Chen, Yida Wang, Xueyang Zhang, et al. Recondreamer: Crafting world models for driving scene reconstruction via online restoration. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1559–1569, 2025a.

Jingcheng Ni, Yuxin Guo, Yichen Liu, Rui Chen, Lewei Lu, and Zehuan Wu. Maskgwm: A generalizable driving world model with video mask reconstruction. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 22381–22391, 2025b.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023.

Xiangyu Peng, Zangwei Zheng, Chenhui Shen, Tom Young, Xinying Guo, Binluo Wang, Hang Xu, Hongxin Liu, Mingyan Jiang, Wenjun Li, Yuhui Wang, Anbang Ye, Gang Ren, Qianran Ma, Wanying Liang, Xiang Lian, Xiwen Wu, Yuting Zhong, Zhuangyan Li, Chaoyu Gong, Guojun Lei, Leijun Cheng, Limin Zhang, Minghao Li, Ruijie Zhang, Silan Hu, Shijie Huang, Xiaokang Wang, Yuanheng Zhao, Yuqi Wang, Ziang Wei, and Yang You. Open-sora 2.0: Training a commercial-level video generation model in 200k. arXiv preprint arXiv:2503.09642, 2025.

Pablo Pernias, Dominic Rampas, Mats L Richter, Christopher J Pal, and Marc Aubreville. W¨urstchen: An efficient architecture for large-scale text-to-image diffusion models. arXiv preprint arXiv:2306.00637, 2023.

Xuanchi Ren, Yifan Lu, Tianshi Cao, Ruiyuan Gao, Shengyu Huang, Amirmojtaba Sabour, Tianchang Shen, Tobias Pfaff, Jay Zhangjie Wu, Runjian Chen, et al. Cosmos-drive-dreams: Scalable synthetic driving data generation with world foundation models. arXiv preprint arXiv:2506.09042, 2025.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Lloyd Russell, Anthony Hu, Lorenzo Bertoni, George Fedoseev, Jamie Shotton, Elahe Arani, and Gianluca Corrado. Gaia-2: A controllable multi-view generative world model for autonomous driving. arXiv preprint arXiv:2503.20523, 2025.

Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2446–2454, 2020.

Changyao Tian, Chenxin Tao, Jifeng Dai, Hao Li, Ziheng Li, Lewei Lu, Xiaogang Wang, Hongsheng Li, Gao Huang, and Xizhou Zhu. Addp: Learning general representations for image recognition and generation with alternating denoising diffusion process. arXiv preprint arXiv:2306.05423, 2023.

Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018.

Dongxu Wei, Zhiqi Li, and Peidong Liu. Omni-scene: Omni-gaussian representation for ego-centric sparse-view scene reconstruction. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 22317–22327, 2025.

Benjamin Wilson, William Qi, Tanmay Agarwal, John Lambert, Jagjeet Singh, Siddhesh Khandelwal, Bowen Pan, Ratnesh Kumar, Andrew Hartnett, Jhony Kaesemodel Pontes, et al. Argoverse 2: Next generation datasets for self-driving perception and forecasting. arXiv preprint arXiv:2301.00493, 2023.

Wei Wu, Xi Guo, Weixuan Tang, Tingxuan Huang, Chiyu Wang, Dongyue Chen, and Chenjing Ding. Drivescape: Towards high-resolution controllable multi-view driving video generation. arXiv preprint arXiv:2409.05463, 2024a.

Zehuan Wu, Jingcheng Ni, Xiaodong Wang, Yuxin Guo, Rui Chen, Lewei Lu, Jifeng Dai, and Yuwen Xiong. Holodrive: Holistic 2d-3d multi-modal street scene generation for autonomous driving. arXiv preprint arXiv:2412.01407, 2024b.

Bin Xie, Yingfei Liu, Tiancai Wang, Jiale Cao, and Xiangyu Zhang. Glad: A streaming scene generator for autonomous driving. arXiv preprint arXiv:2503.00045, 2025.

Tianyi Yan, Dongming Wu, Wencheng Han, Junpeng Jiang, Xia Zhou, Kun Zhan, Cheng-zhong Xu, and Jianbing Shen. Drivingsphere: Building a high-fidelity 4d world for closed-loop simulation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 27531–27541, 2025.

Jiawei Yang, Jiahui Huang, Yuxiao Chen, Yan Wang, Boyi Li, Yurong You, Maximilian Igl, Apoorva Sharma, Peter Karkus, Danfei Xu, Boris Ivanovic, Yue Wang, and Marco Pavone. Storm: Spatiotemporal reconstruction model for large-scale outdoor scenes. arXiv preprint arXiv:2501.00602, 2025.

Jiazhi Yang, Shenyuan Gao, Yihang Qiu, Li Chen, Tianyu Li, Bo Dai, Kashyap Chitta, Penghao Wu, Jia Zeng, Ping Luo, Jun Zhang, Andreas Geiger, Yu Qiao, and Hongyang Li. Generalized predictive model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14662–14672, June 2024a.

Jiazhi Yang, Shenyuan Gao, Yihang Qiu, Li Chen, Tianyu Li, Bo Dai, Kashyap Chitta, Penghao Wu, Jia Zeng, Ping Luo, et al. Generalized predictive model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14662–14672,

- 2024b.

Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in Neural Information Processing Systems, 37:21875–21911,

- 2024c.

Xiulong Yang, Sheng-Min Shih, Yinlin Fu, Xiaoting Zhao, and Shihao Ji. Your vit is secretly a hybrid discriminative-generative diffusion model. arXiv preprint arXiv:2208.07791, 2022.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024d.

Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 15703–15712, 2025.

Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In International Conference on Learning Representations, 2025.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 3836–3847, 2023.

Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Xinze Chen, Guan Huang, Xiaoyi Bao, and Xingang Wang. Drivedreamer-2: Llm-enhanced world models for diverse driving video generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 10412–10420, 2025.

Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024.

Brady Zhou and Philipp Kr¨ahenb¨uhl. Cross-view transformers for real-time map-view semantic segmentation. In CVPR, 2022.

- 6 APPENDIX 6.1 ADDITIONAL EXPERIMENTS

Table 3: Comparison of STORM and STORM-VAE Method PSNR ↑ D-RMSE ↓ STORM 20.89 5.52

Method AbsRel ↓ δ1 ↑ UniMLVG + STORM 30.825 49.7

STORM-VAE 21.18 4.55

CVD-STORM 16.05 49.7

(a) Comparison of Reconstruction. We extend the original STORM to a 6-view rendering model and evaluate the performance on NuScene. Our STORM-VAE also slightly outperforms the STORM in the reconstruction task.

(b) Comparison of Zero-shot Depth Estimate. We evaluate the performance of our models on depth estimation in the generation results. We use the pesudogroundtruth produced by Depth Anything V2.

- 6.1.1 VIDEO RESULTS We provide a video in the supplementary material for better visualization.
- 6.1.2 COMPARISON OF STORM AND STORM-VAE.

We evaluate the performance of STORM-VAE in comparison to STORM, as illustrated in Table 3. Specifically, Table 3a demonstrates STORM-VAE’s reconstruction capabilities relative to STORM. For quantitative assessment, we evaluate the reconstructed images and depth maps of STORMVAE on the nuScenes dataset using two metrics: Peak Signal-to-Noise Ratio (PSNR) for image quality and Depth Root Mean Square Error (D-RMSE) for depth accuracy. Our experimental results demonstrate that STORM-VAE even slightly exceeds its performance.

In generation task, we compare the performance of CVD-STORM against UniMLVG + STORM, which first employs UniMLVG to generate videos and subsequently applies STORM to reconstruct the 4D scene. During inference, we set the context timesteps equal to the target timesteps, which are the four adjacent frames spanning the interval [t,t + 3]. The GS Decoder processes frames [t+3, t+6] as context in next iterations and continues this progressive reconstruction strategy until reaching the end of the sequence. To assess its zero-shot depth estimation, we employ two metrics: Absolute

Relative Error (AbsRel) and δ1, where δ1 represents the percentage of pixels satisfying max(ddˆ, ddˆ) < 1.25, shown in Table 3b. Since ground truth depth is unavailable for generated results, we utilize

Depth Anything V2 Yang et al. (2024c) to produce pseudo ground-truth depth maps. While these metrics provide valuable comparative insights, we acknowledge their limitation in assessing absolute depth accuracy, which remains an open challenge in generative depth evaluation. We provide more qualitative results in Figure 6,7.

- 6.1.3 MORE QUALITATIVE RESULTS We provide more qualitative results in Figure 8, 9, 10, 11, 12.

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

[Figure 110]

[Figure 111]

frame = 0

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

[Figure 122]

[Figure 123]

frame = 20

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

[Figure 134]

[Figure 135]

frame = 40

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

frame = 60

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

frame = 80

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

[Figure 170]

[Figure 171]

frame = 80

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

frame = 120

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

frame = 140

FigureFigure 6:6: QualitativeQualitative resultsresults ofof DepthDepth Estimation.Estimation.

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

frame = 0

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

frame = 20

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

frame = 40

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

frame = 60

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

frame = 80

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

frame = 80

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

frame = 120

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

### frame=140 Figure 7: Qualitative results of Depth Estimation.

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

frame=0

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

frame=20

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

frame=40

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

frame=60

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

frame=80

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

frame=100

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

frame=120

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

frame=140

- FigureFigure 8:8: QualitativeQualitative resultsresults ofof VideoVideo GenerationGeneration

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

frame=0

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

frame=20

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

frame=40

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

frame=60

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

frame=80

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

frame=100

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

frame=120

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

frame=140

- FigureFigure 9:9: QualitativeQualitativeresultsresultsofofVideoVideoGeneration.Generation

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

frame=0

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

frame=20

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

frame=40

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

frame=60

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

frame=80

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

frame=100

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

frame=120

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

frame=140

- FigureFigure 10:10: QualitativeQualitativeresultsresultsofofVideoVideoGeneration.Generation

[Figure 580]

- Reference frame 1
- Reference frame 2
- Reference frame 3

[Figure 581]

[Figure 582]

[Figure 583]

Generated frame 4

[Figure 584]

Generated frame 14

[Figure 585]

Generated frame 24

[Figure 586]

Generated frame 34

[Figure 587]

Generated frame 44

[Figure 588]

Generated frame 54

[Figure 589]

Generated frame 64

[Figure 590]

Generated frame 74

[Figure 591]

Generated frame 84

[Figure 592]

Generated frame 94

[Figure 593]

Generated frame 104

[Figure 594]

Generated frame 114

[Figure 595]

Generated frame 124

[Figure 596]

Generated frame 134

[Figure 597]

Generated frame 144

### Figure 11: Qualitative results of Video Generation from 3 reference frames.

[Figure 598]

- Reference frame 1
- Reference frame 2
- Reference frame 3

[Figure 599]

[Figure 600]

[Figure 601]

Generated frame 4

[Figure 602]

Generated frame 14

[Figure 603]

Generated frame 24

[Figure 604]

Generated frame 34

[Figure 605]

Generated frame 44

[Figure 606]

Generated frame 54

[Figure 607]

Generated frame 64

[Figure 608]

Generated frame 74

[Figure 609]

Generated frame 84

[Figure 610]

Generated frame 94

[Figure 611]

Generated frame 104

[Figure 612]

Generated frame 114

[Figure 613]

Generated frame 124

[Figure 614]

Generated frame 134

[Figure 615]

Generated frame 144

### Figure 12: Qualitative results of Video Generation from 3 reference frames at night. Our model imitated the blur of fast motion.

