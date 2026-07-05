# arXiv:2603.03143v2[cs.CV]28Jun2026

## Edit in 2D, Verify in 3D: Reinforcement Learning for Multi-view Consistent Scene Editing

[Figure 1]

Jiyuan Wang1,2,3, Chunyu Lin1, , Lei Sun2,✱, Zhi Cao1, Yuyang Yin1, Lang Nie4, Zhenlong Yuan2, Xiangxiang Chu2, Yunchao Wei1, Kang Liao3, and

Guosheng Lin3,

[Figure 2]

[Figure 3]

[Figure 4]

1BJTU, 2AMap Alibaba Group, 3NTU, 4CQUPT {wangjiyuan,cylin}@bjtu.edu.cn

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

①Motion ③Style

②Replacement

[Figure 14]

[Figure 15]

[Figure 16]

| |
|---|

[Figure 17]

[Figure 18]

[Figure 19]

|[Figure 20]|
|---|

[Figure 21]

| |
|---|

| |
|---|

|[Figure 22]|
|---|

| |
|---|

[Figure 23]

[Figure 24]

[Figure 25]

④Background ⑤Addition

| |
|---|

| |
|---|

|[Figure 26]|
|---|

| |
|---|

Fig. 1: We propose RL3DEdit, a novel RL-based model for single-pass 3D editing. Our method achieves high-quality results across diverse editing scenarios, including ①motion edits (cross arms), ②subject replacement (turn into Hulk), ③style transfer (colored pencil style), ④background changes (change to sandy beach), and ⑤challenging scene addition (add a brick wall with ivy).

Abstract. Leveraging the priors of 2D diffusion models for 3D editing has emerged as a promising paradigm. However, multi-view consistency remains challenging in edited results, and the extreme scarcity of paired 3D-consistent editing data makes supervised fine-tuning (SFT) impractical, despite its effectiveness for editing tasks. In this paper, we observe that, while generating multi-view consistent 3D content is highly challenging, verifying 3D consistency is tractable, naturally positioning reinforcement learning (RL) as a feasible solution. Motivated by this, we propose RL3DEdit, a single-pass framework driven by RL optimization with novel rewards derived from the 3D foundation model, VGGT. Specifically, we leverage VGGT’s robust priors learned from massive realworld data, feed the edited images into it, and utilize the output confidence maps and pose estimation errors as reward signals, effectively anchoring the 2D editing priors onto a 3D-consistent manifold via RL. Extensive experiments demonstrate that RL3DEdit achieves stable multiview consistency and outperforms state-of-the-art methods in editing

Work done during the internship at AMAP Alibaba Group, and NTU. Corresponding author; ✱Project Leader.

quality with high efficiency. To promote the development of 3D editing, we will release the code and model.

Keywords: 3D Editing· Reinforcement Learning

### 1 Introduction

3D scene editing plays a pivotal role in applications such as AR/VR and gaming, demanding both high-fidelity semantic manipulation and strict geometric coherence. To achieve this, generating multi-view consistent images via 2D editors, followed by fine-tuning 3D representations such as 3D Gaussian Splatting (3DGS), has emerged as a promising paradigm for 3D scene editing. Especially, the recent 2D editing models (e.g., FLUX-Kontext [32]) achieved remarkable progress in editing fidelity and multi-image correlated editing [60]. Despite this promising foundation, current 3D editing methods not only underexploit these powerful models, but also remain limited: (i) Geometric-conditioned methods [61,74] guide 3D consistency with depth maps of the source image, failing to handle edits involving geometric changes; (ii) Optimization-based approaches [23, 50] iteratively refine 3D representations with single-view edits, suffering from low efficiency and blurry artifacts due to 3D-inconsistent signals. (iii) Attentionbased models [11,57] reproject attention features across viewpoints, yet struggle to guarantee fine-grained geometric consistency.

Particularly, attention manipulation is similarly a suboptimal solution in the early stage of 2D editing [24], but is ultimately surpassed by data-driven supervision [32], suggesting that supervision remains the most effective pathway. However, the extreme scarcity of 3D-consistent editing paired data makes this approach infeasible. Recently, reinforcement learning (RL) [22] has demonstrated strong potential in advancing 2D editing models [40], offering a promising alternative to address the above challenges. Instead of relying on explicit supervision from carefully constructed datasets, RL algorithms optimize editing models using feedback from verifiable reward models (VRMs). In the 3D area, verifying multi-view consistency is significantly more tractable than generating consistent images, making RL a promising approach to acquire 3D consistency without massive paired data.

Motivated by this, the remaining key problem is to identify a robust 3D verifier that can effectively measure the consistency of editing results. In this paper, we leverage the 3D foundation model, VGGT [55], to serve as this verifier. Drawing an analogy to Score Distillation Sampling (SDS [46])—which leverages a frozen 2D diffusion model to assess image quality—we argue that a frozen VGGT, trained on massive real-world 3D data, can provide intrinsic feedback on multiview consistency. Through empirical analysis, we observe that its confidence maps, originally designed for error tolerance, can serve as an effective proxy for evaluating consistency across views. Furthermore, the camera pose prediction can also offer explicit feedback on viewpoint arrangement. To preserve the 2D editor’s editing fidelity, we additionally design an anchor strategy that aligns

Edit in 2D, Verify in 3D 3

edited outputs with high-quality single-view references. Unlike traditional 3D verifiers that are easily “reward-hacked” [59] by textureless or blurry images, the 3D foundation model acts as a robust, geometry-aware reward model by leveraging data-driven priors, thus providing stable guidance for the RL process.

Based on the above analysis, we propose RL3DEdit, a single-pass framework that augments a 2D editor’s 3D consistency prior with RL guided by geometryaware rewards. To satisfy the prerequisite of RL optimization, we explore base editors’ multi-image joint editing capabilities and select FLUX-Kontext, whose inherent cross-view attention lays the foundation for achieving consistency. During training, the RL process explores diverse editing candidates and evaluates their consistency via the VGGT reward model. The RL algorithm, GRPO [22], optimizes the editing model toward predicting 3D consistent results. At inference, the edited multi-view images are reconstructed into 3DGS to yield the final edited 3D scene.

Our method requires no per-scene/prompt fine-tuning and effectively preserves the powerful 2D foundation’s capabilities after RL optimization. Trained on limited samples, it successfully learns 3D consistency priors and shows promising generalization to unseen conditions. Moreover, RL3DEdit can handle geometrychanging instructions, and avoids iterative optimization, enabling single-pass inference that is over 2× faster than previous methods. Extensive experiments confirm that our approach achieves superior editing quality and multi-view consistency with high efficiency. Our contributions can be summarized as follows:

- – We propose a novel 3D editing RL framework, which effectively empowers 2D editors with 3D capabilities through a tractable 3D consistency verifier, thereby bypassing the scarcity of paired training data.
- – We identify that the 3D foundation model with data-driven priors, like VGGT, can serve as a superior verifier. Furthermore, we explore its usage and design tailored rewards to enforce geometric consistency and preserve editing quality.
- – Our optimization-free RL3DEdit model achieves SoTA 3D editing quality with high efficiency.

### 2 Related Work

###### 2.1 2D Image Editing Models

In the early stage, 2D editing faced a similar dilemma to current 3D editing: the scarcity of paired editing data. To address this, pioneering works manipulated the cross-attention maps of generative models [9,24], achieving fine-grained editing. Subsequently, with the advent of paired editing datasets, several methods began directly training models to follow explicit editing instructions [6,43,70], significantly improving editing quality. Recently, with the support of massive data, high-quality unified editing models have emerged [32,60], providing more powerful backbones for 3D editing.

- 2.2 3D Editing Models
- 3D editing can be broadly categorized into object editing and scene editing. Compared to objects, scenes are significantly more challenging due to the complex backgrounds and multiple entities.

- 3D Object Editing. Recent works [36,62,67] have achieved significant progress in 3D object editing by constructing dedicated datasets and fine-tuning on the voxel foundation model – Trellis [63]. However, scene editing is difficult to represent with voxels and poses greater challenges in data construction. 3D Scene Editing. Scene editing typically adopts 3DGS or NeRF [3] as the 3D representation. We categorize existing methods into four classes:

- (i) SDS-based methods [13,26,29,37,45,50,77] leverage diffusion priors via Score Distillation Sampling (SDS) to optimize 3D representations, but typically suffer from blurry textures and over-smoothing.
- (ii) Iterative optimization-based methods, pioneered by IN2N [23] and extended by [12,26,54,58,61,64,69], alternate between single-view editing and 3D optimization. However, the cross-view information is lacking in the editing process, which leads to inconsistent 3D representation optimization and prolonged editing time.
- (iii) Gaussian parameter manipulation methods [48, 72] directly modify 3DGS parameters via semantic grounding or parameter-change predictors distilled from

- 2D editors. While efficient, these methods struggle with action-based edits (e.g., “bowing head”), as such instructions are difficult to translate into explicit Gaussian parameter modifications.

- (iv) Multi-view consistent editing methods have become the dominant paradigm. Similar to 2D editing, early works [11,16,19,31,39,57,61] propagate, interact, or project attention features across views, yet accumulate alignment errors in geometrically discontinuous regions. The insufficient constraints also lead to residual fine-grained inconsistencies. Subsequent methods like EditSplat [10, 28, 34] employ multi-view fusion guidance, conditioning each newly edited view on adjacent/all previously edited ones. But they still produce visible artifacts under instructions that are difficult to quantify precisely (e.g., “open mouth”—but how wide?).

Furthermore, most aforementioned methods rely on InstructPix2Pix, strictly limiting their 2D editing upper bound. While Tinker [74] recently adopted the powerful FLUX-Kontext, it relies on depth-map guidance (restricting it to geometrypreserving edits), introduces complex video-model pipelines, and demands massive paired data (∼25K samples). In contrast, our RL3DEdit achieves superior generalization with only 5% of the data via RL optimization.

###### 2.3 Reinforcement Learning for 3D Tasks

To our knowledge, RL3DEdit is the first work to introduce RL into 3D editing. The RL paradigm, particularly GRPO [22], has achieved remarkable success in LLMs [21] and 2D editing [40], yet remains underexplored in the 3D domain and is mainly focused on 3D generation. Early works [56,66] train human-preference

[Figure 27]

###### Edit in 2D, Verify in 3D 5

[Figure 28]

[Figure 29]

[Figure 30]

Group

[Figure 31]

[Figure 32]

[Figure 33]

|Origin 3D<br><br>[Figure 34]<br><br>[Figure 35]|
|---|

…

[Figure 36]

[Figure 37]

[Figure 38]

{{𝐼 }   ,   ∪ 𝐼 }

[Figure 39]

[Figure 40]

###### VGGT

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

conf

[Figure 48]

𝑟

FLUX-Kontext

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

𝑟

[Figure 54]

[Figure 55]

conf

[Figure 56]

[Figure 57]

…

𝑟

𝐼

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

pre-edit

𝑇 = 𝑅 𝑡

[Figure 62]

[Figure 63]

𝑟

Offline 𝑅    , 𝑡̃ FLUX-Kontext

[Figure 64]

R

𝐼

optimization

After Finetuning

[Figure 65]

𝐼

[Figure 66]

###### Inference Edited 3D

Group Computation

| | |
|---|---|
|Replace the sweater with blue hoodie and turn him a clown.| |

RL3DEdit

…

3D Reconstruction

𝐴 𝐴 𝐴 … 𝐴

|𝐽 = 𝐽 − 𝛽D 𝜋 ∣ 𝜋|
|---|

Fig. 2: Pipeline of RL3DEdit. Section 3.1 details the pipeline.

reward models and employ RLHF to better align generated 3D assets with human intent. More recently, Nabla-R2D3 [41]/AR3D-R1 [53] adopt GRPO to enhance native 3D diffusion/autoregressive generation pipelines, improving texture detail and overall fidelity. Additionally, several works [8,44,71,76] leverage RL for 3D understanding and spatial layout, applying ensemble critics and physical feedback to enforce spatial and geometric constraints.

- 3 Methods

- 3.1 3D Editing Pipeline with Reinforcement Learning

As shown in Fig. 2, we construct a novel 3D editing method based on RL. Specifically, given a 3D asset to be edited, we first render it from M viewpoints to obtain {Im}Mm=1, and feed them simultaneously into a 2D editor (denoted as π) for joint multi-view editing. Ideally, at inference, the fine-tuned editor (RL3DEdit) produces multi-view consistent images in a single forward pass, which are then fed into 3DGS reconstruction to obtain the final edited 3D scene, yielding an efficient editing workflow.

Therefore, the core challenge is to equip the 2D editor with a 3D-consistency prior. We address this via RL optimization. As shown in the upper part of Fig. 2 (FLUX-Kontext branch), during training, the GRPO algorithm explores a group of G edited results (through G independent inference passes [40]), each containing M edited views {Im′ }Mm=1. To explicitly enforce both editing faithfulness and multi-view coherence, we first select an anchor view Ia′ and substitute it with our pre-edited anchor image I˜a (detailed in Sec. 3.4). Together with the other views, they are fed into a dedicated 3D-aware reward model (implemented via VGGT [55]), which is designed to assess multi-view consistency with rD,rP,rT

[Figure 67]

- Fig. 3: Comparison of 2D editing capabilities before and after RL fine-tuning. Left: Visual editing results. Right: Quantitative evaluation using VIEScore [30] on GEditBench-EN [42] (↑, detailed in Sec. 4.2). Both demonstrate that RL3DEdit successfully preserves the original 2D editing fidelity of FLUX-Kontext.

and editing quality with ra. These complementary rewards are combined to form the final composite reward Ri, which effectively guides our optimization toward consistent and high-quality 3D-aware editing. The rewards {Ri}Gi=1 are used to compute the relative advantage via: Ai = (Ri − mean({Rj}Gj=1))/std({Rj}Gj=1), and the model is optimized by maximizing the following objective, which encourages the model to produce high-reward outputs:

J(θ) = Jclip(θ) − β DKL(πθ∥πref),

where πθ/πref denote the fine-tuned/original 2D editors, Jclip(θ) is computed from Ai (see Appendix for details and the formal definition of GRPO). In this way, the model can learn 3D-consistency priors without curated paired supervision.

Compared with previous methods, our method offers two key advantages. On the one hand, verifying multi-view consistency only requires identifying geometric contradictions like ghosting artifacts. This is significantly more tractable than generating consistent images, which requires complex cross-view interactions and relies on scarce multi-view data. This asymmetry perfectly aligns with the RL paradigm, where a consistency verifier can naturally serve as the reward model. On the other hand, 3D editing, like its 2D counterpart, must handle arbitrary instructions on diverse scenes. Learning such general editing priors in 3D would require prohibitively large paired datasets that currently do not exist. Fortunately, modern 2D foundation editors naturally possess this capability. As shown in Fig. 3, the fine-tuned editor preserves its original 2D editing capability, suggesting that our method primarily augments the 2D model with 3D-consistency priors rather than reshaping it, further highlighting the effectiveness of our method.

In the following, we detail the three core components of our pipeline: the 2D editor (Sec. 3.2), the 3D-aware reward model (Sec. 3.3), and the reward design (Sec. 3.4).

[Figure 68]

Resize

FLUX-Kontext Qwen-Image-Edit Instruction-Pix2Pix

[Figure 69]

[Figure 70]

[Figure 71]

Swap the fur color of the two cats while keeping everything else unchanged.

- Fig. 4: Multi-image joint editing comparison. FLUX-Kontext and Qwen-Image-Edit successfully swap the fur colors, while InstructPix2Pix fails due to the lack of cross-view interaction. Moreover, InstructPix2Pix must resize images to low resolution, causing detail loss in multi-image scenarios.

###### 3.2 Multi-Image Joint Editing

Multi-view consistent editing requires effective cross-view interaction during the editing process. In the RL paradigm, optimization relies on exploring diverse outputs and reinforcing those with high rewards. If a 2D editor processes each view independently, the probability of producing 3D consistent multi-view images is essentially zero. Consequently, RL would have no successful samples to reinforce, causing the optimization to fail. Therefore, multi-image joint editing capability is a necessary prerequisite for RL to be effective.

Previous 3D editing methods typically adopt InstructPix2Pix [7] as their 2D backbone. However, its low processing resolution and local convolutional operations limit cross-image interaction. As shown in Fig. 4, given the instruction to swap the fur colors of two cats, InstructPix2Pix fails to access information from the other images and results in random color changes. In contrast, recent DiT-based models (e.g., FLUX-Kontext [32], Qwen-Image-Edit [68]) are naturally suited for multi-image editing, as their Transformer architecture enables global attention across all inputs. Specifically, after tokenizing K input images as {xk}Kk=1, they are concatenated along the sequence dimension as X = Concat(x1,...,xK) and processed with self-attention, enabling attention weights to be distributed across images. This inherent mechanism facilitates efficient cross-view interaction, enabling RL to leverage it to achieve 3D consistency. Based on this, we adopt FLUX-Kontext as our baseline and further validate the applicability to Qwen-Image-Edit in ablation studies (Sec. 4.3).

###### 3.3 Multi-View Consistent Verification

With the RL framework and a promising foundation editor, the remaining question is how to robustly verify the 3D consistency of edited images. In this paper, we choose a 3D foundation model, VGGT [55]. This is inspired by Score Distillation Sampling (SDS [47]), which employs the 2D foundation model Stable Diffusion (SD [49]) to supervise image quality. SD is trained on high-quality, large-scale images, so feeding it low-quality images produces large loss feedback. Analogously, VGGT is trained on millions of real-world 3D data samples, and

[Figure 72]

- Fig. 5: Empirical analysis of VGGT’s depth confidence under progressively degraded 3D consistency. ①-⑤ visualize VGGT confidence predictions for the same set of 9 views, where individual views are gradually replaced by edited versions. ⑥ reveals a near-linear correlation between consistency degradation and average confidence. This validates VGGT as the multi-view consistency verifier. Detailed analysis is in Sec. 3.3. feeding multi-view inconsistent edited images into VGGT can also produce meaningful corresponding feedback. Below, we explore this feedback form and employ VGGT as our reward model. Experiments (Sec. 4.3) show that such data-priorbased verification is better than traditional methods (e.g., reprojection warping, Structure-from-Motion) in avoiding “reward-hacking”.

- 3D Foundation Model. Formally, given N views (Ii)Ni=1 of the same 3D

scene, VGGT learns a Transformer f such that f (Ii)Ni=1 = gi,Di,Pi,τi Ni=1, where gi,Di,Pi,τi denote the camera parameters, depth map, point map, and tracking features of the i-th view, respectively. During training, to handle varying regional difficulty, VGGT jointly predicts confidence maps with error tolerance. The loss function is represented as follows:

Lgeo = ∥confgeo ⊙(geˆo−geo)∥1 +∥confgeo ⊙(∇geˆo−∇geo)∥1 −α ∥log confgeo∥1, where confgeo denotes geometric confidence, including the uncertainty of depth (confD) and point (confP) predictions.

Confidence Map Analysis. To investigate the feedback, as illustrated in Fig. 5, we render 9 views {Ii}9i=1 from a 3D asset and independently edit them to obtain {Ii′}. We then gradually replace Ii with its edited counterpart Ii′ to disrupt multi-view consistency, and observe the changes in VGGT’s output. Our analysis reveals a strong correlation between the predicted confidence and 3D consistency: ① When all views are originally 3D-consistent, the depth confidence is uniformly high. ② Replacing the background of a single view, its background confidence drops significantly while the foreground character remains unaffected.

- ③ Further adding a pair of glasses in another view reduces the confidence in the corresponding eye region. ④ Replacing the person with Iron Man in a third view leads to globally low confidence, as both the background and foreground now exhibit cross-view inconsistencies. ⑤ Finally, when 8 out of 9 views are replaced, although all inputs conceptually belong to the same scene, they lack mutual 3D

consistency, resulting in very low confidence scores. To quantitatively validate this, we select 100 sets of 9-view data and compute the average confidence under varying replacement ratios. As shown in ⑥, we observe that the predicted confidence decays almost linearly as consistency decreases. This empirical result confirms that VGGT’s confidence maps are a reliable and interpretable indicator of multi-view consistency. Therefore, VGGT’s confidence can effectively serve as an indicator for 3D consistency and act as a reward.

###### 3.4 Overview of Reward Model

Geometric Rewards. As discussed above, we use the average depth and point confidence as geometric rewards:

M

M

1 M

1 M

mean confmD , rP =

mean confmP . (1)

rD =

m=1

m=1

Relative Pose Reward. Beyond multi-view consistency, we also consider the arrangement of camera viewpoints. Considering that editing may alter the absolute camera viewpoint, but the inter-view relationship is still preserved, we use the relative poses between adjacent views to measure camera perspective alignment. Let the VGGT-predicted extrinsics of view m be Tm = [Rm |tm], and define the relative transform Tmrel = Tm+1Tm−1 = [Rmrel |trelm ]. We normalize the translation as t˜relm = trelm /∥trelm ∥2 and use the reference (GT) relative pose (Tmrel)∗, yielding the reward:

M−1

1 M − 1

∥Rmrel − (Rmrel)∗∥2F + ∥t˜relm − (t˜relm )∗∥22 . (2)

rT = exp −

m=1

Anchor Reward. As discussed in Sec. 3.1, RL3DEdit preserves the high editing quality after fine-tuning. This not only benefits from the RL fine-tuning paradigm [40], but also from the proposed anchor reward. We pre-compute singleimage editing results for all views offline using FLUX-Kontext, followed by light quality filtering (empirically, >98% are directly usable). Although these results I˜ are mutually 3D-inconsistent, they all satisfy FLUX-Kontext’s high-quality distribution (i.e., correct semantics, detail preservation, etc.). So we can leverage them independently to guide the RL optimization and preserve FLUX-Kontext’s editing fidelity in 3D scenarios.

During training, we randomly sample an anchor index a ∈ {1,...,M} and retrieve I˜a from the offline results (one GRPO-group shares the same index to ensure fair group-reward comparison; randomness is at the sample level). To

evaluate the editing quality of {Im′ }Mm=1, we consider two cases: (1) For the anchor view Ia′ , we directly measure the editing error:

ra = exp − λLLPIPS(Ia′ ,I˜a) , (3)

where LLPIPS(·,·) denotes the perceptual similarity [73], which aligns well with human perception to prevent edit quality degradation. (2) For other views, we re-

place Ia′ with I˜a in the multi-view input, feeding {I1′,...,Ia′−1,I˜a,Ia′+1,...,IM′ }

into VGGT to evaluate consistency. In this way, Ii′̸=a are optimized for higher geometric consistency rewards while Ia′ is optimized for higher anchor reward, ensuring all views converge toward correct semantics and visual details, thereby preserving FLUX-Kontext’s editing fidelity. Finally, the i-th overall reward in the GRPO-group is defined as:

Ri = wDrD + wPrP + wTrT + wara, (4) which we plug into our framework to optimize FLUX-Kontext.

- 4 Experiments

- 4.1 Implementation Details

We adopt FLUX-Kontext-dev [32] as our baseline and fine-tune it using LoRA with rank 32 and alpha 32. During training, we set M = 9 and wD = wP = wT = wa = 0.25. Following Flow-GRPO [40], we employ Stochastic Differential Equations (SDE) to enhance exploration randomness, setting the SDE noise level to 0.8 and the group size to 16. However, unlike the 6-step denoising exploration used in prior work [40], our experiments show that 3D consistency demands higher image fidelity, so we adopt a 12-step exploration. To accelerate convergence, we incorporate improvements from MixGRPO [35], setting the window size to 4.

For training data, we collect 8 scenes from the IN2N [23], BlendedMVS [65], and Mip-NeRF360 [5] datasets, and construct 7∼9 editing prompts per scene using a VLM [14], yielding 70 prompts in total (Appendix for details). For each scene, we sample 18∼23 sets of M-view images as training samples, where the M=9 views are evenly sampled around the scene to ensure sufficient visual overlap for consistency evaluation, yielding a total of 1,319 samples. Training was conducted for one epoch on an NVIDIA RTX Pro 6000 GPU and took 42 hours. In inference, we employ 3DGS to reconstruct the edited 3D scene from the multiview edited images (Appendix for details).

###### 4.2 Comparison Analysis

Comparative Models. We compare RL3DEdit with the open-source SoTA 3D editing methods, including DGE [11], EditSplat [34], and GaussCtrl [61]. We note that previous methods adopt InstructPix2Pix [6] as the 2D editing model; however, as discussed earlier, its limitations in multi-image joint editing make it unsuitable as our baseline. For a fair comparison, we re-implement the strongest baseline, EditSplat, under the same backbone FLUX-Kontext (see Appendix for implementation details) and compare it with our proposed method. For each scene, we use identical text prompts for editing, then render the edited 3D scene from the same camera poses to obtain results from different methods at consistent novel viewpoints.

Quantitative Comparison and Metrics. We evaluate four dimensions: (i) VIEScore [30], a VLM-based metric (GPT-4.1 [1]) that jointly evaluates

- Table 1: Quantitative comparison with SoTA methods. Ph-Loss stands for photometric reprojection loss, ↑ / ↓ indicates higher/lower is better. Best results are highlighted in bold. Editing time is tested on an RTX Pro 6000 GPU. Due to space constraints, per-split metrics (in-distribution/zero-shot) and user study are provided in Appendix.

VIEScore↑ CLIP-dir↑ Ph-Loss↓ Avg. Editing (Novel View Synthesis) (Edit-Results) Time↓

Methods

DGE [11] 2.81 0.116 0.086 4min GaussCtrl [61] 2.37 0.096 0.077 12min EditSplat [34] 2.72 0.121 0.081 3.5min EditSplat w/ FLUX-Kontext 3.23 0.125 0.082 40min

RL3DEdit (Ours) 5.48 0.147 0.076 1.5min

instruction-following and visual quality—we pioneer its usage in 3D editing to provide reproducible, objective assessment free from the subjective variance of user studies; (ii) CLIP directional similarity [17], following prior work [34,50]; (iii) photometric reprojection loss (Ph-Loss) [18] for multi-view consistency (definition in Appendix); and (iv) average editing time. The test data includes novel views (70 cases), unseen instructions (16 cases), and new scenes (14 cases), totaling 100 test cases (Appendix for detailed split). We note that all prior methods perform per-scene optimization, inherently using the same scene and instruction for both training and testing. Our 70 cases follow the same protocol for fair comparison. For the additional 30 cases, only RL3DEdit operates in a zeroshot setting, while others still optimize on corresponding data. As shown in Tab. 1, RL3DEdit achieves the best performance across all dimensions. It not only achieves a remarkable leap in editing fidelity and semantic alignment (VIEScore of 5.48 vs. 3.23), but also maintains the lowest Ph-Loss for rigorous 3D consistency. Crucially, this high-quality editing is achieved in just 1.5 minutes—over 2× faster than traditional pipelines and over 20× faster than the other FLUX-based baseline.

Qualitative Comparison. Metrics alone cannot fully capture editing quality. We therefore present extensive visual results across Figs. 1, 2, 6, 7, and 8, covering diverse scenes and instructions; additional results are provided in the supplementary video. Specifically, Fig. 6 presents qualitative comparisons under some challenging instructions. (Row 1) EditSplat/GaussCtrl relies on depth warping/guiding, causing severe failures under geometric-changing prompts. DGE misinterprets the semantics. Only RL3DEdit successfully places the ball in front of the bear. (Row 2) Compared to the blurry results of other methods, RL3DEdit achieves a more realistic Minecraft-style appearance. (Row 3) Motion editing often causes artifacts due to the difficulty in quantifying action magnitude. DGE and EditSplat exhibit artifacts around the mouth, while GaussCtrl alters subject identity. Only RL3DEdit produces correct, high-quality results. (Row 4) For winter scene conversion, DGE merely whitens the scene; GaussCtrl generates snow but alters the dinosaur; EditSplat suffers ghost-artifacts, as massive multi-view inconsistency causes warping failure. Only RL3DEdit achieves semantically accurate editing. These improvements on challenging instructions vividly

[Figure 73]

[Figure 74]

12 Jiyuan Wang et al.

[Figure 75]

EditSplat GaussCtrl w/FLUX-Kontext

DGE RL3DEdit

Origin

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

|[Figure 84]|
|---|

| |
|---|

Add a red rubber ball next to the stone base of the bear statue

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

| |
|---|

|[Figure 91]|
|---|

| |
|---|

|[Figure 92]|
|---|

|[Figure 93]|
|---|

[Figure 94]

Transform the person into a blocky Minecraft-style character.

|[Figure 95]|
|---|

|[Figure 96]|
|---|

[Figure 97]

[Figure 98]

|[Figure 99]|
|---|

| |
|---|

|[Figure 100]|
|---|

[Figure 101]

[Figure 102]

[Figure 103]

Make the man open his mouth

| |
|---|

|[Figure 104]|
|---|

|[Figure 105]|
|---|

[Figure 106]

|[Figure 107]|
|---|

|[Figure 108]|
|---|

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Change the scene to winter and cover the ground, base, and dinosaur statue with white snow

- Fig. 6: Qualitative comparison. RL3DEdit achieves much higher 3D editing quality than other methods, in addition (1st row)/replacement (2nd row)/motion (3rd row)/style (4th row) editing.

illustrate the performance advantages shown in Tab. 1, further demonstrating the effectiveness of our method.

###### 4.3 Ablation Study

Due to computational constraints, the ablation models are all trained and tested on face scenes with over 200 samples to validate the effectiveness of each component. Experiments confirm that this data scale is sufficient to draw conclusions. Additional ablations on other design details (e.g., anchor selection) are provided in the Appendix.

Effect of Rewards. Rewards rD and rP (depth and point confidence) both constrain 3D consistency, so we ablate them together. ① Removing rD and rP causes significant degradation in both novel-view editing quality and reprojection loss (Tab. 2). Fig. 7 shows severe ghosting artifacts due to inconsistency, demonstrating the importance of VGGT consistency rewards. ② Without rT, subtle viewpoint shifts occur, which become evident in reprojection evaluation using GT extrinsics (Tab. 2). Fig. 7 shows displacement in wall details. ③ Without the anchor image to preserve editing priors, outputs degrade toward over-smoothed results (Fig. 7), primarily because 3D consistency is easier to achieve with low-

- Table 2: Quantitative ablation study on reward components and alternative designs. RL3DEdit* denotes the model trained with the same method but only on face data. The model with rwarp produces blurry images that achieve high consistency but not the desired editing quality. The extension to Qwen-Image-Edit demonstrates that stronger 2D editing models yield better results.

Methods VIEScore↑ Ph-Loss↓

- ① w/o (rD, rP) 2.11 0.193
- ② w/o rT 4.77 0.131
- ③ w/o ra 4.34 0.091

- ④ replaced w/ rSfM 0.97 0.201
- ⑤ replaced w/ rwarp 1.41 0.065

- ⑥ Qwen-Image-Edit 5.43 0.079
- ⑦ RL3DEdit* 5.26 0.077

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Origin w/o 𝑟 𝑟 w/o 𝑟 w/o 𝑟 𝑟 𝑟 QwenEdit

RL3DEdit*

[Figure 119]

[Figure 120]

[Figure 121]

#### ① ② ③ ④ ⑤ ⑥ ⑦

| |
|---|

[Figure 122]

| |
|---|

- Fig. 7: Qualitative ablation results. To better illustrate the causes of degradation, all variants use the same viewpoint for edited results except w/o (rD, rP), which uses newly rendered views. In w/o rT, red dashed lines highlight subtle viewpoint shifts compared to the original.

frequency details, causing RL to optimize in this direction and fail to preserve FLUX-Kontext’s editing quality. Effect of Alternative Consistency Verifiers. We compare VGGT against two widely used 3D-consistency measures to validate its necessity as a reward.

- ④ SfM-based reward. Structure-from-Motion requires no learned priors but relies on sparse feature matching (definition in Appendix). During RL training, the model quickly learns to produce textureless outputs that yield few matchable keypoints, trivially satisfying the SfM consistency check while destroying editing quality (Fig. 7, Tab. 2).
- ⑤ Reprojection warping reward. We compute photometric reprojection loss (PhLoss) [18] using VGGT-predicted depth and GT poses as the reward signal (definition in Appendix). Tab. 2 shows that the resulting model achieves the lowest Ph-Loss among all variants, yet produces severely blurred outputs (Fig. 7), confirming that Ph-Loss is easily “reward-hacked” by low-frequency images. It is worth noting that we still adopt Ph-Loss as an evaluation metric in Tab. 1,

[Figure 123]

[Figure 124]

[Figure 125]

###### 14 Jiyuan Wang et al.

Make the bear eat apple

Convert Lego loader to metal material

Clear the background objects

Replace stone pedestal with a crystal base

Put cap on person

[Figure 126]

[Figure 127]

| |
|---|

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

Origin

|[Figure 132]|
|---|

|[Figure 133]|
|---|

|[Figure 134]|
|---|

| |
|---|

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

| |
|---|

[Figure 140]

Edited

|[Figure 141]|
|---|

|④|
|---|

|[Figure 142]|
|---|

| |
|---|

①

②

⑤

③

- Fig. 8: Zero-shot editing results, where ①,②,③ are new instructions; ④,⑤ are new scenes. More results are available in the supplementary video.

as it remains effective for measuring consistency among sharp images, while any blurriness is penalized by VIEScore, ensuring the fairness of our comparison.

Compared to these traditional methods, VGGT is trained on massive realworld 3D images. As a reward function, we think it can optimize consistency while preserving the capability to output high-quality images (further discussed in Appendix). This validates the necessity of using data-driven priors like VGGT as reward models.

Extensions on Qwen-Image-Edit. Both results in Tab. 2 ⑥ and Fig. 7 ⑥ demonstrate that our proposed framework can be further enhanced with more powerful editing models. With the rapid advancement of 2D editing models, our framework has strong potential.

Zero-Shot Generalization. As shown in Fig. 8, by preserving the priors of FLUX-Kontext, our method can generalize to unseen instructions and scenes.

### 5 Limitations and Future Work

Limitations of the 2D Backbone. Since RL3DEdit is fine-tuned on a 2D editor, its performance is bounded by the backbone’s inherent constraints. The primary limitation stems from attention sequence length: multi-view images share the same token capacity, forcing a trade-off between the number of views and per-image resolution. However, this is not an inherent defect of our framework. A natural extension is to use the anchor image as guidance and generate edited images in batches to cover more viewpoints, which we leave as future work. Moreover, with the rapid advancement of efficient attention mechanisms—especially streaming and causal attention, which have been successfully applied to longsequence 3D perception [33]—context length continues to expand. As demonstrated in Sec. 4.3, our method can transfer to other backbones, and this limitation is also expected to diminish as foundation models evolve.

Large Structural Deformations. RL3DEdit may struggle with extremely drastic non-rigid deformations (e.g., “make the person bow his head”), where the ambiguity in action magnitude leads to subtle multi-view discrepancies that compromise 3D reconstruction. We note that all baselines also fail on such cases,

and enhancing generative priors for severe non-rigid deformations remains an open direction.

Training Scale. Our training scale is limited, primarily due to the computational overhead of GRPO—each sample requires exploring 16 candidate groups, with each group demanding a 12-step inference pass, resulting in ∼2 days of training under our current setup. Nevertheless, RL3DEdit already achieves superior performance, and we believe the method can further improve with increased training scale.

### 6 Conclusion

We present RL3DEdit, an efficient 3D scene editing framework based on RL. Our core insight is that while generating 3D-consistent images is challenging, verifying 3D consistency is tractable, making RL an ideal solution. Building on this, we leverage the 3D foundation model VGGT to construct geometryaware rewards and employ GRPO to effectively anchor the 2D editor’s prior onto the 3D consistency manifold. Experiments demonstrate that RL3DEdit achieves superior editing quality and multi-view consistency with minimal training data, while delivering over 2× speedup compared to existing methods. Our framework generalizes well and seamlessly transfers to other 2D editing models, offering a new paradigm for future 3D editing research.

### Acknowledgements

This work was supported by the National Natural Science Foundation of China (NSFC) under Grant (62573039, U2441242).

### References

- 1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)
- 2. Bao, Y., Ding, T., Huo, J., Li, W., Gao, Y.: Fisn: Finding spatial neighborhoods for generalizable novel view synthesis. IEEE Transactions on Visualization and Computer Graphics (2026)
- 3. Bao, Y., Ding, T., Huo, J., Li, W., Li, Y., Gao, Y.: Insertnerf: Instilling generalizability into nerf with hypernet modules. In: International Conference on Learning Representations (2024)
- 4. Bao, Y., Liao, J., Huo, J., Gao, Y.: Distractor-free generalizable 3d gaussian splatting. In: The Fourteenth International Conference on Learning Representations

(2026), https://openreview.net/forum?id=G33Iemmj3Z

- 5. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mip-nerf 360: Unbounded anti-aliased neural radiance fields (2022), https://arxiv.org/ abs/2111.12077
- 6. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions (2023), https://arxiv.org/abs/2211.09800

- 7. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18392–18402 (2023)
- 8. Bucher, M.J., Armeni, I.: Respace: Text-driven autoregressive 3d indoor scene synthesis and editing (2025), https://arxiv.org/abs/2506.02459
- 9. Chefer, H., Alaluf, Y., Vinker, Y., Wolf, L., Cohen-Or, D.: Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models (2023), https://arxiv.org/abs/2301.13826
- 10. Chen, J.K., Bulò, S.R., Müller, N., Porzi, L., Kontschieder, P., Wang, Y.X.: Consistdreamer: 3d-consistent 2d diffusion for high-fidelity scene editing (2024), https://arxiv.org/abs/2406.09404
- 11. Chen, M., Laina, I., Vedaldi, A.: Dge: Direct gaussian 3d editing by consistent multi-view editing. arXiv preprint arXiv:2404.18929 (2024)
- 12. Chen, Y., Chen, Z., Zhang, C., Wang, F., Yang, X., Wang, Y., Cai, Z., Yang, L., Liu, H., Lin, G.: Gaussianeditor: Swift and controllable 3d editing with gaussian splatting (2023), https://arxiv.org/abs/2311.14521
- 13. Cheng, X., Yang, T., Wang, J., Li, Y., Zhang, L., Zhang, J., Yuan, L.: Progressive3d: Progressively local editing for text-to-3d content creation with complex semantic prompts (2024), https://arxiv.org/abs/2310.11784
- 14. Comanici, G., Bieber, E., Schaekermann, M., et al., I.P.: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities (2025), https://arxiv.org/abs/2507.06261
- 15. Dang, J., Wu, X., Yan, H., Zheng, H., Zheng, W.S., Lai, J., Hu, B., Chua, T.S.: Decoupled seg tokens make stronger reasoning video segmenter and grounder. IEEE Transactions on Pattern Analysis and Machine Intelligence (2026)
- 16. Dong, J., Wang, Y.X.: Vica-nerf: View-consistency-aware 3d editing of neural radiance fields (2024), https://arxiv.org/abs/2402.00864
- 17. Gal, R., Patashnik, O., Maron, H., Chechik, G., Cohen-Or, D.: Stylegan-nada: Clip-guided domain adaptation of image generators (2021), https://arxiv.org/ abs/2108.00946
- 18. Godard, C., Aodha, O.M., Firman, M., Brostow, G.: Digging into self-supervised monocular depth estimation (2019), https://arxiv.org/abs/1806.01260
- 19. Gomel, E., Wolf, L.: Diffusion-based attention warping for consistent 3d scene editing (2024), https://arxiv.org/abs/2412.07984
- 20. Gong, N., Li, Z., Dong, S., Bai, H., Ying, W., Wang, X., Fu, Y.: Sculpting features from noise: Reward-guided hierarchical diffusion for task-optimal feature transformation. arXiv preprint arXiv:2505.15152 (2025)
- 21. Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., Zhu, e.a.: Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature 645(8081), 633–638 (Sep 2025). https://doi.org/10.1038/s41586-025-09422-z, http: //dx.doi.org/10.1038/s41586-025-09422-z
- 22. Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., Zhu, Q.e.a.: Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature 645(8081), 633–638 (Sep 2025). https://doi.org/10.1038/s41586-025-09422-z, http: //dx.doi.org/10.1038/s41586-025-09422-z
- 23. Haque, A., Tancik, M., Efros, A.A., Holynski, A., Kanazawa, A.: Instruct-nerf2nerf: Editing 3d scenes with instructions. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 19740–19750 (2023)
- 24. Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-Or, D.: Prompt-to-prompt image editing with cross attention control (2022), https:// arxiv.org/abs/2208.01626

- 25. Jiang, L., Mao, Y., Xu, L., Lu, T., Ren, K., Jin, Y., Xu, X., Yu, M., Pang, J., Zhao, F., et al.: Anysplat: Feed-forward 3d gaussian splatting from unconstrained views. ACM Transactions on Graphics (TOG) 44(6), 1–16 (2025)
- 26. Kamata, H., Sakuma, Y., Hayakawa, A., Ishii, M., Narihira, T.: Instruct 3d-to3d: Text instruction guided 3d-to-3d conversion (2023), https://arxiv.org/abs/ 2303.15780
- 27. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics 42(4) (July 2023), https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- 28. Khalid, U., Iqbal, H., Farooq, A., Hua, J., Chen, C.: 3dego: 3d editing on the go!

(2024), https://arxiv.org/abs/2407.10102

- 29. Kim, J., Lee, S., Shin, J., Choi, J., Shim, H.: Dreamcatalyst: Fast and high-quality 3d editing via controlling editability and identity preservation (2025), https:// arxiv.org/abs/2407.11394
- 30. Ku, M., Jiang, D., Wei, C., Yue, X., Chen, W.: Viescore: Towards explainable metrics for conditional image synthesis evaluation (2024), https://arxiv.org/ abs/2312.14867
- 31. Kwon, G., Park, J., Ye, J.C.: Unified editing of panorama, 3d scenes, and videos through disentangled self-attention injection (2024), https://arxiv.org/abs/ 2405.16823
- 32. Labs, B.F., Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., English, J., English, Z., Esser, P., Kulal, S., Lacey, K., Levi, Y., Li, C., Lorenz, D., Müller, J., Podell, D., Rombach, R., Saini, H., Sauer, A., Smith, L.: Flux.1 kontext: Flow matching for in-context image generation and editing in latent space

(2025), https://arxiv.org/abs/2506.15742

- 33. Lan, Y., Luo, Y., Hong, F., Zhou, S., Chen, H., Lyu, Z., Yang, S., Dai, B., Loy, C.C., Pan, X.: STream3R: Scalable sequential 3d reconstruction with causal transformer

(2025), https://arxiv.org/abs/2508.10893

- 34. Lee, D.I., Park, H., Seo, J., Park, E., Park, H., Baek, H.D., Shin, S., Kim, S., Kim, S.: Editsplat: Multi-view fusion and attention-guided optimization for viewconsistent 3d scene editing with 3d gaussian splatting (2025), https://arxiv.org/ abs/2412.11520
- 35. Li, J., Cui, Y., Huang, T., Ma, Y., Fan, C., Yang, M., Zhong, Z., Bo, L.: Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde (2025), https://arxiv. org/abs/2507.21802
- 36. Li, L., Huang, Z., Feng, H., Zhuang, G., Chen, R., Guo, C., Sheng, L.: Voxhammer: Training-free precise and coherent 3d editing in native 3d space (2025), https: //arxiv.org/abs/2508.19247
- 37. Li, Y., Dou, Y., Shi, Y., Lei, Y., Chen, X., Zhang, Y., Zhou, P., Ni, B.: Focaldreamer: Text-driven 3d editing via focal-fusion assembly (2023), https://arxiv. org/abs/2308.10608
- 38. Li, Z., Yan, H., Li, S., Luo, K., Lu, L., Yang, X., Lin, W.: Diffpcn: Latent diffusion model based on multi-view depth images for point cloud completion. arXiv preprint arXiv:2509.23723 (2025)
- 39. Liao, K., Wu, S., Wu, Z., Jin, L., Wang, C., Wang, Y., Wang, F., Li, W., Loy, C.C.: Thinking with camera: A unified multimodal model for camera-centric understanding and generation. arXiv preprint arXiv:2510.08673 (2025)
- 40. Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., Ouyang, W.: Flow-grpo: Training flow matching models via online rl (2025), https://arxiv.org/abs/2505.05470

- 41. Liu, Q., Liu, Z., Zhang, D., Jia, K.: Nabla-r2d3: Effective and efficient 3d diffusion alignment with 2d rewards (2025), https://arxiv.org/abs/2506.15684
- 42. Liu, S., Han, Y., Xing, P., Yin, F., Wang, R., Cheng, W., Liao, J., Wang, Y., Fu, H., Han, C., Li, G., Peng, Y., Sun, Q., Wu, J., Cai, Y., Ge, Z., Ming, R., Xia, L., Zeng, X., Zhu, Y., Jiao, B., Zhang, X., Yu, G., Jiang, D.: Step1x-edit: A practical framework for general image editing (2025), https://arxiv.org/abs/2504.17761
- 43. Pan, X., Dong, L., Huang, S., Peng, Z., Chen, W., Wei, F.: Kosmos-g: Generating images in context with multimodal large language models (2024), https://arxiv. org/abs/2310.02992
- 44. Pan, Z., Liu, H.: Metaspatial: Reinforcing 3d spatial reasoning in vlms for the metaverse (2025), https://arxiv.org/abs/2503.18470
- 45. Park, J., Kwon, G., Ye, J.C.: Ed-nerf: Efficient text-guided editing of 3d scene with latent space nerf (2024), https://arxiv.org/abs/2310.02712
- 46. Poole, B., Jain, A., Barron, J.T., Mildenhall, B.: Dreamfusion: Text-to-3d using 2d diffusion (2022), https://arxiv.org/abs/2209.14988
- 47. Poole, B., Jain, A., Barron, J.T., Mildenhall, B.: Dreamfusion: Text-to-3d using

- 2d diffusion (2022), https://arxiv.org/abs/2209.14988

48. Qin, H., Sun, Y., Wang, M., Kong, M., Lu, M., Zhu, Q.: Variation-aware flexible

- 3d gaussian editing (2026), https://arxiv.org/abs/2602.11638

- 49. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models (2022), https://arxiv.org/abs/ 2112.10752
- 50. Shu, Z., Yu, J., Chao, K., Xin, S., Liu, L.: Gaussedit: Adaptive 3d scene editing with text and image prompts. IEEE Transactions on Visualization and Computer Graphics 31(10), 7769–7780 (Oct 2025). https://doi.org/10.1109/tvcg.2025. 3556745, http://dx.doi.org/10.1109/TVCG.2025.3556745
- 51. Sun, Y., Cheng, H., Lu, C., Li, Z., Wu, M., Lu, H., Zhu, J.: Hyperpoint: Multimodal 3d foundation model in hyperbolic space. Pattern Recognition (2025)
- 52. Sun, Y., Zhu, J., Cheng, H., Lu, C., Yang, Z., Chen, L., Wang, Y.: Align then adapt: Rethinking parameter-efficient transfer learning in 4d perception. IEEE Transactions on Multimedia (2026)
- 53. Tang, Y., Guo, Z., Zhu, K., Zhang, R., Chen, Q., Jiang, D., Liu, J., Zeng, B., Song, H., Qu, D., Bai, T., Xu, D., Zhang, W., Zhao, B.: Are we ready for rl in text-to3d generation? a progressive investigation (2025), https://arxiv.org/abs/2512. 10949
- 54. Wang, B., Dutt, N.S., Mitra, N.J.: Proteusnerf: Fast lightweight nerf editing using 3d-aware image context (2024), https://arxiv.org/abs/2310.09965
- 55. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: VGGT: Visual geometry grounded transformer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2025)
- 56. Wang, W., Xu, H., Yang, Y., Liu, Z., Meng, J., Wang, H.: Mvreward: Better aligning and evaluating multi-view diffusion models with human preferences (2024), https://arxiv.org/abs/2412.06614
- 57. Wang, Y., Yi, X., Wu, Z., Zhao, N., Chen, L., Zhang, H.: View-consistent 3d editing with gaussian splatting (2025), https://arxiv.org/abs/2403.11868
- 58. Wang, Y., Yi, X., Wu, Z., Zhao, N., Chen, L., Zhang, H.: View-consistent 3d editing with gaussian splatting (2025), https://arxiv.org/abs/2403.11868
- 59. Weng, L.: Reward hacking in reinforcement learning. lilianweng.github.io (Nov 2024), https://lilianweng.github.io/posts/2024-11-28-reward-hacking/

- 60. Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., ming Yin, S., Bai, S., Xu, X., Chen, Y., Chen, Y., Tang, Z., Zhang, Z., Wang, Z., Yang, A., Yu, B., Cheng, C., Liu, D., Li, D., Zhang, H., Meng, H., Wei, H., Ni, J., Chen, K., Cao, K., Peng, L., Qu, L., Wu, M., Wang, P., Yu, S., Wen, T., Feng, W., Xu, X., Wang, Y., Zhang, Y., Zhu, Y., Wu, Y., Cai, Y., Liu, Z.: Qwen-image technical report (2025), https://arxiv.org/abs/2508.02324
- 61. Wu, J., Bian, J.W., Li, X., Wang, G., Reid, I., Torr, P., Prisacariu, V.A.: Gaussctrl: Multi-view consistent text-driven 3d gaussian splatting editing (2024), https:// arxiv.org/abs/2403.08733
- 62. Xia, R., Tang, Y., Zhou, P.: Towards scalable and consistent 3d editing (2025), https://arxiv.org/abs/2510.02994
- 63. Xiang, J., Lv, Z., Xu, S., Deng, Y., Wang, R., Zhang, B., Chen, D., Tong, X., Yang, J.: Structured 3d latents for scalable and versatile 3d generation (2025), https://arxiv.org/abs/2412.01506
- 64. Xiao, H., Chen, Y., Huang, H., Xiong, H., Yang, J., Prasad, P., Zhao, Y.: Localized gaussian splatting editing with contextual awareness. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). p. 5207–5217. IEEE (Feb 2025). https://doi.org/10.1109/wacv61041.2025.00509, http://dx.doi. org/10.1109/WACV61041.2025.00509
- 65. Yao, Y., Luo, Z., Li, S., Zhang, J., Ren, Y., Zhou, L., Fang, T., Quan, L.: Blendedmvs: A large-scale dataset for generalized multi-view stereo networks (2020), https://arxiv.org/abs/1911.10127
- 66. Ye, J., Liu, F., Li, Q., Wang, Z., Wang, Y., Wang, X., Duan, Y., Zhu, J.: Dreamreward: Text-to-3d generation with human preference (2024), https://arxiv.org/ abs/2403.14613
- 67. Ye, J., Xie, S., Zhao, R., Wang, Z., Yan, H., Zu, W., Ma, L., Zhu, J.: Nano3d: A training-free approach for efficient 3d editing without masks (2025), https: //arxiv.org/abs/2510.15019
- 68. Yin, F., Liu, S., Han, Y., Wang, Z., Xing, P., Wang, R., Cheng, W., Wang, Y., Li, A., Yin, Z., Chen, P., Zhang, X., Jiang, D., Zeng, X., Yu, G.: Reasonedit: Towards reasoning-enhanced image editing models (2025), https://arxiv.org/abs/2511. 22625
- 69. Yu, L., Xiang, W., Han, K.: Edit-diffnerf: Editing 3d neural radiance fields using 2d diffusion model (2023), https://arxiv.org/abs/2306.09551
- 70. Yu, Q., Chow, W., Yue, Z., Pan, K., Wu, Y., Wan, X., Li, J., Tang, S., Zhang, H., Zhuang, Y.: Anyedit: Mastering unified high-quality image editing for any idea. arXiv preprint arXiv:2411.15738 (2024)
- 71. Yu, T., Li, X., Shen, Y., Liu, Y., Lourentzou, I.: Core3d: Collaborative reasoning as a foundation for 3d intelligence (2025), https://arxiv.org/abs/2512.12768
- 72. Zhang, Q., Xu, Y., Wang, C., Lee, H.Y., Wetzstein, G., Zhou, B., Yang, C.: 3ditscene: Editing any scene via language-guided disentangled gaussian splatting

(2024), https://arxiv.org/abs/2405.18424

- 73. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric (2018), https://arxiv.org/ abs/1801.03924
- 74. Zhao, C., Li, X., Feng, T., Zhao, Z., Chen, H., Shen, C.: Tinker: Diffusion’s gift to 3d–multi-view consistent editing from sparse inputs without per-scene optimization

(2025), https://arxiv.org/abs/2508.14811

- 75. Zhao, Q., Li, Y., Sun, Q., Yan, Z.: Resilphase: Plug-and-play phase mapping and noise-resilient macro-trajectory extrapolation for diffusion acceleration (2026), https://arxiv.org/abs/2606.26769

- 76. Zhao, Y., Sun, S., Zhang, M., Shi, Y., Yang, X., Bian, J.: Scenerevis: A self-reflective vision-grounded framework for 3d indoor scene synthesis via multi-turn rl (2026), https://arxiv.org/abs/2602.09432
- 77. Zhuang, J., Wang, C., Liu, L., Lin, L., Li, G.: Dreameditor: Text-driven 3d scene editing with neural fields (2023), https://arxiv.org/abs/2306.13455

Supplementary Material

- S1 Overview

In this supplementary material, we provide additional technical details, experimental results, and discussions that were omitted from the main paper due to space constraints. The detailed table of contents is provided below:

- S1 Overview ................................................................ 1
- S2 Methodology Details .....................................................1

- S2.1 Formal Definition of GRPO ............................................ 1
- S2.2 Definitions of Alternative Rewards and Evaluation Metrics ............. 3
- S2.3 3DGS Reconstruction Details .......................................... 4
- S2.4 Baseline Implementation: EditSplat with FLUX-Kontext ............... 5

- S3 Dataset and Evaluation Setup ........................................... 5

- S3.1 Training Dataset Construction ......................................... 5
- S3.2 Detailed Test Data Split ............................................... 7

- S4 Additional Experimental Results .........................................8

- S4.1 Per-split Quantitative Metrics & User Study ........................... 8
- S4.2 Additional Ablation Studies ............................................9
- S4.3 Additional Qualitative Results and Failure Cases ......................11

- S5 Further Discussions .....................................................11 S5.1 Why VGGT Serves as a Robust Verifier ...............................11

- S2 Methodology Details

- S2.1 Formal Definition of GRPO

Preliminary: FLUX-Kontext Baseline With the rapid development of deep learning [2,4,15,20,38,51,52,75], we adopt FLUX-Kontext as our base 2D editor, which is built upon a Diffusion Transformer (DiT) architecture and the flowmatching paradigm. During training, a trajectory is constructed from the target edited image x0 to pure noise x1 ∼ N(0,I) over time step t ∈ [0,1], formulated as xt = (1 − t)x0 + tx1. The network vθ is trained to predict the velocity along this path:

0,x1,t ∥vθ(xt,t,c) − (x1 − x0)∥22 , (S1) where c denotes the conditions, including editing instructions and reference images. At inference time, sampling is performed by solving the deterministic Ordinary Differential Equation (ODE) starting from Gaussian noise x1:

LFM(θ) = Ex

dxt = vθ(xt,t,c)dt, t : 1 → 0, (S2) which yields the edited result x0.

Benefiting from the DiT backbone, FLUX-Kontext naturally supports multi-

image correlated editing. The tokens of K input images {xk}Kk=1 are concatenated along the sequence dimension as X = Concat(x1,...,xK). Since all im-

age tokens reside in the same sequence, self-attention mechanisms inherently

distribute attention weights across different images, thereby enabling effective cross-view interactions.

Empowering 2D Editors with 3D Capability via GRPO To align the 2D editor with 3D consistency using Group Relative Policy Optimization (GRPO) [22],

we formulate the editing model as a conditional policy πθ. Unlike Proximal Policy Optimization (PPO), GRPO introduces a group-relative advantage to stabilize policy updates without requiring a separate Critic (Value Model).

Given a condition x (comprising M source views and an editing instruction), the policy independently generates a group of G candidate outputs {yi}Gi=1, where each yi = {Im′i}Mm=1 contains M edited views. To meet GRPO’s requirement for stochastic exploration, we follow Flow-GRPO [40] by converting the deterministic Flow-ODE (Eq. S2) into an equivalent Stochastic Differential Equation (SDE):

σt2 2t

dxt = vθ(xt,t,c) +

xt + (1 − t)vθ(xt,t,c) dt + σtdwt. (S3)

Specifically, during sampling, the model starts from pure noise and executes 12 denoising steps. After each step, we inject Gaussian noise with a strength of σt = 0.8 to successfully transition from a deterministic ODE to an SDE, augmenting exploration diversity.

Each generated candidate yi is evaluated by the VGGT-based reward model to obtain a composite reward Ri. GRPO then computes the relative advantage Ai for each candidate within the group:

Ri − mean({Rj}Gj=1) std({Rj}Gj=1)

Ai =

. (S4)

The policy is optimized by maximizing the following objective, which encourages the network to assign higher probabilities to sample trajectories that yielded higher rewards:

J(θ) = Jclip(θ) − β DKL(πθ∥πref), (S5) where Jclip(θ) is given by:

Jclip(θ) = Ex

G

1 G

min ρi(θ)Ai, clip ρi(θ), 1−ϵ, 1+ϵ Ai . (S6)

i=1

θ(yi|x)

Here, ρi(θ) = π

πold(yi|x) is the importance-sampling ratio tracking the change between the new and old policy. By maximizing this objective, the behavior is explicitly guided by the sign of the relative advantage Ai: for candidates with higher-than-average rewards (Ai > 0), the objective pushes their generation probability πθ(yi | x) up; conversely, for candidates with lower-than-average rewards (Ai < 0), the objective penalizes them and pushes their probability down. The clip function restricts the ratio ρi(θ) within the interval [1 − ϵ,1 + ϵ],

ensuring that the policy does not update too drastically in any single step, which mitigates destructive updates and provides stable optimization. Finally, the Kullback-Leibler (KL) divergence penalty DKL(πθ∥πref) strongly constrains the updated policy πθ to stay close to the reference policy πref (the original FLUX-Kontext editor). This effectively prevents the policy from deviating too far, mitigating mode collapse and preserving its powerful 2D image editing priors without curated paired supervision.

###### S2.2 Definitions of Alternative Rewards and Evaluation Metrics

Photometric Reprojection Loss (Ph-Loss). For reprojection warping, we use depth maps inferred by VGGT. Given an image pair (IA,IB) with corresponding inferred depths (DA,DB) and camera extrinsics (TA,TB), we warp IA to view B using:

IA→B = IA σ(TBTA−1 · DA · σ−1(pA)) , (S7)

where pA denotes the 2D pixel coordinates, σ denotes camera projection, and ⟨·⟩ denotes warp sampling. To handle occlusions and missing regions, the valid region mask M is determined by depth consistency:

M = ⊮[|DA→B − DB| < τ], (S8)

where DA→B is warped similarly to IA→B, and τ is the depth threshold. Following Monodepth2 [18], we use photometric loss within the valid mask to measure consistency. In our evaluation metric, we calculate this over M −1 adjacent view pairs (IA,IB):

1 M − 1

Lph =

(A,B)

0.15·(IB −IA→B)+0.85·LSSIM(IB,IA→B) ⊙M

. (S9)

1

When using it as an alternative reward (Experiment ⑤ in paper Table 2), we also average adjacent pairs and define the reward as rwarp = exp(−Lph). This reward is unsuitable for training for two reasons: first, lacking GT depth for the newly edited contents leads to accumulated errors during warping; second, blurry image outputs trivially minimize the photometric differences, yielding deceitfully high rewards. Such “reward-hacking” severely degrades image quality during optimization.

Despite these limitations as a training reward, it is worth noting that we still adopt Lph as a rigorous evaluation metric in paper Table 1. This is because, during evaluation, the photometric loss effectively measures geometric consistency among sharp images, while any potential blurriness collapse is heavily penalized by the independent VIEScore, ensuring a fair and comprehensive comparison between models.

SfM-based Reward. Structure-from-Motion (SfM) sequentially performs feature extraction, matching, and triangulation to obtain 3D point clouds, which

are then reprojected to corresponding views to assess consistency. Using the GT extrinsics on the M edited views, we define the SfM-based reward as:

rSfM = (Nreg/M) · exp − e¯reproj , (S10)

where Nreg is the number of successfully registered views and e¯reproj is the mean reprojection error across all successfully matched keypoints.

However, because SfM fundamentally relies on sparse feature matching, it is highly susceptible to “reward-hacking” [59]. As shown in paper Figure 7, the RL policy quickly learns to generate textureless, flat, or severely distorted scenes. These textureless images yield very few keypoints, trivially minimizing the reprojection error e¯reproj and producing spuriously high rewards, but entirely destroying the editing quality.

CLIP Directional Similarity. To evaluate how well the edited 3D scene aligns with the semantic instruction without losing the identity of the original scene, we report the CLIP Directional Similarity [17] in our benchmark (Table 1). This metric measures the cosine similarity between the change in the image embeddings and the change in the text embeddings. Specifically, letting EI and ET denote the CLIP image and text encoders respectively, we compute:

∆I = EI(Iedit) − EI(Isource), ∆T = ET(tedit) − ET(tsource), (S11)

∆I · ∆T ∥∆I∥∥∆T∥

CLIP-dir =

, (S12)

where Isource and Iedit are the rendered views before and after editing, tsource is the text description of the original scene, and tedit is the provided editing instruction. We average this score across all evaluated novel viewpoints.

###### S2.3 3DGS Reconstruction Details

As discussed in paper Sec. 3.1, we employ 3D Gaussian Splatting (3DGS) to reconstruct the final edited 3D scene using the M = 9 generated multi-view images and the original GT camera parameters. Since iteratively reconstructing using only 9 sparse views is highly prone to severe overfitting and geometric degeneration, we avoid initializing the 3DGS randomly. Instead, we perform a warm-up initialization using AnySplat [25], a feed-forward 3DGS model related to recent generalizable feed-forward 3DGS methods [2,4], which yields a reliable initial 3D Gaussian state in under 1 second. Starting from this initialization, we perform iterative optimization using the standard 3DGS [27] codebase. We fine-tune the initialized Gaussians for 3,000 iterations using a traditional photometric loss on the edited views, guided by the default learning rate schedulers (Taking about 40s). This combination of feed-forward initialization and light-weight iterative optimization successfully yields a high-fidelity edited 3DGS model capable of rendering geometrically consistent images from any novel viewpoint.

###### S2.4 Baseline Implementation: EditSplat with FLUX-Kontext

As discussed in main paper Sec. 4.2, we re-implement the strongest baseline, EditSplat [34], using the same powerful backbone, FLUX-Kontext, for a fair comparison. In its original implementation, EditSplat utilizes InstructPix2Pix [7] as the 2D diffusion prior. We substitute this single-image editor with FLUXKontext while strictly preserving EditSplat’s core optimization mechanisms.

Specifically, we retain its sequential Multi-view Fusion Guidance (MFG). We first obtain initial FLUX-Kontext edits on all source images. Then, utilizing depth-guided iterative alpha blending, these multi-view edits are sequentially projected and fused into each target view to form a fused guidance image hM. During the underlying diffusion process, classifier-free guidance is generalized to simultaneously align the generated output with hM, the original source image hS, and the text prompt hT.

It is worth noting that, due to the high resolution and slower inference speed of FLUX-Kontext compared to InstructPix2Pix, the overall optimization process of this strong baseline increases from roughly 3.5 minutes to approximately 40 minutes per scene.

### S3 Dataset and Evaluation Setup

###### S3.1 Training Dataset Construction

- Table S1: Comprehensive list of 70 editing instructions in our training dataset.

Scene Edit Instruction

Add a red rubber ball next to the stone base of the bear statue.

Convert the bear statue material to bronze with green patina.

Remove the rectangular stone base beneath the bear statue.

Change the bear statue material to dark brown carved wood.

Bear

Replace the gray stone bear with a black-and-white stone panda statue.

Replace background trees and bushes with dense bamboo forest.

Change scene lighting to a moonlit night.

Apply watercolor painting style to the scene.

Modify the bear statue’s pose to sitting on its base.

Place a canvas backpack on the bench seat near the bicycle handlebars.

Bicycle

Replace the grass and paved path with dry desert sand.

Change the black metal bench material to distressed wood.

Install a blue child safety seat on the bicycle’s rear rack.

Change the scene to winter with snow covered.

Add a brick wall with ivy along the road, blocking the bushes.

Convert the entire scene to black-and-white pencil sketch style.

Remove the white bicycle entirely from the scene.

Shrink the white bicycle to miniature size at its original position. Bonsai

Change all pink LEGO flowers on the bonsai tree to white. Continued on next page

Change the purple knitted blanket to a deep blue velvet blanket.

Place a small brown hardcover book on the left corner of the blanket.

Make the LEGO bonsai tree trunk taller and upright.

Convert the entire scene to Van Gogh-style oil painting.

Change the brown wooden stand material to white marble.

Convert the bonsai into a pine tree bonsai with green needles.

Place a red Santa hat on the dinosaur statue’s head.

Change the wooden dinosaur statue material to shiny gold.

Open the dinosaur statue’s mouth.

Transform into a realistic living dinosaur with lifelike skin texture.

Dinosaur

Change the scene to winter and cover it with white snow.

Cover the surrounding gravel and grass with thick autumn fallen leaves.

Convert the entire scene to colored pencil sketch art style.

Remove all distant buildings, billboards from the background. Add a colorful woven scarf around the dinosaur statue’s neck. Remove the circular stone base beneath the dinosaur statue.

Convert the entire garden scene to cartoon style.

Remove the sphere ball on the base under the table.

Change the table material to marble with gray veining.

Garden

Replace the vase and dried flowers with a glass vase of red roses.

Replace the visible grass areas with a pond.

Replace background and garden with extending sandy beach ground.

Turn to winter and add white snow to the environment.

Convert the scene to Monet-style oil painting.

Place a red-and-white checkered tablecloth on the round table.

Add fashionable black-framed glasses to the man.

Make the man cross his arms in front of his chest.

Make the man laugh heartily.

Person

Replace the office background with a beach scene.

Transform the person into a blocky Minecraft-style character.

Place a ceramic mug in the man’s right hand.

Change the man’s clothes to a dark gray business suit.

Convert the entire scene to comic illustration style.

Transform the man into Iron Man in full battle armor.

Add fashionable black-framed glasses to the person.

Face

Remove the stubble from the person’s face.

Change the gray sweater to a deep blue hoodie.

Add a baseball cap to the person.

Convert the portrait to oil painting style.

Make the man open his mouth.

Apply clown makeup to the person’s face.

Transform the person into the Hulk. StoneHorse

Transform the stone horse into a realistic living horse with fur texture.

Add a red traditional Chinese knot tassel around the horse’s neck. Continued on next page

Change the horse statue’s surface to blue-and-white porcelain style.

Colorize the stone horse with brown horse and saddle colors.

Add yellow fallen leaves on the horse’s back and ground.

Convert the scene to cyberpunk style.

Switch to nighttime with bright red glowing eyes on the horse statue.

Switch the scene to rainy weather with wet and reflective surfaces.

###### S3.2 Detailed Test Data Split

- Table S2: The 16 unseen test instructions utilized for evaluating instruction-level zeroshot generalization on the 8 training scenes.

Scene Unseen Edit Instruction

Make the bear eat apple

Bear

Add a thick layer of snow on the bear statue.

Paint the white bicycle bright red.

Bicycle

Remove the black metal bench.

Clear the background objects and replace with clean wall and floor.

Bonsai

Place a small blue ceramic bird next to the bonsai.

Replace the wooden dinosaur with a metallic robot dinosaur.

Dinosaur

Place a small puddle of water in front of the dinosaur.

Add a white ceramic rabbit statue on the table.

Garden

Change the garden lighting to a warm sunset glow.

Dress the man in a red T-shirt with a white star logo on the chest.

Person

Change the person’s hair color to blonde.

Make the person close their eyes.

Face

Add a thick brown beard to the person’s face.

Replace stone pedestal with a crystal base.

Stone Horse

Add a heavy metal armor to the horse.

As stated in Section 4.2, our evaluation split consists of 100 testing cases categorized into:

- 1. 70 Novel Views (In-distribution): These evaluate view synthesis generalization. They use scenes and instructions seen during training but render them from completely novel viewpoints not used in the optimization trajectories.
- 2. 16 Unseen Instructions (Zero-shot instruction): Two novel, unseen instructions are drafted for each of the 8 training scenes, evaluating the model’s generalization beyond the training instruction. We list them in Table S2.

- Table S3: The 14 new scene instructions utilized for evaluating fully zero-shot scene generalization capability across 4 newly introduced test assets.

Test Scene New Scenes Edit Instruction

Change the wooden floor to marble.

Room

Remove the TV from the room.

Convert the scene to line drawing style.

Change the scene to winter with snow covering the stump.

Place a small camping lantern on the stump.

Stump

Turn the wooden stump into solid stone.

Surround the stump with vibrant spring flowers.

Put cap on this person.

Fangzhou

Add a mustache to this person.

Transform this person into blocky Minecraft style.

Convert Lego loader to metal material.

Place a small LEGO figure next to the loader.

Kitchen

Add a bowl of fruit on the table.

Convert the scene into an underwater environment.

3. 14 New Scenes (Zero-shot scene): Unseen new scenes (room, stump, fangzhou, kitchen) are tested with new instructions to verify the zero-shot generalization capabilities. See detailed instructions in Table S3.

### S4 Additional Experimental Results

###### S4.1 Per-split Quantitative Metrics & User Study

As mentioned in paper Table 1, we present quantitative evaluations partitioned into in-distribution and zero-shot settings to better demonstrate the generalization capabilities of our method. Notably, baseline methods necessitate iterative per-scene optimization across all test cases (including both seen and unseen instructions/scenes), inherently remaining within the in-distribution paradigm.

3D editing is essentially a subjective task, so we further conduct a user study. To comprehensively assess both image quality and multi-view consistency simultaneously, we present side-by-side rendered video comparisons of the editing results (our method versus baselines) to 25 participants. The participants are asked to select their overall preferred result based on how well it follows the editing instruction and maintains geometric stability throughout the video. We report the preference rate (%) for each method.

As shown in Table S4, our method not only achieves state-of-the-art performance on in-distribution data, but also maintains significant advantages in the zero-shot setting over other baselines that directly optimize on the testing scenes. Furthermore, RL3DEdit achieves an overwhelming advantage in the user

###### Table S4: Per-split quantitative metrics and user study preference rates. “In-dist.” refers to the 70 novel-view cases, while “Zero-shot” encompasses 30 cases with unseen instructions or newly introduced scenes. Detailed in Sec. S4.1.

In-dist. (70 cases) Zero-shot (30 cases) VIEScore↑ User Study↑ Ph-Loss↓ VIEScore↑ User Study↑ Ph-Loss↓

Methods

DGE [11] 2.76 8.6% 0.086 2.92 10.0% 0.085 GaussCtrl [61] 2.40 4.3% 0.077 2.30 6.7% 0.078 EditSplat [34] 2.62 5.7% 0.082 2.94 6.7% 0.080 EditSplat w/ FLUX-Kontext

3.22 10.0% 0.083 3.26 16.6% 0.079

RL3DEdit (Ours) 5.52 71.4% 0.075 5.40 60.0% 0.078

study, securing the highest preference rate in both seen and unseen scenarios, which strongly corroborates its superior visual quality and 3D consistency.

###### S4.2 Additional Ablation Studies

As mentioned in paper Sec. 4.3, due to space constraints, we present additional ablation studies on other design details in this appendix. All ablation models are trained and evaluated on the same Face dataset split following the identical protocol as in the main paper.

Effect of Anchor Selection Strategy. To explicitly enforce editing fidelity during the RL process, we utilize a pre-calculated, high-quality single-view edit as an anchor to guide multi-view optimization. We compare two anchor selection strategies: (a) Fixed: Always choosing the first rendered view as the anchor. (b) Random: Randomly selecting one view per sample during training (our default).

As shown in the top section of Table S5, the Random strategy achieves better performance. This is because the Random strategy allows all viewpoints to seamlessly learn the editing prior, ensuring uniform editing quality.

Effect of Denoising Steps. In standard 2D RL editing methodologies, a 6step denoising process using an SDE formulation is typically sufficient to explore high-quality trajectories [40]. However, in our 3D consistent editing framework, we observe that evaluating geometry consistency demands higher image fidelity and minimal artifacts. As shown in the middle section of Table S5, utilizing only 6 denoising steps fails to resolve fine-grained details efficiently, obstructing the VGGT reward model from accurately assessing cross-view 3D consistency, leading to poor Ph-Loss. Increasing the denoising steps to 12 significantly improves the novel view render quality and enables effective 3D alignment. Further increasing to 20 steps yields negligible gains while disproportionately increasing inference and training time. Therefore, we safely adopt 12 steps as our default configuration.

Effect of Reward Weights. Our composite reward integrates four critical components: depth consistency (rD), point consistency (rP), relative pose alignment

- Table S5: Additional ablation studies on anchor selection strategies, denoising steps, and reward weights. “Ours (Default)” denotes the RL3DEdit configuration used in the main paper. All experiments are conducted on the face dataset.

Configuration VIEScore↑ Ph-Loss↓ Anchor Selection Strategy

Fixed Anchor (1st view) 4.54 0.087 Ours (Random, 1 view) 5.26 0.077

###### Denoising Steps

6 steps 3.91 0.093 20 steps 5.28 0.078 Ours (12 steps) 5.26 0.077

###### Reward Weights

Geometry-heavy (wD, wP = 0.4) 4.31 0.076 Quality-heavy (wa = 0.7) 3.97 0.091 Ours (Equal weights 0.25) 5.26 0.077

(rT), and anchor-based image quality (ra). Our default configuration equally weights them (wD = wP = wT = wa = 0.25). We systematically ablate this design by skewing the weight distribution: (a) Geometry-heavy: wD = wP = 0.4,wT = wa = 0.1. (b) Quality-heavy: wa = 0.7,wD = wP = wT = 0.1.

The bottom section of Table S5 reveals the delicate balance required for 3D editing. A Geometry-heavy setting aggressively forces geometric alignment, resulting in a similar Ph-Loss but inevitably compromising the semantic richness and fine details inherent in the 2D prior (VIEScore drops to 4.31). Conversely, a Quality-heavy setting aggressively prioritizes 2D visual appeal. However, this lack of geometric constraint produces severe multi-view inconsistencies. Consequently, the novel view reconstruction suffers from critical artifacts, which destroy the overall fidelity of novel-view editing. The equal weighting strategy successfully negotiates these competing objectives, proving to be the optimal configuration.

Stylized Edits. We specifically evaluated performance on stylized editing instructions (e.g., artistic style transfer). While the performance is slightly lower than on realistic edits, it remains acceptable, because the VGGT reward guides

- 3D consistency over all samples and the consistency prior learned from realistic edits generalizes to stylized ones.

Different Inference Views. RL3DEdit is primarily a multi-view framework that allows flexible view handling: fewer views (e.g., 4) can be accommodated by duplicating them into 9-view inputs with similar performance, while scaling to more views requires dedicated designs such as a streaming pipeline, which we leave to future work.

[Figure 143]

###### Supplementary Material 11

[Figure 144]

[Figure 145]

Remove the white bicycle entirely from the scene.

[Figure 146]

[Figure 147]

Change the brown wooden stand material to white marble.

[Figure 148]

[Figure 149]

Turn to winter and add white snow to the environment.

[Figure 150]

Colorize the stone horse with brown horse and saddle colors.

- Fig. S1: Additional qualitative comparisons. (Please zoom in for details and refer to the supplementary video for novel view synthesis).

###### S4.3 Additional Qualitative Results and Failure Cases

We present further qualitative editing results of RL3DEdit in Figure S1. For a more comprehensive multi-view evaluation, we strongly encourage reviewing the supplementary video, which dynamically shows the superior 3D consistency and editing quality of our method.

Failure Cases Analysis. As discussed in Sec. 5, RL3DEdit may struggle with extremely drastic non-rigid deformations (e.g., “make the person bow his head”), where action magnitude ambiguity leads to multi-view discrepancies that compromise 3D reconstruction. All baselines also fail on such cases.

S5 Further Discussions

###### S5.1 Why VGGT Serves as a Robust Verifier

As mentioned in paper Sec. 4.3, we argue that VGGT is significantly harder to be “reward hacked” than traditional 3D consistency verifiers. For example: (1) SfM relies on handcrafted feature matching algorithms (e.g., SIFT). If a model collapses and generates textureless images, feature extraction fails, resulting in zero matching points and artificially high SfM rewards. Consequently, an RL policy can maximize this reward by generating distorted or even blank images. (2) Photometric Loss (Ph-Loss) relies on pixel-level photometric comparisons. If the generated images are excessively blurry or dominated by low-frequency noise, spatially adjacent pixels become nearly identical, leading to artificially

[Figure 151]

[Figure 152]

12 Supplementary Material

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

##### ① ② ③

[Figure 157]

[Figure 158]

[Figure 159]

- Fig. S2: Visualization of VGGT confidence maps. For blurry or distorted inputs (① and

②), VGGT assigns lower confidence (darker regions). In ③, high-fidelity regions yield high confidence (the wall textures naturally receive slightly lower confidence due to the increased consistency difficulty.) This ensures that the reward effectively penalizes low-quality or “reward-hacked” outputs.

minimized errors. Therefore, this metric cannot distinguish between meaningful, high-quality edits and low-frequency artifacts. Conversely, VGGT possesses the following advantages: (a) Implicit Real-World Priors. VGGT is trained on millions of multi-view sets of real-world scenes. Through this extensive pretraining, it learns a robust implicit prior regarding the natural structure and appearance of real-world imagery. (b) Fidelity-Conditioned Confidence. As shown in Fig S2, when evaluating blurry, textureless, or highly distorted images, VGGT inherently outputs lower confidence maps. (c) Dual Constraint Synergy. The VGGT reward functions as a powerful “dual constraint.” It simultaneously demands strict geometric consistency across viewpoints while explicitly enforcing that the generated images remain natural and realistic.

