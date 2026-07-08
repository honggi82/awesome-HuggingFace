## Go-with-the-Flow: Motion-Controllable Video Diffusion Models Using Real-Time Warped Noise

Ryan Burgert1,3 Yuancheng Xu1,4 Wenqi Xian1 Oliver Pilarski1 Pascal Clausen1 Mingming He1 Li Ma1 Yitong Deng2,5 Lingxiao Li2 Mohsen Mousavi1 Michael Ryoo3 Paul Debevec1 Ning Yu1†

1Netflix Eyeline Studios 2Netflix 3Stony Brook University 4University of Maryland 5Stanford University

{ryan.burgert,yuancheng.xu,wenqi.xian,oliver.pilarski,pascal.clausen, mingming.he,li.ma,mohsen.mousavi,debevec,ning.yu}@scanlinevfx.com lingxiaol@netflix.com {rburgert,mryoo}@cs.stonybrook.edu ycxu@umd.edu yitongd@stanford.edu

# arXiv:2501.08331v5[cs.CV]6Aug2025

https://eyeline-labs.github.io/Go-with-the-Flow/

Local object motion control

[Figure 1]

[Figure 2]

[Figure 3]

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

[Figure 15]

x

Global camera movement control

[Figure 16]

Input Image Depth Map

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Motion Transfer

Input Video Optical Flows

Image-To-Video: Edited First Frame Text-To-Video: “An outdoor hot tub”

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

Motion Transfer

Figure 1. Go-with-the-Flow presents a simple, robust, and easy-to-implement method for motion-controllable video diffusion models based on optical flow and noise warping. It only requires fine-tuning video diffusion models as a black box using warped noise patterns. Leveraging our models, we can (1) control the motion of individual objects or parts of those objects, (2) direct the camera movement by providing global flow fields corresponding to the desired movements, and (3) transfer the motion from input videos to target contexts.

#### Abstract

tured noise. Consequently, our method is agnostic to diffusion model design, requiring no changes to model architectures or training pipelines. Specifically, we propose a novel noise warping algorithm, fast enough to run in real time, that replaces random temporal Gaussianity with correlated warped noise derived from optical flow fields, while preserving the spatial Gaussianity. The efficiency of our algorithm enables us to fine-tune modern video diffusion base models using warped noise with minimal overhead,

Generative modeling aims to transform random noise into structured outputs. In this work, we enhance video diffusion models by allowing motion control via structured latent noise sampling. This is achieved by just a change in data: we pre-process training videos to yield struc-

†Project lead

and provide a one-stop solution for a wide range of userfriendly motion control: local object motion control, global camera movement control, and motion transfer. The harmonization between temporal coherence and spatial Gaussianity in our warped noise leads to effective motion control while maintaining per-frame pixel quality. Extensive experiments and user studies demonstrate the advantages of our method, making it a robust and scalable approach for controlling motion in video diffusion models. Video results are available on our webpage; source code and model checkpoints are available on GitHub.

#### 1. Introduction

“We adore chaos because we love to produce order.” — M. C. Escher, Dutch artist

The essence of generative modeling lies in producing order from chaos, learning to transform random noise from the latent space into structured outputs that align with the distribution of training data. In this paper, we propose a novel approach to enhance generative model learning by proactively introducing partial order into the chaos of latent space sampling.

Our work is motivated by the remarkable progress in video diffusion generative models [3, 4, 7, 15, 16, 64] and the equally significant challenges they face in terms of controllability beyond text and image guidance. Finegrained interactive control over motion dynamics remains an under-explored area due to the intricate spatiotemporal correlations among video frames. The complexity of modern video diffusion architectures [7, 64], which leverage 3D autoencoders [68] and spatiotemporal tokenizers [35], further complicates efforts to adapt models for effective motion control. The optimal format for defining and disentangling motion control from other guidance remains an open question.

Within the domain of motion-controllable video diffusion models, current applications typically fall into three categories: (1) local object motion control, represented by object bounding boxes or masks with motion trajectories [13, 25, 39, 45, 49, 58, 60, 63]; (2) global camera movement control, parameterized by camera poses and trajectories [17, 30, 58, 59, 62, 63] or categorized by common directional patterns such as panning and tilting [16, 63]; and (3) motion transfer from reference videos to target contexts specified by prompts or initial frames [1, 14, 29, 34, 38, 57, 65, 66]. However, these approaches share three key limitations: (1) they often necessitate complex modifications to the base model design such as guidance attention [45], limiting compatibility with modern full-attention architectures involving spatiotemporal tokens [64]; (2) they are constrained to specific applications, requiring detailed param-

eterized motion signals, such as camera parameters, which are challenging to acquire or estimate accurately [72], thus restricting generalizability across diverse scenarios; and (3) they are over-rigid to motion control at the cost of spatiotemporal visual quality.

To address these limitations, we propose a novel and straightforward method to incorporate motion control as a structured component within the chaos of video diffusion’s latent space. We achieve this by correlating the temporal distribution of latent noise. Specifically, starting with a 2D Gaussian noise slice, we temporally concatenate it with warped noise slices, given the optical flow field [53] extracted from a training video sample. Fig. 2 illustrates the diagram of our method. Our approach requires only a change in data: we pre-process training videos to yield warped noise and then fine-tune a video diffusion model. As it occurs solely during noise sampling, our method is agnostic to diffusion model design, requiring no modifications to model architectures or training pipelines. Surprisingly, removing temporal Gaussianity from the noise distribution does not deteriorate model fine-tuning. Instead, it can be quickly adapted after fine-tuning because temporal structure in the chaos of latent space facilitates generative learning and enables motion correspondence. Temporal coherence occurring in the latent space also harmonizes motion control with per-frame pixel quality by inheriting the high-quality prior from the base model.

It is worth noting that video diffusion fine-tuning relies on efficient noise warping algorithms that introduce minimal overhead during data pre-processing and noise sampling. The existing noise warping algorithm, How I Warped Your Noise (HIWYN) [8], that maintains spatial Gaussianity and enables temporal flow warping, however, suffers from the quadratic computation costs w.r.t. frame count, making it much slower than in real time and therefore impractical for large-scale video diffusion model training. To address this, we propose a novel noise warping algorithm that runs fast enough in real time. Rather than warping each frame through a chain of operations from the initial frame, our algorithm iteratively warps noise between consecutive frames. This is achieved by carefully tracking the noise and the flow density along a forward and a backward flow at the pixel level, accounting for both expansion and contraction dynamics, supplemented with conditional white noise sampling from HIWYN Chang et al. [8] to preserve Gaussianity. Algorithm 1 provides further details. We validate the spatial Gaussianity and time complexity of our noisewarping algorithm and apply it to training-free image diffusion models for quantitative and qualitative assessments of controllability and temporal consistency.

During video diffusion inference, our method offers a one-stop solution for diverse motion control applications by adapting noise warping based on motion type. (1) For local

[Figure 56]

Figure 2. Our method consists of three components: flow field extraction, real-time noise warping, and diffusion model finetuning/inference. During fine-tuning, we use the original captions of video samples. At inference, our method enables adaptation of reference motion to various prompts and/or initial frames, offering creativity and diversity in generation.

object motion, we interactively transform noise elements within object masks given users’ dragging signals. (2) For global camera movement control, we reuse the optical flows from reference videos to warp input noise, and regenerate videos conditioned on different texts or initial frames. (3) For arbitrary motion transfer, the motion representations are not limited to optical flows [53], but also include flows from 3D rendering engines [55], depth warping [67], etc. We validate the effectiveness of our solution across various video generation tasks, demonstrating its ability to preserve consistent motion across different contexts or render distinct motions for the same context. Extensive experiments and user studies indicate the advantages of our solution in pixel quality, motion control, text alignment, temporal consistency, and user preference.

In summary, our contributions include:

- (1) A novel and simple one-stop solution for motion-

controllable video diffusion models, integrating motion control as a flow field for noise warping in latent space sampling, plug-and-play for any video diffusion base models as a black box, and compatible with any other types of controls.

- (2) An efficient noise warping algorithm that main-

tains spatial Gaussianity and follows temporal motion flows across frames, facilitating motion-controllable video diffusion model fine-tuning with minimal overhead.

- (3) Comprehensive experiments and user studies demon-

strating the overall advantageous pixel quality, controllability, temporal consistency, and subjective preference of our method on diverse motion control applications, including but not limited to: local object motion control, motion transfer to new contexts, and reference-based global camera movement control.

#### 2. Related work

##### 2.1. Image and video diffusion models

With the theoretical establishments of diffusion models [20, 27, 50, 51] and their practical advancements [19, 40], and when sophisticated text encoders [46] and language mod-

els [47] meet diffusion models, great breakthroughs in textto-image generation [42, 48, 52] have revolutionized how we digitize and create visual worlds. Building upon these, image-to-image diffusion models [6, 28, 73] enable image editing applications like stylization [36], relighting [18], and super-resolution [52, 70], expanding creativity in recreating or enhancing visual worlds.

A natural extension of image generation use cases is to cover the temporal dimension for video generation. The most cost-efficient way is to reuse the well-trained image diffusion model weights. Directly querying the above image diffusion models using random noise to generate videos frame-by-frame often struggles with temporal inconsistency, flickering, or semantic drifting. Noise warping, HIWYN [8], as a method for creating a sequence of temporally-correlated latent noise from optical flow while claiming spatial Gaussianity preservation, yields temporally consistent motion patterns after querying image diffusion models without further fine-tuning. To overcome its defective spatial Gaussianity preservation and undesired time complexity, we propose a novel warped noise sampling algorithm that guarantees spatial Gaussianity and runs fast enough in real time. We validate its efficacy by applying it to the training-free image diffusion models like DifFRelight [18] for video relighting and DeepFloyd IF [52] for video super-resolution.

Video diffusion model training is a more costly yet more effective way for video generation [3, 4, 7, 9, 16, 44, 61, 64]. AnimateDiff [16] upgrades pre-trained image diffusion models by fine-tuning temporal attention layers on large-scale video datasets. CogVideoX [64], a state-of-the-art open-source video diffusion model, combines spatial and temporal dimensions by encoding/decoding videos via 3D causal VAE [68] and diffusing/denoising spatiotemporal tokens via diffusion transformers [41]. We use CogVideoX [64] as a base model and incorporate our warped noise sampling for motion-controllable fine-tuning. We also fine-tune on AnimateDiff [16] to show our method is model-agnostic.

##### 2.2. Motion controllable video generation

Beyond text [16, 64] and image controls [15, 61, 75] for video diffusion models, motion control makes video generation more interactive, dynamically targeted, and spatiotemporally fine-grained. Current approaches to motion control follow three main paradigms:

Firstly, local object motion control is represented by object bounding boxes or masks with motion trajectories [13, 25, 39, 45, 49, 58, 60, 63]. DragAnything [60] allows precise object motion manipulation in images without retraining, while SG-I2V [39] generates realistic, continuous video from single images using self-guided motion trajectories. These serve as recent baselines for local object motion control. Our method is plug-and-play, treating diffusion models as a black box while using synthetic flows to mimic and densify object trajectories at the pixel level.

Secondly, global camera movement control is parameterized by camera poses and trajectories [17, 30, 58, 59, 62, 63] or categorized by common directional patterns like panning and tilting [16, 63]. These methods introduce additional modules that accept camera parameters, trained in a supervised manner. Other approaches [22, 69] leverage rendering priors as input for camera control. Approaches like ReCapture [71] enable reconfiguration of camera trajectories in given videos. Our method bypasses the need for extensive camera parameter collection, and directly generalizes new camera movements from reference videos at inference.

Lastly, motion transfer happens from reference videos to target contexts [1, 14, 29, 34, 38, 57, 65, 66]. DiffusionMotionTransfer [65] introduces a loss that maintains scene layout and motion fidelity in target videos, while MotionClone [34] uses temporal attention as motion representation, streamlining motion transfer. Using them as motion transfer baselines, we demonstrate our model’s flexibility in combining reference geometries with target text guidance.

#### 3. Method

Go-with-the-Flow is comprised of two separate parts: our noise warping algorithm and video diffusion fine-tuning. The noise warping algorithm operates independently from the diffusion model training process: we use the noise patterns it produces to train the diffusion model. Our motion control is based entirely on noise initializations, introducing no extra parameters to the video diffusion model.

Inspired by the existing noise warping algorithm HIWYN [8], which introduced noise warping for image diffusion models, we introduce a new use case for the warped noise: we use it as a form of motion conditioning for video generation models. After fine-tuning a video diffusion model on a large corpus of videos paired with warped noise, we can control the motion of videos at inference time.

Algorithm 1 Go-with-the-Flow next-frame warping

- 1: Input: previous-frame noise q ∈ RD, previous-frame density p ∈ RD, forward flow f : D → N2, backward flow f′ : D → N2.
- 2: Let G = (V,V ′,E) be a bipartite graph with V = D, V ′ = D and edge set E = {} to be constructed.
- 3: for v in V do ▷ Contraction
- 4: E ← E ∪ (v,v + f(v)) if v + f(v′) ∈ D
- 5: end for
- 6: for v′ in V ′ do ▷ Expansion
- 7: if degG(v′) = 0 then ▷ degG(v) denote the degree of v in G
- 8: E ← E ∪ (v′ + f′(v′),v′) if v′ + f′(v′) ∈ D
- 9: end if
- 10: end for
- 11: for v in V do ▷ Conditional white noise sampling
- 12: d ← degG(v)
- 13: Sample Z ∼ N(0,Id), and set S ← di=1 Zi
- 14: Xi ← q(dv) + √1d(Zi − Sd ) for i ∈ [d]

- 15: R(v) ← {Xi}i∈[d]
- 16: end for
- 17: for (v′) in V ′ do ▷ Compute next-frame noise and density
- 18: q′(v′) ← 0, p′(v′) ← 0, s ← 0
- 19: for v in V such that (v,v′) ∈ E do
- 20: d ← degG(v), α ← p(dv)

- 21: q′(v′) ← q′(v′) + αR(v).pop()
- 22: p′(v′) ← p′(v′) + α
- 23: s ← s + α2d1

- 24: end for
- 25: if s = 0 then
- 26: Sample q′(v′) ∼ N(0,1)
- 27: else
- 28: q′(v′) ← q

′(v′) √s ▷ Renormalize to unit variance

- 29: end if
- 30: end for
- 31: return next-frame noise and density q′,p′.

##### 3.1. Go-with-the-Flow noise warping

###### 3.1.1 Algorithm

To facilitate the large-scale noise warping required by this new use case, we introduce a fast noise warping algorithm (Algorithm 1) that warps noise frame-by-frame, storing just the previous frame’s noise (with dimensions H × W × C, where H is height, W is width, and C is the number of channels) and a matrix of per-pixel flow density values (with dimensions H × W). The density values indicate how much noise has been compressed into a given region. Unlike HIWYN [8] which requires time-consuming polygon rasterization and upsampling of each pixel, our algo-

rithm directly tracks the necessary expansion and contraction between frames according to the optical flow and uses only pixel-level operations that are easily parallelizable. We show that our algorithm retains the same Gaussianity guarantee as HIWYN [8] (Proposition 1).

Next-frame noise warping. Our noise warping algorithm calculates noise iteratively, where the noise for a given frame depends only on the state of the previous frame.

Let H × W be the dimensions of each video frame. Let D = [H] × [W] denote a 2D matrix with height H and width W, where we use the notation [n] := 1,...,n. Given the previous frame’s noise1 q ∈ RD and the flow density p ∈ RD together with forward and backward flows2 f,f′ : D → N2, our algorithm computes the next-frame noise and density q′,p′ ∈ RD such that q′ (resp. p′) is temporally correlated with q (resp. p) via the flows.

At a high level, our algorithm (in Algorithm 1) combines two types of dynamics: expansion and contraction. In the case of expansion, such as when a region of the video zooms in or an object moves towards the camera, one noise pixel is mapped to one or more noise pixels in the next frame (hence it “expands”). In the case of contraction, we adopt the Lagrangian fluid dynamics viewpoint of treating noise pixels as particles moving along the forward flow f. This often leaves gaps that need to be filled. Hence, for regions not reached when flowing along f, we use the backward flow f′ to pull back a noise pixel. That gap is filled with noise calculated with the expansion case.

Additionally, to preserve the distribution correctly over long time periods, we use density values to keep track of how many noise pixels were aggregated into a given region, so that when mixed with other nearby particles in the contraction case, these higher density particles have a larger weight. This is illustrated in Fig. 3.

We unify both expansion and contraction cases by building a bipartite graph G where edges represent how noise and density should be transferred from the previous frame to the next. When aggregating the influence from graph edges to form the next-frame noise q′, we scale the noise in accordance with the flow density to ensure the preservation of the original frame’s distribution, as detailed in Algorithm 1. The expansion and contraction cases are calculated in tandem to prevent any cross-correlation, guaranteeing the output will be perfectly Gaussian.

###### 3.1.2 Theoretical analysis

Proposition 1 (Preservation of Gaussian white noise). If the pixels of the previous-frame noise q in Algorithm 1 are i.i.d. standard Gaussians, then the output next-frame noise

- 1Since different channels are treated independently, we will assume a single channel in images.
- 2We allow flows to go out of bounds, i.e., f and f′ can land in N2 \ D.

[Figure 57]

Figure 3. Diagram of our noise warping algorithm. A case example of our algorithm illustrates both the expansion and contraction cases, along with example density values. Each node represents some noise pixel ‘q’. Noise values q0..3 are transferred from frame 0 to frame 1 using forward optical flow, and the remaining pixels in frame 1 that did not receive any values obtain their values from frame 0 using reverse optical flow (the expansion case). In the contraction cases such as q2′ , their densities become the sum of their sources. And in the expansion case, where one source pixel spreads out into multiple target pixels, such as q2 spreading out into q1′ and q3′ , its density is dispersed.

q′ also has i.i.d. standard Gaussian pixels. Please check the appendix for a formal mathematical proof.

Proposition 2 (Time Complexity). For a given frame, the time complexity of this algorithm is O(D), linear time with respect to the number of noise pixels processed. Proof: There are only two cases - contraction and expansion. Because each previous-frame pixel can only be contracted to one current-frame pixel, and during expansion each current-frame pixel can only be mapped to one previousframe pixel, the total number of edges E will never exceed

- 2D.
- 3.2. Training-free image diffusion models with warped noise

As shown by Chang et al. [8] and Deng et al. [11], noise warping can be combined with image diffusion models to yield temporally consistent video edits without training. To do this, we first take an input video and calculate its optical flows using RAFT [53]. Then, with Algorithm 1, we use the flow fields to create sequences of Gaussian noise for each frame in the input video, ensuring that the noise moves along the flow fields. These noises are used during the perframe diffusion processes in place of what would normally be temporally independently sampled Gaussian noise. This enables temporally consistent inference for video tasks, such as relighting [18] and super-resolution [52], using image-based diffusion models.

##### 3.3. Fine-tuning video diffusion models with warped noise

We use warped noise to condition a video diffusion model on optical flow. In particular, we fine-tune two variants of a latent video diffusion model CogVideoX [64], both the

[Figure 58]

- Figure 4. Showcasing the effect of noise degradation level γ on generated videos. A few frames from the driving video are shown in the leftmost column. Our model outputs are in the next 3 columns. As degradation decreases (γ from 0.7 to 0.5), the video more strictly adheres to the input flow. This allows us to control video movement with a user-specified level of precision.

text-to-video (T2V) and image-to-video (I2V) variants. We regard CogVideoX as a black box without changing its architecture.

We use the same training objective as in normal finetuning, i.e., the mean squared loss between denoised samples and samples with noise added. In fact, we use the exact same training pipeline as the original CogVideoX repository, with exactly one difference: during training, we use warped noise instead of regular Gaussian noise. For each training video, we calculate its optical flow for each frame, and create a warped noise tensor Q ∈ RF×C×H×W, where F,C,H,W are the number of frames, the number of channels, the height and width of encoded video samples respectively by applying our algorithm iteratively.

We also introduce the concept of noise degradation, which lets us control the strength of our motion conditioning at inference time. After calculating the clean warped noise, we then degrade it by a random degradation level γ ∈ [0,1], by first sampling uncorrelated gaussian noise ζ ∼ N(0,1) and modifying the warped noise Q ← √(1−γ)Q+ζγ

. As degradation level γ → 1, Q approaches

(1−γ)2+γ2

an uncorrelated Gaussian, and as γ → 0, Q approaches clean warped noise. At inference, the user can control how strictly the resulting video should adhere to the input flow. Please see Fig. 4 for a qualitative depiction of the effect of γ.

In practice, because the diffusion model works on latent embeddings, we calculate the optical flow and warped noise in image space and then downsample that noise into latent

space, which in the case of CogVideoX means downscaling by a factor of 8 × 8 spatially and 4 temporally. We use nearest-neighbor interpolation along the temporal axis and mean-pooling along the two spatial axes, which are then multiplied by 8 to preserve unit variance.

##### 3.4. Video diffusion inference with warped noise

At inference, we generate warped noise from an input video to guide the motion of the output video. Then, using a deterministic sampling process such as DDIM [50], we use that warped noise to initialize the diffusion process of our fine-tuned video diffusion model. This method of control is much simpler than other motion control methods, as it does not require any changes to the diffusion pipeline or architecture - using exactly the same amount of memory and runtime as the base model.

In the case of local object motion control, we allow the user to specify object movements through a simple user interface as shown in Fig. 5. It is used to generate synthetic optical flows, where multiple layers of polygons are overlaid on an image. Then, these polygons are translated, rotated and scaled with paths defined by the user. We warp the noise accordingly, and use that noise to initialize the diffusion process, along with a text prompt, and in the case of the image-to-video model, a given first frame image. By controlling the extent to which the output video follows these polygons, users can simulate camera movement by shifting the background, or even 3D motion effects by overlaying two polygons in parallax and moving them at different

speeds. We find that this motion control representation is quite robust to user error, where even if the polygon only roughly matches the object or area of interest it will still produce high quality results. For synthetic object motion control, we typically use a degradation value γ between 0.5 and 0.7, depending on the level of motion precision the user desires, which is a higher level than we would normally use for motion transfer.

The case of motion transfer and camera motion control are very similar – the only difference is the source of the flows used to generate the warped noise. In the case of motion transfer, we calculate the optical flow of a driving video, get warped noises that match the motion. Like in local object motion control, we use that warped noise to initialize a diffusion process. In the case of motion transfer, we typically use a lower degradation value γ between 0.2 and 0.5, as we usually want the output video’s motion to match the driving video’s motion as closely as possible.

- 3.5. Implementation details

We fine-tune the recent state-of-the-art open-source video diffusion model, CogVideoX-5B [64], on both its T2V and I2V variants. We use a large general-purpose video dataset composed of 4M videos with resolution ≥720×480 ranging from approximately 10 to 120 seconds in length, with paired texts captioned by CogVLM2 [56]. We used 8 NVIDIA A100 80GB GPUs over the course of 40 GPU days, for 30,000 iterations using a rank-2048 LoRA [23] with a learning rate of 10−5 and a batch size of 8.

Our method is data agnostic and model agnostic. It can be used to add motion control to arbitrary video diffusion models, while only processing the noise sampling during fine-tuning. For example, it also works with AnimateDiff [16] fine-tuned with the WebVid dataset [2], trained on 8×40GB A100 GPUs over a period of 2 days with 12 frames and 256 × 320 resolution. See its qualitative results in Fig. 16 in the supplementary material.

- 4. Experiments

- 4.1. Gaussianity

Evaluation metrics. To validate the preservation of spatial i.i.d. Gaussianity, we follow the evaluation protocol outlined by InfRes [11]. Specifically, we use Moran’s I to measure the spatial correlation of warped noise and the Kolmogorov-Smirnov (K-S) test to assess normality.

Baselines. Following HIWYN [8], we choose the per-frame fixed and independently-sampled noise as oracle baselines for perfect spatial Gaussianity but zero temporal correlation. We choose bilinear, bicubic, and nearest neighbor temporal interpolation as oracle baselines for sufficient temporal correlation but no spatial Gaussianity. We also compare with the recent noise warping algorithms including HI-

WYN [8] and InfRes [11]. In line with these papers, we also include baselines Preserve Your Own Correlation (PYoCo) [12] and Control-A-Video (CaV) [10], which have perfect Gaussianity but zero and insufficient temporal correlation, respectively.

Results. According to Tab. 1 1st section, we observe:

- (1) For Moran’s I, a value close to 0 indicates no spatial

cross-correlation, which is desirable for i.i.d. noise. Our method achieves a Moran’s I index of 0.00014 and a high p-value of 0.84, indicating strong evidence for no spatial autocorrelation. Similarly low Moran’s I values and high pvalues are observed for PYoCo, CaV, HIWYN and InfRes, because they also aim to generate spatially gaussian outputs.

- (2) The K-S test compares the empirical distribution of

the warped noise to a standard normal distribution. A small K-S statistic and a high p-value indicate the two distributions are similar. Our method obtains a K-S statistic of 0.060 and p-value of 0.44, suggesting the warped noise follows a normal distribution. Comparable results are seen for the other Gaussianity-preserving methods.

- (3) In contrast, the bilinear, bicubic, and nearest neigh-

bor warping methods fail to maintain Gaussianity, exhibiting Moran’s I values an order of magnitude higher (0.24 to 0.30) with p-values of 0.0, and K-S statistics 3-6 times larger (0.17 to 0.37) with very low p-values (<0.05). These results provide strong evidence for the presence of spatial autocorrelation and deviation from normality in the warped noise from these interpolation-based methods.

##### 4.2. Efficiency

Noise generation efficiency is measured by wall time profiling on an NVIDIA A100 40GB GPU, generating noise at a resolution of 1024×1024 pixels. We compare with the same baselines as above. According to Tab. 1 2nd section, our method runs faster than the concurrent InfRes and significantly outperforms the most recent published baseline HIWYN by 26×, due to our algorithm’s linear time complexity. The efficiency is one order of magnitude faster than real time, validating our feasibility to apply noise warping on the fly during video diffusion model fine-tuning.

##### 4.3. Video editing via image diffusion

To further validate the effectiveness of our noise warping algorithm, we repurpose off-the-shelf image-to-image diffusion models to perform video-to-video editing tasks in a frame-by-frame manner, without training. Noise is warped using our algorithm and the above baselines based on the RAFT optical flow [53] from input video and fed to two image pre-trained diffusion models: DeepFloyd IF [52] for super-resolution and DifFRelight [18] for portrait relighting. By measuring the quality and temporal consistency of the output video, we can effectively evaluate the spatial Gaussianity and temporal consistency of different noise

- Table 1. Noise warping algorithm benchmarking in terms of Gaussianity, efficiency, and spatial quality and temporal consistency for two image diffusion based applications. ⇑/⇓ indicates a higher/lower value is better.

Noise w/o warping Noise warping method

Fixed Random Bilinear Bicubic Nearest PYoCo CaV HIWYN InfRes Ours

Gaussianity Moran’s I (index) ⇓ -0.00027 0.00019 0.30 0.24 0.26 0.00023 -0.00079 0.0011 0.00036 0.00014 Moran’s I (p-value) ⇑ 0.29 0.36 0.0 0.0 0.0 0.73 0.25 0.11 0.60 0.84 K-S Test (index) ⇓ 0.089 0.075 0.34 0.37 0.17 0.13 0.073 0.062 0.055 0.060 K-S Test (p-value) ⇑ 0.12 0.19 0.0005 0.0004 0.04 0.08 0.27 0.42 0.50 0.44

###### Efficiency at 1024×1024 resolution

GPU time (ms) ⇓ < 1 < 1 4.41 4.33 6.82 3.54 2.31 55.2 2.61 2.14

Super-resolution - DeepFloyd IF LPIPS ⇓ 0.29 0.29 0.60 0.62 0.55 0.28 0.28 0.29 0.28 0.29 SSIM ⇑ 0.88 0.88 0.72 0.70 0.65 0.88 0.88 0.87 0.88 0.88 PSNR ⇑ 29.36 29.41 28.68 28.55 28.59 29.40 29.39 29.31 29.38 29.39 Warping error ⇓ 163.84 233.65 165.90 167.95 244.72 186.63 220.28 164.35 190.82 152.04

Relighting - DifFRelight LPIPS ⇓ 0.33 0.31 0.40 0.41 0.73 0.35 0.35 0.36 0.35 0.33 SSIM ⇑ 0.69 0.77 0.73 0.70 0.38 0.58 0.67 0.64 0.60 0.70 PSNR ⇑ 28.91 29.02 28.87 28.82 28.21 28.83 28.87 28.82 28.81 28.92 Warping error ⇓ 86.65 128.11 47.53 43.57 164.42 95.24 106.77 87.72 87.97 85.82

warping algorithms.

Evaluation metrics. We use LPIPS [74], SSIM [21], and PSNR [21] to measure the quality of the output frames w.r.t. ground truth frames. We use warping error [31] to measure temporal consistency (mean square error) between two adjacent generated frames after flow warping.

###### 4.3.1 DeepFloyd IF video super-resolution

We evaluate noise warping on DeepFloyd IF [52] superresolution using 43 videos from the DAVIS dataset [43]. The videos were downsampled to the 64×64 and superresolved to 256×256.

Results. According to Tab. 1 3rd section, our algorithm outperforms all the baselines in terms of temporal consistency (warping error). Our supplementary video also shows that our algorithm is more stable for the foreground, background, and edges, in contrast to InfRes which is often unstable in the background and HIWNY which is much less stable around moving edges. Our algorithm is comparable to other methods in PSNR, SSIM, and LPIPS image quality metrics, apart from the bilinear, bicubic, and nearest methods which result in low quality generation due to spatial non-Gaussianity. See Fig. 12 in the supplementary material for more details.

###### 4.3.2 DifFRelight video relighting

We evaluate noise warping on DifFRelight [18] portrait video relighting using their own dataset, which includes

- 4 subjects in 4 scenarios: a 180-degree view animation, a 720-degree view animation, a zigzag camera movement se-

quence, and an interpolating camera path through several fixed stage capture positions, all with fixed lighting conditions. During inference, we center crop a 1024×1024 region out of a 1080×1920 Gaussian splat rendering and infer with various noises using conditioned lighting.

Results. According to Tab. 1 4th section, throughout all baseline comparisons, our algorithm shows consistently advantageous scores in both image and temporal metrics, validating its fundamental benefits to the image diffusion model. Although our visual results at first glance are comparable to HIWYN and InfRes in the supplementary Fig. 13 and our webpage, its visual improvements can be seen in the beard regions and skin reflections. We also notice quite low warping error values on the bilinear and bicubic noise inferences, likely coming from the long blurry streaks generated along the flow, while at the same time image quality deteriorates significantly.

##### 4.4. Video diffusion with motion control

###### 4.4.1 Local object motion control

We introduce a novel method for controlling object motion, by leveraging the flows of input templates. These templates include user-defined local region masks and cut-and-drag trajectories that allow users to specify the motion of one or more objects built with a simple, intuitive UI (Fig. 5), and synthetic flows of a camera rotating around 3D objects (Fig. 6).

During inference, we use the precise flow computed from the input template frames to guide noise warping for video generation. This enables our I2V model to apply accurate, localized movements and adjustments to the input

Source Template Ours SG-I2V DragAnything

Baselines. We evaluate our video generation model against five state-of-the-art baselines, SG-I2V [39], MotionClone [34], DragAnything [60], to benchmark its ability to accurately control object and camera movements derived from a given input template. One of the most recent works, SG-I2V, is an I2V model for object motion transfer guided by bounding box trajectories. We adapt our user-defined polygons to bounding boxes as its input.

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

Results. From Fig. 5, Fig. 6, Tab. 2 and our webpage, we observe:

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

- (1) Existing methods struggle to handle complex, local-

ized object motions. Specifically, when specifying local adjustments, such as rotating a dog’s head while keeping the rest of the body static, these methods often fail, applying unnatural translational or global transformations to the entire object.

- (2) We find that SG-I2V frequently misinterprets object-

specific movements as global camera shifts, resulting in scene-wide translations rather than accurate object manipulations.

- (3) DragAnything, which employs single-line trajectory

control, lacks temporal and 3D consistency, leading to significant distortions and reduced fidelity in complex motion scenarios.

- (4) MotionClone also fails to capture subtle object dy-

namics, as it relies on sparse temporal attention for motion guidance and is likely limited by the low spatial resolution of its diffusion features.

- (5) Qualitatively, our model outperforms these baselines

by maintaining high object fidelity and 3D consistency, even in scenarios with intricate or overlapping motions. Notably, our approach preserves object integrity and introduces plausible physical interactions, such as generating realistic splashes when moving a duck within a tub. Extensive user studies and quantitative evaluations validate our superior performance in motion consistency, visual fidelity, and overall realism.

- (6) Our quantitative evaluation matches our qualitative

- Figure 5. Qualitative comparisons of local object motion control. Zoom in for details. The user selects any number of polygons, then scales, rotates, or translates them along arbitrary paths, which are then used to create the warped noise flow.

[Figure 83]

- Figure 6. Qualitative comparisons of camera movement video generation of our method (b) and MotionClone (c) using a turning source video (a).

observations. On both VIPSeg and the 40 videos from our user-study, our method outperforms all the training-based and training-free baselines.

User study. We conducted a comprehensive user study with 40 participants, asking them to evaluate and rate different methods based on their effectiveness in object motion control and maintaining 3D and temporal consistency. Our method stands out significantly, achieving a win percentage of 82% for cut-and-draglocal object motion control like Fig. 5 and 90% for the turnable camera movement control like Fig. 6. The three baselines have substantially lower performance levels. More user study details are included in the supplementary material Sec. 10 and Fig. 15.

image while preserving object structure and faithfully following the intended motion trajectory.

We also provide quantitative benchmarks. Following [39], we use the VIPSeg [37] to benchmark our method on local object motion control, as well as the 40 videos from our user study.

[Figure 84]

###### Davis Input Ours MotionClone

DMT

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Prompt: A snowboarder racing down a snowy slope.

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

- Figure 9. We explore a 3D scene, flying into a given image. Similar to Fig. 8, we take an image as an input and use a monocular depth estimator DepthPro [5] to get a depth map. Then, we use that depth to generate a crudely warped video (note the pixelation on the rough depth warp column when zoomed in) - and from the movement in that video get warped noise. From there, we run our motion-conditioned I2V model.

[Figure 107]

[Figure 108]

[Figure 109]

Original Video Ours MotionClone AnyV2V

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

012FrameFrameFrame

- Figure 10. Comparison of initial frame video editing results across different methods. All methods start with the same edited initial frame derived from the original video.

Prompt: A hot air ballon souring over a scenic village.

- Figure 7. Qualitative comparisons of motion transfer T2V on the DAVIS dataset. Zoom-in needed.

[Figure 119]

- Figure 8. We apply our method to a sequence of frames warped using monocular depth estimation, enabling consistent 3D scene generation from a single image. In this example, we use results from WonderJourney. Zoom-in needed.

###### 4.4.2 Motion transfer and camera movement control

Our method also supports motion transfer and camera movement control, working with both T2V and I2V video diffusion models. By using reference videos and applying noise warping based on their optical flows, it can effectively capture and transfer complex motions.

Datasets. We choose the DAVIS video dataset [43] containing 43 videos of general object motion with ground truth object segmentation annotations, a random subset of 100 videos from the DL3DV dataset [33], and 19 videos generated with WonderJourney [67] that predominantly feature camera movements (Fig. 8), which itself uses depthwarping.

Evaluation metrics. For pixel quality, we calculate Fr´echet Inception Distance (FID) between a set of real and generated frames. For motion controllability, we calculate (1) the mean Interaction over Union (mIoU) of CoTracker’s tracking bounding boxes [26] between ground-truth and generated videos, and (2) the pixel MSE between ground-truth

and generated videos, considering an I2V diffusion model is conditioned on ground-truth prompts, ground-truth initial frames, and ground-truth motion trajectories/flows. For text controllability, we calculate the cosine similarity between the prompt’s CLIP [46] text embedding and the generated frames’ CLIP image embeddings, and average over frames of a generated video. For temporal consistency, we calculate (1) the cosine similarity of the CLIP image embeddings between two consecutive generated frames and average over all pairs in a generated video, and (2) the Fr´echet Video Distance (FVD) [54] between a set of real and generate videos. In addition, we also benchmark on four metrics of VBench [24], specifically for the temporal consistency/smoothness dimension.

Baselines. For the motion transfer T2V scenario, we compare with the recent state-of-the-art methods Diffusion Motion Transfer (DMT) [65], MotionClone [34], and MotionC-

- Table 2. Quantitative comparisons of motion transfer. ⇑/⇓ indicates a higher/lower value is better. Bold indicates the best results. Gray background rows indicate our final model. Dashed lines separate ablation study from baseline benchmarking.

CoTracker Optical Pixel CLIP CLIP FVD VBench ⇑

Training FID mIoU flow MSE text image ×103 Subject Background Motion Temperal free? ⇓ ⇑ err. ⇓ ⇓ ⇑ ⇑ ⇓ consistency consistency smoothness flickering

| | | |
|---|---|---|
|MotionClone SG-I2V Ours<br><br>|✓ ✓ ×<br><br>|Local object motion control on VIPSeg<br><br>85.2 0.71 0.48 0.086 0.31 0.95 1.26 0.88 0.85 0.94 0.90 61.4 0.63 0.84 0.065 0.31 0.97 1.06 0.93 0.95 0.96 0.94 41.1 0.75 0.36 0.039 0.32 0.98 0.47 0.91 0.92 0.97 0.95<br><br>|
|MotionClone SG-I2V DragAnything Ours<br><br>|✓ ✓ × ×<br><br>|Local object motion control on our 40 samples in the user study<br><br>96.6 - 0.80 0.048 0.33 0.98 1.38 0.86 0.93 0.97 0.95 79.9 - 0.64 0.042 0.32 0.98 1.27 0.95 0.95 0.98 0.94 82.8 - 0.62 0.047 0.31 0.97 1.30 0.93 0.95 0.98 0.95 74.3 - 0.56 0.028 0.32 0.98 0.94 0.96 0.95 0.98 0.96<br><br>|
|DMT MotionClone MotionCtrl Ours Ours-CogVideoX-2B<br><br>|✓ ✓ × × ×<br><br>|Motion transfer T2V on DAVIS<br><br>- 0.85 0.28 - 0.31 0.95 - 0.86 0.92 0.94 0.91<br>- 0.75 0.38 - 0.32 0.93 - 0.78 0.89 0.86 0.81 0.47 0.85 - 0.32 0.97 - 0.97 0.93 0.98 0.92<br><br>- 0.70 0.41 - 0.33 0.98 - 0.88 0.93 0.97 0.89<br><br>- 0.64 0.48 - 0.32 0.95 - 0.89 0.91 0.97 0.90<br><br><br>|
|MotionClone ImageConductor Original CogVideoX-5B Ours (γ = 0.5) Our (γ = 0.9) Our (γ = 0.8) Our (γ = 0.4) Our (γ = 0.2) Our (33% data) Our (12.5% data)<br><br>|✓ × ✓ × × × × × × ×<br><br>|Motion transfer I2V on DAVIS<br><br>99.4 0.72 0.42 0.068 0.31 0.94 1.84 0.75 0.85 0.92 0.87<br><br>104.6 0.66 0.64 0.072 0.31 0.93 1.58 0.77 0.88 0.93 0.90<br><br>76.62 0.52 0.67 0.088 0.31 0.96 1.36 0.85 0.91 0.96 0.92 78.6 0.74 0.36 0.053 0.31 0.97 1.21 0.88 0.92 0.98 0.93 92.5 0.50 0.65 0.072 0.31 0.95 1.59 0.80 0.89 0.94 0.91 80.6 0.68 0.47 0.067 0.31 0.96 1.50 0.85 0.91 0.96 0.92<br><br>77.7 0.74 0.36 0.056 0.31 0.97 1.27 0.87 0.91 0.97 0.93 77.1 0.74 0.37 0.058 0.32 0.97 1.29 0.86 0.91 0.97 0.93<br><br><br>100.1 0.73 0.40 0.066 0.31 0.97 1.46 0.85 0.90 0.97 0.92<br><br>105.2 0.71 0.39 0.072 0.31 0.96 1.93 0.84 0.89 0.97 0.91<br><br><br>|
|MotionClone ImageConductor Ours<br><br>|✓ × ×<br><br>|Camera movement transfer I2V on DL3DV<br><br>82.7 0.71 0.44 0.104 0.33 0.94 1.11 0.74 0.85 0.91 0.86 89.2 0.61 0.78 0.068 0.31 0.95 0.91 0.85 0.90 0.95 0.93 48.4 0.83 0.20 0.046 0.32 0.97 0.34 0.88 0.92 0.97 0.93<br><br>|
|MotionClone ImageConductor Ours<br><br>|✓ × ×<br><br>|Camera movement transfer I2V on WonderJourney<br><br>177.9 0.81 0.17 0.103 0.32 0.96 1.93 0.75 0.87 0.93 0.87 166.1 0.79 0.39 0.085 0.32 0.94 1.63 0.79 0.88 0.93 0.90 128.3 0.85 0.15 0.072 0.31 0.98 1.55 0.82 0.91 0.98 0.94<br><br>|

trl [58]. For the motion transfer I2V, we compare MotionClone and ImageConductor [32]) as DMT does not take an image as input.

In addition, we demonstrate video first-frame editing, a challenge where a user starts with an original video and an edited version of its initial frame. The goal is to seamlessly propagate the edits made to the first frame throughout the entire video while preserving the original motion. We qualitatively compare with MotionClone [34] and the state-ofthe-art video editing method AnyV2V [29] on real videos with photoshopped first frames.

We also source a few images for image-based depth warping, where we take an image, use a monocular depth estimator, DepthPro [5], to get a depth map, and crudely warp it to simulate a desired camera trajectory.

Results. We present both qualitative and quantitative comparisons with baselines in Tab. 2, Fig. 7, Fig. 8, Fig. 9, Fig. 10, and our webpage. We observe:

- (1) Our superior object motion transfer: On the DAVIS

dataset, which includes object motion along with some degree of camera movement, our method demonstrates improved motion fidelity and overall video quality, as measured by Vbench. In particular, in the I2V setting, where both the initial frame and the source video are provided, our method achieves significantly better scores in FID, FVD, and motion metrics, indicating much closer reconstructions of the ground truth videos.

- (2) Our superior camera movement control: On the

DL3DV and WonderJourney datasets, which involve substantial camera movement, our method notably outperforms MotionClone in both motion fidelity and general video quality. This highlights our method’s ability to effectively replicate intricate camera movements while maintaining visual coherence. For our depth-warping example Fig. 9, our results are far better than simply warping an image from its depth map, resulting in a smooth, realistic camera trajec-

tory. See our webpage for videos.

(3) Our superior video first-frame editing: In Fig. 10, our method seamlessly integrates the added object into the scene while accurately preserving the camera movement from the original video. In contrast, both baselines exhibit significant identity loss: MotionClone generates additional, unintended objects, and in AnyV2V, the foreground object gradually disappears. This demonstrates the superiority of our method in maintaining the original video’s motion while faithfully preserving the identity of the object added to the first frame.

###### 4.4.3 Ablation studies

In Tab. 2 for the DAVIS I2V task, we compare our method with a variant that excludes motion conditioning using warped noise (“Original CogVideoX-5B”), relying solely on textual prompts describing the objects. We observe: (1) Better video reconstruction: Our method, which incorporates motion conditioning, achieves superior FID, FVD, and CoTracker mIoU scores, indicating more accurate reconstruction of the source video. This is because textual prompts and the initial frame alone are insufficient to capture a video’s future dynamics, whereas incorporating real video-derived motion guidance enables the generation of more realistic sequences. (2) Improved video quality: By utilizing warped noise for motion conditioning, our approach not only maintains but also enhances overall video quality, as measured by Vbench, demonstrating that integrating realistic motion cues improves the plausibility of the generated videos without compromising quality.

Further exploring the effects of degradation, we find that as the degradation values γ increase, the motion control becomes tighter - resulting in higher optical flow and CoTracker-mIoU scores, along with a closer per-frame similarity to target videos on the I2V DAVIS task. We find that, in general, γ ≈ 0.5 is a good value for most tasks.

We also perform ablations on dataset size, comparing models trained on different fractions of our dataset: training with a fraction of dataset yields worse performance than our full model.

In addition, we perform an ablation where we use the weaker base model CogVideoX-2B, and find its performance is weaker than our main T2V model, based on CogVideoX-5B.

#### 5. Conclusion

In this work, we introduce a novel and faster-than-real-time noise warping algorithm that seamlessly incorporates motion control into video diffusion noise sampling, bridging the gap between chaos and order in generative modeling. By leveraging this noise warping technique to preprocess

video data for video diffusion fine-tuning, we provide a unified paradigm for a wide range of user-friendly, motioncontrollable video generation applications. Extensive experiments and user studies demonstrate the superiority of our method in terms of visual quality, motion controllability, and temporal consistency, making it a robust and versatile solution for motion control in video diffusion models.

#### Acknowledgments

We would like to express our gratitude to Stephan Trojansky and Jeffrey Shapiro for their initial and ongoing executive support; Sebastian Sylwan, Daniel Heckenberg, Jitendra Agarwal, Matheus Le˜ao, and Sungmin Lee for their IT support; Xueming Yu and David George for their hardware support; Jennifer Lao and Lianette Alnaber for their operational support; and Winnie Lin, Ahmet Tasel, Yiqun Mei, Lukas Lepicovsky, Rahul Garg, Ashish Rastogi, Ritwik Kumar, Cornelia Carapcea, and Girish Balakrishnan for their insightful technical discussions.

#### Social impact statement

Our work contributes to the growing field of video generative models by advancing motion-controllable video generation, which has the potential to revolutionize creative industries such as filmmaking and animation. By introducing a computationally efficient and accessible framework, our method democratizes high-quality video generation, enabling creators, developers, and artists to produce dynamic content with minimal resources or specialized training.

However, we acknowledge the potential misuse of such technology, including the creation of deepfakes or misleading media. To mitigate these risks, we advocate for responsible use, proper content labeling, and the integration of detection mechanisms to ensure ethical deployment. Our approach also emphasizes compatibility with diverse models, encouraging transparency and collaboration within the research community to address societal concerns effectively while maximizing the positive impact of this technology.

#### References

- [1] Luca Savant Aira, Antonio Montanaro, Emanuele Aiello, Diego Valsesia, and Enrico Magli. Motioncraft: Physicsbased zero-shot video generation. In NeurIPS, 2024. 2, 4
- [2] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In ICCV, 2021. 7, 17
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv, 2023. 2, 3
- [4] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023. 2, 3
- [5] Aleksei Bochkovskii, Ama¨el Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, and Vladlen Koltun. Depth pro: Sharp monocular metric depth in less than a second, 2024. 10, 11
- [6] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023. 3
- [7] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. 2024. URL https://openai. com/research/videogeneration-models-as-world-simulators, 2024. 2, 3
- [8] Pascal Chang, Jingwei Tang, Markus Gross, and Vinicius C Azevedo. How i warped your noise: a temporally-correlated noise prior for diffusion models. In ICLR, 2024. 2, 3, 4, 5, 7, 16, 17
- [9] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv, 2023. 3
- [10] Weifeng Chen, Yatai Ji, Jie Wu, Hefeng Wu, Pan Xie, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable text-to-video generation with diffusion models. arXiv, 2023. 7
- [11] Yitong Deng, Winnie Lin, Lingxiao Li, Dmitriy Smirnov, Ryan Burgert, Ning Yu, Vincent Dedun, and Mohammad H Taghavi. Infinite-resolution integral noise warping for diffusion models. arXiv, 2024. 5, 7
- [12] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, MingYu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. In ICCV, 2023. 7
- [13] Daniel Geng, Charles Herrmann, Junhwa Hur, Forrester Cole, Serena Zhang, Tobias Pfaff, Tatiana Lopez-Guevara, Carl Doersch, Yusuf Aytar, Michael Rubinstein, et al. Motion prompting: Controlling video generation with motion trajectories. arXiv, 2024. 2, 4
- [14] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. In ICLR, 2024. 2, 4

- [15] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls to text-to-video diffusion models. In ECCV, 2024. 2, 4
- [16] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-toimage diffusion models without specific tuning. In ICLR,

2024. 2, 3, 4, 7, 17

- [17] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv, 2024. 2,

- 4

[18] Mingming He, Pascal Clausen, Ahmet Levent Tas¸el, Li Ma, Oliver Pilarski, Wenqi Xian, Laszlo Rikker, Xueming Yu, Ryan Burgert, Ning Yu, et al. Diffrelight: Diffusion-based facial performance relighting. In SIGGRAPH Asia, 2024. 3,

- 5, 7, 8

- [19] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv, 2022. 3
- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 3
- [21] Alain Hore and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In ICPR, 2010. 8
- [22] Chen Hou, Guoqiang Wei, Yan Zeng, and Zhibo Chen. Training-free camera control for video generation. arXiv,

2024. 4

- [23] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. 7
- [24] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, 2024. 10
- [25] Yash Jain, Anshul Nasery, Vibhav Vineet, and Harkirat Behl. Peekaboo: Interactive video generation via maskeddiffusion. In CVPR, 2024. 2, 4
- [26] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. arXiv, 2023. 10
- [27] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022. 3
- [28] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In CVPR, 2024. 3
- [29] Max Ku, Cong Wei, Weiming Ren, Huan Yang, and Wenhu Chen. Anyv2v: A plug-and-play framework for any videoto-video editing tasks. arXiv, 2024. 2, 4, 11
- [30] Zhengfei Kuang, Shengqu Cai, Hao He, Yinghao Xu, Hongsheng Li, Leonidas Guibas, and Gordon Wetzstein. Collaborative video diffusion: Consistent multi-video generation with camera control. arXiv, 2024. 2, 4
- [31] Wei-Sheng Lai, Jia-Bin Huang, Oliver Wang, Eli Shechtman, Ersin Yumer, and Ming-Hsuan Yang. Learning blind video temporal consistency. In ECCV, 2018. 8

- [32] Yaowei Li, Xintao Wang, Zhaoyang Zhang, Zhouxia Wang, Ziyang Yuan, Liangbin Xie, Yuexian Zou, and Ying Shan. Image conductor: Precision control for interactive video synthesis. arXiv, 2024. 11
- [33] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In CVPR, 2024. 10
- [34] Pengyang Ling, Jiazi Bu, Pan Zhang, Xiaoyi Dong, Yuhang Zang, Tong Wu, Huaian Chen, Jiaqi Wang, and Yi Jin. Motionclone: Training-free motion cloning for controllable video generation. arXiv, 2024. 2, 4, 9, 10, 11
- [35] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv, 2024. 2
- [36] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In ICLR, 2022. 3
- [37] Jiaxu Miao, Yunchao Wei, Yu Wu, Chen Liang, Guangrui Li, and Yi Yang. Vspw: A large-scale dataset for video scene parsing in the wild. In CVPR, 2021. 9
- [38] Chong Mou, Mingdeng Cao, Xintao Wang, Zhaoyang Zhang, Ying Shan, and Jian Zhang. Revideo: Remake a video with motion and content control. ArXiv, abs/2405.13865, 2024. 2, 4
- [39] Koichi Namekata, Sherwin Bahmani, Ziyi Wu, Yash Kant, Igor Gilitschenski, and David B Lindell. Sg-i2v: Self-guided trajectory control in image-to-video generation. arXiv, 2024.

- 2, 4, 9

[40] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv, 2021.

- 3

- [41] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 3
- [42] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv, 2023. 3
- [43] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv, 2017. 8, 10
- [44] Zhiwu Qing, Shiwei Zhang, Jiayu Wang, Xiang Wang, Yujie Wei, Yingya Zhang, Changxin Gao, and Nong Sang. Hierarchical spatio-temporal decoupling for text-to-video generation. In CVPR, 2024. 3
- [45] Haonan Qiu, Zhaoxi Chen, Zhouxia Wang, Yingqing He, Menghan Xia, and Ziwei Liu. Freetraj: Tuning-free trajectory control in video diffusion models. arXiv, 2024. 2, 4
- [46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language super-

vision. In ICML, 2021. 3, 10

- [47] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020. 3
- [48] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 3
- [49] Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, et al. Motion-i2v: Consistent and controllable image-to-video generation with explicit motion modeling. In SIGGRAPH, 2024. 2, 4
- [50] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 3, 6
- [51] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021. 3
- [52] StabilityAI. Deepfloyd if. URL https://github.com/deepfloyd/IF?tab=readme-ov-file, 2023. 3, 5, 7, 8
- [53] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In ECCV, 2020. 2, 3, 5, 7
- [54] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv, 2018. 10
- [55] Oliver Villar. Learning Blender. Addison-Wesley Professional, 2021. 3
- [56] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. Cogvlm: Visual expert for pretrained language models. arXiv, 2024. 7
- [57] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. In NeurIPS, 2024. 2, 4
- [58] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In SIGGRAPH, 2024. 2, 4, 11
- [59] Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T Barron, and Aleksander Holynski. Cat4d: Create anything in 4d with multi-view video diffusion models. arXiv, 2024. 2, 4
- [60] Weijia Wu, Zhuang Li, Yuchao Gu, Rui Zhao, Yefei He, David Junhao Zhang, Mike Zheng Shou, Yan Li, Tingting Gao, and Di Zhang. Draganything: Motion control for anything using entity representation. In ECCV, 2024. 2, 4, 9
- [61] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images with video diffusion priors. In ECCV,

2024. 3, 4

- [62] Dejia Xu, Weili Nie, Chao Liu, Sifei Liu, Jan Kautz, Zhangyang Wang, and Arash Vahdat. Camco: Cameracontrollable 3d-consistent image-to-video generation. arXiv,

2024. 2, 4

- [63] Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. Direct-a-video: Customized video generation with userdirected camera movement and object motion. In SIGGRAPH, 2024. 2, 4
- [64] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv, 2024. 2, 3, 4, 5, 7
- [65] Danah Yatim, Rafail Fridman, Omer Bar-Tal, Yoni Kasten, and Tali Dekel. Space-time diffusion features for zero-shot text-driven motion transfer. In CVPR, 2024. 2, 4, 10
- [66] Wenjie Yin, Yi Yu, Hang Yin, Danica Kragic, and M˚arten Bj¨orkman. Scalable motion style transfer with constrained diffusion generation. In AAAI, 2024. 2, 4
- [67] Hong-Xing Yu, Haoyi Duan, Junhwa Hur, Kyle Sargent, Michael Rubinstein, William T Freeman, Forrester Cole, Deqing Sun, Noah Snavely, Jiajun Wu, et al. Wonderjourney: Going from anywhere to everywhere. In CVPR, 2024. 3, 10
- [68] Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. In ICLR, 2024. 2, 3
- [69] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv, 2024. 4
- [70] Zongsheng Yue, Jianyi Wang, and Chen Change Loy. Resshift: Efficient diffusion model for image superresolution by residual shifting. NeurIPS, 2024. 3
- [71] David Junhao Zhang, Roni Paiss, Shiran Zada, Nikhil Karnad, David E Jacobs, Yael Pritch, Inbar Mosseri, Mike Zheng Shou, Neal Wadhwa, and Nataniel Ruiz. Recapture: Generative video camera controls for user-provided videos using masked video fine-tuning. arXiv, 2024. 4
- [72] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and MingHsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. arXiv, 2024. 2
- [73] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 3
- [74] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 8
- [75] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent selfattention for long-range image and video generation. arXiv,

2024. 4

## Go-with-the-Flow: Motion-Controllable Video Diffusion Models Using Real-Time Warped Noise

### Supplementary Material

#### 6. Gaussianity preservation of our noise warping algorithm

In this section, we discuss our noise warping algorithm, providing a formal proof of its Gaussianity preservation properties. We also present an illustrative example that demonstrates how noise that undergoes expansion and subsequent contraction returns to its original state, showcasing how our noise warping algorithm maintains the underlying Gaussian distribution throughout the warping process.

Proof. For each (x,y) ∈ V , R(x,y) is a collection of upsampled noise Xi, where

1 √

S d

q(x,y) d

] + E[

E[Xi] = E[

(Zi −

)] = 0

d

q(x,y) d

1 √

S d

Var(Xi) = Var(

) + Var(

(Zi −

))

d

d − 1 d

1 d2

1 d

Zj d

Var(

Zi −

=

+

)

j̸=i

(d − 1)2 + (d − 1) d2

1 d2

1 d

1 d

=

+

=

,

where we used the fact that q(x,y) and Zi’s are i.i.d. standard Gaussians. Since Xi is constructed as a weighted sum of Gaussians, itself is also a Gaussian. Moreover, for i ̸= j, we compute

###### Cov(Xi,Xj)

q(x,y) d

1 √

S d

q(x,y) d

1 √

S d

=Cov(

(Zi −

(Zj −

+

),

+

))

d

d

1 d

1 d2

S d

S d

E[(Zi −

)(Zj −

+

=

)]

E[ZiS] d

E[S2] d2

1 d

1 d2

(0 − 2

+

+

)

=

1 d

2 d

1 d

1 d2

(−

+

+

) = 0.

=

Hence all Xi’s are independent.

For each (x′,y′) ∈ V ′, if degG((x′,y′)) = 0, then q′(x′,y′) is sampled as an independent standard Gaussian. Otherwise, the output noise pixel q′(x′,y′) is built as a weighted sum of R(x,y).pop() for each edge ((x,y),(x′,y′)) ∈ E, where R(x,y).pop() is an independent Gaussian of mean 0 and variance deg 1

G((x,y)). Hence q′(x′,y′) is also a Gaussian with mean 0. The variable s after executing the inner for loop thus represents the variance of q′(x′,y′), so the renormalization at the end brings

q′(x′,y′) back to a standard Gaussian. Since the composing Xi’s are independent, the resulting noise q′ should also have an independent Gaussian in each pixel.

| |
|---|

Example 1 (Exact recovery of expansion-contraction). Consider the following evolution of noise across three frames with forward flows fi→j going from frame i to frame j with i + 1 = j (and backward flow if i − 1 = j). Suppose at frame 1, a pixel v ∈ D with density 1 has noise q. Suppose further that va′ is a pixel at frame 2 such that f1−→12(va′ ) = {v}, and vb′ ∈ D is the only pixel at frame 2 such that f1−→12(vb′) = ∅ and f2→1(vb′) = v. This represents the scenario where v is expanded into two pixels va′ ,vb′. Then Algorithm 1 with forward flow f1→2 and backward flow f2→1 will result in va′ having density 1/2 and noise 2q + √12(Z

a−Zb

2 ), and vb′ having density 1/2 and noise

q 2 + √12(Z

b−Za

2 ), where Za and Zb are i.i.d. standard Gaussians. Now, from frame 2 to frame 3, suppose there exists a pixel v′′ such that f2−→13(v′′) = {va′ ,vb′}, i.e., they both va′ and vb′ contract to v′′, and that f3→2(D) ∩ {va′ ,vb′} = ∅. Then Algorithm 1 with forward flow f2→3 and backward flow f3→2 will result in v′′ having density 1 and noise q, hence deterministically recovering the noise and density of v in frame 0.

#### 7. Qualitative results of training-free image diffusion based video editing

Noise warping methods that do not preserve Gaussianity degrade per-frame performance, as originally pointed out in [8]. For example, using nearest neighbor and bilinear interpolation destroys the Gaussianity (see Fig. 11) and consequently deteriorates the per-frame performance on pretrained image-to-image diffusion models (see Fig. 12 and Fig. 13).

#### 8. The advantage of noise warping

By using noise warping as a condition for motion, we effectively discard all structural information from our input video that cannot be inferred from motion alone. This can be advantageous, as demonstrated in Fig. 14. MotionClone does not use optical flow to guide the video trajectory, instead relying on manipulating activations within the diffusion model. As a result, the windmill gains an extra set of arms, whereas our method, which relies solely on motion information from optical flow via warped noise, does not introduce such artifacts.

[Figure 120]

- Figure 11. A direct visualization of the noise produced by our noise warping algorithm, HIWYN [8], bilinear, and nearest neighbor interpolations. The forward movement in this long roller-coaster video forces the noise to expand significantly. Early in the video, the HIWYN baseline produces visibly non-Gaussian results. See the full video on our webpage.

[Figure 121]

- Figure 12. Using different noise warping algorithms on DeepFloyd IF for video super-resolution on the DAVIS dataset.

[Figure 122]

Figure 13. Using different noise warping algorithms on DifFRelight for portrait video relighting.

#### 9. Comparison to the video diffusion base model without finetuning

Interestingly, video diffusion models respond to noise warping even without training. In Fig. 14 the rightmost column, even though the per-frame quality suffers, the flow of the output video still roughly follows the flow of the warped noise. However, because warped noise is statisically distinct from the pure Gaussian noise CogVideoX was trained on, without fine-tuning it can result in visual artifacts.

#### 10. User study settings and statistics

Fig. 15 presents our user study questionnaires and statistics for two applications: (1) local object motion control, and (2) turnable camera movement video generation. Our questions focus on users’ overall subjective preference, controllability, and temporal consistency.

#### 11. Model Agnostic

Our method is data- and model-agnostic. It can be used to add motion control to arbitrary video diffusion models by only processing the noise sampling during fine-tuning. For example, it also works with AnimateDiff [16] fine-tuned on the WebVid dataset [2] (the weights for this model on our

GitHub page). See its qualitative results in Fig. 16. Since release, the community has also trained a version of Go-withthe-Flow on HunyuanVideo (linked on our GitHub page). Therefore, our method will generalize to future more advanced video diffusion base model.

#### 12. Pseudo code

See Fig. 17 for our noise warping pseudo code. See our source code and model checkpoints on GitHub.

Input Video

Optical Flow Ours MotionClone

[Figure 123]

[Figure 124]

[Figure 125]

Frame0Frame1

- Frame2

CogVideoX

(No Finetuning)

- Frame3Frame4Frame6Frame5Frame7

- Figure 14. We show a cut-and-drag animation of a windmill rotating clockwise, next to the derived optical flow, our outputs, a baseline and an ablation. Note that the input video column appears to have two sets of panels because it’s being cut and dragged over itself to create rotational motion. When using noise warping is better: Per-frame structural information can poison the result of MotionClone, giving the windmill an extra set of arms - whereas ours only receives motion information from optical flow alone via warped noise (there are no double-windmills in the optical flow patterns). Ablation in rightmost column: warped noise with γ = .5 on the CogVideoX base model before we fine-tune it. Because warped noise is statisically distinct from the pure Gaussian noise CogVideoX was trained on, without fine-tuning it can result in visual artifacts. Note how although the per-frame quality suffers here, it still picks up on motion queues from the warped noise (the camera zooms into the windmill).

[Figure 126]

(a) User study interface and questions for local object motion control, corresponding to Fig. 5 in the main paper.

[Figure 127]

(b) User study interface and questions for turnable camera movement video generation, corresponding to Fig. 6 in the main paper.

[Figure 128]

(c) User study statistics for local object motion control on the first question “Which video is the best overall?”

[Figure 129]

(d) User study statistics for local object motion control on the second question “Which video best aligns with the user intent for controlling the object movement based on the input?”

[Figure 130]

(e) User study statistics for local object motion control on the third question “Which video best preserves the intended camera movement from the input?”

[Figure 131]

(f) User study statistics for local object motion control on the fourth question “Which video maintains the most consistent and stable motion throughout?”

[Figure 132]

(g) User study statistics for motion transfer on the first question “Which video has better overall quality?”

- Figure 15. User study questionnaires screenshots and statistics. For all the questions of both applications, our method (the rightmost bar plot) significantly wins the most user preferences.

###### Go-with-the-Flow + AnimateDiff Output Videos

Soaring through a post-apocalyptic wasteland

Soaring over waterfalls in National Park

Soaring through New York City

Soaring through a forest

Soaring through a magical crystal cave

Soaring through the grand canyon

Soaring through the great wall of China

Input Video Warped Noise

[Figure 133]

- Figure 16. Fine-tuning AnimateDiff with our warped noise flow. We used Go-with-the-Flow to fine-tune AnimateDiff T2V, and display the results above. The input video is on the left, and from that video we derive warped noise which is used to initialize AnimateDiff on the columns to its right with different text prompts.

- 1 def warp_noise(prev_frame, cur_frame, prev_noise, prev_weight):

- 2

- 3 height, width, _ = prev_frame.shape

- 4

- 5 flow = optical_flow(prev_frame, cur_frame) # Agnostic to the optical flow algorithm

- 6 backwards_flow = -flow # A cheap approximation of optical_flow(cur_frame, prev_frame)

- 7

- 8 expansion_noise = zeros(height, width)

- 9 contraction_noise = prev_noise.copy()

- 10

- 11 expansion_mask = ones (height, width, type=bool)

- 12 contraction_mask = zeros(height, width, type=bool)

- 13

- 14 for x in range(width): for y in range(height):

- 15 dx, dy = flow[x,y]

- 16 if 0 <= x+dx <= width-1 and 0 <= y+dy <= height-1:

- 17 # This particle stays in bounds

- 18 expansion_mask [x+dx, y+dx] = False

- 19 contraction_mask[x , y ] = True # Contraction mask is True where

- 20

- 21 for x in range(width): for y in range(height):

- 22 if expansion_mask[x, y]:

- 23 dx, dy = backwards_flow[x,y]

- 24 expansion_noise [x, y] = prev_noise[x+dx, y+dy]

- 25

- 26 # We’ve decided which source pixels are involved in contraction and expansion now

- 27 contraction_noise &= contraction_mask

- 28 expansion_noise, contraction_noise, cur_weight = jointly_regaussianize_and_rebalance_weights(

- 29 expansion_noise, contraction_noise, prev_weight

- 30 ) # Regaussianize all noise values here, and divide the weights by the number of pixels in each bin

- 31

- 32 contraction_weight = zeros(height, width)

- 33 for x in range(width): for y in range(height):

- 34 if contraction_mask[x, y]:

- 35 # Contraction treats the noise pixels as particles, each moving from the source to the

- 36 # destination with this flow

- 37 dx, dy = flow[x,y]

- 38 # Contraction is a weighted sum of source pixels to a destination pixel

- 39 pixel_weight = cur_weight[x, y]

- 40 # Sum all the source noise pixels that contract to the same destination

- 41 contraction_noise [x+dx, y+dy] += prev_noise[x, y] * pixel_weight

- 42 # When we multiply a noise pixel by a weight, the variance changes by that weight squared

- 43 contraction_weight[x+dx, y+dy] += pixel_weight ** 2

- 44 contraction_noise /= sqrt(contraction_weight) # Adjust the variance of the summed contracted noise

- 45

- 46 # Mixing contraction and expansion noises with their respective masks

- 47 cur_noise = contraction_noise & contraction_mask + expansion_noise & expansion_mask

- 48

- 49 return cur_noise, cur_weight

Figure 17. Our noise warping pseudo code.

