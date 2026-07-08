# arXiv:2604.00093v1[cs.CV]31Mar2026

## RawGen: Learning Camera Raw Image Generation

Dongyoung Kim1 , Junyong Lee1∗ , Abhijith Punnappurath1∗, Mahmoud Afifi1∗ , Sangmin Han2 , Alex Levinshtein1, and Michael S. Brown1

1AI Center - Toronto, Samsung Electronics 2Yonsei University ∗Equal contribution Project Page: https://dy112.github.io/rawgen-page/

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

Target camera: Samsung NX2000 Canon EOS-1Ds Mark III Fujifilm X-M1

Text prompt: Two kids sitting on swings, laughing together.

Fig. 1: We present RawGen, a diffusion-based method for generating realistic camera raw images. RawGen produces a latent representation of linear CIE XYZ images, conditioned on an sRGB image or a text prompt. The latent is decoded to CIE XYZ and mapped to arbitrary target camera raw spaces.

Abstract. Cameras capture scene-referred linear raw images, which are processed by onboard image signal processors (ISPs) into display-referred 8-bit sRGB outputs. Although raw data is more faithful for low-level vision tasks, collecting large-scale raw datasets remains a major bottleneck, as existing datasets are limited and tied to specific camera hardware. Generative models offer a promising way to address this scarcity; however, existing diffusion frameworks are designed to synthesize photofinished sRGB images rather than physically consistent linear representations. This paper presents RawGen, to our knowledge the first diffusionbased framework enabling text-to-raw generation for arbitrary target cameras, alongside sRGB-to-raw inversion. RawGen leverages the generative priors of large-scale sRGB diffusion models to synthesize physically meaningful linear outputs, such as CIE XYZ or camera-specific raw representations, via specialized processing in latent and pixel spaces. To handle unknown and diverse ISP pipelines and photo-finishing effects in diffusion-model training data, we build a many-to-one inverse-ISP dataset where multiple sRGB renditions of the same scene generated using diverse ISP parameters are anchored to a common scene-referred target. Fine-tuning a conditional denoiser and specialized decoder on this dataset allows RawGen to obtain camera-centric linear reconstructions that effectively invert the rendering pipeline. We demonstrate RawGen’s superior performance over traditional inverse-ISP methods that assume a fixed ISP. Furthermore, we show that augmenting training pipelines with RawGen’s scalable, text-driven synthetic data can benefit downstream low-level vision tasks.

### 1 Introduction

Camera sensors natively record scene-referred linear raw data. Raw images are subsequently processed by onboard image signal processors (ISPs) through a lossy rendering pipeline to produce display-referred outputs, typically in the 8bit sRGB format. While raw captures preserve physically meaningful radiometric information, they are rarely available at scale and often tied to specific camera models. Collecting large raw datasets is costly and labor-intensive, and data must be re-captured whenever the sensor or the camera pipeline changes, limiting reproducibility and cross-device coverage. This scarcity remains a fundamental bottleneck for low-level vision and computational photography research that relies on linear, sensor-centric measurements.

Generative models offer a potential route to mitigate this limitation. Recent diffusion-based frameworks have achieved remarkable success in producing high-fidelity, semantically coherent sRGB images [23,29,30,36,45,47,49,63–66]. In particular, text-to-image (T2I) systems [7, 10, 19, 24, 50, 53, 54, 57, 60, 74, 76] enable strong semantic controllability via textual prompts, facilitating scalable and diverse image synthesis. However, these models operate primarily in the 8-bit sRGB domain, which is optimized for display and storage rather than preserving physical linearity. This representation typically includes nonlinear photofinishing operations (e.g., gamma correction, tone mapping, and color styling) that distort the underlying radiometric structure of the scene. Consequently, diffusion model outputs are not directly compatible with scene-referred linear signals, such as camera raw measurements or device-independent linear representations that can serve as canonical references.

A natural strategy is to reconstruct linear radiometric data from diffusiongenerated sRGB images using inverse-ISP techniques [3,9,55,70]. However, most existing inverse-ISP methods are trained on paired datasets generated under a fixed imaging pipeline, such as software ISPs (e.g., RawPy, DCRAW) or a specific in-camera ISP configuration. These methods therefore implicitly assume a single camera response and a known sequence of color and tone transformations. In contrast, diffusion models are trained on large-scale sRGB corpora that encompass diverse and heterogeneous photo-finishing styles originating from unknown ISPs and post-processing pipelines.

This discrepancy creates a fundamental domain gap. Conventional inverseISP models expect sRGB inputs produced by a fixed and known rendering pipeline, whereas diffusion-generated sRGB images entangle unknown, diverse, and often highly nonlinear photo-finishing effects. Directly applying standard inverse-ISP methods to diffusion model outputs fails to reliably recover a consistent scene-referred linear representation. Enabling physically meaningful linear signal synthesis from generative models therefore remains an open challenge.

In this paper, we introduce RawGen, a diffusion-based framework designed to bridge this gap. RawGen preserves the strong semantic priors and promptability of large-scale diffusion models while incorporating learned latent-level and pixel-level unprocessing to recover a canonical scene-referred linear representation from diverse sRGB inputs. Rather than assuming a fixed inverse mapping,

RawGen is trained to suppress uncontrolled and heterogeneous photo-finishing effects implicitly embedded in diffusion-generated images, thereby enabling consistent and physically grounded linear synthesis.

The central idea of RawGen is to factor out ISP-induced variability and recover a shared linear anchor that remains consistent across multiple sRGB versions of the same underlying scene. To achieve this, we adopt a many-toone reconstruction objective in which multiple photo-finished sRGB images of a single scene serve as inputs and are mapped to a common linear reference. This formulation explicitly encourages the recovered linear representation to remain invariant to differences in post-processing operations—such as tone mapping, gamma correction, and color styling—that are embedded in the sRGB inputs, while preserving the underlying scene structure and semantic content.

With this learned invariance, RawGen supports two complementary modes of generation. First, it converts an sRGB image into a canonical scene-referred linear representation that is robust to diverse post-processing styles. Second, it maps nonlinear image representations produced from text prompts by a pretrained diffusion prior into canonical linear representations. When a target camera raw space is required, the canonical linear output can be transformed into the camera-specific raw domain using standard color-space mappings and camera metadata, enabling camera-agnostic raw synthesis without retraining (Fig. 1).

We demonstrate that RawGen consistently outperforms conventional inverseISP methods that rely on fixed imaging assumptions, yielding a more stable and scene-referred linear representation under diverse and uncontrolled photofinishing variations. Moreover, RawGen enables scalable text-driven raw generation as a practical source of training data for downstream tasks such as illuminant estimation, neural ISP learning, and denoising, providing broad scene diversity without additional capture effort or scene asset design.

#### Contribution

In summary, our main contributions are as follows:

- • We propose RawGen, a diffusion-based camera-agnostic raw generation framework that preserves strong generative priors while unprocessing diverse sRGB inputs with unknown photo-finishing into a canonical scene-referred linear domain via latent- and pixel-level processing.
- • We introduce a many-to-one linear reconstruction task and propose a method to construct a new type of dataset that maps multiple photo-finished sRGB observations to a single common linear reference, enabling robustness to heterogeneous and unknown ISP transformations.
- • We develop, to our knowledge, the first unified framework enabling text-driven synthesis of physically meaningful linear and camera-specific raw data for arbitrary target cameras, and demonstrate that the resulting synthetic data alleviates data acquisition challenges in downstream low-level vision tasks.

### 2 Related Works

Forward & Inverse Camera ISP Pipelines. Traditional camera hardware ISPs consist of carefully engineered signal-processing stages [22], such as denoising, demosaicing, white balancing, color correction, tone mapping, and more, often tuned to produce a manufacturer-specific visual aesthetic. Because incamera pipelines are proprietary and largely inaccessible, the rendered sRGB photographs available on the internet reflect diverse and unknown photo-finishing operations. Large-scale generative models trained on such data therefore inherit these uncontrolled rendering characteristics, producing outputs in the nonlinear sRGB domain with implicit and heterogeneous ISP effects.

Recent work has explored learning-based ISPs that replace handcrafted modules with end-to-end neural mappings (e.g., [28,32,33,52,75]). These networks are trained to convert raw images, typically from a specific camera, to their display-referred outputs assuming a one-to-one relationship between the sRGB images and their originating ISPs.

Recovering linear sensor data from rendered images has long been studied for low-level vision. Early approaches to raw reconstruction relied on calibrationbased techniques [13,14,21,27,38,43]. Later work, such as UPI [11], performed sRGB-to-raw reconstruction by sequentially inverting the ISP with non-learnable parametric operations. Nam et al. [44] proposed one of the earliest deep learning frameworks to jointly model the forward and inverse ISP mappings. Building on this idea, subsequent methods such as CIE-XYZ-Net [3], CycleISP [73], InvISP [70], ParamISP [39], and model-based ISP approaches [17] further advanced the field through convolutional networks and differentiable, model-driven architectures trained on paired raw-sRGB data.

Despite their differences, existing data-driven inverse ISP approaches largely assume a one-to-one relationship between rendered images and their underlying ISPs. This assumption is reasonable when training data is captured or synthesized under controlled pipelines but breaks down for diffusion-generated imagery, where rendering parameters are unknown and highly variable. Consequently, applying conventional inverse-ISP strategies to generative outputs cannot reliably recover a consistent scene-referred representation.

Diffusion Models for Low-Level Vision Tasks. Denoising diffusion models [29,57,63] are powerful generative models that synthesize images by reversing a gradual noising process. Their state-of-the-art performance has been demonstrated across diverse low-level vision tasks [40,59], including super-resolution [61], image restoration [37, 68], and low-light enhancement [34, 69]. Most methods approach these tasks as conditional generation problems, guiding the diffusion process with auxiliary inputs like degraded images or semantic maps.

Recently, diffusion-based approaches have been explored for ISP-related tasks. Some works perform inverse-ISP by reconstructing raw from sRGB inputs [55], while others learn forward ISP mappings (raw-to-sRGB) [15,56]. These methods, however, typically rely on synthetic training pairs generated via fixed one-to-one ISP logic (e.g., using RawPy). More recently, Yuan et al. [71] proposed controlling camera parameters (e.g., white balance, exposure) by guiding the diffusion

process to maintain scene consistency. While effective for guided synthesis, this approach embeds parameter adjustment within the iterative generation, potentially requiring a new generative pass for each modification. For HDR reconstruction, diffusion-based works [8,67] focus on fusing multiple exposures, generally outputting tone-mapped sRGB or display-oriented HDR formats.

Consequently, these approaches do not address the challenge of reconstructing camera-centric, physically linear representations (e.g., CIE XYZ or raw) from in-the-wild sRGB inputs. Furthermore, they do not explore text-driven synthesis of raw data.

Such an approach that generates raw data would decouple the initial generation from subsequent adjustments, thereby enabling flexible and efficient postprocessing in standard dedicated software.

To handle unknown ISP pipelines, we adopt a many-to-one training paradigm that anchors diverse photo-finished observations to a shared linear target, enabling physically consistent raw generation at scale. RawGen is the first framework capable of synthesizing physically meaningful raw images directly from open-world text prompts without assuming a fixed imaging pipeline.

### 3 Method

RawGen synthesizes physically meaningful linear data by repurposing pretrained sRGB diffusion models. As discussed in Sec. 2, the same raw capture can yield substantially different sRGB renderings under different ISP and photo-finishing settings, and large-scale diffusion training corpora entangle these unknown, heterogeneous effects. Rather than assuming a fixed inverse mapping as in conventional inverse-ISP methods, RawGen learns to suppress this uncontrolled variability and recover a canonical, scene-referred linear representation.

We adopt CIE XYZ as the canonical space. It is linear and device-independent, and it relates to camera-specific raw spaces through standard color transforms, enabling camera-agnostic synthesis without retraining.

As shown in Fig. 2, our method builds on a pretrained rectified-flow DiT [47] with native image conditioning capability. We first describe the many-to-one training data construction (Sec. 3.1) that underlies our training stages, then present the two fine-tuning stages: denoiser fine-tuning (Sec. 3.2) and decoder fine-tuning (Sec. 3.3), corresponding to panels (A) and (B) of Fig. 2, and finally detail inference for image-to-raw and text-to-raw generation (Sec. 3.4), corresponding to Fig. 2(C).

#### 3.1 Many-to-One Training Data Construction

To invert diverse, unknown photo-finishing effects into a single linear target, we first construct paired data that links multiple sRGB renditions of the same scene to one common anchor. Starting from a raw image Iraw, we derive a linear CIE XYZ image IXYZ by applying per-scene white balancing and a camerato-XYZ color conversion matrix, yielding an illumination-neutral, scene-referred

[Figure 14]

[Figure 15]

[Figure 16]

Frozen weights Trainable weights ∨ OR operation C Concatenation + Addition - Subtraction

Non-trainable operations

…

(A) Denoiser Fine-tune (B) Decoder Fine-tune

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

EVAE

ISP

EVAE

DiT

sRGB latent

𝑣

C

Raw

sRGB

XYZ

…

XYZ latent

+

Params: { Color matrices, Illuminant color }

Reconstruction loss

v-prediction loss

[Figure 30]

[Figure 31]

[Figure 32]

Params: { Color matrices, Illuminant color }

[Figure 33]

[Figure 34]

[Figure 35]

XYZ latent

[Figure 36]

To CIE XYZ

| | |
|---|---|
| | |

noise

[Figure 37]

DVAE

[Figure 38]

- 𝑣

XYZ

Training camera settings

Pred XYZ

(C) Inference Phase

[Figure 39]

|Text prompt: Close-up of a bicycle wheel with roses between the spokes.| |
|---|---|
| | |

Params: { Color matrices, Illuminant colors, <noise profile>

Text-to-latent

[Figure 40]

I2R T2R

Target camera settings

T2R

|[Figure 41]<br><br>[Figure 42]|
|---|

[Figure 43]

sRGB latent

... }

∨

|[Figure 44]<br><br>sRGB| |
|---|---|
| | |

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

EVAE C

[Figure 49]

[Figure 50]

DiT DVAE

noise To camera raw

Raw

XYZ

I2R

- Fig. 2: Overview of the RawGen framework. During training, raw images are converted to CIE XYZ and sRGB representations to (A) fine-tune the DiT to denoise CIE XYZ latents conditioned on an sRGB image, and (B) fine-tune the VAE decoder to reconstruct CIE XYZ images. During inference (C), either an sRGB image (image-to-raw, I2R) or a text prompt (text-to-raw, T2R) is used to condition the DiT to generate a CIE XYZ latent, which is decoded to obtain a CIE XYZ image and subsequently mapped to the target camera’s raw space using its calibration metadata, which can be easily acquired from a single DNG file of target camera.

representation that precedes any photo-finishing. We treat IXYZ as the anchor, shared across all sRGB variants of the same scene.

Using this anchor, we generate N sRGB variants {IsRGB(n) }Nn=1 by sampling diverse photo-finishing parameter vectors P(n), which control global color tone, contrast, and tone mapping. Concretely, we randomize three ISP parameter groups: (i) white balance, by perturbing red and blue gains around the DNG AsShotNeutral initialization; (ii) tone mapping, by varying the per-channel tone-curve shape; and (iii) contrast, via a global gain about a mid-gray pivot in sRGB. Together, these perturbations yield diverse photo-finishing styles while preserving the same scene content. By pairing multiple sRGB variants with a single XYZ anchor, this strategy approximates the heterogeneous ISP behavior found in real-world sRGB images, encouraging invariance to photo-finishing while preserving scene content. Each image is encoded by the frozen VAE encoder EVAE:

zsRGB(n) = EVAE IsRGB(n) , zXYZ = EVAE(IXYZ). (1)

The resulting latent pairs {(zsRGB(n) , zXYZ)}Nn=1 are cached for both training stages. More details of the ISP pipeline and photo-finishing parameter sampling are pro-

vided in the supplementary material.

#### 3.2 Denoiser Fine-Tuning

The first training stage, shown in Fig. 2(A), fine-tunes the pretrained DiT to map sRGB-conditioned inputs to the linear XYZ anchor latent.

Objective. Given the anchor latent zXYZ from Eq. (1), we randomly sample one sRGB variant latent zsRGB(n) under the many-to-one setting. We corrupt the anchor latent by interpolating with Gaussian noise ϵ ∼ N(0,I):

zt = (1 − t)zXYZ + tϵ, t ∈ [0,1], (2)

and tune the DiT parameterized by θ to predict the rectified-flow velocity target (also referred to as a v-prediction target) vgt = ϵ − zXYZ:

2 2

Ldenoise = En, t, ϵ vgt − vθ zt, t; zsRGB(n)

. (3)

By tuning with randomly sampled variants, the denoiser learns to suppress ISPinduced variability in the sRGB condition and consistently recover the shared latent representation of the linear anchor XYZ image.

Conditioning. We leverage the image conditioning mechanism of the pretrained DiT. The sRGB context latent zsRGB(n) and the noisy target latent zt are patchified into token sequences and concatenated along the sequence dimension, and the combined sequence is processed jointly through the DiT transformer blocks. Only the target portion of the output contributes to the loss. We fine-tune the model using LoRA adapters [31] on attention projection layers, while keeping the pretrained backbone weights frozen.

#### 3.3 Decoder Fine-Tuning

The second training stage, shown in Fig. 2(B), fine-tunes the VAE decoder to reconstruct linear XYZ images from XYZ-domain latents. Since the pretrained decoder is optimized for sRGB, directly decoding XYZ latents may yield degraded outputs. We fine-tune DVAE using the ground-truth anchor latent zXYZ from Eq. (1), minimizing an ℓ1 loss:

IXYZ = DVAE(zXYZ), Lrecon = IXYZ − IXYZ

. (4)

1

This retargets the VAE decoder from sRGB to linear XYZ while preserving the spatial representations learned during pretraining.

#### 3.4 Inference: Image-to-Raw and Text-to-Raw

As illustrated in Fig. 2(C), RawGen uses a unified inference pipeline that first synthesizes a scene-referred CIE XYZ image and then renders it into a target camera’s linear raw space. The only difference between image-to-raw (I2R) and text-to-raw (T2R) lies in how the sRGB conditioning latent is obtained

within the same pretrained backbone that shares the core DiT and the VAE encoder/decoder. Specifically, I2R encodes an input sRGB image using the frozen VAE encoder, whereas T2R follows the conventional diffusion model’s standard text-conditioned generation route and directly takes the intermediate latent produced from the text prompt prior to VAE decoding (i.e., the latent that would otherwise be decoded to an sRGB image). In both cases, this sRGB latent conditions the same DiT to generate an XYZ latent, which is decoded with our fine-tuned VAE decoder and finally mapped deterministically to the target camera’s raw space using its calibration metadata under a chosen illuminant. In our implementation, we instantiate the backbone with FLUX.1 and reuse its native text-to-latent pathway for T2R conditioning.

Image-to-Raw (I2R). Given an sRGB image IsRGB, we obtain the conditioning latent zsRGB = EVAE(IsRGB). We initialize the target tokens with noise and solve the reverse ODE conditioned on zsRGB:

- 0
- 1

vθ(zt, t; zsRGB)dt, (5)

zˆXYZ = z1 +

approximated using Euler steps. We retain only the target tokens, reshape them into the spatial latent zˆXYZ, and decode IXYZ = DVAE(ˆzXYZ). The decoded XYZ image is then converted to the target camera raw space using the mapping described below.

Text-to-Raw (T2R). Given a text prompt, we first obtain an sRGB conditioning latent zsRGB using the pretrained text-to-latent pathway of the base model. We then run the same conditional generation and decoding steps as in I2R to produce zˆXYZ and IXYZ, and apply the same XYZ-to-camera mapping to obtain camera-specific linear raw outputs.

CIE XYZ to Camera Raw. To convert IXYZ into a target camera’s linear rawRGB space, we apply calibration transforms derived from the target camera’s DNG metadata together with a chosen illuminant parameterized by correlated color temperature (CCT). In practice, we (i) interpolate camera calibration matrices based on the selected CCT, (ii) map IXYZ to a white-balanced camera RGB using the interpolated forward model, and (iii) apply an illuminant gain in camera RGB space to obtain the illuminated camera RGB, which we treat as the target camera’s linear raw-RGB representation. This mapping is deterministic and non-trainable, enabling re-rendering of the same scene-referred output across cameras without retraining. Full details of matrix interpolation and illuminant sampling are provided in the supplementary material.

Optional Noise and CFA. If required, we synthesize sensor noise using a heteroscedastic noise model to better match the noise characteristics of real raw camera images; we refer the reader to the supplementary material for details. While more advanced noise synthesis models (e.g., [1]) can produce more realistic noise characteristics, detailed noise modeling is beyond the scope of this paper. Finally, if required by the downstream task, the resulting linear RGB raw image can be re-mosaiced according to a specified color filter array (CFA) pattern to produce a single-channel mosaiced raw image.

### 4 Experiments

#### 4.1 Training Details and Experiment Overview

Training Details. We train RawGen using the MIT-Adobe FiveK [12] and RAISE [20] datasets. Both datasets provide large-scale raw DNG images covering diverse scenes and subjects, including urban and natural environments, as well as animals and human portraits. To obtain CIE XYZ anchor targets, we parse the AsShotNeutral and ForwardMatrix tags from each DNG file. We then apply the software ISP [62] with randomized photo-finishing parameters to render diverse sRGB variants from the same anchor (Sec. 3.1). For the DiT backbone, we adopt FLUX.1-Kontext [7], which natively supports image-context conditioning. We fine-tune the DiT model via LoRA with rank r=64 and scaling α=64.

Experiment Overview. As described in Sec. 3, once trained, RawGen synthesizes scene-referred linear representations under two conditioning modes: (i) an input sRGB image (I2R) and (ii) a text prompt (T2R). We evaluate RawGen from three perspectives. First, we assess its many-to-one reconstruction capability by recovering a canonical linear XYZ representation from sRGB inputs with diverse and unknown photo-finishing (Sec. 4.2). Second, we evaluate devicespecific raw synthesis by mapping canonical outputs to target camera domains and examining their distribution alignment with real raw data (Sec. 4.3). Third, we study practical applications enabled by RawGen, including scalable T2Rbased data generation for downstream low-level vision tasks and raw-domain image editing. (Sec. 4.4). Together, these experiments demonstrate that RawGen enables physically grounded linear synthesis while supporting camera-aware and scalable raw generation.

#### 4.2 Evaluation of Many-to-One sRGB-to-XYZ Invertability

RawGen projects sRGB inputs containing unknown and diverse photo-finishing effects into a shared canonical scene-referred XYZ domain, suppressing photofinishing variability while preserving scene content. To evaluate this many-to-one inverse-ISP capability, we conduct two complementary analyses using the Imageto-Raw (I2R) inference pathway (Fig. 2C): (1) expert-retouched real variations and (2) text-guided synthetic variations. We compare RawGen against CIE XYZ Net [3], InvISP [70], and Raw-Diffusion [55]. Since these baselines typically assume a one-to-one correspondence between an sRGB input and its reconstruction target (e.g., raw or XYZ), we train all methods to invert the ISP induced by the default configuration of our software ISP, using the corresponding (default sRGB, anchor XYZ) pair as supervision.

Expert-Retouched Real Variations. For this experiment, we use the MITAdobe FiveK dataset [12], where each scene is edited by five experts (A–E) with distinct aesthetic preferences. For each scene, the five expert-retouched sRGB images are fed into RawGen, and reconstruction accuracy is measured against the reference scene-referred XYZ target using PSNR and SSIM. All evaluations

###### Table 1: Quantitative comparison of sRGB-to-XYZ reconstruction across five photofinishing styles on the MIT-Adobe FiveK dataset [12]. We report results for ablated variant of RawGen. Best results are highlighted in yellow .

Expert A Expert B Expert C Expert D Expert E Method

PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM PSNR SSIM CIE XYZ Net [3] 19.60 0.7861 21.04 0.8431 19.49 0.7795 19.44 0.8058 18.64 0.7918 InvISP [70] 16.04 0.6854 16.04 0.6854 14.92 0.6323 14.24 0.6802 13.30 0.6473 Raw-Diffusion [55] 19.30 0.7832 20.66 0.8397 18.79 0.7705 18.69 0.7966 17.49 0.7751 RawGenone-to-one 19.57 0.7782 21.21 0.8349 19.48 0.7741 18.74 0.7929 18.07 0.7839 RawGen (Ours) 23.20 0.8432 24.35 0.8581 23.37 0.8387 23.51 0.8531 23.89 0.8500

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Expert A

Expert B

Expert C

Expert D

Expert E

sRGBimageRaw-Diffusion

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

PSNR: 13.53

PSNR: 21.67

PSNR: 12.09

PSNR: 16.59

- PSNR: 12.62

[Figure 61]

- PSNR: 13.63

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

PSNR: 12.54

PSNR: 15.56

PSNR: 12.56

PSNR: 18.11

CIEXYZNet

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

PSNR: 37.12

PSNR: 34.29

PSNR: 36.53

PSNR: 30.76

PSNR: 37.21

RawGen(ours)

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

Groundtruth

- Fig. 3: CIE XYZ reconstruction results. Shown is an sRGB input image rendered with different rendering preferences (Expert A–E) and the corresponding CIE XYZ reconstructions produced by Raw-Diffusion [55], CIE XYZ Net [3], and our RawGen. The last row shows the ground-truth CIE XYZ image.

are conducted on test splits unseen during training for both RawGen and the compared methods. As a baseline, we evaluate RawGenone-to-one, which uses the same architecture as RawGen but is trained with strict one-to-one supervision on (default sRGB, anchor XYZ) pairs rather than our many-to-one training scheme.

- Table 2: Latent space compactness of sRGB-to-XYZ conversion methods. 100 color-graded sRGB variants per prompt are converted to XYZ and encoded into the VAE latent space; lower mean L2 distance to the centroid indicates more compact clusters.

Mean distance to centroid ↓ PCA t-SNE UMAP

Method

sRGB 286.8 27.65 2.099 InvISP 265.4 24.33 1.913 XYZNet 256.1 25.52 2.014 RAW-Diffusion 318.2 19.58 2.557 RawGen 160.0 10.69 1.067

[Figure 96]

Fig. 4: t-SNE visualization of VAE latent spaces for each sRGB-to-XYZ conversion method. Dashed contours indicate the estimated density region per method. RawGen produces the most compact cluster.

As shown in Tab. 1, RawGen consistently generalizes to unseen photo-finishing

styles and more accurately recovers scene-referred linear XYZ representations than prior methods. The performance drop of RawGenone-to-one further highlights the importance of domain-level many-to-one supervision. Qualitative reconstructions and error maps in Fig. 3 further corroborate RawGen’s robustness under diverse and previously unseen edits.

Text-Augmented Synthetic Variations. To further examine many-to-one inverse-ISP capability, we construct a larger and more diverse set of sRGB variants using text-guided generation. Unlike the real-variant experiment above, which relies on five expert retouchings per scene, this setting synthesizes controlled color variations for a broader evaluation of linear invertibility.

We use the pre-trained FLUX.1-Kontext [7] in a two-stage process. First, an anchor sRGB image is generated from a prompt (e.g., “capture narrow alley after rainfall”). The model is then conditioned on this anchor together with 100 color-grading prompts (e.g., “warm cast:”, “cool cast:”, “teal-and-orange

grade:”), producing 100 variants. Leveraging the image-conditioning feature of FLUX.1-Kontext, scene structure and texture are preserved, while the text prompts induce diverse global color shifts. These variants are converted to XYZ by each evaluated method, with RawGen operating via the Image-to-Raw inference pathway (Fig. 2C).

To assess many-to-one consistency, the reconstructed outputs are encoded into the VAE latent space. The latent embeddings are projected using PCA [46], t-SNE [41], and UMAP [42]. PCA captures dominant linear variance, while tSNE and UMAP preserve local neighborhood structure under nonlinear embeddings. Within each projected space, we compute the mean L2 distance to the centroid across variants. Lower distances indicate stronger suppression of photo-finishing variability and greater convergence toward a canonical representation. Tab. 2 reports the quantitative results, showing that RawGen produces more compact latent clusters than competing methods under PCA, t-SNE, and UMAP projections, whereas other approaches yield more dispersed embeddings,

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

PSNR: 32.36

Image-to-Raw

Text-to-Raw

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

PSNR: 33.23

(A) sRGB image (B) GT raw image (C) Reconstructed raw (D) Example of generated raw images from image caption

- Fig. 5: Raw reconstruction and generation results. Shown are the input sRGB images

- (A) and corresponding ground-truth raw images (B). Our reconstructed raw results are shown in (C). Additional generated raw images of the same semantic scene are shown in (D), where InstructBLIP-2 [18] is used to generate text descriptions from image (A).

[Figure 109]

(A) Generated XYZ image

Text prompt: Volcanic night landscape

[Figure 110]

|[Figure 111]|
|---|

(B) Mapped to raw

| |
|---|

[Figure 112]

|[Figure 113]|
|---|

(D) Modular ISP

| |
|---|

[Figure 114]

[Figure 115]

(C) PyNet

| |
|---|

|[Figure 116]|
|---|

Fig. 6: Rendering results of pre-trained neural ISPs on real data from the Samsung S24 main camera (S24 dataset [6]). (A) CIE XYZ image generated by our RawGen.

- (B) Image mapped to the S24 main camera raw space with synthetic noise. (C) Result of PyNet [33]. (D) Result of Modular Neural ISP [5].

indicating weaker suppression of photo-finishing variability. The t-SNE visualization in Fig. 4 further confirms that RawGen forms a tight and well-separated cluster with reduced overlap. This result aligns with the many-to-one objective, which encourages diverse sRGB variants of the same scene to converge toward a canonical XYZ representation and promotes invariance to photo-finishing effects.

#### 4.3 Device-Specific Raw Synthesis

RawGen produces canonical linear XYZ representations independent of any specific camera. When a target device is specified, these outputs are mapped to the corresponding raw domain using device-specific color correction matrices (e.g., ForwardMatrix) and metadata, with heteroscedastic Gaussian noise injected to approximate sensor measurements. This decoupled design enables cameraagnostic linear generation and device-specific raw synthesis without retraining. Figure 5 presents examples, including reconstruction from sRGB inputs with unknown edits (A–C) and multiple raw samples generated from text prompts via InstructBLIP-2 [18] (D), supporting scalable raw-domain augmentation.

We further assess distribution alignment between RawGen-synthesized and real-world raw images. Specifically, we evaluate neural ISP models trained on

real raw data from the Samsung Galaxy S24 main camera (S24 dataset [6]) on RawGen-generated raw images mapped to the same device domain, without retraining. As shown in Figure 6, both ISP models produce plausible sRGB renderings. Notably, among the two evaluated ISPs, Modular Neural ISP includes an explicit denoising module and effectively suppresses the injected synthetic noise, indicating strong distribution alignment with real device-specific raw statistics.

#### 4.4 Application of RawGen

Scalable Raw Synthesis for Downstream Tasks. Various low-level vision studies that rely on raw images have faced difficulties in data acquisition due to device-specific domain shifts, and collecting raw data at scale has required substantial human labor and extensive capture time. RawGen’s Text-to-Raw (T2R) synthesis flow enables scalable generation of diverse device-specific raw images without physical capture, making large-scale augmentation practical for low-level vision tasks. Having established the realism of these synthesized raw samples (Sec. 4.3), we evaluate whether they can effectively support downstream tasks that require camera-specific raw data.

We evaluate impact of the scalability on three tasks: illuminant estimation, neural ISP learning, and raw-domain denoising, all requiring device-specific raw data. For controlled comparison, we follow the same datasets, training procedures, and evaluation protocols as Graphics2RAW [62]. For each task, we synthesize 3K samples. These synthetic samples are used only for training, and performance is measured on the corresponding real-world test splits without modification, isolating the effect of the synthetic data source while enabling direct comparison with prior methods.

The scalability advantage is particularly salient for illuminant estimation, which requires training data for nine cameras. In the real NUS-8 dataset, each camera provides only 217 images on average, and prior graphics-based raw synthesis [62] is further constrained by limited assets, yielding 125 synthetic raw images for training. In contrast, RawGen can generate diverse, camera-specific raw images directly from text prompts at scale, substantially expanding training coverage without additional capture effort or scene asset design. Detailed experimental settings are provided in the supplementary material.

Tab. 3 shows the results. RawGen-based T2R improves performance across all three tasks compared to prior synthetic raw generation approaches. These results indicate that scalable and physically grounded raw synthesis can alleviate data acquisition bottlenecks in device-specific low-level vision tasks.

Generative Image Editing. Image editing is more reliable in a scene-referred linear space, where radiometric linearity and a wider dynamic range are preserved, enabling physically grounded operations such as white balance and exposure scaling [3]. Conventional inverse-ISP methods [3,9,55,70] enable lineardomain editing but are often tuned to specific devices and fixed imaging assumptions, which can lead to failures when inverting heterogeneous sRGB inputs outside the training distribution (Sec. 4.2) and degrade editing quality.

- Table 3: Quantitative comparison of downstream tasks: illuminant estimation (reported as angular error statistics), neural ISP learning, and denoising at two ISO levels. Each row corresponds to a model trained using data generated by the method listed. Models trained on real data are shown in the bottom row as a reference. The best results of models trained on non-real raw data are highlighted in yellow .

Denoising Illuminant estimation Neural ISP

Method ISO 1600 ISO 3200

Mean Median Worst 25% PSNR SSIM ∆E PSNR SSIM PSNR SSIM EnlightenGAN [35] 7.01 6.82 11.07 35.58 0.965 3.137 48.82 0.991 47.25 0.988 UPI [11] 6.26 5.89 10.33 36.43 0.966 2.907 49.05 0.990 47.51 0.988 Graphics2RAW [62] 4.21 3.38 8.57 38.10 0.974 2.301 49.37 0.991 48.16 0.989 RawGen 3.14 2.11 7.37 38.42 0.970 2.183 50.63 0.994 48.57 0.992 Real 3.02 2.17 6.77 38.32 0.974 2.133 49.80 0.993 48.25 0.990

Camera-controllable generation frameworks such as Generative Photography [71] support parameter-driven edits (e.g., white balance, shutter speed, and f -number) while preserving scene context, but they operate entirely in the sRGB domain without access to an underlying raw image. Consequently, applying a new parameter setting typically requires rerunning an iterative diffusion process, rather than performing ISP operations on linear sensor data. Moreover, their controllability is limited to the camera parameters explicitly learned by the model, and supporting additional types of edits or new control dimensions generally requires retraining or additional fine-tuning.

In contrast, RawGen decouples content synthesis from rendering by producing a scene-referred linear image from either an input image or a text prompt, and then enabling edits via standard software-ISP operations. After generating the linear representation once, a wide range of edits, including white balance, exposure compensation, and tone mapping, can be applied directly in the linear domain using an arbitrary software ISP pipeline without re-invoking the generative model. This supports efficient exploration of diverse editing operations beyond a fixed set of learned camera parameters.

### 5 Conclusion and Discussion

We present RawGen, a diffusion-based framework for camera-agnostic raw generation that bridges display-referred generative models and scene-referred linear representations. Using a many-to-one objective, RawGen suppresses photofinishing effects and recovers a canonical XYZ representation consistent across diverse sRGB variants. We demonstrated many-to-one linear XYZ reconstruction and device-specific raw synthesis. Moreover, scalable text-to-raw data synthesis enables RawGen to improve downstream low-level vision tasks.

Limitations. While RawGen generates canonical scene-referred XYZ representations, accurate device-specific raw synthesis also depends on factors beyond color mapping, such as noise characteristics and PSFs. Future work may explore conditioning RawGen on device-specific priors to incorporate more physically grounded sensor modeling, including spatially varying noise, lens shading, and optical blur, further improving device fidelity and physical consistency.

### Acknowledgments

We thank Ran Zhang for assistance with rendering the generated raw images using neural ISP models.

## Supplementary Material

In the main paper, we present RawGen, a method for generating realistic raw images conditioned on text or sRGB inputs. In the supplementary material, we provide additional methodological details and extended experimental results.

### A Photo-Finishing Simulation Details

ISP Pipeline. Starting from a DNG raw file, we employ a physically grounded ISP simulator based on the software ISP [62]. The pipeline processes raw sensor data through the ordered stages: raw → normalize → lens_shading_correction

→ white_balance → demosaic → xyz → srgb → gamma. We use edge-aware (EA) demosaicing, adopt a D50 white point for XYZ-to-sRGB conversion, and apply the camera’s embedded forward color matrix (the ForwardMatrix tag in the DNG) for color transformation.

XYZ Anchor. The XYZ anchor IXYZ is obtained by running the ISP pipeline with the DNG’s embedded white-balance coefficients and terminating at the xyz stage, prior to sRGB rendering. No photo-finishing augmentation is applied; the anchor serves as a fixed, illumination-neutral, scene-referred reference for all variants of the same scene.

sRGB Variant Generation. For each scene, the ISP pipeline is executed multiple times with independently sampled photo-finishing parameters to produce a

set of sRGB renditions {IsRGB(n) }. Since parameters are sampled independently per scene, the augmentation space is effectively continuous; the number of renditions

reflects computational budget rather than a fixed style count. Three parameter groups are randomized:

- 1. White Balance. Channel gains for red and blue are initialized from the DNG AsShotNeutral metadata and independently perturbed using multiplicative factors r ∼ U(0.7,1.3) and b ∼ U(0.7,1.3), whereas the green gain is held constant (g = 1.0).
- 2. Tone Mapping. The ISP output is decoded to linear RGB via the inverse sRGB electro-optical transfer function (OETF). A parametric tone-mapping operator is applied per channel:

T(Ei) =

(1 + β)Eiγ β + Eiγ

, (6)

where Ei ∈ [0,1] denotes the linear per-channel intensity. Parameters are sampled as β ∼ N(0.6, 0.12) (clipped to [0.1,2.0]) and γ ∼ N(0.9, 0.12) (clipped to [0.5,1.5]). The tone-mapped result is re-encoded to sRGB using the standard OETF.

- 3. Contrast. A contrast factor c ∼ U(0.7,1.3) is applied in the display-referred sRGB domain about a mid-gray pivot (0.5):

Iout = (Iin − 0.5)c + 0.5. (7)

Table 4: Photo-finishing parameter ranges used for sRGB variant generation.

|Parameter|Distribution<br><br>|Range|Stage|
|---|---|---|---|
|Red WB multiplier r<br><br>|Uniform<br><br>|[0.7, 1.3]|White balance|
|Blue WB multiplier b<br><br>|Uniform|[0.7, 1.3]<br><br>|White balance|
|Tone-map β<br><br>|Normal|µ=0.6, σ=0.1 (clip [0.1, 2.0])<br><br>|Tone mapping|
|Tone-map γ<br><br>|Normal|µ=0.9, σ=0.1 (clip [0.5, 1.5])<br><br>|Tone mapping|
|Contrast c|Uniform<br><br>|[0.7, 1.3]<br><br>|sRGB|

Image Resolution and Format. Each image is center-cropped to the largest square region and resized to 1024 × 1024 using Lanczos interpolation. XYZ anchors are stored as 16-bit PNG to preserve precision, while sRGB variants are stored as 8-bit PNG.

Parameter Ranges. Tab. 4 summarizes all stochastic parameters.

### B XYZ-to-Camera Raw Mapping

To render a decoded CIE XYZ image into a target camera’s linear raw-RGB space, we use calibration metadata from a representative DNG of the target camera. Specifically, we leverage the DNG-provided matrix pairs under two reference illuminants and interpolate them according to the chosen correlated color temperature (CCT). This enables camera-specific rendering while keeping the XYZ generation stage camera-agnostic.

CCT-Based Calibration Matrix Interpolation. Let C1,C2 denote the ColorMatrix under two calibration illuminants, and F1,F2 denote the corresponding ForwardMatrix extracted from the DNG file. Given a target CCT T, we compute an interpolation weight using the reciprocal-temperature rule:

g = clip

and form interpolated matrices

1/T − 1/T1 1/T2 − 1/T1

, 0, 1 , (8)

C(T) = g C1 + (1 − g)C2, F(T) = g F1 + (1 − g)F2. (9)

XYZ to Camera Raw-RGB. Given a decoded linear XYZ image IXYZ, we first map it to a white-balanced camera RGB by inverting the forward model:

IWB = clip IXYZ · F(T)−T, 0, 1 , (10)

where the matrix is applied per pixel. We then convert an illuminant in XYZ, ℓXYZ, into camera RGB via the color matrix,

ℓRGB = C(T)ℓXYZ, (11)

normalize it by the green channel to obtain a relative gain vector, and apply it to obtain an illuminated camera RGB, which we treat as the target camera’s linear raw-RGB output:

Iraw = clip(IWB ⊙ ℓRGB, 0, 1). (12)

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

- (A) Generated XYZ image (B) Generated raw image with varying CCT values

(C) Generated raw image (CCT = 7000 K) with varying noise scales

CCT = 7000 K CCT = 5500 K CCT = 4500 K CCT = 3500 K

s = 0.3 s = 0.7 s = 1.0 s = 1.5 s = 2.0

Text prompt: Snowy hills at dusk

Fig. 7: Variations of raw images generated from the CIE XYZ image shown in (A).

- (B) shows raw images mapped to the Samsung Galaxy S24 main camera raw space under different correlated color temperatures (CCTs). (C) shows raw images mapped with a fixed CCT of 7000 K and different noise strength factor values s. All images are gamma-corrected for visualization throughout this supplementary material.

Unless otherwise noted, we export Iraw as a normalized 16-bit image for dataset generation.

Heteroscedastic Noise Simulation. We optionally simulate target sensor noise by applying a heteroscedastic noise model to the resulting raw image Iraw. For each color channel, the noise variance is modeled as a linear function of the signal intensity, scaled by a global noise strength factor s:

σc2(Iraw) = s αc Iraw(c) + βc , (13)

where αc and βc represent the signal-dependent (shot) and signal-independent (read) noise components for channel c, respectively, and s controls the overall noise magnitude. The parameters αc and βc are obtained from camera noise calibration [25] or derived from the camera’s noise profile [2].

Zero-mean Gaussian noise is then sampled according to this variance and added to the input image, producing the noisy output:

Iraw′ = Iraw + N 0, σ2(Iraw) . (14)

Fig. 7 shows an example of mapping a generated CIE XYZ image to the raw space of the Samsung Galaxy S24 main camera under different CCTs and noise strength factor values.

- C Downstream Task Setup Details

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

|[Figure 127]|
|---|

|[Figure 128]|
|---|

|[Figure 129]|
|---|

|[Figure 130]|
|---|

|[Figure 131]|
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

##### Illumination Estimation. We follow Graphics2RAW and evaluate illuminant estimation on the nine-camera NUS dataset [16]. We generate 3K synthetic raw

images from text prompts to increase scene diversity. The network architecture [26] and optimization settings are kept identical to the reference protocol, and performance is measured using angular error. This controlled setup isolates the impact of RawGen’s scalable generative data on learning color constancy.

Neural ISP. To assess RawGen for neural ISP training, we use the nighttime dataset [51] and testing configuration introduced in Graphics2RAW. 3K synthetic raw images generated with RawGen are paired with rendered sRGB targets and used for supervision. The ISP architecture [58], loss functions, and evaluation metrics are kept fixed, and testing is performed on real raw inputs. This setup evaluates whether RawGen provides effective supervision for real-world ISP rendering.

Image Denoising. We further evaluate RawGen on raw-domain denoising using the nighttime dataset [51] and protocol introduced in Graphics2RAW. For training, we synthesize 3K clean linear images from text prompts and generate noisy counterparts with the same heteroscedastic Gaussian noise model as the baseline. The denoiser architecture [72] and training configuration remain unchanged, and testing is conducted exclusively on real noisy captures at ISOs 1600 and 3200. This controlled setup isolates the benefit of RawGen for improving noise-robust learning through scalable training data.

### D Additional Results

CIE XYZ Reconstruction. In Sec. 4.2 of the main paper, we report quantitative results on CIE XYZ reconstruction under unseen photo-finishing styles using the MIT-Adobe FiveK dataset [12], where each scene is edited by five experts (A–E) with distinct aesthetic preferences. Here, we provide additional qualitative comparisons in Figs. 8 to 12.

sRGB-to-Raw Synthesis. In Sec. 4.3 of the main paper, we compared RawGen-

synthesized raw images with ground-truth raw data and presented additional raw samples for the same semantic scene generated from image descriptions extracted from the input sRGB images using InstructBLIP-2 [18]. Additional results are shown in Fig. 13. We further present Text-to-Raw (T2R)-based raw generation results, where the CIE XYZ images generated by RawGen are mapped to arbitrary target camera raw domains (Figs. 15 to 18).

Compatibility on Pre-Trained ISP. In Sec. 4.3 of the main paper, we evaluate the compatibility of RawGen-generated raw images with neural ISP models trained exclusively on real raw data from the Samsung Galaxy S24 camera (S24 dataset [6]).

Here, we provide additional compatibility results for Lite ISP [75], Invertible ISP [70], and Modular Neural ISP [5]. All ISP models are fed RawGengenerated raw images with synthesized sensor noise, and their rendered sRGB outputs are shown in Fig. 19. The visually plausible renderings—despite known generalization limitations when raw distributions deviate from the training cam-

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

Expert A

Expert B

Expert C

Expert D

Expert E

sRGBimageRaw-Diffusion

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

PSNR: 25.33

PSNR: 22.71

PSNR: 16.51

PSNR: 25.11

PSNR: 29.04

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

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

PSNR: 24.24

PSNR: 21.74

PSNR: 16.27

PSNR: 23.50

PSNR: 29.04

CIEXYZNet

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

PSNR: 33.12

PSNR: 31.73

PSNR: 31.97

PSNR: 29.70

PSNR: 29.04

RawGen(ours)

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

Groundtruth

- Fig. 8: Additional CIE XYZ reconstruction results. Shown is an sRGB input image rendered with different rendering preferences (Expert A–E) and the corresponding CIE XYZ reconstructions produced by Raw-Diffusion [55], CIE XYZ Net [3], and our RawGen. The last row shows the ground-truth CIE XYZ image.

##### era [4,48]—indicate close alignment between RawGen-generated raw images and real S24 training data.

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

Expert A

Expert B

Expert C

Expert D

Expert E

sRGBimageRaw-Diffusion

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

PSNR: 15.62

PSNR: 28.38

PSNR: 19.49

- PSNR: 19.35

[Figure 187]

- PSNR: 20.25

PSNR: 14.78

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

PSNR: 18.74

PSNR: 21.92

PSNR: 25.00

PSNR: 22.40

CIEXYZNet

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

PSNR: 26.25

PSNR: 35.22

PSNR: 31.05

PSNR: 28.44

PSNR: 33.71

RawGen(ours)

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

###### Groundtruth

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

Expert A

Expert B

Expert C

Expert D

Expert E

sRGBimageRaw-Diffusion

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

PSNR: 24.37

PSNR: 18.14

PSNR: 17.64

PSNR: 18.66

PSNR: 14.73

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

[Figure 244]

[Figure 245]

[Figure 246]

PSNR: 22.96

PSNR: 17.89

PSNR: 17.43

PSNR: 18.17

PSNR: 14.97

CIEXYZNet

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

PSNR: 28.79

PSNR: 29.75

PSNR: 28.44

PSNR: 26.40

PSNR: 29.05

RawGen(ours)

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

###### Groundtruth

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

Expert A

Expert B

Expert C

Expert D

Expert E

sRGBimageRaw-Diffusion

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

PSNR: 21.71

PSNR: 18.58

PSNR: 20.71

PSNR: 20.93

PSNR: 18.26

[Figure 277]

[Figure 278]

[Figure 279]

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

PSNR: 20.37

PSNR: 17.77

PSNR: 19.37

PSNR: 19.82

PSNR: 17.65

CIEXYZNet

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

PSNR: 30.15

PSNR: 28.13

PSNR: 28.29

PSNR: 27.74

PSNR: 29.20

RawGen(ours)

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

###### Groundtruth

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

Expert A

Expert B

Expert C

Expert D

Expert E

sRGBimageRaw-Diffusion

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

PSNR: 26.41

PSNR: 24.64

PSNR: 24.10

PSNR: 19.04

PSNR: 23.09

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

- PSNR: 26.81

RawGen(ours)

[Figure 337]

- PSNR: 27.75

PSNR: 23.24

PSNR: 22.33

PSNR: 18.76

PSNR: 22.04

CIEXYZNet

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

PSNR: 27.46

PSNR: 26.85

PSNR: 35.05

PSNR: 28.70

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

###### Groundtruth

- PSNR: 29.04

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

PSNR: 40.99

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

- PSNR: 33.10

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

- PSNR: 34.10

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

PSNR: 34.31

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

PSNR: 32.61

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

PSNR: 31.88

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

- PSNR: 30.42

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

(A) sRGB image (B) GT raw image (C) Reconstructed raw (D) Example of generated raw images from image caption

###### Fig. 13: Additional raw image reconstruction and generation results. Shown are the input sRGB images (A) and corresponding ground-truth raw images (B). Our reconstructed raw results are shown in (C). Additional generated raw images of the same semantic scene are shown in (D), where InstructBLIP-2 [18] is used to generate text descriptions from image (A).

[Figure 405]

[Figure 406]

[Figure 407]

Target camera: Canon EOS 600D Samsung NX2000 Olympus PEN E-PL6

Text prompt: A close-up image of a smiling baby astronaut on the moon.

[Figure 408]

[Figure 409]

[Figure 410]

Target camera: Nikon D5200 Sony Alpha 57 Fujifilm X-M1

Text prompt: A portrait of a young woman in a room.

[Figure 411]

[Figure 412]

[Figure 413]

Target camera: Nikon D40 Canon EOS-1Ds Mark III Panasonic Lumix DMC-GX1

Text prompt: A photo of a woman typing on her laptop in a café.

###### Fig. 14: Additional results of our RawGen, generated from the text prompts shown at the bottom of each image. For each prompt, we show generated raw images for three target cameras.

[Figure 414]

[Figure 415]

[Figure 416]

Target camera: Canon 1Ds MkIII

[Figure 417]

[Figure 418]

[Figure 419]

- Fig. 15: Examples of generated raw images for the Canon 1Ds MkIII DSLR camera. The images were generated using the following text prompts: 1) Cracked ceramic mug with visible hairline fractures, natural lighting, 2) convenience store at 2am, viewed through a glass facade, 3) commuters stepping onto an escalator at the exact moment of boarding, 4) mountain valley just before sunrise with rising ground fog, 5) peeling paint on a weathered metal door, and 6) city sidewalk just as a shallow puddle finishes forming.

[Figure 420]

[Figure 421]

[Figure 422]

- Target camera: Olympus EPL6
- Fig. 16: Examples of generated raw images for the Olympus EPL6 DSLR camera. The images were generated using the following text prompts: 1) ocean waves striking a seawall at the exact moment of impact, 2) shadow band phenomenon in the final seconds before totality, 3) wind-driven sand crossing a desert road at sunset, 4) coastal fishing village harbor at dawn before activity begins, 5) suburban parking lot after freezing rain, and 6) pedestrian mid-step with both feet briefly off the ground.

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

- Target camera: Nikon D5200
- Fig. 17: Examples of generated raw images for the Nikon D5200 DSLR camera. The images were generated using the following text prompts: 1) Close-up of human skin with visible pores and fine hairs, 2) narrow alley after rainfall, 3) roadside gas station in midday heat shimmer, 4) tree pollen drifting in golden backlight, 5) coastal cliff facing the ocean in strong onshore winds, and 6) subway platform during non-peak hours.

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

- Target camera: Samsung NX2000
- Fig. 18: Examples of generated raw images for the Samsung NX2000 DSLR camera. The images were generated using the following text prompts: 1) Thin ice forming across a pond at the transition point, 2) person adjusting glasses at the precise midpoint of the motion, 3) visible breath at the exact moment of exhale in cold air, 4) lightning illuminating low cloud cover without direct bolt visibility, 5) coastal fog rolling over a highway overpass, and

[Figure 435]

[Figure 436]

[Figure 437]

- 6) newspaper print under fluorescent lighting with slight glare.

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

Text prompt: A man at a crosswalk

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

Text prompt: Leather wallet and keys

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

Text prompt: Open book in a library

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

Text prompt: Street market after rain

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

Text prompt: A pool at closing time

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

Text prompt: Local restaurant at night

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

Text prompt: River bend at dusk

###### (A) Generated XYZ image

(B) Mapped to raw (C) Lite ISP (D) Invertible ISP (E) Modular ISP

Fig. 19: Rendering results of pre-trained neural ISPs on real data from the Samsung S24 main camera (S24 dataset [6]). (A) CIE XYZ images generated by our RawGen.

###### (B) Images mapped to the S24 main camera raw space with synthetic noise. (C) Result of Lite ISP [75]. (D) Result of Invertible ISP [70]. (E) Result of Modular Neural ISP [5].

### References

- 1. Abdelhamed, A., Brubaker, M.A., Brown, M.S.: Noise flow: Noise modeling with conditional normalizing flows. In: ICCV (2019) 8
- 2. Abdelhamed, A., Lin, S., Brown, M.S.: A high-quality denoising dataset for smartphone cameras. In: CVPR (2018) 18
- 3. Afifi, M., Abdelhamed, A., Abuolaim, A., Punnappurath, A., Brown, M.S.: CIE XYZ Net: Unprocessing images for low-level computer vision tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence 44(9), 4688–4700 (2021) 2, 4, 9, 10, 13, 20, 21, 22, 23, 24
- 4. Afifi, M., Abuolaim, A.: Semi-supervised raw-to-raw mapping. In: BMVC (2021) 20
- 5. Afifi, M., Wang, Z., Zhang, R., Brown, M.S.: Modular neural image signal processing. arXiv preprint arXiv:2512.08564 (2025) 12, 19, 31
- 6. Afifi, M., Zhao, L., Punnappurath, A., Abdelsalam, M.A., Zhang, R., Brown, M.S.: Time-aware auto white balance in mobile photography. In: ICCV (2025) 12, 13, 19, 31
- 7. Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., English, J., English, Z., Esser, P., Kulal, S., et al.: FLUX. 1 Kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742 (2025) 2, 9, 11
- 8. Bemana, M., Leimkühler, T., Myszkowski, K., Seidel, H.P., Ritschel, T.: Bracket diffusion: HDR image generation by consistent LDR denoising. Computer Graphics Forum 44(2), e70086:1–13 (2025) 5
- 9. Berdan, R., Besbinar, B., Reinders, C., Otsuka, J., Iso, D.: ReRAW: RGB-to-raw image reconstruction via stratified sampling for efficient object detection on the edge. In: CVPR (2025) 2, 13
- 10. Betker, J., Goh, G., Jing, L., Brooks, T., Wang, J., Li, L., Ouyang, L., Zhuang, J., Lee, J., Guo, Y., Manassra, W., Dhariwal, P., Chu, C., Jiao, Y., Ramesh, A.: Improving image generation with better captions. Tech. rep., OpenAI (2023), https://cdn.openai.com/papers/dall-e-3.pdf, technical report 2
- 11. Brooks, T., Mildenhall, B., Xue, T., Chen, J., Sharlet, D., Barron, J.T.: Unprocessing images for learned raw denoising. In: CVPR (2019) 4, 14
- 12. Bychkovsky, V., Paris, S., Chan, E., Durand, F.: Learning photographic global tonal adjustment with a database of input / output image pairs. In: CVPR (2011) 9, 10, 19
- 13. Chakrabarti, A., Scharstein, D., Zickler, T.E.: An empirical camera model for internet color vision. In: BMVC (2009) 4
- 14. Chakrabarti, A., Xiong, Y., Sun, B., Darrell, T., Scharstein, D., Zickler, T., Saenko, K.: Modeling radiometric uncertainty for vision with tone-mapped color images. IEEE Transactions on Pattern Analysis and Machine Intelligence 36(11), 2185– 2198 (2014) 4
- 15. Chen, Y., Wen, Y., Li, W., Liu, J., Guo, Y., Hu, J., Chen, X.: RDDM: Practicing raw domain diffusion model for real-world image restoration. arXiv preprint arXiv:2508.19154 (2025) 4
- 16. Cheng, D., Prasad, D.K., Brown, M.S.: Illuminant estimation for color constancy: Why spatial-domain methods work and the role of the color distribution. Journal of the Optical Society of America A 31(5), 1049–1058 (2014) 18
- 17. Conde, M.V., McDonagh, S., Maggioni, M., Leonardis, A., Pérez-Pellitero, E.: Model-based image signal processors via learnable dictionaries. In: AAAI (2022) 4

- 18. Dai, W., Li, J., Li, D., Tiong, A., Zhao, J., Wang, W., Li, B., Fung, P.N., Hoi, S.: InstructBLIP: Towards general-purpose vision-language models with instruction tuning. NeurIPS (2023) 12, 19, 25
- 19. Dai, X., Hou, J., Ma, C.Y., Tsai, S., Wang, J., Wang, R., Zhang, P., Vandenhende, S., Wang, X., Dubey, A., et al.: Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807 (2023) 2
- 20. Dang-Nguyen, D.T., Pasquini, C., Conotter, V., Boato, G.: Raise: A raw images dataset for digital image forensics. In: ACM Multimedia Systems (2015) 9
- 21. Debevec, P.E., Malik, J.: Recovering high dynamic range radiance maps from photographs. In: ACM SIGGRAPH (2008) 4
- 22. Delbracio, M., Kelly, D., Brown, M.S., Milanfar, P.: Mobile computational photography: A tour. Annual review of vision science 7(1), 571–604 (2021) 4
- 23. Dhariwal, P., Nichol, A.: Diffusion models beat GANs on image synthesis. NeurIPS

(2021) 2

- 24. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: ICML (2024) 2
- 25. Foi, A., Trimeche, M., Katkovnik, V., Egiazarian, K.: Practical PoissonianGaussian noise modeling and fitting for single-image raw-data. IEEE Transactions on Image Processing 17(10), 1737–1754 (2008) 18
- 26. Gong, H.: Convolutional mean: A simple convolutional neural network for illuminant estimation. In: BMVC (2019) 19
- 27. Grossberg, M.D., Nayar, S.K.: Determining the camera response from images: What is knowable? IEEE Transactions on Pattern Analysis and Machine Intelligence 25(11), 1455–1467 (2003) 4
- 28. He, X., Hu, T., Wang, G., Wang, Z., Wang, R., Zhang, Q., Yan, K., Chen, Z., Li, R., Xie, C., et al.: Enhancing raw-to-sRGB with decoupled style structure in Fourier domain. In: AAAI (2024) 4
- 29. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. NeurIPS

(2020) 2, 4

- 30. Ho, J., Salimans, T.: Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022) 2
- 31. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: LoRA: Low-rank adaptation of large language models. In: ICLR (2022) 7
- 32. Ignatov, A., Sycheva, A., Timofte, R., Tseng, Y., Xu, Y.S., Yu, P.H., Chiang, C.M., Kuo, H.K., Chen, M.H., Cheng, C.M., et al.: MicroISP: Processing 32MP photos on mobile devices with deep learning. In: ECCV (2022) 4
- 33. Ignatov, A., Van Gool, L., Timofte, R.: Replacing mobile camera ISP with a single deep learning model. In: CVPRW (2020) 4, 12
- 34. Jiang, H., Luo, A., Fan, H., Han, S., Liu, S.: Low-light image enhancement with wavelet-based diffusion models. ACM Transactions on Graphics (TOG) 42(6), 1– 14 (2023) 4
- 35. Jiang, Y., Gong, X., Liu, D., Cheng, Y., Fang, C., Shen, X., Yang, J., Zhou, P., Wang, Z.: EnlightenGAN: Deep light enhancement without paired supervision. IEEE Transactions on Image Processing 30, 2340–2349 (2021) 14
- 36. Karras, T., Aittala, M., Aila, T., Laine, S.: Elucidating the design space of diffusionbased generative models. NeurIPS (2022) 2
- 37. Kawar, B., Elad, M., Ermon, S., Song, J.: Denoising diffusion restoration models. NeurIPS 35, 23593–23606 (2022) 4

- 38. Kim, S.J., Lin, H.T., Lu, Z., Süsstrunk, S., Lin, S., Brown, M.S.: A new in-camera imaging model for color computer vision and its application. IEEE Transactions on Pattern Analysis and Machine Intelligence 34(12), 2289–2302 (2012) 4
- 39. Kim, W., Kim, G., Lee, J., Lee, S., Baek, S.H., Cho, S.: ParamISP: Learned forward and inverse ISPs using camera parameters. In: CVPR (2024) 4
- 40. Luo, A., Li, X., Yang, F., Liu, J., Fan, H., Liu, S.: FlowDiffuser: Advancing optical flow estimation with diffusion models. In: CVPR (2024) 4
- 41. van der Maaten, L., Hinton, G.: Visualizing data using t-sne. Journal of Machine Learning Research 9, 2579–2605 (2008) 11
- 42. McInnes, L., Healy, J., Melville, J.: Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426 (2018) 11
- 43. Mitsunaga, T., Nayar, S.K.: Radiometric self calibration. In: CVPR (1999) 4
- 44. Nam, S., Joo Kim, S.: Modelling the scene dependent imaging in cameras with a deep neural network. In: ICCV (2017) 4
- 45. Nichol, A.Q., Dhariwal, P.: Improved denoising diffusion probabilistic models. In: ICML (2021) 2
- 46. Pearson, K.: On lines and planes of closest fit to systems of points in space. Philosophical Magazine 2(11), 559–572 (1901) 11
- 47. Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: ICCV (2023) 2, 5
- 48. Perevozchikov, G., Mehta, N., Afifi, M., Timofte, R.: Rawformer: Unpaired rawto-raw translation for learnable camera ISPs. In: ECCV (2024) 20
- 49. Pernias, P., Rampas, D., Richter, M.L., Pal, C.J., Aubreville, M.: Würstchen: An efficient architecture for large-scale text-to-image diffusion models. arXiv preprint arXiv:2306.00637 (2023) 2
- 50. Podell, D., Weng, C.I., Onoe, Y., et al.: SDXL: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023) 2
- 51. Punnappurath, A., Abuolaim, A., Abdelhamed, A., Levinshtein, A., Brown, M.S.: Day-to-night image synthesis for training nighttime neural ISPs. In: CVPR (2022) 19
- 52. Raimundo, D.W., Ignatov, A., Timofte, R.: LAN: Lightweight attention-based network for raw-to-RGB smartphone image processing. In: CVPRW (2022) 4
- 53. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical textconditional image generation with CLIP latents. arXiv preprint arXiv:2204.06125

(2022) 2

- 54. Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., Sutskever, I.: Zero-shot text-to-image generation. In: ICML (2021) 2
- 55. Reinders, C., Berdan, R., Besbinar, B., Otsuka, J., Iso, D.: Raw-diffusion: RGBguided diffusion models for high-fidelity raw image generation. In: WACV (2025) 2, 4, 9, 10, 13, 20, 21, 22, 23, 24
- 56. Ren, Y., Jiang, H., Yang, M., Li, W., Liu, S.: ISPDiffuser: Learning RAW-tosRGB mappings with texture-aware diffusion models and histogram-guided color consistency. In: AAAI (2025) 4
- 57. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: CVPR (2022) 2, 4
- 58. Ronneberger, O., Fischer, P., Brox, T.: U-Net: Convolutional networks for biomedical image segmentation. In: MICCAI (2015) 19
- 59. Saharia, C., Chan, W., Chang, H., Lee, C., Ho, J., Salimans, T., Fleet, D., Norouzi, M.: Palette: Image-to-image diffusion models. In: SIGGRAPH (2022) 4

- 60. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., et al.: Photorealistic textto-image diffusion models with deep language understanding. In: NeurIPS (2022) 2
- 61. Saharia, C., Ho, J., Chan, W., Salimans, T., Fleet, D.J., Norouzi, M.: Image superresolution via iterative refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence 45(4), 4713–4726 (2022) 4
- 62. Seo, D., Punnappurath, A., Zhao, L., Abdelhamed, A., Tedla, S.K., Park, S., Choe, J., Brown, M.S.: Graphics2RAW: Mapping computer graphics images to sensor raw images. In: ICCV (2023) 9, 13, 14, 16
- 63. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020) 2, 4
- 64. Song, Y., Dhariwal, P., Chen, M., Sutskever, I.: Consistency models. arXiv e-prints

(2023) 2

- 65. Song, Y., Ermon, S.: Generative modeling by estimating gradients of the data distribution. NeurIPS (2019) 2
- 66. Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Scorebased generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 (2020) 2
- 67. Wang, C., Xia, Z., Leimkuhler, T., Myszkowski, K., Zhang, X.: LEDiff: Latent exposure diffusion for HDR generation. In: CVPR (2025) 5
- 68. Wang, Y., Yu, J., Zhang, J.: Zero-shot image restoration using denoising diffusion null-space model. arXiv preprint arXiv:2212.00490 (2022) 4
- 69. Wang, Y., Yu, Y., Yang, W., Guo, L., Chau, L.P., Kot, A.C., Wen, B.: ExposureDiffusion: Learning to expose for low-light image enhancement. In: ICCV (2023) 4
- 70. Xing, Y., Qian, Z., Chen, Q.: Invertible image signal processing. In: CVPR (2021) 2, 4, 9, 10, 13, 19, 31
- 71. Yuan, Y., Wang, X., Sheng, Y., Chennuri, P., Zhang, X., Chan, S.: Generative photography: Scene-consistent camera control for realistic text-to-image synthesis. In: CVPR (2025) 4, 14
- 72. Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.H.: Restormer: Efficient transformer for high-resolution image restoration. In: CVPR (2022) 19
- 73. Zamir, S.W., Arora, A., Khan, S., et al.: CycleISP: Real image restoration via improved data synthesis. In: CVPR (2020) 4
- 74. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image diffusion models. arXiv preprint arXiv:2302.05543 (2023) 2
- 75. Zhang, Z., Wang, H., Liu, M., Wang, R., Zhang, J., Zuo, W.: Learning raw-tosRGB mappings with inaccurately aligned supervision. In: ICCV (2021) 4, 19, 31
- 76. Zhou, C., Yu, L., Babu, A., Tirumala, K., Yasunaga, M., Shamis, L., Kahn, J., Ma, X., Zettlemoyer, L., Levy, O.: Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039 (2024) 2

