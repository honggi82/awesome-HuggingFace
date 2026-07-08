## AnyStory: Towards Unified Single and Multiple Subject Personalization in Text-to-Image Generation

Junjie He Yuxiang Tuo Binghui Chen Chongyang Zhong Yifeng Geng Liefeng Bo Institute for Intelligent Computing, Alibaba Tongyi Lab

{hejunjie.hjj, yuxiang.tyx}@alibaba-inc.com chenbinghui@bupt.cn {zhongchongyang.zzy, cangyu.gyf, liefeng.bo}@alibaba-inc.com

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

# arXiv:2501.09503v2[cs.CV]1May2025

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

The fearless explorer faces challenges in snowy mountains, lush jungles, misty forests, and vast deserts in stunning and vibrant art styles.

A witch harnesses the power of magic in a series of fantastical and otherworldly scenes.

A pirate finds treasure on a deserted island, navigates through fog-enshrouded ancient ruins, and peers around on the deck of his ship under a star-filled sky.

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

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

The Captain is portrayed in different dramatic and adventurous scenes on the ship. The panda stood against a mountain backdrop, rolled in cherry blossoms, wandered through the bamboo forest, and lounged in a field of golden wheat.

A determined avocado competes in track and field events.

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

A boy and girl play in a meadow, gaze at the stars, and walk to school

hand in hand in a vibrant

paper cut-out style.

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Two men pause mid-air in a battle, exchanging rapid punches and kicks, before reaching a mutual respect and understanding.

Figure 1. Example generations I from AnyStory. Our approach demonstrates excellence in preserving subject details, aligning text descriptions, and personalizing multiple subjects. Here, the image with a plain white background serves as the reference. For more examples, please refer to Fig. 7 and Fig. 8.

### Abstract

alization for single subjects, but also for multiple subjects, without sacrificing subject fidelity. Specifically, AnyStory models the subject personalization problem in an “encodethen-route” manner. In the encoding step, AnyStory utilizes a universal and powerful image encoder, i.e., ReferenceNet, in conjunction with CLIP vision encoder to achieve highfidelity encoding of subject features. In the routing step, AnyStory utilizes a decoupled instance-aware subject router to accurately perceive and predict the potential location of

Recently, large-scale generative models have demonstrated outstanding text-to-image generation capabilities. However, generating high-fidelity personalized images with specific subjects still presents challenges, especially in cases involving multiple subjects. In this paper, we propose AnyStory, a unified approach for personalized subject generation. AnyStory not only achieves high-fidelity person-

the corresponding subject in the latent space, and guide the injection of subject conditions. Detailed experimental results demonstrate the excellent performance of our method in retaining subject details, aligning text descriptions, and personalizing for multiple subjects. The project page is at https://aigcdesigngroup.github. io/AnyStory/.

### 1. Introduction

Recently, with the rapid development of diffusion models [14, 27, 58, 59], many large generative models [4, 8, 9, 43, 44, 48, 50, 52] have demonstrated remarkable textto-image generation capabilities. However, generating personalized images with specific subjects still presents challenges. Early efforts [2, 7, 15, 21, 28, 38, 53] utilize finetuning at test time to achieve personalized content generation. These methods require extensive fine-tuning time and their generalization ability is limited by the number and diversity of tuning images. Recent works [17, 40, 42, 47, 57, 65, 67, 68, 70, 71] have explored zero-shot settings. They have introduced specialized subject encoders to retrain text-to-image models on large-scale personalized image datasets, without the need for model fine-tuning at test time. However, these methods are either limited by the encoder’s capability to provide high-fidelity subject details [40, 47, 57, 67, 70, 71], or focus on specific categories of objects (such as face identities [42, 65, 68]) and cannot extend to general subjects (such as human clothing, accessories, and non-human entities), limiting their applicability.

In addition, previous methods mainly focus on singlesubject personalization. Problems with subject blending often occur in multi-subject generation due to semantic leakage [11, 68]. Some methods [19, 32, 35, 39, 45, 47, 66, 72] address this issue by introducing pre-defined subject masks, but this restricts the diversity and creativity of generative models. Additionally, providing precise masks for subjects with complex interactions and occlusions is difficult. Recent research, i.e., UniPortrait [24], proposes a subject router to adaptively perceive and constrain the effect region of each subject condition in the diffusion denoising process. However, the routing features used by UniPortrait are highly coupled with subject identity features, limiting the accuracy and flexibility of the routing module. Furthermore, it primarily focuses on the domain of face identity and does not consider the impact of subject conditions on the background.

In this paper, we propose AnyStory, a unified single- and multi-subject personalization framework. We aim to personalize general subjects while achieving fine-grained control over multi-subject conditions. Additionally, we aim to allow the variation of subjects’ backgrounds, poses, and views through text prompts while maintaining subject de-

tails, thus creating complex and fantastical narratives.

To achieve this, we have introduced two key modules, i.e., an enhanced subject representation encoder and a decoupled instance-aware subject router. To be specific, we adopt the “encode-then-route” design of UniPortrait. In order to achieve a general subject representation, we abandon domain-specific expert models, such as the face encoders [13, 30], and instead use a powerful and versatile model, i.e., ReferenceNet [29], combined with the CLIP vision encoder [51] to encode the subject. CLIP vision encoder is responsible for encoding the subject’s coarse concepts, while ReferenceNet is responsible for encoding the appearance details to enhance subject fidelity. To improve efficiency, we also simplify the architecture of ReferenceNet, skipping all cross-attention layers to save storage and computation costs. In order to avoid the copy-paste effect, we further collect a large amount of paired subject data, which is sourced from image, video, and 3D rendering databases. These paired data contain instances of the same subject in different contexts, effectively aiding the encoder in understanding and encoding provided subject concepts.

For the subject router, in contrast to UniPortrait, we implement a separate branch to allow for a specialized and flexible routing guidance. Additionally, we improve the structure of the routing module by modeling it as a mini-image segmentation decoder, with a masked crossattention [10, 23] and a background routing representation being introduced. Combined with instance-aware routing regularization loss, the proposed router can accurately perceive and predict the potential location of the corresponding subject in the latent during the denoising process. In practice, we observe that the behavior of this enhanced subject router is similar to image instance segmentation, which may provide a potential approach for image-prompted visual subject segmentation.

The experimental results demonstrate the outstanding performance of our method in preserving the fidelity of the subject details, aligning text descriptions, and personalizing for multiple subjects. Our contributions can be summarized as follows:

- • We propose a unified single- and multi-subject personalization framework called AnyStory. It achieves consistency in personalizing both single-subject and multisubject while adhering to text prompts;
- • We introduce an enhanced subject representation encoder, composed of a simplified lightweight ReferenceNet and CLIP vision encoder, capable of high-fidelity detail encoding for general subjects.
- • We propose a decoupled instance-aware routing module that can accurately perceive and predict the potential conditioning areas of the subject, thereby achieving flexible and controllable personalized generation of single or multiple subjects.

### 2. Related Work

Single-subject personalization. Personalized image generation with specific subjects is a popular and challenging topic in text-to-image generation. Early works [2, 7, 15, 16, 21, 28, 38, 53, 62] rely on fine-tuning during testing. These methods typically require several minutes to even hours to achieve satisfactory results, and their generalization abilities are limited by the number of fine-tuned images. Recently, some methods [17, 33, 40, 47, 57, 60, 67, 70] have sought to achieve personalized image generation for subjects without additional fine-tuning. IP-Adapter [70] encodes subjects into text-compatible image prompts for subject personalization. BLIP-Diffusion [40] introduces a pretrained multimodal encoder to provide subject representation. SSR-Encoder [71] proposes a token-to-patch aligner and detail-preserved subject encoder to learn selective subject embedding. FaceStudio [69], InstantID [65], and PhotoMaker [42] utilize face embeddings derived from face encoders as the condition. Although these methods have made progress, they are either limited by the ability of the image encoder to preserve subject details [17, 40, 47, 57, 67, 70, 71], or focus on specific domains, e.g., face identity, without the ability to generalize to other objects [20, 24, 42, 65, 69]. Multi-subject personalization. Significant progress has been made in single-subject personalization. However, the personalized generation of multi-subject images still presents challenges due to the problem of subject blending [11, 68]. To overcome these challenges, recent studies [19, 35, 39, 45, 47, 66, 72] have utilized predefined layout masks to guide multi-subject generation. However, these layout-dependent methods limit the creativity of the generation models and the diversity of resulting images. Additionally, providing precise layout masks for each subject in complex contexts is challenging. Some methods obtain subject masks from attention maps corresponding to subject tokens [5, 6, 25, 56, 61, 63] or from segmentation of existing images [22, 37], which may result in inaccurate masks for the target subject instances. FastComposer [68], Subject-Diffusion [47], and StoryMaker [74] impose constraints on cross-attention maps for different subjects during training, but this impacts the injection of subject conditions. Recently, UniPortrait [24] introduces a subject router to perceive and predict subject potential positions during denoising, avoiding blending adaptively. However, its routing features are highly coupled with subject features, limiting the precision of the routing module.

Story visualization. Generating visual narratives based on given scripts, known as story visualization [3, 22, 61, 74], is rapidly evolving. StoryDiffusion [73] proposes a consistent self-attention calculation to ensure the consistency of characters throughout the story sequence. ConsiStory [61] proposes a training-free approach that shares the internal activations of the pre-trained diffusion model to achieve sub-

ject consistency. DreamStory [22] utilizes a Large Language Model (LLM) and a multi-subject consistent diffusion model, incorporating masked mutual self-attention and masked mutual cross-attention modules, to generate consistent multi-subject story scenes. The proposed method in this paper achieves subject consistency in image sequence generation through routed subject conditioning.

### 3. Methods

We introduce AnyStory, a pioneering method for unified single- and multi-subject personalization in text-to-image generation. We first briefly review the background of the diffusion model in Sec. 3.1, and then detail the two proposed key components, i.e., the enhanced subject encoder and the decoupled instance-aware subject router, in Sec. 3.2 and Sec. 3.3, respectively. Finally, we outline our training scheme in Sec. 3.4. The framework of our method is illustrated in Fig. 2.

#### 3.1. Preliminary

The underlying text-to-image model we used in this paper is Stable Diffusion XL (SDXL) [50]. SDXL takes a text prompt P as input and produces the image x0. It contains three modules: an autoencoder (E(·),D(·)), a CLIP text encoder τ(·), and a U-Net ϵθ(·). Typically, it is trained using the following diffusion loss:

0,P,ϵ∼N(0,1),t[∥ϵ − ϵθ(zt,t,τ(P))∥22] (1)

Ldiff = Ez

where ϵ ∼ N(0,1) is the sampled Gaussian noise, t is the time step, z0 = E(x0) is the latent code of x0, and zt is computed by zt = αtz0 + σtϵ with the coefficients αt and σt provided by the noise scheduler.

#### 3.2. Enhanced subject representation encoding

Personalizing subject images in an open domain while ensuring fidelity to subject details and textual descriptions remains an unresolved issue. A key challenge lies in the encoding of subject information, which requires maximal preservation of subject characteristics while maintaining a certain level of editing capability. Current mainstream methods [17, 40, 45, 47, 57, 67, 70, 71] largely rely on CLIP vision encoder to encode subjects. However, CLIP’s features are primarily semantic (for the reason of contrastive image-text training paradigm) and of low-resolution (typically 224 × 224), thus limited to providing thorough details of the subjects. Alternative approaches [20, 42, 49, 65] incorporate domain-specific expert models, such as face encoders [13, 30], to enhance subject identity representation. Despite their success, they are limited in their domain and are not extendable to general subjects. To address these issues, we introduce ReferenceNet [29], a powerful and versatile image encoder, to encode the subject in conjunction

| |
|---|

###### Router Module

Identity (No Cross Att.)

[Figure 53]

[Figure 54]

|(𝑁 + 1) × ℎ𝑤|
|---|

···

Refined Routing Map

[Figure 55]

Lightweight ReferenceNet

Self Att.

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

| |
|---|

| |
|---|

|Linear B| |
|---|---|
| | |

| |Linear A|
|---|---|
| | |

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Routed Cross Att.

| |
|---|

| |
|---|

C

···

··· ···

···

VAE

[Figure 64]

Routed Reference Att.

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

|Masked Cross Attention|
|---|

Router

Subject QFormer

CLIP

···

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

k, v q

mask

Subject features Subject features

U-Net

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

Coarse Routing Map

··· ··· ···

|Linear B| |
|---|---|
| | |

| |Linear A|
|---|---|
| | |

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

···

···

[Figure 88]

Routing QFormer

Latent features

Routing features

CLIP

|ℎ𝑤 × 𝑑|
|---|

|(𝑁 + 1) × 𝑑𝑟|
|---|

Routing features

- Figure 2. Overview of AnyStory framework. AnyStory follows the “encode-then-route” conditional generation paradigm. It first utilizes a simplified ReferenceNet combined with a CLIP vision encoder to encode the subject (Sec. 3.2), and then employs a decoupled instanceaware subject router to guide the subject condition injection (Sec. 3.3). The training process is divided into two stages: the subject encoder training stage and the router training stage (Sec. 3.4). For brevity, we omit the text conditional branch here.

with the CLIP vision encoder. ReferenceNet utilizes a variational autoencoder (VAE) [36, 50] to encode reference images and then extracts their features through a network with the same architecture as U-Net. It boasts three prominent advantages: (1) it supports higher resolution inputs, thereby enabling it to retain more subject details; (2) it has a feature space aligned with the denoising U-Net, facilitating the direct extraction of subject features at different depths and scales by U-Net; (3) it uses pre-trained U-Net weights for initialization, which possess a wealth of visual priors and demonstrate good generalization ability for learning general subject concepts.

CLIP encoding. Following previous approaches [24, 70], we utilize the hidden states from the penultimate layer of the CLIP image encoder, which align well with image captions, as a rough visual concept representation of the subject. We first segment the subject area in the reference image to remove background information, and then input the segmented image into the CLIP image encoder to obtain a 257-length patch-level feature Fclip. Subsequently, we compress Fclip using a QFormer [1, 41] into m tokens. The final result, denoted as E, serves as the subject representation derived from the CLIP vision encoder:

E = QFormer(Fclip), (2) where E ∈ Rm×d

c, and dc is the same as the text feature dimension in the pre-trained diffusion model. Empirically, we set m to 64 in our experiments.

ReferenceNet encoding. In the original implementation [29], ReferenceNet adopts the same architecture as U-Net, including cross-attention blocks with text condition injection. However, since ReferenceNet is only used as a visual feature extractor in our task and does not require text condition injection, we skip all cross-attention

|Architecture|#Params (B) Speed (ms/img)<br><br>|
|---|---|
|Original ReferenceNet [29] Simplified ReferenceNet<br><br>|2.57 62.0 2.02 53.2|

Table 1. Statistics of the simplified ReferenceNet. The speed is measured on an A100 GPU with a batch size of 1 and an input (latent) resolution of 64 × 64.

blocks, reducing the number of parameters and computational complexity (see Table 1). Additionally, in order to label the subject areas, we also add a subject mask channel to the input of ReferenceNet. Specifically, we feed the segmented subject reference to the VAE encoder for encoding and then concatenate the encoded result with the downsampled subject mask to obtain Fvae. Next, Fvae undergoes the reduced non-cross attention ReferenceNet. The hidden states of each self-attention layer, denoted as {Gl|Gl ∈ Rh

GwG×d}l, are extracted as the ReferenceNetencoded representation of the subject:

{Gl}Ll=2 = ReferenceNet(Fvae), (3)

where l denotes the layer index, and L denotes the total number of layers. Here, we ignore the features of the first self-attention layer in order to better align with the routing module (see Sec. 3.3).

#### 3.3. Decoupled instance-aware subject routing

The injection of subject conditions requires careful consideration of injecting positions to avoid the influence on unrelated targets. Previous methods [38, 40, 57, 67, 70, 71] have typically injected the conditional features into the latent through a naive attention module. However, due to the soft weighting mechanism, these approaches are prone to semantic leakage [11, 68], leading to subject characteristics

blending, especially in the generation of instances with similar appearance. Some methods [19, 35, 39, 45, 47, 66, 72] introduce predefined layout masks to address this issue, but this limits their practical applicability. UniPortrait [24] proposes a router to perceive and confine the effect region of subject conditions adaptively; however, its routing features are completely coupled with subject features, which limits the ability of the routing module; also, it does not consider the impact of subject conditions on the background. In this study, we propose a decoupled instance-aware subject routing module, which can accurately and effectively route subject features to the corresponding areas while reducing the impact on unrelated areas.

Decoupled routing mechanism. Different from UniPortrait [24], we employ an independent branch to specifically predict the potential location of subjects in the latent during the denoising process. As depicted in the Fig. 2, given a series of segmented subject images, they are respectively passed through CLIP image encoder and an additional one-query QFormer to obtain the routing features {Ri|Ri ∈ Rd

}Ni=1, where N represents the number of reference subjects. Particularly, we include an additional routing feature RN+1 for the background (zero image as input) to further confine the subject’s conditioning areas. The idea behind this is to alleviate the undesirable biases inherent in subject features on the generated image backgrounds (e.g., we used a large amount of pure white background data from 3D rendering to train the subject encoder). To accurately route the subjects to their respective positions, we employ an image segmentation decoder [10, 23] to model the router. Specifically, in each cross-attention layer of the U-Net, we first predict a coarse routing map by taking the linearly projected inner product of {Ri}Ni=1+1 and Zl. Here, Zl ∈ Rhw×d represents the latent features in the l-th layer. Subsequently, we refine the routing features {Ri}Ni=1+1 using a masked cross attention [10] with the latent feature Zl, where the coarse routing map serves as the attention mask. The updated routing features are then subjected to the projected inner product with Zl again to obtain the refined routing maps {Mli | Mli ∈ [0,1]hw}Ni=1+1. {Mli}Ni=1+1 are finally used to guide the injection of information related to the subjects at that layer. For a detailed structure of the router, please refer to the right half of Fig. 2.

r

Instance-aware routing regularization loss. In order to facilitate router learning and to differentiate between different instances of the subjects, we introduce an instanceaware routing regularization loss. The loss function is defined as:

1 N

Llroute = λ ·

N

||Mli − Mgti ||22 (4)

i=1

where Mgti ∈ {0,1}hw represents the downsampled ground truth mask of the i-th subject in the target image. Typically,

we consider the entire subject instance, such as the full human body, as the routing target, regardless of whether the input subject has been cropped.

Routing-guided subject information injection. For CLIP encoded subject representations, we use the decoupled cross attention [70] to incorporate them into the U-Net, but with additional routing-guided localization constraints:

##### QKT √

Zˆl =Softmax(

##### )V

d

##### QKˆ liT

N+1

)Vˆ il,

σ(Mli) ⊙ Softmax(

√

+ η

d

i=1

(5)

where Q = ZlWql , K = CWkl , and V = CWvl represent the query, key, and value matrices for text condi-

tions, C represents text embeddings, Kˆ li = EiWˆ kl and Vˆ il = EiWˆ vl represent the key and value matrices for the CLIP-encoded i-th subject, Wˆ kl and Wˆ vl are both trainable parameters, σ(Mli) represents the “0-1” version of Mli after operations argmax and one-hot for {Mli}i over i dimension, ⊙ represents element-wise multiplication, and η represents the strength of conditions. It should be noted that here we also include an additional background representation, i.e., EN+1, from CLIP, which similarly corresponds to zero-valued image inputs. This embedding (EN+1) is also utilized as an unconditional embedding during classifierfree guidance sampling training [26].

In regard to the injection of ReferenceNet encoded subject features, we adopt the original reference attention [29] but with an additional attention mask induced from routing maps. With a slight abuse of notation, this process can be formulated as follows:

Q[K,K˜ l1,··· ,K˜ lN]T

Z˜l =Softmax(

√

- (6)

where Q = ZlWql , K = ZlWkl , and V = ZlWvl represent the query, key, and value matrices for self-attention,

K˜ li = GliW˜ kl and V˜ il = GliW˜ vl indicate the key and value matrices for the ReferenceNet-encoded features of the i-th

subject at the l-th layer, [·] represents concat operation, Bias({Mli−1}i,γ) represents the applied attention bias, Bias({Mli−1}i,γ) = [0,g(Ml1−1)+γ,··· ,g(MlN−1)+γ],

- (7)

d

+ Bias({Mli−1}i,γ))[V,V˜ 1l ,··· ,V˜ Nl ],

where γ controls the overall strength of ReferenceNet conditions, g(Mli−1) ∈ {0,−∞}hw×h

GwG represents the attention bias derived from the routing maps of the preceding cross-attention layer, and its specific calculation is as follows:

g(Mli−1)u,v =

0 if σ(Mli−1)u = 1 −∞ otherwise

. (8)

Reference Prompt w/o subject router w/ subject router

Reference

Prompt

SDXL

[Figure 89]

[Figure 90]

[Figure 91]

As the two women strolled

[Figure 92]

[Figure 93]

In the vibrant, colorful style of a Japanese manga, a young man sits at a small, wooden table in a sunlit kitchen, eating noodles with his chopsticks skillfully held between his fingers, while steam rises up from the bowl in a whimsical swirl.

through the bustling streets, colorful banners and neon

signs illuminated the sky,

casting a soft glow over their excited expressions.

[Figure 94]

[Figure 95]

[Figure 96]

As the sun set outside the

[Figure 97]

window, the two plump apples sat side by side on

the wooden table, casting

playful shadows on the delicate, hand-drawn tablecloth with intricate

w/ CLIP vision encoder &

w/ CLIP vision encoder

w/ ReferenceNet encoder

ReferenceNet encoder

[Figure 98]

floral patterns, evoking a

[Figure 99]

[Figure 100]

[Figure 101]

sense of cozy nostalgia.

[Figure 102]

[Figure 103]

[Figure 104]

The fox sat on the chair,

holding a phone in its hands, and playing with it while

surrounded by a beautiful

garden full of colorful flowers, the children's illustration style making the

scene delightful and

whimsical.

- Figure 3. Effect of ReferenceNet encoding. The ReferenceNet encoder enhances the preservation of subject details.

[Figure 105]

[Figure 106]

[Figure 107]

On a snowy Christmas Eve,

under a sky adorned with shimmering stars, a majestic

elk gracefully trotted

through a forest blanketed with glistening white snow, capturing the essence of a

nostalgic winter wonderland.

Similar to UniPortrait, to ensure proper gradient backpropagation through σ(·) during training, we employ the Gumbelsoftmax trick [31]. In practice, we observed that the routing map behaves similarly to the instance segmentation mask, providing a potential method for reference-prompted image segmentation (first encode the image with VAE, then feed the encoded image and reference into denoising U-Net and router respectively to predict the masks, see Fig. 5).

Figure 4. The effectiveness of the router. The router restricts the influence areas of the subject conditions, thereby avoiding the blending of characteristics between multiple subjects and improving the quality of the generated images.

training loss in this stage is the same as the original diffusion loss, as shown in Eq. 1.

#### 3.4. Training

Router training. We fix the subject encoder and train the router. The primary training data consists of an additional 300k unpaired multi-person images from LAION [54, 55]. Surprisingly, despite the training dataset of the router being predominantly focused on human images, it is able to effectively generalize to general subjects. We attribute this to the powerful generalization ability of the CLIP model and the highly compressed single-token routing features. The training loss for this stage includes the diffusion loss (Eq. 1) and the routing regularization loss (Eq. 4), with the balancing parameter λ set to 0.1.

Following UniPortrait, the training process of AnyStory is divided into two stages: subject encoder training stage and router training stage.

Subject encoder training. We train the subject QFormer, ReferenceNet, and corresponding key, value matrices in attention blocks. The ReferenceNet utilizes pre-trained UNet weights for initialization. To avoid the copy-paste effect caused by fine-grained encoding of subject features, we collect a large amount of paired data that maintains consistent subject identity while displaying variations in background, pose, and views. These data are sourced from image, video, and 3D rendering databases, captioned by Qwen2-VL [64]. Specifically, the image (about 410k) and video (about 520k) data primarily originate from human-centric datasets such as DeepFashion2 [18] and human dancing videos, while the 3D data (about 5,600k) is obtained from the Objaverse [12], where images of objects from seven different perspectives are rendered as paired data. During the training process, one image from these pairs is utilized as the reference input, while another image, depicting the same subject identity but in a different context, serves as the prediction target. Additionally, data augmentation techniques, including random rotation, cropping, and zero-padding, are applied to the reference image to further prevent subject overfitting. The

### 4. Experiments

#### 4.1. Setup

We use the stable diffusion XL [50] as the base model. The CLIP image encoder employed is the OpenAI’s clip-vit-huge-patch14. Both the subject QFormer and the routing QFormer consist of 4 layers. The input image resolution for ReferenceNet is 512×512. All training is conducted on 8 A100 GPUs with a batch size of 64, utilizing the AdamW [46] optimizer with a learning rate of 1e-4. In order to facilitate classifier-free guidance sampling [26], we drop the CLIP subject conditioning during training on 10% of the images. During the inference process, we employ 25

Reference Prompt Personalized Results t=25 t=24 t=22 t=18 t=10 t=1

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

As the two women strolled through the

bustling streets,

colorful banners and neon signs illuminated

the sky, casting a soft

glow over their excited expressions.

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

As the sun set outside the window, the two plump apples sat side by

[Figure 130]

side on the wooden table,

casting playful shadows on the delicate, handdrawn tablecloth with intricate floral patterns, evoking a sense of cozy nostalgia.

[Figure 131]

[Figure 132]

[Figure 133]

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

The man lies peacefully in bed, his body curled up in a fetal position, his chest rising and falling

with each serene breath,

the soft glow of the moon casting an ethereal light over the tranquil scene, capturing the essence of peaceful serenity in this anime-

inspired depiction.

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

A spiky hedgehog sits at the edge of the tranquil

lake, fishing with a

makeshift rod as the golden sunlight dances on the water, capturing

the whimsical charm of a

Pixar illustration.

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

The cartoon dragon sits

on a vintage-style

armchair, holding a small cup of coffee in one claw, with the other

claw resting on the arm

of the chair, and a mischievous glint in his

eye, set against a

backdrop of a cozy, oldfashioned living room with intricate patterns

and warm, rich colors.

Figure 5. Visualization of routing maps. We visualize the routing maps within each cross-attention layer in the U-Net at different diffusion time steps. There are a total of 70 cross-attention layers in the SDXL U-Net, and we sequentially display them in each subfigure in a top-to-bottom and left-to-right order (yellow represents the effective region). We utilize T = 25 steps of EDM sampling. Each complete row corresponds to one entity. The background routing map has been ignored, which is the complement of the routing maps of all subjects. Best viewed in color and zoomed in.

#### 4.3. Effect of the decoupled instance-aware router

steps of EDM [34] sampling and a 7.5 classifier-free guidance scale, and to achieve more realistic image generation, we employ the RealVisXL V4.0 model from huggingface.

- Fig. 4 demonstrates the effectiveness of the proposed router, which can effectively avoid feature blending between subjects in multi-subject generation. Additionally, we observe that the use of the router in single-subject settings also improves the quality of generated images, particularly in the image background. This is because the router restricts the influence area of subject conditions, thereby reducing the potential bias inherent in subject features (e.g., simple white background preference learned from a large amount of 3D rendering data) on the quality of generated images.
- Fig. 5 visualizes the routing maps of the diffusion model

#### 4.2. Effect of ReferenceNet encoder

Fig. 3 illustrates the effectiveness of the ReferenceNet encoder, which enhances the preservation of fine details in the subject compared to using only CLIP vision encoder. However, it is also evident that using ReferenceNet alone does not yield satisfactory results. In fact, in our extensive testing, we found that the ReferenceNet encoder only achieves alignment of the subject details and does not guide subject generation. We still need to rely on CLIP-encoded features, which are well-aligned with text embeddings, to trigger subject generation.

at different time steps during the denoising process. These results demonstrate that the proposed router can accurately

Reference Prompt Personalized Results t=25 t=24 t=22 t=18 t=10 t=1

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

The two boys sat at the kitchen table,

surrounded by books and

papers, earnestly discussing their homework as the evening

sunlight flooded through

the window, casting warm, golden highlights

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

on their faces and the

tabletop.

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

The two boys sat at the kitchen table, surrounded by books and

Reference Prompt Personalized Results t=25 t=24 t=22 t=18 t=10 t=1

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

The two boys sat at the kitchen table,

papers, earnestly

discussing their homework as the evening sunlight flooded through the window, casting warm, golden highlights on their faces and the

surrounded by books and

papers, earnestly discussing their homework as the evening

sunlight flooded through

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

the window, casting warm, golden highlights

tabletop.

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

on their faces and the

tabletop.

(a) Coarse routing maps

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

The two boys sat at the kitchen table, surrounded by books and

Reference Prompt Personalized Results t=25 t=24 t=22 t=18 t=10 t=1

papers, earnestly

discussing their homework as the evening sunlight flooded through the window, casting warm, golden highlights on their faces and the

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

The two boys sat at the kitchen table,

surrounded by books and

papers, earnestly discussing their homework as the evening sunlight flooded through the window, casting warm, golden highlights

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

tabletop.

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

on their faces and the

tabletop.

(b) Refined routing maps

Reference Prompt Personalized Results t=25 t=24 t=22 t=18 t=10 t=1

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

The two boys sat at the kitchen table,

Figure 6. Effectiveness of the proposed router structure. For the meaning of each illustration, please refer to Fig. 5.

surrounded by books and

papers, earnestly discussing their homework as the evening sunlight flooded through the window, casting warm, golden highlights

ject condition injection. Experimental results demonstrate that our method excels in retaining subject details, aligning with textual descriptions, and personalizing for multiple subjects.

perceive and locate the effect regions of each subject condition during the denoising process. The displayed routing maps are similar to image segmentation masks, indicating the potential for achieving guided image segmentation based on reference images through denoising U-Net and trained routers. Additionally, as mentioned in Sec. 3.4, despite our router being trained predominantly on humancentric datasets, it generalizes well to general subjects such as the cartoon dinosaur in Fig. 5. We attribute this to the powerful generalization capability of the CLIP model and the highly compressed single-token routing features.

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

on their faces and the

tabletop.

Limitations and future work. Currently, AnyStory is unable to generate personalized backgrounds for images. However, maintaining consistency in the image background is equally important in sequential image generation. In the future, we will expand AnyStory’s control capabilities from the subject domain to the background domain. Additionally, the copy-paste effect still exists in the subjects generated by AnyStory, and we aim to mitigate this further in the future through data augmentation and the use of more powerful text-to-image generation models.

Fig. 6 demonstrates the effectiveness of modeling the router as a miniature image segmentation decoder. Compared to the coarse routing map obtained by a simple dot product, the refined routing map through a lightweight masked cross-attention module can more accurately predict the potential position of each subject.

#### 4.4. Example generations

In Fig. 1, Fig. 7, and Fig. 8, we visualize further results of our approach, demonstrating its outstanding performance in preserving subject details, aligning text prompts, and enabling multi-subject personalization.

### 5. Conclusion

We propose AnyStory, a unified method for personalized generation of both single and multiple subjects. AnyStory utilizes a universal and powerful ReferenceNet in addition to a CLIP vision encoder to achieve high-fidelity subject encoding, and employs a decoupled, instance-aware routing module for flexible and accurate single/multiple sub-

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

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

[Figure 268]

[Figure 269]

A playful illustration of a friendly shark adds a charming and unexpected twist to the traditional image of a fearsome predator.

An egg baby explores a beautiful and enchanting meadow, creating a whimsical scene.

Mr. Crocodile is a multi-talented and versatile individual, adept at excelling in various professions and environments.

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

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

A heartwarming scene of snowmen in a cozy winter wonderland. A brave monster warrior explores a dark cave, encountering dangers and treasure along the way.

A zombie cautiously explores a dark, eerie cave filled with bats, bones, and glowing crystals.

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

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

[Figure 304]

[Figure 305]

A futuristic driver navigates a cyberpunk city in a sleek, bright orange car. A camel journeys through the desert, encountering various sights and experiences along the way.

A determined warrior triumphs in battle across both desolate and fantastical landscapes.

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

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

A hunter leads a young boy through a mysterious forest. Two little fish swim and play together joyfully in the magical world beneath the

A bee and a cat playfully interact in a whimsically painted scene.

waves.

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

A boy and girl enjoy a

sunny day in a meadow

and share a tender moment in the rain.

###### Figure 7. Example generations II from AnyStory.

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

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

The elegant and regal bear enjoyed a gourmet meal at an ornate dining table. A mischievous and studious monkey worked diligently on homework in a cozy, candlelit library surrounded by clutter and the scent of old parchment.

A martial artist performs intense training in the rain and showcases his skills in an urban setting.

[Figure 350]

[Figure 351]

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

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

A cartoon doctor is depicted in various medical settings, including working with medical tools, flying on a magical bed, and performing a complex medical procedure.

A powerful wizard uses magic to defeat an army of skeletons on the battlefield. A robot explores different themed playgrounds and cityscapes, showcasing its abilities and personality.

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

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

A cartoon dinosaur leads a lively dance party with children, and has a delightful picnic with a little girl.

A poet finds inspiration in nature and tranquility, expressing his thoughts in verses and classical art.

A young college student is depicted running, laughing with friends, studying in a café, giving a speech, and presenting in class with confidence and determination.

[Figure 386]

[Figure 387]

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

A young girl works on her homework in her cozy bedroom, adding illustrations with

A man and a woman share a secret conversation and race through the city streets, finding solace in their shared solitude.

a set of colorful markers as the sun sets and the stars begin to twinkle outside.

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

Two women enjoy a

peaceful day together,

dancing and watching the sunset.

###### Figure 8. Example generations III from AnyStory.

### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. NeurIPS, 35: 23716–23736, 2022. 4
- [2] Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel CohenOr, and Dani Lischinski. Break-a-scene: Extracting multiple concepts from a single image. In SIGGRAPH Asia 2023 Conference Papers, pages 1–12, 2023. 2, 3
- [3] Omri Avrahami, Amir Hertz, Yael Vinker, Moab Arar, Shlomi Fruchter, Ohad Fried, Daniel Cohen-Or, and Dani Lischinski. The chosen one: Consistent characters in textto-image diffusion models. In ACM SIGGRAPH 2024 conference papers, pages 1–12, 2024. 3
- [4] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023. 2
- [5] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In ICCV, pages 22560–22570, 2023. 3
- [6] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM TOG, 42(4):1–10, 2023. 3
- [7] Hong Chen, Yipeng Zhang, Simin Wu, Xin Wang, Xuguang Duan, Yuwei Zhou, and Wenwu Zhu. Disenbooth: Identitypreserving disentangled tuning for subject-driven text-toimage generation. arXiv preprint arXiv:2305.03374, 2023. 2, 3
- [8] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. PixArt-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 2
- [9] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. PixArt-Σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In ECCV, pages 74–91. Springer, 2025. 2
- [10] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In CVPR, pages 1290–1299, 2022. 2, 5
- [11] Omer Dahary, Or Patashnik, Kfir Aberman, and Daniel Cohen-Or. Be yourself: Bounded attention for multisubject text-to-image generation. In ECCV, pages 432–448. Springer, 2025. 2, 3, 4
- [12] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In CVPR, pages 13142– 13153, 2023. 6

- [13] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In CVPR, pages 4690–4699, 2019. 2, 3
- [14] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. NeurIPS, 34:8780–8794,

2021. 2

- [15] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In ICLR, 2023. 2, 3
- [16] Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Designing an encoder for fast personalization of text-to-image models. In Siggraph, 2023. 3
- [17] Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Encoder-based domain tuning for fast personalization of text-to-image models. ACM TOG, 42(4):1–13, 2023. 2, 3
- [18] Yuying Ge, Ruimao Zhang, Xiaogang Wang, Xiaoou Tang, and Ping Luo. Deepfashion2: A versatile benchmark for detection, pose estimation, segmentation and re-identification of clothing images. In CVPR, pages 5337–5345, 2019. 6
- [19] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized lowrank adaptation for multi-concept customization of diffusion models. NeurIPS, 36, 2024. 2, 3, 5
- [20] Zinan Guo, Yanze Wu, Zhuowei Chen, Lang Chen, Peng Zhang, and Qian He. Pulid: Pure and lightning id customization via contrastive alignment. arXiv preprint arXiv:2404.16022, 2024. 3
- [21] Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. arXiv preprint arXiv:2303.11305, 2023. 2, 3
- [22] Huiguo He, Huan Yang, Zixi Tuo, Yuan Zhou, Qiuyue Wang, Yuhang Zhang, Zeyu Liu, Wenhao Huang, Hongyang Chao, and Jian Yin. Dreamstory: Open-domain story visualization by llm-guided multi-subject consistent diffusion. arXiv preprint arXiv:2407.12899, 2024. 3
- [23] Junjie He, Pengyu Li, Yifeng Geng, and Xuansong Xie. Fastinst: A simple query-based model for real-time instance segmentation. In CVPR, pages 23663–23672, 2023. 2, 5
- [24] Junjie He, Yifeng Geng, and Liefeng Bo. Uniportrait: A unified framework for identity-preserving singleand multi-human image personalization. arXiv preprint arXiv:2408.05939, 2024. 2, 3, 4, 5
- [25] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. ICLR, 2023. 3
- [26] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 5, 6
- [27] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33:6840–6851, 2020. 2
- [28] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.

- LoRA: Low-rank adaptation of large language models. In ICLR, 2022. 2, 3
- [29] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In CVPR, pages 8153–8163, 2024. 2, 3, 4, 5
- [30] Yuge Huang, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, and Feiyue Huang. Curricularface: adaptive curriculum learning loss for deep face recognition. In CVPR, pages 5901–5910, 2020. 2, 3
- [31] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144, 2016. 6
- [32] Sangwon Jang, Jaehyeong Jo, Kimin Lee, and Sung Ju Hwang. Identity decoupling for multi-subject personalization of text-to-image models. arXiv preprint arXiv:2404.04243, 2024. 2
- [33] Xuhui Jia, Yang Zhao, Kelvin CK Chan, Yandong Li, Han Zhang, Boqing Gong, Tingbo Hou, Huisheng Wang, and Yu-Chuan Su. Taming encoder for zero fine-tuning image customization with text-to-image diffusion models. arXiv preprint arXiv:2304.02642, 2023. 3
- [34] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. NeurIPS, 35:26565–26577, 2022. 7
- [35] Chanran Kim, Jeongin Lee, Shichang Joung, Bongmo Kim, and Yeul-Min Baek. Instantfamily: Masked attention for zero-shot multi-id image generation. arXiv preprint arXiv:2404.19427, 2024. 2, 3, 5
- [36] Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 4
- [37] Zhe Kong, Yong Zhang, Tianyu Yang, Tao Wang, Kaihao Zhang, Bizhu Wu, Guanying Chen, Wei Liu, and Wenhan Luo. Omg: Occlusion-friendly personalized multiconcept generation in diffusion models. arXiv preprint arXiv:2403.10983, 2024. 3
- [38] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In CVPR, pages 1931–1941,

2023. 2, 3, 4

- [39] Gihyun Kwon, Simon Jenni, Dingzeyu Li, Joon-Young Lee, Jong Chul Ye, and Fabian Caba Heilbron. Concept weaver: Enabling multi-concept fusion in text-to-image models. In CVPR, pages 8880–8889, 2024. 2, 3, 5
- [40] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pretrained subject representation for controllable text-to-image generation and editing. NeurIPS, 36, 2024. 2, 3, 4
- [41] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 4
- [42] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, MingMing Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. In CVPR, pages 8640–8650, 2024. 2, 3
- [43] Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, et al. Hunyuan-dit: A powerful

- multi-resolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748, 2024. 2
- [44] Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Chase Lambert, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-to-image alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024. 2
- [45] Zhiheng Liu, Yifei Zhang, Yujun Shen, Kecheng Zheng, Kai Zhu, Ruili Feng, Yu Liu, Deli Zhao, Jingren Zhou, and Yang Cao. Cones 2: Customizable image synthesis with multiple subjects. arXiv preprint arXiv:2305.19327, 2023. 2, 3, 5
- [46] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 6
- [47] Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subject-diffusion: Open domain personalized text-to-image generation without test-time fine-tuning. arXiv preprint arXiv:2307.11410, 2023. 2, 3, 5
- [48] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, pages 4195–4205, 2023. 2
- [49] Xu Peng, Junwei Zhu, Boyuan Jiang, Ying Tai, Donghao Luo, Jiangning Zhang, Wei Lin, Taisong Jin, Chengjie Wang, and Rongrong Ji. Portraitbooth: A versatile portrait model for fast identity-preserved personalization. In CVPR, pages 27080–27090, 2024. 3
- [50] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2, 3, 4, 6
- [51] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763. PMLR, 2021. 2
- [52] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684– 10695, 2022. 2
- [53] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, pages 22500–22510, 2023. 2, 3
- [54] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 6
- [55] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. In NeurIPS, pages 25278–25294, 2022. 6
- [56] Dazhong Shen, Guanglu Song, Zeyue Xue, Fu-Yun Wang, and Yu Liu. Rethinking the spatial inconsistency in classifierfree diffusion guidance. In CVPR, pages 9370–9379, 2024. 3

- [57] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without testtime finetuning. In CVPR, pages 8543–8552, 2024. 2, 3, 4
- [58] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, pages 2256– 2265, 2015. 2
- [59] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. NeurIPS, 32, 2019. 2
- [60] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 2024. 3
- [61] Yoad Tewel, Omri Kaduri, Rinon Gal, Yoni Kasten, Lior Wolf, Gal Chechik, and Yuval Atzmon. Training-free consistent text-to-image generation. ACM TOG, 43(4):1–18, 2024. 3
- [62] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. p+: Extended textual conditioning in text-toimage generation. arXiv preprint arXiv:2303.09522, 2023. 3
- [63] Jinglong Wang, Xiawei Li, Jing Zhang, Qingyuan Xu, Qin Zhou, Qian Yu, Lu Sheng, and Dong Xu. Diffusion model is secretly a training-free open vocabulary semantic segmenter. arXiv preprint arXiv:2309.02773, 2023. 3
- [64] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 6
- [65] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024. 2, 3
- [66] Xudong Wang, Trevor Darrell, Sai Saketh Rambhatla, Rohit Girdhar, and Ishan Misra. Instancediffusion: Instance-level control for image generation. In CVPR, pages 6232–6242,

2024. 2, 3, 5

- [67] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. arXiv preprint arXiv:2302.13848, 2023. 2, 3, 4
- [68] Guangxuan Xiao, Tianwei Yin, William T Freeman, Fr´edo Durand, and Song Han. Fastcomposer: Tuning-free multisubject image generation with localized attention. arXiv preprint arXiv:2305.10431, 2023. 2, 3, 4
- [69] Yuxuan Yan, Chi Zhang, Rui Wang, Yichao Zhou, Gege Zhang, Pei Cheng, Gang Yu, and Bin Fu. Facestudio: Put your face everywhere in seconds. arXiv preprint arXiv:2312.02663, 2023. 3
- [70] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 2, 3, 4, 5

- [71] Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan,

- et al. Ssr-encoder: Encoding selective subject representation for subject-driven generation. In CVPR, pages 8069–8078, 2024. 2, 3, 4
- [72] Dewei Zhou, You Li, Fan Ma, Xiaoting Zhang, and Yi Yang. Migc: Multi-instance generation controller for text-to-image synthesis. In CVPR, pages 6818–6828, 2024. 2, 3, 5
- [73] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent selfattention for long-range image and video generation. arXiv preprint arXiv:2405.01434, 2024. 3
- [74] Zhengguang Zhou, Jing Li, Huaxia Li, Nemo Chen, and Xu Tang. Storymaker: Towards holistic consistent characters in text-to-image generation. arXiv preprint arXiv:2409.12576,

2024. 3

## Appendix

### A. Referenced subject images and URLs

This section consolidates the sources of the referenced subject images in this paper. We extend our gratitude to the owners of these images for sharing their valuable assets.

|Reference<br><br>|URL|
|---|---|
|[Figure 407]<br><br>[Figure 408]<br><br>[Figure 409]<br><br>[Figure 410]<br><br>[Figure 411]<br><br>[Figure 412]<br><br>[Figure 413]<br><br>[Figure 414]<br><br>[Figure 415]<br><br>[Figure 416]<br><br>[Figure 417]<br><br>[Figure 418]<br><br>[Figure 419]<br><br>[Figure 420]<br><br>[Figure 421]<br><br>[Figure 422]<br><br>|https://pixabay.com/illustrations/ai-generated-dwarf-story-fantasy-8697130/<br><br>https://pixabay.com/illustrations/girl-coat-night-night-city-8836068/<br><br>https://pixabay.com/vectors/man-warrior-art-character-cartoon-9093563/<br><br>https://pixabay.com/photos/mario-figure-game-nintendo-super-1558068/<br><br>https://pixabay.com/illustrations/panda-cartoon-2d-art-character-7918136/<br><br>https://pixabay.com/illustrations/avocado-food-fruit-6931344/<br><br>https://pixabay.com/vectors/guy-anime-cartoon-chibi-character-7330732/<br><br>https://pixabay.com/vectors/guy-anime-cartoon-chibi-character-7330788/<br><br>https://pixabay.com/photos/young-male-man-japanese-anime-3815077/<br><br>https://pixabay.com/photos/young-male-man-japanese-anime-3816557/<br><br>https://pixabay.com/illustrations/shark-jaws-fish-animal-marine-life-2317422/<br><br>https://unsplash.com/photos/white-egg-with-face-illustration-WtolM5hsj14<br><br>https://pixabay.com/vectors/alligator-crocodile-suit-cartoon-576481/<br><br>https://pixabay.com/illustrations/snowman-winter-christmas-time-snow-7583640/<br><br>https://pixabay.com/illustrations/monster-cartoon-funny-creature-8534186/<br><br>https://unsplash.com/photos/a-cartoon-character-wearing-a-face-mask-andrunning-6-adg66qleM|

|Reference|URL<br><br>|
|---|---|
|[Figure 423]<br><br>[Figure 424]<br><br>[Figure 425]<br><br>[Figure 426]<br><br>[Figure 427]<br><br>[Figure 428]<br><br>[Figure 429]<br><br>[Figure 430]<br><br>[Figure 431]<br><br>[Figure 432]<br><br>[Figure 433]<br><br>[Figure 434]<br><br>[Figure 435]<br><br>[Figure 436]<br><br>[Figure 437]<br><br>[Figure 438]<br><br>[Figure 439]<br><br>[Figure 440]<br><br>[Figure 441]<br><br>|https://pixabay.com/illustrations/car-vehicle-drive-transportation-8316057/<br><br>https://pixabay.com/vectors/camel-desert-two-humped-animal-7751098/<br><br>https://pixabay.com/illustrations/cartoon-samurai-characters-4790355/<br><br>https://pixabay.com/illustrations/caveman-prehistoric-character-9211043/<br><br>https://pixabay.com/illustrations/boy-walk-nature-anime-smile-8350034/<br><br>https://pixabay.com/illustrations/fish-jaw-angry-cartoon-parrot-fish-1402423/<br><br>https://pixabay.com/illustrations/fish-telescope-fish-cartoon-1450768/<br><br>https://pixabay.com/vectors/cat-pet-animal-kitty-kitten-cute-6484941/<br><br>https://pixabay.com/vectors/child-costume-bee-character-8320341/<br><br>https://pixabay.com/vectors/guy-anime-cartoon-chibi-character-7330758/<br><br>https://pixabay.com/vectors/girl-anime-chibi-cartoon-character-7346667/<br><br>https://unsplash.com/photos/white-and-blue-cat-figurine-u3ZUSIH_eis<br><br>https://unsplash.com/photos/sock-monkey-plush-toy-on-brown-panel-5INN0oj12u4<br><br>https://pixabay.com/illustrations/karate-fighter-cartoon-character-8537724/<br><br>https://pixabay.com/illustrations/ai-generated-giraffe-doctor-8647702/<br><br>https://pixabay.com/illustrations/ai-generated-skull-character-8124354/<br><br>https://unsplash.com/photos/a-red-robot-is-standing-on-a-pink-backgroundunt3066GV-E<br><br>https://pixabay.com/illustrations/cartoon-dinosaur-dragon-animal-8539364/<br><br>https://pixabay.com/illustrations/man-book-read-hanfu-chinese-hanfu-7364886/|

|Reference|URL<br><br>|
|---|---|
|[Figure 442]<br><br>[Figure 443]<br><br>[Figure 444]<br><br>[Figure 445]<br><br>[Figure 446]<br><br>[Figure 447]<br><br>[Figure 448]<br><br>[Figure 449]<br><br>[Figure 450]<br><br>[Figure 451]<br><br>[Figure 452]<br><br>[Figure 453]<br><br>[Figure 454]<br><br>[Figure 455]<br><br>[Figure 456]<br><br>[Figure 457]<br><br>[Figure 458]<br><br>[Figure 459]|https://pixabay.com/vectors/muslim-hijab-child-cartoon-doodle-7747745/<br><br>https://pixabay.com/illustrations/tambourine-musician-woman-character-9073083/<br><br>https://pixabay.com/illustrations/ai-generated-man-agent-character-9050849/<br><br>https://pixabay.com/illustrations/ai-generated-superhero-hero-heroine-7977051/<br><br>https://unsplash.com/photos/a-woman-in-a-tan-jacket-and-tan-pants-QVyAUDUOlMw<br><br>https://unsplash.com/photos/a-woman-in-a-yellow-shirt-and-black-pantsrdHrrFA1KKg<br><br>https://pixabay.com/vectors/fashion-boy-cartoon-spring-summer-8515751/<br><br>https://pixabay.com/illustrations/woman-girl-fashion-model-female-8859569/<br><br>https://pixabay.com/illustrations/woman-cartoon-character-anime-8926994/<br><br>https://pixabay.com/photos/apple-red-delicious-fruit-vitamins-256268/<br><br>tps://pixabay.com/photos/apple-food-fresh-fruit-green-1239300/<br><br>https://pixabay.com/illustrations/fox-animal-wildlife-wild-mammal-9267914/<br><br>https://pixabay.com/illustrations/christmas-deer-animal-rudolph-8380345/<br><br>https://pixabay.com/illustrations/ai-generated-man-portrait-7953120/<br><br>https://pixabay.com/illustrations/created-by-ai-hedgehog-cartoon-8635844/<br><br>https://pixabay.com/vectors/dragon-creature-baby-dragon-8480029/<br><br>https://pixabay.com/vectors/boy-cartoon-fashion-chibi-kawaii-8515729/<br><br>https://pixabay.com/vectors/blonde-boy-cartoon-character-comic-1300066/|

