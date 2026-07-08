## UniPortrait: A Unified Framework for Identity-Preserving Single- and Multi-Human Image Personalization

Junjie He, Yifeng Geng, and Liefeng Bo Institute for Intelligent Computing, Alibaba Group

{hejunjie.hjj, cangyu.gyf, liefeng.bo}@alibaba-inc.com

Text-to-Single-ID Personalization Text-to-Multi-ID Personalization

Stylized Portrait Synthesis

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

# arXiv:2408.05939v2[cs.CV]6Sep2024

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

“Blonde with hair beaten into loose waves. Tattoos on both arms and multiple bracelets on each wrist. Green lace blouse with high-waisted jeans. Yellow walls.”

[Figure 23]

[Figure 24]

[Figure 25]

“Three men in a coffee shop.”

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

“The man was sitting at a table with a laptop on it. With one hand over his head, he looked at the screen with a worried expression on his face.”

[Figure 36]

Identity-Preserving Style Transfer With ControlNet

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

“Two women, retro comic style artwork, comic book cover, symmetrical, vibrant.”

“An old woman, white hair and a wrinkled face, a serious expression. Wearing a black coat and a straw hat, she stood in a snowy field with a white sky background.”

Figure 1. Example generations from UniPortrait. Our method customizes single- and multi-ID images in a unified manner, providing high-fidelity identity preservation, extensive facial editability, free-form text description, and no requirement for a predetermined layout.

### Abstract

bility with existing generative control tools. The project page is at https://aigcdesigngroup.github. io/UniPortrait-Page/.

This paper presents UniPortrait, an innovative human image personalization framework that unifies single- and multi-ID customization with high face fidelity, extensive facial editability, free-form input description, and diverse layout generation. UniPortrait consists of only two plug-andplay modules: an ID embedding module and an ID routing module. The ID embedding module extracts versatile editable facial features with a decoupling strategy for each ID and embeds them into the context space of diffusion models. The ID routing module then combines and distributes these embeddings adaptively to their respective regions within the synthesized image, achieving the customization of single and multiple IDs. With a carefully designed two-stage training scheme, UniPortrait achieves superior performance in both single- and multi-ID customization. Quantitative and qualitative experiments demonstrate the advantages of our method over existing approaches as well as its good scalability, e.g., the universal compati-

### 1. Introduction

Recent advances in diffusion models [9, 18] have revolutionized the field of text-to-image synthesis, paving the way for a plethora of image customization tasks [27, 31, 40, 61, 62]. Among these, human image personalization has emerged as a key area of focus for its enormous potential in applications such as AI portrait photos, image animation, and virtual try-ons. The objective of this task is to generate images that maintain a consistent face identity (ID) with reference face images while adhering to additional prompts.

Early human image customization methods [11, 19, 49] rely on test-time fine-tuning, with each new identity roughly taking dozens of minutes or even hours to achieve satisfactory results. Recent works [34, 60, 65, 67] resort to

tuning-free setups. They employ a separate trained encoder to encode face ID information into one or several tokens and inject them into the generation process, resembling text conditioning. Thanks to the elimination of training requirements, these tuning-free methods have achieved impressive efficiency when facing new identities. However, they still suffer from two limitations. First, they struggle to preserve facial shape and texture details. This is because the features they use are either losing the spatial information (e.g., the global embedding of the encoder [60, 66]) or not expert at the face domain (e.g. the features from CLIP [45] image encoder [34, 64]). Second, these encoder-based methods often tend towards mere replication of the reference face image, presenting difficulties in facial control and editing. We attribute this problem to the current methods’ insufficient disentanglement between intrinsic face identity and face-relevant yet identity-irrelevant representations. As a result, the model overfits the spurious facial details.

Besides, previous human customization methods primarily focus on single-ID personalization and struggle when personalizing multi-ID images. The main challenge they encountered is identity blending [64], in which a same generated face attends to multiple ID features. Some multiconcept customization methods [2, 23, 29, 64] integrate ID information into text embeddings and use the structure of text to distinguish different subjects. Nonetheless, these techniques necessitate a one-to-one mapping between identities and corresponding text tokens, preventing the use of free-form text prompts for subjects like plural phrases “two women”. Moreover, such direct fusion may compromise the integrity of both identity representation and text semantics because of the distinct nature of visual and textual signals. Another line of research [7, 15, 25, 26, 30, 38] introduces manually designed masks to separate the different IDs’ information. Despite their effectiveness in avoiding ID blending, they require specifying the face location before the generation, which limits the diversity of generated images.

To address the aforementioned issues, we introduce UniPortrait, a solution for unified single- and multi-human image personalization with high-fidelity identity preservation, substantial facial editability, free-form input description, and diverse layout generation (see Fig. 1). UniPortrait consists of only two plug-and-play modules: an ID embedding module and an ID routing module. The ID embedding module extracts versatile high-fidelity facial features for each ID and embeds them into the context space of diffusion models. The ID routing module then combines and distributes these embeddings adaptively to their respective regions within the synthesized image, achieving the customization of single and multiple IDs without the prompt and layout restrictions.

More concretely, instead of the last global feature, we utilize the discriminative features with spatial structure from the penultimate layer of the face recognition backbone

as the base intrinsic ID features in the ID embedding module. To enhance the face fidelity, we further employ CLIP local features following previous works [63, 67] but additionally incorporate shallow features that are empirically low-level and more identity-relevant from the face backbone as the face structure representations. Note that contrary to intrinsic ID features, these face structure representations lack the disentanglement and may couple with other identity-irrelevant information such as gaze, pose, or even face-irrelevant features like lighting. To prevent the model from overfitting these spurious facial details, we explicitly decouple the face structure features from the intrinsic ID features by emphasizing a strong dropping regularization, i.e., DropToken [1] & DropPath [20] on the face structure branch. This forced disentangling strategy makes the model more reliant on the intrinsic ID representation, which in turn, affords us increased flexibility in achieving an optimal balance between ID similarity and facial editability.

In the ID routing module, to prevent feature mixing between IDs, we propose a routing network to adaptively route and assign a unique ID to each potential face area. Specifically, we predict a discrete probability distribution at each spatial location within the cross-attention layer and, during the forward, select the best-matched top-1 ID embedding to engage in the attention of that specific location. To ensure that all ID conditions are routed and only routed to one target face area, we further introduce a routing regularization loss to assist the router learning. Thanks to the adaptiveness of the ID routing module, we allow UniPortrait to customize multi-ID images without any layout predetermination and prompt format restriction, unlocking the creativity of generative models and freedom of prompt designs.

The overall training process of our framework is curated into two stages: the single-ID training stage and the multiID fine-tuning stage. The former trains the ID embedding module and the latter specializes in the ID routing module. After completing two-stage training, our model can be used for either single-ID customization or multi-ID personalization. Extensive experiments demonstrate our method’s advantages over existing approaches and its good scalability, e.g., the universal compatibility with existing generative control tools such as ControlNet [69] and IP-Adapter [67].

The contributions of this paper are summarized as below:

- • We propose UniPortrait, an innovative human image personalization framework that unifies single- and multi-ID customization with high face fidelity and controllability;
- • We propose a novel ID embedding module with a decoupling strategy, which embeds detailed face identity information while maintaining good editability.
- • We introduce the ID routing mechanism, which addresses the identity blending issue in multi-ID customization yet without compromising each identity integrity, generated image diversity, and prompt design flexibility.

### 2. Related Work

Text-to-image generation. The development of diffusion models [9, 18, 43, 56, 58] has significantly advanced recent progress in text-conditioned image generation [5, 10, 14, 24, 44, 46, 47, 50, 51]. State-of-the-art text-to-image models [4, 43, 50] have been able to generate images precisely following user-provided prompts. However, generating images based solely on text cannot produce user-specific concepts, such as their beloved pets or personal items. Consequently, recent research has introduced personalized generation tasks [12, 13, 29, 38, 48, 62]. Among these, identitypreserving human image customization has emerged as a hot and popular topic due to its broad applications.

Single-ID personalization. Early works like FaceStudio [66] and InstantID [60] utilize the global feature of the face backbone as the human ID condition. Due to the loss of spatial information, these methods struggle to capture the intricate details of facial shape and texture. IPAdapter-FaceID-Plus [67] and Infinite-ID [63] introduce local features of CLIP image encoder to enhance face structure representation. Despite their improvements in ID similarity, these methods suffer from subject overfitting due to the insufficient disentanglement between face identity and identity-irrelevant representations. FlashFace [70] exploits ReferenceNet [68] but relies on constructing a large dataset containing multiple images of a single ID. CapHuman [35] customizes human images based on 3D reconstruction [33], with the render parameters, such as head pose and face position, needing to be set in advance. Most of these methods [34, 42, 60, 66, 67, 70] focus on single ID customization and encounter difficulties when generalizing to multi-IDs.

Multi-ID personalization. One challenge in customizing multi-ID images is identity blending [64]. A common solution [2, 23, 28] to this problem involves encoding and integrating all subjects’ information into text embeddings, using the text’s structure to differentiate different IDs. The drawback of this approach lies in its restriction of prompt formats, e.g., we should explicitly specify the subject word and restrict its representation solely to the singular form. Moreover, such integration can compromise identity fidelity and prompt consistency due to the distinct nature of visual and textual signals. Another line of research [15, 25, 26, 30, 38] introduces predefined layout masks to separate the information of different IDs. These methods effectively prevent identity blending but limit the diversity of generated images. FastComposer [64] utilizes localized cross-attention maps to customize multiple IDs without layout constraints. However, it necessitates a specialization of the diffusion model, making it impractical to employ other foundation text-to-image models or control tools from the community. MoA [41] draws on the FastComposer principle and deploys a mixed-attention module to disentangle the prior and personalization branches. De-

spite improving usability, the entanglement of text and image embeddings still results in a less-than-ideal balance between face fidelity and prompt consistency.

Our proposed UniPortrait exploits an independent, versatile ID embedding module with a decoupling strategy to achieve a pleasing trade-off between ID fidelity and prompt consistency, and meanwhile employs an ID routing module to unify single-ID and multi-ID customization by which we do not specify the ID locations in advance or impose constraints on subject text descriptions.

### 3. Methods

This section introduces UniPortrait, an innovative approach to unified single-ID and multi-ID character generation. We first briefly review the background of Stable Diffusion in Sec. 3.1 and then elaborate on the details of two key modules of UniPortrait in Sec. 3.2 and Sec. 3.3. Finally, we illustrate the training scheme of UniPortrait in Sec. 3.4. The overview of our framework is shown in Fig. 2.

#### 3.1. Preliminary

In this paper, the underlying model used for text-to-image synthesis is Stable Diffusion [47]. It takes a text prompt P as input and generates the corresponding image x0. Stable diffusion comprises three main modules: an autoencoder (E(·),D(·)), a CLIP text encoder τ(·), and a U-Net ϵθ(·). Typically, it is trained under the following diffusion loss:

0,P,ϵ∼N(0,1),t[∥ϵ − ϵθ(zt,t,τ(P))∥22] (1) where ϵ ∼ N(0,1) is the randomly sampled Gaussian noise, t is the time step, z0 = E(x0) is the latent representation of x0, and zt is calculated by zt = αtz0 + σtϵ with the coefficients αt and σt provided by the noise scheduler.

Ldiff = Ez

#### 3.2. ID embedding

The ID embedding module is specifically crafted to impart high-fidelity editable face ID information, thereby guiding the diffusion model to generate ID-consistent and -controllable images. Unlike most preceding approaches [60, 66] that harness the final global features of a face recognition backbone for face ID representation, we utilize features from the penultimate layer (prior to the fully connected layer). This adjustment aims to preserve an enhanced degree of spatial information pertaining to ID features. Since the face recognition backbone [8, 21, 36, 59] is commonly trained on a large dataset that contains millions of human IDs, its features are expected to be insensitive to ID-irrelevant facial information such as expression, pose, and gaze. In particular, they are insensitive to facial shape and texture details. This is achieved by only noting that the fluctuations in weight or age influence one’s appearance but do not change his or her identity. We term these recognition features the intrinsic ID features.

|[Figure 43]|
|---|

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Latent feature

···

···

$

| | | | |
|---|---|---|---|
| |!#$(&)|!#$(()|!#$())|

Learnable queries

ID Routing

ID Embedding

···

··· ··· ···

[Figure 48]

Router

[Figure 49]

Attention Block

|CLIP Image Encoder| |
|---|---|
| | |

1/2 ↓ 1/4 ↓ 1/8 ↑

Cross Attention

Face Recognition Backbone

Face Recognition Backbone

[Figure 50]

Stage Ⅰ

Cross Attention

Attention Block

1/16

1/16

[Figure 51]

| | |
|---|---|
| | |

DropPath

Cross Attention

!!

[Figure 52]

Stage Ⅱ

[Figure 53]

MLP

FFN

!"

[Figure 54]

DropToken

MLP

×#

[Figure 55]

[Figure 56]

[Figure 57]

!#$

[Figure 58]

[Figure 59]

“A woman smiling with her hair down and wearing a necklace.”

···

Cross Attention

[Figure 60]

Routing Regularization loss

[Figure 61]

LoRA

[Figure 62]

[Figure 63]

Trainable

[Figure 64]

|CLIP Text Encoder| |
|---|---|
| | |

[Figure 65]

LoRA

Shared

“Three young girls posing together on a swing, dressed elegantly.”

Diffusion loss

|CLIP Text Encoder| |
|---|---|
| | |

Cross Attention

Only in the training phrase

Diffusion loss

Cross Attention

Figure 2. Overview of UniPortrait framework. Our proposed UniPortrait consists of two plug-and-play modules: an ID embedding module and an ID routing module. The ID embedding module extracts versatile editable facial features with a decoupling strategy for each ID (Sec. 3.2), and the ID routing module combines and distributes these embeddings to their respective locations adaptively without the intervention for prompts and layouts (Sec. 3.3). The entire training process of the framework is curated into two stages, i.e., the single-ID training stage and the multi-ID fine-tuning stage (Sec. 3.4).

reliant on intrinsic ID characteristics. Such a strategy allows for a more versatile trade-off between ID similarity and editability, catering to the varied requirements of users seeking identity-preserving portrait generation.

However, there often arises a user demand to personalize ID images to closely match the appearance of a given face reference, i.e., maintaining consistent facial shape and texture details beyond the intrinsic ID characteristics. In response, several prior studies [6, 63, 67] have leveraged local features derived from the CLIP image encoder as face structure conditions. Despite the enhanced face similarity, utilizing CLIP local features poses two significant challenges. Firstly, CLIP is trained on generically weakly aligned image-text pairs, rendering its features less discriminative concerning face identities and predominantly semantic. Secondly, due to the lack of disentanglement, these features may couple with other ID-irrelevant facial information or even face-irrelevant representations like background lighting. Given the typically scarce and non-diverse nature of personalization training data—in which the training reference and target faces often come from the same or similar images—these irrelevant features risk leading to model overfitting on non-essential facial details, which, in turn, complicates the process of facial control and editing.

Specifically, as depicted in the blue section of Fig. 2, we first flatten and apply a Multilayer Perceptron (MLP) to the penultimate layer’s features of the face recognition model to obtain the intrinsic ID features Fr ∈ Rm

r×dr, where mr represents the feature length and dr represents the feature dimension. We then interpolate the shallow features, i.e., the 1/2, 1/4, and 1/8 feature maps from the face backbone and concatenate them with CLIP local features to derive the face structure features Fs ∈ Rm

s×ds through another MLP. Next, we introduce a l layer Q-Former [31, 32] with m learnable queries to aggregate Fr and Fs. Each layer of the Q-Former comprises two attention blocks and one Feed-Forward Network (FFN), with the attention blocks respectively attending to the intrinsic ID information and face structure representations. In the input and output of the second attention block, we further introduce DropToken [1] and DropPath [20] as means of decoupling face structure from intrinsic ID representation. The final output from the Q-Former, denoted as Fid ∈ Rm×d, is then employed as the ID embedding and aligned into the context space of U-Net. Here, we use decoupled cross attention [67] to inject the ID information into U-Net.

In order to solve these problems, we initially integrate the shallow features from the face recognition model to augment the structural representation of the face. Subsequently, we apply a strong dropping regularization to the structure feature branch to decouple it from the intrinsic ID branch. The shallow features of the backbone are empirically lowlevel, containing more texture details, and they are more IDrelevant, facilitating us to generate higher-fidelity portraits. The dropping regularization on the facial structure branch maintains the independence of intrinsic ID features and face structure features, simultaneously making the model more

Single-ID multi-reference embedding. Owing to the scalability of the designed architecture, our ID embedding module can be seamlessly adapted to cater to the context of embedding multiple references for a single ID. We only need to extract the intrinsic ID features Fr(j) and face structure

features Fs(j) for each reference (j is the reference index) and concatenate them as the new intrinsic ID and face structure features. Experiments show that although our method is trained on a single reference image, it can be effectively extended to accommodate multiple reference images and achieve improved personalization results. See experiments in Appendix B for details.

Identity interpolation. We extract the ID embedding Fid(n) for each identity (n is the identity index) and perform linear interpolation on them to achieve ID interpolation. By identity interpolation, we can create a meaningful semantic transition between different IDs or even different states of the same ID. See the experiment in Sec. 4.4 for details.

#### 3.3. ID Routing

Through the ID embedding module, we can obtain versatile editable embeddings for a single ID. For multi-ID scenarios, we leverage the ID embedding module to embed each ID information into the context space. Notably, these embedded ID representations are position-independent, as we have not imposed any positional constraints on them. To avoid identity blending, previous methods have either integrated ID embeddings into text embeddings [2, 23, 28] or employed manually crafted layout masks [25, 30, 38] to segregate the information of different IDs. The former requires adherence to a specific format of text descriptions (e.g., the singular phrase for the subject) and may potentially degrade the fidelity of both textual and identity representations; the latter constrains the diversity of the resultant imagery. In this work, we introduce a position-wise ID routing module integrated within each cross-attention layer to adaptively route and assign a unique ID to each potential face area in the latent features, thereby effectively mitigating the problem of identity blending.

Specifically, assume there are N distinct IDs, with each ID embedding denoted as Fid(n), n = 1,··· ,N. For every spatial position (u,v) within the latent feature Z ∈ Rc×h×w, we route and assign a unique identity k∗ (1 ≤ k∗ ≤ N) to it with k∗ determined as:

ψ(Z,Fid(:),(u,v))k, (2)

k∗ = arg max

k

where ψ represents the router and it outputs a Ndimensional discrete probability distribution as:

ψ(Z,Fid(:),(u,v)) = Softmax([θ(Z)u,v ∗ ϕ(Waggr ∗ Fid(n))]Nn=1).

(3)

Here Waggr ∈ Rm is a weight that aggregates m ID embedding tokens into a singular token, θ and ϕ are two small networks, and ∗ is the matrix multiplication operator. Subsequently, the ID information, as pinpointed by each spatial location, is integrated into the latent feature of that location

via the cross-attention mechanism, analogous to the single ID information injection outlined in Sec. 3.2. In practice, we instantiate θ and ϕ as two 2-layer MLPs for simplicity.

The idea behind the ID routing is that each face in an image is associated with at most one ID feature. By confining each position to cross-attend to solely one ID information, the blending problem between IDs is efficaciously circumvented.

However, directly applying Eq. 2 presents two potential concerns. Primarily, it does not guarantee that all IDs will be routed. Secondly, the same ID still has the possibility of being leaked to multiple target faces by attending to their partial areas. Additionally, the Eq. 2 is non-differentiable. In order to alleviate these issues, we propose the incorporation of a routing regularization loss and leverage the Gumbel softmax trick [22] during the training phase. These measures facilitate router learning, enhancing its capability to effectively manage and distribute ID representations.

Routing regularization loss. Concretely, given a target image that contains N distinct IDs during the training phase, we first detect bounding boxes for all faces on the image and convert them into binary masks, where 1 represents the face area and 0 represents the other area. In this way, we obtain N face region masks. Then, the routing regularization loss is calculated by the L2 loss between the router’s outputs and these face region masks as follows:

1 N ∥Wroute ⊙ (ψ(Z,Fid(:)) − M)∥22 (4)

Lroute = λ ·

where λ is the weight of the loss function, ⊙ denotes the element-wise multiplication, ψ(Z,Fid(:)) ∈ RN×h×w represents the routing outputs across all positions, M ∈ RN×h×w is the stack of N face region masks, and Wroute ∈ Rh×w is the union of N face region masks which means that we only apply regularization loss to the face regions in the image. By emphasizing the routing targets across all face region areas, on the one hand, we encourage all IDs to be routed, and on the other hand, we prompt the ID router that each ID can merely be routed to at most one face area.

Gumbel softmax trick. To ensure the gradients of the routing module can be properly backpropagated during training, we introduce the Gumbel-softmax trick [22]. Specifically, during training, we add Gumbel noise to the output logits of the router to reparameterize the routing sampling process. During inference, we normally select the best-determined top-1 identity from the router for forward propagation.

Finally, it is noteworthy that in the case of a single ID, the router becomes trivial, and the routing-based multi-ID generation degenerates into common single-ID generation.

#### 3.4. Training

The entire training process of UniPortrait is curated into two stages: the single-ID training stage and the multi-ID

IP-Adapter-

Reference

FastComposer PhotoMaker InstantID FlashFace FaceID-PlusV2

UniPortrait (Ours)

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

“A grizzled detective, fedora casting a shadow over his square jaw, a cigar dangling from his lips, his trench coat evocative of film noir, in a rainy alley.”

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

“A woman eating a healthy parfait and yogurt, cartoon.”

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

“A man clad in cyberpunk fashion: neon accents, reflective sunglasses, and a leather jacket with glowing circuit patterns. He stands stoically amidst a soaked cityscape.”

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

“An 8-bit pixel art representation of a skateboarder, constructed

from carefully arranged blocks of vibrant pixels. The background is snowy

mountains and a blue sky.”

###### Figure 3. Qualitative comparison of different methods on single-ID image customization.

|Method|Arch.<br><br>MultiID?<br><br>Face Sim. ↑ (%)<br><br>CLIP-T ↑ (%)<br><br>FID ↓<br><br>LAIONAes ↑<br><br>|
|---|---|
|SD v1-5 [47] PortraitBooth† [42] IP-Adapter∗ [67] FlashFace [70] FastComposer [64] UniPortrait (ours) PhotoMaker [34] InstantID [60]<br><br>|SD15 - 4.4 27.7 - 6.60<br><br>SD15 65.7 24.5 - -<br><br>SD15 68.4 24.7 139.5 6.43 SD15 72.7 25.8 141.7 5.85 SD15 50.8 24.1 134.5 6.17 SD15 71.1 26.1 123.4 6.42 SDXL 41.7 26.9 136.1 6.01 SDXL 77.9 24.1 163.5 6.15<br><br>|

nessed from the target image, an approach adopted to enhance learning of the textural and structural details of the face. Throughout the training process, we employ a dropping regularization on the face structure branch, with probabilities delineated as follows: complete branch dropout occurs with a 0.33 probability; retaining the branch while randomly dropping face structure tokens at a 0.33 probability; and complete preservation of the face structure branch at a 0.34 probability. To extract facial information more comprehensively, Low-Rank Adaptation (LoRA [19]) has been appended to the U-Net architecture. Only the parameters within the ID embedding module and the U-Net’s LoRA are subjected to training in this stage. The training loss is aligned with the conventional diffusion loss, as shown in Eq. 1.

- Table 1. Quantitative comparison with different single-ID customization methods. † denotes the results reported in its original paper. IP-Adapter∗ corresponds to the IP-Adapter-FaceID-PlusV2 variant. “Multi-ID” indicates the capability of the method to customize multi-ID images using only text and ID conditions. The best result is shown in bold, and the second best is underlined.

Stage II: multiple ID fine-tuning. After completing the Stage I training, we introduce the ID routing module. We fix all the parameters in the ID embedding module and only fine-tune the parameters of the ID router and LoRA module, with the learning rate of LoRA module decaying by 0.1. The loss function in the second stage encompasses the original diffusion loss (Eq. 1) and the routing regularization loss (Eq. 4). Herein, the balancing parameter λ is set to 0.1.

fine-tuning stage. After completing this two-stage training, UniPortrait can be used for either single-ID customization or multi-ID personalization.

Stage I: single ID training. In this phase, we only introduce the ID embedding module; the training regimen is limited to images that feature a singular ID, as is depicted on the left side of Fig. 2. We first crop and align the face region of an image to serve as the input for the ID embedding module. If the face has an associated ID label, e.g., an image sourced from the CelebA dataset [37], another cropped and aligned face image of the identical ID is sampled with a 0.1 probability to act as the input for the intrinsic ID branch. Conversely, all inputs for the face structure branch are har-

### 4. Experiments

#### 4.1. Setup

Dataset. The dataset utilized in this work comprises four primary segments: (1) 240k single-ID images filtered from

Ref. FastComposer

UniPortrait (Ours)

Ref. FastComposer UniPortrait (Ours)

Ref. FastComposer

UniPortrait (Ours) Ref. FastComposer UniPortrait (Ours)

[Figure 94]

[Figure 95]

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

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

“A cinematic homage to classic film stars, male and female, in a romantic clasp, black and white, with 1940s Hollywood glamour.”

“Two men are engaged in a heated debate, .... gesturing emphatically, with a sense of urgency and intensity. In the style of pen and ink.”

“A bold and graphic silkscreen print featuring three characters, ...line work and vibrant color blocks, …sense of graphic design and visual impact.”

“Three generations of laughter, a grandpa with his grandsons engaged in a lively baking session, happy faces.”

- Figure 4. Qualitative comparison of different methods on multi-ID image customization. For compatibility with FastComposer, numerical plural expressions (e.g., “two men”) are converted into singular phrases linked by “and” (e.g., “a man and a man”).

Reference UniPortrait (Ours)

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

“A winter wonderland, four Popmart blind boxes, cartoon.”

[Figure 116]

[Figure 117]

[Figure 118]

“Paper cutout, with a couple in Ghibli animation style.”

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

“A photo of three friends laughing.”

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

“A couple eating a candlelit dinner at a romantic restaurant.”

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

- Figure 5. Additional examples of multi-ID customization. UniPortrait is capable of customizing multi-ID images using free-form prompts and generating diverse layouts.

|Method|Arch.<br><br>Face Sim. ↑ (%)<br><br>CLIP-T ↑ (%)<br><br>FID ↓<br><br>LAIONAes ↑<br><br>|
|---|---|
|SD v1-5 [47]<br><br>FastComposer [64] UniPortrait (ours)<br><br>|SD15 1.6 28.9 - 6.25 SD15 38.0 25.5 156.6 5.81 SD15 67.3 27.4 139.5 5.89<br><br>|

Table 2. Quantitative comparison with different multi-ID customization methods.

16 leanable queries. The rank of LoRA in U-Net is set to 128. All experiments are conducted on 8 V100 GPUs using AdamW [39] optimizer with a batch size of 128 and a learning rate of 1e-5. The first stage trains 300k iterations, and the second stage trains 150k iterations. To facilitate classifier-free guidance sampling [17], we train the model without the face conditions on 5% images. During inference, we use 20-step DDIM [57] sampling with a classifierfree guidance scale of 7.5, and to achieve more realistic image generation, we employ the Realistic Vision V4.0 model from huggingface following the previous work [67].

Evaluation Metrics. We assess the image generation quality in terms of identity preservation, prompt consistency, FID [16], and LAION-Aesthetics (LAION-Aes) scores [53]. For identity preservation and prompt consistency, we follow the evaluation protocol established by FastComposer [64]. Specifically, identity preservation is quantified by calculating pairwise face similarities between reference and generated faces with FaceNet [52]. For multiidentity generation, we detect all faces in generated images and use a greedy matching procedure between generated and reference faces. The lowest similarity score across all faces measures overall identity preservation. Prompt consistency is assessed utilizing the average CLIP-L/14 imagetext similarity.

LAION [54, 55]; (2) 100k single-ID portraits filtered from CelebA database [37]; (3) 160k high-quality single-ID images collected from the Internet; (4) 120k high-quality multi-ID portraits filtered from LAION. The first three subsets are utilized for Stage I training, while the last subset is used for Stage II training. Images originating from CeleA and those obtained from the Internet are captioned using Qwen-VL [3], whereas images from LAION are presented with their original text captions. It is noteworthy that within all these data, only the CeleA images are endowed with ID annotations.

#### 4.2. Results

Single-ID personalization. We first evaluate the performance of single-ID customization. We follow FastComposer [64] and use 15 identities from the CelebA dataset [37], which have been deliberately excluded from our training dataset, with 40 unique text prompts assigned to each subject for assessment. These text prompts cover a wide range of scenarios, such as re-contextualization, stylization, accessorization, and various actions. For a fair comparison, all methods accept a single reference face image

Implementation details. Our training starts with the StableDiffusion v1-5 [47] model. The face recognition backbone we used is CurricularFace [21]. For CLIP image encoder, we use OpenCLIP’s clip-vit-huge-patch14. The Q-Former in ID embedding module has 6 layers and

|Intrisic ID branch face backbone global feat.<br><br>face backbone local feat.<br><br>|Face structure branch CLIP local feat<br><br>face backbone shallow feat.<br><br>drop token & path|Face Sim. ↑ (%) CLIP-T ↑ (%) FID ↓ LAION-Aes ↑|
|---|---|---|
| | |45.5 27.2 115.2 6.47 58.3 26.8 120.3 6.50 61.4 26.1 124.7 6.38 64.5 25.9 128.1 6.46 68.4 26.1 122.6 6.30 71.1 26.1 123.4 6.42|

Table 3. Ablation studies for components in ID embedding module.

[Figure 135]

[Figure 136]

| |Face Sim. ↑ (%)<br><br>CLIP-T ↑ (%)<br><br>FID ↓<br><br>LAIONAes ↑<br><br>|
|---|---|
|w/o routing regularization loss w/ routing regularization loss<br><br>|59.1 27.5 139.8 5.89 67.3 27.4 139.5 5.89|

[Figure 137]

[Figure 138]

“Two men sitting in a park together.”

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

w/o routing regularization loss

Table 4. Ablation study for the routing regularization loss.

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

w/ routing regularization loss

and generate 4 images each once. The quantitative results are shown in Table 1. Our approach manifests a commendable balance between identity preservation and prompt consistency, simultaneously achieving the lowest FID score and the second-highest score in LAION-Aesthetics, which markedly surpasses the performance metrics of PortraitBooth [42], IP-Adapter-FaceID-PlusV2 [67], and FastComposer [64]. Notably, InstantID [60] records the highest similarity in face identity; however, its scores for prompt consistency and FID are relatively inferior, a limitation attributed to its requisite for fixed positions of facial landmarks. PhotoMaker [34] showcases notable results in prompt consistency, albeit with mediocre outcomes in facial similarity. While FlashFace [70] manages to accomplish a relative trade-off between face similarity and prompt consistency, the inferior FID and LAION-Aes values indicate its unsatisfactory behavior in the quality and diversity of generated images. It is crucial to note that among all the evaluated methods, only FastComposer and our approach facilitate the personalized generation of images featuring multiple individuals naively. Fig. 3 presents a visual comparison of the qualitative outcomes derived from utilizing different methods to respond to a series of prompts for single ID personalization, where the qualitative analysis aligns with the conclusions made from quantitative metrics.

! = 20 ! = 19 ! = 17 ! = 13 ! = 1

Figure 6. The effect of the routing regularization loss.

exhibits improved fidelity to the text prompts, enabling the direct application of text for the stylized customization of images featuring multiple persons. In addition, thanks to the ID routing mechanism, our approach supports greater flexibility in prompt input. This is particularly advantageous for inputs comprising plural phrases, which, in the case of FastComposer, necessitate conversion into singular phrases interconnected by “and”. More visual examples illustrating the versatility of our method in rendering multi-ID images are presented in Fig. 5, further substantiating the qualitative enhancements our approach brings to multi-ID image customization.

#### 4.3. Ablation study

Components in ID embedding module. Table 3 presents an assessment of the efficacy of the various constituents within the ID embedding module. Using local features from the penultimate layer of the face recognition model instead of its last global feature contributes to a marked enhancement in ID similarity. Introducing face structure features further bolsters the ID similarity, with this effect appearing particularly pronounced when incorporating shallow features from the face backbone. Nevertheless, it is observed that the integration of facial structure features incurs a decrease in both the diversity of generated images, i.e., FID, and their consistency with the associated text prompts. This reduction can be mitigated by the application of DropToken and DropPath regularizations within the face structure branch. Simultaneously, these regularizations help mitigate model overdependence on inaccurate facial details, thereby optimally reinforcing ID similarity. Despite these adjustments, it must be acknowledged that the inclusion of the face structure branch inherently entails a compromise in

Multi-ID personalization. We further assess the performance of multi-ID image generation. We also use the test benchmark from FastComposer [64], which contains the 15 IDs from CelebA database described above and 21 additionally curated test prompts. These 15 IDs are strategically paired, resulting in a total of 105 multi-ID combinations.

- Table 2 shows the quantitative comparison between UniPortrait and FastComposer. Our method outperforms FastComposer on all metrics, manifesting enhanced identity preservation and prompt consistency alongside augmented quality and aesthetics in the generated images. A qualitative analysis is shown in Fig. 4. UniPortrait retains the distinctive attributes of different subjects. Concurrently, UniPortrait

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

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Image Prompt

“Blue eyes, red lips, blush.” “A child.” “A man.”

(a) Face editing, with text prompts (b) Face swap, with ControlNet-Inpainting (c) ID-preserving style transfer, with IP-Adapter and ControlNet-Edge

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

Image Prompt

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 171]

[Figure 172]

| | |
|---|---|
| | |

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

Community Checkpoint ”ToonYou”

“The woman on the left has short brown hair and is wearing a black shirt. The woman on the right has blonde hair styled back and is also dressed in a black shirt. Her head is resting

| | |
|---|---|
| | |

“Best quality, masterpiece, two women and a man sitting on a stone bench.” on the shoulder of the other woman. “

(d) Stylized generation, with IP-Adapter or Community Checkpoint (e) Pose Control, with ControlNet-Pose

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

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

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

ID1 ! = 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0 ID2

(f) ID interpolation, with the text prompt “A face.” and ID embedding interpolation coefficient !. "!" = !"!"($) + (1 − !)"!"(&)

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

“Battle over the last pancake.” “Navigate school halls in panic.” “Laughing under pouring rain.” “Sit together at the desk, brows furrowed in concentration.”

“Chase fireflies, faces illuminated by jarred stars.”

“Sleep quietly on the bed.”

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

“Childhood Glee, instant friends.” “Teenage rivalry, swords clash.” “Young love blossoms, shy smiles.” “Battle worn comrades, fierce looks.” “Midlife reflection, quiet tea house.”

“Serenity in age, lifetime of love.”

(g) Story generation, with the style LoRA “disney-pixar-cartoon-typeb” (the first row) and “chosen-chinese-stylensfw-hentai” (the second row) from the community

Figure 7. Diverse applications of UniPortrait.

prompt consistency to some extent. For more analysis on the branch of face structure, please refer to the Appendix A. Routing regularization loss. Table 4 provides a verification of the efficacy of the routing regularization loss. The results suggest that this approach can substantially enhance ID similarity while concurrently maintaining prompt consistency in multi-ID customization. Fig. 6 depicts the average routing maps derived from all U-Net cross-attention layers at different diffusion steps. It is observable that the employment of routing regularization loss results in a more focused routing result, indicative of improved segregation of information pertaining to distinct IDs. A more detailed layer-by-layer routing graphs are available for review in the Appendix C.

these, face attribute modification stands out – this includes alterations in age, gender, and specific facial characteristics, as shown in Fig. 7(a). Additionally, the adaptable, plugand-play nature of UniPortrait ensures compatibility with a range of existing community-developed tools, such as ControlNet [69], LoRA [19], and IP-Adapter [67]. This integration facilitates the creation of conditionally controllable IDpreserving generations. Examples of such applications are shown in Fig. 7(b-e). Going a step further, UniPortrait’s capacity for identity interpolation among different characters is investigated, showcasing its adeptness in smoothly blending features from multiple identities, as shown in Fig. 7(f). Besides these, we also demonstrate the potential of UniPortrait to generate stories with consistent IDs, as exemplified in Fig. 7(g). For more applications and examples, please refer to the Appendix D.

#### 4.4. Application

The superior performance of UniPortrait in aligning IDs, maintaining prompt consistency, as well as enhancing the diversity and quality of generated images, paves the way for a plethora of potential downstream applications. Among

### 5. Conclusion

We introduce UniPortrait, a model developed for the unified customization of single- and multi-ID images. Uni-

Portrait incorporates an advanced ID embedding module that ensures high-fidelity and editable identity embeddings. Furthermore, a modular, plug-and-play ID routing component has been integrated to address the challenge of identity blendings during the multi-ID generation process. The empirical results demonstrate that UniPortrait outperforms existing methods by delivering a synthesis that is not only of high quality and diversity but also offers robust editability and strong identity fidelity. We hope that our UniPortrait will serve as a new baseline work within this domain, providing a benchmark that can be followed, replicated, and optimized by all research institutions.

Limitations. Considering that the routing decisions of the router are exclusively based on human ID information, our ID routing module is currently incapable of customizing attributes unrelated to face identity, e.g., clothing and actions, for each ID in the multi-ID generation. A possible solution is to feed the representations of all attributes of interest into the router to guide the ID routing, i.e., the attribute-binding ID routing. We leave it for future research.

Ethics considerations. The utilization of human image personalization technologies presents potential social risks, with the misuse of such technologies for deepfakes being a significant concern. In order to address these risks, the implementation of ethical guidelines and responsible usage practices is imperative. At present, the synthesized outcomes display certain visual anomalies that may facilitate the identification of deepfakes.

### Acknowledgments

Most of the face images used in our experiments come from the Pexels, Unsplash, Pixabay, and Wikipedia websites. We thank the owners of these images for sharing their valuable assets. We also thank the StyleGAN2 authors for sharing their high-quality synthesized face images, which constitute another important part of the ID images we used.

### References

- [1] Hassan Akbari, Liangzhe Yuan, Rui Qian, Wei-Hong Chuang, Shih-Fu Chang, Yin Cui, and Boqing Gong. Vatt: Transformers for multimodal self-supervised learning from raw video, audio and text. NeurIPS, 34:24206–24221, 2021. 2, 4
- [2] Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel CohenOr, and Dani Lischinski. Break-a-scene: Extracting multiple concepts from a single image. In SIGGRAPH Asia 2023 Conference Papers, pages 1–12, 2023. 2, 3, 5
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 7
- [4] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce

- Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023. 3
- [5] Shoufa Chen, Mengmeng Xu, Jiawei Ren, Yuren Cong, Sen He, Yanping Xie, Animesh Sinha, Ping Luo, Tao Xiang, and Juan-Manuel Perez-Rua. Gentron: Delving deep into diffusion transformers for image and video generation. arXiv preprint arXiv:2312.04557, 2023. 3
- [6] Siying Cui, Jia Guo, Xiang An, Jiankang Deng, Yongle Zhao, Xinyu Wei, and Ziyong Feng. Idadapter: Learning mixed features for tuning-free personalization of text-toimage models. In CVPR, pages 950–959, 2024. 4
- [7] Omer Dahary, Or Patashnik, Kfir Aberman, and Daniel Cohen-Or. Be yourself: Bounded attention for multi-subject text-to-image generation. arXiv preprint arXiv:2403.16990,

2024. 2

- [8] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In CVPR, pages 4690–4699, 2019. 3
- [9] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. NeurIPS, 34:8780–8794,

2021. 1, 3

- [10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 3
- [11] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 1
- [12] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In ICLR, 2023. 3
- [13] Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Encoder-based domain tuning for fast personalization of text-to-image models. ACM TOG, 42(4):1–13, 2023. 3
- [14] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020. 3
- [15] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized lowrank adaptation for multi-concept customization of diffusion models. NeurIPS, 36, 2024. 2, 3
- [16] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 30, 2017. 7
- [17] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 7
- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33:6840–6851, 2020. 1, 3

- [19] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In ICLR, 2022. 1, 6, 9
- [20] Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In ECCV, pages 646–661. Springer, 2016. 2, 4
- [21] Yuge Huang, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, and Feiyue Huang. Curricularface: adaptive curriculum learning loss for deep face recognition. In CVPR, pages 5901–5910, 2020. 3, 7
- [22] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144, 2016. 5
- [23] Sangwon Jang, Jaehyeong Jo, Kimin Lee, and Sung Ju Hwang. Identity decoupling for multi-subject personalization of text-to-image models. arXiv preprint arXiv:2404.04243, 2024. 2, 3, 5
- [24] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. CVPR, 2023. 3
- [25] Chanran Kim, Jeongin Lee, Shichang Joung, Bongmo Kim, and Yeul-Min Baek. Instantfamily: Masked attention for zero-shot multi-id image generation. arXiv preprint arXiv:2404.19427, 2024. 2, 3, 5
- [26] Zhe Kong, Yong Zhang, Tianyu Yang, Tao Wang, Kaihao Zhang, Bizhu Wu, Guanying Chen, Wei Liu, and Wenhan Luo. Omg: Occlusion-friendly personalized multiconcept generation in diffusion models. arXiv preprint arXiv:2403.10983, 2024. 2, 3
- [27] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In CVPR, 2023. 1
- [28] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. CVPR, 2023. 3, 5
- [29] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In CVPR, pages 1931–1941,

2023. 2, 3

- [30] Gihyun Kwon, Simon Jenni, Dingzeyu Li, Joon-Young Lee, Jong Chul Ye, and Fabian Caba Heilbron. Concept weaver: Enabling multi-concept fusion in text-to-image models. In CVPR, pages 8880–8889, 2024. 2, 3, 5
- [31] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pretrained subject representation for controllable text-to-image generation and editing. NeurIPS, 36, 2024. 1, 4
- [32] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 4
- [33] Tianye Li, Timo Bolkart, Michael J Black, Hao Li, and Javier Romero. Learning a model of facial shape and expression from 4d scans. ACM Trans. Graph., 36(6):194–1, 2017. 3
- [34] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, MingMing Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. In CVPR, pages 8640–8650, 2024. 1, 2, 3, 6, 8

- [35] Chao Liang, Fan Ma, Linchao Zhu, Yingying Deng, and Yi Yang. Caphuman: Capture your moments in parallel universes. In CVPR, pages 6400–6409, 2024. 3
- [36] Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj, and Le Song. Sphereface: Deep hypersphere embedding for face recognition. In CVPR, pages 212–220, 2017. 3
- [37] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In ICCV, pages 3730–3738, 2015. 6, 7
- [38] Zhiheng Liu, Yifei Zhang, Yujun Shen, Kecheng Zheng, Kai Zhu, Ruili Feng, Yu Liu, Deli Zhao, Jingren Zhou, and Yang Cao. Cones 2: Customizable image synthesis with multiple subjects. arXiv preprint arXiv:2305.19327, 2023. 2, 3, 5
- [39] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 7
- [40] Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subject-diffusion: Open domain personalized text-to-image generation without test-time fine-tuning. arXiv preprint arXiv:2307.11410, 2023. 1
- [41] Daniil Ostashev, Yuwei Fang, Sergey Tulyakov, Kfir Aberman, et al. Moa: Mixture-of-attention for subject-context disentanglement in personalized image generation. arXiv preprint arXiv:2404.11565, 2024. 3
- [42] Xu Peng, Junwei Zhu, Boyuan Jiang, Ying Tai, Donghao Luo, Jiangning Zhang, Wei Lin, Taisong Jin, Chengjie Wang, and Rongrong Ji. Portraitbooth: A versatile portrait model for fast identity-preserved personalization. In CVPR, pages 27080–27090, 2024. 3, 6, 8
- [43] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 3
- [44] Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015. 3
- [45] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763. PMLR, 2021. 2
- [46] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125,

2022. 3

- [47] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684– 10695, 2022. 3, 6, 7
- [48] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, pages 22500–22510, 2023. 3
- [49] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for

- fast personalization of text-to-image models. arXiv preprint arXiv:2307.06949, 2023. 1
- [50] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, pages 36479–36494,

2022. 3

- [51] Axel Sauer, Tero Karras, Samuli Laine, Andreas Geiger, and Timo Aila. Stylegan-t: Unlocking the power of gans for fast large-scale text-to-image synthesis. In ICML, pages 30105–

30118. PMLR, 2023. 3

- [52] Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. In CVPR, pages 815–823, 2015. 7
- [53] Christoph Schuhmann. Laion-aesthetics. https:// laion.ai/blog/laion-aesthetics/, 2022. 7
- [54] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 7
- [55] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. In NeurIPS, pages 25278–25294, 2022. 7
- [56] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, pages 2256– 2265, 2015. 3
- [57] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 7
- [58] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. NeurIPS, 32, 2019. 3
- [59] Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou, Zhifeng Li, and Wei Liu. Cosface: Large margin cosine loss for deep face recognition. In CVPR, pages 5265–5274, 2018. 3
- [60] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024. 1, 2, 3, 6, 8
- [61] Zhouxia Wang, Xintao Wang, Liangbin Xie, Zhongang Qi, Ying Shan, Wenping Wang, and Ping Luo. Styleadapter: A single-pass lora-free model for stylized image generation. arXiv preprint arXiv:2309.01770, 2023. 1
- [62] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. arXiv preprint arXiv:2302.13848, 2023. 1, 3
- [63] Yi Wu, Ziqiang Li, Heliang Zheng, Chaoyue Wang, and Bin Li. Infinite-id: Identity-preserved personalization via id-semantics decoupling paradigm. arXiv preprint arXiv:2403.11781, 2024. 2, 3, 4

- [64] Guangxuan Xiao, Tianwei Yin, William T Freeman, Fr´edo Durand, and Song Han. Fastcomposer: Tuning-free multisubject image generation with localized attention. arXiv preprint arXiv:2305.10431, 2023. 2, 3, 6, 7, 8
- [65] Guangxuan Xiao, Tianwei Yin, William T Freeman, Fr´edo Durand, and Song Han. Fastcomposer: Tuning-free multisubject image generation with localized attention. arXiv preprint arXiv:2305.10431, 2023. 1
- [66] Yuxuan Yan, Chi Zhang, Rui Wang, Yichao Zhou, Gege Zhang, Pei Cheng, Gang Yu, and Bin Fu. Facestudio: Put your face everywhere in seconds. arXiv preprint arXiv:2312.02663, 2023. 2, 3
- [67] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 1, 2, 3, 4, 6, 7, 8, 9

- [68] Lvmin Zhang. Reference-only controlnet. https:// github.com/Mikubill/sd-webui-controlnet/ discussions/1236, 2023. 3
- [69] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, pages 3836–3847, 2023. 2, 9
- [70] Shilong Zhang, Lianghua Huang, Xi Chen, Yifei Zhang, Zhi-Fan Wu, Yutong Feng, Wei Wang, Yujun Shen, Yu Liu, and Ping Luo. Flashface: Human image personalization with high-fidelity identity preservation. arXiv preprint arXiv:2403.17008, 2024. 3, 6, 8

## Appendix

|Number of reference images|Face Sim. ↑ (%)<br><br>CLIP-T ↑ (%)<br><br>FID ↓<br><br>LAIONAes ↑<br><br>|
|---|---|
|1<br><br>2 4<br><br><br>|55.5 26.1 123.4 6.42 58.5 26.1 122.8 6.41 61.2 25.9 124.7 6.44|

### A. Face Structure Scale

As illustrated in Table 3 of the main paper, the introduction of the face structure branch has indeed enhanced ID similarity. However, it has also compromised some aspects of prompt consistency. In practical implementation, a scaling factor β (see Fig. I(a)) can be incorporated to modulate the intensity of the face structure branch, thereby enabling a flexible trade-off between face similarity and prompt consistency. Fig. II provides a visualization of the impact of different β values on the generated outcomes. As expected, a larger β value leads to increased facial shape and texture fidelity (attributed to the detailed spatial information in the face structure branch), but also results in decreased facial editability (e.g., expression editing, due to the insufficient disentanglement of face structure features). Empirically, a lower β value (e.g., 0.1) can be selected to achieve a balance between prompt consistency and ID alignment.

Table I. Performance of using multi-reference images.

### B. Single-ID Multi-Reference Generation

As described in Sec. 3.2 of the main paper, our method can seamlessly extend to the generation of multi-reference images for a single ID (see Fig. I(b)). Table I provides a quantitative assessment of the advantages of using multireference images. Here, we continue to use the 15 IDs described in Sec. 4.2 for evaluation, but instead of using the same reference image to compute face similarity, we use a different image of the same ID as the target. The findings demonstrate that multi-reference images can improve ID similarity without compromising prompt consistency, FID, and aesthetic scores.

### C. Visualization of Multi-ID Routing Map

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

···

Learnable queries

Learnable queries

Fig. III and Fig. IV display the routing map of each crossattention layer of the U-Net under different diffusion steps. It is observed that, with the utilization of routing regularization loss, the ID router can discern different IDs at earlier time steps and in a more pronounced manner, particularly within the cross-attention layers of the U-Net’s decoder.

···

···

!!(')

!!(&)

!!(#)

···

Attention Block

Attention Block

!#

!#

!%(') !%(&)

!%(#)

···

Attention Block

Attention Block

!$

!$ ×"

FFN

ID Embedding

FFN

ID Embedding

×#

×#

!!"

!!"

···

### D. More Example Generations

···

(a) (b)

Fig. V-VIII present more personalized human image results of UniPortrait, demonstrating the outstanding perfor-

Figure I. Illustrations of (a) face structure scale β and (b) single-ID multi-reference embedding in the inference phrase.

[Figure 224]

Figure II. The effect of the face structure scale.

[Figure 225]

! = 20

[Figure 226]

- ! = 19

[Figure 227]

! = 17

[Figure 228]

! = 13

[Figure 229]

! = 1

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

“Two men sitting in a park together.”

w/o routing regularization loss

Figure III. Visualization of the routing map within each cross-attention layer of the U-Net without the routing regularization loss.

[Figure 235]

- ! = 20

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

“Two men sitting in a park together.”

w/ routing regularization loss

[Figure 241]

! = 19

[Figure 242]

! = 17

[Figure 243]

! = 13

[Figure 244]

! = 1

Figure IV. Visualization of the routing map within each cross-attention layer of the U-Net with the routing regularization loss.

mance of our approach in single- and multi-ID customization again.

|Model / LoRA|URL<br><br>|
|---|---|
|Realistic Vision V4.0<br><br>|https://huggingface.co/SG161222/Realistic Vision V4.0 noVAE<br><br>|
|ToonYou|https://civitai.com/models/30240/toonyou|
|disney-pixarcartoon-typeb|https://civitai.com/models/75650/disney-pixar-cartoon-typeb<br><br>|
|chosen-chinesestylensfw-hentai|https://civitai.com/models/95643/chosen-chinese-stylensfwhentai|

### E. Open-Sourced Text-to-Image Models

Table II shows the URLs of the utilized open-sourced textto-image models and LoRAs in this paper.

Table II. URLs of the used models and LoRAs in this paper.

[Figure 245]

###### Figure V. Text-to-single-ID personalization examples.

[Figure 246]

###### Figure VI. Text-to-single-ID personalization examples.

[Figure 247]

[Figure 248]

