## Customizing Text-to-Image Models with a Single Image Pair

#### Maxwell Jones1 Sheng-Yu Wang1 Nupur Kumari1 David Bau2 Jun-Yan Zhu1 1Carnegie Mellon University 2Northeastern University

Text Prompt: A photo of a dog

# arXiv:2405.01536v2[cs.CV]28Oct2024

|[Figure 1]|[Figure 2]|[Figure 3]|[Figure 4]|
|---|---|---|---|

Pretrained Text-to-Image Diffusion

Pretrained

Text Prompt: A photo of a dog in digital illustration style

Image Pair

stylistic difference

|[Figure 5]|[Figure 6]|[Figure 7]|[Figure 8]|
|---|---|---|---|

|[Figure 9]|
|---|

|[Figure 10]|
|---|

Pair Customization

Content image

Style image

Applies stylistic difference while preserving structure

|[Figure 11]|[Figure 12]|[Figure 13]|[Figure 14]|
|---|---|---|---|

Standard

only train Customization on style image

Fails to preserve structure and original color tone

Figure 1. Given a single image pair, we present Pair Customization, a method for customizing a pre-trained text-to-image model and learning a new style from the image pair’s stylistic difference. Our method can apply the learned stylistic difference to new input images while preserving the input structure. Compared to Dreambooth LoRA [36,77], a standard customization method that solely uses style images, our method effectively disentangles style and content, resulting in better structure, color preservation, and style application. Style image credit: Jack Parkhouse.

### Abstract

to apply a stylistic change without overfitting to the specific image content in the examples. To address this new task, we employ a joint optimization method that explicitly separates the style and content into distinct LoRA weight spaces. We optimize these style and content weights to reproduce the style and content images while encouraging their orthogonality. During inference, we modify the diffusion process via a new style guidance based on our learned weights. Both qualitative and quantitative experiments show that our method can effectively learn style while avoiding overfitting to image content, highlighting the potential of modeling such stylistic differences from a single image pair.

Art reinterpretation is the practice of creating a variation of a reference work, making a paired artwork that exhibits a distinct artistic style. We ask if such an image pair can be used to customize a generative model to capture the demonstrated stylistic difference. We propose Pair Customization, a new customization method that learns stylistic difference from a single image pair and then applies the acquired style to the generation process. Unlike existing methods that learn to mimic a single concept from a collection of images, our method captures the stylistic difference between paired images. This allows us

### 1. Introduction

Artistic works are often inspired by a reference image, a recurring scene, or even a previous piece of art [57]. Such creations involve re-interpreting an original composition in the artist’s unique style. A notable example is Van Gogh’s Repetitions [66], in which the artist created multiple versions of the same scenes with his distinctive expressiveness, including adaptations of other artists’ work. Such sets of variations allow close comparison of stylized art to a reference image, providing unique insights into an artist’s detailed techniques and choices.

In our work, we explore how such content-style image pairs can be used to customize a generative model to capture the demonstrated stylistic difference. Our goal is to customize a pre-trained generative model to synthesize stylized images, distilling the essence of the style from as few as a single pair without fixating on specific content. We wish to create a model capable of re-interpreting a variety of different content in the style demonstrated by the paired variation.

Prior works on model customization/personalization [19, 46,75] take one or a few images of a single concept to customize large-scale text-to-image models [70, 73]. While they aim to learn styles without using pairs, the generated samples from these customized models often resemble the training images’ content, such as specific objects, persons, and scene layouts. In Figure 1, we observe that standard single-image customization (3rd row) alters the subject, color tone, and pose of the original image (1st row). These issues arise because the artistic intent is difficult to discern from a single image: unlike image pairs that can demonstrate a style through contrasts, a singleton example will always intertwine choices of both style and content. Due to this ambiguity, the model fails to capture the artistic style accurately and, in some cases, overfits and generates the subject-specific details rather than the style, as shown in Figure 5.

On the other hand, our Pair Customization method exploits the contrast between image pairs to generate pairwise consistent images while better disentangling style and content. In Figure 1 (2nd row), our method accurately follows the given style, turning the background into a single color matching the original background and preserving the identity and pose for each dog. Our method achieves this by disentangling the intended style from the image pair.

Our new customization task is challenging since text-toimage models were not initially designed to generate pairwise content. Even when given specific text prompts like “a portrait” and “a portrait with Picasso style”, a textto-image diffusion model often struggles to generate images with consistent structure from the same noise seed. Therefore, it remains unclear how a customized model can generate stylized images while maintaining the original structure.

To address the challenges, we first propose a joint optimization method with separate sets of Low-Rank Adaptation [36] (LoRA) weights for style and content. The optimization encourages the content LoRA to reconstruct the content image and the style LoRA to apply the style to the content. We find that the resulting style LoRA can apply the same style to other unseen content. Furthermore, we enforce row-space orthogonality [67] between style and content LoRA parameters to improve style and content disentanglement. Next, we extend the standard classifier-free guidance method [34] and propose style guidance. Style guidance integrates style LoRA predictions into the original denoising path, which aids in better content preservation and facilitates smoother control over the stylization strength. This method is more effective than the previous technique, where a customized model’s strength is controlled by the magnitude of LoRA weights [78].

Our method is built upon Stable Diffusion XL [68]. We experiment with various image pairs, including different categories of content (e.g., portraits, animals, landscapes) and style (e.g., paintings, digital illustrations, filters). We evaluate our method on the above single image pairs and demonstrate the advantage of our method in preserving diverse structures while applying the stylization faithfully, compared to existing customization methods. Our code, models, and data are available on our webpage.

### 2. Related Works

Text-to-image generative models. Deep generative models aim to model the data distribution of a given training set [16, 27, 33, 45, 88, 93]. Recently, large-scale text-toimage models [4, 11, 26, 41, 55, 65, 68, 70, 73, 80, 81, 101] trained on internet-scale training data [9,82] have shown exceptional generalization. Notably, diffusion models [33,87] stand out as the most widely adopted model class. While existing models can generate a broad spectrum of objects and concepts, they often struggle with rare or unseen concepts. Our work focuses on teaching these models to understand and depict a new style concept. Conditional generative models [8,40,48,60,64,79,102] learn to transform images across different domains, but the training often requires thousands to millions of image pairs. We focus on a more challenging case, where only a single image pair is available.

Customizing generative models. Model customization, or personalization, aims to adapt an existing generative model with additional data, with the goal of generating outputs tailored to specific user preferences. Earlier efforts mainly focus on customizing pre-trained GANs [27,43,44] for smaller datasets [42,61,104], incorporating user edits [6, 95, 96], or aligning with text prompts [22, 62]. Recently, the focus has pivoted towards adapting large-scale text-toimage models to generate user-provided concepts, typically

presented as one or a few images. Simply fine-tuning on the concept leads to overfitting. To mitigate this and enable variations via free text, several works explored different regularizations, including prior preservation [46, 75], human alignment [86], patch-based learning [103], as well as parameter update restriction, where we only update text tokens [1, 15, 19, 94], attention layers [20, 30, 46], low-rank weights [36, 77, 90], or clusters of neurons [53]. More recent methods focus on encoder-based approaches for faster personalization [2,13,14,21,47,56,76,84,92,99,100]. Instead of learning a single concept, several works further focus on learning multiple concepts [3,28,46,67,83]. Other methods [58,72] propose customizing text-to-video models to learn motion, while Guo et al. [29] propose animating customized text-to-image models models by incorporating motion Low-Rank Adapter [36] modules. Our method takes inspiration from these techniques; however, we aim to address an inherently different task. Instead of learning concepts from an image collection, we customize the model to learn stylistic differences from an image pair.

Style and content separation. Various past works have explored learning a style while separating it from content [12, 25, 37, 49, 89]. Our work is inspired by the seminal work Image Analogy [32], a computational paradigm that takes an image pair and applies the same translation to unseen images. Common image analogy methods include patch-wise similarity matching [32,39,50] and data-driven approaches [5, 63, 71, 91, 97, 105]. Different from these, we aim to exploit the text-guided generation capabilities of large-scale models so that we can directly use the style concept with unseen context. Recently, StyleDrop [86] has been proposed to learn a custom style for masked generative transformer models. Unlike StyleDrop, we do not rely on human feedback in the process. Concurrent with our work, Hertz et al. [31] introduced a method for generating images with style consistency, offering the option of using a style reference image. In contrast, we exploit an image pair to better discern the stylistic difference.

### 3. Method

Our method seeks to learn a new style from a single image pair. This task is challenging, as models tend to overfit when trained on a single image, especially when generating images in the same category as the training image (e.g., a model trained and tested on dog photos). To reduce this overfitting, we introduce a new algorithm aimed at disentangling the structure of the subject from the style of the artwork. Specifically, we leverage the image pair to learn separate model weights for style and content. At inference time, we modify the standard classifier-free guidance formulation to help preserve the original image structure when applying the learned style. In this section, we give a brief overview of diffusion models, outline our design choices,

and explain the final method in detail.

##### 3.1. Preliminary: Model Customization

Diffusion models. Diffusion models [33, 85, 88], map Gaussian noise to the image distribution through iterative denoising. Denoising is learned by reversing the forward diffusion process x0,...,xT, where image x0 is slowly diffused to random noise xT over T timesteps, defined by xt = √α¯tx0 + √1 − α¯tϵ for timestep t ∈ [0,T]. Noise ϵ ∼ N(0,I) is randomly sampled, and α¯t controls the noise strength. The training objective of diffusion models is to denoise any intermediate noisy image xt via noise prediction:

Eϵ,x,c,t wt∥ϵ − ϵθ(xt,c,t)∥2 , (1)

where wt is a time-dependent weight, ϵθ(·) is the denoiser that learns to predict noise, and c denotes extra conditioning input, such as text. At inference, the denoiser ϵθ will gradually denoise random Gaussian noise into images. The resulting distribution of generated images approximates the training data distribution [33].

In our work, we use Stable Diffusion XL [68], a largescale text-to-image diffusion model built on Latent Diffusion Models [73]. The model consists of a U-Net [74] trained on the latent space of an auto-encoder, with text conditioning from two text encoders, CLIP [69] and OpenCLIP [38].

Model customization with Low-Rank Adapters. LowRank Adapters (LoRA) [36] is a parameter-efficient finetuning method [35] that applies low-rank weight changes ∆θLoRA to pre-trained model weights θ0. For each layer with an initial weight W0 ∈ Rm×n, the weight update is defined by ∆WLoRA = BA, a product of learnable matrices B ∈ Rm×r and A ∈ Rr×n, where r ≪ min(m,n) to enforce the low-rank constraint. The weight matrix of a particular layer with LoRA is:

WLoRA = W0 + ∆WLoRA = W0 + BA. (2)

At inference time, the LoRA strength is usually controlled by a scaling factor α ∈ [0,1] applied to the weight update ∆WLoRA [78]:

WLoRA = W0 + α∆WLoRA. (3)

LoRA has been applied for customizing text-to-image diffusion models to learn new concepts with as few as three to five images [78].

##### 3.2. Style Extraction from an image pair

We aim to customize a pre-trained model with an artistic style in order to stylize the original model outputs while preserving their content, as shown in Figure 2 (right). To achieve this, we introduce style LoRA weight θstyle = θ0 +

|Sample random timestep t|
|---|

UpdateContentLoRA Image pair

|[Figure 15]|
|---|

|[Figure 16]<br><br>[Figure 17]|
|---|

[Figure 18]

[Figure 19]

|A photo of a dog|
|---|

Content LoRA

Pretrained

Pretrained

Loss

|[Figure 20]|
|---|

Joint Denoising Objective

A photo of a V* dog

Denoised image

Content image

|[Figure 21]|
|---|

|[Figure 22]<br><br>[Figure 23]|
|---|

[Figure 24]

[Figure 25]

Same noise seed

Pretrained

Content LoRA

|A photo of a dog in digital art style|
|---|

Pretrained

Loss

Style LoRA

Style LoRA

A photo of a V* dog in digital art style

Denoised image

Style image

Update Style LoRA

Training: Joint optimize style and content Inference: Generate style images with same layouts

- Figure 2. Method overview. (Left) We disentangle style and content from an image pair by jointly training two low-rank adapters, StyleLoRA and ContentLoRA, representing style and content, respectively. Our training objective consists of two losses: The first loss fine-tunes ContentLoRA to reconstruct content image conditioned on a content prompt. The second loss encourages reconstructing the style image using both StyleLoRA and ContentLoRA conditioned on a style prompt, but we only optimize Style LoRA for this loss. (Right) At inference time, we only apply StyleLoRA to customize the model. Given the same noise seed, the customized model generates a stylized counterpart of the original pre-trained model output. V* is a fixed random rare token that is a prompt modifier for the content image. Style image credits: Jack Parkhouse

∆θstyle. While a pre-trained model generates content from a noise seed and text c, style LoRA’s goal is to generate a stylized counterpart of original content from the same noise seed and a style-specific text prompt cstyle, where cstyle is original text c appended by suffix ‘‘in <desc> style’’. Here, <desc> is a placeholder for some worded description of the style (e.g., “digital art”), and style LoRA θstyle associates <desc> to the desired style.

content of the image, we first employ the standard training objective for diffusion models as described in Section 3.1 with the content image:

(xt,content,ccontent,t)∥2 ,

Lcontent = Eϵ,x

content,t wt∥ϵ − ϵθ

content

(4) where ϵθ

is the denoiser with content LoRA applied, xt,content is a noisy content image at timestep t, and ccontent is text representing the content image, including some rare token V*. Next, we optimize the combined style and content weights to reconstruct the style image. In particular, we only train the style LoRA weights during this step, while stopping the gradient flow to the content LoRA weights via stopgrad sg[·]:

content

Unfortunately, learning style LoRA θstyle from a single style image often leads to copying content (Figure 5). Hence, we explicitly learn disentanglement from a style and content image, denoted by xstyle and xcontent, respectively.

Disentangling style and content. We leverage the fact that the style image shares the same layout and structure as the content image. Our key idea is to learn a separate content LoRA θcontent = θ0 + ∆θcontent to reconstruct the content image. By explicitly modeling the content, we can train the style LoRA to “extract” the stylistic differences between the style and content image. We apply both style and content LoRA to reconstruct the style image, i.e., θcombined = θ0 + ∆θcontent + ∆θstyle. This approach prevents leaking the content image to style LoRA, resulting in a better stylization model.

θcombined = θ0 + sg[∆θcontent] + ∆θstyle. (5)

We then apply diffusion objective to train θcombined to denoise xt,style, a noisy style image at timestep t:

(xt,style,cstyle,t)∥2 ,

Lcombined = Eϵ,x

###### style,t wt∥ϵ − ϵθ

combined

(6) where ϵθ

is the denoiser with both LoRAs applied as in Equation 5, cstyle is ‘‘{ccontent} in <desc> style’’, and <desc> is a worded description of the style (e.g., “digital art”). Finally, we jointly optimize the LoRAs with the two losses:

combined

During training, we feed the content LoRA θcontent with a content-specific text ccontent, which contains a random rare token V*, and feed the combined model θcombined with cstyle, where cstyle is ‘‘{ccontent} in <desc> style’’. Figure 2 (Left) summarizes our training process.

Lcontent + Lcombined (7)

min

∆θcontent,∆θstyle

Figure 2 provides an overview of our method. Next, we discuss the regularization that promotes the disentanglement of style from content.

Jointly learning style and content. We employ two different objectives during every training step. To learn the

LoRA Weight Scale

|[Figure 26]|[Figure 27]|[Figure 28]|[Figure 29]|[Figure 30]|
|---|---|---|---|---|

0.0 0.4 0.6 0.8

0.2

Style Guidance

|[Figure 31]|[Figure 32]|[Figure 33]|[Figure 34]|[Figure 35]|
|---|---|---|---|---|

0.0 2.0 3.0 4.0

1.0

- Figure 3. Style guidance. We compare our style guidance and standard LoRA weight scaling [78]. Style guidance better preserves content when the style is applied. Blue and green stand for the LoRA weight scale and style guidance scale, respectively. More details of style guidance formulation are in Section 3.3.

|[Figure 36]|[Figure 37]|
|---|---|
| |[Figure 38]|

Without Orthogonal Adaption

|[Figure 39]|[Figure 40]|
|---|---|
| |[Figure 41]|

With Orthogonal Adaption

|[Figure 42]|[Figure 43]|
|---|---|
| |[Figure 44]|

Pretrained Stable Diffusion

Training Pair

|[Figure 45]|
|---|

|[Figure 46]|
|---|

Content Image

Style Image

- Figure 4. Orthogonal adaptation. Enforcing row-space orthogonality between style and content LoRA improves image quality, where the images capture the style better and have fewer visual artifacts.

Orthogonality between style and content LoRA. To further encourage style and content LoRAs to represent separate concepts, we enforce orthogonality upon the LoRA weights. We denote by W0 the original weight matrix and Wcontent, Wstyle the LoRA modifications (layer index omitted for simplicity). Reiterating Equation 2, we decompose Wcontent, Wstyle into low-rank matrices:

Wcontent =W0 + BcontentAcontent Wstyle =W0 + BstyleAstyle.

(8)

We initialize Bcontent,Bstyle with the zero matrix and choose the rows of Acontent, Astyle from an orthonormal basis. We then fix Acontent, Astyle and only update Bcontent, Bstyle in training. This forces the style and content LoRA updates to respond to orthogonal inputs, and empirically reduces visual artifacts, as shown in Figure 4. This technique is inspired by Po et al. [67]. While their work focuses on merging multiple customized objects after each is trained separately, we apply the method for style-content separation during joint training.

##### 3.3. Style Guidance

A common technique to improve text-to-image model’s sample quality is via classifier-free guidance [34]:

ϵˆθ(xt,c) = ϵθ(xt,∅) + λcfg(ϵθ(xt,c) − ϵθ(xt,∅)), (9)

where ϵˆθ(xt,c,t) is the new noise prediction, ∅ denotes no conditioning, and λcfg controls the amplification of text guidance. For notation simplicity, we omit the timestep t in this equation and subsequent ones.

To improve pairwise consistency between original and stylized content, we propose an inference algorithm that preserves the original denoising path while adding controllable style guidance:

ϵˆθ

0,θstyle(xt,c,cstyle)

(xt,∅)

= ϵθ

(10)

0

(xt,∅))

(xt,c) − ϵθ

+ λcfg(ϵθ

0

0

(xt,cstyle) − ϵθ

+ λstyle(ϵθ

(xt,c)),

style

0

where style guidance is the difference in noise prediction between style LoRA and the pre-trained model. Style guidance strength is controlled by λstyle, and setting λstyle = 0 is equivalent to generating original content. In Figure 3, we compare our style guidance against scaling LoRA weights (Equation 3), and we find that our method better preserves the layout. More details and a derivation of our style guidance are in Appendix B.

Previous works have also used multiple guidance terms with diffusion models, including guidance from multiple text prompts using the same model [52] and additional image conditions [8]. Unlike these, we obtain additional guidance from a customized model and apply it to the original model. StyleDrop [86] considers a similar formulation with two guidance terms but for masked generative transformers. SINE [103] uses a customized content model to apply textbased image editing to a single image, like adding snow. However, we use a customized style model to generate any image with the desired style.

Blending multiple learned styles. With a collection of models customized by our method, we can blend the learned styles as follows. Given some set of styles S and strengths λstyle

, we blend the style guidance from each model, and our new inference path is represented by

,...,λstyle

0

n

ϵˆθ

0,θstyle(xt,c,cstyle)

(xt,∅)

= ϵθ

0

(11)

(xt,∅))

(xt,c) − ϵθ

+ λcfg(ϵθ

0

0

) − ϵθ

+

λstyle

(ϵθ

(xt,cstyle

(xt,c)),

stylei

0

i

i

stylei∈S

We can vary the strengths of any parameter λstyle

to seamlessly increase or decrease style application while preserving content. Figure 10 gives a qualitative example of blending two different styles while preserving image content.

i

Implementation details. We train all models using an AdamW optimizer [54] and learning rate 1 × 10−5. For baselines, we train for 500 steps. For our method, we first train our content weights on the content image for 250

steps, and then train jointly for 500 additional steps. All image generation is performed using 50 steps of a PNDMScheduler [51]. For all methods using inference with LoRA adapters, we use SDEdit [59] to further preserve structure. Specifically, normal classifier-free guidance on the original prompt without style is used for the first 10 steps. We then apply style guidance/LoRA scale for the rest of the timesteps.

### 4. Experiments

##### 4.1. Dataset

In this section, we show our method’s results on various image pairs and compare them with several baselines. We explain our dataset, baselines, and metrics in detail, then we present quantitative and qualitative results.

Datasets. To enable large-scale quantitative evaluation, we construct a diverse set of paired style and content images as follows. First, we generate 40 content images for each class: headshots, animals, and landscapes. When generating images in the headshot class, we generate 20 images with the prompt “A professional headshot of a man” and 20 images with the prompt

“A professional headshot of a woman”. Similarly, we split the animal class into photos of dogs and cats. To curate synthetic pairs, we then apply image editing or imageto-image translation methods to all the content images to obtain the stylized version. For each unique prompt, we choose a single paired instance as training data and hold out the other pairs with the same prompt as a test set (Same Category). For each prompt, we also choose 5 pairs from each of the other prompts as a secondary test set (Different Category). We show all our synthetic training image pairs in Appendix D. By leveraging synthetic pairs for evaluation, we can train on a single synthetic pair and test our results against held out synthetic style images. Secondly, we qualitatively compare against single artist pairs in Figure 5. Next, we describe the specific methods to create the paired dataset. First, we consider the diffusion based image editing technique LEDITS++ [7] to translate images into paintings. Next, we consider Cartoonization [98], a GAN-based translation technique that aplies a cartoon-like effect. We also consider Stylized Neural Painting [106], which turns photos into painting strokes using a rendering based approach. Finally, we consider the image filtering technique posterization. We provide a more detailed description of each method for creating synthetic pairs in Appendix D.

##### 4.2. Baselines and Evaluation Metrics

Baselines. We compare our method against – (1) DreamBooth LoRA [36,78] (DB LoRA), (2) Concept Sliders [23] (3) IP-adapters [100], (4) IP-adapters w/ T2I, and (5) StyleDrop [86]. DB LoRA uses only the style image and fine-

tunes low-rank adapters in all the linear layers in the attention blocks of the diffusion model. We evaluate different amounts of style applications for DB LoRA using the standard LoRA scale [78] . Concept sliders presents a paired image model customization method that trains a single low-rank adapter jointly on both images, with different reconstruction losses for the style and content images. We also evaluate using the standard LoRA scale . IP-adapters is an encoder-based method that does not require training for every style and takes a style image as an extra condition separate from the text prompt. Increasing or decreasing the guidance from the input style image is possible by scaling the weight of the image conditioning. We consider the SDXL [68] implementation of this method. For the IPAdapter, we compare against the stronger baseline of providing extra conditioning of an edge map of the content image through T2I Adapters [60] to preserve the content image structure. The recently proposed Styledrop [86] technique for learning new styles is based on MUSE [11], and uses human feedback in its method. Since MUSE is not publicly available, we follow Style-Aligned Image Generation’s [31] setup, and implement a version of StyleDrop on SDXL. Specifically, we train low-rank linear layers following each Feed-Forward layer in the attention blocks of SDXL. For a fair comparison, we train Styledrop without human feedback.

Evaluation metrics. When evaluating the performance of each method, we consider two quantitative metrics: perceptual distance to ground truth style images and structure preservation from the original image. A better customization method will have a low perceptual distance to the ground truth style images while still preserving content of the original image before adding style. We measure these using – (1) Distance to GT Styled: given holdout ground truth style images, we measure the perceptual distance between our styled outputs and the ground truth style images using DreamSim [18], a recent method for measuring the perceptual distance between images. DreamSim image embeddings are comprised of an ensemble of image embedding models, including CLIP [69] and DINO [10], which are then fine-tuned so the final embeddings respect human perception. We measure DreamSim distance as (1 - cosine similarity between DreamSim embeddings), where a lower value implies that the images are perceptually more similar. (2) Distance to Content Image: to measure content preservation after style application, we measure the perceptual distance of our generated style image to the original content image with no style guidance. We again use DreamSim, this time comparing styled and content images. Note here that a perceptual distance of zero to the content image is undesirable, as this would require no style to be applied. However, a better-performing method should obtain a better tradeoff between the two distances. (3) We also perform a

Pretrained Output Ours Sliders DB LoRA

|[Figure 47]|[Figure 48]|
|---|---|
| |[Figure 49]|

|[Figure 50]|[Figure 51]|
|---|---|
| |[Figure 52]|

|[Figure 53]|[Figure 54]|
|---|---|
| |[Figure 55]|

|[Figure 56]|[Figure 57]|
|---|---|
| |[Figure 58]|

Artist Created Pairs

|[Figure 59]|
|---|

|[Figure 60]|
|---|

“A photo of a dog in digital art style”

|[Figure 61]|[Figure 62]|
|---|---|
| |[Figure 63]|

|[Figure 64]|[Figure 65]|
|---|---|
| |[Figure 66]|

|[Figure 67]|[Figure 68]|
|---|---|
| |[Figure 69]|

|[Figure 70]|[Figure 71]|
|---|---|
| |[Figure 72]|

Style Image

Content Image

“A photo of a cat in digital art style”

|[Figure 73]|[Figure 74]|
|---|---|
| |[Figure 75]|

|[Figure 76]|[Figure 77]|
|---|---|
| |[Figure 78]|

|[Figure 79]|[Figure 80]|
|---|---|
| |[Figure 81]|

|[Figure 82]|[Figure 83]|
|---|---|
| |[Figure 84]|

|[Figure 85]|
|---|

|[Figure 86]|
|---|

“A dinner plate in drawing style”

|[Figure 87]|[Figure 88]|
|---|---|
| |[Figure 89]|

|[Figure 90]|[Figure 91]|
|---|---|
| |[Figure 92]|

|[Figure 93]|[Figure 94]|
|---|---|
| |[Figure 95]|

|[Figure 96]|[Figure 97]|
|---|---|
| |[Figure 98]|

Content Image

Style Image

“A bowl of soup on a plate in drawing style”

|[Figure 99]|[Figure 100]|
|---|---|
| |[Figure 101]|

|[Figure 102]|[Figure 103]|
|---|---|
| |[Figure 104]|

|[Figure 105]|[Figure 106]|
|---|---|
| |[Figure 107]|

|[Figure 108]|[Figure 109]|
|---|---|
| |[Figure 110]|

###### Synthetic Pairs

|[Figure 111]|
|---|

|[Figure 112]|
|---|

“A photo of a dog in cartoon style”

|[Figure 113]|[Figure 114]|
|---|---|
| |[Figure 115]|

|[Figure 116]|[Figure 117]|
|---|---|
| |[Figure 118]|

|[Figure 119]|[Figure 120]|
|---|---|
| |[Figure 121]|

|[Figure 122]|[Figure 123]|
|---|---|
| |[Figure 124]|

Style Image

Content Image

“A photo of a cat in cartoon style”

|[Figure 125]|[Figure 126]|
|---|---|
| |[Figure 127]|

|[Figure 128]|[Figure 129]|
|---|---|
| |[Figure 130]|

|[Figure 131]|[Figure 132]|
|---|---|
| |[Figure 133]|

|[Figure 134]|[Figure 135]|
|---|---|
| |[Figure 136]|

|[Figure 137]|
|---|

|[Figure 138]|
|---|

“A headshot of a man in painting style”

|[Figure 139]|[Figure 140]|
|---|---|
| |[Figure 141]|

|[Figure 142]|[Figure 143]|
|---|---|
| |[Figure 144]|

|[Figure 145]|[Figure 146]|
|---|---|
| |[Figure 147]|

|[Figure 148]|[Figure 149]|
|---|---|
| |[Figure 150]|

Content Image

Style Image

“A headshot of a woman in painting style”

- Figure 5. Result of our method compared to the strongest baselines. When only training with the style image as in DB LoRA, the image structure is not preserved, and overfitting occurs. While Concept Slider’s training scheme [23] uses both style and content images, it still exhibits overfitting and loss of structure in many cases. Our method preserves the structure of the input image while faithfully applying the desired style. We use style guidance strength 3 and classifier guidance strength 5. Style image credits: Jack Parkhouse (First row) and Aaron Hertzmann (Second row)

Pretrained Output

Style-Align (control)

IP-Adapter (T2I)

Ours

Style-Align

IP-Adapter

Synthetic Pairs

|[Figure 151]|
|---|

|[Figure 152]|
|---|

|[Figure 153]|
|---|

|[Figure 154]|
|---|

|[Figure 155]|
|---|

|[Figure 156]|
|---|

“A photo of a landscape in poster style”

|[Figure 157]|
|---|

|[Figure 158]|
|---|

|[Figure 159]|
|---|

|[Figure 160]|
|---|

|[Figure 161]|
|---|

|[Figure 162]|
|---|

|[Figure 163]|
|---|

|[Figure 164]|
|---|

Style Image

Content Image

“A photo of a cat in poster style”

|[Figure 165]|
|---|

|[Figure 166]|
|---|

|[Figure 167]|
|---|

|[Figure 168]|
|---|

|[Figure 169]|
|---|

|[Figure 170]|
|---|

###### “A photo of a dog in poster style”

|[Figure 171]|
|---|

|[Figure 172]|
|---|

|[Figure 173]|
|---|

|[Figure 174]|
|---|

|[Figure 175]|
|---|

|[Figure 176]|
|---|

“A headshot of a cat in painted style”

|[Figure 177]|
|---|

|[Figure 178]|
|---|

|[Figure 179]|
|---|

|[Figure 180]|
|---|

|[Figure 181]|
|---|

|[Figure 182]|
|---|

|[Figure 183]|
|---|

|[Figure 184]|
|---|

“A headshot of a dog in painted style”

Content Image

Style Image

|[Figure 185]|
|---|

|[Figure 186]|
|---|

|[Figure 187]|
|---|

|[Figure 188]|
|---|

|[Figure 189]|
|---|

|[Figure 190]|
|---|

“A photo of a landscape in painted style”

|[Figure 191]|
|---|

|[Figure 192]|
|---|

|[Figure 193]|
|---|

|[Figure 194]|
|---|

|[Figure 195]|
|---|

|[Figure 196]|
|---|

“A photo of a dog in painted style”

|[Figure 197]|
|---|

|[Figure 198]|
|---|

|[Figure 199]|
|---|

|[Figure 200]|
|---|

|[Figure 201]|
|---|

|[Figure 202]|
|---|

|[Figure 203]|
|---|

|[Figure 204]|
|---|

Content Image

Style Image

“A photo of a cat in painted style”

|[Figure 205]|
|---|

|[Figure 206]|
|---|

|[Figure 207]|
|---|

|[Figure 208]|
|---|

|[Figure 209]|
|---|

|[Figure 210]|
|---|

###### “A photo of a landscape in painted style”

- Figure 6. Result of our method compared to the methods without finetuning (zoom in for best viewing). For the baseline methods, we also add the edge map from the pretrained output as an extra condition (3rd and 5th column). Without this edge map, other methods tend to lose the structure of the pretrained output. In some cases, however, an additional edge map can overly constrain the output of a model, like in the second image pair example. Our method preserves the structure of the Stable Diffusion image while faithfully applying the desired style. We use style guidance strength 3 and classifier guidance strength 5 for our method and set the IP-adapter scale and style-alignment scale to 0.5.

##### 4.3. Results

Same Category

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

Quantitative evaluation. We show quantitative results against the highest performing baselines in Figure 7. Increased marker size (circles) indicates the higher application of style, and line color determines the method. When evaluating style similarity vs. structure preservation in Figure 7, we see that our training method’s Pareto dominates all baselines, yielding lower perceptual distance to style images while still being perceptually similar to the original content image. We find that DB LoRA and Styledrop perform similarly, and report Styledrop results in Appendix A. Finally, we consider our method with LoRA scale during inference and other baselines with our style guidance during inference for ablation, and provide results in Appendix A.

Different Category

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

Qualitative evaluation. We compare our method with the highest performing baselines in Figure 5. The finetuningbased methods DB LoRA [36,78] and Concept Sliders [23] outperform the encoder-based method [100] for our task. Hence, we compare against that in Figure 5. For both baselines, we modulate style application with LoRA scale (Equation 3). We observe that DB LoRA often fails to generate the style-transformed version of the original image and overfits to the training pair image when generating similar concepts. There are two main reasons why this may occur. First, we are in a challenging case where there is only 1 training image instead of the usual 3 − 5 images that customization methods use. Second, we are prompting the model on the same or very similar text prompts to the training prompt, and the baseline method overfits to the training image for these prompts. Our method preserves the structure of the original image while applying the learned style. Moreover, applying our style guidance instead of the LoRA scale benefits the baseline method as well (Figure 5, last 2 columns), as it can better preserve the structure of the original image, though it still tends to overfit to the content of the training image. We observe a similar issue for other baselines as well.

- Figure 7. Quantitative comparison with baselines on learned style. Given a fixed inference path, our method pareto dominates baselines for image generation both on the same category as training (left) and when evaluated on categories different from training, e.g., trained on human portraits but tested on dog images (right). We further evaluate the diversity of generated images in the supplement. We show that baselines often lose diversity, while our method leads to diverse generations while still achieving lower perceptual distance to the ground truth style. Increased marker size corresponds to an increase in the guidance scale.

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

- Figure 8. Human preference study. Our method is preferred over the baselines (≥ 60%). Further, our full method, including orthogonal weight matrices (Section 3.2), is preferred over the one w/o orthogonal weight matrices, specifically for the same category as training pair, e.g., trained on a headshot of a man and tested on other headshots of man. The Gray dashed line denotes 50% chance performance.

We compare our method to non finetuning-based methods in Figure 6. We observe that these methods perform worse than finetuning-based methods, especially when generating images in a different category to the training style image. We also compare with baselines using our style guidance for style application at inference time in Appendix A .

User preference study. We perform a user preference study using Amazon Mechanical Turk. We test our method against all baselines, as well as a version of our method trained without the orthogonality constraint. Specifically, we test on all datasets in Section 4.1. When evaluating against DB LoRA and Concept Sliders, we consider inference with both LoRA scale as in Equation 3 and style guidance as in Equation 10. For each method, we pick a sin-

human preference study of our method against baselines.

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

Original Model

Edited Model 0

Real Image Style Pair

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

- Train Image Pair 0

[Figure 222]

- Train Image Pair 1

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

Edited Model 1

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

- Figure 10. Blending multiple style guidances. We can compose multiple customized models by directly blending each style guidance together. Adjusting the blending strength of each model allows us to acquire a smooth style transition. Each stylized image corresponds to different style guidance strengths. Train Image Pair 0 style image credits: Jack Parkhouse

|[Figure 233]|
|---|

|[Figure 234]|
|---|

Content Image

Style Image

|[Figure 235]|
|---|
|[Figure 236]|

Pretrained Output

|[Figure 237]|
|---|
|[Figure 238]|

Ours

Training Pair

|[Figure 239]|
|---|
|[Figure 240]|

Pretrained Output

|[Figure 241]|
|---|

|[Figure 242]|
|---|

Content Image

Style Image

Ours

Baseline

Baseline

|[Figure 243]|
|---|
|[Figure 244]|

|[Figure 245]|
|---|
|[Figure 246]|

|[Figure 247]|
|---|
|[Figure 248]|

- Figure 11. Limitations. Top: our method can cause structure changes in some instances, like change of body position or background changes. Bottom: Our method can change the content in some cases from pretrained output, like the addition of facial hair. We display Baseline DB LoRA for comparison.

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

- Figure 9. Real image editing. We can edit real images by inverting a real image into a noisy latent code using a diffusion inversion pipeline [24, 87] and an image prompt c. From here, we apply style guidance (Equation 10) with the same prompt c and new style prompt cstyle for the desired style application.

gle style strength that performs most optimally according to quantitative metrics as in Figure 7. Full details are available in Appendix D, and a user study with all baselines is available in Appendix A . We collect 400 responses per paired test of ours vs the other method. The user is shown an image generated via our method and an image generated via the other method and asked to select the image that best applies the given style to the new content image. We provide a detailed setup of the user study as well as a user study on baseline methods using our style guidance in Appendix D As shown in Figure 8, our method is favored by users in comparison to baselines, whether evaluating images generated within the same category as the training image pair or across different categories. Secondly, users prefer our full method to ours without the orthogonality constraint, specifically when evaluating on the same category as training.

Real Image Editing. Our method can also stylize real images. We use DDIM inversion [24, 87] to invert images into their noisy latent codes at some intermediate step using a reference prompt c. From here, we use our style guidance (Equation 10) with reference prompt c and new prompt cstyle to denoise the noisy latent code to a stylized image. In Figure 9, we show real image editing results. We provide more details in Appendix C.

### 5. Discussion and Limitations

In this work, we have introduced a new task: customizing a text-to-image model with a single image pair. To address this task, we have developed a customization method that explicitly disentangles style and content through both training objectives and a separated parameter space. Our method enables us to grasp the style concept without memorizing the content of input examples. While our approach outperforms existing customization methods, it still exhibits

Blending learned styles. We show that we can blend the learned styles by applying a new inference path, defined in Equation 11. In Figure 10, we show the results of blending two models. We can seamlessly blend the two styles at varying strengths while still preserving the content.

several limitations, as discussed below.

Limitations. First, our method may occasionally fail to completely maintain input structure, as demonstrated in Figure 11. This could occur as background/pose change (Top), or as additional features being added (Bottom).

Second, our current method relies on test-time optimization, which takes around 15 minutes on a single A5000 GPU. This can be computationally demanding if we need to process many image styles. Leveraging encoder-based approaches [2,76] for predicting style and content weights in a feed-forward manner could potentially speed up the customization process.

Acknowledgments. We would like to thank Ali Jahanian, Gaurav Parmar, Ruihan Gao, Sean Liu, and Or Patashnik for their insightful feedback and input that contributed to the finished work. We also thank Jack Parkhouse and Aaron Hertzmann for providing style images. Maxwell Jones is supported by the Rales Fellowship and the Siebel Scholar fellowship. This project is partly supported by the Amazon Faculty Research Award, NSF IIS-2239076, Open Philanthropy, and the Packard Fellowship.

### References

- [1] Yuval Alaluf, Elad Richardson, Gal Metzer, and Daniel Cohen-Or. A neural space-time representation for text-toimage personalization. In SIGGRAPH Asia, 2023.
- [2] Moab Arar, Rinon Gal, Yuval Atzmon, Gal Chechik, Daniel Cohen-Or, Ariel Shamir, and Amit H. Bermano. Domainagnostic tuning-encoder for fast personalization of text-toimage models. In SIGGRAPH Asia 2023 Conference Papers, pages 1–10, 2023.
- [3] Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel CohenOr, and Dani Lischinski. Break-a-scene: Extracting multiple concepts from a single image. In SIGGRAPH Asia 2023 Conference Papers, pages 1–12, 2023.
- [4] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. ediffi: Text-toimage diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.
- [5] Amir Bar, Yossi Gandelsman, Trevor Darrell, Amir Globerson, and Alexei A. Efros. Visual prompting via image inpainting. 2022.
- [6] David Bau, Steven Liu, Tongzhou Wang, Jun-Yan Zhu, and Antonio Torralba. Rewriting a deep generative model. In ECCV, 2020.
- [7] Manuel Brack, Felix Friedrich, Katharina Kornmeier, Linoy Tsaban, Patrick Schramowski, Kristian Kersting, and Apolin´ario Passos. Ledits++: Limitless image editing using text-to-image models. arXiv preprint arXiv:2311.16711, 2023.
- [8] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on

- Computer Vision and Pattern Recognition, pages 18392– 18402, 2023.
- [9] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/ kakaobrain/coyo-dataset, 2022.
- [10] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021.
- [11] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jos´e Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Textto-image generation via masked generative transformers. In Proceedings of the 40th International Conference on Machine Learning, pages 4055–4075, 2023.
- [12] Tian Qi Chen and Mark Schmidt. Fast patchbased style transfer of arbitrary style. arXiv preprint arXiv:1612.04337, 2016.
- [13] Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Rui, Xuhui Jia, Ming-Wei Chang, and William W Cohen. Subject-driven text-to-image generation via apprenticeship learning. 2023.
- [14] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. arXiv preprint arXiv:2307.09481, 2023.
- [15] Giannis Daras and Alexandros G Dimakis. Multiresolution textual inversion. arXiv preprint arXiv:2211.17115, 2022.
- [16] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real nvp. In ICLR, 2017.
- [17] Yarden Frenkel, Yael Vinker, Ariel Shamir, and Daniel Cohen-Or. Implicit style-content separation using b-lora. arXiv preprint arXiv:2403.14572, 2024.
- [18] Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. arXiv preprint arXiv:2306.09344, 2023.
- [19] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In The Eleventh International Conference on Learning Representations, 2022.
- [20] Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Designing an encoder for fast personalization of text-to-image models. ACM Transactions on Graphics (TOG), 2023.
- [21] Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Encoder-based domain tuning for fast personalization of text-to-image models. ACM Transactions on Graphics (TOG), 42(4):1–13, 2023.
- [22] Rinon Gal, Or Patashnik, Haggai Maron, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Stylegan-nada: Clipguided domain adaptation of image generators. ACM Transactions on Graphics (TOG), 41(4):1–13, 2022.

- [23] Rohit Gandikota, Joanna Materzynska, Tingrui Zhou, Antonio Torralba, and David Bau. Concept sliders: Lora adaptors for precise control in diffusion models. arXiv preprint arXiv:2311.12092, 2023.
- [24] Daniel Garibi, Or Patashnik, Andrey Voynov, Hadar Averbuch-Elor, and Daniel Cohen-Or. Renoise: Real image inversion through iterative noising. arXiv preprint arXiv:2403.14602, 2024.
- [25] Leon A Gatys, Alexander S Ecker, and Matthias Bethge. A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576, 2015.
- [26] Aaron Gokaslan, A Feder Cooper, Jasmine Collins, Landan Seguin, Austin Jacobson, Mihir Patel, Jonathan Frankle, Cory Stephenson, and Volodymyr Kuleshov. Commoncanvas: An open diffusion model trained with creativecommons images. arXiv preprint arXiv:2310.16825, 2023.
- [27] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020.
- [28] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models. Advances in Neural Information Processing Systems, 36, 2024.
- [29] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.
- [30] Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. In ICCV, 2023.
- [31] Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. Style aligned image generation via shared attention. arXiv preprint arXiv:2312.02133, 2023.
- [32] Aaron Hertzmann, Charles E. Jacobs, Nuria Oliver, Brian Curless, and David H. Salesin. Image analogies. 2001.
- [33] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. 2020.
- [34] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [35] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In International Conference on Machine Learning, pages 2790–2799. PMLR, 2019.
- [36] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In ICLR, 2021.
- [37] Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In ICCV, 2017.
- [38] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave,

- Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, July 2021. If you use this software, please cite it as below.
- [39] Revital Irony, Daniel Cohen-Or, and Dani Lischinski. Colorization by example. In Eurographics Conference on Rendering Techniques, 2005.
- [40] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In CVPR, 2017.
- [41] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In CVPR, 2023.
- [42] Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. 2020.
- [43] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In CVPR, 2019.
- [44] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In CVPR, 2020.
- [45] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014.
- [46] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1931–1941. IEEE Computer Society, 2023.
- [47] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pretrained subject representation for controllable text-to-image generation and editing. Advances in Neural Information Processing Systems, 36, 2024.
- [48] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In CVPR, 2023.
- [49] Yanghao Li, Naiyan Wang, Jiaying Liu, and Xiaodi Hou. Demystifying neural style transfer. arXiv preprint arXiv:1701.01036, 2017.
- [50] Jing Liao, Yuan Yao, Lu Yuan, Gang Hua, and Sing Bing Kang. Visual attribute transfer through deep image analogy. ACM Trans. Graph., 36(4), jul 2017.
- [51] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. In ICLR, 2022.
- [52] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual generation with composable diffusion models. In European Conference on Computer Vision, pages 423–439. Springer, 2022.
- [53] Zhiheng Liu, Ruili Feng, Kai Zhu, Yifei Zhang, Kecheng Zheng, Yu Liu, Deli Zhao, Jingren Zhou, and Yang Cao. Cones: Concept neurons in diffusion models for customized generation. 2023.
- [54] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2018.

- [55] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.
- [56] Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subject-diffusion: Open domain personalized text-toimage generation without test-time fine-tuning. arXiv preprint arXiv:2307.11410, 2023.
- [57] Csaba Markus. How six different artists have reinterpreted da vinci’s ‘mona lisa’. https : / / www.parkwestgallery.com/six-differentartists-da-vinci-mona-lisa/, 11 2019.
- [58] Joanna Materzynska, Josef Sivic, Eli Shechtman, Antonio Torralba, Richard Zhang, and Bryan Russell. Customizing motion in text-to-video diffusion models. arXiv preprint arXiv:2312.04966, 2023.
- [59] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In ICLR, 2022.
- [60] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-toimage diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 4296– 4304, 2024.
- [61] Yotam Nitzan, Kfir Aberman, Qiurui He, Orly Liba, Michal Yarom, Yossi Gandelsman, Inbar Mosseri, Yael Pritch, and Daniel Cohen-Or. Mystyle: A personalized generative prior. In SIGGRAPH ASIA, 2022.
- [62] Yotam Nitzan, Micha¨el Gharbi, Richard Zhang, Taesung Park, Jun-Yan Zhu, Daniel Cohen-Or, and Eli Shechtman. Domain expansion of image generators. In CVPR, 2023.
- [63] Taesung Park, Alexei A Efros, Richard Zhang, and JunYan Zhu. Contrastive learning for unpaired image-to-image translation. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IX 16, pages 319–345. Springer, 2020.
- [64] Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and JunYan Zhu. Semantic image synthesis with spatially-adaptive normalization. In CVPR, 2019.
- [65] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023.
- [66] The Phillips Collection. Vah gogh repetitions. https:// www.phillipscollection.org/event/201310-11-van-gogh-repetitions, 10 2013.
- [67] Ryan Po, Guandao Yang, Kfir Aberman, and Gordon Wetzstein. Orthogonal adaptation for modular customization of diffusion models. arXiv preprint arXiv:2312.02432, 2023.
- [68] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [69] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,

- Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. 2021.
- [70] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.
- [71] Scott Reed, Yi Zhang, Yuting Zhang, and Honglak Lee. Deep visual analogy-making. 2015.
- [72] Yixuan Ren, Yang Zhou, Jimei Yang, Jing Shi, Difan Liu, Feng Liu, Mingi Kwon, and Abhinav Shrivastava. Customize-a-video: One-shot motion customization of text-to-video diffusion models. arXiv preprint arXiv:2402.14780, 2024.
- [73] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.
- [74] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015.
- [75] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500– 22510, 2023.
- [76] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models, 2023.
- [77] Simo Ryu. Lora-stable diffusion. https://github. com/cloneofsimo/lora, 2023.
- [78] Simo Ryu. Low-rank adaptation for fast text-to-image diffusion fine-tuning. https : / / github . com / cloneofsimo/lora, 2023.
- [79] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 Conference Proceedings, pages 1– 10, 2022.
- [80] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022.
- [81] Axel Sauer, Tero Karras, Samuli Laine, Andreas Geiger, and Timo Aila. Stylegan-t: Unlocking the power of gans for fast large-scale text-to-image synthesis. In International conference on machine learning, pages 30105–

30118. PMLR, 2023.

- [82] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-

- 400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021.
- [83] Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. Ziplora: Any subject in any style by effectively merging loras. arXiv preprint arXiv:2311.13600, 2023.
- [84] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without testtime finetuning. arXiv preprint arXiv:2304.03411, 2023.
- [85] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. 2015.
- [86] Kihyuk Sohn, Nataniel Ruiz, Kimin Lee, Daniel Castro Chin, Irina Blok, Huiwen Chang, Jarred Barber, Lu Jiang, Glenn Entis, Yuanzhen Li, et al. Styledrop: Text-to-image generation in any style. arXiv preprint arXiv:2306.00983, 2023.
- [87] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021.
- [88] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. In ICLR, 2021.
- [89] Joshua Tenenbaum and William Freeman. Separating style and content. Advances in neural information processing systems, 9, 1996.
- [90] Yoad Tewel, Rinon Gal, Gal Chechik, and Yuval Atzmon. Key-locked rank one editing for text-to-image personalization. ACM Transactions on Graphics (TOG), 2023.
- [91] Paul Upchurch, Noah Snavely, and Kavita Bala. From a to z: supervised transfer of style and content using deep neural network generators. arXiv preprint arXiv:1603.02003, 2016.
- [92] Dani Valevski, Danny Lumen, Yossi Matias, and Yaniv Leviathan. Face0: Instantaneously conditioning a text-toimage model on a face. In SIGGRAPH Asia 2023 Conference Papers, pages 1–10, 2023.
- [93] Aaron Van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelcnn decoders. 2016.
- [94] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. p+: Extended textual conditioning in text-toimage generation. arXiv preprint arXiv:2303.09522, 2023.
- [95] Sheng-Yu Wang, David Bau, and Jun-Yan Zhu. Sketch your own gan. In ICCV, 2021.
- [96] Sheng-Yu Wang, David Bau, and Jun-Yan Zhu. Rewriting geometric rules of a gan. ACM SIGGRAPH, 2022.
- [97] Xinlong Wang, Wen Wang, Yue Cao, Chunhua Shen, and Tiejun Huang. Images speak in images: A generalist painter for in-context visual learning. In CVPR, 2023.
- [98] Xinrui Wang and Jinze Yu. Learning to cartoonize using white-box cartoon representations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8090–8099, 2020.
- [99] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. In ICCV, 2023.

- [100] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721, 2023.
- [101] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. Transactions on Machine Learning Research, 2022.
- [102] Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023.
- [103] Zhixing Zhang, Ligong Han, Arnab Ghosh, Dimitris N Metaxas, and Jian Ren. Sine: Single image editing with text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6027–6037, 2023.
- [104] Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han. Differentiable augmentation for data-efficient gan training. 2020.
- [105] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycleconsistent adversarial networks. In ICCV, 2017.
- [106] Zhengxia Zou, Tianyang Shi, Shuang Qiu, Yi Yuan, and Zhenwei Shi. Stylized neural painting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15689–15698, 2021.

## Appendix

In Section A, we evaluate our method against baselines on the diversity metric, showing that our method leads to more diverse generations comparatively. We also show more qualitative results along with a comparison to the concurrent work of Style Aligned Image Generation [31]. In Section B, we then present details of our style guidance formulation. In Section C, we provide implementation details for real image editing, as well as quantitative and qualitative comparisons of real image editing with our method vs baselines. Finally, in Section D, we provide more implementation details, including the setup for our human preference study and the full synthetic training dataset used for evaluation.

### A. More Quantitative and Qualitative Results

Style Guidance Ablation We compare our method, DB LoRA [36,77], and sliders [23] with both style guidance and LoRA scale used at inference. We also compare against IP Adapters [100] without T2I adapter conditioning, and our Styledrop [86] implementation. In Figure 12, style guidance outperforms the LoRA scale for Ours (blue vs orange), DB LoRA (green vs. pink), and Concept Sliders (brown vs. purple) while our final method still pareto dominates all others, highlighting the effectiveness of both our training scheme and our inference scheme. In Figure 13, our method is still preferred greater than 60 percent of the time against all baselines, while baselines that use style guidance for inference have a higher success rate than those using DB LoRA.

Diversity metric. To measure the overfitting behavior of our method and baselines, we consider a diversity metric. Concretely, we measure the DreamSim [18] perceptual distance between any two images trained with the same style image pair and generated with the same prompt. We then average results over training pairs and prompts. More formally, we let

DreamSim Diversity

(12)

1,i2∈dataS,P DreamSim(i1,i2)

=ES∈S,P∈P Ei

where S is the set of style image pairs, P is the set of prompts, and dataS,P is the set of images generated with prompt P by a model customized on style S. DreamSim(·,·) is DreamSim perceptual distance. A decrease in DreamSim Diversity indicates that all images in a certain domain are becoming perceptually similar, which may indicate overfitting to the style training image. Methods that do not overfit the style training image should have higher diversity scores while also having a low perceptual distance to the ground truth testing style images. We present our findings in Figure 15. Our method is able to achieve a

low perceptual distance to style ground-truth images while maintaining higher diversity scores. As shown in Figure 5 in the main paper, the baseline results mode collapses to the training image, thus lowering their diversity score as they all become perceptually similar to each other.

Style Aligned Image Generation [31] Baseline Style Aligned Image Generation [31] is a recent work for zeroshot style-consistent image generation from an exemplar style image. Given the exemplar style image, it is first inverted to a noise map; then for a new text prompt, the image is generated by attending to both its own self-attention map and the self-attention map from the style exemplar at every denoising step. We compare against this baseline by using the style image in our training image pair as an exemplar and generating a new style image with a new text prompt using this method. Optionally, we condition this generation on the edge map of the newly generated image without attention sharing using ControlNet [102] to help with content preservation. We show the qualitative results of our method compared to all the variants of this baseline in Figure 6 of the main paper. Figures 16 and 17 show quantitative comparison, where our method outperforms this baseline in terms of both style similarity and diversity metric. We achieve lower perceptual distance to the style ground-truth images, low perceptual distance from content images, and high diversity.

Extra Qualitative Evaluation We compare our method with the highest-performing baselines, but use our style guidance (Equation 10 of the main paper ) to apply stylization during inference for these baselines. We present our results in Figure 14. First, we notice that using style guidance for adding style allows the baseline methods to better preserve original content over LoRA scale (Figure 14 vs Figure 5 in the main paper ). While adding our style guidance is better able to preserve content while applying style for baseline methods, our full method is still able to outperform baselines with style guidance applied.

### B. Style Guidance Details

In this section, we derive our style guidance formulation. We consider the probability of latent x with multiple conditionings [8], i.e., the text prompt ct and a class of style images cstyle. First, we apply Bayes’ rule:

P(x|ct,cstyle)

P(x,ct,cstyle) P(ct,cstyle)

=

P(cstyle|ct,x)P(ct|x)P(x) P(ct,cstyle)

=

(13)

Same Category Different Category

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

- Figure 12. Quantitative comparison with baselines on learned style. Given a fixed inference path, our method’s pareto dominates baselines for image generation both on the same category as training (left) and when evaluated on categories different from training, e.g., trained on human portraits but tested on dog images (right). Secondly, our proposed style guidance outperforms standard LoRA weight scale guidance for our training method (blue vs. orange), DB LoRA (green vs. pink), and Sliders (brown vs. purple). Increased marker size corresponds to an increase in guidance scale.

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

- Figure 13. Human preference study. Our method is preferred over the baselines (≥ 60%), including those using our style guidance. The Gray dashed line denotes 50% chance performance. Applying logarithm on both sides, we get:

###### proximate

(xt,ct,style) (16)

###### ∇x log(P(xt|ct,cstyle)) ≈ ϵθ

style

where ct is the original text prompt, cstyle is the class of stylized images from the training style, θstyle is the UNet with style LoRA adapters applied, and ct,style = ‘‘{ct} in <desc> style’’. Here, we use ct to push the prediction in the text direction, and both text conditioning (‘‘in <desc> style’’) and low-rank adapters (θstyle) to push the prediction into the class of images in the artist’s style denoted by cstyle. Following this, our new score estimate is:

log(P(x|ct,cstyle))

= log(P(cstyle|ct,x)) + log(P(ct|x)) + log(P(x)) − log(P(ct,cstyle))

Next, we take the derivative with respect to x:

(14)

∇x log(P(x|ct,cstyle))

=∇x log(P(cstyle|ct,x)) + ∇x log(P(ct|x)) + ∇x log(P(x))

P(cstyle,ct,x) P(ct,x)

= ∇x log

P(ct,x) P(x)

+ ∇x log

+ ∇x log (P(x))

= (∇x log P(cstyle,ct,x) − ∇x log P(ct,x))

+ (∇x log P(ct,x) − ∇x log P(x))

+ (∇x log (P(x)))

(15) As usual, we approximate ∇x (log P(ct,x)) via ϵθ(xt,ct) and ∇x log (P(x)) via ϵθ(xt,∅). Importantly, we ap-

ϵˆθ(xt,ct,cstyle)

=ϵθ(xt,∅) (17) + λcfg(ϵθ(xt,ct) − ϵθ(xt,∅)) (18)

(xt,ct,style) − ϵθ(xt,ct))

+ λstyle(ϵθ

style

(19)

λcfg and λstyle are guidance scales that can be varied as in classifier-free guidance [34]. Given a fixed λcfg, we can vary the λstyle term as desired to generate an original guidance λcfg image with varying amounts of style. Notice that at λcfg = λstyle, the ϵθ(xt,∅) terms cancel and we are left with the original classifier-freeguidance.

### C. Real Image Editing Details

We provide a quantitative and qualitative analysis of our real image editing results compared to baselines. We provide quantitative results in Figure 18. For quantitative evaluation, we use a set of 17 real images from the B-LoRA dataset [17] and add style with all trained style models (see Figure 20 for all training pairs). We measure the perceptual

Pretrained Output Ours Sliders (Style Guid.) DB LoRA (Style Guid.)

|[Figure 254]|[Figure 255]|
|---|---|
| |[Figure 256]|

|[Figure 257]|[Figure 258]|
|---|---|
| |[Figure 259]|

|[Figure 260]|[Figure 261]|
|---|---|
| |[Figure 262]|

|[Figure 263]|[Figure 264]|
|---|---|
| |[Figure 265]|

Artist Created Pairs

|[Figure 266]|
|---|

|[Figure 267]|
|---|

“A photo of a dog in digital art style”

|[Figure 268]|[Figure 269]|
|---|---|
| |[Figure 270]|

|[Figure 271]|[Figure 272]|
|---|---|
| |[Figure 273]|

|[Figure 274]|[Figure 275]|
|---|---|
| |[Figure 276]|

|[Figure 277]|[Figure 278]|
|---|---|
| |[Figure 279]|

Style Image

Content Image

“A photo of a cat in digital art style”

|[Figure 280]|[Figure 281]|
|---|---|
| |[Figure 282]|

|[Figure 283]|[Figure 284]|
|---|---|
| |[Figure 285]|

|[Figure 286]|[Figure 287]|
|---|---|
| |[Figure 288]|

|[Figure 289]|[Figure 290]|
|---|---|
| |[Figure 291]|

|[Figure 292]|
|---|

|[Figure 293]|
|---|

“A dinner plate in drawing style”

|[Figure 294]|[Figure 295]|
|---|---|
| |[Figure 296]|

|[Figure 297]|[Figure 298]|
|---|---|
| |[Figure 299]|

|[Figure 300]|[Figure 301]|
|---|---|
| |[Figure 302]|

|[Figure 303]|[Figure 304]|
|---|---|
| |[Figure 305]|

Content Image

Style Image

“A bowl of soup on a plate in drawing style”

|[Figure 306]|[Figure 307]|
|---|---|
| |[Figure 308]|

|[Figure 309]|[Figure 310]|
|---|---|
| |[Figure 311]|

|[Figure 312]|[Figure 313]|
|---|---|
| |[Figure 314]|

|[Figure 315]|[Figure 316]|
|---|---|
| |[Figure 317]|

###### Synthetic Pairs

|[Figure 318]|
|---|

|[Figure 319]|
|---|

“A photo of a dog in cartoon style”

|[Figure 320]|[Figure 321]|
|---|---|
| |[Figure 322]|

|[Figure 323]|[Figure 324]|
|---|---|
| |[Figure 325]|

|[Figure 326]|[Figure 327]|
|---|---|
| |[Figure 328]|

|[Figure 329]|[Figure 330]|
|---|---|
| |[Figure 331]|

Style Image

Content Image

“A photo of a cat in cartoon style”

|[Figure 332]|[Figure 333]|
|---|---|
| |[Figure 334]|

|[Figure 335]|[Figure 336]|
|---|---|
| |[Figure 337]|

|[Figure 338]|
|---|

|[Figure 339]|
|---|
|[Figure 340]|

|[Figure 341]|[Figure 342]|
|---|---|
| |[Figure 343]|

|[Figure 344]|
|---|

|[Figure 345]|
|---|

“A headshot of a man in painting style”

|[Figure 346]|[Figure 347]|
|---|---|
| |[Figure 348]|

|[Figure 349]|[Figure 350]|
|---|---|
| |[Figure 351]|

|[Figure 352]|
|---|

|[Figure 353]|[Figure 354]|
|---|---|
| |[Figure 355]|

|[Figure 356]|
|---|
|[Figure 357]|

Content Image

Style Image

“A headshot of a woman in painting style”

- Figure 14. Result of our method compared to the strongest baselines, but replacing LoRA scale (Eq. 3 ) with our style guidance (Eq. 10 ) for the baselines. While our style guidance increases baseline performance over LoRA scale images displayed in Figure 5 , our method is still superior in terms of preserving content while applying style.

distance to content and style images with DreamSim [18]. Our method Pareto dominates other baselines, yielding both lower perceptual distance to style images and lower per-

ceptual distance to the original content image. We provide qualitative results against baselines in Figure 19. Our method is able to better preserve the real image structure

Same Category Different Category

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

- Figure 15. Quantitative comparison on Diversity metric. Our method with style guidance has high diversity and low perceptual distance to ground truth style images both on the same category as training (left) and when evaluated on categories different from training, e.g., trained on human portraits but tested on dog images (right). Methods without edge control tend to lose diversity indicating overfitting, and methods with edge control have similar/higher diversity, but much worse style application. Increased marker size corresponds to an increase in style guidance scale.

[Figure 358]

[Figure 359]

[Figure 360]

Same Category Different Category

- Figure 16. Style similarity with Style Aligned [31]. Our method Pareto dominates both versions of Style Aligned Image Generation for image generation both on the same category as training (left) and when evaluated on categories different from training, e.g., trained on human portraits but tested on dog images (right).

[Figure 361]

[Figure 362]

Same Category Different Category

[Figure 363]

- Figure 17. Image diversity with Style Aligned [31] on learned style (Diversity). Our method has high diversity and low perceptual distance to ground truth style images both on the same category as training (left) and when evaluated on categories different from training, e.g., trained on human portraits but tested on dog images (right) as compared to both versions of Style Aligned Image Generation.

Method Hyperparameter value Same Category Different Category

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

Ours (Style Guid.) 3 4 Ours w/ Orthog (Style Guid.) 3 4 DB LoRA (Style Guid.) 2 4 DB LoRA (LoRA Scale) 0.4 0.8 Concept Sliders (Style Guid.) 2 4 Concept Sliders (LoRA Scale) 0.6 0.8 StyleDrop LoRA (LoRA Scale) 0.6 1 IP Adapter w/T2I (Image Guidance) 0.5 0.5 IP Adapter (Image Guidance) 0.5 0.5

Table 1. Experiment Hyperparameters. We choose a fixed stylization hyperparameter for our own model and each baseline when generating images for Mechanical Turk. When picking a hyperparameter, we try and optimize tradeoffs between style application and content preservation, informed by Figure 12. Our style guidance (Equation 10 in the main body ) generally takes values from 0 to λcfg = 5, while all other stylization hyperparameters generally take values 0 to 1.

- Figure 18. Quantitative comparison with baselines on real image editing. Given a fixed classifier-free guidance scale, our method pareto dominates baselines for image generation. During inference, we use style guidance for our method, classifier free guidance for DB LoRA and Sliders, and Image Guidance for IP Adapter w/ T2I. Increased marker size corresponds to an increase in guidance scale.

nique that reduces the number of distinct colors in a given image to some fixed number N, reducing color variation and creating fixed color areas. We apply posterization to images in our training set with N = 8.

while applying the style from the training pair. We invert the noise to 600 steps for our method and 700 steps for baselines, as we find experimentally that baselines do not apply style when real images are only inverted to 600 steps. We use ReNoise Inversion [24] for our DDIM inversion implementation on SDXL.

Training data. We present our full training set of 20 different style transformations in Figure 20. Each image pair is a standalone training instance used in our method. We consider four different styles (posterization, impressionist, neural painting, cartoonization), with each column corresponding to a single style. For each style, we consider five categories for training (man, woman, dog, cat, landscape).

### D. Implementation Details

Style descriptor Ablation We consider replacing a text style descriptor (i.e., ”digital art style”) with a random rare token (i.e., ”S∗ style”). Applying the same style strength, we quantitatively find that using words yields better distance to content and GT style image (0.340±0.03, 0.194± 0.03) than using rare tokens (0.348 ± 0.04, 0.209 ± 0.03), averaged over all test cases. These experiments confirm the remark in StyleDrop [86] that using a text style descriptor produces better results.

We describe the specific methods used to create the paired dataset in detail.

LEDITS++ [7] is a diffusion-based image editing technique that transforms an image by updating the inference path of a diffusion model. After fine-grained inversion, a global prompt and a set of translation prompts representing a new style or object are used to perform the image translation. We leverage LEDITS++ on all images with the translation prompt “Impressionist style”. Further, we change the word “photo” to “painting” in the original prompt when generating the style image.

Mechanical Turk details. When running Amazon Mechanical Turk, we prompt users with an analogy-style interface. First, we provide the training pair of images, followed by the testing content image, and two options for possible styled examples. After viewing both images, users choose either the left or right image. Figure 21 shows an example. Each individual user is presented with four training examples, as in Figure 21, followed by 16 random testing examples comparing our method with one of our baselines. We survey 75 users for each of the 16 individual studies and use bootstrapping to obtain variance estimates. In total, we collect 19200 user samples. For each method, we pick a stylization hyperparameter based on Figure 12 . For details, see Table 1

White-box cartoonization. Cartoonization [98] is a GAN-based image-to-image translation technique that applies a cartoon-like effect to real images. We apply the cartoonized model to our set of generated images to create image pairs.

Stylized neural painting. Stylized Neural Painting [106] is a rendering based image to image translation technique where an image is reconstructed via N painting strokes, where the strokes are guided by a loss function that encourages the final translated image to resemble the original. We use the Neural Painting model with N = 1000 to create image pairs.

Posterization. Posterization is an image filtering tech-

[Figure 364]

[Figure 365]

Real Image

Style Pair

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

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

Concept Sliders

IP-Adapter (with T2I)

Concept Sliders

IP-Adapter (with T2I)

Ours

DB-LoRA

Ours

DB-LoRA

- Figure 19. Real image editing comparison. We compare our real image editing results to other baselines. Our method is able to best preserve content from the real image while applying style, across both training style pairs and input images. We set classifier-free guidance to 5 in all cases. For our method, we set our style guidance (Eq. 10 in the main body) to 6. For baselines, we set LoRA scale (Eq. 3 in the main body) to 1. We invert all images using DDIM inversion [24,87].

[Figure 384]

- Figure 20. Training Data. We present the synthetic training data set used for evaluation, where each pair is used as a single training instance. Each column corresponds to a different style, and each row corresponds to a different content category.

[Figure 385]

###### Figure 21. Mturk User Interface

