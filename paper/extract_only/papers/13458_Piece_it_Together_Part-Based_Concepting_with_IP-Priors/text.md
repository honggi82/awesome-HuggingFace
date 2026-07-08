## arXiv:2503.10365v1[cs.CV]13Mar2025

### Piece it Together: Part-Based Concepting with IP-Priors

Elad Richardson1 Kfir Goldberg1,2 Yuval Alaluf1 Daniel Cohen-Or1 1Tel Aviv University 2Bria AI

##### Character Ideation Product Design Toy Concepting

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

[Figure 16]

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

Figure 1. Using a dedicated prior for the target domain, our method, Piece it Together (PiT), effectively completes missing information by seamlessly integrating given elements into a coherent composition while adding the necessary missing pieces needed for the complete concept to reside in the prior domain.

###### Abstract

###### 1. Introduction

Advanced generative models excel at synthesizing images but often rely on text-based conditioning. Visual designers, however, often work beyond language, directly drawing inspiration from existing visual elements. In many cases, these elements represent only fragments of a potential concept—such as an uniquely structured wing, or a specific hairstyle—serving as inspiration for the artist to explore how they can come together creatively into a coherent whole. Recognizing this need, we introduce a generative framework that seamlessly integrates a partial set of userprovided visual components into a coherent composition while simultaneously sampling the missing parts needed to generate a plausible and complete concept. Our approach builds on a strong and underexplored representation space, extracted from IP-Adapter+, on which we train IPPrior, a lightweight flow-matching model that synthesizes coherent compositions based on domain-specific priors, enabling diverse and context-aware generations. Additionally, we present a LoRA-based fine-tuning strategy that significantly improves prompt adherence in IP-Adapter+ for a given task, addressing its common trade-off between reconstruction quality and prompt adherence. Project page can be found at https://eladrich.github.io/PiT/

“To create is to recombine”

– Fran¸cois Jacob

Recent advancements in image generation have significantly enhanced our ability to prototype and visualize new ideas. However, while modern generative models achieve impressive results, they are still typically conditioned on text, assuming that concepts can be fully articulated through language. In practice, however, artists and designers often work visually — drawing from references, reconfiguring elements, and refining compositions in ways that cannot always be expressed through text alone [19]. This has led to the development of techniques for conditioning generative models directly on image conditions [51, 52, 76]. However, these methods alone offer limited control on how the visual concepts influence the generated results. Recognizing this, recent works have explored more advanced manipulation of visual concepts using generative models [11, 26, 54, 57, 65], offering a more interactive and intuitive creative process.

Expanding on this line of work, we ask: Can we simultaneously assemble given visual components and sample plausible completions for missing parts to create a coherent whole? To this end, we propose a model that dynamically adapts to user inputs, assembling provided elements into a coherent structure while inferring missing components in a manner consistent with the provided context.

The choice of representation space for our model inputs is crucial. Inspiration Tree [65] and ConceptLab [55] use learnable tokens within the text encoder’s input space [15]. While effective for their respective tasks, this optimizationbased approach would hinder our ability to use our method efficiently at inference time. Alternatively, pOps [54] and IP-Composer [11] operate in CLIP space. This allows

- them to efficiently encode visual concepts via a pretrained CLIP encoder and is well suited for semantic manipulations. However, the CLIP space is limited in its ability to preserve complex concepts, resulting in a loss of details [51, 54]. This intuitively stems from the fact that CLIP was never trained to reconstruct images but rather to learn a joint representation space for text and images [50]. While this encourages a semantic representation, it does not require the representation to encode visual details that cannot be easily described through text. To improve on this, we explore alternative spaces and ultimately converge on the internal representation of IP-Adapter+. This adapter extends the popular model from [76]. While not formally described in a paper, it has gained popularity for its improved reconstruction quality. We show that using this IP+ space not only results in improved reconstructions but also retains the ability to perform semantic manipulations and thus can serve as a new representation for visual concepts, see Figure 2.

With our chosen representation in place, we turn to training a part-conditioned model on a set of generated samples from a given target domain. We train our model to sample both conditionally and unconditionally from that domain while interpreting any given input within that context. For instance, a model tuned on monsters and creatures would always generate a creature from that domain. Importantly, this strong prior allows the artist to reinterpret everyday objects as potential parts within the learned domain (e.g., the broccoli input in Figure 1) and sample multiple plausible results for the same set of inputs. In essence, the trained model captures a prior distribution over the target domain in the IP+ space and is therefore dubbed an IP-Prior model.

Finally, once a complete visual concept has been generated using an IP-Prior, it can be rendered as an image by passing it to a pretrained image generation model [49]. Ideally, at this stage, we would additionally like to introduce additional conditions — for example, incorporating a text prompt to place the generated concept within a specific scene. Unfortunately, a key limitation of IP-Adapter+ is its inherent trade-off between reconstruction quality and prompt adherence. We hypothesize this arises from the expressiveness of the IP+ space, which makes the text conditioning redundant during fine-tuning, and propose a simple LoRA-based mechanism for re-enabling text conditioning on the generated concepts. Together, this results in PiT, a flexible pipeline that first facilitates concept ideation and

- then renders those concepts as high-quality images.

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

CLIP+IP

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Input Scrawny → Muscular

Figure 2. Semantic Manipulation in CLIP Space vs. IP+ Space. We encode the input image (left) into two different embedding spaces, modify its latent representation by traversing each space, and render the edited image using SDXL [49]. As shown, CLIP struggles to both reconstruct the concept and follow the desired edit, whereas in IP+ space, the rendered images are faithful both to the concept and the desired edit across the entire range.

###### 2. Related Work

Image-Conditioned Generation. While text has become the de facto interface for generating visual content [4, 10, 39, 47, 51, 56, 60, 61, 78], its ability to describe specific visual concepts can be limited. As a result, various approaches have been proposed to incorporate image inputs into generative models. These techniques can be roughly grouped based on the mechanism they use to incorporate the visual information into the generative network.

In optimization-based personalization methods, a single visual concept is encoded into the text embedding inputs [1, 16, 66] or directly into the network weights [31, 58, 64]. The model can then generate images of the given concept under different prompts. Numerous efforts have been made to turn these personalization methods optimizationfree. These methods train an encoder to directly map a concept into some form of compact representation that is then passed to the model [2, 17, 43, 59, 62, 68, 77, 79]. Specifically, in [76] the input image is encoded through CLIP and passed to a set of decoupled cross-attention layers. This has proven to be an effective conditioning mechanism that has gained popularity in the community.

Multi-Concept Generation. Successfully depicting multiple concepts in a single image remains a challenging task, even in text-based generation [8, 9, 21, 35, 42, 46]. This challenge becomes even more pronounced in imageconditioned generation. Consequently, a substantial body of work has focused on improving conditioning techniques to support multiple inputs. In essence, these methods build upon existing image-conditioning approaches, extending them to better support multiple visual constraints.

Some approaches aim to solve an optimization problem resulting in a disentangled representation for each visual concept [3, 18, 31]. Another line of work leverages LoRA [25] to learn new concepts and aims to apply multiple LoRA modules in conjunction [20, 30, 48, 75], often through spatial maps or localized prompts. Others follow

the encoder-based approaches and inject a representation of the visual information into the model while accounting for the different objects [22, 42, 44, 70]. Alternatively, numerous works utilize the advancement in the research on vision-language models [34, 37, 74] to define multimodal prompts, interleaving multiple reference images with an input text prompt [41, 71, 72, 81].

Crucially, the aforementioned works primarily focus on embedding complete objects into a scene where all elements are predefined, eliminating the need for the model to generate novel content. In contrast, our approach generates new concepts from partial user-provided parts, requiring the model to both integrate the given inputs and infer missing details to form a single, coherent concept.

Visually Inspired and Creative Generation. The exploration of human creativity within computer graphics has been a significant area of research, with numerous works investigating how computational tools can enhance the creative design process [13, 14, 23, 28, 40, 67, 73]. Interestingly, a fundamental aspect of creativity is the ability to leverage prior knowledge to generate novel ideas [6, 12, 69]. Aligned with these observations, recent works have started

- to explore how strong generative models can serve to inspire creative generation and exploration.

More specifically, Inspiration Tree [65] showed that one can decompose a visual concept into distinct attributes, organizing them hierarchically in a tree structure by building on the literature of concept personalization. This was extended in [33], which learns disentangled concept representations along language-informed axes. In the context of creative generation, ConceptLab [55] uses guidance from a VLM to learn a new token embedding representing novel concepts within a given broad category. However, while their approach leverages VLMs for the creative generation process, they do not offer control over the visual attributes of the learned concept.

In the context of generating images inspired by multiple visual concepts, it has been shown that generative models that are conditioned on the CLIP image embeddings, such as IP-Adapter [76] or Kandinsky [61], allow one to use the CLIP embedding space to manipulate and compose different concepts together [11, 44, 54]. More specifically, in pOps [54], a Diffusion Prior model leveraging the CLIP space is used to learn semantic operators (e.g., union, texturing) over given input concepts. Similarly, IP-Composer [11] enables compositional image generation by extracting and integrating visual concepts from multiple reference images. However, while CLIP provides a semantically meaningful space, it often struggles to accurately reconstruct desired concepts, limiting its usability. We show that the proposed usage of the IP+ space overcomes this limitation while still allowing for semantically meaningful manipulations of image embeddings.

###### 3. Preliminaries

Diffusion Prior. Diffusion models are typically trained with a conditioning vector, c, directly derived from a given text prompt, y. Ramesh et al. [51] introduce a two-stage approach that decomposes the text-to-image generation process into two steps.

First, a diffusion model is trained to generate an image conditioned on an image embedding, c, using the standard diffusion loss:

Ldiffusion = Ez,y,ε,t ||ε − εθ(zt,t,c)||22 . (1)

Here, the denoising network, εθ, learns to remove noise ε added to the latent code zt at time step t, given the conditioning image embedding c.

Next, the Diffusion Prior model, Pθ, is trained to recover a clean image embedding, e, from its noisy version, et, conditioned on the corresponding text prompt, y. The objective function is given by:

Lprior = Ee,y,t ||e − Pθ(et,t,y)||22 . (2)

Once both models are trained separately, they are combined into a full text-to-image generation pipeline. In this work, we extend the Diffusion Prior beyond its conventional role of predicting image embeddings from text. Instead, we adapt it to operate over multiple image parts, enabling finergrained control over the synthesis process.

###### 4. Method

We begin by laying the foundation of Piece-it-Together (PiT), detailing the representation space in which our IPPrior operates. Next, we describe the design and training process of our generative model within this chosen representation space. Finally, we demonstrate how the generated concepts can be integrated with pretrained image generation models alongside existing conditions.

###### 4.1. In Search of a Space

The CLIP space is widely used as an image representation for image-conditioned generation and manipulation tasks [11, 54, 61, 77]. Beyond its alignment with the corresponding text encoding, CLIP has several compelling advantages, including an efficient image encoder and a semantically rich representation. However, as originally highlighted in [51], the CLIP space struggles to encode distinct visual patterns and is prone to attribute leakage. This limitation stems from CLIP’s training objective, which encourages the model to learn a joint text-image representation but does not explicitly require it to capture fine-grained visual details that cannot be easily described through text. While this design choice is reasonable for CLIP’s intended purpose, it poses clear challenges for image-conditioned tasks.

[Figure 39]

[Figure 40]

[Figure 41]

#### IP-A+

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

. . .

SDXL

[Figure 49]

[Figure 50]

[Figure 51]

Query Embeddings

| |
|---|

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

| |
|---|

Perceiver

| |
|---|

| |
|---|

. . .

[Figure 59]

| |
|---|

[Figure 60]

[Figure 61]

[Figure 62]

...

[Figure 63]

| |
|---|

[Figure 64]

###### IP-Prior

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

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

. . .

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

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

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

| |
|---|

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

. . .

. . .

. . .

| |
|---|

| |
|---|

| |
|---|

. . .

[Figure 98]

CLIP

[Figure 99]

Noised 𝐼𝑃 Embedding

[Figure 100]

[Figure 101]

[Figure 102]

IP-A+

###### IP-A+ IP-A+

|[Figure 103]<br><br>[Figure 104]| | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

|[Figure 105]<br><br>[Figure 106]|
|---|

|[Figure 107]<br><br>[Figure 108]|
|---|

|[Figure 109]<br><br>[Figure 110]|
|---|

- Figure 3. Piece-it-Together Overview. Given an input image, we extract its semantic components (e.g., using SAM [29]) and encode each image patch into the IP+ space using frozen IP-Adapter+ (IP-A+) blocks (shown in yellow). The resulting set of compact image embeddings are then passed together through our IP-Prior model (green), which also receives a noised image embedding representing our desired complete concept. The IP-Prior model outputs a cleaned image embedding that captures the intended concept, which is subsequently used to generate the final concept image using SDXL [49] (blue). At inference time, users can provide a varying number of object-part images to generate a new concept that aligns with the learned distribution.

###### 4.2. How to Train Your IP-Prior

Recognizing that the performance of our method is constrained by the reconstruction capabilities of its visual representation, we must first search for an alternative representation that remains compact and inherently semantic while also offering improved reconstructions. To this end, we find that the internal representation of IP-Adapter+ meets these criteria and offers a compelling set of properties. This model was released as an extension of the original IP-Adapter [77] and gained popularity in the community for image-conditioned generation. While the original IPAdapter utilizes the already compact CLIP embedding as input, as illustrated in Figure 3, IP-Adapter+ employs a Perceiver-like architecture [27] operating over the full internal representation of the CLIP model. This approach produces a representation of 16 × 2048 vectors, explicitly optimized with reconstruction in mind. The resulting representation is then fed into a pretrained generative model via a set of trainable attention layers.

A key requirement for our model is the ability to generate multiple plausible outputs for the same set of input parts. This enables designers to explore a range of variations, a crucial property for ideation. To achieve this, we follow pOps [54] and design our model as a generative operator acting on image embeddings. During training, the model takes a sequence of visual concepts representing object parts, each encoded as IP+ vectors, and learns to produce a representation of the complete object, see Figure 3.

Deviating from [54], which fine-tuned a pretrained prior model for the desired generative operator, our approach cannot rely on a pretrained prior, as no existing model has been trained on IP+ vectors. ‘This required us to build and train our network from scratch. Through empirical exploration, we chose to train a 4-block Diffusion Transformer (DiT) [45] using rectified flow [36] instead of the standard denoising loss used in [54, 61]. This approach provides an efficient training on a single GPU while keeping the model lightweight for both training and inference.

While IP-Adapter+ is usually used end-to-end as a method for image conditioning, we propose to conceptually decouple this process into two parts: first, encoding images into a compact representation, and second, conditioning the generative model on this representation. We then demonstrate that this intermediate encoding, which we refer to as the IP+ space, is a semantic representation space with significantly improved reconstructions compared to the standard CLIP image embedding and can serve as an effective representation for our generative model. As we will show, operating in a compact latent space rather than directly on pixels makes training efficient and allows for easy manipulation of the generated concepts.

###### 4.3. Defining a Prior via Data

Having demonstrated how to train a prior given a corresponding dataset, we now turn to the data-gathering process. While any visual dataset could be used to train an IPPrior, we find that leveraging generated data offers a more flexible and efficient approach. Our data generation process is based on the assumption that modern text-based generative models can effectively sample from specific domains. For instance, in a character ideation workflow, one could easily use the prompt “Concept art of an imaginary creature,

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

IP-A+

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

...

[Figure 124]

[Figure 125]

###### “a monster in a bowling alley”

[Figure 126]

[Figure 127]

Text Encoder

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

.

[Figure 137]

.

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

SDXL

(+LoRA)

[Figure 144]

- Figure 4. Generated Data Samples. We present sample images generated using FLUX-Schnell [5], which are used to train our IPPrior model for the “creatures” domain.

Figure 5. Recovering the Text Adherence via IP-LoRA. IPAdapter+ enables rendering generated concepts via SDXL [49] but often struggles with text adherence. To address this, we fine-tune a LoRA adapter over paired examples, where the conditioning image has a clean background and the target image places the object in a scene described using a text prompt. This lightweight training (using just 50 prompts) effectively restores text control while maintaining visual fidelity.

white background” to generate diverse character concepts. That is, the domain is often well-within the text-based generative model. However, text offers a very specific form of control over the sampled results, making the exploration process tedious.

Building on this observation, we first generate a large set of images using a pretrained text-to-image model, specifically Flux-Schnell [5]. To enhance the diversity of the generated dataset, we randomly append relevant adjectives to the base prompt (e.g., “shadowy”, “cryptic”, “magnificent”). A sample training domain generated using this technique is shown in Figure 4. To create the input pairs, we extract semantic parts from the target images using a generalpurpose segmentation method [29] and randomly sample a subset of them. This process encourages the model to better solve the target task — spatially assembling the provided visual concepts while generating the missing parts — all within the context of the learned prior. Interestingly, since we operate in an embedding space, we do not encounter the overfitting issues reported in relevant literature for other tasks when using a simple segmentation-based data generation process [63].

arises from the expressiveness of the IP+ space, which makes the text conditioning redundant during fine-tuning, causing the model to ignore it in inference. Despite this, we argue that since the model functions as an adapter, its text understanding capabilities are still “hidden” inside the model and simply need to be “reactivated”.

To re-enable text conditioning, we show that training a LoRA adapter [25] on a small set of examples can effectively restore this capability. In these examples, the conditioning and target images are not identical. Instead, the conditioning image depicts the object on a clean background, while the target image places the object in a new scene described by a text prompt. We refer to this training setup as IP-LoRA and illustrate it in Figure 5. This setup forces the model to incorporate the text prompt while preserving the visual characteristics of the target concept. Remarkably, even when trained on just 50 text prompts, our model generalizes well to unseen prompts describing different backgrounds, showing that the LoRA adapter enables the model to reuse its existing text comprehension abilities. Furthermore, we show that this mechanism can also be leveraged to personalize the model for specific subdomains based on the given image embedding. For example, it can generate an image depicting a full character sheet from a single generated concept.

###### 4.4. Unleashing Text Adherence with IP-LoRA

Once a complete visual concept has been generated using an IP-Prior, it can be rendered as an image by passing it to a pretrained image generation model [49] via IP-Adapter+. When rendered as-is, the generated concept closely matches its representation in the training data, which in our case often features a simple, clean background. Ideally, however, at this point, one would want to explore how the concept fits into different scenarios and scenes. A natural approach to achieve this is to leverage the text-conditioning of the pretrained image model alongside our outputted IP vector. In practice, however, when using IP-Adapter+, the model tends to exhibit low prompt adherence, see Section 5.2. A common way to address this is by adjusting the scaling factor of the adapter outputs, but this comes at the cost of reconstruction quality, leading to a suboptimal trade-off between fidelity and controllability. We hypothesize this

###### 4.5. Implementation Details

Our IP-Prior consists of approximately 270M parameters. Training is performed with a batch size of 64 for 500K steps, requiring 30GB of RAM on a single GPU. The IPLoRA training is conducted using the AdamW optimizer with a learning rate of 1 × 10−4 for 10K steps. During inference, the IP-Prior model runs for 25 steps, while SDXL uses 50 steps. To enhance fine details, we apply SDEdit [38] over Flux-Dev [5] with a strength of 0.35.

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

CharactersToysProducts

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

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

Figure 6. PiT Results for Different Priors. We show results generated by our approach across three different domains. For each result, we use a varying number of input parts and generate multiple plausible outputs by altering the seed used for learning the representation.

###### 5. Experiments

###### 5.1. IP-Prior Results

Qualitative Results. In Figure 6, we present the results of PiT across three different priors. During inference, our model receives a sequence of tokens of possibly varying lengths and generates a corresponding output for each seed. The parts themselves are gathered from online sources as well as from newly generated images. As shown, our model successfully recognizes the semantic meaning of each given part and integrates it coherently into the generated result. Notably, this process is entirely free of any additional text supervision, requiring the user to provide the desired object. Furthermore, beyond merely integrating the given parts, the model effectively generates meaningful and coherent completions to the missing information, ultimately producing an in-domain result.

Note that, as expected, the model exhibits greater variability when provided with fewer inputs, as there is more room for interpretation. This behavior is further highlighted in Figure 7, where we present results generated from a single conditioning image. Here, the outputs vary significantly from one another, aligning with the intended use case in which an artist iteratively refines their target concept.

To further highlight our model’s ability to interpret the same elements in different ways based on the trained IPPrior, we show results in Figure 8 using multiple different priors for the same set of inputs. As shown, each model

[Figure 177]

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

Input Sampled Results

Figure 7. Sampling From a Single Input. We present concepts generated by PiT using a single input part, which encourages greater variation across the generated results.

correctly interprets the provided elements and seamlessly integrates missing components to produce a complete generation. In the “portrait” domain, the model is trained on “Found Object Portraits” and in the ”Duck” domain, the model is trained on customized rubber ducks. Notably, for the “portrait” domain, the model must interpret each given input as part of a facial structure. For instance, in the first row, the hair is interpreted as an eyebrow.

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

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

Input Character Product Ducks Portraits

- Figure 8. Sampling Across Different Priors Given a single input part, we generate concepts across different learned IP-Prior models, highlighting how each model naturally interprets and adapts the part according to its learned distribution.

Cute→

Scary Frosty→

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Blazing Scrawny→

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

Muscular

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

- Figure 9. Semantic Manipulations in IP+ Space. Using embeddings learned by our IP-Prior, we apply semantic manipulations to edit the concept before rendering the image with SDXL.

Semantic Manipulations in IP+. One of the key advantages of working in embedding spaces for part-based generation is the ability to semantically edit and manipulate generated concepts, further enriching the ideation process. We demonstrate this in Figure 9, where we apply various semantic transformations within the IP+ space. Each edit direction is found by generating 50 samples corresponding to a pair of contrasting words (e.g., “cute” and “scary”), embedding them into our space, and computing the vector direction between the mean embeddings of each set. This simple yet effective approach enables meaningful semantic edits, further highlighting both the potential of working in the IP+ space and the broader benefit of working with embeddings in PiT.

###### 5.2. IP-LoRA Results

In the previous section, all results were rendered as-is using the original IP-Adapter+ over SDXL. As discussed, a desirable property of our approach is the ability to take a generated concept and integrate it into different scenes or styles, enabling a continuous and flexible creative process.

Input PiT IP-A (0.2) IP-A (0.4) IP-A (0.6)

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

“...in space with the milky way behind him”

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

“...in the snow”

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

“...in a lab”

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

“...in a bowling alley”

Figure 10. IP-Lora vs. IP-Adapter+. Given a learned representation, we show results of rendering the concept in a new scene.

Table 1. Quantitative Comparison. We show the average text and visual scores on a scale of 1-5, computed using Qwen [74].

PiT IP-A (0.2) IP-A (0.4) IP-A (0.6) IP-A (0.8)

Text Score ↑ 3.60 3.70 2.45 1.37 1.06 Visual Score ↑ 4.55 2.41 3.95 4.64 4.84

Text-Conditioning. We first present results using an IPLoRA trained to restore text conditioning for describing a background in which we aim to place our generated concept, as shown in Figure 10. In this setup, SDXL is conditioned both on the standard text prompt and on concepts generated using our method, which are passed as encodings to the adapter. As shown, our method effectively balances the text conditioning while remaining faithful to the concept. For comparison, we include the original behavior of IP-Adapter+. When using strong adapter scales (e.g., 0.6), the model completely disregards the text and simply reconstructs the given image embedding. Lowering the scaling value allows the model to incorporate the textual information but significantly degrades reconstruction quality.

We now validate this claim quantitatively. A common metric for measuring text and image similarity is CLIPspace similarity [24], but it is unsuitable for our evaluation, as we aim to preserve information beyond what CLIP embeddings can capture. Instead, we use Qwen 2 [74]. Specifically, we prompt the VLM to rate each image on a scale of 1 to 5 based on its adherence to the given text and its similarity to the reference image. The results, presented in Table 1, align with the visual observation, demonstrating the effectiveness of the IP-LoRA approach.

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

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

IP-Adapter+OmniGenPiT-ECLIPSEλ

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

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

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

Figure 11. Qualitative Comparisons. We provide visual comparisons with alternative methods across various domains.

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

- Figure 12. Style Generation. We train a LoRA to generate character reference sheets when given a concept embedding.

Styled Generations. Another common use case for model fine-tuning is personalization to specific styles or domains [15, 58]. Similarly, in Figure 12, we present a LoRA trained to generate character reference sheets when conditioned on the concept embedding of a given character, where paired data is obtained by extracting a sample from the reference sheet and using its embedding as conditioning.

###### 5.3. Comparisons

Given the nature of our task, there is no direct method that tackles the same task. Still, to offer an analysis of PiT with respect to the state-of-the-art, we compare it to a representative set of multi-image baselines, testing their generalization. More specifically, we consider OmniGen [71], a strong multi-modal model, λ-ECLIPSE [44] as a priorbased technique for multi-image compositions, and IPAdapter+, modified to operate over multiple images by aggregating the generated embeddings.

We present results in Figure 11 across multiple concept domains. As expected, the naive approach of averaging the generated embeddings and using IP-Adapter+ blends the input components together, producing a hybrid of their visual characteristics rather than preserving distinct parts. Similarly, λ-ECLIPSE may capture the target domain (e.g., ducks) and the color palette of the input components, but it fails to integrate the specified parts into the output. Finally, OmniGen, exhibits inconsistent behavior across domains: in some cases, it omits certain parts, while in others, it strictly preserves their spatial structure, preventing it from semantically assembling the parts together, as shown in the third column. In contrast, our approach effectively incorporates the provided parts while naturally completing the missing information based on the target domain.

###### 5.4. Additional Inputs

While our primary focus is on object parts as input visual concepts, our approach is not inherently limited to this specific semantic meaning. As an example, in Figure 13, we show that our model can also be conditioned on a grid-like arrangement of reference images, describing some form of visual style. This allows one to visually describe a target look for the generated concept alongside the target parts.

We additionally demonstrate in Figure 14 that PiT can also be conditioned on sketches depicting parts of an object, providing users with greater flexibility when specific visual parts are unavailable. In this case, the model interprets each sketch based on the learned prior.

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

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

Part Style Sampled Results

- Figure 13. PiT results with reference conditioning. The model is conditioned on both image parts as well as a grid representing a target look.

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

[Figure 324]

[Figure 325]

[Figure 326]

Input Sampled Results

- Figure 14. PiT results for sketch conditioning. The model is conditioned on sketch images, allowing one to specify rough shapes without color.

###### 5.5. Limitations

It is important to note that, as with any embedding-based method, the amount of information encoded in the compact embedding space is inherently limited. While our use of the IP+ space significantly improves the tradeoff between reconstruction and editability — mitigating prominent issues such as attribute mixing discussed in [51, 54] — not all information can be successfully preserved. For example, the model struggles to encode small or high-frequency details, limiting our ability to condition on fine-grained regions of the target concept or small text. Nevertheless, as we demonstrate, embedding-based mechanisms enable simple manipulations that would otherwise be difficult to achieve. Thus, for our ideation task, we find IP+ to be a well-balanced choice.

###### 6. Conclusions

In this work, we introduced PiT, a method for ideating new concepts from a sparse set of input parts. Our approach functions as a generative operator that receives a set of part embeddings and produces an embedding representing a plausible and complete concept within a given target domain. PiT addresses core limitations of current CLIP-based methods by operating in the more expressive IP+ space, enabling improved concept reconstructions while allowing for meaningful semantic manipulations. We hope that PiT not only serves as a strong model for photo-inspired generation and ideation but also provides a foundation for solving additional tasks that would benefit from operating within the same representation space.

###### Acknowledgements

We thank Or Lichter and Yael Vinker for their feedback and insightful comments. This research was supported in part by the Israel Science Foundation (grants no. 2492/20 and 1473/24), Len Blavatnik and the Blavatnik family foundation.

###### References

- [1] Yuval Alaluf, Elad Richardson, Gal Metzer, and Daniel Cohen-Or. A neural space-time representation for text-toimage personalization, 2023. 2
- [2] Moab Arar, Rinon Gal, Yuval Atzmon, Gal Chechik, Daniel Cohen-Or, Ariel Shamir, and Amit H. Bermano. Domainagnostic tuning-encoder for fast personalization of text-toimage models. In SIGGRAPH Asia 2023 Conference Papers, pages 1–10, 2023. 2
- [3] Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel CohenOr, and Dani Lischinski. Break-a-scene: Extracting multiple concepts from a single image. In SIGGRAPH Asia 2023 Conference Papers, pages 1–12, 2023. 2
- [4] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala,

- Timo Aila, Samuli Laine, Bryan Catanzaro, Tero Karras, and Ming-Yu Liu. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers, 2023. 2
- [5] Black-Forest. Flux: Diffusion models for layered image generation, 2024. Accessed: 2024. 5, 13
- [6] Nathalie Bonnardel and Evelyne Marm`eche. Towards supporting evocation processes in creative design: A cognitive approach. International journal of human-computer studies, 63(4-5):422–435, 2005. 3
- [7] Bria-AI. Bria background removal v2.0, 2024. Accessed:

2024. 13

- [8] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models, 2023. 2
- [9] Omer Dahary, Or Patashnik, Kfir Aberman, and Daniel Cohen-Or. Be yourself: Bounded attention for multi-subject text-to-image generation. arXiv preprint arXiv:2403.16990,

2024. 2

- [10] Ming Ding, Wendi Zheng, Wenyi Hong, and Jie Tang. Cogview2: Faster and better text-to-image generation via hierarchical transformers. Advances in Neural Information Processing Systems, 35:16890–16902, 2022. 2
- [11] Sara Dorfman, Dana Cohen-Bar, Rinon Gal, and Daniel Cohen-Or. Ip-composer: Semantic composition of visual concepts. arXiv preprint arXiv:2502.13951, 2025. 1, 2, 3
- [12] Claudia Eckert and Martin Stacey. Sources of inspiration: a language of design. Design studies, 21(5):523–538, 2000. 3
- [13] Mohamed Elhoseiny and Mohamed Elfeki. Creativity inspired zero-shot learning. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5784– 5793, 2019. 3
- [14] Philippe Esling and Ninon Devis. Creativity in the era of artificial intelligence. arXiv preprint arXiv:2008.05959, 2020. 3
- [15] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 2, 8
- [16] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In The Eleventh International Conference on Learning Representations, 2023. 2
- [17] Rinon Gal, Moab Arar, Yuval Atzmon, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. Encoder-based domain tuning for fast personalization of text-to-image models,

2023. 2

- [18] Daniel Garibi, Shahar Yadin, Roni Paiss, Omer Tov, Shiran Zada, Ariel Ephrat, Tomer Michaeli, Inbar Mosseri, and Tali Dekel. Tokenverse: Versatile multi-concept personalization in token modulation space, 2025. 2
- [19] Milene Gonc¸alves, Carlos Cardoso, and Petra BadkeSchaub. What inspires designers? preferences on inspirational approaches during idea generation. Design studies, 35

(1):29–53, 2014. 1

- [20] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, Yixiao Ge, Ying Shan, and Mike Zheng Shou. Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models, 2023. 2
- [21] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized lowrank adaptation for multi-concept customization of diffusion models. Advances in Neural Information Processing Systems, 36:15890–15902, 2023. 2
- [22] Shaozhe Hao, Kai Han, Zhengyao Lv, Shihao Zhao, and Kwan-Yee K Wong. Conceptexpress: Harnessing diffusion models for single-image unsupervised concept extraction. In European Conference on Computer Vision, pages 215–233. Springer, 2024. 3
- [23] Aaron Hertzmann. Can computers create art? In Arts, page 18. MDPI, 2018. 3
- [24] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning, 2022. 7
- [25] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 2, 5
- [26] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. arXiv preprint arXiv:2302.09778, 2023. 1
- [27] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International conference on machine learning, pages 4651–4664. PMLR, 2021. 4
- [28] Anna Kantosalo, Jukka M Toivanen, Ping Xiao, and Hannu Toivonen. From isolation to involvement: Adapting machine creativity software to support human-computer co-creation. In ICCC, pages 1–7, 2014. 3
- [29] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 4, 5, 13
- [30] Zhe Kong, Yong Zhang, Tianyu Yang, Tao Wang, Kaihao Zhang, Bizhu Wu, Guanying Chen, Wei Liu, and Wenhan Luo. Omg: Occlusion-friendly personalized multi-concept generation in diffusion models. In European Conference on Computer Vision, pages 253–270. Springer, 2024. 2
- [31] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. 2023. 2
- [32] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. Interna-

- tional journal of computer vision, 128(7):1956–1981, 2020. 13
- [33] Sharon Lee, Yunzhi Zhang, Shangzhe Wu, and Jiajun Wu. Language-informed visual concept learning. In The Twelfth International Conference on Learning Representations, 2024. 3
- [34] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 3

- [35] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22511–22521, 2023. 2
- [36] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 4
- [37] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 3
- [38] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 5
- [39] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 2
- [40] Jonas Oppenlaender. The creativity of text-to-image generation. In Proceedings of the 25th International Academic Mindtrek Conference. ACM, 2022. 3
- [41] Xichen Pan, Li Dong, Shaohan Huang, Zhiliang Peng, Wenhu Chen, and Furu Wei. Kosmos-g: Generating images in context with multimodal large language models. arXiv preprint arXiv:2310.02992, 2023. 3
- [42] Gaurav Parmar, Or Patashnik, Kuan-Chieh Wang, Daniil Ostashev, Srinivasa Narasimhan, Jun-Yan Zhu, Daniel CohenOr, and Kfir Aberman. Object-level visual prompts for compositional image generation. arXiv preprint arXiv:2501.01424, 2025. 2, 3
- [43] Or Patashnik, Rinon Gal, Daniil Ostashev, Sergey Tulyakov, Kfir Aberman, and Daniel Cohen-Or. Nested attention: Semantic-aware attention values for concept personalization. arXiv preprint arXiv:2501.01407, 2025. 2
- [44] Maitreya Patel, Sangmin Jung, Chitta Baral, and Yezhou Yang. λ-eclipse: Multi-concept personalized text-to-image diffusion models by leveraging clip latent space, 2024. 3, 8
- [45] William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023. 4
- [46] Quynh Phung, Songwei Ge, and Jia-Bin Huang. Grounded text-to-image synthesis with attention refocusing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7932–7942, 2024. 2

- [47] Ryan Po, Wang Yifan, Vladislav Golyanik, Kfir Aberman, Jonathan T. Barron, Amit H. Bermano, Eric Ryan Chan, Tali Dekel, Aleksander Holynski, Angjoo Kanazawa, C. Karen Liu, Lingjie Liu, Ben Mildenhall, Matthias Nießner, Bj¨orn Ommer, Christian Theobalt, Peter Wonka, and Gordon Wetzstein. State of the art on diffusion models for visual computing, 2023. 2
- [48] Ryan Po, Guandao Yang, Kfir Aberman, and Gordon Wetzstein. Orthogonal adaptation for modular customization of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7964–7973, 2024. 2
- [49] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2, 4, 5
- [50] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 2
- [51] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125,

2022. 1, 2, 3, 9

- [52] Anton Razzhigaev, Arseniy Shakhmatov, Anastasia Maltseva, Vladimir Arkhipkin, Igor Pavlov, Ilya Ryabov, Angelina Kuts, Alexander Panchenko, Andrey Kuznetsov, and Denis Dimitrov. Kandinsky: an improved text-to-image synthesis with image prior and latent diffusion. arXiv preprint arXiv:2310.03502, 2023. 1
- [53] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang Li, Qing Jiang, and Lei Zhang. Grounded sam: Assembling open-world models for diverse visual tasks,

2024. 13

- [54] Elad Richardson, Yuval Alaluf, Ali Mahdavi-Amiri, and Daniel Cohen-Or. pops: Photo-inspired diffusion operators. arXiv preprint arXiv:2406.01300, 2024. 1, 2, 3, 4, 9
- [55] Elad Richardson, Kfir Goldberg, Yuval Alaluf, and Daniel Cohen-Or. Conceptlab: Creative concept generation using vlm-guided diffusion prior constraints. ACM Transactions on Graphics, 43(3):1–14, 2024. 2, 3
- [56] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2022. 2
- [57] Ciara Rowles, Shimon Vainer, Dante De Nigris, Slava Elizarov, Konstantin Kutsy, and Simon Donn´e. Ipadapterinstruct: Resolving ambiguity in image-based conditioning using instruct prompts. arXiv preprint arXiv:2408.03209,

2024. 1

- [58] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. 2022. 2, 8

- [59] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6527–6536, 2024. 2
- [60] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 2
- [61] Arseniy Shakhmatov, Anton Razzhigaev, Aleksandr Nikolich, Vladimir Arkhipkin, Igor Pavlov, Andrey Kuznetsov, and Denis Dimitrov. Kandinsky 2. https: //github.com/ai-forever/Kandinsky-2, 2022. 2, 3, 4
- [62] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without testtime finetuning, 2023. 2
- [63] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 3, 2024. 5
- [64] Yoad Tewel, Rinon Gal, Gal Chechik, and Yuval Atzmon. Key-locked rank one editing for text-to-image personalization. In ACM SIGGRAPH 2023 Conference Proceedings,

2023. 2

- [65] Yael Vinker, Andrey Voynov, Daniel Cohen-Or, and Ariel Shamir. Concept decomposition for visual exploration and inspiration. ACM Transactions on Graphics (TOG), 42(6): 1–13, 2023. 1, 2, 3
- [66] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. p+: Extended textual conditioning in text-toimage generation, 2023. 2
- [67] Haonan Wang, James Zou, Michael Mozer, Linjun Zhang, Anirudh Goyal, Alex Lamb, Zhun Deng, Michael Qizhe Xie, Hannah Brown, and Kenji Kawaguchi. Can ai be as creative as humans? arXiv preprint arXiv:2401.01623, 2024. 3
- [68] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. arXiv preprint arXiv:2302.13848, 2023. 2
- [69] Merryl J Wilkenfeld and Thomas B Ward. Similarity and emergence in conceptual combination. Journal of Memory and Language, 45(1):21–38, 2001. 3
- [70] Guangxuan Xiao, Tianwei Yin, William T Freeman, Fr´edo Durand, and Song Han. Fastcomposer: Tuning-free multisubject image generation with localized attention. International Journal of Computer Vision, pages 1–20, 2024. 3
- [71] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024. 3, 8
- [72] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o:

- One single transformer to unify multimodal understanding and generation, 2024. 3
- [73] Kai Xu, Hanlin Zheng, Hao Zhang, Daniel Cohen-Or, Ligang Liu, and Yueshan Xiong. Photo-inspired model-driven 3d object modeling. ACM Transactions on Graphics (TOG), 30(4):1–10, 2011. 3
- [74] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024. 3, 7
- [75] Yang Yang, Wen Wang, Liang Peng, Chaotian Song, Yao Chen, Hengjia Li, Xiaolong Yang, Qinglin Lu, Deng Cai, Boxi Wu, et al. Lora-composer: Leveraging low-rank adaptation for multi-concept customization in training-free diffusion models. arXiv preprint arXiv:2403.11627, 2024. 2
- [76] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 1, 2, 3

- [77] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. 2023. 2, 3, 4
- [78] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models, 2024. 2
- [79] Yu Zeng, Vishal M Patel, Haochen Wang, Xun Huang, TingChun Wang, Ming-Yu Liu, and Yogesh Balaji. Jedi: Jointimage diffusion models for finetuning-free personalized textto-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6786–6795, 2024. 2
- [80] Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang, and Nicu Sebe. Bilateral reference for high-resolution dichotomous image segmentation. CAAI Artificial Intelligence Research, 2024. 13
- [81] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model, 2024. 3

# Appendix

###### A. Additional Details

###### A.1. Training Data for IP-Prior

As explained in the main text, the data for each domain shown in the paper was generated using Flux-Schnell [5] with a dedicated prompt. We next detail the specific prompt used for each such domain

- • Characters - “studio photo pixar style concept art, An imaginary fantasy (adjectives) (character) creature with eyes arms legs mouth, white background studio photo pixar asset”. Where “adjectives” is a set of 4-6 adjectives randomly sampled from a pool of about 200 adjectives, collected using an LLM, and “character” is a set 1-3 characters sampled from a list of characters generated by an LLM concatenated with a list of 18K classes from OpenImages [32].
- • Products - “A product design photo of a (attributes) product with (character-like) attributes, integrated together to create one seamless product. It is set against a light gray background with a soft gradient, creating a neutral and elegant backdrop that emphasizes the contemporary design. The soft, even lighting highlights the contours and textures, lending a professional, polished quality to the composition” Where “attributes” is a similarly generated list of ∼300 product attributes including materials and features and “character-like” is an optional description on an object or animal.
- • Toys - “Professional studio photo of an extremely cute and friendly smiling (animal) plush toy is sitting in a natural frontal position, facing the camera. He is wearing a (outfit) (item). It is set against a white background with soft, even lighting, lending a professional quality to the composition” Where, as before, “animal” is a set of texts of different animals, and “outfit” is a set of attributes covering clothing and hairstyle.
- • Portraits - “An artistic face collage crafted from a sparse and minimal set of large everyday objects, such as (object). The assemblage forms expressive features, such as lips, textured eyes, and a sculpted nose, set against a pristine white background that highlights the intricate details and creative use of materials.” Where “object” is chosen from a set of 60 relevant objects, such as “shells” and “fruits”.
- • Rubber Ducks - “Professional studio photo of a rubber duck. He is wearing a (outfit). It is set against a white background with soft, even lighting, lending a professional quality to the composition”

Where “outfit” is the same list as the one used for toys.

In terms of quantity, the data itself was generated on the fly, alongside the training process. We use four steps for generating the image with FLUX-Schnell model [5], followed by segmentation using SAM [29].

###### A.2. Training Data for IP-LoRA

We follow the following process to train the background generation LoRA, where the model is conditioned on an image embedding representing the concept alongside the text prompt. First, we generate images of different creatures using FLUX, and add to the prompt a description of a background chosen from a set of 50 prompts. Then, we use the Bria RMBG2.0 model [7, 80] to separate the creature from the background. This image is used as the input to the IP-Adapter+ during training alongside the text prompt describing the target background.

Similarly, for the reference sheet LoRA, we first generate a set of images using the prompt “A character sheet displaying an imaginary fantasy (adjectives) (creatures) creature with eyes mouth, from several angles with 1 large front view in the middle, clean white background. In the background, we can see half-completed, partially colored sketches of different parts of the object”. Here, the adjectives and creatures are sampled from the same list as above. Next, we use grounded SAM [53] to find the largest single view of the creature in the generated image. Its segmented and cropped instance then serves as the visual conditioning for the model. In this scenario, the text prompt remains fixed during tuning, as the only control comes from the given image embedding.

###### B. Additional Results and Comparisons

Below, we provide additional qualitative results and comparisons, as follows:

- • In Figure 15 and Figure 16 we present additional results for the domain of character ideation, showing our model ability to successfully a wide set of inputs.
- • In Figure 17 we present additional results for the product design domain.
- • In Figure 18 and Figure 19 we present additional results for the toy concepting domain.
- • In Figure 20 we present additional results for our reference sheet LoRA, showing its ability to successfully condition on a given image embedding and generating an output that resides in the target style while aligning with the given input.

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

###### Input Sampled Results Input Sampled Results Figure 15. Additional PiT results for character ideation

[Figure 399]

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

Input Sampled Results

Figure 16. Additional PiT results for character ideation

[Figure 423]

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

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

Input Sampled Results

Figure 17. Additional PiT results for product design

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

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

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

Input Sampled Results

Figure 18. Additional PiT results for toy concepting

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

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

[Figure 496]

[Figure 497]

[Figure 498]

Input Sampled Results

Figure 19. Additional PiT results for toy concepting

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

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

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

###### Figure 20. Style Generation. We train a LoRA to generate character reference sheets when given a concept embedding.

