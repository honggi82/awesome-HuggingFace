# arXiv:2501.12224v1[cs.CV]21Jan2025

## TokenVerse: Versatile Multi-concept Personalization in Token Modulation Space

Daniel Garibi1,2,∗ Shahar Yadin1,3,∗ Roni Paiss1 Omer Tov1 Shiran Zada1 Ariel Ephrat1 Tomer Michaeli1,3 Inbar Mosseri1 Tali Dekel1,4 1Google DeepMind 2Tel Aviv University 3Technion 4Weizmann Institute

https://token-verse.github.io/

ConceptImages

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

a boat is ﬂoating on the water

a doll wearing a jacket

a dog wearing a hat and a necklace

ﬂip ﬂops with a pattern

a forest with a light

a man wearing a shirt

a rabbit wearing glasses

[Figure 7]

[Figure 8]

[Figure 9]

###### GeneratedImages

§

a doll paddling on a boat in a river under light

a doll wearing a shirt and a necklace a dog wearing a shirt with a pattern and

glasses running on the beach

Figure 1. TokenVerse extracts distinct complex visual concepts from a set of concept images (top), and allows users to generate images that depict these concepts in novel versatile compositions (bottom row). Our framework independently processes each concept image, and learns to disentangle its concepts based solely on an accompanying caption, without any additional supervision or masks. This is achieved by learning a personalized representation for each token in the source caption. Our personalized text tokens, extracted from multiple images, are then flexibly incorporated into new text prompts (colored words) to generate novel creative images.

### Abstract

We present TokenVerse – a method for multi-concept personalization, leveraging a pre-trained text-to-image diffusion model. Our framework can disentangle complex visual elements and attributes from as little as a single image, while enabling seamless plug-and-play generation of combinations of concepts extracted from multiple images. As opposed to existing works, TokenVerse can handle multiple images with multiple concepts each, and supports a wide-range of concepts, including objects, accessories, materials, pose, and lighting. Our work exploits a DiT-based

*Equal contribution, work was done while interns at Google DeepMind

text-to-image model, in which the input text affects the generation through both attention and modulation (shift and scale). We observe that the modulation space is semantic and enables localized control over complex concepts. Building on this insight, we devise an optimization-based framework that takes as input an image and a text description, and finds for each word a distinct direction in the modulation space. These directions can then be used to generate new images that combine the learned concepts in a desired configuration. We demonstrate the effectiveness of TokenVerse in challenging personalization settings, and showcase its advantages over existing methods.

### 1. Introduction

The visual world is a rich tapestry of visual concepts, encompassing diverse objects, poses, lighting conditions, materials, and textures, often intricately combined in complex ways. Text-to-image models have shown a remarkable ability to learn and represent this complexity, generating images that not only capture individual concepts but also seamlessly integrate them into diverse and coherent settings. In this work, our goal is to gain user-control over the unique properties of the generated concepts. Specifically, our approach seeks to: (i) disentangle and learn the distinctive attributes of various visual elements within as little as a single image, and (ii) enable the generation of new images that flexibly combine different subsets of concepts extracted from multiple images (Fig. 1). Such versatile control over a diverse array of concepts is a pivotal ingredient in various real-world content creation tasks, including storytelling and personalized content creation. Yet, existing approaches focus on personalization only for objects or styles, often require segmentation masks or bounding boxes, and can either support a single concept per image or multiple concepts within a single image.

Broadly speaking, there are two main approaches for personalized content creation. The first is fine-tuning a text-to-image generative model [32], teaching it to associate some unique text token with a new concept. This approach is restricted in its ability to seamlessly compose multiple concepts, as it requires the combination of model weights, each specialized for a different concept. The second approach keeps the model fixed and optimizes the input text embedding associated with a word that describes a concept [12]. However, this approach is often not expressive enough to fully capture the nuances of each concept. Furthermore, both approaches are primarily designed for capturing a single concept, and struggle to disentangle multiple concepts encapsulated in a single image. While various methods have been developed to generalize these approaches (e.g. by constraining different concepts to represent distinct image regions), existing methods are still restrictive in either the type or breadth of concepts they can handle. In particular, they struggle to disentangle non-object concepts like pose, materials, and lighting conditions.

Our objective is to enable versatile and flexible concept personalization, where diverse concepts are extracted from single images, and can be combined in different configurations. Our framework builds on a pre-trained text-to-image Diffusion Transformer (DiT) model, in which the input text is processed in two paths: (i) through transformer blocks where the text tokens are jointly processed with the image tokens, and (ii) through a modulation path where a global text embedding is mapped to scale and shift coefficients that modulate the channels of the tokens within each transformer block. Inspired by the successful use of the modulation

space in GANs for semantic image manipulation, here we explore the use of the modulation space in DiTs for our task. We observe that directions within this space, which we call M, correspond to semantic modifications to the generated image, similarly to GANs. However, these manipulations are often not sufficiently localized for personalized content creation. Yet, we find that modifying the modulation vector for only a single text token can lead to semantic modifications only to the concept associated with this token. We denote the space of per-token modulations by M+.

We take advantage of the space M+ for unsupervised disentangled personalization and composition of visual concepts. We show that given an image and its caption, optimizing a modulation vector for each text token is sufficient to personalize the visual element it describes. The disentanglement of the visual concepts in the image is naturally facilitated by the model’s inherent association between text tokens and their corresponding image parts [16]. While the modulation vectors for all text tokens are optimized jointly, we find that each optimized vector personalizes the visual element tied to the text token it modulates. Thus, multiple learned elements can be generated together in new settings by simply describing their composition in the caption and plugging the learned vectors for their text tokens.

We show that our approach is expressive enough to represent complex concepts without requiring adjustment to the model’s weights, thus preserving its prior. In addition, as the visual elements are defined by semantic text tokens rather than visual cues (e.g. segmentation masks), our method supports the personalization of overlapping objects, as well as of non-object concepts (e.g. poses, lighting and materials) simply by describing them in the caption during optimization (Fig. 1). Lastly, this approach is highly modular, allowing concepts extracted from multiple images to be seamlessly combined.

To summarize, our key contributions are as follows:

- • We present TokenVerse – the first method that enables disentangled multi-concept personalization from multiple images and the plug-and-play composition of new images that depict the learned concepts.
- • Our method can personalize semantic concepts beyond objects, such as lighting conditions, materials and poses.
- • We explore the role of text token modulation in the generation process in DiTs, and demonstrate its effectiveness as a localized and semantically-meaningful space.
- • We demonstrate the applicability of TokenVerse for personalized content creation and storytelling.

### 2. Related work

Diffusion transformers. While diffusion models initially relied on UNet-based architectures [27, 28, 31, 33], most recent state-of-the-art text-to-image models use diffusion transformers (DiTs) as their backbone [9, 10]. Unlike

in UNets, where the text condition is queried via crossattention layers, in DiTs the text tokens are processed by the transformer alongside the image tokens. Joint attention layers facilitate bidirectional interaction between image and text tokens, enabling seamless integration of textual and visual information throughout the generation process.

Personalization methods. Personalization methods extend the distribution of a pretrained model to include a specific concept provided by a set of images. Most approaches achieve this either by learning specialized text embeddings to represent the concept [7, 12, 37] or by fine-tuning layers of the model itself [11, 32, 34]. These methods learn a single concept from one or several images, and can then incorporate the learned concept in new images generated according to text prompts. In contrast, we focus on multiconcept disentangled personalization and composition, i.e. extracting several concepts from a single image, and composing them with concepts learned from other images.

Disentangled multi-concept personalization Several recent works aimed to learn multiple concepts from a single or few images, enabling these concepts to be reused in novel compositions. Break-a-Scene [8] employs Dreambooth finetuning [32] while relying on user-provided spatial masks to isolate the concepts in the image. The dependence on spatial masks poses a significant limitation, as does not allow to disentangle e.g. the appearance of an object from its pose. Other methods, like Inspiration Tree [36] and ConceptExpress [14], extract multiple concepts by jointly learning several tokens from a single image, each representing a different concept. However, the disentanglement achieved by these methods is unpredictable, offering no control over which aspects of the image are separated into individual concepts. This limits the flexibility and usability of the approach, making it less effective for generating complex and diverse compositions. Unlike these approaches, our method provides control over the visual elements that should be personalized without requiring any visual cues.

Composition of concepts Complementary to the task of multi-concept personalization, is that of combining concepts from different images into a single generated image. Methods for concept composition typically utilize LoRA [17] to learn new concepts (usually one concept per image) and propose techniques for merging multiple LoRAs to operate simultaneously. Some methods, like LoRA Composer [38] and OMG [23], achieve this by incorporating spatial conditioning, such as masks, and assigning localized prompts to specific regions of the image. However, this limits the ability to compose concepts that naturally overlap, such as a necklace worn by a man. Other approaches [13, 26, 34] fuse multiple LoRAs, allowing them

|[Figure 10]|
|---|

|[Figure 11]|
|---|

SourceImage

|[Figure 12]|
|---|

|[Figure 13]|
|---|

- (a)directionin

Space:

A dog and a cat sitting on the street in New York City

Cat to Sphynx cat Dog to Poodle dog

|[Figure 14]|
|---|

|[Figure 15]|
|---|

|[Figure 16]|
|---|

A toy car and a ball on the ﬂoor

Ball to Baseball

|[Figure 17]|
|---|

|[Figure 18]|
|---|

|[Figure 19]|
|---|

Pose to Tree pose

A man doing a yoga pose in the park

- (b)directionin

|[Figure 20]|
|---|

|[Figure 21]|
|---|

Space:

Figure 2. Directions in the global modulation space (M) and our per-token modulation space (M+). Given a generated image (top row), we modify it using text-driven directions in both M and M+ spaces. (a) Adding a direction to the vector that is used to modulate all the text and image tokens (i.e. a direction in the space M) can be used to effectively modify desired concepts in the generated image. Yet, this often results in non-local changes that also affect other concepts in the generated image. (b) Adding a direction only to the modulation vector of a specific text token, like “dog” or “ball” (i.e. a direction in the space M+) leads to a localized modification that mostly affects the concept of interest.

to work cohesively. However, these methods are either restricted to composing style and content or require a joint optimization process for all source images, limiting the number of distinct images that can be effectively combined. Contrary to these methods, our approach supports plug-andplay composition of a much larger number of personalized objects, without any user-provided spatial conditioning.

### 3. Preliminaries: Diffusion transformers

Diffusion and flow-based models generate images through an iterative process that starts with a sample of white Gaussian noise and gradually evolves into a valid sample. In each step, the current noisy image is passed through a neural network, which is used to obtain a slightly less noisy version of the image. As illustrated in Fig. 3(a), in textto-image diffusion transformers (DiTs), this neural network is a transformer that jointly processes the text and image tokens. Each DiT block contains an attention layer, a feedforward MLP, and a modulation mechanism that serves to incorporate conditioning signals, as we explain next.

The Modulation mechanism in DiTs. modulation refers to modifying the activations of a neural network on a perchannel basis, where each channel is multiplied by a single scale factor and shifted by a single bias scalar. In

|“a chicken made of mosaic under lights”<br><br>CLIP<br><br>MLP<br><br>[Figure 22]<br><br>y<br><br>[Figure 23]<br><br>“..chicken..”<br><br>[Figure 24]<br><br>“..mosaic..”<br><br>[Figure 25]<br><br>“...light...”<br><br>∆chicken<br><br>∆light<br><br>Per-token Modulation Attention Feed-forward DiT<br><br>[Figure 26]<br><br>Concept images<br><br>∆mosaic<br><br>(c) TokenVerse at inference<br><br>ConceptMod<br><br>ConceptMod<br><br>ConceptMod|
|---|

|“a turtle in the beach”<br><br>y<br><br>CLIP<br><br>MLP<br><br>[Figure 27]<br><br>[Figure 28]<br><br>Modulation Attention Feed-forward DiT<br><br>(a) Original DiT model<br><br>|y<br><br>Modulation Block<br><br>Mod Mod Mod Mod<br><br>|
|---|
|
|---|

|y<br><br>(b) TokenVerse training<br><br>∆mosaic<br><br>[Figure 29]<br><br>Concept image<br><br>“mosaic”<br><br>∆holding<br><br>“holding”<br><br>Per-token Modulation Block<br><br>∆ﬂowers<br><br>“ﬂowers”<br><br>∆vase<br><br>“vase”<br><br>Mod<br><br>ConceptMod<br><br>Mod<br><br>ConceptMod<br><br>Mod<br><br>ConceptMod<br><br>Mod<br><br>ConceptMod<br><br>|
|---|

Figure 3. TokenVerse overview. (a) A pre-trained text-to-image DiT model processes both image and text tokens via a series of DiT blocks. Each block consists of modulation, attention and feed-forward modules. We focus on the modulation block, in which the tokens are modulated via a vector y, which is derived from a pooled text embedding. (b) Given a concept image and its corresponding caption, TokenVerse learns a personalized modulation vector offset ∆ for each text token. These offsets represent personalized directions in the modulation space and are learned using a simple reconstruction objective. (c) At inference, the pre-learned direction vectors are used to modulate the text tokens, enabling the injection of personalized concepts into the generated images.

the context of generative models, the modulation mechanism has gained significant popularity following its use in StyleGAN [18–22]. There, small modifications to the modulation parameters have been shown to lead to smooth and semantically meaningful perturbations to the generated image, a property that has been exploited in a wide range of works on image editing and manipulation [2– 6, 15, 24, 29, 30, 35, 39]. In modern text-to-image DiTs, the modulation mechanism is used for incorporating conditioning signals, such as the diffusion timestep and a compact representation of the text prompt. In particular, Stable Diffusion 3 and Flux [9, 10] process the diffusion timestep t and a pooled embedding of the text prompt p (CLIP’s class token) by an MLP, which outputs a vector

y = MLP(t,CLIP(p)). (1)

This vector is then further processed and split into perchannel scale and shift parameters, which are used to modulate the text and image tokens within each block of the diffusion model. Importantly, the same scale and shift parameters are used for all tokens.

### 4. The M+ space

Inspired by the use of the modulation space in StyleGAN for editing, we now explore the space of modulation vectors in DiTs, which we coin M. We start by demonstrating the effect of modifying the modulation vector using simple manipulations on the text embedding injected to the modulation path (keeping the text at the input of the transformer fixed). We show this allows achieving semantically rich modifications, but often changes the generated image

in a non-localized manner. We then introduce our proposed framework, which modifies only the modulation vector that affects specific text tokens. We coin the space of all such modifications the M+ space.

The modulation space M. A naive approach to obtaining a direction in the space of modulation vectors is by using text prompts with and without a specific attribute,

∆attribute = MLP(t,eattribute) − MLP(t,eneutral), (2)

where eneutral is the pooled embedding of the text prompt used to generate the image and eattribute is the pooled embedding of the same prompt but with some attribute added to the object of interest (e.g. “Poodle dog” instead of “dog”). Having obtained such a direction, we can add it to the modulation vector y (Eq. (1)) with some scale factor w, to obtain an updated modulation vector, y + w∆attribute.

Figure 2 (a) illustrates the effect that this has on the generated image. As can be seen, the object of interest is indeed modified into possessing the desired attribute. However, the modification is not localized. Namely, each direction modifies also unrelated attributes in the image (e.g. changing both the dog and the cat when only one of them is specified in the direction).

The Per-token modulation space M+. To overcome the lack of localization, here we propose to modulate individual text tokens differently. We coin the space corresponding to all per-text-token modulation vectors M+. Specifically, rather than using the same vector y to modulate all tokens, we propose modifying the modulation vector only for the

text token corresponding to the concept we wish to affect. Modifying the modulation vector for a single text token affects the corresponding image tokens through the joint attention layer, and this turns out to translate into more localized effects in the areas of the image associated with that token.

Figure 2(b) illustrates the effect of such directions. For changes to the dog we apply the modulation vector y + w∆Poodle (Eq. (2)) only to the “dog” text token while using the unmodified y for the rest of the tokens. Similarly, for affecting the cat we apply the modulation vector y+w∆Sphynx only to the “cat” text token. As can be seen, these directions lead to highly localized changes, mostly affecting the objects corresponding to the manipulated tokens. This approach is not limited to objects and can also be applied to abstract concepts like pose. For example, in the case of a man doing yoga, the pose changes when modifying the modulation vector in the direction ∆Tree Pose. In this case as well, modifying the modulation vector only for the word “pose” better preserves the background of the image compared to modifying all tokens.

### 5. Disentangled concept learning

Having observed that the space M+ enables localized semantic manipulations, we would now like to find customized directions within this space. Specifically, given an example image depicting several desired concepts (a “concept image”) and a caption describing it, our goal is to learn disentangled representations for each of the visual concepts mentioned in the caption. Importantly, we aim to achieve this in an unsupervised manner, without relying on object masks. For example, for the caption “a person dancing at dawn”, we want to associate the words “person”, “dancing”, and “dawn” with directions in M+ that capture the identity, pose and lighting in the image, respectively. Once we extract these directions from the concept image, we can use them to generate new images with any subset of the learned concepts, simply by adding the directions to the appropriate text tokens. Importantly, our approach is modular in the sense that different directions can be extracted separately from different concept images, without the need for joint training.

Our objective is to determine a set of directions

{∆i}leni=1(p), each corresponding to a different token in the prompt p, which represent the concepts associated with

those tokens. As illustrated in Fig. 3(b), we do so by training Concept-Mod – a small MLP that predicts for every token in p a direction in M+,

(∆1,...,∆len(p)) = Concept-Mod(p). (3)

Our goal is to have each ∆i represent the direction in M+ between the ith token (e.g. “person”, which represents a

[Figure 30]

[Figure 31]

Pinput

}

ConceptMod

∆a ∆doll ∆wearing ∆a ∆jacket

“A doll wearing a jacket”

mod

|[Figure 32]|
|---|

[Figure 33]

“A picture divided into two parts. The left part is {Pinput}. The right partis {Pgenerated}.”

DiT

###### L2

mod

|[Figure 34]<br><br>[Figure 35]|+noise| |
|---|---|---|
| | | |

|[Figure 36]|
|---|

[Figure 37]

[Figure 38]

DiT

concat(Iinput, Igenerated)

Figure 4. Concept isolation loss. When training Concept-Mod we apply an additional concept isolation loss in 50% of the training steps. This loss encourages learning directions that do not interfere with other images by enforcing that the parts in the image that should not be affected by the directions remain similar.

generic human) and its customized version (e.g. the specific person appearing in the image). We train ConceptMod on the concept image and its associated prompt by using the same diffusion objective with which the original text-to-image model was trained. At inference, the learned concepts can be incorporated into newly generated images by adding the learned offsets to the appropriate text tokens (Fig. 3(c)).

Per-block optimization. We learn the directions in two stages. In the first stage, we aim to learn the coarse aspects of the concepts in the image. This is done by prioritizing the selection of high noise levels in the optimization of the diffusion loss. In the second stage, we refine the directions by focusing more on the lower noise levels. In this stage, we also train an additional MLP that outputs a vector per transformer block. The outputs of this per-block MLP are added to the output of the Concept-Mod MLP, leading to a per-token per-block direction.

Concept isolation loss. TokenVerse learns for each token in the caption a separate direction in M+. As this space is relatively disentangled, directions learned from the same image are usually well isolated. However, when combining objects learned from different images, their optimized directions might interfere with each other, and degrade concept fidelity. To avoid such cases, we incorporate an additional concept isolation loss in 50% of the training iterations. This loss is designed to steer the optimization such that the optimized directions do not affect concepts that do not appear in the concept image. As illustrated in Fig. 4, we combine the input image with a random generated image and merge their captions into a single sentence describing both parts of the concatenated image. In practice we gener-

Concept Images Generated Images

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

a dog wearing a hat and a necklace

a cat wearing glasses and a shirt

[Figure 44]

[Figure 45]

a laying dog wearing glasses

a man wearing a shirt

a dog wearing a shirt, glasses ﬂying in the sky tied to balloons

a dog with a shirt, glasses on a rollercoaster

a dog wearing a shirt and necklace having a picnic

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

a creature sitting on a bowl

a man leaning against a wall

[Figure 51]

[Figure 52]

a woman posing

a woman posing by the sea

a cartoon of a man posing on a mountain

a creature posing in the snow

a creature posing on mars

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

a doll inside a bucket

a rocking horse on the ﬂoor

[Figure 58]

[Figure 59]

a doll sitting on bench in the garden

a woman standing in a fog holding lamp

a doll riding rocking horse in the garden

a doll on a bench in the park, fog around

###### a doll riding rocking horse in a fog

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

a mosaic vase, holding ﬂowers

a mountain under a light.

[Figure 65]

[Figure 66]

a boat is ﬂoating on the water

a doll wearing a jacket

a light over paris a doll on a boat made of mosaic a doll surﬁng on a surfboard made of mosaic under a light

- Figure 5. Qualitative results. Each row begins with a bank of four source images, from which our method independently extracts concepts. To the right, three generated images are shown, demonstrating the seamless combination of these concepts into new, coherent outputs.

Concept Images Generated Image

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

a dog wearing a hat and a necklace

a dog wearing a hat and a necklace

a man leaning against a wall

a cat wearing glasses and a shirt

a man wearing a shirt

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

a doll sitting on a bench in the garden.

a man wearing shirt, hat, glasses, necklace holding a backpack, sitting on a bench near a table in front of a fog.

a backpack hanging on a chair

a woman standing in a fog

a mirror on a wall and a table

- Figure 6. Extreme multi-concept personalization. Our method has no technical constraint on the number of concepts that can be combined in an image. As can be seen, TokenVerse can generate images composing a significant number of concepts.

ated a fixed set of 25 images using the base model and always randomly chose one image from this set. We then run the model on the combined image and combined caption, applying the optimized directions only over the tokens from the input prompt. Finally, we apply an L2 loss between the output of the model and the output of the base model only on the part corresponding to the image that was concatenated to the concept image. This encourages the learned directions to only affect the parts in the concept image that match their text.

### 6. Experiments

We evaluate TokenVerse quantitatively and qualitatively, and demonstrate that it outperforms existing approaches in accurately extracting multiple concepts from images and in seamlessly using them to generate new images.

#### 6.1. Implementation details

We illustrate our method with Flux-dev, which has 58 DiT blocks and a modulation vector of dimension 3072. The optimization process involves two stages. In the first stage, we optimize a global direction for each text token over 800 steps. During this process, for 92% of the training iterations, we train on timesteps t between 800 and 1000, and for the remaining iterations, we train on timesteps between 0 and 800. In the second stage, we refine the modulation by optimizing a per-block offset for additional 600 steps. In this stage, we reverse the focus by training on t between 0 and 800 for 92% of the steps and t between 800 and 1000 for the rest. We use image augmentations as well as text augmentations. See App. A for additional details.

For personalization of humans, where capturing precise details is crucial, we use multiple input images depicting

the same subject, each accompanied by four augmentations of the input prompt. For all other cases we train on a single concept image with eight augmented prompts.

#### 6.2. Qualitative results

We present qualitative results generated by TokenVerse in Figures 1, 5, 6 and 7. Figure 5 presents three generated images for each set of concept images, each depicting a different combination of concepts. As can be seen, TokenVerse successfully extracts the desired visual elements from the concept images, and seamlessly combines them in new settings. Notably, TokenVerse correctly disentangles not only the featured objects (e.g. the dogs or the cat in the first row) but also their clothing, pose, materials, and even the lighting conditions. Importantly, TokenVerse has no technical limitation on the number of concepts that can be combined into a single image. To illustrate this, Fig. 6 showcases the composition of nine personalized objects, each extracted from a different concept image. Figure 7 further showcases the personalization and combination of abstract concepts such as lighting and pose. As can be seen, the pose concept is completely disentangled from the woman and successfully transferred to the personalized bear. More qualitative results, including examples of storytelling, can be found in App. B, App. E and our project webpage.

#### 6.3. Comparisons

TokenVerse enables (i) disentangled personalization of multiple concepts from a single image, and (ii) plug-andplay composition of multiple concepts learned separately from different images into one generated image. We refer to those sub-tasks as decomposition and composition, respectively. Unlike TokenVerse, existing methods address only one of these sub-tasks. In addition, some of them require

|Method<br><br>|Concept Decomposition<br><br>|Concept Composition|masks free<br><br>|
|---|---|---|---|
|DB LoRA Break-A-Scene ConceptExpress OMG TokenVerse (Ours)<br><br>|× √<br><br>√ × √<br><br>|× × × √ √|√ × √ √ √<br><br>|

- Table 1. Capabilities of competing baselines. The table lists the capabilities of DreamBooth [32], BAS [8], ConceptExpress [14], OMG [23] and our method. Concept decomposition is the task of disentangling and personalizing multiple objects from a single image. Composition is the task of combining separately learned concepts in a new generated image. Unlike existing approaches, our method enables mask-free multi-object disentangled personalization, as well as the composition of multiple objects from several reference images.

object masks as input, which limits their generalization to concepts beyond objects. The capabilities of each method are summarized in Tab. 1.

In order to evaluate these methods on our full task, it is necessary to perform some adaptations, which we describe in App. C.1. Notably, there is no trivial way to adapt BreakA-Scene [8] to work on multiple images with separate training, as it requires fine-tuning the model’s weights on each concept image. We therefore evaluate it on the easier task of jointly learning the concepts from all concept images together (i.e. no plug-and-play composition, but rather a new optimization for each desired combination of objects). For DreamBooth, we provide two adaptations, one for the full task (separate training) and one for the easier task (joint training), as detailed in App. C.1. ConceptExpress, OMG, and TokenVerse are evaluated on the full task.

##### 6.3.1. Qualitative comparison

Figure 8 presents qualitative comparisons on the simplest case of our task: generating an image depicting two concepts, each learned separately from a different concept image. Naturally, each concept image depicts more than one concept. For example, in the first row, the models need to learn the visual attributes of the cat and of the sheep doll, disentangling them from the other elements appearing in the concept images. As observed, even in this simplest case, all competing methods fail to disentangle at least one of the two concepts. For instance, in the first row, OMG fails to properly identify the doll and extracts the vase instead. This is because it was designed for personalization from images depicting a single object.

##### 6.3.2. Quantitative comparison

We next provide quantitative comparisons on the simplest form of our task, as mentioned above, i.e. combining two concepts learned separately from two different images, both containing more than one concept. Since some of the base-

[Figure 77]

[Figure 78]

[Figure 79]

Light

Pose

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

Figure 7. Concepts beyond objects. We demonstrate the composition of three types of personalized concepts: object (the bear; concept image not shown), pose (left column) and lighting (top row). TokenVerse successfully learns the pose and lighting without overfitting to the identity of the poser or the specific lit scene.

lines are originally designed for either composition or decomposition, we additionally report the performance on each of these sub-tasks separately. In the decomposition experiment, the goal is to extract two concepts from a single image. In the composition experiment, the goal is to extract a single concept from each of two images that depict only one concept. In both cases, the final goal is to generate an image that combines the two learned concepts.

Evaluation set For images containing a single concept, we used a subset of the dataset provided by DreamBench++ [25], which includes 20 images and corresponding target prompts. For concept images that contain multiple concepts, there is no existing benchmark suitable for evaluating our task. We therefore created a custom dataset of 30 images, each containing between two and four distinct concepts. The target prompts for these images were generated using an LLM.

Metrics We follow the evaluation protocol of DreamBench++ [25], a benchmark specifically designed to assess personalization in text-to-image models. Given a concept image, a target prompt and the corresponding generated image, DreamBench++ employs a multimodal LLM [1] to output two scores in the range of 0 to 1 (higher is better): (i) Concept Preservation (CP) measures how well the personalized concept is preserved in the generated images, and (ii) Prompt Fidelity (PF) assesses how well the generated image adheres to the target prompt. We calculate CP separately for each concept in the generated image, by providing DreamBench++ with a concept image that depicts only the rele-

###### Concept Images Concept Express BASjoint DBjoint DB OMG Ours

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

a cat next to a doll in the garden

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

a cat wearing an hat

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

a man wearing hat and glasses

- Figure 8. Qualitative comparisons. Each row depicts two concept images (left) and images containing a combination of those concepts, generated by ConceptExpress [14], BAS [8], DreamBooth [32], OMG [23] and our method. The concepts associated with the green and blue words are taken from the left and right concept images, respectively. As can be seen, our method best composes the two concepts while preserving concept fidelity.

vant concept where the rest of the image is masked out. We report the average CP for the two generated concepts.

Figure 9(a-c) quantitatively compares the different methods (see exact numbers in App. C.2). As can be seen, TokenVerse consistently outperforms all other methods in terms of Concept Preservation and achieves Prompt Preservation scores that are competitive with the best competitors. This applies to three evaluated tasks, demonstrating that (i) TokenVerse outperforms other methods in on the tasks they were originally designed to handle (composition / decomposition) without requiring any visual cues such as segmentation masks; (ii) TokenVerse is superior to other methods on the full task of disentangled concept learning and composition; (iii) TokenVerse outperforms DBjoint and Breaka-Scenejoint even though they were evaluated on the easier task of jointly learning the concepts for each combination. Please refer to App. C.2 for a detailed analysis.

##### 6.3.3. User study

We further compare TokenVerse to existing methods by conducting a user study. We select a random subset of examples from the evaluation set described in Sec. 6.3.2, where each concept image depicts more than one concept and each result image fuses concepts from two different images. We followed the user study protocol of Break-AScene [8] and asked the evaluators to rate both the alignment of the generated image to the prompt and the concept preservation on a scale of 1-5. Since jointly rating the identity preservation of two concepts could cause ambiguity in cases where one concept was preserved well and

the other was not, compared to cases where both concepts were slightly preserved, we separated the identity preservation rating into two consecutive questions focusing on one object at a time and averaged the two responses. Our study was conducted with 37 participants, each tasked with voting on five results per method, resulting in a total of 3000 votes. As shown in Fig. 9(d), the user study results are consistent with the DreamBench++ evaluation, giving our method the best concept preservation score with a high prompt fidelity score. Refer to App. D for additional details.

#### 6.4. Ablation study

Figure 10 qualitatively illustrates the contribution of each component in TokenVerse. First, we ablate the use of the M+ space, by applying the directions predicted by Concept-Mod directly to each text token before it enters the transformer, similarly to textual inversion methods [7, 12]. As can be seen in column (a) , this approach fails to faithfully reconstruct the concepts. In column (b), we show the effect of applying the directions in M+, i.e. adding them to the modulation vector of each text token. As can be seen, this space proves to be more expressive, enabling better concept preservation. Subsequently, we introduce per-block directions in column (c). These further enrich the method’s capabilities and lead to improved concept fidelity. However, combining concepts from different images still leads to unsatisfactory results. Column (d) shows the results obtained with our full method, including the isolation loss, which mitigates the interference between concepts learned from different images, and improves concept preservation.

(a) Quantitative Evaluation using DreamBench++ (b) User Study

Concept composition Concept Decomposition Full task Full task

ConceptPreservation

ConceptPreservation

ConceptPreservation

| |Ours<br><br>ConceptExpress<br><br>DB<br><br>OMG<br><br>Break-a-Scenejoint<br><br>DBjoint| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| |ConceptExpress<br><br>OMG<br><br>Break-a-Scenejoint<br><br>Ours<br><br>DB<br><br>DBjoint| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| |ConceptExpress OMG<br><br>DB, DBjoint Ours Break-a-Scenejoint| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| |Ours<br><br>Concept Express OMG<br><br>Break-a-Scenejoint<br><br>DB<br><br>DBjoint| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

ConceptPreservation

Prompt Fidelity Prompt Fidelity Prompt Fidelity Prompt Fidelity

- Figure 9. Quantitative comparison. We compare our method to other baselines on concept preservation and prompt fidelity (higher is better) using DreamBench++ and a user study. (a) We compare three different settings: (i) composing two concepts from different images (concept composition), (ii) decomposing two concepts from the same image (concept decomposition), and (iii) the combination of the two (full task). (b) We conduct a user study, comparing our method to existing methods on our full task. Our method consistently scores best in terms of concept preservation while maintaining high prompt fidelity scores. See App. C.2 for the exact metrics.

[Figure 112]

[Figure 113]

[Figure 114]

a dog in front of a background

a creature sitting on a bowl

[Figure 115]

a doll inside a bucket

a chicken sitting on a stool in the garden

Concept Images

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

dogbowla insidea near

bucketa onatable

chickena

onthetable

Offsets to input text tokens

space modulation

+ Per-block offsets

+ Isolation loss

(a) (b) (c) (d)

- Figure 10. Ablations. The left pane shows all the concepts used to generate the result images. Columns (a) to (d) shows the results of our method as additional components are progressively integrated.

#### 6.5. Limitations

While our method is the first to support both the disentangled learning and the composition of multiple visual concepts, some limitations remain for future work. First, since the directions are learned separately for each concept image, in rare occasions the resulting modulated text tokens may become similar to each other. In such cases, this similarity could lead to blending of the objects in the generated image, as seen in Fig. 11(a). We find that this phenomenon happens mostly when the two concepts correspond to the sole or main subjects in their respective images, and is not linked to specific concept types. In such cases, training on both images jointly fixes the issue. We provide further analysis of this case in App. F . In addition, as seen in Fig. 11(b), our method struggles with combining two concepts that share the same name identifier. This problem can be mitigated by using distinct words for each concept, as demonstrated in App. F . Lastly, our method may fail for incompatible combinations. An example is shown in Fig. 11(c), where attempting to cause a doll with extremely short limbs do a

pose that requires arms and legs results in the generation of undesired human body parts.

### 7. Conclusion

Recent progress in text-to-image diffusion models has ignited a surge of interest in harnessing them for controlled image generation. Many approaches have been proposed for personalized image generation, where objects or styles are extracted from reference images and are used to compose new images. However, existing methods struggle to handle multiple images with multiple concepts each, and do not support non-object concepts like poses, materials and lighting conditions. Here, we introduced TokenVerse – the first method for multi-concept personalization that overcomes these challenges. TokenVerse extracts per-text-token directions in the modulation space of DiTs, which we showed to be rich and semantic. Our method opens the door to a plethora of applications, from story telling to personalized content creation.

Highly similar modulated tokens

Incompatible combinations

(a) (b) (c)

Coliding captions

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

ConceptImagesGeneratedImage

a doll inside a bucket

a woman posing by the sea

a doll wearing a jacket

a doll inside a bucket

a doll wearing a jacket

a dog wearing a hat and a necklace

[Figure 130]

[Figure 131]

[Figure 132]

a doll posing on a table

a doll and a dog in the garden

a doll on a table and a doll on a sofa

- Figure 11. Limitations. Concept images are shown in the top row, with the generated images using TokenVerse below in each case. While our method supports both disentangled learning and multi-concept composition, limitations remain. (a) Rare blending can occur in specific combinations due to independent training of concepts; We provide analysis and mitigations in App. F. (b) Challenges arise with concepts sharing the same name identifier, which can be mitigated by using distinct terms. (c) Certain incompatible combinations, such as a doll with tiny limbs in a complex pose, may result in undesired outputs.

### References

- [1] Openai. introducing gpt-4o and more tools to chatgpt free users., 2024. https://openai.com/ index/gpt-4o-and-moretools-to-chatgpt-free/. 8
- [2] Rameen Abdal, Yipeng Qin, and Peter Wonka. Image2stylegan: How to embed images into the stylegan latent space? In Proceedings of the IEEE international conference on computer vision, pages 4432–4441, 2019. 4
- [3] Rameen Abdal, Yipeng Qin, and Peter Wonka. Image2stylegan++: How to edit the embedded images? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8296–8305, 2020.
- [4] Rameen Abdal, Peihao Zhu, Niloy Mitra, and Peter Wonka. Styleflow: Attribute-conditioned exploration of stylegangenerated images using conditional continuous normalizing flows, 2020.
- [5] Yuval Alaluf, Or Patashnik, and Daniel Cohen-Or. Restyle: A residual-based stylegan encoder via iterative refinement. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.
- [6] Yuval Alaluf, Omer Tov, Ron Mokady, Rinon Gal, and Amit H Bermano. Hyperstyle: Stylegan inversion with hypernetworks for real image editing. arXiv preprint arXiv:2111.15666, 2021. 4
- [7] Yuval Alaluf, Elad Richardson, Gal Metzer, and Daniel Cohen-Or. A neural space-time representation for text-toimage personalization, 2023. 3, 9
- [8] Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel CohenOr, and Dani Lischinski. Break-a-scene: Extracting multiple concepts from a single image. In SIGGRAPH Asia 2023 Conference Papers, page 1–12. ACM, 2023. 3, 8, 9, 13

- [9] Black Forest Labs. Flux, https://github.com/black-forestlabs/flux, 2024. 2, 4
- [10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024. 2, 4
- [11] Yarden Frenkel, Yael Vinker, Ariel Shamir, and Daniel Cohen-Or. Implicit style-content separation using b-lora,

2024. 3

- [12] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion, 2022. 2, 3, 9
- [13] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, Yixiao Ge, Ying Shan, and Mike Zheng Shou. Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models, 2023. 3
- [14] Shaozhe Hao, Kai Han, Zhengyao Lv, Shihao Zhao, and Kwan-Yee K. Wong. Conceptexpress: Harnessing diffusion models for single-image unsupervised concept extraction, 2024. 3, 8, 9, 13, 14
- [15] Erik H¨ark¨onen, Aaron Hertzmann, Jaakko Lehtinen, and Sylvain Paris. Ganspace: Discovering interpretable gan controls. arXiv preprint arXiv:2004.02546, 2020. 4
- [16] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control, 2022. 2
- [17] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. 3
- [18] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196, 2017. 4
- [19] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4401–4410, 2019.
- [20] Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. In Proc. NeurIPS, 2020.
- [21] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8110–8119, 2020.
- [22] Tero Karras, Miika Aittala, Samuli Laine, Erik H¨ark¨onen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Alias-free generative adversarial networks. Advances in Neural Information Processing Systems, 34, 2021. 4
- [23] Zhe Kong, Yong Zhang, Tianyu Yang, Tao Wang, Kaihao Zhang, Bizhu Wu, Guanying Chen, Wei Liu, and Wenhan Luo. Omg: Occlusion-friendly personalized multi-concept generation in diffusion models, 2024. 3, 8, 9, 13, 14

- [24] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. Styleclip: Text-driven manipulation of stylegan imagery. arXiv preprint arXiv:2103.17249, 2021. 4
- [25] Yuang Peng, Yuxin Cui, Haomiao Tang, Zekun Qi, Runpei Dong, Jing Bai, Chunrui Han, Zheng Ge, Xiangyu Zhang, and Shu-Tao Xia. Dreambench++: A human-aligned benchmark for personalized image generation, 2024. 8
- [26] Ryan Po, Guandao Yang, Kfir Aberman, and Gordon Wetzstein. Orthogonal adaptation for modular customization of diffusion models, 2023. 3
- [27] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis, 2023. 2
- [28] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents, 2022. 2
- [29] Elad Richardson, Yuval Alaluf, Or Patashnik, Yotam Nitzan, Yaniv Azar, Stav Shapiro, and Daniel Cohen-Or. Encoding in style: a stylegan encoder for image-to-image translation. arXiv preprint arXiv:2008.00951, 2020. 4
- [30] Daniel Roich, Ron Mokady, Amit H Bermano, and Daniel Cohen-Or. Pivotal tuning for latent-based editing of real images. ACM Transactions on Graphics (TOG), 42(1):1–13,

2022. 4

- [31] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2022. 2
- [32] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation, 2023. 2, 3, 8, 9, 14
- [33] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding, 2022. 2
- [34] Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. Ziplora: Any subject in any style by effectively merging loras, 2023. 3
- [35] Omer Tov, Yuval Alaluf, Yotam Nitzan, Or Patashnik, and Daniel Cohen-Or. Designing an encoder for stylegan image manipulation. arXiv preprint arXiv:2102.02766, 2021. 4
- [36] Yael Vinker, Andrey Voynov, Daniel Cohen-Or, and Ariel Shamir. Concept decomposition for visual exploration and inspiration, 2023. 3
- [37] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. P+: Extended textual conditioning in text-toimage generation, 2023. 3
- [38] Yang Yang, Wen Wang, Liang Peng, Chaotian Song, Yao Chen, Hengjia Li, Xiaolong Yang, Qinglin Lu, Deng Cai, Boxi Wu, and Wei Liu. Lora-composer: Leveraging lowrank adaptation for multi-concept customization in trainingfree diffusion models, 2024. 3
- [39] Jiapeng Zhu, Yujun Shen, Deli Zhao, and Bolei Zhou. Indomain gan inversion for real image editing. arXiv preprint arXiv:2004.00049, 2020. 4

### Appendix

- A. Additional training details We provide additional training details of our method.

- A.1. Training agmentations

During training, each image is paired with a text description. However, we found empirically that assigning multiple text descriptions per image improves the model’s ability to distinguish and separate the different concepts within the image. We achieve this by augmenting the text with several prompts that closely resemble the original description. Each prompt retains the same words describing the objects and actions but rearranges them in a different order, which can be generated using a large language model (LLM). In addition to text augmentations, we applied image augmentations such as random flips and mirroring to further enhance the model’s performance. Figure 12 demonstrates the improvement these augmentations have on the resulting images.

- A.2. Images for concept isolation loss

As described in Sec. 5 in the main text, our method employs a concept isolation loss, where the concept image and its accompanying prompt are concatenated to a random pair of generated image an prompt. In practice, we randomly sample from a fixed set of 25 pairs of captions and generated images for all the examples in the paper. We created these sets of images by asking an LLM to generate 25 prompts that contain objects, and providing the captions to Flux to generate the output images. The entire set of images is presented in Fig. 21.

- B. Additional qualitative results

- Figure 13 demonstrates iterative addition of concepts into a generated image. In each column, we introduce an additional learned concept while retaining all previously added concepts. Figures 16, 17 and 18, present additional qualitative examples of our method.

- C. Quantitative evaluation

We first describe how we adapted existing methods to our task for evaluation (note that they were also evaluated on their original tasks). Then, we provide the exact metrics from our quantitative evaluation and analyze the results.

#### C.1. Adaptations to existing methods

ConceptExpress [14] learns multiple text tokens from a single image without requiring fine-tuning of the model itself, making it directly applicable to our task. In contrast, OMG [23] requires adaptation to support concept decomposition, which we perform by fine-tuning with multiple rare

Concept Images

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

a doll inside a bucket

a creature sitting on a bowl

a dog in front of a background

a chicken sitting on a stool in the garden

(a) (b)

Full method

Without augmentations

[Figure 138]

[Figure 139]

a dog inside a bowl near a bucket on a table

[Figure 140]

[Figure 141]

a chicken on the table

- Figure 12. Augmentations Ablation. The top row shows the concepts used to generate the result images. Column (a) displays the results of our full method, while column (b) shows the results without text and image augmentations.

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

shirt hair pose

Object

Atrributes

Object

Atrributes

pose lighting hair

- Figure 13. Progressive composition of concepts. TokenVerse can be used to progressively add concepts into a generated image, while controlling all other aspects of the generated images via text. In each row, the object, pose, lighting, and hair are personalized, while the background is described by text (e.g. “NY city”, “garden”, and “Mars” for the top row.)

tokens, each linked to a word representing a distinct concept. While Break-A-Scene [8] is a prominent method for image decomposition, there is no trivial way to adapt it to learn concepts from multiple images with separate training. This is due to the fact that it involves fine-tuning the weights of the model on each concept image. As stated in the main

| |Composition|Decomposition|Full task|
|---|---|---|---|
|Metric<br><br>|CP ↑ PF ↑ CP · PF ↑|CP ↑ PF ↑ CP · PF ↑<br><br>|CP ↑ PF ↑ CP · PF↑|
|Dreambooth DreamBoothjoint OMG Break-A-Scenejoint ConceptExpress|0.280242 0.844422 0.236642 0.371304 0.619288 0.229944 0.238606 0.743968 0.177515 0.392912 0.372664 0.154380 0.214718 0.548723 0.117821<br><br>|0.668524 0.660167 0.441337 0.668524 0.660167 0.441337 0.267477 0.791793 0.211787 0.598387 0.641935 0.384126 0.246387 0.695087 0.171260<br><br>|0.207116 0.827521 0.171393 0.306262 0.591337 0.181104 0.207787 0.843395 0.175246 0.499054 0.641236 0.320011 0.187853 0.733286 0.137750<br><br>|
|Ours<br><br>|0.470108 0.688061 0.323463<br><br>|0.669940 0.747698 0.500431<br><br>|0.553125 0.821875 0.454600|

- Table 2. Dreambench++ Evaluation We complement the quantitative comparison graphs of the main paper with the exact measurements of concept preservation (CP) and prompt fidelity (PF).

|Metric|CP ↑ PF ↑|
|---|---|
|Dreambooth DreamBoothjoint OMG Break-A-Scenejoint ConceptExpress|2.2505 4.465 2.7582 3.462<br><br>2.205 3.611<br><br>3.203 3.151<br><br><br>2.211 2.686<br><br>|
|Ours|4.078 4.292|

- Table 3. User Study We complement the user study results graphs of the main paper with the exact measurements of concept preservation and prompt fidelity.

ods on the tasks for which they were originally designed to solve.

Notably, OMG [23] achieves very high Prompt Fidelity scores along with an extremely low Concept Preservation scores. We attribute this to the fact that OMG first performs unconditional generation, and then uses a segmentation map obtained from this image as layout for the personalized generation. Therefore, in cases where the personalized object does not match the shape of the corresponding object generated in the first stage, it cannot be accurately incorporated in the target generation. This usually leads to images with sufficient prompt alignment, but which fail to preserve the personalized concepts. This phenomenon is also demonstrated qualitatively in Fig. 13 in the main text, where OMG correctly follows the prompt, yet consistently fails to preserve objects that can vary in shape, such as the sheep doll, the hats and the glasses. The same type of behavior is also apparent for DreamBooth [32] whenever the two concepts are learned from different images. We believe this arises from the combination of two LoRAs, each trained on a separate image, which can compromise the method’s ability to preserve the concepts. In addition, ConceptExpress [14] takes an input image and returns a token for each concept it identifies in the image. A manual process is then required to map each concept to its corresponding token. ConceptExpress successfully detects only 75% of the concepts in the test set. For the remaining concepts, the generation is performed unconditionally.

text, we therefore evaluate Break-A-Scene on an easier task, where all of the personalized concepts are learned jointly (i.e. performing a separate optimization process per each combination of objects as opposed to the plug-and-play composition supported by our method). We adapt Break-AScene to this easier task by fine-tuning the model jointly on all images from which we wish to extract and compose concepts. We denote the adapted method Break-a-Scenejoint. LoRA-DreamBooth [32] is the prominent method for personalization (generating one concept learned from one image). We adapt it to concept decomposition similarly to OMG, and compose personalized objects by applying several LoRAs together. We also evaluate an additional variant, DreamBoothjoint on the easier task, by training it jointly on all concept images, as we do for Break-a-Scenejoint.

#### C.2. Quantitative comparison

### D. User study

As described in Sec. 6.3.2, we follow the evaluation protocol in DreamBench++, which reports Concept Preservation (CP) and Prompt Fidelity (PF). There often exists a tradeoff between these two criteria. Therefore, in addition to CP and PF, DreamBench++ also reports their product, CP·PF, as a unified measure of success.

As reported in the main text, we further evaluate the performance of our method by conducting a user study. The full numerical results are presented in Table 3.

In the study, we randomly sample 15 of the combinations used for the DreamBench++ full task evaluation. This process results in 90 images in total for each rater (6 methods, 15 results each).

Table 2 presents the numeric results for the quantitative evaluation described in Sec. 6.3.2. As can be seen, our method outperforms all other methods on CP and CP·PF across all three tasks. That is, TokenVerse not only surpasses the performance of the adapted methods on our full task, but also outperforms the original (unadapted) meth-

For each image, we asked participants to answer three questions:

• How well does the prompt align with the generated image?

(a) Common case (b) Failure case (c) Optional mitigation

[Figure 156]

###### ConceptImagesGeneratedImage

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

a doll wearing a jacket next to a dog wearing a hat and a necklace

a doll wearing a jacket

a doll inside a bucket

a dog wearing a shirt

a dog wearing a hat and a necklace

[Figure 161]

[Figure 162]

[Figure 163]

a doll and a dog in the garden

a doll and a dog in the garden

a doll and a dog in the garden

- Figure 14. Limitation – highly similar modulated tokens. (a) The common scenario of combining two distinct objects, such as a doll and a dog, into a single image. (b) A failure case where independent training of concepts leads to the creation of hybrid objects. (c) A potential mitigation for this issue by employing joint training on both concepts.

[Figure 164]

[Figure 165]

[Figure 166]

a doll wearing a jacket

[Figure 167]

a doll inside a bucket

a doll on a table and a doll on a sofa

ConceptImagesGeneratedImage

a rabbit wearing a jacket

[Figure 168]

a sheep inside a bucket

a rabbit on a table and a sheep on a sofa

(a) Coliding captions (b) With proper captions

[Figure 169]

- Figure 15. Limitations – colliding captions. Our method may fail when handling cases of colliding identifiers, such as two dolls (a). This issue can be easily resolved by assigning distinct identifiers to each object during the initial training (b).

directions learned for two objects taken from different images may influence each other, leading to undesired results. This behavior does not depend on the type of the objects, as we show in Fig. 14(a) in which TokenVerse successfully generates the concepts “dog” and “doll” in the same image. We observe that in such failure cases, the personalized directions (learned separately for each concept) result in keys with high inner product (at all transformer blocks, for all timesteps), leading to blending in the attention layer. When the directions are learned jointly, the inner product of the keys remains in normal range, and this problem does not occur, as seen in Fig. 14(c). We leave further analysis of this phenomenon to future work.

In addition, as shown in Fig. 15, the choice of word for each token is important. When the same term is used to identify concepts in two different images (e.g. “doll” in column (a)), the model struggles to generate an image containing both instances. Assigning different identifiers to each object (e.g. “sheep” for a sheep doll, and “rabbit” for the rabbit doll), allows the model to accurately generate an image with both objects as seen in Fig. 15(b).

- • To what extent does the first concept appear in the generated image?
- • To what extent does the second concept appear in the generated image?

The exact format of the user study is shown in Fig. 20.

### E. Application to styorytelling

An immediate application of our method is storytelling, where we aim to generate a narrative consisting of images featuring the same objects and scenes. An example is shown in Fig. 19.

### F. Limitation Analysis

Figure 14 shows a possible limitation of our approach. Specifically, in Fig. 14(b) we show that in rare cases, the

Concept Images Output Image

###### Concept Images Output Image

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

a doll wearing a jacket

a cat wearing glasses and a shirt

[Figure 177]

[Figure 178]

a dog standing in the garden

a dog wearing a shirt

a cat wearing glasses and a shirt

a dog wearing a shirt and glasses doing skydiving

a forest with a light betweeen the trees

a doll wearing a hat and a shirt sitting on a blanket in the park under a light

a dog wearing a hat and necklace

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

a man leaning against a wall

a man wearing a shirt

a cat wearing glasses and a shirt

[Figure 187]

[Figure 188]

[Figure 189]

a standing boy a woman posing house on a tree

a dog wearing a hat and a necklace

a chair and a table in a living room

a man wearing shirt, glasses and necklace sitting on a chair in a living room next to a table

a boy posing in the garden

a mirror on a wall and a table

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

a doll wearing a jacket

a mosaic vase with ﬂowers

a doll on the snow

a lamp next to a chair and a table

a man wearing a shirt

a backpack hanging on a chair

a doll wearing a shirt and a chef's hat made of mosaic cooking a meal

a doll standing on a chair tries to replace the bulb in lamp

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

a creature sitting on a bowl

a dog wearing a hat and a necklace

a lamp next to a chair and a table

[Figure 206]

[Figure 207]

[Figure 208]

a dog wearing a hat and a necklace

a cat wearing glasses and a shirt

a woman wearing a necklace

a dog wearing a shirt, glasses necklace ﬂying in the sky tied to baloons

a pineapple and a bottle on a table

a doll inside a bucket on a cart

a creature wearing necklace and sitting on a chair, a pineapple and a doll on a table

a mirror on a wall and a table

###### Figure 16. Qualitative results. Each row contains two result images and the source images of the concepts that they contain.

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

a doll sitting on a bench

a dog wearing a hat and a necklace

a cat wearing glasses and a shirt

[Figure 217]

[Figure 218]

a dog wearing a hat and a necklace

ﬂip ﬂops with a pattern in the sand

a laying dog wearing glasses

a dog with a shirt with texture pattern, glasses on a rollercoaster

a parking car

a doll wearing a shirt, glasses , and hat driving a car near a brick wall

a bicycle is leaning on a brick wall

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

a doll wearing a jacket

a cat wearing glasses and a shirt

[Figure 226]

[Figure 227]

a man leaning a woman posing house on a tree against a wall

a man posing in the garden

a forest with a light betweeen the trees

a dog wearing shirt, glasses and necklace in the park under a light

a dog wearing a hat and a necklace

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

a man leaning against a wall

a doll sitting on a bench

a doll wearing a jacket

a laying dog wearing glasses and necklace

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

a man on a bench, holding an umbrella by the sea

a doll wearing a necklace and glasses on a boat

a woman is holding an umbrella

a woman posing near the sea.

a boat is ﬂoating on the water

a dog with glasses, necklace and shirt

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

a dog wearing a hat and necklace

a doll wearing a jacket

a cat wearing glasses and a shirt

a woman wearing a necklace

a man wearing a shirt

a rabbit wearing glasses

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

a dog wearing shirt, glasses and ecklace and wide hat in the garden having a picnic

doll wearing a shirt, glasses and necklace ﬂying in the sky using an umbrella

a laying dog wearing glasses

a doll sitting on a bench in the garden

a woman wearing a necklace

a woman is holding an umbrella

###### Figure 17. Qualitative results. Each row contains two result images and the source images of the concepts that they contain.

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

a toy sits near a bowl and seashell

a doll inside a bucket on a cart

a boat is ﬂoating on the water

a dog wearing a hat and necklace

a cat wearing glasses and a shirt

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

a doll sailing a boat in the river, using an umbrella as a sail, under light

a backpack hanging on a chair

a forest with a light betweeen the trees

a toy wearing a hat, a shirt and glasses on a blanket with a backpack in the park under light

a forest with a light betweeen the trees

a rabbit wearing glasses

a woman is holding an umbrella

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

a doll sitting on a bench

a dog wearing a hat and necklace

ﬂip ﬂops with a pattern in the sand

a dog wearing a hat and a necklace

a cat wearing glasses and a shirt

a rabbit wearing glasses

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

a boat is ﬂoating on the water

a doll wearing a shirt with pattern, necklace and glasses in the fog

a woman standing in a fog

a cat wearing glasses and a shirt

dog wearing a shirt, glasses and necklace on a boat

a laying dog wearing glasses and necklace

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

a doll wearing a jacket

a man leaning against a wall

a backpack hanging on a chair

a woman posing

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

a doll holding a backpack and an umbrella on a boat

a boat is ﬂoating on the water

a man posing on a bench under a light, 3d rendered cartoon

a woman is holding an umbrella

a doll sitting on a bench

a tv on a table in under a light

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

a doll inside a bucket on a cart

a dog made of plastic

[Figure 290]

[Figure 291]

a doll inside a bucket on a cart

a man in the snow under a light.

a doll in a bucket made of plastic on a carpet under a light

a rocking horse on a carpet

a tv on a table in under a light

a doll in its bed, light coming through the window

###### Figure 18. Qualitative results. Each row contains two result images and the source images of the concepts that they contain.

###### Characters Storytelling using Concept Images

[Figure 292]

[Figure 293]

[Figure 294]

 e sun rises, cas ng a warm glow on Toby’s  eehouse. Outside, Toby is prac cing his morning yoga. Woolson  e sheep watches him,  ying mimic  e poses wi  a play l bleat.

[Figure 295]

Af r his yoga, Toby feels re eshed and ready for a new inven on. He decides work on a super-duper bubble blower!

Zippy Toby

[Figure 296]

[Figure 297]

[Figure 298]

As he  nkers, Sheldon  e snail arrives, slowly but surely, wi  a special delivery: an invita on Jingles’ ju ling show in  e  wn square!

[Figure 299]

Sheldon Woolson

Toby and Woolson decide head  e  wn square.

###### Scenes

[Figure 300]

[Figure 301]

[Figure 302]

On  eir way,  ey meet Zippy  e bluebird. Zippy, being his usual mischievous self, grabs some of Woolson’s ﬂuﬀy wool make a nest!

[Figure 303]

 ey arrive at  e bustling marketplace. Toby is amazed by a   e sights and sounds.

Treehouse Marketplace

###### Pose

[Figure 304]

[Figure 305]

Zippy gets dis ac d by shiny  inkets, while Woolson eyes some delicious-looking  uits.

Yoga pose

- Figure 19. Storytelling results. Demonstration of our method’s usability for storytelling applications. All the characters, scenes, and poses featured in the story are shown on the left. On the right is the story itself, generated by a language model (LLM). This story was then reprocessed by the LLM to generate prompts, which were used to create the accompanying images.

(a) Prompt Alignment (b) Concept Preservation

[Figure 306]

[Figure 307]

[Figure 308]

- Figure 20. An example of the questions asked in the user study. Given a generated image the users are asked about its alignment with both the text and the input concepts

[Figure 309]

“Sailing through serene turquoise waters, surrounded by lush gree hills and peaceful horizons.”

[Figure 310]

“A caracal moves silently through the wild.”

[Figure 311]

“Paragliding over a beautiful mountain valley.”

[Figure 312]

“A boat sailing down a river at sunset,with lush green trees lining the banks.”

[Figure 313]

“A family having a picnic in front of a volcano.”

[Figure 314]

“A beautiful wedding ceremony taking place in a sunlit field.”

[Figure 315]

“A red fox stands by a calm lake with snow-capped mountains in the background.”

[Figure 316]

“Children standing with their backs to the camera in a forest..”

[Figure 317]

“A couple having a romantic picnic by a cascading waterfall.”

[Figure 318]

“A beautiful wedding ceremony taking place in a sunlit field.”

[Figure 319]

“A vibrant scarlet macaw with its wings outstretched, flying through a tropical rainforest.”

[Figure 320]

[Figure 321]

“A man playing piano” “A woman riding a scooter”

[Figure 322]

[Figure 323]

“A snake as a messenger”

“A lone hiker sitting on a rock on a ridge, meditating and connecting with the peace and tranquility of nature”

[Figure 324]

“A giraffe having a snowball fight with friends”

[Figure 325]

“A giraffe decorating a Christmas tree”

[Figure 326]

“A woman with an owl-shaped dreamcatcher hanging in her car”

[Figure 327]

“A lemur sitting at a chessboard, contemplating its next move, its opponent a wise old owl”

[Figure 328]

“A lemur dribbling a basketball with impressive skill, its eyes on the hoop, ready to shoot”

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

“A fashion accessory, a watch strap crafted from genuine crocodile leather, its unique texture and pattern adding a touch of sophistication and luxury, sparking ethical debates about animal welfare and sustainability”

“A cartoon deer with large, expressive eyes and exaggerated movements, designed to bring laughter and entertainment to viewers of all ages, a lighthearted and whimsical character that embodies the joy of animation.”

“A group of friends volunteering to remove invasive plant species from

“A kestrel with its hovering flight pattern, flying over a playground, its colorful plumage contrasting with the bright colors of the equipment”

“A powerful image of an antelope standing alone in a harsh and unforgiving environment, its silhouette casting a long shadow across the sand dunes, its presence a testament to its ability to survive and thrive in challenging conditions a symbol of strength and adaptability”

a ridge, helping to restore its native ecosystem”

- Figure 21. Generated images used for concept isolation loss. The images were generated with the base Flux model according to the accompanying prompts. 21

