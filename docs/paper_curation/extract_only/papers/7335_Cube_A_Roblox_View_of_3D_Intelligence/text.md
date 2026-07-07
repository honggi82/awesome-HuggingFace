[Figure 1]

# arXiv:2503.15475v3[cs.CV]18Jul2025

## Cube: A Roblox View of 3D Intelligence

Foundation AI team, Roblox1

### Abstract

Foundation models trained on vast amounts of data have demonstrated remarkable reasoning and generation capabilities in the domains of text, images, audio and video. Our goal at Roblox is to build such a foundation model for 3D intelligence, a model that can support developers in producing all aspects of a Roblox experience, from generating 3D objects and scenes to rigging characters for animation to producing programmatic scripts describing object behaviors. We discuss three key design requirements for such a 3D foundation model and then present our first step towards building such a model. We expect that 3D geometric shapes will be a core data type and describe our solution for 3D shape tokenizer. We show how our tokenization scheme can be used in applications for text-to-shape generation, shape-to-text generation and text-to-scene generation. We demonstrate how these applications can collaborate with existing large language models (LLMs) to perform scene analysis and reasoning. We conclude with a discussion outlining our path to building a fully unified foundation model for 3D intelligence. Our code and model weights can be found at: https://github.com/Roblox/cube2.

[Figure 2]

Figure 1 | Scene Generation. We are developing a foundation model for 3D intelligence that will support applications like scene generation. This winter village scene was generated through a multi-turn conversation with our prototype scene generation tool.

1See Appendix for the full author list. 2We have updated the code and model since the release in March 2025. Please see Section 5 for more details.

[Figure 3]

###### Applications

###### Discrete Shape Tokenization

###### Text-to-Shape / Shape-to-Text Generation

[Figure 4]

Text-to-Shape GPT

Shape Tokenization

“A unicorn with mane and tail, […]”

Shape-to-Text GPT

Input Mesh

Input Text

Text Tokens Shape Tokens

Mesh

###### Text-to-Scene Generation

…

[Figure 5]

“A circular end table, […]”

[Figure 6]

[Figure 7]

[Figure 8]

Discrete Shape Tokens

“A cozy living room with a […] sofa and a […] end table on a […] rug. A ceiling fan […] is overhead, and the room includes a TV stand […]”

Rug TV Stand

“A modern threeseater sofa, […]”

Fan

……. …….

…….

[Figure 9]

Table

Shape De-Tokenization

Sofa

“A ceiling fan with a light fixture, […]”

Decoded Mesh

Input Text Generated Scene

Scene Layout

Shape Tokens

- Figure 2 | Overview. We present an important step towards the foundation model for 3D intelligence. Specifically, our report focuses on 3D shape tokenization—a technique for converting between shapes and discrete tokens. We also demonstrate how our tokenization scheme enables multiple applications including text-to-shape generation, shape-to-text generation, and text-to-scene generation.

#### 1. Introduction

Modern generative AI promises to give every human a collaborative assistant for creating visual content. An assistant that can convert simple inputs, such as text prompts, into productionquality visual experiences. Today, these assistants typically rely on an underlying foundation AI model that is trained on vast amounts of broad, multi-modal data, and then fine-tuned to perform specific downstream assistive tasks(Bommasani et al., 2021). Recent foundation models that bridge the domains of text, image, audio and video have demonstrated remarkable reasoning and generation capabilities(Achiam et al., 2023; Team et al., 2024).

At Roblox, our goal is to build a foundation model for 3D intelligence. We envision this model as the base for a variety of collaborative assistants that will aid developers in producing all aspects of Roblox experiences from creating individual 3D objects (e.g., make a motorbike with wings), to full 3D scene layout (e.g., create a futuristic cloud city), to rigged character avatars (e.g., generate an alien ninja capable of doing wall jumps) to scripts describing object behaviors, interactions and game logic (e.g., make the door open when a player is near it and carrying a gold key). We believe that building such a foundation model for 3D reasoning and generation imposes three core design requirements.

- • Learn jointly from sparse, multi-modal data. The amount of readily available 3D experience data suitable for training is much less than the data available for training text, image or video models. But 3D data is multi-modal (e.g., geometric meshes, CSG parts, layouts, textures, rigging, programmed scripts) and the modalities are strongly correlated with each other (e.g., an object’s texture and placement in a scene often strongly correlate with its shape). Thus, our model should jointly learn from all the modalities available in the training data.
- • Handle unbounded input/output size via an autoregressive model. 3D experiences can be of vastly different sizes, such as a micro-scene containing a few static objects (e.g., a room with a dinner table and chairs) to a multi-apartment building with interactive doors and elevators, to a large-scale city with interacting vehicles and non-player characters. To handle inputs and outputs at each of these scales, our model should be autoregressive

[Figure 10]

[Figure 11]

Latent Space Regularization with Self-supervised Loss

Phase-Modulated Positional Encoding

Input Mesh

Pointcloud

KV

Q

Self Attention x 12

Cross Attention

[Figure 12]

[Figure 13]

Learnable Queries

Continuous Latent Vector

Optimal Transport Vector Quantization

KV

Stochastic Gradient Shortcut (50% of training time)

Cross Attention Self Attention x 24

Decoded Mesh Occupancy Field

Q

Phase-Modulated Positional Encoding

Grid Query Points

Discrete Shape Tokens

- Figure 3 | Shape Tokenization. Our pipeline encodes an input mesh into discrete tokens through several steps: (1) We sample points from the mesh surface and embed them using our Phased-Modulated Positional Encoding; (2) A Perceiver-based transformer (Jaegle et al., 2021) encodes these points into continuous latent vectors, regularized with a self-supervised loss; (3) We apply optimal-transport vector quantization (Zhang et al., 2024) to convert these vectors into discrete shape tokens; (4) These tokens can later be decoded into an occupancy field for mesh extraction. To improve training stability and the reconstruction quality, we also introduce a Stochastic Gradient Shortcut layer that allows the decoder to utilize the continuous latent vectors directly during training.

with large context length.

• Collaborate with humans and other AI systems via multi-modal inputs/outputs. Humans are familiar with data modalities such as natural language text, images, sketches and video and should be able to express design intents using any of them. Moreover, existing large language models (LLMs) such as GPT-4o (Achiam et al., 2023), which are trained with vast amounts of multi-modal data including text, programs, images, vector graphics, scene graphs, video, speech, audio, etc. can provide common sense knowledge (i.e., general priors) on a variety of topics. Thus, for our model to seamlessly collaborate with humans as well as other LLMs, multi-modal data should be first-class input and output data types.

In this report we present our first step towards this foundational model for 3D intelligence. Specifically, we focus on discrete tokenization of the 3D shape, recognizing that geometry will be a central data type of our foundation model. We show how our tokenization scheme can be used to build several applications including text-to-shape generation, shape-to-text generation, and text-to-scene generation (Figures1 and2). We also demonstrate how these applications can collaborate with a text-based LLM to perform scene analysis and reasoning tasks. Our code and inference model weights are available at: https://github.com/Roblox/cube.

#### 2. Shape Tokenization

We need a representation of 3D geometry that is expressive enough to capture a wide range of geometric properties faithfully, including smooth surfaces, sharp edges, high-frequency details, and can serve as both input and output tokens to multi-modal autoregressive sequence models. Faced with these needs, we start from an expressive, continuous shape representation such as 3DShape2VecSet (Zhang et al., 2023a), and adapt it into discrete tokens to enable native handling

[Figure 14]

[Figure 15]

(a) Baseline Positional Encoding 𝛾(·) (b) Phase-Modulated Positional Encoding 𝛾′(·)

- Figure 4 | Phase-Modulated Positional Encoding. Comparison of positional encoding methods using 128 base frequencies for 1D inputs in [−1,1], with corresponding dot-product similarity matrices. (a) Traditional positional encoding 𝛾(·) enhances high-frequency details but exhibits periodic variations across embedding dimensions as inputs vary from −1 to 1. This causes dot-product similarities between encoded vectors to poorly reflect their spatial proximity. (b) Our proposed PMPE 𝛾′(·) maintains clear distinctions between spatially distant inputs, as reflected in the dot-product similarity of encoded vectors. Our final approach uses the combined encoding of 𝛾′ +𝛾 that preserves fine-grained details while ensuring similarity patterns align with global spatial proximity.

of input and output tokens across multiple modalities in a manner similar to mixed-modal foundation models such as Chameleon (Team, 2024).

As outlined in Figure 3, our high-level architecture follows an encoder-decoder design that encodes the input 3D mesh into a latent representation capable of being decoded into an implicit occupancy field. One key distinction is that the continuous latents get discretized through an additional vector quantization process, which poses unique training challenges due to its non-differentiability nature. We propose two techniques, namely stochastic gradient shortcut and self-supervised latent space regularization, to address the VQ training challenges ( Section 2.2 and 2.3, respectively). Another architecture improvement is the use of phasemodulated positional encodings that improve the perceiver-based transformer’s ability to disambiguate spatially distinct points in cross-attention layers (Section 2.1). These architecture changes allow us to train a shape tokenizer that faithfully captures a wide range of shapes while yielding discrete tokens that are suitable as a new modality for training token-based mixed-modal foundation models.

###### 2.1. Phase-Modulated Positional Encoding

To encode a shape into a compact latent representation, we first sample 𝑁𝑝 points from its surface to create a point cloud P ∈ R𝑁

𝑝×3 . Prior work (Li et al., 2024; Zhang et al., 2023a; Zhao et al.,

- 2023, 2025) embeds P via a sinusoid positional encoding function 𝛾(·) (Mildenhall et al., 2020) before processing it with a transformer network:

𝛾(𝑝) = [sin (𝜔1𝑝 + 𝜑1), sin (𝜔2𝑝 + 𝜑2), · · · ,sin (𝜔𝐿𝑝 + 𝜑𝐿)] , (1) where 𝛾(𝑝) is applied separately to each of the three coordinate channels 𝑝 ∈ [𝑥, 𝑦, 𝑧] in P, and 𝜔𝑖 = 2⌊𝑖/2⌋𝜋, 𝜑𝑖 = 𝜋2 (𝑖 mod2), for 𝑖 = 1, · · · , 𝐿, with 𝐿 being the number of base frequencies.

The periodicity of 𝛾(·) maps points separated by multiples of 2𝜋/𝜔𝑖 in space to identical encodings in the 𝑖𝑡ℎ channel. The result is that spatially distant points can map to similar embedding vectors (Figure 4a) that are hard to distinguish after the dot-product in crossattention layers. The inability for the embedding to disambiguate spatially distant points, and

Q

- ps
- pt

Student Encoder

MLP Head

[Figure 16]

[Figure 17]

Continuous Latent Vectors

Masked Learnable Queries

KV

Cross-Entropy Loss: →pt log ps

EMA

KV

Input Mesh

Pointcloud

Stop Gradient

Q

Teacher Encoder

MLP Head

Continuous Latent Vectors

Learnable Queries

- Figure 5 | Latent Space Regularization with Self-supervised Loss. We regularize our latent space using a self-supervised loss inspired by DINOv2 (Oquab et al., 2023). This loss is computed as the cross entropy between features extracted from the student and teacher encoders, where the teacher model is an Exponential Moving Average (EMA) of the student. Input queries to the student encoder are randomly masked for robustness.

correspondingly, to disambiguate different shape surface features, results in reduced shape reconstruction quality.

To address this problem, we need a new technique to encode points such that it not only captures the multi-scale features as in the traditional position encoding but preserves the distinctiveness of spatially distant points with the dot-product attention. We take inspiration from phase modulation techniques (Haykin, 2008) and propose Phase-Modulated Position Encoding (PMPE). PMPE modulates phase offsets across all the sinusoidal functions, and uses the embedding function 𝛾PM, defined as:

𝛾PM(𝑝) = 𝛾(𝑝) + 𝛾′(𝑝), 𝛾′(𝑝) = sin (

𝜋

𝜋

𝜋

𝑝 + 𝜑1′), sin (

𝑝 + 𝜑2′), · · · ,sin (

𝑝 + 𝜑′𝐿) ,

(2)

2

2

2

𝑖 𝐿

𝑖 𝐿

𝜑′𝑖 = 2𝜋 (𝛽𝐿)1−

, 𝑖 = 1, · · · , 𝐿,

+

where 𝛾(𝑝) is the traditional encoding function, and 𝛽 is a hyperparameter that controls the rate of phase variation across channels. The term (𝛽𝐿)(1−𝐿𝑖 ) here is to vary the base frequency to avoid resonance between 𝛾(𝑝) and 𝛾′(𝑝).

In contrast to 𝛾(𝑝), which uses exponentially increasing frequencies to capture multi-scale features, 𝛾′(𝑝) uses the same frequency 𝜋/2 for each channel but varies the phase offset 𝜑′

𝑖 via a non-linear function of 𝑖. The non-linear phase modulation ensures the distinctiveness of spatially distant points when mapped to the embedding space, as shown in Figure 4b.

Empirically, we find that PMPE yields significantly improved reconstruction fidelity, particularly for complex geometric details. PMPE also produces fewer artifacts such as disconnected components that appear in methods that use only traditional positional encoding.

###### 2.2. Stochastic linear shortcut for gradient stabilization

After encoding the input shape into a continuous latent vector, we employ a variant of VQVAE (van den Oord et al., 2017) (i.e., Optimal Transport VQ (Zhang et al., 2024)) to convert the latents into a sequence of discrete tokens. The quantization layer in VQ-VAE can introduce

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

(a) Without self-supervised loss (b) With self-supervised loss

- Figure 6 | Impact of Self-Supervised Loss. Cosine similarity between the latent vectors of 3D meshes for models trained (a) with and (b) without the proposed self-supervised loss term. (a) Without regularization, latent cosine similarity scores fail to correlate with geometric structure (e.g., the ice-cream in 4𝑡ℎ row has higher similarity to the car in 1𝑠𝑡 row than the similar car in 2𝑛𝑑 row). (b) With the loss, latent space similarity aligns with ground-truth geometric relationships.

training instabilities due to ill-defined gradients from the non-differentiable code embedding assignment (Huh et al., 2023). Prior works have attempted to use a straight-through estimator (Bengio et al., 2013) or randomly replace a fraction of the quantized embeddings with their continuous approximations during training (Takida et al., 2022; Zhang et al., 2023b). Inspired by the prior works, we make a further adjustment by introducing an additional linear shortcut layer that stochastically skips the entire quantization bottleneck.

With 50% probability, we project the continuous latents from the encoder with a linear layer, and directly feed the projected latents to the decoder, bypassing the quantization layer entirely. Notice that this is different from the direct shortcut approach analyzed in the previous work (Fifty et al., 2024) that is equivalent to using an identity (instead of linear) layer in our formulation. Both Fifty et al. (2024) and our own experiments have found the direct shortcut approach to not perform well. Our intuition is that the additional linear layer in our formulation allows the shortcut pathway to learn a slightly different set of weights but with well-defined gradients, and it allows the shortcut pathway to act as a teacher network for the quantization pathway. As a result, the quantization pathway is less prone to being stuck at a local minima. Empirically, we observe that incorporating the linear shortcut layer leads to a lower training and validation loss and makes training more stable under a broader range of hyperparameters.

###### 2.3. Learning geometrically clustered latents with self-supervised loss

Following prior work in vision models (Caron et al., 2021; Grill et al., 2020; Oquab et al., 2023; Zhou et al., 2021), we employ a self-supervised loss to regularize the latent space to be geometrically clustered and smooth, where similar shapes produce latent vectors that are close to each other. Figure 5 illustrates our encoder’s self-supervised learning pipeline. Following the architecture in DINOv2 (Oquab et al., 2023), we maintain an exponential moving average version of the encoder as the teacher model. The student encoder receives randomly masked

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Groundtruth Ours-KL Ours-VQ Craftsman

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

- Figure 7 | Qualitative Analysis of Shape Reconstruction. Comparison between our method and Craftsman (Li et al., 2024) demonstrates that both of our model variants achieve superior reconstruction quality, preserving finer geometric details while producing fewer artifacts.

input queries, while the teacher encoder has access to the full set of queries. Both encoders use an additional MLP head to produce “prototype scores”, and the self-supervised loss is the cross-entropy between these scores, weighted by a hyperparameter 𝜆𝑆𝑆𝐿 that balances with the occupancy reconstruction loss. As shown in Figure 6, the ability to encode geometrically similar shapes into latent vectors with large cosine similarities naturally emerges from the additional self-supervised loss. We anticipate that this property will prove valuable for a wide range of shape processing applications.

###### 2.4. Results

Architecture details. For training our shape tokenizer, we use similar transformer layers for both the encoder and the decoder, with each layer having 768 width and 12 heads. The encoder and decoder consist of 13 and 24 transformer layers respectively, with a total number of 273M parameters. We use 512 latent code tokens and a codebook size of 16,384 with 32-dimensional embeddings. We set 𝛽 = 0.125 for PMPE and 𝜆𝑆𝑆𝐿 = 0.0005 for the self-supervised loss. For the VQ layer, we use a recent variant called OptVQ (Zhang et al., 2024) that integrates an optimal transport method for index assignment.

Training data. We train our shape tokenizer on roughly 1.5 million 3D object assets that utilize a combination of licensed and public available datasets (e.g. Objaverse (Deitke et al., 2023)),

- as well as the free data assets that are available in Roblox Creator Store that are opted in for training. We center and rescale all assets to lie within a normalized bounding box with each axis ranging from [−1,1]. For each asset within a training batch, we sample 8,192 points directly on its surface to facilitate input encoding. To compute the occupancy loss, we sample another 8,192 points comprising two different sets, one from uniform sampling in the space to capture the overall geometric structure and the other from near-surface to capture high-frequency surface details.

Model comparison. We compare our discrete shape tokenizer with a continuous variant trained on the same 1.5 million object dataset, and with CraftsMan (Li et al., 2024) a recent variant of 3DShape2VecSet trained on a 170K object subset of Objaverse. Our continuous variant is trained using the same architecture as the discrete tokenizer, with the only difference that it skips the

###### CraftsMan (Li et al., 2024) Ours-VQ Ours-KL

S-IoU↑ 68.8% 91.7% 94.8% V-IoU↑ 83.6% 94.5% 95.4%

Table 1 | Comparison of S-IoU and V-IoU results across different models.

VQ process and uses KL-divergence to regularize the latent space (Kingma et al., 2013). We measure the shape reconstruction quality of different methods using surface-IoU (S-IoU) and volumetric IoU (V-IoU) on the Toys4K (Stojanov et al., 2021) dataset. We calculate S-IoU using points near the mesh surface and V-IoU using uniformly sampled points within the bounding volume. None of the three methods was trained on the Toys4K dataset.

As shown in Table 1 and Figure 7, both our VQ-VAE (Ours-VQ) and the continuous variant (Ours-KL) outperform CraftsMan in both S-IoU and V-IoU metrics. Our continuous variant still outperforms its discrete counterpart, highlighting that there remains some loss of geometry fidelity through the vector quantization process. We aim to bridge this gap in future work.

#### 3. Applications

Shape tokenization is only one component of the unified foundation model we envision, but we have already found it to be an effective representation of geometric data. We have begun developing a variety of applications using our shape tokens as the core geometric representation. These applications include text-to-shape generation, shape-to-text generation and text-to-scene generation. We briefly explain how each application takes advantage of our shape tokenization scheme and demonstrate how they might aid users in developing 3D experiences.

###### 3.1. Text-to-shape generation

Our text-to-shape application allows users to convert a text description of a shape into a triangle mesh model that can serve as an element of a 3D experience.

Architecture. Our text-to-shape architecture is a decoder-only transformer similar to GPT-2 (Radford et al., 2019). We tokenize the shape into discrete tokens and use the transformer to generate the shape tokens with text conditioning. Specifically, we encode input text prompts using a pre-trained CLIP text encoder (Radford et al., 2021), and leverage a dual-stream attention (Esser et al., 2024) to inject the text conditioning into the transformer model to output shape tokens in an autoregressive manner. We also incorporate Classifier-free Guidance (Ho and Salimans, 2022) into our GPT training and inference pipeline. During training, we randomly drop the text conditioning and replace it with an empty string 10% of the time following (Sun et al., 2024).

Training data. Training our text-to-shape application requires paired examples of text prompts and their corresponding 3D shapes. For each of the 1.5 million assets used to train the shape tokenizer, we render multiple views and use GPT-4o (Achiam et al., 2023) to provide captions of various lengths.

Mesh extraction. We use marching cubes (Lorensen and Cline, 1998) to extract the iso-surface from the occupancy field, along with an in-house mesh decimation algorithm built upon the quadric errors (Garland and Heckbert, 1997) for simplifying the mesh into desired number of faces. Finally, we apply a post-processing step that removes small disconnected components (i.e. floater artifacts) from the output mesh.

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

A sleek vintage green couch

An alarm clock with metallic bells

A velvet armchair with sleek walnut legs

A large decorative barrel

A crystal blade fantasy sword

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

A sturdy wooden crate with planks

A rectangular bed pillow

A vibrant potted herb plant

A wooden table covered with a cloth

A sturdy, steel warehouse shelf

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

A wooden picnic table

A buggy with knobby tires

A sleek fire extinguisher

A light fixture with a metal frame

A small tealight candle

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

A ceramic pot holds a monstera plant

A coffee maker with a ceramic body

A crystal blade fantasy sword

A glamorous Barbiestyle bed

A cartoon whale

A small bench ottoman

A low poly soap dispenser

A small porcelain espresso cup

A moto leather jacket

A low-polygon stylized crab

- Figure 8 | Text-to-Shape Generation Result Gallery. Our model can generate a diverse set of 3D meshes, capturing sharp edges, smooth surfaces and complex structures.

[Figure 73]

##### Input Shape Caption Short Caption Medium Caption Long

The object is a vintage biplane with a classic, old-fashioned design. the biplane has a single propeller at the front, with visible struts connecting the wings. overall design is compact and robust (...)

A vintage biplane with two stacked wings, a single propeller, narrow fuselage, and a small tail stabilizer.

A vintage biplane.

The object is a whimsical, anthropomorphic character with a smooth, rounded body primarily colored in a bright yellow. it has a playful, cartoonish appearance with exaggerated features such as large, expressive eyes and a wide mouth displaying prominent teeth. (...)

[Figure 74]

A cartoon dog with a rounded body, ﬂoppy ears, short legs, and a protruding snout, standing upright.

A cartoon dog.

[Figure 75]

The object is a stylized hat with a tall, pointed, and slightly curved top that tapers to a point. it features a wide, ﬂoppy brim that curves upward, giving it a classic wizard or wizard-like appearance. the material appears to be a smooth, light-colored fabric, possibly fabric or a similar substance. the hat's design is simple and elegant, with a slight sheen that suggests a polished ﬁnish. (...)

A traditional wizard hat with a tall, pointed, and slightly curved top, a wide circular brim, and a decorative band.

A wizard hat.

- Figure 9 | Shape-to-text examples. Shape-to-text captioning of example shapes from Toys4K dataset using short, medium and long captions. Words highlighted in blue indicate notable differences as captions increase in length. Short captions usually capture the shape category. More details about the shape’s geometry and style are added as caption length increases.

Results. We show a representative set of text-to-shape generations in Figure 8. Our text-to-shape model is able to generate a diverse set of 3D meshes, capturing sharp edges, smooth surfaces and complex structures. These meshes, as produced with our discrete shape tokens are approaching the visual quality to those produced using recent methods based on rectified flow transformers with continuous latents such as Trellis (Xiang et al., 2024), Hunyuan3D-2 (Zhao et al., 2025) and TripoSG (Li et al., 2025). While a benefit of our approach is that it can easily support multi-modal tokens, a full understanding of the relative strengths of the different approaches is an area of future work.

###### 3.2. Shape-to-text generation

Our shape-to-text application produces a descriptive natural language caption for an input 3D shape. These text captions are designed to capture 3D shape information in the sense that they can be fed back to our text-to-shape application to produce corresponding 3D models. And, as we will show in Section3.3, these text captions allow us to build AI tools that can seamlessly collaborate with existing LLMs to leverage their common sense knowledge and reasoning capabilities.

Our architecture is inspired by recent work on vision-language models such as LLaVA (Liu et al., 2023). In order to perceive visual signals in vision-language models, a pre-trained openset visual encoder, such as CLIP(Radford et al., 2021), is commonly used to encode the visual input into the input space of the language decoder. In our case, we encode the 3D shape input with our shape tokenizer and inject the shape tokens into a pre-trained decoder-only transformer (Vaswani et al., 2017) with a single two-layer MLP latent projection layer. For the pre-trained LLM backbone, we chose the language model from InternVL 2.5-2B (Chen et al.,

- 2024), a multimodal large language model pre-trained on large-scale image-text data. We choose

##### Input Shape Shape-to-Text Text-to-Shape

[Figure 76]

[Figure 77]

A futuristic humanoid robot with large, articulated arms, bulky, segmented legs, and a compact, detailed torso. the head is small and integrated into the body.

[Figure 78]

[Figure 79]

A modern farming tractor with large rear wheels, smaller front wheels, a covered cab, and a robust rectangular body, designed for heavy-duty farm work.

- Figure 10 | Shape cycle consistency. Our shape-to-text and text-to-shape models demonstrate the cycle consistency. Given a shape, we caption it using shape-to-text model, then then regenerate the 3D shape using the text-to-shape model. We highlight some keywords in blue for the shape category and style. The process preserves the overall geometry and key characteristics of the original shape, although there can be some loss in fine-scale details.

this backbone due to its outstanding performance on a wide range of multimodal benchmarks. Our architecture design follows a standard multimodal learning practice and it demonstrates that our shape tokenizer can be adapted and applied in standard multimodal setups.

Training. Since the shape tokenizer is trained on the shape dataset only, the shape latent space can be different from that of the language transformer. Following the training recipe of LLaVA (Liu et al., 2023), we adopt a two-stage training pipeline. In the first stage, only the latent projection layer is trained to align the shape features with the text features of the pre-trained transformer. In the second stage, we jointly finetune the latent projection layer and the weights of the language transformer to better align the shape features with the text features. We keep the shape tokenizer frozen during the entire training process. Similar to LLaVA, we train the model using the next-token prediction loss and only apply the loss to the text output sequence.

Implementation details. We use the same dataset we developed for text-to-shape generation (Section3.1). In the dataset, we have captions of various length. In order to control the output caption length in the shape-to-text generation, we append a short text instruction after the input shape tokens to indicate the desired output caption length. Specifically, we use “caption short:“,

“caption medium:“, and “caption long:“ for caption lengths of less than 25 tokens, less than 75 tokens, and greater than 75 tokens respectively.

Results. Figure 9 shows captions of various lengths generated by our shape-to-text model. When producing short captions, the model only gives category information about the input shape (e.g., a cartoon dog). As the caption length increases, the model describes different parts of the object (e.g., rounded body, floppy ears, etc.) and styles (e.g., playful, cartoonish appearance, large, expressive eyes, etc.).

Shape cycle consistency. Our goal is to capture a notion of 3D shape in the natural language captions produced by our shape-to-text model. Figure 10 shows examples where we start from shape input, generate corresponding text captions using our shape-to-text model, and then regenerate the shape by treating the captions as inputs to our text-to-shape model. The results show that the captions produced by our shape-to-text model capture enough 3D information for our text-to-shape model to reproduce the overall 3D shape, albeit with some loss of high-

{

###### "scene_id": "scene_001", "objects": [

{

"object_id": "e60d6ea3a2044786af17e11407f4430f", "object_category": "table", "object_caption": "a modern rectangular table with a thin, circular pedestal base, featuring a sleek, minimalist ↩→ design and a flat top surface", "position": "[-8.73, 4.56, -10.48]", // center of the bounding box "extent": "[3.75, 2.72, 5.32]", // x,y,z axis "rotation": "1.57" // rotation angle around Y (UP) axis in radians

}, "..."

] }

- Figure 11 | Scene graph with text-based object descriptions. For each scene object, the scene description includes the type of object (object_category) and a text description of its shape (object_caption). These descriptions facilitate LLM understanding of the scene. To support the simple case of orienting objects on a ground plane, our current implementation only supports Y axis rotations.

Figure 11 | Scene graph with textual object descriptions. For each scene object, the scene description includes the type of object (object_category) and a text description of its shape (object_caption). These descriptions facilitate LLM understanding of the scene. Our application only supports Y axis rotation for simplicity reason, it should support rotation in all axis.

frequency geometric details. We hypothesize that shape cycle-consistency emerges from the fact that the same shape tokens are used by the shape-to-text and text-to-shape models.

###### 3.3. Text-to-Scene generation

Our text-to-scene application combines the text-to-shape and shape-to-text models from the prior sections with an off-the-shelf LLM to provide users the ability to iteratively construct full scenes using only text descriptions. From simple prompts like “Make a campsite” and also corrective instructions like “Add another tent”, our scene generation application creates a collection of objects and lays them out (sets position, orientation and scale) in a scene configuration. The system can also make suggestions to the user about what additional elements should be added to the scene (e.g., consider adding a camp fire). [MA: Can we do a campsite example, or should we change the example described here.]

###### 3.3. Text-to-Scene generation

Our text-to-scene application combines the text-to-shape and shape-to-text models from the prior sections with an off-the-shelf LLM to provide users the ability to iteratively construct full scenes using only text descriptions. From simple prompts like “Make a campsite” and also corrective instructions like “Add another tent”, our scene generation application creates a collection of objects and lays them out (sets position, orientation and scale) in a scene configuration. The system can also make suggestions to the user about what additional elements should be added to the scene (e.g., consider adding neon lights to a 60’s-style diner scene).

###### 3.3.1. Scene graph representation

Scene graph representation. Our shape-to-text model allows us to convert entire scene descriptions (the individual object shape descriptions and description of their placement in a scene) into a single text-based scene-graph that can be consumed and processed by LLMs. Our scene graph format is JSON-based, and represents a scene as a flat list of 3D objects (Figure 11). Each object includes a text description of the object’s shape (object_caption) and the type of object it is (object_category). These fields provide a sufficiently detailed description of the 3D object for an LLM to be able to reason about some of its spatial and semantic properties. Objects also have scene layout information giving the position, orientation, and bounding box of the object in the scene. Given a 3D scene (such as via a GLTF file), we convert the scene to our textual scene graph by extracting all references to 3D models and replacing them with their bounding box and the appropriate text descriptions produced by our shape-to-text model (Section 3.2). Conversely, given a scene graph containing only a textual description of objects, we can convert all textual object descriptions into 3D geometry using text-to-shape (Section 3.1).

Our shape-to-text model allows us to convert entire scene descriptions (the individual object shape descriptions and description of their placement in a scene) into a single text-based scenegraph that can be consumed and processed by LLMs. Our scene graph format is JSON-based, and represents a scene as a flat list of 3D objects (Figure 11). Each object includes a text description of the object’s shape (object_caption) and the type of object it is (object_category). These fields provide a sufficiently detailed description of the 3D object for an LLM to be able to reason about some of its spatial and semantic properties. Objects also have scene layout information giving the position, orientation, and bounding box of the object in the scene.

Given a 3D scene (such as via a GLTF file), we can easily convert the scene to our textual scene graph by extracting all references to 3D models and replacing them with their bounding box and the appropriate text descriptions produced by our shape-to-text model (Section 3.2). Conversely, given a scene graph containing only a textual description of objects, we can convert all textual object descriptions into 3D geometry using text-to-shape (Section 3.1).

Making a scene. Given a user-specified scene description, our text-to-scene application uses an LLM to generate scene descriptions in the format described above. Modern LLMs exhibit strong capabilities for reasoning about what objects are likely to be present given the input description, but they can struggle with more precise spatial reasoning needed to construct plausible scene layouts. To assist the LLM we generate a database of prompt and JSON scene graph pairs (created by processing scenes using our shape-to-text model described above) and provide exemplars from this database to the LLM via in-context learning(Dong et al., 2022). Once the LLM has generated a JSON scene graph, we generate scene geometry for each object

###### 3.3.2. Making a scene

Initial scene creation. Given a user-specified scene description, our text-to-scene application uses an LLM to generate scene descriptions in the format described above. Modern LLMs exhibit strong capabilities for reasoning about what objects are likely to be present given the input description, but they can struggle with more precise spatial reasoning needed to construct plausible scene layouts. To assist the LLM we generate a database of prompt and scene json

12

[Figure 80]

A garage and oﬃce scene.

[Figure 81]

A low poly palm island scene.

[Figure 82]

An anime Japanese pagoda scene.

- Figure 12 | Scene generation examples. The layout, the rotation, position and scale of each object are produced by the LLM system, while the geometry and texture are generated by our text-to-shape and an in-house text-to-texture model built upon FlashTex (Deng et al., 2024).

###### 3D scene suggestion assistant:

[Figure 83]

(We remove all the scene related meta data, i.e. scene name, object category name from the artists, and only use the shape caption generated by our shape-to-text model to test the 3D scene Q&A)

User Please summarize the scene. Assistant The scene features a modern dining setting with tables, chairs, and various objects like plates, bottles, coins, and decorative items. Pendant lights hang above, and a sleek, minimalist aesthetic dominates the space. There are also some irregularly shaped objects, possibly food items, enhancing the dining atmosphere.

User I’d like to add condiments on top of the counter, where should I place them? Assistant The scene includes modern rectangular tables and a counter-like surface (likely the trapezoidal prisms or modern tables). Consider these positions: 1.Near the center of the counter – Ensures easy access for diners. 2.Close to plates or food items – Keeps condiments within reach of those eating. 3.Evenly spaced along the counter – Prevents crowding in one area.

User I want to replace the stool near the counter, what other type of seat do you recommend? Assistant For a modern dining scene, here are some great alternatives to replace the stool near the counter:

1.High-Back Bar Chairs – Sleek and cushioned for a premium feel. 2.Bench Seating – A continuous, modern look that fits multiple players. 3.Swivel Chairs – Adds a dynamic, interactive element.

User What audio, background music would you recommend in this scene? Assistant For your modern dining scene, I recommend: 1. Lo-fi jazz with soft piano & muffled chatter for a chic,

relaxed vibe. 2. Light orchestral music with faint dish clinks for an elegant touch. 3. Acoustic guitar & café ambiance for a cozy feel.

Table 2 | Scene analysis and suggestion example. Given this diner scene, our scene suggestion assistant is able to summarize the scene, make suggestions for scene enhancements such as where to placement of condiments, what style of seating would be appropriate, and make background audio recommendations.

using our text-to-shape model. Figure 12 shows renderings of scenes generated using our text-to-scene generator. The generated scenes are diverse and the layout placement is physicallyplausible (i.e., the sofa, chair and the office desk are placed in the correct location and the cars in the garage are facing in the consistent direction). The objects in the scene have consistent styles in both the geometry and texture (e.g., the pagoda and the traditional Japanese front gate match in the scale and style). The scenes depicted in the figure were generated by the LLM from a starting scene description prompt, and then iteratively refined by the user through multi-turn conversations with our system. For example the user can issue new text instructions to the LLM, (“Add a tree to the scene”) resulting in a new scene description, or use a 3D scene editor tool to manually adjust the position and orientation of objects after the initial placement, which can be used to occasionally correct the LLM placement errors.

Scene analysis and suggestions. When creating a 3D scene, it can be challenging to consider what objects populate it with. We envision a human user working in tandem with our scene creation application to help them to quickly consider options for how to improve the scene. For example, Table 2 depicts a 3D scene of a 60’s-style diner. The text description of a scene provides sufficient information for the LLM to comprehend how the scene might be used, and to offer

suggestions on how to enhance it. For example, the system is able to suggest relevant places to put condiments on the counter top, recommend styles of seating that might be added, and even offer music suggestions.

#### 4. Conclusion

We have demonstrated the first step towards our goal of developing a foundational model that is capable of 3D generation and reasoning. Specifically, we have shown how our shape tokenization scheme enables a variety of generative 3D applications. Looking ahead, there are multiple challenges that must be addressed to achieve the complete vision of a unified 3D foundational model capable of assisting human developers with building all aspects of a Roblox experience. Here we outline some of the future steps towards this vision.

Mixed generation of meshes and parts. Our text-to-shape application generates a 3D geometric mesh from a text prompt. But the Roblox platform also supports modeling 3D shapes via constructive solid geometry (CSG) operations on basic parts (e.g. cylinder, cube, sphere, etc.). These parts-based objects provide a compact representation that can be rendered efficiently on a variety of edge devices. They also provide a distinct “blocky” visual style that is the hallmark of many Roblox experiences. We are exploring how to incorporate our shape tokens into an autoregressive GPT model that can generate parts-only CSG shapes or shapes that combine CSG and meshes.

Character avatar generation. Avatars are the primary representation of players in Roblox experiences. For many players, their first creation experience involves designing or customizing their avatar. Avatars move and animate as players interact with a scene, so a AI-based avatar generation must produce riggable geometry: detailed head meshes, separate meshes for animatable features such as eyes and mouth, a body mesh, and clothing meshes that are layered on the body.

4D behavior generation. Objects in a Roblox experience can be rigged for animation and/or programmatically scripted to behave in different ways depending on how a player interacts with them. For example, a 3D car model might be rigged so that turning the steering wheel rotates the wheels. A door might be scripted to slide open when a player is near it. We refer to such rigged and/or scripted motions as 4D behavior, and our goal is to build AI-based 4D behavior generators.

Our vision of a unified foundational model for 3D intelligence aims to enable users of all backgrounds to easily create complete 3D experiences. Generative AI tools that can assist users

- at each step of the creation process requires such a foundational model that can reason about 3D shape, object behaviors, player interactions and game logic. Our work so far has taken a step towards the first of these reasoning goals. With the release of our open source model3 we hope to engage others in the research community to address these goals with us.

3https://github.com/Roblox/cube

#### 5. Update Logs (July, 2025)

Since the initial release in March 2025, we have made some key updates to our open-source text-to-shape model, which we describe in more detail below.

###### 5.1. Data

We introduce an additional synthetic data source into the training set. The synthetic data pipeline takes advantage of the generalization capabilities of state-of-the-art text-to-image and imageto-shape models to generate paired (𝑡𝑒𝑥𝑡, 𝑠ℎ𝑎𝑝𝑒) data suitable for our training. For generating a diverse set of prompts to be fed into the pipeline, we start with a list of seed prompts that come from two sources: 1) list of objects found within LLM-generated fictional game concepts; 2) real-life objects (e.g., dishwasher, laptop, etc.). Then the seed prompts are fed into an LLM to create multiple, descriptive variations. In total, we collected about 3 million high-quality synthetic assets for training.

With the addition of synthetic data, we observed substantial improvement in prompt adherence for our text-to-shape model, including for challenging compositional prompts. Please refer to our Github page (https://github.com/Roblox/cube) for a gallery of updated results.

###### 5.2. VQ-VAE Shape Tokenization Improvements

To enhance the quality of our VQ-VAE shape tokenization, several key refinements were made to the architecture and training procedure (see sample comparisons with our previous model in Figure 13). The VQ-VAE latent length was increased from 512 to 1024. This expanded VQ-VAE is optimized using a two-stage training protocol along with other regularization and input data improvements detailed below.

###### 5.2.1. Two-stage training protocol

Our training process is divided into an initial pre-training stage with occupancy supervision, followed by a fine-tuning stage with Truncated Signed Distance Function (TSDF) supervision.

- Stage 1: Occupancy-based pre-training. The new VQ-VAE model is first trained using supervision from a binary occupancy field, similar to our March’25 model. To accelerate convergence, the 1024-d VQ-VAE model is initialized from the weights of our initial 512-d model. All layers are transferred directly, while the new, larger query layer is initialized with the same mean and standard deviation as the 512-d model.
- Stage 2: TSDF-based fine-tuning. Once the occupancy-trained model has converged, it is further fine-tuned using a continuous TSDF. Unlike binary occupancy values, the TSDF provides richer, continuous gradient signals about the surface, enabling the model to capture finer-grained details.

###### 5.2.2. Regularization and input data

In addition to the two-stage protocol, we made some more improvements to the training procedure:

• Eikonal loss: To enforce a valid SDF, Eikonal loss (Gropp et al., 2020) is applied as a regularizer, encouraging smoothness and continuity in the predicted SDF.

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Groundtruth

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Ours-VQ (Mar 2025)

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Ours-VQ (Jul 2025)

- Figure 13 | VQ-VAE reconstruction comparison between our previous model (Mar 2025) and our updated model (Jul 2025) with the improvements introduced in Section 5.2.

- • REPA regularization: Inspired by the Representation Alignment (REPA) work (Yu et al.,

2025) from the image domain that regularizes learned features with a given semantic representation (e.g. DINOv2 (Oquab et al., 2023)), we introduce an additional regularization term in the latent space. This loss minimizes the distance between the VQ quantized latents and the latents just before the final decoder layer of our VQ-VAE. We found that the REPA regularization improves the smoothness of the quantized latents, which is beneficial for the GPT training of the downstream shape generation.

- • Input point cloud density: To provide the encoder with a more detailed input signal, we increase the number of points sampled from the mesh surface from 8,192 to 32,768. This modification enables the shape tokens to capture more surface details without increasing the computation overhead of the subsequent generative model.

###### 5.3. 3D Bounding Box Conditioning

To enable more explicit control of the spatial extent of the generated shapes, we introduce a 3D bounding box as additional input conditioning into our text-to-shape model. The bounding box is represented as a 3D vector containing the unit normalized dimensions of the target shape’s axis-aligned bounding box. This vector is fed directly into an MLP, and the output embedding is appended as an additional token along with the text tokens to condition the shape generation.

One challenge that emerged during training is that since each shape is paired with a single deterministic bounding box but can be associated with multiple diverse text prompts, the model learned to prioritize the bounding-box condition, often disregarding the text prompt during inference. To address this, we apply a random perturbation to each dimension of the bounding box during training. This simple data augmentation technique prevents the model from overfitting to the exact dimensions of the bounding box and forces it to balance the influence of both spatial and textual conditional inputs. As a result, the final model remains

responsive to the text prompt while adhering to the specified spatial constraints. Please refer to our Github page (https://github.com/Roblox/cube) for illustrations of the 3D bounding box conditioning.

###### 5.4. Accelerated shape extraction

To accelerate the shape extraction, we make use of a hierarchical volume decoding technique that reduces the computational complexity of sampling the Signed Distance Function (SDF). Standard iso-surface extraction methods, such as Marching Cubes, necessitate evaluating an SDF, 𝑓 (x) : R3 → R, on a dense volumetric grid of resolution 𝑁3, incurring a computational cost of 𝑂(𝑁3). However, the target surface, defined by the zero-level set 𝑆 = {x ∈ R3 | 𝑓 (x) = 0}, implies that its representation within the volume is sparse. Our approach exploits this sparsity by first identifying occupied voxels on a coarse grid of resolution 𝑁𝑐3, where 𝑁𝑐 ≪ 𝑁. A voxel is classified as occupied if the SDF values at its vertices exhibit a sign change, indicating the presence of the iso-surface. Voxels determined to be entirely inside (𝑓 (x) < 0) or outside (𝑓 (x) > 0) the surface are pruned. Subsequently, only the subset of occupied coarse voxels is subdivided to the target resolution 𝑁 for dense SDF evaluation. This coarse-to-fine strategy constrains the expensive queries to a narrow band surrounding the surface, reducing the complexity from 𝑂(𝑁3) towards 𝑂(𝑁2) and thereby substantially decreasing the number of required cross-attention queries to accelerate inference time.

#### References

- J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

- Y. Bengio, N. Léonard, and A. Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.

- R. Bommasani, D. A. Hudson, E. Adeli, R. Altman, S. Arora, S. von Arx, M. S. Bernstein, J. Bohg, A. Bosselut, E. Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021.

M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin. Emerging properties in self-supervised vision transformers. In IEEE International Conference on Computer Vision (ICCV), 2021.

Z. Chen, W. Wang, Y. Cao, Y. Liu, Z. Gao, E. Cui, J. Zhu, S. Ye, H. Tian, Z. Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.

M. Deitke, D. Schwenk, J. Salvador, L. Weihs, O. Michel, E. VanderBilt, L. Schmidt, K. Ehsani, A. Kembhavi, and A. Farhadi. Objaverse: A universe of annotated 3d objects. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

K. Deng, T. Omernick, A. Weiss, D. Ramanan, J.-Y. Zhu, T. Zhou, and M. Agrawala. Flashtex: Fast relightable mesh texturing with lightcontrolnet. In European Conference on Computer Vision (ECCV), 2024.

Q. Dong, L. Li, D. Dai, C. Zheng, J. Ma, R. Li, H. Xia, J. Xu, Z. Wu, T. Liu, et al. A survey on in-context learning. arXiv preprint arXiv:2301.00234, 2022.

P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Müller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

C. Fifty, R. G. Junkins, D. Duan, A. Iger, J. W. Liu, E. Amid, S. Thrun, and C. Ré. Restructuring vector quantization with the rotation trick. arXiv preprint arXiv:2410.06424, 2024.

M. Garland and P. S. Heckbert. Surface simplification using quadric error metrics. In Proceedings of the 24th annual conference on Computer graphics and interactive techniques, 1997.

J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. Richemond, E. Buchatskaya, C. Doersch, B. Avila Pires, Z. Guo, M. Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to selfsupervised learning. In Advances in Neural Information Processing Systems, 2020.

A. Gropp, L. Yariv, N. Haim, M. Atzmon, and Y. Lipman. Implicit geometric regularization for learning shapes. In International Conference on Machine Learning (ICML), 2020.

- S. Haykin. Communication systems. John Wiley & Sons, 2008. J. Ho and T. Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

M. Huh, B. Cheung, P. Agrawal, and P. Isola. Straightening out the straight-through estimator: Overcoming optimization challenges in vector quantized networks. In International Conference on Machine Learning (ICML), 2023.

- A. Jaegle, F. Gimeno, A. Brock, O. Vinyals, A. Zisserman, and J. Carreira. Perceiver: General perception with iterative attention. In International Conference on Machine Learning (ICML), 2021.

D. P. Kingma, M. Welling, et al. Auto-encoding variational bayes, 2013. W. Li, J. Liu, H. Yan, R. Chen, Y. Liang, X. Chen, P. Tan, and X. Long. Craftsman3d: High-fidelity

mesh generation with 3d native generation and interactive geometry refiner. arXiv preprint arXiv:2405.14979, 2024.

- Y. Li, Z.-X. Zou, Z. Liu, D. Wang, Y. Liang, Z. Yu, X. Liu, Y.-C. Guo, D. Liang, W. Ouyang, et al. Triposg: High-fidelity 3d shape synthesis using large-scale rectified flow models. arXiv preprint arXiv:2502.06608, 2025.

H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

W. E. Lorensen and H. E. Cline. Marching cubes: A high resolution 3d surface construction algorithm. In Seminal graphics: pioneering efforts that shaped the field, pages 347–353. 1998.

- B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In European Conference on Computer Vision (ECCV), 2020.

M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza,

- F. Massa, A. El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021.

S. Stojanov, A. Thai, and J. M. Rehg. Using shape to categorize: Low-shot learning with an explicit shape bias. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

P. Sun, Y. Jiang, S. Chen, S. Zhang, B. Peng, P. Luo, and Z. Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

- Y. Takida, T. Shibuya, W. Liao, C.-H. Lai, J. Ohmura, T. Uesaka, N. Murata, S. Takahashi, T. Kumakura, and Y. Mitsufuji. Sq-vae: Variational bayes on discrete representation with self-annealed stochastic quantization. arXiv preprint arXiv:2205.07547, 2022.

C. Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

- G. Team, P. Georgiev, V. I. Lei, R. Burnell, L. Bai, A. Gulati, G. Tanzer, D. Vincent, Z. Pan, S. Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

- A. van den Oord, O. Vinyals, and k. kavukcuoglu. Neural discrete representation learning. In Advances in Neural Information Processing Systems, 2017.

- A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

J. Xiang, Z. Lv, S. Xu, Y. Deng, R. Wang, B. Zhang, D. Chen, X. Tong, and J. Yang. Structured 3d latents for scalable and versatile 3d generation. arXiv preprint arXiv:2412.01506, 2024.

S. Yu, S. Kwak, H. Jang, J. Jeong, J. Huang, J. Shin, and S. Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In International Conference on Learning Representations (ICLR), 2025.

- B. Zhang, J. Tang, M. Niessner, and P. Wonka. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. In ACM Transactions on Graphics (TOG), 2023a.

- B. Zhang, W. Zheng, J. Zhou, and J. Lu. Preventing local pitfalls in vector quantization via optimal transport. arXiv preprint arXiv:2412.15195, 2024.

J. Zhang, F. Zhan, C. Theobalt, and S. Lu. Regularized vector quantization for tokenized image synthesis. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023b.

- Z. Zhao, W. Liu, X. Chen, X. Zeng, R. Wang, P. Cheng, B. Fu, T. Chen, G. Yu, and S. Gao. Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

- Z. Zhao, Z. Lai, Q. Lin, Y. Zhao, H. Liu, S. Yang, Y. Feng, M. Yang, S. Zhang, X. Yang, et al. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202, 2025.

J. Zhou, C. Wei, H. Wang, W. Shen, C. Xie, A. Yuille, and T. Kong. ibot: Image bert pre-training with online tokenizer. arXiv preprint arXiv:2111.07832, 2021.

#### Appendix A. Contributions and Acknowledgments

Team Leads Kiran Bhat (Overall) Nishchaie Khanna (Infrastructure) Karun Channa (Product) Tinghui Zhou (Modeling)

Contributors Alexander Weiss Christian Reiser Ravi Kiran Chirravuri Ravali Kandur Alejandro Pelaez Akash Garg Michael Palleschi Jessica Wang Skylar Litz Leon Liu Anying Li David Harmon Derek Liu Liangjun Feng Denis Goupil Lukas Kuczynski Jihyun Yoon Naveen Marri Peiye Zhuang Yinan Zhang Brian Yin Haomiao Jiang Marcel van Workum Thomas Lane Bryce Erickson Salil Pathare Kyle Price Steve Han Yiqing Wang

Core Contributors Yiheng Zhu Xiaoxia Sun Charles Shang Anirudh Sudarshan Maurice Chu Daiqing Li Kangle Deng Jean-Philippe Fauconnier Tijmen Verhulsdonck

Research Advisors Maneesh Agrawala Kayvon Fatahalian

Executive Advisors Anupam Singh David Baszucki

