## GPS as a Control Signal for Image Generation

Chao Feng1 Ziyang Chen1 Aleksander Hoły´nski2 Alexei A. Efros2 Andrew Owens1 1University of Michigan 2UC Berkeley

https://cfeng16.github.io/gps-gen/

# arXiv:2501.12390v2[cs.CV]22Jan2025

|[Figure 1]<br><br>[Figure 2]|
|---|

|GPS-to-3D ”statue of liberty”<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]|
|---|

|Average<br><br>“building”<br><br>[Figure 7]|
|---|

|“superman”<br><br>[Figure 8]|
|---|

|“street view in watercolor painting style”<br><br>[Figure 9]|
|---|

|“street view in oil painting style”<br><br>[Figure 10]|
|---|

|“selfie”<br><br>[Figure 11]|
|---|

|“street view in acrylic painting style”<br><br>[Figure 12]|
|---|

|“tourist bus”<br><br>[Figure 13]|
|---|

|[Figure 14]<br><br>[Figure 15]|
|---|

|[Figure 16]<br><br>“superman”|
|---|

|“boat”<br><br>[Figure 17]|
|---|

|[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>|
|---|

|“tourists”<br><br>[Figure 46]|
|---|

|“selfie”<br><br>[Figure 47]|
|---|

|“superman”<br><br>[Figure 48]|
|---|

|“autumn”<br><br>[Figure 49]|
|---|

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

|[Figure 55]<br><br>[Figure 56]|
|---|

|“selfie”<br><br>[Figure 57]|
|---|

|“aerial view”<br><br>[Figure 58]|
|---|

|[Figure 59]<br><br>“bagel”|
|---|

|“sunshine”<br><br>[Figure 60]|
|---|

|“spring”<br><br>[Figure 61]|
|---|

|[Figure 62]<br><br>“aerial view”|
|---|

|“snowing”<br><br>[Figure 63]|
|---|

|[Figure 64]<br><br>“bagel”|
|---|

Figure 1. What can we do with a GPS-conditioned image generation model? We train GPS-to-image models and use them for tasks that require a fine-grained understanding of how images vary within a city. For example, a model trained on densely sampled geotagged photos from Manhattan can generate images that match a neighborhood’s general appearance and capture key landmarks like museums and parks. We show images sampled from a variety of GPS locations and text prompts. For example, an image with the text prompt “bagel” results in a modern-style sculpture when conditioned on the Museum of Modern Art and an impressionist-style painting when conditioned on the Metropolitan Museum of Art. We also “lift” a 3D NeRF of the Statue of Liberty from a landmark-specific 2D GPS-to-image model using score distillation sampling. Please see the project webpage and Sec. A.1.1 for more examples.

### Abstract

### 1. Introduction

Each time a tourist snaps a photo, they capture a tiny sliver of the world. Research on geotagged photo collections has shown that these images, when analyzed collectively, can reveal surprising amounts of information, including the landmarks that people visit [17], the 3D structure of the buildings they see [78], and geographic variations in architecture and fashion [20, 53].

We show that the GPS tags contained in photo metadata provide a useful control signal for image generation. We train GPS-to-image models and use them for tasks that require a fine-grained understanding of how images vary within a city. In particular, we train a diffusion model to generate images conditioned on both GPS and text. The learned model generates images that capture the distinctive appearance of different neighborhoods, parks, and landmarks. We also extract 3D models from 2D GPS-to-image models through score distillation sampling, using GPS conditioning to constrain the appearance of the reconstruction from each viewpoint. Our evaluations suggest that our GPS-conditioned models successfully learn to generate images that vary based on location, and that GPS conditioning improves estimated 3D structure.

In this paper, we show that GPS conditioning is a useful and abundantly available control signal for image generation, which complements other common forms of conditioning like text. We train diffusion models to map GPS coordinates from a particular city (or from a more spatially localized region) to images. To solve this problem, the model needs to capture fine-grained distinctions in how images change their appearance over space. Such a model, for example, needs to know the locations of museums and parks,

the subtle differences between building facades in different neighborhoods, and how landmarks change their appearance from different perspectives. Consequently, these models convey information that would be difficult to obtain from image or language supervision alone.

We demonstrate the utility of this location-based control signal in a variety of ways. First, we train diffusion models on both GPS coordinates and text (obtained from captioning), allowing us to generate images that appear as though they were shot in a given location while capturing a particular text prompt to allow for additional control (Fig. 1). The resulting model exhibits the ability to perform compositional generation and the ability to closely follow location conditioning. For example, the prompt “aerial view” produces a plausible overhead image of Central Park’s Bethesda Fountain. The prompt “superman” results in a statue or a painting when conditioned on locations within the New York Museum of Modern Art, while it generates a photo of a costumed human in Times Square.

Second, we show that 3D geometry can be lifted from 2D GPS-to-image models (Fig. 1, Statue of Liberty). We exploit the fact that GPS conditioning tells us how a landmark should appear from different viewing positions. Given a GPS-to-photo model trained on a specific landmark, we extract a NeRF using score distillation sampling [64, 91], using the learned conditional distribution to ensure that the estimated NeRF is consistent with the visual appearance of photos from every viewing direction. This “3D reconstruction by 3D generation” approach does not require explicit camera pose estimation, matching, or triangulation. Instead, it obtains its signal from cross-modal association between GPS and images.

Our evaluations suggest that GPS conditioning provides a useful control signal for generating images and extracting structure from geotagged image collections. These experiments suggest:

- • GPS-conditioned image generation models can capture subtle variations between locations within a city.
- • GPS conditioning complements language-based conditioning for image generation and 3D generation.
- • 3D reconstructions can be extracted from 2D GPS-toimage models without explicit camera pose estimation.

### 2. Related Work

Exploring geotagged image collections. GPS tags associated with images have been used in many different ways. Some researchers use GPS coordinates as complementary signals for image classification [85] and remote sensing [16]. Some works predict geolocation from images [27, 29, 83, 94, 102] or retrieve GPS or address from images in CLIP style [90, 96]. Other work has applied GPS data for different applications, such as city mapping and landmark identification [17], architectural styles [20], scene

chronology [57], and fashion trends [53]. Mall et al. [54] creates “underground map” of cities by analyzing fashion styles in public social media photos to reveal unique neighborhood information. Snavely et al. [78] used geo-tagged, unordered photos to reconstruct 3D models of tourist sites and formulated it as a photo tourism problem. Shrivastava et al. [76] introduced the painting2GPS task, which estimates the GPS coordinates of a painting by matching it to a collection of real geo-tagged photos. In contrast to previous works, we leverage GPS tags as additional conditional signals in generative models, enabling GPS-guided generation of tourist images and providing free supervisory signals for 3D reconstruction.

Conditioning diffusion models. Diffusion models [9, 19, 34, 63, 66, 67, 69, 79–82] are designed to learn how to restore data that has been deliberately corrupted by adding Gaussian noise. Specifically, the forward process gradually adds noise to data over several time steps, transforming it into pure noise. The reverse process then learns to denoise the data step by step, reconstructing the original data from the noise. Prior work has used many various conditions to guide diffusion models for image/video/3D/4D synthesis, including text [3, 9, 10, 13, 45, 64, 66, 67, 69, 86, 87], depth [11, 101], audio [8, 26, 86, 87], camera poses [4, 41, 42, 48, 71, 74], motion [5, 23, 43, 75, 93, 100], tactile signals [98, 99], segmentation mask [2, 6, 101], and many other conditions. Siglidis et al. [77] utilizes conditional diffusion models as data mining tools. Recently, Deng et al. [18] uses street maps, height maps, and text as conditions for video generative models to synthesize streetscapes. Khanna et al. [39] generate satellite images from GPS coordinates but are limited to satellite imagery and require specialized, calibrated training data.l In contrast, our work learns from “in the wild” geotagged photos using the GPS tags taken from EXIF metadata, a much more diverse and abundantly available data source, and we use our models for a variety of downstream tasks that were not considered in prior work, such as 3D model extraction.

3D reconstruction and generation. Reconstructing 3D models from multiple images [28] is a longstanding problem. The traditional pipeline involves matching and verifying features [52, 60], estimating camera pose and sparse 3D geometry with structure from motion (SfM) [73, 88, 95], and generating dense reconstructions using multi-view stereo [22, 38] or neural fields [58]. State-of-the-art methods for unordered photo collections use specialized matching, filtering, and bundle adjustment to handle all-pairs matching and dense structure estimation [1, 51, 60, 73, 78]. While these approaches have been successful, they remain brittle, as each step can introduce unrecoverable errors. Recent works have generated 3D models “zero shot” solely from models trained solely on 2D images [12, 37, 64, 70, 91]. Poole et al. [64] and Wang et al. [91] used the score

function of a text-to-image generation model, an approach that they called score distillation sampling [64] or score Jacobian chaining [91]. Other work extends this approach with 3D synthetic data for fine-tuning [48, 74] and improved optimization [45, 92]. However, these models still face issues like the Janus problem due to difficulties in assigning pose [64]. We extend this framework to generate NeRFs that are assigned probability under a GPS-to-image model: we seek a NeRF for which every viewpoint has a high probability under the conditional distribution, avoiding the need for explicit camera pose estimation or feature matching.

Compositional generation. A notable characteristic of diffusion models is the ease with which they allow concept composition through the simple addition of noise estimates. This can be interpreted by treating these noise estimates as gradients of a conditional data distribution [81, 82], where their sum points in a direction that jointly enhances multiple conditional likelihoods. This technique has been used to enable compositions of text prompts both globally [47] and spatially [6, 21], of various image transformations [24] and individual image components [25]. It has also been used for diffusion models originating from two distinct modalities (sight and sound) [15]. Another line of work like ControlNet [101] shows the composition of multiple conditions (e.g., text prompt and pose&depth). In this paper, we demonstrate that: 1) two conditions of GPS tags and text prompts can successfully generate images using a single noise estimate; 2) our GPS-to-image diffusion models can obtain images representative of a given concept over a large geographic area by averaging noise estimates.

### 3. Method

We propose GPS-to-image diffusion model to synthesize a tourist image by conditioning on GPS coordinates and text. We then show that a variation of the model can be trained on a single tourist site and can be used to “lift” 3D models.

#### 3.1. GPS-to-image diffusion

We train a model to generate images conditioned on GPS coordinates from a given city, a challenging case that requires capturing fine-grained distinctions between the appearance of different locations. This model is further conditioned on text prompts to improve control of the model.

Preliminaries. Diffusion models [19, 34, 79–82] iteratively denoise the Gaussian noise xT to generate the image x0 of a distribution, which is a reverse of forward process. In the forward process, a clean image x0 is gradually transitioned into random noise xT by adding Gaussian noise. At each time step, the noisy latent zt+1 can be expressed:

zt+1 = αtzt + βtϵt,

where zt is noisy latent of previous timestep and ϵt is a standard Gaussian noise. αt and βt are predefined coeffi-

cients, so zt+1 is also the function of x0. In DDPM [34], the training objective of diffusion models is simplified to:

0,ϵt ω (t)||ϵt − ϵϕ (zt,y,t)||22 , (1) where ω (t) is a weighting function of timestep t (usually set to 1), ϵϕ is the denoiser, and y is the condition such as text prompt. For inference, DDIM [80] and classifier-free guidance (CFG) are usually employed:

L(ϕ) = Et,x

ϵˆϕ (zt;y,t) = (1 + ω)ϵϕ (zt;y,t) − ωϵϕ (zt;∅,t), (2)

where ω is the guidance weight and ϵˆϕ (zt;y,t) is the predicted noise for denoising.

Training a GPS-conditioned diffusion. Given a collection of geotagged tourist photos over a certain area (like a city), we want to learn a diffusion model [67, 79] that can synthesize tourist images controllably, when conditioned on the photo’s GPS position. For a randomly taken tourist image x, we use the GPS coordinates (x,y) to represent its position, where x and y are the longitude and latitude respectively. We use the (x,y) as an extra condition for diffusion models to make them aware of position geospatially. For instance, tourist photos taken at the Louvre Museum usually contain the Louvre Pyramid but not the Arc de Triomphe. In this way, we can endow models with the capability to provide reliable tour guides for walking through Paris by controllably synthesizing tourist photos.

We build our model on top of a pretrained text-to-image latent diffusion model [67]. Since our off-the-shelf model accepts a text prompt, we provide the text caption generated by BLIP-3 [97] on our collected datasets, encoded as a CLIP [65] text embedding p ∈ RL×D, where L is the number of text tokens and D is the token feature dimension. We then learn a GPS-pose embedding g = [f(x),f(y)] ∈ R2×D, and append it to text tokens p to establish a “GPS” CLIP text condition, as shown in Fig. 2. This input representation ensures that the model starts from an initialization that closely resembles what it was trained on.

We finetune the model using the text embedding p and GPS embedding g. Specifically, given a tourist photo dataset X, for training samples {x,p,g}, we optimize the diffusion loss:

t,t ∥ ϵt − ϵϕ(zt;p,g,t) ∥22 , (3) where zt is the noisy latent of image x at timestep t.

Lrecon = Ex,g,ϵ

Inference. During inference, we use the classifier-free guidance strategy from InstructPix2Pix [10] for two conditions (text prompt and GPS tag). Our score estimate is as follows:

ϵ˜ϕ (zt;p,g,t) =ϵϕ (zt;∅,∅,t)

+ ωp (ϵϕ (zt;p,∅,t) − ϵϕ (zt;∅,∅,t)) + ωg (ϵϕ (zt;p,g,t) − ϵϕ (zt;p,∅,t)),

(4)

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

EXIF Metadata

40°46'55.9"N 73°57'57.2"W

[Figure 69]

[Figure 70]

Angle Diffusion

##### +

[Figure 71]

“A skating rink in a park with tall buildings in the background.”

Captioning Model

[Angle] + “Statue of Liberty”

Addingnoise

Rendering

diffuse

SDS Loss

Pose-to-Angle

NeRF

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

GPS Diffusion

[Figure 77]

Frozen Trainable

[Figure 78]

Noisy image

+ EmbConcat.

Reconstruction Loss

(a) GPS-to-image generation (b) GPS-to-3D reconstruction

- Figure 2. Method. (a) After downloading geotagged photos, we train a GPS-to-image generation model conditioned on GPS tags and text prompts. The trained generative model can produce images using both conditioning signals in a compositional manner. (b) We can also extract 3D models from a landmark-specific GPS-to-image model using score distillation sampling. This diffusion model parameterizes the GPS location by the azimuth with respect to a given landmark’s center. + means we concatenate GPS embeddings and text embeddings. where ωp and ωg are gudiance weights.

instead of (x,y) as an extra condition for diffusion models, which means that we would replace g in Eq. (3) with g′ = f′(α) ∈ R1×D. This representation makes it straightforward to combine the approach with DreamFusion [64], which can be easily extended to accept angular conditioning. Additionally, we fix the text prompt to “A photo of {landmark name}” for each landmark. We refer to the diffusion model conditioned on GPS coordinates as the GPS-to-image diffusion, and the model conditioned on the angle from GPS tags as the angle-to-image diffusion.

#### 3.2. GPS-guided 3D reconstruction

Recent work has shown that 3D models can be extracted from 2D text-to-image diffusion models. We build on this idea to obtain 3D reconstructions of specific locations using GPS-to-image models.

Preliminary. Prior work [45, 64, 74, 91, 92] leverages pretrained 2D text-to-image diffusion models like Imagen [69] to synthesize 3D contents by textual descriptions. During optimization, a Gaussian noise would be added to each NeRF rendering x = hθ (q), where q is the camera pose. Then noisy rendering would be fed into pretrained denoiser ϵϕ and score distillation sampling (SDS) loss provides gradient to guide NeRF [7, 58] training:

We found that when the focused area is small, by directly finetuning the diffusion model with only a few thousand self-collected tourist photos, the model adapts to the distribution very easily and sometimes loses the generative diversity of the original model. To avoid this issue, we follow [68] and use a prior preservation loss to maintain the prior knowledge and regularize the pose-conditioned training. More details are presented in Appendix A.2.1. We combine both losses and optimize the final objective:

∂x ∂θ

. (5)

∇θLSDS (ϕ, x = hθ (q)) ≈ Et,ϵ ω (t) (ˆϵϕ (zt; y, t) − ϵ)

The text prompt y is appended by view-dependent texts of “front view”, “side view”, or “back view” based on the randomly sampled camera poses q, which can benefit 3D generation results. However, in some cases, 2D text-to-image diffusion models struggle to control viewpoints accurately, causing a multi-face Janus issue [64].

L = Lrecon + λLpreservation, (6)

where λ is the weight to balance the reconstruction and preservation loss.

GPS-guided score distillation sampling. Using the pose information from GPS, our angle-to-image diffusion model generates photos of monuments from various viewpoints by conditioning on the GPS poses. Our angle conditioning is analogous to view-specific prompting in Poole et al. [64] while providing a better view prior, as shown in Fig. 3. We use score distillation sampling (SDS) to extract a 3D model of the landmark from our diffusion model [64] (Fig. 2(b)). We parameterize this 3D model as a NeRF hθ (·) with pa-

Angle-to-image diffusion. GPS signals provide useful information, e.g., viewpoint details, for 3D landmark reconstruction from tourism photos. Hence, we train a diffusion model to transform these implicit signals into a score function for supervision. For an image x taken from a landmark at GPS coordinate (x,y) ∈ R2, we parameterize the pose using the azimuth angle α, with respect to the center of the

landmark (xo,yo): α = arctan x−x

y−yo . Here we use α

o

GPS tags by querying from Flickr: 1) New York City (Manhattan, 501,592 photos); 2) Paris (315,306 photos). For the landmark reconstruction task, we gather 6 sets of landmark photos following a similar approach. The number of evaluated landmarks aligns with previous work in the field [56]. Please see Appendix A.3 for dataset details.

Longitude Latitude

Arc + ”back view”

Arc + (48.87, 2.26)

Arc + (48.67, 2.29)

[Figure 79]

[Figure 80]

[Figure 81]

Arc+”sideview”

Arc + (48.59, 2.48)

Arc + ”front view”

Arc + (48.77, 2.67)

GPS-to-image diffusion. We use a positional encoding with frequencies of 10 [58] and a two-layer MLP to encode the GPS conditions. For each city, we normalize (x,y) to the range [−1,1]. We finetune Stable Diffusion-v1.4 [67] on Flickr images, at a resolution of 512×512 for 15k steps. We use the AdamW [50] optimizer with a learning rate of 1 × 10−4. We use a batch size of 512 on 8 NVIDIA L40S GPUs. During training, we randomly drop text and GPS conditions, ensuring 5% text-only, 5% GPS-only, and 5% unconditional generation. We caption images using BLIP3 [97]. See Appendix A.1.5 for details.

(a) SfM (b) DreamFusion (c) Ours

- Figure 3. 3D Setup Comparison. We extract 3D models from 2D GPS-to-image models. (a) Traditional approaches require running SfM to estimate camera pose, followed by dense geometry estimation. Since they are based on triangulation, they are susceptible to catastrophic errors due to incorrect pose; (b) DreamFusion [64] samples images from different poses within a scene using viewdependent prompting. However, text has a limited ability to precisely control the position of the camera. (c) Our method extends DreamFusion with GPS conditioning, reducing pose uncertainty.

rameters θ. We optimize the parameters of a NeRF [58] using gradient descent, such that every rendered viewpoint has a high probability under the Angle-conditioned image model.

During each iteration of the optimization process, we sample random virtual camera poses. For each one, we transform the pose q to azimuth α, and create a conditioning embedding g′ (Fig. 2 (b)). We can then render a corresponding image x = hθ (q) from the NeRF.

We use the SDS loss to obtain a gradient from the angleto-image diffusion model to supervise NeRF [59]. Following recent approaches [36, 74], we gradually decrease both the maximum and minimum of the time step sampling interval for the SDS loss during the optimization process. We apply classifier-free guidance (CFG) [33] to the SDS loss. For the unconditional version of the model, we simultaneously zeroed out both text and GPS conditions. This results in a noise estimator:

ϵˆϕ (zt;p,g′,t) = (1+ω)ϵϕ (zt;p,g′,t)−ωϵϕ (zt;∅,∅,t),

(7) where zt is the noisy latent of image rendered by NeRF and ω is the CFG guidance weight. The equation of gradient is presented in Appendix A.2.2.

- 4. Experiments

Angle-to-image diffusion. We train individual Angle-todiffusion models for each landmark. We calculate the azimuth angle α from GPS using α = arctan x−x

y−yo and map it to the nearest 10° angle bin. We use the normalized bin value with positional encoding as conditional input. We fix text prompts with the template “A photo of {landmark name}”. The weight of preservation loss λ in Eq. (6) is set to 1.0. Please refer to Appendix A.2.1 for more details.

o

3D reconstruction. We apply different guidance weights of classifier-free guidance (CFG) in Eq. (7) for score distillation sampling for each landmark. We turn on shading after 1000 steps and use orientation and opacity regularization loss, following [64]. We use the Adam optimizer [40] with a learning rate of 0.01 for 10k training steps. The time step sampling interval is gradually reduced from [0.98,0.98] to

|[Figure 82]<br><br>“Car”|
|---|

|[Figure 83]<br><br>“Tourist”|
|---|

|[Figure 84]<br><br>“Vintage car”|
|---|

|[Figure 85]<br><br>“Selfie”|
|---|

|[Figure 86]<br><br>“Boat”|
|---|

|[Figure 87]<br><br>“Batman”|
|---|

|[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>|
|---|

|[Figure 103]<br><br>“Selfie”|
|---|

|[Figure 104]<br><br>“Cloudy”|
|---|

We evaluate our model using a variety of quantitative and qualitative metrics. We first evaluate our GPS-to-image diffusion model, measuring its ability to successfully generate images that convey GPS and semantics of text prompts. We then evaluate our model’s ability to obtain 3D reconstructions for landmarks guided by GPS.

|[Figure 105]<br><br>“Batman”|
|---|

|[Figure 106]<br><br>“Fountain”|
|---|

|[Figure 107]<br><br>“Afternoon tea”|
|---|

|[Figure 108]<br><br>“Ben Affleck”|
|---|

|[Figure 109]<br><br>“Computer Scientist”|
|---|

|[Figure 110]<br><br>“Street view in oil painting style”|
|---|

|[Figure 111]<br><br>“A cup of coffee”|
|---|

|[Figure 112]<br><br>“Seine”|
|---|

#### 4.1. Implementation details

Tourist photo collection. To train GPS-to-image diffusion models, we obtained two city photo collections with

Figure 4. Qualitative results for Paris. We show images that have been sampled from our GPS-to-image diffusion model for various locations and prompts within Paris.

“dog” 40°46'40.7"N 73°58'09.5"W

”selfie” 40°42'25.1"N 74°00'38.7"W

”snowing” 40°46'09.1"N 73°58'41.9"W

”street view” 40°45'32.9"N 73°58'36.4"W

”a cyclist” 48°52'17.8"N 2°17'43.5"E

”building” 48°50'50.0"N 2°20'13.6"E

”tourist boat” 48°51'15.8"N 2°20'35.7"E

”at night” 48°51'24.0"N 2°17'48.7"E

Address-to-image

|[Figure 113]|
|---|

|[Figure 114]|
|---|

|[Figure 115]|
|---|

|[Figure 116]|
|---|

|[Figure 117]|
|---|

|[Figure 118]|
|---|

|[Figure 119]|
|---|

|[Figure 120]|
|---|

Neighbor Text-

|[Figure 121]|
|---|

|[Figure 122]|
|---|

|[Figure 123]|
|---|

|[Figure 124]|
|---|

|[Figure 125]|
|---|

|[Figure 126]|
|---|

|[Figure 127]|
|---|

|[Figure 128]|
|---|

GPSNearest

|[Figure 129]|
|---|

|[Figure 130]|
|---|

|[Figure 131]|
|---|

|[Figure 132]|
|---|

|[Figure 133]|
|---|

|[Figure 134]|
|---|

|[Figure 135]|
|---|

|[Figure 136]|
|---|

GPS-to-image Google

Ours-

|[Figure 137]|
|---|

|[Figure 138]|
|---|

|[Figure 139]|
|---|

|[Figure 140]|
|---|

|[Figure 141]|
|---|

|[Figure 142]|
|---|

|[Figure 143]|
|---|

|[Figure 144]|
|---|

Streetview

- Figure 5. Qualitative comparison for GPS-to-image diffusion. We compare the qualitative results of our method against baselines using specific pairs of text prompts and GPS tags. Each column shows a text prompt and a GPS tag at the top. Text-address-to-image diffusion model is conditioned on a combination of the text prompt and the address name derived from the GPS tag. We also perform nearest neighbor in the training set based on GPS tags. Our GPS-to-image diffusion model uses a text prompt and raw GPS tag as conditioning. Google Street View images are sampled for reference of geolocation. Our method achieves better compositionality and visual quality.

Table 1. Evaluation of GPS-to-image diffusion. We compare our method with several baselines in terms of CLIP Score and GPS Score. NN represents the nearest neighbor and SD is for stable diffusion. The best results are in bold, and the second bests are colored in blue.

image and GPS embeddings. See Appendix A.1.4 for details.

Baselines. Since we are not aware of any prior work on GPS-to-image generation, we include two baselines: 1) Stable Diffusion (SD)v1.4 [67]; 2) GPS Nearest Neighbor. For SD, we consider two variations: the first accepts a concatenation of the text prompt and address name1, while the second is conditioned only on the text prompt.

Method CLIP Score (↑) GPS Score (↑) Avg (↑)

GPS NN 18.77 13.66 16.22 SD (Text&address) 26.65 4.25 15.45 SD (Text) 29.13 1.21 15.17

Results. As shown in Tab. 1 and Fig. 5, our method achieves the best performance in terms of the average CLIP score and GPS score. Additionally, it demonstrates better compositionality, indicating that our method can successfully generate images from text prompts and GPS tags. Our method, without the text prompt, achieves a better GPS score than the Nearest Neighbor, indicating our model better captures image distributions in a geospatial context. More qualitative results are presented in Fig. 1 and Fig. 4. For instance, our method can successfully generate images under different weather [44] and lighting [46] conditions given a certain GPS location.

Ours 27.88 8.15 18.02 Ours (w/o text) – 13.71 –

[0.02,0.50]. To match the characteristics of tourism photo distribution, we restrict the elevation angle of sampled virtual camera views to below 0, while the azimuth angle is sampled across the full range.

#### 4.2. Evaluation of GPS-to-image generation

We first evaluate our GPS-to-image diffusion models in generating photos conditioned on GPS tags.

Evaluation metrics. To evaluate our model and baselines, we create 1,292 random GPS-text pairs as conditions for generation. We report CLIP score (CS) [65] to measure the alignment between generated image and text prompts. Analogously, we train a GPS-CLIP model on paired GPSimage data with contrastive loss [14, 30, 61, 65] and report GPS score (GS) which measures cosine similarity between

#### 4.3. Average images

Inspired by work that computes average images [20, 89], we apply our GPS-to-image models to the problem of obtaining images that are representative of a given concept over

1The address name is obtained through geocoding using GeoPy.

Champs-Élysées 7th Arr. 1st Arr. Rue de la Croix Nivert Luxembourg Garden

Manhattan Midtown Hudson Street Barclay Street SOHO Central Park

|[Figure 145]|
|---|

|[Figure 146]|
|---|

|[Figure 147]|
|---|

|[Figure 148]|
|---|

|[Figure 149]|
|---|

|[Figure 150]|
|---|

|[Figure 151]|
|---|

|[Figure 152]|
|---|

|[Figure 153]|
|---|

|[Figure 154]|
|---|

AverageGooglemap

|[Figure 155]|
|---|

|[Figure 156]|
|---|

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

(a) Paris (b) New York City

- Figure 6. Average images. We select five areas for Paris and New York City respectively. Using our GPS-to-image models, we obtain representative images of the concept of “building” within these geographic regions to observe architectural styles. More examples can be found on project webpage for different locations and concepts.

a large geographic area. Specifically, we generate a single image that has high probability under all GPS locations within a user-specified area, as measured by our diffusion model. To do this, we follow work on compositional generation [47] and simultaneously estimate noise using a large number of GPS locations and average together the noise vectors during each step of the reverse diffusion process (see Appendix A.1.3 for details). In Fig. 6, we show images generated for the text prompt “building” over a variety of streets in neighborhoods in Paris and New York. The resulting images capture the distinctive architectural styles of buildings in the specified areas. More examples can be found on the project webpage, showcasing different locations and concepts.

#### 4.4. Evaluation of angle-to-image generation

We also evaluate how well our angle-to-image diffusion models can generate photos of monuments conditioned on desired viewpoints (angles) derived from GPS tags.

Evaluation metric. We train a classifier on each landmark dataset individually to predict the discretized angle bins derived from GPS tags. For each angle bin of 10°, we ask generative models to synthesize 10 images and pair them up with input angle bins as ground truth. Then we apply the trained angle classifier to evaluate these images using accuracy as the metric. The evaluation method for diffusion models we adopt is similar to CLIP score [65]. We use this classifier trained on our training dataset to testify whether the finetuned diffusion model has successfully fit the training distribution. It should be noted that the main goal of finetuning a diffusion model is to facilitate the end goal of 3D landmark reconstruction—the reconstruction quality of the reconstructed 3D landmarks should serve as an evaluation that our model has successfully learned the data distribution and angle conditioning signal from GPS.

Results. We compare our method with two baselines: 1) Stable Diffusion (SD) v1.4 [67]; 2) Random chance. The results are reported in Tab. 2. Our angle-to-image diffusion model significantly outperforms both random chance

- Table 2. Evaluation of angle-to-image diffusion. We evaluate the accuracy of our model in generating images with the correct azimuth, as determined by an image-to-azimuth model.

Method Angle acc (%)

Random chance 2.78 Stable Diffusion [67] 3.06 Ours 22.36

- Table 3. Quantitative comparison for 3D. We report results: CLIP Score (CS) [65], Perceptual Quality (PC), and Tourist Score (TS). It shows that our method achieves the highest quality.

Method CS [65] (↑) PQ (↑) TS (↑)

NeRF [84] 20.57 1.32 1.36 Dreamfusion [64] 29.49 2.21 2.09 Ours 31.87 3.31 3.45

and text-to-image Stable Diffusion [67] by a relatively large margin, demonstrating that it effectively learns viewpoint signals from image-GPS pairs. Some generated images from the model are provided in Appendix A.2.1.

- 4.5. Evaluation of 3D landmark reconstruction

We evaluate how good our reconstructed landmarks are based on our angle-to-image diffusion model.

Evaluation metrics. The generated 3D landmarks are evaluated both qualitatively and quantitatively. We calculate CLIP Score (CS) [65] on RGB renderings of the landmarks with their corresponding text prompts of their names. The final CLIP score is presented as an average calculated from 30 randomly selected views. Following prior work [35], we also conducted a user study for evaluation. We asked 36 participants to score the Perceptual Quality (PQ) and Tourist Score (TS) of landmark reconstructions on a scale of 1 to 5, where 5 is the best. We define the perceptual quality metric to evaluate the quality of generated 3D assets, which do not necessarily match ground truth. The tourist score evaluates the 3D landmark reconstruction compared with real tourist photos by human preference. We evaluate 6 landmarks as mentioned in Sec. 4.1.

RGB Surface normal

RGB Surface normal RGB Surface normal

RGB Surface normal

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

|[Figure 171]|
|---|

|[Figure 172]|
|---|

DreamFusionOurs

|[Figure 173]|
|---|

|[Figure 174]|
|---|

|[Figure 175]|
|---|

|[Figure 176]|
|---|

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

|[Figure 185]|
|---|

|[Figure 186]|
|---|

|[Figure 187]|
|---|

|[Figure 188]|
|---|

DreamFusionOurs

|[Figure 189]|
|---|

|[Figure 190]|
|---|

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

- Figure 7. Qualitative comparison for 3D monument reconstruction. We show qualitative results of DreamFusion [64] and our method on t Tower Our reconstructed 3D monuments have better visual quality and more accurate re. rendered e of RGB rendering white. Please see Appendix A.2.2 and project web

|[Figure 197]<br><br>1) Leaning<br><br>We use examples.|
|---|

|[Figure 198]<br><br>two monuments:<br><br>3D structure. webpage for more|
|---|

|[Figure 199]<br><br>of Pisa; 2) Arc depth to make|
|---|

|[Figure 200]<br><br>de Triomphe. the background|
|---|

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

DreamFusionOurs

Baselines. We compare our method to DreamFusion [64] and 7 by F ter wild Ner r sion we Di c with our model. We consider two text

30

GPS tag Text

|[Figure 205]<br><br>COLMAP [ step, we train<br><br>Nerfacto [84] for use Stable|
|---|

|[Figure 206]<br><br>72] followed Nerf in the each scene. Diffusion [67] to|
|---|

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

|[Figure 211]<br><br>NeRF [58]. (NeRF-W)<br><br>For DreamFusion ensure a fair types of|
|---|

|[Figure 212]<br><br>For the lat[55] 2 and<br><br>[64], comparison<br><br>prompts:|
|---|

| |
|---|

20

20

Full W/o PP W/o PP and GPS

10

10

| |
|---|

| |
|---|

0

0

CS GS

CS

(a) Representation of geolocation (b) 3D reconstruction

1) “A photo of {landmark name}”; 2) “{landmark name}”, both of them are with view-dependent conditioning (text prompt is appended with “front/back/side view”). We pick the best one for evaluation.

Figure 8. Ablation. We conducted ablation studies to analyze the effectiveness of different modules in our method for GPS-to-image generation and 3D landmark reconstruction.

Results. As shown in Tab. 3, our method outperforms two baselines in terms of three evaluation metrics. Our qualitative results in Fig. 7 show that renderings from our models have better visual quality and more accurate 3D structure than those from DreamFusion [64]. This suggests that our GPS-conditioned diffusion model can provide a better pose prior than the text-to-image diffusion [67] with view-related prompts. As expected, we found the SfMbased baseline to be “all or nothing”, either providing very high-quality reconstructions or catastrophically failing (as shown in Appendix A.2.3). For example, COLMAP [72] successfully reconstructs camera poses and sparse point clouds for 3 of the 6 scenes, and fails on three. NeRFW [55] estimation completely fails on 6 landmarks and Nerfacto [84] fails on 5. This also reveals one shortcoming:

if COLMAP [72] cannot reconstruct the poses of the input images, NeRF [58] optimization is not possible. In contrast, our method addresses this and is able to reconstruct scenes that COLMAP [72] cannot reconstruct.

#### 4.6. Ablation Study

Attention map visualization. We visualize attention maps [32] for the text and GPS conditions to examine what signals the model focuses on. We show two examples in Fig. 9. We can see that the text prompt effectively controls the semantics of objects in the synthesized image, while the GPS tag (latitude and longitude) significantly influences the background. For instance, in the “A tourist” example, we can observe the shape of the Oculus Center in the attention maps corresponding to the GPS tag.

Representation of geolocation. Geolocation can be represented in two variations: 1) continuous GPS tag; 2) ad-

2We use this popular reimplementation since NeRF-W is not publicly available.

“A yellow cab” + 40°44'14.3"N 73°59'48.8"W

tion that is difficult to fully disentangle.

|[Figure 213]|
|---|

|[Figure 214]|
|---|

|[Figure 215]|
|---|

|[Figure 216]|
|---|

Acknowledgements. We thank David Fouhey, David Crandall, Ayush Shrivastava, Chenhao Zheng, Daniel Geng, and Jeongsoo Park for the helpful discussions. We thank Yiming Dou for helping set up NeRF baselines. Chao especially thanks Xinyu Zhang for her help in this project. This work was supported in part by Cisco Systems, NSF CAREER #2339071, and DARPA Contract No. HR001120C0123.

Generation

Cab Latitude Longitude

“A tourist” + 40°42'42.5"N 74°00'43.8"W

|[Figure 217]|
|---|

|[Figure 218]|
|---|

|[Figure 219]|
|---|

|[Figure 220]|
|---|

### References

Tourist Latitude Longitude

Generation

Figure 9. Attention visualization. We visualize attention maps for text and GPS tokens.

- [1] Sameer Agarwal, Yasutaka Furukawa, Noah Snavely, Ian Simon, Brian Curless, Steven M Seitz, and Richard Szeliski. Building rome in a day. Communications of the ACM, 54(10):105–112, 2011. 2
- [2] Sherwin Bahmani, Jeong Joon Park, Despoina Paschalidou, Xingguang Yan, Gordon Wetzstein, Leonidas Guibas, and Andrea Tagliasacchi. Cc3d: Layout-conditioned generation of compositional 3d scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7171–7181, 2023. 2
- [3] Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7996–8006, 2024. 2
- [4] Sherwin Bahmani, Ivan Skorokhodov, Aliaksandr Siarohin, Willi Menapace, Guocheng Qian, Michael Vasilkovsky, Hsin-Ying Lee, Chaoyang Wang, Jiaxu Zou, Andrea Tagliasacchi, et al. Vd3d: Taming large video diffusion transformers for 3d camera control. arXiv preprint arXiv:2407.12781, 2024. 2
- [5] Sherwin Bahmani, Xian Liu, Wang Yifan, Ivan Skorokhodov, Victor Rong, Ziwei Liu, Xihui Liu, Jeong Joon Park, Sergey Tulyakov, Gordon Wetzstein, et al. Tc4d: Trajectory-conditioned text-to-4d generation. In European Conference on Computer Vision, pages 53–72. Springer,

2025. 2

- [6] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. 2023. 2, 3
- [7] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. CVPR, 2022. 4
- [8] Burak Can Biner, Farrin Marouf Sofian, Umur Berkay Karaka¸s, Duygu Ceylan, Erkut Erdem, and Aykut Erdem. Sonicdiffusion: Audio-driven image generation and editing with pretrained diffusion models. arXiv preprint arXiv:2405.00878, 2024. 2
- [9] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2

dress name in text geodecoded from GPS tag. We finetune stable diffusion [67] on these two variations and results are presented in Fig. 8. As shown in Fig. 8, though CLIP Scores are comparable, our method based on continuous GPS tag outperforms the text-based method by a significant margin for GPS Score. This suggests that using a continuous GPS tag as a conditioning input better controls the geospatial aspects of image generation.

3D reconstruction. We conduct experiments to evaluate the importance of prior preservation loss and GPS conditioning for 3D landmark reconstruction. We train our angleto-image diffusion models without prior preservation loss and also perform experiments where we remove angle conditioning during training (Fig. 8). Our method outperforms these baselines by a large margin, suggesting that GPS is a valuable conditioning signal for reconstruction.

### 5. Conclusion

Our work suggests that GPS coordinates are a useful signal for controllable image generation. We have proposed a method to generate images conditioned on GPS tag and text prompt in a compositional manner, which successfully learns the cross-modal association between GPS tags and images. It can achieve compositional generation for tasks that require a fine-grained understanding of how images vary within a city. We also find that GPS conditioning enables us to reconstruct 3D landmarks by score distillation sampling without explicit camera pose estimation. Our work opens two future directions. The first is to develop models that use GPS-to-image generation methods to analyze geotagged photo collections in additional ways. The second is to develop new generative models that can extract more information from GPS conditioning.

Limitations. Our approach may not be well-suited for cases where few photos have GPS available. Score distillation sampling is known to produce saturated images. In some scenarios, GPS tags carry certain semantic informa-

- [10] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392– 18402, 2023. 2, 3
- [11] Shengqu Cai, Duygu Ceylan, Matheus Gadelha, ChunHao Paul Huang, Tuanfeng Yang Wang, and Gordon Wetzstein. Generative rendering: Controllable 4d-guided video generation with 2d diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7611–7620, 2024. 2
- [12] Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al. Efficient geometry-aware 3d generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16123– 16133, 2022. 2
- [13] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7310– 7320, 2024. 2
- [14] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607. PMLR, 2020. 6
- [15] Ziyang Chen, Daniel Geng, and Andrew Owens. Images that sound: Composing images and sounds on a single canvas. arXiv preprint arXiv:2405.12221, 2024. 3
- [16] Gordon Christie, Neil Fendley, James Wilson, and Ryan Mukherjee. Functional map of the world. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6172–6180, 2018. 2
- [17] David J Crandall, Lars Backstrom, Daniel Huttenlocher, and Jon Kleinberg. Mapping the world’s photos. In Proceedings of the 18th international conference on World wide web, pages 761–770, 2009. 1, 2
- [18] Boyang Deng, Richard Tucker, Zhengqi Li, Leonidas Guibas, Noah Snavely, and Gordon Wetzstein. Streetscapes: Large-scale consistent street view generation using autoregressive video diffusion. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 2
- [19] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 2, 3
- [20] Carl Doersch, Saurabh Singh, Abhinav Gupta, Josef Sivic, and Alexei Efros. What makes paris look like paris? ACM Transactions on Graphics, 31(4), 2012. 1, 2, 6
- [21] Yilun Du, Conor Durkan, Robin Strudel, Joshua B Tenenbaum, Sander Dieleman, Rob Fergus, Jascha SohlDickstein, Arnaud Doucet, and Will Sussman Grathwohl. Reduce, reuse, recycle: Compositional generation with energy-based diffusion models and mcmc. In International conference on machine learning, pages 8489–8510. PMLR,

2023. 3

- [22] Yasutaka Furukawa and Jean Ponce. Accurate, dense, and robust multiview stereopsis. IEEE transactions on pattern analysis and machine intelligence, 32(8):1362–1376, 2009. 2
- [23] Daniel Geng and Andrew Owens. Motion guidance: Diffusion-based image editing with differentiable motion estimators. arXiv preprint arXiv:2401.18085, 2024. 2
- [24] Daniel Geng, Inbum Park, and Andrew Owens. Visual anagrams: Generating multi-view optical illusions with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24154– 24163, 2024. 3
- [25] Daniel Geng, Inbum Park, and Andrew Owens. Factorized diffusion: Perceptual illusions by noise decomposition. In European Conference on Computer Vision, pages 366–384. Springer, 2025. 3
- [26] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15180–15190, 2023. 2
- [27] Lukas Haas, Michal Skreta, Silas Alberti, and Chelsea Finn. Pigeon: Predicting image geolocations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12893–12902, 2024. 2
- [28] Richard Hartley and Andrew Zisserman. Multiple view geometry in computer vision. Cambridge university press,

2003. 2

- [29] James Hays and Alexei A Efros. Im2gps: estimating geographic information from a single image. In 2008 ieee conference on computer vision and pattern recognition, pages 1–8. IEEE, 2008. 2
- [30] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020. 6
- [31] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016. 15
- [32] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 8
- [33] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 5
- [34] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2, 3
- [35] Lukas H”ollein, Ang Cao, Andrew Owens, Justin Johnson, and Matthias Nießner. Text2room: Extracting textured 3d meshes from 2d text-to-image models. International Conference on Computer Vision (ICCV), 2023. 7
- [36] Yukun Huang, Jianan Wang, Yukai Shi, Xianbiao Qi, Zheng-Jun Zha, and Lei Zhang. Dreamtime: An improved optimization strategy for text-to-3d content creation. arXiv preprint arXiv:2306.12422, 2023. 5

- [37] Ajay Jain, Ben Mildenhall, Jonathan T Barron, Pieter Abbeel, and Ben Poole. Zero-shot text-guided object generation with dream fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 867–876, 2022. 2
- [38] Takeo Kanade and Masatoshi Okutomi. A stereo matching algorithm with an adaptive window: Theory and experiment. IEEE transactions on pattern analysis and machine intelligence, 16(9):920–932, 1994. 2
- [39] Samar Khanna, Patrick Liu, Linqi Zhou, Chenlin Meng, Robin Rombach, Marshall Burke, David Lobell, and Stefano Ermon. Diffusionsat: A generative foundation model for satellite imagery. arXiv preprint arXiv:2312.03606,

2023. 2

- [40] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980,

2014. 5

- [41] Zhengfei Kuang, Shengqu Cai, Hao He, Yinghao Xu, Hongsheng Li, Leonidas Guibas, and Gordon Wetzstein. Collaborative video diffusion: Consistent multivideo generation with camera control. arXiv preprint arXiv:2405.17414, 2024. 2
- [42] Nupur Kumari, Grace Su, Richard Zhang, Taesung Park, Eli Shechtman, and Jun-Yan Zhu. Customizing text-to-image diffusion with camera viewpoint control. arXiv preprint arXiv:2404.12333, 2024. 2
- [43] Ruining Li, Chuanxia Zheng, Christian Rupprecht, and Andrea Vedaldi. Dragapart: Learning a part-level motion prior for articulated objects. arXiv preprint arXiv:2403.15382,

2024. 2

- [44] Yuan Li, Zhi-Hao Lin, David Forsyth, Jia-Bin Huang, and Shenlong Wang. Climatenerf: Extreme weather synthesis in neural radiance field. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3227– 3238, 2023. 6
- [45] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: Highresolution text-to-3d content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 300–309, 2023. 2, 3, 4
- [46] Andrew Liu, Shiry Ginosar, Tinghui Zhou, Alexei A Efros, and Noah Snavely. Learning to factorize and relight a city. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16, pages 544–561. Springer, 2020. 6
- [47] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual generation with composable diffusion models. In European Conference on Computer Vision, pages 423–439. Springer, 2022. 3, 7
- [48] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9298–9309, 2023. 2, 3
- [49] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic

- gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016. 15
- [50] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 5, 16
- [51] Yin Lou, Noah Snavely, and Johannes Gehrke. Matchminer: Efficient spanning structure mining in large image collections. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part II 12, 2012. 2
- [52] David G Lowe. Distinctive image features from scaleinvariant keypoints. International journal of computer vision, 60:91–110, 2004. 2
- [53] Utkarsh Mall, Kevin Matzen, Bharath Hariharan, Noah Snavely, and Kavita Bala. Geostyle: Discovering fashion trends and events. In Proceedings of the IEEE/CVF international conference on computer vision, pages 411–420,

2019. 1, 2

- [54] Utkarsh Mall, Kavita Bala, Tamara Berg, and Kristen Grauman. Discovering underground maps from fashion. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3114–3123, 2022. 2
- [55] Ricardo Martin-Brualla, Noha Radwan, Mehdi SM Sajjadi, Jonathan T Barron, Alexey Dosovitskiy, and Daniel Duckworth. Nerf in the wild: Neural radiance fields for unconstrained photo collections. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7210–7219, 2021. 8, 16
- [56] Ricardo Martin-Brualla, Noha Radwan, Mehdi S. M. Sajjadi, Jonathan T. Barron, Alexey Dosovitskiy, and Daniel Duckworth. NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections. In CVPR, 2021. 5, 15, 16, 17
- [57] Kevin Matzen and Noah Snavely. Scene chronology. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part VII 13, pages 615–630. Springer, 2014. 2
- [58] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 2, 4, 5, 8
- [59] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (ToG), 41(4):1–15, 2022. 5
- [60] David Nister and Henrik Stewenius. Scalable recognition with a vocabulary tree. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’06), pages 2161–2168. Ieee, 2006. 2
- [61] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 6
- [62] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 15

- [63] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195– 4205, 2023. 2
- [64] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. ICLR,

2023. 2, 3, 4, 5, 7, 8, 15, 16

- [65] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3, 6, 7
- [66] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022. 2
- [67] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 3, 5, 6, 7, 8, 9, 16
- [68] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500– 22510, 2023. 4
- [69] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 2, 4
- [70] Aditya Sanghi, Hang Chu, Joseph G Lambourne, Ye Wang, Chin-Yi Cheng, Marco Fumero, and Kamal Rahimi Malekshan. Clip-forge: Towards zero-shot text-to-shape generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18603–18613,

2022. 2

- [71] Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann, Hong-Xing Yu, Yunzhi Zhang, Eric Ryan Chan, Dmitry Lagun, Li Fei-Fei, Deqing Sun, et al. Zeronvs: Zero-shot 360-degree view synthesis from a single real image. arXiv preprint arXiv:2310.17994, 2023. 2
- [72] Johannes Lutz Sch¨onberger and Jan-Michael Frahm. Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 8, 16, 17
- [73] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104–4113, 2016. 2, 15, 16
- [74] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d

- generation. arXiv preprint arXiv:2308.16512, 2023. 2, 3, 4, 5
- [75] Yujun Shi, Chuhui Xue, Jun Hao Liew, Jiachun Pan, Hanshu Yan, Wenqing Zhang, Vincent YF Tan, and Song Bai. Dragdiffusion: Harnessing diffusion models for interactive point-based image editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8839–8849, 2024. 2
- [76] Abhinav Shrivastava, Tomasz Malisiewicz, Abhinav Gupta, and Alexei A Efros. Data-driven visual similarity for crossdomain image matching. ACM Trans. Graph., 30(6):154,

2011. 2

- [77] Ioannis Siglidis, Aleksander Holynski, Alexei A Efros, Mathieu Aubry, and Shiry Ginosar. Diffusion models as data mining tools. arXiv preprint arXiv:2408.02752, 2024. 2
- [78] Noah Snavely, Steven M Seitz, and Richard Szeliski. Photo tourism: exploring photo collections in 3d. In ACM siggraph 2006 papers, pages 835–846. 2006. 1, 2
- [79] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 2, 3
- [80] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 3
- [81] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019. 3
- [82] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 2, 3
- [83] Nicholas Collin Suwono, Justin Chih-Yao Chen, Tun Min Hung, Ting-Hao Kenneth Huang, I-Bin Liao, Yung-Hui Li, Lun-Wei Ku, and Shao-Hua Sun. Location-aware visual question generation with lightweight models. arXiv preprint arXiv:2310.15129, 2023. 2
- [84] Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li, Brent Yi, Terrance Wang, Alexander Kristoffersen, Jake Austin, Kamyar Salahi, Abhik Ahuja, et al. Nerfstudio: A modular framework for neural radiance field development. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–12, 2023. 7, 8, 15, 16, 17
- [85] Kevin Tang, Manohar Paluri, Li Fei-Fei, Rob Fergus, and Lubomir Bourdev. Improving image classification with location context. In Proceedings of the IEEE international conference on computer vision, pages 1008–1016, 2015. 2
- [86] Zineng Tang, Ziyi Yang, Mahmoud Khademi, Yang Liu, Chenguang Zhu, and Mohit Bansal. Codi-2: In-context interleaved and interactive any-to-any generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27425–27434, 2024. 2
- [87] Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, and Mohit Bansal. Any-to-any generation via composable diffusion. Advances in Neural Information Processing Systems, 36, 2024. 2

- [88] Carlo Tomasi and Takeo Kanade. Shape and motion from image streams under orthography: a factorization method. International journal of computer vision, 9:137–154, 1992. 2
- [89] Antonio Torralba and Aude Oliva. Statistics of natural image categories. Network: computation in neural systems,

2003. 6

- [90] Vicente Vivanco Cepeda, Gaurav Kumar Nayak, and Mubarak Shah. Geoclip: Clip-inspired alignment between locations and images for effective worldwide geolocalization. Advances in Neural Information Processing Systems, 36, 2024. 2
- [91] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12619–12629, 2023. 2, 3, 4
- [92] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: Highfidelity and diverse text-to-3d generation with variational score distillation. arXiv preprint arXiv:2305.16213, 2023. 3, 4
- [93] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 2
- [94] Tobias Weyand, Ilya Kostrikov, and James Philbin. Planetphoto geolocation with convolutional neural networks. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part VIII 14, pages 37–55. Springer, 2016. 2
- [95] Changchang Wu, Sameer Agarwal, Brian Curless, and Steven M Seitz. Multicore bundle adjustment. In CVPR 2011, pages 3057–3064. IEEE, 2011. 2
- [96] Shixiong Xu, Chenghao Zhang, Lubin Fan, Gaofeng Meng, Shiming Xiang, and Jieping Ye. Addressclip: Empowering vision-language models for city-wide image address localization. arXiv preprint arXiv:2407.08156, 2024. 2
- [97] Le Xue, Manli Shu, Anas Awadalla, Jun Wang, An Yan, Senthil Purushwalkam, Honglu Zhou, Viraj Prabhu, Yutong Dai, Michael S Ryoo, et al. xgen-mm (blip-3): A family of open large multimodal models. arXiv preprint arXiv:2408.08872, 2024. 3, 5, 15
- [98] Fengyu Yang, Jiacheng Zhang, and Andrew Owens. Generating visual scenes from touch. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22070–22080, 2023. 2
- [99] Fengyu Yang, Chao Feng, Ziyang Chen, Hyoungseob Park, Daniel Wang, Yiming Dou, Ziyao Zeng, Xien Chen, Rit Gangopadhyay, Andrew Owens, et al. Binding touch to everything: Learning unified multimodal tactile representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26340– 26353, 2024. 2
- [100] Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. Dragnuwa: Fine-grained

- control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089, 2023. 2
- [101] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 2, 3
- [102] Xiaohan Zhang, Xingyu Li, Waqas Sultani, Chen Chen, and Safwan Wshah. Geodtr+: Toward generic crossview geolocalization via geometric disentanglement. IEEE Transactions on Pattern Analysis and Machine Intelligence,

2024. 2

|“car”<br><br>[Figure 221]|
|---|

|“selfie”<br><br>[Figure 222]|
|---|

|“breakfast”<br><br>[Figure 223]|
|---|

|[Figure 224]<br><br>“spiderman”|
|---|

|[Figure 225]<br><br>“aerial view in oil painting style”|
|---|

|“aerial view”<br><br>[Figure 226]|
|---|

|“musicals”<br><br>[Figure 227]|
|---|

|“batman”<br><br>[Figure 228]|
|---|

|[Figure 229]<br><br>[Figure 230]<br><br>[Figure 231]<br><br>[Figure 232]<br><br>[Figure 233]<br><br>[Figure 234]<br><br>[Figure 235]<br><br>[Figure 236]<br><br>[Figure 237]|
|---|

|“aerial view”<br><br>[Figure 238]|
|---|

|“eiffel tower”<br><br>[Figure 239]|
|---|

|[Figure 240]<br><br>[Figure 241]<br><br>[Figure 242]<br><br>[Figure 243]<br><br>[Figure 244]<br><br>[Figure 245]<br><br>[Figure 246]<br><br>[Figure 247]<br><br>[Figure 248]<br><br>[Figure 249]<br><br>[Figure 250]<br><br>[Figure 251]<br><br>[Figure 252]<br><br>|
|---|

|[Figure 253]<br><br>“batman”|
|---|

|[Figure 254]<br><br>“yellow cab”|
|---|

|“restaurant”<br><br>[Figure 255]|
|---|

|[Figure 256]<br><br>“apple event”|
|---|

|[Figure 257]<br><br>“pedestrian”|
|---|

|“spiderman”<br><br>[Figure 258]|
|---|

|[Figure 259]<br><br>“building”|
|---|

|“vintage car”<br><br>[Figure 260]|
|---|

|[Figure 261]<br><br>“tiger”|
|---|

|[Figure 262]<br><br>“rubber duck”|
|---|

(a) New York City (b) Paris

- Figure 10. More qualitative results for GPS-to-image generation. We present more qualitative results of GPS-to-image generation for New York City and Paris. Images are sampled from a variety of GPS locations and text prompts.

### A.1. GPS-to-image generation

#### A.1.1. More qualitative results

We present more qualitative results for our GPS-to-image generation in Fig. 10. As shown in Fig. 10, our method can successfully generate images conditioned on GPS tag and text prompt in a compositional manner. For instance, in New York City (a): 1) text prompt “tiger” along with GPS location of the Charging Bull statue generates an image of a tiger in a similar pose to the Charging Bull, with an appropriately matching background; 2) given text prompt “spiderman” and “batman”, we can generate an image of either an oil painting of spiderman or a stele of batman, depending on the location within The Metropolitan Museum of Art; 3) when conditioned on the GPS location of Madison Square Garden and the text prompt “apple event”, our model generates an image that appears to have been taken at Madison Square Garden (see ceiling in the image) during a real Apple event (see Apple logo in the image). In Paris (b): 1) with text prompt “spiderman” and GPS location of Rodin Museum, the GPS-to-image diffusion model can generate an image of spiderman statue posed similarly to The Thinker; 2) text prompt “batman” and GPS location of Louvre Museum’s statue gallery can result in an image of statue of batman, while the model can generate an image of painting about Eiffel Tower when GPS location is the painting gallery of Louvre and text prompt is “eiffel tower”; 3) given the text prompt “musicals” and the GPS location of Palais Garnier, an image depicting a musical performance at Palais Garnier is generated; 4) when conditioned on text prompt “breakfast” and GPS location of Orsay Museum, our

GPS-to-image diffusion model can generate an image of oil painting of breakfast; 5) using the text prompt “car” along with the GPS location of the Champs-Elys´´ ees, an image of the car is generated with a background filled with buildings in the Haussmannian architectural style. Additionally, we present randomly sampled images from generation results of our GPS-to-image diffusion models in Fig. 11. Some images have visible artifacts, and this may be due to the limited availability of photos with GPS tags in that area.

#### A.1.2. Evaluation set

We create a test set for New York City and Paris, comprised of 1292 pairs of text prompts and GPS tags in total. We attach two files nyc-eval.json and paris-eval.json to show the lists we use for each city.

#### A.1.3. Average images

Within a selected area, we have a set of sampled locations with GPS coordinates {(x0,y0),(x1,y1),...,(xM−1,yM−1)}, then we could get their corresponding GPS embeddings {g0,...,gM−1}. For the concept like text prompt “building”, we obtain fixed text embedding p for {g0,...,gM−1}. The noise estimate is as follows:

ϵ¯ϕ (zt; p, g, t) = ϵϕ (zt; ∅, ∅, t)

+ ωp (ϵϕ (zt; p, ∅, t) − ϵϕ (zt; ∅, ∅, t))

M−1 i=0 ϵϕ (zt; p, gi, t)

M − ϵϕ (zt; p, ∅, t) ,

+ ωg

(8)

where ωp and ωg are guidance weights also used in Eq. (4), and ϵϕ is the denoiser of our trained GPS-to-image diffusion

”At night” 48°51'24.0"N ”A cyclist” 2°17'48.7"E

”Tourist boat” 48°51'15.8"N 2°20'35.7"E

”Building” 48°50'50.0"N 2°20'13.6"E

48°52'17.8"N 2°17'43.5"E

“antique market stalls” 48°51'39.3"N 2°20'12.0"E

”bench with umbrella” 40°42'51.7"N 74°00'30.2"W

”bench with umbrella” 48°52'18.0"N 2°17'36.6"E

”boat” 40°46'23.7"N 73°58'06.4"W

”building” 48°50'42.7"N 2°19'56.3"E

”building” 48°52'24.6"N 2°17'44.4"E

”city map board for tourists” 40°46'11.9"N 73°58'38.9"W

”coffee kiosk” 40°43'47.6"N 73°59'26.6"W

“corner grocery store” 40°44'29.5"N 73°59'22.4"W

”coffee kiosk” 48°52'24.6"N 2°17'44.4"E

|[Figure 263]|
|---|

|[Figure 264]|
|---|

|[Figure 265]|
|---|

|[Figure 266]|
|---|

|[Figure 267]|
|---|

|[Figure 268]|
|---|

|[Figure 269]|
|---|

|[Figure 270]|
|---|

|[Figure 271]|
|---|

|[Figure 272]|
|---|

“flower boxes on balconies” 48°51'15.7"N 2°20'35.3"E

“flower boxes on balconies” 48°50'34.7"N 2°18'46.4"E

“dog” 40°46'29.0"N 73°58'21.1"W

”flower garden” 48°50'42.0"N 2°20'13.4"E

”flower vendor stall” 40°43'53.0"N 73°59'49.0"W

”flower vendor stall” 48°50'48.7"N 2°20'35.1"E

”green metal chairs” 48°50'40.5"N 2°20'03.2"E

”fountain” 48°50'40.5"N 2°20'03.2"E

“ice cream truck” 40°43'53.0"N 73°59'49.0"W

”ice cream truck” 40°44'29.5"N 73°59'22.4"W

|[Figure 273]|
|---|

|[Figure 274]|
|---|

|[Figure 275]|
|---|

|[Figure 276]|
|---|

|[Figure 277]|
|---|

|[Figure 278]|
|---|

|[Figure 279]|
|---|

|[Figure 280]|
|---|

|[Figure 281]|
|---|

|[Figure 282]|
|---|

”public bulletin board with event flyers”

“ice cream truck” 40°44'59.2"N 73°59'15.8"W

”park clock tower” 40°47'33.3"N 73°57'29.3"W

”rooftop” 48°51'52.1"N 2°18'36.9"E

”selfie” 40°46'08.1"N 73°58'22.3"W

”selfie” 48°50'49.2"N 2°20'01.5"E

”snowing” 48°50'40.5"N 2°20'03.2"E

”snowing” 48°50'44.6"N 2°20'14.3"E

“spring” 40°46'25.1"N 73°58'23.2"W

”statue” 48°50'53.0"N 2°20'21.4"E

40°42'32.9"N 74°00'38.0"W

|[Figure 283]|
|---|

|[Figure 284]|
|---|

|[Figure 285]|
|---|

|[Figure 286]|
|---|

|[Figure 287]|
|---|

|[Figure 288]|
|---|

|[Figure 289]|
|---|

|[Figure 290]|
|---|

|[Figure 291]|
|---|

|[Figure 292]|
|---|

”tourist information booth”

”tourist information booth”

“streetlamp” 48°51'23.5"N 2°17'49.2"E

”summer” 48°50'44.0"N 2°20'05.2"E

”summer” 48°50'49.9"N 2°20'12.9"E

”autumn” 48°50'53.0"N 2°20'21.4"E

”tourists” 40°46'10.8"N 73°58'36.5"W

”tree” 40°46'08.1"N 73°58'22.3"W

“tree” 40°47'49.1"N 73°57'27.5"W

”tree” 48°50'49.2"N 2°20'19.0"E

40°42'17.4"N 74°00'55.7"W

40°45'30.4"N 73°58'44.0"W

|[Figure 293]|
|---|

|[Figure 294]|
|---|

|[Figure 295]|
|---|

|[Figure 296]|
|---|

|[Figure 297]|
|---|

|[Figure 298]|
|---|

|[Figure 299]|
|---|

|[Figure 300]|
|---|

|[Figure 301]|
|---|

|[Figure 302]|
|---|

- Figure 11. Random sampling. We show some randomly sampled images from generation results of our GPS-to-image diffusion models conditioned on text prompts and GPS tags. These sampled results were used in the quantitative evaluation.

Algorithm 1 Pseudocode of training GPS-CLIP.

model. It is worth noting that all average images shown in Fig. 6 share the same initial random noise.

# x, y: batch of longitudes and latitudes # imgs: batch of images # f_gps: shared-weight encoder for longitude and

latitude # f_v: vision encoder of DINOv2

#### A.1.4. GPS-CLIP

- # p: projection layer for f_gps

- # q: projection layer for f_v # t: temperature for imgs, x, y in loader: # load a minibatch

As mentioned in Sec. 4, we use GPS score which measures cosine similarity between image and GPS embeddings as one of evaluation metrics. Specifically, we use pretrained frozen DINOv2 (ViT-B/14) [62] as image encoder. We add a single projection layer to the image encoder. For the GPS encoder, we use a shared-weight 6-layer MLP for latitude and longitude. The resulting embeddings are concatenated and passed through a single layer to produce the final GPS embedding, which has the same dimensionality as DINOv2. We use GELU [31] as activation function for GPS encoder. The batch size is 512, and temperature is 0.07, learning rate is 1 × 10−4 with warmup and cosine learning rate decay [49]. We train GPS-CLIP on a single NVIDIA L40S. The pseudocode for training process is presented in Algorithm 1.

- x_f = f_gps.forward(x)

- y_f = f_gps.forward(y) gps_e = p.forward(cat([x_f, y_f], dim=1)) # GPS

embedding img_f = f_v.forward(imgs) img_e = q.forward(img_f) # image embedding gps_e = gps_e / norm(gps_e) # embedding

normalization img_e = img_e / norm(img_e) # embedding

normalization logits = mm(img_e.view(1, C), gps_e.view(1, C).T)/t labels = torch.arange(n) loss_i = cross_entropy_loss(logits, labels, axis=0) loss_g = cross_entropy_loss(logits, labels, axis=1) loss = (loss_i + loss_g)/2 loss.backward()

mm: matrix multiplication.

### A.2. GPS-guided 3D reconstruction

#### A.1.5. Implementation details

We use xgen-mm-phi3-mini-instruct-r-v1 of BLIP-3 [97] as our captioning model for collected datasets. For classifierfree guidance (CFG), we set ωp to 3.5 and ωg to 7.5 in Eq. (4) for GPS-to-image diffusion.

- • More 3D qualitative comparisons between our method and DreamFusion [64] are presented in Fig. 12 and project webpage. Please refer to Sec. A.2.2 for more details.
- • Qualitative results regarding SfM [73], Nerfacto [84], and NeRF-W [56] are shown in Fig. 13. Please refer

RGB Surface normal

RGB Surface normal RGB Surface normal RGB Surface normal

|[Figure 303]|
|---|

|[Figure 304]|
|---|

|[Figure 305]|
|---|

|[Figure 306]|
|---|

|[Figure 307]|
|---|

|[Figure 308]|
|---|

|[Figure 309]|
|---|

|[Figure 310]|
|---|

DreamFusionOurs

|[Figure 311]|
|---|

|[Figure 312]|
|---|

|[Figure 313]|
|---|

|[Figure 314]|
|---|

|[Figure 315]|
|---|

|[Figure 316]|
|---|

|[Figure 317]|
|---|

|[Figure 318]|
|---|

- Figure 12. More qualitative comparison for 3D monument reconstruction. We show qualitative results of DreamFusion [64] and our method on Stonehenge. Our reconstructed 3D monuments have better visual quality and more accurate 3D structure. We use rendered depth to make the background of RGB rendering white. Please see project webpage for more examples.

Qualitative results. We show more qualitative comparison between our method and DreamFusion [64] in Fig. 12 and project webpage. It should be noted that for all videos, we directly use raw renderings and do not use rendered depth to make the background of RGB rendering white.

to Sec. A.2.3 for more details.

• Some qualitative results of angle-to-image diffusion are presented in Fig. 14, please refer to Sec. A.2.1 for more details.

#### A.2.1. Angle-to-image diffusion

#### A.2.3. Baseline of COLMAP with NeRF

Prior preservation loss. As mentioned in Sec. 3.2, for angle-to-image model training, we utilize prior preservation loss. To be specific, with synthesized images X∗ from original stable diffusion model [67] and text condition p, we optimize the preservation loss:

We present qualitative results of COLMAP [73], NeRFW [56], and Nerfacto [84] in Fig. 13. Since NeRF-W’s [56] official code is not available, we evaluate the popular reimplementation. As shown in Fig. 13, COLMAP [72] successfully reconstructs camera poses and sparse point clouds for 3 of the 6 scenes, and fails on 3. NeRF-W [55] estimation completely fails on 6 landmarks and Nerfacto [84] fails on 5.

Lpreservation = Ex∗,ϵ,t ∥ ϵt − ϵϕ(z∗t;p,∅,t) ∥22 , (9)

where ∅ represents that we zero out the angle condition for these training examples.

Implementation details. For each landmark, we finetune Stable Diffusion-v1.4 [67] on collected Flickr images, at a resolution of 256×256 for 800 steps. After angle discretization, we normalize the angle to the range of [−1,1]. We use a positional encoding and a two-layer MLP to encode the angle condition. For the positional encoding, we use 10 frequencies. We use the AdamW [50] optimizer with a constant learning rate of 5 × 10−6 and gradient accumulation without warm-up. We use a global batch size of 256 on 4 NVIDIA A40 GPUs.

### A.3. Datasets

Tourist photo collection. By querying Flickr, we obtain photo collections for 2 cities: 1) New York City (Manhattan, 501,592 photos); 2) Paris (315,306 photos) and 6 landmarks: 1) Leaning Tower of Pisa (2,967 photos); 2) Arc de Triomphe (2,377 photos); 3) Washington Monument (2,563 photos); 4) Statue of Liberty (1,174 photos); 5) Stonehenge (2,486 photos); 6) Space Needle (1,800 photos). The number of evaluated landmarks is in-line with prior work [56] in the field. It is worth noting that we focus primarily on Manhattan for New York City due to resource constraints. Some examples sampled from datasets are shown in Fig. 15. It should be noted that 2 cities are collected for GPS-toimage generation and 6 landmarks are for angle-to-image generation and 3D landmark reconstruction. As mentioned in Sec. 3.2, the angle of capture is necessary so we use a bespoke dataset for each landmark.

Qualitative results. Some generated images from our angle-to-image diffusion model are presented in Fig. 14.

#### A.2.2. GPS-guided score distillation sampling

Gradient. The gradient in Sec. 3.2 we use to supervise NeRF is as follows:

∇θLSDS (ϕ,x = hθ (q)) ≈ Eg′,ϵ

∂x ∂θ

, (10)

t,t ω (t)(ϵˆϕ (zt;p,g′,t) − ϵt)

where ω (t) is a weighting function, which we set to ω (t) = σt2 following [64].

NeRF-W

Real Image SfM point cloud Nerfacto

NeRF-W

Real Image SfM point cloud Nerfacto

|[Figure 319]|
|---|

|[Figure 320]|
|---|

|[Figure 321]|
|---|

|[Figure 322]|
|---|

|[Figure 323]|
|---|

|[Figure 324]|
|---|

|[Figure 325]|
|---|

|[Figure 326]|
|---|

(b) Leaning Tower of Pisa

(a) Washington Monument

|[Figure 327]|
|---|

|[Figure 328]|
|---|

|[Figure 329]|
|---|

|[Figure 330]|
|---|

|[Figure 331]|
|---|

|[Figure 332]|
|---|

|[Figure 333]|
|---|

|[Figure 334]|
|---|

(d) Arc de Triomphe

(c) Space Needle

|[Figure 335]|
|---|

|[Figure 336]|
|---|

|[Figure 337]|
|---|

|[Figure 338]|
|---|

|[Figure 339]|
|---|

|[Figure 340]|
|---|

|[Figure 341]|
|---|

|[Figure 342]|
|---|

(e) Statue of Liberty

(f) Stonehenge

- Figure 13. SfM/NeRF baselines. We present SfM reconstructions from COLMAP [72], Nerfacto [84] rendering results, and NeRF-W [56] rendering results for 6 evaluated landmarks. SfM reconstruction fails on (a), (b), and (c). Nerfacto [84] only succeeds on (f). NeRF-W [56] completely fails on 6 scenes.

|[Figure 343]|
|---|

|[Figure 344]|
|---|

|[Figure 345]|
|---|

|[Figure 346]|
|---|

|[Figure 347]|
|---|

|[Figure 348]|
|---|

|[Figure 349]|
|---|

|[Figure 350]|
|---|

|[Figure 351]|
|---|

|[Figure 352]|
|---|

|[Figure 353]|
|---|

LeaningTower

ofPisa Arcde

Triomphe Statueof

Liberty

|[Figure 354]|
|---|

|[Figure 355]|
|---|

|[Figure 356]|
|---|

|[Figure 357]|
|---|

|[Figure 358]|
|---|

|[Figure 359]|
|---|

|[Figure 360]|
|---|

0° 60° 120° 180° 240° 300°

- Figure 14. Qualitative results for angle-to-image generation. We show generated images of our angle-to-image diffusion model for the Arc de Triomphe, Statue of Liberty, and Leaning Tower of Pisa. Images are sampled conditioned on different angles estimated by GPS tags.

|[Figure 361]|
|---|

|[Figure 362]<br><br>[Figure 363]|
|---|

|[Figure 364]<br><br>[Figure 365]|
|---|

|[Figure 366]|
|---|

|[Figure 367]|
|---|

|[Figure 368]<br><br>[Figure 369]|
|---|

|[Figure 370]|
|---|

|[Figure 371]|
|---|

[Figure 372]

[Figure 373]

New York City

Paris

48° 51' 12.0" N 02° 20' 57.0" E

48° 51' 40.0" N 02° 20' 11.0" E

48° 51' 29.7" N 02° 17' 38.6" E

40° 42' 19.4" N 74° 00' 48.7" W

48° 51' 12.4" N 02° 20' 55.2" E

40° 45' 05.4" N 73° 58' 33.0" W

40° 45' 28.9" N 73° 59' 02.8" W

40° 44' 35.3" N 73° 59' 16.4" W

[Figure 374]

|[Figure 375]|
|---|

|[Figure 376]|
|---|

|[Figure 377]<br><br>[Figure 378]|
|---|

|[Figure 379]|
|---|

|[Figure 380]<br><br>[Figure 381]|
|---|

|[Figure 382]|
|---|

|[Figure 383]|
|---|

|[Figure 384]|
|---|

[Figure 385]

Arc de Triomphe

Statue of Liberty

40° 41’ 08.46" N 74° 02' 05.33" W

40° 41' 19.53" N 74° 02' 38.40" W

40° 41' 16.20" N 74° 02' 31.20" W

40° 41' 16.20" N 74° 02' 31.20" W

48° 52' 28.50" N 02° 17' 46.40" E

48° 52' 25.37" N 02° 17' 46.37" E

48° 52' 22.94" N 02° 17' 43.90" E

48° 52' 25.20" N 20° 17' 46.92" E

|[Figure 386]|
|---|

|[Figure 387]<br><br>[Figure 388]|
|---|

|[Figure 389]|
|---|

|[Figure 390]|
|---|

|[Figure 391]|
|---|

|[Figure 392]|
|---|

|[Figure 393]|
|---|

|[Figure 394]|
|---|

[Figure 395]

[Figure 396]

Stonehenge

Washington Monument

38° 53' 21.60" N 77° 02' 07.20" W

38° 53' 20.40" N 77° 02' 09.00" W

38° 53' 25.00" N 77° 01' 55.00" W

38° 53' 19.80" N 77° 02' 07.80" W

51° 10' 43.80" N 01° 49' 34.20" W

51° 10' 46.35" N 01° 49' 36.57" W

51° 10' 45.00" N 01° 49' 34.80" W

51° 10' 44.75" N 01° 49' 29.47" W

|[Figure 397]<br><br>[Figure 398]|
|---|

|[Figure 399]<br><br>[Figure 400]|
|---|

|[Figure 401]|
|---|

|[Figure 402]|
|---|

|[Figure 403]<br><br>[Figure 404]<br><br>[Figure 405]|
|---|

|[Figure 406]<br><br>[Figure 407]|
|---|

|[Figure 408]|
|---|

|[Figure 409]|
|---|

[Figure 410]

[Figure 411]

Space Needle

Leaning Tower of Pisa

43° 43' 21.30" N 10° 23' 44.09" E

43° 43' 21.96" N 10° 23' 39.95" E

43° 43' 21.04" N 10° 23' 46.46" E

47° 37' 13.398" N 122° 21’ 06.78" W

40° 37' 07.71" N 122° 21' 1.272" W

47° 37' 12.6" N 122° 21' 7.2" W

47° 37' 13.2" N 122° 20' 54.342" W

43° 43' 22.61" N 10° 23' 46.02" E

Figure 15. Data samples. We show some random photos with their GPS tags from our collected datasets.

