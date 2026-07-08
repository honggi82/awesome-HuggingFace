## Hash3D: Training-free Acceleration for 3D Generation

# arXiv:2404.06091v1[cs.CV]9Apr2024

##### Xingyi Yang Xinchao Wang

###### National University of Singapore xyang@u.nus.edu, xinchao@nus.edu.sg

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

Gaussian-DreamerDream-Gaussian

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

1.5×

[Figure 18]

+Hash3D+Hash3D

[Figure 19]

[Figure 20]

A marble bust of a fox head A porcelain dragon

An old car overgrown by vines and weeds

An orc forging a hammer on an anvil A gummy bear playing the saxophone

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

4.𝟎×

Fig. 1: Examples by applying our Hash3D on Gaussian-Dreamer [58] and DreamGaussian [49]. We accelerate Gaussian-Dreamer by 1.5× and Dream-Gaussian by 4× with comparable visual quality.

Abstract. The evolution of 3D generative modeling has been notably propelled by the adoption of 2D diffusion models. Despite this progress, the cumbersome optimization process per se presents a critical hurdle to efficiency. In this paper, we introduce Hash3D, a universal acceleration for 3D generation without model training. Central to Hash3D is the insight that feature-map redundancy is prevalent in images rendered from camera positions and diffusion time-steps in close proximity. By effectively hashing and reusing these feature maps across neighboring timesteps and camera angles, Hash3D substantially prevents redundant calculations, thus accelerating the diffusion model’s inference in 3D generation tasks. We achieve this through an adaptive grid-based hashing. Surprisingly, this feature-sharing mechanism not only speed up the generation but also enhances the smoothness and view consistency of the synthesized 3D objects. Our experiments covering 5 textto-3D and 3 image-to-3D models, demonstrate Hash3D’s versatility to speed up optimization, enhancing efficiency by 1.3 ∼ 4×. Additionally, Hash3D’s integration with 3D Gaussian splatting largely speeds up 3D model creation, reducing text-to-3D processing to about 10 minutes and image-to-3D conversion to roughly 30 seconds. The project page is at https://adamdad.github.io/hash3D/.

Keywords: Fast 3D Generation · Score Distillation Sampling

### 1 Introduction

In the evolving landscape of 3D generative modeling, the integration of 2D diffusion models [35,51] has led to notable advancements. These methods leverage off-the-the-shelf image diffusion models to distill 3D models by predicting 2D score functions at different views, known as score distillation sampling (SDS).

While this approach has opened up new avenues for creating detailed 3D assets, it also brings forth significant challenges, particularly in terms of efficiency. Particularly, SDS requires sampling thousands of score predictions at different camera poses and denoising timesteps from the diffusion model, causing a extensively long optimization, even for hours to create one object [52]. These prolonged duration create a significant obstacle to apply them in practical application products, calling for new solutions to improve its efficiency.

To mitigate this bottleneck, current efforts concentrate on three strategies. The first strategy trains an inference-only models [7, 11, 18, 24, 56] to bypass the lengthy optimization process. While effective, this method requires extensive training time and substantial computational resources. The second approach [39, 49,58] seeks to reduce optimization times through enhanced 3D parameterization techniques. However, this strategy necessitates a unique design for each specific representation, presenting its own set of challenges. The third approach attempts to directly generate sparse views to model 3D objects, assuming near-perfect view consistency in generation [16,27] which, in practice, is often not achievable.

Returning to the core issue within SDS, a considerable portion of computational effort is consumed in repeated sampling of the 2D image score function [48]. Motivated by methods that accelerate 2D diffusion sampling [3,28,46], we posed the question: Is it possible to reduce the number of inference steps of the diffusion model for 3D generation?

In pursuit of this, our exploration revealed a crucial observation: denoising outputs and feature maps from near camera positions and timesteps are remarkably similar. This discovery directly informs our solution, Hash3D, designed to reduce the computation by leveraging this redundancy.

At its core, Hash3D implements a space-time trade-off through a grid-based hash table. This table stores intermediate features from the diffusion model. Whenever a new sampled view is close to one it has already worked on, Hash3D efficiently retrieves the relevant features from the hash table. By reusing these features to calculate the current view’s score function, it avoids redoing calculations that have already been done. Additionally, we have developed a method to dynamically choose the grid size for each view, enhancing the system’s adaptability. As such, Hash3D not only conserves computational resources, but does so without any model training or complex modifications, making it simple to implement and efficient to apply.

Beyond just being efficient, Hash3D helps produce 3D objects with improved multi-view consistency. Traditional diffusion-based methods often result in 3D objects with disjointed appearances when viewed from various angles [2]. In contrast, Hash3D connects independently sampled views by sharing features within each grid, leading to smoother, more consistent 3D models.

Another key advantage of Hash3D is on its versatility. It integrates seamlessly into a diverse array of diffusion-based 3D generative workflows. Our experiments, covering 5 text-to-3D and 3 image-to-3D models, demonstrate Hash3D’s versatility to speed up optimization, enhancing efficiency by 1.3 ∼ 4×, without compromising on performance. Specifically, the integration of Hash3D with 3D Gaussian Splatting [13] brings a significant leap forward, cutting down the time for text-to-3D to about 10 minutes and image-to-3D to roughly 30 seconds.

The contribution of this paper can be summarized into

- – We introduce the Hash3D, a versatile, plug-and-play and training-free acceleration method for diffusion-based text-to-3D and image-to-3D models.
- – The paper emphasizes the redundancy in diffusion models when processing nearby views and timesteps. This finding motivates the development of Hash3D, aiming to boost efficiency without compromising quality.
- – Hash3D employs an adaptive grid-based hashing to efficiently retrieve features, significantly reducing the computations across view and time.
- – Our extensive testing across a range of models demonstrates that Hash3D not only speeds up the generative process by 1.3 ∼ 4×, but also results in a slight improvement in performance.

- 2 Related Work
- 3D Generation Model. The development of 3D generative models has become a focal point in the computer vision. Typically, these models are trained to produce the parameters that define 3D representations. This approach has been successfully applied across several larger-scale models using extensive and diverse datasets for generating voxel representation [54], point cloud [1,33], implicit function [12], triplane [45,56]. Despite these advances, scalability continues to be a formidable challenge, primarily due to data volume and computational resource constraints. A promising solution to this issue lies in leveraging

- 2D generative models to enhance and optimize 3D representations. Recently, diffusion-based models, particularly those involving score distillation into 3D representations [35], represent significant progress. However, these methods are often constrained by lengthy optimization processes. Efficient Diffusion Model. Diffusion models, known for their iterative denoising process for image generation, are pivotal yet time-intensive. There has been a substantial body of work aimed at accelerating these models. This acceleration can be approached from two angles: firstly, by reducing the sampling steps through advanced sampling mechanisms [3,22,28,46] or timestep distillation [44,47], which decreases the number of required sampling steps. The second approach focuses on minimizing the computational demands of each model inference. This can be achieved by developing smaller diffusion models [9,14,57] or reusing features from adjacent steps [20,29], thereby enhancing efficiency without compromising effectiveness. However, the application of these techniques to
- 3D generative tasks remains largely unexplored.

Hashing Techniques. Hashing, pivotal in computational and storage efficiency, involves converting variable-sized inputs into fixed-size hash code via hash functions. These code index a hash table, enabling fast and consistent data access. Widely used in file systems, hashing has proven effective in a variety of applications, like 3D representation [10,31,34,55], neural network compression [6,15], using hashing as a components in deep network [40] and neural network-based hash function development [4,17,19,60]. Our study explores the application of hashing to retrieve features from 3D generation. By adopting this technique, we aim to reduce computational overhead for repeated diffusion sampling and speed up the creation of realistic 3D objects.

### 3 Preliminary

In this section, we provide the necessary notations, as well as the background on optimization-based 3D generation, focusing on diffusion models and Score Distillation Sampling (SDS) [35].

#### 3.1 Diffusion Models

Diffusion models, a class of generative models, reverse a process of adding noise by constructing a series of latent variables. Starting with a dataset x0 drawn from a distribution q(x0), the models progressively introduce Gaussian noise over T steps. Each step, defined as q(xt|xt−1) = N(xt;√1 − βtxt−1,βtI), is controlled by β1:T, values ranging from 0 to 1. The inherently Gaussian nature

√of1this− α¯noisetϵ, enableswhere directϵ ∼ Nsampling(0,I) withfromαt =q(1x−t)βusingt andtheα¯t formula= ts=1 xαts.= √α¯tx0 +

The reverse process is formulated as a variational Markov chain, parameterized by a time-conditioned denoising neural network ϵ(xt,t,y), with y being the conditional input for generation, such as text for text-to-image model [43] or camera pose for novel view synthesis [25]. The training of the denoiser aims to minimize a re-weighted evidence lower bound (ELBO), aligning with the noise:

0,ϵ ||ϵ − ϵ(xt,t,y)||22 (1) Here, ϵ(xt,t,y) approximates the score function ∇xt

LDDPM = Et,x

log p(xt|x0). Data generation is achieved by denoising from noise, often enhanced using classifier-free guidance with scale parameter ω: ˆϵ(xt,t,y) = (1 + ω)ϵ(xt,t,y) − ωϵ(xt,t,∅).

Extracting Feature from Diffusion Model. A diffusion denoiser ϵ is typically parameterized with a U-Net [42]. It uses l down-sampling layers {Di}li=1 and up-sampling layers {Ui}li=1, coupled with skip connections that link features from Di to Ui. This module effectively merges high-level features from Ui+1 with low-level features from Di, as expressed by the equation:

vi(+1U) = concat(Di(vi(−D1)),Ui+1(vi(U))) (2)

In this context, vi(U) and vi(+1D) represent the up-sampled and down-sampled features after the i-th layer, respectively.

#### 3.2 Score Distillation Sampling (SDS)

The Score Distillation Sampling (SDS) [35] represents an optimization-based 3D generation method. This method focuses on optimizing the 3D representation, denoted as Θ, using a pre-trained 2D diffusion models with its noise prediction network, denoted as ϵpretrain(xt,t,y).

Given a camera pose c = (θ,ϕ,ρ) ∈ R3 defined by elevation ϕ, azimuth θ and camera distances ρ, and the its corresponding prompt yc, a differentiable rendering function g(·;Θ), SDS aims to refine the parameter Θ, such that each rendered image x0 = g(c;θ) is perceived as realistic by ϵpretrain. The optimization objective is formulated as follows:

LSDS = Et,c

min

Θ

σt αt

ω(t)KL qΘ(xt|yc,t)∥p(xt|yc;t) (3)

By excluding the Jacobian term of the U-Net, the gradient of the optimization problem can be effectively approximated:

∂x ∂Θ

∇ΘLSDS ≈ Et,c,ϵ ω(t)(ϵpretrain(xt,t,yc) − ϵ)

(4)

To optimize Eq. 4, we randomly sample different time-step t, camera c, and random noise ϵ, and compute gradient of the 3D representation, and update θ accordingly. This approach ensures that the rendered image from 3D object aligns with the distribution learned by the diffusion model.

Efficiency Problem. The main challenge lies in the need for thousands to tens of thousands of iterations to optimize Eq 4, each requiring a separate diffusion model inference. This process is time-consuming due to the model’s complexity. We make it faster by using a hash function to reuse features from similar inputs, cutting down on the number of calculations needed.

### 4 Hash3D

This section introduces Hash3D, a plug-and-play booster for Score Distillation Sampling (SDS) to improve its efficiency. We start by analyzing the redundancy presented in the diffusion model across different timestep and camera poses. Based on the finding, we present our strategy that employ a grid-based hashing to reuse feature across different sampling iterations.

#### 4.1 Probing the Redundancy in SDS

Typically, SDS randomly samples camera poses and timesteps to ensure that the rendered views align with diffusion model’s prediction. A critical observation here is that deep feature extraction at proximate c and t often reveals a high degree of similarity. This similarity underpins our method, suggesting that reusing features from nearby points does not significantly impact model’s prediction.

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

𝜃 = 0° 𝜃 = 90° 𝜃 = 180° 𝜃 = 270°

[Figure 52]

[Figure 53]

𝜃azimuth

𝜌polar

azimuth 𝜃

polar 𝜌

###### Fig. 2: Feature similarity extracted from different camera poses.

Full Prediction 𝛿 = 1° 𝛿 = 5° 𝛿 = 10° 𝛿 = 20° Full Prediction 𝛿 = 1° 𝛿 = 5° 𝛿 = 10° 𝛿 = 20°

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

- Fig. 3: By interpolating latent between generated views, we enable the synthesis of novel views with no computations.

Measuring the Similarity. Intuitively, images captured with close up camera and times results in similar visual information. We hypothesize that features produced by diffusion model would exhibit a similar pattern. In terms of the temporal similarity, previous studies [20,29] have noted that features extracted from adjacent timesteps from diffusion models show a high level of similarity.

To test the hypothesis about the spatial similarity, we conducted a preliminary study using the diffusion model to generate novel views of the same object from different camera positions. In practice, we use Zero-123 [25] to generate image from different cameras poses conditioned on single image input. For each

specific camera angle and timestep, we extracted features vl(−U1) as the input of the last up-sampling layer at each timestep. By adjusting elevation angles (ϕ)

and azimuth angles (θ), we were able to measure the cosine similarity of these features between different views, averaging the results across all timesteps.

The findings, presented in Figure 2, reveal a large similarity score in features from views within a [−10◦,10◦] range, with the value higher than 0.8. This phenomenon was not unique to Zero-123; we observed similar patterns in textto-image diffusion models like Stable Diffusion [41]. These findings underscore the redundancy in predicted outputs within the SDS process.

Synthesising Novel View for Free. Exploiting redundancy, we conducted an initial experiment to create new views by simply reusing and interpolating scores from precomputed nearby cameras. We started by generating 2 images using Zero-123 at angles (θ,ϕ) = (10◦ ± δ,90◦) and saved all denoising predictions from each timestep. Our goal was to average all pairs of predictions

to synthesize a 3-nd view at (10◦,90◦) for free. We experimented with varying δ ∈ {1◦,5◦,10◦,20◦}, and compared them with the full denoising predictions.

Figure 3 demonstrates that for angles (δ) up to 5◦, novel views closely match fully generated ones, proving effective for closely positioned cameras. Yet, interpolations between cameras at wider angles yield blurrier images. Additionally, optimal window sizes vary by object; for example, a δ = 5◦ suits the ghost but not the capybara, indicating that best window size is sample-specific.

Based on these insights, we presents a novel approach: instead of computing the noise prediction for every new camera pose and timestep, we create a memory system to store previously computed features. As such, we can retrieve and reuse these pre-computed features whenever needed. Ideally, this approach could reduces redundant calculations and speeds up the optimization process.

#### 4.2 Hashing-based Feature Reuse

In light of our analysis, we developed Hash3D, a solution that incorporates hashing techniques to optimize the SDS. Hash3D is fundamentally designed to minimize the repetitive computational burden typically associated with the diffusion model, effectively trading storage space for accelerated 3D optimization.

At the core of Hash3D is a hash table for storing and retrieving previously extracted features. When Hash3D samples a specific camera pose c and timestep t, it first checks the hash table for similar features. If a match is found, it’s reused directly in the diffusion model, significantly cutting down on computation. If there’s no match in the same hash bucket, the model performs standard inference, and the new feature is added to the hash table for future use.

Grid-based Hashing. For efficient indexing in our hash table, we use a gridbased hashing function with keys composed of camera poses c = (θ,ϕ,ρ) and timestep t. This function assigns each camera and timestep to a designated grid cell, streamlining data organization and retrieval.

Firstly, we define the size of our grid cells in both the spatial and temporal domains, denoted as ∆θ,∆ϕ,∆ρ and ∆t respectively. For each input key [θ,ϕ,ρ,t], the hashing function calculates the indices of the corresponding grid cell. This is achieved by dividing each coordinate by its respective grid size

i =

θ ∆θ

, j =

ϕ ∆ϕ

, k =

ρ ∆ρ

, l =

t ∆t

(5)

Upon obtaining these indices, we combine them into a single hash code that uniquely identifies each bucket in the hash table. The hash function idx = (i + N1 · j + N2 · k + N3 · l) mod n is used, where N1,N2,N3 are large prime numbers [34,50], and n denotes the size of the hash table.

Through this hash function, keys that are close in terms of camera pose and timestep are likely to be hashed to the same bucket. This grid-based approach not only making the data retrieval faster but also maintains the spatial-temporal relationship inherent in the data, which is crucial for our method.

Collision Resolution. When multiple keys are assigned to the same hash value, a collision occurs. We address these collisions using separate chaining. In this

[Figure 64]

###### Reuse Fe

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

|atu|re| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

###### Update Hash Table

###### Extract Feature

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Grid-based Hash

Add 𝝐

False

[Figure 70]

[Figure 71]

[Figure 72]

True

[Figure 73]

3D Model 𝜃

𝑔(𝒄;𝜃)

Camera Pose 𝒄 Timestep 𝑡

New Camera

Retrieve and Reuse

Used Camera

- Fig. 4: Overall pipeline of our Hash3D. Given the sampled camera and time-step, we retrieve the intermediate diffusion feature from hash table. If no matching found, it performs a standard inference and stores the new feature in the hash table; otherwise, if a feature from a close-up view already exists, it is reused without re-calculation.

context, each hash value idx is linked to a distinct queue, denoted as qidx. To ensure the queue reflects the most recent data and remains manageable in size, it is limited to a maximum length Q = 3. When this limit is reached, the oldest elements is removed to accommodate the new entry, ensuring the queue stays relevant to the evolving 3D representation.

Feature Retrieval and Update. Once the hash value idx is determined, we either retrieve existing data from the hash table or update it with new features. We set a hash probability 0 < η < 1 to make sure the balanced behavior between retrieval and update. In other words, with probability η, we retrieve the feature; otherwise, it performs hash table updates.

For feature updates, following prior work [29], we extract the feature vl(−U1), which is the input of last up-sampling layer in the U-net. Once extracted, we compute the hash code idx and append the data to the corresponding queue qidx. The stored data includes input noisy latent x, camera pose c, timestep t, and extracted diffusion features vl(−U1).

For feature retrieval, we aggregate data from qidx through weighted averaging. This method considers the distance of each noisy input xi from the current query point x. The weighted average v for a given index is calculated as follows:

|qidx|

e(−||x−xi||22)

Wivi, where Wi =

|qidx| i=1 e(−||x−xi||22) (6)

v =

i=1

Here, Wi is the weight assigned to vi based on its distance from the query point, and |qidx| is the current length of the queue. An empty queue |qidx| indicates unsuccessful retrieval, necessitating feature update.

#### 4.3 Adaptive Grid Hashing

In grid-based hashing, the selection of an appropriate grid size ∆θ,∆ϕ,∆ρ,∆t plays a pivotal role. As illustrated in Section 4.1, we see three insights related to grid size. First, feature similarity is only maintained at a median grid size; overly large grids tend to produce artifacts in generated views. Second, it is suggested that ideal grid size differs across various objects. Third, even for a single object,

optimal grid sizes vary for different views and time steps, indicating the necessity for adaptive grid sizing to ensure optimal hashing performance.

Learning to Adjust the Grid Size. To address these challenges, we propose to dynamically adjusting grid sizes. The objective is to maximize the average cosine similarity cos(·,·) among features within each grid. In other words, only if the feature is similar enough, we can reuse it. Such problem is formulated as

|qidx|

1 |qidx|

cos(vj, vi), s.t.|qidx| > 0 [Non-empty] (7)

max

∆θ,∆ϕ,∆ρ,∆t

i,j

Given our hashing function is non-differentiale, we employ a brute-force approach. Namely, we evaluate M predetermined potential grid sizes, each corresponding to a distinct hash table, and only use best one.

For each input [θ,ϕ,ρ,t], we calculate the hash code {idx(m)}Mm=1 for M times, and indexing in each bucket. Feature vectors are updated accordingly, with new elements being appended to their respective bucket. We calculate the cosine similarity between the new and existing elements in the bucket, maintaining a running average sidx(n) of these similarities

|qidx(m)|

1 |qidx(m)|

cos(vnew, vi) (8)

sidx(m) ← γsidx(m) + (1 − γ)

i=1

During retrieval, we hash across all M grid sizes but only consider the grid with the highest average similarity for feature extraction.

Computational and Memory Efficiency. Despite employing a brute-force approach that involves hashing M times for each input, our method maintains computational efficiency due to the low cost of hashing. It also maintains memory efficiency, as hash tables store only references to data. To prioritize speed, we deliberately avoid using neural networks for hashing function learning.

### 5 Experiment

In this section, we assess the effectiveness of our HS by integrating it with various

- 3D generative models, encompassing both image-to-3D and text-to-3D tasks.

#### 5.1 Experimental Setup

Baselines. To validate the versatility of our method, we conducted extensive tests across a wide range of baseline text-to-3D and image-to-3D methods.

- – Image-to-3D. Our approach builds upon techniques such as Zero-123+SDS [26], DreamGaussian [49] and Magic123 [37]. For Zero-123+SDS, we have incorporated Instant-NGP [32] and Gaussian Splatting [13] as its representation. We call these two variants Zero-123 (NeRF) and Zero-123 (GS).
- – Text-to-3D. Our tests also covered a range of methods, such as Dreamfusion [35], Fantasia3D [5], Latent-NeRF [30], Magic3D [21], and GaussianDreamer [58].

For DreamGaussian and GaussianDreamer, we implement Hash3D on top of the official code. And for other methods, we use the reproduction from threestudio1. Implementation Details. We stick to the same hyper-parameter setup within their original implementations of these methods. For text-to-3D, we use the stable-diffusion-2-12 as our 2D diffusion model. For image-to-3D, we employ the stable-zero1233. We use a default hash probability setting of η = 0.1. We use M = 3 sets of grid sizes, with ∆θ,∆ϕ,∆t ∈ {10,20,30} and ∆ρ ∈ {0.1,0.15,0.2}. We verify this hyper-parameter setup in the ablation study.

Dataset and Evaluation Metrics. To assess our method, we focus on evaluating the computational cost and visual quality achieved by implementing Hash3D.

- – Image-to-3D. For the image-to-3D experiments, we leverage the Google Scanned Objects (GSO) dataset [8] for evaluation [24, 25]. We focused on evaluating novel view synthesis (NVS) performance using established metrics such as PSNR, SSIM [53], and LPIPS [59]. We selected 30 objects from the dataset. For each object, we generated a 256×256 input image for 3D reconstruction. We then rendered 16 different views at a 30-degree elevation, varying azimuth angles, to compare the reconstructed models with their ground-truth. To ensure semantic consistency, we also calculated CLIPsimilarity scores between the rendered views and the original input images.
- – Text-to-3D. We generated 3D models from 50 different prompts, selected based on a prior study. To evaluate our methods, we focused on two primary metrics: mean±std CLIP-similarity [23, 36, 38] and the average generation time for each method. For assessing CLIP-similarity, we calculated the similarity between the input text prompt and 8 uniformly rendered views at elevation ϕ = 0◦ and azimuth θ = [0◦,45◦,90◦,135◦,180◦,225◦,270◦,315◦]. Additionally, we recorded and reported the generation time for each run.
- – User Study. To evaluate the visual quality of generated 3D objects, we carried out a study involving 44 participants. They were shown 12 videos of 3D renderings, created using two methods: Zero-123 (NeRF) for imagesto-3D, and Gaussian-Dreamer for text-to-3D. These renderings were made both with and without Hash3D. Participants were asked to rate the visual quality of each pair of renderings, distributing a total of 100 points between the two in each pair to indicate their perceived quality difference.
- – Computational Cost. We report the running time for each experiment using a single RTX A5000. Besides, we report MACs in the tables. Given that feature retrieval is stochastic — implying that retrieval of features is not guaranteed with attempt in empty bucket — we provide the theoretical average MACs across all steps, pretending that all retrieval succeeded.

#### 5.2 3D Generation Results

Image-to-3D Qualitative Results. Figure 5 demonstrates the outcomes of incorporating Hash3D into the Zero-123 framework to generate 3D objects. This

- 1 https://github.com/threestudio-project/threestudio
- 2 https://huggingface.co/stabilityai/stable-diffusion-2-1
- 3 https://huggingface.co/stabilityai/stable-zero123

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

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Zero-123 + Hash3D (6 min) Zero-123 (20 min)

- Fig. 5: Qualitative Results using Hash3D along with Zero123 for image-to-3D generation. We mark the visual dissimilarity in yellow.

integration not only preserves visual quality and consistency across views but also markedly decreases the processing time. In specific instances, Hash3D outperforms the baseline, as evidenced by the enhanced clarity of the dragon wings’ boundaries in row 1 and the more distinct taillights of the train in row 4. A similar level of visual fidelity is observed in Figure 1, where Hash3D is applied in conjunction with DreamGaussian, indicating that the integration effectively maintains quality while improving efficiency.

Image-to-3D Quantitative Results. For a detailed numerical analysis, refer to Table 1, which outlines the novel view synthesis performance, CLIP scores, running times for on top of all 4 baseline methods. Notably, For DreamGaussian and Zero-123(NeRF), we speed up the running time by 4× and 3× respectively. This reduction in running times is mainly due to the efficient feature retrieval and reuse mechanism employed by Hash3D. Additionally, our approach not only speeds up the process but also slightly improves performance. We believe this enhancement stems from the sharing of common features across different camera views, which reduces the inconsistencies found in independently sampled noise predictions, resulting in the smoother generation of 3D models.

Text-to-3D Qualitative Results. In Figure 6, we present the results generated by our method, comparing Hash3D with DreamFusion [35], SDS+GS, and Fantasia3D [5]. The comparison demonstrates that Hash3D maintains comparable visual quality to these established methods.

Text-to-3D Quantitative Results. Table 2 provides a detailed quantitative evaluation of Hash3D. Across various methods, Hash3D markedly decreases processing times, showcasing its adaptability in speeding up 3D generation. Significantly, this reduction in time comes with minimal impact on the CLIP score, effectively maintaining visual quality. Notably, with certain methods such as

- Table 1: Speed and performance comparison when integrated image-to-3D models with Hash3D. We report the original running time in their paper.

Method Time↓ Speed↑ MACs↓ PSNR↑ SSIM↑ LPIPS↓ CLIP-G/14↑

DreamGaussian 2m - 168.78G 16.202±2.501 0.772±0.102 0.225±0.111 0.693±0.105 + Hash3D 30s 4.0× 154.76G 16.356±2.533 0.776±0.103 0.223±0.113 0.694±0.104

Zero-123(NeRF) 20m - 168.78G 17.773±3.074 0.787±0.101 0.198±0.097 0.662±0.0107 + Hash3D 7m 3.3× 154.76G 17.961±3.034 0.789±0.095 0.196±0.0971 0.665±0.104

Zero-123(GS) 6m - 168.78G 18.409±2.615 0.789±0.100 0.204±0.101 0.643±0.105 + Hash3D 3m 2.0× 154.76G 18.616±2.898 0.793±0.099 0.204±0.099 0.632±0.106

Magic123 120m - 847.38G 18.718±2.446 0.803±0.093 0.169±0.092 0.718±0.099 + Hash3D 90m 1.3× 776.97G 18.631±2.726 0.803±0.091 0.174±0.093 0.715±0.107

- Table 2: Speed and performance comparison between various text-to-3D baseline when integrated with Hash3D.

Method Time↓ Speed↑ MACs↓ CLIP-G/14↑ CLIP-L/14↑ CLIP-B/32↑

|Dreamfusion<br><br>+ Hash3D|1h 00m 40m<br><br>|1.5×<br><br>|678.60G 622.21G<br><br>|0.407± 0.088 0.411±0.070<br><br>|0.267±0.058 0.266± 0.050<br><br>|0.314 ±0.049 0.312±0.044|
|---|---|---|---|---|---|---|
|Latent-NeRF<br><br>+ Hash3D|30m 17m<br><br>|1.8×<br><br>|678.60G 622.21G|0.406±0.033 0.406±0.038<br><br>|0.254±0.039 0.258±0.045|0.306±0.037 0.305±0.038<br><br>|
|SDS+GS<br><br>+ Hash3D|1h 18m 40m<br><br>|1.9×<br><br>|678.60G 622.21G|0.413±0.048 0.402±0.062<br><br>|0.263±0.034 0.252±0.041<br><br>|0.313±0.036 0.306±0.036|
|Magic3D<br><br>+ Hash3D|1h 30m 1h<br><br>|1.5×<br><br>|678.60G 622.21G|0.399±0.012 0.393±0.011<br><br>|0.257±0.064 0.250±0.054|0.303±0.059<br><br>0.304±0.052<br>|

GaussianDreamer 15m - 678.60G 0.412±0.049 0.267±0.035 0.312±0.038 + Hash3D 10m 1.5× 622.21G 0.416±0.057 0.271±0.036 0.312±0.037

GaussianDreamer, Hash3D goes beyond maintaining quality; it subtly improves visual fidelity. This improvement suggests that Hash3D’s approach, which considers the relationship between nearby camera views, has the potential to enhance existing text-to-3D generation processes.

User preference study. As shown in Figure 7, Hash3D received an average preference score of 52.33/100 and 56.29/100 when compared to Zero-123 (NeRF) and Gaussian-Dreamer. These scores are consistent with previous results, indicating that Hash3D slightly enhances the visual quality of the generated objects.

Gaussian-Dreamer (+ Hash3D)

Gaussian-Dreamer

Zero-123 (+ Hash3D)

Zero-123

0 20 40 60 80 100

Preference [%]

Fig. 7: User preference study for Hash3D.

- 5.3 Ablation Study and Analysis In this section, we study several key components in our Hash3D framework.

- Ablation 1: Hashing Features vs. Hashing Noise. Our Hash3D involves hashing intermediate features in the diffusion U-Net. Alternatively, we explored hashing the predicted noise estimation directly, leading to the development of a variant named Hash3D with noise (Hash3D w/n). This variant hashes and reuses

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

DreamFusion 1 h

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

###### + Hash3D 40 min (1.5×)

a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes

A oil and small monster that is playing with guitar

a delicious hamburger an astronaut riding a horse

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

SDS + 3DGS 1.3 h

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

###### + Hash3D 40 min (1.9×)

a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes

A oil and small monster that is playing with guitar

a delicious hamburger an astronaut riding a horse

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

Fantasia3d 2 h

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

+ Hash3D 1.2 h (1.7×)

a DSLR photo of an ice cream sundae

a teddy bear with christmas hat and leather boot

batman The leaning tower of Pisa

- Fig. 6: Visual comparison for text-to-3D task, when applying Hash3D to DreamFusion [35], SDS+GS and Fantasia3D [5].

the predicted score function directly. We applied this approach to the image-to3D task using Zero123, and the results are detailed in Table 9. Interestingly, while Hash3D w/n demonstrates a reduction in processing time, it yields considerably poorer results in terms of CLIP scores. This outcome underscores the effectiveness of our initial choice to hash features rather than noise predictions.

- Ablation 2: Influence of Hash Probability η. A crucial factor in our Hash3D is the feature retrieval probability η. To understand its impact, we conducted an ablation experiment with Dreamfusion, testing various η values {0.01,0.05,0.1,0.3,0.5,0.7}.

The relationship between CLIP score, time, and different η values is depicted in Figure 8. We observed that running time steadily decrease across all values. Interestingly, with smaller η values (less than 0.3), Hash3D even improved the visual quality of the generated 3D models. We speculate this improvement results from the enhanced smoothness in predicted noises across different views, attributable to feature sharing via a grid-based hash table. However, when η > 0.3, there was negligible impact on running time reduction. Figure 10 showcases the same trend in terms of visual quality. A moderately small η effectively balances

60

| |
|---|

0.42

CLIP-G/14Score

0.41

55

Time(min)

0.40

50

0.39

| |
|---|

45

0.38

40

0.37

| |
|---|

| |
|---|

0.36

| |
|---|

35

| |
|---|

| |
|---|

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7

Hash Probility

Fig. 8: Ablation study with different hash probability η.

Method Time CLIP-G/14

Zero-123 (NeRF) + Hash3D w/n 6 min 0.631±0.090 Zero-123 (NeRF) + Hash3D 7 min 0.665±0.104

Zero-123 (GS) + Hash3D w/n 3 min 0.622±0.083 Zero-123 (GS) + Hash3D 3 min 0.632±1.06

Fig. 9: Comparison between Hashing Features vs. Hashing Noise, applied to Zero-123.

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

𝜂 = 0 (No Hash) 𝜂 = 0.01 𝜂 = 0.1 𝜂 = 0.3 𝜂 = 0.5 𝜂 = 0.7

Fig. 10: Quantitative ablation study with different hash probability η

performance and efficiency. Consequently, we opted for η = 0.1 for the experiments presented in our main paper.

- Ablation 3: Adaptive Grid Size. In this study, we introduce a dynamic adjustment of the grid size for hashing, tailored to each individual sample. This adaptive approach, termed AdaptGrid, is evaluated against a baseline method that employs a constant grid size, within the context of Dreamfusion. As illustrated in Table 3, the AdaptGrid strategy surpasses the performance of the constant grid size method. Larger grid sizes tend to compromise the visual quality of generated 3D objects. Conversely, while smaller grid sizes preserve performance to a greater extent, they significantly reduce the likelihood of matching nearby features, resulting in increased computation time.

∆θ, ∆ϕ, ∆ρ, ∆t (10, 10, 0.1, 10) (20, 20, 0.15, 20) (30, 30, 0.2, 30) AdaptGrid (Ours)

CLIP-G/14↑ 0.408±0.033 0.345±0.055 0.287±0.078 0.411±0.070 Time↓ 48m 38m 32m 40m

Table 3: Ablation study on the Adaptive v.s. Constant Grid Size.

### 6 Conclusion

In this paper, we present Hash3D, a training-free technique that improves the efficiency of diffusion-based 3D generative modeling. Hash3D utilizes adaptive grid-based hashing to efficiently retrieve and reuse features from adjacent camera poses, to minimize redundant computations. As a result, Hash3D not only speeds up 3D model generation by 1.3 ∼ 4× without the need for additional training, but it also improves the smoothness and consistency of the generated 3D models.

### References

- 1. Achlioptas, P., Diamanti, O., Mitliagkas, I., Guibas, L.: Learning representations and generative models for 3d point clouds. In: International conference on machine learning. pp. 40–49. PMLR (2018)
- 2. Armandpour, M., Zheng, H., Sadeghian, A., Sadeghian, A., Zhou, M.: Re-imagine the negative prompt algorithm: Transform 2d diffusion into 3d, alleviate janus problem and beyond. arXiv preprint arXiv:2304.04968 (2023)
- 3. Bao, F., Li, C., Zhu, J., Zhang, B.: Analytic-DPM: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. In: International Conference on Learning Representations (2022), https://openreview.net/forum?id= 0xiJLKH-ufZ
- 4. Cao, Z., Long, M., Wang, J., Yu, P.S.: Hashnet: Deep learning to hash by continuation. In: Proceedings of the IEEE international conference on computer vision. pp. 5608–5617 (2017)
- 5. Chen, R., Chen, Y., Jiao, N., Jia, K.: Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (October 2023)
- 6. Chen, W., Wilson, J.T., Tyree, S., Weinberger, K.Q., Chen, Y.: Compressing neural networks with the hashing trick. In: Proceedings of the 32nd International Conference on International Conference on Machine Learning - Volume 37. p. 2285–2294. ICML’15, JMLR.org (2015)
- 7. Chen, Y., Li, Z., Liu, P.: Et3d: Efficient text-to-3d generation via multi-view distillation (2023)
- 8. Downs, L., Francis, A., Koenig, N., Kinman, B., Hickman, R., Reymann, K., McHugh, T.B., Vanhoucke, V.: Google scanned objects: A high-quality dataset of 3d scanned household items. In: 2022 International Conference on Robotics and Automation (ICRA). pp. 2553–2560. IEEE (2022)
- 9. Fang, G., Ma, X., Wang, X.: Structural pruning for diffusion models. In: Advances in Neural Information Processing Systems (2023)
- 10. Girish, S., Shrivastava, A., Gupta, K.: Shacira: Scalable hash-grid compression for implicit neural representations. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 17513–17524 (2023)
- 11. Jun, H., Nichol, A.: Shap-e: Generating conditional 3d implicit functions (2023)
- 12. Jun, H., Nichol, A.: Shap-e: Generating conditional 3d implicit functions. arXiv preprint arXiv:2305.02463 (2023)
- 13. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics 42(4) (July 2023), https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- 14. Kim, B.K., Song, H.K., Castells, T., Choi, S.: Bk-sdm: A lightweight, fast, and cheap version of stable diffusion. arXiv preprint arXiv:2305.15798 (2023), https: //arxiv.org/abs/2305.15798
- 15. Kitaev, N., Kaiser, L., Levskaya, A.: Reformer: The efficient transformer. In: International Conference on Learning Representations (2020), https://openreview. net/forum?id=rkgNKkHtvB
- 16. Kong, X., Liu, S., Lyu, X., Taher, M., Qi, X., Davison, A.J.: Eschernet: A generative model for scalable view synthesis. arXiv preprint arXiv:2402.03908 (2024)
- 17. Lai, H., Pan, Y., Liu, Y., Yan, S.: Simultaneous feature learning and hash coding with deep neural networks. In: 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 3270–3278. IEEE Computer Society, Los

- Alamitos, CA, USA (jun 2015). https://doi.org/10.1109/CVPR.2015.7298947, https://doi.ieeecomputersociety.org/10.1109/CVPR.2015.7298947
- 18. Li, J., Tan, H., Zhang, K., Xu, Z., Luan, F., Xu, Y., Hong, Y., Sunkavalli, K., Shakhnarovich, G., Bi, S.: Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. https://arxiv.org/abs/2311.06214 (2023)
- 19. Li, Q., Sun, Z., He, R., Tan, T.: Deep supervised discrete hashing. Advances in neural information processing systems 30 (2017)
- 20. Li, S., Hu, T., Khan, F.S., Li, L., Yang, S., Wang, Y., Cheng, M.M., Yang, J.: Faster diffusion: Rethinking the role of unet encoder in diffusion models (2023)
- 21. Lin, C.H., Gao, J., Tang, L., Takikawa, T., Zeng, X., Huang, X., Kreis, K., Fidler, S., Liu, M.Y., Lin, T.Y.: Magic3d: High-resolution text-to-3d content creation. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2023)
- 22. Liu, L., Ren, Y., Lin, Z., Zhao, Z.: Pseudo numerical methods for diffusion models on manifolds. In: International Conference on Learning Representations (2022), https://openreview.net/forum?id=PlKWVd2yBkY
- 23. Liu, M., Xu, C., Jin, H., Chen, L., T, M.V., Xu, Z., Su, H.: One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization (2023), https://openreview.net/forum?id=A6X9y8n4sT
- 24. Liu, M., Xu, C., Jin, H., Chen, L., Varma T, M., Xu, Z., Su, H.: One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. Advances in Neural Information Processing Systems 36 (2024)
- 25. Liu, R., Wu, R., Van Hoorick, B., Tokmakov, P., Zakharov, S., Vondrick, C.: Zero1-to-3: Zero-shot one image to 3d object. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9298–9309 (2023)
- 26. Liu, R., Wu, R., Van Hoorick, B., Tokmakov, P., Zakharov, S., Vondrick, C.: Zero1-to-3: Zero-shot one image to 3d object. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 9298–9309 (October 2023)
- 27. Liu, Y., Lin, C., Zeng, Z., Long, X., Liu, L., Komura, T., Wang, W.: Syncdreamer: Generating multiview-consistent images from a single-view image. In: The Twelfth International Conference on Learning Representations (2024), https: //openreview.net/forum?id=MN3yH2ovHb
- 28. Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Zhu, J.: Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems 35, 5775–5787 (2022)
- 29. Ma, X., Fang, G., Wang, X.: Deepcache: Accelerating diffusion models for free. arXiv preprint arXiv:2312.00858 (2023)
- 30. Metzer, G., Richardson, E., Patashnik, O., Giryes, R., Cohen-Or, D.: Latent-nerf for shape-guided generation of 3d shapes and textures. In: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 12663–12673. IEEE Computer Society, Los Alamitos, CA, USA (jun 2023). https://doi.org/ 10.1109/CVPR52729.2023.01218, https://doi.ieeecomputersociety.org/10. 1109/CVPR52729.2023.01218
- 31. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (ToG) 41(4), 1– 15 (2022)
- 32. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph. 41(4), 102:1–102:15 (Jul 2022). https://doi.org/10.1145/3528223.3530127, https://doi.org/10. 1145/3528223.3530127

- 33. Nichol, A., Jun, H., Dhariwal, P., Mishkin, P., Chen, M.: Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751

(2022)

- 34. Nießner, M., Zollhöfer, M., Izadi, S., Stamminger, M.: Real-time 3d reconstruction at scale using voxel hashing. ACM Trans. Graph. 32(6) (nov 2013). https://doi. org/10.1145/2508363.2508374, https://doi.org/10.1145/2508363.2508374
- 35. Poole, B., Jain, A., Barron, J.T., Mildenhall, B.: Dreamfusion: Text-to-3d using 2d diffusion. In: The Eleventh International Conference on Learning Representations

(2023), https://openreview.net/forum?id=FjNys5c7VyY

- 36. Qian, G., Mai, J., Hamdi, A., Ren, J., Siarohin, A., Li, B., Lee, H.Y., Skorokhodov, I., Wonka, P., Tulyakov, S., Ghanem, B.: Magic123: One image to highquality 3d object generation using both 2d and 3d diffusion priors. arXiv preprint arXiv:2306.17843 (2023)
- 37. Qian, G., Mai, J., Hamdi, A., Ren, J., Siarohin, A., Li, B., Lee, H.Y., Skorokhodov, I., Wonka, P., Tulyakov, S., Ghanem, B.: Magic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors. In: The Twelfth International Conference on Learning Representations (ICLR) (2024), https://openreview.net/forum?id=0jHkUDyEO9
- 38. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: Meila, M., Zhang, T. (eds.) Proceedings of the 38th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 139, pp. 8748–8763. PMLR (18–24 Jul 2021), https://proceedings.mlr.press/v139/radford21a.html
- 39. Ren, J., Pan, L., Tang, J., Zhang, C., Cao, A., Zeng, G., Liu, Z.: Dreamgaussian4d: Generative 4d gaussian splatting. arXiv preprint arXiv:2312.17142 (2023)
- 40. Roller, S., Sukhbaatar, S., Weston, J., et al.: Hash layers for large sparse models. Advances in Neural Information Processing Systems 34, 17555–17566 (2021)
- 41. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2022), https: //github.com/CompVis/latent-diffusionhttps://arxiv.org/abs/2112.10752
- 42. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. pp. 234–241. Springer (2015)
- 43. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic textto-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems 35, 36479–36494 (2022)
- 44. Salimans, T., Ho, J.: Progressive distillation for fast sampling of diffusion models. In: International Conference on Learning Representations (2022), https: //openreview.net/forum?id=TIdIXIpzhoI
- 45. Shue, J.R., Chan, E.R., Po, R., Ankner, Z., Wu, J., Wetzstein, G.: 3d neural field generation using triplane diffusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20875–20886 (2023)
- 46. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. In: International Conference on Learning Representations (2021), https://openreview.net/forum? id=St1giarCHLP
- 47. Song, Y., Dhariwal, P., Chen, M., Sutskever, I.: Consistency models. arXiv preprint arXiv:2303.01469 (2023)

- 48. Song, Y., Ermon, S.: Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems 32 (2019)
- 49. Tang, J., Ren, J., Zhou, H., Liu, Z., Zeng, G.: Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653 (2023)
- 50. Teschner, M., Heidelberger, B., Müller, M., Pomerantes, D., Gross, M.H.: Optimized spatial hashing for collision detection of deformable objects. In: Vmv. vol. 3, pp. 47–54 (2003)
- 51. Wang, H., Du, X., Li, J., Yeh, R.A., Shakhnarovich, G.: Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12619– 12629 (2023)
- 52. Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., Zhu, J.: Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems 36 (2024)
- 53. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13(4), 600–612 (2004)
- 54. Wu, J., Zhang, C., Xue, T., Freeman, B., Tenenbaum, J.: Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling. Advances in neural information processing systems 29 (2016)
- 55. Xie, X., Gherardi, R., Pan, Z., Huang, S.: Hollownerf: Pruning hashgrid-based nerfs with trainable collision mitigation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3480–3490 (2023)
- 56. Xu, Y., Tan, H., Luan, F., Bi, S., Wang, P., Li, J., Shi, Z., Sunkavalli, K., Wetzstein, G., Xu, Z., Zhang, K.: DMV3d: Denoising multi-view diffusion using 3d large reconstruction model. In: The Twelfth International Conference on Learning Representations (2024), https://openreview.net/forum?id=H4yQefeXhp
- 57. Yang, X., Zhou, D., Feng, J., Wang, X.: Diffusion probabilistic model made slim. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22552–22562 (2023)
- 58. Yi, T., Fang, J., Wang, J., Wu, G., Xie, L., Zhang, X., Liu, W., Tian, Q., Wang, X.: Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models. arXiv preprint arXiv:2310.08529 (2023)
- 59. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586–595 (2018)
- 60. Zhu, H., Long, M., Wang, J., Cao, Y.: Deep hashing network for efficient similarity retrieval. In: Proceedings of the AAAI conference on Artificial Intelligence. vol. 30

(2016)

