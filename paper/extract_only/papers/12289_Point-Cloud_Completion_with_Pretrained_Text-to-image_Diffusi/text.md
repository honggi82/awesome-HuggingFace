# arXiv:2306.10533v1[cs.CV]18Jun2023

## Point-Cloud Completion with Pretrained Text-to-image Diffusion Models

#### Yoni Kasten1 Ohad Rahamim2 Gal Chechik1,2

1NVIDIA Research 2Bar-Ilan University

[Figure 1]

[Figure 2]

[Figure 3]

SDS Complete

[Figure 4]

SDS Complete

+ “A Chair”

+ “A teapot”

Figure 1: We present SDS-Complete: A test-time optimization method for completing point clouds, captured by depth sensors, into complete surface representation using pre-trained text-to-image diffusion model. Our method takes two inputs: an incomplete point cloud (blue) and a textual description of the object ("A chair"). It outputs a complete surface (gray) that is consistent with the input points (blue).

### Abstract

Point-cloud data collected in real-world applications are often incomplete. Data is typically missing due to objects being observed from partial viewpoints, which only capture a specific perspective or angle. Additionally, data can be incomplete due to occlusion and low-resolution sampling. Existing completion approaches rely on datasets of predefined objects to guide the completion of noisy and incomplete, point clouds. However, these approaches perform poorly when tested on Out-OfDistribution (OOD) objects, that are poorly represented in the training dataset. Here we leverage recent advances in text-guided image generation, which lead to major breakthroughs in text-guided shape generation. We describe an approach called SDS-Complete that uses a pre-trained text-to-image diffusion model and leverages the text semantics of a given incomplete point cloud of an object, to obtain a complete surface representation. SDS-Complete can complete a variety of objects using test-time optimization without expensive collection of 3D information. We evaluate SDS-Complete on incomplete scanned objects, captured by real-world depth sensors and LiDAR scanners. We find that it effectively reconstructs objects that are absent from common datasets, reducing Chamfer loss by 50% on average compared with current methods. Project page: https://sds-complete.github.io/

Preprint. Under review.

### 1 Introduction

Depth cameras and LiDAR scanners enable us to capture the 3D geometrical structure of various objects in space. However, when used in the real world, various factors may significantly limit how well we can capture and reconstruct the full 3D geometry of objects from data alone. Specifically, factors like self-occlusions, partial camera viewpoints, or limitations in sensor resolution may cause the scanner to capture incomplete or partially sampled 3D objects. To fully understand the threedimensional world, one must address partial data and missing object parts.

Current approaches for point cloud completion demonstrate impressive results in handling in-domain shapes. However, due to the absence of large-scale datasets with a wide variety of shapes, these methods often face difficulties in dealing with shape classes that are outside their domain. The need for extensive 3D data collection poses a significant challenge in developing a model capable of effectively completing diverse object classes encountered in real-world scenarios that require depth perception, such as indoor scene reconstruction or autonomous driving setups.

Surface completion has been extensively explored [6]. Broadly speaking, some approaches focus on training models for specific object classes [53, 45, 26, 33] and some train class-agnostic models [47, 51, 44]. In general, existing methods achieve impressive results when tested on objects from the distribution they are trained on (in-domain). Unfortunately, their performance deteriorates dramatically for out-of-domain objects, namely objects and classes that were not present in the training distribution.

In this work, we address this challenge of OOD objects by leveraging a pretrained text-to-image diffusion model. It has been shown that these models, even though never trained on 3D data, can be used for text-guided 3D shape generation [34]. This is done through the SDS loss, which measures the agreement of the 3D shape’s rendered images with the model prior. Our key idea is that since text-to-image diffusion models were trained on a vast number of diverse objects, they contain a strong prior about the shape and texture of objects, and that prior can be used for completing missing parts. For example, given a partial point cloud, knowing that it corresponds to a chair can guide the completion process, because objects from this class are expected to exhibit some types of symmetries and parts.

The key challenge in this approach is to combine the prior information from the diffusion model with the observed partial point cloud, to generate a complete shape that is faithful to the partial observations. We introduce SDS-Complete: a point cloud completion method that uses the SDS-loss [34] to accurately complete object surfaces (Fig. 1, right) while being guided by input constraints of text and point clouds (Fig. 1, left). To be consistent with the input points, we use a Signed Distance Function (SDF) surface representation [32, 15, 3, 50], and constrain the zero level set of the SDF to go through the input points. SDS-Complete enables overcoming the limitations of working with OOD objects as it brings the semantics from a pretrained text-to-image diffusion model. That allows us to generate accurate and realistic 3D shapes from partial observations.

We demonstrate that SDS-Complete generates completions for various objects with different shape types from two real-world datasets: the Redwood dataset [10], which contains various incomplete real-world depth camera scans, and the KITTI dataset [5], which contains object LiDAR scans from driving scenarios. In both cases, we outperform state-of-the-art methods for OOD objects, while showing comparable results on object classes that were used to train these methods.

In summary, this paper makes the following contributions: (1) We formulate point cloud completion as a test-time optimization problem, avoiding collecting large datasets of 3D geometries for training. (2) We develop a new approach to combine the SDS loss [34], with an empirical point-cloud, by using an SDF surface representation. (3) We present a practical and unified approach for completing and preserving existing 3D content captured by different depth sensors (LiDAR or depth camera) while sampling realistic novel camera views for the SDS loss, that would complete the shape consistently. (4) We demonstrate state-of-the-art completion results for objects considered to be out-of-distribution for point cloud completion.

### 2 Related work

Surface Completion from Point Clouds. Over the last years, neural network-driven approaches [53, 45, 26, 33] have demonstrated remarkable capabilities in reconstructing objects from incomplete or partial inputs. Early attempts with neural networks [11, 12, 16, 41] utilized voxel grid representations of 3D geometries due to their straightforward processing with off-the-shelf 3D convolutional layers. While voxels proved to be useful, they suffer from a space complexity issue, as their representation grows cubically. Consequently, these methods can only generate shapes with limited resolution. In contrast, point cloud representations [13, 1] have been leveraged to model higher-resolution geometries using neural networks. Several methods [46, 52] use such techniques for predicting the completed point cloud given a partial input point cloud. However, to obtain a surface representation, a surface reconstruction technique [21] needs to be applied as a post-processing step, which can introduce additional errors. Recently, an alternative approach has emerged where the output surface is represented using neural representations [26, 32]. The advantage of these representations lies in their ability to represent the surface continuously without any discretization. [26, 33] trained deep neural networks and latent conditioned implicit neural representations on a dataset of predefined object classes [7], to perform point cloud completion. While most deep methods for surface completion train a different model per object class, very recent methods have focused on training multi-class models, allowing for better generalization [47, 51]. PoinTr [51] uses a transformer encoder-decoder architecture for translating a given input point cloud into a set of point proxies. These point proxies are then converted into completed point clouds using FoldingNet [49]. ShapeFormer [47] directly reconstructs surfaces from incomplete point clouds using a transformer.

Other recent works [9, 24, 30] show progress in the task of shape completion given a partial surface, where [30] uses a transformer and autoregressive modeling, and [9, 24] employ diffusion processes that allow controlling the completion with text. However, these methods require a surface as input and cannot handle incomplete point clouds. Furthermore, their applicability is limited to the domain they are trained on.

In contrast to the above-mentioned methods, our method performs point cloud completion as a test-time optimization process using pre-trained available diffusion models, and therefore, we do not rely on any collection of 3D shapes for training, and we work on much broader domains.

##### 3D models from text using 2D supervision. Several approaches used large vision-and-language models like CLIP [35] to analyze and synthesize 3D objects. Text2Mesh [28], CLIP-Mesh [22] and DreamFields [20] present approaches for editing meshes, generating 3D models, and synthesizing NeRFs [29] respectively, based on input text prompts. The methods employ differentiable renderers to generate images while maximizing their similarity with the input text in CLIP space.

Diffusion models have recently gained attention for their ability to generate high-quality images [48]. One application of interest is textual-guided image generation [38, 39], where these models generate images based on text prompts, enabling control over the generated visual content. DreamFusion [34] pioneered the use of text-to-image diffusion models as guidance for text-guided 3D object generation. Latent-NeRF [27] enables the training of DreamFusion with higher-resolution images by optimizing the NeRF with diffusion model features instead of RGB colors. TEXTure [37] and Text2Tex [8] use depth-aware text-to-image diffusion models to synthesize textures for meshes. Other recent works predict shapes directly from 2D images [36, 43, 25]. In contrast, our method uses the input text for completing partial point clouds, rather than editing or synthesizing 3D content.

### 3 Preliminaries

#### 3.1 Volume Rendering

Neural Radiance Field A neural radiance field [29] is a pair of two functions: σ : R3 → R+ and c : (R3,S2) → R3, each represented by a Multilayer Perceptron (MLP). The function σ maps a 3D point x ∈ R3 into a density value, and the function c maps a 3D point x and a view direction v ∈ S2 into an RGB color. A neural radiance field can represent the geometric and appearance properties of a 3D object and is used as a differentiable renderer of 2D images from the 3D scene. Let I be an image with a camera center t ∈ R3, the pixel coordinate u = (u,v)T ∈ R2 is backprojected into a

- 3D ray ru, starting at t and going through the pixel u with a direction v ∈ S2. Let µ1,µ2,...,µN

r

be sample distances from t on the ray ru, then the densities and colors of the radiance field are alpha

composited from the camera center through the ray. The RGB image color I(u,v) is calculated by:

I(u,v) =

Nr

wic(t + µiv,v) (1)

i=1

where wi = αi j<i(1 − αj) is the color contribution of the ith segment to the rendered pixel, and αi = 1 − exp(−σ(t + µiv)(µi+1 − µi)) is the opacity of segment i. Eq. (1) is differentiable with respect to the learned parameters of c and σ and therefore, is used to train the neural radiance field. Let I¯be the ground truth image, then the MSE loss is used to train the neural radiance field:

n

1 n

LMSE =

i=1

where n is the number of pixels in the batch.

I(ui) − I¯(ui) 2 (2)

Volume Rendering of Neural Implicit Surfaces While the neural radiance field shows impressive performances in synthesizing novel views, extracting object geometries from a trained radiance field is not trivial. Defining the surface by simply thresholding the density σ results in noisy and inaccurate geometry. We adopt the solution proposed by [50]. Let Ω ⊂ R3 be the space occupied by the object, and M denotes the boundary of the surface. Then the SDF f : R3 → R is defined by

f(x) = (−1)1

Ω(x) min

∥x − y∥ (3)

y∈M

1 x ∈ Ω 0 otherwise

. Given f, the surface M is defined by its zero level set, i.e.

where 1Ω(x) =

M = {x ∈ R3 : f(x) = 0} (4)

A signed distance function can be utilized for defining a neural radiance field density. Let x ∈ R3 and f : R3 → R be a 3D point and an SDF respectively, the density σ(x) is defined by:

σ(x) = αΨβ(−f(x)) (5)

where Ψβ(s) is the Cumulative Distribution Function (CDF) of the Laplace distribution with zero mean and β scale:

 

- 1

- 2exp β s s ≤ 0

(6)

Ψβ(s) =



1 − 21exp −βs s > 0

and α and β are parameters that can be learned during training (in our case, we set them to be constant, see details in the supplementary). It is then possible to train a neural radiance field, defined by the SDF f and the neural color function c, using the loss function defined by Eq. (2).

#### 3.2 Score Distillation Sampling (SDS)

Diffusion Models A diffusion model [31, 40, 38] generates image samples from a Gaussian noise image, by inverting the process of gradually adding noise to an image. This process is defined as follows: at time t = 1,...,T, a Gaussian noise ϵ ∼ N(0,I) is added to the image:

It = √α¯tI + √1 − α¯tϵ, where α¯t = ti=1 αi, αt = 1 − βt and βt ∈ (0,1) defines the amount of added noise. A denoising neural network ϵˆ = Φ(It;t) is trained to predict the added noise ϵˆgiven the noisy image It and the noise level t. The diffusion models are trained on large image collections C for minimizing the loss

Φ(√α¯tI + √1 − α¯tϵ;t) − ϵ 2 (7)

LD = E

I∈C

Given a pretrained Φ, an image sample is generated by sampling a Gaussian noise image IT ∼ N(0,I) and gradually denoising it using Φ.

Diffusion models can be extended to be conditioned on additional inputs. Text-to-image diffusion models [38] condition Φ on a textual prompt embedding input y, and train Φ(It;t,y). Therefore, they can generate images given text and sampled Gaussian noise.

[Figure 5]

“a sofa”

Neural Surface (Signed Distance Function) represented using 𝑓 . Optimize 𝜃,𝜑

Input Text Prompt y

Input Incomplete Point cloud P

[Figure 6]

|Sensor-compatibility loss ℒ 𝑃,𝑓|
|---|

𝑐 ,𝑓

[Figure 7]

[Figure 8]

[Figure 9]

| |
|---|

𝐼𝑚 ,…, 

[Figure 10]

𝐼𝑚

⋅⋅⋅

[Figure 11]

[Figure 12]

Stable Diffusion

SDS Loss ℒSDS (𝑦,𝐼𝑚 )

𝐼𝑚

𝐼𝑚

Radiance field rendering using (𝑐 ,𝑓 )

- Figure 2: The components of our SDS-Complete approach. Our method optimizes two neural functions: A signed distance function fθ representing the surface and a volumetric coloring function cφ (introduced in Sec. 3.1). Together, (cφ,fθ) define a radiance field, which is used to render novel image views Im0,...Imn. The SDS-Loss is applied to these renderings and encourages them to be compatible with the input text y (bottom left). To constrain the surface to lie on the input points, we encourage the signed distance function to be zero at the input points (Sensor compatibility loss).

DreamFusion[34] uses a pretrained, and fixed, text condition diffusion model Φ(It;t,y) and uses it to train a NeRF model from scratch, given a textural description embedding y0. In each iteration, a camera view is sampled and used to render an image I0 from the NeRF model. I0 is differentiable with respect to the learned parameters of the NeRF model (θNeRF), and used as an input to Φ(I0;t,y). The Score Distillation Sampling (SDS) loss is then applied:

∇θNeRFLSDS(I0) = Et,ϵ (w(t)Φ(√α¯tI0 + √1 − α¯tϵ;t,y0) − ϵ)∇θNeRF

I0 (8)

Note that ∇θNeRFLSDS is the gradient with respect to θNeRF of Eq. (7), where the Jacobian of Φ is omitted for stability and efficiency. Intuitively, if I0 looks like a natural image, and is compatible with y0, then the pretrained diffusion model predicts the added noise successfully, resulting in low values for LD. By updating the NeRF’s weights according to Eq. (8), LD is reduced, and as a result, the rendered images become more compatible with y0.

### 4 Our Method

Inputs and components of our system. The overall scheme for our method is depicted in Fig. 2. We address the problem of completing a surface given incomplete point cloud measurements captured by a depth sensor. These input measurements (Fig. 2 top-right), include a set of 3D input points P = {p1,p2,...,pN} and a text description embedding y of the incomplete object. We assume that P is captured by a depth sensor like a depth camera or a LiDAR sensor, and that the internal parameters of the sensor are known. We further assume that the point cloud is segmented out from the original scan, namely, that all the points in P belong to a single object that is described by y. A sensor ray i is associated with a binary value mask Mi ∈ {0,1}, indicating whether this ray intersects the surface at a point that belongs to P. The ray i is also associated with the ray’s distance from the sensor to the surface Di ∈ R if Mi = 1. Lastly, for our camera sampling process (see Sec.4.2), we assume that the original, non-segmented scan, contains points from the world’s ground plane, that are used to estimate the plane’s parameters l ∈ P3 [17].

#### 4.1 Loss Terms

Our method optimizes for the complete object surface represented by a neural signed distance function fθ : R3 → R, (see Eq. 3), and a neural color function cφ : R3 → R3, where θ and φ represent the learned parameters of the neural functions. As described in Section 3.1, these two functions form a neural radiance field and can be optimized using the rendered images of the 3D volumetric functions. In contrast to [34], the object surface is defined directly by fθ, as its zero level set (Eq. (4)). To constrain the surface to go through the input points we encourage the signed distance to be zero at these points (Fig. 2, middle-right), using the following point-cloud loss:

N

1 N

|fθ (pi)|. (9)

Lp =

i=1

At each iteration, we render the radiance field from the sensor perspective. Each rendered pixel i is associated with its expected rendered opacity and distance from the surface, denoted by M˜i and D˜i respectively. We use the input opacities and distances to constrain the optimized surface to match the mask and depth sensor observations:

K

K

2

1 K

1 K

Di − D˜i

|Mi − M˜i| , Ld =

, (10)

Lm =

i=1

i=1

where K is the number of sensor rays. To constrain fθ to form a valid SDF, we apply the Eikonal loss regularization introduced in [15]:

1 |Peik| p∈P

|∥∇fθ (pi)∥ − 1| , (11)

Leikonal =

eik

where Peik contains both P and uniformly sampled points from the region of interest.

While Lm,Ld,Lp and Leikonal, constrain the optimized surface to match the information that is captured by the sensor, the losses do not provide any signal for the occluded missing content that cannot be captured by the depth sensor. A semantic prior is required in order to complete the unobserved part of the surface. For that, we utilize the input text embedding y and a pretrained text-to-image diffusion model Φ. Our goal is to use Φ to supply the semantic prior for the unobserved parts, such that any rendered image of the object would be compatible with y. To this end, we render random object views using our radiance field and apply the SDS loss (Eq. (8)) with the input text embedding y to optimize fθ and cφ (Fig. 2, bottom-right).

Finally, we use the known world plane to further regularize the surface from drifting below the ground:

max(−fθ(p),0), (12)

Lplane =

p∈Puniform

where Puniform is a set of uniformly sampled 3D points below the plane in the region of interest. Our total loss is:

Ltotal = δmLm + δdLd + δpLp + δeikonalLeikonal + δplaneLplane + LSDS, (13)

where δm,δd,δp,δeikonal and δplane are the coefficients that define the weights of the different loss terms relative to the SDS loss.

#### 4.2 Camera handling

To keep the generated content consistent with the existing partially observed object, careful handling of camera sampling is needed. In contrast to [34] where the SDS loss is used to generate a 3D object “from scratch", in our case sampling camera positions uniformly at random, results in inferior results (see ablation study in Fig. 6).

Instead, we developed a “curriculum" for sampling camera poses. Let C0 = (R0,t0) be the original camera-to-world pose of the depth sensor. To preserve the roll angle of C0 with respect to the object and prevent rendering flipped or unrealistically rotated images, we define the azimuth and elevation deviation from C0 with respect to the segmented world plane. Specifically, let nl ∈ S2 be the normal

#### GT Ours ShapeFormer Sinv cGAN PoinTr Input

[Figure 13]

[Figure 14]

[Figure 15]

- Figure 3: In-domain completion. Comparison with traditional shape completion methods on the Redwood dataset for in-domain objects. Red represents methods that output point-clouds. SDSComplete completes unseen parts with accuracy that is comparable to previous work.

to the plane l, we define the azimuth rotation update to be Razimuth = R(nl,γazimuth), where R(n,γ) is the Rodrigues’ rotation formula for a rotation around the unit vector n, with γ degrees. Similarly,

let a0 be the normalized principal axis direction of C0, we define the elevation rotation update by Relevation = R(nl×a0,γelevation). Assuming that the origin is located at the object’s center, an updated camera, Cupdate, for γazimuth and γelevation degrees, is given by:

Cupdate = (RazimuthRelevationR0,RazimuthRelevationt0). (14)

During training, we start by applying the SDS loss on the rendered image from C0 pose, and then we gradually increase the sampling range of the deviation angles until the entire object is covered. By initially applying the SDS loss on images rendered from the depth sensor’s perspective, the colors of the observed part of the object are optimized first to be consistent with y, and then, when the sampling range increases, the rest of the object’s colors and geometry are completed accordingly.

### 5 Experiments

We conduct an evaluation of our model, on two real-world datasets that encompass a diverse array of general objects. Our primary objective in selecting these datasets is to demonstrate that our proposed method can handle diverse variations of object types that are not confined to any specific domain.

Datasets. We assessed the performance of our model by utilizing partial point clouds obtained from depth images and LiDAR scans. The Redwood dataset [10] comprises a diverse collection of object scans. On Tab. 1 we compare the Chamfer distance in mm, for objects with existing 360◦ scans that we refer to as a GT. We further tested our model on the KITTI LiDAR dataset [4, 14], which contains incomplete point clouds of objects in real-world scenes captured by LiDAR sensors. In contrast to the Redwood dataset, for KITTI, there is no ground-truth data available for quantitative evaluation, and therefore, we only compare the outputs qualitatively. We note that there exist large datasets for 3D car shapes e.g. ShapeNetCars [52] which were used for training the baseline methods and therefore we consider cars as in-domain objects. To evaluate our performance on OOD classes compared to the baseline methods, we further include other object classes such as trucks and motorcycles.

Baselines We conducted a comparative analysis of SDS-Complete in comparison to several other point-cloud completion approaches, including PoinTr [51] and ShapeFormer [47]. Both PoinTr and ShapeFormer are characterized by their adoption of multi-model training methodologies. Furthermore, we extended our analysis to include methods that specialize in specific object-class: cGAN [45] and shape-inversion [53] have per-class trained models for chairs and tables. During the inference stage,

[Figure 16]

- Figure 4: Out-of-domain comparisons on the Redwood dataset. A qualitative comparison between SDS-Complete to multi-class methods that aim to exhibit generalization capabilities. Notably, SDSComplete produces more accurate completions.

object alignment was executed by utilizing the world’s ground plane, denoted as l ∈ P3 [17], to employ the same alignment procedure that these methods applied during their training phase.

Results on Redwood We measure the Chamfer distances to quantify the dissimilarity between the generated completions and their corresponding ground-truth shapes. Our evaluation encompasses two distinct groups of objects: one contains objects from the same object classes of the baselines’ training data ([7]), whereas the other group contains OOD objects that lack comprehensive preexisting training data. The results are presented in Tab. 1 and demonstrate that SDS-Complete gets state-of-the-art results on OOD objects while remaining comparable to the baselines on in-domain objects. We further show qualitative results on in-domain and out-domain objects in Fig. 3 and 4 respectively. It can be observed that our method has the capability to maintain consistent performance across both in-distribution and OOD objects, while the completions generated by other methods for OOD objects exhibit unpredictability and deviate from the intended shapes, leading to inferior performance.

To demonstrate the importance of each component of our method we present an ablation study in Fig.

- 6. As can be seen, without the SDS-loss, our model has no understanding of object characteristics like the fact that the chair has four legs and a straight back-side. Without the SDF representation, it is not possible to apply the point cloud constraints directly on the surface which results in an inferior ability to follow the partial input. Lastly, it can be seen that our camera sampling “curriculum" improves the completion compared to a random camera sampling, by preserving the consistency of the generated content with the existing sensor measurements.

Results on KITTI We present qualitative comparisons in Fig. 5. Besides our method, we include ShapeFormer [47] and PoinTr [51], both of which were trained on ShapeNetCars [52]. Notably, SDS-Complete exhibited better completion results, particularly when confronted with objects with no trainable data. In the supplementary, we further present a user-study evaluation to compare the performance of the 3 methods.

Limitations The major factor limiting our method is the application of the SDS loss with low resolution images due to GPU memory limitation, which requires a lot of sampling views until the object is completed. Our SDF model is initialized to a sphere and therefore cannot handle well objects with components that have disc topology. In the supplementary, we show failure cases of our method and list additional implementation details.

Object Shape PoinTr cGAN Sinv SDS-Complete Former (ours)

old chair† 23.2 34.1 33.2 36.7 19.3 outside chair† 25.9 29.6 42.8 28.7 22.6 one lag table† 39.7 21.6 99.4 24.9 20.3 executive chair† 33.6 43.9 208 20.6 23.7

Average (in) 30.6 32.3 95.8 27.7 21.5

trash can 136.4 137 - - 36.4 plant in vase 60.8 41 - - 29.5 vespa 79.4 70.3 - - 57.6 tricycle 65.2 60.4 - - 39 couch 43.9 87.4 - - 36.5 office trash 68.8 49.7 - - 20.5

Average (out) 75.7 74.3 - - 36.6 Average 60.4 59 - - 30.5

Table 1: Chamfer loss (lower is better) for objects from the Redwood dataset. † represents in-domain objects. cGAN and Sinv models solely focused on chairs and tables. “Average" denotes the mean performance on all 10 objects, “Average (in)" refers to in-domain objects, and “Average (out)" refers to OOD objects.

[Figure 17]

- Figure 5: Completion results on the KITTI dataset. A qualitative comparison with previous methods that are trained on a dataset containing car shapes. Notably, SDS-Complete, produces results that better complete the shape. Other methods fail to produce meaningful shapes for OOD objects

[Figure 18]

Input points Random cameras

No SDS loss

No SDF representation

Full method

Average Chamfer ↓ 43.5 43.8 59.2 30.5

- Figure 6: An ablation study for demonstrating the contribution of each part of our method. Random cameras: running without our camera handling that is described in Sec. 4.2. No SDS loss: using all losses but the SDS loss. No SDF representation: running with a density function as in [29]. Below, we compare the average Chamfer distance over the evaluated 10 Redwood scans.

### 6 Conclusions

We presented SDS-Complete, a novel test time optimization approach for 3D completion that leverages a text-to-2D pre-trained model to enable the reconstruction of a wide variety of objects. To adapt the SDS-loss for the use of point clouds, we incorporated an SDF representation and constrained the surface to lie on the input points. We successfully applied the SDS-loss on images rendered from novel views and completed the missing part of the object by aligning the images with an input textual description. By handling the camera sampling carefully we maintained the consistency of the completed part with the input captured part. This enabled us to produce superior results even on previously unconsidered objects for completion. In the future, we would like to utilize advances in text-to-3D for achieving higher-quality completions.

### 7 Acknowledgments

We thank Lior Yariv, Dolev Ofri, Or Perel and Haggai Maron for their insightful comments. We thank Lior Bracha and Chen Tessler for helping with the user study. This work was funded by a grant to GC from the Israel Science Foundation (ISF 737/2018), and by an equipment grant to GC and Bar-Ilan University from the Israel Science Foundation (ISF 2332/18). OR is supported by a PhD fellowship from Bar-Ilan data science institute (BIU DSI).

### References

- [1] P. Achlioptas, O. Diamanti, I. Mitliagkas, and L. Guibas. Learning representations and generative models for 3d point clouds. In International conference on machine learning, pages 40–49. PMLR, 2018.
- [2] K. S. Arun, T. S. Huang, and S. D. Blostein. Least-squares fitting of two 3-d point sets. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-9:698–700, 1987.
- [3] M. Atzmon and Y. Lipman. Sal: Sign agnostic learning of shapes from raw data. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.
- [4] J. Behley, M. Garbade, A. Milioto, J. Quenzel, S. Behnke, J. Gall, and C. Stachniss. Towards 3D LiDARbased semantic scene understanding of 3D point cloud sequences: The SemanticKITTI Dataset. The International Journal on Robotics Research, 40(8-9):959–967, 2021.
- [5] J. Behley, M. Garbade, A. Milioto, J. Quenzel, S. Behnke, C. Stachniss, and J. Gall. Semantickitti: A dataset for semantic scene understanding of lidar sequences. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9297–9307, 2019.
- [6] M. Berger, A. Tagliasacchi, L. M. Seversky, P. Alliez, G. Guennebaud, J. A. Levine, A. Sharf, and C. T. Silva. A survey of surface reconstruction from point clouds. In Computer graphics forum, volume 36, pages 301–329. Wiley Online Library, 2017.
- [7] A. X. Chang, T. Funkhouser, L. Guibas, P. Hanrahan, Q. Huang, Z. Li, S. Savarese, M. Savva, S. Song, H. Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015.
- [8] D. Z. Chen, Y. Siddiqui, H.-Y. Lee, S. Tulyakov, and M. Nießner. Text2tex: Text-driven texture synthesis via diffusion models. arXiv preprint arXiv:2303.11396, 2023.
- [9] Y.-C. Cheng, H.-Y. Lee, S. Tulyakov, A. G. Schwing, and L.-Y. Gui. Sdfusion: Multimodal 3d shape completion, reconstruction, and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4456–4465, 2023.
- [10] S. Choi, Q.-Y. Zhou, S. Miller, and V. Koltun. A large dataset of object scans. arXiv preprint arXiv:1602.02481, 2016.
- [11] C. B. Choy, D. Xu, J. Gwak, K. Chen, and S. Savarese. 3d-r2n2: A unified approach for single and multiview 3d object reconstruction. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part VIII 14, pages 628–644. Springer, 2016.
- [12] A. Dai, C. Ruizhongtai Qi, and M. Nießner. Shape completion using 3d-encoder-predictor cnns and shape synthesis. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5868–5877, 2017.

- [13] H. Fan, H. Su, and L. J. Guibas. A point set generation network for 3d object reconstruction from a single image. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 605–613, 2017.
- [14] A. Geiger, P. Lenz, and R. Urtasun. Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite. In Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), pages 3354–3361, 2012.
- [15] A. Gropp, L. Yariv, N. Haim, M. Atzmon, and Y. Lipman. Implicit geometric regularization for learning shapes. In International Conference on Machine Learning, pages 3789–3799. PMLR, 2020.
- [16] C. Häne, S. Tulsiani, and J. Malik. Hierarchical surface prediction for 3d object reconstruction. In 2017 International Conference on 3D Vision (3DV), pages 412–420. IEEE, 2017.
- [17] R. Hartley and A. Zisserman. Multiple view geometry in computer vision. Cambridge university press, 2003.
- [18] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.
- [19] D. Hendrycks and K. Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.
- [20] A. Jain, B. Mildenhall, J. T. Barron, P. Abbeel, and B. Poole. Zero-shot text-guided object generation with dream fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 867–876, 2022.
- [21] M. Kazhdan and H. Hoppe. Screened poisson surface reconstruction. ACM Transactions on Graphics (ToG), 32(3):1–13, 2013.
- [22] N. M. Khalid, T. Xie, E. Belilovsky, and T. Popa. CLIP-mesh: Generating textured meshes from text using pretrained image-text models. In SIGGRAPH Asia 2022 Conference Papers. ACM, nov 2022.
- [23] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
- [24] M. Li, Y. Duan, J. Zhou, and J. Lu. Diffusion-sdf: Text-to-shape via voxelized diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12642–12651, 2023.
- [25] L. Melas-Kyriazi, I. Laina, C. Rupprecht, and A. Vedaldi. Realfusion: 360deg reconstruction of any object from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8446–8455, 2023.
- [26] L. Mescheder, M. Oechsle, M. Niemeyer, S. Nowozin, and A. Geiger. Occupancy networks: Learning 3d reconstruction in function space. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4460–4470, 2019.
- [27] G. Metzer, E. Richardson, O. Patashnik, R. Giryes, and D. Cohen-Or. Latent-nerf for shape-guided generation of 3d shapes and textures. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12663–12673, 2023.
- [28] O. Michel, R. Bar-On, R. Liu, S. Benaim, and R. Hanocka. Text2mesh: Text-driven neural stylization for meshes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13492–13502, 2022.
- [29] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16, pages 405–421. Springer, 2020.
- [30] P. Mittal, Y.-C. Cheng, M. Singh, and S. Tulsiani. Autosdf: Shape priors for 3d completion, reconstruction and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 306–315, 2022.
- [31] A. Q. Nichol and P. Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pages 8162–8171. PMLR, 2021.
- [32] J. J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.

- [33] S. Peng, M. Niemeyer, L. Mescheder, M. Pollefeys, and A. Geiger. Convolutional occupancy networks. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part III 16, pages 523–540. Springer, 2020.
- [34] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.
- [35] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [36] A. Raj, S. Kaza, B. Poole, M. Niemeyer, N. Ruiz, B. Mildenhall, S. Zada, K. Aberman, M. Rubinstein, J. Barron, et al. Dreambooth3d: Subject-driven text-to-3d generation. arXiv preprint arXiv:2303.13508, 2023.
- [37] E. Richardson, G. Metzer, Y. Alaluf, R. Giryes, and D. Cohen-Or. Texture: Text-guided texturing of 3d shapes. arXiv preprint arXiv:2302.01721, 2023.
- [38] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.
- [39] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans, J. Ho, D. J. Fleet, and M. Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 36479–36494. Curran Associates, Inc., 2022.
- [40] J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.
- [41] D. Stutz and A. Geiger. Learning 3d shape completion under weak supervision. International Journal of Computer Vision, 128(5):1162–1181, oct 2018.
- [42] J. Tang. Stable-dreamfusion: Text-to-3d with stable-diffusion, 2022. https://github.com/ashawkey/stabledreamfusion.
- [43] J. Tang, T. Wang, B. Zhang, T. Zhang, R. Yi, L. Ma, and D. Chen. Make-it-3d: High-fidelity 3d creation from a single image with diffusion prior. arXiv preprint arXiv:2303.14184, 2023.
- [44] F. Williams, Z. Gojcic, S. Khamis, D. Zorin, J. Bruna, S. Fidler, and O. Litany. Neural fields as learnable kernels for 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18500–18510, June 2022.
- [45] R. Wu, X. Chen, Y. Zhuang, and B. Chen. Multimodal shape completion via conditional generative adversarial networks. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16, pages 281–296. Springer, 2020.
- [46] P. Xiang, X. Wen, Y.-S. Liu, Y.-P. Cao, P. Wan, W. Zheng, and Z. Han. Snowflakenet: Point cloud completion by snowflake point deconvolution with skip-transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5499–5509, 2021.
- [47] X. Yan, L. Lin, N. J. Mitra, D. Lischinski, D. Cohen-Or, and H. Huang. Shapeformer: Transformer-based shape completion via sparse representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6239–6249, 2022.
- [48] L. Yang, Z. Zhang, Y. Song, S. Hong, R. Xu, Y. Zhao, Y. Shao, W. Zhang, B. Cui, and M.-H. Yang. Diffusion models: A comprehensive survey of methods and applications. arXiv preprint arXiv:2209.00796, 2022.
- [49] Y. Yang, C. Feng, Y. Shen, and D. Tian. Foldingnet: Interpretable unsupervised learning on 3d point clouds. arXiv preprint arXiv:1712.07262, 2(3):5, 2017.
- [50] L. Yariv, J. Gu, Y. Kasten, and Y. Lipman. Volume rendering of neural implicit surfaces. Advances in Neural Information Processing Systems, 34:4805–4815, 2021.
- [51] X. Yu, Y. Rao, Z. Wang, Z. Liu, J. Lu, and J. Zhou. Pointr: Diverse point cloud completion with geometryaware transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12498–12507, 2021.

- [52] W. Yuan, T. Khot, D. Held, C. Mertz, and M. Hebert. Pcn: Point completion network. In 2018 international conference on 3D vision (3DV), pages 728–737. IEEE, 2018.
- [53] J. Zhang, X. Chen, Z. Cai, L. Pan, H. Zhao, S. Yi, C. K. Yeo, B. Dai, and C. C. Loy. Unsupervised 3d shape completion through gan inversion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1768–1777, 2021.

### A Sensitivity to Textual Description

#### A.1 Completion with Different Text Descriptions

Our approach operates by combining a partial input point cloud with a text description that guides the model when completing missing parts of the object.

We tested the effect of changing the text prompt while keeping the same input point cloud. Fig. 7 shows results for completing Redwood’s scan "08754" of a partially captured teapot (Fig.1 in the main paper). Completing the point cloud with other text descriptions demonstrates how the text controls the shape.

[Figure 19]

Input points “A teapot” “A coffee pot” “A watering can” “A pitcher” "A cup"

Figure 7: Completion of the same input, with different text descriptions. Results obtained with our method for the partial point cloud of scan "08754". While the handle and the top part of the object are constrained by the input point cloud, the model completes the other side of the object according to the input text.

[Figure 20]

(a) Input points

(b) “An object” (c) “A thing” (d) “A plant” (e) "A plant in a large vase"

Figure 8: The effect of using a specific or generic text description. Results are shown for reconstructing scan "06127" from the Redwood dataset. (a) The input point cloud. (b, c) Completion using two generic texts. Completion quality is poor. (d) Completion using the object class name. (e) Completion using a detailed textual description.

#### A.2 Generic vs Detailed Text Prompts

To evaluate the contribution of selecting an appropriate text prompt per object, we repeated reconstruction experiments of the 10 objects evaluated in the main paper, but varied the text prompts. Specifically, we used three levels of description specificity. First, for a fully generic prompt (class agnostic), we tested two alternatives: "An object", "A thing". Second, we used the class name as the prompt. Finally, we used a more detailed description.

Table 2 provides the Chamfer distances between our reconstruction and the ground truth for all prompts. Using generic text yields inferior reconstructions. Adding specific details did not provide a significant improvement over using the class name. A qualitative comparison is shown in Fig. 8.

Object "An object" "A thing" "A <class name>" Full text

executive chair 32.9 33.7 28.9 23.7 trash can 37.8 44.0 37.2 36.4 old chair 86.6 87.6 22.4 19.3 outside chair 58.0 52.7 23.4 22.6 plant in vase 63.6 44.5 36.4 29.5 one leg table 48.3 34.2 24.2 20.3 vespa 70.6 74.5 57.6 57.6 tricycle 40.3 44.7 36.0 39.0 couch 74.7 69.5 36.5 36.5 office trash 26.1 26.7 23.9 20.5

Average 53.9 51.2 32.6 30.5

Table 2: The effect of generic vs detailed prompt in terms of Chamfer distances (lower is better). Columns 1 and 2: two generic configurations where a global text is used for all objects. Column 3: only the class name is used e.g. both "executive chair" and "outside chair" are reconstructed with the text "A chair". Column 4: the results of our model with the text prompts from Table 5.

Input points

Incorrect text Correct text

[Figure 21]

“A chair” “A one leg square table”

“A table” ”An outside chair”

“A table”

”An old chair”

Figure 9: The effect of incorrect textual descriptions. Each row corresponds to a different object. Left: The partial scans that are given as input to our model. Middle: Completion performed using incorrect text descriptions. Right: the completion results of our method with our final text prompts. In the first two rows, the completion is inferior when given the wrong text. In the bottom row, even with an incorrect text ("A table") the model still completes the chair correctly. This is because the input provides strong constraints. To make the shape more similar to a table, the method still needs to reconstruct the missing leg.

#### A.3 Reconstruction with incorrect prompts

We further check the sensitivity of our model to wrong text prompts. Specifically, we used the text: "A table" for a chair, and the text "A chair" for a table. The visualizations are presented in Fig. 9.

### B Additional Results

Supplementary Redwood Results We evaluate our method on 4 additional Redwood cases with available ground truth surfaces: "08712","05456","00034" and "06912". Qualitative and quantitative comparisons are presented in Fig.11 and Table 3 respectively. We can see that our method completes the shapes better than the baselines.

We applied our method to additional Redwood cases of various object types with no available ground truth. Qualitative results, including RGB renderings, are shown in Fig. 10.

Video Results We attach to the supplementary folder, 360o video visualizations of our reconstructed objects for both, KITTI and Redwood datasets.

[Figure 22]

[Figure 23]

- Figure 10: Qualitative outputs of our method, when applied on Redwood cases with no available 360o GT scans for quantitative comparison. The figure is arranged as 2 columns of different objects, where for each object we show (from left to right): the input point cloud, the completed surface, the completed surface together with the input points, and image rendering of our optimized coloring function c.

object Shape PoinTr SDS-Complete Former (ours)

plant in vase 2 31.3 37.6 21 park trash can 130 119.9 39.8 bench 29 32.6 27.4 sofa 106.6 129.3 29.2

average 82 83.9 33.7

Table 3: Quantitative evaluation for additional objects from the Redwood dataset. Chamfer distance (in mm) comparisons between SDS-Complete to the baseline methods. Our method performs better than the baselines.

### C KITTI User Study.

We conduct a user study for evaluating the various methods on the KITTI dataset given 15 real object scans of cars, motorcycles, trucks, and an excavator. Specifically, we gather a group of 5 participants to rank the quality of each completed surface and its faithfulness to the input partial point cloud. For each object, the participants were given three anonymous shapes produced by the three methods: SDS-Complete, ShapeFormer [47], and PoinTr [51]. While the outputs of SDS-Complete and ShapeFormer are surfaces, PoinTr only outputs a point cloud. Therefore, we applied Screened Poisson Surface Reconstruction [21] to each output of PoinTr to base the user study comparisons on surface representations. The participants were instructed to choose the best shape, while the order of the methods was shuffled for each object. The best completion method for each input case is selected

[Figure 24]

- Figure 11: Additional comparisons with four objects from the Redwood dataset. Qualitative comparisons between SDS-Complete to baseline multi-class methods. Notably, SDS-Complete, produces more accurate completions.

by the majority vote. Among the three methods, our method stands out as the method with the highest number of wins (11 out of 15). The results of the user study are presented in Table 4.

object Shape PoinTr SDS-Complete

Former (ours) best quality [%] 26.6 0.0 73.4

Table 4: Human evaluation on KITTI. A user study for evaluating our method’s surface completion quality compared to the baselines, when tested on real object scans from the KITTI dataset. For each method, we present the percentage of cases that this method was selected by the participants as the best method. Our method gets the highest number of wins compared to the baselines.

### D Failure examples

Fig. 12 shows failure examples. In general, our method does not reconstruct well thin surfaces. We hypothesize that the initialization of the SDF to sphere [3], prevents the model from minimizing the occluded part at early training stages. Then, the SDS loss usually tries to paint this redundant content according to the text prompt, instead of removing it. Different initializations to the SDF, or other regularizations, need to be explored and left as future work.

### E Implementation Details

Running Time We run our method for 2000 epochs, where each epoch uses 100 iterations. That takes about 1950 and 1380 minutes for Redwood and KITTI scans respectively, on NVIDIA RTX

[Figure 25]

(a) (b) (c) (d)

- Figure 12: Failure cases of our method. (a),(b) Input points and surface completion respectively, for Redwood scan "05492" (standing sign). (c),(d) Input points and surface completion respectively for Redwood scan "01373" (picnic table).

A6000. We note that many scans need much fewer iterations for converging, but to complete the fine details, e.g. the chair’s legs, many iterations are needed due to the low resolution of the image rendering for the SDS-loss.

Network Architecture For our optimized coloring function cφ, we use 4 linear layers with 96 channels, where the two intermediate ones are incorporated in ResNet [18] blocks, as implemented

by [42], with SiLU activations [19]. For the SDF network fθ, we use 4 linear layers with 96 channels and ReLU activations. fθ is initialized to a sphere [3] with radius lengths of 0.5 and 0.9 for Redwood and KITTI scans respectively. For both, cφ and fθ we use Positional Encoding with 6 levels [29]. For extracting density from fθ (Equations (5) and (6) in the main paper) we use α = 100,β = 10−3.

SDS-loss Implementation Details We base our code on the implementation of [42]. During training, for each iteration, we randomly sample a background color, to prevent the model from "changing" the geometry by just coloring it with the background color. For Redwood cases, we render 80 × 80 images for the SDS-loss using the sampled camera and the known internal parameters of the sensor. For KITTI, at initialization time we first project the object’s LiDAR points to a 2D spherical projection [5], with height and width of 64 and 1024 pixels respectively. We use the projected 2D mask to select the 2D bounding box area of 64 pixels height, where the width is determined by the min and max horizontal coordinates of the object ±5 pixels. The LiDAR rays that define this selected bounding box are used to render the object during training, where a novel camera pose is defined by rotating these rays around the object’s centroid. As a text-to-image diffusion model we use Stable Diffusion v2 [38].

Training Details We optimize the networks using the Adam optimizer [23] with a learning rate 10−4. The coefficients for our loss for all the experiments are δm = 105, δd = 105, δp = 105, δeikonal = 104,δplane = 105. At each iteration we sample 1000 uniform points for Lplane and Leikonal. For Lm,Ld, at each iteration, we randomly sample 2000 pixels for Redwood cases, whereas for KITTI, we render the entire bounding box.

Camera Sampling As described in Section 4.2 of the main paper, during training, we start by applying the SDS loss on the rendered image from C0 pose, and then we gradually increase the sampling range of the deviation angles until the entire object is covered. In more detail, we gradually increase the sampling range of the azimuth angles: γazimuth ∼ U(−ν,ν), starting from ν = 0 to ν = 180. Specifically, we set ν = 30,45,60,90,180 at epochs 20,50,80,100,120 respectively. γelevation is set to 0 for 20 epochs and then uniformly sampled according to: γelevation ∼ U(−ξ0,0) for Redwood scans, where ξ0 is the elevation of C0 from the plane l in degrees. For KITTI scans (after epoch 20) we use γelevation ∼ U(−ξ0,ξ0) since the original viewpoint is usually low, and we also scale the distance from the source to the object uniformly by ∼ U(1,2) after epoch 20. As in [34], we augment the input text according to the viewing direction, with a text that describes the viewpoint orientation. Specifically, as in [42] we use "*, front view", "*, side view", "*, back view", "*, overhead view" and "*, bottom view", where * denotes the input text. Unlike [34], the orientation of the object is determined by the input points. Therefore, we use an extra input from the user of γ0

= 90 if the object is viewed from the side. Then, during training, we use γ0

, which explains the original viewpoint, e.g. γ0

azimuth

azimuth

and γazimuth to calculate the azimuth with respect to the

azimuth

object, and γelevation to compute the elevation with respect to the plane l. These orientations are used to augment the text with the corresponding view direction description.

Object Centralization Given the input points we centralize them at the origin. This is done in general by subtracting their center of mass. When the object’s largest dimension is aligned with the viewing axis, the center of mass is usually biased toward the camera. To handle this, we extract an oriented 3D bounding box for the input points and measure the ratio between the largest distance to the smallest distance from the center of mass to any bounding box point. If this ratio is above 1.7 we use the bounding box center as our centroid instead of using the center of mass. In the KITTI dataset, which mostly includes non-isotropic objects, we always use the bounding box center as our centroid. We then scale the points such that the largest point norm is 0.5.

Baseline Runnings For running the baseline methods, we tried to locate the input points as much as possible according to the method’s expectations to prevent them from failing. This includes using our knowledge about the world plane l and the object orientation with respect to the camera γ0

. For ShapeFormer, each time we took the best shape out of the 5 that it outputs.

azimuth

Data Processing For the Redwood dataset, we segmented out the foreground object manually. As a preprocessing, we manually aligned the GT scan with the partial point cloud and applied ICP for refinement [2]. Each KITTI scan that we used, is the aggregation of 5 timestamps. The segmentation map for KITTI is given by [5]. For both, KITTI and Redwood datasets and for each scan, the plane l is segmented out from the original point cloud using RANSAC [17].

### F List of Textual Prompts

The text prompts for the Redwood shapes that we tested are presented in Table 5. For the KITTI dataset, we used the text "A *", where "*" denotes the shape class: "car", "truck", "motorcycle", or "mini excavator".

### G Broader Impact

Our approach uses the SDS loss that builds on a text-to-image diffusion model [38]. As such, it inherits possible biases that such a model may have.

Scan ID Scan name Text Prompt

08712 plant in vase 2 "A plant" 06912 bench "A bench" 05492 standing sign "A standing sign" 05456 park trash can "Park trash can" 01373 picnic table "A picnic table" 00034 sofa "A sofa" 01184 trash can "An outdoor trash can with wheels" 06127 plant in vase "A plant in a large vase" 06830 tricycle "Children’s tricycle with adult’s handle" 07306 office trash "An office trash can" 05452 outside chair "An a outside chair" 06145 one leg table "A one leg square table" 05117 old chair "An old chair" 09639 executive chair "An executive chair" 06188 vespa "A motorcycle" 07136 couch "A couch" 08754 teapot "A teapot" 09664 traffic cone "A traffic cone" 04797 street faucet "A street faucet" 02426 clothes iron "A clothes iron" 09424 backpack "A backpack" 04014 lamp "A lamp" 04210 globe "A globe" 06271 vase "A vase" 04154 fire extinguisher "A fire extinguisher" 00030 vase 2 "A vase" 09484 rocking horse "A rocking horse" 04919 duck spring swing "A duck spring swing" 06177 printer "A printer" 04457 vacuum cleaner "Vacuum cleaner" 06124 vase 3 "A vase" 09488 basketball "A basketball"

Table 5: Scan IDs from the Redwood dataset [10], and their corresponding textual prompts.

