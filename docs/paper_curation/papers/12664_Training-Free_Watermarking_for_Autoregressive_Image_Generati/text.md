arXiv:2505.14673v1[cs.CV]20May2025

# Training-Free Watermarking for Autoregressive Image Generation

###### Yu Tong1,2 Zihao Pan3 Shuai Yang4 Kaiyang Zhou1,

1Hong Kong Baptist University 2Wuhan University 3Sun Yat-sen University 4Peking University https://github.com/maifoundations/IndexMark

#### Abstract

Invisible image watermarking can protect image ownership and prevent malicious misuse of visual generative models. However, existing generative watermarking methods are mainly designed for diffusion models while watermarking for autoregressive image generation models remains largely underexplored. We propose IndexMark, a training-free watermarking framework for autoregressive image generation models. IndexMark is inspired by the redundancy property of the codebook: replacing autoregressively generated indices with similar indices produces negligible visual differences. The core component in IndexMark is a simple yet effective match-then-replace method, which carefully selects watermark tokens from the codebook based on token similarity, and promotes the use of watermark tokens through token replacement, thereby embedding the watermark without affecting the image quality. Watermark verification is achieved by calculating the proportion of watermark tokens in generated images, with precision further improved by an Index Encoder. Furthermore, we introduce an auxiliary validation scheme to enhance robustness against cropping attacks. Experiments demonstrate that IndexMark achieves state-of-the-art performance in terms of image quality and verification accuracy, and exhibits robustness against various perturbations, including cropping, noises, Gaussian blur, random erasing, color jittering, and JPEG compression.

#### 1 Introduction

With the remarkable success of Large Language Models (LLMs) [31, 2, 41] in natural language processing, recent advancements have seen autoregressive image generation models, such as LlamaGen [25] and VAR [27], demonstrating substantial potential in the domain of visual generation. These models leverage a Vector Quantization (VQ) tokenizer [30] to transform images into discrete tokens. Subsequently, they autoregressively predict the “next token” within a codebook to generate images. Notably, these models exhibit significant advantages in terms of both image quality and generation speed. The proliferation of open-source high-quality autoregressive image generation models empowers the general public to create diverse customized content. However, it also brings potential risks of model misuse [3, 43, 32], such as fake news fabrication, ambiguous copyright attribution, and improper use of public figures’ portraits. Amidst growing calls for government regulation and industry compliance [16, 36], model developers need to enhance image traceability to ensure accountability in legal liability determination, copyright protection, and content moderation.

Invisible watermarking [15, 1, 24] provides a technical pathway for image traceability. This technology embeds imperceptible watermarks into images to help model developers achieve user-level attribution tracking of AI-generated content. Existing watermarking methods can be broadly categorized into two types: post-processing watermarks embedded after generation [5, 37, 1], and generative

Corresponding author

Preprint. Under review.

watermarks integrated during the generation process [38, 10]. Since the former introduces additional inference and storage overhead, generative watermarks are generally more practical and hence more popular. However, current generative watermarking techniques primarily focus on diffusion models and lack exploration for the emerging autoregressive image generation models. Due to substantial architectural differences between the two paradigms—diffusion models employ progressive denoising [13] whereas autoregressive models rely on sequential generation [25]—current diffusion-based watermarking methods cannot be directly applied to autoregressive models. This motivates us to investigate efficient and effective autoregressive watermarking strategies.

We believe leveraging the characteristics of the autoregressive image generation models is the key to the design of an effective water strategy. Recent research in autoregressive image ion models has identified a notable redundancy their codebooks [14, 11]: a large number of vectors are associated with different indices but highly similar to each other. This feature naturally leads to an elegant solution of watermarking that has minimal impact content of the image. Instead of using an image as ark, we embed the watermark by altering the distribution of generated indices [18]. Specifically, divide the codebook into “red” and “green” groups by pairing similar indices. After generating the indices, we replace as many red indices as possible with their paired green indices (called watermark tokens), thus changing the distribution ratio of red-green indices in the final sequence to embed watermark information (see Figure 1). This type of watermark has three advantages. 1) It is inherently robust and can only be removed by extensively modifying the color blocks of the image. 2) The redundancy of the codebook allows this match-then-replace strategy to imperceptibly embed watermarks. 3) Moreover, different red-green division schemes can correspond to different identity identifiers (IDs), assisting model developers in image tracing.

|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>1|[Figure 4]<br><br>1|[Figure 5]<br><br>0|[Figure 6]<br><br>0|
|---|---|---|---|
|[Figure 7]<br><br>0|[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>1|[Figure 11]<br><br>1|[Figure 12]<br><br>0|
|[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>0|[Figure 17]<br><br>0|[Figure 18]<br><br>0|[Figure 19]<br><br>0|
|[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>1|[Figure 23]<br><br>1|[Figure 24]<br><br>1|[Figure 25]<br><br>0|

[Figure 26]

[Figure 27]

|1|15|86|57|
|---|---|---|---|
|59<br><br>m|51<br><br>ark|65<br><br>in|28<br><br>g|
|23|56<br><br>ge|73<br><br>ne|83<br><br>ra|
|6|12<br><br>ssu|17<br><br>e|29<br><br>in|

|1|15|86|57|
|---|---|---|---|
|59|51|65|28|
|23|56|73|83|
|6|12|17|29|

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

50% Green Index

Unwatermarked Image

replace

replace

[Figure 35]

|[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>1|[Figure 39]<br><br>1|[Figure 40]<br><br>1|[Figure 41]<br><br>1|
|---|---|---|---|
|[Figure 42]<br><br>1|[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>1|[Figure 46]<br><br>1|[Figure 47]<br><br>1|
|[Figure 48]<br><br>1|[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>1|[Figure 53]<br><br>1|[Figure 54]<br><br>1|
|[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>1|[Figure 58]<br><br>1|[Figure 59]<br><br>1|[Figure 60]<br><br>1|

[Figure 61]

|on1|15th|e74|90|
|---|---|---|---|
|11a|51wa|65te|rm26|
|24st|ati53|stic71|13al|
|6|12|17w|82e|

[Figure 62]

|1|15|74|90|
|---|---|---|---|
|11|51|65|26|
|24|53|71|13|
|6|12|17|82|

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

100% Green Index Watermarked Image

Figure 1: Watermark embedding by index replacement to attain a higher proportion of watermark tokens (green index).

Building on the above insights, this paper proposes IndexMark, the first training-free watermarking framework for autoregressive image generation models. We first formulate the pairing of similar indices as a maximum weight perfect matching problem [22], and solve it with top-K pruning and the Blossom algorithm [7]. Then, we randomly assign each pair of indices to a red list or a green list. After autoregressive index generation, red indices are selectively replaced with their paired green indices according to index confidence, thereby embedding an invisible image watermark. This matchthen-replace strategy robustly embeds watermarks with image quality and content well preserved. During the watermark verification stage, indices of the generated image are reconstructed via VQVAE to compute the “green-index rate” for verification. To compensate the index reconstruction errors of VQ-VAE, we introduce an Index Encoder for accurate index reconstruction. Although the red-green watermark is intrinsically robust against various image perturbations, the verifier is still vulnerable to cropping attacks due to VQ-VAE’s block-level processing characteristics. Therefore, we propose a corresponding cropping-robust validation scheme specific to the modern autoregressive image generation models.

Our key contributions are summarized as follows: 1) We propose a training-free watermarking framework that can be directly applied to autoregressive image generation models without requiring any additional fine-tuning or training. 2) We introduce a match-then-replace approach, which enables training-free watermark embedding with minimal impact on the visual quality of the images. 3) We design a precise image indexing validation framework that can verify the presence of watermarks with higher statistical confidence. 4) Thanks to the Index Encoder and the designed cropping-robust watermark verification method, our approach demonstrates strong robustness towards a wide range of image perturbations.

#### 2 Related Work

###### 2.1 Image Watermarking

Watermarks in generative models can be embedded either after images are generated (i.e., postprocessing) or during the generation process (i.e., in-processing). Post-processing methods are mainly divided into transformation-based and deep encoder-decoder methods. Representative post-processing methods include Discrete Wavelet Transform (DWT) [37], Discrete Cosine Transform (DCT) [5], and DWT-DCT methods [1], which embed watermarks into the spatial or frequency domain with minimal impact on image quality. A drawback of these methods is that simple attacks in the pixel space can significantly affect the accuracy of watermark extraction. Deep encoder-decoder methods often generate watermarked images in an end-to-end manner, e.g., using adversarial training to add imperceptible noise into image pixels [42, 40]. However, these methods often struggle to generalize to images outside the training data distribution.

Research on in-processing watermarking primarily focuses on diffusion-based models. The Tree-Ring watermark [35] embeds watermark into the noisy image before denoising. ROBIN [15] injects watermark into an intermediate diffusion state while maintaining consistency between the watermarked image and the generated image and robustness of the watermark. Though the performance is strong, these methods cannot be directly transferred to autoregressive architectures. Concurrent to our work, Safe-VAR [34] explores watermark embedding for the VAR model [27] through multi-scale interaction and fusion techniques. However, Safe-VAR lacks scalability as it only works for VAR-like models and suffers from high training costs as it requires over 200K images for fine-tuning the decoder and training an additional multi-scale module. In contrast, our approach does not require any training and can be applied more broadly to codebook-based autoregressive models.

###### 2.2 Autoregressive Image Generation

In autoregressive image generation, image data is typically transformed into one-dimensional sequences of pixels or tokens, and the model predicts the next image token based on the existing context. Early autoregressive image generation models [28, 29] perform image generation by predicting continuous pixels, which have high computational complexity. The seminal work, VQ-VAE [30], builds a codebook containing feature representations and casts image generation into a discrete label prediction problem. VQ-GAN [8] extends VQ-VAE by using adversarial training to improve the image quality. Recently, LlamaGen [26] and Open-MAGVIT2 [20] apply the concept of next-token prediction, which has been widely used in large language models, to autoregressive image generation, achieving performance that even surpass diffusion-based methods.1 However, the problem of embedding watermarks into autoregressive image generation models remains largely understudied, exposing huge risks of model misuse for these models. Our IndexMark fills this gap by offering a simple, training-free solution.

#### 3 Methodology

Task Definition Autoregressive model watermarking aims to embed an invisible, verifiable watermark w into an autoregressively generated image I during creation using a watermarking algorithm E, resulting in Iw. After potential real-world image transformations O such as JPEG compression, the model owner can extract the watermark from the altered image O(Iw) via a validation algorithm D, facilitating image traceability.

Our Framework As illustrated in Figure 2, our IndexMark framework is composed of two parts: watermark embedding (Sec. 3.1) and watermark verification (Sec. 3.2). In the watermark embedding part, we first divide the codebook of an autoregressive model into pairs of indices such that each pair contains similar vectors. Then, for each index pair we randomly assign one index into a red list and the other into a green list. Finally, we perform selective red-green index replacement based on index confidence during the image decoding process to embed invisible watermarks into images (i.e., replacing as many red indices as possible with green indices). In the watermark verification part, we propose a method based on statistical probability, where an Index Encoder is introduced to achieve

1The background on autoregressive image generation can be found in the Appendix B.1.

##### mage

Watermark embedding Watermark verification

|[Figure 69]<br><br>1|[Figure 70]<br><br>1|[Figure 71]<br><br>0|[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>0|
|---|---|---|---|
|[Figure 75]<br><br>0|[Figure 76]<br><br>1|[Figure 77]<br><br>1|[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>0|
|[Figure 82]<br><br>0|[Figure 83]<br><br>0|[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>0|[Figure 87]<br><br>0|
|[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>1|[Figure 92]<br><br>1|[Figure 93]<br><br>1|[Figure 94]<br><br>0|

[Figure 95]

###### Autoregressive Model

[Figure 96]

|1|15|86|57|
|---|---|---|---|
|59|51|65|28|
|23|56|73|83|
|6|12|17|29|

- h w i i
- i

[Figure 97]

- p p q q c



 

- q  ∣

( ) ( , )

[Figure 98]

[Figure 99]

1

[Figure 100]

|1|15|86|57|59|51|
|---|---|---|---|---|---|

?

[Figure 101]

qi qi

[Figure 102]

[Figure 103]

[Figure 104]

Codebook

[Figure 105]

|[Figure 106]<br><br>1|[Figure 107]<br><br>1|[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>1|[Figure 111]<br><br>1|
|---|---|---|---|
|[Figure 112]<br><br>1|[Figure 113]<br><br>1|[Figure 114]<br><br>1|[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>1|
|[Figure 119]<br><br>1|[Figure 120]<br><br>1|[Figure 121]<br><br>1|[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]<br><br>1|
|[Figure 125]<br><br>1|[Figure 126]<br><br>1|[Figure 127]<br><br>1|[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>1|

|11| |
|---|---|

[Figure 132]

- p1
- p2

|1|15|74|90|
|---|---|---|---|
|11|51|65|26|
|24|53|71|13|
|6|12|17|82|

[Figure 133]

|59| |
|---|---|

[Figure 134]

|53| |
|---|---|
|56| |

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

|N-1| |
|---|---|
|N| |

[Figure 139]

pN/2

[Figure 140]

Figure 2: Watermark embedding and verification of IndexMark. During autoregressive index generation, IndexMark selectively replaces red indices with green indices from the same index pair based on confidence to embed the watermark. The watermarked image is fed into the Index Encoder to calculate the green index rate for watermark verification.

precise reconstruction of indices. We also propose a cropping-invariant watermark verification scheme for cropped images.

###### 3.1 Watermark Embedding

Construction of Index Pairs We aim to divide the indices in the codebook, I = {1,2,...,N}, into index pairs P = {p1,p2,...,pN/2}, where each pair pk = {ik,jk} contains two distinct indices from I such that k N/=12 pk = I and pk ∩ pk′ = ∅ for k ̸= k′. The objective of the red-green index allocation is to compute an optimal assignment that maximizes the sum of the intra-pair similarity Ssum for all red-green index pairs:

N/2

sim(ik,jk), (1)

Ssum = max P∈P

k=1

where P is the set of all possible such partitions P and sim(i,j) is the cosine similarity between the vectors of index i and index j in the codebook. We cast this problem into a maximum weight perfect matching problem. Specifically, we construct a complete graph G = (V,E), where each vertex in the vertex set V represents an index from the codebook and the edge set E connects all pairs of vertices, with the edge weights w set as the cosine similarity between the two connected vertices. After constructing this complete graph, our objective is to find a maximum weight perfect matching M∗, which is a subset of E containing N/2 edges such that every vertex in V is linked to only one edge in M∗, and the sum of the weights of these N/2 edges is maximized:

M∗ = arg max

w(i,j), (2)

M∈M

(i,j)∈M

where M represents the set of all possible perfect matchings on the graph G, and w(i,j) represents the weight of the edge connecting vertex i and vertex j.

We solve this problem using the Blossom algorithm [7]. Considering the large number of indices in the codebook, directly applying the Blossom algorithm would result in extremely high computational complexity. For this reason, we perform top-K pruning on the complete graph, retaining only the K edges with the highest weights for each vertex. We then apply the Blossom algorithm [7] to the pruned sparse graph to obtain the maximum weight perfect matching M∗. Please refer to Appendix B.2 for the details on the Blossom algorithm.

Red-Green Index Assignment After obtaining the maximum weight perfect matching M∗, we need to assign indices to red and green lists for each index pair in M∗. For simplicity, we randomly assign the two indices in each index pair to the red and green lists. In practical applications, users

can customize the assignment of red and green indices. The total number of possible assignments is as high as 2N/2, providing model developers and users with extremely abundant identity identifiers (IDs) for image tracing.

Confidence-Guided Index Replacement Autoregressive image generation models produce token index sequences in an autoregressive manner. Our objective is to replace as many red indices as possible with green indices, while avoiding bad replacements that harms image quality. To achieve controllable watermark strength that balances between watermark strength and image quality, we propose a confidence-guided index replacement strategy. Specifically, we use the classification probability of an index predicted by the autoregressive model as the confidence measure and calculate relative confidence (will be detailed in Eq. (3)). The greater the relative confidence, the larger the gap between the red index and the paired green index at the current index position. Replacing these indices with significant gaps can lead to a noticeable decline in image quality. Based on the relative confidence distribution of two indices within each pair, we calculate a quantile as the replacement threshold to control watermark strength. For a given replacement threshold, we prioritize replacing index pairs with smaller relative confidence to balance watermark strength and image quality.

When the autoregressive model generates the k-th red index Idxk, we record the classification probability P(Idxk) for Idxk and the classification probability P(Idx′k) for its paired green index Idx′k. After the autoregressive model generates all indices, we obtain the confidence set for red indices, conf = {P(Idx1),P(Idx2),...,P(IdxN

)}, and the confidence set for paired green indices, conf′ = {P(Idx′1),P(Idx′2),...,P(Idx′N

red

)}, where Nred represents the total number of red indices. Based on these two sets of confidence, we calculate the relative confidence for each index pair:

red

relative-confk = log(P(Idxk)/P(Idx′k)), (3)

where k represents the relative confidence of the k-th index pair. We achieve controllable watermark strength by setting a distribution quantile, with the relative confidence distribution illustrated in Figure 3. Specifically, when replacing red indices with paired green indices, we only replace index pairs on the left side of the quantile. Therefore, when the quantile is set to 0%, the model does not perform red index replacement, resulting in a non-watermarked image. When the quantile is set to 100%, the model replaces all red indices with green indices from their index pairs. For other quantile values, the model prioritizes replacing red indices with green indices of lower relative confidence. Additionally, the confidence distribution exhibits a characteristic pattern of low density at both ends and high density in the middle, making index pairs with high relative confidence more likely to be filtered out. For example, by simply setting the quantile to 95%, we can filter out all index pairs with a relative confidence greater than 5. This design not only achieves controllability of watermark strength but also optimizes the balance between watermark strength and image quality by “filtering” index pairs with high relative confidence.

[Figure 141]

[Figure 142]

[Figure 143]

Figure 3: Index pair distribution of one hundred generated images.

###### 3.2 Watermark Verification

Statistical Probability-Based Watermark Verification Since the red and green indices are randomly assigned, the proportion of green indices in an image without a watermark is approximately 50%, while the proportion in an ideal watermarked image approaches 100%. In fact, the process of autoregressive image generation can be regarded as NIdx independent Bernoulli trials with equal probability of taking the value 0 (red index) or 1 (green index), where NIdx is the total number of indices in an image. According to the Central Limit Theorem [9], when NIdx is sufficiently large, the sample mean follows a normal distribution. Thus, we can calculate the confidence interval CI for the mean of NIdx trials at a confidence level of 1 − β as follows:

zβ/2 2√NIdx

zβ/2 2√NIdx

CI = 0.5 −

, (4)

, 0.5 +

where zβ/2 represents the two-tailed critical value of the standard normal distribution. After calculating the confidence interval, we can use the right endpoint of the confidence interval as a decision threshold. If the proportion of green indices in an image is below the threshold, the image is classified as a non-watermarked image; otherwise, it is classified as a watermarked image.

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

[Figure 157]

|51|64|77|
|---|---|---|
|55|12|61|
|29|41|24|

|51|64|77|
|---|---|---|
|55|19|61|
|10|41|24|

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

[Figure 177]

[Figure 178]

[Figure 179]

real rc rc

[Figure 180]

[Figure 181]

[Figure 182]

- Figure 4: Training of Index Encoder. The Encoder, Codebook, and Decoder are frozen while the Index Encoder is updated to achieve accurate index reconstruction.

Index Encoder VQ-VAE is designed solely for pixel-level reconstruction. However, the objective of the watermark validator in this paper is index reconstruction. In practice, the autoregressively generated indices Idx = {Idx1,Idx2,...,IdxN

Idx} are processed through the decoder to produce the image I. However, the reconstructed indices Idxrc obtained by feeding I into the encoder and vector quantization module, may differ from the original Idx. As illustrated in Figure 6(b), the original VQ-VAE encoder struggles to accurately identify watermarks in high-confidence scenarios. Therefore, we propose an Index Encoder to assist users in high-confidence scenarios in achieving high-precision index reconstruction.

As shown in Figure 4, we freeze the encoder, codebook, and decoder, and retrain a new encoder, termed Index Encoder, with the goal of achieving accurate reconstruction of the indices Idx. We first input the original image Ireal into the encoder and decoder to obtain the vector z and image I. Then, we input I into the Index Encoder and decoder to obtain the reconstructed vector zrc and reconstructed image Irc. The optimization of the Index Encoder is performed by minimizing two loss terms: (1) the mean squared error (MSE) between the vector z and the vector zrc, and (2) the MSE between the image I and the reconstructed image Irc:

Lencoder = ∥zrc − z∥22 + γ∥Irc − I∥22, (5) where γ represents the weight hyperparameter.

Cropped Image Watermark Verification Although the red–green index watermark itself is highly robust, as demonstrated in Table 1, the VQ-VAE encoding paradigm is inherently vulnerable to cropping attacks. During index reconstruction, VQ-VAE first divides an image into fixed-size, nonoverlapping patches (e.g., 8 × 8 pixels) and independently encodes each patch to retrieve the index. Then, even a slight crop to the image can drastically alter the patch composition. For instance, consider cropping an image such that the new top-left corner lands at position (4, 3) within the neighborhood of the original patch. Such a tiny shift entails that every subsequent pixel now belongs to completely different spatial segments compared to the original image. When encoded, these reconfigured patches will lead to entirely different codebook indices, thereby significantly weakening watermark-verification robustness.

To address this weakness, we propose traversing every pixel in the local image block of the cropped image to achieve alignment of the local image blocks. Taking an 8 × 8 pixel block as an example, suppose the top-left corner of the cropped image is originally located at (4, 3) of a block. We enumerate the cropped image to traverse all pixel positions within the first local image block. That is, the top-left corner of the cropped image moves from (1, 1) to (8, 8), stopping after enumerating 64 candidate images. For each candidate, we calculate its green index rate. As long as the green index rate of one of the candidates reaches the decision threshold, the image is considered to contain a watermark. For example, as the image moves from (1, 1), the green index rate remains close to 50%. However, when the top-left corner of the cropped image moves to (4, 3), the green index rate reaches 100%, indicating the presence of a watermark. The example of cropped image verification can be found in the Appendix B.3.

#### 4 Experiments

Model and Datasets We conduct experiments using a state-of-the-art autoregressive image generation model, LlamaGen [25]. For text-to-image generation tasks, we generate images at 256×256 and 512×512 resolutions. For class-conditioned image generation tasks, we generat images at 256×256

###### Table 1: Comparison of IndexMark with post- and in-processing watermarking methods in terms of quality and robustness against various attacks.

Image quality Accuracy ↑ PSNR ↑ SSIM ↑ MSSIM ↑ CLIP ↑ FID↓ Clean Blur Noise JPEG Bright Erase Crop Avg

Model Method

MSCOCO Dataset Post processing (256 × 256)

37.71 0.970 0.992 0.325 25.85 0.603 0.501 0.607 0.500 0.571 0.567 0.500 0.549 DwtDctSvd 37.57 0.979 0.992 0.325 27.60 0.996 0.982 0.994 0.963 0.556 0.994 0.500 0.855

DwtDct

| | |
|---|---|
| | |

RivaGAN 40.44 0.980 0.992 0.324 25.78 0.930 0.919 0.929 0.727 0.862 0.847 0.500 0.816 LlamaGen (AR) (256 × 256)

W/o watermark ∞ 1.000 1.000 0.328 26.55 − − − − − − − −

IndexMark 23.54 0.838 0.930 0.326 24.73 1.000 0.991 0.995 0.978 0.988 0.997 0.998 0.992 Post processing (512 × 512)

37.61 0.963 0.990 0.279 54.30 0.741 0.512 0.739 0.500 0.680 0.734 0.500 0.629 DwtDctSvd 37.38 0.972 0.989 0.280 55.60 0.999 0.990 0.998 0.988 0.673 0.998 0.500 0.878

DwtDct

| | |
|---|---|
| | |
| | |

RivaGAN 40.41 0.978 0.989 0.279 56.49 0.973 0.967 0.970 0.900 0.930 0.945 0.958 0.949 Stable Diffusion (512 × 512)

W/o watermark ∞ 1.000 1.000 0.403 25.53 − − − − − − − − Tree-Ring 15.37 0.568 0.626 0.364 25.93 1.000 1.000 0.994 0.999 1.000 1.000 0.833 0.975

ROBIN 24.03 0.768 0.881 0.396 26.86 1.000 1.000 0.998 0.971 1.000 1.000 0.918 0.983 LlamaGen (AR) (512 × 512)

W/o watermark ∞ 1.000 1.000 0.282 54.57 − − − − − − − − IndexMark 24.15 0.838 0.930 0.281 54.35 1.000 0.988 0.994 0.984 0.989 0.992 0.993 0.991

ImageNet Dataset Post processing (256 × 256)

38.73 0.974 0.993 0.288 15.17 0.583 0.501 0.588 0.500 0.584 0.568 0.500 0.546 DwtDctSvd 38.44 0.979 0.991 0.288 15.32 0.994 0.991 0.989 0.960 0.552 0.994 0.500 0.854

DwtDct

| | |
|---|---|
| | |
| | |

RivaGAN 40.44 0.980 0.991 0.288 15.29 0.951 0.930 0.950 0.746 0.919 0.914 0.500 0.844 ImageNet Diffusion (256 × 256)

W/o watermark ∞ 1.000 1.000 0.271 16.25 − − − − − − − − Tree-Ring 15.68 0.663 0.607 0.267 17.68 1.000 0.994 0.999 0.998 0.798 0.995 0.924 0.958

ROBIN 24.98 0.875 0.872 0.275 18.26 1.000 0.999 0.999 0.999 0.928 0.999 0.994 0.988 LlamaGen (AR) (256 × 256)

W/o watermark ∞ 1.000 1.000 0.289 15.08 − − − − − − − −

IndexMark 23.86 0.738 0.903 0.288 13.89 1.000 1.000 1.000 1.000 0.998 1.000 0.998 0.999 Post processing (384 × 384)

39.36 0.972 0.991 0.286 12.50 0.720 0.521 0.725 0.500 0.780 0.696 0.500 0.634 DwtDctSvd 39.08 0.979 0.989 0.285 12.62 0.999 0.990 0.999 0.542 0.664 0.999 0.500 0.813

DwtDct

| | |
|---|---|
| | |

RivaGAN 40.45 0.977 0.989 0.286 12.79 0.966 0.947 0.964 0.846 0.949 0.999 0.953 0.946 LlamaGen (AR) (384 × 384)

W/o watermark ∞ 1.000 1.000 0.287 12.65 − − − − − − − − IndexMark 25.45 0.783 0.913 0.286 11.81 1.000 1.000 1.000 1.000 0.998 1.000 0.993 0.998

and 384×384 resolutions. We conduct pre-training for index reconstruction at various resolutions on LlamaGen’s text-to-image VQ-VAE and class-conditioned VQ-VAE using the MS-COCO-2017 training dataset [19] and the ImageNet-1k validation dataset [6], respectively.

Evaluation Metrics and Baselines To evaluate the effectiveness of IndexMark, we employ the watermark verification method based on statistical probability, calculating accuracy (ACC) to measure the watermark verification performance. In addition, we utilize Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Multiscale SSIM (MSSIM) [33] to quantify the pixel-level differences between watermarked and original images. We further assess the fidelity of the watermarked image distribution with the FID [12] and evaluate the alignment between generated images and their text prompts with the CLIP score [23]. We compare IndexMark with five baseline approaches: three post-processing methods including DwtDct [1], DwtDctSvd [21], and RivaGAN [40], and two diffusion-based methods including Tree-Ring [35] and ROBIN [15]. 2

Implementation Details For the construction of index pairs, we set top-K pruning with K = 10. For the text-conditioned generation task, we set top-K sampling with K = 1000, CFG-scale to 7.5, and downsample-size to 16. For the class-conditioned generation task, we set top-K sampling with K = 2000, CFG-scale to 4.0, downsample-size to 16, and default to a full-green index watermark. For the Index Encoder, we used the Adam optimizer [17] with a learning rate of 1e-5, and set γ to 0.5. All experiments are conducted on an NVIDIA A100 GPU.

###### 4.1 Image Quality and Watermark Robustness

Image Quality Traditional post-hoc watermarking methods often cause slight visual distortions and suffer from poor robustness. In contrast, generative methods can seamlessly embed watermarks into the generated content without altering the semantics. Diffusion-based methods typically operate in the latent spacebut tend to cause significant semantic changes due to the difficulty in precisely controlling the perturbation magnitude. In comparison, our approach is built upon VQ-VAE and autoregressive image generation models and therefore performs better in faithfully preserving image details and structures. As shown in Table 1, our approach achieves significant improvements in the PSNR, SSIM, and MSSIM scores, while causing much less image quality degradation compared to watermark-free generations, as evidenced by the CLIP and FID scores. Moreover, unlike diffusionbased methods, we observe that the FID of watermarked images in our method is even lower than that of non-watermarked ones, further demonstrating our superiority in preserving visual fidelity.

2More details about how the evaluation metrics are computed can be found in the Appendix C.

Diffusion W/o Watermark ROBIN AR W/o Watermark IndexMark Many bunches of bright yellow bananas hanging on display.

Green index generation

A table with plates of food with drinks in the middle.

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

Diffusion W/o Watermark ROBIN AR W/o Watermark IndexMark Many bunches of bright yellow bananas hanging on display.

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

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

A table with plates of food with drinks in the middle.

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

- Figure 5: ROBIN vs. IndexMark. ROBIN embeds watermarks during the intermediate diffusion state, which may lead to changes in the image content. In contrast, IndexMark uses the match-then-replace strategy to embed watermarks, effectively preserving the image’s quality and content.

(a) Evaluation on confidence-guided index replacement.

(b) Evaluation on Index Encoder.

Figure 6: Ablation results on confidence-guided index replacement and Index Encoder.

Figure 5 shows that the ROBIN method may remove certain objects from the original image (such as chopsticks and pastries on the plate), which affects parts of the generated image content. In contrast, our method effectively preserves the overall image content and semantic structure, demonstrating its superiority. More qualitative results can be found in the Appendix D.3.

Robustness To evaluate the robustness of our watermarking method, we select six common data augmentations as attack methods. These include Gaussian blur with a kernel size of 11, Gaussian noise with a standard deviation of σ = 0.01, JPEG compression with a quality factor of 70, color jitter with brightness set to 0.5, random erasing of 10% of the region, and random cropping of 75%. We select the right endpoint of the confidence interval at a 99.9% confidence level as the threshold for watermark determination. As shown in Table 1, we report the ACC under each attack setting. Notably, our method demonstrates strong robustness against most perturbations, significantly outperforming the baselines at image resolutions of 256, 384, and 512. While Stable Diffusion-based methods perform better than traditional approaches, they still fall noticeably short of our method.

###### 4.2 Ablation Study and Further Analyses

Confidence-Guided Index Replacement We substitute the confidence-guided method with random index selection based on watermark strength. The results shown in Figure 6(a) justify the effectiveness of our design as the PSNR scores of random index selection are significantly lower than IndexMark.

(a) Training Loss. (b) Index Reconstruction Rate. (c) Green index Validation Rate.

Figure 7: Images showing the variation of training loss, index reconstruction rate, and green index verification rate of the Index Encoder with respect to epochs.

Watermark strength

0 1

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

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

Figure 8: Generated images under different watermark strengths.

Index Encoder We compare watermark verification rates with and without Index Encoder at different confidence levels on 256×256 resolution images. As shown in Figure 6(b), differences are minimal at lower confidence levels, but at higher levels, Index Encoder significantly improves verification rates. More robustness experiments can be found in the Appendix D.2.

Index Reconstruction To investigate whether the Index Encoder improves index reconstruction capability, we conduct index-to-index reconstruction experiments at a resolution of 256 across multiple epochs using the Index Encoder, as well as validation experiments on pure green index images. Additionally, Figure 7(a) illustrates the training loss across multiple resolutions. As shown in Figure 7(b), after only 20 epochs of training, the index reconstruction capability of the Index Encoder surpasses that of the original encoder. Furthermore, as depicted in Figure 7(c), the Index Encoder’s validation capability for images with all indices being green significantly exceeds that of the original encoder, allowing users to verify watermarks with a higher confidence level.

Watermark Strength We explore the impact of watermark strength on images. The qualitative results, as shown in Figure 8, indicate that an increase in IndexMark watermark strength does not cause noticeable changes in image quality.

#### 5 Conclusion

This paper proposes IndexMark, the first training-free watermarking method for autoregressive image generation models. IndexMark carefully selects watermark tokens from the codebook based on token similarity and promotes the use of watermark tokens through token replacement, thereby embedding the watermark in the image. We believe that our method offers a novel perspective for watermark design in autoregressive image generation models.

#### References

- [1] Ali Al-Haj. Combined dwt-dct digital image watermarking. Journal of computer science, 3(9):740–746, 2007.
- [2] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [3] Miles Brundage, Shahar Avin, Jack Clark, Helen Toner, Peter Eckersley, Ben Garfinkel, Allan Dafoe, Paul Scharre, Thomas Zeitzoff, Bobby Filar, et al. The malicious use of artificial intelligence: Forecasting, prevention, and mitigation. arXiv preprint arXiv:1802.07228, 2018.
- [4] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2818–2829, 2023.
- [5] Ingemar Cox, Matthew Miller, Jeffrey Bloom, Jessica Fridrich, and Ton Kalker. Digital watermarking and steganography. Morgan kaufmann, 2007.
- [6] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.
- [7] Jack Edmonds. Paths, trees, and flowers. Canadian Journal of mathematics, 17:449–467, 1965.
- [8] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.
- [9] William Feller. An introduction to probability theory and its applications, Volume 2, volume 2. John Wiley & Sons, 1991.
- [10] Pierre Fernandez, Guillaume Couairon, Hervé Jégou, Matthijs Douze, and Teddy Furon. The stable signature: Rooting watermarks in latent diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22466–22477, 2023.
- [11] Ziyao Guo, Kaipeng Zhang, and Michael Qizhe Shieh. Improving autoregressive image generation through coarse-to-fine token prediction. arXiv preprint arXiv:2503.16194, 2025.
- [12] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.
- [13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [14] Teng Hu, Jiangning Zhang, Ran Yi, Jieyu Weng, Yabiao Wang, Xianfang Zeng, Zhucun Xue, and Lizhuang Ma. Improving autoregressive visual generation with cluster-oriented token prediction. arXiv preprint arXiv:2501.00880, 2025.
- [15] Huayang Huang, Yu Wu, and Qian Wang. Robin: Robust and invisible watermarks for diffusion models with adversarial optimization. Advances in Neural Information Processing Systems, 37:3937–3963, 2024.
- [16] Makena Kelly. White house rolls out plan to promote ethical ai, 2023.
- [17] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
- [18] John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. A watermark for large language models. In International Conference on Machine Learning, pages 17061–17084. PMLR, 2023.

- [19] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, pages 740–755. Springer, 2014.
- [20] Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Openmagvit2: An open-source project toward democratizing auto-regressive visual generation. arXiv preprint arXiv:2409.04410, 2024.
- [21] KA Navas, Mathews Cheriyan Ajay, M Lekshmi, Tampy S Archana, and M Sasikumar. Dwtdct-svd based watermarking. In 2008 3rd international conference on communication systems software and middleware and workshops (COMSWARE’08), pages 271–274. IEEE, 2008.
- [22] Constantine NK Osiakwan and Selim G Akl. The maximum weight perfect matching problem for complete weighted graphs is in pc. In Proceedings of the Second IEEE Symposium on Parallel and Distributed Processing 1990, pages 880–887. IEEE, 1990.
- [23] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [24] Ahmad Rezaei, Mohammad Akbari, Saeed Ranjbar Alvar, Arezou Fatemi, and Yong Zhang. Lawa: Using latent space for in-generation image watermarking. In European Conference on Computer Vision, pages 118–136. Springer, 2024.
- [25] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.
- [26] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.
- [27] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024.
- [28] Aaron Van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelcnn decoders. Advances in neural information processing systems, 29, 2016.
- [29] Aäron Van Den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. In International conference on machine learning, pages 1747–1756. PMLR, 2016.
- [30] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.
- [31] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [32] James Vincent. An online propaganda campaign used ai-generated headshots to create fake journalists. Verge. com, 2020.
- [33] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600– 612, 2004.
- [34] Ziyi Wang, Songbai Tan, Gang Xu, Xuerui Qiu, Hongbin Xu, Xin Meng, Ming Li, and Fei Richard Yu. Safe-var: Safe visual autoregressive model for text-to-image generative watermarking. arXiv preprint arXiv:2503.11324, 2025.

- [35] Yuxin Wen, John Kirchenbauer, Jonas Geiping, and Tom Goldstein. Tree-ring watermarks: Fingerprints for diffusion images that are invisible and robust. arXiv preprint arXiv:2305.20030, 2023.
- [36] Kyle Wiggers. Microsoft pledges to watermark ai-generated images and videos, 2023.
- [37] Xiang-Gen Xia, Charles G Boncelet, and Gonzalo R Arce. Wavelet transform based watermark for digital images. Optics Express, 3(12):497–511, 1998.
- [38] Ning Yu, Vladislav Skripniuk, Dingfan Chen, Larry Davis, and Mario Fritz. Responsible disclosure of generative models using scalable fingerprinting. arXiv preprint arXiv:2012.08726, 2020.
- [39] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. Advances in Neural Information Processing Systems, 37:128940–128966, 2024.
- [40] Kevin Alex Zhang, Lei Xu, Alfredo Cuesta-Infante, and Kalyan Veeramachaneni. Robust invisible video watermarking with attention. arXiv preprint arXiv:1909.01285, 2019.
- [41] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.
- [42] Jiren Zhu, Russell Kaplan, Justin Johnson, and Li Fei-Fei. Hidden: Hiding data with deep networks. In Proceedings of the European conference on computer vision (ECCV), pages 657–672, 2018.
- [43] Hazem Zohny, John McMillan, and Mike King. Ethics of generative ai, 2023.

Appendix

- A Limitations and Social Impact

- A.1 Limitations

The verification of IndexMark watermark relies on the index reconstruction capability of the VQ-VAE model. A more robust encoder can enhance the robustness of our method, such as index reconstruction based on image semantics [39]. Additionally, our current match-then-replace method uses simple pairwise matching. By exploring diverse matching methods, we can further leverage the redundancy of the codebook, thereby improving the quality of the watermarked images.

- A.2 Social Impact

With the rapid advancement of autoregressive image generation models, developers have the responsibility and obligation to ensure the safety of these models. We provide developers with an efficient and effective method to help them counteract the misuse of models, marking a step towards responsible AI in autoregressive image generation models.

- B Model Details

- B.1 VQ-VAE and Autoregressive Image Generation

VQ-VAE The Vector Quantized Variational Autoencoder (VQ-VAE) provides a framework for encoding images into a discrete latent representation. Given an input image x ∈ RH×W×3, the encoder produces a continuous latent feature map:

z = encoder(x) ∈ Rh×w×d. (6) For every spatial location (i,j) we find the nearest entry in the VQ-VAE’s codebook C = {e1,e2,...,eK} ⊂ Rd:

∥zij − ek∥2, zijq = ek

, (7)

kij = arg min

ij

k∈{1,...,K}

where kij is a discrete index, and zijq is the corresponding quantised vector. By flattening the quantized vector zq, a sequence of discrete tokens T = {T1,T2,...,Th×w} is obtained, where each token Ti represents an index in the codebook C. During the reconstruction phase, the quantized latent vector zq is retrieved using the token indices and the codebook. This vector is then passed through a decoder to reconstruct the original image: xˆ = decoder(zq). During the training phase, the model is constrained by the image reconstruction loss, codebook loss, and commitment loss, defined as:

L = ∥x − xˆ∥22 + β∥q − sg[z]∥22 + γ∥z − sg[q]∥22, (8) where sg denotes the stop gradient operation. Autoregressive Image Generation The autoregressive model defines the generation process as the prediction of the next token:

n

n

p(xi | x<i). (9)

p xi | x1,x2,...,xi−1 =

p(x) =

i=1

i=1

In autoregressive image generation, xi denotes the image token in the discrete latent space, and the image generation process can be formulated as:

- h×w
- i=1

p(qi | q<i,c), (10)

p(q) =

where qi denotes the discretized image token, c denotes the embedding of the class label or the text, and h × w represents the total number of image tokens. During the training phase, the model is trained by maximizing the likelihood of the observed token sequences:

- h×w
- i=1

log p qi | q<i,c . (11)

Ltrain = −log p(q) = −

During inference, the model generates the sequence of token indices autoregressively by sampling each next index. Once the full sequence of image token indices is produced, the codebook is used to reconstruct the latent vector zq from those indices, and zq is then fed into the VQ-VAE decoder to synthesize the final image.

###### B.2 Blossom

The core principle of the Blossom algorithm is to iteratively approach the optimal matching by dynamically handling odd-length cycle structures within the graph. Its key steps are as follows:

- • Blossom Shrinking: When the algorithm verifies an odd cycle, it contracts the cycle into a super vertex, preserving the connections between the cycle and external vertices, thereby simplifying the complex structure into a recursively manageable subgraph.
- • Augmenting Path Search: The current matching is expanded by traversing a path that alternates between matched and unmatched edges. During each expansion, the matching status of the edges on the path is flipped to increase the total weight.
- • Dual Variable Adjustment: Utilizing the duality theory of linear programming, the potentials of vertices and odd sets are adjusted to ensure that each operation converges toward maximizing the total weight.

The pseudocode of the Blossom Algorithm is shown in Algorithm 1. Algorithm 1: Blossom Algorithm

Input: Graph G = (V,E), edge weights w : E → R Output: Maximum-weight perfect matching M ⊆ E // Initialize

- 1 M ← ∅
- 2 y(v) ← 21 maxe∈δ(v) w(e) for all v ∈ V

- 3 B ← ∅
- 4 while M is not perfect do

- 5 Search for augmenting paths via BFS/DFS // Build alternating trees
- 6 if any odd-length cycle B found then // Blossom Shrinking

- 7 Contract B into super-node b
- 8 Update B ← B ∪ {b}
- 9 Adjust dual variables y and zB for b // Maintain LP feasibility

- 10 if augmenting path P found then // Augment matching

- 11 M ← M ⊕ P // Symmetric difference
- 12
- 13 Expand blossoms in B along P // Restore original graph
- 14
- 15 Reset search structures // Dual Variable Adjustment

- 16 Compute δ = min{slack(e) | e ∈ E}
- 17 Update y(v) ← y(v) ± δ and zB ← zB + 2δ // Converge to optimality

- 18 return M

###### B.3 Watermark Verification on Cropped Image

In the Figure 9, we show watermark verification process on a cropped image. As an example, with an 8 × 8 input image and using a patch side length of 2, by traversing the first image patch, the watermark in the cropped image can be successfully verified with at most 2 × 2 checks.

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

1 2 3 4

1

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

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

(a) Watermarked image (b) Cropped image (c) Traversing the first patch (d) Traversal: Step 1

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

2

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

(e) Traversal: Step 2

| |[Figure 785]<br><br>[Figure 786]| | |
|---|---|---|---|
| |[Figure 787]<br><br>[Figure 788]| | |
| |[Figure 789]<br><br>[Figure 790]<br><br>[Figure 791]<br><br>[Figure 792]| | |
| | |[Figure 793]<br><br>[Figure 794]| |

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

3 4

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

[Figure 883]

[Figure 884]

[Figure 885]

[Figure 886]

[Figure 887]

[Figure 888]

[Figure 889]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

[Figure 894]

[Figure 895]

[Figure 896]

[Figure 897]

[Figure 898]

[Figure 899]

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

[Figure 904]

[Figure 905]

[Figure 906]

[Figure 907]

[Figure 908]

[Figure 909]

[Figure 910]

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

[Figure 915]

[Figure 916]

[Figure 917]

[Figure 918]

[Figure 919]

[Figure 920]

[Figure 921]

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

[Figure 927]

[Figure 928]

[Figure 929]

[Figure 930]

[Figure 931]

[Figure 932]

[Figure 933]

[Figure 934]

[Figure 935]

[Figure 936]

[Figure 937]

[Figure 938]

[Figure 939]

[Figure 940]

[Figure 941]

[Figure 942]

[Figure 943]

[Figure 944]

[Figure 945]

[Figure 946]

[Figure 947]

[Figure 948]

[Figure 949]

[Figure 950]

[Figure 951]

[Figure 952]

[Figure 953]

[Figure 954]

(f) Traversal: Step 3 (g) Traversal: Step 4 (h) Verify watermark

Figure 9: Visualization of the traversal process for watermark verification on the cropped image.

#### C Experimental Details

###### C.1 Details About Evaluation Metrics

FID For text-to-image tasks, we generate 5,000 images to evaluate the Fréchet Inception Distance (FID) score [12] on the MS-COCO-2017 training dataset. For class-conditioned image generation tasks, we generate 10,000 images to evaluate the FID score on the ImageNet-1k validation dataset.

CLIP Score We use OpenCLIP-ViT model [4] to compute the CLIP score [23] between generated images and their corresponding text prompts. For class-conditioned generation, we use “a photo of category” as the input.

C.2 Details of the Threshold for Watermark Determination

For 256 × 256 and 384 × 384 resolutions, we select a green index rate of 0.615 near the 99.9% confidence level as the determination threshold. For 512 × 512 resolution, we choose a green index rate of 0.60 near the 99.99% confidence level as the determination threshold. Regarding cropping attacks, since the image is reduced to approximately 50% of its original size, we use a higher confidence level to detect the watermark. Specifically, for 512 × 512 resolution, we use a green index rate of 0.65 as the determination threshold, while for 256 × 256 and 384 × 384 resolutions, we adopt 0.7 as the determination threshold.

D More Experimental Results

###### D.1 Green Index Generation

We explored the possibility of generating images using only the green indices from the codebook, referring to this variant as GreenGen. As shown in Figure 10, the watermarked images generated by GreenGen exhibit significant differences compared to the watermark-free images. The quantitative results are shown in Table 2. GreenGen differs significantly from the watermarked image at the pixel level. Although GreenGen achieves a CLIP score similar to that of IndexMark, its performance in terms of FID is not as good as IndexMark. This result indicates that there is a substantial amount of redundancy in the codebook, and our method effectively leverages this redundancy to achieve better watermark embedding while maintaining image quality and content integrity.

Diffusion W/o Watermark ROBIN AR W/o Watermark IndexMark Many bunches of bright yellow bananas hanging on display.

Green index generation

A table with plates of food with drinks in the middle.

[Figure 1038]

[Figure 1039]

[Figure 1043]

[Figure 1044]

[Figure 1048]

[Figure 1049]

[Figure 1053]

[Figure 1054]

Diffusion W/o Watermark ROBIN AR W/o Watermark IndexMark Many bunches of bright yellow bananas hanging on display.

[Figure 1058]

[Figure 1062]

[Figure 1066]

[Figure 1070]

[Figure 1074]

[Figure 1078]

[Figure 1082]

[Figure 1086]

[Figure 1090]

[Figure 1094]

A table with plates of food with drinks in the middle.

[Figure 1098]

[Figure 1102]

[Figure 1106]

[Figure 1110]

[Figure 1114]

[Figure 1118]

[Figure 1122]

[Figure 1126]

[Figure 1130]

[Figure 1134]

AR W/o Watermark IndexMark Many bunches of bright yellow bananas hanging on display.

GreenGen

[Figure 1135]

[Figure 1136]

[Figure 1137]

[Figure 1138]

[Figure 1139]

[Figure 1140]

[Figure 1141]

[Figure 1142]

[Figure 1143]

[Figure 1144]

[Figure 1145]

[Figure 1146]

[Figure 1147]

[Figure 1148]

[Figure 1149]

[Figure 1150]

[Figure 1151]

[Figure 1152]

[Figure 1153]

[Figure 1154]

[Figure 1155]

[Figure 1156]

[Figure 1157]

[Figure 1158]

[Figure 1159]

[Figure 1160]

[Figure 1161]

[Figure 1162]

[Figure 1163]

[Figure 1164]

A table with plates of food with drinks in the middle.

[Figure 1165]

[Figure 1166]

[Figure 1167]

[Figure 1168]

[Figure 1169]

[Figure 1170]

[Figure 1171]

[Figure 1172]

[Figure 1173]

[Figure 1174]

[Figure 1175]

[Figure 1176]

[Figure 1177]

[Figure 1178]

[Figure 1179]

[Figure 1180]

[Figure 1181]

[Figure 1182]

[Figure 1183]

[Figure 1184]

[Figure 1185]

[Figure 1186]

[Figure 1187]

[Figure 1188]

[Figure 1189]

[Figure 1190]

[Figure 1191]

[Figure 1192]

[Figure 1193]

[Figure 1194]

- Figure 10: GreenGen vs. IndexMark. GreenGen generates autoregressive images by removing red indices from the codebook and using only green indices, resulting in significant differences between the watermarked images and non-watermarked ones. In contrast, IndexMark achieves smaller differences through a match-then-replace method.

Table 2: Comparison results of image quality between IndexMark and GreenGen.

Model Method PSNR ↑ SSIM ↑ MSSIM ↑ CLIP ↑ FID ↓

MSCOCO Dataset LlamaGen (AR) (256 × 256)

W/o watermark ∞ 1.000 1.000 0.328 26.55 GreenGen 9.76 0.267 0.111 0.326 26.35 IndexMark 23.54 0.838 0.930 0.326 24.73

W/o watermark ∞ 1.000 1.000 0.282 54.57 GreenGen 10.11 0.280 0.129 0.281 54.51 IndexMark 24.15 0.838 0.930 0.281 54.35

LlamaGen (AR) (512 × 512)

ImageNet Dataset LlamaGen (AR) (256 × 256)

W/o watermark ∞ 1.000 1.000 0.289 15.08 GreenGen 9.46 0.186 0.106 0.288 15.30 IndexMark 23.86 0.738 0.903 0.288 13.89

W/o watermark ∞ 1.000 1.000 0.287 12.65 GreenGen 9.454 0.230 0.131 0.286 12.46 IndexMark 25.45 0.783 0.913 0.286 11.81

LlamaGen (AR) (384 × 384)

###### D.2 Robustness Experiment without the Index Encoder

Under lower watermark-verification confidence thresholds, we removed the Index Encoder (w/o IE) and conducted robustness experiments. As shown in Table 3, even at low confidence settings, the model without the Index Encoder maintains strong robustness, thereby reducing training costs for users with less stringent security requirements.

Table 3: Comparison of ACC across different watermarking methods under various attacks. Clean indicates watermark verification results on unaltered images, while Avg represents the average accuracy across all attack scenarios.

Model Method Clean Blur Noise JPEG Bright Erase Crop Avg

MSCOCO Dataset LlamaGen (AR) (256 × 256)

IndexMark w/o IE 1.000 0.972 0.990 0.970 0.974 0.997 0.917 0.974

IndexMark 1.000 0.991 0.995 0.978 0.988 0.997 0.998 0.992 LlamaGen (AR) (512 × 512)

IndexMark w/o IE 1.000 0.969 0.992 0.980 0.981 0.992 0.939 0.978 IndexMark 1.000 1.000 0.998 0.971 1.000 1.000 0.918 0.983

ImageNet Dataset LlamaGen (AR) (256 × 256)

IndexMark w/o IE 1.000 1.000 1.000 1.000 0.995 1.000 0.996 0.998

IndexMark 1.000 1.000 1.000 1.000 0.998 1.000 0.998 0.999 LlamaGen (AR) (384 × 384)

IndexMark w/o IE 1.000 0.999 0.999 1.000 0.994 0.999 0.905 0.985 IndexMark 1.000 1.000 1.000 1.000 0.998 1.000 0.993 0.998

###### D.3 More Qualitative Results

## W/o Watermark IndexMark

Plate of food with gravy on mesh table with knife.

[Figure 1195]

[Figure 1196]

[Figure 1197]

[Figure 1198]

[Figure 1199]

[Figure 1200]

[Figure 1201]

[Figure 1202]

[Figure 1203]

[Figure 1204]

[Figure 1205]

[Figure 1206]

[Figure 1207]

[Figure 1208]

[Figure 1209]

[Figure 1210]

[Figure 1211]

[Figure 1212]

A black and white chicken is walking through tall plants.

[Figure 1213]

[Figure 1214]

[Figure 1215]

[Figure 1216]

[Figure 1217]

[Figure 1218]

[Figure 1219]

[Figure 1220]

[Figure 1221]

[Figure 1222]

[Figure 1223]

[Figure 1224]

[Figure 1225]

[Figure 1226]

[Figure 1227]

[Figure 1228]

[Figure 1229]

[Figure 1230]

A cat curled up in a box with a Pirates Hat on.

[Figure 1231]

[Figure 1232]

[Figure 1233]

[Figure 1234]

[Figure 1235]

[Figure 1236]

[Figure 1237]

[Figure 1238]

[Figure 1239]

[Figure 1240]

[Figure 1241]

[Figure 1242]

[Figure 1243]

[Figure 1244]

[Figure 1245]

[Figure 1246]

[Figure 1247]

[Figure 1248]

- Figure 11: More qualitative comparison results between non-watermarked images and IndexMark watermarked images.

### W/o Watermark IndexMark

A gigantic black bear roams around with his head hanging low.

[Figure 1249]

[Figure 1250]

[Figure 1251]

[Figure 1252]

[Figure 1253]

[Figure 1254]

[Figure 1255]

[Figure 1256]

[Figure 1257]

[Figure 1258]

[Figure 1259]

[Figure 1260]

[Figure 1261]

[Figure 1262]

[Figure 1263]

[Figure 1264]

[Figure 1265]

[Figure 1266]

A bunch of fruit sits in front of a portrait.

[Figure 1267]

[Figure 1268]

[Figure 1269]

[Figure 1270]

[Figure 1271]

[Figure 1272]

[Figure 1273]

[Figure 1274]

[Figure 1275]

[Figure 1276]

[Figure 1277]

[Figure 1278]

[Figure 1279]

[Figure 1280]

[Figure 1281]

[Figure 1282]

[Figure 1283]

[Figure 1284]

A cat lying in the sun on a table.

[Figure 1285]

[Figure 1286]

[Figure 1287]

[Figure 1288]

[Figure 1289]

[Figure 1290]

[Figure 1291]

[Figure 1292]

[Figure 1293]

[Figure 1294]

[Figure 1295]

[Figure 1296]

[Figure 1297]

[Figure 1298]

[Figure 1299]

[Figure 1300]

[Figure 1301]

[Figure 1302]

- Figure 12: More qualitative comparison results between non-watermarked images and IndexMark watermarked images.

