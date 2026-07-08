# arXiv:2605.25969v1[cs.CL]25May2026

## Triplet-Block Diffusion RWKV

### Ke Lin*

William & Mary

leonard.keilin@gmail.com

### Yiyang Luo*

HKUST

yluodq@connect.ust.hk

### Zhaolong Su

Cornell

zs494@cornell.edu

### Yunya Song

HKUST

yunyasong@ust.hk

### Anyi Rao

HKUST

anyirao@ust.hk

### Abstract

Causal Transformer language models suffer from strictly sequential decoding and a quadratic per-step attention cost. While lineartime causal models and discrete diffusion models each address these weaknesses, their integration remains inherently inconsistent: diffusion requires bidirectional attention, while causal models are unidirectional. To unify these architectures, we propose B3D-RWKV, a diffusion RWKV variant that integrates the model’s O(L) inference efficiency with parallel, bidirectional discrete-diffusion through a triplet-block layout method. B3D-RWKV7.2B reaches comparable accuracy on an 8task suite versus existing models while significantly outperforming baselines in decoding throughput with an average of 1.6× speedup. Code is available at https://github.com/ leonardodalinky/B3D-RWKV.

### 1 Introduction

Large language models (LLMs) have advanced rapidly under the dominance of the strictly causal Transformer architecture (Vaswani et al., 2017), yet the left-to-right design of most modern decoders introduces two structural limitations: sequential decoding, which prevents parallelization, and quadratic attention costs, which make longcontext inference expensive. These drawbacks have driven the development of alternative architectures designed to challenge the Transformer’s dominance: (1) Discrete-diffusion language models (Nie et al., 2025; Bie et al., 2025; Ye et al., 2025; Gong et al., 2024) avoid sequential decoding, instead denoising token blocks in parallel using bidirectional attention (Arriola et al., 2025). (2) The RWKV family (Peng et al., 2023, 2024, 2025) reformulates the classical Recurrent Neural Network (RNN) with attention-like channel mixing to obtain O(L) inference at Transformer-level quality.

*Equal Contribution.

This motivates us to combine these alternative architectures to improve generation efficiency over standard Transformers. However, using a strictly causal backbone for diffusion language models presents an architectural mismatch: diffusion requires bidirectional attention, while causal models are unidirectional.

To achieve this combination, we introduce a triplet-block layout method that converts a causal RNN-style language model into a block-diffusion language model without altering the backbone. Each logical generation block of size B appears three times consecutively in a training sample: a masked copy b1, an identical masked copy b2 on which the denoising loss is computed, and a clean ground-truth copy b3 that refreshes the recurrent state before the next block. Because the backbone model reads strictly left-to-right, the hidden state arriving at any masked position of b2 has already absorbed every unmasked token of b1, so b2 gains pseudo-bidirectional access to its own unmasked context on a strictly causal model.

Our contributions are as follows:

- • We release B3D-RWKV-7.2B, the first diffusionstyle linear-time RNN language model trained at the 7B scale with the mask-prediction objective. We train the model using our triplet-block diffusion framework, which integrates parallel token selection into the RWKV-7 backbone without modifying its original parameters.
- • We provide a comprehensive comparison between our model and other strictly causal language models on an 8-task suite. We also demonstrate that our 7.2B model matches the reasoning capabilities of the RWKV-7 baseline while achieving 1.6× the decoding throughput at comparable generation lengths.

###### (a) Triplet-Block Layout (Training)

###### (b) Per-Block Iterative Denoising (Inference)

Prefix

M M M M M M M M

Step

(Masked) (Lossable Masked) (Clean)

CommittedPrefix

Denoising Logic

Tokens

M M M M

If , commit token

M M M M

Strictly Causal LM Backbone (RNN/SSM)

Fallback: Commit top-1 token in edge cases

Causal State Propagation

Append to Prefix

Step (Final)

Next Block

Pseudo-Bidirectional Context

Figure 1: (a) Triplet-block layout for diffusion training on a strictly causal LM. Each logical block i unfolds left to right as three contiguous physical blocks: a masked copy b(1i), an identical lossable masked copy b(2i), and a clean ground-truth copy b(3i) that refreshes the recurrent hidden state for block i + 1. (b) Per-block iterative denoising at inference. At each step, the sampler commits every position whose top-1 probability exceeds τ. The loop terminates when every position is committed; the now-clean block is appended to c, and the next logical block begins.

### 2 Related Work

Discrete-diffusion and masked language models. The thread traces back to BERT-style maskedlanguage pretraining (Devlin et al., 2019) and Mask-Predict’s parallel decoder (Ghazvininejad et al., 2019), which MaskGIT (Chang et al., 2022) carried to image transformers with a confidencethresholded commit schedule that almost every later masked generator reuses. The discretediffusion family proper was introduced by D3PM (Austin et al., 2021), with SEDD (Lou et al., 2023), MDLM (Sahoo et al., 2024), and MD4 (Shi et al., 2024) reformulating and simplifying the absorbingstate objective. More recent scaled-up systems, including LLaDA (Nie et al., 2025), LLaDA 2.x (Bie et al., 2025), Dream 7B (Ye et al., 2025), DiffuLLaMA (Gong et al., 2024), Block Diffusion (Arriola et al., 2025), WeDLM (Liu et al., 2025), and Nemotron-Labs-Diffusion (Fu et al., 2026), combine these objectives with instruction tuning and parallel decoding. The concurrent DiffuMamba (Singh et al., 2026) is the closest design point to ours and the only prior recipe that pairs a maskeddiffusion objective with a linear-time backbone, but it does so by architecturally modifying Mamba into a bidirectional block and so trains from scratch at the 1.3B scale on DCLM (Li et al., 2024).

Linear-time recurrent and state-space backbones. A parallel thread has produced strictly causal, linear-time alternatives to softmax atten-

tion: the RWKV family from RWKV-4 (Peng et al., 2023) through Eagle/Finch (Peng et al., 2024) to RWKV-7 (Peng et al., 2025); the selective state-space models (SSM) Mamba and Mamba-2 (Gu and Dao, 2023; Dao and Gu, 2024); RetNet (Sun et al., 2023); Gated Linear Attention (Yang et al., 2023); and the Hyena Hierarchy (Poli et al., 2023). These backbones report perplexity parity with quadratic-attention Transformers at large wallclock and memory savings; to our knowledge, none have been combined with a discrete-diffusion training objective at a large scale.

### 3 Method

To enable diffusion paradigm within strictly causal language models, we propose a triplet-block layout for efficient training and inference. This method comprises a triplet-block layout (§3.1) and a blockwise iterative denoising sampler (§3.2). Implementation details of training and inference are provided in Appendix A.1 and A.2.

#### 3.1 Triplet-block layout

Let the training context length be L, and the logical generation block size be B. We partition each training sample into N = L/B contigu-

- ous logical blocks. For each logical block index i ∈ {1,...,N}, denote the clean ground-truth tokens by g(i) ∈ VB. Each logical block is then laid
- out as the concatenation of three physical blocks of

General Tasks Math & Science MMLU(5) ARC-C(0) ARC-E(0) PIQA(0) RACE(0) GSM8K(8) MATH(4) GPQA(5)

Model

Causal LM LLaMA3-8B 66.6 53.6 81.1 79.8 41.9 78.9 41.1 35.5

Qwen3-8B 76.9 56.6 81.7 79.1 - 89.9 60.8 44.4 RWKV-7-7.2B 65.1 55.5 83.8 80.7 43.5 83.9 48.8 30.8

Diffusion LM LLaDA-8B 65.9 47.5 71.8 74.8 38.7 70.9 30.7 30.4 Dream-7B 69.5 59.8 83.9 75.8 44.7 77.2 39.6 36.6

Strictly Causal Diffusion LM

DiffuMamba - 28.3 49.1 62.6 - - - B3D-RWKV-7.2B 64.8 61.6 79.3 73.5 49.7 71.5 23.8 25.6

Table 1: Benchmark results on the 8-task suite. The number of few-shot examples for each benchmark is indicated in brackets. Results outperforming the RWKV baseline are bolded, while comparable results are underlined.

length B (Fig. 1(a):

b(1i) masked copy

∥ b(2i)

∥ b(3i)

. (1)

masked copy (lossable)

clean copy

The two masked copies b(1i) and b(2i) are identical: they share the same mask pattern m(i) ∈ {0,1}B, replacing masked positions with [MASK] and re-

taining g(i) elsewhere. The clean copy b(3i) is also identical to g(i). Let ℓ(ji) ∈ {0,1} be the lossable flag, π(i,j) the physical position of the j-th token of b(2i), and pij(·)≜pθ(· | x<π(i,j)) the next-token distribution there. Writing S ={(i,j):m(ji)ℓ(ji) = 1} for the supervised positions and Nv = |S|, the training loss is the mean cross-entropy on S:

1 Nv

log pij gj(i) . (2)

LCE(θ) = −

(i,j)∈S

Following the Confidence-Aware Parallel training scheme of LLaDA-2.0 (Bie et al., 2025), we further sharpen pij on supervised positions that are already correctly predicted, so that the inference-time threshold sampler (§3.2) can commit more positions per denoising step. Let gˆj(i) = arg maxv pij(v) be the model’s current top-1, H(p) = − v p(v)log p(v) the entropy, C = {(i,j) ∈ S : gˆj(i) = gj(i)} the gated subset, and Nc=|C|:

1 Nc

H pij . (3)

LCAP(θ) =

(i,j)∈C

The membership in C is computed without gradient, so the entropy flows only on the selected subset. The total objective is

L(θ) = LCE(θ) + λCAP LCAP(θ). (4)

Pseudo-bidirectional access. Fix any masked position with block-local index j∈{0,...,B−1}

in b(2i), whose physical position π(i,j) sits after every token of b(1i). Two complementary streams of context are visible there. (i) Left context. Within b(2i) itself, the unmasked tokens at indices k<j lie to the left of π(i,j) and supply the standard causal left context that a vanilla decoder would use. (ii) Right context via b(1i). Because b(1i) has been processed in full before π(i,j) and carries the same mask pattern m(i), its unmasked tokens at every block-local index k are already absorbed into the hidden state at π(i,j). The union of streams (i) and (ii) is exactly the set of unmasked tokens of the logical block, so position j receives full bidirectional conditioning over block i while the backbone still reads the sample strictly left-to-right (Fig. 1). Although the training context is 3× the original length, the architecture of strictly causal models remains more computationally efficient than standard attention-based models.

Requirements and universal claims. The construction in Eq. (2) and the pseudo-bidirectionalaccess argument depend on only two properties of the backbone pθ:

- • (R1) Strict causality: the predictive distribution at any position depends solely on positions strictly to its left.
- • (R2) Forward-propagating state: an internal state that allows the predictive distribution at positions

b(2i) to access unmasked tokens from b(1i).

Every member of the linear-time backbone family currently in use, such as RWKV-v4 through v7, Mamba and Mamba-2, RetNet, Gated Linear Attention, and Hyena, satisfies (R1) and (R2) by construction. Standard causal Transformers also

satisfy these, but their triple sequence-length cost is unattractive. Therefore, the triplet construction defines a universally no-architectural-change training recipe over the class of strictly causal backbones.

#### 3.2 Inference: block-wise iterative denoising

At inference, the model generates one logical block at a time with only 2 replicates of physical blocks. Let c denote the prefix of already-committed tokens. For each new block, the sampler initializes to an all-MASK input of length B and runs at most T denoising iterations. At each iteration the sampler forwards c concatenated with the current best guess of the block, reads the top-1 probability pj at each still-masked position, and commits any position with pj > τ; a low-confidence fallback commits the top-kmin positions whenever fewer than kmin clear the threshold, guaranteeing strictly positive progress per iteration. Appendix A.2 records the per-iteration loop and Figure 1 summarizes it.

4 Experiments

#### 4.1 Setup

The backbone is the public RWKV-7-g1f-7.2B (Peng et al., 2025) causal-LM checkpoint. Training data is the mixture of TÜLU3 SFT dataset (Lambert et al., 2024) and curated trajectories of GLM-

###### 5.1 and Claude Opus 4.6. In the first training round, we set the triplet layout (§3.1) to B = 32 and N = 64, expanding 2,048-token samples into 6,144-token sequences for 1.8 epochs. In the sec-

- ond round, we increase the layout to N = 256, expanding 8,192-token samples into 24,576-token

sequences for 0.2 epochs. The λCAP is set to 0.5. The model is trained on 8× H100 80GB SXM GPUs. The full setup is in Appendix A and B.

#### 4.2 Benchmarks

We evaluate B3D-RWKV-7.2B on an 8-task suite: MMLU (Hendrycks et al., 2020), ARC-Challenge, ARC-Easy (Clark et al., 2018), PIQA (Bisk et al., 2019), RACE (Lai et al., 2017), GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021) and GPQA (Rein et al., 2023). For a fair comparison, we restrict baselines to backbones of comparable parameter scale released in roughly the same time window as RWKV-7.

- Table 1 presents downstream performance on

general and math reasoning tasks. B3D-RWKV performs comparably to other diffusion LMs of similar scale and matches the performance of the RWKV-7

Decoding Throughput w.r.t. Context Length

Throughput(Tokens/Second)

200

150

100

LLaDA-8B

50

RWKV-7-7.2B

B3D-RWKV-7.2B (Ours)

0

1 2 4 8 16 32 64 128 256 512

Context Length (K tokens)

Figure 2: Inference throughput of LLaDA-8B, RWKV7.2B, and B3D-RWKV-7.2B on an H100 80GB GPU.

baseline. Notably, our method outperforms others on benchmarks like ARC-C and RACE, likely due to pseudo-bidirectional perception-enhancing reasoning capabilities. Conversely, parallel decoding may slightly reduce math reasoning accuracy, as these problems involve highly complex structures. For example, MATH is graded by a LaTeXlevel answer verifier that demands exact symbolic and numerical agreement, leaving no partial credit for minor local errors, which is exactly the failure mode that parallel decoding is most exposed to.

These results show that B3D-RWKV achieves comparable or superior performance on simpler tasks, but experiences an acceptable drop on complex structural problems, likely due to parallel decoding issues of diffusion language models.

#### 4.3 Throughput

Figure 2 compares the inference throughput of LLaDA-8B and our model against an RWKV-7 baseline across context lengths from 1K to 512K. LLaDA-8B utilizes Fast-dllm (Wu et al., 2026) for efficient inference, with batch size fixed at 1, block size B = 32, and T = 32 diffusion steps. The commit threshold is set to 0.9 in our settings. Our model consistently achieves an average of 1.6× higher throughput than RWKV-7 while maintaining nearly identical performance. Adjusting sampling parameters achieves a 2.02× speedup with a slight drop in quality. More details of throughput is in Appendix C.

### 5 Conclusion

We propose a triplet-block layout training method to adapt strictly causal language models into generative diffusion language models, and it requires no architectural changes. B3D-RWKV achieves

a general 1.6× throughput compared to the original RWKV model, while maintaining comparable performance to existing models, offering an efficient way to transform pre-trained causal language models into diffusion language models.

### Limitations

Universality argued structurally; demonstrated on one backbone. The universality claim is structural: the architectural requirements (R1, R2 in §3.1) are stated precisely, and every member of the linear-time backbone family we cite satisfies them by construction. Due to computational constraints, we empirically validate the recipe using a single 7.2B-parameter RWKV-7 backbone. Empirical confirmation on smaller RWKV-v7 checkpoints, on other RWKV variants (Peng et al., 2023, 2024), and on non-RWKV linear-time backbones, such as Mamba, is left to future work.

- 3× physical-sequence cost. The triplet layout applies a multiplicative 3× factor to the physical sequence length per logical block in exchange for pseudo-bidirectional access. RWKV-7’s linearin-length complexity (Peng et al., 2025) makes this approach feasible, whereas the quadratic cost of Transformers has previously restricted discretediffusion training to short context lengths.

Small-scale SFT data and no RL alignment. We continued training using only the TÜLU 3 SFT mixture and a curated set of reasoning trajectories (Section 2), totaling 4.9B tokens. Additionally, we run neither large-scale further pretraining nor any subsequent reinforcement-learning alignment stage. Compared with the trillion-token corpus that produced the parent RWKV-7 “Goose” checkpoint, this is a narrow and stylistically biased distribution, so a degree of catastrophic forgetting on capabilities the base model originally acquired from broad pretrain data is essentially unavoidable, and likely accounts for part of the accuracy regression we observe on a subset of evaluation tasks. We expect that scaling the diffusion-style post-training corpus and adding an RL alignment stage on top of B3D-RWKV would recover, and likely exceed, the parent checkpoint’s accuracy on the affected tasks; both are left to future work.

More complicated scenarios. Due to computational constraints, we have not specifically optimized this model for scenarios like tool calling or

coding. However, the pretrained RWKV’s inherent capabilities allow for success in some simple coding tasks, as demonstrated in Appendix D. We intend to improve this in future work.

### Ethical Considerations

B3D-RWKV is initialized from a publicly released causal language model, and our fine-tuning only changes its architectural behavior, not its safety properties. We did not run any additional alignment or content filtering, so B3D-RWKV inherits whatever biases and inaccuracies already exist in its base checkpoint and pre-training corpus. We recommend reviewing its outputs before using B3DRWKV in any user-facing system or in settings where factual accuracy matters. Furthermore, the training dataset is publicly available and we do not impose any security check upon them.

### References

Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhi-Hong Qi, Jiaqi Han, S. Sahoo, and V. Kuleshov. 2025. Block diffusion: Interpolating between autoregressive and diffusion language models. International Conference on Learning Representations.

Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. 2021. Structured denoising diffusion models in discrete state-spaces. Neural Information Processing Systems.

Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, Chengxi Li, Chongxuan Li, Jianguo Li, Zehuan Li, Huabin Liu, Lin Liu, Guoshan Lu, Xiaocheng Lu, Yuxin Ma, and 12 others. 2025. Llada2.0: Scaling up diffusion language models to 100b. arXiv.org.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2019. Piqa: Reasoning about physical commonsense in natural language. In AAAI Conference on Artificial Intelligence.

Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and W. Freeman. 2022. Maskgit: Masked generative image transformer. Computer Vision and Pattern Recognition.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv.org.

K. Cobbe, Vineet Kosaraju, Mo Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher

Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv.org.

Tri Dao and Albert Gu. 2024. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. International Conference on Machine Learning.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. North American Chapter of the Association for Computational Linguistics.

Yonggan Fu, Lexington Whalen, Abhinav Garg, Chengyue Wu, Maksim Khadkevich, Nicolai Oswald, Enze Xie, Daniel Egert, Sharath Turuvekere Sreenivas, Shizhe Diao, Chenhan Yu, Ye Yu, Weijia Chen, Sajad Norouzi, Shiyi Lan, Ligeng Zhu, Jin Wang, Jindong Jiang, Morteza Mardani, and 6 others. 2026. Nemotron-labs-diffusion: A tri-mode language model unifying autoregressive, diffusion, and self-speculation decoding. Technical report, NVIDIA. Technical report.

Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and Luke Zettlemoyer. 2019. Mask-predict: Parallel decoding of conditional masked language models. Conference on Empirical Methods in Natural Language Processing.

Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, Hao Peng, and Lingpeng Kong. 2024. Scaling diffusion language models via adaptation from autoregressive models. International Conference on Learning Representations.

Albert Gu and Tri Dao. 2023. Mamba: Linear-time sequence modeling with selective state spaces. Conference on Language Modeling.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, D. Song, and J. Steinhardt. 2020. Measuring massive multitask language understanding. International Conference on Learning Representations.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, D. Song, and J. Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. In NeurIPS Datasets and Benchmarks.

Diederik P. Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. International Conference on Learning Representations.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and E. Hovy. 2017. Race: Large-scale reading comprehension dataset from examinations. Conference on Empirical Methods in Natural Language Processing.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri,

Shane Lyu, and 1 others. 2024. Tulu 3: Pushing frontiers in open language model post-training. Conference on Language Model.

Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, Saurabh Garg, Rui Xin, Niklas Muennighoff, Reinhard Heckel, Jean Mercat, Mayee Chen, Suchin Gururangan, Mitchell Wortsman, Alon Albalak, and 40 others. 2024. Datacomplm: In search of the next generation of training sets for language models. Neural Information Processing Systems.

Aiwei Liu, Minghua He, Shaoxun Zeng, Linhao Zhang, Chuhan Wu, Wei Jia, Yuan Liu, Yang Yu, Xiao Zhou, and Jie Zhou. 2025. Wedlm: Reconciling diffusion language models with standard causal attention for fast inference. arXiv preprint arXiv:2512.22737.

Aaron Lou, Chenlin Meng, and Stefano Ermon. 2023. Discrete diffusion modeling by estimating the ratios of the data distribution. International Conference on Machine Learning.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Jirong Wen, and Chongxuan Li. 2025. Large language diffusion models. Neural Information Processing Systems.

Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, G. Kranthikiran, Xingjian Du, Xuming He, Haowen Hou, Przemyslaw Kazienko, Jan Koco´n, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, and 13 others. 2023. Rwkv: Reinventing rnns for the transformer era. Findings of Conference on Empirical Methods in Natural Language Processing.

Bo Peng, Daniel Goldstein, Quentin Anthony, Alon Albalak, Eric Alcaide, Stella Biderman, Eugene Cheah, Teddy Ferdinan, Haowen Hou, Przemyslaw Kazienko, G. Kranthikiran, Jan Koco’n, Bartlomiej Koptyra, Satyapriya Krishna, Ronald McClelland, Niklas Muennighoff, Fares Obeid, Atsushi Saito, Guangyu Song, and 9 others. 2024. Eagle and finch: Rwkv with matrix-valued states and dynamic recurrence. Conference on Language Model.

Bo Peng, Ruichong Zhang, Daniel Goldstein, Eric Alcaide, Xingjian Du, Haowen Hou, Jiaju Lin, Jiaxin Liu, Janna Lu, William Merrill, Guangyu Song, Kaifeng Tan, Saiteja Utpala, Nathan Wilce, J. S. Wind, Tianyi Wu, D. Wuttke, and Christian ZhouZheng. 2025. Rwkv-7 "goose" with expressive dynamic state evolution. Conference on Language Modeling.

Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, S. Baccus, Y. Bengio, Stefano Ermon, and Christopher Ré. 2023. Hyena hierarchy: Towards larger convolutional language models. International Conference on Machine Learning.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. 2019. Zero: Memory optimizations toward training trillion parameter models. International Conference for High Performance Computing, Networking, Storage and Analysis.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2023. Gpqa: A graduate-level google-proof q&a benchmark. Conference on Language Model.

S. Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and V. Kuleshov. 2024. Simple and effective masked diffusion language models. Neural Information Processing Systems.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K. Titsias. 2024. Simplified and generalized masked diffusion for discrete data. Neural Information Processing Systems.

Vaibhav Singh, Oleksiy Ostapenko, Pierre-André Noël, Eugene Belilovsky, and Torsten Scholak. 2026. Diffumamba: High-throughput diffusion lms with mamba backbone. arXiv.org.

Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. 2023. Retentive network: A successor to transformer for large language models. arXiv.org.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and I. Polosukhin. 2017. Attention is all you need. Neural Information Processing Systems.

Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. 2026. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. International Conference on Learning Representations.

Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. 2023. Gated linear attention transformers with hardware-efficient training. International Conference on Machine Learning.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. 2025. Dream 7b: Diffusion large language models. arXiv.org.

### A Model Implementation A.1 Mask Sampling Rules

Following LLaDA (Nie et al., 2025), each logical block i independently samples a mask ratio r(i) ∼ Uniform[0,1], then draws ⌊r(i)B⌋ mask positions uniformly without replacement from the lossable subset; with probability 0.10 the ratio is overridden to 1.0 (full-mask augmentation) so the training distribution matches the inference distribution, where each new block begins fully masked. Two role-aware rules are necessary and we treat them as non-negotiable: force-mask EOS (the document-final EOS is in every sample’s mask set) and force-mask PAD inside the EOS-containing block (trailing PAD positions are extended to the mask). Both supervise the stopping decision and block a trailing-PAD shortcut that does not transfer to inference.

Per-block uniform mask ratio. Following LLaDA (Nie et al., 2025), each logical block i independently samples its mask ratio r(i) ∼ Uniform[rmin,rmax] with rmin = 0 and rmax = 1 in all our runs. The mask pattern m(i) ∈ {0,1}B is then drawn by sampling ⌊r(i)B⌋ positions uniformly without replacement from the lossable subset of block i.

Full-mask trick. To shrink the distribution gap between training and inference, where each new block begins fully masked, we override r(i) to 1.0 with probability 0.10 at sample time, so roughly

- one in ten logical blocks has every lossable position masked. This is the LLaDA full-mask augmentation (Nie et al., 2025) applied per-block rather than per-sample, and it is the only training-time hyperparameter we vary on the mask-sampling axis.

Force-mask EOS. Every document carries a document-final EOS token at g(i

⋆)

j⋆ for some (i⋆,j⋆), where i⋆ is the EOS-containing logical block. We force-include this position in m(i⋆) in every sample

- of every epoch. Without this rule, a uniform perblock mask ratio of r ∼ Uniform[0,1] supervises the EOS position in fewer than half of all samples and contributes only a small fraction of the total loss signal. Qualitatively, an unsupervised EOS produces an inference-time model that never stops generating, while force-masking the EOS yields well-calibrated stopping behavior.

Force-mask PAD inside the EOS-containing block. A document shorter than B tokens leaves

a tail of PAD symbols inside the EOS-containing logical block. Without intervention, the model learns the shortcut “MASK followed by visible PAD implies EOS”, a cue that does not transfer to inference because PAD is suppressed in the decoder. We therefore extend the forced mask to all PAD positions in the EOS-containing block. The model is then forced to predict the document-final EOS from the upstream content rather than from a trailingPAD shortcut.

Vocabulary slot reuse. The RWKV-world tokenizer (Peng et al., 2025) uses 65,530 real token slots out of its 65,536-padded embedding table. We re-purpose two of the unused trailing slots, setting ID 65,535 as MASK and ID 65,534 as PAD, while preserving ID 0 as the original EOS. The embedding table and output projection are therefore not extended; no new parameters are introduced for the diffusion training objective.

#### A.2 Inference Sampler

This appendix expands the per-iteration commit rule of §3.2.

Per-iteration commit rule. At each iteration t, the sampler forwards c concatenated with the current best guess of the block through the backbone, reads the top-1 probability pj at each stillmasked position j, and commits any position whose pj exceeds a fixed confidence threshold τ. If fewer than a floor kmin positions are committed in iteration t, the sampler falls back to committing the top-kmin most confident positions even when they sit below τ. This guarantees strictly positive progress per iteration. The rule is the LLaDA 2.0 (Bie et al., 2025) confidence-thresholdplus-low-confidence-fallback rule, which itself descends from the MaskGIT (Chang et al., 2022) confidence-thresholded commit schedule for nonautoregressive masked generators. Iteration t + 1 then conditions on the newly-committed tokens. The loop exits as soon as every position in the block is committed; the now-clean block is appended to c, and the next logical block begins.

#### A.3 Architecture and Model Layout

Table 2 lists the architectural configuration of B3D-RWKV-7.2B and the corresponding physicalsequence layout induced by the triplet block training scheme. All values match the public RWKV7G1f-7.2B checkpoint we initialize from; we do not modify the backbone architecture.

Throughput Accuracy

(a) Eﬀect of Sampling Steps

(b) Eﬀect of Commit Threshold τ

800

100

800

100

Throughput(Tokens/Second)

Throughput(Tokens/Second)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

700

700

80

80

###### Accuracy(%)

Accuracy(%)

600

600

60

60

500

500

40

40

400

400

20

20

300

300

200

0

200

0

8 16 24 32

0.3 0.5 0.7 0.9

Sampling Steps

Commit Threshold τ

Figure 3: Effect of (a) sampling steps and (b) commit threshold τ on decoding throughput and accuracy on ARC-E benchmark.

Metric Value Layers 32 Embedding dim 4,096 Head size 64 Number of heads 64 RWKV LoRA rank 128/128/96/480 Total parameters 7.2 B Vocabulary slots (padded) 65,536 Vocabulary slots (used) 65,530 MASK token ID 65,535 PAD token ID 65,534 Context length (training) 6,144 Block size B 32 Logical blocks per sample 64 Raw content len. per sample 2,048

- Table 2: Architecture and physical layout of B3DRWKV-7.2B. The triplet layout maps each 2,048-token raw content sample to a 6,144-token physical sequence consisting of 64 logical blocks of size B = 32, each unfolded as three contiguous physical blocks (b1,b2,b3).

### B Implementation Notes

This appendix records the precise distributedtraining configuration used by our single B3DRWKV-7.2B run, for reproducibility. The notes here are not claimed as a contribution; they allow exact reproduction of the training run and flag two configuration pitfalls that are easy to overlook in the DeepSpeed (Rajbhandari et al., 2019) + PyTorch-Lightning stack. Training takes a total of approximately 500 H100 hours using 8× H100 80GB SXM GPUs to complete two epochs.

Training Dataset. We trained the model on a 4.97 billion tokens mixture of the “allenai/tulu-3sft-mixture”, “Jackrong/GLM-5.1-Reasoning-1MCleaned”, and “angrygiraffe/claude-opus-4.6-4.7reasoning-8.7k” datasets on Huggingface Datasets.

Precision and gradient accumulation. The run uses bf16 mixed-precision activations and a fp32 master copy of the optimizer state. Gradient accumulation precision must be set explicitly to fp32 via the DeepSpeed configuration key data_types.grad_accum_dtype = fp32. The Adam optimizer (Kingma and Ba, 2014) master weights, first-moment buffer, and secondmoment buffer are all fp32 inside DeepSpeed’s BF16_Optimizer.

Inter-rank communication precision. Inter-rank communication uses fp32 via communication_data_type = fp32 in the DeepSpeed configuration. Default communicationbucket sizes (allgather_bucket_size, reduce_bucket_size) are kept at their 200 MB defaults; reducing them traded fragmentation for throughput in our exploratory tests.

Gradient clipping. The PyTorch-Lightning Trainer’s gradient_clip_val argument is silently ignored on the DeepSpeed strategy path. The same value must be propagated through the strategy’s configuration as strategy.config[”gradient_clipping”] for it to take effect; we set this to 0.5 matching the Trainer-side value, so both code paths agree.

Adam optimizer epsilon. We set the Adam (Kingma and Ba, 2014) optimizer epsilon to ϵ = 10−8. The upstream RWKV training-code default

#### Component Approx. peak (GB)

Model weights (bf16) 14.4 Gradients (bf16, ZeRO-2) 1.8 Adam state (fp32) 10.8 Activations ∼30 CUDA context overhead ∼1.5 Total observed 58–60

- Table 3: Single-GPU peak memory footprint (trainingtime) for the post-fix B3D-RWKV-7.2B configuration on an H100 80GB. Activations and the wkv scratchpad dominate the budget.

of 10−18 is intended for the small-scale fp32 setting and is not appropriate for the GPU FusedAdam kernel that DeepSpeed’s BF16 Optimizer dispatches to in our configuration; we therefore raise it to 10−8, the standard Adam default.

Parallelism. We run DeepSpeed (Rajbhandari et al., 2019) ZeRO Stage 2 w/o offload on 8× NVIDIA H100 80GB GPUs. Effective batch size is 4 batches per GPU ×8 GPUs ×4 gradientaccumulation steps = 128 samples per step.

Memory footprint (Training). Table 3 reports the observed single-GPU peak memory split for the post-fix configuration. The activation plus wkvscratchpad budget dominates and is the binding constraint for raising the micro-batch beyond 4 at the current context length.

Open-source License. We follow the Apache2.0 License from RWKV directly.

### C Performance

Throughput (Training). On 8× H100 80GB SXM GPUs, the post-fix training configuration sustains approximately 43,000 tokens per second at micro-batch 4 and context length 6,144.

Throughput (Inference). Across n = 6,284 requests sampled from a production-style workload, the mean decoding throughput is 222.1 tok/s, with per-request rates ranging from 74.7 to 785.4 tok/s (Table 4). The large standard deviation (124.6, about 56% of the mean) comes mostly from differences in prompt length and in how many tokens the speculative decoder commits per draft step.

Latency (Inference). End-to-end latency grows close to linearly with the prefilled context length, from 91 ms at 1K tokens to 45.8 s at 512K (Fig. 4).

Avg Std Min Max Throughput 222.1 124.6 74.7 785.4

Table 4: Per-request decoding throughput statistics over n = 6284 requests (tokens/second).

Decoding Latency w.r.t. Context Length

B3D-RWKV-7.2B (Ours)

- 102

- 103

- 104

Latency(ms)

1 2 4 8 16 32 64 128 256 512

Context Length (K tokens)

Figure 4: Latency in milliseconds by prefilled context length.

An increase in context length produces a corresponding linear increase in latency, which is what we expect from the RWKV-7 backbone since it has no quadratic attention to dominate at long context.

Sampling steps. More sampling steps trade throughput for accuracy (Fig. 3 (a)). At 8 steps, the model runs at 581 tok/s but only reaches 18.7% accuracy; at 32 steps, accuracy climbs to 79.3% while throughput drops to 213 tok/s. The largest single jump in accuracy happens between 16 and 24 steps (+30.8 points for a 26% throughput drop), which is roughly the point at which iterative denoising starts to produce coherent output.

Commit threshold. The commit threshold τ controls how confident the model must be before committing a draft token. At τ = 0.3 the decoder commits aggressively and reaches 772 tok/s, but accuracy collapses to 11.2% (Fig. 3 (b)). Raising τ to 0.9 recovers 79.3% accuracy at 213 tok/s, which is the same operating point reached by 32 sampling steps. Both knobs move along the same speed–accuracy frontier; we use τ = 0.9 as the default and treat lower values as a tuning knob when latency matters more than accuracy.

### D Samples

We include several samples generated by B3DRWKV-7.2B in the Appendix to demonstrate the model’s proficiency in solving both general and

complex tasks. The default system prompt is the standard “You are a helpful assistant”.

### E Declaration of AI Usage

We use Grammarly and Gemini to proofread our papers, and Claude Code for coding.

Sample: Knowledge User: Could you tell me the life story of Albert Einstein?

Assistant: <think>The user is asking for a detailed biography of Albert Einstein’s life. This is a straightforward request. I should cover his early life, his education, his scientific contributions, his challenges, and his later years. I need to present this in a clear, engaging way. Let me think about what aspects of Einstein’s life are most interesting. His early life in Munich, his time in Zurich, his work on the Einstein equation, his struggles with quantum mechanics, and his Nobel Prize are all key points. I should also mention his later work on unified field theory. I should avoid getting bogged down in the technical details of his equations and focus on the human story. The tension of his relationship with quantum mechanics is a great story to include, as it shows his character and the limits of his work. I’ll structure this chronologically, starting from his birth in 1879 and going to his death in 1955. I’ll include key events like his marriage, the Nobel Prize, and his later work. </think> Albert Einstein (1879-1955) was one of the most influential physicists of the 20th century. Born in Munich, Germany, on March 14, 1879, he was the son of a Jewish family. His father, Hermann, was a successful manufacturer of electrical goods, and his mother, Pauline, was a seammaker. Einstein showed an interest in mathematics from a young age.

#### Sample: Biology

User: To prevent desiccation and injury, the embryos of terrestrial vertebrates are encased within a fluid secreted by the: A. Amnion, B. Chorion, C. Allantois, D. Yolk sac. Which is the right answer?

Assistant: <think>The user is asking about the embryonic sacs in terrestrial vertebrates. The correct answer is A. Amnion. Let me think about this carefully. In terrestrial vertebrates, the embryo is enclosed in the amniotic sac, which contains the amniotic fluid. The amniotic fluid is produced by the amnion. The chorrion is the outermost layer of the embryonic sac, and the allantois is the waste absorption sac. The yolk sac is for nutrient absorption. The chorrion is the outermost layer of the amniotic sac, and the fluid is the amniotic sac. The amnion is the membrane that produces the amniotic fluid. The chorrion is the outermost layer of the amniotic sac. The allantois is a sac that develops from the embryo. So, the amnion is the structure that secretes the amniotic fluid. </think> The correct answer is **A. Amnion**.

Sample: Math User: Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages? Assistant: <think>The user wants to find out how many hours it will take Joy to read 120 pages. Joy reads 8 pages in 20 minutes. First, find the number of pages she reads per minute. Total pages = 8 pages in 20 minutes. Pages per minute = 8 pages / 20 minutes = 0.4 pages per minute. Next, find the number of minutes needed to read 120 pages. Total pages = 120 pages. Pages per minute = 8 pages. Total minutes = 120 pages / 8 pages/minute = 15 minutes. Then, convert minutes to hours. 15 minutes = 15 / 60 hours = 0.25 hours. Let me double check. Pages per minute = 8 / 20 = 0.4 Minutes per hour = 120 / 0.4 = 15 minutes. Hours = 15 / 60 = 0.25 hours. Alternatively: Pages per hour = 8 / 20 = 0.4 pages per minute. Pages per hour

= 8 / 20 = 0.4 pages per minute. Pages per hour = 0.4 * 60 = 24 pages per hour. Total hours = 120 / 24 = 5 hours. Both methods yield 5 hours. </think> It will take Joy **5** hours to read 120 pages.

Sample: Commonsense User: Where is the capital of US? Assistant: <think>The user is asking for the capital of the United States. This is a straightforward geography question. </think> The capital of the United States is Washington, D.C.

Sample: Translation User: Translate the following sentence to English: Hoy hace muy buen tiempo. Assistant: <think>The user wants me to translate a Spanish sentence into English. The sentence is "Hoy hace muy bu tiempo." "Hoy" = today, "hace" = makes, "muy buen" = very good, "tiempo" = "Hoy" = today, "weather" = weather Translation: "It makes very good weather today." </think> "It makes very good weather today.“

Sample: Multi-turn User: Calculate the result of 13 + 17. Assistant: The answer is **30**. User: Now multiply the previous result by 3. Assistant: <think>The user wants me to multiply the previous result by 3. The previous result was 30. 30 * 3 = 90. </think> The answer is **90**.

Sample: Coding User: Write a python function to calculate Fibonacci numbers. Assistant: <think>

We need to write a Python function to calculate Fibonacci numbers. The user says "Write a Python function to calculate Fibonacci numbers." That's straightforward. But we need to consider the context: The user likely wants a function that returns the nth Fibonacci number. Possibly they want an iterative or recursive approach. Since it's a simple request, we can provide a function that computes the nth Fibonacci number efficiently. We'll handle edge cases: n=0 returns 0, n=1 returns 1, maybe negative numbers? Typically Fibonacci is defined for non-negative integers. We'll assume n is a non-negative integer. We'll implement iterative for efficiency. We should output the code. The instruction: "Write a Python function to calculate Fibonacci numbers." So we just need to provide the function. Possibly include docstring and example usage. We'll write: def fibonacci(n): if n < 0:

raise ValueError("n must be non-negative") a, b = 0, 1 for _ in range(n):

a, b = b, a + b

return a That's it. </think> ```python def fibonacci(n: int) -> int:

"""Return the nth Fibonacci number. The Fibonacci sequence is defined as:

- F(0) = 0
- F(1) = 1 F(n) = F(n-1) + F(n-2) for n >= 1

Args: n (int): A non-negative integer representing the position in the sequence. Returns:

int: The nth Fibonacci number. Raises:

ValueError: If n is negative.""" if n < 0:

raise ValueError("n must be non-negative") a, b = 0, 1 for _ in range(n):

a, b = b, a + b return a

##### ```

