## FedRand: Enhancing Privacy in Federated Learning with Randomized LoRA Subparameter Updates

Sangwoo Park1 Seanie Lee1 Byungjoo Kim1 Sung Ju Hwang12

# arXiv:2503.07216v2[cs.LG]11Mar2025

### Abstract

Federated Learning (FL) is a widely used framework for training models in a decentralized manner, ensuring that the central server does not have direct access to data from local clients. However, this approach may still fail to fully preserve data privacy, as models from local clients are exposed to the central server during the aggregation process. This issue becomes even more critical when training vision-language models (VLMs) with FL, as VLMs can easily memorize training data instances, making them vulnerable to membership inference attacks (MIAs). To address this challenge, we propose the FedRand framework, which avoids disclosing the full set of client parameters. In this framework, each client randomly selects subparameters of LowRank Adaptation (LoRA) from the server and keeps the remaining counterparts of the LoRA weights as private parameters. After training both parameters on the client’s private dataset, only the non-private client parameters are sent back to the server for aggregation. This approach mitigates the risk of exposing client-side VLM parameters, thereby enhancing data privacy. We empirically validate that FedRand improves robustness against MIAs compared to relevant baselines while achieving accuracy comparable to methods that communicate full LoRA parameters across several benchmark datasets.

### 1. Introduction

Vision-language models (VLMs) (Alayrac et al., 2022; Zhu et al., 2023; Liu et al., 2023) have demonstrated remarkable performance in various multi-modal tasks, such as visual question answering (Dai et al., 2023; Liu et al., 2023) and image captioning (Li et al., 2023). However, deploy-

1Graduate School of AI, KAIST 2DeepAuto.ai. Correspondence to: Sangwoo Park <swgger@kaist.ac.kr>, Seanie Lee <lsnfamily02@kaist.ac.kr>.

Preprint. Copyright 2025 by the author(s).

ing VLMs in real-world scenarios raises significant concerns about data privacy. These models can easily memorize training data (Carlini et al., 2021, 2023), including sensitive information such as private photographs or medical diagnosis records. Adversarial attackers can exploit this vulnerability to perform a membership inference attack (Shokri et al., 2017), which aims to detect whether a specific data instance is part of the training dataset.

Federated learning (FL; McMahan et al., 2017) is a distributed learning framework in which each local client receives global parameters from a central server, trains a local model on its private dataset, and periodically sends the local model back to the server for aggregation. It offers a potential solution to address privacy concerns, as the central server cannot directly access the private dataset. However, naively transmitting local model parameters back to the central server remains vulnerable to membership inference attacks, as attackers can potentially reconstruct the local client model by intercepting its parameters during the aggregation stage. This issue is particularly critical when fine-tuning vision-language models (VLMs), as their large capacity to memorize private training data amplifies the privacy risks.

To address the privacy issue, we propose a simple yet privacy-enhanced federated learning (FL) framework, dubbed FedRand. In this framework, clients randomly select a subset of parameters provided by the server and keep the remaining parameters as client-specific private ones. After updating both the selected parameters and their client-specific private parameters, only the non-private parameters are transmitted back to the server for the model update.

Specifically, we first apply Low-Rank Adaptation (LoRA; Hu et al., 2022) matrices A and B to the pre-trained weight W0 of a VLM. The pre-trained weight W0 is fixed and shared across all clients and the server. At each round of updates, each local client model receives the LoRA weights A and B from the server. Each client then randomly selects either A or B and initializes the counterpart of the LoRA weights using the parameters from the previous round as client-specific private ones(Figure 1a). After updating both parameters on the client’s private training

[Figure 1]

- Client 1

(b) Aggregation

Client 4

[Figure 2]

[Figure 3]

Private Dataset

[Figure 4]

Private

Dataset

|[Figure 5]|[Figure 6]|
|---|---|
| | |

|[Figure 7]|[Figure 8]|
|---|---|
| |Private Parameter|

|[Figure 9]|
|---|

[Figure 10]

- Client 2

[Figure 11]

Private Dataset

|[Figure 12]|[Figure 13]|
|---|---|
| |Private Parameter|

|[Figure 14]|
|---|

|[Figure 15]|[Figure 16]|
|---|---|
| |Private Parameter|

|[Figure 17]|
|---|

Pretrained Weight

|[Figure 18]|
|---|

Updated

LoRA Weights

|[Figure 19]|
|---|

|[Figure 20]|
|---|

Avg

[Figure 21]

Avg

|[Figure 22]|
|---|

|[Figure 23]|
|---|

[Figure 24]

- Client 3

|[Figure 25]|
|---|

|[Figure 26]|[Figure 27]|
|---|---|
| | |

|[Figure 28]|[Figure 29]|
|---|---|
| |Private<br><br>Parameter|

|[Figure 30]|
|---|

[Figure 31]

Private

Pretrained Weight LoRA Weights

Dataset

Central Server

(a) Local Update

- Figure 1. (a). At each round r, each local client selects a LoRA weight either Ar or Br for initialization from the server and initializes the other counterparts of LoRA weights using the previous round’s client model parameters as private parameters. (b). After updating both parameters, only the non-private parameters are sent back to the server and aggregated to update the LoRA weights of the central server.

68.5 69.0 69.5 70.0 70.5

AUROC of MIA ( )

- 98

- 99

- 100

- 101

- 102

- 103

- 104

- 105

CIDEr()

MSCOCO

FedAvg

FedPer

FedPara (layer 4) FedPara (layer 2) FedRand (Ours)

- Figure 2. Trade-off between task performance (CIDEr) and vulnerability to membership inference attacks (AUROC of MIA) on MSCOCO dataset.

perimental results demonstrate that FedRand significantly improves the trade-off between accuracy and robustness against membership inference attacks (Figure 2) while reducing communication costs between the server and clients compared to other relevant baselines.

Our contributions and findings are summarized below:

- • We show that even fine-tuning VLMs with FL remains vulnerable to membership inference attacks due to the exposure of client model parameters, posing significant privacy concerns.
- • To address these privacy concerns, we propose FedRand. First, a client randomly selects subparameters of LoRA weights from the server and updates both the selected parameters and client-specific private parameters. Only the non-private parameters are sent back to the server, preventing the exposure of the full local model parameters.
- • We experimentally demonstrate that FedRand enhances robustness against membership inference attacks while achieving performance comparable to models that communicate full LoRA weights between the server and clients.

dataset, the client-specific parameters remain hidden, and only the remaining parameters are sent back to the server. Finally, the parameters A and B from the clients are averaged to form the new LoRA weights of the server model (Figure 1b). Since the client-specific private parameters are kept hidden, adversarial attackers cannot fully reconstruct the client model parameters by intercepting the parameters transmitted to the server. This design makes FedRand more robust against membership inference attacks. Furthermore, sending only non-private parameters significantly reduces the communication cost between the server and clients compared to the model that communicates all LoRA weights between the server and clients.

### 2. Related Work

Federated learning. Federated Learning (FL) is a decentralized machine learning approach that allows multiple clients to collaboratively train a shared model without sharing their private data, thereby preserving privacy and security. FedAvg (McMahan et al., 2017), one of the most widely used algorithms in FL, updates a global model by averaging the model parameters trained on each client’s pri-

We empirically validate our proposed FedRand on visual question answering and image captioning tasks using the ScienceQA (Lu et al., 2022), MSCOCO (Lin et al., 2014), and NoCaps (Agrawal et al., 2019) datasets. Ex-

vate dataset. While many variants of FedAvg have been proposed (Li et al., 2020; Yu et al., 2020; Acar et al., 2021; Zhang et al., 2024), they remain vulnerable to membership inference attacks because clients’ parameters are exposed to the server. In another line of work, methods like FedPer (Arivazhagan et al., 2019) and FedPara (HyeonWoo et al., 2022) distinguish client-specific private parameters from global parameters shared between the server and clients to reduce communication costs. However, although these methods avoid exposing client parameters to the server, they fail to strike a balance between accuracy and robustness against membership inference attacks.

Membership inference attack. Although FL avoids sharing private data between clients and a server by training client models locally and aggregating only the parameters of the client models at the server, clients are still vulnerable to the leakage of privacy-sensitive information. This can occur through membership inference attacks (Shokri et al., 2017), where an attacker detects whether a specific data instance is included in a private client’s dataset. While both the central server and clients can potentially deduce private details from shared information such as model parameters, the majority of works (Hitaj et al., 2017; Melis et al., 2019) focus on client-based membership inference attacks under the strong assumption of a secure server. However, server-based membership inference attacks pose a significant threat, particularly due to the memorization capacities of VLMs. Jayaraman et al. (2024) have demonstrated this vulnerability through k-nearest neighbor retrieval tests on open-source image datasets, showing that VLMs are prone to retaining training data. Moreover, Li et al. (2024) utilize average top-k Rényi entropy of VLMs’ output probabilities to distinguish training data from other data, highlighting the vulnerability of VLMs to membership inference attacks. This suggests that malicious use of client models on the server side could lead to data leakage through the memorization of training data by the client models. To address this issue, we propose FedRand, which prevents the exposure of client models to the server and thus enhances robustness against server-based membership inference attacks.

### 3. Method

###### 3.1. Preliminaries

Let pθ : X × Z → Y be a vision language model (VLM) with its parameter θ, which takes as input a sequence of tokens x ∈ X and an image z ∈ Z, and outputs another sequence of tokens y ∈ Y as a response to the input. Here, X is the set of all possible input sequences, Z is the set of all possible images, and Y is the set of all possible output sequences. In the FL framework, each

client k ∈ [K] := {1,...,K} has access only to its local training dataset Dk = {(x(ik),z(ik),yi(k))}n

i=1, where Dk ∩ Dk′ = ∅ for all k,k′ ∈ [K] with k ̸= k′. Furthermore, the central server does not have direct access to any of the local datasets. For each round of update r ∈ [R], a subset of client indices Sr ⊂ [K] is randomly chosen with |Sr| = K′. Then each client k ∈ Sr receives the parameter θr from the central server and trains its local model pθ(k) on the dataset Dk as follows:

k

θr,t(k)+1 = θr,t(k) − η∇θL(θr,t(k);Dk) L(θr,t(k);Dk) = −

(1)

1 nk

(y | x,z),

log pθ(k)

r,t

(x,z,y)∈Dk

for t = 0,...,Tk − 1, where η > 0 is a learning rate and θ0(k,0) is initialized with θr. Since fully fine-tuning the VLM is computationally expensive, we apply Low Rank Adaptation (LoRA; Hu et al., 2022) for fine-tuning the weight matrix of the VLM at the l-th layer as:

Wr,t(k,l) = W0(l) + A(r,tk,l)Br,t(k,l), (2)

where W0(l) is the frozen pre-trained weight matrix of the VLM, and A(r,tk,l) and Br,t(k,l) are low-rank matrices, i.e., rank(A(r,tk,l)Br,t(k,l)) ≪ rank(W0(l)). With a slight abuse of notation of θr,t(k), we denote the parameter θr,t(k) = {(W0(l),A(r,tk,l),Br,t(k,l))}Ll=1 as the set of the initial pretrained weight matrices and LoRA weight matrices for the client k at step t in round r. After the local client update, following the FedAvg (McMahan et al., 2017) and FedIT (Zhang et al., 2024), we aggregate the parameters of the local client models and update the server parameter θr = {(W0(l),A(rl),Br(l))}Ll=1 to θr+1 as follows:

A(rl+1) =

nk mr

A(r,Tk,l)

k

k∈Sr

,Br(l+1) =

nk mr

Br,T(k,l)

k

k∈Sr

(3)

where mr = k∈S

nk and nk = |Dk|. At the next round r+1, the central server model pθ

r

uses its updated weight matrix,

r+1

Wr(+1l) = W0(l) + A(rl+1) Br(l+1) (4) for each layer l ∈ [L].

###### 3.2. Privacy Enhanced FL: FedRand

However, aggregating the parameters of client models at the central server poses a serious privacy issue. An adversarial attacker can fully reconstruct the local model by hijacking the LoRA parameters. Since VLMs easily memorize training data (Carlini et al., 2021, 2023; Jayaraman et al., 2024), the attacker can detect whether a

###### Algorithm 1 FedRand

Algorithm 2 client_update(k, θ, E, ρ, η, b, r)

- 1: Input: VLM pθ with pre-trained weights θ =

{W0(l)}Ll=1, learning rate η, total round R, number of clients K, number of clients participating for update

K′, probability ρ of choosing A, and batch size b.

- 2: Randomly initialize LoRA weights {(A(0l),B0(l))}Ll=1.
- 3: for r = 0,...,R − 1 do
- 4: mr ← 0, θr ← {(W0(l),A(rl),Br(l))}Ll=1
- 5: Choose client indices Sr from [K] s.t. |Sr| = K′.
- 6: for each k in Sr do
- 7: (θ(k),ak,nk) ← client_update(k,θr,E,ρ,η,b,r)
- 8: mr ← mr + nk
- 9: end for
- 10: α ← k∈Sr

nk mr · {ak=1}, β ←

k∈Sr

nk mr · {ak̸=1}

- 11: for l = 1,...,L do
- 12: if α > 0 then
- 13: A(rl+1) ← k∈Sr,ak=1

nk

αmr A(r,Tk,l)

k

- 14: else
- 15: A(rl+1) ← A(rl)
- 16: end if
- 17: if β > 0 then
- 18: Br(l+1) ← k∈Sr,ak̸=1

nk

βmr Br,T(k,l)

k

- 19: else
- 20: Br(l+1) ← Br(l)
- 21: end if
- 22: end for
- 23: end for
- 24: θ∗ ← {(W0(l),A(Rl),BR(l))}Ll=1
- 25: return pθ

Input: Client index k, server parameter θr = {(W0(l),A(rl),Br(l))}Ll=1, train epochs E, probability ρ of choosing A(l), learning rate η, batch size b, and current round r.

2: Tk ← ⌈|Dk|/b⌉ · E

uk ← Uniform(0,1), ak ← {uk<ρ} 4: if r = 0 then

{A(0k,l,0 )}Ll=1 ← {rand_init(A(rl))}Ll=1 6: {B0(k,l,0 )}Ll=1 ← {zero_init(Br(l))}Ll=1

else

8: if ak = 1 then

{A(r,k,l0 )}Ll=1 ← {A(rl)}Ll=1 10: {Br,(k,l0 )}Ll=1 ← {Br(k,l−1),T

}Ll=1 else

k

12: {A(r,k,l0 )}Ll=1 ← {A(rk,l−1),T

}Ll=1 {Br,(k,l0 )}Ll=1 ← {Br(l)}Ll=1

k

14: end if

end if

16: for t = 0,...,Tk − 1 do

Sample a mini-batch B from the client dataset Dk. 18: θr,t(k) ← {(W0(l),A(r,tk,l),Br,t(k,l))}Ll=1

L(θt(k);B) ← −|B|1 (x,z,y)∈B log pθ(k)

(y | x,z) 20: θr,t(k)+1 ← θr,t(k) − η∇θ(k)

t

L(θr,t(k);B) end for

r,t

22: Cache {(A(r,Tk,l)

,Br,T(k,l)

)} if ak = 1 then 24: return {A(Tk,l)

k

k

}Ll=1,ak,|Dk| else

∗

k

26: return {BT(k,l)

}Ll=1,ak,|Dk| end if

particular training data instance is included in the local client’s training dataset Dk using a membership inference attack (Shokri et al., 2017; Li et al., 2024).

k

Specifically, at each round r ∈ [R], each client k ∈ Sr first samples ak with a probability ρ of choosing {A(rl)}Ll=1 as follows:

To address the issue of exposing the full parameters of local client models to an attacker, we propose FedRand, a method in which, during each up-

date round, each client randomly selects either {A(rl)}Ll=1 or {Br(l)}Ll=1 LoRA weights from the server as initialization, while the remaining components are initialized using the previous round’s client model parameters θr(k−)1,T

u(k) ∼ Uniform(0,1), ak = {u(k)<ρ}, (5)

where is an indicator function. The binary variable ak ∈ {0,1} indicates whether A(rl) is selected. If ak = 1, we initialize A(r,k,l0 ) with A(rl) from the server and randomly initialize its counterpart, Br,(k,l0 ), with the client parameter Br(k,l−1),T

,Br(k,l−1),T

= {(W0(l),A(rk,l−1),T

)}Ll=1 as private parameters. Only the selected parameters are sent back to the server after updating the client model, whereas the clientspecific private LoRA weights remain hidden. This randomized LoRA subparameter update prevents the attacker from fully recovering the parameters of the local client model, thereby enhancing robustness against membership inference attacks. Furthermore, our proposed method, FedRand, helps save communication costs by reducing the number of parameters sent from clients to the server compared to the FedAvg method.

k

k

k

from the previous round r − 1. Otherwise, we reverse the procedure as follows:

k

A(rl), if ak = 1, A(rl−)1,T

A(r,k,l0 ) =

(6)

, otherwise,

k

- Br(k,l−1),T

k

, if ak = 1,

- Br(l), otherwise.

Br,(k,l0 ) =

(7)

for all layers l ∈ [L]. Note that A(0k,l,0 ) is randomly initialized and B0(k,l,0 ) is initialized as a zero matrix, regardless of the choice of ak. Then, we update the local client model, initialized with θ0(k) = {(W0(l),A(0k,l),B0(k,l))}Ll=1, as described in Equation 1, for Tk steps, yielding θT(k)

= {(W0(l),A(Tk,l)

k

,BT(k,l)

)}Ll=1. After the local update, only the selected LoRA parameters are sent back to the central server and the parameter of the central server model is up-

k

k

dated to θr+1 = {(W0(l),A(rl),Br(l))}Ll=1 as follows:

nk mr · {ak̸=1} (8)

nk mr · {ak=1}, β =

α =

k∈Sr

k∈Sr

αmr A(r,Tk,l)

nk

, if α > 0 A(rl), otherwise,

A(rl+1) = k∈S

(9)

r,ak=1

k

βmr Br,T(k,l)

nk

, if β > 0 Br(l), otherwise,

Br(l+1) = k∈S

r,ak̸=1

k

(10)

nk and nk = |Dk|. The parameters {A(r,Tk,l)

where mr = k∈S

r

}Ll=1 are aggregated from the clients whose ak = 1, while {Br,T(k,l)

k

}Ll=1 are aggregated from the clients whose ak ̸= 1. If none of the clients choose the server parameters {A(rl)}Ll=1, the parameters are not updated and remain the same for {A(rl+1) }Ll=1. The same rule applies to the update of {Br(l)}Ll=1. Note that we need normalization factors α and β to ensure that the summation of the coefficients in Equation 9 and Equation 10 equals one, respectively. Otherwise, the summation of coefficients would not equal to one, since some of the weight matrices from the clients are not sent back to the server. After R rounds of updates, we use θ∗ = {(W0(l),A(Rl),BR(l))}Ll=1 as the parameters of the final server model pθ

k

. We outline our method in Algorithm 1 and Algorithm 2.

∗

### 4. Experiments

###### 4.1. Setup

Dataset. To evaluate both the effectiveness and privacy robustness of FedRand, we conduct two experiments: (a) accuracy evaluation on visual question answering (VQA) and image captioning tasks, and (b) a membership inference attack using models trained in experiment (a). For the VQA task, we use the ScienceQA (Lu et al., 2022) dataset, while for the image captioning task, we use MSCOCO (Lin et al., 2014). To assess out-of-distribution (OOD) generalization and robustness against membership inference attacks, we employ the NoCaps (Agrawal et al., 2019) dataset. For the non-IID scenarios, we use the Dirichlet distribution to randomly split each dataset, where ScienceQA is divided based on topics, while MSCOCO is partitioned according to object classes in images. We set the Dirichlet

parameter to 0.5 as suggested by FedML (He et al., 2020). Detailed descriptions of each dataset can be found in Appendix A.1.

Evaluation metrics. For ScienceQA dataset, we measure the exact match between ground truth answers and model predictions as an accuracy. For MSCOCO and NoCaps datasets, BLEU (Papineni et al., 2002), ROUGE (Lin, 2004), and CIDEr (Oliveira dos Santos et al., 2021) score are utilized to evaluate the quality of the responses. Lastly, we use the MaxRényi-K% (Li et al., 2024) metric as a score for binary classification between member and non-member data, defined as follows:

MaxRény-K%(X)

1 |Max-K%(X)|

###### Hα(pθ(· | x1:i)), (11)

=

i∈Max-K%(X)

where X = (x1,...,xT) is an input token sequence, pθ(· | x1:i) denotes the next-token distribution after the i-th token, and Max-K%(X) is the set of token positions in X with the highest K% Rény entropy Hα. With this score, we compute the AUROC score to measure the robustness against membership inference attacks. Note that MaxRény-0% is the maximum Rényi entropy among all positions from 1 to T−1, i.e., maxi∈[T−1] Hα(pθ(· | x1:i)).

Implementation details. We use a pre-trained model trained with the TinyLLava (Zhou et al., 2024) framework, which consists of an image encoder, CLIP (Radford et al., 2021), an instruction-tuned language model, OpenELM (Mehta et al., 2024), with 450M parameters, and a linear transformation layer that maps the output of CLIP to the word embedding space of OpenELM. We finetune only the language model using LoRA with a rank of 8, while keeping the rest of the model frozen. For each round of FL updates, we fine-tune a client model using the AdamW (Loshchilov & Hutter, 2019) optimizer for one epoch, with a learning rate of 3 · 10−4, weight decay of 10−6, a batch size of 8, and ρ = 0.5. We set the total number of clients K to 12 and sample 30% of clients at each round during FL (i.e., K′ = 4). The total number of FL update rounds is set to 30.

Baselines. We compare our proposed method, FedRand, against the following relevant baselines.

- 1. FedAvg (McMahan et al., 2017) trains local clients using the full LoRA weights provided by a central server and averages the updated full LoRA weights from clients to update the server model’s parameters.
- 2. FedPer (Arivazhagan et al., 2019) communicates the LoRA weights of certain top layers between the server and clients while keeping the remaining LoRA weights

Table 1. We train each method on the VQA, ScienceQA, and MSCOCO datasets and report its performance on the server, as well as the average performance of the clients. The best results are bolded, and the second-best ones are underlined.

Server ScienceQA MSCOCO Method Acc BLEU-1 BLEU-2 BLEU-3 BLEU-4 ROUGE CIDEr

FedAvg (oracle) 81.50 (0.53) 75.49 (0.44) 58.53 (0.37) 43.43 (0.32) 31.80 (0.17) 55.29 (0.22) 111.08 (0.76) FedPer (2 layer) 42.11 74.53 57.11 41.97 30.12 54.13 106.60 FedPer (4 layer) 44.59 74.43 57.31 42.14 30.22 54.17 107.44 FedPara 64.78 73.73 56.94 41.36 29.91 53.75 106.96

FedRand (Ours) 80.12 (0.42) 75.37 (0.35) 58.66 (0.38) 43.63 (0.23) 31.89 (0.25) 55.15 (0.19) 110.27 (0.54) Client ScienceQA MSCOCO Method Acc BLEU-1 BLEU-2 BLEU-3 BLEU-4 ROUGE CIDEr

FedAvg (oracle) 79.90 (1.26) 73.86 (0.56) 56.62 (0.57) 41.43 (0.54) 29.76 (0.51) 54.04 (0.32) 104.48 (1.21) FedPer (2 layer) 56.93 (5.40) 71.82 (1.52) 54.20 (1.78) 39.10 (1.61) 27.67 (1.35) 52.45 (0.93) 101.00 (3.63) FedPer (4 layer) 58.94 (5.73) 72.52 (1.39) 54.99 (1.61) 39.84 (1.51) 28.34 (1.26) 52.96 (0.72) 101.32 (2.86) FedPara 58.57 (5.20) 71.36 (1.60) 53.51 (2.18) 38.25 (2.13) 26.86 (1.69) 52.90 (1.23) 97.53 (5.03)

FedRand (Ours) 76.01 (1.15) 73.90 (0.89) 56.76 (0.97) 41.72 (0.94) 29.94 (0.63) 53.64 (0.69) 105.10 (1.30)

as client-specific private parameters. We share the top 2 or 4 layers of LoRA weights across clients as global parameters. The other layers of LoRA weights are kept hidden as client-specific private parameters and are never shared. Since LoRA parameters of certain layers remain entirely private in FedPer, the LoRA A and B matrices of these non-shared layers were initialized using the aggregation results from the first round to ensure training stability.

- 3. FedPara (Hyeon-Woo et al., 2022) parameterizes private LoRA weights for each client and global LoRA weights shared across the server and clients. Each client performs elementwise multiplication between its private LoRA weights and the global ones, then adds the result to the initial pre-trained weights. The global parameters are aggregated from the clients and averaged to serve as the parameters of the server model.

FedAvg serves as the oracle method for accuracy evaluation experiments, as it always communicates the full LoRA weights between the server and clients. The other two baselines are selected because they share the concept of partial parameter sharing with our method, enabling a comparative analysis of different strategies. The details of the implementation for FedPer and FedPara are provided in Appendix A.3.

- 4.2. Experimental Results

Main results. Table 1 presents the performance of FedRand and other baselines on the ScienceQA and MSCOCO datasets. The upper table reports the statistics of the server-side aggregated global model, while the lower table summarizes the average statistics of individual client models. Given the dynamic client participation in FL, we

Table 2. We evaluate each method trained on the MSCOCO dataset to measure OOD generalization on the NoCaps dataset.

Server NoCaps Method BLEU-1 BLEU-2 BLEU-3 BLEU-4 ROUGE CIDEr

FedAvg (oracle) 78.74 62.24 46.38 33.49 54.66 79.82 FedPer (2 layer) 78.10 61.30 45.30 32.20 53.20 78.10 FedPer (4 layer) 78.40 61.80 45.80 32.80 54.30 78.50 FedPara 77.10 59.90 43.90 31.10 53.40 77.20

FedRand (Ours) 78.81 62.23 46.42 33.61 54.57 79.23

conducted three runs with different random seeds for the top two performing methods: FedAvg and FedRand. On both the server and client sides, the results indicate that FedRand achieves comparable performance to FedAvg an oracle method that communicates full LoRA parameters between the server and clients in every round without considering membership inference attacks. This highlights the effectiveness of our proposed method, FedRand, while reducing communication costs between the server and clients by sharing only a subset of client parameters in each round.

In contrast, FedPer and FedPara exhibit significantly lower performance on both the server and client sides compared to FedAvg and FedRand across the ScienceQA, and MSCOCO datasets. This underperformance is attributed to their client-specific private parameters. Since these parameters are never aggregated, knowledge transfer between clients is limited, leading to overfitting on small client datasets and a degradation in generalization performance. On the other hand, our method, FedRand, stochastically shares a random subset of client parameters at each round, encouraging knowledge transfer between clients. This mitigates the overfitting issue and improves generalization.

OOD generalization. Furthermore, we evaluate the models trained on the MSCOCO dataset using the NoCaps dataset to measure out-of-distribution (OOD) generalization performance. As shown in Table 2, we observe sim-

Table 3. We ablate each component of our FedRand and measure its performance (BLEU, ROUGE, and CIDEr) on MSCOCO dataset and robustness (MaxRény-10%) against the membership inference attack.

MSCOCO (↑) MaxRényi-10% (↓) Component BLEU-1 BELU-2 BLEU-3 BELU-4 ROUGE CIDEr Image Caption

ρ = 0.3 75.57 58.58 43.36 31.47 54.90 109.37 53.89 (2.79) 65.53 (3.07) ρ = 0.7 75.23 58.20 42.97 31.11 54.86 108.98 52.79 (1.37) 65.40 (4.22) w/o past parameters 76.30 59.29 44.23 32.42 55.27 110.83 58.04 (5.35) 67.44 (4.33) w/o normalization 72.50 54.90 39.61 28.04 52.79 98.83 51.03 (2.12) 62.21 (1.71)

FedRand 75.37 (0.35) 58.66 (0.38) 43.63 (0.23) 31.89 (0.25) 55.15 (0.19) 110.27 (0.54) 53.84 (2.50) 66.61 (3.22)

ilar trends to those in the previous experiments. FedRand achieves performance comparable to FedAvg, while FedPer and FedPara significantly degrade in performance compared to both FedAvg and FedRand. These results once again highlight the effectiveness of our method, FedRand.

Membership inference attack (MIA). We perform a membership inference attack on the models trained on the MSCOCO dataset. Following Li et al. (2024), we use MaxRényi-K%, described in Equation 11, as a score for binary classification to distinguish member data instances in the MSCOCO dataset from non-member ones in the NoCaps dataset, and report the AUROC score in Table 4. A sample of 300 is drawn from each population for member and non-member data, consisting of 600 images in total. Notably, the non-member data primarily consists of object images that rarely appear in MSCOCO.

We consider two plausible scenarios: (a) the server attempts a MIA using the aggregated model (denoted as ‘server’ in the table), and (b) the server maliciously reconstructs the client model and performs MIA (denoted as ‘client’ in the table). In the case of FedAvg, the server can exactly reconstruct client models using the full client LoRA parameters transmitted to it. However, in our FedRand, since only a subset of parameters is sent to the server per round, the timing at which a client sends the other set of parameters varies across clients. Thus, we first intercept one part of LoRA weights from each client in the final round. Then we obtain the rest of the LoRA weights at the secondto-last round in which each corresponding client participates. For FedPer and FerPara, the client model cannot be fully reconstructed under any circumstance; therefore, we report only the ‘server’ results for those two methods.

As shown in Table 4, FedRand demonstrates stronger resistance to MIA compared to the other baseline methods. This is due to the fact that clients send only a subset of parameters to the server, which helps prevent the exposure of their full client parameters. Both FedAvg and FedRand show that reconstructed client models are more vulnerable than server models, with this trend being more pronounced in FedAvg, as it can fully reconstruct client models at the end of any round. FedPer and FedPara are expected to be effective against MIA since they do not share client-specific

Table 4. Membership inference attack to distinguish the training dataset MSCOCO from the NoCaps dataset using Rényi Entropy Max_0% and Max_10%. Lower scores indicate better robustness against the membership inference attack. Statistics are presented in percentage.

MaxRényi-0% (↓) MaxRényi-10% (↓) image caption image caption

FedAvg (server) 49.96 (3.11) 70.22 (2.56) 54.57 (4.07) 70.22 (2.56) FedAvg (client) 51.68 (4.17) 70.68 (3.82) 54.71 (4.11) 70.69 (3.80) FedPer (2 layers) 50.73 (4.36) 70.01 (3.87) 56.76 (1.51) 70.03 (3.84) FedPer (4 layers) 51.77 (3.40) 69.71 (4.14) 57.74 (2.25) 69.73 (4.16) FedPara 53.48 (1.77) 69.67 (2.97) 57.07 (2.93) 69.63 (2.99)

FedRand (server) 48.90 (4.75) 67.02 (3.74) 53.84 (2.50) 66.61 (3.22) FedRand (client) 47.83 (3.56) 68.51 (3.69) 54.99 (4.22) 68.51 (3.69)

private parameters at all; however, they show worse robustness than FedRand. This may be attributed to the fact that their private parameters are never shared across clients, limiting knowledge transfer. As a result, the shared global parameters must compensate by fitting each client’s dataset more closely, making them more prone to overfitting and leading to more severe memorization.

Ablation studies. We conduct a comprehensive ablation study on each component of our method to evaluate its effectiveness. First, we vary the probability ρ of selecting the LoRA weight matrix A, setting it to ρ = 0.3 and ρ = 0.7. Additionally, we ablate the normalization factors α and β, as defined in Equation 8, referring to this case as “w/o normalization.” Lastly, instead of using the client-specific private parameters in lines 10 and 13 of Algorithm 2, we initialize with the full LoRA weights from the server and send either the updated A or B back to the server, depending on the variable ak, referring to this case as “w/o past parameters”.

As shown in Table 3, selecting the LoRA weight matrix A either more or less frequently than B degrades the performance of image captioning on MSCOCO while slightly improving robustness against MIA. Similarly, removing normalization significantly degrades BLEU, ROUGE, and CIDEr scores, while making the model more robust to MIA due to underfitting. In contrast, initializing all the client

FedAvg

1.00

FedPer (top 25% layers)

Algorithms

0.25

FL

FedPara

1.00

FedRand (Ours)

0.75

Ratio of communicated # parameters to FedAvg per round

- Figure 3. The ratio of number of communicated LoRA parameters, compared to FedAvg per round under LoRA configuration.

parameters with the LoRA weights of the server without using the client’s past parameters significantly boosts the performance on the MSCOCO dataset but drastically sacrificing robustness against the MIA. These experimental results support the choice of hyperparameters ρ = 0.5 and

- our algorithm design.

Communication cost. Figure 3 illustrates the communication cost between a server and clients required for each method. Although FedPer reduces the the cost to 25% by sharing only the upper layers, it significantly underperforms compared to FedAvg as shown in previous experiments. In the case of our proposed FedRand, receives the same number of parameters received from the server as FedAvg, but only sends half of them are back to the server, reducing the communication cost by approximately 25% per round, while retaining accuracy similar to FedAvg.

5. Conclusion

In this work, we proposed the FedRand framework to mitigate the vulnerability of vision-language models (VLMs) fine-tuned with federated learning to membership inference attacks. Instead of communicating the full LoRA weights of VLMs between the server and clients — which an attacker could intercept to perform membership inference attacks — each client randomly selected a subset of LoRA weights from the server and initialized the remaining LoRA weights using its private parameters from the previ-

- ous round. After updating both sets of parameters, only the non-private parameters were sent back to the server for aggregation, reducing the risk of disclosing the full parameters of the client model. We extensively validated that our proposed FedRand achieved performance comparable to FedAvg, which communicated full LoRA weights between the server and clients, while demonstrating improved robustness against membership inference attacks compared to other relevant baselines. Additionally, our method reduced communication costs between the server and clients

by transmitting only a subset of the client model parameters to the server. As future work, we suggested randomly selecting sub-layers of clients for training or quantizing client parameters sent to the server to further enhance the security of client model parameters.

### Impact Statements

This paper presents a framework, FedRand, aimed at improving privacy in Federated Learning (FL), particularly when training vision-language models (VLMs). Our work contributes to advancing the field of privacy-preserving machine learning by mitigating the risks of membership inference attacks without significantly compromising model performance. By enhancing data privacy in FL, our approach can benefit various real-world applications, including healthcare, finance, and other domains where sensitive data is distributed across multiple entities. FedRand reduces the exposure of client-side model parameters, thereby strengthening privacy guarantees for users participating in federated training. However, as with any privacy-preserving method, FedRand does not eliminate all risks. Adversarial attackers may still attempt more sophisticated attacks beyond membership inference, and further research is needed to address emerging privacy threats. Additionally, while our method enhances privacy, it does not directly address fairness or bias in FL, which remain important considerations for real-world deployment. Overall, this work aligns with the broader goal of developing privacy-preserving AI systems and does not introduce any foreseeable ethical concerns or negative societal impacts.

### References

Acar, D. A. E., Zhao, Y., Matas, R., Mattina, M., Whatmough, P., and Saligrama, V. Federated learning based on dynamic regularization. International Conference on Learning Representations (ICLR), 2021.

Agrawal, H., Desai, K., Wang, Y., Chen, X., Jain, R., Johnson, M., Batra, D., Parikh, D., Lee, S., and Anderson, P. NoCaps: novel object captioning at scale. Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems (NeurIPS), 2022.

Arivazhagan, M. G., Aggarwal, V., Singh, A. K., and Choudhary, S. Federated learning with personalization layers. arXiv preprint arXiv:1912.00818, 2019.

Carlini, N., Tramer, F., Wallace, E., Jagielski, M., Herbert-

Voss, A., Lee, K., Roberts, A., Brown, T., Song, D., Erlingsson, U., et al. Extracting training data from large language models. 30th USENIX Security Symposium (USENIX Security 21), 2021.

Carlini, N., Ippolito, D., Jagielski, M., Lee, K., Tramer, F., and Zhang, C. Quantifying memorization across neural language models. International Conference on Learning Representations (ICLR), 2023.

Dai, W., Li, J., Li, D., Tiong, A., Zhao, J., Wang, W., Li, B., Fung, P., and Hoi, S. InstructBLIP: Towards generalpurpose vision-language models with instruction tuning. Advances in Neural Information Processing Systems (NeurIPS), 2023.

He, C., Li, S., So, J., Zhang, M., Wang, H., Wang, X., Vepakomma, P., Singh, A., Qiu, H., Shen, L., Zhao, P., Kang, Y., Liu, Y., Raskar, R., Yang, Q., Annavaram, M., and Avestimehr, S. Fedml: A research library and benchmark for federated machine learning. ArXiv, 2020.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

Hitaj, B., Ateniese, G., and Perez-Cruz, F. Deep models under the gan: information leakage from collaborative deep learning. ACM SIGSAC conference on computer and communications security, 2017.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA: Low-rank adaptation of large language models. International Conference on Learning Representations (ICLR), 2022.

Hyeon-Woo, N., Ye-Bin, M., and Oh, T.-H. Fedpara: Lowrank hadamard product for communication-efficient federated learning. International Conference on Learning Representations (ICLR), 2022.

Jayaraman, B., Guo, C., and Chaudhuri, K. Déjà vu memorization in vision–language models. Advances in Neural Information Processing Systems (NeurIPS), 2024.

Li, J., Li, D., Savarese, S., and Hoi, S. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. International Conference on Machine Learning (ICML), 2023.

Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., and Smith, V. Federated optimization in heterogeneous networks. Machine learning and systems (MLSys), 2020.

Li, Z., Wu, Y., Chen, Y., Tonin, F., Rocamora, E. A., and Cevher, V. Membership inference attacks against large vision-language models. Advances in Neural Information Processing Systems (NeurIPS), 2024.

Lin, C.-Y. ROUGE: A package for automatic evaluation of summaries. Text Summarization Branches Out, 2004.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., and Zitnick, C. L. Microsoft COCO: Common objects in context. European Conference Computer Vision (ECCV), 2014.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. Advances in Neural Information Processing Systems (NeurIPS), 2023.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. iclr, 2019.

Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems (NeurIPS), 2022.

McMahan, B., Moore, E., Ramage, D., Hampson, S., and y Arcas, B. A. Communication-efficient learning of deep networks from decentralized data. International Conference on Artificial Intelligence and Statistics (AISTATS), 2017.

Mehta, S., Sekhavat, M., Cao, Q., Horton, M., Jin, Y., Sun, F., Mirzadeh, I., Najibikohnehshahri, M., Belenko, D., Zatloukal, P., and Rastegari, M. Openelm: An efficient language model family with open training and inference framework. ICML Workshop, 2024.

Melis, L., Song, C., De Cristofaro, E., and Shmatikov, V. Exploiting unintended feature leakage in collaborative learning. IEEE symposium on security and privacy (SP), 2019.

Oliveira dos Santos, G., Colombini, E. L., and Avila, S. CIDEr-R: Robust consensus-based image description evaluation. Workshop on Noisy User-generated Text (WNUT 2021), 2021.

Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. BLEU: a method for automatic evaluation of machine translation. Association for Computational Linguistics (ACL), 2002.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. International Conference on Machine Learning (ICML), 2021.

Shokri, R., Stronati, M., Song, C., and Shmatikov, V. Membership inference attacks against machine learning models. 2017 IEEE symposium on security and privacy (SP), 2017.

Waswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A., Kaiser, L., and Polosukhin, I. Attention is all you need. Advances in Neural Information Processing Systems (NeurIPS), 2017.

Yu, F., Rawat, A. S., Menon, A., and Kumar, S. Federated learning with only positive labels. International Conference on Machine Learning (ICML), 2020.

Zhang, J., Vahidian, S., Kuo, M., Li, C., Zhang, R., Yu, T., Wang, G., and Chen, Y. Towards building the federatedgpt: Federated instruction tuning. International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024.

Zhou, B., Hu, Y., Weng, X., Jia, J., Luo, J., Liu, X., Wu, J., and Huang, L. Tinyllava: A framework of small-scale large multimodal models. arXiv preprint arXiv:2402.14289, 2024.

Zhu, D., Chen, J., Shen, X., Li, X., and Elhoseiny, M. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

### A. Experimental Details

###### A.1. Dataset

- • ScienceQA (Lu et al., 2022) is a multiple choice visual question answering dataset derived from elementary and high school science curricula, covering three subjects: natural science, language science, and social science. We focus exclusively on the 10,327 questions that include accompanying images, representing 48.7% of the entire dataset.
- • MSCOCO (Lin et al., 2014) contains over 330K images with dense annotations for image recognition, segmentation and captioning tasks. Among the 83K instances specifically created for captioning, 50K images are sampled for training and 5K images each for validation and testing.
- • NoCaps (Agrawal et al., 2019) is designed to evaluate the ability of image captioning models to describe objects not present in the MSCOCO dataset. 45K validation sets, each with 10 captions, are used to assess OOD generalization.

###### A.2. Prompt Template

We present a prompt template for each dataset. Note that the presence of contextual information in ScienceQA depends on the question.

ScienceQA

Based on the image, respond to the question with a given options. USER: {image}\n Context: {context}. Options: {options}. Answer: ASSISTANT: ...

MSCOCO & NoCaps

Briefly describe given image. USER: {image}\n A short image description: ASSISTANT: ...

###### A.3. Communication process of FedPer and FedPara

In the original FedPer framework, the classifier and top N basic blocks of a ResNet (He et al., 2016) model are designated as personalization layers. To adapt this approach for LoRA settings, we instead share the LoRA parameters of the top 2 or

###### 4 transformer (Waswani et al., 2017) layers with the server.

Similarly, the FedPara method originally parameterize weight of base models with Hadamard product between two sets of low rank matrices. To extend this idea to transformer architecture LLMs with LoRA, we introduce an additional pair of LoRA A and B matrices per layer, ensuring the additional LoRA weight matrices remain private on the client side.

|Non-member data|
|---|

[Figure 32]

Label: Two men in swim caps are swimming in lanes in a pool.

Generated: Man swimming in a pool with a rope.

[Figure 33]

Label: A person is rollerblading while holding a hockey stick.

Generated: Man on a skateboard is holding a hockey stick.

[Figure 34]

Label: Shrub shaved into a lion shape with ﬂowers along the mane.

Generated: Large bear statue with ﬂowers growing around it.

|(a) Member data - Defense Fail<br><br>|
|---|

|(b) Member data - Defense Success<br><br>|
|---|

[Figure 35]

[Figure 36]

Label: a female tennis player on the tennis court holding a racket.

Label: A plate of food on a table near a drink. Generated: plate of food sitting on a table.

Generated: Woman is swinging a tennis racket ball.

[Figure 37]

[Figure 38]

Label: A couple are standing behind a table with a cake and bouquet.

Label: a small table and couch in the living room

Generated: Living room with a table and chairs.

Generated: couple of people standing next to a cake.

[Figure 39]

[Figure 40]

Label: A laptop computer sitting on top of a desk.

Label: a person with a white shirt and a black tie

Generated: Man wearing a blue shirt and a black tie.

Generated: Desk with a laptop and a computer monitor.

≥ 3.0 ≥ 2.0 and < 3.0 ≥ 1.0 and < 2.0

< 1.0

- Figure 4. An example of token-wise Rényi entropy measurement for member (MSCOCO) and non-member (NoCaps) data. The higher the entropy is, the more robust to MIA.

### B. Membership Inference Attack Example

We show two sets of membership inference attack (MIA) examples in Figure 4, where color denotes token-wise Rényi entropy with FedRand. On the left (a), the model is confident in next-token prediction for member data (MSCOCO), indicating a failed defense against MIA. On the right (b), the model is highly uncertain for both member and non-member data (NoCaps), leading to a successful defense against MIA.

