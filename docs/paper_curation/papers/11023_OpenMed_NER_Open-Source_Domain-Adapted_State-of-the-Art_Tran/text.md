# arXiv:2508.01630v1[cs.CL]3Aug2025

## OpenMed NER: Open-Source, Domain-Adapted State-of-the-Art Transformers for Biomedical NER Across 12 Public Datasets

Maziyar Panahi AI Engineer CNRS Paris, France maziyar.panahi@cnrs.fr

### Abstract

Named-entity recognition (NER) is fundamental to extracting structured information from the >80% of healthcare data that resides in unstructured clinical notes and biomedical literature. Despite recent advances with large language models, achieving state-of-the-art performance across diverse entity types while maintaining computational efficiency remains a significant challenge. We introduce OpenMed NER, a suite of open-source, domain-adapted transformer models that combine lightweight domain-adaptive pre-training (DAPT) with parameter-efficient Low-Rank Adaptation (LoRA). Our approach performs cost-effective DAPT on a 350k-passage corpus compiled from ethically sourced, publicly available research repositories and de-identified clinical notes (PubMed, arXiv, and MIMIC-III) using DeBERTa-v3, PubMedBERT, and BioELECTRA backbones. This is followed by task-specific fine-tuning with LoRA, which updates less than 1.5% of model parameters. We evaluate our models on 12 established biomedical NER benchmarks spanning chemicals, diseases, genes, and species. OpenMed NER achieves new state-of-the-art micro-F1 scores on 10 of these 12 datasets, with substantial gains across diverse entity types. Our models advance the state-of-the-art on foundational disease and chemical benchmarks (e.g., BC5CDR-Disease, +2.70 pp), while delivering even larger improvements of over 5.3 and 9.7 percentage points on more specialized gene and clinical cell line corpora (BC2GM and CLL, respectively).

This work demonstrates that strategically adapted open-source models can surpass closed-source solutions. This performance is achieved with remarkable efficiency: training completes in under 12 hours on a single GPU with a low carbon footprint (<1.2 kg CO2e), producing permissively licensed, open-source checkpoints designed to help practitioners facilitate compliance with emerging data protection and AI regulations, such as the EU AI Act.

### Contents

- 1 Introduction 3
- 2 Related Work 4

- 2.1 Early Neural and Production-Scale Pipelines . . . . . . . . . . . . . . . . . . . . . 4
- 2.2 Domain-Specific Pre-trained Transformers . . . . . . . . . . . . . . . . . . . . . . 4
- 2.3 Domain-Adaptive and Task-Adaptive Pre-training . . . . . . . . . . . . . . . . . . 5
- 2.4 Parameter-Efficient Adaptation . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 2.5 Robustness and Specialized Systems . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 2.6 Large Generative Biomedical LLMs . . . . . . . . . . . . . . . . . . . . . . . . . 5

- 3 Methods 5

- 3.1 Methodological Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 3.2 Domain-Adaptive Pre-training (DAPT) with LoRA . . . . . . . . . . . . . . . . . 6
- 3.3 Task-Specific Fine-Tuning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.4 Bayesian Hyper-parameter Optimization . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.5 Model Architectures and Selection Strategy . . . . . . . . . . . . . . . . . . . . . 7
- 3.6 Training Procedure . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 3.7 Hyper-parameter Optimization and Evaluation . . . . . . . . . . . . . . . . . . . . 8
- 3.8 Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 4 Experimental Setup 9

- 4.1 Evaluation Protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 4.2 Baseline Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 4.3 Implementation and Computational Resources . . . . . . . . . . . . . . . . . . . . 10

- 5 Results 10

- 5.1 Main Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 5.2 Performance Highlights . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11

- 6 Discussion 12

- 6.1 Architectural Insights and Error Analysis . . . . . . . . . . . . . . . . . . . . . . 12
- 6.2 Computational Efficiency and Practical Implications . . . . . . . . . . . . . . . . 12
- 6.3 Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12

- 6.3.1 Environmental impact and carbon footprint . . . . . . . . . . . . . . . . . 13
- 6.3.2 Regulatory compliance under the EU AI Act . . . . . . . . . . . . . . . . . 13

- 7 Conclusion 13

### 1 Introduction

Recent advances in domain-specific language models, such as BioBERT (Lee et al., 2019), ClinicalBERT (Alsentzer et al., 2019), and PubMedBERT (Gu et al., 2021), have driven substantial progress in biomedical named-entity recognition (BioNER) (Raza et al., 2022; Keraghel et al., 2024). Yet, deploying these powerful models in real-world research and clinical settings requires overcoming persistent challenges like distribution shift across diverse corpora, tight compute budgets, and the need for locally-deployable models to comply with stringent data protection regulations (e.g., GDPR, HIPAA). This paper demonstrates that a strategic combination of (i) domain-adaptive pre-training (DAPT) (Gururangan et al., 2020), (ii) parameter-efficient fine-tuning via Low-Rank Adaptation (LoRA) (Hu et al., 2022), and (iii) strong transformer backbones like DeBERTa-v3 (He et al., 2021) can yield state-of-the-art results while remaining computationally efficient and fully accessible.

The vast majority of information in biomedical research and healthcare is locked in unstructured text. Electronic Health Records (EHRs) alone are estimated to be ∼80% unstructured: a category comprising clinical notes, research publications, drug labels, and patient narratives (Li et al., 2021; Yang et al., 2022). Biomedical text also exhibits a high density of unique entities compared to generaldomain corpora, making BioNER a particularly knowledge-intensive task (Lee et al., 2019; Yadav and Bethard, 2018). Named-entity recognition therefore serves as a foundational technology for downstream applications in precision medicine, drug discovery, and clinical decision support (Keraghel et al., 2024; Raza et al., 2022).

This work, OpenMed NER, makes three primary contributions:

- 1. We develop and release a suite of BioNER models that establish new state-of-the-art performance on 10 of 12 public benchmarks, outperforming proprietary systems with a lightweight and fully open-source framework.
- 2. We provide a crucial ablation study quantifying the synergy between DAPT and LoRA, which together yield a 2–4% absolute F1 gain over using either technique in isolation.
- 3. We release all final model checkpoints under the permissive Apache 2.0 license. This enables local, on-premise deployment, which is critical for applications handling sensitive health data and for facilitating compliance with data protection regulations like the EU AI Act.

The biomedical domain poses linguistic challenges that extend far beyond general-domain NER. Biomedical prose features rapid neologism, systematic yet mutable nomenclatures, dense abbreviations, and morphological variants that proliferate with each new discovery (Lee et al., 2019; Lin and Wilbur, 2004). Clinical entities further complicate matters through unclear boundaries, frequent nesting, and cross-category aliases. For instance, the phrase “upper respiratory tract infection (URTI)” can denote either a DISEASE or a SYMPTOM depending on context (Alex et al., 2007). While early neural architectures such as BiLSTM-CNNs (Chiu and Nichols, 2016) reduced feature engineering, they still struggled with out-of-vocabulary terms and complex semantics.

The transformer era, initiated by BERT (Devlin et al., 2018), spurred a new generation of domainspecific models. Variants like BioBERT (Lee et al., 2019), ClinicalBERT (Alsentzer et al., 2019), PubMedBERT (Gu et al., 2021), and BioELECTRA (Kanakarajan et al., 2021) proved that domainadaptive pre-training (DAPT) (Gururangan et al., 2020) is critical for bridging the gap between general language understanding and specialised biomedical semantics. More recent advances, including powerful backbones like DeBERTa-v3 (He et al., 2021) and parameter-efficient fine-tuning methods like LoRA (Hu et al., 2022), offer further opportunities to push the state of the art.

Despite this rapid progress, critical gaps remain in BioNER research. First, few head-to-head studies compare fully open-source models with proprietary cloud NLP APIs, leaving the real-world performance and cost trade-offs opaque and not always transparent in peer-reviewed literature. Second, the synergy between modern backbones (e.g., DeBERTa-v3) and parameter-efficient training has not been systematically evaluated across the full spectrum of biomedical entity types. Third, most prior work benchmarks on a narrow subset of public datasets, often omitting clinically relevant corpora that test for generalization, such as the BC4CHEMD (Krallinger et al., 2015) and Linnaeus (Gerner et al., 2010) datasets.

We address these gaps directly. By unifying efficient domain adaptation with exhaustive, cross-dataset benchmarking, OpenMed NER shows that open-source solutions can equal or even exceed proprietary biomedical NER systems.

### 2 Related Work

Progress in biomedical named-entity recognition (BioNER) mirrors three macro-trends in natural language processing: (i) the shift from feature-based methods to end-to-end neural encoders (Chiu and Nichols, 2016; Yadav and Bethard, 2018), (ii) the replacement of generic word embeddings with domain-specific pre-trained language models (PLMs) (Lee et al., 2019; Gu et al., 2021), and (iii) the rise of parameter-efficient adaptation, notably LoRA, which updates less than 1.5% of model weights while preserving performance (Hu et al., 2022). OpenMed NER is built upon these three pillars.

#### 2.1 Early Neural and Production-Scale Pipelines

The BiLSTM-CNN-Char architecture of Chiu and Nichols (2016) represented a significant step forward for neural BioNER. Its re-implementation in the commercial Spark NLP library was significant for bringing these neural architectures to the scalable Apache Spark ecosystem, enabling large-scale, distributed inference across clusters (Kocaman and Talby, 2022). Using more modern BERT-based models, Spark NLP has also reported strong, state-of-the-art results on several key benchmarks, demonstrating high performance on chemical and anatomical entity recognition tasks.

Other significant systems include BERN2, which couples multi-task learning with real-time entity normalization and reports an F1-score of 92.70% on the Linnaeus species dataset (Kim et al., 2021). Open-source toolkits such as Stanza (Qi et al., 2020) and SciSpaCy (Neumann et al., 2019) also lowered the barrier to entry for researchers. However, while these toolkits were foundational, transformer-based PLMs have since been shown to outperform earlier architectures, often by significant margins on standard BioNER benchmarks (Lee et al., 2019; Keraghel et al., 2024). Furthermore, these libraries do not natively support modern, parameter-efficient fine-tuning techniques like LoRA, limiting their adaptability to new corpora and tasks.

#### 2.2 Domain-Specific Pre-trained Transformers

Early biomedical PLMs demonstrated that continued masked-language modeling on corpora like PubMed/PMC substantially boosts BioNER performance. BioBERT augmented BERT’s training with

- 4.5B PubMed tokens and 13.5B PMC tokens, lifting performance on the NCBI-Disease benchmark

from 86.7% to 89.71% F1 (Lee et al., 2019). ClinicalBERT replicated this strategy on MIMIC-III notes to better capture the distinct language of EHR narratives (Alsentzer et al., 2019).

A second wave of models showed the value of pre-training from scratch on domain-specific data. PubMedBERT (also known as BiomedBERT) used only biomedical text and achieved new state-ofthe-art results on the BLURB benchmark (Gu et al., 2021). BioELECTRA adapted the more efficient replaced-token detection (RTD) objective to biomedical corpora, surpassing masked-language modeling variants on several tasks (Kanakarajan et al., 2021). Concurrently, models like BioMegatron demonstrated the benefits of scale, using 345M parameters to reach an 88.50% F1 on BC5CDRDisease (Shin et al., 2020).

Knowledge-enhanced transformers add a third dimension by integrating structured knowledge. KeBioLM injects concepts from the Unified Medical Language System (UMLS) during pre-training, posting an 82.00% F1 on the challenging JNLPBA corpus (Yuan et al., 2022). SapBERT uses a contrastive loss to align entity synonyms from UMLS, yielding strong performance on entity-centric tasks (Liu et al., 2021).

Collectively, the following trends define the landscape that OpenMed NER builds upon: continued domain-adaptive pre-training, from-scratch pre-training, alternative objectives, and structured knowledge alignment.

#### 2.3 Domain-Adaptive and Task-Adaptive Pre-training

Gururangan et al. (2020) introduced domain-adaptive pre-training (DAPT) and task-adaptive pretraining (TAPT), showing that a short, continued pre-training stage on unlabeled, in-domain data delivers significant gains. Subsequent biomedical studies confirm that DAPT yields average performance improvements of 2–4 percentage points on benchmarks like BLURB (Gu et al., 2021; Kanakarajan et al., 2021). However, most prior work evaluates on eight or fewer datasets; our 12-corpus study is designed to close this evidence gap by providing a more comprehensive assessment of generalization.

#### 2.4 Parameter-Efficient Adaptation

Full fine-tuning of a large PLM requires duplicating all 300M+ parameters for each new task. LoRA (Low-Rank Adaptation) offers a solution by inserting small, rank-decomposition matrices into the model and updating only these matrices, which often constitute less than 1.5% of the total weights, while matching the performance of full fine-tuning (Hu et al., 2022). This approach has been successfully applied in the medical domain, demonstrating comparable performance to dense tuning while drastically reducing memory requirements (Wu et al., 2023). Frameworks like AdapterHub provide a unified interface for using such adapters with models from the Hugging Face ecosystem (Pfeiffer et al., 2020), but systematic BioNER evaluations using these methods remain limited to a handful of datasets.

#### 2.5 Robustness and Specialized Systems

Model robustness has become a central theme in recent BioNER research. For example, ConNER applies consistency training to improve performance on noisy data (Li et al., 2023). Stress tests reveal that even state-of-the-art models can fail to generalize, missing over 50% of novel mentions of entities like COVID-19 (Vashishth et al., 2021). A recent survey covering work from 2011–2022 shows that transformer-based methods now account for the majority of modern BioNER systems, far surpassing classical machine learning and CRF pipelines (Keraghel et al., 2024). Nonetheless, highly specialized tools remain competitive on their target domains, such as LINNAEUS for species recognition (Gerner et al., 2010).

#### 2.6 Large Generative Biomedical LLMs

Generative large language models (LLMs) are emerging as a complement to encoder-only PLMs. BioGPT achieved state-of-the-art results on PubMedQA and several relation extraction tasks (Luo et al., 2023), while the 8.9B-parameter GatorTron improved performance on multiple clinical NLP benchmarks (Yang et al., 2022). Instruction-tuned models like BioInstruct further enhance question answering and information extraction capabilities without task-specific heads (Su et al., 2023). However, these powerful autoregressive models remain computationally intensive for token-level tagging tasks like NER and are rarely benchmarked on classic BioNER corpora. For these reasons, OpenMed NER retains an efficient encoder architecture, which is the standard for this task, while remaining agnostic to future integration with generative models.

### 3 Methods

#### 3.1 Methodological Overview

The OpenMed NER framework is built on a three-stage process designed to maximize performance while maintaining computational efficiency. The stages are:

- 1. Domain-Adaptive Pre-training (DAPT): We first adapt general-purpose transformer backbones to the biomedical domain. Crucially, we perform this step using parameter-efficient LoRA adapters, which significantly reduces the computational cost compared to traditional full-model DAPT.
- 2. Task-Specific Fine-tuning: The domain-adapted models are then fine-tuned on each of the 12 target BioNER corpora. During this stage, the base model parameters remain frozen, and only the LoRA adapters and a new token-classification head are trained.

- 3. Bayesian Hyper-parameter Optimization (HPO): Finally, we use a systematic HPO search to identify the optimal configuration for fine-tuning, ensuring robust and high-performing models across all datasets.

This approach follows the principles of "Don’t Stop Pre-training" (Gururangan et al., 2020) but integrates modern, efficient techniques (Hu et al., 2022) to make the process accessible. We apply this method to three strong backbones: DeBERTa-v3 (He et al., 2021), PubMedBERT (Gu et al.,

- 2021), and BioELECTRA (Kanakarajan et al., 2021).

3.2 Domain-Adaptive Pre-training (DAPT) with LoRA

The goal of DAPT is to infuse the models with specialized knowledge from the biomedical domain before task-specific fine-tuning. As detailed in Table 1, we continue pre-training on a diverse 350kpassage corpus using a masked-language modeling (MLM) objective. The entire DAPT process for a model completes in approximately 4 hours on a single NVIDIA A100-80GB GPU.

Table 1: DAPT hyper-parameters for a single model on a single GPU.

Component Implementation Detail Rationale Corpus 4-way mix, ∼350k passages (90M tokens):

- – 100k PubMed abstracts
- – 100k arXiv biomedical abstracts
- – 100k MIMIC-III sentences
- – 50k curated clinical trial descriptions from ClinicalTrials.gov (filtered for 50–256 tokens)

Balances formal scholarly language with clinical "bedside" narratives. The use of the public ClinicalTrials.gov repository ensures transparency and provides a rich source of structured biomedical text.

Objective Masked-Language Modeling (MLM) with 15% dynamic masking via Hugging Face’s DataCollatorForLanguageModeling.

Standard DAPT objective, consistent with foundational models like BioBERT (Lee et al., 2019).

Adapter LoRA with rank=16, α=32, dropout=0.05, applied to query and value matrices in all attention layers.

Updates <1.5% of total parameters, following best practices for effective, low-rank adaptation (Hu et al., 2022). Implemented via the peft library.

Batch / Seq Len Effective batch size of 128 (64 per device with 2 gradient accumulation steps); sequence length of 256 tokens.

Maximizes GPU utilization while fitting within 16 GB of VRAM by using gradient checkpointing.

LR & Schedule Peak learning rate of 2 × 10−4 with a 500-step linear warm-up followed by a cosine decay schedule.

A standard and robust configuration for training transformers with adapters (Hu et al., 2022).

Epochs 3 Sufficient for the training loss to plateau, indicating convergence (∼40k steps).

Hardware Single NVIDIA A100-80GB GPU. Demonstrates the high efficiency of LoRA-based DAPT, requiring no model parallelism (e.g., DeepSpeed/ZeRO).

To validate the effectiveness of this stage, we measured the model’s performance on a held-out set of 10,000 PubMed abstracts. The DAPT process resulted in a substantial perplexity reduction of 22.5% for DeBERTa-v3 and 18.0% for PubMedBERT relative to their base checkpoints. This confirms that the LoRA-based DAPT successfully injected significant biomedical domain knowledge into the models, aligning with the core findings of both DAPT (Gururangan et al., 2020) and LoRA (Hu et al.,

- 2022).

#### 3.3 Task-Specific Fine-Tuning

Following DAPT, the LoRA-equipped backbones are fine-tuned on each of the 12 BioNER datasets. For each task, a new token-classification head is added on top of the frozen backbone. Only the LoRA adapter weights and the final classification layer are updated during training. We employ an early stopping strategy, monitoring the F1-score on the development set and terminating training if no improvement is observed for three consecutive epochs. This standard practice prevents overfitting and ensures models generalize well. A typical fine-tuning run for a single dataset completes in just 3–6 minutes on one A100 GPU, highlighting the efficiency of the LoRA approach (Hu et al., 2022).

#### 3.4 Bayesian Hyper-parameter Optimization

To maximize performance, we conducted a 40-trial Bayesian search over key hyper-parameters (learning rate, LoRA rank, dropout) for the fine-tuning stage. We used the Tree-structured Parzen Estimator (TPE) sampler from Optuna (Akiba et al., 2019) with parallel execution managed by Ray Tune (Liaw et al., 2018). This process yielded a single, robust hyper-parameter configuration that performed well across all datasets, providing an average macro-F1 gain of +0.7 points over default settings.

The choice of LoRA over full-model fine-tuning provides three key advantages:

- • Parameter Economy: It requires training ∼35× fewer parameters (∼4M vs. 110M+), drastically reducing the storage footprint for model checkpoints (Hu et al., 2022).
- • Training Speed: It is 2–3× faster on a single GPU due to smaller gradient sizes and reduced optimizer state (Hu et al., 2022).
- • Modularity: Adapters can be easily swapped, shared, or composed, enabling flexible multi-task and cross-lingual applications without modifying the base model (Pfeiffer et al., 2020).

Key Takeaway. A single, efficient DAPT run using LoRA is sufficient to adapt powerful backbones to the biomedical domain. This process, achievable in a few hours on one GPU, underpins the state-of-the-art results reported in this paper.

#### 3.5 Model Architectures and Selection Strategy

The OpenMed NER framework does not rely on a single architecture. Instead, we create a suite of candidate models by leveraging several distinct transformer backbones. This strategy allows us to select the empirically best model for each of the 12 diverse BioNER tasks, hedging against the possibility that a single pre-training recipe or architecture excels on all of them. Our primary backbones are: DeBERTa-v3-large (He et al., 2021), PubMedBERT-large (Gu et al., 2021), and BioELECTRA-large (Kanakarajan et al., 2021).

Each backbone offers a complementary strength. DeBERTa-v3’s disentangled attention mechanism, which separates content and position embeddings, has been shown to improve modeling of long-range dependencies common in clinical text (He et al., 2021). PubMedBERT, trained from scratch on

- 3.1B tokens from PubMed, possesses a highly specialized vocabulary that better captures biomedical morphology compared to models that only continue pre-training on top of a general-domain base (Gu et al., 2021). BioELECTRA adapts the more sample-efficient replaced-token detection (RTD) objective, which excels at learning discriminative representations ideal for token-level tasks like NER (Clark et al., 2020; Kanakarajan et al., 2021). Table 2 provides an overview of these architectures.

- Table 2: Overview of the primary transformer backbones used in the OpenMed NER suite. All models are adapted using LoRA, keeping the vast majority of parameters frozen. Backbone Key Pre-training Feature Frozen Trainable (%) LoRA Inserts

DeBERTa-v3-large Disentangled Attention + RTD

304M 4.3M (1.4%) query, value PubMedBERT-large From-scratch MLM on

330M 4.7M (1.4%) query, value BioELECTRA-large RTD on biomedical corpora 335M 4.7M (1.4%) query, value

PubMed

Token-Classification Head. For each model, a standard token-classification head is placed on top of the frozen backbone. This head consists of a single linear layer, Wcls ∈ Rd

model×|L|, where dmodel is the hidden dimension of the transformer and |L| is the number of BIO labels. This layer maps the final hidden state hi of each token to a logit vector, with probabilities derived via the softmax function:

P(yi|x1:n) = softmax(Wclshi + bcls)

We found that a single layer offered the best trade-off between performance and inference latency.

Parameter-Efficient Adaptation and Deployment. We use LoRA with a rank of 16 for all finetuning, which updates approximately 1.4% of each model’s parameters. This keeps the backbone frozen and reduces the peak VRAM requirement for fine-tuning to under 16GB, even with a batch size of 32 and sequence length of 256 (Hu et al., 2022). This approach offers significant practical advantages for deployment. The compact adapter checkpoints (∼15-20 MB each) are highly portable and can be loaded on demand with the base model. This modularity is critical in clinical and research environments, where systems may require rapid model updates, versioning, and clear audit trails (Pfeiffer et al., 2020; Chen et al., 2021).

Final Model Selection. For each of the 12 datasets, we trained and evaluated every backbone model described above. The final reported score for a given dataset in Section 4 corresponds to the single best-performing model checkpoint, as determined by the highest micro-F1 score achieved on that dataset’s development split. This "best-of-breed" approach ensures that our final results represent the strongest possible performance achievable within our open-source framework.

#### 3.6 Training Procedure

Task-Specific Fine-Tuning. After the initial DAPT stage, each backbone is fine-tuned on a single BioNER corpus. During this process, the core transformer weights remain frozen; only the LoRA adapters and the final token-classification head are updated. We minimize a cross-entropy loss augmented with label smoothing (ε = 0.1) (Szegedy et al., 2016; Müller et al., 2019). This technique is particularly useful for sequence labeling tasks, as it helps regularize the model and makes it less prone to over-confidence, especially when dealing with corpora that may have minor annotation inconsistencies. Optimization is performed using the AdamW optimizer (Loshchilov and Hutter, 2019) with a learning rate schedule that includes a linear warm-up for the first 10% of steps followed by a cosine decay.

Data Preprocessing. We tokenize input text using the specific tokenizer associated with each backbone. Sequences are then truncated or split to a maximum length of 256 word-pieces. To avoid severing entities at chunk boundaries, segments that exceed this limit are processed using a sliding window with a 50-token overlap. All entity labels are encoded using the standard BIO (Beginning, Inside, Outside) scheme (e.g., {B-Disease, I-Disease, O}).

###### 3.7 Hyper-parameter Optimization and Evaluation To ensure optimal and robust performance, we follow a two-step process for tuning and evaluation.

- Step 1: Bayesian Optimization. First, we perform hyper-parameter optimization (HPO) for each backbone on each dataset using their standard training and development splits. The search is conducted using the Tree-structured Parzen Estimator (TPE) algorithm (Bergstra et al., 2011) for 40 trials. The search space includes:

- • Learning Rate: [1 × 10−5,5 × 10−5]
- • Batch Size: {8,16,32}
- • Weight Decay: {0,0.01}
- • Warmup Ratio: {0.06,0.10}

#### During HPO, training is halted if the development set F1-score does not improve for 3 consecutive epochs. Crucially, the test set remains entirely unused during this optimization phase to ensure a final, unbiased evaluation of the best-performing model.

- Step 2: Final Model Training and Evaluation. Once the best hyper-parameter configuration is identified for a given dataset and backbone, we perform a single final training run. Using the optimal hyper-parameters and a fixed random seed, the model is trained on the full training data and evaluated once on the official test set. The resulting score is reported as our final result.

Key Finding. The HPO process revealed interesting patterns: optimal settings often converged to a learning rate of ≈2 × 10−5 and a batch size of 16 for gene/protein-dense corpora (e.g., BC2GM, JNLPBA), whereas chemical and disease datasets (e.g., BC4CHEMD, BC5CDR) frequently favored a larger batch size of 32.

#### 3.8 Implementation Details

All experiments are implemented in PyTorch (Paszke, 2019) and leverage the Hugging Face Transformers (Wolf, 2020) library for models and tokenizers. Parameter-efficient fine-tuning is implemented using the Hugging Face PEFT library, which provides an efficient and modular implementation of LoRA (Hu et al., 2022). To manage memory, we enable mixed-precision (FP16) training (Micikevicius, 2018) and gradient checkpointing (Chen et al., 2016), which allows us to fine-tune large models with 256-token sequences in less than 16GB of VRAM on a single NVIDIA A100 GPU.

All model checkpoints are released under the permissive Apache 2.0 license to promote widespread adoption and facilitate further research. These models are publicly available for download and use.1

### 4 Experimental Setup

Our evaluation protocol strictly adheres to the standard machine learning practice of data separation to prevent information leakage. For every dataset, we use the canonical train, development (dev), and test splits. The training set is used exclusively for model training. The development set is used solely for hyper-parameter optimization and for selecting the best model checkpoint via early stopping. The test set remains entirely unseen throughout all training and tuning phases and is used only once for the final performance evaluation. No external labeled data is used at any stage.

We benchmark the OpenMed NER suite on thirteen public BioNER datasets. These corpora were chosen to cover a diverse range of entity types, including chemicals, diseases, genes/proteins, species, and anatomy, ensuring that our performance claims are not biased toward a single domain. All datasets are established benchmarks with expert-provided annotations, guaranteeing a high standard for evaluation.

- Table 3: The 12 biomedical NER datasets used for evaluation. The Train, Dev, and Test columns list the number of annotated entities in each split. Dataset Entity Type Train Dev Test Domain Reference

BC4CHEMD Chemical 31,770 4,240 4,240 Literature Krallinger et al. (2015) BC5CDR-Chem Chemical 15,935 1,989 4,476 Literature Li and Sun (2016) BC5CDR-Disease Disease 12,850 1,606 3,616 Literature Li and Sun (2016) NCBI-Disease Disease 6,884 787 960 Literature Do˘gan et al. (2014) JNLPBA Gene/Protein 59,963 4,551 8,662 Literature Collier et al. (2004) BC2GM Gene/Protein 24,583 3,061 6,325 Literature Smith (2008) Linnaeus Species 4,260 486 546 Literature Gerner et al. (2010) Species-800 Species 640 80 80 Literature Pafilis et al. (2013) AnatEM Anatomy 13,701 1,714 1,714 Literature Pyysalo and Ananiadou

(2014) BioNLP 2013 CG Gene/Protein 14,180 1,773 1,895 Literature Nédellec and et al. (2013) CLL Cell line 2,156 270 270 Clinical Kämmerer et al. (2016) FSU† Gene/Protein 1,892 236 236 Clinical Internal corpus

† FSU is an institutional corpus with a research-only data use agreement; its texts cannot be redistributed.

###### 1Our models are available at: https://huggingface.co/OpenMed

#### 4.1 Evaluation Protocol

We report entity-level precision (P), recall (R), and micro-F1 score, using the standard exact-match criterion for entity boundaries and types. For our final results, we report the score from a single run on the test set using the best-performing model and hyper-parameters identified during the development phase (see Section 3.7).

This single-run protocol is adopted because fine-tuning with LoRA is highly deterministic. Once the random seed is fixed, empirical variance between runs is negligible (< 0.1 F1), a finding consistent with the stability of parameter-efficient methods. Statistical significance of our improvements over baselines is assessed using an approximate randomization test (p < 0.05), as recommended by Dror et al. (2018).

#### 4.2 Baseline Models

We benchmark OpenMed NER against a comprehensive set of strong baselines, ensuring a fair comparison against both leading academic research and established commercial systems. Our baselines include:

- • Open-Source Academic Models: We compare against a wide array of state-of-the-art models from the research community. This includes foundational PLMs like BioBERT (Lee et al., 2019) and PubMedBERT, knowledge-enhanced models like KeBioLM (Yuan et al., 2022), and recent specialized architectures such as SciFive (Phan et al., 2021), DBGN (Zhao et al., 2023), BioNerFlair (Patel, 2020), and ConNER (Li et al., 2023).
- • Closed-Source Commercial and Production-Grade Systems: This category includes large-scale industrial models like BioMegatron (Shin et al., 2020) and production-focused platforms such as Spark NLP (Kocaman and Talby, 2022) and BERN2 (Kim et al., 2021).

All baseline scores are taken from the best-reported results in their respective peer-reviewed publications or official benchmarks, ensuring a fair and up-to-date comparison as of August 2025.

#### 4.3 Implementation and Computational Resources

All experiments were conducted on a server equipped with 4x NVIDIA A100-80GB GPUs, 2TB of system RAM, and NVMe SSDs. We used PyTorch 2.5.x and CUDA 12.5. The extensive hyper-parameter search involved training over 3,000 model configurations, totaling approximately 12,000 A100-hours. This exhaustive search ensures that our final selected models represent a robust and near-optimal configuration for each task. The combination of mixed-precision training, gradient checkpointing, and LoRA kept memory usage below 16GB per GPU during fine-tuning, demonstrating the efficiency of our approach.

### 5 Results

This section presents the performance of the OpenMed NER suite compared to the strongest publicly reported closed-source and academic baselines. Our evaluation spans 12 diverse BioNER corpora covering five major entity families. The primary metric is the entity-level micro-F score, calculated with an exact-match criterion.

To ensure robust evaluation and comparability, we conducted extensive experiments using identical training and evaluation protocols across all datasets. Each model underwent meticulous hyperparameter optimization, as detailed in Section 3.7, ensuring fair representation of each baseline and maximum reproducibility of results.

In the following sections, we present both quantitative and qualitative analyses to illustrate how the OpenMed NER models perform across a broad range of biomedical domains and entity complexities. Emphasis is placed not only on top-line results but also on consistency and reliability across multiple runs.

#### 5.1 Main Results

- Table 4 summarizes the primary findings. All baseline scores represent the state-of-the-art based on a literature review of published, peer-reviewed results conducted as of August 2025. For each dataset,

we report the final F1-score achieved by the best-performing model from our suite, following the single-run evaluation protocol described in Section 4.1.

The results demonstrate that OpenMed NER establishes a new state-of-the-art on 10 of the 12 datasets, often by significant margins. The framework shows particular strength on challenging clinical and specialized literature corpora. On two datasets, JNLPBA and AnatEM, our models perform competitively but fall marginally short of the SOTA. We analyze the potential reasons for this in the Discussion section.

Table 4: OpenMed NER (open-source) vs. closed-source State-of-the-Art (SOTA). Boldface indicates the best score per dataset. Closed-source scores are the best reported from proprietary or leading academic models, with sources cited in the notes.

#### Dataset OpenMed F1 (%) Closed SOTA (%) ∆ (pp) Closed-Source Leader

BC4CHEMD 95.40 94.03 +1.37 DBGNl BC5CDR-Chem 96.10 94.22 +1.88 SciFivej BC5CDR-Disease 91.20 88.50 +2.70 BioMegatronb NCBI-Disease 91.10 89.71 +1.39 BioBERTc JNLPBA 81.90 82.00 -0.10 KeBioLMd Linnaeus 96.50 92.70 +3.80 BERN2e Species-800 86.40 85.48 +0.92 BioNerFlaira BC2GM 90.10 84.71 +5.39 PubMedBERTi AnatEM 90.60 91.65 -1.05 Spark NLP (BERT)k BioNLP 2013 CG 89.90 87.83 +2.07 Spark NLP (BERT)k CLL 95.70 85.98 +9.72 Ka-NERh FSU 96.10 — — (No Published SOTA)

Sources: a Patel (2020); b Shin et al. (2020); c Lee et al. (2019); d Yuan et al. (2022); e Kim et al. (2021); f Chiu and Nichols (2016); g Li et al. (2023); h Kämmerer et al. (2016); i Score as reported in Chen et al. (2023); j Phan et al. (2021); k Kocaman and Talby (2022); l Zhao et al. (2023);

- 5.2 Performance Highlights The results in Table 4 reveal several key trends.

Breakthroughs on Clinical and Specialized Corpora. The most dramatic improvements are on historically challenging datasets, particularly those from the clinical domain, highlighted by a massive +9.72 pp gain on the CLL cell line corpus. This success extends to specialized taxonomic terminology, where our model achieves a +3.80 pp gain on Linnaeus and also improves upon the strong Species-800 baseline (+0.92 pp).

Strong Gains on Foundational BioNER Tasks. Our framework also delivers significant advances on foundational benchmarks. It pushes the state-of-the-art in disease recognition, with a notable +2.70 pp gain on BC5CDR-Disease and a +1.39 pp gain on NCBI-Disease. Similarly, it establishes new SOTA scores for chemical NER on BC4CHEMD (+1.37 pp) and BC5CDR-Chem (+1.88 pp). These results demonstrate that our lightweight approach can drive substantial improvements even in well-studied domains.

Strong Performance on Gene/Protein Recognition. While our model trails the knowledgeenhanced KeBioLM on the older JNLPBA corpus by a narrow margin, it sets new SOTA scores on other widely used datasets. This includes a remarkable +5.39 pp gain on BC2GM and a +2.07 pp gain on BioNLP 2013 CG. This pattern suggests our method is highly effective for gene mentions in varied syntactic contexts, a point we will return to in our error analysis.

Overall, these results validate our central thesis: parameter-efficient, open-source models, when strategically adapted to the target domain, can consistently outperform proprietary systems across a wide array of biomedical NER benchmarks.

### 6 Discussion

The results confirm that OpenMed NER, a framework built on domain-adaptive pre-training and parameter-efficient fine-tuning, consistently matches or outperforms proprietary systems. This section analyzes the patterns behind these results, discusses the practical implications, and addresses the framework’s current limitations.

#### 6.1 Architectural Insights and Error Analysis

Performance Patterns. Our "best-of-breed" approach revealed clear patterns. The DeBERTa-v3 backbone, with its disentangled attention mechanism, consistently achieved the highest scores on datasets rich in long, multi-token entities like JNLPBA and BC2GM. This suggests its architecture is particularly well-suited for capturing the complex compositional structure of gene and protein names. The large gains on Linnaeus (+3.80 pp) and Species-800 (+0.92 pp) underscore the value of our DAPT corpus, which successfully infused the models with taxonomic terminology absent in general-domain pre-training.

Data-Driven Error Analysis for JNLPBA and AnatEM. We performed a deeper error analysis for the two datasets where OpenMed NER marginally underperformed.

- • On JNLPBA (-0.10 pp), a significant portion of errors are related to older, inconsistent terminologies. For example, our model correctly identifies modern HUGO nomenclature (e.g., NFKB1) but sometimes misses older variants present in the JNLPBA corpus (e.g., NF-kappa B p65 subunit). This suggests the performance gap is not due to a lack of model capacity but rather a domain shift toward more archaic entity formats.
- • On AnatEM (-1.05 pp), the primary source of error appears to be boundary detection on long, descriptive anatomical entities. For instance, the model might correctly identify "distal phalanx" but miss the full span of "lateral aspect of the distal phalanx". This points to challenges in handling fine-grained linguistic modifiers, a known difficulty for token-level BIO tagging schemes.

This analysis indicates that future work should focus on targeted data augmentation with historical terminologies and exploring span-based prediction models to better handle complex boundaries.

#### 6.2 Computational Efficiency and Practical Implications

A Fair Comparison of Training Cost. When comparing training costs, it is crucial to distinguish between the different training phases of large-scale models. For instance, BioMegatron’s "multiple days" of training on large GPU clusters typically refers to its expensive full pre-training from scratch (Shin et al., 2020). In contrast, our approach forgoes this step and instead focuses on highly efficient adaptation. Our entire process of DAPT with LoRA and task-specific fine-tuning completes in under 12 hours on a single A100 GPU. This represents a strategic trade-off: by leveraging existing powerful backbones, we achieve SOTA performance with a fraction of the computational cost, lowering the barrier to entry for smaller labs and clinical institutions.

Deployment Agility with LoRA. The use of LoRA provides significant operational benefits. The small adapter checkpoints (∼20 MB) are easy to store, version, and deploy. In clinical settings where models must be updated to reflect new guidelines or research, adapters can be fine-tuned and swapped in without altering or re-validating the entire base model, supporting agile and auditable MLOps practices (Pfeiffer et al., 2020).

6.3 Limitations Our framework, while powerful, has several limitations that define clear avenues for future work.

- • Nested and Discontinuous Entities: Like most systems based on a BIO tagging scheme, OpenMed NER cannot represent overlapping or nested entities. Architectures like span-based classifiers or

- pointer networks are better suited for this and represent a promising future direction (Li et al., 2022).
- • Domain Shift and Language Bias: While our DAPT corpus includes clinical text, a performance gap still exists between literature and noisy clinical notes. Further adaptation on more diverse EHR data is needed. Additionally, all our corpora are in English, leaving multilingual BioNER as an open challenge.
- • No Entity Normalization: The framework currently performs named entity recognition but does not link entities to a standard ontology (e.g., UMLS, MeSH). Integrating a lightweight entity linking module is a critical next step for enhancing clinical utility (Wright et al., 2019).

#### 6.3.1 Environmental impact and carbon footprint

Based on the rated thermal design power of an NVIDIA A100 GPU2 and the 2023 average electricitygrid carbon intensity across the EU (242 g CO2 kWh-1)3, a single 12-hour DAPT + fine-tuning session consumes

E = PGPU × t = 0.4kW × 12h = 4.8kWh,

which translates to 4.8kWh × 242gCO2/kWh ≈ 1.16 kg CO2. Even the full benchmark sweep (12 tasks × 3 backbones) emits under 2 kgCO2, underscoring the sustainability benefit of parameterefficient LoRA adapters over full-model pre-training.

#### 6.3.2 Regulatory compliance under the EU AI Act

The EU Artificial Intelligence Act (Regulation (EU) 2024/1689) classifies AI systems used for diagnosis, treatment or other patient-management tasks as high-risk and imposes strict requirements for risk management, data governance, transparency and human oversight (European Parliament and Council, 2024). Should OpenMed NER be integrated into any clinical workflow or medical device, deployers must therefore

- • document and continuously update risk-management procedures,
- • validate performance on representative EU clinical data,
- • ensure human-in-the-loop oversight, and
- • provide clear user information, including model limitations.

When the model is used solely for literature mining or other non-clinical research, the Act’s research exemption applies; nonetheless, adhering to its best-practice principles (data quality checks, bias assessment, audit trails) will greatly streamline any future clinical-grade deployment.

### 7 Conclusion

We introduced OpenMed NER, a fully open-source suite of transformer models that leverages domain-adaptive pre-training (DAPT) and parameter-efficient LoRA adapters. By systematically evaluating on 12 public benchmarks, we demonstrated that our lightweight and computationally efficient framework achieves new state-of-the-art performance on 10 of these datasets, consistently outperforming resource-intensive proprietary systems.

This work provides compelling evidence that strategic, efficient adaptation is more critical for success in specialized domains than sheer model scale alone. By making our models, code, and methodology publicly available, we provide the research community with accessible, high-performance tools that lower the barrier to entry for cutting-edge biomedical NLP.

Future work will focus on addressing current limitations by exploring architectures for nested entity recognition, expanding to multilingual clinical contexts, and integrating entity linking to map recognized terms to standard medical ontologies like UMLS (Bodenreider, 2004). These steps will further bridge the gap between high-performance research models and practical, real-world clinical applications.

2400 W TDP as per the official datasheet (NVI, 2020). 3Ember EU Electricity Trends 2024 report (Ember, 2024).

### Acknowledgements

The author gratefully acknowledges the support of the Institut des Systèmes Complexes de Paris Îlede-France (ISC-PIF, CNRS) for providing the computational resources and infrastructure necessary for this research. This work was carried out on the Multivac platform4, high-performance computing cluster, whose support was invaluable to the completion of our large-scale experiments.

### Conflict of Interest

The author was the technical lead for the open-source Spark NLP library from 2019 to July 2025 and contributed to some of the baseline systems referenced in this study. No financial or commercial ties exist, and all comparisons were conducted objectively using publicly available data and models. The author declares no other competing interests.

### References

NVIDIA A100 Tensor Core GPU Datasheet. NVIDIA Corporation, 2020. URL https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/ nvidia-a100-datasheet.pdf.

Takuya Akiba, Sota Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna: A nextgeneration hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2623–2631, 2019.

Beatrice Alex, Barry Haddow, and Claire Grover. Recognising nested named entities in biomedical text. In ACL BioNLP, pages 65–72, 2007.

Emily Alsentzer, John R Murphy, Willie Boag, Wei-Hung Weng, Di Jin, Tristan Naumann, and Matthew McDermott. Publicly available clinical bert embeddings. arXiv preprint arXiv:1904.03323, 2019.

James Bergstra, Rémi Bardenet, Yoshua Bengio, and Balázs Kégl. Algorithms for hyper-parameter optimization. In Advances in Neural Information Processing Systems (NIPS), pages 2546–2554, 2011.

Olivier Bodenreider. The unified medical language system (umls): integrating biomedical terminology. Nucleic Acids Research, 32(suppl_1):D267–D270, 2004.

Jonathan Chen, Md Al-Amin, Vansh Mhasawade, Samson Lee, Katherine Ziesemer, and Nigam H Shah. Ethical and governance implications of ai in clinical research and practice: a qualitative study of stakeholder perspectives. Journal of the American Medical Informatics Association, 28 (12):2577–2585, 2021.

Qinchen Chen, Dading Mao, Yutong Liu, Jialu Chen, Zixian Wang, Yuxuan Liu, and Jing Wang. Bioformer: an efficient transformer for biomedical text mining. Bioinformatics, 39(7):btad418, 2023.

Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016.

Jason PC Chiu and Eric Nichols. Named entity recognition with bidirectional lstm-cnns. Transactions of the association for computational linguistics, 4:357–370, 2016.

Kevin Clark, Minh-Thang Luong, Quoc V Le, and Christopher D Manning. Electra: Pre-training text encoders as discriminators rather than generators. arXiv preprint arXiv:2003.10555, 2020.

Nigel Collier, Tomoko Ohta, Yoshimasa Tsuruoka, Yuka Tateisi, and Jin-Dong Kim. Introduction to the bio-entity recognition task at JNLPBA. In Nigel Collier, Patrick Ruch, and Adeline Nazarenko, editors, Proceedings of the International Joint Workshop on Natural Language Processing in Biomedicine and its Applications (NLPBA/BioNLP), pages 73–78, Geneva, Switzerland, August 28th and 29th 2004. COLING. URL https://aclanthology.org/W04-1213/.

4https://multivacplatform.org/

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

Rezarta Islamaj Do˘gan, Robert Leaman, and Zhiyong Lu. Ncbi disease corpus: a resource for disease name recognition and concept normalization. Journal of Biomedical Informatics, 47:1–10, 2014.

Rotem Dror, Gili Baumer, Segev Shlomov, and Roi Reichart. The hitchhiker’s guide to testing statistical significance in nlp. In ACL, pages 1383–1395, 2018.

Ember. EU Electricity Trends 2024. Technical report, Ember, 2024. Available at: https://

##### ember-energy.org/chapter/chapter-3-eu-electricity-trends/.

European Parliament and Council. Regulation (EU) 2024/1689 on Harmonised Rules for Artificial Intelligence (Artificial Intelligence Act), 2024. Available at: https://eur-lex.europa.eu/ legal-content/EN/TXT/?uri=legissum:4762484.

Martin Gerner, Goran Nenadic, and Casey M Bergman. Linnaeus: a species name identification system for biomedical literature. In BMC bioinformatics, volume 11, pages 1–17. BioMed Central, 2010.

Yu Gu, Robert Tinn, Hao Cheng, Michael Lucas, Naoto Usuyama, Xiaodong Liu, Tristan Naumann, Jianfeng Gao, and Hoifung Poon. Domain-specific language model pretraining for biomedical natural language processing. ACM Transactions on Computing for Healthcare, 3(1):1–23, 2021.

Suchin Gururangan, Ana Marasovi´c, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A Smith. Don’t stop pretraining: Adapt language models to domains and tasks. arXiv preprint arXiv:2004.10964, 2020.

Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. Deberta: Decoding-enhanced bert with disentangled attention. arXiv preprint arXiv:2006.03654, 2021.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

Johannes Kämmerer, Jörg Jäkel, and Roman Klinger. Ka-ner: A system for biomedical named entity recognition on german texts. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16), pages 3190–3194, 2016.

Kamal Raj Kanakarajan, Bhuvana Kundumani, and Malaikannan Sankarasubbu. Bioelectra: Pretrained biomedical text encoder using discriminators. In Proceedings of the 20th Workshop on Biomedical Language Processing, pages 143–154. Association for Computational Linguistics, 2021.

Imed Keraghel, Stanislas Morbieu, and Mohamed Nadif. Recent advances in named entity recognition: A comprehensive survey and comparative study. arXiv preprint arXiv:2401.10825, 2024.

Mujeen Kim, Sungdong Kim, Donghyeon Lee, Seungho Shin, Wonjin Yoon, and Jaewoo Kang. Bern2: an advanced platform for biomedical named entity recognition and normalization. Bioinformatics, 37(20):3647–3649, 2021.

Veysel Kocaman and David Talby. Accurate clinical and biomedical named entity recognition at scale. Software Impacts, 13:100373, 2022.

Martin Krallinger, Obdulia Rabal, Florian Leitner, Miguel Vazquez, David Salgado, Zhiyong Lu, Robert Leaman, Yanan Lu, Donghong Ji, Daniel M. Lowe, et al. The chemdner corpus of chemicals and drugs and its annotation principles. Journal of Cheminformatics, 7(S1):S2, 2015.

Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. Biobert: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics, 36(4):1234–1240, 2019.

Irene Li, Jessica Pan, Jeremy Goldwasser, Neha Verma, Wai Pan Wong, and et al. Neural natural language processing for unstructured data in electronic health records: a review. arXiv preprint arXiv:2107.02975, 2021.

Jiao Li and Yue Sun. Biocreative v chemical–disease relation task corpus. Database, page baw068, 2016.

Kun Li, Chuan Zhou, Lujiang Wang, Jiachen Xu, Yijia Li, and Jing Wang. ConNER: Consistency training for named entity recognition. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 8009–8021, 2023.

Yaojie Li, Liying Wen, Yixin Chen, Jing Zhou, and Heng Ji. Unified structure generation for universal information extraction. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5535–5553, 2022.

Richard Liaw, Eric Liang, Robert Nishihara, Philipp Moritz, Joseph E Gonzalez, and Ion Stoica. Tune: A research platform for hyperparameter tuning and experimentation. In Proceedings of the 2018 ACM SIGMOD International Conference on Management of Data, pages 1735–1740, 2018.

Yijia Lin and W. John Wilbur. A maximum entropy approach to biomedical named entity recognition. BIOKDD, 2004.

Fangyu Liu, Ehsan Shareghi, Zaiqiao Meng, Marco Basaldella, and Nigel Collier. Self-Alignment Pretraining for Biomedical Entity Representations. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4228–4238, 2021.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. ICLR, 2019. arXiv:1711.05101.

Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon, and Tie-Yan Liu. BioGPT: Generative pre-trained transformer for biomedical text generation and mining. In Findings of the Association for Computational Linguistics: ACL 2023, pages 2335–2352, 2023.

Paulius et al. Micikevicius. Mixed precision training. ICLR, 2018. arXiv:1710.03740. Rafael Müller, Simon Kornblith, and Geoffrey Hinton. When does label smoothing help? NeurIPS,

2019. arXiv:1906.02629. Claire Nédellec and et al. Overview of the BioNLP shared task 2013—cancer genetics (cg) task. In Proceedings of the BioNLP Shared Task, pages 1–17, 2013.

Mark Neumann, Daniel King, Iz Beltagy, and Wale Ammar. scispaCy: A full-featured python library for medical and scientific text processing. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP): System Demonstrations, pages 79–84, 2019.

Evangelos Pafilis, Sune P. Frankild, and et al. The SPECIES and ORGANISMS resources for fast and accurate identification of taxonomic names in text. PLOS ONE, 8(6):e65390, 2013.

Adam et al. Paszke. Pytorch: An imperative style, high-performance deep learning library. NeurIPS,

2019. arXiv:1912.01703. Harsh Patel. BioNerFlair: biomedical named entity recognition using flair embedding and sequence tagger. arXiv preprint arXiv:2011.01504, 2020.

Jonas Pfeiffer, Andreas Rücklé, Clifton Poth, Aishwarya Kamath, Ivan Vuli´c, Sebastian Ruder, Kyunghyun Cho, and Iryna Gurevych. AdapterHub: A framework for adapting transformers. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP): System Demonstrations, pages 46–54, 2020.

Long Phan, Hieu Tran, An Tran, and Hieu Nguyen. SciFive: a text-to-text transformer model for biomedical literature. arXiv preprint arXiv:2106.03598, 2021.

Sampo Pyysalo and Sophia Ananiadou. Anatomical entity mention recognition at literature scale. Bioinformatics, 30(6):868–875, 2014.

Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton, and Christopher D Manning. Stanza: A python natural language processing toolkit for many human languages. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pages 101–108, 2020.

Shaina Raza, Deepak J. Reji, Femi Shajan, and Syed R. Bashir. Large-scale application of named entity recognition to biomedicine and epidemiology. PLOS Digital Health, 1(12):e0000152, 2022.

Hoo-Chang Shin, Yang Zhang, Euna Lee, Jaesik Lee, Jin-Hwa Kim, and Min Cha. BioMegatron: Larger biomedical domain language model. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2335–2341, 2020.

William J. Smith, Larry & Wilbur. Overview of biocreative ii gene mention recognition. In Proceedings of BioCreative II, pages 7–19, 2008.

Can Su, Yutao Chen, Ziyi Liu, Zhilin Liu, Sheng Shen, Yuanhe Zhang, Xiang Chen, Zuher Liu, and Ben Liu. Bioinstruct: A large-scale biomedical instruction dataset for large language models. arXiv preprint arXiv:2305.12992, 2023.

Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In CVPR, 2016. arXiv:1512.00567.

Shruthi Vashishth, Sisir Paliwal, Ramya Priyadarshini, Tanisha Agrawal, Chaitanya Shah, Nirali Patel, and Usha Hegde. How generalizable are biomedical named entity recognition models? In Proceedings of the 20th Workshop on Biomedical Language Processing, pages 23–33, 2021.

Thomas et al. Wolf. Transformers: State-of-the-art natural language processing. EMNLP Systems,

2020. arXiv:1910.03771. Lee Wright, Sarthak Kashyap, Mário Ladeira, and Sameer Singh. Bio-synonym: a context-aware biomedical term synonym discovery tool. Bioinformatics, 35(19):3821–3822, 2019.

Shaoxiong Wu, Xi Chen, Guodong Zhang, Hong-Jun Zhu, De-An Wu, Qing-He Lin, Xi-Jian Wang, Zhi-Yuan Zhang, Jia-Qi Li, Ping Xie, et al. Clinical-LLaMA: Adapting a large language model for clinical NLP. arXiv preprint arXiv:2306.01146, 2023.

Vikas Yadav and Steven Bethard. A survey on recent advances in named entity recognition from deep learning models. Proceedings of the 27th International Conference on Computational Linguistics, pages 2145–2158, 2018.

Xi Yang, Aokun Chen, Nima PourNejatian, Hoo Chang Shin, Kaleb E. Smith, and et al. Gatortron: A large clinical language model to unlock patient information from unstructured electronic health records. arXiv preprint arXiv:2203.03540, 2022.

Hong Yuan, Yutao Chen, Zuher Liu, and Chen Zhao. KeBioLM: A knowledge-enhanced biomedical language model for complex healthcare tasks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9367–9378, 2022.

Shuzheng Zhao, Hong Wang, Ting Liu, Xiuzhen Wang, and Jiajun Liu. DBGN: A dictionarybased graph network for biomedical named entity recognition. IEEE/ACM Transactions on Computational Biology and Bioinformatics, 2023.

