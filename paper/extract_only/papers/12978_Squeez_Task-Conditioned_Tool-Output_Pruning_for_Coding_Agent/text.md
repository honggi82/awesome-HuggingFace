## Squeez: Task-Conditioned Tool-Output Pruning for Coding Agents

Ad´am´ Kov´acs KR Labs kovacs@krlabs.eu https://github.com/KRLabsOrg/squeez

# arXiv:2604.04979v1[cs.SE]4Apr2026

[Figure 1]

### Abstract

Coding agents repeatedly consume long tool observations even though only a small fraction of each observation matters for the next step. We study task-conditioned tooloutput pruning: given a focused query and one tool output, return the smallest verbatim evidence block the agent should inspect next. We introduce a benchmark of 11,477 examples built from SWE-bench repository interactions and synthetic multiecosystem tool outputs, with a manually curated 618-example test set. We fine-tune Qwen 3.5 2B with LoRA and compare it against larger zero-shot models and heuristic pruning baselines. Our model reaches 0.86 recall and 0.80 F1 while removing 92% of input tokens, outperforming zero-shot Qwen 3.5 35B A3B by 11 recall points and all heuristic baselines by a wide margin.

### 1 Introduction

Coding agents work over streams of file reads, grep hits, stack traces, build logs, API responses, and version-control history (Yang

- et al., 2024; Wang et al., 2025). In such observations, only a small fraction of the tokens is relevant to the next decision, yet the agent often has to re-read the entire output. This makes pruning a central efficiency problem for agent systems (Jiang et al., 2023, 2024; Hwang
- et al., 2025; Kerboua et al., 2025; Wang et al., 2026).

We study a narrow but practical version of this problem: given a focused query and a single tool observation, return the smallest verbatim evidence block the agent should inspect next. The task is not to solve the issue from one observation, but to preserve the relevant evidence and discard the rest. This distinguishes

our setting from generic prompt compression and from document-level retrieval compression, which typically operate on prose documents or retrieved passages rather than mixed-format tool output.

We make three contributions. We formulate task-conditioned tool-output pruning as a benchmark task for coding agents, release a dataset spanning real SWE-bench repository interactions and synthetic multi-ecosystem tool outputs, and show that a compact Qwen 3.5 2B model fine-tuned for this task substantially outperforms larger zero-shot models and simple heuristic baselines. Because the model can be served through vLLM or used as a CLI filter on piped tool output, it can be inserted into existing coding-agent stacks such as Codex or Claude Code with minimal changes to the surrounding agent loop. The model, dataset, and evaluation code and CLI are publicly available under the Apache 2.0 license.

### 2 Related Work

Prompt compression methods such as LLMLingua (Jiang et al., 2023) and LongLLMLingua (Jiang et al., 2024) compress long prompts before generation. LLMLingua uses a coarseto-fine compression pipeline with a budget controller and iterative token removal, while LongLLMLingua adapts compression to longcontext settings with query-aware compression and document reordering. These methods target prompt efficiency at the token or promptblock level rather than query-conditioned extraction from a single mixed-format tool observation. Abstractive summarization addresses a different problem again, since it rewrites salient content rather than preserving verbatim source evidence (See et al., 2017; Liu and Lapata, 2019).

Document-level context pruners are closer to our work. EXIT (Hwang et al., 2025) performs extractive compression for retrievalaugmented generation by selecting relevant portions of retrieved textual context for a downstream question-answering stage. Provence (Chirkova et al., 2025) formulates context pruning as sequence labeling, combining pruning with reranking, and training on diverse QA domains to obtain robust out-of-the-box RAG pruning. Open-source models such as Zilliz Semantic Highlight (Zilliz, 2025) adapt this idea to lightweight semantic highlighting, producing token- and sentence-level relevance scores over retrieved documents for real-time highlighting. These systems, however, assume retrieved passages or document text. They do not target the mixed observations that arise in coding agents, where code, logs, shell traces, metadata, and structured outputs are interleaved in a single artifact.

Our setting is also related to agent-specific pruning. FocusAgent (Kerboua et al., 2025) reduces the context available to web agents by trimming web observations, while SWE-Pruner (Wang et al., 2026) targets coding agents more directly by pruning repository code context. The nearest of these to our work is SWEPruner, but its emphasis is repository-level code context rather than single tool observations spanning files, logs, build outputs, and other non-code modalities.

Finally, our task is related to extractive QA and supporting-evidence supervision, where outputs are grounded in explicit spans or sentence sets rather than free generation (Rajpurkar et al., 2016; Thorne et al., 2018; Yang et al., 2018). Verbatim evidence extraction has also been applied in domain-specific QA, where lightweight extractive pipelines select grounded sentences to prevent hallucination (Kovacs et al., 2025). The common concern is faithfulness: outputs should remain traceable to source evidence. Our setting differs in focusing on query-conditioned pruning of tool observations rather than answer production from prose documents.

### 3 Task Definition

Figure 1 illustrates the overall pipeline. The benchmark input is a pair (q,o), where q is a

Input Query: “Find the traceback that explains the ImportError.” Tool output: 501 lines from read file, grep, pytest, git log, . . .

annotate

##### Grounded spans

- 183: ):

- 184: super(). init (

- 185: import name=...

191: self.name = name 193: self.subdomain = ... Gold: (start line, end line)

derive

Generative output <relevant lines>

- 184: super(). init (

- 185: import name=... 191: self.name = name </relevant lines> Eval: recall, F1, compression

Figure 1: System overview. The benchmark stores grounded spans over raw tool output, then derives a generative training target. At evaluation, predicted lines are compared to gold spans using recall, F1, and compression.

short task-conditioned extraction query and o is one raw tool observation. The output is a set of one or more contiguous spans over o:

Y = {(s1,e1),...,(sk,ek)},

where each span refers to line indices in the original output.

The task is intentionally narrower than bug solving. The model is not asked to infer the correct patch from one observation, but to extract the minimal evidence block that would help an agent on the next reasoning step. Queries are short, concrete, and tool-aware. They may be easy, medium, or moderately semantic; they need not avoid lexical cues if those cues reflect the underlying agent need. The key requirement is that they define a plausible pruning decision rather than a full diagnosis.

### 4 Dataset

4.1 Data Sources The benchmark draws on two sources of raw tool output.

SWE-bench derived. We clone repository snapshots from SWE-bench (Jimenez et al., 2024) and execute 14 tool types against them file reads, grep, Git log and blame, test runners, linters, type checkers, pip install, curl, and others — collecting 10,713 raw observations that reflect the kind of output a coding agent encounters during issue resolution.

Synthetic multi-ecosystem. To extend coverage beyond SWE-bench’s Python-heavy distribution, we use openai/gpt-oss-120b to generate 2,039 realistic tool outputs for representative tasks in TypeScript, Go, Rust, Java, Docker, Terraform, Kubernetes, and related build or deployment workflows. We also construct 575 explicit negative examples by pairing mismatched queries and tool outputs, where the correct pruning decision is to return nothing.

The released benchmark contains 11,477 examples in total: 9,205 SWE-derived, 1,697 synthetic positives, and 575 synthetic negatives, covering 27 tool types; Table 2 lists the toolfamily counts.

#### 4.2 Example Construction

Each example pairs one focused query with one raw tool observation. Both sources go through the same two-stage teacher-labeling pipeline using openai/gpt-oss-120b. Given the background task and the raw tool output, the teacher first writes a focused, tool-aware extraction query: not the full issue statement, but a narrower local information need such as finding the failure block, the relevant code region, or the commit entry that matters for the next debugging step. It then selects the smallest contiguous span or set of spans that answers that query.

The teacher sees a numbered rendering of the tool output for stable span selection, but the released labels are always mapped back onto the original raw text, so every target is a verbatim subset of the source. Positive examples whose query cannot be supported by the observation are discarded. For generative training, gold spans are linearized as XML-wrapped extracted text, while the released benchmark stores the span coordinates over the original tool output.

Source Raw inputs Released rows SWE-derived 10,713 9,205 Synthetic positives 2,039 1,697 Synthetic negatives — 575 Total 12,752 11,477

- Table 1: Source composition of the released benchmark. Synthetic negatives are constructed by pairing mismatched queries and tool outputs.

Tool Rows Tool Rows Tool Rows read file 3768 ls 347 build output 168 grep 1330 type check 317 docker build 151 git log 720 git blame 291 make cmake 133 python 698 npm build 230 kubectl 123 test output 546 tsc 229 cargo build 122 curl 493 npm install 227 go build 120 pip install 441 coverage 198 mvn gradle 114 lint output 184 docker logs 198 terraform 103 mypy pyright 95 eslint 78 git diff 53

- Table 2: Tool-family counts in the released benchmark.

#### 4.3 Curation and Statistics

We split SWE-derived examples by repository and synthetic examples by tool family, yielding 10,508 training examples, 240 development examples, and 618 test examples. The held-out set was manually curated: from 729 candidate test examples, 111 (15.2%) were excluded as near-duplicates, trivial 1–2 line outputs, overly broad spans, or incorrect annotations. The final released test set therefore contains only manually reviewed examples.

Table 1 summarizes the source composition of the released benchmark. Table 3 shows the largest tool families. The distribution is intentionally heterogeneous: the benchmark mixes very short observations such as Python exceptions and test summaries with long file reads, type-check outputs, and container logs. This variation is one reason simple truncation and lexical retrieval are weak baselines: relevant evidence may occur at the beginning, middle, or end of the observation, and the useful pruning unit depends on the tool type.

### 5 Model and Evaluation

Our model is Qwen 3.5 2B (Qwen Team, 2026) fine-tuned with LoRA (Hu et al., 2021; Dettmers et al., 2023). The model receives the focused query and raw tool output and is trained to emit the verbatim extracted text wrapped in <relevant lines> tags. We finetune for three epochs with maximum sequence length 20,000, per-device batch size 8, gradient accumulation 4, learning rate 2 × 10−4,

Tool Rows Avg. input Avg. gold read file 3768 1677 84 grep 1330 779 19 git log 720 161 11 python 698 60 28 curl 493 723 68 pip install 441 438 79 type check 317 3418 39 tsc 229 1444 56 remaining tools 3481 914 51

Table 3: Largest tool families in the dataset. Token counts are averages over released examples.

warmup ratio 0.05, and weight decay 0.01. For serving we merge the LoRA adapter into the base model and run the merged checkpoint with vLLM. Training was performed on a single NVIDIA A100 80GB GPU.

We choose Qwen 3.5 2B because the Qwen 3.5 family offers strong performance on code, reasoning, and agent benchmarks while providing small-scale variants suitable for efficient deployment (Qwen Team, 2026). In our setting, the aim is not to maximize zero-shot reasoning depth with the largest possible generator, but to learn a narrow supervised extraction policy that can run cheaply inside existing agent systems. A 2B backbone is therefore a better fit for this use case than a much larger decoder, while still remaining strong enough to benefit from task-specific supervision.

We compare against three zero-shot generative baselines — Qwen 3.5 35B A3B, Kimi K2, and the unfine-tuned Qwen 3.5 2B base

— and four simple heuristics: BM25, First-N, Last-N, and Random. The heuristic baselines keep approximately 10% of the input lines so that they operate at a compression level similar to the gold annotations.

The matching unit is the line. Given a predicted line set P and a gold line set G, we report recall, exact match, compression, and two overlap scores. In the main text, F1 denotes a tolerant line-matching score in which a predicted line matches a gold line if fuzzy substring similarity exceeds 0.5, reducing brittleness from minor formatting differences in generative output. We also report strict exacttext overlap F1 in Table 4 for completeness. Our analysis focuses on recall under strong compression, with F1 as the summary metric. This reflects the task itself: dropping relevant evidence is usually more harmful than keeping a slightly larger block.

In practice, the model is intended as a

Model Prec. Recall Strict F1 Exact F1 Compression

Squeez-2B 0.80 0.86 0.79 0.49 0.80 0.92 Qwen 3.5 35B A3B 0.74 0.75 0.70 0.39 0.73 0.92 Kimi K2 0.61 0.53 0.53 0.30 0.68 0.94 Qwen 3.5 2B (base) 0.42 0.53 0.41 0.19 0.55 0.82

BM25 (10%) 0.13 0.22 0.13 0.01 0.23 0.90 First-N (10%) 0.07 0.14 0.08 0.02 0.16 0.91 Random (10%) 0.07 0.10 0.07 0.01 0.20 0.91 Last-N (10%) 0.05 0.05 0.04 0.01 0.14 0.91

Table 4: Held-out test results. Compression is the fraction of input removed. Squeez-2B is LoRAtuned; other generative models use the same prompt zero-shot.

lightweight pre-processing step for agent systems. In our release it is exposed both as a CLI that can consume piped tool output and as a vLLM-served model, so existing coding agents can insert pruning before the next reasoning step without changing their core planning logic.

### 6 Results

Setup. All models are evaluated on the heldout 618-example test set. All generative models receive the same <query>/<tool output> prompt format as the fine-tuned model. The fine-tuned model is served with vLLM; the larger zero-shot models are accessed via OpenAI-compatible APIs.

Squeez-2B attains the highest recall among all systems while maintaining 92% compression. It outperforms the 18× larger Qwen 3.5 35B A3B by 11 recall points and the unfine-tuned 2B base by 33 points. The gain is not just recall: the fine-tuned model is also the most precise system in the comparison, which indicates that it has learned a tool-specific extraction policy rather than generic instruction following.

The zero-shot baselines differ in how they fail. Qwen 35B is the strongest of them, but still misses benchmark-specific regularities: in repetitive logs or Git history it often selects a semantically adjacent but incorrect block. Kimi K2 is the most aggressive compressor, but pays for that with lower recall, often dropping lines that look boilerplate-like but are actually part of the gold evidence. The unfinetuned 2B base under-compresses and produces noisier extractions. Heuristic baselines perform substantially worse than any generative model. BM25, which is effective for document retrieval, reaches only 0.22 recall on tool output. This is a direct consequence of the task: relevant lines may occur at the beginning, middle, or end of an observation, and usefulness depends on the query rather than lexical overlap alone.

###### Pattern Query / Observation Squeez-2B Baseline error

Precise selection in structured output

git log, 21 lines. Query asks for the commit relevant to the dimension-order change in xr.polyval.

Selects the single correct commit entry.

Qwen 35B chooses a plausible but wrong transpose-related commit; Qwen 2B base overselects several polyval commits.

Failure-block extraction

service log, 176 lines. Query asks for the TLS handshake failure affecting the health-check request.

Returns the correct 5-line health-check failure block.

Qwen 35B selects a later payment-request TLS failure; Kimi K2 keeps only part of the correct block.

Correct empty prediction

docker logs, 316 lines. Query asks for a numpy version conflict that is not present.

Returns empty output. Qwen 35B generates explanatory text (“No relevant lines found. . . ”); Qwen 2B base returns unrelated database errors.

Adjacent overselection

build output, 110 lines. Query asks for the Dockerfile syntax error on line 12.

Finds the Dockerfile error but also includes a nearby Python SyntaxError.

Qwen 35B misses the Dockerfile error entirely and selects only the Python block.

Table 5: Representative held-out examples. The fine-tuned model is strongest on precise hits, compact failure blocks, and true negatives; its residual errors are usually adjacent over-selection.

Figure 2 shows the recall–compression trade-off across all systems.

Recall

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

0.8

0.6

0.4

0.2

0

0.85 0.9 0.95

Compression (fraction of input removed)

Figure 2: Recall–compression trade-off on the heldout test set. • Squeez-2B, ■ Qwen 35B, ▲ Kimi K2, ♦ Qwen 2B, ◦ heuristic baselines.

Qualitatively, the fine-tuned model succeeds for three recurring reasons. It learns to isolate a single relevant hit in structured outputs such as grep and git log; it learns to keep compact failure blocks in noisy test, build, and service logs; and it is much better than larger zero-shot models at returning empty output on explicit negatives. On the 59 negative examples in the held-out test set, it returns empty output 80% of the time, compared with 7% for Qwen 35B. Its strongest remaining failures are usually semantically adjacent selections rather than wholly irrelevant output: the wrong file from an ls listing, a related commit from the same module, or a correct failure block with nearby extra context attached. Table 5 summarizes representative held-out examples. Figure 3 shows the same behavior on a compact kubectl observation, where the relevant evidence is a two-line failure block inside a much longer output.

Query: Find the block showing the OOMKilled reason and exit code for the analytics-worker container.

|23: State: Waiting<br>24: Reason: CrashLoopBackOff<br>25: Last State: Terminated<br><br>26: Reason: OOMKilled<br><br>27: Exit Code: 137<br>28: Started: Mon, 10 Mar 2026 08:12:45 +0000<br><br><br>...<br><br>30: Ready: False<br>31: Restart Count: 3<br>|
|---|

Gold span

|26: Reason: OOMKilled<br>27: Exit Code: 137<br>|
|---|

Figure 3: Compact qualitative example from a kubectl observation. The full output has 250 lines; the gold evidence has two.

### 7 Limitations

The benchmark evaluates pruning quality on single tool observations rather than full agent trajectories. It therefore measures evidence preservation directly, but not the downstream effect on end-to-end task completion. In addition, usefulness is approximated with span overlap, which cannot capture every valid alternative pruning decision. Finally, some tool families remain noisier than others, especially grep and lint output.

### 8 Conclusion

Squeez is a task-conditioned tool-output pruner for coding agents. The benchmark pairs grounded span annotations over 27 tool types with real SWE-bench workflows and synthetic multi-ecosystem observations. A compact LoRA-tuned Qwen 3.5 2B model reaches 0.86 recall while removing 92% of input tokens, outperforming the 18× larger Qwen 3.5 35B A3B and all heuristic baselines by a wide margin. The broader lesson is that mixed-format tool output is not handled well by zero-shot gen-

erative models or retrieval heuristics alone; it responds well to narrow, task-specific supervision.

The model and evaluation code are available on GitHub, and the released model and dataset are available on Hugging Face.

### References

Nadezhda Chirkova, Thibault Formal, Vassilina Nikoulina, and St´ephane Clinchant. 2025. Provence: efficient and robust context pruning for retrieval-augmented generation. Preprint, arXiv:2501.16214.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. Qlora: Efficient finetuning of quantized llms. In Advances in Neural Information Processing Systems, volume 36, pages 10088–10115. Curran Associates, Inc.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. Preprint, arXiv:2106.09685.

Taeho Hwang, Sukmin Cho, Soyeong Jeong, Hoyun Song, SeungYoon Han, and Jong C. Park. 2025. EXIT: Context-aware extractive compression for enhancing retrieval-augmented generation. In Findings of the Association for Computational Linguistics: ACL 2025, pages 4895–4924, Vienna, Austria. Association for Computational Linguistics.

Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023. LLMLingua: Compressing prompts for accelerated inference of large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 13358–13376, Singapore. Association for Computational Linguistics.

Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2024. LongLLMLingua: Accelerating and enhancing LLMs in long context scenarios via prompt compression. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1658–1677, Bangkok, Thailand. Association for Computational Linguistics.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. 2024. SWE-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations.

Imene Kerboua, Sahar Omidi Shayegan, Megh Thakkar, Xing Han L`u, L´eo Boisvert, Massimo Caccia, Je´re´my Espinas, Alexandre Aussem, V´eronique Eglin, and Alexandre Lacoste. 2025. Focusagent: Simple yet effective ways of trimming the large context of web agents.

Adam Kovacs, Paul Schmitt, and Gabor Recski. 2025. KR labs at ArchEHR-QA 2025: A verbatim approach for evidence-based question answering. In Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks), pages 69–74, Vienna, Austria. Association for Computational Linguistics.

Yang Liu and Mirella Lapata. 2019. Text summarization with pretrained encoders. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3730–3740, Hong Kong, China. Association for Computational Linguistics.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents. https://qwen.ai/blog?id= qwen3.5. Blog post.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392, Austin, Texas. Association for Computational Linguistics.

Abigail See, Peter J. Liu, and Christopher D. Manning. 2017. Get to the point: Summarization with pointer-generator networks. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1073–1083, Vancouver, Canada. Association for Computational Linguistics.

James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. FEVER: a large-scale dataset for fact extraction and VERification. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 809–819, New Orleans, Louisiana. Association for Computational Linguistics.

Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, and 5 others. 2025. Openhands: An open platform for ai software developers as generalist agents. Preprint, arXiv:2407.16741.

Yuhang Wang, Yuling Shi, Mo Yang, Rongrui Zhang, Shilin He, Heng Lian, Yuting Chen, Siyu

Ye, Kai Cai, and Xiaodong Gu. 2026. Swe-pruner: Self-adaptive context pruning for coding agents. Preprint, arXiv:2601.16746.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. 2024. Swe-agent: agent-computer interfaces enable automated software engineering. In Proceedings of the 38th International Conference on Neural Information Processing Systems, NIPS ’24, Red Hook, NY, USA. Curran Associates Inc.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380, Brussels, Belgium. Association for Computational Linguistics.

Zilliz. 2025. Semantic highlight bilingual model. https://huggingface.co/zilliz/ semantic-highlight-bilingual-v1. Model card.

