## arXiv:2308.16458v5[cs.LG]20May2024

Bioinformatics

doi.10.1093/bioinformatics/xxxxxx Advance Access Publication Date: Day Month Year

Manuscript Category

### BioCoder: A Benchmark for Bioinformatics Code Generation with Large Language Models

###### Xiangru Tang1, Bill Qian1, Rick Gao1, Jiakang Chen1, Xinyun Chen2, and Mark B. Gerstein1,†

- 1 Yale University, New Haven, CT 06520, USA.
- 2 Google Deepmind, Mountain View, CA 94043, USA. †To whom correspondence should be addressed.

###### Abstract

Pre-trained large language models (LLMs) have significantly improved code generation. As these models scale up, there is an increasing need for the output to handle more intricate tasks and to be appropriately specialized to particular domains. Here, we target bioinformatics due to the amount of domain knowledge, algorithms, and data operations this discipline requires. We present BioCoder, a benchmark developed to evaluate LLMs in generating bioinformatics-specific code. BioCoder spans much of the field, covering cross-file dependencies, class declarations, and global variables. It incorporates 1,026 Python functions and 1,243 Java methods extracted from GitHub, along with 253 examples from the Rosalind Project, all pertaining to bioinformatics. Using topic modeling, we show that the overall coverage of the included code is representative of the full spectrum of bioinformatics calculations. BioCoder incorporates a fuzztesting framework for evaluation. We have applied it to evaluate various models including InCoder, CodeGen, CodeGen2, SantaCoder, StarCoder, StarCoder+, InstructCodeT5+, GPT-3.5, and GPT4. Furthermore, we fine-tuned one model (StarCoder), demonstrating that our training dataset can enhance the performance on our testing benchmark (by >15% in terms of Pass@K under certain prompt configurations and always >3%). The results highlight two key aspects of successful models: (1) Successful models accommodate a long prompt (> 2,600 tokens) with full context, including functional dependencies. (2) They contain domain-specific knowledge of bioinformatics, beyond just general coding capability. This is evident from the performance gain of GPT-3.5/4 compared to the smaller models on our benchmark (50% vs. up to 25%).

Availability and implementation: All datasets, benchmark, Docker images, and scripts required for testing are available at: https://github.com/gersteinlab/biocoder and https://biocoder-benchmark. github.io/.

###### 1 Introduction

Large language models (LLMs) have demonstrated great success in code generation (Chen et al., 2021; Chowdhery et al., 2022; Chen et al., 2023; Barke et al., 2023; Li et al., 2023). The landscape of existing coding benchmarks for LLMs is largely populated with simple functions, often limited to a handful of lines (Chen et al., 2021; Austin et al., 2021; Du et al., 2023; Wong et al., 2023). Combined with a significant lack of closed-domain datasets across diverse fields, this landscape highlights the need for a more robust benchmarking system. Although domainspecific datasets, such as DS1000 (Lai et al., 2022) for data science, have emerged, they fall short of adequately addressing specific tasks in fields like bioinformatics. Open-domain alternatives, including HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021), and APPS (Hendrycks et al., 2021), offer entry-level programming tasks, but their utility is limited as they lack the ability to test more niche, domain-specific code blocks. This shortfall is largely due to a lack of appropriate fine-tuning and context (Muennighoff et al., 2023a). Therefore, a more comprehensive approach to benchmarking is clearly needed.

To address these limitations, we introduce BioCoder (see Figure 1), a benchmark for code generation incorporating 2,269 bioinformatics-specific coding problems. Our BioCoder benchmark mainly targets bioinformatics data analysis, which tasks such as managing various biological data formats, understanding processing workflows, and utilizing APIs of various packages. This domain encapsulates the majority of daily tasks encountered by bioinformaticians in data analysis. However, BioCoder also touches upon aspects of writing bioinformatics software, particularly when tool development intersects with data analysis. Further expanding the scope of BioCoder, we included an additional 253 questions from the Rosalind project. This project specializes in generating Python functions that address key bioinformatics topics such as genetic sequencing and DNA/RNA analysis. BioCoder assures the inclusion of all potential external packages and code that could be utilized by the generated program. This consideration extends to recognizing that real-world functions often necessitate managing multiple external function calls and using global variables; hence, we included all potentially required class declarations in the input. Lastly, we performed ablation studies to determine whether the models are strictly memorizing the solutions rather than being proficient at generating code (see Appendix M).

The key highlights of our work can be outlined as follows: (1) We create a new high-quality dataset for code generation, curated

1

[Figure 1]

2 Tang et al.

###### Code Parsing Tools

###### Processed Data

###### Testing and Docker

###### Models

Tools for parsing large-scale projects and repos, including AST (Abstract Syntax Tree) parsers for code and utilities to extract code attributes.

The Pragmatic dataset encompasses functional code for bioinformatics repos, comprehensive specifications, and a scalable interface. It has undergone rigorous filtering, extensive data cleaning, and preprocessing to prevent models from memorizing.

Code generation with large language models such as SantaCoder, StarCoder, CodeGen, in conjunction with APIs based on OpenAI.

Testing incorporates a Docker environment, an abundance of required dependencies, and a multitude of fuzz test cases. This robust setup not only facilitates testing in realistic project scenarios, but also promotes exceptional scalability and transferability.

Fig. 1: Overview of the contributions of BioCoder.

from 1,720 bioinformatics repositories referenced in peer-reviewed bioinformatics articles. We processed the data, rephrasing more detailed text descriptions, as well as associated comments and specifications, including considerations needed in coding. (2) We provide an extendable parsing tool capable of extracting all pertinent information associated with the target function in expansive projects. (3) We provide a library for code LLMs, similar to Bui et al. (2023), for both training and inference in code generation tasks. (4) We provide a fuzz testing tool capable of scaling to handle substantial datasets. Our benchmark results, derived from 1,000 iterations, indicate the Pass@K rate.

has also seen the release of several open-source code LLMs, such as bilingual GLM-130B (Zeng et al., 2022), CodeGeeX-13B (Zheng et al., 2023), OctoPack (Muennighoff et al., 2023b), WizardCoder (Luo et al., 2023), SantaCoder (Allal et al., 2023), and StarCoder (Li et al., 2023). Salesforce’s CodeGen (Nijkamp et al., 2023a,b), Huawei’s PanguCoder (Christopoulou et al., 2022; Shen et al., 2023), Meta’s LLaMA (Touvron et al., 2023), and CMU’s InCoder model (Fried et al., 2023) also contribute to the field. To integrate code LLMs into real scenarios, researchers have explored methods to incorporate dependencies of relevant code in the prompt (Shrivastava et al., 2023; Zhang et al., 2023a).

###### 2 Related Work

2.2 Code Generation Datasets and Benchmarks

BIOCODER is a code generation benchmark designed for challenging, practical bioinformatics scenarios, offering an extensible testing framework for evaluating the performance of LLMs. We provide a brief overview of the related work in both code generation models and benchmarks.

Early work on code generation benchmarks used lexical exact match, data flow, and abstract syntax tree (AST) methods. However, these measures proved to be unreliable due to their sensitivity to inconsequential differences in the generated code. In response, execution-based evaluation approaches have become more prevalent (Chen et al., 2021; Athiwaratkun et al., 2023; Li et al., 2022; Wang et al., 2022b; Lai et al., 2022; Khlaaf et al., 2022). These approaches execute tests on the generated code to verify its functional correctness, ensuring unbiased evaluations irrespective of implementation method or style variations.

- 2.1 Code Generation with LLMs

LLMs have demonstrated remarkable performances across various domains (Askell et al., 2021; Bai et al., 2022; Biderman et al., 2023; Bommasani

- et al., 2022; Gao et al., 2022; Patil et al., 2023; Xu et al., 2023; Qin
- et al., 2023; Zhang et al., 2023a). Furthermore, LLMs trained with code data have shown promising results in code, exhibiting impressive zeroshot performance on several benchmarks (Zhang et al., 2023b; Olausson et al., 2023; Li et al., 2023; Fried et al., 2023; Wang et al., 2021; Allal et al., 2023). A proven strategy to improve model performance involves increasing both the model parameters and the volume of training data (Radford et al., 2019; Brown et al., 2020; Mitchell et al., 2023), and many large-scale LLMs have been developed to support this endeavor (Chowdhery et al., 2022; Thoppilan et al., 2022; Hoffmann et al.,

As a result, the field of code generation has seen a burgeoning number of execution-based benchmarks (Table 1) (Yuan et al., 2023; Lee et al., 2023; Pan et al., 2023; Wong et al., 2023; Zan et al., 2023), each presenting unique properties in terms of size, language coverage (Orlanski et al., 2023), complexity (Du et al., 2023; Zhuo, 2023), and practical applicability (Yu et al., 2023). For instance, HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021) are frequently used code generation benchmarks that consist of 164 and 974 simple Python functions, respectively, representing a small sample size. These benchmarks also overlook the multi-language coding scenarios gap, which is partially bridged by benchmarks like HumanEval-X (Zheng et al., 2023) and MCoNaLa (Wang et al., 2023b).

- 2022). These models have demonstrated their prowess in code generation (Brown et al., 2020; Chen et al., 2021; OpenAI, 2023), and the field

Data Statistics

Benchmark Num Language

Scenario

Test P.C. P.L. C.C. C.L. HumanEval (2021) 164 Python 7.8 450.6 13.7 180.9 6.8 Code Exercise MBPP (2021) 974 Python 3.1 78.6 1.0 181.1 6.7 Code Exercise APPS (2021) 5, 000 Python 21.0 1743.4 41.6 473.8 21.4 Competitions DS-1000 (2022) 1, 000 Python 1.6 879.1 31.6 137.4 5.0 Data Science HumanEval-X (2023) 164∗ Multi. 7.8 468.4 15.5 264.6 12.1 Multilingual NumpyEval (2022a) 101 Python 3.5 222.9 7.0 29.9 1.1 Public Library TorchDataEval (2022b) 50 Python 1.1 329.0 8.6 50.7 1.3 Private Library BioCoder (public set) 460 Multi. 1000 10465.6 243.5 706.8 26.2 Bioinformatics BioCoder (hidden set) 2,269 Multi. 1000 12296.7 298.8 919.5 26.2 Bioinformatics BioCoder (similar set) 460 Multi. 1000 9885.6 240.8 767.5 26.8 Bioinformatics

- Table 1. Comparison of the statistics of BioCoder to previous benchmarks. Num is the benchmark size. Test refers to the average amount of test cases. P.C. and P.L. indicate the average number of characters and lines in each prompt, respectively, and C.C. and C.L. indicate the average number of characters and lines in the original code solutions. This table is derived from Zan et al. (2023). Please refer to Zan et al.

(2023) for a more comprehensive survey.

AST parser

Write code perturbation

Generated functions

[Figure 2]

[Figure 3]

|Project 1 Project 2<br><br>Project 3 ...<br><br>|
|---|

Indicates manual human interaction

Sort functions by programming language

Execute

[Figure 4]

Extract all functions from Github repositories with an parser

|Standard library imports<br><br>Local imports<br><br>Importing specific members from a module<br><br>Class declarations<br><br>Global variables<br><br>Additional function declarations<br><br>Comment<br><br>Function definition|
|---|

Filter remaining functions on criteria

Create a context file for each problem.

Remove any invalid functions

[Figure 5]

[Figure 6]

[Figure 7]

Determine final functions

1740 bioinformatics Github repositories

[Figure 8]

Projects Sort

[Figure 9]

Domain knowledge check

Specification check

Write test cases for each function

Runnable check

[Figure 10]

[Figure 11]

Environment variables are randomly created by the fuzz tester right before runtime

Add all the required dependencies to the context file

Write code description and specification for each function

Line number check

Dependency check

[Figure 12]

Code Quality check

Human Selection

Data cleaning and data inspection Benchmark collection

Fuzz Test

- Fig. 2: A diagram of the BIOCODER construction process involving custom GitHub repository cleaning, parsing, and function selection, along with context and test case creation and a massively dockerized testing framework.

For a more comprehensive survey on the previous benchmarks of code generation, refer to Zan et al. (2023).

comprehensive comparison between our benchmark and CoderEval can be found in Appendix G.

However, all datasets discussed above share the same shortcoming of only benchmarking generic functions, rather than domain-specific ones. DS-1000 (Lai et al., 2022) represents a more domain-specific dataset, featuring 1,000 data science workflows extracted from Python functions. Li et al. (2023) reported that the performance on HumanEval and MBPP benchmarks does not always align with those on the DS-1000 benchmark. This discrepancy underscores the need for benchmarks that more accurately emulate real-world, domain-specific code generation.

###### 3 The BIOCODER Benchmark

3.1 Initial Dataset Filtering to a Set of 28 Repositories

Our dataset originates from an initial web scrape of 1,743 bioinformaticsrelated GitHub repositories (see Figure 2). Specifically, we utilized the list of 1,743 bioinformatics-adjacent repositories from Russell et al. (2018) as the foundation for BIOCODER. This list contains a curated selection of 1,720 bioinformatics repositories sourced from the literature. The collected repositories include code written in various programming languages such as C, C++, PHP, Python, R, Ruby, SQL, Perl, Java, Matlab, and C#. However, for the scope of this study, we focus on Python and Java, with the intention to expand to other languages in the future. The decision to prioritize Java and Python was based on an empirical investigation into the prevalence of different programming languages across bioinformatics repositories. A more detailed discussion of this language selection process can be found in Appendix P.

In addition, the context supplied greatly influences the performance of existing LLMs (Wang et al., 2022a). While DS-1000 includes eight packages, it fails to fully reflect a typical coding environment. This gap is partially bridged through benchmarks such as CoderEval (Yu et al., 2023), which incorporate some dependencies and function calls. However, these benchmarks are rudimentary in nature and consist primarily of domainagnostic functions. As LLMs continue to evolve, we are now beginning to see repository-level benchmarks that provide a high amount of context, such as RepoBench (Liu et al., 2023). However, these benchmarks remain new and untried.

Our work shares common aspects with CoderEval in its ability to evaluate models beyond the simple generation of standalone functions. Both methodologies employ Docker-based testing to handle the necessity of context-dependent code. However, our approach distinguishes itself from CoderEval by its specific emphasis on bioinformatics. We ensure that each function in our dataset requires a certain level of domain expertise in bioinformatics through a combination of automatic filtering, GPT-assisted filtering, and manual inspection. Furthermore, our dataset surpasses the scale of CoderEval, which consists of 230 functions from 43 Python projects and 230 methods from ten Java projects. In contrast, we source 2,522 functions from over two thousand repositories, providing a more extensive and challenging context for code generation tasks. A

The repositories were then filtered based on popularity, community ratings, and a manual review process. This resulted in a set of 28 highquality, highly domain-specific repositories commonly used in the field of bioinformatics. After determining this set of repositories, we developed custom Python and Java parsers to automatically analyze the selected GitHub repositories. These parsers generated an AST for each code file in the repositories and extracted relevant data, including function content, function signatures, important imports, and cross-file dependencies for each function within the code files. Upon parsing all the repositories, we obtained a large set of over 20,000 Python functions and more than 50,000 Java functions. Given this extensive baseline of functions, we conducted two rounds of automatic filtering, resulting in a final count of 1,026 Python functions and 1,243 Java functions (Table 2).

Public Hidden Similar PY JAVA Overall PY JAVA Overall PY JAVA Overall

Avg. Comment Lines 4.96 2.66 4.40 8.77 4.90 6.65 5.75 3.14 5.12 Avg. Tokens of G.T. 189.25 106.54 169.28 353.67 107.88 219.02 216.62 100.92 188.68 Avg. Lines of G.T. 24.30 11.10 21.11 43.28 12.19 26.25 26.50 10.32 22.59 Avg. Parameters of G.T. 2.39 1.70 2.23 2.92 1.25 2.00 2.48 1.10 2.15 Avg. Classes/Function Decl. 20.25 2.52 15.97 19.45 32.96 26.85 20.20 1.16 15.60 Avg. Global Variables 1.90 - - 2.26 - - 1.87 - Avg. Imports 11.91 1.52 9.40 10.37 5.00 7.43 11.63 1.16 9.10 Avg. Function Calls 7.26 4.56 6.61 14.39 6.47 10.05 9.47 5.92 8.61

- Table 2. Summary statistics for the BioCoder dataset. G.T. stands for the ground truth function. “Public data” represents datasets with test cases. “Hidden data” encompasses a wider array of intricate issues. “Similar data” is a subset of the hidden data, mimicking the distribution of the public data (Appendix T).

- 3.2 Topic Distribution in the Selected Repositories

To gain an understanding of the distribution of bioinformatics within our set of 28 repositories, we applied latent Dirichlet allocation (LDA) to the abstracts of articles citing each repository. Each of these selected repositories contains the codebase associated with a single bioinformatics journal article. We used LDA to infer topics for the abstracts of articles citing each repository in the main dataset. Specifically, from the LDA model, we identified terms that were primarily associated with a single topic. We chose a model with eight topics due to its maximal coherence of concepts within the top topic-specialized terms. Finally, these eight topics were then manually labeled to summarize the top terms, resulting in the following categories: 1) Cancer and epigenetics, 2) Proteomics and microscopy, 3) Variant calling, 4) Genetics and population analysis, 5) Structure and molecular interaction, 6) Web and graphical applications, 7) Assembly and sequence analysis, and 8) Transcription and RNA sequencing. A detailed description of each topic can be found in Appendix N. Our function topic filtering process can be found in Appendix V.

- 3.3 Filtering the Repositories to a List of Core Functions

To further filter and find a small set of functions, we started with a large baseline of functions – i.e. all the functions in the 28 repositories above – and initiated two rounds of automatic filtering to reduce the manual workload. The first round involved keyword filtering, where each function and its comments required at least 10 matches with bioinformatics-related keywords scraped from Wikipedia articles, as mentioned earlier. The methodology for obtaining this Wikipedia-based wordlist can be found in Appendix V. Subsequently, we performed a second round of filtering, during which the OpenAI GPT-3.5 model assessed the bioinformatics relevance of each function. Finally, we manually sorted through the remaining functions, resulting in 1,026 Python functions and 1,243 Java functions (see Table 2). The “similar data” set in Table 2 includes an additional 157 Python functions and 50 Java functions, maintaining the same 253 Rosalind function count, reflecting the composition of the public data. These additional functions were selected to closely align with the same statistics of the public data, such as the distribution of comment lines and token counts.

Our function selection process aimed to strike a balance, ensuring that the final dataset comprises truly bioinformatics-focused functions applicable to our study. This filtering process was undertaken by experts with knowledge in bioinformatics, highlighting the essential role of bioinformatics understanding in this work.

Although our benchmark for code generation is general in nature, it is rooted in the context of bioinformatics, utilizing curated and filtered datasets based on bioinformatics problems (see Appendix N for more details on the topic modeling and statistics regarding the overall topic coverage of the dataset). While an understanding of bioinformatics and biology may not be essential for using the benchmark, it was built to reflect the complexity and domain specifics of bioinformatics.

3.4 Benchmark Construction

BIOCODER-PY and BIOCODER-JAVA .

For each function that passed all rounds of filtering described in Section 3.1, we manually wrote custom code context, including necessary imports, cross-file dependencies, and relevant fuzz test cases (detailed in Section 3.6). We then created custom prompts based on the parsed function data and summaries, ensuring the inclusion of required imports and crossfile dependencies (see Figure 4). As we are testing function-level code generation, imports and classes are predefined and included in the context. We are not prompting the model to generate the classes needed to pass the tests, but rather testing its ability to extract pertinent imports and classes from the context for use in the generated function. Table 7 provides prompt statistics. Finally, we presented the model with a prompt to generate the function, offering the function signature as a starting point. Appendix B and H contain examples of different prompt types. Prompts were partially generated using GPT-3.5, which was used to create function summaries

Model Context limit # Parameters InCoder (Fried et al., 2023) 2048 6B SantaCoder (Allal et al., 2023) 2048 1.1B

StarCoder (Li et al., 2023) 8192 15.5B

StarCoderPlus (Li et al., 2023) 8192 15.5B InstructCodeT5+ (Wang et al., 2023a) 2048 16B CodeGen-6B (Nijkamp et al., 2023a) 2048 6B CodeGen-16B (Nijkamp et al., 2023a) 2048 16B

CodeGen2 (Nijkamp et al., 2023b) 2048 7B*

GPT-3.5-Turbo 8192 Unknown GPT-4 8192 Unknown

Table 3. Context length limits and sizes of different code LLMs.

for all functions in the public dataset. These summaries were incorporated into the prompts to efficiently describe the functions. Appendix E provides more details on this method. Figure 3 shows two examples of the resulting prompt.

BIOCODER-ROSALIND. To compile the Rosalind portion of the benchmark, we began by scraping the problem descriptions from the Rosalind website, identifying problems with available solutions, and gathering all possible solutions. Subsequently, we developed a custom scraper to assemble ten test cases for each Rosalind problem. Using these test cases, we crafted a script to automatically assess whether the available solutions were successfully executed against the collected test cases.

Solutions that successfully executed against all test cases formed the ‘golden code’ section of the Rosalind benchmark, producing correct outputs when run with the test cases. Each Rosalind benchmark context is custommade, incorporating the scraped test cases and injecting them into the generated code. The prompts for the Rosalind problems are constructed using the scraped problem descriptions, supplemented with a brief section outlining the context into which the generated code would be integrated. This rigorous filtering process resulted in 253 functions meeting all our criteria. Selected examples for the Rosalind dataset are shown in Appendix C. Statistics of token counts, comment lines per function, and parameters per function can be found in Appendix A.

- 3.5 Metric

We used the Pass@K metric to measure the functional accuracy (Chen et al., 2021, 2022; Cassano et al., 2023) of code generation models. This metric quantifies, for a certain value K, the probability that the model can solve a particular programming problem when generating K candidate solutions. A problem is deemed “solved” if at least one of the K generated code samples passes all the test cases. E represents the numerical estimation for a particular problem. Each code sample represents a complete function or program intended to solve the problem. The mathematical estimation of Pass@K for a particular problem is articulated as follows:

Pass@K := E

Problems

1 −

n−c k n k

,

where n is the number of samples generated by the model, c is the number of samples that pass all test cases, and K is the number of samples considered for the Pass@K evaluation (Chen et al., 2021).

- 3.6 Testing Framework

Our testing framework begins with a manual review of selected functions, followed by the creation of a context file and golden code for each problem (see Figure 4 for an example), as discussed in 3.4.

Our testing strategy is a hybrid of unit testing and fuzz testing methods, which shares similarities with the metamorphic testing methodology described in Chen et al. (2009). In metamorphic testing, both a reference implementation and the test code are provided with parametrically generated input data to ensure identical behavior. While our approach is not strictly metamorphic testing, it leverages similar principles by using a golden code as a reference and generating random test inputs to compare outputs.

|Here are the imports: import org.apache.commons.math3.distribution.HypergeometricDistribution; import java.text.DecimalFormat; import java.util.*; import java.util.function.Function; import htsjdk.samtools.util.SequenceUtil; import java.util.Objects;|Here are the imports: from collections import defaultdict import re import numpy as np|
|---|---|
| |Here are the global variables: trans_dict = defaultdict(lambda : 5, A=0, C=1, G=2, T=3) trans_dict['-'] = 4|
|Here are the class declarations: class KmerOperations {<br><br>public static long[] rightNeighbours(long kmer, int k) { long mask = (1L << (2 * k)) - 1; long[] ans = new long[] {(kmer << 2) & mask,<br><br>((kmer << 2) & mask) | 1,<br>((kmer << 2) & mask) | 2, ((kmer << 2) & mask) | 3};<br><br><br>for (int i = 0; i < ans.length; i++) { long rc = rc(ans[i], k); if (rc < ans[i]) {<br><br>ans[i] = rc; }<br><br>} return ans;<br><br>} }| |
| |Here are the class declarations: class Sequence(object):<br><br>attributes: self.label,self.sequence,self.length,self.unaligned_length,self.frequency,self.np_seque nce<br><br>methods: def __init__(self, label, sequence):<br><br>summary: Initializes a class instance with the specified label and sequence information.<br><br>param: label (str) - the label of the sequence. param: sequence (str) - the nucleotide sequence. return: None - the function does not return any value.<br><br>def __eq__(self, other): Parameters:<br><br>- self (object) - the first object to be compared<br>- other (object) - the second object to be compared Return:<br>- (bool) - returns True if the objects are equal and False if they are not<br><br><br>equal.|
|summary: Reverses and complement a kmer sequence. param: kmer (long) - the kmer sequence to be reversed and complemented. param: k (long) - the length of the kmer sequence. return: (long) - the reversed and complemented kmer sequence. The function is located in the class KmerOperations| |
| |summary: Returns a string with the sequence in fasta format param: None return: str The FASTA representation of the sequence The function is located in the class Sequence|
| |def to_fasta(self):|
|public static long rc(long kmer, long k)| |

| |
|---|

Area where imports for function is defined

| |
|---|

Area where global variabls for function is defined (Python only)

| |
|---|

Area where external classes for function is defined

Area where summary for function is defined

Area where function signature is defined

- Fig. 3: Sample prompts for code generation. Our prompts follow the same general outline. First, imports are declared at the top of the prompt, then global variables (if any), followed by function declarations, class dependencies, and finally, our actual instructions regarding the function to be generated.

For Python and Java functions, we use a custom syntax in the context file to indicate insertion points for randomly generated test cases, representing four data types: integers, floats, strings, and Boolean values. During runtime, these insertion points are replaced with language-specific code to insert dynamically generated test cases. The tester can be run for any number of iterations, depending on the desired number of fuzz tests.

For Rosalind functions, the process is simpler and more efficient as the functions are less complex. The output of the golden code is generated and cached ahead of time. During testing, the tester executes the generated code within the corresponding context and compares the output with the cached golden code output.

We ran the golden output against itself for every fuzz and Rosalind test case to ensure 100% reliability. To ensure system security and test reliability, we ran our tests in Docker environments using Amazon Web Services, coordinating tasks across multiple nodes to accelerate the process without compromising result validity. After creating a generalized Docker image with all necessary Python requirements, we summarized our testing framework in Appendix K and addressed potential concerns about testing issues due to package changes in Appendix S.

- 4 Models and Results

InstructCodeT5+ (Wang et al., 2023a), and GPT3.5-Turbo and GPT4 (OpenAI, 2023) through Azure OpenAI Service. Full details of the model context lengths and model sizes can be found in Table 3.

To target specific performance characteristics, we came up with hundreds of variations of the prompt. We chose three goals: test the performance of models with extraneous context, without extraneous context, and any context. These goals allow us to better analyze failure reasons and the effectiveness of our context-driven approach. After careful experimentation, we settled on the prompt type shown in Figure 3, which we call Summary at Bottom. Following the instruction paradigm of some considered models, we test a version with the summary moved to the top, along with the text “# Here is an instruction. Complete the function using the required context”. To test without extraneous context, we used human annotators to manually determine the required context and used the structure of the Summary at Top prompt. Further prompt explanations can be found in Appendix H.

Below is an explanation of the prompt types:

- 1. Summary Only These prompts only contain the summary and the function signature, with the uncommented summary coming before the signature. Note that the summary includes nearly complete details about the task; however, it intentionally does not thoroughly explain what the context is. Therefore, this result is best treated as a baseline when compared with other prompt types.
- 2. Uncommented These prompts contain the full parsed context (including the imports, global variables classes, internal class functions, etc.), the summary, and the function signature, in that

To test BIOCODER, we opted to benchmark StarCoder-15B (Li et al.,

- 2023), StarCoder+-15B (Li et al., 2023), InCoder (Fried et al., 2023), SantaCoder (Allal et al., 2023), CodeGen (6B-mono and 16Bmono) (Nijkamp et al., 2023a), CodeGen2-7B (Nijkamp et al., 2023b),

import random import hashlib import numpy import skimage import skimage.measure import scipy.ndimage import os import logging F_ANGLE = 'Angle' LOGGER = logging.getLogger(__name__) class UntangleWorms:

Context file def """Returnangle_features(self):a list of angle feature names"""

try:

return ['_'.join((F_ANGLE, str(n))) for n in range(1, self.

ncontrol_points() - 1)] except:

LOGGER.error(

''Failed to get # of control points from training file. Unknown number of angle measurements''

, exc_info=True)

###### return [] Golden Code

def ncontrol_points(self):

return <|int;range=3,10|> <<insert solution here>>

|Generated Code<br><br>def angle_features(self): # Define angle feature names angle_feature_names = ['Feature1',<br><br>'Feature2', 'Feature3'] return angle_feature_names|
|---|

def main():

print(UntangleWorms().angle_features()) if __name__ == "__main__":

main()

Fig. 4: Test case for UntangleWorms example (an image analysis tool). The context file includes various import dependencies and a class definition with a method placeholder for the solution. The UntangleWorms class comes from a GitHub repository file (https://github.com/CellProfiler/ CellProfiler/blob/master/cellprofiler/modules/untangleworms.py) that was scraped in our study.

|Here are the imports: Import org.apache.commons.math3.distribution.HypergeometricDistribution; import java.text.DecimalFormat; import java.util.*; import java.util.function.Function; import htsjdk.samtools.util.SequenceUtil; import java.util.Objects;|Here are the imports: from collections import defaultdict import re import numpy as np|
|---|---|
| |Here are the global variables: trans_dict = defaultdict(lambda : 5, A=0, C=1, G=2, T=3) trans_dict['-'] = 4|
|Here are the class declarations: class KmerOperations {<br><br>public static long[] rightNeighbours(long kmer, int k) { long mask = (1L << (2 * k)) - 1; long[] ans = new long[] {(kmer << 2) & mask,<br><br>((kmer << 2) & mask) | 1,<br>((kmer << 2) & mask) | 2, ((kmer << 2) & mask) | 3};<br><br><br>for (int i = 0; i < ans.length; i++) { long rc = rc(ans[i], k); if (rc < ans[i]) {<br><br>ans[i] = rc; }<br><br>} return ans;<br><br>} }| |
| |Here are the class declarations: class Sequence(object):<br><br>attributes: self.label,self.sequence,self.length,self.unaligned_length,self.frequency,self.np_seque nce<br><br>methods: def __init__(self, label, sequence):<br><br>summary: Initializes a class instance with the specified label and sequence information.<br><br>param: label (str) - the label of the sequence. param: sequence (str) - the nucleotide sequence. return: None - the function does not return any value.<br><br>def __eq__(self, other): Parameters:<br><br>- self (object) - the first object to be compared<br>- other (object) - the second object to be compared Return:<br>- (bool) - returns True if the objects are equal and False if they are not<br><br><br>equal.|
|summary: Reverses and complement a kmer sequence. param: kmer (long) - the kmer sequence to be reversed and complemented. param: k (long) - the length of the kmer sequence. return: (long) - the reversed and complemented kmer sequence. The function is located in the class KmerOperations| |
| |summary: Returns a string with the sequence in fasta format param: None return: str The FASTA representation of the sequence The function is located in the class Sequence|
| |def to_fasta(self):|
|public static long rc(long kmer, long k)| |

Java Python Pass@1 Pass@5 Pass@10 Pass@20 Pass@1 Pass@5 Pass@10 Pass@20

Model Prompt

Summary at Top 0 0 0 0 0.828 2.016 3.006 4.459 Uncommented 0 0 0 0 0.032 0.159 0.318 0.637 Summary Only 0 0 0 0 1.688 5.320 8.332 12.006 Necessary Only 0 0 0 0 0.032 0.159 0.318 0.637

InCoder-6B

Summary at Top 0 0 0 0 0.637 1.338 1.844 2.548 Uncommented 0 0 0 0 0.287 0.764 0.955 1.274 Summary Only 0 0 0 0 2.965 9.848 14.227 18.181 Necessary Only 0 0 0 0 0.032 0.159 0.318 0.637

SantaCoder-1.1B

Summary at Top 0 0 0 0 3.694 13.197 19.359 24.554 Uncommented 0 0 0 0 0.318 1.062 1.591 2.548 Summary Only 0 0 0 0 4.682 15.225 21.200 27.166 Necessary Only 0 0 0 0 0.127 0.603 1.123 1.911

StarCoder-15.5B

Summary at top 0 0 0 0 5.818 16.562 21.091 27.048 Uncommented 0 0 0 0 3.312 9.073 12.574 17.536 Summary Only 0.200 1.000 2.000 4.000 7.295 20.838 26.143 39.570 Necessary Only 3.300 12.097 19.545 30.000 0.597 1.173 1.813 2.611

StarCoder-15.5B (finetuned)

Summary at Top 0 0 0 0 2.675 9.133 14.019 19.650 Uncommented 0 0 0 0 0.510 0.955 1.274 1.911 Summary Only 1.300 5.031 8.042 12.000 2.548 8.279 12.864 18.057 Necessary Only 0 0 0 0 0.127 0.457 0.609 0.637

StarCoder+

InstructCodeT5+ All prompt types 0 0 0 0 0 0 0 0

Summary at Top 0 0 0 0 0.637 0.637 0.637 0.637 Uncommented 0 0 0 0 0 0 0 0 Summary Only 0 0 0 0 0.637 0.637 0.637 0.637 Necessary Only 0 0 0 0 0 0 0 0

CodeGen-6B-mono

Summary at Top 0 0 0 0 0.637 0.637 0.637 0.637 Uncommented 0 0 0 0 0 0 0 0 Summary Only 0 0 0 0 0.637 0.637 0.637 0.637 Necessary Only 0 0 0 0 0 0 0 0

CodeGen-16B-mono

Summary at Top 0 0 0 0 0.637 0.637 0.637 0.637 Uncommented 0 0 0 0 0.510 0.637 0.637 0.637 Summary Only 0 0 0 0 0.860 2.494 3.962 6.242 Necessary Only 0 0 0 0 0 0 0 0

CodeGen2-7B

Summary at Top 4.100 7.235 8.989 11.600 22.771 33.461 36.551 39.490 Uncommented 6.300 11.563 14.436 18.000 11.019 19.075 21.680 24.204 Summary Only 17.400 33.199 37.878 42.000 24.682 33.997 37.132 40.127 Necessary Only 43.500 52.582 53.995 55.400 28.758 39.529 44.029 47.771

GPT-3.5-Turbo

Summary at top 1.100 5.500 11.000 22.000 10.701 25.500 32.910 39.490 Uncommented 6.367 11.234 15.897 18.562 12.654 20.129 24.387 27.932 Summary Only 19.483 24.721 29.634 2.543 13.172 24.578 28.394 31.938 Necessary Only 45.011 55.350 57.616 60.000 38.439 48.491 50.619 52.229

GPT-4

- Table 4. Zero-shot and fine-tuned performance with five prompt versions of BioCoder. For examples of each prompt version see Appendix H. For all settings, we performed trials twice for Pass@K. Results are expressed in percentages. We only fine-tuned StarCoder for 2,000 steps; all others are zero-shot results. Additional results can be found in Appendix Q (Summary at Bottom results are omitted here).

order. For functions exceeding ten lines in the context, we summarize the parameters, return type, and purpose instead of including the full function code. This step streamlines the number of input tokens and eliminates extraneous data.

- 3. Summary at Bottom These prompts have the same structure as the uncommented ones, but we add the context as a comment. In addition, there are no results for "summary at bottom" for Java due to incompatibility with Java syntax. We were unable to generate this type of prompt for Java in a similar manner to how we generated it for Python.
- 4. Summary at Top These prompts contain the summary, the full (commented) parsed context, and the function signature, in that order. For Java, the summary is not copied at the bottom. This is intended for models with shorter context lengths, as when we truncated the prompt (usually only affecting the context), the summary would still be intact, along with a portion of the context.

5. Necessary Only We use a mixture of our syntax solving algorithm and hand annotation to select precisely which objects within the context are necessary for the function to execute. Note that this is very similar to the environment used for testing the functions.

To accurately represent the performance of the LLM outputs, we implemented basic correction mechanisms to rectify minor syntax and style errors that did not impact functionality. For instance, all StarCoder outputs were appended with a post-script. Each LLM output was then passed through these correction mechanisms before being sent to the testing framework for evaluation (see Tables 4 and 5).

Furthermore, to empirically evaluate the hypothesis regarding the efficacy of smaller, specialized LLMs in closed-domain code generation, as opposed to large open-domain pre-trained models like GPT-3.5 and GPT4, we fine-tuned StarCoder and documented the resulting performance. We chose StarCoder as a representative sample of currently popular models. Due to computing constraints, we were unable to fine-tune all the models,

Examining the other models, it is interesting to note that while InstructCodeT5+, CodeGen, and CodeGen2 are all larger than InCoder and SantaCoder, they perform far worse. This is likely due to the former being trained for single-line completions rather than function completion. Furthermore, InstructCodeT5+, CodeGen, and CodeGen2 have relatively small context limits (Mikolov et al., 2013; MOI et al., 2022), which likely hurts their performance. As for the remaining model, SantaCoder notably performs impressively well for being only a roughly 1B parameter model, which is an indication of aggressive fine-tuning on Python code.

Rosalind Pass@1 Pass@5 Pass@10 Pass@20 InCoder Description 0.020 0.099 0.198 0.395

Model Prompt

SantaCoder Description 0.158 0.658 1.075 1.581

StarCoder Description 0.534 2.042 3.228 4.743 StarCoderPlus Description 0.356 1.313 1.978 2.767

StarCoder (fine-tuned) Description 1.623 3.109 5.328 7.036 InstructCodeT5+ Description 0.059 0.296 0.593 1.186

We also note that the context length limit has a substantial impact on how different models perform on different prompts. Except for GPT-3.5 and GPT-4, models performed the best on the Summary Only prompt style, likely because of its shorter length. Summary-only prompts are shortened prompts utilized across all our LLM models to ensure that context-limited LLMs still receive all the necessary information necessary to potentially generate functions. Within the summary-only prompts, we optimized our prompts to contain only the absolute minimum of necessary information, without including much of the additional context that provides detail regarding the functionality of other dependencies. Looking at Figure 3, which contains the complete prompt structure, summary-only prompts would reduce the class declarations to only their declarations and one sentence describing their output and input. This is especially pronounced for InCoder and SantaCoder, as they both have small context limits of 2,048 tokens. Their Pass@K performance for Python decreases dramatically when switching from short Summary Only prompts to longer Summary at Top/Bottom ones.

CodeGen Description 0.692 2.088 3.055 3.953 CodeGen2 Description 0.059 0.296 0.593 1.186

GPT-3.5 Turbo Description 23.671 31.953 36.702 40.725 GPT-4 Description 24.308 39.551 44.864 50.198

- Table 5. Performance on Rosalind. In this table, we have omitted the percentage symbol (%), although these figures represent the Pass@K in the form of percentages. For all settings, n=20.

but we encourage contributions from the broader community. The results in Table 4 and Table 5 align with our initial hypothesis, which proposed that larger models would likely outperform their smaller counterparts. However, the significant performance gap between GPT-3.5, GPT-4, and all other code-generation models was surprising. This underscores the crucial role of both the dataset size and parameter size of the base models in accomplishing closed-domain code generation prompts. Java performance improved significantly, as the structure is similar between the training set and testing set. Interestingly, despite the rudimentary nature of our fine-tuning on StarCoder, the results still highlighted a significant improvement compared with the non-fine-tuned model. This stark contrast in performance bolsters our original assertion: achieving success in closeddomain tasks can be realized either through large open-domain LLMs or via fine-tuning smaller models. These smaller models could potentially achieve comparable performance but with significantly reduced computational and memory requirements. Furthermore, Table 4 demonstrates that the performance of models improves with the inclusion of dependencies in prompts. Without additional training, ChatGPT models performed notably better than other models. Their performance underscores the crucial role of both the dataset scale and model size. That said, the performance of other models (e.g. StarCoder) could be improved through fine-tuning.

- 5 Analysis and Discussion

Looking more closely at the results in Table 4, it is clear that the larger models with more parameters generally perform better than the smaller models. The GPT-4 model dwarfs the other models in this study in both size and performance. However, it is clear that BIOCODER remains a challenge as GPT-3.5 and GPT-4, the best models, only achieved an accuracy of slightly under 60%.

Failure/Success Count Percent (%) Mismatched output 8661 4.567

Invalid syntax 117665 62.038 Runtime error 55351 29.184

Time out 4 0.002 Successfully Passed 7982 4.209

Total Testcases 189663 100

- Table 6. Aggregated error distribution across all models. Prompt Mean Median STDev

Java 2278.82 2599.00 1331.81 Python 2790.75 2194.00 2539.79 Rosalind 564.49 509.00 286.47 Overall 1510.66 812.50 1882.80

- Table 7. Prompt token distribution.

The scatterplots in Appendix J reveal an inverse relationship between the number of tokens in the prompt and the Pass@K score for models with an average Pass@K score of at least 2%. Furthermore, for models such as SantaCoder and GPT, the performance sharply declines after around 500 tokens. This could be due to the massive amount of context “confusing” the models. However, model performance cannot only be attributed to prompt length. We can see that even though the Necessary Only prompts are relatively shorter when compared to the Summary at Top or Uncommented prompts, the Pass@k performance of the "Uncommented" prompts is worse for many of the models. For further analysis of this and prompt structure in general, please refer to Appendix U.

Focusing on Java’s performance, it is clear that most of the publicly available LLMs have not been fine-tuned for Java, resulting in near 0% Pass@K values. Finally, the performance results for Rosalind in Table 5 are roughly in line with Python’s performance in Table 4.

Table 6 provides an overview of the error statistics collected from our test runs. The errors include: ‘different output’, where the generated code’s output did not match the golden code’s output; ‘invalid syntax’, where syntax errors in the generated code prevented code execution; ‘function timed out’, where code execution exceeded time limits; and ‘runtime error’, where the generated code compiled successfully but failed to run. The vast majority of the generated code samples tested encountered a syntax or runtime error without resulting in an output. See Table 8 for more detail. Additional error statistics per model can be found in Appendix O. Looking at Appendix O, it appears that the models struggle the most with writing code that will successfully compile or run. For the code that did produce outputs, however, GPT-based models produced more correct samples than incorrect ones (differing output), while other models generated more incorrect but syntactically valid code. Therefore, it seems that the betterperforming models have the most trouble generating syntactically correct code rather than understanding the logic required to complete the problems outlined in the prompts. Further discussion on the results of each model can be found in Appendix I. Despite these challenges, we believe that this dataset holds importance for benchmarking future models, especially ones with larger context limits, such as GPT-4-32k and Claude2.

###### 6 Limitations

Our study has several limitations that warrant discussion. First, the use of closed-source LLMs, may introduce a degree of circularity in our methodology. The lack of transparency regarding the training data and

Model Prompt Pass Differing Runtime Err. Syntax Err. Other Model Prompt Pass Differing Runtime Err. Syntax Err. Other

Summary at top 756 664 1509 1207 4

Summary at top 358 119 2321 1342 0 Uncommented 409 254 435 3040 2 Uncommented 15 9 2796 1320 0 Summary Only 949 863 1528 799 1 Summary Only 411 356 1765 1608 0

GPT-3

GPT-4

Summary at bottom 422 319 854 1545 0 Summary at bottom 25 2 1174 1939 0 Necessary Only 1338 1103 1275 420 4 Necessary Only 1660 1094 996 388 2

Summary at top 116 160 1473 2390 1

Summary at top 20 0 2326 1794 0 Uncommented 10 18 484 3628 0 Uncommented 16 1 508 3612 3 Summary Only 147 258 1885 1849 1 Summary Only 27 136 1589 2386 2

StarCoder

Codegen2-7B

Summary at bottom 203 395 857 1682 3 Summary at bottom 16 69 2182 873 0 Necessary Only 4 9 85 4042 0 Necessary Only 0 5 113 4022 0

- Table 8. Failure modes for selected (best-performing) models, corresponding to the results shown in Table 4. "Pass" means the generated and golden code exhibited the same functionality. "Differing" indicates code that compiled and ran, but gave different outputs with respect to the golden code when executed. "Runtime Err." indicates that the code compiled but unexpectedly crashed during execution. "Syntax Err" indicates the code did not compile. "Other" includes timeouts resulting from mistakes such as infinite loops. Note that we intentionally set the execution limits to be generous, so if a generated code sample failed given our constraints, then the sample would not be used in practice anyway.

model architecture of these LLMs makes it challenging to determine whether the test cases used in our benchmark are truly independent of the models’ knowledge base. Despite our obfuscation step (including the ablation study in Appendix M), this circularity could potentially lead to inflated performance metrics and hinder the generalizability of our findings. We partially mitigated this issue by including open-source models in the evaluation. Future research should prioritize the use of open-source LLMs or collaborate with LLM providers to ensure a clear separation between training and testing data.

Moreover, the closed-source nature of some LLMs used in our study raises concerns about reproducibility and fairness in performance comparisons. The constant evolution and updates to these models, often without detailed release notes, may make it difficult to replicate our findings or conduct longitudinal studies. To address this issue, we encourage the development and adoption of standardized benchmarking protocols and the use of versioned, open-source models whenever possible. This includes the usage of mechanisms such as the OpenAI’s “system fingerprint” and “seed” parameters.

Lastly, by using OpenAI GPT to assist in identifying bioinformaticsrelated code samples, we may have favored examples that align with its preexisting knowledge or biases. This bias could result in a dataset that is more easily solved by OpenAI GPT-like models, potentially skewing the performance evaluation. Future work should explore alternative approaches to dataset curation that minimize reliance on the same type of models being evaluated.

Despite these limitations, our study serves as a valuable starting point for evaluating the performance of LLMs in bioinformatics code generation. By acknowledging and addressing the identified challenges, future research can build upon our work to develop more robust, transparent, and comprehensive benchmarking frameworks. Such efforts will be essential in advancing the responsible and effective application of LLMs in bioinformatics research and practice.

###### 7 Conclusions and Future Work

Our study underscores the challenges in code generation, emphasizing the shortcomings of current models in the face of complex tasks. We present highly challenging natural language to code tasks, providing input rich with dependencies and imports. Existing models struggle to comprehend the application of these imported toolkits or functions contained in other files. Our tasks are marked by extensive input and a high level of specialization. These programs are closer to real-world scenarios, requiring professionallevel code-writing skills, rather than merely catering to beginners. This suggests that the code in question can typically be produced only by professional programmers.

We welcome contributions to our benchmark. Our data are stored in the JSON format, with the fields documented in our GitHub repository. To ensure compatibility and ease of integration, we recommend that contributors follow the existing data structure and provide the necessary

information for each new test case, such as the problem description, input/output examples, and any additional context or dependencies.

As a novel benchmark within the field of bioinformatics, there remain a multitude of areas for future exploration. While we have covered most existing models and included a few well-established repositories, future work could expand the dataset to include more niche sub-studies and programming languages. Additionally, our benchmark may not exhaustively cover all relevant domains and emerging techniques. As bioinformatics evolves, it is crucial to update and expand the benchmark to reflect the latest challenges and methodologies, moving beyond functionlevel code generation to include more complex, multi-step workflows that require planning and better simulate real-world use cases. An example of such a task for realistic assessment could involve initially interpreting a count matrix and subsequently identifying different cell types present based on the count data. These types of tasks require the model to execute multiple routines in a sequence and to adapt based on prior outcomes.

###### 8 Acknowledgment

Mark Gerstein and Xiangru Tang acknowledge support from Schmidt Futures.

###### References

Allal, L. B., Li, R., Kocetkov, D. et al. (2023). Santacoder: don’t reach for the stars! arXiv preprint arXiv:2301.03988. Askell, A., Bai, Y., Chen, A. et al. (2021). A general language assistant as a laboratory for alignment. arXiv preprint arXiv:2112.00861. Athiwaratkun, B., Gouda, S. K., Wang, Z. et al. (2023). Multi-lingual evaluation of code generation models. Austin, J., Odena, A., Nye, M. et al. (2021). Program synthesis with large language models.

Bai, Y., Jones, A., Ndousse, K. et al. (2022). Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862.

Barke, S., James, M. B. and Polikarpova, N. (2023). Grounded copilot: How programmers interact with code-generating models. Proceedings of the ACM on Programming Languages, 7(OOPSLA1), 85–111.

Biderman, S., Schoelkopf, H., Anthony, Q. et al. (2023). Pythia: A suite for analyzing large language models across training and scaling. Bommasani, R., Hudson, D. A., Adeli, E. et al. (2022). On the opportunities and risks of foundation models. Brown, T. B., Mann, B., Ryder, N. et al. (2020). Language models are few-shot learners. arXiv preprint arXiv:2005.14165. Bui, N. D., Le, H., Wang, Y. et al. (2023). Codetf: One-stop transformer library for state-of-the-art code llm. arXiv preprint arXiv:2306.00029.

Cassano, F., Gouwar, J., Nguyen, D. et al. (2023). MultiPL-E: a scalable and polyglot approach to benchmarking neural code generation. IEEE Transactions on Software Engineering, pages 1–17.

Chen, B., Zhang, F., Nguyen, A. et al. (2022). Codet: Code generation with generated tests. arXiv preprint arXiv:2207.10397. Chen, M., Tworek, J., Jun, H. et al. (2021). Evaluating large language models trained on code.

Chen, T. Y., Ho, J. W., Liu, H. et al. (2009). An innovative approach for testing bioinformatics programs using metamorphic testing. BMC Bioinformatics, 10(1), 24.

Chen, X., Lin, M., SchÃd’rli, N. et al. (2023). Teaching large language models to self-debug. Chowdhery, A., Narang, S., Devlin, J. et al. (2022). Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311.

Christopoulou, F., Lampouras, G., Gritta, M. et al. (2022). PanGu-Coder: program synthesis with function-level language modeling. arXiv preprint arXiv:2207.11280.

Du, X., Liu, M., Wang, K. et al. (2023). Classeval: A manually-crafted benchmark for evaluating llms on class-level code generation. arXiv preprint arXiv:2308.01861.

Fried, D., Aghajanyan, A., Lin, J. et al. (2023). Incoder: A generative model for code infilling and synthesis. Gao, L., Tow, J., Biderman, S. et al. (2022). Eleutherai/lm-evaluationharness: v0.3.0. Hendrycks, D., Basart, S., Kadavath, S. et al. (2021). Measuring coding challenge competence with apps. NeurIPS. Hoffmann, J., Borgeaud, S., Mensch, A. et al. (2022). Training computeoptimal large language models.

Khlaaf, H., Mishkin, P., Achiam, J. et al. (2022). A hazard analysis framework for code synthesis large language models. arXiv preprint arXiv:2207.14157.

Lai, Y., Li, C., Wang, Y. et al. (2022). DS-1000: a natural and reliable benchmark for data science code generation. ArXiv, abs/2211.11501. Lee, T., Hong, S., Ahn, J. et al. (2023). Who wrote this code? watermarking for code generation. arXiv preprint arXiv:2305.15060. Li, R., Allal, L. B., Zi, Y. et al. (2023). Starcoder: may the source be with

you! Li, Y., Choi, D., Chung, J. et al. (2022). Competition-level code generation with alphacode. arXiv preprint arXiv:2203.07814. Liu, T., Xu, C. and McAuley, J. (2023). Repobench: Benchmarking repository-level code auto-completion systems.

Luo, Z., Xu, C., Zhao, P. et al. (2023). Wizardcoder: Empowering code large language models with evol-instruct. arXiv preprint arXiv:2306.08568.

Mikolov, T., Chen, K., Corrado, G. et al. (2013). Efficient estimation of word representations in vector space.

Mitchell, M., Luccioni, A. S., Lambert, N. et al. (2023). Measuring data. MOI, A., Patry, N., Cistac, P. et al. (2022). huggingface/tokenizers: Rust

0.13.2. Muennighoff, N., Wang, T., Sutawika, L. et al. (2023a). Crosslingual generalization through multitask finetuning.

Muennighoff, N., Liu, Q., Zebaze, A. et al. (2023b). Octopack: Instruction tuning code large language models. arXiv preprint arXiv:2308.07124. Nijkamp, E., Pang, B., Hayashi, H. et al. (2023a). CodeGen: an open large language model for code with multi-turn program synthesis. In The Eleventh International Conference on Learning Representations. Nijkamp, E., Hayashi, H., Xiong, C. et al. (2023b). Codegen2: Lessons

for training llms on programming and natural languages. arXiv preprint. Olausson, T. X., Inala, J. P., Wang, C. et al. (2023). Demystifying gpt

self-repair for code generation. arXiv preprint arXiv:2306.09896. OpenAI (2023). Gpt-4 technical report. Orlanski, G., Xiao, K., Garcia, X. et al. (2023). Measuring the impact

of programming language distribution. In International Conference on Machine Learning, pages 26619–26645. PMLR.

Pan, R., Ibrahimzada, A. R., Krishna, R. et al. (2023). Understanding the effectiveness of large language models in code translation. arXiv preprint arXiv:2308.03109.

Patil, S. G., Zhang, T., Wang, X. et al. (2023). Gorilla: Large language model connected with massive apis. arXiv preprint arXiv:2305.15334.

Qin, Y., Liang, S., Ye, Y. et al. (2023). Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789.

Radford, A., Wu, J., Child, R. et al. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.

Russell, P. H., Johnson, R. L., Ananthan, S. et al. (2018). A large-scale analysis of bioinformatics code on github. PloS one, 13(10), e0205898.

Shen, B., Zhang, J., Chen, T. et al. (2023). Pangu-coder2: Boosting large language models for code with ranking feedback. arXiv preprint arXiv:2307.14936.

Shrivastava, D., Kocetkov, D., de Vries, H. et al. (2023). Repofusion: Training code models to understand your repository. Thoppilan, R., Freitas, D. D., Hall, J. et al. (2022). Lamda: Language models for dialog applications. Touvron, H., Lavril, T., Izacard, G. et al. (2023). LLaMA: open and

efficient foundation language models. arXiv preprint arXiv:2302.13971. Wang, S., Li, Z., Qian, H. et al. (2022a). Recode: Robustness evaluation

of code generation models. arXiv preprint arXiv:2212.10264.

- Wang, Y., Wang, W., Joty, S. et al. (2021). Codet5: Identifier-aware unified pre-trained encoder-decoder models for code understanding and generation.

- Wang, Y., Le, H., Gotmare, A. D. et al. (2023a). Codet5+: Open code large language models for code understanding and generation. arXiv preprint arXiv:2305.07922.
- Wang, Z., Zhou, S., Fried, D. et al. (2022b). Execution-based evaluation for open-domain code generation. arXiv preprint arXiv:2212.10481.

- Wang, Z., Cuenca, G., Zhou, S. et al. (2023b). Mconala: A benchmark for code generation from multiple natural languages.

Wong, M.-F., Guo, S., Hang, C.-N. et al. (2023). Natural language generation and understanding of big code for ai-assisted programming: A review. Entropy, 25(6), 888.

Xu, Q., Hong, F., Li, B. et al. (2023). On the tool manipulation capability of open-source large language models. arXiv preprint arXiv:2305.16504.

Yu, H., Shen, B., Ran, D. et al. (2023). Codereval: A benchmark of pragmatic code generation with generative pre-trained models. arXiv preprint.

Yuan, Z., Liu, J., Zi, Q. et al. (2023). Evaluating instruction-tuned large language models on code comprehension and generation. arXiv preprint arXiv:2308.01240.

Zan, D., Chen, B., Yang, D. et al. (2022a). Cert: Continual pre-training on sketches for library-oriented code generation.

Zan, D., Chen, B., Lin, Z. et al. (2022b). When language model meets private library. In Conference on Empirical Methods in Natural Language Processing.

Zan, D., Chen, B., Zhang, F. et al. (2023). Large language models meet nl2code: A survey. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7443–7464.

Zeng, A., Liu, X., Du, Z. et al. (2022). GLM-130B: an open bilingual

pre-trained model. arXiv preprint arXiv:2210.02414. Zhang, F., Chen, B., Zhang, Y. et al. (2023a). Zhang, S. D., Tigges, C., Biderman, S. et al. (2023b). Can transformers

learn to solve problems recursively? arXiv preprint arXiv:2305.14699. Zheng, Q., Xia, X., Zou, X. et al. (2023). Codegeex: A pre-trained model

for code generation with multilingual evaluations on humaneval-x. Zhuo, T. Y. (2023). Large language models are state-of-the-art evaluators of code generation. arXiv preprint arXiv:2304.14317.

# arXiv:2308.16458v5[cs.LG]20May2024

A FUNCTION STATISTICS

Below are our distributions of comment lines, token lengths, and the number of parameters per function across our entire dataset. Generally, our number of parameters and comments are leftskewed. Our prompt lengths, while left-skewed, had a much more even distribution.

Prompt Comments

| |java prompts python prompts<br><br>| |
|---|
|
|---|---|
| | |
| | |
| | |
| | |

400

300

Count

200

100

0

0 10 20 30 40 50 60 70 num lines

Figure 1: Comment lines per function distribution

Prompt Token Lengths

java prompts python prompts

| |
|---|

200

150

Count

100

50

0

0 500 1000 1500 2000 2500 3000 3500 Token Count

Figure 2: Prompt token length distribution

Prompt Parameters

java prompts python prompts

| |
|---|

500

400

Count

300

200

100

0

0 5 10 15 20 25 30 num params

Figure 3: Parameters per function distribution

Below is the distribution of the number of output tokens, averaged across either Python or Java. The vast concentration of the generations is concentrated at 0-250, as those are the minimum reserved tokens for generations if the prompts are too large. Across each language, we utilized nine models to generate outputs.

Figure 4: The distribution of the number of output tokens of the models.

LLM Outputs Tokens Length

Python Java

| |
|---|

20000

15000

Count

10000

5000

0

0 250 500 750 1000 1250 1500 1750 2000 Tokens

B PYTHON EXAMPLES

B.1 PYTHON EXAMPLE 1 B.1.1 FULL PROMPT

- 1 #This is in python

- 2 #Write a function called "unpipe_name" that takes in a string parameter called "name". The function takes a string containing multiple gene names separated by pipes , and returns a single gene name devoid of duplicates and pipe characters. The function checks if there are any duplicates , and removes any meaningless target names. If there are multiple gene names present , the function takes the longest name as the final name. If there are any ambiguous gene names , the function logs a warning and selects the longest name as the final name. The function should also import "cnvlib.params".

- 3 #

- 4 #def unpipe_name(name):

- 5 #

- 6 #Here are the imports:

- 7 #import numpy as np

- 8 #import logging

- 9 #from . import params

- 10 #from skgenome import tabio

- 11 #Here are the global variables:

- 12 #MIN_REF_COVERAGE = -5.0

- 13 #MAX_REF_SPREAD = 1.0

- 14 #NULL_LOG2_COVERAGE = -20.0

- 15 #GC_MIN_FRACTION = 0.3

- 16 #GC_MAX_FRACTION = 0.7

- 17 #INSERT_SIZE = 250

- 18 #IGNORE_GENE_NAMES = ’-’, ’.’, ’CGH ’

- 19 #ANTITARGET_NAME = ’Antitarget ’

- 20 #ANTITARGET_ALIASES = ANTITARGET_NAME , ’Background ’

- 21 #Here are the class declarations:

- 22 #Here are the additional function declarations:

- 23 #def do_import_picard(fname , too_many_no_coverage):

- 24 # summary: Reads a file in ’picardhs ’ format , processes the data , and returns a modified dataframe.

- 25 # param: fname (string) - the file name/path to be read in ’picardhs ’ format.

- 26 # param: too_many_no_coverage (int) - if the number of bins with no coverage is greater than this value , a warning message is logged. Default is 100.

- 27 # return: garr (pandas dataframe) - a modified dataframe with added columns ’gene ’ and ’log2 ’ based on the original dataframe read from the input file.

- 28 #def unpipe_name(name):

- 29 # summary: Remove duplicate gene names and pipe characters from a given string.

- 30 # param: name (str) - the string containing the gene names.

- 31 # return: new_name (str) - the cleaned gene name string.

- 32 #def do_import_theta(segarr , theta_results_fname , ploidy):

- 33 # summary: A function for importing theta results and estimating copy number and log2 ratios of segments.

- 34 # param: segarr (numpy array) - array of segments

- 35 # param: theta_results_fname (str) - name of theta results file

- 36 # param: ploidy (int) - ploidy of genome (default is 2)

- 37 # return: generator of numpy arrays - array of segments with estimated copy number and log2 ratios.

- 38 #def parse_theta_results(fname):

- 39 # summary: Parses THetA results into a data structure with NLL , mu, C, and p* columns.

- 40 # param: fname (str) - name of the file to parse the results from

- 41 # return: (dict) - a dictionary containing the NLL , mu_normal , mu_tumors , C, and p* values

- 42 #Here are the comments and the specs:

- 43 #Write a function called "unpipe_name" that takes in a string parameter called "name". The function takes a string containing multiple gene names separated by pipes , and returns a single gene name devoid of duplicates and pipe characters. The function checks if there are any duplicates , and removes any meaningless target names. If there are multiple gene names present , the function takes the longest name as the final name. If there are any ambiguous gene names , the function logs a warning and selects the longest name as the final name. The function should also import "cnvlib.params".

- 44 #def unpipe_name(name):

B.1.2 SMALL PROMPT

- 1 Write a function with the following specs:

- 2 --specs begin here --

- 3 #Write a function called "unpipe_name" that takes in a string parameter called "name". The function takes a string containing multiple gene names separated by pipes , and returns a single gene name devoid of duplicates and pipe characters. The function checks if there are any duplicates , and removes any meaningless target names. If there are multiple gene names present , the function takes the longest name as the final name. If there are any ambiguous gene names , the function logs a warning and selects the longest name as the final name. The function should also import "cnvlib.params".

- 4 param: name (str) - the string containing the gene names.

- 5 return: new_name (str) - the cleaned gene name string.

- 6 --specs end here --

- 7 Note the function will be embedded in the following context

- 8 --context begins here --

- 9 import random

- 10 import hashlib

- 11 import numpy as np

- 12 import skimage

- 13 import skimage.measure

- 14 import scipy.ndimage

- 15 import os

- 16 import logging

- 17 from functools import wraps

- 18 from scipy import stats

- 19 import sys

- 20 import math

- 21 IGNORE_GENE_NAMES = ("-", ".", "CGH")

- 22 <<insert solution here >>

- 23 def main():

- 24 string1 = <|string|>

- 25 string2 = ’CGH’

- 26 name=f’{string1}|{string2}’

- 27 print(unpipe_name(name))

- 28 // context continues

- 29 --context ends here --

- 30 Make sure to only generate the function and not any of the context. Make sure you are generating valid , runnable code. Begin your solution with:

- 31 def unpipe_name(name):

- 32 MAKE SURE TO INDENT THE BODY OF YOUR FUNCTION BY A TAB

B.1.3 NO COMMENT PROMPT

- 1 This is in python

- 2 Write a function called "unpipe_name" that takes in a string parameter called "name". The function takes a string containing multiple gene names separated by pipes , and returns a single gene name devoid of duplicates and pipe characters. The function checks if there are any duplicates , and removes any meaningless target names. If there are multiple gene names present , the function takes the longest name as the final name. If there are any ambiguous gene names , the function logs a warning and selects the longest name as the final name. The function should also import "cnvlib.params".

- 3

- 4 def unpipe_name(name):

- 5

- 6 Here are the imports:

- 7 import numpy as np

- 8 import logging

- 9 from . import params

- 10 from skgenome import tabio

- 11 Here are the global variables:

- 12 MIN_REF_COVERAGE = -5.0

- 13 MAX_REF_SPREAD = 1.0

- 14 NULL_LOG2_COVERAGE = -20.0

- 15 GC_MIN_FRACTION = 0.3

- 16 GC_MAX_FRACTION = 0.7

- 17 INSERT_SIZE = 250

- 18 IGNORE_GENE_NAMES = ’-’, ’.’, ’CGH ’

- 19 ANTITARGET_NAME = ’Antitarget ’

- 20 ANTITARGET_ALIASES = ANTITARGET_NAME , ’Background ’

- 21 Here are the class declarations:

- 22 Here are the additional function declarations:

- 23 def do_import_picard(fname , too_many_no_coverage):

- 24 summary: Reads a file in ’picardhs ’ format , processes the data , and returns a modified dataframe.

- 25 param: fname (string) - the file name/path to be read in ’picardhs ’ format.

- 26 param: too_many_no_coverage (int) - if the number of bins with no coverage is greater than this value , a warning message is logged. Default is 100.

- 27 return: garr (pandas dataframe) - a modified dataframe with added columns ’gene ’ and ’log2 ’ based on the original dataframe read from the input file.

- 28 def unpipe_name(name):

- 29 summary: Remove duplicate gene names and pipe characters from a given string.

- 30 param: name (str) - the string containing the gene names.

- 31 return: new_name (str) - the cleaned gene name string.

- 32 def do_import_theta(segarr , theta_results_fname , ploidy):

- 33 summary: A function for importing theta results and estimating copy number and log2 ratios of segments.

- 34 param: segarr (numpy array) - array of segments

- 35 param: theta_results_fname (str) - name of theta results file

- 36 param: ploidy (int) - ploidy of genome (default is 2)

- 37 return: generator of numpy arrays - array of segments with estimated copy number and log2 ratios.

- 38 def parse_theta_results(fname):

- 39 summary: Parses THetA results into a data structure with NLL , mu, C, and p* columns.

- 40 param: fname (str) - name of the file to parse the results from

- 41 return: (dict) - a dictionary containing the NLL , mu_normal , mu_tumors , C, and p* values

- 42 Here are the comments and the specs:

- 43 Write a function called "unpipe_name" that takes in a string parameter called "name". The function takes a string containing multiple gene names separated by pipes , and returns a single gene name devoid of duplicates and pipe characters. The function checks if there are any duplicates , and removes any meaningless target names. If there are multiple gene names present , the function takes the longest name as the final name. If there are any ambiguous gene names , the function logs a warning and selects the longest name as the final name. The function should also import "cnvlib.params".

- 44 def unpipe_name(name):

B.1.4 REFERENCE CODE

- 1 def unpipe_name(name):

- 2 """Fix the duplicated gene names Picard spits out.

- 3

- 4 Return a string containing the single gene name , sans duplications and pipe

- 5 characters.

- 6

- 7 Picard CalculateHsMetrics combines the labels of overlapping intervals

- 8 by joining all labels with ’|’, e.g. ’BRAF|BRAF ’ -- no two distinct

- 9 targeted genes actually overlap , though , so these dupes are redundant

.

- 10 Meaningless target names are dropped , e.g. ’CGH|FOO|-’ resolves as ’ FOO ’.

- 11 In case of ambiguity , the longest name is taken , e.g. "TERT|TERT Promoter"

- 12 resolves as "TERT Promoter".

- 13 """

- 14 if ’|’ not in name:

- 15 return name

- 16 gene_names = set(name.split(’|’))

- 17 if len(gene_names) == 1:

- 18 return gene_names.pop()

- 19 cleaned_names = gene_names.difference(IGNORE_GENE_NAMES)

- 20 if cleaned_names:

- 21 gene_names = cleaned_names

- 22 new_name = sorted(gene_names , key=len , reverse=True)[0]

- 23 if len(gene_names) > 1:

- 24 logging.warning(’WARNING: Ambiguous gene name %r; using %r’, name ,

- 25 new_name)

- 26 return new_name

B.1.5 ANNOTATION PROMPTS

- 1 #This is in python

- 2 #Write a function called "unpipe_name" that takes in a string parameter called "name". The function takes a string containing multiple gene names separated by pipes , and returns a single gene name devoid of duplicates and pipe characters. The function checks if there are any duplicates , and removes any meaningless target names. If there are multiple gene names present , the function takes the longest name as the final name. If there are any ambiguous gene names , the function logs a warning and selects the longest name as the final name. The function should also import "cnvlib.params".

- 3 #

- 4 #def unpipe_name(name):

- B.1.6 CONTEXT

- 1 import random

- 2 import hashlib

- 3 import numpy as np

- 4 import skimage

- 5 import skimage.measure

- 6 import scipy.ndimage

- 7 import os

- 8 import logging

- 9 from functools import wraps

- 10 from scipy import stats

- 11 import sys

- 12 import math

- 13 IGNORE_GENE_NAMES = ("-", ".", "CGH")

- 14 <<insert solution here >>

- 15 def main():

- 16 string1 = <|string|>

- 17 string2 = ’CGH’

- 18 name=f’{string1}|{string2}’

- 19 print(unpipe_name(name))

- 20 if __name__ == "__main__":

- 21 main()

B.2 PYTHON EXAMPLE 2

- B.2.1 FULL PROMPT

- 1 #This is in python

- 2 #write a function called "UnifyLevels" that takes three parameters: baseLevel (a constant array of integers representing the base levels) , addonLevel (a constant array of integers representing the additional levels), and windowSize (an integer representing the window size). The function should merge the two lists of breakpoints ,

but drop addonLevel values that are within windowSize of baseLevel values. The function should return an array of integers called joinedLevel. If addonLevel is an empty array , the function should return baseLevel. The output should be sorted and of dtype=np.int_.

- 3 #

- 4 #def UnifyLevels(baseLevel , addonLevel , windowSize):

- 5 #

- 6 #Here are the imports:

- 7 #from scipy import stats

- 8 #import math

- 9 #import numpy as np

- 10 #import logging

- 11 #import pandas as pd

- 12 #Here are the global variables:

- 13 #Here are the class declarations:

- 14 #Here are the additional function declarations:

- 15 #def segment_haar(cnarr , fdr_q):

- 16 # summary: Segment CNVkit data using HaarSeg algorithm

- 17 # param: cnarr (CopyNumArray) - binned , normalized copy ratios

- 18 # param: fdr_q (float) - false discovery rate q-value

- 19 # return: segarr (CopyNumArray) - CBS data table as a CNVkit object

- 20 #def one_chrom(cnarr , fdr_q , chrom):

- 21 # summary: This function segments copy number data for a single chromosome using the HaarSeg algorithm.

- 22 # param: cnarr (pandas.DataFrame) - a dataframe with columns ’ chromosome ’, ’start ’, ’end ’, ’log2 ’, and ’probes ’.

- 23 # param: fdr_q (float) - the false discovery rate threshold for segmenting the copy number data.

- 24 # param: chrom (str) - the chromosome to segment.

- 25 # return: table (pandas.DataFrame) - a dataframe with columns ’ chromosome ’, ’start ’, ’end ’, ’log2 ’, ’gene ’, and ’probes ’, representing the segmented copy number data.

- 26 #def variants_in_segment(varr , segment , fdr_q):

- 27 # summary: Generates a table of variant segments based on allele frequencies

- 28 # param: varr (object) - variant data

- 29 # param: segment (object) - genomic segment data

- 30 # param: fdr_q (float) - false discovery rate threshold

- 31 # return: table (object) - pandas DataFrame with segmented data

- 32 #def haarSeg(I, breaksFdrQ , W, rawI , haarStartLevel , haarEndLevel):

- 33 # summary: Perform segmentation on a 1D array of log -ratio values according to the HaarSeg algorithm.

- 34 # param: I (array) - A 1D array of log -ratio values , sorted according to their genomic location.

- 35 # param: W (array) - Weight matrix , corresponding to quality of measurement , with values :math:‘1/(\\sigma^2) ‘. Must have the same size as I.

- 36 # param: rawI (array) - The minimum between the raw test -sample and control -sample coverages (before applying log ratio , but after any background reduction and/or normalization). These raw red / green measurments are used to detect low -value probes , which are more sensitive to noise. Used for the non -stationary variance compensation

. Must have the same size as I.

- 37 # param: breaksFdrQ (float) - The FDR q parameter. This value should lie between 0 and 0.5.

- 38 # param: haarStartLevel (int) - The detail subband from which we start to detect peaks.

- 39 # param: haarEndLevel (int) - The detail subband until which we use to detect peaks.

- 40 # return: dict - Returns a dictionary containing the start and end points of each segment and the mean value of each segment.

- 41 #def FDRThres(x, q, stdev):

- 42 # summary: Calculates the False Discovery Rate (FDR) threshold.

- 43 # param: x (unknown type) - array of values.

- 44 # param: q (unknown type) - a floating -point number.

- 45 # param: stdev (unknown type) - a floating -point number representing the standard deviation.

- 46 # return: T (unknown type) - a floating -point number representing the FDR threshold.

- 47 #def SegmentByPeaks(data , peaks , weights):

- 48 # summary: Average the values of the probes within each segment.

- 49 # param: data (array) - the probe array values

- 50 # param: peaks (array) - Positions of copy number breakpoints in the original array

- 51 # param: weights (None or array) - optional array of weights of same length as the data array

- 52 # return: segs (array) - array of segment values obtained by averaging the values of the probes within each segment.

- 53 #def HaarConv(signal , weight , stepHalfSize):

- 54 # summary: Convolve haar wavelet function with a signal , applying circular padding.

- 55 # param: signal (array of floats) - signal to be convolved.

- 56 # param: weight (array of floats) - optional weights for the steps of the convolution.

- 57 # param: stepHalfSize (int) - half size of the step to be used in the convolution.

- 58 # return: array (array of floats) - of floats , representing the convolved signal.

- 59 #def FindLocalPeaks(signal):

- 60 # summary: Finds local maxima on positive values , local minima on negative values.

- 61 # param: signal (const array of floats): an array of floating point numbers

- 62 # return: peakLoc (array of ints): Locations of extrema in ‘signal ‘

- 63 #def UnifyLevels(baseLevel , addonLevel , windowSize):

- 64 # summary: Merge two lists of breakpoints and drop addonLevel values that are too close to baseLevel values.

- 65 # param: baseLevel (const array of ints) - a list of integers representing the base level.

- 66 # param: addonLevel (const array of ints) - a list of integers representing the addon level.

- 67 # param: windowSize (int) - an integer representing the maximum distance between addonLevel and baseLevel values.

- 68 # return: joinedLevel (array of ints) - a sorted array of integers representing the merged list of breakpoints.

- 69 #def PulseConv(signal , pulseSize):

- 70 # summary: Convolve a pulse function with a signal applying circular padding to the signal for non -stationary variance compensation.

- 71 # param: signal (const array of floats) - the signal to be convolved.

- 72 # param: pulseSize (int) - the size of the pulse function.

- 73 # return: array of floats - the convolved signal.

- 74 #def AdjustBreaks(signal , peakLoc):

- 75 # summary: Improve localization of breaks in a signal by adjusting peak locations.

- 76 # param: signal (const array of floats) - the signal to adjust the peak locations in.

- 77 # param: peakLoc (const array of ints) - the current peak locations in the signal.

- 78 # return: newPeakLoc (array of ints) - the adjusted peak locations in the signal.

- 79 #def table2coords(seg_table):

- 80 # summary: Returns x and y arrays for plotting with the help of the input segment Table

- 81 # param: seg_table (list) - a list of tuples containing start , size , and value.

- 82 # return: x (list) - a list of x-coordinates for plotting.

- 83 # return: y (list) - a list of y-coordinates for plotting.

- 84 #Here are the comments and the specs:

- 85 #write a function called "UnifyLevels" that takes three parameters: baseLevel (a constant array of integers representing the base levels) , addonLevel (a constant array of integers representing the additional levels), and windowSize (an integer representing the window size). The function should merge the two lists of breakpoints ,

but drop addonLevel values that are within windowSize of baseLevel values. The function should return an array of integers called joinedLevel. If addonLevel is an empty array , the function should return baseLevel. The output should be sorted and of dtype=np.int_.

- 86 #def UnifyLevels(baseLevel , addonLevel , windowSize):

B.2.2 SMALL PROMPT

- 1 Write a function with the following specs:

- 2 --specs begin here --

- 3 #write a function called "UnifyLevels" that takes three parameters: baseLevel (a constant array of integers representing the base levels) , addonLevel (a constant array of integers representing the additional levels), and windowSize (an integer representing the window size). The function should merge the two lists of breakpoints ,

but drop addonLevel values that are within windowSize of baseLevel values. The function should return an array of integers called joinedLevel. If addonLevel is an empty array , the function should return baseLevel. The output should be sorted and of dtype=np.int_.

- 4 param: baseLevel (const array of ints) - a list of integers representing the base level.

- 5 param: addonLevel (const array of ints) - a list of integers representing the addon level.

- 6 param: windowSize (int) - an integer representing the maximum distance between addonLevel and baseLevel values.

- 7 return: joinedLevel (array of ints) - a sorted array of integers representing the merged list of breakpoints.

- 8 --specs end here --

- 9 Note the function will be embedded in the following context

- 10 --context begins here --

- 11 import random

- 12 import hashlib

- 13 import numpy as np

- 14 import skimage

- 15 import skimage.measure

- 16 import scipy.ndimage

- 17 import os

- 18 import logging

- 19 from functools import wraps

- 20 from scipy import stats

- 21 import sys

- 22 import math

- 23 <<insert solution here >>

- 24 def main():

- 25 np.random.seed(<|int;range=0,1000|>)

- 26 baseLevel = np.random.randint(20, size=(10))

- 27 np.random.seed(<|int;range=0,1000|>)

- 28 addonLevel = np.random.randint(20, size=(10))

- 29 print(UnifyLevels(baseLevel , addonLevel , 3))

- 30 if __name__ == "__main__":

- 31 // context continues

- 32 --context ends here --

- 33 Make sure to only generate the function and not any of the context. Make sure you are generating valid , runnable code. Begin your solution with:

- 34 def UnifyLevels(baseLevel , addonLevel , windowSize):

- 35 MAKE SURE TO INDENT THE BODY OF YOUR FUNCTION BY A TAB

B.2.3 NO COMMENT PROMPT

- 1 This is in python

- 2 write a function called "UnifyLevels" that takes three parameters: baseLevel (a constant array of integers representing the base levels) , addonLevel (a constant array of integers representing the additional levels), and windowSize (an integer representing the window size). The function should merge the two lists of breakpoints ,

but drop addonLevel values that are within windowSize of baseLevel values. The function should return an array of integers called joinedLevel. If addonLevel is an empty array , the function should return baseLevel. The output should be sorted and of dtype=np.int_.

- 3

- 4 def UnifyLevels(baseLevel , addonLevel , windowSize):

- 5

- 6 Here are the imports:

- 7 from scipy import stats

- 8 import math

- 9 import numpy as np

- 10 import logging

- 11 import pandas as pd

- 12 Here are the global variables:

- 13 Here are the class declarations:

- 14 Here are the additional function declarations:

- 15 def segment_haar(cnarr , fdr_q):

- 16 summary: Segment CNVkit data using HaarSeg algorithm

- 17 param: cnarr (CopyNumArray) - binned , normalized copy ratios

- 18 param: fdr_q (float) - false discovery rate q-value

- 19 return: segarr (CopyNumArray) - CBS data table as a CNVkit object

- 20 def one_chrom(cnarr , fdr_q , chrom):

- 21 summary: This function segments copy number data for a single chromosome using the HaarSeg algorithm.

- 22 param: cnarr (pandas.DataFrame) - a dataframe with columns ’ chromosome ’, ’start ’, ’end ’, ’log2 ’, and ’probes ’.

- 23 param: fdr_q (float) - the false discovery rate threshold for segmenting the copy number data.

- 24 param: chrom (str) - the chromosome to segment.

- 25 return: table (pandas.DataFrame) - a dataframe with columns ’ chromosome ’, ’start ’, ’end ’, ’log2 ’, ’gene ’, and ’probes ’, representing the segmented copy number data.

- 26 def variants_in_segment(varr , segment , fdr_q):

- 27 summary: Generates a table of variant segments based on allele frequencies

- 28 param: varr (object) - variant data

- 29 param: segment (object) - genomic segment data

- 30 param: fdr_q (float) - false discovery rate threshold

- 31 return: table (object) - pandas DataFrame with segmented data

- 32 def haarSeg(I, breaksFdrQ , W, rawI , haarStartLevel , haarEndLevel):

- 33 summary: Perform segmentation on a 1D array of log -ratio values according to the HaarSeg algorithm.

- 34 param: I (array) - A 1D array of log -ratio values , sorted according to their genomic location.

- 35 param: W (array) - Weight matrix , corresponding to quality of measurement , with values :math:‘1/(\\sigma^2) ‘. Must have the same size as I.

- 36 param: rawI (array) - The minimum between the raw test -sample and control -sample coverages (before applying log ratio , but after any background reduction and/or normalization). These raw red / green measurments are used to detect low -value probes , which are more sensitive to noise. Used for the non -stationary variance compensation

. Must have the same size as I.

- 37 param: breaksFdrQ (float) - The FDR q parameter. This value should lie between 0 and 0.5.

- 38 param: haarStartLevel (int) - The detail subband from which we start to detect peaks.

- 39 param: haarEndLevel (int) - The detail subband until which we use to detect peaks.

- 40 return: dict - Returns a dictionary containing the start and end points of each segment and the mean value of each segment.

- 41 def FDRThres(x, q, stdev):

- 42 summary: Calculates the False Discovery Rate (FDR) threshold.

- 43 param: x (unknown type) - array of values.

- 44 param: q (unknown type) - a floating -point number.

- 45 param: stdev (unknown type) - a floating -point number representing the standard deviation.

- 46 return: T (unknown type) - a floating -point number representing the FDR threshold.

- 47 def SegmentByPeaks(data , peaks , weights):

- 48 summary: Average the values of the probes within each segment.

- 49 param: data (array) - the probe array values

- 50 param: peaks (array) - Positions of copy number breakpoints in the original array

- 51 param: weights (None or array) - optional array of weights of same length as the data array

- 52 return: segs (array) - array of segment values obtained by averaging the values of the probes within each segment.

- 53 def HaarConv(signal , weight , stepHalfSize):

- 54 summary: Convolve haar wavelet function with a signal , applying circular padding.

- 55 param: signal (array of floats) - signal to be convolved.

- 56 param: weight (array of floats) - optional weights for the steps of the convolution.

- 57 param: stepHalfSize (int) - half size of the step to be used in the convolution.

- 58 return: array (array of floats) - of floats , representing the convolved signal.

- 59 def FindLocalPeaks(signal):

- 60 summary: Finds local maxima on positive values , local minima on negative values.

- 61 param: signal (const array of floats): an array of floating point numbers

- 62 return: peakLoc (array of ints): Locations of extrema in ‘signal ‘

- 63 def UnifyLevels(baseLevel , addonLevel , windowSize):

- 64 summary: Merge two lists of breakpoints and drop addonLevel values that are too close to baseLevel values.

- 65 param: baseLevel (const array of ints) - a list of integers representing the base level.

- 66 param: addonLevel (const array of ints) - a list of integers representing the addon level.

- 67 param: windowSize (int) - an integer representing the maximum distance between addonLevel and baseLevel values.

- 68 return: joinedLevel (array of ints) - a sorted array of integers representing the merged list of breakpoints.

- 69 def PulseConv(signal , pulseSize):

- 70 summary: Convolve a pulse function with a signal applying circular padding to the signal for non -stationary variance compensation.

- 71 param: signal (const array of floats) - the signal to be convolved.

- 72 param: pulseSize (int) - the size of the pulse function.

- 73 return: array of floats - the convolved signal.

- 74 def AdjustBreaks(signal , peakLoc):

- 75 summary: Improve localization of breaks in a signal by adjusting peak locations.

- 76 param: signal (const array of floats) - the signal to adjust the peak locations in.

- 77 param: peakLoc (const array of ints) - the current peak locations in the signal.

- 78 return: newPeakLoc (array of ints) - the adjusted peak locations in the signal.

- 79 def table2coords(seg_table):

- 80 summary: Returns x and y arrays for plotting with the help of the input segment Table

- 81 param: seg_table (list) - a list of tuples containing start , size , and value.

- 82 return: x (list) - a list of x-coordinates for plotting.

- 83 return: y (list) - a list of y-coordinates for plotting.

- 84 Here are the comments and the specs:

- 85 write a function called "UnifyLevels" that takes three parameters: baseLevel (a constant array of integers representing the base levels) , addonLevel (a constant array of integers representing the additional levels), and windowSize (an integer representing the window size). The function should merge the two lists of breakpoints ,

but drop addonLevel values that are within windowSize of baseLevel values. The function should return an array of integers called joinedLevel. If addonLevel is an empty array , the function should return baseLevel. The output should be sorted and of dtype=np.int_.

- 86 def UnifyLevels(baseLevel , addonLevel , windowSize):

- B.2.4 REFERENCE CODE

- 1 def UnifyLevels(baseLevel , addonLevel , windowSize):

- 2 """Unify several decomposition levels.

- 3

- 4 Merge the two lists of breakpoints , but drop addonLevel values that are too

- 5 close to baseLevel values.

- 6

- 7 Parameters

- 8 ----------

- 9 baseLevel : const array of ints

- 10 addonLevel : const array of ints

- 11 windowSize : int

- 12

- 13 Returns

- 14 -------

- 15 joinedLevel : array of ints

- 16

- 17 Source: HaarSeg.c

- 18 """

- 19 if not len(addonLevel):

- 20 return baseLevel

- 21 joinedLevel = []

- 22 addon_idx = 0

- 23 for base_elem in baseLevel:

- 24 while addon_idx < len(addonLevel):

- 25 addon_elem = addonLevel[addon_idx]

- 26 if addon_elem < base_elem - windowSize:

- 27 joinedLevel.append(addon_elem)

- 28 addon_idx += 1

- 29 elif base_elem - windowSize <= addon_elem <= base_elem + windowSize:

- 30 addon_idx += 1

- 31 else:

- 32 assert base_elem + windowSize < addon_elem

- 33 break

- 34 joinedLevel.append(base_elem)

- 35 last_pos = baseLevel[-1] + windowSize if len(baseLevel) else -1

- 36 while addon_idx < len(addonLevel) and addonLevel[addon_idx] <= last_pos:

- 37 addon_idx += 1

- 38 if addon_idx < len(addonLevel):

- 39 joinedLevel.extend(addonLevel[addon_idx:])

- 40 return np.array(sorted(joinedLevel), dtype=np.int_)

- B.2.5 CONTEXT

- 1 import random

- 2 import hashlib

- 3 import numpy as np

- 4 import skimage

- 5 import skimage.measure

- 6 import scipy.ndimage

- 7 import os

- 8 import logging

- 9 from functools import wraps

- 10 from scipy import stats

- 11 import sys

- 12 import math

- 13 <<insert solution here >>

- 14 def main():

- 15 np.random.seed(<|int;range=0,1000|>)

- 16 baseLevel = np.random.randint(20, size=(10))

- 17 np.random.seed(<|int;range=0,1000|>)

- 18 addonLevel = np.random.randint(20, size=(10))

- 19 print(UnifyLevels(baseLevel , addonLevel , 3))

- 20 if __name__ == "__main__":

- 21 main()

C ROSALIND EXAMPLES

The Rosalind dataset consists of 253 problems from the bioinformatics training website rosalind.info. For each of the 253 problems that make up the Rosalind portion of BIOCODER, we created both a prompt that we fed into the LLMs as well as a golden code solution that we either wrote ourselves or found on public GitHub repositories.

For each problem, to create the prompt, we first scraped the problem’s page on the Rosalind website to obtain a problem description. This problem description would then make up the first section of the prompt. For instance, for the Rosalind problem ba1a, here is the scraped problem description:

- 1 # This is the first problem in a collection of "code challenges" to accompany Bioinformatics Algorithms: An Active -Learning Approach by Phillip Compeau & Pavel Pevzner.

- 2 # A k-mer is a string of length k.

- 3 # We define Count(Text , Pattern) as the number of times that a k-mer Pattern appears as a substring of Text.

- 4 # For example ,

- 5 # $\textit{Count}(\text{ACA}\color{green}\textbf{ACTAT}\color{black}\text {GCAT}\color{green}\textbf{ACTAT}\color{black}\text{CGGGA}\color{ green}\textbf{ACTAT}\color{black}\text{CCT}, {\color{green}\textbf{ ACTAT}}) = 3$.

- 6 # We note that Count($\text{CG}\color{green}\textbf{ATATA}\color{black}\ text{TCC}\color{green}\textbf{ATA}\color{black}\text{G}$, $\color{ green}\textbf{ATA}$) is equal to 3 (not 2) since we should account for overlapping occurrences of Pattern in Text.

- 7 # To compute Count(Text , Pattern), our plan is to "slide a window" down Text , checking whether each k-mer substring of Text matches Pattern. We will therefore refer to the k-mer starting at position i of Text as Text(i, k). Throughout this book , we will often use 0-based indexing , meaning that we count starting at 0 instead of 1. In this case , Text begins at position 0 and ends at position |Text| - 1 (| Text| denotes the number of symbols in Text). For example , if Text = GACCATACTG ,

- 8 # then Text(4, 3) = ATA. Note that the last k-mer of Text begins at position |Text| - k, e.g., the last 3-mer of GACCATACTG starts at position 10 - 3 = 7. This discussion results in the following pseudocode for computing Count(Text , Pattern).

- 9 # PatternCount(Text , Pattern)

- 10 # count <- 0

- 11 # for i <- 0 to |Text| - |Pattern|

- 12 # if Text(i, |Pattern|) = Pattern

- 13 # count <-s count + 1

- 14 # return count

- 15 # Implement PatternCount

- 16 # Given: {{DNA strings}} Text and Pattern.

- 17 # Return: Count(Text , Pattern).

- 18 # Sample Dataset

- 19 # GCGCG

- 20 # GCG

- 21 # Sample Output

- 22 # 2

From the sample problem description above you can see that a problem description generally consists of an explanation of the problem at hand and a sample test case. The second section of the prompt consists of a custom block of text that prompts the LLM to generate its code to fit the context in which the generated Rosalind code will be run. For instance, for the same Rosalind problem ba1a, here is the custom block of text that makes up the second section of the prompt:

- 1 #write the solve() function to solve the above problem

- 2 #Do NOT print anything

- 3 #Do NOT make comments

- 4 #Do NOT call the main() function.

- 5 #Use 4 spaces for tabs.

- 6 #input_str: a string

- 7 #return output: another string

- 8

- 9

- 10 def main():

- 11 with open("input.txt", "r") as f:

- 12 output = solve(f.read())

- 13 print(output)

- 14

- 15 # Write your solution here

- 16 # Begin with: def solve(input_str):

From the example above, you can see that the custom block of text consists of a prompt telling the LLM to generate a solve function as well as the context in which the generated code will be run, namely the main function that is included in the custom text. Each prompt consists of these two sections, i.e., the first section of the prompt will include a problem description consisting of an explanation of the problem as well as a sample test case, and the second section of the prompt will consist of custom prompting text that instructs the LLM to generate a solve function and provides the context in which the Rosalind code will be run.

For each of the 253 problems that make up the Rosalind portion of BIOCODER, we either found online or wrote custom golden code solutions. Each golden code solution is meant to be run in the custom context, following the main and solve structure that is described in every Rosalind prompt. For instance, here is the golden code solution for the Rosalind problem ba1a:

- 1 # Compute the Number of Times a Pattern Appears in a Text

- 2

- 3 def substrings(text , size):

- 4 for i in range(len(text) - size + 1):

- 5 yield text[i : i + size]

- 6

- 7 def pattern_count(text , pattern):

- 8 return sum(pattern == x for x in substrings(text , len(pattern)))

- 9

- 10 def solve(input_str):

- 11 text , pattern = input_str.splitlines()

- 12

- 13 return pattern_count(text , pattern)

As you can see, a golden code solution consists of the solve function requested by the corresponding prompt, as well as any additional helper functions that may be called (possibly recursively) by the solve function. For more information on the Rosalind portion of the BIOCODER dataset, three detailed examples of Rosalind prompts and golden code solutions are provided below to further illustrate the differences between the codes and the prompts.

C.1 EXAMPLE 1

The given input is a problem description in the field of bioinformatics, namely the Reverse Complement Problem. The problem is related to the DNA sequences, which are strings containing the characters ‘A’, ‘T’, ‘G’, and ‘C’. Each of these characters represents a nucleotide in the DNA.

The problem requires the model to find the reverse complement of a given DNA string. The reverse complement of a DNA string is obtained by replacing each nucleotide in the string with its complement (‘A’ with ‘T’, ‘T’ with ‘A’, ‘C’ with ‘G’, and ‘G’ with ‘C’), and then reversing the entire string. For example, the reverse complement of “GTCA” is “TGAC.”

The problem description includes a sample dataset (DNA string) and its corresponding output, which is its reverse complement. The problem also specifies the format for the solution. For instance, a Python function named ‘solve’ takes a DNA string as an input and returns its reverse complement as output. The function should not print anything or include any comments.

- 1

- 2 Reverse Complement Problem:

- 3 In DNA strings , symbols ’A’ and ’T’ are complements of each other , as are ’C’ and ’G’.

- 4 Given a nucleotide p, we denote its complementary nucleotide as p. The reverse complement of a DNA string Pattern = p1âĂępn is the string Pattern = pn âĂę p1 formed by taking the complement of each nucleotide in Pattern , then reversing the resulting string.

- 5

- 6 For example , the reverse complement of Pattern = "GTCA" is Pattern = " TGAC".

- 7

- 8

- 9 Problem:

- 10 Find the reverse complement of a DNA string.

- 11 Given: A DNA string Pattern.

- 12 Return: Pattern , the reverse complement of Pattern.

- 13

- 14 Sample Dataset:

- 15 AAAACCCGGT

- 16

- 17 Sample Output:

- 18 ACCGGGTTTT

- 19

- 20 write the solve() function to solve the above problem.

- 21

- 22 Do NOT print anything

- 23 Do NOT make comments

- 24 Do NOT call the main() function.

- 25 Use 4 spaces for tabs.

- 26

- 27 input_str: a string

- 28 return output: another string

- 29

- 30 def main():

- 31 with open("input.txt", "r") as f:

- 32 output = solve(f.read())

- 33 print(output)

- 34

- 35

- 36 Write your solution here:

- 37 Begin with: def solve(input_str):

The given sample solution to the problem is written in Python. It defines two functions: ‘revcomp’ and ‘main’. The ‘revcomp’ function computes the reverse complement of a DNA string by first reversing the string and then replacing each nucleotide with its complement using Python’s ‘str.maketrans’ method. The ‘main’ function reads a DNA string from a file, computes its reverse complement using the ‘revcomp’ function, and then prints the result.

The ground-truth code is a solution to the problem presented in the Rosalind platform, a platform providing a collection of bioinformatics problems. This platform is known for helping students and researchers learn bioinformatics by solving provided computational problems.

- 1 Find the Reverse Complement of a String

- 2

- 3

- 4 def revcomp(seq):

- 5 return seq[::-1].translate(str.maketrans("ACGT", "TGCA"))

- 6

- 7

- 8 def main(file):

- 9 seq = open(file).read().splitlines()[0]

- 10 print(revcomp(seq))

C.2 EXAMPLE 2 For this example, the input consists of several parts.

The initial portion discusses the Hardy Weinberg Principle and Mendel’s laws, which are used in genetics to predict how genes will distribute throughout a population over time. The Hardy Weinberg Principle assumes that a population is large and remains in genetic equilibrium, meaning the frequency of each allele (a version of a gene) remains constant across generations, under certain conditions.

The problem statement is presented: if we know the proportion of homozygous recessive individuals (those with two copies of the recessive allele) for each of several genetic factors in a population, can we predict the probability that a randomly selected individual carries at least one copy of the recessive allele for each factor? The task is to write a Python function named ’solve’ to solve this problem.

A sample dataset is given, represented as an array of floating-point numbers, each of which indicates the proportion of homozygous recessive individuals for a specific genetic factor in the population. The ‘solve’ function will receive this array as a string and should return another string representing the solution array.

The final portion of the input is a sample Python code that reads the dataset from a file, calls the ‘solve’ function to compute the solution, and then prints the solution. However, the user is instructed not to print anything, not to write comments, and not to call this ‘main’ function in the solution. The user is also instructed to use four spaces for indentation.

- 1 Genetic Drift and the Hardy -Weinberg Principle Mendel ’s laws of segregation and independent assortment are excellent for the study of

individual organisms and their progeny , but they say nothing about how alleles move through a population over time.

- 2

- 3 Our first question is: when can we assume that the ratio of an allele in a population , called the allele frequency , is stable? G. H. Hardy and

Wilhelm Weinberg independently considered this question at the turn of the 20th Century , shortly after Mendel ’s ideas had been rediscovered. They concluded that the percentage of an allele in a population of individuals is in genetic equilibrium when five conditions are satisfied: The population is so large that random changes in the allele frequency are negligible. No new mutations are affecting the gene of interest; The gene does not influence survival or reproduction , so natural selection is not occurring; Gene flow , or the change in allele frequency due to migration into and out of the population , is negligible. Mating occurs randomly with respect to the

gene of interest. The Hardy -Weinberg principle states that if a population is in genetic equilibrium for a given allele , then its frequency will remain constant and evenly distributed through the population. Unless the gene in question is important to survival or reproduction , Hardy -Weinberg usually offers a reasonable enough model

of population genetics. One of the many benefits of the Mendelian theory of inheritance and simplifying models like Hardy -Weinberg is that they help us predict the probability with which genetic diseases will be inherited , so as to take appropriate preventative measures. Genetic diseases are usually caused by mutations to chromosomes , which are passed on to subsequent generations. The simplest and most widespread case of a genetic disease is a single gene disorder , which

is caused by a single mutated gene. Over 4,000 such human diseases have been identified , including cystic fibrosis and sickle -cell anemia. In both of these cases , the individual must possess two recessive alleles for a gene in order to contract the disease. Thus , carriers can live their entire lives without knowing that they can pass the disease on to their children. The above introduction to genetic equilibrium leaves us with a basic and yet very practical question regarding gene disorders: if we know the number of people who have a disease encoded by a recessive allele , can we predict the number of carriers in the population?

- 4

- 5 Problem:

- 6

- 7 To model the Hardy -Weinberg principle , assume that we have a population of $N$ diploid individuals. If an allele is in genetic equilibrium , then because mating is random , we may view the $2N$ chromosomes as

receiving their alleles uniformly. In other words , if there are $m$ dominant alleles , then the probability of a selected chromosome exhibiting the dominant allele is simply $p = \frac{m}{2N}$. Because the first assumption of genetic equilibrium states that the population is so large as to be ignored , we will assume that $N$ is infinite , so that we only need to concern ourselves with the value of

$p$.

- 8

- 9 Given: An array $A$ for which $A[k]$ represents the proportion of homozygous recessive individuals for the $k$ -th Mendelian factor in a

diploid population. Assume that the population is in genetic equilibrium for all factors.

- 10

- 11 Return: An array $B$ having the same length as $A$ in which $B[k]$ represents the probability that a randomly selected individual carries at least one copy of the recessive allele for the $k$ -th factor.

- 12

- 13 Sample Dataset

- 14 0.1 0.25 0.5

- 15

- 16 write the solve() function to solve the above problem

- 17

- 18 Do NOT print anything

- 19 Do NOT make comments

- 20 Do NOT call the main() function.

- 21

- 22 Use 4 spaces for tabs.

- 23

- 24 input_str: a string

- 25 return output: another string

- 26

- 27 def main():

- 28 with open("input.txt", "r") as f:

- 29 output = solve(f.read())

- 30 print(output)

- 31

- 32 Write your solution here

- 33 Begin with: def solve(input_str):

In the output portion, a Python function afrq is presented, which takes an array of proportions of homozygous recessive individuals for each genetic factor in a population and returns an array of probabilities that a randomly selected individual carries at least one copy of the recessive allele for each factor. The main function uses this afrq function to solve the problem based on the input data. It takes the input data from a file, applies the afrq function to compute the solution, and then prints the solution in the form of a space-separated string of floating-point numbers, each rounded to three decimal places.

- 1 # Counting Disease Carriers

- 2

- 3 from math import sqrt

- 4 from .helpers import Parser

- 5

- 6 def afrq(a):

- 7 """Counting Disease Carriers"""

- 8 return [2 * sqrt(x) - x for x in a]

- 9

- 10 def main(file):

- 11 b = afrq(Parser(file).floats())

- 12 print(*[round(x, 3) for x in b])

C.3 EXAMPLE 3

This Python script is designed to find the most frequent k-mers (substrings of length k) in a given DNA sequence (a string), while allowing for a specified number of mismatches. This problem is referred to as the “Frequent Words with Mismatches Problem.”

The input to this problem is a string containing two distinct parts separated by a new line.

The DNA sequence is represented as a string of characters ‘A’, ‘C’, ‘G’, and ‘T’. Two integers, k and d, are separated by a space. The integer k refers to the length of the substrings (k-mers) of interest, and d specifies the maximum number of mismatches that are allowed in a k-mer.

- 1

- 2 We defined a mismatch in ‘‘Compute the Hamming Distance Between Two StringsâĂİ. We now generalize âĂ Find the Most Frequent Words in a StringâĂİ to incorporate mismatches as well. Given strings Text and Pattern as well as an integer d, we define Countd(Text , Pattern) as the total number of occurrences of Pattern in Text with at most d mismatches. For example , Count1(AACAAGCTGATAAACATTTAAAGAG , AAAAA) = 4

because AAAAA appears four times in this string with at most one mismatch: AACAA , ATAAA , AAACA , and AAAGA. Note that two of these occurrences overlap.

- 3

- 4 A most frequent k-mer with up to d mismatches in Text is simply a string Pattern maximizing Countd(Text , Pattern) among all k-mers. Note that Pattern does not need to actually appear as a substring of Text; for example , AAAAA is the most frequent 5-mer with 1 mismatch in AACAAGCTGATAAACATTTAAAGAG , even though AAAAA does not appear exactly in this string. Keep this in mind while solving the following problem

.

- 5

- 6 Frequent Words with Mismatches Problem

- 7

- 8 Find the most frequent k-mers with mismatches in a string. Given: A string Text as well as integers k and d. Return: All most frequent kmers with up to d mismatches in Text.

- 9

- 10 Sample Dataset

- 11 ACGTTGCATGTCGCATGATGCATGAGAGCT

- 12 4 1

- 13

- 14 Sample Output

- 15 GATG ATGC ATGT

- 16

- 17 write the solve() function to solve the above problem

- 18 Do NOT print anything

- 19 Do NOT make comments

- 20 Do NOT call the main() function.

- 21 Use 4 spaces for tabs.

- 22 input_str: a string

- 23 return output: another string

- 24

- 25 def main():

- 26 with open("input.txt", "r") as f:

- 27 output = solve(f.read())

- 28 print(output)

- 29

- 30 Write your solution here

- 31 Begin with: def solve(input_str): Here is the corresponding reference code:

- 1 # Find the Most Frequent Words with Mismatches in a String

- 2

- 3 from .ba1g import hamming

- 4 from .ba1b import count_kmers , most_frequent

- 5 from itertools import product

- 6

- 7 # Note , the best kmer might not be observed in our sequence. The simplistic

- 8 # method here simply checks all possible kmers (which is ~17M for k = 12)

- 9

- 10 def generate_kmers(k):

- 11 return ("".join(x) for x in product(["A", "C", "G", "T"], repeat=k))

- 12

- 13 def count_hamming_kmers(kmers , d, k):

- 14 for x in generate_kmers(k):

- 15 count = sum(kmers[y] for y in kmers if hamming(x, y) <= d)

- 16 if count > 0:

- 17 yield [x, count]

- 18

- 19 def main(file):

- 20 seq, ints = open(file).read().splitlines()

- 21 k, d = list(map(int, ints.split()))

- 22 kmers = count_kmers(seq, k)

- 23 hkmers = dict(count_hamming_kmers(kmers , d, k))

- 24 print(*most_frequent(hkmers))

- D ADDITIONAL MODEL INFORMATION

We were unable to run the 16B models due to issues with scaling and sharding on the A100s. We found that the following parameters produced the best results across all models.

- • top_k = 50
- • top_p = 0.95
- • temperature = 0.7
- • early_stopping = True
- • num_return_sequences = 1
- • do_sample = True

We utilized similar parameters to make model testing consistent across all tested models. This approach allowed for a more unbiased comparison of how each model performed on our benchmark. We determined these weights by averaging the recommended parameters for the above models and then used Bayesian optimization to determine the most optimal parameters.

Note that we used the version of GPT-3.5-Turbo hosted on Azure OpenAI Service (gpt-35-turbo-0301). There may be some minor differences compared with the OpenAI Platform version.

- E UTILIZING CHATGPT FOR PROMPT GENERATION

ChatGPT served as a pivotal component in our prompt generation, enabling us to assemble comprehensive annotations for the ground-truth functions and their dependencies. This eliminated the need for extensive manual labor that would have otherwise been spent on annotation. Utilizing GPT-3.5 Turbo calls further ensures consistency across annotations, mitigating the variability that might arise from human annotators.

The prompt was constructed as follows:

- 1 Generate a prompt for the following function , such that a programmer could reproduce it based solely on the description. Emphasize crucial

components like the return statement , complex function calls , and

ambiguous variables. However , avoid line -by-line descriptions; aim to provide a high -level overview. The response should be in plain text , free from any formatting such as Markdown. Keep the length under 200 words or 15 lines , whichever is shorter.

- 2 Here is the function for you to describe:

- 3 <function >

- 4 Here are the additional dependencies that may or may not be used within the function:

- 5 <dependencies >

- 6 Again , limit your responses to no more than 200 words or 15 lines , whichever comes first.

- 7 Start your prompt with "write a function".

- F MODEL CONTEXT LIMITS

Since the limit includes both input and output tokens, we reserved 256 tokens for the output, therefore reducing the input length limit by 256. In our best effort to continue testing within these context limits, instead of immediately treating a failing test case, we attempted to truncate the prompts such that they fit barely below the context limit of each respective model. For all models, we also tested a version with the function summary at the top. Note that for all models, the context limit was identical regardless of the parameter count.

- G CODEREVAL COMPARISON

To validate the performance of code LLMs, multiple existing benchmarks have been proposed, including only cases of generating a standalone function, i.e., a function that invokes or accesses only built-in functions and standard libraries. However, standalone functions constitute only about 30% of the functions from real open-source projects. To assess a model’s performance for pragmatic code generation (i.e., code generation for real settings of open-source or proprietary code), CoderEval (?) proposes a benchmark named CoderEval of pragmatic code generation with generative pre-trained models.

In comparison to CoderEval, our approach is more focused on bioinformatics, as we ensure each function requires a certain level of domain knowledge in bioinformatics, as opposed to entry-level data science or software engineering tasks. Moreover, we employ a more advanced parser, implementing a comprehensive parsing of the respective GitHub repositories. This includes the imported third-party packages and the classes that have been imported from another actual file.

We also conduct function testing at the file level, rather than the repository level, manually extracting the context at the end and running it through a framework that catches more errors than CoderEval. Additionally, our dataset is larger than CoderEval’s, which consists only of 230 functions from 43 Python projects and 230 methods from 10 Java projects, while we have data from more than two thousand sources.

CoderEval classifies tasks into six levels according to the source of dependency outside the function, while we directly use the most complex dependencies. Moreover, CoderEval employs a humanlabeled version description from 13 experienced engineers, while we leverage ChatGPT to generate function descriptions. Similarly, for test cases, our use of fuzz testing allows us to scale to large data volumes without the need for intensive manual annotation.

However, we share some similarities with CoderEval. Both BIOCODER and CoderEval can be used to assess the performance of models beyond merely generating standalone functions. Given the need to address dependencies with context, we both establish testing based on Docker, cloning GitHub repositories and their associated import dependencies. With ‘k’ candidate outputs generated by different models (e.g., 20), we simulate scenarios in actual IDEs.

- H PROMPT EXAMPLES Below are examples of each prompt type:

- 1. Summary Only An example is shown below:

- 1 #This is in python

- 2 #Write a function named "planewise_morphology_closing" that accepts two parameters , "x_data" and "structuring_element". Inside the function , create a new variable named "y_data" that will hold the result of the morphological closing operation. Loop through each item in "x_data" and perform the same operation on each. Do this by using the "enumerate" function to

get both the index and plane for each item. Use the "closing" function from the "skimage.morphology" module to perform the morphological closing operation on each plane using the " structuring_element" parameter. Finally , return the newly created "y_data" array that holds the result of the operation. Ensure that the function uses the "numpy" and "skimage. morphology" modules , as they are dependencies for the function.

- 3 #

- 4 #def planewise_morphology_closing(x_data , structuring_element):

- 5

- 6

- 2. Uncommented An example is shown below:

- 1 This is in python

- 2 Here are the imports:

- 3 from cellprofiler.library.modules import closing

- 4 from ._help import HELP_FOR_STREL

- 5 from cellprofiler_core.module import ImageProcessing

- 6 from cellprofiler_core.setting import StructuringElement

- 7 Here are the global variables:

- 8 Here are the class declarations:

- 9 class Closing(ImageProcessing):

- 10 attributes: self.structuring_element ,self.function

- 11 methods:

- 12 def create_settings(self):

- 13 summary: creates settings for the Closing function

- 14 param: self (unknown) - instance of the Closing function

- 15 return: None

- 16 def settings(self):

- 17 summary: Returns settings from superclass with self. structuring_element added.

- 18 param: None

- 19 return: list - __settings__ with self.structuring_element appended.

- 20 def visible_settings(self):

- 21 summary: This function returns the visible settings of a Closing object.

- 22 param: self (Closing) - the Closing object whose settings are to be returned.

- 23 return: list - the visible settings of the Closing object.

- 24 def run(self , workspace):

- 25 summary: Applies morphological closing to an image in a workspace.

- 26 param: workspace (object) - the workspace containing the image.

- 27 return: None

- 28 Here are the additional function declarations:

- 29 def planewise_morphology_closing(x_data , structuring_element):

- 30 summary: Applies morphology closing operation to each plane in

- x_data using structuring_element and returns the resulting

- y_data.

- 31 param: x_data (numpy array) - 3D array containing the input data.

- 32 param: structuring_element (numpy array) - Structuring element used to perform the closing operation.

- 33 return: y_data (numpy array) - 3D array containing the result of closure operation on each plane in x_data.

- 34 Here are the comments and the specs:

- 35 Write a function named "planewise_morphology_closing" that accepts two parameters , "x_data" and "structuring_element". Inside the function , create a new variable named "y_data" that will hold

the result of the morphological closing operation. Loop through each item in "x_data" and perform the same operation on each. Do this by using the "enumerate" function to get both the index

and plane for each item. Use the "closing" function from the " skimage.morphology" module to perform the morphological closing

operation on each plane using the "structuring_element" parameter. Finally , return the newly created "y_data" array that holds the result of the operation. Ensure that the function uses the "numpy" and "skimage.morphology" modules , as they are dependencies for the function.

- 36 def planewise_morphology_closing(x_data , structuring_element):

- 37

- 38

Note that it is nearly certain that each prompt will contain extraneous data that does not need to be used for the function. The goal is for the LLM to ensure it uses the correct context for the function. Note that for prompts that exceed the model’s context limit, this prompt is likely to fail.

#### 3. Summary at Bottom

- 1 #Here are the imports:

- 2 #from cellprofiler.library.modules import closing

- 3 #from ._help import HELP_FOR_STREL

- 4 #from cellprofiler_core.module import ImageProcessing

- 5 #from cellprofiler_core.setting import StructuringElement

- 6 #Here are the global variables:

- 7 #Here are the class declarations:

- 8 #class Closing(ImageProcessing):

- 9 # attributes: self.structuring_element ,self.function

- 10 # methods:

- 11 # def create_settings(self):

- 12 # summary: creates settings for the Closing function

- 13 # param: self (unknown) - instance of the Closing function

- 14 # return: None

- 15 # def settings(self):

- 16 # summary: Returns settings from superclass with self. structuring_element added.

- 17 # param: None

- 18 # return: list - __settings__ with self.structuring_element appended.

- 19 # def visible_settings(self):

- 20 # summary: This function returns the visible settings of a Closing object.

- 21 # param: self (Closing) - the Closing object whose settings are to be returned.

- 22 # return: list - the visible settings of the Closing object.

- 23 # def run(self , workspace):

- 24 # summary: Applies morphological closing to an image in a workspace.

- 25 # param: workspace (object) - the workspace containing the image.

- 26 # return: None

- 27 #Here are the additional function declarations:

- 28 #def planewise_morphology_closing(x_data , structuring_element):

- 29 # summary: Applies morphology closing operation to each plane in

- x_data using structuring_element and returns the resulting

- y_data.

- 30 # param: x_data (numpy array) - 3D array containing the input data.

- 31 # param: structuring_element (numpy array) - Structuring element used to perform the closing operation.

- 32 # return: y_data (numpy array) - 3D array containing the result of closure operation on each plane in x_data.

- 33 #Here are the comments and the specs:

- 34 #Write a function named "planewise_morphology_closing" that accepts two parameters , "x_data" and "structuring_element". Inside the function , create a new variable named "y_data" that will hold the result of the morphological closing operation. Loop through each item in "x_data" and perform the same operation on each. Do this by using the "enumerate" function to

get both the index and plane for each item. Use the "closing" function from the "skimage.morphology" module to perform the morphological closing operation on each plane using the " structuring_element" parameter. Finally , return the newly created "y_data" array that holds the result of the operation. Ensure that the function uses the "numpy" and "skimage. morphology" modules , as they are dependencies for the function.

- 35 def planewise_morphology_closing(x_data , structuring_element):

- 36

#### 4. Summary at Top

- 1 #This is in python

- 2 #Write a function named "planewise_morphology_closing" that accepts two parameters , "x_data" and "structuring_element". Inside the function , create a new variable named "y_data" that will hold the result of the morphological closing operation. Loop through each item in "x_data" and perform the same operation on each. Do this by using the "enumerate" function to

get both the index and plane for each item. Use the "closing" function from the "skimage.morphology" module to perform the morphological closing operation on each plane using the " structuring_element" parameter. Finally , return the newly created "y_data" array that holds the result of the operation. Ensure that the function uses the "numpy" and "skimage. morphology" modules , as they are dependencies for the function.

- 3 #

- 4 #def planewise_morphology_closing(x_data , structuring_element):

- 5 #

- 6 #Here are the imports:

- 7 #from cellprofiler.library.modules import closing

- 8 #from ._help import HELP_FOR_STREL

- 9 #from cellprofiler_core.module import ImageProcessing

- 10 #from cellprofiler_core.setting import StructuringElement

- 11 #Here are the global variables:

- 12 #Here are the class declarations:

- 13 #class Closing(ImageProcessing):

- 14 # attributes: self.structuring_element ,self.function

- 15 # methods:

- 16 # def create_settings(self):

- 17 # summary: creates settings for the Closing function

- 18 # param: self (unknown) - instance of the Closing function

- 19 # return: None

- 20 # def settings(self):

- 21 # summary: Returns settings from superclass with self. structuring_element added.

- 22 # param: None

- 23 # return: list - __settings__ with self.structuring_element appended.

- 24 # def visible_settings(self):

- 25 # summary: This function returns the visible settings of a Closing object.

- 26 # param: self (Closing) - the Closing object whose settings are to be returned.

- 27 # return: list - the visible settings of the Closing object.

- 28 # def run(self , workspace):

- 29 # summary: Applies morphological closing to an image in a workspace.

- 30 # param: workspace (object) - the workspace containing the image.

- 31 # return: None

- 32 #Here are the additional function declarations:

- 33 #def planewise_morphology_closing(x_data , structuring_element):

- 34 # summary: Applies morphology closing operation to each plane in

- x_data using structuring_element and returns the resulting

- y_data.

- 35 # param: x_data (numpy array) - 3D array containing the input data.

- 36 # param: structuring_element (numpy array) - Structuring element used to perform the closing operation.

- 37 # return: y_data (numpy array) - 3D array containing the result of closure operation on each plane in x_data.

- 38 #Here are the comments and the specs:

- 39 #Write a function named "planewise_morphology_closing" that accepts two parameters , "x_data" and "structuring_element". Inside the function , create a new variable named "y_data" that will hold the result of the morphological closing operation. Loop through each item in "x_data" and perform the same operation on each. Do this by using the "enumerate" function to

get both the index and plane for each item. Use the "closing" function from the "skimage.morphology" module to perform the morphological closing operation on each plane using the " structuring_element" parameter. Finally , return the newly created "y_data" array that holds the result of the operation. Ensure that the function uses the "numpy" and "skimage. morphology" modules , as they are dependencies for the function.

- 40 #def planewise_morphology_closing(x_data , structuring_element):

- 41

#### 5. Necessary Only

- 1 Write a function with the following specs:

- 2 --specs begin here --

- 3 #Write a function named "planewise_morphology_closing" that accepts two parameters , "x_data" and "structuring_element". Inside the function , create a new variable named "y_data" that will hold the result of the morphological closing operation. Loop through each item in "x_data" and perform the same operation on each. Do this by using the "enumerate" function to

get both the index and plane for each item. Use the "closing" function from the "skimage.morphology" module to perform the morphological closing operation on each plane using the " structuring_element" parameter. Finally , return the newly created "y_data" array that holds the result of the operation. Ensure that the function uses the "numpy" and "skimage. morphology" modules , as they are dependencies for the function.

- 4 param: x_data (numpy array) - 3D array containing the input data.

- 5 param: structuring_element (numpy array) - Structuring element used to perform the closing operation.

- 6 return: y_data (numpy array) - 3D array containing the result of closure operation on each plane in x_data.

- 7 --specs end here --

- 8 Note the function will be embedded in the following context

- 9 --context begins here --

- 10 import numpy

- 11 import skimage.morphology

- 12 import os

- 13 <<insert solution here >>

- 14 def main():

- 15 numpy.random.seed(<|int;range=0,100|>)

- 16 pixels = numpy.random.randint(2, size=(10, 10, 10))

- 17 structuring_element = skimage.morphology.square(3)

- 18 print(planewise_morphology_closing(pixels , structuring_element ))

- 19 if __name__ == "__main__":

- 20 main()

- 21 --context ends here --

- 22 Make sure to only generate the function and not any of the context

. Make sure you are generating valid , runnable code. Begin your solution with:

- 23 def planewise_morphology_closing(x_data , structuring_element):

- 24

- I MODEL ABLATION STUDY

For the ablation study, we considered two representative functions, one for Java and one for Python, and determined how the various models performed in generating the following functions. The Java function we selected is as follows:

- 1 public static String getReverseComplementedSequence(SAMRecord record , int startIndex , int length) {

- 2 if (startIndex < 0) {

- 3 startIndex = record.getReadLength() + startIndex;

- 4 }

- 5 byte[] rangeBytes = Arrays.copyOfRange(record.getReadBases(), startIndex , startIndex + length);

- 6 SequenceUtil.reverseComplement(rangeBytes);

- 7 return new String(rangeBytes);

- 8 }

The (abridged) context to which the function getReverseComplementedSequence had access is as follows:

- 1 import org.apache.commons.math3.distribution.HypergeometricDistribution;

- 2 import java.text.DecimalFormat;

- 3 import java.util.*;

- 4 import java.util.function.Function;

- 5 import htsjdk.samtools.util.SequenceUtil;

- 6 import java.util.Objects;

- 7

- 8 class SAMRecord {

- 9 public byte[] arr2;

- 10

- 11 public SAMRecord(byte[] arr) {

- 12 arr2 = arr;

- 13 }

- 14

- 15 public byte[] getReadBases() {

- 16 return arr2;

- 17 }

- 18

- 19 public int getReadLength() {

- 20 return arr2.length;

- 21 }

- 22 }

- 23

- 24 final class Utils {

- 25 /**

- 26 * Method returns reverse complemented sequence for the part of the record. Can work with 3’ and 5’ ends

- 27 * (if start index < 0, then it will found the index in the end of sequence by adding the length of record).

- 28 * @param record read from SAM file to process

- 29 * @param startIndex index where start the sequence

- 30 * @param length length of pert of sequence

- 31 * @return reverse complemented part of record

- 32 */

- 33 <<insert solution here >>

- 34

- 35 public static String reverse(String string) {

- 36 return new StringBuffer(string).reverse().toString();

- 37 }

- 38

- 39 public static String complement(String string) {

- 40 final byte[] bases = htsjdk.samtools.util.StringUtil. stringToBytes(string);

- 41 complement(bases);

- 42 return htsjdk.samtools.util.StringUtil.bytesToString(bases);

- 43 }

- 44 }

We chose this as the representative Java function because it is highly related to bioinformatics, and the solution will require an understanding of the context surrounding it, namely the custom SAMRecord class. Furthermore, the model will have to understand that there are many helpful utility functions available in the Utils class that can be used to assist in completing the function, such as the reverse and complement functions.

The Python function that we selected for this ablation study is as follows:

- 1 def absolute_clonal(cnarr , ploidy , purity , is_reference_male , is_sample_female

- 2 ):

- 3 """Calculate absolute copy number values from segment or bin log2 ratios."""

- 4 absolutes = np.zeros(len(cnarr), dtype=np.float_)

- 5 for i, row in enumerate(cnarr):

- 6 ref_copies , expect_copies = _reference_expect_copies(row. chromosome ,

- 7 ploidy , is_sample_female , is_reference_male)

- 8 absolutes[i] = _log2_ratio_to_absolute(row.log2 , ref_copies ,

- 9 expect_copies , purity)

- 10 return absolutes The (abridged) context to which the function absolute_clonal had access is as follows:

- 1 import random

- 2 import hashlib

- 3 import numpy as np

- 4 import os

- 5

- 6 def _reference_expect_copies(chrom , ploidy , is_sample_female , is_reference_male

- 7 ):

- 8 """Determine the number copies of a chromosome expected and in reference.

- 9 For sex chromosomes , these values may not be the same ploidy as the

- 10 autosomes. The "reference" number is the chromosome ’s ploidy in the

- 11 CNVkit reference , while "expect" is the chromosome ’s neutral ploidy in the

- 12 given sample , based on the specified sex of each. E.g., given a female

- 13 sample and a male reference , on chromosome X the "reference" value is 1 but

- 14 "expect" is 2.

- 15 Returns

- 16 -------

- 17 tuple

- 18 A pair of integers: number of copies in the reference , and expected in

- 19 the sample.

- 20 """

- 21 chrom = chrom.lower()

- 22 if chrom in [’chrx ’, ’x’]:

- 23 ref_copies = ploidy // 2 if is_reference_male else ploidy

- 24 exp_copies = ploidy if is_sample_female else ploidy // 2

- 25 elif chrom in [’chry ’, ’y’]:

- 26 ref_copies = ploidy // 2

- 27 exp_copies = 0 if is_sample_female else ploidy // 2

- 28 else:

- 29 ref_copies = exp_copies = ploidy

- 30 return ref_copies , exp_copies

- 31

- 32 def _log2_ratio_to_absolute_pure(log2_ratio , ref_copies):

- 33 """Transform a log2 ratio to absolute linear scale (for a pure sample ).

- 34 Purity adjustment is skipped. This is appropriate if the sample is germline

- 35 or if scaling for tumor heterogeneity was done beforehand.

- 36 .. math :: n = r*2^v

- 37 """

- 38 ncopies = ref_copies * 2 ** log2_ratio

- 39 return ncopies

- 40

- 41 def _log2_ratio_to_absolute(log2_ratio , ref_copies , expect_copies , purity

=None

- 42 ):

- 43 """Transform a log2 ratio to absolute linear scale (for an impure sample).

- 44 Does not round to an integer absolute value here.

- 45 Math::

- 46 log2_ratio = log2(ncopies / ploidy)

- 47 2^log2_ratio = ncopies / ploidy

- 48 ncopies = ploidy * 2^log2_ratio

- 49 With rescaling for purity::

- 50 let v = log2 ratio value , p = tumor purity ,

- 51 r = reference ploidy , x = expected ploidy ,

- 52 n = tumor ploidy ("ncopies" above);

- 53 v = log_2(p*n/r + (1-p)*x/r)

- 54 2^v = p*n/r + (1-p)*x/r

- 55 n*p/r = 2^v - (1-p)*x/r

- 56 n = (r*2^v - x*(1-p)) / p

- 57 If purity adjustment is skipped (p=1; e.g. if germline or if scaling for

- 58 heterogeneity was done beforehand)::

- 59 n = r*2^v

- 60 """

- 61 if purity and purity < 1.0:

- 62 ncopies = (ref_copies * 2 ** log2_ratio - expect_copies * (1 purity)

- 63 ) / purity

- 64 else:

- 65 ncopies = _log2_ratio_to_absolute_pure(log2_ratio , ref_copies)

- 66 return ncopies

- 67

- 68 <<insert solution here >>

We chose this as the representative Python function because, like the Java function we selected, it is highly related to bioinformatics, and the solution will require an understanding of the context surrounding it, namely the custom _reference_expect_copies and _log2_ratio_to_absolute functions. Additionally, solving the problem will require the use of certain general imports, such as the numpy package.

- 1. InCoder: Similar to many other models, InCoder exhibits very poor performance in Java, mainly because it was trained and fine-tuned on a Python corpus. Therefore, InCoder achieves 0% Pass@K rate for the Java prompts. The main issue that InCoder encounters is its struggle to understand the objective presented in the prompt for Java tasks, leading to failure to generate the intended function. Instead, it appears to regurgitate the prompt, inevitably resulting in failure. For instance, here is a sample InCoder output for the function getReverseComplementedSequence:

- 1 # Do not write any comments in your code.

- 2 #This is in java

- 3 #Write a function called "getReverseComplementedSequence" that takes in a SAMRecord object , an integer called "startIndex", and an integer called "length". The function should check if startIndex is less than 0, and if it is, set startIndex to the

SAMRecord object ’s read length plus startIndex. The function should then create a byte array called "rangeBytes" that is a copy of a portion of the SAMRecord object ’s read bases starting

from the startIndex and ending at startIndex + length. The function should then call SequenceUtil.reverseComplement on rangeBytes , and finally return a new String created from the rangeBytes byte array.

- 4 #Here are the imports:

- 5 #import org.apache.commons.math3.distribution. HypergeometricDistribution;

- 6 #import java.text.DecimalFormat;

- 7 #import java.util.*;

- 8 #import java.util.function.Function;

- 9 #import htsjdk.samtools.util.SequenceUtil;

- 10 #import java.util.Objects;

- 11 #Here are the class declarations:

- 12 #class SAMRecord {

- 13 # public byte[] arr2;

- 14 # public SAMRecord(byte[] arr) {

- 15 # arr2 = arr;

- 16 # }

- 17 # public byte[] getReadBases() {

- 18 # return arr2;

- 19 # }

- 20 # public int getReadLength() {

- 21 # return arr2.length;

- 22 # }

- 23 #}

- 24 #class Region {

- 25 # /**

- 26 # * Chromosome name

- 27 # */

- 28 # public final String chr;

- 29 # /**

- 30 # * Region start position

- 31 # */

- 32 # public final int start;

- 33 # /**

- 34 # * Region end position

- 35 # */

- 36

Although InCoder performs comparatively better on Python, it still struggles greatly to understand the objective of the prompt and to generate the function accurately. Instead, for Python code, the InCoder models appear to simply regurgitate the summary of the Python function provided in the prompt. For instance, here is a sample InCoder output for the function absolute_clonal:

- 1 # summary: Returns the number of copies of a chromosome in reference and in sample.

- 2 # param: chrom (string) - the chromosome to be evaluated

- 3 # param: ploidy (int) - the ploidy of the chromosome

- 4 # param: is_reference_male (bool) - indicates if the reference is male or not

- 5 # return: tuple - a pair of integers indicating the number of copies in the reference and in the sample.

- 6 #def _log2_ratio_to_absolute(cnarr , ref_copies , expect_copies , purity):

- 7 # summary: Converts log2 ratio values into absolute copy number values.

- 8 # param: cnarr (array -like) - Copy number segments or bins

- 9 # param: ref_copies (array -like) - An array of integers indicating the number of copies in the reference for each segment or bin.

- 10 # param: expect_copies (array -like) - An array of integers indicating the number of copies in the expected for each segment or bin.

- 11 # param: purity (float) - estimated tumor pur

- 12

Overall, it seems that InCoder struggles to understand longer and more complex prompts, and it struggles to understand the objective of these prompts. Often it defaults to regurgitating parts of the prompt it has encountered, without generating the desired function.

- 2. SantaCoder: Similar to many other models, SantaCoder also exhibits poor Java performance, primarily because it was trained and fine-tuned on a Python corpus. Therefore, SantaCoder achieves 0% Pass@K rate for the Java prompts. The main issue that SantaCoder encounters is its struggle to understand the objective presented in the prompt and to generate the full function. Instead, it often only generates part of the function. This limitation may also be due to the smaller context limit of SantaCoder, preventing it from fully generating the functions. For instance, here is a sample SantaCoder output for the function getReverseComplementSequence:

- 1 # double x = perc;

- 2 # int a = (int) x;

- 3 # int b = (int) (x - a);

- 4 # int c = (int) (b - a);

- 5 # if(c==0){

- 6 # return 0;

- 7 # }else if(c==1){

- 8 # return 1;

- 9 # }else{

- 10 # return 2;

- 11 # }

- 12 # //return getColor(value , perc);

- 13

- 14 # int c1 = (int) ((perc - a)/b);

- 15 # int c2 = (int) ((perc - a - b)/a);

- 16 # return 1 + c1 * 3 + c2 * 5;

- 17

- 18 # int a1 = (int) (c1 / 2);

- 19 # int a2 = (int) ((c1 - a1) / 2);

- 20 # int a3 = (int) (c2 / 2);

- 21 # int a4 = (int) ((c2 - a3) / 2);

- 22 # int a5 = (int) (c1 +

- 23

Although SantaCoder performs comparatively better on Python, especially considering its small model size, it still struggles greatly to generate many of the functions. This is often due to the small context size of the SantaCoder model. Thus, for more complicated functions, SantaCoder is unable to generate the model. Overall, it seems that SantaCoder struggles to understand the longer and more complex prompts, and it often struggles to understand the objective of the prompts. Furthermore, the small context limit of SantaCoder restricts its ability to generate more complex and context-rich code.

- 3. StarCoder: Similar to many of the other models, StarCoder exhibits very poor Java performance, primarily because it was trained and fine-tuned on a Java corpus. Therefore, StarCoder achieves 0% Pass@K rate for the Java prompts. However, unlike some of the other models, it seems to understand the objective of the prompt and uses context appropriately. For instance, it correctly uses the reverseComplement helper function when generating the getReverseComplementedSequence function, as demonstrated below:

- 1 def getReverseComplementedSequence(samRecord , startIndex , length):

- 2 if startIndex < 0:

- 3 startIndex = samRecord.getReadLength() + startIndex

- 4 rangeBytes = samRecord.getReadBases().getBytes()[startIndex : startIndex + length]

- 5 reverseComplemented = SequenceUtil.reverseComplement( rangeBytes)

- 6 return reverseComplemented.toString()

- 7

However, StarCoder, being fine-tuned on Python, appears to have a hard time generating proper Java syntax, and instead generates a Python function. Although StarCoder performs comparatively better on Python, it still struggles greatly to generate many of the functions. Python seems to understand the objective in the prompt and generates proper syntax, however, some of its logic is incorrect, resulting in some failed tests. For instance, here is a sample StarCoder output for the function absolute_clonal:

- 1 #Write your function here!

- 2 def absolute_clonal(cnarr , ploidy , purity , is_reference_male , is_sample_female):

- 3 absolutes = np.zeros(len(cnarr), dtype=np.float_)

- 4 for i, row in enumerate(cnarr):

- 5 chromosome , log2_value = row

- 6 ref_copies , expect_copies = _reference_expect_copies( chromosome , ploidy , is_sample_female , is_reference_male)

- 7 absolute_value = _log2_ratio_to_absolute(log2_value , ref_copies , expect_copies , purity)

- 8 absolutes[i] = absolute_value

- 9 return absolutes

- 10

- 4. StarCoder+: Unlike many of the other models, StarCoder+ has a non-zero Pass@K performance on Java. This is likely because StarCoder+ is trained and fine-tuned on a larger corpus that consists of more languages than just Python. It seems to understand the objective of the prompt and uses the context appropriately, for instance using the reverseComplement helper function correctly when generating the getReverseComplementedSequence function, as demonstrated below:

- 1 import htsjdk.samtools.util.SequenceUtil;

- 2 public class getReverseComplementedSequence

- 3 {

- 4 public static String getReverseComplementedSequence(SAMRecord samRecord , int startIndex , int length)

- 5 {

- 6 if(startIndex < 0)

- 7 {

- 8 startIndex = samRecord.getReadLength() + startIndex;

- 9 }

- 10 byte[] rangeBytes = new byte[length];

- 11 samRecord.getReadBases(rangeBytes , startIndex , length);

- 12 byte[] rcBytes = SequenceUtil.reverseComplement(rangeBytes);

- 13 return new String(rcBytes);

- 14 }

- 15 }

- 16

- 5. CodeGen: The performance of CodeGen1 was not great. This could be due to a variety of factors, including the fact that CodeGen models are not intended for full multi-line function generation, or the fact that the dataset used to train CodeGen models does not contain much Java code. This is demonstrated in some of the generations, as CodeGen ended up generating Python code, despite us mentioning that this prompt is in Java and using Java code in the context. Sometimes, it also generates the comment again, such as below:

- 1 #Note: This is not a string method.

- 2

- 6. CodeGen2: CodeGen2 outperforms CodeGen as expected in our normal test cases; however, unexpectedly, it is slightly outperformed in the Rosalind benchmark.

- 7. GPT-3.5: GPT clearly performs the best out of all models tested, in both Java and Python and across all prompt types. The version we used on Azure had an 8k context size, so it had an advantage over some of the models by being able to understand more of the context. Furthermore, the fact that some of the function descriptions were generated by GPT might mean that it had an easier time generating code from its own description. Furthermore, given the size of the model, it is possible that some of the functions were memorized, although we tried to combat this by slightly changing the function signature in some cases. Here is an example of the GPT output: [Note the entire example has changed.]

- 1 import numpy as np

- 2

- 3 def absolute_clonal(cnarr , ploidy , purity , is_reference_male , is_sample_female):

- 4 absolutes = np.zeros(len(cnarr), dtype=np.float_)

- 5

- 6 for i, row in enumerate(cnarr):

- 7 chromosome = row.chromosome

- 8 ref_copies , expect_copies = _reference_expect_copies( chromosome , ploidy , is_sample_female , is_reference_male)

- 9 abs_copy = _log2_ratio_to_absolute(row.log2 , ref_copies , expect_copies , purity)

- 10 absolutes[i] = abs_copy

- 11

- 12 return absolutes

- 13

It is also important to note that in every prompt, GPT-3.5 generated usable, cleaner code without comments as directed. Therefore, most test cases were run without syntax errors but mostly with logic errors. Below is an example of a longer sample:

- 1 def FindLocalPeaks(signal):

- 2 peakLoc = []

- 3 maxSuspect = None

- 4 minSuspect = None

- 5 for i in range(1, len(signal) -1):

- 6 if signal[i] > 0:

- 7 if signal[i] > signal[i-1] and signal[i] == signal[i

+1]:

- 8 maxSuspect = i

- 9 elif signal[i] == signal[i-1] and signal[i] > signal[i

+1]:

- 10 if maxSuspect is not None:

- 11 peakLoc.append(maxSuspect)

- 12 maxSuspect = None

- 13 elif signal[i] > signal[i-1] and signal[i] > signal[i

+1]:

- 14 peakLoc.append(i)

- 15 elif signal[i] < 0:

- 16 if signal[i] < signal[i-1] and signal[i] == signal[i

+1]:

- 17 minSuspect = i

- 18 elif signal[i] == signal[i-1] and signal[i] < signal[i

+1]:

- 19 if minSuspect is not None:

- 20 peakLoc.append(minSuspect)

- 21 minSuspect = None

- 22 elif signal[i] < signal[i-1] and signal[i] < signal[i

+1]:

- 23 peakLoc.append(i)

- 24 return np.array(peakLoc , dtype=int)

- 25

Next, we performed a horizontal analysis on a single prompt to compare the outputs of each of the models with each other, aiming to understand the differences between the models and gain insight

into what parts of the prompts the various models seem to understand. We applied the following problem for the horizontal analysis:

- 1 @property

- 2 def outlines(self):

- 3 """Get a mask of all the points on the border of objects"""

- 4 if self._outlines is None:

- 5 for i, labels in enumerate(self.labels):

- 6 if i == 0:

- 7 self._outlines = centrosome.outline.outline(labels) != 0

- 8 else:

- 9 self._outlines |= centrosome.outline.outline(labels) != 0

- 10 if self.line_width is not None and self.line_width > 1:

- 11 hw = float(self.line_width) / 2

- 12 d = scipy.ndimage.distance_transform_edt(~self._outlines)

- 13 dti, dtj = numpy.where((d < hw + 0.5) & ~self._outlines)

- 14 self._outlines = self._outlines.astype(numpy.float32)

- 15 self._outlines[dti, dtj] = numpy.minimum(1, hw + 0.5 - d[dti , dtj])

- 16 return self._outlines.astype(numpy.float32) The reasons we chose this function for the horizontal analysis are the following:

- 1. It is a Python function, and all of the models generally seem to be fine-tuned on Python code and thus perform better on Python code; therefore, it would be a fairer comparison to use a Python function for this analysis.
- 2. Furthermore, this is also a relatively tricky function to generate as it has many moving parts and relies on many different levels of understanding of the context. It is important to note that it has a @property decorator and is a function that is part of a class; thus, the models must understand this and generate the @property decorator accordingly. Furthermore, the model must understand and correctly use the self._outlines, self._line_width, and self.labels instance variables. Finally, the model must understand and correctly use the highly bioinformatics-specific external library, centrosome.

The prompt that we used for the horizontal analysis is the following “Summary Only” prompt:

- 1 #This is in python

- 2 #Write a function called "get_border_mask" that takes in a class object as an argument and returns a binary mask of all the points on the border of objects. The function should check if the instance variable

"_outlines" is None and if it is, then use the labels attribute of the class object to generate the outlines mask using "centrosome. outline.outline(labels)" method. If the "_outlines" variable is not None , then it should iterate over "centrosome.outline.outline(labels) " for each i, and update the "_outlines" variable using the bitwise OR operator with the new mask. If the line_width attribute of the class object is not None and greater than 1, the function should calculate the distance transform using the "~self._outlines" mask and

compute the binary mask for borders with the given line width (hw) using numpy.where(). Finally , the function should return the mask as a float32 numpy array. Make sure to import the necessary dependencies

including "centrosome.outline" and "numpy" and also initialize the " _outlines" variable in the class.

- 3 #

- 4 #The function is located in the class OutlinesMixin

- 5 #

- 6 #@property

- 7 #def get_border_mask(self):

We chose the “Summary Only” format of the prompt because generally, the models performed best when the prompt was written in the “Summary Only” format. The outputs of the various models for this prompt are as follows:

#### 1. InCoder:

- 1 def get_border_mask(self):

- 2 #If outlines is none , generate outlines using centrosome. outline.outline(labels) method and set outlines to the generated mask.

- 3 if self.outlines is None:

- 4 self.outlines = self.outline.outline(labels=self.labels)

- 5

- 6 # Get the mask using numpy

- 7 border_mask = np.where(self.outlines > 0, 1, 0 )

- 8

- 9 # Check if line width is greater than 1

- 10 if self.line_width > 1:

- 11 # Get distance transform

- 12 dt = np.abs(self.outlines - border_mask) * 255

- 13 dt[dt==0] = 255

- 14

- 15 # Use np.where to create border mask

- 16 border_mask = np.where(

- 17 dt > 0,

- 18 1,

- 19 0

- 20 )

- 21

- 22 # Return border mask as float32 numpy array

- 23 return border_mask.astype(np.float32)

- 24

#### 2. SantaCoder:

- 1 def get_border_mask(self):

- 2 if self._outlines is None:

- 3 self._outlines = centrosome.outline.outline(self.labels)

- 4 outlines_mask = np.where(self._outlines , self._outlines , self. labels)

- 5 #outlines_mask = np.where(self._outlines , self._outlines , self

._outlines)

- 6 #outlines_mask = np.where(outlines_mask > 0, outlines_mask , 0)

- 7 return outlines_mask

- 8

#### 3. StarCoder:

- 1 def get_border_mask(self):

- 2 if self._outlines is None:

- 3 self._outlines = centrosome.outline.outline(self.labels)

- 4 else:

- 5 for i in range(self._outlines.shape[0]):

- 6 self._outlines[i] = self._outlines[i] | centrosome. outline.outline(self.labels[i])

- 7 if self.line_width is not None and self.line_width > 1:

- 8 dist_trans = ndimage.distance_transform_edt(~self. _outlines)

- 9 border_mask = np.where(dist_trans <= self.line_width , 1, 0)

- 10 else:

- 11 border_mask = self._outlines

- 12 return border_mask.astype(np.float32)

- 13

#### 4. StarCoder+:

- 1 @property

- 2 def outlines(self):

- 3 if not hasattr(self , ’_outlines ’):

- 4 self._outlines = centrosome.outline.outline(self.labels)

- 5 return self._outlines

- 6 else:

- 7 for i in range(len(self.labels)):

- 8 self._outlines[i] = centrosome.outline.outline(self. labels)

- 9 self._outlines = self._outlines | self._outlines

- 10 return self._outlines

- 11

#### 5. InstructCodeT5+:

- 1 def get_border_mask(self ,labels):

- 2

- 3 if self._outlines is None:

- 4 self._outlines = outline(labels)

- 5

- 6 mask = self._outlines

- 7

- 8 if self.line_width is not None:

- 9 hw = self.line_width

- 10

- 11 if self.line_width >1:

- 12

#### 6. CodeGen:

- 1 # Do not write any comments in your code.

- 2 #This is in python

- 3 #Write a function called "get_border_mask" that takes in a class object as an argument and returns a binary mask of all the points on the border of objects. The function should check if the instance variable "_outlines" is None and if it is, then use the labels attribute of the class object to generate the outlines mask using "centrosome.outline.outline(labels)" method

. If the "_outlines" variable is not None , then it should iterate over "centrosome.outline.outline(labels)" for each i, and update the "_outlines" variable using the bitwise OR operator with the new mask. If the line_width attribute of the class object is not None and greater than 1, the function should calculate the distance transform using the "~self. _outlines" mask and compute the binary mask for borders with the given line width (hw) using numpy.where(). Finally , the function should return the mask as a float32 numpy array. Make sure to import the necessary dependencies including "centrosome

.outline" and "numpy" and also initialize the "_outlines" variable in the class.

- 4 #

- 5 #The function is located in the class OutlinesMixin

- 6 #

- 7 #@property

- 8 #def get_border_mask(self):

- 9

#### 7. CodeGen2:

- 1 def get_border_mask(self):

- 2 """

- 3 Computes the mask for the outlines of a morphology.

- 4

- 5 Parameters

- 6 ----------

- 7 self : morphology object

- 8

- 9 Returns

- 10 -------

- 11 border_mask :

- 12

#### 8. GPT-3.5:

- 1 @property

- 2 def get_border_mask(self):

- 3 if self._outlines is None:

- 4 self._outlines = outline(self.labels)

- 5 else:

- 6 for i in range(1, np.max(self.labels)):

- 7 self._outlines |= outline(self.labels == i)

- 8

- 9 if self.line_width is not None and self.line_width > 1:

- 10 dt = np.array(self._outlines , dtype=np.uint8)

- 11 dt = np.where(dt < 0.5, 1, 0)

- 12 dt = ndimage.distance_transform_edt(dt)

- 13 dt = (dt < self.line_width).astype(float)

- 14 return dt

- 15 else:

- 16 return np.array(self._outlines , dtype=np.float32)

- 17

#### 9. GPT-4:

- 1 def outlines(self):

- 2 if self._outlines is None:

- 3 self._outlines = centrosome.outline.outline(self.labels)

- 4 else:

- 5 for i in range(len(self.labels)):

- 6 self._outlines |= centrosome.outline.outline(self. labels[i])

- 7

- 8 if self.line_width is not None and self.line_width > 1:

- 9 distances = scipy.ndimage.distance_transform_edt(~self. _outlines)

- 10 hw = self.line_width / 2

- 11 self._outlines = numpy.where(distances <= hw, 1, 0)

- 12

- 13 return self._outlines.astype(numpy.float32)

- 14

In comparing the outputs of the various models to the chosen prompt, it is clear that there are various levels of understanding of the prompt among the models. By far, the worst-performing models are the InstructCodeT5+, CodeGen, and CodeGen2 models. All of these models failed to generate a full function. This is likely due to these models being trained to perform one-line completions rather than full-function generation. Among the three, InstructCodeT5+ performed the best as it at least generated part of a function and showed an understanding of the self._outlines instance variable, unlike both the CodeGen and CodeGen2 models. However, InstructCodeT5+ also got the function signature wrong, indicating a lack of understanding of the function’s structure in the prompt. The CodeGen2 model did not generate any function body; however, it did correctly generate the function signature and part of a useful docstring for the function. Among these three models, the original CodeGen model performed the worst by far, as it simply regurgitated the prompt, not generating any new code.

For the remaining five models, there were different levels of understanding. The two models that demonstrated the deepest levels of understanding of the prompt were the StarCoder+ and GPT-3.5 models, as they were the only two models to recognize that the get_border_mask function has a @property function decorator. Furthermore, they both showed some understanding of the external function calls that were necessary to complete the function, with StarCoder+ calling the centrosome.outline.outline function correctly and GPT-3.5 using the ndimage.distance_transform_edt function, although not completely correctly, as the correct function call would be scipy.ndimage.distance_transform_edt. However, the logic that each of these models uses to perform the get_border_mask function is correct, with GPT-3.5 getting closer to the intended logic. For instance, GPT-3.5 includes the check self.line_width is None.

The remaining three models not yet mentioned in this analysis are InCoder, SantaCoder, and StarCoder. These three models generated complete functions, unlike InstructCodeT5+, CodeGen, and CodeGen2; however, they did not include the @property function decorator, unlike the StarCoder+ and GPT-3.5 models. Among these three “middle-of-the-pack" models, StarCoder performs especially well, as it demonstrates an understanding of the three necessary instance variables, namely self._outlines, self.labels, and self.line_width, and uses both the centrosome.outline.outline and ndimage.distance_transform_edt external function calls. However, like GPT-3.5, it does not quite make the ndimage external call correctly. Furthermore, the structural logic of StarCoder’s code is similar to the golden code, on the same level as GPT-3.5. As for InCoder and SantaCoder, although they both generated full functions, their functions were somewhat off in their structural logic. SantaCoder performs decently, using the centrosome.outline.outline external package correctly; however, the function it generates is overly simple and does not meet the prompt description guidelines. As for InCoder, it uses a nonexistent self.outlines instance variable instead of the intended self._outlines instance variable. Furthermore, it calls the nonexistent function self.outline.outline instead of the intended centrosome.outline.outline function. By contrast, InCoder writes a more involved function than SantaCoder, which more closely mirrors the guidelines provided by the prompt.

- J PROMPT LENGTH STUDY

In these scatterplots, each dot represents a single generation from the model, with a corresponding prompt length as input and a Pass@K value as the result. Note that there are many points with a Pass@K of 0 along the bottom; this is because most test cases did not pass, as indicated by the main results.

gpt4 Performance Tokens v. Pass@K Average

1.0

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.8

0.6

PassatK

0.4

0.2

0.0

0 200 400 600 800 1000 1200 Tokens of Prompt

Figure 5: Scatterplots of the correlation of Pass@K and tokens of prompt. Statistics for GPT-4.

starcoderplus Performance Tokens v. Pass@K Average

1.0

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.8

0.6

PassatK

0.4

0.2

0.0

0 1000 2000 3000 4000 5000 Tokens of Prompt

Figure 6: Statistics for Starcoderplus

incoder-6B Performance Tokens v. Pass@K Average

1.0

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

0.8

0.6

PassatK

0.4

0.2

0.0

0 250 500 750 1000 1250 1500 1750 2000 Tokens of Prompt

Figure 7: Statistics for Incoder

codegen-6B-mono Performance Tokens v. Pass@K Average

1.0

| | |
|---|---|
| | |
| | |
| | |
| | |

0.8

0.6

PassatK

0.4

0.2

0.0

0 250 500 750 1000 1250 1500 1750 Tokens of Prompt

Figure 8: Statistics for CodeGen

codegen2-7B Performance Tokens v. Pass@K Average

1.0

| | |
|---|---|
| | |
| | |
| | |
| | |

0.8

0.6

PassatK

0.4

0.2

0.0

0 200 400 600 800 1000 1200 1400 1600 Tokens of Prompt

Figure 9: Statistics for CodeGen2

instructcodet5p-16b Performance Tokens v. Pass@K Average

1.0

| | |
|---|---|
| | |
| | |
| | |
| | |

0.8

0.6

PassatK

0.4

0.2

0.0

0 250 500 750 1000 1250 1500 1750 Tokens of Prompt

Figure 10: Statistics for InstructCodeT5+

starcoder Performance Tokens v. Pass@K Average

1.0

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

0.8

0.6

PassatK

0.4

0.2

0.0

0 1000 2000 3000 4000 5000 6000 7000 Tokens of Prompt

Figure 11: Statistics for Starcoder

santacoder Performance Tokens v. Pass@K Average

1.0

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

0.8

0.6

PassatK

0.4

0.2

0.0

0 250 500 750 1000 1250 1500 1750 2000 Tokens of Prompt

Figure 12: Statistics for Santacoder

gpt Performance Tokens v. Pass@K Average

1.0

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.8

0.6

PassatK

0.4

0.2

0.0

0 200 400 600 800 1000 1200 Tokens of Prompt

Figure 13: Statistics for GPT-3.5

- K TESTING FRAMEWORK An outline of our testing frame is as follows:

- 1. Preprocessing: Each generated output is cleaned and minor corrections are applied in accordance with the above steps. We append a call to the main() function at the end of the file.
- 2. Container Preparation: The output from LLMs, alongside the environment files and sample test cases, are copied into the container.
- 3. Fuzz Execution: The test cases are executed with a memory cap of 7GB and a time limit of 60 seconds. A test is considered passed if the actual output precisely matches the expected output, for all cases tested.
- 4. Analysis: We gather and analyze the results from these executions.

Through this approach, we provide a secure and efficient testing framework, promising robustness in the evaluation of generated code.

K.1 MOTIVATION BEHIND FUZZ TESTING

We decided to utilize concepts from fuzz testing, as fuzz testing is widely used in the industry to capture bugs, crashes, security vulnerabilities, etc. in functions. However, in these cases, they do not have access to a "correct" version of the function; instead, they are merely creating inputs to intentionally try to crash the program, find out of bounds memory accesses, etc. Our situation is unique because we have the "golden code," or the ground truth version of the code, so given any input, we know the expected output, which is not typically available in fuzz testing frameworks.

Our testing approach shares similarities with the methodology described by ?, where both a reference implementation (analogous to our "golden code") and the test code are provided with parametrically generated input data to ensure identical behavior. However, while their approach is presented in the context of metamorphic testing, our testing methodology can be considered a hybrid of unit testing and fuzz testing.

Given this requirement, and the goal of large-scale prompt generation, we decided to implement our framework. We set out to accomplish two tasks: make the annotation process easier for human editors

and support our feature set that combines both elements from unit testing and elements from fuzz testing. We believe that our resulting pipeline is more intuitive than piecing together other testing frameworks, and it proved to improve efficiency in our annotation process, enabling larger-scale annotation to support the goal of the paper.

Furthermore, note that while handwritten test cases would likely target edge cases of a program (e.g., branch coverage, conditional coverage), the probability of our fuzz testing framework hitting all of the same edge cases is high given 1,000 iterations of randomly generated inputs. This means that we can save a significant amount of time building the dataset, as we only need to write an outline of a test case, letting the framework handle the rest. Therefore, we can think of the framework as thousands of "random unit tests," with a high probability that these unit tests would include handwritten test cases, if we had written them.

In terms of variable generation, we replace the <|var_type;parameter|> syntax with random values for each iteration, for an unlimited number of iterations. These parameters are modifiable. We implemented this system to be flexible, allowing us to target specific scopes for fuzz testing. We can check correctness by substituting the same variables in the original code, and determining if the outputs of the two functions match. This indicates identical functionality with the original code.

Here is an example of random integer and numpy array generation:

- 1 import numpy

- 2 import skimage.morphology

- 3 import os

- 4 <<insert solution here >>

- 5 def main():

- 6 numpy.random.seed(<|int;range=0,100|>)

- 7 labels = numpy.random.randint(2, size=(3, 3))

- 8 diameter = <|int;range=2,10|>

- 9 print(fill_object_holes(labels , diameter))

- 10 if __name__ == "__main__":

- 11 main() Here is an example of random string generation:

- 1 import random

- 2 [IMPORTS REDACTED FOR CONCISENESS]

- 3 import warnings

- 4 from textwrap import wrap

- 5 import string

- 6 import zlib

- 7 import io

- 8 from os.path import isfile

- 9 class GenipeError(Exception):

- 10 pass

- 11 _CHECK_STRING = b’GENIPE INDEX FILE ’

- 12 def dosage_from_probs(homo_probs , hetero_probs , scale=2):

- 13 """Computes dosage from probability matrix (for the minor allele).

- 14 Args:

- 15 homo_probs (numpy.array): the probabilities for the homozygous genotype

- 16 hetero_probs (numpy.array): the probabilities for the heterozygous

- 17 genotype

- 18 scale (int): the scale value

- 19 Returns:

- 20 numpy.array: the dosage computed from the probabilities

- 21 """

- 22 return (homo_probs + (hetero_probs / 2)) * scale

- 23 <<insert solution here >>

- 24 def main():

- 25 np.random.seed(<|int;range=0,100|>)

- 26 prob_matrix = np.random.rand(10, 10)

- 27 a1 = <|string|>

- 28 a2 = <|string|>

- 29 print(maf_dosage_from_probs(prob_matrix , a1, a2))

- 30 if __name__ == "__main__":

- 31 main()

<|int|> denotes an integer. If left without parameters, this will be replaced with a random integer between INT_MIN and INT_MAX before compile time (or in this case, before the Python file is executed). Parameters that can be passed in include range, even/odd, etc. Similarly, for <|string|> this generates a random ASCII string of any type. It can be further narrowed down into ASCII strings only, lowercase only, specific characters only, etc. by passing in the relevant parameters. These random inserted values can be manipulated to become part of a larger data structure, for example, a Numpy array, or a mock Python object.

When these files are executed, we replace «insert solution here» with the golden code in one iteration and the error corrected generated code in a corresponding iteration. The fuzz framework is designed so that the same inputs will be passed to this pair of iterations, meaning that we should get the same output (none of the functions have a non-deterministic component). Therefore, this supports one aspect of the "secure" testing framework, as we have created an environment where everything else is equal, except for the generated/golden code.

L PROMPT LENGTH ABLATION STUDY

We have conducted additional experiments to illustrate the effect of different prompts.

As a result, our study revealed that more detailed text descriptions improve code generation, particularly for Python, but they have less impact on Java. However, irrespective of the level of detail, the core structure of the generated code remains similar. Hence, ambiguity in descriptions will not dramatically influence the outcome.

Specifically, in response to concerns about the ambiguity of the Python text descriptions, we conducted a study to examine the effects of ambiguous prompts on code generation. Using ChatGPT, we generated both “short” and “long” summaries, with the former being more concise and potentially omitting some details, and the latter providing more detail to reduce ambiguity.

For the “short” summaries, we used the following prompt:

1 Please write a clear and succinct prompt in one paragraph directing the user to write the following function. In the prompt make sure to include sufficient details on what the following function does.

For the “long” summaries, we used the following prompt:

1 Please write a highly detailed prompt in one paragraph directing the user

to write the following function. In the one paragraph prompt make sure to include all the details on what the following function does. If you are unsure of some. of the variable and function references make an educated guess.

Passing the above prompts into ChatGPT, we generated “short” and “long” summaries. Then, we used these summaries to construct “short” and “long” prompts.

We calculated Pass@K results on ChatGPT generations for both “short” and “long” prompts. The results are as follows:

Prompt Java Python Pass@1 Pass@5 Pass@10 Pass@20 Pass@1 Pass@5 Pass@10 Pass@20

Short Summary 34.700 42.313 43.000 43.800 25.960 36.371 39.825 42.484 Long Summary 36.800 46.412 49.842 54.000 38.567 52.419 56.439 60.510

Below are examples of “short” and “long” prompts:

- 1 SHORT PROMPT EXAMPLE

- 2 #This is in python

- 3 # Write a Python function called "UnifyLevels" that takes in three parameters: "baseLevel" (a constant array of integers), "addonLevel" (a constant array of integers), and "windowSize" (an integer). The function merges the two lists of breakpoints , but drops addonLevel values that are too close to baseLevel values. The function then returns the merged list of breakpoints as an array of integers. If the "addonLevel" list is empty , the function should return the " baseLevel" as is.

- 4 # Return only the code in the completion. I don ’t want any other comments

. Don’t say "here is your code" or similar remarks.

- 5 # def UnifyLevels(baseLevel , addonLevel , windowSize):

- 1 LONG PROMPT EXAMPLE

- 2 #This is in python

- 3 # Write a function called "UnifyLevels" that takes in three parameters: " baseLevel" (a list of integers), "addonLevel" (a list of integers), and "windowSize" (an integer). The purpose of this function is to merge the two lists of breakpoints (baseLevel and addonLevel), but drop any values from addonLevel that are too close to values in baseLevel. The function should first check if addonLevel is empty. If

it is, the function should simply return baseLevel. Otherwise , the

function should create an empty list called "joinedLevel" and an integer variable called "addon_idx" initialized to 0. The function should then iterate over each element in baseLevel using a for loop. Inside the loop , there should be a while loop that continues as long as addon_idx is less than the length of addonLevel. Within the while loop , the function should retrieve the element at index addon_idx from addonLevel and assign it to a variable called "addon_elem". Next , there should be three cases to handle:

- 4 # - If addon_elem is less than base_elem minus windowSize , the function should append addon_elem to joinedLevel , increment addon_idx by 1, and continue to the next iteration of the while loop.

- 5 # - If addon_elem is between base_elem minus windowSize and base_elem plus windowSize (inclusive), the function should increment addon_idx by 1 and continue to the next iteration of the while loop.

- 6 # - If addon_elem is greater than base_elem plus windowSize , the function should assert that base_elem plus windowSize is less than addon_elem

, break out of the while loop , and move to the next iteration of the for loop.

- 7 After each iteration of the for loop , the function should append base_elem to joinedLevel. Once the for loop is complete , the function

should check if there are any remaining elements in addonLevel starting from addon_idx. If there are , the function should extend joinedLevel with those elements. Finally , the function should return a numpy array containing the sorted elements of joinedLevel , with the

data type set to np.int_.

- 8 # Return only the code in the completion. I don ’t want any other comments

. Don’t say "here is your code" or similar remarks.

- 9 # def UnifyLevels(baseLevel , addonLevel , windowSize):

According to our study, using more detailed prompts improved the performance, especially for Python generations. Python Pass@1 performance notably increased from 26.0% to 38.6% with more detailed prompts. However, the improvement for Java was less pronounced, with only a minor increase from 34.7% to 36.8%. Despite the difference in performance, the structural similarity between code generated from both short and long prompts remained high. Therefore, even if a prompt is ambiguous or trivial, it will not significantly affect the fundamental structure of the generated code. Consider the following short prompt and its generated example:

- 1

- 2 SHORT PROMPT

- 3

- 4

- 5 #This is in python

- 6 # Write a function called "FindLocalPeaks" that takes in a constant array of floats called "signal" as a parameter. The function should find

the locations of local maxima on positive values and local minima on negative values in the "signal" array. The first and last index of the array should not be considered as extrema. The function should iterate through the array and compare each element with its adjacent elements to determine if it is a local peak. If a local peak is found , its index should be stored in an array called "peakLoc". The function should return the "peakLoc" array as the output. Make sure to include the necessary import statement for numpy.

- 7 # Return only the code in the completion. I don ’t want any other comments

. Don’t say "here is your code" or similar remarks.

- 8 # def FindLocalPeaks(signal):

- 1

- 2 SAMPLE GENERATION

- 3 import numpy as np

- 4 def FindLocalPeaks(signal):

- 5 peakLoc = []

- 6 for i in range(1, len(signal) -1):

- 7 if signal[i-1] < signal[i] > signal[i+1] and signal[i] > 0:

- 8 peakLoc.append(i)

- 9 elif signal[i-1] > signal[i] < signal[i+1] and signal[i] < 0:

- 10 peakLoc.append(i)

- 11 return peakLoc Here is the corresponding long prompt and its sample generated example:

- 1 LONG PROMPT

- 2 #This is in python

- 3 # Write a function called "FindLocalPeaks" that takes in one parameter , " signal", which is a constant array of floats. The function is used to

find local maxima on positive values and local minima on negative values. The function excludes the first and last indices of the " signal" array as possible extrema. The function should return an array of integers , "peakLoc", which represents the locations of these extrema in the "signal" array. The function should follow the logic provided in the source code provided below , which includes conditions

for various scenarios when comparing adjacent elements in the " signal" array. When a local maxima or minima is found , its location should be appended to the "peakLoc" array. If you encounter equal values while comparing adjacent elements , there are additional conditions that need to be followed. Remember to import the necessary

modules and define any necessary variables.

- 4 # Return only the code in the completion. I don ’t want any other comments

. Don’t say "here is your code" or similar remarks.

- 5 # def FindLocalPeaks(signal):

- 1

- 2 SAMPLE GENERATION

- 3 def FindLocalPeaks(signal):

- 4 import numpy as np

- 5 peakLoc = []

- 6 for i in range(1, len(signal) -1):

- 7 if signal[i] > 0:

- 8 if signal[i] > signal[i-1] and signal[i] > signal[i+1]:

- 9 peakLoc.append(i)

- 10 elif signal[i] < 0:

- 11 if signal[i] < signal[i-1] and signal[i] < signal[i+1]:

- 12 peakLoc.append(i)

- 13 return np.array(peakLoc)

As can be seen in the above example, using a more detailed prompt (10 lines of description vs. 7 lines of description) does not have an outsized effect on the structure of the sample generations, as the functions generated from the two different prompts are structurally similar. Therefore, the improved performance likely stems from the small details that might be omitted in the concise prompts that are relevant to functional performance.

M OBFUSCATION ABLATION STUDY

We conducted a study using “obfuscated prompts.” The framework for this “obfuscated prompts” ablation study drew inspiration from ??. The prompts used were primarily “short” GPT summaries in their original form, with function or parameter names substituted with placeholder terminology, deliberately adjusted to hinder a model’s ability to memorize portions of the training data.

Our analysis of the results shows that there is a performance decrease with obfuscated prompts compared to their non-obfuscated counterparts. However, this difference doesn’t reach a magnitude justifying attributing ChatGPT’s performance exclusively to the memorization of training data.

Below, we provide an example showing a comparison between an obfuscated prompt with a nonobfuscated prompt.

- 1 SAMPLE ORIGINAL PROMPT

- 2 #This is in python

- 3 # Write a Python function called "unpipe_name" that takes in a string argument called "name" and returns a new string that is the single gene name after removing any duplications and pipe characters. This function is designed to fix the duplicated gene names produced by Picard CalculateHsMetrics , which combines the labels of overlapping intervals by joining all labels with ’|’. If the name only consists of a single gene name , then that name is returned. If there are duplicated gene names and a meaningless target name , such as ’-’ or any name in the IGNORE_GENE_NAMES set , then they are removed. If there are still duplicated gene names , then the longest name is taken

. If there are multiple equally long gene names , then one of them is arbitrarily selected and a warning is logged.

- 4

- 5 # Return only the code in the completion. I don ’t want any other comments

. Don’t say "here is your code" or similar remarks.

- 6 # def unpipe_name(name):

- 1 SAMPLE OBFUSCATED PROMPT

- 2 # This is in python

- 3 # Write a Python function called "FUNCTION" that takes in a string argument called "VAR0" and returns a new string that is the single gene VAR0 after removing any duplications and pipe characters. This function is designed to fix the duplicated gene names produced by Picard CalculateHsMetrics , which combines the labels of overlapping intervals by joining all labels with ’|’. If the VAR0 only consists of a single gene VAR0 , then that VAR0 is returned. If there are duplicated gene names and a meaningless target VAR0 , such as ’-’ or any VAR0 in the IGNORE_GENE_NAMES set , then they are removed. If there are still duplicated gene names , then the longest VAR0 is taken

. If there are multiple equally long gene names , then one of them is arbitrarily selected and a warning is logged.

- 4

- 5 # Return only the code in the completion. I don ’t want any other comments

. Don’t say "here is your code" or similar remarks.

- 6 # def FUNCTION(VAR0):

By replacing specific function names and parameters with generic monikers like FUNCTION and VAR0 in our obfuscated prompts, we sought to limit the potential influence of prior training data memorization in ChatGPT’s code generation. Overall, this process aimed to increase the difficulty for the model to rely on memorization as a primary means of accurate code generation, thus ensuring that the model’s performance is based on interpreting the structure and requirement of the task, rather than recollecting learned patterns.

We carried out Pass@K tests on the obfuscated prompts, and the results are displayed in table 1.

Our experiment shows that obfuscation of the prompts has an impact on code generation performance, but not to the extent that would suggest memorization is the primary driver. Obfuscated prompts did underperform compared to non-obfuscated ones, with a drop of over 10% in Pass@1. While a

Prompt Java Python Pass@1 Pass@5 Pass@10 Pass@20 Pass@1 Pass@5 Pass@10 Pass@20

Short Summary 34.700 42.313 43.000 43.800 25.960 36.371 39.825 42.484 Obfuscated Summary 30.700 40.684 44.250 47.900 22.771 32.733 35.872 38.790

Table 1: Performance on obfuscated prompts

10% difference is noteworthy, it also means the model still generated code with acceptable accuracy compared to the non-obfuscated ones, indicating a strong ability to understand and interpret the task rather than relying solely on memorization.

Here is an example that further shows minimal data leakage, along with more details on the generated outputs:

- 1 SHORT PROMPT

- 2 #This is in python

- 3 # Write a function called ‘_neighbors ‘ that takes in a two -or-three dimensional image and returns the number of neighbor pixels for each pixel of the image. The function should first pad the image with zeros , then create a mask where any pixel value greater than zero is considered a neighbor. Next , it should compute the response using a uniform filter and multiply it by a scaling factor. The labels are then computed by multiplying the response with the mask and slicing the padded image. Finally , the function should return the labels as an array of unsigned integers.

- 4 # Return only the code in the completion. I don ’t want any other comments

. Don’t say "here is your code" or similar remarks.

- 5 # def _neighbors(image):

- 1

- 2

- 3 SAMPLE GENERATION with ORIGINAL PROMPT

- 4

- 5

- 6 def _neighbors(image):

- 7 from skimage.filters import uniform_filter

- 8 from scipy import ndimage

- 9 scaling_factor = 10

- 10 mask = (image > 0).astype(int)

- 11 padded_image = np.pad(image , ((1,1), (1,1), (0,0)), ’constant ’, constant_values=0)

- 12 response = uniform_filter(padded_image , size=3)

- 13 labels = (response * scaling_factor * mask)[1:-1, 1:-1, :]

- 14 labels = ndimage.label(labels)[0]

- 15 return labels.astype(np.uint16)

- 16

- 17

- 1 SAMPLE OBFUSCATED PROMPT

- 2

- 3

- 4 # This is in python

- 5 # Write a function called ‘FUNCTION ‘ that takes in a two -or-three dimensional VAR0 and returns the number of neighbor pixels for each pixel of the VAR0. The function should first pad the VAR0 with zeros ,

then create a mask where any pixel value greater than zero is considered a neighbor. Next , it should compute the response using a uniform filter and multiply it by a scaling factor. The labels are then computed by multiplying the response with the mask and slicing the padded VAR0. Finally , the function should return the labels as an

array of unsigned integers.

- 6 # Return only the code in the completion. I don ’t want any other comments

. Don’t say "here is your code" or similar remarks.

- 7 # def FUNCTION(VAR0):

- 8

- 9

- 10

- 1

- 2 SAMPLE GENERATION with OBFUSCATED PROMPT

- 3 def FUNCTION(VAR0):

- 4 import numpy as np

- 5 from scipy.ndimage import uniform_filter

- 6

- 7 VAR0 = np.pad(VAR0 , ((1, 1), (1, 1)), mode=’constant ’)

- 8 mask = VAR0 > 0

- 9 response = uniform_filter(mask , size=3)

- 10 scaling_factor = 10

- 11 labels = response * scaling_factor * mask * VAR0[1:-1, 1:-1]

- 12 return labels.astype(np.uint16)

It is important to acknowledge the limitations of our study. Given the nature of our data sources, we cannot fully guarantee that our results are completely unaffected by memorization, as many of the repositories used are likely present in the model’s training data. Our obfuscation approach, which focused on changing function names and parameters to combat function signature memorization, may not eliminate all effects of memorization. A more thorough obfuscation process that also alters context and dependencies would be ideal, but current limitations in symbol solving and refactoring algorithms for dynamic languages make this challenging to implement at scale.

Despite these constraints, the obfuscation process did have a meaningful, though not drastic, impact on Pass@K performance. This suggests that while memorization plays a role in GPT’s performance, the model’s ability to comprehend and interpret the task described in the prompt is also a significant factor in its code generation capabilities. The provided example comparing code generated from original and obfuscated prompts supports this conclusion. Although the generated functions are not identical, they are functionally similar, demonstrating the model’s capacity to grasp and interpret the task requirements even when key details are abstracted.

In summary, our findings indicate that while memorization influences GPT’s code generation performance, it is not the dominant factor. The model’s understanding and interpretation of the presented task also play a substantial role in its ability to generate accurate code. Additional research using more extensive obfuscation techniques could help further clarify the relative impact of memorization and understanding on the model’s performance.

N TOPIC ANALYSIS

The eight topics in the LDA model are described below.

- 1. Cancer and epigenetics: Cancer and epigenetics refer to the study of heritable changes in gene expression that do not involve changes to the underlying DNA sequence but can lead to the development and progression of cancer. These epigenetic modifications, which include DNA methylation, histone modifications, and small RNA-associated gene silencing, can turn genes on or off, influencing cancer susceptibility, initiation, progression, and response to treatment. Understanding these processes can aid in developing targeted therapies and better diagnostic tools for cancer patients.
- 2. Proteomics and microscopy: Proteomics and microscopy refer to the integration of proteomic analyses (the large-scale study of proteins, their structures, and functions) with advanced microscopy techniques. This integration allows scientists to visualize and quantify the spatial and temporal distribution of proteins within cells and tissues. By combining the detailed molecular information from proteomics with the high-resolution imaging from microscopy, researchers can gain deeper insights into cellular processes, protein-protein interactions, and the structural organization of cellular components.
- 3. Variant calling: Variant calling is a process in bioinformatics where sequence data (often from next-generation sequencing) are analyzed to identify differences, or variants, between a given sample and a reference genome. These variants can include single-nucleotide polymorphisms, insertions, deletions, and more. Identifying these variants is crucial for understanding genetic diversity, disease susceptibility, and personalizing medical treatments based on an individual’s genetic makeup.
- 4. Genetics and population analysis: Genetics and population analysis in bioinformatics refers to the study of the genetic composition of populations and how it changes over time. This involves analyzing large datasets of genetic information to understand patterns of genetic variation, inheritance, and evolution within and between populations. Such analyses can provide insights into population structure, migration, adaptation, and the history of species or populations, as well as help identify genetic factors associated with diseases or traits in specific populations.
- 5. Structure and molecular interaction: Structure and molecular interaction in bioinformatics pertains to the study and prediction of the three-dimensional structures of biomolecules (like proteins and nucleic acids) and how they interact with one another. By using computational methods to model and analyze these structures, scientists can gain insights into the molecular mechanisms underlying biological processes. This understanding can be pivotal for drug design, predicting protein functions, and understanding the effects of genetic mutations on molecular interactions and stability.
- 6. Web and graphical applications: Web and graphical applications in bioinformatics refer to the development and use of online tools and software with user-friendly graphical interfaces to analyze, visualize, and interpret biological data. These applications allow both experts and non-experts to interact with complex datasets, facilitating tasks like sequence alignment, gene annotation, pathway analysis, and more. By providing accessible platforms, these applications help democratize the analysis and understanding of vast and intricate bioinformatics data.
- 7. Assembly and sequence analysis: Assembly and sequence analysis in bioinformatics involves the process of taking raw sequence data, typically from next-generation sequencing technologies, and reconstructing the original DNA or RNA sequences. Assembly might mean piecing together short DNA reads into longer sequences or entire genomes. Once assembled, sequence analysis is used to identify genes, predict their functions, detect variations, and compare sequences across different organisms or individuals. This foundational process is key to many downstream analyses in genomics and personalized medicine.
- 8. Transcription and RNA-seq: Transcription and RNA-seq (RNA sequencing) in bioinformatics relate to the study of the transcriptome, i.e., the complete set of RNA molecules expressed from genes in a cell or tissue. RNA-seq is a technique that uses next-generation sequencing to capture and quantify the RNA in a sample, providing insights into which genes are active (or “expressed”) under specific conditions. By analyzing RNA-seq data, researchers can uncover patterns of gene expression, identify novel RNA species, and study regulatory pathways, aiding in understanding cellular responses, disease mechanisms, and developmental processes.

All eight topics were present across our subset of 28 repositories from which we created the benchmark dataset. A detailed breakdown of each repository, their descriptions taken from GitHub, and the topics to which they correspond, is as follows:

Note that we added an entire column for the description, and fixed the spacing. Nothing else has changed.

Repository Description Topic Areas AdmiralenOla/ Scoary

Pan-genome wide association studies

Genetics and population analysis, Assembly and sequence analysis

Transcription and RNA-seq

biocore/deblur Deblur is a greedy deconvolution algorithm based on known read error profiles.

An open-source application for biological image analysis

Cancer and epigenetics, Proteomics and microscopy

CellProfiler/ CellProfiler

Tools for handling Unique Molecular Identifiers in NGS data sets

Genetics and population analysis

CGATOxford/ UMI-tools

Automated omics-scale protein modeling and simulation setup.

Structure and molecular interaction

choderalab/ ensembler

etal/cnvkit Copy number variant detection from targeted DNA sequencing

Variant calling

Detection of macromolecular systems in protein datasets using systems modelling and similarity search.

Structure and molecular interaction, Web and graphical applications

gem-pasteur/ macsyfinder

microRNA PREdiction From small RNA-seq data

Transcription and RNA-seq

hangelwen/ miR-PREFeR

jnktsj/DNApi De novo adapter prediction algorithm for small RNA sequencing data

Web and graphical applications, Transcription and RNA-seq

juliema/aTRAM Automated Target Restricted Assembly Method

Genetics and population analysis, Assembly and sequence analysis

RNF framework for NGS: simulation of reads, evaluation of mappers, conversion of RNF-compliant data.

Variant calling, Structure and molecular interaction

karel-brinda/ rnftools

mad-lab/transit Repository for the TRANSIT and TPP tools developed by the Ioerger Lab

Web and graphical applications

makson96/Dynamics Dynamics PyMOL Plugin Structure and molecular interaction

Pipelines and tools for the processing of ancient and modern HTS data.

Variant calling, Assembly and sequence analysis

MikkelSchubert/ paleomix

Some hopefully useful tools for mass spectrometry applied to proteomics

Proteomics and microscopy

msproteomicstools/ msproteomicstools

LipidFinder: A computational workflow for discovery of new lipid molecular species

Structure and molecular interaction

ODonnell-Lipidomics/ LipidFinder

pgxcentre/genipe Genome-wide imputation pipeline Proteomics and microscopy, Genetics and population analysis

Tool for the detection and quantification of alternative splicing events from RNA-Seq data.

Transcription and RNA-seq

ratschlab/ spladder

Locating genetic regions that are "just right"

Web and graphical applications

SamStudio8/ goldilocks

#### Repository Description Topic Areas

simonvh/fluff Fluff is a Python package that contains several scripts to produce pretty, publication-quality figures for next-generation sequencing experiments.

Web and graphical applications

Python library to simulate evolutionary sequence data

Structure and molecular interaction, Web and graphical applications

sjspielman/ pyvolve

ST Pipeline contains the tools and scripts needed to process and analyze the raw files generated with the Spatial Transcriptomics method in FASTQ format.

Proteomics and microscopy, Web and graphical applications, Transcription and RNA-seq

Research/st_ pipeline

ursgal/ursgal Ursgal - universal Python module combining common bottom-up proteomics tools for large-scale analysis

Proteomics and microscopy

vals/umis Tools for processing UMI RNA-tag data

Transcription and RNA-seq

A Python API for the RCSB Protein Data Bank (PDB)

Proteomics and microscopy, Web and graphical applications

williamgilpin/ pypdb

zhanglab/psamm Curation and analysis of metabolic models

Structure and molecular interaction, Web and graphical applications

NEAT read simulation tools Variant calling, Assembly and se-

zstephens/ neat-genreads

quence analysis samtools/htsjdk A Java API for high-throughput se-

Assembly and sequence analysis, Transcription and RNA-seq

quencing data (HTS) formats.

Note that in this study, we primarily focused on categorizing based on biological disciplines. We recognize that other groupings, such as by data modality, could offer additional valuable perspectives to our work.

O MODEL ERROR DISTRIBUTIONS

Models Failure: Syntax Error

Failure: Runtime Error

Failure: Timeout Error

Failure: Output Disagreement

Passed Tests

CodeGen-6B-Mono 11268 8176 1 148 105 CodeGen2-7B 12687 6718 0 211 79 GPT-3.5 Turbo 9231 10603 0 5624 6643 InCoder-6B 11268 8176 1 148 105 InstructCodeT5P-16B 19667 33 0 0 0 SantaCoder 14391 4601 1 555 139 StarCoder 26233 10688 0 1660 808

The failure modes are similar between the models but not exactly the same. Below is the failure breakdown for each model:

- 1

- 2 CodeGen -6B-Mono

- 3

- 4 Failure: Syntax Error = 11268

- 5 Failure: Runtime Error = 8176

- 6 Failure: Timeout Error = 1

- 7 Failure: Output Disagreement = 148

- 8 Passed Tests = 105

- 9

- 10 CodeGen2 -7B

- 11

- 12 Failure: Syntax Error = 12687

- 13 Failure: Runtime Error = 6718

- 14 Failure: Output Disagreement = 211

- 15 Passed Tests = 79

- 16

- 17 GPT3.5-Turbo

- 18

- 19 Failure: Syntax Error = 9231

- 20 Failure: Runtime Error = 10603

- 21 Failure: Output Disagreement = 5624

- 22 Passed Tests = 6643

- 23

- 24 InCoder -6B

- 25

- 26 Failure: Syntax Error = 12777

- 27 Failure: Runtime Error = 6502

- 28 Failure: Timeout Error = 2

- 29 Failure: Output Disagreement = 309

- 30 Passed Tests = 100

- 31

- 32 InstructCodeT5P -16B

- 33

- 34 Failure: Syntax Error = 19667

- 35 Failure: Runtime Error = 33

- 36

- 37 SantaCoder

- 38

- 39 Failure: Syntax Error = 14391

- 40 Failure: Runtime Error = 4604

- 41 Failure: Timeout Error = 1

- 42 Failure: Output Disagreement = 555

- 43 Passed Tests = 139

- 44

- 45 StarCoder

- 46

- 47 Failure: Syntax Error = 26233

- 48 Failure: Runtime Error = 10688

- 49 Failure: Output Disagreement = 1660

- 50 Passed Tests = 808

- P LANGUAGE ANALYSIS

Our decision to include Java and Python was based on an empirical investigation into the prevalence of different programming languages across bioinformatics repositories. We computed the total byte sizes of various languages across all repositories we surveyed. Out of the 13 languages analyzed (Python, Bourne Shell, R, Perl, Cpp, C, Java, Bourne_Again_Shell, MATLAB, m4, SQL, Ruby, PHP), Java was the most prevalent, with a byte size of 242,318,285 bytes, followed closely by Python at 192,324,018 bytes. The presence of Java as the most extensive language indicates that it plays a crucial role in the bioinformatics community, perhaps in the form of underlying tools or frameworks that support high-level bioinformatics operations. A detailed breakdown of the byte size of the various languages in our repository is as follows:

#### Programming Language Number of Bytes

Java 242,318,285 Python 192,324,018

C 184,967,144 C++ 184,694,473 Perl 129,213,485 R 40,708,273 Bourne Shell 35,495,605

PHP 33,876,889

MATLAB 28,889,990 SQL 15,630,061 Ruby 8,935,640

m4 7,956,980 Bourne Again Shell 2,851,620

- Q SUMMARY AT BOTTOM RESULTS

Summary At Bottom results were omitted from Table 4 in the main text for brevity, but we include them here. More details about how Summary At Bottom prompts are constructed can be found in Appendix h.

Model Pass@1 Pass@5 Pass@10 Pass@20

InCoder-6B 0.610 2.587 4.303 6.274 SantaCoder-1.1B 0.510 1.949 3.013 4.459 StarCoder-15.5B 6.465 13.824 16.746 19.076

StarCoder-15.5B (finetuned) \ \ \ \

StarCoder+ 4.172 11.772 14.933 17.197 CodeGen-6B-mono 2.070 4.535 5.896 7.006

CodeGen-16B-mono 2.166 5.137 6.022 6.369 CodeGen2-7B 0.510 1.019 1.207 1.274 GPT-3.5-Turbo 13.439 20.040 22.460 25.478

GPT-4 6.316 15.272 20.223 24.823

Table 3: Performance on "Summary at Bottom" rows for Python columns

- R IMPORTANCE OF THE CONTEXT

Imports and classes are predefined and are included in the context because, as we are testing functionlevel code generation, we are not prompting the model nor expecting the model to generate the classes it needs to pass the tests. Instead, we are testing the model’s ability to extract the pertinent imports and classes from the context to use in the generated function.

To further illustrate how the prompt needs to contain information on the context to correctly generate the golden code, we provide an example below. The golden-code function uses the global variable "IGNORE_GENE_NAMES." We include this information in the prompt because it embeds the task’s necessary foundations. Essentially, we ensure the prompt has enough context for an informed human to reproduce the function, implying that the function generated ought to incorporate the environment’s existing resources (imports & classes) rather than generating new ones.

- 1 PROMPT:

- 2

- 3 This is in python.

- 4 Write a function called "unpipe_name" that takes in a string parameter called "name". The function takes a string containing multiple gene names separated by pipes , and returns a single gene name devoid of duplicates and pipe characters. The function checks if there are any duplicates , and removes any meaningless target names. If there are multiple gene names present , the function takes the longest name as the final name. If there are any ambiguous gene names , the function logs a warning and selects the longest name as the final name. The function should also import "cnvlib.params".

- 5

- 6 def unpipe_name(name):

- 7

- 8 Here are the imports:

- 9 import numpy as np

- 10 import logging

- 11 from . import params

- 12 from skgenome import tabio

- 13 Here are the global variables:

- 14 MIN_REF_COVERAGE = -5.0

- 15 MAX_REF_SPREAD = 1.0

- 16 NULL_LOG2_COVERAGE = -20.0

- 17 GC_MIN_FRACTION = 0.3

- 18 GC_MAX_FRACTION = 0.7

- 19 INSERT_SIZE = 250

- 20 IGNORE_GENE_NAMES = ’-’, ’.’, ’CGH ’

- 21 ANTITARGET_NAME = ’Antitarget ’

- 22 ANTITARGET_ALIASES = ANTITARGET_NAME , ’Background ’

- 23 Here are the class declarations:

- 24 Here are the additional function declarations:

- 25 def do_import_picard(fname , too_many_no_coverage):

- 26 summary: Reads a file in ’picardhs ’ format , processes the data , and returns a modified dataframe.

- 27 param: fname (string) - the file name/path to be read in ’picardhs ’ format.

- 28 param: too_many_no_coverage (int) - if the number of bins with no coverage is greater than this value , a warning message is logged. Default is 100.

- 29 return: garr (pandas dataframe) - a modified dataframe with added columns ’gene ’ and ’log2 ’ based on the original dataframe read from the input file.

- 30 def unpipe_name(name):

- 31 summary: Remove duplicate gene names and pipe characters from a given string.

- 32 param: name (str) - the string containing the gene names.

- 33 return: new_name (str) - the cleaned gene name string.

- 34 def do_import_theta(segarr , theta_results_fname , ploidy):

- 35 summary: A function for importing theta results and estimating copy number and log2 ratios of segments.

- 36 param: segarr (numpy array) - array of segments

- 37 param: theta_results_fname (str) - name of theta results file

- 38 param: ploidy (int) - ploidy of genome (default is 2)

- 39 return: generator of numpy arrays - array of segments with estimated copy number and log2 ratios.

- 40 def parse_theta_results(fname):

- 41 summary: Parses THetA results into a data structure with NLL , mu, C, and p* columns.

- 42 param: fname (str) - name of the file to parse the results from

- 43 return: (dict) - a dictionary containing the NLL , mu_normal , mu_tumors , C, and p* values

- 44 Here are the comments and the specs:

- 45 Write a function called "unpipe_name" that takes in a string parameter called "name". The function takes a string containing multiple gene names separated by pipes , and returns a single gene name devoid of duplicates and pipe characters. The function checks if there are any duplicates , and removes any meaningless target names. If there are multiple gene names present , the function takes the longest name as the final name. If there are any ambiguous gene names , the function logs a warning and selects the longest name as the final name. The function should also import "cnvlib.params".

- 46

- 47 def unpipe_name(name):

- 48 GOLDEN CODE:

- 49

- 50 def unpipe_name(name):

- 51 """Fix the duplicated gene names Picard spits out.

- 52

- 53 Return a string containing the single gene name , sans duplications and pipe

- 54 characters.

- 55

- 56 Picard CalculateHsMetrics combines the labels of overlapping intervals

- 57 by joining all labels with ’|’, e.g. ’BRAF|BRAF ’ -- no two distinct

- 58 targeted genes actually overlap , though , so these dupes are redundant

.

- 59 Meaningless target names are dropped , e.g. ’CGH|FOO|-’ resolves as ’ FOO ’.

- 60 In case of ambiguity , the longest name is taken , e.g. "TERT|TERT Promoter"

- 61 resolves as "TERT Promoter".

- 62 """

- 63 if ’|’ not in name:

- 64 return name

- 65 gene_names = set(name.split(’|’))

- 66 if len(gene_names) == 1:

- 67 return gene_names.pop()

- 68 cleaned_names = gene_names.difference(IGNORE_GENE_NAMES)

- 69 if cleaned_names:

- 70 gene_names = cleaned_names

- 71 new_name = sorted(gene_names , key=len , reverse=True)[0]

- 72 if len(gene_names) > 1:

- 73 logging.warning(’WARNING: Ambiguous gene name %r; using %r’, name ,

- 74 new_name)

- 75 return new_name

S POTENTIAL CONCERNS ABOUT CHANGES IN PACKAGES

In addressing potential concerns about code generation issues due to changes in packages, we have taken steps to ensure stability and consistency in our testing environment. Our testing environment utilizes standardized and version-locked packages that guarantee stable results for our "golden" code samples. As such, we do not need to worry about package updates or feature deprecation.

Furthermore, when giving prompts, we have taken care to specifically mention the packages that need to be used, which guides the LLM to correctly utilize the expected versions of these packages. Any remaining discrepancies will be highlighted in our error analysis for the respective LLMs.

In the prompts we provide, we clearly specify which packages and functions should be used, including any external dependencies. This way, the LLM knows exactly which packages to use.

Take, for example, this dataset-generated prompt in Python:

- 1 #This is in python

- 2

- 3 #write a function called "binary_erosion" that takes in two parameters: " x_data" and "structuring_element". The function should first check if

the "structuring_element" is a 2D array or not. If it is, and " x_data" is not a 2D array , the function should create a new array called "y_data" that is the same size as "x_data" and contains only zeros. The function should then loop through each slice of "x_data" and apply the skimage.morphology.binary_erosion function using the " structuring_element" as the structuring element. The result of each slice should be added to the corresponding slice of "y_data". Finally , the function should return "y_data". If the "structuring_element" is not 2D and "x_data" is 2D, the function should raise a NotImplementedError with the message "A 3D structuring element cannot

be applied to a 2D image." Otherwise , the function should simply apply the skimage.morphology.binary_erosion function to "x_data" using "structuring_element" as the structuring element and return the

result.

- 4 #

- 5 #def binary_erosion(x_data , structuring_element):

The prompt meticulously guides the LLM to employ the "binary_erosion" function derived from the "skimage.morphology package." Any inaccuracies in using the specified packages, likely due to the model training on outdated or more advanced versions, would deem its output incorrect, leading us to categorize it as a failed generation. A paramount facet we are evaluating is the proficiency of the LLM in accurately implementing specified external packages as mentioned in the prompts.

T DIFFERENCES AMONG PUBLIC, HIDDEN, AND SIMILAR SETS

While constructing the datasets, we observed the need for a categorization that simultaneously caters to present-day LLMs and future, more capable versions. Consequently, we delineated our public and hidden datasets, which admittedly, resulted in a considerable divergence between their summary statistics.

The public test set encapsulates relatively moderate challenges an LLM might encounter, keeping in mind the token limits of current-generation code LLMs, which range from 2,048 to 4,096 tokens. This constraint dictated that the public test set should comprise smaller and simpler functions.

Conversely, the hidden dataset was configured to house a broader, more challenging set of problems, aiming to benchmark future models, for instance, GPT4 with its anticipated 32K token limit. This two-tiered framework ensures the longevity of this benchmark as advancements unfold in the field of LLMs and facilitates additional investigations into domain-specific generative models.

Moreover, we have devised a "Similar Dataset," which is essentially a subset of the hidden dataset harboring statistics comparable to the public dataset. This new dataset ensures direct, fair comparisons between the public and the hidden tests.

The "public data" represents datasets with specific crafted contexts and corresponding test cases. The "hidden data" encompasses a wider array of intricate issues. Furthermore, the "similar data" is a subset of the hidden data, curated to mimic the overall distribution of the public data. This set includes an additional 157 Python problems and 50 Java problems, maintaining the same 253 Rosalind problem tally, thus reflecting the composition of the public data. The issues in this set were cherry-picked based on their L2 similarity norm of calculated statistics, ensuring close alignment to the mean distribution of the public data.

In addition, the reason for having a much larger number of hidden tests compared to public tests is to ensure the integrity of our benchmark. Thus, we divided our dataset into a much larger hidden set rather than a public set. We believe that access to too many public test cases might lead to potential overfitting while fine-tuning models, which would fundamentally undermine the benchmark.

U PROMPT STRUCTURE ANALYSIS

As demonstrated by the scatterplots in Appendix j, there is a general negative correlation between the length of the prompt and the performance of the model. We also plot the number of models that perform the best for each prompt type as follows:

Number of Models that Performed the Best vs. Prompt Style

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

- 0
- 1
- 2
- 3
- 4
- 5
- 6

NumberofModels

SummaryAtTop SummaryAtBottom Uncommented SummaryOnly NecessaryOnly

Prompt Style

Figure 14: Number of best-performing models over all prompt styles

As demonstrated in Figure 14, overall, the prompts with the fewest tokens, i.e., the "Summary Only" and "Necessary Only" prompt styles, had the most models that performed the best. However, some models performed better with the longer "Summary At Top" and "Summary At Bottom" prompts. In this section, we analyze some of the possible reasons for these discrepancies and variations in the results by considering prompt structure. Consider the following "Summary Only" and "Necessary Only" prompts for the following "binary_erosion" function:

- 1 def binary_erosion(x_data , structuring_element):

- 2 is_strel_2d = structuring_element.ndim == 2

- 3 is_img_2d = x_data.ndim == 2

- 4 if is_strel_2d and not is_img_2d:

- 5 y_data = numpy.zeros_like(x_data)

- 6 for index , plane in enumerate(x_data):

- 7 y_data[index] = skimage.morphology.binary_erosion(plane ,

- 8 structuring_element)

- 9 return y_data

- 10 if not is_strel_2d and is_img_2d:

- 11 raise NotImplementedError(

- 12 ’A 3D structuring element cannot be applied to a 2D image.’)

- 13 y_data = skimage.morphology.binary_erosion(x_data , structuring_element)

- 14 return y_data Here is the "Summary Only" prompt:

- 1 #This is in python

- 2 #write a function called "binary_erosion" that takes in two parameters: " x_data" and "structuring_element". The function should first check if

the "structuring_element" is a 2D array or not. If it is, and "

x_data" is not a 2D array , the function should create a new array called "y_data" that is the same size as "x_data" and contains only zeros. The function should then loop through each slice of "x_data" and apply the skimage.morphology.binary_erosion function using the " structuring_element" as the structuring element. The result of each slice should be added to the corresponding slice of "y_data". Finally , the function should return "y_data". If the "structuring_element" is not 2D and "x_data" is 2D, the function should raise a NotImplementedError with the message "A 3D structuring element cannot

be applied to a 2D image." Otherwise , the function should simply apply the skimage.morphology.binary_erosion function to "x_data" using "structuring_element" as the structuring element and return the

result.

- 3 #

- 4 #def binary_erosion(x_data , structuring_element): Here is the "Necessary Only" prompt:

- 1 Write a function with the following specs:

- 2 --specs begin here --

- 3 #write a function called "binary_erosion" that takes in two parameters: " x_data" and "structuring_element". The function should first check if

the "structuring_element" is a 2D array or not. If it is, and " x_data" is not a 2D array , the function should create a new array called "y_data" that is the same size as "x_data" and contains only zeros. The function should then loop through each slice of "x_data" and apply the skimage.morphology.binary_erosion function using the " structuring_element" as the structuring element. The result of each slice should be added to the corresponding slice of "y_data". Finally , the function should return "y_data". If the "structuring_element" is not 2D and "x_data" is 2D, the function should raise a NotImplementedError with the message "A 3D structuring element cannot

be applied to a 2D image." Otherwise , the function should simply apply the skimage.morphology.binary_erosion function to "x_data" using "structuring_element" as the structuring element and return the

result.

- 4 param: x_data (numpy.ndarray) - input data to be eroded.

- 5 param: structuring_element (numpy.ndarray) - structuring element for erosion.

- 6 return: y_data (numpy.ndarray) - eroded data.

- 7 --specs end here --

- 8 Note the function will be embedded in the following context

- 9 --context begins here --

- 10 import numpy

- 11 import skimage.morphology

- 12 import os

- 13 numpy.random.seed(<|int;range=0,100|>)

- 14 <<insert solution here >>

- 15 def main():

- 16 x_data = numpy.random.randint(2, size=(10, 10))

- 17 structuring_element = skimage.morphology.square(3)

- 18 print(binary_erosion(x_data , structuring_element))

- 19 if __name__ == "__main__":

- 20 main()

- 21 --context ends here --

- 22 Make sure to only generate the function and not any of the context. Make sure you are generating valid , runnable code. Begin your solution with:

- 23 def binary_erosion(x_data , structuring_element):

- 24 MAKE SURE TO INDENT THE BODY OF YOUR FUNCTION BY A TAB

From the structure of these prompts, we can hypothesize why some of the simpler models perform better with the "Summary Only" prompts, while the GPT models generally perform better with the "Necessary Only" prompts. Since the "Necessary Only" prompts include the GPT-generated summary of the function embedded within the prompt, along with additional context, it is reasonable

to hypothesize that simpler models such as InCoder, SantaCoder, StarCoder, or CodeGen are unable to digest the context information embedded within the "Necessary Only" prompts. Instead, the context information ends up being regarded as noise that confuses the simpler models, resulting in poorer performance. However, larger models, such as the GPT-3.5 and GPT-4 models released by OpenAI, are able to properly incorporate the context information provided in the "Necessary Only" prompts, and as a result, they perform significantly better than their smaller counterparts with the additional context information.

It is also worth examining the structure of the "Uncommented" prompt style, to understand why these prompts perform poorly when compared to the commented versions. For instance, consider the "Uncommented" prompt style for the "binary_erosion" problem:

- 1 This is in python

- 2 write a function called "binary_erosion" that takes in two parameters: " x_data" and "structuring_element". The function should first check if

the "structuring_element" is a 2D array or not. If it is, and " x_data" is not a 2D array , the function should create a new array called "y_data" that is the same size as "x_data" and contains only zeros. The function should then loop through each slice of "x_data" and apply the skimage.morphology.binary_erosion function using the " structuring_element" as the structuring element. The result of each slice should be added to the corresponding slice of "y_data". Finally , the function should return "y_data". If the "structuring_element" is not 2D and "x_data" is 2D, the function should raise a NotImplementedError with the message "A 3D structuring element cannot

be applied to a 2D image." Otherwise , the function should simply apply the skimage.morphology.binary_erosion function to "x_data" using "structuring_element" as the structuring element and return the

result.

- 3

- 4 def binary_erosion(x_data , structuring_element):

- 5

- 6 Here are the imports:

- 7 import skimage.morphology

- 8 import scipy.ndimage

- 9 import numpy

- 10 Here are the global variables:

- 11 Here are the class declarations:

- 12 Here are the additional function declarations:

- 13 def dilation(x_data , structuring_element):

- 14 summary: Performs dilation on input image data with a structuring element

- 15 param: x_data (numpy array) - input image data to perform dilation on

- 16 param: structuring_element (numpy array) - structuring element for the dilation operation

- 17 return: y_data (numpy array) - resulting dilated image data

- 18 def erosion(x_data , structuring_element):

- 19 summary: Performs erosion operation on input data using a structuring element.

- 20 param: x_data (numpy.ndarray) - input data to apply erosion on.

- 21 param: structuring_element (numpy.ndarray) - structuring element used for erosion operation.

- 22 return: y_data (numpy.ndarray) - erosion result as a 2D or 3D numpy array.

- 23 def binary_erosion(x_data , structuring_element):

- 24 summary: Performs binary erosion on input data using a structuring element.

- 25 param: x_data (numpy.ndarray) - input data to be eroded.

- 26 param: structuring_element (numpy.ndarray) - structuring element for erosion.

- 27 return: y_data (numpy.ndarray) - eroded data.

- 28 def morphological_gradient(x_data , structuring_element):

- 29 summary: Computes the morphological gradient of an image.

- 30 param: x_data (numpy array) - input data.

- 31 param: structuring_element (numpy array) - structuring element used for the operation.

- 32 return: y_data (numpy array) - output data.

- 33 Here are the comments and the specs:

- 34 write a function called "binary_erosion" that takes in two parameters: " x_data" and "structuring_element". The function should first check if

the "structuring_element" is a 2D array or not. If it is, and " x_data" is not a 2D array , the function should create a new array called "y_data" that is the same size as "x_data" and contains only zeros. The function should then loop through each slice of "x_data" and apply the skimage.morphology.binary_erosion function using the " structuring_element" as the structuring element. The result of each slice should be added to the corresponding slice of "y_data". Finally , the function should return "y_data". If the "structuring_element" is not 2D and "x_data" is 2D, the function should raise a NotImplementedError with the message "A 3D structuring element cannot

be applied to a 2D image." Otherwise , the function should simply apply the skimage.morphology.binary_erosion function to "x_data" using "structuring_element" as the structuring element and return the

result.

- 35 def binary_erosion(x_data , structuring_element):

Here is the corresponding commented version of the prompt:

- 1 #This is in python

- 2 #write a function called "binary_erosion" that takes in two parameters: " x_data" and "structuring_element". The function should first check if

the "structuring_element" is a 2D array or not. If it is, and " x_data" is not a 2D array , the function should create a new array called "y_data" that is the same size as "x_data" and contains only zeros. The function should then loop through each slice of "x_data" and apply the skimage.morphology.binary_erosion function using the " structuring_element" as the structuring element. The result of each slice should be added to the corresponding slice of "y_data". Finally , the function should return "y_data". If the "structuring_element" is not 2D and "x_data" is 2D, the function should raise a NotImplementedError with the message "A 3D structuring element cannot

be applied to a 2D image." Otherwise , the function should simply apply the skimage.morphology.binary_erosion function to "x_data" using "structuring_element" as the structuring element and return the

result.

- 3 #

- 4 #def binary_erosion(x_data , structuring_element):

- 5 #

- 6 #Here are the imports:

- 7 #import skimage.morphology

- 8 #import scipy.ndimage

- 9 #import numpy

- 10 #Here are the global variables:

- 11 #Here are the class declarations:

- 12 #Here are the additional function declarations:

- 13 #def dilation(x_data , structuring_element):

- 14 # summary: Performs dilation on input image data with a structuring element

- 15 # param: x_data (numpy array) - input image data to perform dilation on

- 16 # param: structuring_element (numpy array) - structuring element for the dilation operation

- 17 # return: y_data (numpy array) - resulting dilated image data

- 18 #def erosion(x_data , structuring_element):

- 19 # summary: Performs erosion operation on input data using a structuring element.

- 20 # param: x_data (numpy.ndarray) - input data to apply erosion on.

- 21 # param: structuring_element (numpy.ndarray) - structuring element used for erosion operation.

- 22 # return: y_data (numpy.ndarray) - erosion result as a 2D or 3D numpy array.

- 23 #def binary_erosion(x_data , structuring_element):

- 24 # summary: Performs binary erosion on input data using a structuring element.

- 25 # param: x_data (numpy.ndarray) - input data to be eroded.

- 26 # param: structuring_element (numpy.ndarray) - structuring element for erosion.

- 27 # return: y_data (numpy.ndarray) - eroded data.

- 28 #def morphological_gradient(x_data , structuring_element):

- 29 # summary: Computes the morphological gradient of an image.

- 30 # param: x_data (numpy array) - input data.

- 31 # param: structuring_element (numpy array) - structuring element used for the operation.

- 32 # return: y_data (numpy array) - output data.

- 33 #Here are the comments and the specs:

- 34 #write a function called "binary_erosion" that takes in two parameters: " x_data" and "structuring_element". The function should first check if

the "structuring_element" is a 2D array or not. If it is, and " x_data" is not a 2D array , the function should create a new array called "y_data" that is the same size as "x_data" and contains only zeros. The function should then loop through each slice of "x_data" and apply the skimage.morphology.binary_erosion function using the " structuring_element" as the structuring element. The result of each slice should be added to the corresponding slice of "y_data". Finally , the function should return "y_data". If the "structuring_element" is not 2D and "x_data" is 2D, the function should raise a NotImplementedError with the message "A 3D structuring element cannot

be applied to a 2D image." Otherwise , the function should simply apply the skimage.morphology.binary_erosion function to "x_data" using "structuring_element" as the structuring element and return the

result.

- 35 #def binary_erosion(x_data , structuring_element):

Comparing the two prompt styles, the only difference is the inclusion of the "#" symbol in front of each line in the commented version. These "#" symbols appear to have a large effect on LLM code generation. For instance, here is a sample ChatGPT code generation after being passed in a "Uncommented" prompt:

- 1 Here is the Python function as you described:

- 2

- 3 ‘‘‘Python

- 4 import numpy as np

- 5 from skimage.morphology import binary_erosion

- 6

- 7 def binary_erosion(x_data , structuring_element):

- 8

- 9 # Check if structuring_element is 2D

- 10 if len(structuring_element.shape) == 2:

- 11 # Check if x_data is not 2D

- 12 if len(x_data.shape) != 2:

- 13 # Create y_data (same shape as x_data , filled with zeros)

- 14 y_data = np.zeros_like(x_data)

- 15

- 16 # Apply binary_erosion slice by slice

- 17 for idx in range(x_data.shape[0]):

- 18 y_data[idx] = binary_erosion(x_data[idx], structuring_element)

- 19

- 20 return y_data

- 21

- 22 else:

- 23 # Apply binary_erosion directly

- 24 return binary_erosion(x_data , structuring_element)

- 25

- 26 else:

- 27 # Check if x_data is 2D

- 28 if len(x_data.shape) == 2:

- 29 # Raise error

- 30 raise NotImplementedError("A 3D structuring element cannot be applied to a 2D image.")

- 31

- 32 else:

- 33 # Apply binary_erosion directly

- 34 return binary_erosion(x_data , structuring_element)

- 35 ‘‘‘

- 36 If 2D structuring element is provided and x_data isn ’t 2D, function applies the skimage.morphology.binary_erosion function to each slice of x_data separately. Then adds result to y_data. Otherwise , if structuring_element isn’t 2D and x_data is 2D, it raises an error. If

both conditions don’t apply , it directly applies the binary_erosion function to x_data.

Here is a sample ChatGPT code generation after being passed in a corresponding commented prompt:

- 1 import skimage.morphology

- 2 import numpy as np

- 3

- 4 def binary_erosion(x_data , structuring_element):

- 5 if len(structuring_element.shape) == 2:

- 6 if len(x_data.shape) != 2:

- 7 y_data = np.zeros_like(x_data)

- 8 for i in range(x_data.shape[0]):

- 9 y_data[i] = skimage.morphology.binary_erosion(x_data[i], structuring_element)

- 10 return y_data

- 11

- 12 elif len(structuring_element.shape) != 2 and len(x_data.shape) == 2:

- 13 raise NotImplementedError("A 3D structuring element cannot be applied to a 2D image.")

- 14

- 15 else:

- 16 return skimage.morphology.binary_erosion(x_data , structuring_element)

We can observe that the models respond far better to the commented prompt, only returning runnable Python code without unnecessary text that may result in runtime errors in the testing environment. Overall, it is not just prompt length that has a tangible effect on LLM code generation performance. For instance, the slightly longer "Necessary Only" prompts can outperform the shorter "Summary Only" prompts for larger, more complex models that can properly digest and understand the given code context. Furthermore, the inclusion of certain symbols, such as the presence of a "#" at the beginning of every line in the commented prompts, can prompt models to generate more desirable code outputs.

V WIKIPEDIA SCRAPING METHODOLOGY AND RESULTS

To obtain a list of keywords related to bioinformatics, we employ a two-step method. First, we utilize a recursive search of Wikipedia pages. We start with a few seed topics, scrape the Wikipedia pages of those topics for the next set of keywords, and scrape all Wikipedia pages corresponding to the next set of keywords. We run this recursive process twice. The seed words are as follows:

bioinformatics, genomics, proteomics, transcriptomics, metabolomics, phylogenetics, sequence alignment, gene prediction, molecular modeling, FASTA, FASTQ, SAM, BAM, VCF

From this, we obtained a list of 133,504 terms. Then, using the ‘word2vec-google-news-300‘ model ?, we computed the cosine similarity of each word with the word "Bioinformatics," and chose the 290 (an arbitrary number based on what we observed in the cosine similarity values) most similar to obtain the final wordlist. The final wordlist we used is as follows:

ADME, Aerobiology, Altix, Angiogenesis, Annotation, Antibody, Apoptosis, Aptamer, Arbovirus, Astrobiology, Astrophysics, Autoimmunity, Bacteriology, Bacteriophage, Beamline, Bioanalysis, Biobank, Biocatalysis, Biochemical, Biochemist, Biochemistry, Biochip, Biocontainment, Biodefense, Bioelectromagnetics, Bioelectronics, Bioenergetics, Biogeochemistry, Bioinformatic, Bioinformatics, Biology, Biomarker, Biomaterial, Biomaterials, Biomedicine, Biometry, Biomimetic, Biomimetics, Biopharmaceutical, Biophotonics, Biophysical, Biophysics, Bioscience, Biosciences, Biosensor, Biostatistics, Biosynthesis, Biotechnology, Carcinogenesis, Catalysis, Cheminformatics, Chemistry, Chirality, Chromatin, Chromatography, Chromosome, Cognate, Combinatorics, Computation, Computing, Connectome, Cosmochemistry, Cryptography, Crystallography, Cyberinfrastructure, Cybernetics, Cytogenetics, Cytokine, Cytopathology, Cytoscape, DECIPHER, DESY, DNASTAR, Dendrimer, Dermatopathology, Dosimetry, Drosha, ENCODE, Econometrics, Electrochemistry, Electromechanics, Electrophysiology, Electroporation, Endocrinology, Engineering, Entomology, Entrez, Epidemiology, Epigenetic, Epigenetics, Epigenome, Epithelium, Epitope, Ethology, Eukaryotes, Eukaryotic, Exome, Foldit, Frizzled, GISAID, GPCR, GenBank, GeneCards, Geneticist, Genetics, Genome, Genomic, Genomics, Genotyping, Geobiology, Geochemistry, Geomorphology, Germline, Glaciology, Glycobiology, Glycomics, Glycoprotein, GridPP, Haplotype, Hemagglutinin, Hematopathology, Hemostasis, Histocompatibility, Histology, Histone, Histopathology, Hydrometallurgy, Hypertext, Immunochemistry, Immunogenetics, Immunohistochemistry, Immunology, Immunotherapy, Inference, Informatics, Integrase, Integrin, KEGG, KNIME, Kinase, Laboratory, Limnology, Linguistics, MChip, Macromolecule, Meiosis, Metabolism, Metabolomics, Metagenomics, Methylation, MicroRNA, MicroRNAs, Microanalysis, Microarray, Microarrays, Microbiology, Microeconomics, Microprobe, Microscopy, Mitochondrial, Modelica, Molecular, Morphogenesis, Morpholino, Motility, Multiplexing, Mutagenesis, Mycology, NAMD, Nanobiotechnology, Nanoelectronics, Nanoengineering, Nanomaterials, Nanomedicine, Nanopore, Nanotechnology, Nanotoxicology, Nematology, Neuroanatomy, Neurochemistry, Neuroeconomics, Neuroendocrinology, Neuroethics, Neurogenesis, Neurogenetics, Neuroimaging, Neuroimmunology, Neuroinflammation, Neuroinformatics, Neuropathology, Neuropharmacology, Neurophysiology, Neuroprotection, Neuropsychiatry, Neuropsychology, Neuroradiology, Neurorehabilitation, Neuroscience, Neurovirology, Nucleation, Nucleotide, Nutrigenomics, Oceanography, Oligonucleotide, Omics, Oncogene, Ontologies, Ontology, OpenMM, Otorhinolaryngology, PRNP, Palaeontology, Paleobiology, Paleontology, Panspermia, Parasitology, Pathogen, Pathogenesis, Pathology, Pathophysiology, Peptide, Peptides, Petrology, Pharmaceutics, Pharmacogenetics, Pharmacogenomics, Pharmacognosy, Pharmacology, Pharmacovigilance, Phenotype, Phenotypes, Phenotypic, Philology, Phosphorylation, Photobiology, Photochemistry, Phylogenetics, Phylogeny, Physics, Physiology, Polymerase, Polymerization, Postharvest, Primatology, Proteome, Proteomic, Proteomics, Proteostasis, Psycholinguistics, PubChem, Pulmonology, Pyrosequencing, QSAR, Qubit, RNAi, Radiochemistry, RefSeq, Retrovirus, Rheology, Ribosome, SBML, Scilab, Sequencing, Serology, Speciation, Spectroscopy, Spintronics, Sulfolobus, Supercomputer, Supercomputers, Supercomputing, Synchrotron, Systematics, TaqMan, Taxonomy, Technology, Teratology, Tetrahymena, Toxicogenomics, Transcriptional, Transcriptome, Transfection, Tribology, Ubiquitin, Vertebrate, Vertebrates, Virology, WEHI, WikiBhasha, Zoology

