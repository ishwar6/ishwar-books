# Appendix A · The Research Papers, in the Order to Learn Them

> You will not have time to read these papers. This appendix gives you, for each one, the
> 5–10 points that are the paper, and the one thing to say about it in an interview.
> Read the tiers in order; within a tier the order is also deliberate. Every arXiv link
> was checked on 8 September 2026.
>
> Format per paper: **Title** (authors, venue, year) link (*why it matters*) crux
> points: **interview line**.

Time budget if you read only this appendix: about 3 hours. The two-week plan and the
"only 10" list are at the end.

---

## Tier 1: Foundations (why any of this works)

### 1. Attention Is All You Need
Vaswani et al., NeurIPS 2017: [arXiv 1706.03762](https://arxiv.org/abs/1706.03762)
*The architecture every embedding model and every LLM is built on.*
- Replaces recurrence with **self-attention**: each token forms query/key/value vectors;
  output = softmax(QKᵀ/√d)·V, a weighted mix of all tokens' values.
- **Multi-head** attention: several attention maps in parallel, each learning a different
  relation (syntax, coreference, position).
- **Positional encodings** inject order since attention itself is permutation-invariant.
- Encoder–decoder stack of attention + feed-forward layers with residual connections and
  layer norm; trains fully in parallel over the sequence (RNNs cannot).
- Cost is **O(n²)** in sequence length: the root reason long context is expensive and
  retrieval matters.
- Set new SOTA on translation with far less training compute than prior models.
**Interview line:** "Self-attention lets every token see every other in one step, in
parallel; the price is quadratic cost in length."

### 2. BERT
Devlin et al., NAACL 2019: [arXiv 1810.04805](https://arxiv.org/abs/1810.04805)
*The encoder that made contextual text representations a commodity.*
- Encoder-only transformer pre-trained with **masked language modelling** (predict hidden
  tokens from both sides) and next-sentence prediction.
- **Bidirectional** context: unlike GPT-style left-to-right models, each token's vector sees
  the whole sentence: better for understanding/embedding, useless for generation.
- Pre-train once, **fine-tune** cheaply per task; dominated NLP benchmarks in 2019.
- The `[CLS]` token or mean-pooled outputs became the default "sentence vector": but raw
  BERT vectors are poor for similarity (see Sentence-BERT).
- Cross-encoders (rerankers) are BERT with query and passage concatenated.
**Interview line:** "BERT-style encoders are what embedding models and rerankers are; GPT-style
decoders are what generate."

### 3. Sentence-BERT
Reimers & Gurevych, EMNLP 2019: [arXiv 1908.10084](https://arxiv.org/abs/1908.10084)
*Turned BERT into a usable embedding model.*
- Raw BERT similarity needs both sentences in one forward pass (cross-encoder): finding the
  most similar pair among 10K sentences took ~65 hours. Unusable for search.
- **Siamese/bi-encoder** setup: encode each sentence independently, mean-pool, train with
  classification/regression/triplet objectives so cosine similarity means semantic similarity.
- Reduced the 10K-sentence task to ~5 seconds with comparable accuracy.
- Established the bi-encoder vs cross-encoder trade-off: precomputable and fast vs precise.
- The `sentence-transformers` library came from this paper and is still the standard tool.
**Interview line:** "SBERT is why we can precompute one vector per chunk and compare with
cosine; the cross-encoder is what we lose and why rerankers exist."

### 4. Dense Passage Retrieval (DPR)
Karpukhin et al., EMNLP 2020: [arXiv 2004.04906](https://arxiv.org/abs/2004.04906)
*Proved dense retrieval could beat BM25 for QA.*
- Two BERT encoders (question, passage), trained so the dot product is high for gold
  question–passage pairs.
- Training trick that matters: **in-batch negatives** plus **hard negatives** from BM25:
the model learns from passages that look relevant but are not.
- Beat BM25 top-20 accuracy by 9–19 points on open-domain QA with only a few thousand
  training pairs.
- Retrieval with FAISS over 21M Wikipedia passages; showed end-to-end QA improves when
  retrieval improves.
- Weakness admitted in the paper: fails on rare entities and exact strings where BM25 wins:
the argument for hybrid search.
**Interview line:** "DPR showed learned dense retrievers beat BM25 on semantic matching but
lose on rare exact terms; hybrid search takes both."

### 5. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (the RAG paper)
Lewis et al., NeurIPS 2020: [arXiv 2005.11401](https://arxiv.org/abs/2005.11401)
*The paper that named the technique.*
- Combine a **parametric** memory (a seq2seq generator, BART) with a **non-parametric**
  memory (a DPR index over Wikipedia) and train them **jointly** end-to-end.
- Retrieved documents are treated as a **latent variable**: marginalise over the top-k
  passages, either per sequence (RAG-Sequence) or per token (RAG-Token).
- The generator learned to use retrieved text; retriever gradients flowed through the
  query encoder (the document index was frozen).
- Outperformed parametric-only models on open-domain QA and produced more specific,
  factual language; knowledge could be **updated by swapping the index** without retraining.
- Modern RAG is the *frozen* version: no joint training, a prompt instead of marginalisation:
but the framing (parametric + non-parametric memory) is the same.
**Interview line:** "Lewis et al. 2020: pair a generator with a retriever over an external
index so knowledge lives in the index, not the weights, and can be swapped."

### 6. REALM
Guu et al., ICML 2020: [arXiv 2002.08909](https://arxiv.org/abs/2002.08909)
*Retrieval as part of pre-training.*
- Adds a latent **knowledge retriever** to masked-LM pre-training; the model learns to
  retrieve documents that help predict masked tokens: trained from unsupervised text only.
- Backpropagates through retrieval by treating the retrieved document as a latent variable
  over millions of candidates (using MIPS for the top-k).
- Index must be **refreshed asynchronously** during training as the encoder changes: the
  engineering problem that made joint training rare in practice.
- Large gains on open-domain QA over parametric models of the same size; also
  interpretable (you can see what was retrieved) and modular (update the corpus).
**Interview line:** "REALM taught the retriever during pre-training; the stale-index problem
is why most systems today freeze the retriever."

### 7. Fusion-in-Decoder (FiD)
Izacard & Grave, EACL 2021: [arXiv 2007.01282](https://arxiv.org/abs/2007.01282)
*How to use many passages at once.*
- Encode each retrieved passage **independently** with the question (encoder), then let the
  **decoder attend across all** encoded passages jointly.
- Encoding cost is linear in the number of passages (not quadratic), so it scaled to 100
  passages: and accuracy kept improving with more passages.
- Simple generative reader beat extractive readers on NaturalQuestions and TriviaQA.
- Conceptual ancestor of "stuff k chunks in the prompt": aggregation of evidence happens in
  the decoder.
**Interview line:** "FiD showed the reader improves as you give it more passages if it can
attend across them: the argument for retrieval recall over precision."

---

## Tier 2: Retrieval methods (the machinery)

### 8. The Probabilistic Relevance Framework: BM25 and Beyond
Robertson & Zaragoza, Foundations and Trends in IR, 2009:
[PDF (author copy)](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
*The definitive explanation of BM25.*
- Ranking as probability of relevance; derives BM25 from the binary independence model
  with term-frequency saturation and document-length normalisation.
- Score = Σ_t IDF(t) · tf·(k₁+1) / (tf + k₁·(1−b+b·|d|/avgdl)); **k₁ ≈ 1.2–2**, **b ≈ 0.75**.
- IDF down-weights common terms; saturation means the 10th occurrence adds little; `b`
  controls how much long documents are penalised.
- BM25F extends it to fielded documents (title vs body weights).
- Remains the strongest zero-training baseline and the standard sparse half of hybrid search.
**Interview line:** "BM25 = IDF × saturated TF × length normalisation; two parameters,
no training, unbeatable on rare exact terms."

### 9. HNSW
Malkov & Yashunin, 2016 (IEEE TPAMI 2018): [arXiv 1603.09320](https://arxiv.org/abs/1603.09320)
*The index inside Qdrant, Weaviate, pgvector, FAISS-HNSW and most others.*
- Proximity graph where each node links to ~`M` neighbours; **multiple layers**, upper layers
  sparse (long jumps), bottom layer contains all points (short jumps): like a skip list.
- Search: greedy descent from the top layer's entry point; at the bottom, best-first search
  with a candidate heap of size **`ef`**; return top-k.
- Insert: same search to find neighbours, connect with a heuristic that keeps links
  diverse; `ef_construction` controls build quality.
- **Logarithmic** query complexity in practice; recall tunable via `ef` at query time
  without rebuilding.
- Memory: vectors + ~`M`×2 links per node; incremental inserts supported, deletes are
  awkward (tombstones).
**Interview line:** "HNSW is a layered proximity graph; `M` trades memory for recall,
`ef` trades latency for recall at query time."

### 10. ColBERT and ColBERTv2
Khattab & Zaharia, SIGIR 2020: [arXiv 2004.12832](https://arxiv.org/abs/2004.12832);
Santhanam et al., NAACL 2022: [arXiv 2112.01488](https://arxiv.org/abs/2112.01488)
*Late interaction: one vector per token.*
- Encode query and passage separately into **per-token** embeddings (128 dims each).
- Score = **MaxSim**: for each query token, max cosine over passage tokens; sum over query
  tokens. Cheap enough to precompute passage side and index it.
- Near cross-encoder quality at bi-encoder-like latency; supports end-to-end retrieval
  with an ANN index over token vectors followed by exact MaxSim.
- v2: **residual compression** (centroid id + quantised residual) cuts index size 6–10×;
  denoised supervision via distillation from a cross-encoder.
- Basis of ColPali (images) and of Qdrant's multivector `MAX_SIM` support.
**Interview line:** "Keep token vectors, score with MaxSim: most of the cross-encoder's
precision at a fraction of the cost, for ~100× the storage."

### 11. Contriever
Izacard et al., 2021: [arXiv 2112.09118](https://arxiv.org/abs/2112.09118)
*Dense retrieval without labelled pairs.*
- **Unsupervised contrastive** training: two random spans from the same document are
  positives, spans from other documents are negatives (large batch, MoCo-style queue).
- Competitive with BM25 zero-shot across BEIR and much better after light fine-tuning;
  strong multilingual and cross-lingual transfer.
- Showed that the *pretraining recipe* matters more than the labels: the template for E5,
  BGE, GTE and every open embedding model since.
**Interview line:** "Contriever: contrastive learning on random spans from the same document
gives you a retriever with no labels."

### 12. E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training
Wang et al., 2022: [arXiv 2212.03533](https://arxiv.org/abs/2212.03533)
*The recipe behind modern general-purpose embedding models.*
- Curated **CCPairs**: ~270M weakly labelled text pairs (post–comment, title–passage,
  question–answer) filtered by consistency.
- Two-stage: contrastive pre-training on pairs with in-batch negatives, then fine-tuning
  with hard negatives and knowledge distillation.
- **"query: " / "passage: " prefixes** so the same model embeds questions and documents
  asymmetrically: why many APIs have an `input_type`.
- First model to beat BM25 on BEIR zero-shot; strong on MTEB across tasks.
**Interview line:** "E5's prefixes are why you set 'query' vs 'document' type when embedding;
the model was trained to treat them differently."

### 13. Matryoshka Representation Learning (MRL)
Kusupati et al., NeurIPS 2022: [arXiv 2205.13147](https://arxiv.org/abs/2205.13147)
*Why you can truncate an embedding.*
- Train so that the **first d dimensions** (for several d: 8, 16, …, full) each form a good
  embedding: losses applied at every prefix length, like nested dolls.
- One model serves many budgets: truncate to 256 dims for coarse search, rescore with the
  full vector: "adaptive retrieval" with up to 14× fewer FLOPs at the same accuracy.
- Adopted by OpenAI text-embedding-3 (`dimensions` parameter), Voyage, Nomic, Qwen3.
- Combines with quantisation for large memory savings in vector DBs.
**Interview line:** "MRL puts the most information in the leading dimensions so you can
truncate 1,536 → 512 with small loss and 3× less RAM."

### 14. Billion-scale similarity search with GPUs (FAISS)
Johnson, Douze, Jégou, 2017: [arXiv 1702.08734](https://arxiv.org/abs/1702.08734)
*Product quantisation and IVF at scale.*
- **IVF**: coarse k-means quantiser; search only `nprobe` nearest cells.
- **Product quantisation (PQ)**: split a vector into m sub-vectors, quantise each to 256
  centroids → 1 byte per sub-vector; distances via lookup tables (asymmetric distance).
- GPU implementation of k-selection and IVF-PQ; indexed 1B vectors on 8 GPUs.
- The compression ideas (scalar/product quantisation, rescoring with exact vectors) are what
  Qdrant's quantisation config exposes.
**Interview line:** "PQ compresses vectors to a few bytes via sub-vector codebooks; pair with
IVF or HNSW and rescore the top candidates exactly."

### 15. Reciprocal Rank Fusion (RRF)
Cormack, Clarke & Buettcher, SIGIR 2009:
[PDF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
*The two-line algorithm behind hybrid search.*
- Fuse ranked lists by **score(d) = Σ_lists 1/(k + rank_list(d))**, with **k = 60**.
- No score normalisation, no training; outperformed Condorcet fusion and learned
  combinations in their tests.
- Robust because ranks are comparable across systems while raw scores are not.
- Qdrant implements it server-side (`RrfQuery`); LangChain's `EnsembleRetriever` does the
  same client-side.
**Interview line:** "RRF sums 1/(60+rank) across lists: rank-based, scale-free, and the
default fusion for dense + BM25."

---

## Tier 3: Making RAG better (2021–2024)

### 16. Improving language models by retrieving from trillions of tokens (RETRO)
Borgeaud et al., DeepMind, 2021: [arXiv 2112.04426](https://arxiv.org/abs/2112.04426)
- Retrieval **inside the transformer**: chunked cross-attention to neighbours fetched from a
  2-trillion-token database by frozen BERT embeddings.
- A 7B RETRO matched GPT-3 175B on some benchmarks: retrieval as a substitute for
  parameters.
- Showed memorisation vs generalisation split: gains largest on test data overlapping the
  database.
**Interview line:** "RETRO: retrieval can substitute for ~25× parameters when the knowledge
is in the database."

### 17. Atlas: Few-shot Learning with Retrieval Augmented Language Models
Izacard et al., 2022: [arXiv 2208.03299](https://arxiv.org/abs/2208.03299)
- FiD reader + Contriever retriever, **jointly trained** with retriever supervision derived
  from the reader (which passages helped).
- 11B Atlas beat 540B PaLM on few-shot NaturalQuestions with 64 examples.
- Studied index updating and temporal knowledge; showed you can swap the index to update
  facts.
**Interview line:** "Atlas: a small model with a good retriever beats a huge model without one."

### 18. In-Context Retrieval-Augmented Language Models (In-context RALM)
Ram et al., TACL 2023: [arXiv 2302.00083](https://arxiv.org/abs/2302.00083)
- Simply **prepend retrieved documents to the prompt** of a frozen LM (no architecture
  change, no training) and perplexity improves substantially.
- Even **BM25** retrieval gave large gains; a reranker trained for the LM gave more.
- Legitimised the "frozen LLM + retrieved context" pattern everyone uses.
**Interview line:** "In-context RALM showed that just putting retrieved text in the prompt
of a frozen model works: that is modern RAG."

### 19. Lost in the Middle
Liu et al., TACL 2024: [arXiv 2307.03172](https://arxiv.org/abs/2307.03172)
- Multi-document QA where the answer's position among the documents is varied: accuracy is
  **U-shaped**: high when the answer is first or last, low in the middle.
- Holds for GPT-3.5/4, Claude, and long-context open models; more documents → worse.
- Models that claim long context do not use it uniformly.
- Practical consequence: put the most relevant chunks at the **start and end**; keep k small;
  rerank.
**Interview line:** "Models attend to the edges of the context; order your chunks and keep k
small."

### 20. Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)
Gao et al., ACL 2023: [arXiv 2212.10496](https://arxiv.org/abs/2212.10496)
- Ask the LLM to write a **hypothetical answer** to the query, embed *that*, and search:
answer-to-answer similarity is easier than question-to-answer.
- Zero-shot, no training; strong gains on out-of-domain and low-resource settings.
- Costs an LLM call per query and can hallucinate the wrong domain; average the hypothetical
  with the raw query embedding to hedge.
**Interview line:** "HyDE embeds a hypothetical answer instead of the question; a query
transform that helps when questions and passages are phrased very differently."

### 21. Query Rewriting for Retrieval-Augmented LLMs (Rewrite-Retrieve-Read)
Ma et al., EMNLP 2023: [arXiv 2305.14283](https://arxiv.org/abs/2305.14283)
- Insert a **rewriter** between the user query and the retriever; the reader is frozen.
- A small rewriter trained with RL from the reader's answer quality beats prompting the LLM
  to rewrite.
- Formalised the family: multi-query, step-back, decomposition, HyDE are all "rewrite".
**Interview line:** "Rewrite-Retrieve-Read: fix the query before retrieval; train or prompt
a rewriter."

### 22. Active Retrieval Augmented Generation (FLARE)
Jiang et al., EMNLP 2023: [arXiv 2305.06983](https://arxiv.org/abs/2305.06983)
- Retrieve **during** generation, not just before: generate a tentative next sentence; if it
  contains low-confidence tokens, use it as a query, retrieve, and regenerate.
- Helps long-form generation where information needs change across the answer.
- Ancestor of agentic retrieval loops: retrieval triggered by the model's own uncertainty.
**Interview line:** "FLARE retrieves when the model is unsure mid-generation: retrieval as
an on-demand action."

### 23. Self-RAG
Asai et al., ICLR 2024: [arXiv 2310.11511](https://arxiv.org/abs/2310.11511)
- Train the LM to emit **reflection tokens**: `Retrieve?`, `IsRel` (passage relevant),
  `IsSup` (answer supported), `IsUse` (answer useful).
- The model decides whether to retrieve, critiques each passage and its own output, and
  selects the best continuation via the tokens' scores.
- Beat ChatGPT and retrieval-augmented Llama2-chat on QA, reasoning and long-form factuality.
- In practice implemented with prompts/graders instead of special tokens: Chapter 15.
**Interview line:** "Self-RAG: the model grades relevance and support itself and decides
when to retrieve."

### 24. Corrective RAG (CRAG)
Yan et al., 2024: [arXiv 2401.15884](https://arxiv.org/abs/2401.15884)
- A lightweight **retrieval evaluator** scores retrieved documents as
  **Correct / Incorrect / Ambiguous**.
- Correct → refine (decompose into knowledge strips, keep relevant ones); Incorrect → fall
  back to **web search** with a rewritten query; Ambiguous → both.
- Plug-in: works with any generator; consistent gains across four datasets.
**Interview line:** "CRAG grades retrieval and takes a corrective action (refine, web search,
or both) before generating."

### 25. Adaptive-RAG
Jeong et al., NAACL 2024: [arXiv 2403.14403](https://arxiv.org/abs/2403.14403)
- A small **classifier predicts query complexity** and routes to: no retrieval, single-step
  retrieval, or multi-step (iterative) retrieval.
- Matches the multi-step system's accuracy at much lower average cost, because most
  questions are simple.
- The pattern behind "router" nodes in production graphs.
**Interview line:** "Adaptive-RAG routes by question complexity so easy questions do not pay
for the agent loop."

### 26. Chain-of-Note
Yu et al., 2023: [arXiv 2311.09210](https://arxiv.org/abs/2311.09210)
- Before answering, the model writes a **reading note** per retrieved document: what it
  says and whether it is relevant.
- Improves robustness to **noisy** and **irrelevant** retrieval and lets the model answer
  "unknown" when nothing is relevant.
- Simple prompt-level technique; large gains on noisy-retrieval benchmarks.
**Interview line:** "Chain-of-Note: make the model annotate each passage first: cheap
robustness to bad retrieval and a natural 'I don't know'."

### 27. RAPTOR
Sarthi et al., ICLR 2024: [arXiv 2401.18059](https://arxiv.org/abs/2401.18059)
- Build a **tree**: cluster chunks by embedding, summarise each cluster with an LLM, embed
  the summaries, cluster again: up to a root.
- Retrieve across **all levels** (collapsed tree), so questions needing a whole-document
  view hit summaries while detail questions hit leaves.
- Big gains on questions requiring multi-step reasoning over long documents (QuALITY).
- The "hierarchical index" idea reappears in GraphRAG's community summaries.
**Interview line:** "RAPTOR indexes summaries of clusters of chunks recursively so you can
retrieve at the right granularity."

### 28. Retrieval-Augmented Generation for LLMs: A Survey
Gao et al., 2023 (updated 2024): [arXiv 2312.10997](https://arxiv.org/abs/2312.10997)
- The taxonomy everyone quotes: **Naive RAG** (retrieve–read), **Advanced RAG**
  (pre-retrieval query optimisation, post-retrieval reranking/compression), **Modular RAG**
  (routing, memory, fusion, iterative/adaptive retrieval).
- Catalogues retrieval sources, granularity (token/phrase/chunk/document/graph), indexing
  tricks, query transforms, fusion, fine-tuning of retriever and generator.
- Evaluation section lists retrieval and generation metrics and benchmarks (RGB, RECALL,
  CRUD).
**Interview line:** "Naive → Advanced → Modular RAG is the survey's taxonomy; Modular is
what agentic systems are."

### 29. Promptriever: Instruction-Trained Retrievers
Weller et al., 2024: [arXiv 2409.11136](https://arxiv.org/abs/2409.11136)
- Train a bi-encoder on **instruction–query–passage** triples so the retriever follows
  natural-language instructions ("find documents that argue against…").
- Robust to phrasing changes; can be *prompted* like an LLM.
- Precursor to reasoning-aware retrievers (ReasonIR, 2025) and instruction fields in
  commercial embedding APIs.
**Interview line:** "Instruction-trained retrievers let the embedding follow the task
instruction, not just the topic."

---

## Tier 4: Agents and evaluation

### 30. ReAct
Yao et al., ICLR 2023: [arXiv 2210.03629](https://arxiv.org/abs/2210.03629)
- Interleave **reasoning traces** ("Thought") with **actions** (tool calls such as
  `Search[...]`, `Lookup[...]`) and **observations** in one prompt loop.
- Reasoning helps choose actions; actions ground reasoning in real information; fewer
  hallucinated facts than chain-of-thought alone.
- The loop every agent framework implements; LangGraph's agent is ReAct with structure.
**Interview line:** "ReAct = think, act, observe, repeat: the agent loop."

### 31. Toolformer
Schick et al., 2023: [arXiv 2302.04761](https://arxiv.org/abs/2302.04761)
- LM **teaches itself** to call APIs (calculator, search, QA, translation) by inserting
  call tokens into text and keeping the ones that reduce perplexity on the continuation.
- Self-supervised: no human tool-use annotations.
- Showed tool use is learnable and improves factuality of small models.
**Interview line:** "Toolformer: models can learn when a tool call helps from self-supervision."

### 32. Reflexion
Shinn et al., NeurIPS 2023: [arXiv 2303.11366](https://arxiv.org/abs/2303.11366)
- After a failed attempt, the agent writes a **verbal reflection** ("I failed because…")
  stored in an episodic memory and used in the next attempt.
- "Verbal reinforcement learning": no weight updates, large gains on coding and reasoning
  tasks with a few retries.
- The template for retry-with-feedback loops in RAG agents.
**Interview line:** "Reflexion stores natural-language lessons between attempts: memory as
the learning signal."

### 33. Ragas
Es et al., 2023: [arXiv 2309.15217](https://arxiv.org/abs/2309.15217)
- **Reference-free** RAG metrics via an LLM: **faithfulness** (fraction of answer claims
  supported by context), **answer relevance** (generate questions from the answer, compare
  to the real question), **context relevance/precision** (fraction of context sentences
  needed).
- Metrics correlate with human judgement better than GPT-scored baselines.
- Defined the vocabulary the whole industry uses; this book implements the metrics by hand
  (Chapter 10) because the library is version-fragile.
**Interview line:** "Ragas: faithfulness = supported claims / total claims, judged by an
LLM, no reference answer needed."

### 34. ARES
Saad-Falcon et al., NAACL 2024: [arXiv 2311.09476](https://arxiv.org/abs/2311.09476)
- Train **small judge models** (fine-tuned classifiers) on **synthetic** query–passage–answer
  data to score context relevance, answer faithfulness and answer relevance.
- Uses a small **human-labelled validation set** with prediction-powered inference to
  produce confidence intervals: calibrated automatic evaluation.
- Cheaper than an LLM judge per call; more setup.
**Interview line:** "ARES trains cheap judges on synthetic data and calibrates them against
~150 human labels."

### 35. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
Zheng et al., NeurIPS 2023: [arXiv 2306.05685](https://arxiv.org/abs/2306.05685)
- Strong LLMs agree with human preferences >80%: as much as humans agree with each other.
- Documented biases: **position** (first answer preferred), **verbosity**, **self-enhancement**,
  limited maths/reasoning grading ability.
- Mitigations: swap positions, reference-guided grading, chain-of-thought judging.
**Interview line:** "LLM judges match humans ~80%, but swap answer order and use rubrics to
counter position and verbosity bias."

### 36. FActScore
Min et al., EMNLP 2023: [arXiv 2305.14251](https://arxiv.org/abs/2305.14251)
- Break long-form generations into **atomic facts**; check each against a knowledge source;
  score = fraction supported.
- Human evaluation showed even GPT-4 had many unsupported facts in biographies; automated
  FActScore within ~2% of human error rate.
- The claim-decomposition method behind faithfulness metrics and RAGChecker.
**Interview line:** "FActScore: decompose into atomic claims and verify each: precision of
facts, not vibes."

### 37. RAGChecker
Ru et al., NeurIPS 2024: [arXiv 2408.08067](https://arxiv.org/abs/2408.08067)
- Claim-level checking of answer vs ground truth **and** vs retrieved context, producing
  **diagnostic** metrics: context precision/recall, faithfulness, noise sensitivity,
  hallucination rate, self-knowledge, context utilisation.
- Attributes errors to **retriever vs generator**; validated against human judgements.
- Insight from their benchmark: generators are sensitive to noise; better retrieval helps
  more than bigger generators up to a point.
**Interview line:** "RAGChecker tells you whether the retriever or the generator is at
fault, claim by claim."

---

## Tier 5: Frontier (2024–2026)

### 38. From Local to Global: A Graph RAG Approach to Query-Focused Summarization (GraphRAG)
Edge et al., Microsoft, 2024: [arXiv 2404.16130](https://arxiv.org/abs/2404.16130)
- LLM extracts **entities and relationships** from every chunk → knowledge graph →
  **Leiden** community detection → LLM-written **community summaries** at multiple levels.
- **Global** questions answered by map-reduce over community summaries; large gains in
  comprehensiveness and diversity vs vector RAG on sensemaking questions.
- Indexing cost (an LLM pass over the whole corpus plus summaries) was the objection;
  **LazyGraphRAG** (Nov 2024, Microsoft blog) removed it by deferring LLM work to query time.
**Interview line:** "GraphRAG answers 'what are the themes' questions via community
summaries; LazyGraphRAG made indexing cost the same as vector RAG."

### 39. HippoRAG and LightRAG
Gutiérrez et al., NeurIPS 2024: [arXiv 2405.14831](https://arxiv.org/abs/2405.14831);
Guo et al., 2024: [arXiv 2410.05779](https://arxiv.org/abs/2410.05779)
- HippoRAG: knowledge graph + **Personalized PageRank** from query entities, inspired by
  hippocampal indexing; strong on multi-hop QA at lower cost than iterative retrieval.
- LightRAG: **dual-level** retrieval (specific entities and abstract topics) over a graph
  combined with vectors; **incremental** index updates; much cheaper than GraphRAG.
**Interview line:** "Open graph-RAG alternatives: HippoRAG (PageRank over entities),
LightRAG (dual-level, incremental)."

### 40. When to use Graphs in RAG
Xiang et al., 2025: [arXiv 2506.05690](https://arxiv.org/abs/2506.05690)
- Systematic comparison: graph methods help on **multi-hop, relational and global**
  questions; **hurt** on simple factoid questions and add cost.
- Proposes GraphRAG-Bench and analyses when graph structure is actually used.
**Interview line:** "Benchmark first: graphs pay off on relational/global questions only."

### 41. ColPali
Faysse et al., 2024 (ICLR 2025): [arXiv 2407.01449](https://arxiv.org/abs/2407.01449)
- Embed **page images** with a vision-language model (PaliGemma) into ~1,000 patch vectors;
  score queries with **MaxSim** (ColBERT-style): no OCR, no layout parsing.
- Introduced **ViDoRe**, the visual document retrieval benchmark; ColPali beat OCR+text
  pipelines substantially and indexed faster.
- Successors ColQwen2/ColQwen3; production via Qdrant multivectors with pooled prefetch.
**Interview line:** "ColPali retrieves PDF pages as images with late interaction: tables and
charts without OCR."

### 42. Late Chunking
Günther et al., Jina AI, 2024: [arXiv 2409.04701](https://arxiv.org/abs/2409.04701)
- Embed the **whole document** with a long-context embedding model, then split the
  **token embeddings** into chunks and mean-pool: each chunk vector carries document context.
- No LLM calls, no text changes; consistent retrieval gains over naive chunking.
- Requires access to token-level embeddings (open models; not the OpenAI API).
**Interview line:** "Late chunking: embed first, chunk second."

### 43. Contextual Retrieval (Anthropic engineering post, Sep 2024: a blog, not a paper)
[anthropic.com/engineering/contextual-retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
- Prepend an LLM-generated **context sentence** to each chunk before embedding and BM25
  indexing.
- Reported: top-20 retrieval failure rate −35% (embeddings only), **−49%** with contextual
  BM25, **−67%** adding a reranker.
- Cheap with prompt caching (the document is the cached prefix).
**Interview line:** "Contextual retrieval: one LLM sentence per chunk at index time cut
retrieval failures by half."

### 44. RAG or Long-Context LLMs? (Self-Route) and LongRAG
Li et al., 2024: [arXiv 2407.16833](https://arxiv.org/abs/2407.16833);
Jiang et al., 2024: [arXiv 2406.15319](https://arxiv.org/abs/2406.15319)
- Long-context beats RAG on average when the corpus fits, at far higher cost; **Self-Route**
  lets the model decide per query whether retrieved chunks suffice: long-context quality
  at RAG-like cost.
- LongRAG: retrieve **long units** (4K-token groups), far fewer of them, and read with a
  long-context model; recall and answer quality both rise.
**Interview line:** "Hybrid wins: retrieve long units, read with long context, route per query."

### 45. In Defense of RAG in the Era of Long-Context Language Models
Yu et al., 2024: [arXiv 2409.01666](https://arxiv.org/abs/2409.01666)
- Ordering retrieved chunks by **original document position** instead of relevance
  (order-preserve RAG) beats long-context models on the same benchmarks.
- Too many chunks hurts (inverted-U); the sweet spot is well below the context limit.
**Interview line:** "Keep chunks in document order and keep k modest: RAG beats stuffing."

### 46. Agentic RAG surveys and SoK
Singh et al., 2025: [arXiv 2501.09136](https://arxiv.org/abs/2501.09136);
Li et al., 2025: [arXiv 2507.09477](https://arxiv.org/abs/2507.09477);
SoK, 2026: [arXiv 2603.07379](https://arxiv.org/abs/2603.07379)
- Taxonomy: single-agent, multi-agent, hierarchical, corrective, adaptive, graph-based
  agentic RAG; reasoning-integrated RAG.
- SoK (Mar 2026) frames agentic RAG as **sequential decision making** (state, actions
  (retrieve/tool/answer), reward (quality − cost)) and surveys evaluation gaps.
**Interview line:** "Agentic RAG = retrieval as an action in a decision loop; the surveys give
the taxonomy."

### 47. Search-R1, R1-Searcher, Search-o1
[arXiv 2503.09516](https://arxiv.org/abs/2503.09516), [arXiv 2503.05592](https://arxiv.org/abs/2503.05592),
[arXiv 2501.05366](https://arxiv.org/abs/2501.05366): all 2025
- Train LLMs with **reinforcement learning** (outcome reward only) to interleave reasoning
  and search-engine calls; the model learns *when* to search and how to reformulate.
- Search-o1 adds a **reason-in-documents** module so retrieved text is condensed before it
  enters the reasoning chain.
- Large gains on multi-hop QA over prompting-based agents.
**Interview line:** "2025: RL taught models to search: retrieval behaviour is now learned,
not scripted."

### 48. A-RAG: Hierarchical Retrieval Interfaces
2026: [arXiv 2602.03442](https://arxiv.org/abs/2602.03442)
- Expose **three tools** to the agent: keyword search, semantic search, chunk read
  (neighbourhood): instead of one flat search.
- The agent composes cheap tools first and escalates; better accuracy per token than
  fixed pipelines.
**Interview line:** "Give the agent a hierarchy of retrieval tools, not one search box."

### 49. PoisonedRAG and Securing RAG
Zou et al., 2024 (USENIX Security 2025): [arXiv 2402.07867](https://arxiv.org/abs/2402.07867);
taxonomy, 2026: [arXiv 2604.08304](https://arxiv.org/abs/2604.08304)
- **Five** injected passages per target question in a corpus of millions → attacker-chosen
  answers ~90% of the time, black-box.
- Two conditions: the passage must be *retrieved* (semantically close to the question) and
  must *steer* generation.
- 2026 taxonomy maps attacks/defences across ingestion, retrieval, context assembly,
  generation and delivery.
**Interview line:** "A handful of poisoned documents can hijack RAG; treat retrieved text as
untrusted input and defend at every stage."

### 50. Qwen3 Embedding and Gemini Embedding
Zhang et al., 2025: [arXiv 2506.05176](https://arxiv.org/abs/2506.05176);
Lee et al., 2025: [arXiv 2503.07891](https://arxiv.org/abs/2503.07891)
- Qwen3: open **embedding + reranker** family (0.6B/4B/8B), multi-stage training with
  synthetic data from the LLM itself, instruction-aware, MRL dims; top of open MTEB.
- Gemini Embedding: distilled from Gemini, top of MTEB(Multilingual) at release; one
  model, 100+ languages, code.
**Interview line:** "2025 embedding models are LLM-derived, instruction-aware and
Matryoshka: pick by your own eval, not the leaderboard."

### 51. When Retrieval Succeeds and Fails
2025: [arXiv 2510.09106](https://arxiv.org/abs/2510.09106)
- A review, not a benchmark: as LLMs grow, the advantage of *traditional* RAG shrinks;
  the paper catalogues RAG's weak points (retrieval noise, integration, evaluation) and the
  cases where LLMs alone still fail and RAG clearly helps: fresh and domain-specific
  knowledge, sparse evidence a strong model does not hold in its weights.
**Interview line:** "RAG's edge over a bare LLM is now conditional (fresh, domain-specific,
sparse facts) so measure whether retrieval helps on your questions."

### 52. Deep Research: A Survey of Autonomous Research Agents
2025: [arXiv 2508.12752](https://arxiv.org/abs/2508.12752)
- Architecture of deep-research systems: planning, iterative search, reading, note-taking,
  reflection, report synthesis with citations; evaluation challenges.
**Interview line:** "Deep research = agentic RAG with a long loop, a scratchpad and a budget."

### 53. MemGPT (memory as paging)
Packer et al., 2023: [arXiv 2310.08560](https://arxiv.org/abs/2310.08560)
- Treat the context window as **main memory** and external stores as **disk**; the model
  issues function calls to page information in and out.
- Enables unbounded conversations and document analysis; the template for agent memory
  products.
**Interview line:** "MemGPT: the LLM as an OS managing its own context via retrieval and writes."

---

## The 2-week reading plan (≈ 20 minutes/day, this appendix only)

| Day | Papers | Focus |
|---|---|---|
| 1 | 1, 2 | attention; encoders vs decoders |
| 2 | 3, 4 | bi-encoders, in-batch negatives, dense vs BM25 |
| 3 | 5, 6, 7 | the RAG paper; joint training; fusion-in-decoder |
| 4 | 8, 15 | BM25 formula; RRF: write both from memory |
| 5 | 9, 14 | HNSW parameters; IVF/PQ |
| 6 | 10, 11, 12, 13 | late interaction; embedding training recipes; MRL |
| 7 | 16, 17, 18, 19 | retrieval vs parameters; lost in the middle |
| 8 | 20, 21, 22 | query transforms; active retrieval |
| 9 | 23, 24, 25, 26, 27 | Self-RAG, CRAG, Adaptive, Chain-of-Note, RAPTOR |
| 10 | 28, 29, 30, 31, 32 | the survey taxonomy; agents |
| 11 | 33, 34, 35, 36, 37 | evaluation: derive faithfulness and nDCG yourself |
| 12 | 38, 39, 40, 41, 42, 43 | graphs; visual retrieval; contextual retrieval |
| 13 | 44, 45, 46, 47, 48 | long context vs RAG; agentic RAG; RL search |
| 14 | 49, 50, 51, 52, 53 | security; embeddings 2025; memory: then reread Chapter 20 |

## If you only have time for 10

1. **RAG (Lewis 2020)**: the framing.
2. **DPR**: dense retrieval and hard negatives.
3. **HNSW**: the index you will be asked to explain.
4. **BM25 (Robertson & Zaragoza)** + **RRF**: the sparse half and how to fuse.
5. **ColBERT**: late interaction; unlocks ColPali.
6. **Lost in the Middle**: why k and order matter.
7. **Self-RAG** (with CRAG as a footnote): the agentic loop.
8. **Ragas** + **Judging LLM-as-a-Judge**: evaluation and its biases.
9. **GraphRAG**: the alternative you must be able to position.
10. **Contextual Retrieval**: the cheapest big retrieval win of the last two years.
