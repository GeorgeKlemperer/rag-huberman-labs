import os, re, math
import numpy as np

DOCS = "cleaned-test-transcripts"


def load_and_chunk(directory, chunk_size=500, overlap=50):
    chunks = []
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith((".txt", ".md")):
            continue
        text = open(os.path.join(directory, fn)).read()
        for s in range(0, len(text), chunk_size - overlap):
            ct = text[s:s + chunk_size]
            if ct.strip():
                chunks.append({"text": ct, "source": fn, "start": s})
    return chunks


def _tok(t):
    return re.findall(r"[a-z0-9]+", t.lower())


class Embedder:
    def __init__(self, ngram=1):
        self.ngram = ngram

    def _feats(self, t):
        ws = _tok(t)
        f = list(ws)
        if self.ngram >= 2:
            f += [ws[i] + "_" + ws[i + 1] for i in range(len(ws) - 1)]
        return f

    def fit(self, texts):
        df = {}
        for t in texts:
            for w in set(self._feats(t)):
                df[w] = df.get(w, 0) + 1
        self.vocab = {w: i for i, w in enumerate(df)}
        self.idf = np.zeros(len(self.vocab), dtype=np.float32)
        for w, i in self.vocab.items():
            self.idf[i] = math.log((1 + len(texts)) / (1 + df[w])) + 1.0
        return self

    def encode(self, texts, show_progress_bar=False):
        V = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for r, t in enumerate(texts):
            for w in self._feats(t):
                j = self.vocab.get(w)
                if j is not None:
                    V[r, j] += 1.0
            V[r] *= self.idf
            n = np.linalg.norm(V[r])
            if n > 0:
                V[r] /= n
        return V


def embed_chunks(chunks, model):
    return model.encode([c["text"] for c in chunks], show_progress_bar=True)


def build_corpus(directory, chunk_size=500, ngram=1):
    chunks = load_and_chunk(directory, chunk_size=chunk_size)
    model = Embedder(ngram=ngram).fit([c["text"] for c in chunks])
    return chunks, embed_chunks(chunks, model), model


def make_retriever(chunks, embeddings, model):
    def retrieve(query, k):
        q = model.encode([query])[0]
        sims = embeddings @ q
        return [int(i) for i in np.argsort(sims)[::-1][:k]]
    return retrieve


def top_similarity(query, embeddings, model):
    return float(np.max(embeddings @ model.encode([query])[0]))


def precision_at_k(retrieved, relevant):
    retrieved, relevant = set(retrieved), set(relevant)
    return 0.0 if not retrieved else len(retrieved & relevant) / len(retrieved)


def max_precision_at_k(retrieved, relevant):
    retrieved, relevant = set(retrieved), set(relevant)
    if not retrieved:
        return 0.0
    return min(len(retrieved), len(relevant)) / len(retrieved)


def normalized_precision_at_k(retrieved, relevant):
    p = precision_at_k(retrieved, relevant)
    p_max = max_precision_at_k(retrieved, relevant)
    return 0.0 if p_max == 0 else p / p_max


def recall_at_k(retrieved, relevant):
    retrieved, relevant = set(retrieved), set(relevant)
    return 0.0 if not relevant else len(retrieved & relevant) / len(relevant)


def reciprocal_rank(retrieved, relevant):
    relevant = set(relevant)
    for i, d in enumerate(retrieved):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_retrieval(eval_set, retrieve_fn, k=5):
    res = []
    for it in eval_set:
        got = retrieve_fn(it["query"], k)
        rel = it["relevant_chunk_ids"]
        res.append({"query": it["query"], "type": it.get("type", ""),
                    "precision": precision_at_k(got, rel),
                    "normalized_precision": normalized_precision_at_k(got, rel),
                    "recall": recall_at_k(got, rel), "rr": reciprocal_rank(got, rel),
                    "retrieved": got, "relevant": rel})
    return {"per_query": res, "avg_precision": float(np.mean([r["precision"] for r in res])),
            "avg_normalized_precision": float(np.mean([r["normalized_precision"] for r in res])),
            "avg_recall": float(np.mean([r["recall"] for r in res])),
            "mrr": float(np.mean([r["rr"] for r in res])), "k": k}


def print_report(res, label=""):
    if label:
        print(label)
    print(f"  k={res['k']} queries={len(res['per_query'])} "
          f"P@k={res['avg_precision']:.3f} Pn@k={res['avg_normalized_precision']:.3f} "
          f"R@k={res['avg_recall']:.3f} MRR={res['mrr']:.3f}")


def print_per_query(res):
    print(f"  {'Query':<50}{'Type':>15}{'#rel':>5}{'P@k':>6}{'Pn@k':>7}{'R@k':>6}{'RR':>7}")
    for r in res["per_query"]:
        print(f"  {r['query'][:48]:<50}{r['type']:>15}{len(r['relevant']):>5}"
              f"{r['precision']:>6.2f}{r['normalized_precision']:>7.2f}"
              f"{r['recall']:>6.2f}{r['rr']:>7.2f}")


def print_by_type(res):
    types = {}
    for r in res["per_query"]:
        types.setdefault(r["type"], []).append(r)
    print(f"  {'Type':<16}{'n':>3}{'P@k':>7}{'Pn@k':>7}{'R@k':>7}{'MRR':>7}")
    for t, rows in sorted(types.items()):
        p = float(np.mean([x["precision"] for x in rows]))
        pn = float(np.mean([x["normalized_precision"] for x in rows]))
        rec = float(np.mean([x["recall"] for x in rows]))
        mrr = float(np.mean([x["rr"] for x in rows]))
        print(f"  {t:<16}{len(rows):>3}{p:>7.2f}{pn:>7.2f}{rec:>7.2f}{mrr:>7.2f}")


def _load_text(path):
    with open(path) as f:
        return f.read()


def _find_spans(text, phrases):
    lowered = text.lower()
    spans = []
    for phrase in phrases:
        start = lowered.find(phrase.lower())
        if start == -1:
            raise ValueError(f"Could not find evidence phrase {phrase!r} in source text")
        spans.append({"phrase": phrase, "start": start, "end": start + len(phrase)})
    return spans


def _chunk_overlaps_span(chunk, span):
    chunk_start = chunk["start"]
    chunk_end = chunk_start + len(chunk["text"])
    return max(chunk_start, span["start"]) < min(chunk_end, span["end"])


def print_eval_set(eval_set, chunks, max_text=140):
    print("\nEval set preview")
    for i, item in enumerate(eval_set, start=1):
        print(f"  {i}. query: {item['query']}")
        print(f"     type: {item.get('type', '')}")
        print(f"     source: {item['source']}")
        for span in item["evidence_spans"]:
            print(f"     evidence: {span['phrase']} @ {span['start']}..{span['end']}")
        print(f"     relevant_chunk_ids: {item['relevant_chunk_ids']}")
        for chunk_id in item["relevant_chunk_ids"]:
            chunk = chunks[chunk_id]
            text = chunk["text"].replace("\n", " ")
            if len(text) > max_text:
                text = text[:max_text - 3] + "..."
            print(f"     - chunk {chunk_id} ({chunk['source']} @ {chunk['start']}): {text}")


# In-corpus eval queries. Each item declares the query, the source transcript that
# contains the answer, the query type, and verbatim evidence_phrases used to anchor
# the gold span(s). Anchors are chosen to be unique (or nearly so) within the source
# so the ground truth stays narrow. Lean strict: only phrases that themselves contain
# the answer are anchored, not merely related context.
IN_CORPUS = [
    # --- Direct lookup: a specific fact stated plainly in one place ---
    {
        "query": "What does the HPA axis stand for?",
        "type": "direct_lookup",
        "source": "004 - Essentials Erasing Fears & Traumas Using Modern Neuroscience.txt",
        "evidence_phrases": ["HPA axis stands for hypothalamic-pituitary-adrenal axis"],
    },
    {
        "query": "What is the large lymphatic reservoir in the abdomen called?",
        "type": "direct_lookup",
        "source": "007 - Improve Your Lymphatic System for Overall Health & Appearance.txt",
        "evidence_phrases": ["large compartment that sits within your abdomen called the \"cisterna chyli.\""],
    },
    {
        "query": "What is the root word of amygdala?",
        "type": "direct_lookup",
        "source": "004 - Essentials Erasing Fears & Traumas Using Modern Neuroscience.txt",
        "evidence_phrases": ["Amygdala means almond"],
    },
    # --- Detail buried in context: an incidental fact inside a passage about something else ---
    {
        "query": "How often do humans naturally sigh?",
        "type": "detail_buried",
        "source": "002 - Essentials Breathing for Mental & Physical Health & Performance Dr. Jack Feldman.txt",
        "evidence_phrases": ["we sigh about every five minutes"],
    },
    {
        "query": "What is the maximum time delay between the two ears used to localize a sound?",
        "type": "detail_buried",
        "source": "003 - How Your Thoughts Are Built & How You Can Shape Them Dr. Jennifer Groh.txt",
        "evidence_phrases": ["about a half a millisecond, is the largest delay you can experience"],
    },
    {
        "query": "Who was the writer of Braveheart?",
        "type": "detail_buried",
        "source": "009 - How to Overcome Inner Resistance Steven Pressfield.txt",
        "evidence_phrases": ["Randall Wallace, who wrote \"Braveheart\""],
    },
    # --- Multi-hop: answer requires combining two facts from far-apart chunks ---
    {
        # Mom's cancer (~line 545) and his own remission drug (~line 863) sit in very
        # different parts of the transcript, so retrieval must surface two separate chunks.
        "query": "What type of brain cancer did Dr. Fajgenbaum's mother have, and what drug ultimately put his own disease into remission?",
        "type": "multi_hop",
        "source": "005 - Using Existing Drugs in New Ways to Treat & Cure Diseases of Brain & Body Dr. David Fajgenbaum.txt",
        "evidence_phrases": ["Glioblastoma brain tumors", "Since starting Rapamycin"],
    },
    {
        # Both facts are about Matt's self-review methods but live far apart in the
        # talk (~line 471 vs ~line 521), so one coherent question still needs two chunks.
        "query": "How does Matt Abrahams suggest reviewing a recording of yourself, and what daily habit does he keep to reflect on his own communication?",
        "type": "multi_hop",
        "source": "001 - How to Speak Clearly & With Confidence Matt Abrahams.txt",
        "evidence_phrases": ["watch it three times", "Every night before I go to bed"],
    },
    # --- Synthesis: one coherent question requiring multiple separated evidence spans ---
    {
        "query": "According to Huberman, what are some ineffective and effective forms of gratitude practice, and what rapid brain/immune changes does it produce?",
        "type": "synthesis",
        "source": "008 - Essentials The Science of Gratitude & How to Build a Gratitude Practice.txt",
        "evidence_phrases": [
            "that style of gratitude practice is not particularly effective in shifting your neural circuitry",
            "most potent form of gratitude practice is not a gratitude practice where you give gratitude or express gratitude, but rather where you receive gratitude",
            "reductions in amygdala activation, and large reductions in the production of something called \"TNF-alpha,\"",
        ],
    },
]

# Out-of-corpus queries: topics the corpus does not cover. A good system should surface
# nothing relevant (low top similarity) and the generator should refuse to answer.
OUT_OF_CORPUS = [
    "What's the population of Tokyo?",
    "Who won the 2022 FIFA World Cup?",
]


def build_eval_set(chunks, directory=DOCS):
    eval_set = []
    for item in IN_CORPUS:
        source_path = os.path.join(directory, item["source"])
        source_text = _load_text(source_path)
        evidence_spans = _find_spans(source_text, item["evidence_phrases"])
        relevant_chunk_ids = [i for i, c in enumerate(chunks)
                              if c["source"] == item["source"] and any(_chunk_overlaps_span(c, span) for span in evidence_spans)]
        eval_set.append({"query": item["query"], "type": item.get("type", ""), "source": item["source"],
                         "evidence_spans": evidence_spans, "relevant_chunk_ids": relevant_chunk_ids})
    return eval_set


if __name__ == "__main__":
    chunks, embeddings, model = build_corpus(DOCS)
    eval_set = build_eval_set(chunks)
    retrieve = make_retriever(chunks, embeddings, model)

    # Sanity check: every in-corpus query must anchor to at least one chunk, otherwise
    # the evidence phrase is wrong/missing and the query silently scores 0.
    empty = [it["query"] for it in eval_set if not it["relevant_chunk_ids"]]
    if empty:
        print("WARNING: queries with no relevant chunks (check evidence_phrases):")
        for q in empty:
            print(f"  - {q}")

    print_eval_set(eval_set, chunks)

    baseline = evaluate_retrieval(eval_set, retrieve, k=5)
    print_report(baseline, "Baseline (in-corpus queries):")
    print_per_query(baseline)

    print("\nBy query type (k=5):")
    print_by_type(baseline)

    print("\nOut-of-corpus queries (scored separately; lower top-sim = better abstention)")
    incorpus_top = float(np.mean([top_similarity(it["query"], embeddings, model) for it in eval_set]))
    print(f"  in-corpus average top similarity: {incorpus_top:.3f}")
    for q in OUT_OF_CORPUS:
        print(f"  {q[:48]:<50}top-sim={top_similarity(q, embeddings, model):.3f}")

    print("\nExperiment 1: vary k")
    for k in [1, 3, 5, 10, 20]:
        r = evaluate_retrieval(eval_set, retrieve, k=k)
        print(f"  k={k:<3} P@k={r['avg_precision']:.3f} R@k={r['avg_recall']:.3f} MRR={r['mrr']:.3f}")

    print("\nExperiment 2: vary chunk size")
    for cs in [128, 256, 512, 1024]:
        c, e, m = build_corpus(DOCS, chunk_size=cs)
        r = evaluate_retrieval(build_eval_set(c), make_retriever(c, e, m), k=5)
        print(f"  size={cs:<5} chunks={len(c):<4} P@5={r['avg_precision']:.3f} R@5={r['avg_recall']:.3f} MRR={r['mrr']:.3f}")

    print("\nExperiment 3: vary embedding model")
    for name, ng in [("unigram", 1), ("uni+bigram", 2)]:
        c, e, m = build_corpus(DOCS, ngram=ng)
        r = evaluate_retrieval(build_eval_set(c), make_retriever(c, e, m), k=5)
        print(f"  {name:<12} P@5={r['avg_precision']:.3f} R@5={r['avg_recall']:.3f} MRR={r['mrr']:.3f}")
