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
        res.append({"query": it["query"], "precision": precision_at_k(got, rel),
                    "recall": recall_at_k(got, rel), "rr": reciprocal_rank(got, rel),
                    "retrieved": got, "relevant": rel})
    return {"per_query": res, "avg_precision": float(np.mean([r["precision"] for r in res])),
            "avg_recall": float(np.mean([r["recall"] for r in res])),
            "mrr": float(np.mean([r["rr"] for r in res])), "k": k}


def print_report(res, label=""):
    if label:
        print(label)
    print(f"  k={res['k']} queries={len(res['per_query'])} "
          f"P@k={res['avg_precision']:.3f} R@k={res['avg_recall']:.3f} MRR={res['mrr']:.3f}")


def print_per_query(res):
    print(f"  {'Query':<50}{'#rel':>5}{'P@k':>6}{'R@k':>6}{'RR':>7}")
    for r in res["per_query"]:
        print(f"  {r['query'][:48]:<50}{len(r['relevant']):>5}{r['precision']:>6.2f}{r['recall']:>6.2f}{r['rr']:>7.2f}")


IN_CORPUS = [
    ("What is Dr. Jennifer Groh a Professor of?", ["psychology and neuroscience"]),
    ("Where is the vestibular system located and what part of the brain processes its signals?", ["flocculus", "cerebellum", "inner ear"]),
    ("What was the name of Fajgenbaum's partner during his sickness", ["Kaitlin"]),
]
OUT_OF_CORPUS = ["What's the population of Tokyo?"]


def build_eval_set(chunks):
    return [{"query": q, "relevant_chunk_ids": [i for i, c in enumerate(chunks) if any(p in c["text"] for p in ph)]}
            for q, ph in IN_CORPUS]


if __name__ == "__main__":
    chunks, embeddings, model = build_corpus(DOCS)
    eval_set = build_eval_set(chunks)
    retrieve = make_retriever(chunks, embeddings, model)

    print_report(evaluate_retrieval(eval_set, retrieve, k=5), "Baseline (in-corpus queries):")
    print_per_query(evaluate_retrieval(eval_set, retrieve, k=5))

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
