import numpy as np

from semantic_poc.features import TfIdf, terms_of, tokenize


def test_tokenize_lowercase_and_apostrophes():
    assert tokenize("Card's NOT here, ok?") == ["card's", "not", "here", "ok"]


def test_terms_include_bigrams():
    assert terms_of("replace my card") == [
        "replace", "my", "card", "replace my", "my card"]


def test_tfidf_deterministic_and_l2_normalized():
    docs = ["replace my card", "my card is lost", "check my balance", "my card"]
    a = TfIdf.fit(docs)
    b = TfIdf.fit(list(docs))
    assert a.vocab == b.vocab
    assert np.allclose(a.idf, b.idf)
    X = a.transform(["my card is lost lost"]).toarray()
    assert np.isclose(np.linalg.norm(X), 1.0)


def test_tfidf_is_float64_and_matches_go_math_exactly():
    # The Go server computes in float64 with tolerance 1e-9; float32 anywhere
    # in the Python pipeline breaks parity.
    docs = ["replace my card", "my card is lost", "check my balance", "my card"]
    t = TfIdf.fit(docs)
    assert t.idf.dtype == np.float64
    X = t.transform(["my card is lost lost"])
    assert X.dtype == np.float64
    # Reproduce the documented formula in plain float64 (what Go executes).
    row = {}
    for term, count in [("my", 1), ("card", 1), ("my card", 1), ("is", 1), ("lost", 2)]:
        if term in t.vocab:
            row[t.vocab[term]] = (1 + np.log(count)) * t.idf[t.vocab[term]]
    norm = np.sqrt(sum(v * v for v in row.values()))
    dense = X.toarray()[0]
    for j, v in row.items():
        assert dense[j] == v / norm  # exact, not approx


def test_tfidf_min_df_filters_rare_terms():
    docs = ["replace my card", "my card is lost", "unique snowflake"]
    t = TfIdf.fit(docs, min_df=2)
    assert "unique" not in t.vocab and "my card" in t.vocab


def test_unknown_terms_ignored():
    t = TfIdf.fit(["replace my card", "my card"], min_df=2)
    X = t.transform(["completely unseen words"])
    assert X.nnz == 0
