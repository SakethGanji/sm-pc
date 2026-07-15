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


def test_tfidf_min_df_filters_rare_terms():
    docs = ["replace my card", "my card is lost", "unique snowflake"]
    t = TfIdf.fit(docs, min_df=2)
    assert "unique" not in t.vocab and "my card" in t.vocab


def test_unknown_terms_ignored():
    t = TfIdf.fit(["replace my card", "my card"], min_df=2)
    X = t.transform(["completely unseen words"])
    assert X.nnz == 0
