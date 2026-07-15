from semantic_poc.policy import decide

FAM = {"a": "f1", "b": "f1", "c": "f2", "d": "f2"}
TH = {"a": 0.9, "b": 0.8, "c": 0.85, "d": 0.7}


def d(probs, **kw):
    args = dict(thresholds=TH, families=FAM, tau_low=0.3, max_accepted=3,
                exclusive_groups=[])
    args.update(kw)
    return decide(probs, **args)


def test_single_accept():
    r = d({"a": 0.95, "b": 0.1, "c": 0.1, "d": 0.1})
    assert r["decision"] == "accepted"
    assert r["intents"][0]["name"] == "a"
    assert r["intents"][0]["main_intent"] == "f1"


def test_multi_accept_sorted_desc():
    r = d({"a": 0.91, "b": 0.99, "c": 0.1, "d": 0.1})
    assert r["decision"] == "multi_accepted"
    assert [i["name"] for i in r["intents"]] == ["b", "a"]


def test_no_supported_intent_below_tau_low():
    r = d({"a": 0.2, "b": 0.1, "c": 0.25, "d": 0.05})
    assert r["decision"] == "no_supported_intent" and r["intents"] == []


def test_abstained_between_tau_low_and_threshold():
    r = d({"a": 0.5, "b": 0.1, "c": 0.1, "d": 0.1})
    assert r["decision"] == "abstained"


def test_exclusive_group_keeps_argmax():
    r = d({"a": 0.95, "b": 0.85, "c": 0.1, "d": 0.1},
          exclusive_groups=[["a", "b"]])
    assert [i["name"] for i in r["intents"]] == ["a"]


def test_cap_and_overflow_flag():
    r = d({"a": 0.95, "b": 0.9, "c": 0.9, "d": 0.8}, max_accepted=3)
    assert len(r["intents"]) == 3 and r["overflow"] is True


def test_equal_probabilities_tie_break_by_name():
    # Same rule as the Go implementation: prob desc, then name asc.
    r = d({"d": 0.92, "b": 0.92, "a": 0.92, "c": 0.1})
    assert [i["name"] for i in r["intents"]] == ["a", "b", "d"]
