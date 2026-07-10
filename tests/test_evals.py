from tracegate.detect import default_detectors
from tracegate.evals.corpus import build_corpus
from tracegate.evals.metrics import DetectorScore, evaluate_detector


def test_corpus_composition():
    corpus = build_corpus()
    assert len(corpus) == 14
    assert sum(1 for c in corpus if not c.expected_detectors) == 4
    assert sum(1 for c in corpus if "loop" in c.expected_detectors) == 5
    assert sum(1 for c in corpus if "tool_misuse" in c.expected_detectors) == 5
    assert sum(1 for c in corpus if "contradiction" in c.expected_detectors) == 2


def test_deterministic_detectors_perfect_on_corpus():
    corpus = build_corpus()
    for detector in default_detectors():
        score = evaluate_detector(detector, corpus)
        assert score.precision == 1.0, f"{detector.name} fp={score.fp}"
        assert score.recall == 1.0, f"{detector.name} fn={score.fn}"


def test_score_math():
    score = DetectorScore(detector="d", tp=3, fp=1, fn=2)
    assert score.precision == 0.75
    assert score.recall == 0.6


def test_score_zero_denominators():
    score = DetectorScore(detector="d", tp=0, fp=0, fn=0)
    assert score.precision == 1.0
    assert score.recall == 1.0
