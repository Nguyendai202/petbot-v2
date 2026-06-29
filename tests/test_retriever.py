from vetbot.retriever import _tokenize_vi, _build_sparse_vector, _format_points


class FakePoint:
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload


class TestTokenizeVi:
    def test_removes_stopwords(self):
        tokens = _tokenize_vi("chó của tôi không ăn")
        assert "của" not in tokens
        assert "không" not in tokens
        assert "chó" in tokens

    def test_lowercases(self):
        assert "parvo" in _tokenize_vi("Bệnh Parvo nguy hiểm")

    def test_empty_text(self):
        assert _tokenize_vi("") == []


class TestBuildSparseVector:
    def test_empty_text_returns_valid_vector(self):
        vec = _build_sparse_vector("")
        assert vec.indices == [0]
        assert vec.values == [0.0]

    def test_values_normalized_to_sum_one(self):
        vec = _build_sparse_vector("chó nôn nôn mèo")
        assert abs(sum(vec.values) - 1.0) < 1e-9

    def test_same_text_same_vector(self):
        v1 = _build_sparse_vector("chó nôn vàng")
        v2 = _build_sparse_vector("chó nôn vàng")
        assert v1.indices == v2.indices
        assert v1.values == v2.values


class TestFormatPoints:
    def test_formats_with_disease_and_score(self):
        points = [FakePoint(0.812, {"page_content": "nội dung", "disease_name": "Parvo"})]
        result = _format_points(points)
        assert "[Bệnh: Parvo | Độ liên quan: 0.812]" in result
        assert "nội dung" in result

    def test_skips_empty_content(self):
        points = [FakePoint(0.5, {"page_content": "", "disease_name": "Parvo"})]
        assert _format_points(points) == ""

    def test_no_disease_name_omits_label(self):
        points = [FakePoint(0.5, {"page_content": "nội dung"})]
        result = _format_points(points)
        assert "Bệnh:" not in result
        assert "Độ liên quan: 0.5" in result
