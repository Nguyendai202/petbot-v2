from vetbot.ingest import _normalize, _match_heading, split_by_disease_heading


class TestNormalize:
    def test_collapses_whitespace(self):
        assert _normalize("BỆNH   CARRE\n") == "BỆNH CARRE"


class TestMatchHeading:
    def test_matches_known_heading(self):
        assert _match_heading("BỆNH CẦU TRÙNG") == "Bệnh Cầu Trùng"

    def test_normalizes_parvovirus_aliases(self):
        assert _match_heading("PARVOVIRUS") == "Bệnh Viêm Ruột Truyền Nhiễm Do Parvovirus"
        assert _match_heading("VIÊM RUỘT TRUYỀN NHIỄM DO PARVOVIRUS") == \
            "Bệnh Viêm Ruột Truyền Nhiễm Do Parvovirus"

    def test_non_heading_returns_none(self):
        assert _match_heading("Chó nôn liên tục từ sáng") is None


class TestSplitByDiseaseHeading:
    def test_splits_sections_by_heading(self):
        text = (
            "Đoạn mở đầu không thuộc bệnh nào\n"
            "BỆNH CẦU TRÙNG\n"
            "Triệu chứng: tiêu chảy có máu\n"
            "BỆNH CARRE\n"
            "Triệu chứng: sốt, ho\n"
        )
        sections = split_by_disease_heading(text)
        diseases = [name for name, _ in sections]
        assert "Chưa xác định" in diseases
        assert "Bệnh Cầu Trùng" in diseases
        assert "Bệnh Carre" in diseases

    def test_content_under_each_heading(self):
        text = "BỆNH CẦU TRÙNG\nTriệu chứng đặc trưng\n"
        sections = split_by_disease_heading(text)
        disease, content = sections[0]
        assert disease == "Bệnh Cầu Trùng"
        assert "Triệu chứng đặc trưng" in content
