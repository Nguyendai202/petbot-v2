from vetbot.chain import (
    _is_emergency,
    _count_followup_turns,
    _count_disambiguate_turns,
    _trim,
    _extract_disease_names,
)


class TestIsEmergency:
    def test_detects_seizure(self):
        assert _is_emergency("bé nhà em đang co giật")

    def test_detects_unable_to_urinate(self):
        assert _is_emergency("mèo không tiểu được từ sáng")

    def test_case_insensitive(self):
        assert _is_emergency("Bé BẤT TỈNH rồi")

    def test_normal_symptom_not_emergency(self):
        assert not _is_emergency("chó nôn vàng mấy lần")

    def test_empty_string_not_emergency(self):
        assert not _is_emergency("")


class TestCountFollowupTurns:
    def test_counts_assistant_questions(self):
        messages = [
            {"role": "user", "content": "bé nôn"},
            {"role": "assistant", "content": "Bé mấy tháng tuổi?"},
            {"role": "user", "content": "3 tháng"},
            {"role": "assistant", "content": "Đã tiêm phòng chưa?"},
        ]
        assert _count_followup_turns(messages) == 2

    def test_assistant_statement_not_counted(self):
        messages = [
            {"role": "assistant", "content": "Bé bị viêm ruột do Parvovirus."},
        ]
        assert _count_followup_turns(messages) == 0

    def test_empty_messages(self):
        assert _count_followup_turns([]) == 0


class TestCountDisambiguateTurns:
    def test_counts_disease_specific_questions(self):
        messages = [
            {"role": "assistant", "content": "Mắt bé có bị đục không?"},
            {"role": "assistant", "content": "Phân màu gì, có mùi tanh không?"},
        ]
        assert _count_disambiguate_turns(messages) == 2

    def test_generic_question_not_counted(self):
        messages = [
            {"role": "assistant", "content": "Bạn cần hỗ trợ thêm gì không?"},
        ]
        assert _count_disambiguate_turns(messages) == 0


class TestTrim:
    def test_keeps_last_n_turns(self):
        messages = [{"role": "user", "content": str(i)} for i in range(30)]
        trimmed = _trim(messages)
        assert len(trimmed) == 20
        assert trimmed[-1]["content"] == "29"

    def test_shorter_than_limit_unchanged(self):
        messages = [{"role": "user", "content": "hi"}]
        assert _trim(messages) == messages


class TestExtractDiseaseNames:
    def test_extracts_unique_disease_names(self):
        context = (
            "[Bệnh: Parvovirus | Độ liên quan: 0.8]\nnội dung 1\n\n---\n\n"
            "[Bệnh: Parvovirus | Độ liên quan: 0.6]\nnội dung 2\n\n---\n\n"
            "[Bệnh: Carre | Độ liên quan: 0.5]\nnội dung 3"
        )
        assert _extract_disease_names(context) == ["Parvovirus", "Carre"]

    def test_no_disease_returns_empty(self):
        assert _extract_disease_names("") == []
