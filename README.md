# VetBot v2 — Backlog & Plan

## Đã làm ✅
- Sort + dedup chunks theo score
- Coarse-to-fine retrieval (2 vòng)
- Disease name normalization (`_DISEASE_NAME_MAP`)
- Split `routing_model` (gpt-4o-mini) vs `diagnosis_model` (gpt-4o)
- Cross-reference detect lúc ingest → lưu `cross_ref_disease` vào payload
- Adaptive RAG: `_disambiguate()` — hỏi câu targeted khi nhiều bệnh candidate
- Improve `_query_understanding` prompt: giữ triệu chứng vận động, mắt, vaccine

---

## Backlog — theo thứ tự ưu tiên

### 🔴 Cao

**1. Direct treatment query path**
- Vấn đề: "chó tôi bị ghẻ nên dùng thuốc gì" → classify nhầm `product_query`, symptom query quá ngắn → trả lời thiếu
- Fix: thêm state `treatment_query` vào classify, query_understanding thêm flag `direct_treatment + disease_name` → skip coarse/disambiguate → `get_treatment_points(disease_name)` trực tiếp

**2. Google/Shopee drug search**
- Sau khi chẩn đoán xong, extract tên thuốc từ response
- Search Google hoặc Shopee API → trả link sản phẩm cụ thể cho user mua
- Cần: tool call trong agent, hoặc post-processing sau stream
- Lưu ý: ưu tiên sản phẩm có bán tại VN, tránh kết quả nước ngoài

### 🟡 Trung bình

**3. Score threshold**
- Filter bỏ chunk score < 0.3–0.4 trước khi đưa vào context
- Giảm noise bệnh không liên quan (case Sán Lá Gan lọt vào)
- Nếu sau filter không còn chunk → trả "chưa tìm được thông tin"

**4. Reranker**
- Sau hybrid retrieve, dùng cross-encoder rerank lại theo relevance thực sự
- Model gợi ý: `cross-encoder/ms-marco-MiniLM-L-6-v2` hoặc Cohere Rerank API
- Đặc biệt giúp khi RRF score nhiều bệnh bằng nhau

**5. Chunk size tăng lên 800–1000 tokens**
- Chunk 600 hiện tại đôi khi cắt giữa phần triệu chứng và điều trị
- Tăng chunk_overlap lên 200 để không mất context biên

### 🟢 Thấp / Dài hạn

**6. Merge classify + disambiguate**
- Hiện có 2 LLM call riêng: classify (basic info) + disambiguate (disease-specific)
- Gộp thành 1 call thông minh hơn sau khi có retrieval results

**7. Step-back prompting cho query**
- Từ câu hỏi cụ thể → rút ra câu hỏi tổng quát hơn để retrieve thêm
- VD: "chó nôn vàng" → step-back: "viêm dạ dày ruột chó triệu chứng"

**8. Query decomposition**
- Câu hỏi phức tạp → tách thành nhiều sub-query
- VD: "bé vừa nôn vừa ho" → query riêng cho tiêu hóa + hô hấp

**9. Parent-child indexing**
- Chunk nhỏ để retrieve chính xác, nhưng trả về chunk lớn hơn (parent) cho context
- Tránh LLM bị thiếu context do chunk bị cắt

**10. Sparse vector upgrade**
- Thay TF hash hiện tại bằng BM25 thực sự hoặc SPLADE
- Cải thiện keyword matching, đặc biệt tên thuốc chuyên ngành

**11. Multi-turn context cho retrieve**
- Hiện coarse query chỉ dùng message gần nhất
- Nên tổng hợp toàn bộ triệu chứng từ nhiều turn trước

---

## Ý tưởng Google/Shopee search — chi tiết kỹ thuật

```
Diagnosis response có: "Atropin sulfat 0,1%, 1ml/10kgP"
                       "Glucoza + chất điện giải"

→ Extract drug names (LLM hoặc regex)
→ Search Shopee: "atropin sulfat thú y" 
→ Trả về: [tên sản phẩm, giá, link]
```

Options:
- **Shopee Affiliate API** — có SDK, cần đăng ký
- **Google Custom Search API** — $5/1000 queries, dễ setup
- **SerpAPI** — $50/mo, search Google/Shopee/Lazada tổng hợp

Lưu ý quan trọng: một số thuốc thú y (Atropin, kháng sinh) cần kê đơn → cần disclaimer rõ ràng trước link mua.
