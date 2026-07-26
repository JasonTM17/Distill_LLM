# Báo cáo trạng thái dự án Distill GPT-5.5 → Qwen2.5-1.5B

**Ngày:** 2026-07-25
**Mục đích:** Tóm tắt chi tiết tình hình huấn luyện model hiện tại + đề xuất hướng đi tiếp.

---

## 1. Trạng thái tổng quan (Live evidence)

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Pipeline code | ✅ Hoàn chỉnh | 11 file `.py` ở root (config, gen_batch, format_dataset, train_student, evaluate, evaluate_extended, chat, test_model, test_connection, download_model, download_student) |
| Base model | ✅ Có sẵn | `D:/models/qwen15-1.5b/` (Qwen2.5-1.5B-Instruct, 3GB) + `D:/models/qwen25-3b/` (3B, 5.76GB, chưa dùng) |
| Dataset prompts | ✅ Đầy đủ | `data/prompts.json` = 530 prompts / 10 categories |
| Teacher outputs | ⚠️ Một phần | 396/530 success, 134 fail (đặc biệt philosophy + health = 0 thành công) |
| Processed dataset | ✅ Format xong | `dataset_train.json` = 395 mẫu (1 mẫu trùng/khuyết so với 396) |
| Adapter LoRA | ✅ Đã train | `checkpoints/adapter/` 8.7 MB |
| Merged model | ✅ Đã merge | `checkpoints/merged/model.safetensors` = 1.6 GB |
| Evaluation | ⚠️ Sơ sài | Chỉ perplexity (4.70) + heuristic `len>20` |

## 2. Kết quả training thực tế (từ `trainer_state.json`)

| Step | Epoch | Loss | Token Acc |
|---|---|---|---|
| 10 | 0.81 | 1.432 | 66.0% |
| 20 | 1.57 | 1.329 | 68.5% |
| 30 | 2.32 | 1.140 | **72.6%** |

- 3 epochs × ~13 steps/epoch = 39 steps total.
- `max_steps=39` ⇒ chỉ chạy vừa đủ số sample / (batch × grad_accum) ≈ 395 / 8 ≈ 49 → thực tế là 39 steps × 8 = 312 effective samples × 3 epochs (không trượt sample nào vì `train_dataset` chỉ ~395).
- Loss giảm 20%, accuracy tăng 6.6pp — hội tụ ổn, **không có dấu hiệu overfitting** (loss vẫn đang giảm ở step cuối).
- Perplexity = 4.70 trên 20 held-out → xếp loại "Excellent" theo `evaluate.py`.

## 3. Điểm MẠNH của dự án

1. **Pipeline production-grade:** resumable generation, retry 3 lần, save sau mỗi prompt, UTF-8 wrapper cho Windows console, dotenv cho secrets.
2. **Configuration tập trung** ở `config.py` — không hardcode.
3. **Hyperparameter hợp lý cho 6GB VRAM:** QLoRA NF4, float16, LoRA r=16 trên q/k/v/o, batch=1 × grad_accum=8, AdamW 8-bit. Không gradient checkpointing (đã biết là OOM trên RTX 3060).
4. **Docs chuẩn CK:** PDR, architecture, code standards, roadmap — đầy đủ template.
5. **Git hygiene tốt:** conventional commits, .gitignore bỏ qua model/data raw, API key đã chuyển vào `.env`.
6. **Inference sẵn sàng:** `chat.py` interactive + `test_model.py` nhanh, load merged 4-bit chỉ ~3.5GB VRAM.

## 4. Điểm YẾU / Rủi ro thực tế

### 4.1 Dataset chưa đầy đủ — đây là vấn đề LỚN NHẤT
- **134/530 prompts fail** ở bước teacher generation. Trong đó:
  - `philosophy` (50 prompts): **0 thành công** — toàn bộ `output = ""`.
  - `health` (51 prompts): không có mặt trong danh sách success (chỉ có 50 entries coding + 50 science + 50 ml_ai + 50 vietnamese + 50 math + 50 creative + 46 business + 40 reasoning = 396). Suy ra health = **0 thành công** luôn.
  - `reasoning` chỉ đạt 40/43.
  - `business` đạt 46/51.
  - Có nhiều entries id 455–530 chứa text **corrupted UTF-8** (ví dụ `Vi���t m��Tt`) — đây có thể là encoding bug trong `gen_batch.py` khi request gặp Vietnamese prompts mà API trả lỗi hoặc timeout.
- **Hệ quả:** Model không được dạy philosophy, health, và cũng thiếu nhiều business/reasoning. README tự nhận: "Vietnamese + Business + Health + Philosophy not yet tested."

### 4.2 README nói "Qwen2.5-3B" nhưng code train 1.5B
- `config.py`: `STUDENT_MODEL_ID = "D:/models/qwen15-1.5b"`
- `train_student.py` docstring: `"""QLoRA fine-tune Qwen2.5-3B..."""`
- `system-architecture.md`: "Qwen2.5-1.5B"
- → Inconsistency giữa docstring và config. Phải sửa.

### 4.3 `merged_model/` (root) trống — config có 2 chỗ conflict
- `config.MERGED_DIR = merged_model/` (root, **trống**)
- `config.MERGED_MODEL_DIR = checkpoints/merged/` (có model 1.6GB)
- `chat.py`, `evaluate.py`, `evaluate_extended.py`, `test_model.py` đều dùng `config.MERGED_MODEL_DIR` ⇒ đúng.
- Nhưng đọc nhanh qua `MERGED_DIR` ở `config.py` rất dễ nhầm — nên gộp/xóa.

### 4.4 Evaluation quá yếu
- `evaluate.py` chỉ đo perplexity trên 20 sample (lại là sample **trong** training set, vì không thực sự tách held-out — file `dataset_train.json` là toàn bộ).
- `evaluate_extended.py` "12/12 = 100%" chỉ là heuristic `len(response) > 20` — không có ground truth. Một model hallucinate dài cũng pass.
- Chưa có benchmark chuẩn (HumanEval, GSM8K, MMLU).
- Chưa so sánh side-by-side với teacher trên held-out.

### 4.5 Không có held-out test set thật
- `config.TEST_SPLIT_RATIO = 0.1` được define nhưng không dùng.
- Tất cả 395 mẫu đều đi vào training → perplexity đo trên "đã thấy" sample, không phản ánh generalization thật.

### 4.6 3B model đã download nhưng không dùng
- `D:/models/qwen25-3b/` tốn 5.76GB nhưng `config.STUDENT_MODEL_ID` vẫn trỏ về 1.5B.
- Roadmap có "v0.6 Upgrade to Qwen2.5-3B" nhưng chưa làm.

### 4.7 Vietnamese prompts bị encoding bug
- Các id 501–510, 520–525 hiển thị `Vi���t m��Tt` ⇒ UTF-8 bị corrupt khi lưu JSON (PowerShell `Get-Content` không xử lý đúng, hoặc đường truyền API trả lỗi). Đây là **mất dữ liệu gốc** ở client side — cần điều tra.

## 5. Đề xuất hướng đi tiếp (theo thứ tự ưu tiên)

### P0 — Ngay bây giờ (rẻ, khôi phục chất lượng)

**A. Hoàn thiện dataset 530 prompts**
- Mở `data/prompts.json` và `data/raw/teacher_outputs.json`, xác định chính xác 134 prompt bị fail.
- Fix bug encoding UTF-8 trong `gen_batch.py` (ghi file với `ensure_ascii=False` đã có, nhưng vẫn lỗi ⇒ có thể là lỗi ở payload gửi đi hoặc trong `chat_template.jinja` của teacher). Test với 5 prompts tiếng Việt trước khi chạy lại.
- Bật lại 9Router, chạy `python gen_batch.py` — script đã có resume nên sẽ tiếp tục từ 396.
- Kiểm tra category `philosophy`, `health` đặc biệt — nếu API liên tục fail, có thể teacher model không phù hợp cho domain này hoặc cần tăng `MAX_RETRIES`.

**B. Tách held-out test set**
- Sửa `format_dataset.py` hoặc tạo `split_dataset.py` để tách 10% (~40 sample) làm `dataset_test.json` — đảm bảo stratified theo category.
- Đo lại perplexity trên held-out ⇒ metric mới có ý nghĩa.

**C. Sửa inconsistency tài liệu**
- `train_student.py` docstring đổi thành `QLoRA fine-tune Qwen2.5-1.5B-Instruct`.
- Xóa `MERGED_DIR` (root) trong `config.py` hoặc dùng nó để chuyển output merge ra đó thay vì `checkpoints/merged/`.
- Cập nhật `README.md` để khớp (loss, perplexity, dataset size mới nhất).

### P1 — Nâng chất lượng model (1–2 ngày)

**D. Retrain trên full 530 + held-out split**
- Sau khi (A) xong, `python format_dataset.py` ⇒ `python train_student.py`.
- So sánh 3 mốc: 200 / 396 / 530 samples × 3 epochs.
- Kỳ vọng: token accuracy > 75%, perplexity held-out < 5.

**E. Evaluation nghiêm túc hơn**
- Thay heuristic `len>20` bằng so sánh từng prompt với teacher output (BLEU/ROUGE hoặc LLM-as-judge dùng chính 9Router).
- Thêm 5–10 prompt benchmark coding chuẩn (HumanEval-mini, MBPP) và reasoning (GSM8K-mini).
- Ghi log kết quả ra `plans/reports/evaluation-v04.md`.

### P2 — Mở rộng (1 tuần)

**F. Train thử trên Qwen2.5-3B**
- `D:/models/qwen25-3b/` đã có sẵn, đổi `STUDENT_MODEL_ID`.
- Có thể cần: giảm `MAX_SEQ_LENGTH` còn 384, bật `gradient_checkpointing=True` (vẫn có thể OOM — chuẩn bị fallback về 1.5B).
- So sánh accuracy với 1.5B để quyết định model nào làm "production".

**G. True distillation với soft labels**
- Hiện tại chỉ SFT trên teacher output (hard labels).
- Teacher API không trả logits ⇒ không làm được KL-divergence matching.
- Workaround: dùng teacher API với `temperature=1.0` rồi `n=4` lần, lấy majority vote làm "pseudo-soft target". Hoặc: distill từ một open-weight teacher (Qwen2.5-7B-Instruct nếu có VRAM).

**H. Export deployment**
- GGUF qua `llama.cpp` cho Ollama/llama.cpp deploy.
- ONNX cho cross-platform.
- So sánh Q4_K_M vs Q5_K_M quantization giữ perplexity.

### P3 — V1 production (2–4 tuần)

**I. CI/CD retrain pipeline**
- Bash/PowerShell script tự động: generate → format → train → eval → commit adapter (DVC hoặc git-lfs).
- Track loss/perplexity qua W&B hoặc MLflow local.

**J. REST API inference**
- FastAPI wrapper cho merged model, support streaming.
- Container hóa bằng Docker (CPU/GPU image).

## 6. Câu hỏi cần user quyết định

1. **Có muốn train lại ngay trên 396 mẫu (chất lượng tốt hơn 300 hiện tại) trước khi đợi generate đủ 530?** Hay muốn generate đủ 530 trước rồi train 1 lần?
2. **Có muốn thử train trên 3B không?** Hay giữ 1.5B làm baseline cho đến khi đủ dữ liệu?
3. **Đối với philosophy + health = 0 success:** có nên dùng teacher khác (gpt-5.6-terra) hay bỏ qua 2 category này?
4. **Có muốn benchmark chuẩn (HumanEval, GSM8K) không?** Tốn công nhưng cho metric khách quan.

## 7. Kết luận ngắn

Dự án đã chạy được **end-to-end pipeline** với kết quả training tốt (loss 1.43→1.14, perplexity 4.7). Model 1.5B distilled từ GPT-5.5-xhigh đã có thể dùng local trên RTX 3060. **Tuy nhiên**, chất lượng thực sự chỉ đánh giá được khi:
- Dataset đủ đa dạng (đang thiếu philosophy + health + nhiều business/reasoning/Vietnamese).
- Có held-out test set thật.
- Có benchmark chuẩn thay vì heuristic `len>20`.

**Hướng đi khuyến nghị nhất cho tuần này:** hoàn thiện dataset 530 (P0-A) → fix inconsistency (P0-C) → retrain + đánh giá nghiêm túc (P1-D/E). Sau đó mới quyết định có nên lên 3B hay không.

---

**Bằng chứng đã kiểm tra:** `trainer_state.json` (training metrics), `teacher_outputs.json` (396/530 success), `dataset_train.json` (395 samples), `checkpoints/merged/` (1.6GB merged model tồn tại), `docs/project-roadmap.md` (v0.3 stable), 11 git commits gần nhất, 2 điểm inconsistency trong config + docstring.