# MultiDelete trên ALBEF: Flickr30K và NLVR2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tnde0401/multimodal-unlearning-multidelete/blob/main/MultiDelete_all.ipynb) Notebook điều khiển (`MultiDelete_all.ipynb`)  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tnde0401/multimodal-unlearning-multidelete/blob/main/MultiDelete_Demo1.ipynb) Demo 1 (`MultiDelete_Demo1.ipynb`)

Mã nguồn: https://github.com/tnde0401/multimodal-unlearning-multidelete

Tái lập thí nghiệm của bài báo *MultiDelete for Multimodal Machine Unlearning* (Cheng & Amiri, ECCV 2024) trên mô hình ALBEF, kèm các thí nghiệm thêm của nhóm: MD-ITC, Neg-Dr, ablation (-MD, -MKR, -UKR).

Thư mục này gồm:

| File | Vai trò |
|---|---|
| `MultiDelete_all.ipynb` | Notebook điều khiển |
| `multidelete.py` | Điều phối: tham số dòng lệnh, cài đặt, tải dữ liệu Flickr30K, chạy tuần tự, in bảng kết quả. |
| `run_md_retrieval.py` | Engine Flickr30K: huấn luyện và đánh giá MultiDelete và các baseline. |
| `nlvr2_engine.py` | Engine NLVR2: dữ liệu, huấn luyện, đánh giá, bảng kết quả. |
| `demo_engine.py` | Demo định tính. |
| `MultiDelete_Demo1.ipynb` | Demo 1 (notebook riêng, tự chứa code): truy hồi văn bản sang ảnh trước và sau khi xoá, có kết quả đã chạy. Xem mục 5b. |
| `README.md` | File này. |

Bốn file `.py` phải nằm cạnh nhau. Ba engine là code lấy từ ba notebook đã dùng để chạy ra số liệu trong báo cáo; không cần đọc chúng để chạy. Các lệnh ở mục 2 đến 5 dưới đây là lệnh của `multidelete.py`; trong notebook mỗi lệnh đã nằm sẵn trong một cell.

## 1. Cần gì để chạy

- **GPU NVIDIA có ít nhất 15 GB VRAM.** Các kết quả trong báo cáo chạy trên Tesla T4 (Colab và Kaggle).
- Python 3.10 trở lên, **PyTorch đã cài sẵn với CUDA.** Colab và Kaggle có sẵn. Trên máy riêng, cài torch trước theo hướng dẫn của PyTorch. Nên dùng môi trường ảo mới vì `setup` ghim `transformers==4.36.2` và `timm==0.9.16`.

## 2. Chạy nhanh

```bash
python multidelete.py setup        # 1 lần: cài thư viện, clone repo tác giả, vá lỗi
python multidelete.py flickr30k    # toàn bộ thí nghiệm Flickr30K
python multidelete.py nlvr2        # toàn bộ thí nghiệm NLVR2
```

`python multidelete.py all` là `setup` + `flickr30k` + `nlvr2`. Các lệnh `flickr30k`, `nlvr2`, `demo` tự gọi `setup` nếu chưa làm.

Chạy thử luồng bằng `--profile smoke` (tập xóa nhỏ, vài bước huấn luyện, chạy trong vài chục phút):

```bash
python multidelete.py flickr30k --profile smoke
python multidelete.py nlvr2 --profile smoke
```

Xem các lệnh sẽ chạy mà không chạy gì: thêm `--dry-run`. In lại bảng Flickr30K từ kết quả đã có: `python multidelete.py flickr30k --summary-only`.

## 3. Flickr30K

```bash
python multidelete.py flickr30k [--profile smoke|fast|paper] [--only LIST] [--flickr-images DIR] [--force]
```

Các bước:

1. Lấy ảnh. Nếu đã có ảnh Flickr30K thì truyền `--flickr-images THƯ_MỤC` (thư mục chứa `1000092795.jpg` và hơn 30000 ảnh `.jpg`). Nếu không, script tải `adityajn105/flickr30k` bằng Kaggle API. Annotation train/val/test lấy từ nguồn của LAVIS.
2. Tạo tập xóa `D_f`: với mỗi seed (42, 87, 21) chọn ngẫu nhiên 5000 ảnh train, `D_f` là 1000 ảnh đầu (= 5000 cặp ảnh-caption). Lưu kèm SHA-256 trong `split_manifest.json`.
3. Chạy lần lượt 10 mục, mỗi mục là một lệnh con gọi `run_md_retrieval.py` (thứ tự cố định, `original` luôn đầu tiên để các mục sau dùng lại kết quả "trước khi quên"):

| Mục (`--only`) | Nội dung | Thời gian ước tính trên T4 |
|---|---|---|
| `original` | ALBEF đã fine-tune của LAVIS, chỉ đánh giá | khoảng 15 phút |
| `main` | MultiDelete gốc (MD trên CLS của fusion) | khoảng 41 phút |
| `itc` | MD-ITC: MD trên ma trận tương đồng ITC, sát code repo | khoảng 39 phút |
| `negdr` | Neg-Dr: cặp âm lấy từ `D_r` | khoảng 41 phút |
| `noMD`, `noMKR`, `noUKR` | Ablation, bỏ lần lượt `L_MD`, `L_MKR`, `L_UKR` | khoảng 50 phút mỗi mục |
| `ft` | Baseline FineTune trên `D_r` (1 epoch, lr 5e-5) | khoảng 17 phút |
| `neggrad` | Baseline NegGrad có clip gradient (lr 2e-6, 25 bước) | khoảng 11 phút |
| `dtd` | Baseline Descent-to-Delete | khoảng 10 phút |
| `mdF`, `neggradraw` | Tùy chọn, chỉ chạy khi gọi rõ bằng `--only`: MultiDelete-F (chỉ cập nhật cross-attention và ITM head) và NegGrad nguyên bản không clip | |

Tổng cộng ước tính khoảng 5 đến 6 giờ ở profile `fast` (thời gian ở bảng là số đo từ các lần chạy trên T4, riêng `original` là ước lượng). Muốn chạy một phần: `--only original,main` (đây là phần tối thiểu để có dòng chính của báo cáo).

4. In bảng tổng hợp và lưu `summary_<profile>.csv`.

Thêm `--save-ckpt` để lưu checkpoint đầy đủ (~420 MB) của MultiDelete và FineTune vào `<work>/checkpoints`, như notebook Kaggle gốc. Mặc định không lưu.

**Profile:** `fast` (mặc định, 3 epoch, `k_test=16`) là cấu hình của các số trong báo cáo. `paper` dùng 5 epoch (bài báo không nêu số epoch trong phần chính) và `k_test=128`, chậm hơn nhiều. `smoke` chỉ để thử luồng.

Retrain (huấn luyện lại từ đầu trên `D_r`) **không** có trong script vì tốn khoảng 27 giờ trên T4 (ước lượng). Báo cáo không có cột Retrain.

## 4. NLVR2

```bash
python multidelete.py nlvr2 [--profile smoke|fast|paper|full] [--no-baselines] [--ablation] [--fusion-only]
```

Mặc định là profile `paper` (`D_f` = 1000 mẫu, MultiDelete 5 epoch), đúng cấu hình của báo cáo.

Các bước:

1. Tải mô hình `albef_nlvr` đã fine-tune của LAVIS (Original).
2. Lấy tập huấn luyện từ `HuggingFaceM4/the_cauldron`, cấu hình `nlvr2`, bằng streaming. **Chỉ lấy một tập con** gồm `6 × |D_f| + 2000` mẫu (8000 mẫu ở `paper`), không phải toàn bộ train NLVR2. Tách câu và nhãn từ prompt bằng regex; script in ra vài ví dụ trước và sau khi tách để bạn kiểm tra bằng mắt.
3. Tập kiểm thử là `TIGER-Lab/NLVR2` (6967 mẫu).
4. Kiểm tra trước khi chạy thật: hàm forward tự viết phải khớp `model.forward` của LAVIS, và accuracy của Original trên mẫu train phải cao (nếu dưới 65% thì rất có thể parse sai nhãn, script cảnh báo). Script cũng tự phát hiện ảnh trái/phải có bị đảo thứ tự không.
5. Chạy: Original, MultiDelete, rồi FineTune và NegGrad (hai baseline này chạy 1 epoch). `--ablation` thêm -MD/-MKR/-UKR, `--fusion-only` thêm MultiDelete-F, `--no-baselines` bỏ FineTune và NegGrad.
6. Lưu `summary.csv`, `summary.png`, danh sách `D_f` (theo id mẫu trong pool).

Thời gian thực tế của từng lần chạy được ghi trong `summary.csv` (cột `unlearn_min`, `total_min`).

## 5. Demo

```bash
python multidelete.py demo
```

Huấn luyện lại MultiDelete và **lưu checkpoint** (khoảng 40 phút), rồi với 3 ảnh trong `D_f` và 2 ảnh trong `D_r` in ra 3 caption được xếp cao nhất của mô hình gốc và mô hình đã quên (ITC xếp hạng, ITM xếp lại 16 ứng viên đầu). Caption đúng của ảnh được đánh dấu ✓. Kho truy hồi gồm 1000 ảnh `D_f` và 1000 ảnh `D_r`. Hình và văn bản được lưu trong `<save>/flickr30k/demo/`.

**Lưu ý:** demo là phần bổ sung sau các thí nghiệm chính và không có số liệu trong báo cáo. Các lệnh `flickr30k` và `nlvr2` không phụ thuộc vào nó. Nếu cần xem lại, xem `demo.txt` và log.

## 5b. Demo 1 (notebook riêng)

`MultiDelete_Demo1.ipynb` là notebook độc lập, **không dùng bốn file `.py` ở trên**: ở §0b nó tự ghi ra ba file `.py` của riêng nó (`multidelete.py`, `run_md_retrieval.py`, `demo1_engine.py`) vào `/content/md_code`. Chạy trên Colab (GPU T4), lần đầu mất khoảng 40 phút để huấn luyện MultiDelete (cấu hình `fast`), sau đó lưu checkpoint trên Drive.

Demo so sánh Top-K ảnh được truy hồi cho một câu truy vấn trước và sau khi xoá, trên tập `D_f` và tập đối chứng `D_r`. Kết quả đã chạy sẵn trong notebook (200 truy vấn mỗi nhóm, hạng ảnh đích trước và sau khi xoá):

| Nhóm | R@1 | R@5 | Hạng trung vị | % truy vấn tụt hạng |
|---|---|---|---|---|
| `D_f` | 79,0 → 22,5 | 96,0 → 65,0 | #1 → #3 | 72,0 |
| `D_r` | 88,0 → 82,0 | 98,0 → 96,5 | #1 → #1 | 15,5 |

**Phạm vi:** kho truy hồi chỉ có 2000 ảnh (1000 `D_f` và 1000 `D_r`), truy vấn lấy từ dữ liệu huấn luyện, một seed. Các số này **không so được** với bảng ở mục 7 và chỉ dùng để minh hoạ. `D_r` cũng giảm nhẹ, nên đây là so sánh mô tả, không phải bằng chứng dữ liệu đã được xoá. Trước khi chạy, sửa `DRIVE_ROOT` và `FLICKR_SRC` ở §0 cho khớp với Drive của bạn.

## 6. Kết quả đầu ra

Trong thư mục `--save` (mặc định `./results`):

```
results/
  flickr30k/
    results/        <tag>.json cho từng lần chạy, summary_<profile>.csv, Df/ (tập xóa + split_manifest.json)
    logs/           log đầy đủ của từng lần chạy
    demo/           (nếu chạy demo) ảnh .png và demo.txt
  nlvr2/
    results/        <tag>.json, summary.csv, summary.png, config_used.json
    Df/nlvr2/       danh sách mẫu thuộc D_f
```

Thư mục `--work` (mặc định `./work`) chứa repo đã clone, dữ liệu, cache và checkpoint tạm; có thể xóa sau khi chạy xong.

## 7. Kết quả đã đạt được (để đối chiếu)

Đây là các số trong báo cáo, chạy một seed (42), `D_f` = 1000 ảnh, profile `fast`. Giữa hai lần chạy cùng cấu hình trên Kaggle và Colab số liệu lệch dưới 0,2. Chênh lệch dưới khoảng 1 điểm giữa hai cấu hình nên coi là nhiễu.

**Flickr30K** (TR và IR là R@1/5/10 trung bình trên tập kiểm thử; AUC là AUC của điểm ITC hoặc ITM, `D_f` là lớp dương):

| Phương pháp | TR | IR | AUC ITC | AUC ITM |
|---|---|---|---|---|
| Original | 98,33 | 93,63 | 48,71 | 48,65 |
| FineTune | 97,67 | 92,69 | 46,89 | 47,77 |
| NegGrad | 98,30 | 93,53 | 48,49 | 48,49 |
| DtD | 0,83 | 0,54 | 50,18 | 49,16 |
| MultiDelete | 96,80 | 89,01 | 24,49 | 3,59 |
| MD-ITC | 98,07 | 91,66 | 48,16 | 48,68 |
| Neg-Dr | 97,07 | 90,30 | 24,01 | 3,96 |
| -MD | 98,23 | 93,57 | 49,11 | 48,88 |
| -MKR | 52,20 | 22,86 | 47,96 | 51,35 |
| -UKR | 96,77 | 89,07 | 21,43 | 3,61 |

**NLVR2** (Test Acc là accuracy trên tập kiểm thử; Df-AUC tính trên margin của nhãn đúng, `D_f` là lớp dương):

| Phương pháp | Test Acc | Forget Acc (`D_f`) | Retain Acc (`D_r^eval`) | Df-AUC |
|---|---|---|---|---|
| Original | 82,78 | 97,2 | 98,2 | 49,49 |
| FineTune | 77,67 | 88,1 | 91,2 | 46,74 |
| NegGrad | 50,67 | 52,2 | 50,5 | 48,61 |
| MultiDelete | 81,11 | 65,6 | 95,3 | 44,24 |

FineTune và NegGrad ở lần chạy này là 1 epoch, MultiDelete là 5 epoch.

**Cách đọc AUC:** `D_f` và `D_r` đều là dữ liệu huấn luyện của Original, nên Original đã có AUC khoảng 50. AUC khoảng 50 **không** có nghĩa là mô hình quên tốt hay "như chưa từng thấy". Ví dụ DtD và -MKR trên Flickr30K có AUC gần 50 nhưng mô hình đã hỏng (độ chính xác sụp). Luôn xem AUC cùng với TR/IR. Chiều của AUC ở đây (lớp `D_f` là dương) có thể ngược với code repo gốc: cột `AUC_ITC_repo` trong bảng Flickr30K của script bằng `100 − AUC_ITC`.

## 8. Giới hạn đã biết

- `k_test=16` ở profile `fast`; không có Retrain.
- NLVR2 chỉ dùng tập con 8000 mẫu của train: 1000 mẫu cho `D_f`, 7000 mẫu còn lại làm `D_r` (dùng cho MKR khi xoá). `D_r^eval` (5000 mẫu đầu của `D_r`, dùng để tính AUC và Retain Acc) là tập con của `D_r`, nên không phải dữ liệu chưa thấy.
- Chỉ thử ALBEF.
- Số ngẫu nhiên tiêu thụ khác nhau giữa các lần chạy nên chênh lệch dưới khoảng 1 điểm là nhiễu.
- Các số trong báo cáo lấy từ ba notebook gốc (Kaggle và Colab). `multidelete.py` và `MultiDelete_all.ipynb` là bản gộp viết sau đó để chạy lại từ đầu bằng một lệnh.

