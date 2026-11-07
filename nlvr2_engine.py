# ===================== NLVR2 engine: sinh từ notebook cv_nlvr2_colab.ipynb =====================
import os, sys, json, time, glob, re, gc, copy, math, random, shutil, subprocess, platform, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")

PROFILE           = os.environ.get("MD_PROFILE", "paper")      # "smoke" -> "fast" -> "paper" -> "full"
SEED              = int(os.environ.get("MD_SEED", "42"))        # 42 | 87 | 21
RUN_BASELINES     = os.environ.get("MD_BASELINES", "1") == "1"  # FineTune + NegGrad
RUN_ABLATION      = os.environ.get("MD_ABLATION", "0") == "1"   # -MD / -MKR / -UKR
RUN_FUSION_ONLY   = os.environ.get("MD_FUSION_ONLY", "0") == "1"
UKR_MODE          = "repo"    # "repo": MSE trên toàn bộ token ảnh (đúng mã nguồn tác giả cho NLVR) | "paper": Eq.6
UNLEARN_EVAL_MODE = False
MD_LR, FT_LR, NEG_LR = 1e-5, 5e-5, 1e-5
NEG_STOP_CE       = 5.0       # NegGrad: dừng sớm khi CE trên batch Df vượt ngưỡng (chống sụp mô hình). None = tắt

PROFILES = {
    #            Df (list)                       epoch  batch  max_steps  Dr_eval x|Df|  N_test (0=tất cả)
    "smoke": dict(DF=[100],                          EPOCHS=1, BS=4, MAX_STEPS=25, DR_MULT=5, N_TEST=300),
    "fast":  dict(DF=[1000],                         EPOCHS=3, BS=4, MAX_STEPS=0,  DR_MULT=5, N_TEST=0),
    "paper": dict(DF=[1000],                         EPOCHS=5, BS=4, MAX_STEPS=0,  DR_MULT=5, N_TEST=0),
    "full":  dict(DF=[1000, 2000, 3000, 4000, 5000], EPOCHS=5, BS=4, MAX_STEPS=0,  DR_MULT=5, N_TEST=0),
}
P = PROFILES[PROFILE]
N_POOL = 6 * max(P["DF"]) + 2000      # số mẫu train cần lấy: Df + 5xDf (đánh giá AUC) + phần Dr huấn luyện
NW = 2                                # num_workers

WORK      = os.environ["MD_WORK"]
SAVE_ROOT = os.environ["MD_SAVE"]
REPO      = f"{WORK}/MultiDelete"
CACHE     = f"{WORK}/lavis_cache"
POOL_DIR  = f"{WORK}/nlvr2_train_pool"
RESULTS   = f"{SAVE_ROOT}/results"
LOGS      = f"{SAVE_ROOT}/logs"
STUBS     = f"{WORK}/stubs"
for d in (WORK, CACHE, POOL_DIR, RESULTS, LOGS):
    os.makedirs(d, exist_ok=True)

os.environ.update({
    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": "disabled",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONUNBUFFERED": "1",
})
print("PROFILE:", PROFILE, P, "| N_POOL:", N_POOL)
print("Kết quả lưu tại:", RESULTS)

# ---- nạp LAVIS từ repo đã clone + vá (do lệnh `setup` làm) ----
sys.path.insert(0, STUBS)
sys.path.insert(0, REPO)
os.chdir(REPO)
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG CÓ GPU")
assert torch.cuda.is_available(), "Chưa có GPU"
import lavis
from lavis.models import load_model
print("IMPORT LAVIS OK")


# ===================== §2a. HÀM PARSE (thuần Python) =====================
INSTR  = re.compile(r"true or false|true/false|true or not|answer with|answer the|answer:|^\(?[A-D][\.\)]\s|^(options|choices)\s*:", re.I)
PREFIX = re.compile(r"^(question|statement|sentence|text|caption)\s*:\s*", re.I)
MARKERS = [r"true or false\s*\?\s*(.+)",              # "... Is it true or false? <statement>"
           r"statement[^:\n]{0,80}:\s*(.+)",             # "... statement describing them: <statement>"
           r"true or false[^:\n]{0,20}:\s*(.+)"]
TAIL   = re.compile(r"^the first image is the image on the left|^(is|are|does|do)\s+(this|the|that|it|the following)\b.*\?$|^(answer|please|respond|choose|select)\b|true or false", re.I)
OPT    = re.compile(r"\b([A-D])[\.\):]\s*(true|false|yes|no)\b", re.I)

QUOTED = re.compile(r'["\u201c](.+)["\u201d]', re.S)       # statement nằm trong dấu ngoặc kép: ... is the sentence "<statement>" ...
PARSE_VERSION = "v2-quoted"

def extract_statement(user):
    """Lấy câu statement từ prompt của user. Ưu tiên chuỗi trong dấu ngoặc kép (định dạng thật của the_cauldron/nlvr2),
    nếu không có thì dùng heuristic dự phòng."""
    u = user.replace("<image>", " ").strip()
    mq = QUOTED.search(u)
    if mq and len(mq.group(1).strip()) > 3:
        return mq.group(1).strip().strip("\"'\u201c\u201d ")
    for pat in MARKERS:
        m = re.search(pat, u, flags=re.I | re.S)
        if m:
            u = m.group(1); break
    lines = [PREFIX.sub("", l.strip()) for l in u.split("\n") if l.strip()]
    keep = [l for l in lines if not INSTR.search(l)]
    s = max(keep, key=len) if keep else (lines[0] if lines else "")
    sents = re.split(r"(?<=[.?!])\s+", s)              # bỏ các câu hướng dẫn dính phía sau ("Is this statement correct? Answer yes or no.")
    kept = [x for x in sents if not TAIL.search(x.strip())]
    if kept: s = " ".join(kept)
    return s.strip(" \"'")

def parse_label(user, assistant):
    """Trả về 1 (True) / 0 (False) / None (không nhận ra)."""
    a = (assistant or "").strip()
    opts = {m.group(1).upper(): m.group(2).lower() in ("true", "yes") for m in OPT.finditer(user or "")}
    ml = re.fullmatch(r"(?:answer\s*:\s*)?\(?([A-D])\)?\.?", a, flags=re.I)
    if opts and ml and ml.group(1).upper() in opts:
        return int(opts[ml.group(1).upper()])
    m = re.search(r"\b(true|false|yes|no)\b", a, flags=re.I)
    if m:
        return int(m.group(1).lower() in ("true", "yes"))
    return None

def parse_turn(user, assistant):
    lab = parse_label(user, assistant)
    st = extract_statement(user or "")
    if lab is None or not st:
        return None
    return st, lab

def parse_test_row(question, options, answer):
    """TIGER-Lab/NLVR2: statement nằm sau 'Is it true or false?'; nhãn lấy theo NỘI DUNG option (thứ tự bị xáo)."""
    stmt = question.split("Is it true or false?", 1)[-1].strip()
    mp = {}
    for o in options:
        m = re.match(r"\(([A-Z])\)\s*(true|false)", o.strip(), flags=re.I)
        if m:
            mp[m.group(1).upper()] = int(m.group(2).lower() == "true")
    return stmt, mp[answer.strip().upper()]

def make_splits(pool, df_size, seed, dr_mult):
    """Df: df_size mẫu đầu của hoán vị theo seed (nên Df 1K ⊂ 2K ⊂ ... như notebook Flickr30K).
    Dr = phần còn lại; dr_eval = dr_mult*|Df| mẫu đầu của Dr (dùng tính AUC)."""
    perm = np.random.RandomState(seed).permutation(len(pool))
    assert len(pool) >= df_size * (1 + dr_mult), f"Pool {len(pool)} mẫu < {df_size*(1+dr_mult)} cần cho Df={df_size}"
    df_idx, rest = perm[:df_size], perm[df_size:]
    dr_eval_idx = rest[: dr_mult * df_size]
    pick = lambda idx: [pool[i] for i in idx]
    return pick(df_idx), pick(rest), pick(dr_eval_idx), df_idx

def df_dr_auc(df_scores, dr_scores, n_chunks=5):
    """ROC-AUC (x100) phân biệt Df (nhãn 1) với Dr (nhãn 0); chia Dr thành n_chunks khối cỡ |Df| (như notebook gốc)."""
    from sklearn.metrics import roc_auc_score
    n = len(df_scores); aucs = []
    for c in range(n_chunks):
        chunk = dr_scores[c * n:(c + 1) * n]
        if len(chunk) == 0:
            break
        y = np.r_[np.ones(n), np.zeros(len(chunk))]
        aucs.append(100.0 * roc_auc_score(y, np.r_[df_scores, chunk]))
    return float(np.mean(aucs)), [float(a) for a in aucs]
print("Hàm parse đã sẵn sàng.")

# ===================== §2b. TẠO POOL TỪ the_cauldron/nlvr2 =====================
from datasets import load_dataset
try:
    from datasets import Sequence, Image as HFImage
except Exception:
    Sequence = HFImage = None

def _write_bytes(b, stem):
    ext = ".png" if b[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
    with open(stem + ext, "wb") as f:
        f.write(b)
    return stem + ext

def build_pool(n_samples, pool_dir):
    meta_f = f"{pool_dir}/pool.json"
    ver_f = f"{pool_dir}/parse_version.txt"
    same_ver = os.path.exists(ver_f) and open(ver_f).read().strip() == PARSE_VERSION
    if os.path.exists(meta_f) and not same_ver:
        print("Pool cũ được tạo bằng parser khác -> tạo lại.")
    if os.path.exists(meta_f) and same_ver:
        pool = json.load(open(meta_f))
        if len(pool) >= n_samples and all(os.path.exists(p["img0"]) and os.path.exists(p["img1"]) for p in pool[:50]):
            print(f"Dùng lại pool đã có: {len(pool)} mẫu")
            return pool[:n_samples]
    os.makedirs(f"{pool_dir}/img", exist_ok=True)
    ds = load_dataset("HuggingFaceM4/the_cauldron", "nlvr2", split="train", streaming=True)
    raw = False
    if Sequence is not None:
        try:
            ds = ds.cast_column("images", Sequence(HFImage(decode=False))); raw = True
        except Exception as e:
            print("Không cast được decode=False, sẽ giải mã rồi lưu PNG (chậm hơn):", repr(e)[:150])
    pool, bad_parse, bad_imgs, raw_examples, t0 = [], 0, 0, [], time.time()
    for r_idx, row in enumerate(ds):
        imgs = row["images"]
        if len(imgs) != 2:
            bad_imgs += 1; continue
        paths = []
        for k, im in enumerate(imgs):
            stem = f"{pool_dir}/img/{r_idx:06d}_{k}"
            if isinstance(im, dict):                       # decode=False -> {"bytes":..., "path":...}
                b = im.get("bytes")
                if b is None:
                    b = open(im["path"], "rb").read()
                paths.append(_write_bytes(b, stem))
            else:                                          # PIL
                im.convert("RGB").save(stem + ".png", compress_level=1); paths.append(stem + ".png")
        if len(raw_examples) < 6:
            raw_examples.append(row["texts"])
        for t_idx, t in enumerate(row["texts"]):
            out = parse_turn(t.get("user", ""), t.get("assistant", ""))
            if out is None:
                bad_parse += 1; continue
            pool.append(dict(id=f"{r_idx}_{t_idx}", img0=paths[0], img1=paths[1], sentence=out[0], label=out[1]))
        if r_idx % 500 == 0:
            print(f"  hàng {r_idx}: {len(pool)} mẫu | {time.time()-t0:.0f}s")
        if len(pool) >= n_samples:
            break
    pool = pool[:n_samples]
    json.dump(pool, open(meta_f, "w"))
    open(ver_f, "w").write(PARSE_VERSION)
    json.dump(raw_examples, open(f"{pool_dir}/raw_examples.json", "w"), ensure_ascii=False, indent=1)
    print(f"Pool: {len(pool)} mẫu | bỏ qua: {bad_parse} turn không parse được, {bad_imgs} dòng không có đúng 2 ảnh | raw={raw}")
    return pool

POOL = build_pool(N_POOL, POOL_DIR)
print("Tỉ lệ nhãn True trong pool:", round(float(np.mean([p["label"] for p in POOL])), 3))

raw_ex = json.load(open(f"{POOL_DIR}/raw_examples.json"))
print("=== PROMPT GỐC (3 dòng đầu) ===")
for texts in raw_ex[:3]:
    for t in texts[:2]:
        print("USER     :", repr(t.get("user", ""))[:400])
        print("ASSISTANT:", repr(t.get("assistant", ""))[:120], "\n")
print("=== SAU KHI PARSE ===")
for p in random.Random(0).sample(POOL, 8):
    print(f"[{p['label']}] {p['sentence']}")
from PIL import Image
im0 = Image.open(POOL[0]["img0"]); im1 = Image.open(POOL[0]["img1"])
print("\nKích thước ảnh mẫu đầu:", im0.size, im1.size)

# ===================== §2d. TEST SET =====================
try:
    HF_TEST = load_dataset("TIGER-Lab/NLVR2", "nlvr2", split="test")
except Exception as e:
    print("Thử lại không chỉ định config:", repr(e)[:100])
    HF_TEST = load_dataset("TIGER-Lab/NLVR2", split="test")
meta = HF_TEST.remove_columns([c for c in HF_TEST.column_names if c not in ("question", "options", "answer", "id")])
TEST_ITEMS = [dict(id=r["id"], sentence=s, label=l) for r in meta
              for s, l in [parse_test_row(r["question"], r["options"], r["answer"])]]
print("Test:", len(TEST_ITEMS), "mẫu | tỉ lệ True:", round(float(np.mean([t["label"] for t in TEST_ITEMS])), 3))
print(TEST_ITEMS[0]); print(TEST_ITEMS[1])
rng_t = np.random.RandomState(0)
TEST_IDX = (list(range(len(TEST_ITEMS))) if P["N_TEST"] == 0
            else sorted(rng_t.permutation(len(TEST_ITEMS))[: P["N_TEST"]].tolist()))
print("Số mẫu test dùng:", len(TEST_IDX))

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from lavis.processors.blip_processors import BlipImageEvalProcessor, BlipCaptionProcessor

DEVICE = "cuda"
AMP = True
VIS = BlipImageEvalProcessor(image_size=384)
TXT = BlipCaptionProcessor()
SWAP_POOL, SWAP_TEST = False, False       # được cập nhật ở §4b nếu thứ tự ảnh trái/phải bị đảo

class PairDS(Dataset):
    """Mẫu NLVR2 lưu trên đĩa: (img0, img1, sentence, label)."""
    def __init__(self, items, swap=False):
        self.items, self.swap = items, swap
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        a = VIS(Image.open(it["img0"]).convert("RGB")); b = VIS(Image.open(it["img1"]).convert("RGB"))
        if self.swap: a, b = b, a
        return {"image0": a, "image1": b, "text_input": TXT(it["sentence"]), "label": int(it["label"]), "idx": i}

class HFTestDS(Dataset):
    """Test set đọc trực tiếp từ HuggingFace dataset (giải mã ảnh khi cần)."""
    def __init__(self, hf_images, items, indices, swap=False):
        self.hf, self.items, self.indices, self.swap = hf_images, items, list(indices), swap
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        j = self.indices[i]
        ims = self.hf[j]["images"]
        a = VIS(ims[0].convert("RGB")); b = VIS(ims[1].convert("RGB"))
        if self.swap: a, b = b, a
        it = self.items[j]
        return {"image0": a, "image1": b, "text_input": TXT(it["sentence"]), "label": int(it["label"]), "idx": i}

HF_TEST_IMG = HF_TEST.select_columns(["images"])
def make_test_ds(indices=None, swap=None):
    return HFTestDS(HF_TEST_IMG, TEST_ITEMS, TEST_IDX if indices is None else indices, SWAP_TEST if swap is None else swap)

# ---------------- mô hình ----------------
def load_model_nlvr():
    """Tải ALBEF-NLVR đã fine-tune của LAVIS (tự tải albef_nlvr_lavis.pt vào lavis_cache)."""
    return load_model("albef_nlvr", "nlvr", is_eval=False, device="cpu")

def drop_momentum(m):
    """Bỏ encoder động lượng (chỉ cần khi huấn luyện có distillation) để giải phóng VRAM."""
    for n in ("visual_encoder_m", "text_encoder_m", "cls_head_m"):
        if hasattr(m, n): delattr(m, n)
    m.model_pairs = []; m.use_distill = False
    return m

def nlvr_forward(model, img0, img1, sentences):
    """Sao chép đúng AlbefNLVR.forward; trả (logits, CLS fusion, (token ảnh 0, token ảnh 1))."""
    tok = model.tokenizer(sentences, padding="longest", truncation=True,
                          max_length=model.max_txt_len, return_tensors="pt").to(img0.device)
    B = img0.size(0)
    emb = model.visual_encoder.forward_features(torch.cat([img0, img1], dim=0))
    att = torch.ones(emb.shape[:-1], dtype=torch.long, device=emb.device)
    e0, e1 = torch.split(emb, B)
    out = model.text_encoder(tok.input_ids, attention_mask=tok.attention_mask,
                             encoder_hidden_states=[e0, e1],
                             encoder_attention_mask=[att[:B], att[B:]], return_dict=True)
    cls = out.last_hidden_state[:, 0, :]
    return model.cls_head(cls), cls, (e0, e1)

def text_only_cls(model, sentences, device):
    """CLS của 6 lớp văn bản thuần (mode='text'), dùng cho UKR_MODE='paper'."""
    tok = model.tokenizer(sentences, padding="longest", truncation=True,
                          max_length=model.max_txt_len, return_tensors="pt").to(device)
    out = model.text_encoder(tok.input_ids, attention_mask=tok.attention_mask, return_dict=True, mode="text")
    return out.last_hidden_state[:, 0, :]

@torch.no_grad()
def evaluate(model, ds, bs=32):
    """Accuracy + margin của nhãn đúng (logit[y] - logit[1-y]) cho từng mẫu."""
    model.eval()
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=NW, pin_memory=True)
    margins, labels, preds = [], [], []
    for b in loader:
        im0 = b["image0"].to(DEVICE, non_blocking=True); im1 = b["image1"].to(DEVICE, non_blocking=True)
        y = b["label"].to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            logits, _, _ = nlvr_forward(model, im0, im1, list(b["text_input"]))
        lg = logits.float()
        margins.append((lg.gather(1, y[:, None]) - lg.gather(1, (1 - y)[:, None])).squeeze(1).cpu())
        labels.append(y.cpu()); preds.append(lg.argmax(1).cpu())
    margins = torch.cat(margins).numpy(); labels = torch.cat(labels).numpy(); preds = torch.cat(preds).numpy()
    return dict(acc=float(100 * (preds == labels).mean()), margin=margins, label=labels)

def quick_eval(model, df_ds, dr_ds):
    """Df-AUC (margin) + accuracy trên Df và trên Dr-eval."""
    d, r = evaluate(model, df_ds), evaluate(model, dr_ds)
    auc, chunks = df_dr_auc(d["margin"], r["margin"], n_chunks=max(1, len(r["margin"]) // max(1, len(d["margin"]))))
    return dict(df_acc=d["acc"], dr_acc=r["acc"], df_auc=auc, df_auc_chunks=chunks,
                df_margin=float(d["margin"].mean()), dr_margin=float(r["margin"].mean()))
print("Đã định nghĩa dataset / mô hình / đánh giá.")

# ===================== §4a. TẢI ORIGINAL + KIỂM TRA KHỚP FORWARD =====================
t0 = time.time()
model = drop_momentum(load_model_nlvr()).to(DEVICE).eval()
print(f"Đã tải Original trong {time.time()-t0:.0f}s | tham số: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

# nlvr_forward phải cho đúng logits như model.forward(is_train=False)
x0 = torch.randn(2, 3, 384, 384, device=DEVICE); x1 = torch.randn(2, 3, 384, 384, device=DEVICE)
sents = ["there is a ferret in tall grass", "the left image shows two dogs"]
with torch.no_grad():
    mine, _, _ = nlvr_forward(model, x0, x1, sents)
    ref = model({"image0": x0, "image1": x1, "text_input": sents, "label": torch.tensor([0, 1], device=DEVICE)}, is_train=False)["predictions"]
print("max |logits_mine - logits_lavis| =", float((mine - ref).abs().max()))
assert torch.allclose(mine, ref, atol=1e-3), "nlvr_forward KHÔNG khớp model.forward -> kiểm tra lại phiên bản LAVIS"
print("nlvr_forward khớp model.forward ✔")

def pick_order(make_ds, n_label, items_or_idx):
    r0 = evaluate(model, make_ds(items_or_idx, False))["acc"]
    r1 = evaluate(model, make_ds(items_or_idx, True))["acc"]
    print(f"  {n_label}: acc thứ tự gốc = {r0:.1f}% | đảo ảnh = {r1:.1f}%")
    return r1 > r0 + 3.0, max(r0, r1)

N_CHECK = 300
sub_pool = random.Random(0).sample(POOL, N_CHECK)
print("Kiểm tra mẫu TRAIN (Original đã học các mẫu này):")
SWAP_POOL, acc_pool = pick_order(lambda it, sw: PairDS(it, swap=sw), f"train ({N_CHECK} mẫu)", sub_pool)
sub_test = sorted(np.random.RandomState(1).permutation(len(TEST_ITEMS))[:N_CHECK].tolist())
print("Kiểm tra mẫu TEST:")
SWAP_TEST, acc_test = pick_order(lambda idx, sw: make_test_ds(idx, sw), f"test ({N_CHECK} mẫu)", sub_test)
print("\nSWAP_POOL =", SWAP_POOL, "| SWAP_TEST =", SWAP_TEST)
if acc_pool < 65:
    print("⚠️  Accuracy trên mẫu train quá thấp (<65%): rất có thể parse nhãn/statement SAI (nhãn bị đảo, statement dính hướng dẫn...). "
          "Xem lại §2c, sửa hàm parse, xoá pool.json rồi chạy lại §2b. ĐỪNG chạy tiếp §5.")
elif acc_pool < 75:
    print("⚠️  Accuracy train hơi thấp (65–75%): kiểm tra thêm statement parse có chính xác không.")
else:
    print("✔ Parse có vẻ đúng.")
if acc_test < 70:
    print("⚠️  Accuracy test thấp hơn kỳ vọng (~80%): kiểm tra parse test hoặc thứ tự ảnh.")
del model; gc.collect(); torch.cuda.empty_cache()

def T(name, df):
    """Tên lần chạy: có PROFILE để profile khác nhau (smoke/fast/paper) không bỏ qua nhầm nhau."""
    return f"{PROFILE}_{name}_df{df}_s{SEED}"

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def forever(loader):
    while True:
        for b in loader:
            yield b

def to_dev(b):
    return {k: (v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}

def rotate_texts(texts):
    """Dịch vòng >=1 vị trí trong batch: không cặp nào ghép lại đúng câu của chính nó."""
    B = len(texts)
    if B < 2: return list(texts)
    shift = random.randint(1, B - 1)
    return [texts[(i + shift) % B] for i in range(B)]

def multidelete_step(model, model_ori, s_df, s_dr, scaler, alpha, beta, gamma):
    """Hai pha backward: (MD + UKR) trên Df, rồi MKR trên Dr. Trả về dict giá trị loss."""
    # ---- Pha 1: Df ----
    with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
        _, cls_df, (e0, e1) = nlvr_forward(model, s_df["image0"], s_df["image1"], s_df["text_input"])
        with torch.no_grad():
            _, cls_rand, (o0, o1) = nlvr_forward(model_ori, s_df["image0"], s_df["image1"], rotate_texts(s_df["text_input"]))
        l_md = F.mse_loss(cls_df.float(), cls_rand.float())                       # Eq.2: f'(I,T) ~ f(I,T_q)
        l_ukr = torch.zeros((), device=DEVICE)
        if gamma > 0:                                                              # Eq.6: giữ biểu diễn đơn phương thức trên Df
            if UKR_MODE == "repo":
                l_ukr = 0.5 * (F.mse_loss(e0.float(), o0.float()) + F.mse_loss(e1.float(), o1.float()))
            else:
                with torch.no_grad():
                    t_ori = text_only_cls(model_ori, s_df["text_input"], DEVICE)
                t_new = text_only_cls(model, s_df["text_input"], DEVICE)
                v_new = torch.cat([e0[:, 0], e1[:, 0], t_new], dim=1).float()
                v_ori = torch.cat([o0[:, 0], o1[:, 0], t_ori], dim=1).float()
                l_ukr = F.mse_loss(v_new, v_ori)
        loss_df = alpha * l_md + gamma * l_ukr
    if loss_df.requires_grad:
        scaler.scale(loss_df).backward()
    # ---- Pha 2: Dr ----
    l_mkr = torch.zeros((), device=DEVICE)
    if beta > 0:
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            _, cls_dr, _ = nlvr_forward(model, s_dr["image0"], s_dr["image1"], s_dr["text_input"])
            with torch.no_grad():
                _, cls_dr_o, _ = nlvr_forward(model_ori, s_dr["image0"], s_dr["image1"], s_dr["text_input"])
            l_mkr = F.mse_loss(cls_dr.float(), cls_dr_o.float())                  # Eq.4
            loss_dr = beta * l_mkr
        scaler.scale(loss_dr).backward()
    return dict(md=float(l_md), ukr=float(l_ukr), mkr=float(l_mkr))

def get_before(model, df_size, df_ds, dr_eval_ds):
    f = f"{RESULTS}/before_df{df_size}_s{SEED}_pool{len(POOL)}_t{len(TEST_IDX)}_sw{int(SWAP_POOL)}{int(SWAP_TEST)}.json"
    if os.path.exists(f):
        return json.load(open(f))
    q = quick_eval(model, df_ds, dr_eval_ds)
    t = evaluate(model, make_test_ds())
    d = dict(test_acc=t["acc"], **q)
    json.dump(d, open(f, "w"), indent=1)
    return d

def _run_method(method, df_size, tag, lr, epochs, alpha, beta, gamma, fusion_only, bs):
    set_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    df_items, dr_items, dr_eval_items, _ = make_splits(POOL, df_size, SEED, P["DR_MULT"])
    df_ds, dr_ds, dr_eval_ds = (PairDS(x, SWAP_POOL) for x in (df_items, dr_items, dr_eval_items))
    model = drop_momentum(load_model_nlvr()).to(DEVICE)
    rec = dict(tag=tag, method=method, df_size=df_size, seed=SEED, bs=bs, lr=lr, epochs=epochs, profile=PROFILE,
               alpha=alpha, beta=beta, gamma=gamma, fusion_only=fusion_only, ukr_mode=UKR_MODE,
               n_df=len(df_ds), n_dr=len(dr_ds), n_dr_eval=len(dr_eval_ds), n_test=len(TEST_IDX),
               swap_pool=SWAP_POOL, swap_test=SWAP_TEST, gpu=torch.cuda.get_device_name(0))
    rec["before"] = get_before(model, df_size, df_ds, dr_eval_ds)
    print(f"[{tag}] BEFORE: test_acc={rec['before']['test_acc']:.2f} | Df-AUC={rec['before']['df_auc']:.2f} "
          f"| df_acc={rec['before']['df_acc']:.1f} dr_acc={rec['before']['dr_acc']:.1f}")

    if method != "original":
        model_ori = None
        if method == "multidelete":
            model_ori = copy.deepcopy(model).eval()
            for p_ in model_ori.parameters(): p_.requires_grad = False
        params = [(n, p_) for n, p_ in model.named_parameters() if p_.requires_grad]
        if fusion_only:
            keep = [(n, p_) for n, p_ in params if "crossattention" in n or n.startswith("cls_head")]
            names = {n for n, _ in keep}
            for n, p_ in model.named_parameters():
                if n not in names: p_.requires_grad = False
            params = keep
        rec["trainable_params_M"] = sum(p_.numel() for _, p_ in params) / 1e6
        opt = torch.optim.AdamW([p_ for _, p_ in params], lr=lr, weight_decay=0.02)
        scaler = torch.amp.GradScaler("cuda", enabled=AMP)
        mk = lambda ds: DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=NW, pin_memory=True)
        df_loader = mk(df_ds); dr_iter = forever(mk(dr_ds))
        steps = len(df_loader) if not P["MAX_STEPS"] else min(P["MAX_STEPS"], len(df_loader))
        n_epochs = epochs if method == "multidelete" else 1
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_epochs * steps), eta_min=lr * 0.1)
        hist, step_idx, stopped, t_un = [], 0, None, time.time()
        for ep in range(n_epochs):
            model.train(not (UNLEARN_EVAL_MODE and method == "multidelete"))
            for i, s_df in enumerate(df_loader):
                if i >= steps: break
                s_df = to_dev(s_df)
                opt.zero_grad(set_to_none=True)
                if method == "multidelete":
                    parts = multidelete_step(model, model_ori, s_df, to_dev(next(dr_iter)), scaler, alpha, beta, gamma)
                else:
                    batch = s_df if method == "neggrad" else to_dev(next(dr_iter))
                    with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                        logits, _, _ = nlvr_forward(model, batch["image0"], batch["image1"], batch["text_input"])
                        ce = F.cross_entropy(logits.float(), batch["label"])
                    if method == "neggrad" and NEG_STOP_CE is not None and ce.item() > NEG_STOP_CE:
                        stopped = f"dừng sớm ở bước {step_idx}: CE={ce.item():.2f} > {NEG_STOP_CE}"
                        print("  ", stopped); break
                    scaler.scale(-ce if method == "neggrad" else ce).backward()
                    parts = dict(ce=float(ce))
                if method == "neggrad":
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p_ for _, p_ in params], 1.0)
                scaler.step(opt); scaler.update(); sched.step(); step_idx += 1
                if step_idx % 20 == 0 or step_idx == 1:
                    parts.update(step=step_idx, epoch=ep, lr=sched.get_last_lr()[0], gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
                    hist.append(parts); print(f"  {tag} " + json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in parts.items()}))
            if stopped: break
        rec["time_unlearn_s"] = time.time() - t_un; rec["train_history"] = hist; rec["stopped"] = stopped
        rec["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
        q = quick_eval(model, df_ds, dr_eval_ds); t = evaluate(model, make_test_ds())
        rec["after"] = dict(test_acc=t["acc"], **q)
        print(f"[{tag}] AFTER : test_acc={t['acc']:.2f} | Df-AUC={q['df_auc']:.2f} | df_acc={q['df_acc']:.1f} dr_acc={q['dr_acc']:.1f} "
              f"| unlearn {rec['time_unlearn_s']/60:.1f} phút")
    rec["time_total_s"] = time.time() - t_start
    del model; gc.collect(); torch.cuda.empty_cache()
    return rec

def run_method(method, df_size, tag, lr=MD_LR, epochs=None, alpha=1.0, beta=1.0, gamma=1.0, fusion_only=False, bs=None):
    out = f"{RESULTS}/{tag}.json"
    if os.path.exists(out):
        print(f"[{tag}] đã có kết quả -> bỏ qua"); return json.load(open(out))
    bs = bs or P["BS"]; epochs = epochs or P["EPOCHS"]
    oom, rec = False, None
    try:
        rec = _run_method(method, df_size, tag, lr, epochs, alpha, beta, gamma, fusion_only, bs)
    except torch.cuda.OutOfMemoryError:
        oom = True
    if oom:
        gc.collect(); torch.cuda.empty_cache()
        if bs <= 1: raise RuntimeError("Hết VRAM ngay cả với bs=1")
        print(f"!! OOM với bs={bs} -> chạy lại với bs={bs // 2} (SAI KHÁC so với cấu hình gốc, ghi vào báo cáo)")
        return run_method(method, df_size, tag, lr, epochs, alpha, beta, gamma, fusion_only, bs // 2)
    json.dump(rec, open(out, "w"), indent=1, default=float)
    return rec
print("Sẵn sàng chạy.")

for df in P["DF"]:
    # --- Original (chỉ đánh giá BEFORE) ---
    r = run_method("original", df, T("original", df))
    b = r["before"]
    print(f"\nOriginal  Df={df}: test_acc={b['test_acc']:.2f} | Df-AUC={b['df_auc']:.2f} (kỳ vọng ≈ 50) | "
          f"df_acc={b['df_acc']:.1f} dr_acc={b['dr_acc']:.1f}\n")
    # --- MultiDelete ---
    run_method("multidelete", df, T("multidelete", df), lr=MD_LR)
    if RUN_FUSION_ONLY:
        run_method("multidelete", df, T("multidelete-F", df), lr=MD_LR, fusion_only=True)
    if RUN_ABLATION:
        for nm, kw in [("noMD", dict(alpha=0.0)), ("noMKR", dict(beta=0.0)), ("noUKR", dict(gamma=0.0))]:
            run_method("multidelete", df, T(f"multidelete-{nm}", df), lr=MD_LR, **kw)

for df in P["DF"]:
    if RUN_BASELINES:
        run_method("ft", df, T("ft", df), lr=FT_LR)
        run_method("neggrad", df, T("neggrad", df), lr=NEG_LR)

import pandas as pd
rows = []
for f in sorted(glob.glob(f"{RESULTS}/*.json")):
    if os.path.basename(f).startswith("before_"): continue
    r = json.load(open(f))
    if "tag" not in r: continue
    a = r.get("after") or r["before"]
    rows.append(dict(run=r["tag"], profile=r.get("profile"), method=r["method"], df_size=r["df_size"], seed=r["seed"], bs=r["bs"],
                     epochs=r["epochs"] if r["method"] == "multidelete" else (0 if r["method"] == "original" else 1),
                     test_acc=a["test_acc"], df_acc=a["df_acc"], dr_acc=a["dr_acc"], df_AUC=a["df_auc"],
                     unlearn_min=round(r.get("time_unlearn_s", 0) / 60, 1), total_min=round(r["time_total_s"] / 60, 1),
                     peak_gpu_gb=r.get("peak_gpu_gb"), note=r.get("stopped") or ""))
df_res = pd.DataFrame(rows)
if len(df_res) and "profile" in df_res: df_res = df_res[df_res.profile == PROFILE]   # chỉ hiện profile hiện tại (bỏ dòng này để xem tất cả)
df_res = df_res.round(2)
pd.set_option("display.width", 250, "display.max_columns", 30)
df_res.to_csv(f"{RESULTS}/summary.csv", index=False)
print(df_res.to_string())

paper = pd.DataFrame([
    ["Retrain", 80.3, 50.3], ["FineTune", 80.3, 49.8], ["NegGrad", 77.3, 50.6], ["UL", 76.3, 50.4],
    ["MultiDelete", 80.8, 23.5], ["MultiDelete-F", 80.4, 26.4],
    ["Ablation: -MD", 80.3, 50.3], ["Ablation: -UKR", 79.2, 25.8], ["Ablation: -MKR", 77.1, 25.6]],
    columns=["method (paper, NLVR2/ALBEF)", "D_Test", "D_f"])
print("\nPaper – Table 2 / Table 4 (NLVR2, ALBEF):"); print(paper.to_string())

import matplotlib.pyplot as plt
d = df_res[df_res.method != "original"]
if len(d):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for _, g in d.iterrows():
        ax[0].scatter(g.df_AUC, g.test_acc, s=60, label=g.run)
    o = df_res[df_res.method == "original"]
    if len(o): ax[0].scatter(o.df_AUC, o.test_acc, marker="*", s=200, c="k", label="original")
    ax[0].axvline(50, ls="--", c="gray")
    ax[0].set_xlabel("Df-AUC (margin; 50 = không phân biệt được Df/Dr)"); ax[0].set_ylabel("Test accuracy (%)")
    ax[0].legend(fontsize=6); ax[0].set_title("Quên Df  vs  giữ chất lượng")
    d.plot.bar(x="run", y="unlearn_min", ax=ax[1], legend=False); ax[1].set_ylabel("phút"); ax[1].tick_params(axis="x", labelsize=6)
    ax[1].set_title("Thời gian unlearn")
    plt.tight_layout(); plt.savefig(f"{RESULTS}/summary.png", dpi=150); plt.close()
else:
    print("Chưa có kết quả unlearn để vẽ.")

# Lưu danh sách Df (theo id trong pool) và đóng gói bàn giao
os.makedirs(f"{SAVE_ROOT}/Df/nlvr2", exist_ok=True)
for df in P["DF"]:
    _, _, _, df_idx = make_splits(POOL, df, SEED, P["DR_MULT"])
    open(f"{SAVE_ROOT}/Df/nlvr2/sample-{SEED}-df{df}.txt", "w").write("\n".join(POOL[i]["id"] for i in df_idx) + "\n")
json.dump(dict(profile=PROFILE, P=P, seed=SEED, ukr_mode=UKR_MODE, swap_pool=SWAP_POOL, swap_test=SWAP_TEST,
               neg_stop_ce=NEG_STOP_CE, lrs=dict(md=MD_LR, ft=FT_LR, neg=NEG_LR), n_pool=len(POOL)),
          open(f"{RESULTS}/config_used.json", "w"), indent=1)
print("Df lưu tại:", f"{SAVE_ROOT}/Df/nlvr2")
