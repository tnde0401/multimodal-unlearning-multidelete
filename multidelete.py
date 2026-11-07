
"""
multidelete.py  -  Tái lập MultiDelete (ECCV 2024) trên ALBEF với Flickr30K và NLVR2
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_URL = "https://github.com/CLU-UML/MultiDelete.git"
BACKBONE = "albef"
CFG = f"configs/original/{BACKBONE}/retrieval_flickr30k.yaml"

PIP_PACKAGES = [
    "transformers==4.36.2", "tokenizers>=0.14,<0.19", "timm==0.9.16",
    "omegaconf", "iopath", "einops", "ftfy", "fairscale", "webdataset", "contexttimer", "pycocoevalcap",
    "pycocotools", "sentencepiece", "scikit-learn", "pandas", "matplotlib", "opencv-python-headless",
    "datasets", "pyarrow", "kaggle",
]

# Cấu hình theo profile cho Flickr30K. "fast" là cấu hình đã dùng cho số liệu trong báo cáo.
FLICKR_PROFILES = {
    "smoke": dict(df=100, epochs=1, k_test=8, bs=8, max_steps=25),      # kiểm tra luồng chạy, ~30 phút
    "fast": dict(df=1000, epochs=3, k_test=16, bs=16, max_steps=0),     # báo cáo, ~5 giờ cho toàn bộ trên T4
    "paper": dict(df=1000, epochs=5, k_test=128, bs=16, max_steps=0),   # sát paper, rất lâu
}
NLVR2_PROFILES = ("smoke", "fast", "paper", "full")

# Tên thí nghiệm Flickr30K: (nhãn trong bảng, hậu tố tag, các tham số thêm cho engine, mô tả)
FLICKR_ORDER = ["original", "main", "itc", "negdr", "noMD", "noMKR", "noUKR", "ft", "neggrad", "dtd"]
FLICKR_OPTIONAL = ["mdF", "neggradraw"]      # chỉ chạy khi gọi rõ bằng --only


def log(*a):
    print("[multidelete]", *a, flush=True)


# ----------------------------------------------------------------------------------------------------------
# Đường dẫn và tiện ích
# ----------------------------------------------------------------------------------------------------------
class Paths:
    def __init__(self, a):
        self.work = os.path.abspath(a.work)
        self.save = os.path.abspath(a.save)
        self.repo = f"{self.work}/MultiDelete"
        self.cache = f"{self.work}/lavis_cache"
        self.stubs = f"{self.work}/stubs"
        self.ckpts = f"{self.work}/checkpoints"
        self.fl_results = f"{self.save}/flickr30k/results"
        self.fl_logs = f"{self.save}/flickr30k/logs"
        self.nl_root = f"{self.save}/nlvr2"
        self.demo_out = f"{self.save}/flickr30k/demo"


def make_env(P):
    env = dict(os.environ)
    env.update({
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",      # PyTorch >= 2.6 khi nạp checkpoint LAVIS
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONUNBUFFERED": "1",
    })
    env["PYTHONPATH"] = os.pathsep.join([P.stubs, P.repo, env.get("PYTHONPATH", "")])
    return env


def run_logged(cmd, name, P, logdir, cwd=None, check=True, echo_keys=(), dry=False):
    """Chạy lệnh, ghi toàn bộ output vào logdir/name.log, chỉ in các dòng chứa echo_keys. Trả về returncode."""
    os.makedirs(logdir, exist_ok=True)
    shown = cmd if isinstance(cmd, str) else " ".join(cmd)
    if dry:
        print(f"  $ (cwd={cwd}) {shown}")
        return 0
    path = f"{logdir}/{name}.log"
    t0 = time.time()
    log(f"{name}: bắt đầu  (log: {path})")
    with open(path, "w", encoding="utf-8") as lf:
        p = subprocess.Popen(cmd, cwd=cwd, env=make_env(P), shell=isinstance(cmd, str), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1, encoding="utf-8", errors="replace")
        for line in p.stdout:
            lf.write(line)
            if echo_keys and any(k in line for k in echo_keys):
                print("   ", line.rstrip()[:300], flush=True)
        p.wait()
    log(f"{name}: returncode={p.returncode}  {(time.time() - t0) / 60:.1f} phút")
    if p.returncode and check:
        print(open(path, encoding="utf-8", errors="replace").read()[-3000:])
        raise RuntimeError(f"[{name}] lỗi, xem {path}")
    return p.returncode


def need_gpu():
    try:
        import torch
    except ImportError:
        sys.exit("Chưa cài PyTorch. Hãy cài torch bản có CUDA trước (xem README), rồi chạy lại `setup`.")
    if not torch.cuda.is_available():
        sys.exit("Không thấy GPU. Colab: Runtime -> Change runtime type -> T4 GPU. Máy riêng: cần GPU NVIDIA >= 15 GB.")
    log("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)


# ----------------------------------------------------------------------------------------------------------
# setup: cài thư viện, clone repo, vá lỗi, ghi engine
# ----------------------------------------------------------------------------------------------------------
STUB_SRC = {
    "decord": "class VideoReader: pass\nclass _B:\n    def set_bridge(self, *a, **k): pass\nbridge=_B()\n",
    "spacy": "",
}

# Vá --neg_source (biến thể Neg-Dr): thêm lựa chọn lấy cặp "không liên quan" từ Dr thay vì xoay caption trong Df.
NEG_SOURCE_EDITS = [
    (
        '    p.add_argument("--md_space", default="fusion", choices=["fusion", "itc", "both"])\n',
        '    p.add_argument("--md_space", default="fusion", choices=["fusion", "itc", "both"])\n'
        '    p.add_argument("--neg_source", default="df", choices=["df", "dr"],\n'
        '                   help="Nguồn cặp không liên quan cho MD: df = xoay caption trong batch Df (repo); '
        'dr = cặp (I_p, T_q) lấy từ batch Dr (Eq. 2 của paper)")\n',
    ),
    (
        "def multidelete_loss_df(model, model_ori, s_df, args, device):",
        "def multidelete_loss_df(model, model_ori, s_df, args, device, s_dr=None):",
    ),
    (
        '            out_ori_rand = get_embed(model_ori, s_df["image"], shuffled, device, need_fusion=md_fusion)\n',
        '            if getattr(args, "neg_source", "df") == "dr" and s_dr is not None:\n'
        '                Bd = len(s_dr["text_input"])\n'
        '                sh = int(torch.randint(1, Bd, (1,)))  # dịch vòng >= 1: ảnh p ghép caption q != p (cả hai thuộc Dr)\n'
        '                dperm = (torch.arange(Bd) + sh) % Bd\n'
        '                dr_shuffled = [s_dr["text_input"][i] for i in dperm]\n'
        '                out_ori_rand = get_embed(model_ori, s_dr["image"], dr_shuffled, device, need_fusion=md_fusion)\n'
        "            else:\n"
        '                out_ori_rand = get_embed(model_ori, s_df["image"], shuffled, device, need_fusion=md_fusion)\n',
    ),
    (
        "loss_df, parts = multidelete_loss_df(model, model_ori, s_df, args, device)",
        "loss_df, parts = multidelete_loss_df(model, model_ori, s_df, args, device, s_dr=s_dr)",
    ),
]


def patch_neg_source(path):
    s = open(path, encoding="utf-8").read()
    if '"--neg_source"' in s:
        return
    for old, new in NEG_SOURCE_EDITS:
        if s.count(old) != 1:
            raise RuntimeError(f"Không vá được --neg_source (đoạn mã xuất hiện {s.count(old)} lần):\n{old[:120]}")
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8", newline="\n").write(s)
    log("đã vá --neg_source vào", path)


def install_engine(P):
    """Sao run_md_retrieval.py vào repo của tác giả (engine import lavis từ repo) rồi vá --neg_source lên bản sao."""
    for fn in ("run_md_retrieval.py", "nlvr2_engine.py", "demo_engine.py"):
        if not os.path.isfile(f"{HERE}/{fn}"):
            sys.exit(f"Thiếu {HERE}/{fn}. Các file .py phải nằm cạnh multidelete.py.")
    shutil.copyfile(f"{HERE}/run_md_retrieval.py", f"{P.repo}/run_md_retrieval.py")
    patch_neg_source(f"{P.repo}/run_md_retrieval.py")


def setup(a, P):
    for d in (P.work, P.save, P.cache, P.stubs, P.ckpts, P.fl_results, P.fl_logs):
        os.makedirs(d, exist_ok=True)
    mark = f"{P.work}/.setup_done"
    if os.path.exists(mark) and not a.force_setup:
        install_engine(P)       # luôn sao lại engine để khớp với file hiện tại
        log("setup đã xong trước đó (dùng --force-setup để làm lại).")
        return
    if a.dry_run:
        log("[dry-run] setup: pip install, git clone, vá cache_root + dtype, cài engine")
        return
    log("1/4 cài thư viện Python (cần PyTorch có sẵn)")
    run_logged([sys.executable, "-m", "pip", "install", "-q"] + PIP_PACKAGES, "01_pip_install", P, P.fl_logs, check=False)
    for mod in ("decord", "spacy"):          # LAVIS import 2 gói này ở mức module nhưng retrieval/NLVR không dùng
        try:
            __import__(mod)
        except Exception:
            run_logged([sys.executable, "-m", "pip", "install", "-q", mod], f"02_pip_{mod}", P, P.fl_logs, check=False)
            try:
                __import__(mod)
            except Exception:
                log(f"{mod} không cài được -> dùng stub rỗng (không ảnh hưởng kết quả)")
                os.makedirs(f"{P.stubs}/{mod}", exist_ok=True)
                open(f"{P.stubs}/{mod}/__init__.py", "w").write(STUB_SRC[mod])
    log("2/4 clone repo của tác giả")
    if not os.path.isdir(P.repo):
        run_logged(["git", "clone", "--depth", "1", REPO_URL, P.repo], "03_git_clone", P, P.fl_logs)
    log("3/4 vá repo")
    # Vá 1: cache_root cứng '/data/jiali/lavis_cache' -> thư mục làm việc
    p_cfg = f"{P.repo}/lavis/configs/default.yaml"
    s = open(p_cfg, encoding="utf-8").read()
    s2 = re.sub(r'^(\s*)cache_root:\s*"/data/jiali/lavis_cache"', rf'\1cache_root: "{P.cache}"', s, flags=re.M)
    assert s != s2 or P.cache in s, "Không vá được cache_root, kiểm tra lavis/configs/default.yaml"
    open(p_cfg, "w", encoding="utf-8").write(s2)
    # Vá 2: lỗi lệch dtype (Half vs Float32) khi đánh giá chính thức dưới autocast fp16
    pat = re.compile(r"^([ \t]*)(score_matrix_(?:i2t|t2i))\[start \+ i, topk_idx\] = score \+ topk_sim[ \t]*$", re.M)
    import glob
    for fp in sorted(glob.glob(f"{P.repo}/lavis/models/**/*.py", recursive=True)):
        t = open(fp, encoding="utf-8").read()
        t2, n = pat.subn(r"\1\2[start + i, topk_idx] = (score + topk_sim).to(\2.dtype)", t)
        if n:
            open(fp, "w", encoding="utf-8").write(t2)
            log(f"vá dtype: {os.path.relpath(fp, P.repo)} ({n} chỗ)")
    chk = open(f"{P.repo}/lavis/models/albef_models/__init__.py", encoding="utf-8").read()
    assert ".to(score_matrix_i2t.dtype)" in chk and ".to(score_matrix_t2i.dtype)" in chk, "Vá dtype chưa áp dụng được"
    log("4/4 cài engine vào repo")
    install_engine(P)
    open(mark, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    log("setup xong.")


# ----------------------------------------------------------------------------------------------------------
# Flickr30K: dữ liệu
# ----------------------------------------------------------------------------------------------------------
def find_flickr_dir(root):
    for dp, dn, fn in os.walk(root):
        if "1000092795.jpg" in fn and sum(f.endswith(".jpg") for f in fn) > 30000:
            return dp
    return None


def generate_splits(train_json, out_dir, seeds=(42, 87, 21), max_df=5000):
    """Tập xóa D_f: với mỗi seed, chọn ngẫu nhiên tối đa 5000 ảnh train; D_f = 1000 ảnh đầu. Ghi kèm SHA-256."""
    import hashlib
    import numpy as np
    ann = json.load(open(train_json))
    train_imgs = sorted({x["image"] for x in ann})
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"dataset": "flickr30k", "total_train_images": len(train_imgs), "total_train_pairs": len(ann), "seeds": {}}
    for sd in seeds:
        n_sel = min(max_df, len(train_imgs) // 2)
        ids = np.random.RandomState(sd).choice(train_imgs, n_sel, replace=False).tolist()
        txt_full = "\n".join(ids) + "\n"
        txt_1000 = "\n".join(ids[:1000]) + "\n"
        open(os.path.join(out_dir, f"image-{sd}.txt"), "w").write(txt_full)
        manifest["seeds"][str(sd)] = {
            "file": f"Df/flickr30k/image-{sd}.txt", "num_images_total": len(ids),
            "sha256_5000": hashlib.sha256(txt_full.encode()).hexdigest(),
            "sha256_1000": hashlib.sha256(txt_1000.encode()).hexdigest(),
            "first_5_images": ids[:5],
        }
    json.dump(manifest, open(os.path.join(out_dir, "split_manifest.json"), "w"), indent=2)
    return manifest


def flickr_data(a, P):
    ann_dir = f"{P.cache}/flickr30k/annotations"
    img_link = f"{P.cache}/flickr30k/images/flickr30k-images"
    if a.dry_run:
        log("[dry-run] flickr_data: ảnh (--flickr-images hoặc Kaggle), annotation, Df split")
        return
    os.makedirs(ann_dir, exist_ok=True)
    raw = f"{P.work}/flickr_raw"
    os.makedirs(raw, exist_ok=True)
    src = find_flickr_dir(a.flickr_images) if a.flickr_images else find_flickr_dir(raw)
    if src is None:
        if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
                or os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))):
            sys.exit("Không có ảnh Flickr30K. Hãy một trong hai: (1) đặt KAGGLE_USERNAME và KAGGLE_KEY rồi chạy lại "
                     "(script tải adityajn105/flickr30k), (2) tải ảnh sẵn rồi truyền --flickr-images THƯ_MỤC "
                     "(thư mục phải có 1000092795.jpg và > 30000 ảnh .jpg).")
        kaggle = shutil.which("kaggle") or os.path.join(os.path.dirname(sys.executable), "kaggle")
        run_logged([kaggle, "datasets", "download", "-d", "adityajn105/flickr30k", "-p", raw, "--unzip"],
                   "06_kaggle_download", P, P.fl_logs)
        src = find_flickr_dir(raw)
    assert src, "Không thấy ảnh Flickr30K (cần thư mục có 1000092795.jpg và > 30000 ảnh .jpg)."
    os.makedirs(os.path.dirname(img_link), exist_ok=True)
    if os.path.lexists(img_link):
        os.remove(img_link) if os.path.islink(img_link) else shutil.rmtree(img_link)
    os.symlink(src, img_link)
    log("ảnh:", src, "->", img_link)
    base = "https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_{}.json"
    for sp in ("train", "val", "test"):
        f = f"{ann_dir}/{sp}.json"
        if not os.path.exists(f):
            urllib.request.urlretrieve(base.format(sp), f)
        log(sp, len(json.load(open(f))), "cặp")
    m = generate_splits(f"{ann_dir}/train.json", f"{P.repo}/Df/flickr30k")
    shutil.copytree(f"{P.repo}/Df", f"{P.fl_results}/Df", dirs_exist_ok=True)
    log("train:", m["total_train_images"], "ảnh,", m["total_train_pairs"], "cặp | Df đã lưu vào", f"{P.fl_results}/Df")


# ----------------------------------------------------------------------------------------------------------
# Flickr30K: thí nghiệm
# ----------------------------------------------------------------------------------------------------------
def flickr_experiments(a):
    """nhãn -> (tag, method, extra, mô tả). Hậu tố profile để profile khác nhau không bỏ qua nhầm nhau."""
    pf = FLICKR_PROFILES[a.profile]
    sfx = "" if a.profile == "fast" else f"_{a.profile}"
    base = f"{BACKBONE}_df{pf['df']}_s{a.seed}"
    M = "multidelete"
    return {
        "original": (f"original_{BACKBONE}{sfx}", "original", [], "Original (ALBEF fine-tuned của LAVIS), chỉ đánh giá"),
        "main": (f"multidelete_{base}{sfx}", M, [], "MultiDelete gốc (MD trên fusion CLS)"),
        "itc": (f"multidelete_{base}_mditc{sfx}", M, ["--md_space", "itc"], "MD-ITC: MD trên ma trận ITC (sát code repo)"),
        "negdr": (f"multidelete_{base}_negdr{sfx}", M, ["--neg_source", "dr"], "Neg-Dr: cặp âm lấy từ Dr"),
        "noMD": (f"multidelete-noMD_{base}{sfx}", M, ["--alpha", "0"], "Ablation: bỏ L_MD"),
        "noMKR": (f"multidelete-noMKR_{base}{sfx}", M, ["--beta", "0"], "Ablation: bỏ L_MKR"),
        "noUKR": (f"multidelete-noUKR_{base}{sfx}", M, ["--gamma", "0"], "Ablation: bỏ L_UKR"),
        "ft": (f"ft_{base}{sfx}", "ft", ["--lr", "5e-5"], "Baseline FineTune trên Dr (1 epoch, lr=5e-5)"),
        "neggrad": (f"neggrad_{base}{sfx}", "neggrad", ["--lr", "2e-6", "--max_steps", "25", "--grad_clip", "1.0"],
                    "Baseline NegGrad có kiểm soát (lr=2e-6, 25 bước, clip 1.0)"),
        "dtd": (f"dtd_{base}{sfx}", "dtd", [], "Baseline Descent-to-Delete"),
        # tùy chọn (không chạy mặc định, gọi bằng --only): có trong notebook Kaggle
        "mdF": (f"multidelete-F_{base}{sfx}", M, ["--fusion_only"], "MultiDelete-F: chỉ cập nhật cross-attention + ITM head"),
        "neggradraw": (f"neggrad_raw_{base}{sfx}", "neggrad", ["--lr", "1e-5"],
                       "NegGrad nguyên bản (1 epoch, lr 1e-5, không clip) để minh chứng sập mô hình"),
    }


def build_cmd(a, P, tag, method, extra, bs, save_ckpt=False, ckpt_dir=None, out_dir=None, skip_official=False):
    pf = FLICKR_PROFILES[a.profile]
    cmd = [sys.executable, "run_md_retrieval.py", "--cfg-path", CFG, "--method", method, "--backbone", BACKBONE,
           "--df_size", str(pf["df"]), "--seed", str(a.seed), "--epochs", str(pf["epochs"]), "--bs", str(bs),
           "--max_steps", str(pf["max_steps"]), "--dr_eval_mult", "5", "--amp",
           "--out_dir", out_dir or P.fl_results, "--ckpt_dir", ckpt_dir or P.ckpts, "--tag", tag, "--num_workers", "2"]
    if a.profile != "fast":                  # notebook Kaggle: EVAL_EVERY_EPOCH = (PROFILE not in ("full", "fast"))
        cmd.append("--eval_every_epoch")
    if method == "multidelete":
        cmd += ["--md_space", "fusion"]      # mặc định của notebook Kaggle; extra (vd itc) ghi đè phía sau
    cmd += list(extra)
    if save_ckpt:
        cmd.append("--save_ckpt")
    if skip_official:
        cmd.append("--skip_official")
    return cmd + ["--options", f"run.k_test={pf['k_test']}", f"run.output_dir={P.work}/output/tmp",
                  "run.batch_size_eval=64", "run.num_workers=2", "model.load_finetuned=True"]


def run_flickr_experiment(a, P, name, spec):
    tag, method, extra, desc = spec
    out_json = f"{P.fl_results}/{tag}.json"
    if os.path.exists(out_json) and not a.force:
        log(f"[{name}] đã có {os.path.basename(out_json)} -> bỏ qua (--force để chạy lại)")
        return True
    log(f"===== [{name}] {desc} =====")
    keys = ("BEFORE", "AFTER", "EPOCH", "Saved", "Error", "Traceback", "OutOfMemory")
    bs = FLICKR_PROFILES[a.profile]["bs"]
    ck = bool(a.save_ckpt and name in ("main", "ft"))     # notebook Kaggle chỉ lưu checkpoint của MultiDelete và FineTune
    rc = run_logged(build_cmd(a, P, tag, method, extra, bs, ck), f"extra_{name}", P, P.fl_logs, cwd=P.repo, check=False,
                    echo_keys=keys, dry=a.dry_run)
    lp = f"{P.fl_logs}/extra_{name}.log"
    if rc and bs > 4 and os.path.exists(lp) and "OutOfMemoryError" in open(lp, encoding="utf-8").read()[-30000:]:
        log(f"OOM với bs={bs} -> chạy lại với bs={bs // 2} (khác cấu hình gốc, ghi vào báo cáo)")
        rc = run_logged(build_cmd(a, P, tag, method, extra, bs // 2, ck), f"extra_{name}_bs{bs // 2}", P, P.fl_logs,
                        cwd=P.repo, check=False, echo_keys=keys)
    if rc:
        log(f"!! [{name}] lỗi, xem log trong {P.fl_logs}")
    return rc == 0


LABELS = {"original": "Original", "main": "MultiDelete", "itc": "MD-ITC", "negdr": "Neg-Dr", "noMD": "-MD",
          "noMKR": "-MKR", "noUKR": "-UKR", "ft": "FineTune", "neggrad": "NegGrad", "dtd": "DtD",
          "mdF": "MultiDelete-F", "neggradraw": "NegGrad (raw)"}


def flickr_summary(a, P):
    exps = flickr_experiments(a)
    rows = []
    for name in FLICKR_ORDER + FLICKR_OPTIONAL:
        f = f"{P.fl_results}/{exps[name][0]}.json"
        if not os.path.exists(f):
            continue
        r = json.load(open(f, encoding="utf-8"))
        o = r.get("after_official") or r.get("before_official") or {}
        q = r.get("after_quick") or r.get("before_quick") or {}
        rows.append(dict(method=LABELS[name], TR=o.get("txt_r_mean"), IR=o.get("img_r_mean"),
                         AUC_ITC=q.get("df_auc"), AUC_ITM=q.get("df_auc_itm"),
                         AUC_ITC_repo=None if q.get("df_auc") is None else 100 - q["df_auc"],   # hướng của code repo (Df=0)
                         df_pair_sim=q.get("mean_pair_sim_df"), dr_pair_sim=q.get("mean_pair_sim_dr"),
                         df_itm=q.get("mean_itm_margin_df"), dr_itm=q.get("mean_itm_margin_dr"),
                         unlearn_min=round(r.get("time_unlearn_s", 0) / 60, 1)))
    if not rows:
        log("Chưa có kết quả Flickr30K để tổng hợp.")
        return
    cols = list(rows[0].keys())
    print("\n=== Flickr30K (ALBEF), test = TR/IR R-mean; AUC = Df-AUC (Df là lớp dương, ~50 = không phân biệt được) ===")
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join("-" if r[c] is None else (f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])) for c in cols))
    path = f"{P.fl_results}/summary_{a.profile}.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    log("đã lưu", path)
    print("Lưu ý: Df và Dr đều là dữ liệu train, nên Original đã có AUC ~ 50. AUC ~ 50 KHÔNG có nghĩa là mô hình quên tốt; "
          "phải xem kèm TR/IR (xem báo cáo, mục Phân tích lỗi).")


def flickr30k(a, P):
    a.profile = a.raw_profile if a.raw_profile in FLICKR_PROFILES else "fast"
    setup(a, P)
    if not a.dry_run:
        need_gpu()
    flickr_data(a, P)
    exps = flickr_experiments(a)
    names = [n.strip() for n in a.only.split(",") if n.strip()] if a.only else list(FLICKR_ORDER)
    bad = [n for n in names if n not in exps]
    if bad:
        sys.exit(f"Tên không hợp lệ: {bad}. Chọn trong {list(exps)}")
    names = [n for n in FLICKR_ORDER + FLICKR_OPTIONAL if n in names]    # luôn chạy original trước để cache kết quả BEFORE
    t0 = time.time()
    status = {n: run_flickr_experiment(a, P, n, exps[n]) for n in names}
    log(f"Flickr30K xong sau {(time.time() - t0) / 60:.1f} phút. Trạng thái: {status}")
    flickr_summary(a, P)


# ----------------------------------------------------------------------------------------------------------
# NLVR2 và demo: gọi engine
# ----------------------------------------------------------------------------------------------------------
def nlvr2(a, P):
    setup(a, P)
    prof = a.raw_profile if a.raw_profile in NLVR2_PROFILES else "paper"   # mặc định "paper" = cấu hình của báo cáo
    env_over = {"MD_PROFILE": prof, "MD_SEED": str(a.seed), "MD_WORK": P.work, "MD_SAVE": P.nl_root,
                "MD_BASELINES": "0" if a.no_baselines else "1", "MD_ABLATION": "1" if a.ablation else "0",
                "MD_FUSION_ONLY": "1" if a.fusion_only else "0"}
    if a.dry_run:
        log("[dry-run] nlvr2:", env_over)
        print(f"  $ {sys.executable} {HERE}/nlvr2_engine.py")
        return
    need_gpu()
    os.makedirs(P.nl_root, exist_ok=True)
    env = make_env(P)
    env.update(env_over)
    log("NLVR2: engine in log trực tiếp; mỗi lần chạy lưu JSON vào", f"{P.nl_root}/results")
    rc = subprocess.run([sys.executable, f"{HERE}/nlvr2_engine.py"], env=env, cwd=P.work).returncode
    if rc:
        sys.exit(f"nlvr2_engine kết thúc với returncode={rc}")
    log("NLVR2 xong. Bảng: ", f"{P.nl_root}/results/summary.csv")


def demo(a, P):
    a.profile = a.raw_profile if a.raw_profile in FLICKR_PROFILES else "fast"
    setup(a, P)
    ckpt_tag = f"demo_multidelete_{BACKBONE}_df{FLICKR_PROFILES[a.profile]['df']}_s{a.seed}"
    ckpt = f"{P.ckpts}/{ckpt_tag}_ckpt.pth"
    if a.dry_run:
        print(" ", " ".join(build_cmd(a, P, ckpt_tag, "multidelete", [], 16, True, P.ckpts, f"{P.work}/demo_results", True)))
        print(f"  $ {sys.executable} {HERE}/demo_engine.py")
        return
    need_gpu()
    flickr_data(a, P)
    if not os.path.exists(ckpt):
        log("demo 1/2: huấn luyện MultiDelete và lưu checkpoint (~40 phút trên T4)")
        bs = FLICKR_PROFILES[a.profile]["bs"]
        run_logged(build_cmd(a, P, ckpt_tag, "multidelete", [], bs, True, P.ckpts, f"{P.work}/demo_results", True),
                   "demo_train", P, P.fl_logs, cwd=P.repo, echo_keys=("EPOCH", "AFTER", "Saved", "Error", "Traceback"))
    assert os.path.exists(ckpt), "chưa có checkpoint, xem logs/demo_train.log"
    os.makedirs(P.demo_out, exist_ok=True)
    env = make_env(P)
    env.update({"MD_WORK": P.work, "MD_DEMO_CKPT": ckpt, "MD_DEMO_OUT": P.demo_out, "MD_SEED": str(a.seed),
                "MD_DF": str(FLICKR_PROFILES[a.profile]["df"])})
    log("demo 2/2: so sánh mô hình gốc và mô hình đã quên")
    with open(f"{P.demo_out}/demo.txt", "w", encoding="utf-8") as fh:
        p = subprocess.Popen([sys.executable, f"{HERE}/demo_engine.py"], env=env, cwd=P.work, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            print(line, end="")
            fh.write(line)
        p.wait()
    if p.returncode:
        sys.exit(f"demo_engine returncode={p.returncode} (xem {P.demo_out}/demo.txt)")
    log("đã lưu ảnh và văn bản demo tại", P.demo_out)


# ----------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="MultiDelete trên Flickr30K và NLVR2 (xem README.md).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["setup", "flickr30k", "nlvr2", "demo", "all"])
    ap.add_argument("--profile", default="", help="smoke (thử luồng) | fast | paper | full (chỉ NLVR2). Mặc định như báo cáo: Flickr30K=fast, NLVR2=paper")
    ap.add_argument("--work", default="./work", help="thư mục làm việc: repo, dữ liệu, cache, checkpoint (nhiều GB)")
    ap.add_argument("--save", default="./results", help="thư mục kết quả: JSON, log, bảng, hình. Trên Colab nên trỏ vào Drive")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default="", help=f"Flickr30K: chỉ chạy các mục này, phân tách bằng dấu phẩy, trong {FLICKR_ORDER + FLICKR_OPTIONAL}")
    ap.add_argument("--flickr-images", default="", help="thư mục ảnh Flickr30K có sẵn (bỏ qua bước tải từ Kaggle)")
    ap.add_argument("--no-baselines", action="store_true", help="NLVR2: bỏ FineTune và NegGrad")
    ap.add_argument("--ablation", action="store_true", help="NLVR2: chạy thêm -MD / -MKR / -UKR")
    ap.add_argument("--fusion-only", action="store_true", help="NLVR2: chạy thêm MultiDelete-F")
    ap.add_argument("--save-ckpt", action="store_true", help="Flickr30K: lưu checkpoint đầy đủ (~420 MB) của MultiDelete và FineTune vào <work>/checkpoints")
    ap.add_argument("--force", action="store_true", help="chạy lại cả những lần đã có kết quả (Flickr30K)")
    ap.add_argument("--force-setup", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="chỉ in các lệnh sẽ chạy, không cài, không tải, không huấn luyện")
    ap.add_argument("--summary-only", action="store_true", help="chỉ in bảng tổng hợp từ kết quả đã có")
    a = ap.parse_args()
    a.raw_profile = a.profile
    if a.profile and a.profile not in FLICKR_PROFILES and a.profile not in NLVR2_PROFILES:
        sys.exit(f"profile '{a.profile}' không hợp lệ. Chọn: smoke | fast | paper | full")
    if a.profile == "full" and a.command in ("flickr30k", "demo"):
        sys.exit("profile 'full' chỉ có cho NLVR2")
    P = Paths(a)
    if a.summary_only:
        if a.command != "flickr30k":
            sys.exit("--summary-only chỉ dùng với `flickr30k`")
        a.profile = a.raw_profile if a.raw_profile in FLICKR_PROFILES else "fast"
        flickr_summary(a, P)
        return
    if a.command == "setup":
        setup(a, P)
    elif a.command == "flickr30k":
        flickr30k(a, P)
    elif a.command == "nlvr2":
        nlvr2(a, P)
    elif a.command == "demo":
        demo(a, P)
    elif a.command == "all":
        flickr30k(a, P)
        nlvr2(a, P)


if __name__ == "__main__":
    main()
