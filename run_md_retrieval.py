"""
run_md_retrieval.py
===================
Thực thi MultiDelete và các Baselines (Original, FineTune, NegGrad, DTD) cho tác vụ Image–Text Retrieval
trên bộ dữ liệu Flickr30K với kiến trúc ALBEF / BLIP.
"""
import argparse
import copy
import datetime
import glob
import hashlib
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.utils import now
from lavis.datasets.builders import *  # noqa: F401,F403
from lavis.models import *             # noqa: F401,F403
from lavis.processors import *         # noqa: F401,F403
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *              # noqa: F401,F403


# ----------------------------------------------------------------------------------------
# Các hàm tính toán độ đo (Evaluation Metrics)
# ----------------------------------------------------------------------------------------
def recall_from_sims(sims, img2txt, txt2img, ks=(1, 5, 10)):
    """Tính Recall@K (%) cho TR (Image -> Text) và IR (Text -> Image) từ ma trận tương đồng [n_img, n_txt]."""
    n_img, n_txt = sims.shape
    tr_rank = np.zeros(n_img)
    for i in range(n_img):
        gt = img2txt[i]
        gt_scores = sims[i, gt]
        tr_rank[i] = (sims[i] > gt_scores.max()).sum()
    ir_rank = np.zeros(n_txt)
    for j in range(n_txt):
        ir_rank[j] = (sims[:, j] > sims[txt2img[j], j]).sum()
    out = {}
    for k in ks:
        out[f"txt_r{k}"] = 100.0 * (tr_rank < k).mean()
        out[f"img_r{k}"] = 100.0 * (ir_rank < k).mean()
    out["txt_r_mean"] = float(np.mean([out[f"txt_r{k}"] for k in ks]))
    out["img_r_mean"] = float(np.mean([out[f"img_r{k}"] for k in ks]))
    out["r_mean"] = (out["txt_r_mean"] + out["img_r_mean"]) / 2
    return out


def paired_scores(img_emb, txt_emb, img2txt):
    """Điểm tương đồng cosine trung bình của mỗi ảnh với 5 câu mô tả đúng của chính ảnh đó."""
    return np.array([(txt_emb[img2txt[i]] @ img_emb[i]).mean() for i in range(img_emb.shape[0])])


def df_dr_auc(df_scores, dr_scores, n_chunks=5):
    """ROC-AUC (x100) phân biệt tập xóa Df (nhãn 1) và tập giữ lại Dr (nhãn 0), chia thành n_chunks khối."""
    from sklearn.metrics import roc_auc_score

    n = len(df_scores)
    aucs = []
    for c in range(n_chunks):
        chunk = dr_scores[c * n : (c + 1) * n]
        if len(chunk) == 0:
            break
        y = np.array([1] * n + [0] * len(chunk))
        s = np.concatenate([df_scores, chunk])
        aucs.append(100.0 * roc_auc_score(y, s))
    return float(np.mean(aucs)), aucs


# ----------------------------------------------------------------------------------------
# Chuẩn bị và phân tách dữ liệu Df / Dr
# ----------------------------------------------------------------------------------------
def read_df_ids(path, df_size):
    with open(path) as f:
        ids = [l.strip() for l in f if l.strip()]
    return ids[:df_size]


def make_eval_ds(dtest, annotation):
    """Khởi tạo RetrievalEvalDataset từ danh sách cặp {image, caption} của tập train."""
    ds = copy.deepcopy(dtest)
    by_img = {}
    for a in annotation:
        by_img.setdefault(a["image"], []).append(a["caption"])
    grouped = [{"image": k, "caption": v} for k, v in sorted(by_img.items())]
    ds.annotation = grouped
    ds.text, ds.image, ds.txt2img, ds.img2txt = [], [], {}, {}
    tid = 0
    for iid, ann in enumerate(grouped):
        ds.image.append(ann["image"])
        ds.img2txt[iid] = []
        for cap in ann["caption"]:
            ds.text.append(ds.text_processor(cap))
            ds.img2txt[iid].append(tid)
            ds.txt2img[tid] = iid
            tid += 1
    return ds


def build_splits(dtrain, dtest, df_ids, seed, dr_eval_mult=5):
    df_set = set(df_ids)
    df_train = copy.deepcopy(dtrain)
    df_train.annotation = [a for a in dtrain.annotation if a["image"] in df_set]
    dr_train = copy.deepcopy(dtrain)
    dr_train.annotation = [a for a in dtrain.annotation if a["image"] not in df_set]

    rng = np.random.RandomState(seed)
    dr_images = sorted({a["image"] for a in dr_train.annotation})
    n_eval = min(len(dr_images), dr_eval_mult * len(df_ids))
    dr_eval_imgs = set(rng.choice(dr_images, n_eval, replace=False).tolist())

    df_eval = make_eval_ds(dtest, [a for a in dtrain.annotation if a["image"] in df_set])
    dr_eval = make_eval_ds(dtest, [a for a in dr_train.annotation if a["image"] in dr_eval_imgs])
    return df_train, dr_train, df_eval, dr_eval


# ----------------------------------------------------------------------------------------
# Đánh giá mô hình (Quick Eval & Official LAVIS Test)
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def embed_pass(model, ds, device, amp, bs=64, nw=2):
    """Trích xuất ITC embedding cho toàn bộ ảnh và văn bản của một eval-dataset."""
    model.eval()
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=ds.collater)
    img_embs = []
    for batch in loader:
        img = batch["image"].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            feat = model.visual_encoder.forward_features(img)
            emb = F.normalize(model.vision_proj(feat[:, 0, :]), dim=-1)
        img_embs.append(emb.float().cpu())
    txt_embs = []
    texts = ds.text
    for i in range(0, len(texts), 256):
        t = model.tokenizer(
            texts[i : i + 256], padding="max_length", truncation=True, max_length=35, return_tensors="pt"
        ).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model.text_encoder.forward_text(t)
            emb = F.normalize(model.text_proj(out.last_hidden_state[:, 0, :]), dim=-1)
        txt_embs.append(emb.float().cpu())
    return torch.cat(img_embs).numpy(), torch.cat(txt_embs).numpy()


@torch.no_grad()
def itm_pair_scores(model, ds, device, amp, bs=32, nw=2, max_len=35):
    """Tính ITM margin (logit_match - logit_nomatch) trung bình trên các câu mô tả đúng của mỗi ảnh."""
    model.eval()
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=ds.collater)
    out, ptr = [], 0
    for batch in loader:
        img = batch["image"].to(device)
        B = img.size(0)
        margins = []
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            img_emb = model.visual_encoder.forward_features(img)
            img_att = torch.ones(img_emb.size()[:-1], dtype=torch.long, device=device)
            n_caps = min(len(ds.img2txt[ptr + b]) for b in range(B))
            for k in range(n_caps):
                caps = [ds.text[ds.img2txt[ptr + b][k]] for b in range(B)]
                t = model.tokenizer(
                    caps, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt"
                ).to(device)
                t_emb = model.text_encoder.forward_text(t).last_hidden_state
                cls = model.text_encoder(
                    encoder_embeds=t_emb,
                    attention_mask=t.attention_mask,
                    encoder_hidden_states=img_emb,
                    encoder_attention_mask=img_att,
                    return_dict=True,
                    mode="fusion",
                ).last_hidden_state[:, 0, :]
                lg = model.itm_head(cls).float()
                margins.append((lg[:, 1] - lg[:, 0]).cpu())
        out.append(torch.stack(margins, 0).mean(0))
        ptr += B
    return torch.cat(out).numpy()


def quick_eval(model, df_eval, dr_eval, test_ds, device, amp, nw, n_chunks=5, itm=False):
    """Đánh giá nhanh: AUC Df-vs-Dr (ITC cosine & ITM margin), cùng Recall chỉ dùng ITC trên Df/Test."""
    res = {}
    emb = {}
    for name, ds in [("df", df_eval), ("dr", dr_eval), ("test", test_ds)]:
        emb[name] = embed_pass(model, ds, device, amp, nw=nw)
    for name, ds in [("df", df_eval), ("test", test_ds)]:
        ie, te = emb[name]
        r = recall_from_sims(ie @ te.T, ds.img2txt, ds.txt2img)
        res[f"{name}_itc_txt_r_mean"] = r["txt_r_mean"]
        res[f"{name}_itc_img_r_mean"] = r["img_r_mean"]
        res[f"{name}_itc_r1_tr"] = r["txt_r1"]
        res[f"{name}_itc_r1_ir"] = r["img_r1"]
    df_sc = paired_scores(*emb["df"], df_eval.img2txt)
    dr_sc = paired_scores(*emb["dr"], dr_eval.img2txt)
    auc, aucs = df_dr_auc(df_sc, dr_sc, n_chunks=n_chunks)
    res["df_auc"] = auc
    res["df_auc_chunks"] = aucs
    res["mean_pair_sim_df"] = float(df_sc.mean())
    res["mean_pair_sim_dr"] = float(dr_sc.mean())
    if itm:
        t0 = time.time()
        df_itm = itm_pair_scores(model, df_eval, device, amp, nw=nw)
        dr_itm = itm_pair_scores(model, dr_eval, device, amp, nw=nw)
        auc_i, aucs_i = df_dr_auc(df_itm, dr_itm, n_chunks=n_chunks)
        res["df_auc_itm"] = auc_i
        res["df_auc_itm_chunks"] = aucs_i
        res["mean_itm_margin_df"] = float(df_itm.mean())
        res["mean_itm_margin_dr"] = float(dr_itm.mean())
        res["time_itm_eval_s"] = time.time() - t0
    return res


@torch.no_grad()
def official_test(runner, model, amp=False, split="test"):
    """Đánh giá Recall chính thức của LAVIS (ITC top-k_test + ITM re-rank)."""
    model.eval()
    try:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            return runner.task.evaluation(model, runner.dataloaders[split], split_name=split)
    except RuntimeError as e:
        if amp and "dtypes match" in str(e):
            logging.warning("official_test: lỗi dtype dưới fp16 autocast (%s) -> chạy lại bằng fp32", e)
            return runner.task.evaluation(model, runner.dataloaders[split], split_name=split)
        raise


# ----------------------------------------------------------------------------------------
# Các thành phần hàm mất mát của MultiDelete (Eq. 2, Eq. 4, Eq. 6)
# ----------------------------------------------------------------------------------------
def get_embed(model, image, caption, device, need_fusion=True):
    """Biểu diễn đơn phương thức (ITC) + biểu diễn đa phương thức (CLS của bộ Fusion Encoder)."""
    image_embeds = model.visual_encoder.forward_features(image)
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
    image_feat = F.normalize(model.vision_proj(image_embeds[:, 0, :]), dim=-1)

    text = model.tokenizer(
        caption, padding="max_length", truncation=True, max_length=model.max_txt_len, return_tensors="pt"
    ).to(device)
    text_embeds = model.text_encoder.forward_text(text).last_hidden_state
    text_feat = F.normalize(model.text_proj(text_embeds[:, 0, :]), dim=-1)

    fusion = None
    if need_fusion:
        fusion = model.text_encoder(
            encoder_embeds=text_embeds,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode="fusion",
        ).last_hidden_state[:, 0, :]
    return {
        "image_feat": image_feat,
        "text_feat": text_feat,
        "fusion": fusion,
        "sim_i2t": image_feat @ text_feat.t(),
        "sim_t2i": text_feat @ image_feat.t(),
    }


def drop_momentum(m):
    """Loại bỏ các momentum encoders không dùng khi unlearn để giải phóng ~1.8GB VRAM mỗi bản sao."""
    for n in ("visual_encoder_m", "text_encoder_m", "vision_proj_m", "text_proj_m"):
        if hasattr(m, n):
            delattr(m, n)
    if hasattr(m, "model_pairs"):
        m.model_pairs = []
    return m


def multidelete_loss_df(model, model_ori, s_df, args, device):
    """Các số hạng trên Df: alpha * L_MD (Eq. 2) + gamma * L_UKR (Eq. 6)."""
    mse = nn.MSELoss()
    zero = torch.zeros((), device=device)
    md_fusion = args.alpha > 0 and args.md_space in ("fusion", "both")
    md_itc = args.alpha > 0 and args.md_space in ("itc", "both")
    out_df = get_embed(model, s_df["image"], s_df["text_input"], device, need_fusion=md_fusion)
    l_md = zero
    if args.alpha > 0:
        with torch.no_grad():
            B = len(s_df["text_input"])
            shift = int(torch.randint(1, B, (1,)))  # Dịch vòng >= 1 để không cặp nào ghép trùng lại chính nó
            perm = (torch.arange(B) + shift) % B
            shuffled = [s_df["text_input"][i] for i in perm]
            out_ori_rand = get_embed(model_ori, s_df["image"], shuffled, device, need_fusion=md_fusion)
        if md_fusion:
            l_md = l_md + mse(out_df["fusion"].float(), out_ori_rand["fusion"].float())
        if md_itc:
            l_md = (
                l_md
                + mse(out_df["sim_i2t"].float(), out_ori_rand["sim_i2t"].float())
                + mse(out_df["sim_t2i"].float(), out_ori_rand["sim_t2i"].float())
            )
    l_ukr = zero
    if args.gamma > 0:
        with torch.no_grad():
            out_ori_df = get_embed(model_ori, s_df["image"], s_df["text_input"], device, need_fusion=False)
        l_ukr = mse(out_df["image_feat"].float(), out_ori_df["image_feat"].float()) + mse(
            out_df["text_feat"].float(), out_ori_df["text_feat"].float()
        )
    return args.alpha * l_md + args.gamma * l_ukr, {"md": float(l_md), "ukr": float(l_ukr)}


def multidelete_loss_dr(model, model_ori, s_dr, args, device):
    """Số hạng trên Dr: beta * L_MKR (Eq. 4)."""
    zero = torch.zeros((), device=device)
    if args.beta <= 0:
        return zero, {"mkr": 0.0}
    out_dr = get_embed(model, s_dr["image"], s_dr["text_input"], device)
    with torch.no_grad():
        out_ori_dr = get_embed(model_ori, s_dr["image"], s_dr["text_input"], device)
    l_mkr = nn.MSELoss()(out_dr["fusion"].float(), out_ori_dr["fusion"].float())
    return args.beta * l_mkr, {"mkr": float(l_mkr)}


def save_unlearned_checkpoints(model, args, tag):
    """Lưu checkpoint mô hình sau khi unlearn:
    1. Bản đầy đủ (fp16 state_dict) tại args.ckpt_dir/<tag>_ckpt.pth (~420 MB).
    2. Bản trọng số đầu dự đoán (vision_proj, text_proj, itm_head) tại args.out_dir/checkpoints/<tag>_heads.pth (~2 MB)
       để đóng gói trực tiếp vào bangiao_multidelete.zip.
    """
    os.makedirs(args.ckpt_dir, exist_ok=True)
    full_ckpt_path = os.path.join(args.ckpt_dir, f"{tag}_ckpt.pth")
    state_fp16 = {
        k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu())
        for k, v in model.state_dict().items()
    }
    torch.save({"model": state_fp16, "args": vars(args), "tag": tag}, full_ckpt_path)

    head_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(head_dir, exist_ok=True)
    head_ckpt_path = os.path.join(head_dir, f"{tag}_heads.pth")
    head_keys = ("vision_proj", "text_proj", "itm_head")
    state_heads = {
        k: v for k, v in state_fp16.items() if any(k.startswith(hk) for hk in head_keys)
    }
    torch.save({"model_heads": state_heads, "args": vars(args), "tag": tag}, head_ckpt_path)

    full_mb = round(os.path.getsize(full_ckpt_path) / (1024 * 1024), 2)
    head_mb = round(os.path.getsize(head_ckpt_path) / (1024 * 1024), 2)
    logging.info("Saved full checkpoint: %s (%.2f MB) | head checkpoint: %s (%.2f MB)", full_ckpt_path, full_mb, head_ckpt_path, head_mb)
    return {"full_ckpt_path": full_ckpt_path, "full_ckpt_mb": full_mb, "head_ckpt_path": head_ckpt_path, "head_ckpt_mb": head_mb}


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg-path", required=True)
    p.add_argument("--options", nargs="+", default=None)
    p.add_argument("--method", default="multidelete", choices=["original", "multidelete", "ft", "neggrad", "dtd"])
    p.add_argument("--backbone", default="albef")
    p.add_argument("--df_size", type=int, default=1000)
    p.add_argument("--df_dir", default="Df/flickr30k")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3, help="Số epoch trên Df (MultiDelete); ft/neggrad chạy 1 epoch")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--md_space", default="fusion", choices=["fusion", "itc", "both"])
    p.add_argument("--fusion_only", action="store_true")
    p.add_argument("--grad_clip", type=float, default=0.0, help="Giới hạn chuẩn gradient (clip_grad_norm_) để chống sập NegGrad")
    p.add_argument("--save_ckpt", action="store_true", help="Lưu checkpoint (.pth) sau khi unlearn")
    p.add_argument("--ckpt_dir", default="/content/checkpoints")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--ori_ckpt", default="")
    p.add_argument("--dr_eval_mult", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--skip_official", action="store_true")
    p.add_argument("--eval_every_epoch", action="store_true")
    p.add_argument("--unlearn_eval_mode", action="store_true")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--tag", default="")
    return p.parse_args()


def main():
    t_start = time.time()
    args = parse_args()
    job_id = now()
    cfg = Config(args)
    cfg.run_cfg.seed = args.seed
    init_distributed_mode(cfg.run_cfg)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and device.type == "cuda"

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or f"{args.method}_{args.backbone}_df{args.df_size}_s{args.seed}"
    log_path = os.path.join(args.out_dir, f"{tag}.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    logging.info("ARGS: %s", vars(args))

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    if args.ori_ckpt:
        ck = torch.load(args.ori_ckpt, map_location="cpu")
        miss = model.load_state_dict(ck["model"], strict=False)
        logging.info("Nạp Original từ %s | missing=%d unexpected=%d", args.ori_ckpt, len(miss.missing_keys), len(miss.unexpected_keys))
        del ck
    ds_name = list(cfg.datasets_cfg.keys())[0]
    dtrain, dtest = datasets[ds_name]["train"], datasets[ds_name]["test"]
    runner = RunnerBase(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
    model = runner.model if runner.model is not None else model.to(device)

    df_file = os.path.join(args.df_dir, f"image-{args.seed}.txt")
    df_ids = read_df_ids(df_file, args.df_size)
    n_total_imgs = len({a["image"] for a in dtrain.annotation})
    n_in_train = len(set(df_ids) & {a["image"] for a in dtrain.annotation})
    assert n_in_train == len(df_ids), f"Df có {len(df_ids)} ảnh nhưng chỉ {n_in_train} ảnh nằm trong train.json."
    df_train, dr_train, df_eval, dr_eval = build_splits(dtrain, dtest, df_ids, args.seed, args.dr_eval_mult)
    logging.info(
        "Dtrain: %d ảnh | Df: %d ảnh / %d cặp | Dr: %d ảnh / %d cặp | Df-eval: %d txt | Dr-eval: %d ảnh",
        n_total_imgs, len(df_ids), len(df_train), n_total_imgs - len(df_ids), len(dr_train), len(df_eval.text), len(dr_eval.image),
    )

    model_ori = None
    if args.method == "multidelete":
        drop_momentum(model)
        model_ori = copy.deepcopy(model).eval()
        for p_ in model_ori.parameters():
            p_.requires_grad = False

    result = {
        "tag": tag, "args": vars(args),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "n_train_images": n_total_imgs, "df_images": len(df_ids), "df_pairs": len(df_train),
        "dr_images": n_total_imgs - len(df_ids), "dr_pairs": len(dr_train),
        "torch": torch.__version__,
    }

    # ---- Đánh giá trước khi unlearn (có cache kết quả BEFORE của Original để tiết kiệm thời gian) ----
    t0 = time.time()
    ck_id = hashlib.md5((args.ori_ckpt or "lavis_finetuned").encode()).hexdigest()[:6]
    q_cache = os.path.join(args.out_dir, f"quick_before_{args.backbone}_df{args.df_size}_s{args.seed}_{ck_id}.json")
    if args.method != "original" and os.path.exists(q_cache):
        result["before_quick"] = json.load(open(q_cache))
        logging.info("BEFORE (quick) lấy từ cache %s", q_cache)
    else:
        result["before_quick"] = quick_eval(model, df_eval, dr_eval, dtest, device, amp, args.num_workers, args.dr_eval_mult, itm=True)
        json.dump(result["before_quick"], open(q_cache, "w"))
    result["time_eval_quick_s"] = time.time() - t0
    logging.info("BEFORE (quick): %s", json.dumps(result["before_quick"]))

    if not args.skip_official:
        t0 = time.time()
        cache_f = os.path.join(args.out_dir, f"official_before_{args.backbone}_k{cfg.run_cfg.k_test}_amp{int(amp)}_{ck_id}.json")
        if args.method != "original" and os.path.exists(cache_f):
            result["before_official"] = json.load(open(cache_f))
            logging.info("BEFORE (official test) lấy từ cache %s", cache_f)
        else:
            result["before_official"] = official_test(runner, model, amp)
            json.dump(result["before_official"], open(cache_f, "w"))
        result["time_eval_official_s"] = time.time() - t0
        logging.info("BEFORE (official test): %s", result["before_official"])

    torch.cuda.empty_cache()

    # ---- Vòng lặp Unlearning ----
    if args.method != "original":
        params = [(n, p_) for n, p_ in model.named_parameters() if p_.requires_grad]
        if args.fusion_only:
            params = [(n, p_) for n, p_ in params if "crossattention" in n or n.startswith("itm_head")]
            keep = {n for n, _ in params}
            for n, p_ in model.named_parameters():
                if n not in keep:
                    p_.requires_grad = False
        n_train_params = sum(p_.numel() for _, p_ in params)
        logging.info("Tham số được cập nhật: %.1fM (fusion_only=%s)", n_train_params / 1e6, args.fusion_only)
        result["trainable_params_M"] = n_train_params / 1e6

        history, t_unlearn = [], time.time()
        if args.method == "dtd":
            # Descent-to-Delete (Neel et al., 2021) theo công thức trong runner_unlearn.py của tác giả
            train_size = len(dr_train)
            wd = 0.02
            gamma_dtd = (4.0 - 2.0 * wd) / 4.0
            delta = 1.0 / (train_size * train_size)
            num = 4.0 * np.sqrt(2.0) * (1.0 + wd) * gamma_dtd
            den = (wd * train_size * (1.0 - gamma_dtd)) * (np.sqrt(np.log(1.0 / delta) + 5.0) - np.sqrt(np.log(1.0 / delta)))
            sigma = float(num / den)
            with torch.no_grad():
                for _, p_ in params:
                    p_.add_(torch.randn_like(p_) * sigma)
            history.append({"step": 1, "epoch": 0, "sigma": sigma, "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0})
            logging.info("train %s", json.dumps(history[-1]))
        else:
            opt = torch.optim.AdamW([p_ for _, p_ in params], lr=args.lr, weight_decay=0.02)
            scaler = torch.cuda.amp.GradScaler(enabled=amp)

            mk = lambda ds, bs: DataLoader(
                ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=args.num_workers, collate_fn=ds.collater, pin_memory=True
            )
            df_loader = mk(df_train, args.bs)
            dr_loader = mk(dr_train, args.bs)

            def forever(loader):
                while True:
                    for b in loader:
                        yield b

            dr_iter = forever(dr_loader)
            steps = len(df_loader) if not args.max_steps else min(args.max_steps, len(df_loader))
            n_epochs = args.epochs if args.method == "multidelete" else 1
            total = n_epochs * steps
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total, 1), eta_min=args.lr * 0.1)

            step_idx = 0
            for ep in range(n_epochs):
                model.train(not (args.unlearn_eval_mode and args.method == "multidelete"))
                for i, s_df in enumerate(df_loader):
                    if i >= steps:
                        break
                    s_df = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in s_df.items()}
                    opt.zero_grad(set_to_none=True)
                    if args.method == "multidelete":
                        s_dr = next(dr_iter)
                        s_dr = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in s_dr.items()}
                        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                            loss_df, parts = multidelete_loss_df(model, model_ori, s_df, args, device)
                        if loss_df.requires_grad:
                            scaler.scale(loss_df).backward()
                        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                            loss_dr, parts_dr = multidelete_loss_dr(model, model_ori, s_dr, args, device)
                        if loss_dr.requires_grad:
                            scaler.scale(loss_dr).backward()
                        parts.update(parts_dr)
                        parts["loss"] = loss_df.item() + loss_dr.item()
                    else:
                        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                            batch = s_df if args.method == "neggrad" else next(dr_iter)
                            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                            batch.update({"epoch": 5, "iters": i, "num_iters_per_epoch": steps})
                            l = model(batch)["loss"]
                            loss = -l if args.method == "neggrad" else l
                        parts = {"loss": loss.item()}
                        scaler.scale(loss).backward()
                    if args.grad_clip > 0:
                        scaler.unscale_(opt)
                        gnorm = torch.nn.utils.clip_grad_norm_([p_ for _, p_ in params], max_norm=args.grad_clip)
                        parts["grad_norm"] = round(float(gnorm), 4) if torch.isfinite(gnorm) else -1.0
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    step_idx += 1
                    if step_idx % 20 == 0 or step_idx == 1 or step_idx == total:
                        parts.update(
                            {
                                "step": step_idx,
                                "epoch": ep,
                                "lr": sched.get_last_lr()[0],
                                "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0,
                            }
                        )
                        history.append(parts)
                        logging.info("train %s", json.dumps(parts))
                if args.method == "multidelete" and args.eval_every_epoch:
                    q = quick_eval(model, df_eval, dr_eval, dtest, device, amp, args.num_workers, args.dr_eval_mult)
                    q["epoch"] = ep
                    torch.cuda.empty_cache()
                    result.setdefault("per_epoch_quick", []).append(q)
                    logging.info("EPOCH %d quick: %s", ep, json.dumps(q))

        result["time_unlearn_s"] = time.time() - t_unlearn
        result["train_history"] = history

        # ---- Lưu Checkpoint mô hình sau khi unlearn ----
        if args.save_ckpt:
            result["checkpoint"] = save_unlearned_checkpoints(model, args, tag)

        # ---- Đánh giá sau khi unlearn ----
        result["after_quick"] = quick_eval(
            model, df_eval, dr_eval, dtest, device, amp, args.num_workers, args.dr_eval_mult, itm=True
        )
        logging.info("AFTER (quick): %s", json.dumps(result["after_quick"]))
        if not args.skip_official:
            result["after_official"] = official_test(runner, model, amp)
            logging.info("AFTER (official test): %s", result["after_official"])

    result["time_total_s"] = time.time() - t_start
    result["time_total_str"] = str(datetime.timedelta(seconds=int(result["time_total_s"])))
    out_json = os.path.join(args.out_dir, f"{tag}.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=float)
    logging.info("Saved %s (total %s)", out_json, result["time_total_str"])


if __name__ == "__main__":
    main()