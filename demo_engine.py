# ===================== Demo  =====================
import os, sys, json, re
import matplotlib
matplotlib.use("Agg")
WORK = os.environ["MD_WORK"]
REPO = f"{WORK}/MultiDelete"
CACHE = f"{WORK}/lavis_cache"
ANN = f"{CACHE}/flickr30k/annotations"
DEMO_CKPT = os.environ["MD_DEMO_CKPT"]
OUT = os.environ["MD_DEMO_OUT"]
sys.path.insert(0, f"{WORK}/stubs")
sys.path.insert(0, REPO)
os.chdir(REPO)

import copy, random, textwrap
import numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from lavis.models import load_model_and_preprocess

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_o, vis_proc, txt_proc = load_model_and_preprocess("albef_retrieval", "flickr", is_eval=True, device=dev)
model_o.eval()
model_u = copy.deepcopy(model_o)
ck = torch.load(DEMO_CKPT, map_location="cpu")
miss = model_u.load_state_dict(ck["model"], strict=False)
del ck
model_u.to(dev).eval()
print("nạp checkpoint đã quên: missing =", len(miss.missing_keys), "| unexpected =", len(miss.unexpected_keys))

# ---- dữ liệu: ảnh D_f (từ file split), ảnh D_r ngẫu nhiên, và kho caption để truy hồi ----
ann = json.load(open(f"{ANN}/train.json"))
caps_of = {}
for a in ann:
    caps_of.setdefault(a["image"], []).append(a["caption"])
df_ids = [l.strip() for l in open(f"{REPO}/Df/flickr30k/image-" + os.environ.get("MD_SEED", "42") + ".txt") if l.strip()][:int(os.environ.get("MD_DF", "1000"))]
df_set = set(df_ids)
rng = random.Random(0)
dr_ids = sorted(set(caps_of) - df_set)
gallery_imgs = df_ids + rng.sample(dr_ids, 1000)          # kho: 1000 ảnh Df + 1000 ảnh Dr (10000 caption)
gal_txt, gal_owner = [], []
for im in gallery_imgs:
    for c in caps_of[im]:
        gal_txt.append(c); gal_owner.append(im)
print("kho truy hồi:", len(gal_txt), "caption")
IMG_ROOT = f"{CACHE}/flickr30k/images"

@torch.no_grad()
def build_text_bank(m):
    embs, ids, atts = [], [], []
    for i in range(0, len(gal_txt), 256):
        t = m.tokenizer([txt_proc["eval"](x) for x in gal_txt[i:i+256]], padding="max_length",
                        truncation=True, max_length=35, return_tensors="pt").to(dev)
        out = m.text_encoder.forward_text(t).last_hidden_state
        embs.append(F.normalize(m.text_proj(out[:, 0, :]), dim=-1).float()); ids.append(t.input_ids); atts.append(t.attention_mask)
    return torch.cat(embs), torch.cat(ids), torch.cat(atts)

@torch.no_grad()
def retrieve(m, bank, img_path, k=16, top=3):
    emb, ids, atts = bank
    x = vis_proc["eval"](Image.open(img_path).convert("RGB")).unsqueeze(0).to(dev)
    img_emb = m.visual_encoder.forward_features(x)
    sim = (F.normalize(m.vision_proj(img_emb[:, 0, :]), dim=-1).float() @ emb.t())[0]
    topk = sim.topk(k).indices
    # ITM xếp lại k ứng viên đầu (cùng cách với hàm itm_pair_scores của script)
    class _T:  # đối tượng giả có .input_ids/.attention_mask cho forward_text
        pass
    tt = _T(); tt.input_ids = ids[topk]; tt.attention_mask = atts[topk]
    te = m.text_encoder.forward_text(tt).last_hidden_state
    att = torch.ones(img_emb.size()[:-1], dtype=torch.long, device=dev).expand(k, -1)
    cls = m.text_encoder(encoder_embeds=te, attention_mask=atts[topk], encoder_hidden_states=img_emb.expand(k, -1, -1),
                         encoder_attention_mask=att, return_dict=True, mode="fusion").last_hidden_state[:, 0, :]
    lg = m.itm_head(cls).float()
    margin = (lg[:, 1] - lg[:, 0])
    order = margin.argsort(descending=True)[:top]
    return [(topk[j].item(), margin[j].item()) for j in order]

bank_o, bank_u = build_text_bank(model_o), build_text_bank(model_u)

def show(img_id, group):
    path = f"{IMG_ROOT}/{img_id}"
    ro, ru = retrieve(model_o, bank_o, path), retrieve(model_u, bank_u, path)
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.4)); ax.imshow(Image.open(path)); ax.axis("off")
    ax.set_title(f"[{group}] {img_id.split('/')[-1]}", fontsize=9)
    fig.savefig(os.path.join(OUT, re.sub(r'\W+', '_', img_id) + '.png'), dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f'ảnh: {img_id}  [{group}]')
    for name, r in (("GỐC       ", ro), ("ĐÃ QUÊN    ", ru)):
        print(name)
        for j, mg in r:
            ok = "✓" if gal_owner[j] == img_id else "✗"
            print(f"   {ok} ({mg:+.1f}) {textwrap.shorten(gal_txt[j], 95)}")
    print("-" * 100)

rs = random.Random(1)
for im in rs.sample(df_ids, 3):
    show(im, "D_f - ảnh bị xóa")
for im in rs.sample(dr_ids, 2):
    show(im, "D_r - ảnh giữ lại (đối chứng)")
