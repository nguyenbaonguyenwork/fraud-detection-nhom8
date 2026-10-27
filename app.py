import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import FEATURE_COLS
from model_pipeline import FraudPipeline  # noqa: F401  (cần để joblib tải được pipeline)

st.set_page_config(page_title="Đánh giá rủi ro giao dịch", page_icon="💳", layout="centered")

# ---------------------------------------------------------------- tải mô hình
ART = next((p for p in ("artifacts", ".") if os.path.exists(f"{p}/pipeline.joblib")), None)
if ART is None:
    st.error("Không tìm thấy file `pipeline.joblib` trong repo.")
    st.stop()


@st.cache_resource
def load_pipeline():
    return joblib.load(f"{ART}/pipeline.joblib")


pipe = load_pipeline()
CATEGORIES = sorted(c.replace("category_", "") for c in pipe.ohe_names)

TIERS = {
    1: ("🟢 TIER 1 – AN TOÀN", "Phê duyệt tự động (Approve)", "success",
        "Giao dịch phù hợp với hành vi thông thường. Có thể duyệt ngay."),
    2: ("🟠 TIER 2 – NGHI NGỜ", "Xem xét thủ công (Manual Review)", "warning",
        "Có dấu hiệu bất thường. Cần nhân viên xác minh với chủ thẻ trước khi duyệt."),
    3: ("🔴 TIER 3 – RỦI RO CAO", "Chặn / Từ chối (Block)", "error",
        "Khả năng gian lận cao. Nên chặn giao dịch và liên hệ chủ thẻ."),
}

# ---------------------------------------------------------------- giá trị mặc định + kịch bản mẫu
DEFAULTS = dict(
    amt=45.0, category="grocery_pos", hour=14, dow=2, dist=12.0, new_merchant=False,
    has_hist=True, avg_amt=60.0, std_amt=40.0, last_gap_min=240.0, prev_amt=50.0,
    max_cat=150.0, t5=0, t1=0, t24=2, s1=0.0, s24=90.0, nso=0,
    typ_dist=76.0, std_dist=29.0,
)
PRESETS = {
    "✅ Giao dịch bình thường": {},
    "⚠️ Mua online ban đêm, số tiền lớn": dict(
        amt=850.0, category="shopping_net", hour=2, dist=95.0, new_merchant=True,
        last_gap_min=600.0, prev_amt=40.0, max_cat=120.0, t24=1, s24=40.0),
    "🚨 Thử thẻ: nhiều giao dịch nhỏ liên tiếp": dict(
        amt=3.5, category="misc_net", hour=3, new_merchant=True, last_gap_min=1.0,
        prev_amt=2.9, max_cat=15.0, t5=4, t1=7, t24=12, s1=22.0, s24=60.0, nso=9),
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset(name):
    for k, v in {**DEFAULTS, **PRESETS[name]}.items():
        st.session_state[k] = v


# ---------------------------------------------------------------- tạo 23 đặc trưng từ thông tin nhập
def build_input_row(s):
    amt, has_hist = float(s["amt"]), bool(s["has_hist"])
    avg, std = float(s["avg_amt"]), max(float(s["std_amt"]), 1e-9)
    cz = (amt - avg) / std if has_hist else 0.0
    max_cat = float(s["max_cat"])
    t5, t1, t24 = int(s["t5"]), int(s["t1"]), int(s["t24"])
    t1, t24 = max(t1, t5), max(t24, t1)           # đảm bảo 5 phút ≤ 1 giờ ≤ 24 giờ
    prev = float(s["prev_amt"])
    sd = max(float(s["std_dist"]), 1e-9)
    hour = int(s["hour"])
    return pd.DataFrame([{
        "time_diff": float(s["last_gap_min"]) * 60 if has_hist else 0.0,
        "transactions_last_5min": t5, "transactions_last_1h": t1, "transactions_last_24h": t24,
        "amount_sum_1h": float(s["s1"]), "amount_sum_24h": float(s["s24"]),
        "amt": amt, "customer_avg_amount": avg if has_hist else np.nan,
        "customer_amt_zscore": cz,
        "amt_zscore_cat": cz if (has_hist and max_cat > 0) else 0.0,
        "amt_ratio_cat_max": amt / max_cat if max_cat > 0 else 0.0,
        "has_customer_history": int(has_hist),
        "amt_jump_ratio": amt / prev if (has_hist and prev > 0) else 0.0,
        "is_small_txn": int(amt < 20), "n_small_online_24h": int(s["nso"]),
        "category": s["category"],
        "is_online_cat": int(s["category"] in ("shopping_net", "misc_net")),
        "is_new_merchant": int(s["new_merchant"]),
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "is_night": int(hour >= 22 or hour <= 6), "day_of_week": int(s["dow"]),
        "distance_zscore": (float(s["dist"]) - float(s["typ_dist"])) / sd if has_hist else 0.0,
    }])[FEATURE_COLS]


def explain(row):
    r, out = row.iloc[0], []
    if r.is_night: out.append("Giao dịch vào ban đêm (22h–6h)")
    if r.is_online_cat: out.append("Danh mục mua sắm trực tuyến")
    if r.has_customer_history and r.customer_amt_zscore >= 3:
        out.append(f"Số tiền cao bất thường so với thói quen thẻ (z = {r.customer_amt_zscore:.1f})")
    if r.amt_ratio_cat_max > 1.5: out.append("Vượt xa mức chi lớn nhất từng có ở danh mục này")
    if r.transactions_last_5min >= 3: out.append(f"{int(r.transactions_last_5min)} giao dịch trong 5 phút gần nhất")
    if r.n_small_online_24h >= 3: out.append("Nhiều giao dịch nhỏ trực tuyến trong 24h (dấu hiệu thử thẻ)")
    if r.is_new_merchant: out.append("Merchant lần đầu xuất hiện với thẻ này")
    if r.distance_zscore >= 2: out.append("Khoảng cách tới merchant xa bất thường")
    if r.amt_jump_ratio >= 5: out.append("Số tiền tăng vọt so với giao dịch liền trước")
    if not r.has_customer_history: out.append("Thẻ chưa có lịch sử giao dịch")
    return out


# ---------------------------------------------------------------- giao diện
st.title("💳 Đánh giá rủi ro giao dịch")
st.caption("Nhập thông tin giao dịch → hệ thống xếp vào 1 trong 3 tier. Mô hình: Isolation Forest + XGBoost.")

with st.sidebar:
    st.header("Ngưỡng phân tier")
    low = st.slider("Dưới ngưỡng này → Tier 1", 0.01, 0.30, 0.05, 0.01)
    high = st.slider("Từ ngưỡng này trở lên → Tier 3", 0.02, 0.90, 0.12, 0.01)
    if high <= low:
        st.warning("Ngưỡng Tier 3 phải lớn hơn ngưỡng Tier 1.")
    st.caption("Mặc định theo notebook: 0.05 và 0.12.")

st.markdown("**Thử nhanh với kịch bản mẫu:**")
cols = st.columns(len(PRESETS))
for c, name in zip(cols, PRESETS):
    c.button(name, on_click=apply_preset, args=(name,), use_container_width=True)

st.subheader("1. Thông tin giao dịch")
a, b = st.columns(2)
a.number_input("Số tiền (USD)", 0.01, 100000.0, step=1.0, key="amt")
b.selectbox("Danh mục", CATEGORIES, key="category")
a.slider("Giờ giao dịch (0–23)", 0, 23, key="hour")
b.selectbox("Thứ trong tuần", range(7), key="dow", format_func=lambda d:
            ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"][d])
a.number_input("Khoảng cách từ chủ thẻ tới cửa hàng (km)", 0.0, 20000.0, step=1.0, key="dist")
b.checkbox("Cửa hàng này mới với thẻ (chưa từng giao dịch)", key="new_merchant")

st.subheader("2. Lịch sử của thẻ")
st.checkbox("Thẻ đã có lịch sử giao dịch", key="has_hist")
if st.session_state["has_hist"]:
    c, d = st.columns(2)
    c.number_input("Số tiền giao dịch trung bình của thẻ (USD)", 0.0, 100000.0, step=1.0, key="avg_amt")
    d.number_input("Số tiền giao dịch liền trước (USD)", 0.0, 100000.0, step=1.0, key="prev_amt")
    c.number_input("Cách giao dịch trước bao nhiêu phút?", 0.0, 1e6, step=1.0, key="last_gap_min")
    d.number_input("Mức chi lớn nhất từng có ở danh mục này (USD, 0 = chưa có)", 0.0, 100000.0, step=1.0, key="max_cat")
    st.markdown("**Hoạt động gần đây (không tính giao dịch đang xét)**")
    e, f, g = st.columns(3)
    e.number_input("Số GD trong 5 phút qua", 0, 100, key="t5")
    f.number_input("Số GD trong 1 giờ qua", 0, 200, key="t1")
    g.number_input("Số GD trong 24 giờ qua", 0, 500, key="t24")
    f.number_input("Tổng tiền 1 giờ qua (USD)", 0.0, 1e6, step=1.0, key="s1")
    g.number_input("Tổng tiền 24 giờ qua (USD)", 0.0, 1e6, step=1.0, key="s24")
    e.number_input("Số GD nhỏ (<$20) online trong 24h", 0, 100, key="nso")
    with st.expander("Nâng cao: mức dao động thông thường của thẻ"):
        st.number_input("Độ lệch chuẩn số tiền của thẻ (USD)", 0.01, 100000.0, step=1.0, key="std_amt")
        st.number_input("Khoảng cách thường ngày trung bình (km)", 0.0, 20000.0, step=1.0, key="typ_dist")
        st.number_input("Độ lệch chuẩn khoảng cách (km)", 0.01, 20000.0, step=1.0, key="std_dist")

if st.button("🔎 Đánh giá giao dịch", type="primary", use_container_width=True):
    row = build_input_row(st.session_state)
    prob, if_score = pipe.score_frame(row)
    p = float(prob[0])
    tier = 1 if p < low else 2 if p < high else 3
    title, action, kind, advice = TIERS[tier]

    st.divider()
    getattr(st, kind)(f"## {title}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Xác suất gian lận", f"{p:.2%}")
    m2.metric("Quyết định", action.split(" (")[0])
    m3.metric("Điểm bất thường (IF)", f"{if_score[0]:.3f}")
    st.progress(min(p / max(high * 2, 1e-9), 1.0), text="Mức rủi ro (thang 0 → 2× ngưỡng Tier 3)")
    st.write(advice)

    reasons = explain(row)
    st.subheader("Các dấu hiệu đáng chú ý")
    if reasons:
        for r in reasons:
            st.write(f"• {r}")
    else:
        st.write("Không có dấu hiệu bất thường rõ rệt.")
    st.caption("Các dấu hiệu chỉ mang tính gợi ý; quyết định tier do mô hình đưa ra. "
               "Một số đặc trưng (z-score) được ước tính từ các số bạn nhập ở trên.")
