"""FraudGuard – Nhóm 8: ứng dụng Streamlit phát hiện gian lận thẻ tín dụng.

Chạy: streamlit run app.py   (cần thư mục artifacts/ tạo bởi train.py)
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import RAW_REQUIRED, build_feature_frame
from model_pipeline import (C_REVIEW, FINAL_THRESHOLD, L_FN, LOW_THRESHOLD,
                            decide)

ART = "artifacts"

st.set_page_config(page_title="FraudGuard | Nhóm 8", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1300px; padding-top: 1.5rem;}
    [data-testid="stMetric"] {background:#fff; border:1px solid #E2E8F0;
        border-radius:12px; padding:14px 16px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Tải artifacts
# ------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load(f"{ART}/pipeline.joblib")


@st.cache_data
def load_json(name):
    with open(f"{ART}/{name}", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_demo():
    return pd.read_csv(f"{ART}/demo_test_sample.csv")


if not os.path.exists(f"{ART}/pipeline.joblib"):
    st.error("Chưa có thư mục `artifacts/`. Hãy chạy: "
             "`python train.py --data fraudTrain.csv` rồi mở lại ứng dụng.")
    st.stop()

pipe = load_pipeline()
eda = load_json("eda.json")
metrics = load_json("metrics.json")
demo = load_demo()


def badge(p, low, high):
    return decide(p, low, high)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ FraudGuard")
    st.caption("Isolation Forest + XGBoost · Nhóm 8")
    st.markdown("### Ngưỡng quyết định")
    low = st.slider("Ngưỡng duyệt tự động (dưới mức này)", 0.01, 0.50,
                    float(LOW_THRESHOLD), 0.01)
    high = st.slider("Ngưỡng chặn (từ mức này trở lên)", 0.02, 0.99,
                     float(FINAL_THRESHOLD), 0.01)
    if low >= high:
        st.warning("Ngưỡng duyệt nên nhỏ hơn ngưỡng chặn.")
    st.markdown("### Chi phí")
    st.write(f"Bỏ sót 1 gian lận: **{L_FN:,.0f}$**")
    st.write(f"Xem xét 1 giao dịch: **{C_REVIEW:,.0f}$**")

st.title("Hệ thống phát hiện gian lận thẻ tín dụng")
tabs = st.tabs(["📊 Tổng quan dữ liệu", "🔍 Chấm điểm giao dịch",
                "📈 Hiệu năng mô hình", "🧪 Demo trên tập Test"])

# ------------------------------------------------------------
# Tab 1: EDA
# ------------------------------------------------------------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số giao dịch", f"{eda['n_rows']:,}")
    c2.metric("Số gian lận", f"{eda['n_fraud']:,}")
    c3.metric("Tỷ lệ gian lận", f"{eda['fraud_rate_pct']:.2f}%")
    c4.metric("TB số tiền (gian lận / hợp lệ)",
              f"{eda['amt_mean_fraud']:.0f}$ / {eda['amt_mean_normal']:.0f}$")
    st.caption(f"Giai đoạn dữ liệu: {eda['date_min'][:10]} → {eda['date_max'][:10]}")

    left, right = st.columns(2)
    with left:
        st.subheader("Tỷ lệ gian lận theo danh mục (%)")
        cat = pd.DataFrame(eda["fraud_by_category"])
        cat["rate"] *= 100
        st.bar_chart(cat.set_index("category")["rate"])
    with right:
        st.subheader("Tỷ lệ gian lận theo giờ (%)")
        hr = pd.DataFrame(eda["fraud_by_hour"])
        hr["rate"] *= 100
        st.bar_chart(hr.set_index("hour")["rate"])

# ------------------------------------------------------------
# Tab 2: Chấm điểm
# ------------------------------------------------------------
with tabs[1]:
    st.subheader("Tải file giao dịch thô để chấm điểm")
    st.caption("Cột bắt buộc: " + ", ".join(RAW_REQUIRED) +
               ". Đặc trưng lịch sử được tính theo từng thẻ (cc_num) trong chính file tải lên.")
    up = st.file_uploader("Chọn file CSV", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        missing = [c for c in RAW_REQUIRED if c not in raw.columns]
        if missing:
            st.error("Thiếu cột: " + ", ".join(missing))
        else:
            with st.spinner("Đang tạo đặc trưng và chấm điểm..."):
                feat = build_feature_frame(raw)
                prob, if_s = pipe.score_frame(feat)
            res = feat[["trans_date_trans_time", "cc_num", "merchant",
                        "category", "amt"]].copy()
            res["probability"] = prob
            res["if_score"] = if_s
            res["decision"] = badge(prob, low, high)

            counts = res["decision"].value_counts()
            cols = st.columns(3)
            for col, name in zip(cols, ["Phê duyệt tự động (Approve)",
                                        "Xem xét thủ công (Manual Review)",
                                        "Chặn / Từ chối (Block)"]):
                col.metric(name, f"{int(counts.get(name, 0)):,}")

            st.dataframe(res.sort_values("probability", ascending=False),
                         use_container_width=True, height=420)
            st.download_button("⬇️ Tải kết quả (CSV)",
                               res.to_csv(index=False).encode("utf-8"),
                               "ket_qua_cham_diem.csv", "text/csv")
    else:
        st.info("Chưa có file. Bạn có thể thử bằng tab **Demo trên tập Test**.")

# ------------------------------------------------------------
# Tab 3: Hiệu năng
# ------------------------------------------------------------
with tabs[2]:
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC (Test)", f"{metrics['roc_auc']:.4f}")
    c2.metric("PR-AUC (Test)", f"{metrics['pr_auc']:.4f}")
    c3.metric("Ngưỡng chốt", f"{metrics['final_threshold']:.2f}")

    st.subheader("So sánh tại ngưỡng 0.50 và ngưỡng chốt")
    comp = pd.DataFrame([metrics["at_default"], metrics["at_final"]])
    comp.insert(0, "kịch bản", ["Mặc định 0.50", "Ngưỡng chốt"])
    st.dataframe(comp, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Chi phí theo ngưỡng (Validation)")
        cc = metrics["val_cost_curve"]
        st.line_chart(pd.DataFrame({"cost": cc["cost"]}, index=cc["threshold"]))
        st.caption(f"Ngưỡng tối ưu trên Validation: {metrics['best_val_threshold']:.2f}")
    with right:
        st.subheader("Mức độ quan trọng của đặc trưng (gain)")
        imp = pd.DataFrame(metrics["importance"]).head(15)
        st.bar_chart(imp.set_index("feature")["gain"])

    st.subheader("Phân bổ quyết định trên tập Test")
    st.bar_chart(pd.Series(metrics["decision_counts"]))

# ------------------------------------------------------------
# Tab 4: Demo
# ------------------------------------------------------------
with tabs[3]:
    st.caption("Mẫu Test đã lấy lại (toàn bộ gian lận + giao dịch hợp lệ ngẫu nhiên) "
               "nên Precision trên mẫu này không đại diện cho thực tế.")
    d = demo.copy()
    d["decision"] = badge(d["probability"].to_numpy(), low, high)

    f1, f2 = st.columns(2)
    only = f1.selectbox("Hiển thị", ["Tất cả", "Chỉ gian lận thật", "Chỉ hợp lệ"])
    dec = f2.multiselect("Lọc theo quyết định", sorted(d["decision"].unique()),
                         default=sorted(d["decision"].unique()))
    if only == "Chỉ gian lận thật":
        d = d[d["is_fraud"] == 1]
    elif only == "Chỉ hợp lệ":
        d = d[d["is_fraud"] == 0]
    d = d[d["decision"].isin(dec)]

    pred_block = (d["probability"] >= high)
    st.write(f"**{len(d):,}** giao dịch · bị chặn: **{int(pred_block.sum()):,}** · "
             f"gian lận thật: **{int(d['is_fraud'].sum()):,}**")
    st.dataframe(d.sort_values("probability", ascending=False),
                 use_container_width=True, height=450)

    st.subheader("Thử một giao dịch")
    if len(d):
        i = st.number_input("Chỉ số dòng trong bảng đã lọc", 0, len(d) - 1, 0)
        row = d.iloc[int(i)]
        p = float(row["probability"])
        st.progress(min(max(p, 0.0), 1.0), text=f"Xác suất gian lận: {p:.1%}")
        st.write(f"Quyết định: **{row['decision']}** · Nhãn thật: "
                 f"**{'Gian lận' if row['is_fraud'] == 1 else 'Hợp lệ'}**")

st.markdown("---")
st.caption("FraudGuard · Nhóm 8 · Bản demo phục vụ mục đích học thuật.")
