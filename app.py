import json, os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import RAW_REQUIRED, build_feature_frame
from model_pipeline import (C_REVIEW, COST_FN, COST_FP, COST_TN, COST_TP, FINAL_THRESHOLD,
                            L_FN, LOW_THRESHOLD, FraudPipeline, decide, total_cost)

st.set_page_config(page_title="Phát hiện gian lận thẻ tín dụng – Nhóm 8", page_icon="💳", layout="wide")
ART = "artifacts"


@st.cache_resource
def load_pipeline():
    return joblib.load(f"{ART}/pipeline.joblib")


@st.cache_data
def load_json(name):
    with open(f"{ART}/{name}", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_demo():
    return pd.read_csv(f"{ART}/demo_test_sample.csv", parse_dates=["trans_date_trans_time"])


if not os.path.exists(f"{ART}/pipeline.joblib"):
    st.error("Chưa có file `artifacts/pipeline.joblib`. Hãy chạy `python train.py --data fraudTrain.csv` "
             "rồi commit thư mục `artifacts/` (xem README).")
    st.stop()

pipe, eda, met = load_pipeline(), load_json("eda.json"), load_json("metrics.json")

st.title("💳 Phát hiện gian lận thẻ tín dụng")
st.caption("Nhóm 8 · Isolation Forest + XGBoost (bộ đặc trưng E2) · Tối ưu theo chi phí kinh tế")

tabs = st.tabs(["📊 Tổng quan", "🔍 Chấm điểm 1 giao dịch", "📁 Chấm điểm file CSV",
                "🧪 Demo trên tập Test", "⚙️ Mô hình & chi phí"])

# ------------------------------------------------------------------ Tổng quan
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Số giao dịch", f"{eda['n_rows']:,}")
    c[1].metric("Số giao dịch gian lận", f"{eda['n_fraud']:,}")
    c[2].metric("Tỷ lệ gian lận", f"{eda['fraud_rate_pct']:.2f}%")
    c[3].metric("Giai đoạn", f"{eda['date_min'][:10]} → {eda['date_max'][:10]}")
    st.subheader("Kết quả trên tập Test (XGBoost + E2)")
    f = met["at_final"]
    c = st.columns(5)
    c[0].metric("ROC-AUC", f"{met['roc_auc']:.4f}")
    c[1].metric("PR-AUC", f"{met['pr_auc']:.4f}")
    c[2].metric(f"Recall @ {f['threshold']:.2f}", f"{f['recall']:.3f}")
    c[3].metric(f"Precision @ {f['threshold']:.2f}", f"{f['precision']:.3f}")
    c[4].metric("F1", f"{f['f1']:.3f}")

    l, r = st.columns(2)
    cat = pd.DataFrame(eda["fraud_by_category"]).sort_values("rate", ascending=False)
    l.subheader("Tỷ lệ gian lận theo danh mục (%)")
    l.bar_chart(cat.set_index("category")["rate"] * 100)
    hr = pd.DataFrame(eda["fraud_by_hour"])
    r.subheader("Tỷ lệ gian lận theo giờ (%)")
    r.bar_chart(hr.set_index("hour")["rate"] * 100)
    st.subheader("Mức độ quan trọng của đặc trưng (gain)")
    imp = pd.DataFrame(met["importance"]).head(15).set_index("feature")
    st.bar_chart(imp["gain"])

# ------------------------------------------------------------------ 1 giao dịch
with tabs[1]:
    st.write("Nhập thông tin giao dịch. Các đặc trưng hành vi (lịch sử thẻ) có thể để mặc định "
             "nếu thẻ chưa có lịch sử, hoặc chỉnh trong phần *Nâng cao*.")
    a, b, c3 = st.columns(3)
    amt = a.number_input("Số tiền (USD)", 0.01, 50000.0, 120.0, step=1.0)
    category = b.selectbox("Danh mục", eda["categories"], index=eda["categories"].index("shopping_net")
                           if "shopping_net" in eda["categories"] else 0)
    hour = c3.slider("Giờ giao dịch", 0, 23, 2)
    dow = a.selectbox("Thứ trong tuần", list(range(7)), format_func=lambda d:
                      ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"][d])
    dist_z = b.number_input("distance_zscore (khoảng cách so với thông lệ thẻ)", -5.0, 10.0, 0.0)
    new_m = c3.checkbox("Merchant mới với thẻ này", value=True)

    with st.expander("Nâng cao: đặc trưng hành vi / lịch sử thẻ"):
        has_hist = st.checkbox("Thẻ đã có lịch sử giao dịch", value=True)
        g1, g2, g3 = st.columns(3)
        avg_amt = g1.number_input("customer_avg_amount", 0.0, 50000.0, 80.0)
        cz = g1.number_input("customer_amt_zscore", -10.0, 50.0, 0.0)
        td = g1.number_input("time_diff (giây từ giao dịch trước)", 0.0, 1e8, 3600.0)
        t5 = g2.number_input("transactions_last_5min", 0, 100, 0)
        t1 = g2.number_input("transactions_last_1h", 0, 200, 0)
        t24 = g2.number_input("transactions_last_24h", 0, 500, 1)
        s1 = g2.number_input("amount_sum_1h", 0.0, 1e6, 0.0)
        s24 = g3.number_input("amount_sum_24h", 0.0, 1e6, 0.0)
        zc = g3.number_input("amt_zscore_cat", -10.0, 50.0, 0.0)
        rmax = g3.number_input("amt_ratio_cat_max", 0.0, 50.0, 1.0)
        jump = g3.number_input("amt_jump_ratio", 0.0, 1000.0, 1.0)
        nso = g3.number_input("n_small_online_24h", 0, 100, 0)

    if st.button("Chấm điểm giao dịch", type="primary"):
        row = pd.DataFrame([{
            "time_diff": td, "transactions_last_5min": t5, "transactions_last_1h": t1,
            "transactions_last_24h": t24, "amount_sum_1h": s1, "amount_sum_24h": s24,
            "amt": amt, "customer_avg_amount": avg_amt if has_hist else np.nan,
            "customer_amt_zscore": cz, "amt_zscore_cat": zc, "amt_ratio_cat_max": rmax,
            "has_customer_history": int(has_hist), "amt_jump_ratio": jump,
            "is_small_txn": int(amt < 20), "n_small_online_24h": nso, "category": category,
            "is_online_cat": int(category in ("shopping_net", "misc_net")),
            "is_new_merchant": int(new_m), "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "is_night": int(hour >= 22 or hour <= 6), "day_of_week": dow,
            "distance_zscore": dist_z}])
        p, s = pipe.score_frame(row)
        p = float(p[0])
        d = decide([p])[0]
        c = st.columns(3)
        c[0].metric("Xác suất gian lận", f"{p:.2%}")
        c[1].metric("Isolation Forest score", f"{s[0]:.4f}")
        c[2].metric("Quyết định", d.split(" (")[0])
        (st.error if "Block" in d else st.warning if "Review" in d else st.success)(d)
        st.caption(f"Ngưỡng: < {LOW_THRESHOLD} duyệt tự động · {LOW_THRESHOLD}–{FINAL_THRESHOLD} xem xét thủ công · "
                   f"≥ {FINAL_THRESHOLD} chặn.")

# ------------------------------------------------------------------ CSV
with tabs[2]:
    st.write("Tải lên CSV cùng định dạng `fraudTrain.csv`. Cột bắt buộc: "
             + ", ".join(f"`{c}`" for c in RAW_REQUIRED) + ".")
    st.info("Đặc trưng lịch sử (số giao dịch 5 phút/1h/24h, z-score…) được tính **từ chính file tải lên**, "
            "vì vậy file nên chứa đủ lịch sử theo từng thẻ để kết quả giống notebook.")
    up = st.file_uploader("Chọn file CSV", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        missing = [c for c in RAW_REQUIRED if c not in raw.columns]
        if missing:
            st.error(f"Thiếu cột: {missing}")
        elif len(raw) > 300_000:
            st.error("File quá lớn cho bản demo online (tối đa 300.000 dòng).")
        else:
            with st.spinner("Đang tạo đặc trưng và chấm điểm..."):
                feat = build_feature_frame(raw)
                p, s = pipe.score_frame(feat)
            out = feat[["trans_date_trans_time", "cc_num", "merchant", "category", "amt"]].copy()
            if "is_fraud" in feat.columns:
                out["is_fraud"] = feat["is_fraud"]
            out["probability"], out["if_score"] = p, s
            out["decision"] = decide(p)
            c = st.columns(3)
            vc = out["decision"].value_counts()
            for col, k in zip(c, ["Phê duyệt tự động (Approve)", "Xem xét thủ công (Manual Review)",
                                  "Chặn / Từ chối (Block)"]):
                col.metric(k, f"{int(vc.get(k, 0)):,}")
            st.dataframe(out.sort_values("probability", ascending=False).head(1000), use_container_width=True)
            st.download_button("⬇️ Tải kết quả (CSV)", out.to_csv(index=False).encode("utf-8"),
                               "ket_qua_cham_diem.csv", "text/csv")

# ------------------------------------------------------------------ Demo Test
with tabs[3]:
    st.write("Mẫu giao dịch từ tập **Test** (đã chấm điểm sẵn). Kéo ngưỡng để xem đánh đổi giữa "
             "bỏ sót gian lận và cảnh báo nhầm.")
    demo = load_demo()
    scale = (met["n_test"]) / len(demo)
    t = st.slider("Ngưỡng chặn (threshold)", 0.01, 0.99, float(FINAL_THRESHOLD), 0.01)
    y, pr = demo["is_fraud"].to_numpy(), demo["probability"].to_numpy()
    pred = (pr >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    c = st.columns(4)
    c[0].metric("Recall", f"{tp / max(tp + fn, 1):.3f}")
    c[1].metric("Precision", f"{tp / max(tp + fp, 1):.3f}")
    c[2].metric("Bỏ sót (FN)", fn)
    c[3].metric("Cảnh báo nhầm (FP)", fp)
    st.write(pd.DataFrame([[tp, fn], [fp, tn]], index=["Thực tế: Fraud", "Thực tế: Normal"],
                          columns=["Dự đoán: Fraud", "Dự đoán: Normal"]))
    st.caption("Mẫu demo đã được lấy lại với nhiều gian lận hơn thực tế nên Precision ở đây cao hơn "
               "so với toàn bộ Test; số liệu chuẩn nằm ở tab *Mô hình & chi phí*.")
    demo["decision"] = decide(pr)
    st.dataframe(demo.sort_values("probability", ascending=False).head(500), use_container_width=True)

# ------------------------------------------------------------------ Chi phí
with tabs[4]:
    st.subheader("Ma trận chi phí giả định (USD)")
    st.write(pd.DataFrame([[COST_TP, COST_FN], [COST_FP, COST_TN]],
                          index=["Thực tế: Fraud", "Thực tế: Normal"],
                          columns=["Dự đoán: Fraud", "Dự đoán: Normal"]).style.format("${:,.2f}"))
    st.caption(f"Tổn thất bỏ sót 1 gian lận = ${L_FN:g}; xác minh 1 cảnh báo = ${C_REVIEW:g}; "
               "chấm điểm = $0.05/giao dịch.")
    st.subheader("So sánh ngưỡng trên tập Test")
    rows = []
    for k, lab in [("at_default", "Mặc định 0.50"), ("at_final", f"Đã chốt {FINAL_THRESHOLD:.2f}")]:
        m = met[k]
        rows.append({"Cấu hình": lab, "TP": m["tp"], "FP": m["fp"], "FN": m["fn"], "TN": m["tn"],
                     "Precision": round(m["precision"], 4), "Recall": round(m["recall"], 4),
                     "F1": round(m["f1"], 4), "Tổng chi phí ($)": round(m["cost"], 2)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.subheader("Chi phí theo ngưỡng trên Validation")
    cv = pd.DataFrame(met["val_cost_curve"]).set_index("threshold")
    st.line_chart(cv["cost"])
    st.caption(f"Ngưỡng chi phí thấp nhất trên Validation: {met['best_val_threshold']:.2f}.")
    st.subheader("Vùng quyết định trên Test")
    st.write(pd.Series(met["decision_counts"], name="Số giao dịch").to_frame())
