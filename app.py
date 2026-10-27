import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from model_pipeline import (
    C_REVIEW,
    COST_FN,
    COST_FP,
    COST_TN,
    COST_TP,
    FINAL_THRESHOLD,
    L_FN,
    LOW_THRESHOLD,
    decide,
)

# ============================================================
# CẤU HÌNH TRANG + GIAO DIỆN
# ============================================================
st.set_page_config(
    page_title="FraudGuard | Nhóm 8",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ART = "artifacts"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --navy: #142D4E;
        --blue: #2F6FED;
        --ink: #172B4D;
        --muted: #64748B;
        --line: #E2E8F0;
        --bg-soft: #F5F8FC;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background: #F7F9FC;
        color: var(--ink);
    }
    [data-testid="stHeader"] {
        background: rgba(247, 249, 252, 0.92);
    }
    [data-testid="stSidebar"] {
        background: #102641;
        border-right: 1px solid #203B5D;
    }
    [data-testid="stSidebar"] * {
        color: #EAF1FA;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #C3D2E5;
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }
    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--navy);
        letter-spacing: -0.025em;
    }
    h1 { font-weight: 800 !important; }
    h2, h3 { font-weight: 700 !important; }
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--line);
        padding: 16px 18px;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(20, 45, 78, 0.035);
    }
    [data-testid="stMetricLabel"] p {
        color: #64748B !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--navy) !important;
        font-weight: 800 !important;
    }
    div[data-testid="stTabs"] button {
        border-radius: 10px 10px 0 0;
        font-weight: 600;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--blue);
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background: var(--blue);
        border: 1px solid var(--blue);
        border-radius: 10px;
        padding: 0.65rem 1.15rem;
        font-weight: 700;
    }
    div[data-testid="stButton"] button {
        border-radius: 10px;
    }
    div[data-testid="stDownloadButton"] button {
        border-radius: 10px;
        font-weight: 600;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    div[data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 12px;
    }
    .hero {
        background: linear-gradient(115deg, #102641 0%, #1A416C 72%, #285B91 100%);
        border-radius: 20px;
        padding: 28px 30px;
        color: #FFFFFF;
        margin-bottom: 22px;
        box-shadow: 0 10px 28px rgba(16, 38, 65, 0.12);
    }
    .hero-eyebrow {
        color: #BBD3F4;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 9px;
    }
    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.65rem, 3vw, 2.35rem);
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.2;
        margin: 0 0 10px 0;
        color: #FFFFFF;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #D8E5F5;
        max-width: 760px;
        line-height: 1.65;
        margin: 0;
    }
    .section-kicker {
        color: #64748B;
        font-size: 0.73rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 4px;
    }
    .info-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 17px 18px;
        height: 100%;
    }
    .info-title {
        color: #142D4E;
        font-weight: 700;
        font-size: 0.98rem;
        margin-bottom: 7px;
    }
    .info-text {
        color: #64748B;
        font-size: 0.88rem;
        line-height: 1.6;
    }
    .decision-card {
        border-radius: 16px;
        padding: 20px 22px;
        border: 1px solid;
        margin: 12px 0 16px 0;
    }
    .decision-card h3 {
        margin: 0 0 7px 0;
        font-size: 1.2rem;
    }
    .decision-card p {
        margin: 0;
        line-height: 1.6;
        font-size: 0.92rem;
    }
    .decision-approve {
        background: #F0FDF4;
        border-color: #BBF7D0;
    }
    .decision-approve h3 { color: #166534; }
    .decision-approve p { color: #276749; }
    .decision-review {
        background: #FFFBEB;
        border-color: #FDE68A;
    }
    .decision-review h3 { color: #92400E; }
    .decision-review p { color: #854D0E; }
    .decision-block {
        background: #FFF1F2;
        border-color: #FECDD3;
    }
    .decision-block h3 { color: #9F1239; }
    .decision-block p { color: #9F1239; }
    .small-note {
        color: #64748B;
        font-size: 0.82rem;
        line-height: 1.55;
    }
    .sidebar-brand {
        font-family: 'Manrope', sans-serif;
        font-size: 1.25rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.03em;
        padding: 8px 0 2px 0;
    }
    .sidebar-caption {
        color: #BBD3F4;
        font-size: 0.76rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    hr {
        border-color: #E2E8F0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TẢI TÀI NGUYÊN
# ============================================================
@st.cache_resource
def load_pipeline():
    return joblib.load(f"{ART}/pipeline.joblib")


@st.cache_data
def load_json(name):
    with open(f"{ART}/{name}", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_demo():
    return pd.read_csv(
        f"{ART}/demo_test_sample.csv",
        parse_dates=["trans_date_trans_time"],
    )


if not os.path.exists(f"{ART}/pipeline.joblib"):
    st.error(
        "Chưa có `artifacts/pipeline.joblib`. Hãy chạy "
        "`python train.py --data fraudTrain.csv` rồi commit thư mục `artifacts/`."
    )
    st.stop()

try:
    pipe = load_pipeline()
    eda = load_json("eda.json")
    met = load_json("metrics.json")
except FileNotFoundError as exc:
    st.error(f"Thiếu file tài nguyên: `{exc.filename}`. Hãy kiểm tra thư mục `artifacts/`.")
    st.stop()


# ============================================================
# HÀM HỖ TRỢ HIỂN THỊ
# ============================================================
def decision_type(decision):
    text = str(decision).lower()
    if "block" in text or "chặn" in text or "từ chối" in text:
        return "block"
    if "review" in text or "xem xét" in text:
        return "review"
    return "approve"


def render_decision_explanation(probability, decision):
    """Giải thích quyết định theo ngưỡng; không suy diễn đây là giải thích SHAP."""
    kind = decision_type(decision)
    pct = f"{probability:.2%}"

    if kind == "block":
        title = "Giao dịch được đề xuất chặn"
        body = (
            f"Xác suất do mô hình ước tính là {pct}, nằm tại hoặc cao hơn ngưỡng "
            f"chặn ({FINAL_THRESHOLD:.4f}). Theo quy tắc quyết định hiện tại, "
            "giao dịch được xếp vào nhóm rủi ro cao."
        )
    elif kind == "review":
        title = "Giao dịch cần được xem xét thủ công"
        body = (
            f"Xác suất do mô hình ước tính là {pct}, nằm trong vùng xem xét thủ công "
            f"từ {LOW_THRESHOLD:.4f} đến dưới {FINAL_THRESHOLD:.4f}. Kết quả chưa "
            "đủ để tự động chấp nhận hoặc chặn theo ngưỡng đang cấu hình."
        )
    else:
        title = "Giao dịch được đề xuất phê duyệt tự động"
        body = (
            f"Xác suất do mô hình ước tính là {pct}, thấp hơn ngưỡng xem xét "
            f"({LOW_THRESHOLD:.4f}). Theo quy tắc quyết định hiện tại, giao dịch "
            "được xếp vào nhóm rủi ro thấp hơn."
        )

    css_class = {
        "approve": "decision-approve",
        "review": "decision-review",
        "block": "decision-block",
    }[kind]

    st.markdown(
        f"""
        <div class="decision-card {css_class}">
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Cách đọc quyết định**")
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f"""<div class="info-card">
        <div class="info-title">Phê duyệt tự động</div>
        <div class="info-text">Xác suất &lt; {LOW_THRESHOLD:.4f}. Rủi ro ước tính thấp hơn vùng cảnh báo.</div>
        </div>""",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"""<div class="info-card">
        <div class="info-title">Xem xét thủ công</div>
        <div class="info-text">{LOW_THRESHOLD:.4f} ≤ xác suất &lt; {FINAL_THRESHOLD:.4f}. Cần kiểm tra thêm trước khi quyết định.</div>
        </div>""",
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"""<div class="info-card">
        <div class="info-title">Chặn giao dịch</div>
        <div class="info-text">Xác suất ≥ {FINAL_THRESHOLD:.4f}. Đạt ngưỡng rủi ro cao theo cấu hình hiện tại.</div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.caption(
        "Lưu ý: Đây là giải thích theo xác suất và ngưỡng ra quyết định, không phải "
        "giải thích nhân quả hay SHAP cho từng đặc trưng. Kết quả mô hình là tín hiệu "
        "hỗ trợ ra quyết định, không tự nó chứng minh có gian lận."
    )


# ============================================================
# THANH ĐIỀU HƯỚNG
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🛡️ FraudGuard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-caption">Credit card risk analytics</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**Nhóm 8 · FinTech**")
    st.markdown(
        "Công cụ minh họa phát hiện gian lận thẻ tín dụng bằng Machine Learning."
    )
    st.markdown("---")
    st.markdown("**Các khu vực**")
    st.markdown("01 · Tổng quan dữ liệu")
    st.markdown("02 · Phân tích giao dịch")
    st.markdown("03 · Kiểm thử ngưỡng")
    st.markdown("04 · Mô hình & chi phí")
    st.markdown("---")
    st.markdown(
        '<div class="small-note">Bản demo học thuật. Các quyết định cần được '
        'đánh giá cùng bối cảnh nghiệp vụ.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">Group 8 · FinTech / Risk Analytics</div>
        <div class="hero-title">Phát hiện gian lận thẻ tín dụng</div>
        <p class="hero-subtitle">
            Phân tích rủi ro giao dịch, kiểm tra kết quả mô hình và hỗ trợ diễn giải
            quyết định dựa trên xác suất cùng ngưỡng chi phí.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "Tổng quan",
        "Chấm điểm giao dịch",
        "Demo trên tập Test",
        "Mô hình & chi phí",
    ]
)


# ============================================================
# TAB 1 — TỔNG QUAN
# ============================================================
with tabs[0]:
    st.markdown('<div class="section-kicker">Risk monitoring</div>', unsafe_allow_html=True)
    st.header("Tổng quan dữ liệu và hiệu suất")
    st.write(
        "Tóm tắt dữ liệu và chất lượng phát hiện trên tập Test. "
        "Các chỉ số hiệu suất được tính tại ngưỡng đã chọn cho mô hình."
    )

    c = st.columns(4)
    c[0].metric("Tổng giao dịch", f"{eda['n_rows']:,}")
    c[1].metric("Giao dịch gian lận", f"{eda['n_fraud']:,}")
    c[2].metric("Tỷ lệ gian lận", f"{eda['fraud_rate_pct']:.2f}%")
    c[3].metric("Thời gian dữ liệu", f"{eda['date_min'][:10]} – {eda['date_max'][:10]}")

    st.markdown("### Hiệu suất mô hình trên tập Test")
    f = met["at_final"]
    c = st.columns(5)
    c[0].metric("ROC-AUC", f"{met['roc_auc']:.4f}", help="Khả năng xếp hạng giao dịch dương cao hơn giao dịch âm.")
    c[1].metric("PR-AUC", f"{met['pr_auc']:.4f}", help="Chỉ số hữu ích khi tỷ lệ gian lận thấp và dữ liệu mất cân bằng.")
    c[2].metric("Recall", f"{f['recall']:.3f}", help="Tỷ lệ giao dịch gian lận được phát hiện.")
    c[3].metric("Precision", f"{f['precision']:.3f}", help="Tỷ lệ dự đoán gian lận thực sự là gian lận.")
    c[4].metric("F1-score", f"{f['f1']:.3f}", help="Trung bình điều hòa giữa Precision và Recall.")

    st.markdown("### Mẫu hình gian lận trong dữ liệu")
    left, right = st.columns(2)
    cat = pd.DataFrame(eda["fraud_by_category"]).sort_values("rate", ascending=False)
    with left:
        st.markdown("**Tỷ lệ gian lận theo danh mục**")
        st.bar_chart(cat.set_index("category")["rate"] * 100, y_label="Tỷ lệ (%)")
        st.caption("Tỷ lệ cho biết mức độ phổ biến tương đối trong từng danh mục; không đồng nghĩa danh mục gây ra gian lận.")
    with right:
        hr = pd.DataFrame(eda["fraud_by_hour"])
        st.markdown("**Tỷ lệ gian lận theo giờ**")
        st.bar_chart(hr.set_index("hour")["rate"] * 100, y_label="Tỷ lệ (%)")
        st.caption("Dùng để quan sát mẫu hình theo thời gian trong dữ liệu, không phải quy luật chắc chắn.")

    st.markdown("### Các đặc trưng đóng góp vào mô hình")
    imp = pd.DataFrame(met["importance"]).head(15).copy()
    if not imp.empty and {"feature", "gain"}.issubset(imp.columns):
        imp = imp.sort_values("gain", ascending=True).set_index("feature")
        st.bar_chart(imp["gain"], horizontal=True, x_label="Gain")
        st.caption(
            "Gain phản ánh mức đóng góp của đặc trưng trong các lần phân tách của mô hình cây. "
            "Đây là mức độ quan trọng tổng thể, không giải thích nguyên nhân của một dự đoán riêng lẻ."
        )
    else:
        st.info("Không tìm thấy bảng feature importance hợp lệ trong metrics.json.")


# ============================================================
# TAB 2 — CHẤM ĐIỂM MỘT GIAO DỊCH
# ============================================================
with tabs[1]:
    st.markdown('<div class="section-kicker">Transaction assessment</div>', unsafe_allow_html=True)
    st.header("Đánh giá một giao dịch")
    st.write(
        "Nhập thông tin giao dịch để xem xác suất gian lận ước tính và quyết định theo ngưỡng. "
        "Các trường trong mục nâng cao dành cho việc thử nghiệm hành vi thẻ."
    )

    with st.container(border=True):
        st.markdown("### 01 · Thông tin giao dịch")
        a, b, c3 = st.columns(3)
        amt = a.number_input("Số tiền giao dịch (USD)", 0.01, 50000.0, 120.0, step=1.0)
        category = b.selectbox(
            "Danh mục giao dịch",
            eda["categories"],
            index=eda["categories"].index("shopping_net") if "shopping_net" in eda["categories"] else 0,
        )
        hour = c3.slider("Giờ giao dịch", 0, 23, 2, format="%d:00")
        dow = a.selectbox(
            "Ngày trong tuần",
            list(range(7)),
            format_func=lambda d: ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"][d],
        )
        dist_z = b.number_input(
            "Khoảng cách so với thông lệ (distance z-score)",
            -5.0,
            10.0,
            0.0,
            help="Giá trị chuẩn hóa mô tả độ lệch khoảng cách so với thông lệ được dùng trong pipeline.",
        )
        new_m = c3.checkbox("Merchant mới với thẻ này", value=True)

    with st.expander("02 · Đặc trưng hành vi và lịch sử thẻ", expanded=False):
        st.caption(
            "Các giá trị này là đầu vào của mô hình. Nếu không biết giá trị thực tế, "
            "hãy giữ mặc định chỉ để minh họa; kết quả khi đó không thay thế đánh giá từ dữ liệu thật."
        )
        has_hist = st.checkbox("Thẻ đã có lịch sử giao dịch", value=True)
        g1, g2, g3 = st.columns(3)
        avg_amt = g1.number_input("Số tiền trung bình của khách hàng (USD)", 0.0, 50000.0, 80.0)
        cz = g1.number_input("Độ lệch số tiền so với khách hàng", -10.0, 50.0, 0.0)
        td = g1.number_input("Thời gian từ giao dịch trước (giây)", 0.0, 1e8, 3600.0)
        t5 = g2.number_input("Số giao dịch trong 5 phút", 0, 100, 0)
        t1 = g2.number_input("Số giao dịch trong 1 giờ", 0, 200, 0)
        t24 = g2.number_input("Số giao dịch trong 24 giờ", 0, 500, 1)
        s1 = g2.number_input("Tổng tiền trong 1 giờ (USD)", 0.0, 1e6, 0.0)
        s24 = g3.number_input("Tổng tiền trong 24 giờ (USD)", 0.0, 1e6, 0.0)
        zc = g3.number_input("Độ lệch số tiền theo danh mục", -10.0, 50.0, 0.0)
        rmax = g3.number_input("Tỷ lệ so với mức tối đa theo danh mục", 0.0, 50.0, 1.0)
        jump = g3.number_input("Tỷ lệ tăng số tiền so với trước", 0.0, 1000.0, 1.0)
        nso = g3.number_input("Giao dịch online nhỏ trong 24 giờ", 0, 100, 0)

    run_prediction = st.button("Phân tích giao dịch", type="primary", use_container_width=True)

    if run_prediction:
        row = pd.DataFrame(
            [{
                "time_diff": td,
                "transactions_last_5min": t5,
                "transactions_last_1h": t1,
                "transactions_last_24h": t24,
                "amount_sum_1h": s1,
                "amount_sum_24h": s24,
                "amt": amt,
                "customer_avg_amount": avg_amt if has_hist else np.nan,
                "customer_amt_zscore": cz,
                "amt_zscore_cat": zc,
                "amt_ratio_cat_max": rmax,
                "has_customer_history": int(has_hist),
                "amt_jump_ratio": jump,
                "is_small_txn": int(amt < 20),
                "n_small_online_24h": nso,
                "category": category,
                "is_online_cat": int(category in ("shopping_net", "misc_net")),
                "is_new_merchant": int(new_m),
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                "is_night": int(hour >= 22 or hour <= 6),
                "day_of_week": dow,
                "distance_zscore": dist_z,
            }]
        )

        try:
            probabilities, if_scores = pipe.score_frame(row)
            probability = float(probabilities[0])
            if_score = float(if_scores[0])
            decision = decide([probability])[0]

            st.markdown("---")
            st.markdown('<div class="section-kicker">Prediction result</div>', unsafe_allow_html=True)
            st.subheader("Kết quả phân tích")
            m1, m2, m3 = st.columns(3)
            m1.metric("Xác suất gian lận", f"{probability:.2%}")
            m2.metric("Isolation Forest score", f"{if_score:.4f}")
            m3.metric("Ngưỡng chặn", f"{FINAL_THRESHOLD:.4f}")

            render_decision_explanation(probability, decision)

            st.markdown("### Thông tin giao dịch đã phân tích")
            summary = pd.DataFrame(
                [
                    ("Số tiền", f"${amt:,.2f}"),
                    ("Danh mục", str(category)),
                    ("Thời điểm", f"{hour:02d}:00 · {['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'Chủ nhật'][dow]}"),
                    ("Merchant mới", "Có" if new_m else "Không"),
                    ("Có lịch sử thẻ", "Có" if has_hist else "Không"),
                    ("Số giao dịch trong 1 giờ", f"{t1:,}"),
                    ("Số giao dịch trong 24 giờ", f"{t24:,}"),
                ],
                columns=["Thông tin", "Giá trị"],
            )
            st.dataframe(summary, hide_index=True, use_container_width=True)

        except Exception as exc:
            st.error(
                "Không thể chấm điểm giao dịch với dữ liệu hiện tại. "
                "Hãy kiểm tra pipeline và các đặc trưng đầu vào."
            )
            st.exception(exc)
    else:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">Bạn sẽ nhận được gì?</div>
                <div class="info-text">
                    Sau khi nhấn <b>Phân tích giao dịch</b>, hệ thống hiển thị xác suất gian lận,
                    điểm Isolation Forest, vùng quyết định và cách diễn giải ngưỡng đã áp dụng.
                    Phần giải thích không khẳng định nguyên nhân của dự đoán nếu chưa có phương pháp
                    giải thích theo từng đặc trưng như SHAP.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# TAB 3 — DEMO TRÊN TẬP TEST
# ============================================================
with tabs[2]:
    st.markdown('<div class="section-kicker">Threshold lab</div>', unsafe_allow_html=True)
    st.header("Kiểm thử ngưỡng phân loại")
    st.write(
        "Thử thay đổi ngưỡng nhị phân trên mẫu giao dịch từ tập Test để quan sát "
        "sự đánh đổi giữa bỏ sót gian lận và cảnh báo nhầm."
    )

    try:
        demo = load_demo()
    except FileNotFoundError:
        st.error("Thiếu `artifacts/demo_test_sample.csv`. Hãy kiểm tra thư mục artifacts.")
        st.stop()

    if demo.empty or not {"is_fraud", "probability"}.issubset(demo.columns):
        st.warning("File demo không có dữ liệu hoặc thiếu cột `is_fraud` / `probability`.")
    else:
        t = st.slider(
            "Ngưỡng dự đoán gian lận",
            min_value=0.01,
            max_value=0.99,
            value=float(np.clip(FINAL_THRESHOLD, 0.01, 0.99)),
            step=0.01,
            help="Giao dịch có xác suất lớn hơn hoặc bằng ngưỡng được dự đoán là gian lận trong phần kiểm thử này.",
        )
        y = demo["is_fraud"].to_numpy().astype(int)
        pr = demo["probability"].to_numpy(dtype=float)
        pred = (pr >= t).astype(int)

        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        c = st.columns(5)
        c[0].metric("Recall", f"{recall:.3f}")
        c[1].metric("Precision", f"{precision:.3f}")
        c[2].metric("F1-score", f"{f1:.3f}")
        c[3].metric("Bỏ sót gian lận (FN)", f"{fn:,}")
        c[4].metric("Cảnh báo nhầm (FP)", f"{fp:,}")

        st.markdown("### Ma trận nhầm lẫn")
        cm = pd.DataFrame(
            [[tp, fn], [fp, tn]],
            index=["Thực tế: gian lận", "Thực tế: bình thường"],
            columns=["Dự đoán: gian lận", "Dự đoán: bình thường"],
        )
        st.dataframe(cm, use_container_width=True)

        st.caption(
            "Mẫu demo được lấy có chủ đích với tỷ lệ gian lận cao hơn thực tế; "
            "Precision trong mẫu này không đại diện cho Precision trên toàn bộ tập Test. "
            "Các số liệu đánh giá chuẩn được trình bày ở tab Mô hình & chi phí."
        )

        demo_view = demo.copy()
        demo_view["Dự đoán theo ngưỡng đang chọn"] = np.where(
            pred == 1, "Gian lận", "Bình thường"
        )
        demo_view["Khớp nhãn thực tế"] = np.where(
            pred == y, "Đúng", "Sai"
        )
        st.markdown("### Các giao dịch có xác suất cao")
        st.dataframe(
            demo_view.sort_values("probability", ascending=False).head(100),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# TAB 4 — MÔ HÌNH & CHI PHÍ
# ============================================================
with tabs[3]:
    st.markdown('<div class="section-kicker">Model transparency</div>', unsafe_allow_html=True)
    st.header("Mô hình, ngưỡng và chi phí")
    st.write(
        "Tab này mô tả cách đánh giá mô hình và các giả định chi phí được sử dụng "
        "để lựa chọn ngưỡng. Các mức chi phí là giả định của bài toán, không mặc nhiên "
        "đại diện cho chi phí thực tế của mọi tổ chức."
    )

    st.markdown("### Cấu hình quyết định")
    a, b, c = st.columns(3)
    a.metric("Ngưỡng duyệt tự động", f"{LOW_THRESHOLD:.4f}")
    b.metric("Ngưỡng chặn", f"{FINAL_THRESHOLD:.4f}")
    c.metric("Chi phí chấm điểm", "$0.05 / giao dịch")

    st.markdown("### Ma trận chi phí giả định (USD)")
    cost_matrix = pd.DataFrame(
        [[COST_TP, COST_FN], [COST_FP, COST_TN]],
        index=["Thực tế: gian lận", "Thực tế: bình thường"],
        columns=["Dự đoán: gian lận", "Dự đoán: bình thường"],
    )
    st.dataframe(cost_matrix.style.format("${:,.2f}"), use_container_width=True)
    st.caption(
        f"Tổn thất bỏ sót một giao dịch gian lận = ${L_FN:g}; "
        f"chi phí xác minh một cảnh báo = ${C_REVIEW:g}."
    )

    st.markdown("### So sánh hiệu suất trên tập Test")
    rows = []
    for key, label in [
        ("at_default", "Ngưỡng mặc định 0.50"),
        ("at_final", f"Ngưỡng được chọn {FINAL_THRESHOLD:.4f}"),
    ]:
        m = met[key]
        rows.append(
            {
                "Cấu hình": label,
                "TP": m["tp"],
                "FP": m["fp"],
                "FN": m["fn"],
                "TN": m["tn"],
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1-score": round(m["f1"], 4),
                "Tổng chi phí (USD)": round(m["cost"], 2),
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Chi phí theo ngưỡng trên Validation")
    cv = pd.DataFrame(met["val_cost_curve"]).set_index("threshold")
    st.line_chart(cv["cost"], y_label="Chi phí giả định (USD)", x_label="Threshold")
    st.caption(
        f"Ngưỡng có chi phí thấp nhất trên Validation theo kết quả lưu sẵn: "
        f"{met['best_val_threshold']:.4f}. Tập Test được dùng để đánh giá, "
        "không nên dùng để tiếp tục điều chỉnh ngưỡng."
    )

    st.markdown("### Phân bố quyết định trên tập Test")
    counts = pd.Series(met["decision_counts"], name="Số giao dịch").to_frame()
    st.dataframe(counts, use_container_width=True)

    st.markdown("### Cách diễn giải các chỉ số")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">Recall và chi phí bỏ sót</div>
                <div class="info-text">
                    Recall cao hơn thường đồng nghĩa phát hiện được nhiều giao dịch gian lận hơn,
                    nhưng có thể làm tăng số cảnh báo nhầm. Cần đánh giá cùng chi phí của FN và FP.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">Precision và tỷ lệ gian lận</div>
                <div class="info-text">
                    Precision phụ thuộc đáng kể vào tỷ lệ gian lận trong dữ liệu đánh giá.
                    Vì vậy, không nên suy rộng Precision từ một mẫu demo đã được lấy lại
                    sang môi trường giao dịch thực tế.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown(
    '<div class="small-note">FraudGuard · Nhóm 8 · Bản demo phục vụ mục đích học thuật. '
    'Mọi quyết định chặn giao dịch trong môi trường thực tế cần có quy trình kiểm soát và xác minh phù hợp.</div>',
    unsafe_allow_html=True,
)
