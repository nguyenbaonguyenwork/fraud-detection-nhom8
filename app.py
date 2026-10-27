import datetime as dt
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import FEATURE_COLS, RAW_REQUIRED, build_feature_frame
from model_pipeline import (
    C_MODEL,
    C_REVIEW,
    COST_FN,
    COST_FP,
    COST_TN,
    COST_TP,
    L_FN,
    FraudPipeline,
    total_cost,
)  # Giữ các import để joblib tải pipeline


# ================================================================
# 1. CẤU HÌNH ỨNG DỤNG
# ================================================================

st.set_page_config(
    page_title="Đánh giá rủi ro giao dịch thẻ",
    page_icon="💳",
    layout="wide",
)

LOW_THRESHOLD = 0.05
HIGH_THRESHOLD = 0.12

TIERS = {
    1: (
        "TIER 1 – RỦI RO THẤP",
        "Phê duyệt tự động",
        "success",
        "Điểm rủi ro nằm dưới ngưỡng cảnh báo. "
        "Theo quy tắc phân loại hiện tại, giao dịch được đề xuất phê duyệt tự động.",
    ),
    2: (
        "TIER 2 – CẦN XÁC MINH",
        "Xem xét thủ công",
        "warning",
        "Điểm rủi ro nằm trong vùng cần xem xét. "
        "Nên xác minh thêm thông tin giao dịch trước khi đưa ra quyết định.",
    ),
    3: (
        "TIER 3 – RỦI RO CAO",
        "Chặn hoặc từ chối giao dịch",
        "error",
        "Điểm rủi ro đạt hoặc vượt ngưỡng cảnh báo cao. "
        "Đề xuất tạm giữ hoặc từ chối giao dịch theo chính sách kiểm soát rủi ro.",
    ),
}


def tier_of(p):
    """Phân loại giao dịch theo xác suất gian lận và ngưỡng đã thiết lập."""
    return 1 if p < LOW_THRESHOLD else 2 if p < HIGH_THRESHOLD else 3


# ================================================================
# 2. TẢI MÔ HÌNH
# ================================================================

ART = next(
    (
        path
        for path in ("artifacts", ".")
        if os.path.exists(os.path.join(path, "pipeline.joblib"))
    ),
    None,
)

if ART is None:
    st.error(
        "Không tìm thấy `pipeline.joblib`. "
        "Hãy chạy `train.py` và kiểm tra thư mục `artifacts/`."
    )
    st.stop()


@st.cache_resource
def load_pipeline():
    return joblib.load(os.path.join(ART, "pipeline.joblib"))


@st.cache_data
def load_json(name):
    path = os.path.join(ART, name)

    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as f:
        return json.load(f)


pipe = load_pipeline()

CATEGORIES = sorted(
    c.replace("category_", "", 1)
    for c in pipe.ohe_names
)

DOW = [
    "Thứ 2",
    "Thứ 3",
    "Thứ 4",
    "Thứ 5",
    "Thứ 6",
    "Thứ 7",
    "Chủ nhật",
]


# ================================================================
# 3. HÀM CHẤM ĐIỂM GIAO DỊCH
# ================================================================

def score_transaction(history: pd.DataFrame, tx: dict):
    """
    Tạo đặc trưng bằng build_feature_frame() của dự án,
    sau đó chấm điểm bằng pipeline Isolation Forest + XGBoost.

    history: các giao dịch trước giao dịch đang đánh giá.
    tx: thông tin giao dịch mới.
    """

    t_new = tx["time"]
    rows = []

    for r in history.itertuples(index=False):
        if (
            pd.isna(r.minutes_ago)
            or pd.isna(r.amt)
            or pd.isna(r.category)
            or pd.isna(r.merchant)
            or pd.isna(r.distance_km)
        ):
            continue

        minutes_ago = float(r.minutes_ago)

        if minutes_ago <= 0:
            continue

        rows.append(
            {
                "trans_date_trans_time": (
                    t_new - pd.Timedelta(minutes=minutes_ago)
                ),
                "cc_num": 1,
                "amt": float(r.amt),
                "category": str(r.category),
                "merchant": str(r.merchant),
                "distance_km": float(r.distance_km),
            }
        )

    # Đưa giao dịch mới vào cuối dữ liệu để tạo đặc trưng
    rows.append(
        {
            "trans_date_trans_time": t_new,
            "cc_num": 1,
            "amt": float(tx["amt"]),
            "category": tx["category"],
            "merchant": tx["merchant"],
            "distance_km": float(tx["distance_km"]),
        }
    )

    raw = pd.DataFrame(rows)
    feat = build_feature_frame(raw)

    # Giao dịch đang đánh giá là dòng cuối cùng sau khi sắp xếp
    row = feat.iloc[[-1]]

    prob, ifs = pipe.score_frame(row)

    return (
        float(prob[0]),
        float(ifs[0]),
        row[FEATURE_COLS].reset_index(drop=True),
        len(rows) - 1,
    )


# ================================================================
# 4. GIẢI THÍCH DẤU HIỆU HÀNH VI THEO QUY TẮC
# ================================================================

def explain(r):
    """
    Liệt kê dấu hiệu dựa trên các quy tắc kiểm tra đặc trưng.
    Đây không phải SHAP/LIME và không phải lời giải thích trực tiếp
    cho dự đoán của XGBoost.
    """

    out = []

    if r.is_night:
        out.append("Giao dịch diễn ra ban đêm (22h–6h59).")

    if r.is_online_cat:
        out.append(
            "Giao dịch thuộc danh mục mua sắm trực tuyến "
            "(shopping_net / misc_net)."
        )

    if r.has_customer_history and r.customer_amt_zscore >= 3:
        out.append(
            "Số tiền cao bất thường so với lịch sử chi tiêu "
            f"(z-score = {r.customer_amt_zscore:.1f})."
        )

    if r.amt_ratio_cat_max > 1.5:
        out.append(
            "Số tiền bằng "
            f"{r.amt_ratio_cat_max:.1f} lần mức chi tiêu lớn nhất "
            "từng ghi nhận ở danh mục này."
        )

    if r.transactions_last_5min >= 3:
        out.append(
            f"Có {int(r.transactions_last_5min)} giao dịch "
            "trong khoảng 5 phút gần nhất."
        )

    if r.n_small_online_24h >= 3:
        out.append(
            f"Có {int(r.n_small_online_24h)} giao dịch nhỏ trực tuyến "
            "trong 24 giờ; cần kiểm tra khả năng thử thẻ."
        )

    if r.is_new_merchant:
        out.append("Cửa hàng chưa xuất hiện trong lịch sử của thẻ.")

    if r.distance_zscore >= 2:
        out.append(
            "Khoảng cách tới cửa hàng cao bất thường "
            f"(z-score = {r.distance_zscore:.1f})."
        )

    if r.amt_jump_ratio >= 5:
        out.append(
            "Số tiền giao dịch cao gấp "
            f"{r.amt_jump_ratio:.1f} lần giao dịch liền trước."
        )

    if not r.has_customer_history:
        out.append(
            "Chưa có lịch sử giao dịch đủ điều kiện để đánh giá "
            "hành vi trước đó của thẻ."
        )

    return out


# ================================================================
# 5. DỮ LIỆU MINH HỌA
# ================================================================

def _hist(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "minutes_ago",
            "amt",
            "category",
            "merchant",
            "distance_km",
        ],
    )


_NORMAL_HIST = [
    (300, 42.0, "grocery_pos", "Cửa hàng A", 15.0),
    (1500, 65.0, "gas_transport", "Trạm xăng B", 40.0),
    (2900, 38.0, "food_dining", "Nhà hàng C", 22.0),
    (4400, 80.0, "grocery_pos", "Cửa hàng A", 18.0),
    (6000, 55.0, "home", "Cửa hàng D", 60.0),
    (8800, 47.0, "food_dining", "Nhà hàng C", 25.0),
    (11000, 72.0, "gas_transport", "Trạm xăng B", 35.0),
    (14000, 60.0, "grocery_pos", "Cửa hàng A", 20.0),
]


PRESETS = {
    "Giao dịch thông thường": {
        "hist": _NORMAL_HIST,
        "amt": 45.0,
        "category": "grocery_pos",
        "merchant": "Cửa hàng A",
        "dist": 20.0,
        "t": dt.time(14, 0),
    },

    "Mua sắm trực tuyến ban đêm": {
        "hist": _NORMAL_HIST,
        "amt": 850.0,
        "category": "shopping_net",
        "merchant": "Shop online X",
        "dist": 95.0,
        "t": dt.time(2, 30),
    },

    "Nhiều giao dịch nhỏ liên tiếp": {
        "hist": [
            (1, 2.9, "misc_net", "Web 1", 80.0),
            (2, 1.5, "shopping_net", "Web 2", 80.0),
            (3, 3.2, "misc_net", "Web 3", 80.0),
            (4, 2.1, "shopping_net", "Web 4", 80.0),
            (7, 4.0, "misc_net", "Web 5", 80.0),
            (30, 3.5, "shopping_net", "Web 6", 80.0),
            (600, 42.0, "grocery_pos", "Cửa hàng A", 15.0),
            (1500, 65.0, "gas_transport", "Trạm xăng B", 40.0),
            (2900, 38.0, "food_dining", "Nhà hàng C", 22.0),
            (4400, 80.0, "grocery_pos", "Cửa hàng A", 18.0),
        ],
        "amt": 4.5,
        "category": "shopping_net",
        "merchant": "Web 7",
        "dist": 80.0,
        "t": dt.time(3, 10),
    },
}


def apply_preset(name):
    p = PRESETS[name]

    st.session_state.update(
        hist_df=_hist(p["hist"]),
        hist_ver=st.session_state.get("hist_ver", 0) + 1,
        amt=p["amt"],
        category=p["category"],
        merchant=p["merchant"],
        dist=p["dist"],
        new_time=p["t"],
    )


# ================================================================
# 6. KHỞI TẠO SESSION STATE
# ================================================================

if "hist_df" not in st.session_state:
    st.session_state.update(
        hist_df=_hist(_NORMAL_HIST),
        hist_ver=0,
        card_ref="CARD-0001",
        amt=45.0,
        category="grocery_pos",
        merchant="Cửa hàng A",
        dist=20.0,
        new_time=dt.time(14, 0),
        new_date=dt.date.today(),
    )
else:
    st.session_state.setdefault("card_ref", "CARD-0001")
    st.session_state.setdefault("new_date", dt.date.today())


# ================================================================
# 7. TIÊU ĐỀ VÀ THANH ĐIỀU HƯỚNG
# ================================================================

st.title("Đánh giá rủi ro giao dịch thẻ")

st.caption(
    "Mô hình Isolation Forest + XGBoost · Bộ đặc trưng E2 · "
    f"Rủi ro thấp: p < {LOW_THRESHOLD:.2f} · "
    f"Cần xác minh: {LOW_THRESHOLD:.2f} ≤ p < {HIGH_THRESHOLD:.2f} · "
    f"Rủi ro cao: p ≥ {HIGH_THRESHOLD:.2f}"
)

st.write(
    "Ứng dụng hỗ trợ nhận diện giao dịch có dấu hiệu bất thường "
    "dựa trên thông tin giao dịch, lịch sử chi tiêu và mô hình học máy."
)

tab_score, tab_csv, tab_res = st.tabs(
    [
        "Đánh giá giao dịch",
        "Phân tích file giao dịch",
        "Kết quả nghiên cứu",
    ]
)


# ================================================================
# TAB 1: ĐÁNH GIÁ GIAO DỊCH
# ================================================================

with tab_score:

    st.subheader("1. Thông tin thẻ và giao dịch")

    st.write(
        "Nhập thông tin giao dịch cần đánh giá. "
        "Các trường dữ liệu được sử dụng để tạo đặc trưng đầu vào "
        "theo quy trình của mô hình."
    )

    with st.expander("Chọn kịch bản minh họa", expanded=False):
        st.caption(
            "Các kịch bản sử dụng dữ liệu giả lập để trình diễn chức năng. "
            "Áp dụng một kịch bản sẽ thay thế thông tin giao dịch và lịch sử "
            "hiện tại bằng dữ liệu mẫu."
        )

        selected_preset = st.selectbox(
            "Kịch bản",
            options=list(PRESETS.keys()),
            key="selected_preset",
        )

        if st.button("Áp dụng kịch bản", key="apply_preset_btn"):
            apply_preset(selected_preset)
            st.rerun()

    st.text_input(
        "Mã tham chiếu thẻ",
        key="card_ref",
        help=(
            "Mã nhận diện hiển thị trên giao diện. "
            "Trường này không được sử dụng làm đầu vào của mô hình "
            "và chưa kết nối với cơ sở dữ liệu thẻ."
        ),
    )

    col_date, col_time = st.columns(2)

    with col_date:
        st.date_input("Ngày giao dịch", key="new_date")

    with col_time:
        st.time_input("Giờ giao dịch", key="new_time", step=60)

    col_amt, col_category = st.columns(2)

    with col_amt:
        st.number_input(
            "Số tiền giao dịch (USD)",
            min_value=0.01,
            max_value=100000.0,
            step=1.0,
            key="amt",
        )

    with col_category:
        st.selectbox(
            "Danh mục giao dịch",
            options=CATEGORIES,
            key="category",
        )

    col_merchant, col_distance = st.columns(2)

    with col_merchant:
        st.text_input(
            "Tên cửa hàng hoặc đơn vị chấp nhận thanh toán",
            key="merchant",
        )

    with col_distance:
        st.number_input(
            "Khoảng cách từ chủ thẻ đến cửa hàng (km)",
            min_value=0.0,
            max_value=20000.0,
            step=1.0,
            key="dist",
            help=(
                "Nhập khoảng cách ước tính. Với giao dịch trực tuyến, "
                "khoảng cách có thể không mang cùng ý nghĩa như giao dịch trực tiếp."
            ),
        )

    # ------------------------------------------------------------
    # LỊCH SỬ GIAO DỊCH
    # ------------------------------------------------------------

    st.divider()
    st.subheader("2. Lịch sử giao dịch của thẻ")

    st.write(
        "Nhập các giao dịch đã xảy ra trước giao dịch đang đánh giá. "
        "Lịch sử được sử dụng để tính các đặc trưng hành vi như tần suất "
        "giao dịch, mức chi tiêu tương đối và giao dịch tại cửa hàng mới."
    )

    st.info(
        "Lịch sử hiện được nhập thủ công hoặc lấy từ kịch bản mẫu. "
        "Ứng dụng chưa tự truy xuất lịch sử theo mã tham chiếu thẻ. "
        "Lịch sử thiếu hoặc không chính xác có thể làm giảm độ tin cậy "
        "của kết quả đánh giá."
    )

    hist = st.data_editor(
        st.session_state["hist_df"],
        num_rows="dynamic",
        key=f"hist_{st.session_state['hist_ver']}",
        use_container_width=True,
        column_config={
            "minutes_ago": st.column_config.NumberColumn(
                "Số phút trước giao dịch hiện tại",
                min_value=0.01,
                format="%.1f",
                help="Ví dụ: 30 nghĩa là giao dịch xảy ra trước đó 30 phút.",
            ),
            "amt": st.column_config.NumberColumn(
                "Số tiền (USD)",
                min_value=0.0,
                format="%.2f",
            ),
            "category": st.column_config.SelectboxColumn(
                "Danh mục",
                options=CATEGORIES,
            ),
            "merchant": st.column_config.TextColumn("Cửa hàng"),
            "distance_km": st.column_config.NumberColumn(
                "Khoảng cách (km)",
                min_value=0.0,
                format="%.1f",
            ),
        },
    )

    st.caption(
        "Mỗi dòng là một giao dịch trước giao dịch đang đánh giá. "
        "Không thêm giao dịch hiện tại vào bảng lịch sử."
    )

    # ------------------------------------------------------------
    # NÚT ĐÁNH GIÁ VÀ KẾT QUẢ
    # ------------------------------------------------------------

    st.divider()

    evaluate = st.button(
        "Đánh giá mức độ rủi ro",
        type="primary",
        use_container_width=True,
    )

    if evaluate:

        card_ref = str(st.session_state["card_ref"]).strip()
        merchant = str(st.session_state["merchant"]).strip()

        if not card_ref:
            st.warning("Vui lòng nhập mã tham chiếu thẻ.")

        elif not merchant:
            st.warning(
                "Vui lòng nhập tên cửa hàng hoặc đơn vị chấp nhận thanh toán."
            )

        elif hist.empty:
            st.warning(
                "Bảng lịch sử đang trống. "
                "Hãy bổ sung lịch sử giao dịch trước khi đánh giá."
            )

        elif hist[
            [
                "minutes_ago",
                "amt",
                "category",
                "merchant",
                "distance_km",
            ]
        ].isna().any().any():
            st.warning(
                "Lịch sử có dữ liệu chưa đầy đủ. "
                "Vui lòng kiểm tra các cột trước khi tiếp tục."
            )

        elif (hist["minutes_ago"] <= 0).any():
            st.warning(
                "Số phút trước giao dịch hiện tại phải lớn hơn 0."
            )

        elif (
            (hist["amt"] < 0).any()
            or (hist["distance_km"] < 0).any()
        ):
            st.warning(
                "Số tiền và khoảng cách trong lịch sử không được âm."
            )

        elif not hist["category"].isin(CATEGORIES).all():
            st.warning(
                "Lịch sử chứa danh mục không có trong bộ danh mục của mô hình."
            )

        else:
            t_new = pd.Timestamp(
                dt.datetime.combine(
                    st.session_state["new_date"],
                    st.session_state["new_time"],
                )
            )

            tx = {
                "time": t_new,
                "amt": st.session_state["amt"],
                "category": st.session_state["category"],
                "merchant": merchant,
                "distance_km": st.session_state["dist"],
            }

            try:
                with st.spinner(
                    "Đang tạo đặc trưng và đánh giá giao dịch..."
                ):
                    p, ifs, feats, n_hist = score_transaction(hist, tx)

                tier = tier_of(p)
                title, action, kind, advice = TIERS[tier]

                st.divider()
                st.subheader("3. Kết quả đánh giá")

                st.caption(
                    f"Mã tham chiếu: {card_ref} · "
                    f"Thời điểm: {t_new.strftime('%d/%m/%Y %H:%M')} · "
                    f"Cửa hàng: {merchant}"
                )

                getattr(st, kind)(f"### {title}")

                st.write(f"**Đề xuất xử lý:** {action}")
                st.write(advice)

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Xác suất gian lận",
                    f"{p:.2%}",
                )

                m2.metric(
                    "Điểm bất thường (Isolation Forest)",
                    f"{ifs:.3f}",
                )

                m3.metric(
                    "Giao dịch lịch sử được sử dụng",
                    f"{n_hist:,}",
                )

                st.markdown("#### Các dấu hiệu cần lưu ý")

                r = feats.iloc[0]
                reasons = explain(r)

                if reasons:
                    for reason in reasons:
                        st.markdown(f"- {reason}")
                else:
                    st.write(
                        "Không có dấu hiệu cảnh báo nào được phát hiện "
                        "theo các quy tắc giải thích hiện tại."
                    )

                st.caption(
                    "Các dấu hiệu trên được xác định bằng quy tắc kiểm tra "
                    "đặc trưng hành vi, không phải lời giải thích trực tiếp "
                    "cho dự đoán của XGBoost. Không có cảnh báo theo quy tắc "
                    "không đồng nghĩa với việc giao dịch chắc chắn an toàn."
                )

                if tier == 3:
                    st.info(
                        f"Theo ma trận chi phí của nhóm, chi phí cho một "
                        f"trường hợp dự đoán gian lận đúng là ${COST_TP:.2f}; "
                        f"tổn thất giả định khi bỏ sót một giao dịch gian lận "
                        f"là ${COST_FN:.2f}."
                    )

                with st.expander(
                    "Xem chi tiết 23 đặc trưng đầu vào mô hình"
                ):
                    feature_display = feats[FEATURE_COLS].T.copy()
                    feature_display.columns = ["Giá trị"]

                    st.dataframe(
                        feature_display,
                        use_container_width=True,
                    )

                if n_hist == 0:
                    st.warning(
                        "Không có giao dịch lịch sử hợp lệ được sử dụng. "
                        "Kết quả có thể thiếu thông tin về hành vi trước đó "
                        "của thẻ."
                    )

            except Exception as e:
                st.error(
                    "Không thể đánh giá giao dịch. "
                    "Vui lòng kiểm tra dữ liệu đầu vào và cấu hình mô hình."
                )
                st.exception(e)


# ================================================================
# TAB 2: PHÂN TÍCH FILE CSV
# ================================================================

with tab_csv:

    st.subheader("Phân tích danh sách giao dịch")

    st.write(
        "Tải lên file CSV để đánh giá rủi ro cho nhiều giao dịch. "
        "File cần có các cột đầu vào theo định dạng dữ liệu mà dự án hỗ trợ."
    )

    st.caption(
        "Các cột bắt buộc: "
        + ", ".join(f"`{c}`" for c in RAW_REQUIRED)
        + ". Nếu có cột `is_fraud`, ứng dụng sẽ dùng nhãn thực tế "
        "để tính các chỉ số đánh giá và tổng chi phí."
    )

    st.info(
        "Các đặc trưng hành vi được tính từ dữ liệu trong file. "
        "Để kết quả có ý nghĩa, file cần chứa lịch sử giao dịch phù hợp "
        "và dữ liệu thời gian, mã thẻ nhất quán."
    )

    up = st.file_uploader(
        "Chọn file CSV",
        type="csv",
    )

    if up is not None:

        try:
            raw = pd.read_csv(up)

            miss = [
                c for c in RAW_REQUIRED
                if c not in raw.columns
            ]

            if miss:
                st.error(f"Thiếu cột bắt buộc: {miss}")

            elif len(raw) > 300_000:
                st.error(
                    "File quá lớn cho bản online. "
                    "Giới hạn hiện tại là 300.000 dòng."
                )

            elif raw.empty:
                st.warning("File CSV không có dữ liệu.")

            else:
                with st.spinner(
                    "Đang tạo đặc trưng và chấm điểm giao dịch..."
                ):
                    feat = build_feature_frame(raw)
                    p, s = pipe.score_frame(feat)

                out = feat[
                    [
                        "trans_date_trans_time",
                        "cc_num",
                        "merchant",
                        "category",
                        "amt",
                    ]
                ].copy()

                if "is_fraud" in feat.columns:
                    out["is_fraud"] = feat["is_fraud"].to_numpy()

                out["probability"] = p
                out["if_score"] = s
                out["tier"] = [tier_of(x) for x in p]

                tc = out["tier"].value_counts()

                st.markdown("#### Phân bố mức rủi ro")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Tier 1 – Rủi ro thấp",
                    f"{int(tc.get(1, 0)):,}",
                )

                c2.metric(
                    "Tier 2 – Cần xác minh",
                    f"{int(tc.get(2, 0)):,}",
                )

                c3.metric(
                    "Tier 3 – Rủi ro cao",
                    f"{int(tc.get(3, 0)):,}",
                )

                if (
                    "is_fraud" in out.columns
                    and out["is_fraud"].nunique() == 2
                ):
                    y = out["is_fraud"].to_numpy(dtype=int)
                    pred = (p >= HIGH_THRESHOLD).astype(int)

                    tp = int(((pred == 1) & (y == 1)).sum())
                    fp = int(((pred == 1) & (y == 0)).sum())
                    fn = int(((pred == 0) & (y == 1)).sum())
                    tn = int(((pred == 0) & (y == 0)).sum())

                    st.markdown(
                        f"#### Đánh giá tại ngưỡng {HIGH_THRESHOLD:.2f}"
                    )

                    c1, c2, c3 = st.columns(3)

                    c1.metric("Recall", f"{tp / max(tp + fn, 1):.4f}")
                    c2.metric("Precision", f"{tp / max(tp + fp, 1):.4f}")
                    c3.metric(
                        "Tổng chi phí",
                        f"${total_cost(tp, fp, fn, tn):,.2f}",
                    )

                    st.caption(
                        f"TP = {tp:,} · FP = {fp:,} · "
                        f"FN = {fn:,} · TN = {tn:,}"
                    )

                st.markdown("#### Các giao dịch có xác suất gian lận cao")

                st.dataframe(
                    out.sort_values(
                        "probability",
                        ascending=False,
                    ).head(1000),
                    use_container_width=True,
                )

                st.download_button(
                    "Tải kết quả chấm điểm (CSV)",
                    data=out.to_csv(index=False).encode("utf-8-sig"),
                    file_name="ket_qua_cham_diem.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(
                "Không thể xử lý file. "
                "Hãy kiểm tra định dạng CSV và các cột dữ liệu."
            )
            st.exception(e)


# ================================================================
# TAB 3: KẾT QUẢ NGHIÊN CỨU CỦA NHÓM
# ================================================================

with tab_res:

    st.subheader("Quy trình và dữ liệu")

    st.write(
        "Dữ liệu `fraudTrain.csv` gồm 1.296.675 giao dịch, "
        "được chia theo thời gian theo tỷ lệ 70/15/15: "
        "Train 907.672 · Validation 194.501 · Test 194.502. "
        "Tập Train có 5.121 giao dịch gian lận, với "
        "scale_pos_weight = 176,25. "
        "Quy trình sử dụng 23 biến đầu vào, tiền xử lý bằng "
        "RobustScaler và One-Hot Encoding, sử dụng Isolation Forest "
        "để tạo `if_score`, sau đó đưa bộ đặc trưng E2 vào XGBoost."
    )

    st.subheader("So sánh mô hình trên Validation (threshold 0.50)")

    val = pd.DataFrame(
        [
            ["LR", "E1", 1202, 5010, 50, 188239, 0.1935, 0.9601, 0.3221, 0.9905, 0.4871, 33361.05],
            ["LR", "E2", 1188, 5128, 64, 188121, 0.1881, 0.9489, 0.3140, 0.9916, 0.5791, 35073.05],
            ["DT", "E1", 1036, 124, 216, 193125, 0.8931, 0.8275, 0.8590, 0.9134, 0.7401, 34805.05],
            ["DT", "E2", 1054, 125, 198, 193124, 0.8940, 0.8419, 0.8671, 0.9206, 0.7536, 33062.05],
            ["RF", "E1", 1146, 164, 106, 193085, 0.8748, 0.9153, 0.8946, 0.9979, 0.9615, 24255.05],
            ["RF", "E2", 1136, 151, 116, 193098, 0.8827, 0.9073, 0.8948, 0.9983, 0.9600, 25186.05],
            ["XGBoost", "E1", 1213, 262, 39, 192987, 0.8224, 0.9688, 0.8896, 0.9998, 0.9836, 18050.05],
            ["XGBoost", "E2", 1217, 255, 35, 192994, 0.8268, 0.9720, 0.8935, 0.9998, 0.9829, 17641.05],
            ["LightGBM", "E1", 1147, 8596, 105, 184653, 0.1177, 0.9161, 0.2086, 0.9367, 0.1069, 49454.05],
            ["LightGBM", "E2", 1176, 9308, 76, 183941, 0.1122, 0.9393, 0.2004, 0.9443, 0.1140, 48777.05],
        ],
        columns=[
            "Mô hình",
            "Bộ đặc trưng",
            "TP",
            "FP",
            "FN",
            "TN",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "PR-AUC",
            "Tổng chi phí ($)",
        ],
    )

    st.dataframe(
        val.style.format(
            {
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "PR-AUC": "{:.4f}",
                "Tổng chi phí ($)": "${:,.2f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Theo bảng kết quả Validation được cung cấp, XGBoost + E2 "
        "có tổng chi phí thấp nhất trong các cấu hình được so sánh."
    )

    # ------------------------------------------------------------
    # MA TRẬN CHI PHÍ
    # ------------------------------------------------------------

    st.subheader("Ma trận chi phí giả định (USD)")

    cost_matrix = pd.DataFrame(
        [
            [COST_TP, COST_FN],
            [COST_FP, COST_TN],
        ],
        index=[
            "Thực tế: Fraud",
            "Thực tế: Normal",
        ],
        columns=[
            "Dự đoán: Fraud",
            "Dự đoán: Normal",
        ],
    )

    st.dataframe(
        cost_matrix.style.format("${:,.2f}"),
        use_container_width=True,
    )

    st.caption(
        f"Tổn thất bỏ sót một giao dịch gian lận = ${L_FN:g}; "
        f"chi phí xác minh một cảnh báo = ${C_REVIEW:g}; "
        f"chi phí chấm điểm = ${C_MODEL:g}/giao dịch."
    )

    # ------------------------------------------------------------
    # CHỌN THRESHOLD
    # ------------------------------------------------------------

    st.subheader(
        "Chọn threshold trên Validation "
        "(10 cấu hình chi phí thấp nhất)"
    )

    thr = pd.DataFrame(
        [
            [0.12, 1236, 685, 16, 192564, 0.6434, 0.9872, 0.7791, 17088.05],
            [0.11, 1237, 718, 15, 192531, 0.6327, 0.9880, 0.7714, 17090.05],
            [0.10, 1238, 756, 14, 192493, 0.6209, 0.9888, 0.7628, 17107.05],
            [0.13, 1234, 654, 18, 192595, 0.6536, 0.9856, 0.7860, 17189.05],
            [0.15, 1232, 597, 20, 192652, 0.6736, 0.9840, 0.7997, 17212.05],
            [0.16, 1231, 567, 21, 192682, 0.6846, 0.9832, 0.8072, 17219.05],
            [0.08, 1240, 864, 12, 192385, 0.5894, 0.9904, 0.7390, 17237.05],
            [0.25, 1227, 446, 25, 192803, 0.7334, 0.9800, 0.8390, 17244.05],
            [0.09, 1238, 803, 14, 192446, 0.6066, 0.9888, 0.7519, 17248.05],
            [0.07, 1242, 935, 10, 192314, 0.5705, 0.9920, 0.7244, 17256.05],
        ],
        columns=[
            "Threshold",
            "TP",
            "FP",
            "FN",
            "TN",
            "Precision",
            "Recall",
            "F1",
            "Tổng chi phí ($)",
        ],
    )

    st.dataframe(
        thr.style.format(
            {
                "Threshold": "{:.2f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "Tổng chi phí ($)": "${:,.2f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    # ------------------------------------------------------------
    # KẾT QUẢ TEST
    # ------------------------------------------------------------

    st.subheader("Kết quả cuối trên tập Test (194.502 giao dịch)")

    tst = pd.DataFrame(
        [
            ["Mặc định 0.50", 1096, 262, 37, 193107, 0.8071, 0.9673, 0.8800, 17499.10],
            ["Đã chốt 0.12", 1116, 704, 17, 192665, 0.6132, 0.9850, 0.7558, 16885.10],
        ],
        columns=[
            "Cấu hình",
            "TP",
            "FP",
            "FN",
            "TN",
            "Precision",
            "Recall",
            "F1",
            "Tổng chi phí ($)",
        ],
    )

    st.dataframe(
        tst.style.format(
            {
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "Tổng chi phí ($)": "${:,.2f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.success(
        "Theo số liệu được cung cấp, threshold 0.12 giảm tổng chi phí "
        "$614,00 so với threshold 0.50, bắt thêm 20 giao dịch gian lận "
        "và đồng thời tạo thêm 442 cảnh báo nhầm."
    )

    # ------------------------------------------------------------
    # DECISION ENGINE
    # ------------------------------------------------------------

    st.subheader("Phân bố Tier trên tập Test")

    decision_df = pd.DataFrame(
        {
            "Mức rủi ro": [
                "Tier 1 – Phê duyệt tự động (p < 0.05)",
                "Tier 2 – Xem xét thủ công (0.05 ≤ p < 0.12)",
                "Tier 3 – Rủi ro cao (p ≥ 0.12)",
            ],
            "Số giao dịch": [
                192253,
                429,
                1820,
            ],
            "Tỷ lệ (%)": [
                98.84,
                0.22,
                0.94,
            ],
        }
    )

    st.dataframe(
        decision_df,
        hide_index=True,
        use_container_width=True,
    )

    # ------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------------

    met = load_json("metrics.json")

    if met and met.get("importance"):
        st.subheader("Mức quan trọng của đặc trưng (XGBoost, gain)")

        importance = (
            pd.DataFrame(met["importance"])
            .sort_values("gain", ascending=False)
            .head(15)
            .set_index("feature")
        )

        st.bar_chart(importance["gain"])
