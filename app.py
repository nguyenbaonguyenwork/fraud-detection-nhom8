"""
FraudGuard - Ứng dụng hỗ trợ đánh giá rủi ro giao dịch thẻ.

Chạy:
    streamlit run app.py

Yêu cầu:
    artifacts/pipeline.joblib
    features.py
    model_pipeline.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import RAW_REQUIRED, build_feature_frame
from model_pipeline import FINAL_THRESHOLD, LOW_THRESHOLD, decide


# ============================================================
# 1. CẤU HÌNH ỨNG DỤNG
# ============================================================

ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "pipeline.joblib")

st.set_page_config(
    page_title="FraudGuard | Đánh giá giao dịch",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1450px;
            padding-top: 1.8rem;
            padding-bottom: 3rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid #E5E7EB;
            border-radius: 10px;
            padding: 14px 16px;
            background-color: #FFFFFF;
        }

        div[data-testid="stMetricLabel"] {
            color: #64748B;
        }

        div[data-testid="stMetricValue"] {
            color: #0F172A;
        }

        .section-note {
            color: #64748B;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. TẢI MÔ HÌNH
# ============================================================

@st.cache_resource
def load_pipeline(path):
    return joblib.load(path)


if not os.path.isfile(MODEL_PATH):
    st.error(
        "Không tìm thấy mô hình đã huấn luyện tại "
        f"`{MODEL_PATH}`. Hãy kiểm tra thư mục artifacts."
    )
    st.stop()

try:
    pipeline = load_pipeline(MODEL_PATH)
except Exception as error:
    st.error("Không thể tải mô hình. Hãy kiểm tra file pipeline.joblib.")
    with st.expander("Chi tiết lỗi"):
        st.code(str(error))
    st.stop()


# ============================================================
# 3. CÁC HÀM XỬ LÝ
# ============================================================

DECISION_APPROVE = "Phê duyệt tự động (Approve)"
DECISION_REVIEW = "Xem xét thủ công (Manual Review)"
DECISION_BLOCK = "Chặn / Từ chối (Block)"


def format_decision(decision):
    """Chuẩn hóa tên quyết định để hiển thị thống nhất."""

    value = str(decision).strip().lower()

    if "approve" in value or "phê duyệt" in value:
        return DECISION_APPROVE

    if "review" in value or "xem xét" in value:
        return DECISION_REVIEW

    if "block" in value or "chặn" in value or "từ chối" in value:
        return DECISION_BLOCK

    return str(decision)


def classify_transactions(probabilities, low, high):
    """Sử dụng hàm quyết định đã được định nghĩa trong dự án."""

    decisions = decide(probabilities, low, high)

    if isinstance(decisions, str):
        decisions = [decisions] * len(probabilities)

    return [format_decision(value) for value in decisions]


def safe_value(row, feature, default=None):
    """Lấy giá trị đặc trưng an toàn khi cột không tồn tại."""

    if feature not in row.index:
        return default

    value = row[feature]

    if pd.isna(value):
        return default

    return value


def is_true(value):
    """Chuyển giá trị đặc trưng về kiểu boolean."""

    if value is None:
        return False

    if isinstance(value, str):
        return value.strip().lower() in {
            "true", "1", "yes", "y"
        }

    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def explain_transaction(row):
    """
    Tạo phần giải thích từ các đặc trưng hành vi đã có.

    Đây là giải thích theo quy tắc, không phải kết quả SHAP/LIME.
    Chỉ hiển thị những dấu hiệu mà dữ liệu đặc trưng thực sự hỗ trợ.
    """

    explanations = []

    # Số tiền bất thường so với lịch sử thẻ
    amount_zscore = safe_value(row, "customer_amt_zscore")

    if amount_zscore is not None:
        try:
            if float(amount_zscore) >= 3:
                explanations.append(
                    "Số tiền giao dịch cao bất thường so với lịch sử "
                    "chi tiêu của thẻ."
                )
        except (TypeError, ValueError):
            pass

    # Số tiền so với giao dịch trong cùng danh mục
    category_ratio = safe_value(row, "amt_ratio_cat_max")

    if category_ratio is not None:
        try:
            if float(category_ratio) > 1.5:
                explanations.append(
                    "Số tiền cao hơn đáng kể so với mức tham chiếu "
                    "của danh mục giao dịch."
                )
        except (TypeError, ValueError):
            pass

    # Tần suất giao dịch
    recent_count = safe_value(row, "transactions_last_5min")

    if recent_count is not None:
        try:
            if float(recent_count) >= 3:
                explanations.append(
                    "Có nhiều giao dịch trong khoảng thời gian ngắn. "
                    "Nên kiểm tra xem đây có phải chuỗi giao dịch bất thường."
                )
        except (TypeError, ValueError):
            pass

    # Nhiều giao dịch trực tuyến nhỏ
    small_online_count = safe_value(row, "n_small_online_24h")

    if small_online_count is not None:
        try:
            if float(small_online_count) >= 3:
                explanations.append(
                    "Ghi nhận nhiều giao dịch nhỏ trực tuyến trong 24 giờ. "
                    "Nên kiểm tra khả năng thử thẻ hoặc giao dịch liên tiếp."
                )
        except (TypeError, ValueError):
            pass

    # Cửa hàng mới
    if is_true(safe_value(row, "is_new_merchant", False)):
        explanations.append(
            "Cửa hàng chưa xuất hiện trong lịch sử giao dịch được sử dụng "
            "để tạo đặc trưng."
        )

    # Khoảng cách bất thường
    distance_zscore = safe_value(row, "distance_zscore")

    if distance_zscore is not None:
        try:
            if float(distance_zscore) >= 2:
                explanations.append(
                    "Khoảng cách giao dịch khác biệt đáng kể so với "
                    "mức tham chiếu được tính từ lịch sử."
                )
        except (TypeError, ValueError):
            pass

    # Số tiền tăng đột biến so với giao dịch trước
    amount_jump = safe_value(row, "amt_jump_ratio")

    if amount_jump is not None:
        try:
            if float(amount_jump) >= 5:
                explanations.append(
                    "Số tiền cao gấp nhiều lần giao dịch liền trước."
                )
        except (TypeError, ValueError):
            pass

    # Thời điểm giao dịch
    if is_true(safe_value(row, "is_night", False)):
        explanations.append(
            "Giao dịch diễn ra trong khung giờ ban đêm."
        )

    # Giao dịch trực tuyến
    if is_true(safe_value(row, "is_online_cat", False)):
        explanations.append(
            "Giao dịch thuộc nhóm danh mục trực tuyến."
        )

    # Thiếu lịch sử
    if (
        "has_customer_history" in row.index
        and not is_true(safe_value(row, "has_customer_history", False))
    ):
        explanations.append(
            "Lịch sử thẻ chưa đủ để đánh giá đầy đủ hành vi chi tiêu "
            "trước đó."
        )

    return explanations


def decision_description(decision):
    if decision == DECISION_APPROVE:
        return (
            "Điểm rủi ro nằm dưới ngưỡng xem xét. Giao dịch có thể được "
            "chuyển sang quy trình phê duyệt thông thường."
        )

    if decision == DECISION_REVIEW:
        return (
            "Giao dịch nằm trong vùng cần xem xét. Nên kiểm tra thêm thông "
            "tin trước khi đưa ra quyết định cuối cùng."
        )

    return (
        "Điểm rủi ro đạt ngưỡng chặn. Giao dịch cần được ưu tiên kiểm tra "
        "và xử lý theo chính sách của đơn vị phát hành thẻ."
    )


def prepare_results(raw, low, high):
    """Tạo đặc trưng, chấm điểm và tạo bảng kết quả."""

    feature_frame = build_feature_frame(raw)

    probabilities, anomaly_scores = pipeline.score_frame(feature_frame)

    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    anomaly_scores = np.asarray(anomaly_scores).reshape(-1)

    if len(probabilities) != len(raw):
        raise ValueError(
            "Số điểm dự đoán không khớp số giao dịch đầu vào."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Mô hình trả về xác suất không hợp lệ. "
            "Hãy kiểm tra dữ liệu đầu vào."
        )

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(
            "Mô hình trả về giá trị ngoài khoảng xác suất từ 0 đến 1."
        )

    decisions = classify_transactions(
        probabilities,
        low,
        high,
    )

    # Chỉ đưa các trường cần thiết lên bảng kết quả.
    preferred_columns = [
        "trans_date_trans_time",
        "cc_num",
        "merchant",
        "category",
        "amt",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in raw.columns
    ]

    results = raw[available_columns].copy().reset_index(drop=True)

    results.insert(
        0,
        "Mã dòng",
        np.arange(1, len(results) + 1),
    )

    results["Xác suất gian lận"] = probabilities
    results["Quyết định"] = decisions

    # Điểm IF giữ trong dữ liệu kết quả để phục vụ kiểm tra kỹ thuật,
    # nhưng không hiển thị mặc định trong bảng.
    results["_if_score"] = anomaly_scores

    # Lưu vị trí tương ứng với feature_frame để giải thích chi tiết.
    results["_source_index"] = np.arange(len(feature_frame))

    return results, feature_frame


# ============================================================
# 4. THANH ĐIỀU HƯỚNG
# ============================================================

with st.sidebar:
    st.title("FraudGuard")
    st.caption("Hệ thống hỗ trợ đánh giá rủi ro giao dịch")

    st.divider()

    st.subheader("Ngưỡng quyết định")

    low_threshold = st.number_input(
        "Ngưỡng duyệt tự động",
        min_value=0.0,
        max_value=0.98,
        value=float(LOW_THRESHOLD),
        step=0.01,
        format="%.2f",
        help="Xác suất thấp hơn ngưỡng này thuộc nhóm rủi ro thấp.",
    )

    high_threshold = st.number_input(
        "Ngưỡng rủi ro cao",
        min_value=0.01,
        max_value=1.0,
        value=float(FINAL_THRESHOLD),
        step=0.01,
        format="%.2f",
        help="Xác suất đạt hoặc vượt ngưỡng này thuộc nhóm rủi ro cao.",
    )

    if low_threshold >= high_threshold:
        st.error(
            "Ngưỡng duyệt tự động phải nhỏ hơn ngưỡng rủi ro cao."
        )

    st.divider()

    st.caption(
        "Kết quả là khuyến nghị từ mô hình, không thay thế quy trình "
        "xác minh và chính sách xử lý giao dịch."
    )


# ============================================================
# 5. GIAO DIỆN CHÍNH
# ============================================================

st.title("Đánh giá rủi ro giao dịch thẻ")

st.write(
    "Tải lên dữ liệu giao dịch để hệ thống đánh giá mức độ rủi ro, "
    "xác định các giao dịch cần ưu tiên kiểm tra và cung cấp giải thích "
    "cho từng kết quả."
)

st.divider()

uploaded_file = st.file_uploader(
    "Chọn file giao dịch",
    type=["csv"],
    help=(
        "Sử dụng file dữ liệu thô có cấu trúc cột phù hợp với dữ liệu "
        "đã dùng để xây dựng mô hình."
    ),
)

if uploaded_file is None:
    st.subheader("Bắt đầu đánh giá")

    st.write(
        "Chọn file CSV để bắt đầu. Hệ thống sẽ kiểm tra các cột bắt buộc "
        "trước khi chấm điểm."
    )

    with st.expander("Yêu cầu đối với file dữ liệu"):
        st.write(
            "File cần có đầy đủ các trường đầu vào mà pipeline yêu cầu. "
            "Tên cột phải khớp với dữ liệu được dùng trong dự án."
        )

        st.code(", ".join(RAW_REQUIRED))

    st.stop()


# ============================================================
# 6. ĐỌC VÀ KIỂM TRA FILE
# ============================================================

try:
    raw = pd.read_csv(uploaded_file)
except Exception as error:
    st.error("Không thể đọc file CSV. Hãy kiểm tra định dạng file.")

    with st.expander("Chi tiết lỗi"):
        st.code(str(error))

    st.stop()


if raw.empty:
    st.error("File không có giao dịch nào.")
    st.stop()


missing_columns = [
    column
    for column in RAW_REQUIRED
    if column not in raw.columns
]

if missing_columns:
    st.error(
        "File dữ liệu thiếu các cột bắt buộc: "
        + ", ".join(missing_columns)
    )

    st.write("Các cột cần có:")
    st.code(", ".join(RAW_REQUIRED))

    st.stop()


if len(raw) > 300_000:
    st.error(
        "File có hơn 300.000 giao dịch. Hãy chia nhỏ file để tránh "
        "sử dụng quá nhiều bộ nhớ trong một lần xử lý."
    )
    st.stop()


# ============================================================
# 7. TÓM TẮT DỮ LIỆU ĐẦU VÀO
# ============================================================

st.subheader("Dữ liệu đầu vào")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Số giao dịch",
    f"{len(raw):,}",
)

col2.metric(
    "Số trường dữ liệu",
    f"{len(raw.columns):,}",
)

if "cc_num" in raw.columns:
    col3.metric(
        "Số thẻ khác nhau",
        f"{raw['cc_num'].nunique(dropna=True):,}",
    )
else:
    col3.metric("Số thẻ khác nhau", "Không có dữ liệu")


with st.expander("Xem trước dữ liệu đã tải lên"):
    # Không hiển thị các thông tin nhận dạng cá nhân không cần thiết.
    preview_columns = [
        column
        for column in [
            "trans_date_trans_time",
            "merchant",
            "category",
            "amt",
            "city",
            "state",
        ]
        if column in raw.columns
    ]

    if preview_columns:
        st.dataframe(
            raw[preview_columns].head(10),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.dataframe(
            raw.head(10),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 8. CHẤM ĐIỂM GIAO DỊCH
# ============================================================

st.divider()

if low_threshold >= high_threshold:
    st.warning(
        "Hãy điều chỉnh lại ngưỡng quyết định trong thanh bên trước "
        "khi chấm điểm."
    )
    st.stop()


if st.button(
    "Đánh giá giao dịch",
    type="primary",
    use_container_width=True,
):
    try:
        with st.spinner(
            "Đang tạo đặc trưng hành vi và đánh giá giao dịch..."
        ):
            results, feature_frame = prepare_results(
                raw,
                low_threshold,
                high_threshold,
            )

        st.session_state["fraudguard_results"] = results
        st.session_state["fraudguard_features"] = feature_frame
        st.session_state["fraudguard_thresholds"] = (
            low_threshold,
            high_threshold,
        )

    except Exception as error:
        st.error(
            "Không thể đánh giá dữ liệu. Hãy kiểm tra định dạng cột, "
            "kiểu dữ liệu và tính tương thích với mô hình đã huấn luyện."
        )

        with st.expander("Chi tiết lỗi kỹ thuật"):
            st.code(str(error))


# ============================================================
# 9. KẾT QUẢ ĐÁNH GIÁ
# ============================================================

results = st.session_state.get("fraudguard_results")
feature_frame = st.session_state.get("fraudguard_features")
saved_thresholds = st.session_state.get("fraudguard_thresholds")

if results is None or feature_frame is None:
    st.info(
        "Sau khi kiểm tra file, nhấn **Đánh giá giao dịch** để xem kết quả."
    )
    st.stop()


# Nếu ngưỡng đã thay đổi, yêu cầu đánh giá lại để tránh kết quả không đồng bộ.
if saved_thresholds != (low_threshold, high_threshold):
    st.warning(
        "Ngưỡng quyết định đã thay đổi. Hãy nhấn **Đánh giá giao dịch** "
        "để cập nhật kết quả theo ngưỡng mới."
    )
    st.stop()


st.divider()
st.subheader("Kết quả đánh giá")


# ------------------------------------------------------------
# 9.1. TỔNG QUAN QUYẾT ĐỊNH
# ------------------------------------------------------------

decision_counts = results["Quyết định"].value_counts()

approve_count = int(decision_counts.get(DECISION_APPROVE, 0))
review_count = int(decision_counts.get(DECISION_REVIEW, 0))
block_count = int(decision_counts.get(DECISION_BLOCK, 0))

m1, m2, m3 = st.columns(3)

m1.metric(
    "Rủi ro thấp",
    f"{approve_count:,}",
)

m2.metric(
    "Cần xem xét",
    f"{review_count:,}",
)

m3.metric(
    "Rủi ro cao",
    f"{block_count:,}",
)


# ------------------------------------------------------------
# 9.2. LỌC VÀ TRA CỨU
# ------------------------------------------------------------

st.subheader("Danh sách giao dịch")

filter_col1, filter_col2 = st.columns([1, 2])

with filter_col1:
    selected_decisions = st.multiselect(
        "Lọc theo quyết định",
        options=[
            DECISION_APPROVE,
            DECISION_REVIEW,
            DECISION_BLOCK,
        ],
        default=[
            DECISION_REVIEW,
            DECISION_BLOCK,
        ],
    )

with filter_col2:
    search_text = st.text_input(
        "Tìm kiếm",
        placeholder="Nhập tên cửa hàng, danh mục hoặc mã thẻ...",
    )


filtered = results[
    results["Quyết định"].isin(selected_decisions)
].copy()


if search_text.strip():
    searchable_columns = [
        column
        for column in [
            "merchant",
            "category",
            "cc_num",
        ]
        if column in filtered.columns
    ]

    if searchable_columns:
        search_mask = pd.Series(
            False,
            index=filtered.index,
        )

        for column in searchable_columns:
            search_mask |= (
                filtered[column]
                .astype(str)
                .str.contains(
                    search_text.strip(),
                    case=False,
                    na=False,
                    regex=False,
                )
            )

        filtered = filtered[search_mask]


filtered = filtered.sort_values(
    "Xác suất gian lận",
    ascending=False,
)


display_columns = [
    column
    for column in [
        "Mã dòng",
        "trans_date_trans_time",
        "merchant",
        "category",
        "amt",
        "Xác suất gian lận",
        "Quyết định",
    ]
    if column in filtered.columns
]

display_table = filtered[display_columns].copy()

if "Xác suất gian lận" in display_table.columns:
    display_table["Xác suất gian lận"] = (
        display_table["Xác suất gian lận"].map(
            lambda value: f"{value:.2%}"
        )
    )

st.caption(
    f"Đang hiển thị {len(filtered):,} / {len(results):,} giao dịch."
)

st.dataframe(
    display_table,
    use_container_width=True,
    hide_index=True,
    height=380,
)


# ------------------------------------------------------------
# 9.3. CHỌN GIAO DỊCH ĐỂ XEM GIẢI THÍCH
# ------------------------------------------------------------

st.divider()
st.subheader("Chi tiết giao dịch")

if filtered.empty:
    st.info(
        "Không có giao dịch phù hợp với bộ lọc hiện tại."
    )

else:
    selected_row_id = st.selectbox(
        "Chọn giao dịch cần xem",
        options=filtered["Mã dòng"].tolist(),
        format_func=lambda row_id: (
            f"Giao dịch #{row_id} · "
            f"{filtered.loc[filtered['Mã dòng'] == row_id, 'Quyết định'].iloc[0]} · "
            f"{filtered.loc[filtered['Mã dòng'] == row_id, 'Xác suất gian lận'].iloc[0]:.2%}"
        ),
    )

    result_row = results.loc[
        results["Mã dòng"] == selected_row_id
    ].iloc[0]

    feature_index = int(result_row["_source_index"])
    feature_row = feature_frame.iloc[feature_index]

    probability = float(result_row["Xác suất gian lận"])
    decision = result_row["Quyết định"]

    detail_col1, detail_col2 = st.columns([1, 2])

    with detail_col1:
        st.markdown("**Xác suất gian lận ước tính**")

        st.title(f"{probability:.2%}")

        st.progress(
            min(max(probability, 0.0), 1.0)
        )

        st.markdown("**Quyết định**")

        if decision == DECISION_APPROVE:
            st.success(decision)
        elif decision == DECISION_REVIEW:
            st.warning(decision)
        else:
            st.error(decision)

    with detail_col2:
        st.markdown("**Thông tin giao dịch**")

        detail_fields = {
            "Thời gian": "trans_date_trans_time",
            "Cửa hàng": "merchant",
            "Danh mục": "category",
            "Số tiền": "amt",
        }

        for label, column in detail_fields.items():
            if column not in result_row.index:
                continue

            value = result_row[column]

            if column == "amt":
                try:
                    value = f"${float(value):,.2f}"
                except (TypeError, ValueError):
                    pass

            st.write(f"**{label}:** {value}")

        st.markdown("**Diễn giải quyết định**")
        st.write(decision_description(decision))

    # --------------------------------------------------------
    # GIẢI THÍCH CÁC DẤU HIỆU
    # --------------------------------------------------------

    st.markdown("#### Dấu hiệu hành vi cần chú ý")

    explanations = explain_transaction(feature_row)

    if explanations:
        for explanation in explanations:
            st.markdown(f"- {explanation}")
    else:
        st.write(
            "Không có dấu hiệu nổi bật được nhận diện bởi các quy tắc "
            "giải thích hiện có. Điều này không có nghĩa là giao dịch "
            "chắc chắn hợp lệ."
        )

    st.caption(
        "Các dấu hiệu trên được tổng hợp từ những đặc trưng hành vi "
        "có sẵn. Chúng không phải lời giải thích SHAP/LIME cho dự đoán "
        "của mô hình và không chứng minh rằng giao dịch là gian lận."
    )


# ============================================================
# 10. TẢI KẾT QUẢ
# ============================================================

st.divider()
st.subheader("Xuất kết quả")

export_columns = [
    column
    for column in results.columns
    if not column.startswith("_")
]

export_data = results[export_columns].copy()

export_data["Xác suất gian lận"] = (
    export_data["Xác suất gian lận"].round(6)
)

csv_data = export_data.to_csv(
    index=False,
).encode("utf-8-sig")

st.download_button(
    label="Tải kết quả đánh giá (CSV)",
    data=csv_data,
    file_name="fraudguard_results.csv",
    mime="text/csv",
    use_container_width=True,
)

st.caption(
    "FraudGuard · Nhóm 8 · Công cụ hỗ trợ đánh giá rủi ro giao dịch."
)
