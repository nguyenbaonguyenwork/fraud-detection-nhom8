import datetime as dt
import os

import joblib
import pandas as pd
import streamlit as st

from features import FEATURE_COLS, build_feature_frame
from model_pipeline import FraudPipeline  # Cần để joblib nạp lớp pipeline tùy chỉnh


# ============================================================
# 1. CẤU HÌNH
# ============================================================

st.set_page_config(
    page_title="Đánh giá rủi ro giao dịch",
    page_icon="💳",
    layout="wide",
)

LOW_THRESHOLD = 0.05
HIGH_THRESHOLD = 0.12

TIER_INFO = {
    1: {
        "title": "Rủi ro thấp",
        "action": "Đề xuất phê duyệt tự động",
        "description": (
            "Xác suất gian lận thấp hơn 5%. Theo ngưỡng đang thiết lập, "
            "giao dịch thuộc nhóm rủi ro thấp. Đây không phải cam kết rằng "
            "giao dịch chắc chắn hợp lệ."
        ),
    },
    2: {
        "title": "Cần xác minh",
        "action": "Đề xuất kiểm tra thủ công",
        "description": (
            "Xác suất gian lận nằm trong khoảng từ 5% đến dưới 12%. "
            "Nên kiểm tra thêm thông tin trước khi phê duyệt hoặc từ chối."
        ),
    },
    3: {
        "title": "Rủi ro cao",
        "action": "Đề xuất tạm giữ hoặc từ chối theo chính sách",
        "description": (
            "Xác suất gian lận từ 12% trở lên. Giao dịch cần được xem xét "
            "ưu tiên; không nên tự động kết luận gian lận chỉ dựa vào điểm số."
        ),
    },
}


def tier_of(probability: float) -> int:
    if probability < LOW_THRESHOLD:
        return 1
    if probability < HIGH_THRESHOLD:
        return 2
    return 3


# ============================================================
# 2. TẢI MÔ HÌNH ĐÃ HUẤN LUYỆN
# ============================================================

MODEL_PATHS = (
    os.path.join("artifacts", "pipeline.joblib"),
    "pipeline.joblib",
)

MODEL_PATH = next(
    (path for path in MODEL_PATHS if os.path.isfile(path)),
    None,
)

if MODEL_PATH is None:
    st.error(
        "Không tìm thấy `pipeline.joblib`. Hãy đặt file trong thư mục "
        "`artifacts/` hoặc cùng thư mục với `app.py`."
    )
    st.stop()


@st.cache_resource
def load_pipeline(path: str):
    return joblib.load(path)


pipe = load_pipeline(MODEL_PATH)

CATEGORIES = sorted(
    value.replace("category_", "", 1)
    for value in getattr(pipe, "ohe_names", [])
    if str(value).startswith("category_")
)

if not CATEGORIES:
    st.error(
        "Không đọc được danh mục giao dịch từ pipeline. "
        "Kiểm tra thuộc tính `ohe_names` trong mô hình đã lưu."
    )
    st.stop()


# ============================================================
# 3. TẠO ĐẶC TRƯNG VÀ CHẤM ĐIỂM
# ============================================================

def score_transaction(history: pd.DataFrame, transaction: dict):
    """
    Kết hợp lịch sử do người dùng nhập với giao dịch mới,
    tạo đặc trưng bằng build_feature_frame() và chấm điểm bằng pipeline.
    """

    current_time = pd.Timestamp(transaction["time"])
    rows = []

    required_history_columns = [
        "minutes_ago",
        "amt",
        "category",
        "merchant",
        "distance_km",
    ]

    if history is not None and not history.empty:
        for row in history[required_history_columns].itertuples(
            index=False,
            name=None,
        ):
            minutes_ago, amount, category, merchant, distance = row

            if any(pd.isna(value) for value in row):
                continue

            minutes_ago = float(minutes_ago)

            if minutes_ago <= 0:
                continue

            rows.append(
                {
                    "trans_date_trans_time": (
                        current_time - pd.Timedelta(minutes=minutes_ago)
                    ),
                    "cc_num": 1,
                    "amt": float(amount),
                    "category": str(category),
                    "merchant": str(merchant),
                    "distance_km": float(distance),
                }
            )

    rows.append(
        {
            "trans_date_trans_time": current_time,
            "cc_num": 1,
            "amt": float(transaction["amt"]),
            "category": str(transaction["category"]),
            "merchant": str(transaction["merchant"]),
            "distance_km": float(transaction["distance_km"]),
        }
    )

    raw = pd.DataFrame(rows)
    features = build_feature_frame(raw)

    # Dòng cuối là giao dịch hiện tại sau khi hàm tạo đặc trưng xử lý dữ liệu.
    scored_row = features.iloc[[-1]].copy()

    probabilities, anomaly_scores = pipe.score_frame(scored_row)

    # Chỉ sử dụng các đặc trưng mà pipeline đã được huấn luyện với chúng.
    feature_row = scored_row[FEATURE_COLS].reset_index(drop=True)

    return (
        float(probabilities[0]),
        float(anomaly_scores[0]),
        feature_row,
        len(rows) - 1,
    )


# ============================================================
# 4. GIẢI THÍCH CÁC DẤU HIỆU HÀNH VI
# ============================================================

def explain_behavior(row: pd.Series):
    """
    Giải thích bằng các quy tắc kiểm tra đặc trưng.
    Đây không phải SHAP/LIME và không phải lời giải thích trực tiếp
    cho từng dự đoán của XGBoost.
    """

    reasons = []

    def get_value(name, default=0):
        value = row.get(name, default)

        if pd.isna(value):
            return default

        return value

    # Dấu hiệu 1: Giao dịch ban đêm
    if bool(get_value("is_night", False)):
        reasons.append(
            "Giao dịch diễn ra ban đêm (22:00–06:59), thời điểm cần "
            "được xem xét cùng các thông tin khác."
        )

    # Dấu hiệu 2: Giao dịch trực tuyến
    if bool(get_value("is_online_cat", False)):
        reasons.append(
            "Giao dịch thuộc nhóm mua sắm trực tuyến "
            "(shopping_net/misc_net)."
        )

    # Dấu hiệu 3: Số tiền bất thường so với lịch sử khách hàng
    if (
        bool(get_value("has_customer_history", False))
        and float(get_value("customer_amt_zscore", 0)) >= 3
    ):
        reasons.append(
            "Số tiền cao bất thường so với lịch sử chi tiêu của thẻ "
            f"(z-score {float(get_value('customer_amt_zscore')):.1f})."
        )

    # Dấu hiệu 4: Số tiền bất thường trong danh mục
    category_ratio = float(get_value("amt_ratio_cat_max", 0))

    if category_ratio > 1.5:
        reasons.append(
            "Số tiền cao hơn đáng kể so với mức lớn nhất từng ghi nhận "
            f"ở danh mục này (khoảng {category_ratio:.1f} lần)."
        )

    # Dấu hiệu 5: Nhiều giao dịch trong thời gian ngắn
    recent_count = float(get_value("transactions_last_5min", 0))

    if recent_count >= 3:
        reasons.append(
            f"Có {int(recent_count)} giao dịch trong khoảng 5 phút gần nhất."
        )

    # Dấu hiệu 6: Nhiều giao dịch trực tuyến nhỏ
    small_online_count = float(get_value("n_small_online_24h", 0))

    if small_online_count >= 3:
        reasons.append(
            f"Có {int(small_online_count)} giao dịch nhỏ trực tuyến trong "
            "24 giờ; nên kiểm tra khả năng thử thẻ."
        )

    # Dấu hiệu 7: Cửa hàng mới
    if bool(get_value("is_new_merchant", False)):
        reasons.append(
            "Cửa hàng chưa xuất hiện trong lịch sử giao dịch được cung cấp."
        )

    # Dấu hiệu 8: Khoảng cách bất thường
    distance_z = float(get_value("distance_zscore", 0))

    if distance_z >= 2:
        reasons.append(
            "Khoảng cách đến cửa hàng khác biệt đáng kể so với lịch sử "
            f"(z-score {distance_z:.1f})."
        )

    # Dấu hiệu 9: Số tiền tăng đột biến
    amount_jump = float(get_value("amt_jump_ratio", 0))

    if amount_jump >= 5:
        reasons.append(
            "Số tiền cao gấp "
            f"{amount_jump:.1f} lần giao dịch liền trước."
        )

    # Dấu hiệu 10: Thiếu lịch sử
    if not bool(get_value("has_customer_history", False)):
        reasons.append(
            "Chưa có đủ lịch sử hợp lệ để đánh giá đầy đủ hành vi trước đó."
        )

    return reasons


def make_interpretation(probability: float, tier: int) -> str:
    percentage = probability * 100

    if tier == 1:
        return (
            f"Mô hình ước tính xác suất gian lận là {percentage:.2f}%, "
            "thấp hơn ngưỡng 5%. Giao dịch được xếp vào nhóm rủi ro thấp "
            "theo quy tắc hiện tại."
        )

    if tier == 2:
        return (
            f"Mô hình ước tính xác suất gian lận là {percentage:.2f}%, "
            "nằm trong vùng từ 5% đến dưới 12%. Điểm số chưa đủ để đưa ra "
            "quyết định tự động chắc chắn; nên xác minh thêm."
        )

    return (
        f"Mô hình ước tính xác suất gian lận là {percentage:.2f}%, "
        "đạt hoặc vượt ngưỡng 12%. Giao dịch được đưa vào nhóm cần ưu tiên "
        "kiểm tra, nhưng vẫn cần đối chiếu thông tin thực tế."
    )


# ============================================================
# 5. DỮ LIỆU MẪU TÙY CHỌN
# ============================================================

SAMPLE_HISTORY = pd.DataFrame(
    [
        [300, 42.0, "grocery_pos", "Cửa hàng A", 15.0],
        [1500, 65.0, "gas_transport", "Trạm xăng B", 40.0],
        [2900, 38.0, "food_dining", "Nhà hàng C", 22.0],
        [4400, 80.0, "grocery_pos", "Cửa hàng A", 18.0],
        [6000, 55.0, "home", "Cửa hàng D", 60.0],
    ],
    columns=[
        "minutes_ago",
        "amt",
        "category",
        "merchant",
        "distance_km",
    ],
)


# ============================================================
# 6. GIAO DIỆN CHÍNH
# ============================================================

st.title("Đánh giá rủi ro giao dịch thẻ")

st.write(
    "Nhập thông tin giao dịch và lịch sử trước đó để nhận điểm rủi ro, "
    "đề xuất xử lý và giải thích các dấu hiệu hành vi được phát hiện."
)

with st.expander("Cách đọc kết quả", expanded=False):
    st.markdown(
        f"""
- **Rủi ro thấp:** xác suất gian lận dưới {LOW_THRESHOLD:.0%}.
- **Cần xác minh:** xác suất từ {LOW_THRESHOLD:.0%} đến dưới {HIGH_THRESHOLD:.0%}.
- **Rủi ro cao:** xác suất từ {HIGH_THRESHOLD:.0%} trở lên.

Xác suất là đầu ra của mô hình, không phải bằng chứng chắc chắn về gian lận.
Phần giải thích hành vi dựa trên các quy tắc kiểm tra đặc trưng, không phải
SHAP/LIME. Chất lượng kết quả phụ thuộc vào độ đầy đủ và chính xác của lịch sử
giao dịch được nhập.
"""
    )


# ============================================================
# 6.1. NHẬP THÔNG TIN GIAO DỊCH
# ============================================================

st.subheader("1. Thông tin giao dịch")

with st.form("transaction_form"):
    col1, col2 = st.columns(2)

    with col1:
        transaction_date = st.date_input(
            "Ngày giao dịch",
            value=dt.date.today(),
        )

        amount = st.number_input(
            "Số tiền giao dịch (USD)",
            min_value=0.01,
            max_value=100000.0,
            value=45.0,
            step=1.0,
        )

        category = st.selectbox(
            "Danh mục giao dịch",
            options=CATEGORIES,
            index=(
                CATEGORIES.index("grocery_pos")
                if "grocery_pos" in CATEGORIES
                else 0
            ),
        )

    with col2:
        transaction_time = st.time_input(
            "Giờ giao dịch",
            value=dt.time(14, 0),
            step=60,
        )

        merchant = st.text_input(
            "Tên cửa hàng/đơn vị chấp nhận thanh toán",
            value="Cửa hàng A",
        )

        distance_km = st.number_input(
            "Khoảng cách đến cửa hàng (km)",
            min_value=0.0,
            max_value=20000.0,
            value=20.0,
            step=1.0,
            help=(
                "Với giao dịch trực tuyến, khoảng cách có thể không mang "
                "cùng ý nghĩa như giao dịch trực tiếp."
            ),
        )

    submitted = st.form_submit_button(
        "Đánh giá giao dịch",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# 6.2. NHẬP LỊCH SỬ GIAO DỊCH
# ============================================================

st.subheader("2. Lịch sử giao dịch trước đó")

st.caption(
    "Mỗi dòng là một giao dịch xảy ra trước giao dịch đang đánh giá. "
    "Nhập số phút trước giao dịch hiện tại, số tiền, danh mục, cửa hàng "
    "và khoảng cách. Không nhập giao dịch hiện tại vào bảng này."
)

if "history_df" not in st.session_state:
    st.session_state["history_df"] = SAMPLE_HISTORY.copy()

history = st.data_editor(
    st.session_state["history_df"],
    num_rows="dynamic",
    use_container_width=True,
    key="history_editor",
    column_config={
        "minutes_ago": st.column_config.NumberColumn(
            "Số phút trước",
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
        "merchant": st.column_config.TextColumn(
            "Cửa hàng",
        ),
        "distance_km": st.column_config.NumberColumn(
            "Khoảng cách (km)",
            min_value=0.0,
            format="%.1f",
        ),
    },
)

history = history.copy()
st.session_state["history_df"] = history

col_reset, col_note = st.columns([1, 3])

with col_reset:
    if st.button("Khôi phục lịch sử mẫu"):
        st.session_state["history_df"] = SAMPLE_HISTORY.copy()
        st.rerun()

with col_note:
    st.caption(
        "Nếu không có lịch sử, hãy xóa các dòng mẫu và đánh giá. "
        "Kết quả khi thiếu lịch sử có thể kém tin cậy hơn."
    )


# ============================================================
# 7. XỬ LÝ VÀ LƯU KẾT QUẢ ĐÁNH GIÁ
# ============================================================

if submitted:
    clean_merchant = merchant.strip()

    if not clean_merchant:
        st.session_state.pop("last_result", None)

        st.error(
            "Vui lòng nhập tên cửa hàng hoặc đơn vị chấp nhận thanh toán."
        )

    else:
        expected_columns = [
            "minutes_ago",
            "amt",
            "category",
            "merchant",
            "distance_km",
        ]

        missing_columns = [
            column
            for column in expected_columns
            if column not in history.columns
        ]

        if missing_columns:
            st.session_state.pop("last_result", None)

            st.error(
                "Bảng lịch sử thiếu cột: "
                + ", ".join(missing_columns)
            )

        else:
            valid_history = history.copy()

            if not valid_history.empty:
                for column in [
                    "minutes_ago",
                    "amt",
                    "distance_km",
                ]:
                    valid_history[column] = pd.to_numeric(
                        valid_history[column],
                        errors="coerce",
                    )

                invalid_rows = (
                    valid_history[expected_columns].isna().any(axis=1)
                    | (valid_history["minutes_ago"] <= 0)
                    | (valid_history["amt"] < 0)
                    | (valid_history["distance_km"] < 0)
                    | (~valid_history["category"].isin(CATEGORIES))
                    | (
                        valid_history["merchant"]
                        .astype(str)
                        .str.strip()
                        == ""
                    )
                )

                if invalid_rows.any():
                    st.session_state.pop("last_result", None)

                    st.error(
                        "Lịch sử có dòng thiếu dữ liệu hoặc giá trị không "
                        "hợp lệ. Hãy kiểm tra các ô được nhập trước khi "
                        "đánh giá."
                    )

                    st.stop()

            transaction = {
                "time": pd.Timestamp(
                    dt.datetime.combine(
                        transaction_date,
                        transaction_time,
                    )
                ),
                "amt": float(amount),
                "category": category,
                "merchant": clean_merchant,
                "distance_km": float(distance_km),
            }

            try:
                with st.spinner("Đang đánh giá giao dịch..."):
                    (
                        probability,
                        anomaly_score,
                        features,
                        history_count,
                    ) = score_transaction(
                        valid_history,
                        transaction,
                    )

                tier = tier_of(probability)
                reasons = explain_behavior(features.iloc[0])

                st.session_state["last_result"] = {
                    "probability": probability,
                    "anomaly_score": anomaly_score,
                    "features": features,
                    "history_count": history_count,
                    "tier": tier,
                    "reasons": reasons,
                    "transaction": transaction,
                }

            except Exception as error:
                st.session_state.pop("last_result", None)

                st.error(
                    "Không thể đánh giá giao dịch. Hãy kiểm tra dữ liệu đầu "
                    "vào và bảo đảm `features.py`, `model_pipeline.py` cùng "
                    "mô hình đã huấn luyện nằm đúng vị trí."
                )

                with st.expander("Chi tiết lỗi kỹ thuật"):
                    st.code(str(error))


# ============================================================
# 8. HIỂN THỊ KẾT QUẢ VÀ GIẢI THÍCH
# ============================================================

result = st.session_state.get("last_result")

if result:
    probability = result["probability"]
    anomaly_score = result["anomaly_score"]
    tier = result["tier"]

    tier_info = TIER_INFO[tier]
    transaction = result["transaction"]

    st.divider()
    st.subheader("3. Kết quả đánh giá")

    st.caption(
        f"{transaction['time'].strftime('%d/%m/%Y %H:%M')} · "
        f"{transaction['merchant']} · "
        f"{transaction['category']} · "
        f"${transaction['amt']:,.2f}"
    )

    # Hiển thị trạng thái rủi ro
    if tier == 1:
        st.success(f"**{tier_info['title']}**")

    elif tier == 2:
        st.warning(f"**{tier_info['title']}**")

    else:
        st.error(f"**{tier_info['title']}**")

    st.markdown(
        f"**Đề xuất xử lý:** {tier_info['action']}"
    )

    # Các chỉ số chính
    metric1, metric2 = st.columns(2)

    metric1.metric(
        "Xác suất gian lận ước tính",
        f"{probability:.2%}",
    )

    metric2.metric(
        "Số giao dịch lịch sử đã dùng",
        f"{result['history_count']:,}",
    )

    # Giải thích ý nghĩa xác suất và mức rủi ro
    st.markdown("#### Ý nghĩa của kết quả")

    st.write(
        make_interpretation(
            probability,
            tier,
        )
    )

    st.write(tier_info["description"])

    # Giải thích các dấu hiệu hành vi
    st.markdown("#### Vì sao giao dịch cần được chú ý?")

    if result["reasons"]:
        for reason in result["reasons"]:
            st.markdown(f"- {reason}")

    else:
        st.write(
            "Các quy tắc giải thích hiện tại không phát hiện dấu hiệu cảnh "
            "báo nổi bật từ những đặc trưng được kiểm tra. Điều này không "
            "đồng nghĩa với việc giao dịch chắc chắn an toàn."
        )

    # Chỉ hiển thị thông tin kỹ thuật khi người dùng muốn xem
    with st.expander("Thông tin kỹ thuật bổ sung"):
        st.write(
            "Điểm Isolation Forest là một điểm số bất thường do pipeline "
            "trả về. Không nên tự diễn giải điểm này là xác suất gian lận "
            "nếu chưa xác định rõ quy ước chiều điểm số trong "
            "`model_pipeline.py`."
        )

        st.write(
            f"Điểm Isolation Forest: `{anomaly_score:.4f}`"
        )

        st.write(
            "Các dấu hiệu hành vi ở trên là giải thích dựa trên quy tắc, "
            "không phải SHAP/LIME và không khẳng định quan hệ nhân quả."
        )

    st.caption(
        "Lưu ý: ứng dụng đưa ra kết quả hỗ trợ ra quyết định. "
        "Không nên từ chối giao dịch chỉ dựa trên một điểm số mà không "
        "xem xét quy trình xác minh và chính sách quản trị rủi ro."
    )
