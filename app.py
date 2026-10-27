import os
import datetime as dt

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import RAW_REQUIRED, build_feature_frame
from model_pipeline import FINAL_THRESHOLD, LOW_THRESHOLD, decide


# ============================================================
# 1. CẤU HÌNH
# ============================================================

st.set_page_config(
    page_title="FraudGuard",
    page_icon="🛡️",
    layout="wide",
)

ARTIFACT_PATH = os.path.join("artifacts", "pipeline.joblib")


@st.cache_resource
def load_pipeline():
    return joblib.load(ARTIFACT_PATH)


if not os.path.exists(ARTIFACT_PATH):
    st.error(
        "Không tìm thấy mô hình đã huấn luyện. "
        "Vui lòng kiểm tra artifacts/pipeline.joblib."
    )
    st.stop()

try:
    pipe = load_pipeline()
except Exception as error:
    st.error("Không thể tải mô hình.")
    st.code(str(error))
    st.stop()


# ============================================================
# 2. GIAO DIỆN
# ============================================================

st.title("FraudGuard")
st.caption("Đánh giá rủi ro giao dịch thẻ tín dụng")

st.write(
    "Nhập thông tin giao dịch cần kiểm tra. "
    "Hệ thống sẽ đánh giá rủi ro và đưa ra gợi ý xử lý."
)

st.divider()


# ============================================================
# 3. NGƯỠNG PHÂN LOẠI
# ============================================================

with st.expander("Thiết lập ngưỡng đánh giá", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        low_threshold = st.number_input(
            "Ngưỡng rủi ro thấp",
            min_value=0.0,
            max_value=0.98,
            value=float(LOW_THRESHOLD),
            step=0.01,
            format="%.2f",
        )

    with col2:
        high_threshold = st.number_input(
            "Ngưỡng rủi ro cao",
            min_value=0.01,
            max_value=1.0,
            value=float(FINAL_THRESHOLD),
            step=0.01,
            format="%.2f",
        )

    st.caption(
        "Ngưỡng mặc định lấy từ model_pipeline.py. "
        "Chỉ thay đổi nếu bạn muốn thử các mức quyết định khác."
    )


# ============================================================
# 4. NHẬP THÔNG TIN MỘT GIAO DỊCH
# ============================================================

st.subheader("Thông tin giao dịch")

with st.form("single_transaction_form"):

    col1, col2 = st.columns(2)

    with col1:
        transaction_date = st.date_input(
            "Ngày giao dịch",
            value=dt.date.today(),
        )

        transaction_time = st.time_input(
            "Thời gian giao dịch",
            value=dt.datetime.now().time().replace(second=0, microsecond=0),
        )

        amount = st.number_input(
            "Số tiền giao dịch (USD)",
            min_value=0.01,
            max_value=1_000_000.0,
            value=100.0,
            step=10.0,
        )

        category_options = sorted(
            str(name).replace("category_", "", 1)
            for name in getattr(pipe, "ohe_names", [])
            if str(name).startswith("category_")
        )

        if not category_options:
            category_options = [
                "grocery_pos",
                "shopping_net",
                "misc_net",
                "gas_transport",
                "food_dining",
                "entertainment",
                "personal_care",
                "home",
            ]

        category = st.selectbox(
            "Danh mục giao dịch",
            category_options,
        )

        merchant = st.text_input(
            "Tên cửa hàng",
            value="fraud_test_merchant",
        )

    with col2:
        card_number = st.text_input(
            "Mã thẻ / số thẻ dùng để đối chiếu lịch sử",
            value="1000000000000000",
            help=(
                "Đây là thông tin định danh để tạo đặc trưng hành vi. "
                "Với dữ liệu thật, không nên nhập số thẻ đầy đủ; "
                "hãy sử dụng mã định danh nội bộ đã được ẩn danh."
            ),
        )

        gender = st.selectbox(
            "Giới tính theo dữ liệu",
            ["F", "M"],
        )

        dob = st.date_input(
            "Ngày sinh theo dữ liệu",
            value=dt.date(1995, 1, 1),
        )

        city_pop = st.number_input(
            "Dân số khu vực chủ thẻ",
            min_value=0,
            value=50000,
            step=1000,
        )

    st.markdown("#### Vị trí giao dịch")

    col3, col4 = st.columns(2)

    with col3:
        customer_lat = st.number_input(
            "Vĩ độ chủ thẻ",
            min_value=-90.0,
            max_value=90.0,
            value=16.0544,
            format="%.6f",
        )

        customer_long = st.number_input(
            "Kinh độ chủ thẻ",
            min_value=-180.0,
            max_value=180.0,
            value=108.2022,
            format="%.6f",
        )

    with col4:
        merchant_lat = st.number_input(
            "Vĩ độ cửa hàng",
            min_value=-90.0,
            max_value=90.0,
            value=16.0600,
            format="%.6f",
        )

        merchant_long = st.number_input(
            "Kinh độ cửa hàng",
            min_value=-180.0,
            max_value=180.0,
            value=108.2100,
            format="%.6f",
        )

    st.markdown("#### Thông tin bổ sung")

    col5, col6 = st.columns(2)

    with col5:
        city = st.text_input("Thành phố", value="Da Nang")
        state = st.text_input("Bang / tỉnh", value="DN")
        zip_code = st.text_input("Mã bưu chính", value="550000")

    with col6:
        street = st.text_input("Địa chỉ đường", value="Test Street")
        job = st.text_input("Nghề nghiệp theo dữ liệu", value="Other")

    submitted = st.form_submit_button(
        "Đánh giá giao dịch",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# 5. TẠO DỮ LIỆU ĐẦU VÀO CHO MÔ HÌNH
# ============================================================

def build_single_transaction():
    """
    Tạo một dòng dữ liệu thô theo cấu trúc đầu vào của dự án.

    Những trường không được nhập trực tiếp nhưng có trong RAW_REQUIRED
    được gán giá trị mặc định để đáp ứng cấu trúc dữ liệu.
    """

    transaction_timestamp = pd.Timestamp(
        dt.datetime.combine(transaction_date, transaction_time)
    )

    raw_row = {
        "trans_date_trans_time": transaction_timestamp,
        "cc_num": card_number.strip(),
        "merchant": merchant.strip(),
        "category": category,
        "amt": float(amount),
        "first": "Test",
        "last": "User",
        "gender": gender,
        "street": street,
        "city": city,
        "state": state,
        "zip": zip_code,
        "lat": float(customer_lat),
        "long": float(customer_long),
        "city_pop": int(city_pop),
        "job": job,
        "dob": pd.Timestamp(dob),
        "trans_num": "MANUAL-TRANSACTION-001",
        "merch_lat": float(merchant_lat),
        "merch_long": float(merchant_long),
        "is_fraud": 0,
    }

    # Bổ sung các cột bắt buộc khác nếu dự án có yêu cầu.
    defaults = {
        "first": "Test",
        "last": "User",
        "gender": gender,
        "street": street,
        "city": city,
        "state": state,
        "zip": zip_code,
        "lat": float(customer_lat),
        "long": float(customer_long),
        "city_pop": int(city_pop),
        "job": job,
        "dob": pd.Timestamp(dob),
        "trans_num": "MANUAL-TRANSACTION-001",
        "merch_lat": float(merchant_lat),
        "merch_long": float(merchant_long),
        "is_fraud": 0,
    }

    for column in RAW_REQUIRED:
        if column not in raw_row:
            raw_row[column] = defaults.get(column, 0)

    raw = pd.DataFrame([raw_row])

    return raw


# ============================================================
# 6. GIẢI THÍCH CÁC DẤU HIỆU
# ============================================================

def get_feature(row, name):
    if name not in row.index:
        return None

    value = row[name]

    if pd.isna(value):
        return None

    return value


def explain_transaction(row):
    explanations = []

    amount_zscore = get_feature(row, "customer_amt_zscore")
    if amount_zscore is not None:
        try:
            if float(amount_zscore) >= 3:
                explanations.append(
                    "Số tiền cao bất thường so với mức chi tiêu tham chiếu."
                )
        except (ValueError, TypeError):
            pass

    recent_count = get_feature(row, "transactions_last_5min")
    if recent_count is not None:
        try:
            if float(recent_count) >= 3:
                explanations.append(
                    "Có nhiều giao dịch trong khoảng thời gian ngắn."
                )
        except (ValueError, TypeError):
            pass

    amount_jump = get_feature(row, "amt_jump_ratio")
    if amount_jump is not None:
        try:
            if float(amount_jump) >= 5:
                explanations.append(
                    "Số tiền tăng mạnh so với giao dịch liền trước."
                )
        except (ValueError, TypeError):
            pass

    category_ratio = get_feature(row, "amt_ratio_cat_max")
    if category_ratio is not None:
        try:
            if float(category_ratio) > 1.5:
                explanations.append(
                    "Số tiền cao hơn đáng kể so với mức tham chiếu "
                    "của danh mục giao dịch."
                )
        except (ValueError, TypeError):
            pass

    new_merchant = get_feature(row, "is_new_merchant")
    if new_merchant is not None:
        try:
            if bool(new_merchant):
                explanations.append(
                    "Cửa hàng chưa xuất hiện trong dữ liệu lịch sử "
                    "được dùng để tạo đặc trưng."
                )
        except (ValueError, TypeError):
            pass

    if not explanations:
        explanations.append(
            "Không phát hiện dấu hiệu cảnh báo nổi bật từ các quy tắc "
            "giải thích hiện có. Điều này không đảm bảo giao dịch an toàn."
        )

    return explanations


# ============================================================
# 7. ĐÁNH GIÁ VÀ HIỂN THỊ KẾT QUẢ
# ============================================================

if submitted:

    if low_threshold >= high_threshold:
        st.error(
            "Ngưỡng rủi ro thấp phải nhỏ hơn ngưỡng rủi ro cao."
        )
        st.stop()

    if not card_number.strip():
        st.error("Vui lòng nhập mã thẻ để đánh giá giao dịch.")
        st.stop()

    if not merchant.strip():
        st.error("Vui lòng nhập tên cửa hàng.")
        st.stop()

    try:
        with st.spinner("Đang đánh giá giao dịch..."):

            raw = build_single_transaction()

            missing = [
                column
                for column in RAW_REQUIRED
                if column not in raw.columns
            ]

            if missing:
                raise ValueError(
                    "Thiếu cột dữ liệu bắt buộc: "
                    + ", ".join(missing)
                )

            features = build_feature_frame(raw)

            probabilities, anomaly_scores = pipe.score_frame(features)

            probability = float(
                np.asarray(probabilities).reshape(-1)[0]
            )

            feature_row = features.iloc[0]

            decision = decide(
                np.array([probability]),
                low_threshold,
                high_threshold,
            )

            if isinstance(decision, (list, tuple, np.ndarray, pd.Series)):
                decision = decision[0]

            decision = str(decision)

            explanations = explain_transaction(feature_row)

        st.divider()
        st.subheader("Kết quả đánh giá")

        result_col1, result_col2 = st.columns([1, 2])

        with result_col1:
            st.metric(
                "Xác suất gian lận ước tính",
                f"{probability:.2%}",
            )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

        with result_col2:
            st.markdown("**Đề xuất xử lý**")

            if "Approve" in decision or "Phê duyệt" in decision:
                st.success(decision)
                st.write(
                    "Giao dịch nằm dưới ngưỡng rủi ro đã thiết lập. "
                    "Có thể chuyển sang quy trình xử lý thông thường."
                )

            elif "Review" in decision or "Xem xét" in decision:
                st.warning(decision)
                st.write(
                    "Nên xác minh thêm thông tin trước khi đưa ra "
                    "quyết định cuối cùng."
                )

            else:
                st.error(decision)
                st.write(
                    "Giao dịch đạt ngưỡng rủi ro cao. Nên ưu tiên kiểm tra "
                    "và xử lý theo chính sách của đơn vị phát hành thẻ."
                )

        st.markdown("#### Thông tin giao dịch")

        detail1, detail2, detail3 = st.columns(3)

        detail1.write(f"**Số tiền:** ${amount:,.2f}")
        detail2.write(f"**Danh mục:** {category}")
        detail3.write(f"**Cửa hàng:** {merchant}")

        st.markdown("#### Dấu hiệu cần chú ý")

        for explanation in explanations:
            st.write(f"- {explanation}")

        st.caption(
            "Lưu ý: kết quả là dự đoán từ mô hình, không phải bằng chứng "
            "chắc chắn về gian lận. Vì ứng dụng đang đánh giá một giao dịch "
            "đơn lẻ, các đặc trưng phụ thuộc lịch sử có thể không phản ánh "
            "đúng hành vi nếu chưa cung cấp lịch sử thẻ."
        )

    except Exception as error:
        st.error(
            "Không thể đánh giá giao dịch. Có thể dữ liệu nhập chưa tương "
            "thích với cấu trúc mà mô hình đã huấn luyện yêu cầu."
        )

        with st.expander("Chi tiết lỗi kỹ thuật"):
            st.code(str(error))
