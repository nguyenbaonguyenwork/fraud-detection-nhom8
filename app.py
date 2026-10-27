            transaction = {
                "time": pd.Timestamp(
                    dt.datetime.combine(transaction_date, transaction_time)
                ),
                "amt": float(amount),
                "category": category,
                "merchant": clean_merchant,
                "distance_km": float(distance_km),
            }

            try:
                with st.spinner("Đang đánh giá giao dịch..."):
                    probability, anomaly_score, features, history_count = (
                        score_transaction(valid_history, transaction)
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
                    "Không thể đánh giá giao dịch. Hãy kiểm tra dữ liệu đầu vào "
                    "và bảo đảm `features.py`, `model_pipeline.py` cùng mô hình "
                    "đã huấn luyện nằm đúng vị trí."
                )
                with st.expander("Chi tiết lỗi kỹ thuật"):
                    st.code(str(error))


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

    if tier == 1:
        st.success(f"**{tier_info['title']}**")
    elif tier == 2:
        st.warning(f"**{tier_info['title']}**")
    else:
        st.error(f"**{tier_info['title']}**")

    st.markdown(f"**Đề xuất xử lý:** {tier_info['action']}")

    metric1, metric2 = st.columns(2)
    metric1.metric("Xác suất gian lận ước tính", f"{probability:.2%}")
    metric2.metric("Số giao dịch lịch sử đã dùng", f"{result['history_count']:,}")

    st.markdown("#### Ý nghĩa của kết quả")
    st.write(make_interpretation(probability, tier))
    st.write(tier_info["description"])

    st.markdown("#### Vì sao giao dịch cần được chú ý?")
    if result["reasons"]:
        for reason in result["reasons"]:
            st.markdown(f"- {reason}")
    else:
        st.write(
            "Các quy tắc giải thích hiện tại không phát hiện dấu hiệu cảnh báo "
            "nổi bật từ những đặc trưng được kiểm tra. Điều này không đồng nghĩa "
            "với việc giao dịch chắc chắn an toàn."
        )

    with st.expander("Thông tin kỹ thuật bổ sung"):
        st.write(
            "Điểm Isolation Forest là một điểm số bất thường do pipeline trả về. "
            "Không nên tự diễn giải điểm này là xác suất gian lận nếu chưa xác "
            "định rõ quy ước chiều điểm số trong `model_pipeline.py`."
        )
        st.write(f"Điểm Isolation Forest: `{anomaly_score:.4f}`")
        st.write(
            "Các dấu hiệu hành vi ở trên là giải thích dựa trên quy tắc, "
            "không phải SHAP/LIME và không khẳng định quan hệ nhân quả."
        )

    st.caption(
        "Lưu ý: ứng dụng đưa ra kết quả hỗ trợ ra quyết định. "
        "Không nên từ chối giao dịch chỉ dựa trên một điểm số mà không xem xét "
        "quy trình xác minh và chính sách quản trị rủi ro."
    )
