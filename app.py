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
