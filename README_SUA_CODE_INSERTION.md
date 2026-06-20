# Bổ sung Learning-based Insertion Heuristic cho repo CVRP

## Mục tiêu sửa

Code cũ chủ yếu là **sequential construction / append-based decoding**:
- mô hình chọn node tiếp theo,
- node được thêm vào cuối tuyến hiện tại.

Để khớp hơn với tên đề tài **heuristic chèn**, bản sửa này bổ sung:

**PPO + Attention + Beam Search sinh thứ tự ưu tiên khách hàng**
+
**Best Insertion Heuristic chèn khách hàng vào vị trí tăng cost nhỏ nhất**

## File mới

1. `learning_based_insertion.py`

Chứa công thức và thuật toán chèn thật sự:

```python
Delta(a,j,b) = c(a,j) + c(j,b) - c(a,b)
```

Với mỗi khách hàng `j`, thuật toán thử mọi vị trí giữa hai node liên tiếp trong các tuyến hiện có và chọn vị trí có Delta nhỏ nhất, không vượt tải trọng.

2. `app_insertion.py`

Streamlit app mới để chạy thực nghiệm:
- Greedy Decoder: đối chứng 1.
- Sampling Decoder: đối chứng 2.
- PPO + Attention + Best Insertion: thuật toán đề xuất đúng hướng "chèn".

## Cách dùng

Copy 2 file này vào thư mục gốc repo `-ACN`, cùng cấp với `app.py`.

Chạy:

```bash
streamlit run app_insertion.py
```

## Câu giải thích khi bảo vệ

Phần cài đặt mới không chỉ append khách hàng vào cuối tuyến. Mô hình PPO + Attention sinh thứ tự ưu tiên khách hàng, sau đó Best Insertion thử chèn từng khách hàng vào nhiều vị trí khả thi trong các tuyến hiện có. Vị trí được chọn là vị trí có chi phí tăng thêm nhỏ nhất theo công thức:

```text
Delta(a,j,b) = c(a,j) + c(j,b) - c(a,b)
```

Nhờ vậy, thuật toán phù hợp hơn với tên đề tài "heuristic chèn dựa trên học máy".
