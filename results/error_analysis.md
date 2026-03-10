# NLU Error Analysis Report

Comparative error analysis for SVM, JointBERT, and LLM models.

## Summary

| Model | Total Errors | Error Rate | Ambiguous | Rare | Slot Boundary | OOV |
|-------|-------------|------------|-----------|------|---------------|-----|
| SVM | 198 | 22.2% | 9 | 18 | 17 | 0 |
| JointBERT | 250 | 28.0% | 2 | 6 | 0 | 1 |
| LLM | 100 | 100.0% | 7 | 34 | 0 | 0 |

## SVM Model Analysis

- Total samples: 893
- Total errors: 198
- Error rate: 22.17%

### Top Confusion Pairs

- flight -> airfare (6 errors)
- UNK -> flight (3 errors)
- city -> flight (3 errors)
- airfare#flight -> airfare (3 errors)
- airfare#flight -> flight (3 errors)

### Sample Errors

**1. vào ngày mùng 1 tháng 4 tôi cần một vé từ sevilla đến hà nội...**
   - True: flight, Pred: airfare
   - Categories: ambiguous_intent

**2. tôi muốn có một chuyến bay từ quy nhơn đến thành phố hà nội ...**
   - True: flight, Pred: flight
   - Categories: other

**3. chuyến bay khởi hành từ hạ long đi thành phố hồ chí minh vào...**
   - True: flight, Pred: flight
   - Categories: other

**4. hiển thị các chuyến bay và giá tương ứng từ hạ long đến thàn...**
   - True: airfare#flight, Pred: airfare#flight
   - Categories: other

**5. tìm giúp tôi các chuyến bay đến thành phố hà nội vào thứ bảy...**
   - True: flight, Pred: flight
   - Categories: other

## JointBERT Model Analysis

- Total samples: 893
- Total errors: 250
- Error rate: 28.00%

### Top Confusion Pairs

- flight -> ground_service (5 errors)
- flight -> quantity (2 errors)
- flight -> flight_no (2 errors)
- flight -> abbreviation (2 errors)
- flight -> airfare#flight (2 errors)

### Sample Errors

**1. tôi muốn tìm một chuyến bay từ đà nẵng đến phú quốc và có mộ...**
   - True: flight, Pred: flight
   - Categories: other

**2. vào ngày 12 tháng 8 tôi cần một chuyến bay đi từ tuy hòa đến...**
   - True: flight, Pred: flight
   - Categories: other

**3. tôi muốn một chuyến bay một chiều đi từ phú quốc đến buôn ma...**
   - True: flight, Pred: flight
   - Categories: other

**4. tôi muốn có một chuyến bay từ quy nhơn đến thành phố hà nội ...**
   - True: flight, Pred: flight
   - Categories: other

**5. sau 12 giờ trưa thứ tư ngày mùng sáu tháng tư tôi muốn bay t...**
   - True: flight, Pred: flight
   - Categories: other

## LLM Model Analysis

- Total samples: 100
- Total errors: 100
- Error rate: 100.00%

### Top Confusion Pairs

- abbreviation -> flight (9 errors)
- airfare -> flight (7 errors)
- ground_service -> flight (4 errors)
- ground_fare -> flight (3 errors)
- distance -> flight (2 errors)

### Sample Errors

**1. tôi muốn tìm một chuyến bay từ đà nẵng đến phú quốc và có mộ...**
   - True: flight, Pred: flight
   - Categories: other

**2. vào ngày mùng 1 tháng 4 tôi cần một vé từ sevilla đến hà nội...**
   - True: flight, Pred: flight
   - Categories: other

**3. vào ngày 12 tháng 8 tôi cần một chuyến bay đi từ tuy hòa đến...**
   - True: flight, Pred: flight
   - Categories: other

**4. tôi muốn một chuyến bay một chiều đi từ phú quốc đến buôn ma...**
   - True: flight, Pred: flight
   - Categories: other

**5. tôi muốn có một chuyến bay từ quy nhơn đến thành phố hà nội ...**
   - True: flight, Pred: flight
   - Categories: other
