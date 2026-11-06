# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:**  T4 GPU

**Python / PyTorch / Ultralytics:** 

- Python: 3.13.16 

- PyTorch: 2.11.0+cu130

- Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

   Record này nói về cả bức ảnh. Model phân loại chỉ xếp hạng các lớp cho toàn ảnh, trong record không có box hay tọa độ nào chỉ vào một chiếc xe cụ thể. Vì vậy `cab` ở đây là lớp model chọn cho cả ảnh, không có nghĩa là nó đã tìm ra một chiếc taxi nào đó.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

   Danh sách này đi kèm taxonomy và checkpoint. Checkpoint này dùng ImageNet-1K nên model chỉ chọn trong các lớp có sẵn của bộ đó, không tự đặt ra tên lớp mới.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

   Vì chỉ có tên lớp thì dễ hiểu nhầm. Cùng một mã số có thể là lớp khác nhau ở mỗi taxonomy: `class_id`=0 trong COCO-80 là `person`, nhưng trong ImageNet-1K lại là lớp khác. Cùng một tên ở hai taxonomy cũng có thể được định nghĩa khác nhau. Ghi đủ cả ba thì người đọc biết chính xác nhãn đó thuộc danh sách nào.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

   Guideline cần nói rõ tiêu chí chọn. Ví dụ: ưu tiên vật lớn nhất ở tiền cảnh, hay ưu tiên chủ đề mà dự án đang quan tâm. Cũng cần có thứ tự ưu tiên khi hai lớp ngang nhau, và nói rõ khi nào phải hỏi thay vì tự chọn. Ảnh `traffic` có nhiều loại xe, nếu không có quy tắc thì mỗi người có thể chọn một nhãn khác nhau.

6. Vì sao không được lấy `score` của model làm ground truth?

   Score chỉ cho biết model tin vào kết quả của nó đến đâu, không chứng minh kết quả đó đúng. Model có thể sai dù score cao, và cũng có thể đúng dù score thấp. Ground truth phải do người gán theo guideline và được kiểm tra lại. Nếu lấy score làm chuẩn thì chẳng khác nào để model tự chấm bài của mình.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.524862 ; `bbox_xyxy`=[531.91,151.43, 549.57, 195.15] ;
 `bbox_width`=17.67 ; `bbox_height`=43.72

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

   Góc trên-trái của box nằm ở điểm (531.91, 151.43), góc dưới-phải nằm ở điểm (549.57, 195.15). Tọa độ tính bằng pixel, gốc (0, 0) ở góc trên-trái ảnh, x tăng sang phải và y tăng xuống dưới. Box rộng 17.67 pixel và cao 43.72 pixel.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

   Ngưỡng 0.20 có 17 prediction, ngưỡng 0.60 có 6 prediction. Ngưỡng 0.35 dùng cho JSON có 11 prediction. Ngưỡng càng thấp thì càng giữ lại nhiều box.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

   Reviewer phải xem nhiều box hơn, trong đó có nhiều box score thấp. Họ cần loại box sai hoặc trùng, sửa box lệch và sửa lớp sai. Họ vẫn phải kiểm tra vật bị bỏ sót, vì ngưỡng chỉ lọc prediction chứ không quyết định vật nào được gán nhãn.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

   Mỗi vật thuộc danh sách lớp có một box riêng. Box vẽ sát phần nhìn thấy của vật: bốn cạnh chạm điểm ngoài cùng, không chừa nền thừa và không cắt vào vật. Hai vật cùng lớp vẫn vẽ hai box.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

   Mình chọn vùng model gán `person` 0.61 ở sát mép trái ảnh. Chỗ này nằm lẫn với khu bếp màu đen và các đồ vật trên bàn, nên mình không nhận rõ đó có phải người hay không. Nếu đúng là người thì phần lớn cơ thể đã nằm ngoài ảnh hoặc bị che.

   Với vật bị cắt mép nhưng vẫn nhận ra, mình viết thành quy tắc: chỉ vẽ box quanh phần còn thấy trong ảnh. Còn chỗ này thì mình hỏi Lab Coach, vì nhìn vào ảnh không đủ để biết đó là gì, nên không thể chỉ dựa vào score 0.61 để gán nhãn người.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

   Box chỉ là hình chữ nhật bao ngoài nên thường lẫn cả phần nền. Polygon đi theo biên của vật, nên cho biết hình dạng thật và vùng ảnh nào thuộc về vật, vùng nào là nền.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

   `instance_id` dùng để phân biệt từng đối tượng trong output của bài lab, ví dụ `kitchen-001`. Nó không phải class ID vì class ID cho biết lớp, nên hai người vẫn cùng lớp `person` nhưng có hai `instance_id` khác nhau. Nó cũng không phải tracking ID vì tracking ID dùng để theo một vật qua nhiều khung hình video, còn ở đây chỉ là ảnh tĩnh và không có theo dõi nào.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

   Mỗi instance có một mask riêng. Chấm điểm đúng theo biên phần nhìn thấy của vật: chỗ cong chấm dày hơn, chỗ thẳng chấm thưa hơn. Mask không lấn sang nền hoặc vật bên cạnh, và cũng không cắt vào vật.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

   Mình chọn chỗ các bát (bowl) chồng lên nhau hoặc thìa nằm sát đĩa. 

   Nếu còn thấy ranh giới giữa hai vật thì mình dùng guideline: vẽ theo ranh giới nhìn thấy, không đoán phần bị che, và tách thành hai instance. Nếu biên nhòe đến mức phóng to vẫn không phân biệt được thì mình hỏi Lab Coach, vì tự đoán sẽ làm mask không đáng tin.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho cả ảnh, không có tọa độ | Ảnh `traffic` có nhiều loại xe nhưng model chỉ cho top-1 là `cab`, score 0.51. Rất khó chọn một lớp đại diện cho cả cảnh. | Xem toàn ảnh và chọn một lớp theo thứ tự ưu tiên trong guideline | Nhãn có đúng quy tắc ưu tiên không; các ca chưa rõ đã được đưa đi hỏi chưa |
| Phát hiện vật thể | Mỗi vật có một lớp và một box `[x_min, y_min, x_max, y_max]` theo pixel | Ở ngưỡng 0.35, ảnh `kitchen` có 11 box. Vùng `person` 0.61 ở mép trái rất khó xác nhận là người. Nồi chảo treo trên tường cũng hầu như không có box, vì COCO-80 không có lớp nồi hay chảo. | Vẽ box sát từng vật và chọn đúng lớp | Box có quá rộng hoặc quá hẹp không; có vật bị sót hoặc sai lớp không |
| Instance segmentation | Mỗi instance có một lớp và một polygon/mask riêng | Các chiếc bát (bowl) xếp chồng lên nhau ở góc dưới bên trái có phần biên (mask) bám chưa sát, ranh giới giữa các bát chạm nhau bị mờ và mô hình gặp khó khăn trong việc tách biệt chính xác đa giác cho từng chiếc bát riêng lẻ. | Chấm điểm theo biên từng vật và tách riêng các vật chạm nhau | Mask có bám biên không; hai vật có bị gộp làm một không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

   Mình chỉ dùng ba ảnh COCO công khai có sẵn trong notebook và không đưa ảnh cá nhân, dữ liệu nội bộ hay thông tin nhạy cảm lên Colab hoặc GitHub.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

   Mình sẽ dừng ngay và báo cho Lab Coach hoặc giảng viên, không tự xử lý tiếp.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS