import cv2
from ultralytics import YOLO
import numpy as np
import os
import math
import csv
from PIL import Image


def PredictionYOLOv8DehuskedRice(img_path):
    #Chạy mô hình Yolo với Model vừa huấn luyện
    model=YOLO('../YOLOv8/Train_final_weight/last.pt')
    img = cv2.imread(img_path)
    results = model.predict(img)
    with Image.open(img_path) as imginf:
      # Lấy thông tin DPI
      dpi = imginf.info.get('dpi') if 'dpi' in imginf.info else 'Không rõ'

    #Các thông số để thực hiện chỉnh sửa ảnh
    himg, wimg, channels = img.shape
    thick_line = int(himg * 1 / 640)
    thick_text = int(himg * 2 / 640)
    fsize_text = himg * 0.4 / 640
    fsixe_text_count = himg * 0.5 / 640
    spacing = int(himg * 10/ 640)
    up_spacing = int(himg * 20 / 640)

    #Khai báo các biến cần thiết
    counter={}
    output_lines = []  #Thông số dòng trong file CSV
    rice_info = []  # To store rice information for CSV

    #In tổng số hạt lên góc ảnh
    w=spacing
    h=spacing-20
    namemodel="YOLOv8"
    tsize=cv2.getTextSize(namemodel, cv2.FONT_HERSHEY_SIMPLEX, fsixe_text_count, thick_text)[0]
    cv2.putText(img, namemodel, (wimg-tsize[0],spacing+up_spacing),
                cv2.FONT_HERSHEY_SIMPLEX, fsixe_text_count, (255, 255, 255), thick_text)
    names = model.names
    for i in range(0,len(names)):
      h+=up_spacing
      rice_id = list(names)[list(names.values()).index(str(names[i]))]
      if (names[i] == 'Rice'):
          cv2.putText(img, names[i] + "= " + str(results[0].boxes.cls.tolist().count(rice_id)), (w, h),
                      cv2.FONT_HERSHEY_SIMPLEX, fsixe_text_count, (0, 0, 255), thick_text)
      elif (names[i] == 'Inferior'):
          cv2.putText(img, names[i] + "= " + str(results[0].boxes.cls.tolist().count(rice_id)), (w, h),
                      cv2.FONT_HERSHEY_SIMPLEX, fsixe_text_count, (0,255,0), thick_text)
      else:
          cv2.putText(img, names[i] + "= " + str(results[0].boxes.cls.tolist().count(rice_id)), (w, h),
                      cv2.FONT_HERSHEY_SIMPLEX, fsixe_text_count, (0,255,255), thick_text)

    #Xử lý vẽ các hình đa giác và đường thẳng thể hiện chiều dài và rộng của đối tượng
    for result in results:
        boxes = result.boxes.cpu().numpy() #Vẽ hình đa giác bao quanh đối tượng

        for i, (arrays, box) in enumerate(zip(result.masks.xy, boxes)):
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            points = box.xyxy[0].astype(int)
            (x, y), (w, h), angle = cv2.minAreaRect(arrays)
            if (w > h): #Đảm bảo thỏa mãn về chiều dài và rộng
              w, h = h, w
            if class_name in counter: #Tạo các biến đếm đối tượng khi có nhiều lớp
              counter[class_name]+=1
            else:
              counter[class_name]=1
            label = f"#{counter[class_name]}"
            points = np.array([(points[0], points[1]), (points[2], points[1]), (points[2], points[3]),
                               (points[0], points[3])])  # Vị trí điểm của box
            # Vẽ số thứ tự của đối tượng
            if (class_name == 'Rice'):
                cv2.putText(img, label, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, fsize_text,
                            (0, 0, 255), thick_line)
            elif (class_name == 'Inferior'):
                cv2.putText(img, label, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, fsize_text,
                            (0,255,0), thick_line)
            else:
                cv2.putText(img, label, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, fsize_text,
                            (0,255,255), thick_line)
            #Đổi số liệu sang mm
            if(dpi!='Không rõ' and dpi[0]>200):
              w=round(w/(dpi[0]/25.4),3)
              h=round(h/(dpi[0]/25.4),3)
              #Lưu thông tin từng đối tượng
              rice_info.append({
                    'Hạt gạo': counter[class_name],
                    'Loại': class_name,
                    'Chiều rộng (mm)': w,
                    'Chiều dài (mm)': h
                })
            else:
                rice_info.append({
                    'Hạt gạo': counter[class_name],
                    'Loại': class_name,
                    'Chiều rộng (pixel)': w,
                    'Chiều dài (pixel)': h
                })
        for i, (mask, box) in enumerate(zip(result.masks.xy, boxes)):
            class_id = (box.cls[0])
            class_name = model.names[class_id]
            if (class_name == 'Rice'):
                cv2.drawContours(img, [mask.astype(int)], -1, (0, 255, 0),
                                 thick_line)  # Vẽ hình đa giác bao quanh đối tượng
            elif (class_name == 'Inferior'):
                cv2.drawContours(img, [mask.astype(int)], -1, (0, 255, 255), thick_line)
            else:
                cv2.drawContours(img, [mask.astype(int)], -1, ((255,255,0)), thick_line)
        #Vẽ chiều rộng lên đối tượng
        for i, (arrays, box) in enumerate(zip(result.masks.xy, boxes)):
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            (x, y), (w, h), angle = cv2.minAreaRect(arrays)
        #Thay đổi góc xoay
            if (w > h):
                w, h = h, w
                angle+=90
            #Vẽ chiều rộng dựa theo góc độ
            angle_rad = math.radians(angle)
            dx = int(math.cos(angle_rad) * (w / 2))
            dy = int(math.sin(angle_rad) * (w / 2))
            pt1 = (int(x - dx), int(y - dy))
            pt2 = (int(x + dx), int(y + dy))
            cv2.line(img, pt1, pt2, (0, 0, 255), thick_line)

        #Vẽ chiều dài đối tượng
        for i, (arrays, box) in enumerate(zip(result.masks.xy, boxes)):
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            (x, y), (w, h), angle = cv2.minAreaRect(arrays)
            if (w < h):
              angle+=90
            else:
              w, h = h, w
            #Vẽ chiều rộng dựa theo góc độ
            angle_rad = math.radians(angle)
            dx = int(math.cos(angle_rad) * (h / 2))
            dy = int(math.sin(angle_rad) * (h / 2))
            pt1 = (int(x - dx), int(y - dy))
            pt2 = (int(x + dx), int(y + dy))
            cv2.line(img, pt1, pt2, (0, 0, 255), thick_line)
    return img,rice_info,counter

def create_csv_file(rice_info, img_filename,result_folder,img_path):
    with Image.open(img_path) as imginf:
      # Lấy thông tin DPI
        dpi = imginf.info.get('dpi') if 'dpi' in imginf.info else 'Không rõ'
    # Tạo tên file CSV từ tên file ảnh
    csv_filename = os.path.splitext(img_filename)[0] + '.csv'

    # Tạo đường dẫn đến file CSV trong thư mục test_result
    csv_path = os.path.join(result_folder, csv_filename)

    # Ghi thông tin rice_info vào file CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        if (dpi != 'Không rõ' and dpi[0] > 200):
            fieldnames = ['Hạt gạo', 'Loại', 'Chiều rộng (mm)', 'Chiều dài (mm)']
        else:
            fieldnames = ['Hạt gạo', 'Loại', 'Chiều rộng (pixel)', 'Chiều dài (pixel)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for rice in rice_info:
            writer.writerow(rice)
def AnalyFolderYOLO(test_folder,result_folder):
    # Lặp qua tất cả các tệp tin trong thư mục test
    for filename in os.listdir(test_folder):
        try:
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                # Xây dựng đường dẫn đầy đủ đến tệp tin ảnh
                img_path = os.path.join(test_folder, filename)

                # Xử lý ảnh bằng hàm PredictionYOLOv8DehuskedRice
                result_img, result_rice_info, counter = PredictionYOLOv8DehuskedRice(img_path)

                # Lưu kết quả vào thư mục
                result_path = os.path.join(result_folder, filename)
                create_csv_file(result_rice_info, filename, result_folder,img_path)
                cv2.imwrite(result_path, result_img)
        except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return 'Quá trình xử lý không thành công'
    return 'Xử lý hoàn thành. Kết quả đã được lưu vào thư mục.'

