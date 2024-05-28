import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog,DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import cv2
import os
import math
import csv
from PIL import Image

#Đăng ký dataset
register_coco_instances("my_dataset_train", {},
                        "../DehuskedRice/final_data.coco/train/_annotations.coco.json",
                        "../DehuskedRice/final_data.coco/train")
train_metadata = MetadataCatalog.get("my_dataset_train")
train_dataset_dicts = DatasetCatalog.get("my_dataset_train")


def Detectron2_segmentation(img_path):
  #Chuẩn bị mô hình xử lý
  model ="../Detectron2_Model/Detectron2_Models_Final/config.yaml"
  weight="../Detectron2_Model/OUTPUT_FINAL/model_final.pth"
  cfg = get_cfg()
  cfg.merge_from_file(model)
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
  cfg.MODEL.WEIGHTS = weight
  predictor = DefaultPredictor(cfg)
  #Xử lý ảnh với mô hình Detectron2
  new_im = cv2.imread(img_path)
  outputs = predictor(new_im)
  pred_classes = outputs['instances'].pred_classes.cpu().tolist()
  class_names = MetadataCatalog.get("my_dataset_train").thing_classes
  pred_boxes = outputs["instances"].pred_boxes.tensor.to("cpu").numpy().squeeze()

  with Image.open(img_path) as imginf:
      # Lấy thông tin DPI
      dpi = imginf.info['dpi'] if 'dpi' in imginf.info else 'Không rõ'
  #Khai báo thông số cần thiết
  list_name_class=[]
  contours = []
  output_lines = []
  rice_info = []
  counter={}

  #Các thông số để thực hiện chỉnh sửa ảnh
  himg, wimg, channels = new_im.shape
  thick_line = int(himg * 1 / 640)
  thick_text = int(himg * 2 / 640)
  fsize_text = himg * 0.4 / 640
  fsixe_text_count = himg * 0.5 / 640
  spacing = int(himg * 10 / 640)
  up_spacing = int(himg * 20 / 640)

  #Chuyển đổi thông số mặt nạ thành dạng xy
  for pred_mask in outputs['instances'].pred_masks:
      mask = pred_mask.to("cpu").numpy().astype('uint8')
      contour, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
      contours.append(contour[0])
  image_with_overlaid_predictions = new_im.copy()


  #Vẽ hình đa giác lên đối tượng trong ảnh
  for i,contour in enumerate(contours):
      class_i = outputs['instances'].pred_classes[i]
      class_name = class_names[class_i]
      if (class_name == 'Rice'):
          cv2.drawContours(image_with_overlaid_predictions, [contour], -1, (0, 255, 0),
                           thick_line)  # Vẽ hình đa giác bao quanh đối tượng
      elif (class_name == 'Inferior'):
          cv2.drawContours(image_with_overlaid_predictions, [contour], -1, (0, 255, 255), thick_line)
      else:
          cv2.drawContours(image_with_overlaid_predictions, [contour], -1, (255, 255, 0), thick_line)
  #Vẽ số thứ tự lên ảnh và lưu thông tin
  for i,(box,contour) in enumerate(zip(pred_boxes,contours)):
      class_i= outputs['instances'].pred_classes[i]
      class_name=class_names[class_i]
      list_name_class.append(class_name)
      if isinstance(box, np.ndarray):
        points = box.astype(int)
      else:
        points = pred_boxes.astype(int)
      points = np.array([(points[0], points[1]), (points[2], points[1]), (points[2], points[3]), (points[0], points[3])])
      (x, y), (w, h), angle = cv2.minAreaRect(contour)
      if (w > h):
        w, h = h, w
      if class_name in counter: # Tạo các biến đếm đối tượng khi có nhiều lớp
              counter[class_name]+=1
      else:
        counter[class_name]=1
      label=f'#{str(counter[class_name])}'
      # Vẽ số thứ tự đối tượng
      if (class_name == 'Rice'):
          cv2.putText(image_with_overlaid_predictions, label, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, fsize_text,
                      (0, 0, 255), thick_line)
      elif (class_name == 'Inferior'):
          cv2.putText(image_with_overlaid_predictions, label, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, fsize_text,
                      (0,255,0), thick_line)
      else:
          cv2.putText(image_with_overlaid_predictions, label, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, fsize_text,
                      (0,255,255), thick_line)
      #Đổi số liệu sang mm
      if (dpi != 'Không rõ' and dpi[0] > 200):
          w = round(w / (dpi[0] / 25.4), 3)
          h = round(h / (dpi[0] / 25.4), 3)
          # Lưu thông tin từng đối tượng
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

  #Vẽ chiều rộng lên đối tượng
  for contour in contours:
    (x, y), (w, h), angle = cv2.minAreaRect(contour)
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
    cv2.line(image_with_overlaid_predictions, pt1, pt2, (0, 0, 255), thick_line)

  #Vẽ chiều dài lên đối tượng
  for contour in contours:
    (x, y), (w, h), angle = cv2.minAreaRect(contour)
    #Thay đổi góc xoay
    if (w < h):
      angle+=90
    else:
      w, h = h, w
    #Vẽ chiều dài dựa theo góc độ
    angle_rad = math.radians(angle)
    dx = int(math.cos(angle_rad) * (h / 2))
    dy = int(math.sin(angle_rad) * (h / 2))
    pt1 = (int(x - dx), int(y - dy))
    pt2 = (int(x + dx), int(y + dy))
    cv2.line(image_with_overlaid_predictions, pt1, pt2, (0, 0, 255), thick_line)
  #In tổng số hạt lên góc ảnh
  w=spacing
  h=spacing-20
  namemodel = "Detectron2"
  tsize = cv2.getTextSize(namemodel, cv2.FONT_HERSHEY_SIMPLEX, fsixe_text_count, thick_text)[0]
  cv2.putText(image_with_overlaid_predictions, namemodel, (wimg - tsize[0], spacing + up_spacing),
              cv2.FONT_HERSHEY_SIMPLEX, fsixe_text_count, (255, 255, 255), thick_text)
  names=class_names
  for i in range(0,len(names)):
      if names[i] != 'rice-segmentation':
        h+=up_spacing
        rice_id = list(names)[list(names).index(str(names[i]))]
        if (names[i] == 'Rice'):
            cv2.putText(image_with_overlaid_predictions, names[i]+"= "+str(list_name_class.count(rice_id)), (w, h),
                        cv2.FONT_HERSHEY_SIMPLEX, fsixe_text_count, (0, 0, 255), thick_text)
        elif (names[i] == 'Inferior'):
            cv2.putText(image_with_overlaid_predictions, names[i]+"= "+str(list_name_class.count(rice_id)), (w, h),
                        cv2.FONT_HERSHEY_SIMPLEX, fsixe_text_count, (0, 255, 0), thick_text)
        else:
            cv2.putText(image_with_overlaid_predictions, names[i]+"= "+str(list_name_class.count(rice_id)), (w, h),
                        cv2.FONT_HERSHEY_SIMPLEX, fsixe_text_count, (0,255,255), thick_text)
  return image_with_overlaid_predictions,rice_info,counter


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
def AnalyFolderDetectron2(test_folder,result_folder):
    # Lặp qua tất cả các tệp tin trong thư mục test
    for filename in os.listdir(test_folder):
        try:
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                # Xây dựng đường dẫn đầy đủ đến tệp tin ảnh
                img_path = os.path.join(test_folder, filename)

                # Xử lý ảnh bằng hàm PredictionYOLOv8DehuskedRice
                result_img, result_rice_info, counter = Detectron2_segmentation(img_path)

                # Lưu kết quả vào thư mục
                result_path = os.path.join(result_folder, filename)
                create_csv_file(result_rice_info, filename, result_folder,img_path)
                cv2.imwrite(result_path, result_img)
        except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return 'Quá trình xử lý không thành công'
    return 'Xử lý hoàn thành. Kết quả đã được lưu vào thư mục.'
