from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request,render_template
import os
import YOLO_Analy
import Detectron2_Analy
import cv2
from random import random

def has_image_files(folder_path):
    # Lặp qua tất cả các tệp trong thư mục
    for file in os.listdir(folder_path):
        # Kiểm tra nếu tệp là ảnh (có phần mở rộng là .jpg, .jpeg, .png)
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            return True
    return False

# Khởi tạo
app = Flask(__name__)

#Cho máy khác truy cập
CORS(app)
app.config['CORS_HEADERS']='Content-Type'
app.config['UPLOAD_FOLDER']='static'

@app.route('/',methods=['GET','POST'])
@cross_origin(origins='*')
def home_process():
    if request.method == "POST":
        try:
            image = request.files['file']
            print(image)
            model_select = request.form['model_select']
            if image:
                # Luu
                path_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                print('Vị trí lưu: ', path_save)
                image.save(path_save)
                # Nhận diện qua mô hình
                if model_select == "YOLOv8":
                    result_img, info, counter = YOLO_Analy.PredictionYOLOv8DehuskedRice(path_save)
                else:
                    result_img, info, counter = Detectron2_Analy.Detectron2_segmentation(path_save)
                # Trả về đường dẫn tới file ảnh đã thực hiện
                print(info)
                print(counter)
                if not info:
                    return render_template('index.html', msg='Không nhận diện được vật thể')
                else:
                    cv2.imwrite(path_save, result_img)
                    # Trả về kết quả
                    return render_template("index.html", user_image=image.filename,info=info,counter=counter, rand=str(random()),
                                           msg="Tải file lên thành công")
            else:
                return render_template('index.html', msg='Hãy chọn file để tải lên')
        except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return render_template('index.html', msg='Không nhận diện được vật thể')
    else:
        return render_template('index.html')
@app.route('/folder',methods=['GET','POST'])
@cross_origin(origins='*')
def folder_process():
    if request.method == "POST":
        try:
            # Lấy đường dẫn đến thư mục chứa ảnh đầu vào từ form
            input_folder = request.form['folderPath']
            model_select = request.form['model_select']
            if input_folder:
                if has_image_files(input_folder):
                    if model_select == "YOLOv8":
                        result_folder = os.path.join(input_folder, 'YOLO_result')
                        if not os.path.exists(result_folder):
                            os.makedirs(result_folder)
                        answer = YOLO_Analy.AnalyFolderYOLO(input_folder,result_folder)
                    else:
                        result_folder = os.path.join(input_folder, 'Detectron2_result')
                        if not os.path.exists(result_folder):
                            os.makedirs(result_folder)
                        answer = Detectron2_Analy.AnalyFolderDetectron2(input_folder,result_folder)
                    return render_template('folder.html', msg=answer)
                else:
                    return render_template('folder.html', msg='Thư mục không chứa ảnh.')
            else:
                return render_template('folder.html', msg='Hãy chọn thư mục thư mục')
        except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return render_template('folder.html', msg='Không nhận diện được vật thể')
    else:
        return render_template('folder.html')
#BackEnd
if __name__ =='__main__':
    app.run(host='0.0.0.0',port='9999')