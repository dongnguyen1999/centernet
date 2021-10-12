from inference.main import main
# from inference.model.decode import CtDetDecode, CountDecode
from centernet_detect.dataset.vn_vehicle import DataGenerator, load_data
from inference.model.decode import create_mask, get_masked_img, visualize
from inference.model.model import Model, create_models
from inference.model.bg_subtraction import BackgroundSubtractorMOG2
from utils.config import Config
import cv2
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time
# main()

config = Config(
    name='hourglass_centernet_vehicle_v1',
    num_classes=3, 
    train_path='video1_all\\test', 
    valid_path='video1_all\\test',
    test_paths=['video1_all\\test'],
    checkpoint_path='models\hourglass_centernet_1stack_512\\rf_v1',
    annotation_filename='_annotations_custom_v2.txt',
    data_base='C:\\vehicle-data',
    epochs=1,
    batch_size=1,
    image_id='filename',
    input_size=512,
    enable_augmentation=True,
    # weights_path='/kaggle/working/centernet.hdf5',
)

crowd_model, detect_model = create_models(r'D:\CenterNet\inference\weight\vgg16_fineturning_epoch2.hdf5', r'D:\CenterNet\inference\weight\hg1stack_tfmosaic_epoch6.hdf5')
background_subtractor = BackgroundSubtractorMOG2()

model = Model(crowd_model, detect_model, background_subtractor, 0.5, 0.25, [10, 25], [0.5, 2.0])

# image = cv2.imread(r"C:\vehicle-data\video1_all\train\video1_1.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# print(image.shape)

# y_pred, detections = model.predict(image)
# print(y_pred, detections)


# cap = cv2.VideoCapture(r"C:\vehicle-data\raw\DLngoaiBenhVien\video1.avi")
# cap = cv2.VideoCapture(r"C:\Users\nvdkg\Documents\Camtasia\dnnew_crowd.autosave\dnnew_crowd.autosave.mp4")

# cap = cv2.VideoCapture(r"C:\Users\nvdkg\Documents\Camtasia\Small cut\Small cut.mp4")
cap = cv2.VideoCapture(r"C:\vehicle-data\raw\DLngoaiBenhVien\video1.avi")
# cap = cv2.VideoCapture(r"C:\Users\nvdkg\Documents\Camtasia\test3_crowd\test3_crowd.mp4")
# cap = cv2.VideoCapture(r"C:\Users\nvdkg\Documents\Camtasia\dnnew_crowd.autosave\dnnew_crowd.autosave.mp4")
writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 15, (640,480))

# cap = cv2.VideoCapture(r'C:\Users\nvdkg\Documents\Camtasia\test3_crowd\test3_crowd.mp4')

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
count = 0

mouseX,mouseY = 0,0
def get_mouse_click_coord(event, x, y, flags, param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX,mouseY = x,y

# cv2.namedWindow('Predicted')
# cv2.setMouseCallback('Predicted', get_mouse_click_coord)

# dnnew mask
pts = np.array([[0, 317], [286, 230], 
                [512, 232], [512, 512], 
                [0, 512]], np.int32)

# test3 mask
# pts = np.array([[109, 446], [86, 246], 
#                 [191, 249], [478, 318]], np.int32)

env_mask = create_mask(pts)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.resize(frame, (512, 512))
        
        # cv2.imshow("Frame", frame)

        label, detections, diff_rate = model.predict(frame, mask=env_mask)
        # label, detections, diff_rate = model.predict(frame)

        print('Processing video...%s remain (Speed: %.2f FPS; Progress: %d%% - %d/%d)' % (time.strftime('%H:%M:%S', time.gmtime(int((length-count)/(1000 / model.current_pred_time)))), 1000 / model.current_pred_time, count * 100 / length, count, length), end="\r")
        # frame = visualize(frame, label, detections, diff_rate, model.current_classify_time, model.current_bs_time, model.current_detect_time, model.current_pred_time, display=False, output_size=(512,512))
        frame = visualize(frame, label, detections, diff_rate, model.current_classify_time, model.current_bs_time, model.current_detect_time, model.current_pred_time, display=False, output_size=(640,480))
            
        # cv2.imshow("Predicted", frame)
        writer.write(frame)
        count += 1

        key = cv2.waitKey(30)
        if key == 27:
            break
        # elif key == ord('a'):
        #     print(mouseX,mouseY)
    else:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

# visualize(detections, cv2.resize(image, (512, 512)), display=True)
# plt.show()

