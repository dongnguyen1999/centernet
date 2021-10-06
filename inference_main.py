from inference.main import main
# from inference.model.decode import CtDetDecode, CountDecode
from centernet_detect.dataset.vn_vehicle import DataGenerator, load_data
from inference.model.decode import visualize
from inference.model.model import Model, create_models
from inference.model.bg_subtraction import BackgroundSubtractorMOG2
from utils.config import Config
import cv2
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
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

model = Model(crowd_model, detect_model, background_subtractor, 0.5, 0.25, [5, 15], [0.03, 0.08])

# image = cv2.imread(r"C:\vehicle-data\video1_all\train\video1_1.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# print(image.shape)

# y_pred, detections = model.predict(image)
# print(y_pred, detections)


# cap = cv2.VideoCapture(r"C:\vehicle-data\raw\DLngoaiBenhVien\video1.avi")
# cap = cv2.VideoCapture(r"C:\Users\nvdkg\Documents\Camtasia\dnnew_crowd.autosave\dnnew_crowd.autosave.mp4")
cap = cv2.VideoCapture(r"C:\Users\nvdkg\Documents\Camtasia\Small cut\Small cut.mp4")
# cap = cv2.VideoCapture(r'C:\Users\nvdkg\Documents\Camtasia\test3_crowd\test3_crowd.mp4')

while True:
    _, frame = cap.read()

    frame = cv2.resize(frame, (512, 512))
    # cv2.imshow("Frame", frame)

    label, detections = model.predict(frame)

    if label < 3:
        cv2.rectangle(frame, (1, 1), (511, 511), (0, 255, 0), 3)
        cv2.imshow("Predicted", visualize(detections, frame, display=False))
    else:
        cv2.rectangle(frame, (1, 1), (511, 511), (0, 0, 255), 3)
        cv2.imshow("Predicted", frame,)


    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# visualize(detections, cv2.resize(image, (512, 512)), display=True)
# plt.show()

