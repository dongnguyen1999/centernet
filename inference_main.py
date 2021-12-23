
from inference.utils import create_mask, get_masked_img, visualize
from inference.models.model import Model, create_models
from inference.models.frame_difference import FrameDiffEstimator
from utils.config import Config
import cv2
import numpy as np
import tensorflow as tf
# main()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

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

# from crowd_classification.models.cnn import create_model as create_classify_model
# from centernet_detect.models.hourglass import create_model as create_detect_model
# from centernet_count.models.hourglass import create_model as create_count_model
# from inference.models.dtdecode import CtDetDecode
# from inference.models.countdecode import CountDecode

# crowd_model = create_classify_model(config, architecture="pretrained_vgg16", freeze_feature_block=False)
# crowd_model.load_weights(r"D:\flask_video_stream\inference\weight\vgg16_fineturning_epoch2.hdf5")
# crowd_model.save('vgg16_fineturning_epoch2_model.h5')

# crowd_model_focalbin = create_classify_model(config, architecture="pretrained_vgg16", freeze_feature_block=False)
# crowd_model_focalbin.load_weights(r"D:\flask_video_stream\inference\weight\vgg16_fineturning_focalbin_epoch10.hdf5")
# crowd_model_focalbin.save('vgg16_fineturning_focalbin_epoch10_model.h5')

# detect_model = create_detect_model(config, num_stacks=1)
# detect_model.load_weights(r"D:\flask_video_stream\inference\weight\detect_hg1stack_tfmosaic_epoch6.hdf5")
# detect_model = CtDetDecode(detect_model)
# detect_model.save('detect_hg1stack_tfmosaic_epoch6_model.h5')

# count_model = create_count_model(config, num_stacks=1)
# count_model.load_weights(r"D:\flask_video_stream\inference\weight\count_hg1stack_epoch6.hdf5")
# count_model = CountDecode(count_model)
# count_model.save('count_hg1stack_epoch6_model.h5')

# count_model = create_count_model(config, num_stacks=2)
# count_model.load_weights(r"D:\flask_video_stream\inference\weight\count_hg2stack_epoch6.hdf5")
# count_model = CountDecode(count_model)
# count_model.save('count_hg2stack_epoch6_model.h5')


vgg16_fineturning_weights = r'D:\centernet\inference\weights\vgg16_fineturning_epoch2.hdf5'
detect_hg1stack_tfmosaic_weights = r'D:\centernet\inference\weights\detect_hg1stack_tfmosaic_epoch6.hdf5'
vgg16_fineturning_focalbin_weights = r'D:\centernet\inference\weights\vgg16_fineturning_focalbin_epoch10.hdf5'
count_hg1stack_weights = r'D:\centernet\inference\weights\count_hg1stack_epoch6.hdf5'
count_hg2stack_weights = r'D:\centernet\inference\weights\count_hg2stack_epoch6.hdf5'

model_configs = {
    'crowd_model': {
        'architecture': 'ClsVgg16',
        'weights': vgg16_fineturning_weights
    },
    'crowd_model_focalbin': {
        'architecture': 'ClsVgg16',
        'weights': vgg16_fineturning_focalbin_weights
    },
    'count_model_1stack': {
        'architecture': 'HmOnlyHourglass1Stack',
        'weights': count_hg1stack_weights
    },
    'count_model_2stack': {
        'architecture': 'HmOnlyHourglass2Stack',
        'weights': count_hg2stack_weights
    },
    'detect_model': {
        'architecture': 'DtHourglass1Stack',
        'weights': detect_hg1stack_tfmosaic_weights
    },
}

crowd_model_config = model_configs['crowd_model']
count_model_config = model_configs['count_model_1stack']

crowd_model, count_model = create_models(crowd_model_config, count_model_config)

frame_diff_estimator = FrameDiffEstimator(debug=True)


is_hmonly_model = 'HmOnly' in count_model_config['architecture']
model = Model(crowd_model, count_model, frame_diff_estimator, debug=True, heatmap_only=is_hmonly_model)

# image = cv2.imread(r"C:\vehicle-data\video1_all\train\video1_1.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# print(image.shape)

# y_pred, detections = model.predict(image)
# print(y_pred, detections)


# cap = cv2.VideoCapture(r"C:\vehicle-data\raw\DLngoaiBenhVien\video1.avi")
# cap = cv2.VideoCapture(r"C:\Users\nvdkg\Documents\Camtasia\dnnew_crowd.autosave\dnnew_crowd.autosave.mp4")

# cap = cv2.VideoCapture(r"C:\Users\nvdkg\Documents\Camtasia\Small cut\Small cut.mp4")
# cap = cv2.VideoCapture(r"C:\vehicle-data\raw\DLngoaiBenhVien\video1.avi")
# cap = cv2.VideoCapture(r"C:\Users\nvdkg\Documents\Camtasia\test3_crowd\test3_crowd.mp4")
# cap = cv2.VideoCapture(r"C:\Users\nvdkg\Documents\Camtasia\dnnew_crowd.autosave\dnnew_crowd.autosave.mp4")

cap = cv2.VideoCapture(r"D:\dnnew_crowd.mp4")
writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 15, (512,512))

# cap = cv2.VideoCapture(r'C:\Users\nvdkg\Documents\Camtasia\test3_crowd\test3_crowd.mp4')

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
count = 0

# mouseX,mouseY = 0,0
# def get_mouse_click_coord(event, x, y, flags, param):
#     global mouseX,mouseY
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         mouseX,mouseY = x,y

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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (512, 512))
        
        # cv2.imshow("Frame", frame)

        label, result = model.predict(frame, mask=env_mask)
        # label, detections, diff_rate = model.predict(frame)

        # print('Processing video...%s remain (Speed: %.2f FPS; Progress: %d%% - %d/%d)' % (time.strftime('%H:%M:%S', time.gmtime(int((length-count)/(1000 / model.current_pred_time)))), 1000 / model.current_pred_time, count * 100 / length, count, length), end="\r")
        frame, result = visualize(frame, label, result, model.current_classify_time, model.current_bs_time, model.current_count_time, model.current_pred_time, display=False)
        print(result)            
        cv2.imshow("Predicted", frame)
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

