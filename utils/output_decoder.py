from utils.config import Config
import numpy as np
import cv2
import matplotlib.pyplot as plt

class OutputDecoder:
    def __init__(self, config: Config, score_threshold=0.5, iou_threshold=0.5):
        self.config = config
        self.num_classes = config.num_classes
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def decode_boxes(self, score, y_c, x_c, height, width):
        # flatten
        score = score.reshape(-1)
        y_c = y_c.reshape(-1)
        x_c = x_c.reshape(-1)
        height = height.reshape(-1)
        width = width.reshape(-1)
        size = height*width
        if len(score) == 0: return np.array([])


        top = y_c - height/2
        left = x_c - width/2
        bottom = y_c + height/2
        right= x_c + width/2

        inside_pic = (top > 0) * (left > 0) * (bottom < self.output_size) * (right < self.output_size)
        # outside_pic = len(inside_pic) - np.sum(inside_pic)
        # if outside_pic>0:
        #     print("{} boxes are out of picture".format(outside_pic))

        normal_size = (size < (np.mean(size[size > 0])*20)) * (size > (np.mean(size[size > 0])/20))
        # print('lentop', len(score[inside_pic*normal_size]), np.mean(size[size > 0])*50)
        # print('lentop', normal_size, size[normal_size == False])
        score = score[inside_pic*normal_size]
        top = top[inside_pic*normal_size]
        left = left[inside_pic*normal_size]
        bottom = bottom[inside_pic*normal_size]
        right = right[inside_pic*normal_size]

        

        #sort  
        score_sort = np.argsort(score)[::-1]
        score = score[score_sort]  
        top = top[score_sort]
        left = left[score_sort]
        bottom = bottom[score_sort]
        right = right[score_sort]

        area = ((bottom-top)*(right-left))

        boxes = np.concatenate((score.reshape(-1,1), top.reshape(-1,1), left.reshape(-1,1), bottom.reshape(-1,1), right.reshape(-1,1)), axis=1)

        box_idx = np.arange(len(top))
        alive_box = []
        while len(box_idx) > 0:
            alive_box.append(box_idx[0])

            y1 = np.maximum(top[0],top)
            x1 = np.maximum(left[0],left)
            y2 = np.minimum(bottom[0],bottom)
            x2 = np.minimum(right[0],right)

            cross_h=np.maximum(0,y2-y1)
            cross_w=np.maximum(0,x2-x1)

            still_alive=(((cross_h*cross_w)/area[0]) < self.iou_threshold)

            if np.sum(still_alive)==len(box_idx):
                print("error")
                print(np.max((cross_h*cross_w)), area[0])

            top=top[still_alive]
            left=left[still_alive]
            bottom=bottom[still_alive]
            right=right[still_alive]
            area=area[still_alive]
            box_idx=box_idx[still_alive]

        return boxes[alive_box] #[score, top, left, bottom, right]

    def decode_y_pred(self, y_pred):
        y_c = y_pred[..., self.num_classes] + np.arange(self.output_size).reshape(-1,1)
        x_c = y_pred[..., self.num_classes+1] + np.arange(self.output_size).reshape(1,-1)
        height = y_pred[..., self.num_classes+2] * self.output_size
        width = y_pred[..., self.num_classes+3] * self.output_size
        # print(height[height > 0])
        count = 0
        for category in range(self.num_classes):
            category_heatmap = y_pred[..., category]
            mask = (category_heatmap > self.score_threshold)
            # print("category", category, "boxes num", np.sum(mask))
            score_boxes = self.decode_boxes(category_heatmap[mask], y_c[mask], x_c[mask], height[mask], width[mask])
            if np.size(score_boxes) != 0:
                score_boxes = np.insert(score_boxes, 0, category, axis=1) #category,score,top,left,bottom,right
                if count == 0:
                    all_score_boxes = score_boxes
                else:
                    all_score_boxes = np.concatenate((all_score_boxes, score_boxes),axis=0)
                count+=1
        if count != 0:
            score_sort = np.argsort(all_score_boxes[:, 1])[::-1] #sort by score
            all_score_boxes = all_score_boxes[score_sort]
            # print(all_score_boxes)
            _,unique_idx = np.unique(all_score_boxes[:, 2], return_index=True)
            return all_score_boxes[sorted(unique_idx)] #[category, score, top, left, bottom, right]
        return np.array([])

    def decode_y_true(self, y_true):
        y_c = y_true[..., self.num_classes] + np.arange(self.output_size).reshape(-1,1)
        x_c = y_true[..., self.num_classes+1] + np.arange(self.output_size).reshape(1,-1)
        height = y_true[..., self.num_classes+2] * self.output_size
        width = y_true[..., self.num_classes+3] * self.output_size

        count = 0
        for category in range(self.num_classes):
            category_centerpoint = y_true[..., self.num_classes+4 + category]
            mask = np.sign(category_centerpoint).astype(np.bool)
            # print("category", category, "boxes num", np.sum(mask))

            _y_c, _x_c, _height, _width =  (y_c[mask], x_c[mask], height[mask], width[mask])
            _y_c = _y_c.reshape(-1)
            _x_c = _x_c.reshape(-1)
            _height = _height.reshape(-1)
            _width = _width.reshape(-1)

            top = _y_c - _height/2
            left = _x_c - _width/2
            bottom = _y_c + _height/2
            right= _x_c + _width/2

            score_boxes = np.concatenate((np.ones(top.shape).reshape(-1, 1), top.reshape(-1,1), left.reshape(-1,1), bottom.reshape(-1,1), right.reshape(-1,1)), axis=1)
            # print(score_boxes)
            if np.size(score_boxes) != 0:
                score_boxes = np.insert(score_boxes, 0, category, axis=1) #category,score,top,left,bottom,right
                if count == 0:
                    all_score_boxes = score_boxes
                else:
                    all_score_boxes = np.concatenate((all_score_boxes, score_boxes),axis=0)
                count+=1
        
        score_sort = np.argsort(all_score_boxes[:, 1])[::-1] #sort by score
        all_score_boxes = all_score_boxes[score_sort]
        # print(all_score_boxes)

        _,unique_idx = np.unique(all_score_boxes[:, 2], return_index=True)
        return all_score_boxes[sorted(unique_idx)] #[category, score, top, left, bottom, right]

    def visualize(self, box_and_score, img, max_nb_object=100, le=None, display=False):
        boxes = []
        scores = []
        color_scheme = [(0,0,255), (255,0,0), (0,255,255), (0,127,127), (127,255,127), (255,255,0)]
        number_of_rect = np.minimum(max_nb_object, len(box_and_score))

        for i in reversed(list(range(number_of_rect))):
            predicted_class, score, top, left, bottom, right = box_and_score[i, :]
            # top = np.floor(top + 0.5).astype('int32')
            # left = np.floor(left + 0.5).astype('int32')
            # bottom = np.floor(bottom + 0.5).astype('int32')
            # right = np.floor(right + 0.5).astype('int32')
            top = np.floor(top * self.input_size / self.output_size).astype('int32')
            left = np.floor(left * self.input_size / self.output_size).astype('int32')
            bottom = np.floor(bottom * self.input_size / self.output_size).astype('int32')
            right = np.floor(right * self.input_size / self.output_size).astype('int32')
            predicted_class = int(predicted_class)
            label = '{:.2f}'.format(score)
            if le != None:
                class_name = le.inverse_transform([predicted_class])[0]
                label = '{class_name} {label}'.format(class_name = class_name, label = label)

            #print(label)
            #print(top, left, right, bottom)
            cv2.rectangle(img, (left, top), (right, bottom), color_scheme[predicted_class], 2)
            cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX ,  
                        0.5, color_scheme[predicted_class], 2, cv2.LINE_AA) 
            boxes.append([left, top, right-left, bottom-top])
            scores.append(score)
            
        if display == True:
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            ax.set_axis_off()
            ax.imshow(img)

        return np.array(boxes), np.array(scores)