import numpy as np
from mtcnn import MTCNN


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def aligner():
    return MTCNN()

# Margin is % larger to make box around face
def align(orig_img, aligner, margin=0.1, detect_multiple_faces=True):
    """ run MTCNN face detector """

    if orig_img.ndim < 2:
        return None
    if orig_img.ndim == 2:
        orig_img = to_rgb(orig_img)
    orig_img = orig_img[:, :, 0:3]

    bounding_boxes = aligner.detect_faces(orig_img)
    nrof_faces= len(bounding_boxes)

    if nrof_faces > 0:
        det = bounding_boxes[0]['box']
        det_arr = []
        img_size = np.asarray(orig_img.shape)[0:2]
        if nrof_faces > 1:
            if detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(bounding_boxes[i]['box']))
            else:
                bounding_box_size = (det[1] + det[3])
                img_center = img_size / 2
                offsets = np.vstack([(det[0] + det[2]) / 2 - img_center[1],
                                     (det[1] + det[3]) / 2 - img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                det_arr.append(det[index, :])
        else:
            det_arr.append(np.squeeze(det))

        cropped_arr = []
        bounding_boxes_arr = []
        for i, det in enumerate(det_arr):
            det = np.squeeze(det)

            x, y, width, height = det

            bb = np.zeros(4, dtype=np.int32)

            margin_x = width * margin // 2
            margin_y = height * margin // 2

            bb[0] = max(x - margin_x, 0)
            bb[1] = max(y - margin_y, 0)
            bb[2] = min(x + width + margin_x, img_size[1])
            bb[3] = min(y + height + margin_y, img_size[0])
            
            # Don't let them overlap!
            bb[2] = max(bb[0], bb[2])
            bb[3] = max(bb[1], bb[3])
            
            cropped = orig_img[bb[1]:bb[3], bb[0]:bb[2],:]
            cropped_arr.append(cropped)
            bounding_boxes_arr.append([bb[0], bb[1], bb[2], bb[3]])
        return cropped_arr, bounding_boxes_arr
    else:
        return None