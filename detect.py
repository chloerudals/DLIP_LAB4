''' -------------------
Created Date : 17/06/2021
Edited Date  : 21/06/2021

Original file  : https://github.com/ultralytics/yolov5/blob/master/detect.py
Referenced file: https://github.com/dudesparsh/Parking_detector/blob/master/identify_parking_spots.ipynb

Edited by JiHoon Kang and Kyungmin Do
-------------------- '''

from __future__ import division
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


import matplotlib.pyplot as plt
import glob
import numpy as np
import math

global arr_x1; global arr_x2; global arr_x3; global arr_x4
global arr_y1; global arr_y2; global arr_y3; global arr_y4

@torch.no_grad()
def detect(opt):
    CountCar = 0
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    start_time = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels') # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    if names[int(c)] == 'car':
                        CountCar = n
                
                # file open & save
                if save_txt:
                    with open(txt_path + '/counting_result.txt', 'a') as f:
                        f.write(f'{frame}, {CountCar}\n')
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                        if opt.save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                spot_cnt = 1
                Parked_Space_Array = []
                for px1, py1, px2, py2, px3, py3, px4, py4 in zip(arr_x1, arr_y1, arr_x2, arr_y2, arr_x3, arr_y3, arr_x4, arr_y4):
                    poly_points = np.array([[px1, py1], [px2, py2], [px3, py3], [px4, py4]], dtype = np.int32).reshape((-1,1,2))
                    text_coordinate = int((px1 + px4)/2 - 10) , int(py1 - 30)
                    if spot_cnt < 10:
                        box_w1 = 10; box_h1 = 30; box_w2 = 30; box_h2 = 10
                        box_x1, box_y1, box_x2, box_y2 = int((px1 + px4)/2 - 10 - box_w1), int(py1 - 30 - box_h1), int((px1 + px4)/2 - 10 + box_w2), int(py1 - 30 + box_h2)
                    else:
                        box_w1 = 10; box_h1 = 30; box_w2 = 50; box_h2 = 10
                        box_x1, box_y1, box_x2, box_y2 = int((px1 + px4)/2 - 10 - box_w1), int(py1 - 30 - box_h1), int((px1 + px4)/2 - 10 + box_w2), int(py1 - 30 + box_h2)
                    cv2.rectangle(im0, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)
                    pcx, pcy = abs(px3 + px1) / 2, abs(py3 + py1) / 2
                    cv2.circle(im0, (int(pcx), int(pcy)), 5, (255, 255, 255), 2)

                    for *xyxy, conf, cls in reversed(det):
                        bx1, by1, bx2, by2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                        bcx, bcy = abs(bx2 + bx1) / 2, abs(by2 + by1) / 2 + 10
                        cv2.circle(im0, (int(bcx), int(bcy)), 5, (255, 255, 255), 2)
                        parking_distance = math.sqrt((bcx - pcx)**2 + (bcy - pcy)**2)
                        
                        if parking_distance < 40:
                            cv2.polylines(im0, [poly_points], 1, [0, 0, 255], 2)
                            cv2.line(im0, (int(bcx), int(bcy)), (int(pcx), int(pcy)), (255, 255, 255), 2)
                            cv2.putText(im0, "%d" %spot_cnt, text_coordinate, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            Parked_Space_Array.append(spot_cnt)
                            break
                    
                    spot_cnt += 1
                
                Empty_Space_Array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                for P in Parked_Space_Array:
                    Empty_Space_Array.remove(P)
                
                adj = 5
                for idx in Empty_Space_Array:
                    idx = idx-1
                    px1, py1, px2, py2, px3, py3, px4, py4 = arr_x1[idx], arr_y1[idx], arr_x2[idx], arr_y2[idx], arr_x3[idx], arr_y3[idx], arr_x4[idx], arr_y4[idx]
                    poly_points = np.array([[px1, py1], [px2, py2], [px3, py3], [px4, py4]], dtype = np.int32).reshape((-1,1,2))
                    fillPoly_points = np.array([[px1+adj, py1+adj], [px2+adj, py2-adj], [px3-adj, py3-adj], [px4-adj, py4+adj]], dtype = np.int32).reshape((-1,1,2))
                    text_coordinate = int((px1 + px4)/2 - 10) , int(py1 - 30)
                    cv2.putText(im0, "%d" %(idx+1), text_coordinate, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.polylines(im0, [poly_points], 1, [255, 255, 255], 2)
                    cv2.fillPoly(im0, [fillPoly_points], [0, 200, 0])
                    
                EmptySpace = len(Empty_Space_Array)
                Entering = CountCar - len(Parked_Space_Array) - 1
                if Entering > 0:
                    if Entering == 1:
                        EnteringStr = "{0} car is entering to the parking lot".format(Entering)
                        cv2.putText(im0, EnteringStr, (770, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    else:
                        EnteringStr = "{0} cars are entering to the parking lot".format(Entering)
                        cv2.putText(im0, EnteringStr, (735, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                Empty_Space_Array = [str(int) for int in Empty_Space_Array]
                strEmpty = "Available parking spaces: "+", ".join(Empty_Space_Array)
                cv2.putText(im0, strEmpty, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                current_time = time.time()
                videoFPS = 1 / (current_time - start_time)
                strFPS = f"FPS : {videoFPS:3.0f}"
                strSpace = "Empty Space : {0}".format(EmptySpace)
                cv2.putText(im0, strFPS, (1100, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(im0, strSpace, (1010, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) & 0xFF == ord('q'): # 1 millisecond
                    break
                start_time = current_time

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

# image is expected be in RGB color space# image 
def select_rgb_white_yellow(image): 
    # white color mask
    lower = np.uint8([120, 120, 120])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([190, 190,   0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def detect_edges(image, low_threshold=500, high_threshold=1000):
    return cv2.Canny(image, low_threshold, high_threshold)

def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image, mask)
 
def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    pt_1  = [cols*0.0, rows*0.43]
    pt_2 = [cols*0.0, rows*0.64]
    pt_3 = [cols*1, rows*0.6]
    pt_4 = [cols*1, rows*0.43]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[pt_1, pt_2, pt_3, pt_4]], dtype=np.int32)
    return filter_region(image, vertices)

def hough_lines(image):
    """
    `image` should be the output of a Canny transform.
    
    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=30, minLineLength=30, maxLineGap=10)

def draw_lines(image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
    # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
    if make_copy:
        image = np.copy(image) # don't want to modify the original
    cleaned = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if abs(y2-y1) >=10 and abs(x2-x1) >=0 and abs(x2-x1) <= 10000:
                cleaned.append((x1,y1,x2,y2))
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    print(" No lines detected: ", len(cleaned))
    return image

def identify_blocks(image, lines, make_copy=True):
    if make_copy:
        new_image = np.copy(image)
    #Step 1: Create a clean list of lines
    cleaned = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if abs(y2-y1) >=10 and abs(x2-x1) >=0 and abs(x2-x1) <= 100000:
                cleaned.append((x1,y1,x2,y2))
    
    #Step 2: Sort cleaned by y1 position
    import operator
    list1 = sorted(cleaned, key=operator.itemgetter(1, 1))
    
    #Step 3: Find clusters of y1 close together - clust_dist apart
    clusters = {}
    dIndex = 0
    clus_dist = 1000

    for i in range(len(list1) - 1):
        distance = abs(list1[i+1][1] - list1[i][1])
        if distance <= clus_dist:
            if not dIndex in clusters.keys(): clusters[dIndex] = []
            clusters[dIndex].append(list1[i])
            clusters[dIndex].append(list1[i + 1])

        else:
            dIndex += 1
    
    #Step 4: Identify coordinates of rectangle around this cluster
    rects = {}
    i = 0
    for key in clusters:
        all_list = clusters[key]
        cleaned = list(set(all_list))
        if len(cleaned) > 10:
            cleaned = sorted(cleaned, key=lambda tup: tup[0])
            avg_x1 = cleaned[0][2]
            avg_x2 = cleaned[-1][2]
            avg_y1 = 0
            avg_y2 = 0
            for tup in cleaned:
                if tup[1] > tup[-1]:
                    avg_y1 += tup[-1]
                    avg_y2 += tup[1]
                else:
                    avg_y1 += tup[1]
                    avg_y2 += tup[-1]

            avg_y1 = avg_y1/len(cleaned)
            avg_y2 = avg_y2/len(cleaned)
            rects[i] = (avg_x1, avg_y1, avg_x2, avg_y2)
            i += 1
    
    #Step 5: Draw the rectangles on the image
    buff = 5
    for key in rects:
        tup_topLeft = (int(rects[key][0]), int(rects[key][1]- buff))
        tup_botRight = (int(rects[key][2]), int(rects[key][3] + buff))
        cv2.rectangle(new_image, tup_topLeft, tup_botRight, (0,255,0), 2)
    return new_image, rects

def draw_parking(image, rects, make_copy = True, color=[255, 0, 0], thickness=1, save = True):
    if make_copy:
        new_image = np.copy(image)
    rows, cols = image.shape[:2]
    gap = cols / 12 - 0.1
    adj_gap_x1 = {0:40, 1:35, 2:31, 3:24, 4:15, 5:7, 6:-2, 7:-10, 8:-19, 9:-28, 10:-36, 11:-45, 12:-51 }
    adj_gap_x2 = {0:-10, 1:-10, 2:-7, 3:-4, 4:-3, 5:-3, 6:-2, 7:0, 8:0, 9:1, 10:3, 11:3, 12:4 }
    spot_dict = {} # maps each parking ID to its coords
    poly_dict = {}
    tot_spots = 0

    adj_x1 = {0:(0 - rects[0][0])}
    adj_x2 = {0:(cols - rects[0][2])}

    adj_y1 = {0:-5}
    adj_y2 = {0:5}

    for key in rects:
        # Horizontal lines
        tup = rects[key]
        x1 = int(tup[0]+ adj_x1[key])
        x2 = int(tup[2]+ adj_x2[key])
        y1 = int(tup[1] + adj_y1[key])
        y2 = int(tup[3] + adj_y2[key])
        cv2.rectangle(new_image, (x1, y1),(x2,y2),(0,255,0),2)
        num_splits = int(abs(x2-x1)//gap)
        for i in range(0, num_splits+1):
            x = int(x1 + i*gap)
            cv2.line(new_image, (x + adj_gap_x1[i], y1), (x + adj_gap_x2[i], y2), color, thickness)
        if key > 0 and key < len(rects) -1 :        
            #draw vertical lines
            y = int((y1 + y2)/2)
            cv2.line(new_image, (x1, y), (x2, y), color, thickness)
        # Add up spots in this lane
        if key == 0 or key == (len(rects) -1):
            tot_spots += num_splits +1
        else:
            tot_spots += 2*(num_splits +1)
            
        # Dictionary of spot positions
        if key == 0 or key == (len(rects) -1):
            for i in range(0, num_splits):
                cur_len = len(spot_dict)
                x = int(x1 + i*gap)
                spot_dict[(x + adj_gap_x1[i], y1, x + gap + adj_gap_x2[i+1], y2)] = cur_len +1
                poly_dict[(x + adj_gap_x1[i], y1, x + adj_gap_x2[i], y2, x + gap + adj_gap_x2[i+1], y2, x + gap + adj_gap_x1[i+1], y1)] = cur_len +1
        else:
            for i in range(0, num_splits+1):
                cur_len = len(spot_dict)
                x = int(x1 + i*gap)
                y = int((y1 + y2)/2)
                spot_dict[(x, y1, x+gap, y)] = cur_len +1
                spot_dict[(x, y, x+gap, y2)] = cur_len +2   
    
    print("total parking spaces: ", tot_spots, cur_len)
    if save:
        filename = 'with_parking.jpg'
        cv2.imwrite(filename, new_image)
    return new_image, poly_dict

if __name__ == '__main__':
    # line detection
    test_images = [plt.imread(path) for path in glob.glob('data/images/Img.jpg')]
    white_yellow_images = list(map(select_rgb_white_yellow, test_images))
    gray_images = list(map(convert_gray_scale, white_yellow_images))
    edge_images = list(map(lambda image: detect_edges(image), gray_images))
    roi_images = list(map(select_region, edge_images)) # images showing the region of interest only
    list_of_lines = list(map(hough_lines, roi_images))
    line_images = []
    for image, lines in zip(test_images, list_of_lines):
        line_images.append(draw_lines(image, lines))
    # images showing the region of interest only
    rect_images = []
    rect_coords = []
    for image, lines in zip(test_images, list_of_lines):
        new_image, rects = identify_blocks(image, lines)
        rect_images.append(new_image)
        rect_coords.append(rects)

    delineated = []
    spot_pos = []
    for image, rects in zip(test_images, rect_coords):
        new_image, poly_dict = draw_parking(image, rects)
        delineated.append(new_image)
        spot_pos.append(poly_dict)
    final_spot_dict = spot_pos[0]
    
    arr_x1 = []; arr_x2 = []; arr_x3 = []; arr_x4 = []
    arr_y1 = []; arr_y2 = []; arr_y3 = []; arr_y4 = []
    scaling_x = 1280 / 602; scaling_y = 720 / 338
    for spot in final_spot_dict.keys():
        (x1, y1, x2, y2, x3, y3, x4, y4) = spot
        arr_x1.append(x1 * scaling_x + 2); arr_x2.append(x2 * scaling_x + 2); arr_x3.append(x3 * scaling_x - 2); arr_x4.append(x4 * scaling_x- 2)
        arr_y1.append(y1 * scaling_y); arr_y2.append(y2 * scaling_y); arr_y3.append(y3 * scaling_y); arr_y4.append(y4 * scaling_y)
    arr_x1 = np.array(arr_x1); arr_x2 = np.array(arr_x2); arr_x3 = np.array(arr_x3); arr_x4 = np.array(arr_x4)
    arr_y1 = np.array(arr_y1); arr_y2 = np.array(arr_y2); arr_y3 = np.array(arr_y3); arr_y4 = np.array(arr_y4)
    
    # object detection
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='Parking_management_test_video_short.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_false', help='display results')
    parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, default=2, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    if opt.update:  # update all models (to fix SourceChangeWarning)
        for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            detect(opt=opt)
            strip_optimizer(opt.weights)
    else:
        detect(opt=opt)
