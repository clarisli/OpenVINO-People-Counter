"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from time import time
from inference import Network
from counter import PeopleCounter
from yolo import YoloDetector


# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

log.basicConfig(level=log.INFO)


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("--labels", type=str, default=None, help="Optional. Labels mapping file")
    parser.add_argument("-r", "--raw_output_message", help="Optional. Output inference results raw values showing",
                      default=False, action="store_true")
    parser.add_argument("-iout", "--iou_threshold", help="Optional. Intersection over union threshold for overlapping "
                                                       "detections filtering", default=0.4, type=float)
    return parser



def connect_mqtt():
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    object_detector = YoloDetector()
    log.debug('object detector {}'.format(object_detector))

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device)
    n, c, h, w = infer_network.get_input_shape()
    labels_map = get_labels(args.labels)

    ### Handle the input stream ###
    input_stream = get_input_stream(args.input)
    cap = init_video_capture(input_stream)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    dist_threshold = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 5
    counter = PeopleCounter(dist_threshold, fps)
    
    single_image_mode = args.input.endswith('.jpg') or args.input.endswith('.bmp')
    wait_key_code = 1
    request_id = 0

    ### Loop until stream is over ###
    while cap.isOpened():
        ### Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        
        ### Pre-process the image as needed ###
        in_frame = preprocess_frame(frame, n,c,h,w)
        
        ### Start asynchronous inference for specified request ###
        start_time = time()
        infer_network.exec_net(in_frame, request_id)
        det_time = time() - start_time
        
        ### Wait for the result ###
        if infer_network.wait(request_id) == 0:
             ### Get the results of the inference request ###
            output = infer_network.get_output(request_id)
            qualified_objects = object_detector.get_qualified_objects(output, infer_network,  prob_threshold, args.iou_threshold, in_frame.shape[2:],
                                             frame.shape[:-1])
            frame = draw_boxes(frame, qualified_objects, args.raw_output_message, labels_map)

            ### Extract any desired stats from the results ###
            ### Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            is_new_entry = counter.is_new_entry(qualified_objects)
            if is_new_entry:
                client.publish("person", json.dumps({"total": counter.total_count}))
            new_exit_durations = counter.get_new_exit_durations()
            for duration in new_exit_durations:
                client.publish("person/duration", json.dumps({"duration": duration}))
            client.publish("person", json.dumps({"count": len(qualified_objects)}))    
        
            frame = draw_performance_stats(frame, det_time)
            counter.increment_frame_count()
        ### Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
        ### Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)


        key_pressed = cv2.waitKey(wait_key_code)
        if key_pressed == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

def draw_performance_stats(frame, det_time):
    # Draw performance stats over frame
    inf_time_message = "Inference time: {:.3f} ms".format(det_time * 1e3)
    cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
    log.debug(inf_time_message)
    return frame


def get_input_stream(args_input):
    if args_input == 'CAM':
        input_stream = 0
    else:
        assert os.path.isfile(args_input), "File doesn't exist"
        input_stream = args_input
    return input_stream

def init_video_capture(input_stream):
    cap = cv2.VideoCapture(input_stream)
    if input_stream:
        cap.open(input_stream)
    if not cap.isOpened():
        log.error("ERROR: failed to open the video file")
    return cap

def preprocess_frame(frame, n, c, h, w):
    p_frame = cv2.resize(frame, (w,h))
    p_frame = p_frame.transpose((2,0,1))
    p_frame = p_frame.reshape((n, c, h, w))
    return p_frame


def draw_boxes(frame, objects, args_raw_output_message, labels_map):
    if len(objects) and args_raw_output_message:
        log.info("\nDetected boxes for batch {}:".format(1))
        log.info(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")

    origin_im_size = frame.shape[:-1]
    for obj in objects:
        # Validation bbox of detected object
        if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
            continue
        color = (int(min(obj['class_id'] * 12.5, 255)),
                min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
        det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
            str(obj['class_id'])

        if args_raw_output_message:
            log.info(
                "{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(det_label, obj['confidence'], obj['xmin'],
                                                                            obj['ymin'], obj['xmax'], obj['ymax'],
                                                                            color))

        cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
        cv2.putText(frame,
                    "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
                    (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
    return frame

def get_labels(args_labels):
    if args_labels:
        with open(args_labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None
    return labels_map

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
