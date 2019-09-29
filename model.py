# Import libs
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# Create command line args
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input file")
ap.add_argument("-o", "--output", required=True, help="path to output file")
ap.add_argument("-m", "--mask-rcnn", required=True,
                help="base path to mask-rcnn directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="minimum threshold for pixel-wise weak segmentation")
args = vars(ap.parse_args())

# Get COCO classes
label_path = os.path.sep.join(
    [args["mask_rcnn"], "object_detection_classes_coco.txt"])
LABELS = open(label_path).read().strip().split("\n")

# Colors to present classes
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Get path for weights and config
weight_path = os.path.sep.join(
    [args["mask_rcnn"], "frozen_inference_graph.pb"])
config_path = os.path.sep.join(
    [args["mask_rcnn"], "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# Load model
print("Loading mask r-cnn from disk...")
model = cv2.dnn.readNetFromTensorflow(weight_path, config_path)

# Get video feed
vs = cv2.VideoCapture(args["input"])
writer = None

try:
    # Get approx. num of frames
    num_frames = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT

    total = int(vs.get(num_frames))
    print("{} total frames are in the video".format(total))
except:
    # Couldn't get number of frames
    print("Error getting the num of frames...")
    total = -1

# Go through the frames
while True:
    # Get current and next frame
    next_frame, frame = vs.read()

    # If next doesn't exist
    if not next_frame:
        break

    # Forward pass
    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    model.setInput(blob)

    # Timer
    start = time.time()
    (boxes, masks) = model.forward(["detection_out_final", "detection_masks"])
    end = time.time()

    # Go through num of detected objects
    for i in range(0, boxes.shape[2]):
        # Get id and confidences
        class_id = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        # Filter weak predictions
        if confidence > args["confidence"]:
            # Scale box to size of img
            (H, W) = frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            box_w = end_x - start_x
            box_h = end_y - start_y

            # Get pixel-wise segmentation
            mask = masks[i, class_id]
            mask = cv2.resize(mask, (box_w, box_h),
                              interpolation=cv2.INTER_NEAREST)
            mask = (mask > args["threshold"])

            # Get ROI of masked region
            roi = frame[start_y:end_y, start_x:end_x][mask]

            # Create overlay with ROI
            color = COLORS[class_id]
            combined = ((0.4 * color) + (0.6 * roi)).astype("uint8")

            # Store overlay with frame
            frame[start_y:end_y, start_x:end_x][mask] = combined

            # Draw box
            color = [int(c) for c in color]
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

            # Print pred label and confidence
            text = "{}: {:.4f}".format(LABELS[class_id], confidence)
            cv2.putText(frame, text, (start_x, start_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Check if video writer is available
    if writer is None:
        # Init writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (
            frame.shape[1], frame.shape[0]
        ), True)

        # Info for computing one frame
        if total > 0:
            elapsed = (end - start)

            print("One frame took {:.4f} seconds".format(elapsed))
            print("Estimated time to finish: {:.4f} ...".format(
                elapsed * total))

    # Write to disk
    writer.write(frame)

# Finish
print("Finishing up...")
writer.release()
vs.release()
