#!python3
"""
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Original *nix 2.7: https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version: https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py

@author: Philip Kahn, Aymeric Dujardin
@date: 20180911
"""
# pylint: disable=R, W0401, W0614, W0703
import os
import sys
import time
import logging
import random
from random import randint
import math
import statistics
import getopt
from ctypes import *
import numpy as np
import cv2
import pyzed.sl as sl
import gi
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import paho.mqtt.subscribe as subscribe
import time
import time, threading
os.system ("sudo /etc/init.d/qhmiserver stop")



# Application Variables
client = mqtt.Client("localhost", 1883, 60)
WAIT_SECONDS = 1
frame_numberx = 0
num_rectsx = 0
counter1 = 0
counter2 = 0
no_display = False
message = 'ON'
topic = 'ON'
payload = 'ON'
relay1 = 'OFF'
relay2 = 'OFF'
relay3 = 'OFF'
relay4 = 'OFF'
relay5 = 'OFF'
relay6 = 'OFF'
Object1 = 0
Object2 = 0
Object3 = 0
Object4 = 0
Object5 = 0
Object6 = 0
Object7 = 0
Object8 = 0
Object9 = 0
Object10 = 0
Object11 = 0
Object12 = 0
Object13 = 0
Object14 = 0
Object15 = 0
Object16 = 0
Object17 = 0
Object18 = 0
Object19 = 0
Object20 = 0
newValue1 = 0
newValue2 = 0
newValue3 = 0
newValue4 = 0
newValue5 = 0
newValue6 = 0
newValue7 = 0
newValue8 = 0
newValue9 = 0
newValue10 = 0
newValue11 = 0
newValue12 = 0
newValue13 = 0
newValue14 = 0
newValue15 = 0
newValue16 = 0
newValue17 = 10
newValue18 = 10
newValue19 = 10
newValue20 = 10
newValue21 = 10
newValue22 = 10




# Get the top-level logger object
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

######################################################################
# The callback for when the client receives a CONNACK response from the server.
######################################################################
def on_connect(client, userdata, flags, rc):
    
    client.subscribe("input/Device Input1", 0)
    client.subscribe("input/Device Input2", 0)
    client.subscribe("input/Device Input3", 0)
    client.subscribe("input/Device Input4", 0)
    client.subscribe("input/Device Input5", 0)
    client.subscribe("input/Device Input6", 0)
    client.subscribe("input/Device Input7", 0)
    client.subscribe("input/Device Input8", 0)
    client.subscribe("input/Device Input9", 0)
    client.subscribe("input/Device Input10", 0)
    client.subscribe("input/Device Input11", 0)
    client.subscribe("input/Device Input12", 0)
    client.subscribe("input/Device Input13", 0)
    client.subscribe("input/Device Input14", 0)
    client.subscribe("input/Device Input15", 0)
    client.subscribe("input/Device Input16", 0)
    client.subscribe("input/Device Input17", 0)
    client.subscribe("input/Device Input18", 0)
    client.subscribe("input/Device Input19", 0)
    client.subscribe("input/Device Input20", 0)
    client.subscribe("input/Device Input21", 0)
    client.subscribe("input/Device Input22", 0)
    print("rc: " + str(rc))
     
    msgs = [{'topic':"input/Device Input1", 'payload':newValue1},{'topic':"input/Device Input2", 'payload':newValue2},{'topic':"input/Device Input3", 'payload':newValue3},{'topic':"input/Device Input4", 'payload':newValue4},{'topic':"input/Device Input5", 'payload':newValue5},{'topic':"input/Device Input6", 'payload':newValue6},{'topic':"input/Device Input7", 'payload':newValue7},{'topic':"input/Device Input8", 'payload':newValue8},{'topic':"input/Device Input9", 'payload':newValue9},{'topic':"input/Device Input10", 'payload':newValue10},{'topic':"input/Device Input11", 'payload':newValue11},{'topic':"input/Device Input12", 'payload':newValue12},{'topic':"input/Device Input13", 'payload':newValue13},{'topic':"input/Device Input14", 'payload':newValue14},{'topic':"input/Device Input15", 'payload':newValue15},{'topic':"input/Device Input16", 'payload':newValue16},{'topic':"input/Device Input17", 'payload':newValue17},{'topic':"input/Device Input18", 'payload':newValue18},{'topic':"input/Device Input19", 'payload':newValue19},{'topic':"input/Device Input20", 'payload':newValue20},{'topic':"input/Device Input21", 'payload':newValue21},{'topic':"input/Device Input22", 'payload':newValue22}]

    publish.multiple(msgs, hostname="ubuntu")
    print ("Done publishingx")

######################################################################
# The callback for when a PUBLISH message is received from the server.
######################################################################

def on_message(mosq, obj, msg):
    global message
    print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))
    message = msg.payload
    client.publish("f2",msg.payload);
    global newValue1
    global newValue2
    global newValue3
    global newValue4
    global newValue5
    global newValue6
    global newValue7
    global newValue8
    global newValue9
    global newValue10
    global newValue11
    global newValue12
    global newValue13
    global newValue14
    global newValue15
    global newValue16
    global newValue17
    global newValue18
    global newValue19
    global newValue20
    global newValue21
    global newValue22
    
    
    payload =int(msg.payload)
    topic = msg.topic
    if topic == "input/Device Input1":
       newValue1 = payload
    elif topic == "input/Device Input2":
       newValue2 = payload
    elif topic == "input/Device Input3":
       newValue3 = payload
    elif topic == "input/Device Input4":
       newValue4 = payload
    elif topic == "input/Device Input5":
       newValue5 = payload
    elif topic == "input/Device Input6":
       newValue6 = payload
    elif topic == "input/Device Input7":
       newValue7 = payload  
    elif topic == "input/Device Input8":
       newValue8 = payload
    elif topic == "input/Device Input9":
       newValue9 = payload
    elif topic == "input/Device Input10":
       newValue10 = payload 
    elif topic == "input/Device Input11":
       newValue11 = payload
    elif topic == "input/Device Input12":
       newValue12 = payload  
    elif topic == "input/Device Input13":
       newValue13 = payload
    elif topic == "input/Device Input14":
       newValue14 = payload
    elif topic == "input/Device Input15":
       newValue15 = payload 
    elif topic == "input/Device Input16":
       newValue16 = payload 
    elif topic == "input/Device Input17":
       newValue17 = payload
    elif topic == "input/Device Input18":
       newValue18 = payload  
    elif topic == "input/Device Input19":
       newValue19 = payload
    elif topic == "input/Device Input20":
       newValue20 = payload
    elif topic == "input/Device Input21":
       newValue21 = payload 
    elif topic == "input/Device Input22":
       newValue22 = payload 
    print(topic)
   

#####################################################################
def on_publish(mosq, obj, mid):
    print("mid: " + str(mid))
######################################################################
def on_subscribe(mosq, obj, mid, granted_qos):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))
######################################################################
def on_log(mosq, obj, level, string):
    print(string)
######################################################################
# Publish the DBIRTH certificate
######################################################################


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1


def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("darknet.so", RTLD_GLOBAL)
hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                log.info("Flag value '"+tmp+"' not forcing CPU mode")
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError:
                pass
            # log.info(os.environ.keys())
            # log.warning("FORCE_CPU flag undefined, proceeding with GPU")
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            log.warning("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was
            # compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            log.warning("Environment variables indicated a CPU run, but we didn't find `" +
                        winNoGPUdll+"`. Trying a GPU run anyway.")
else:
    lib = CDLL("../libdarknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(
    c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            name_tag = meta.names[i]
        else:
            name_tag = altNames[i]
        res.append((name_tag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    """
    Performs the detection
    """
    custom_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    custom_image = cv2.resize(custom_image, (lib.network_width(
        net), lib.network_height(net)), interpolation=cv2.INTER_LINEAR)
    im, arr = array_to_image(custom_image)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(
        net, image.shape[1], image.shape[0], thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    res = []
    if debug:
        log.debug("about to range")
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    name_tag = meta.names[i]
                else:
                    name_tag = altNames[i]
                res.append((name_tag, dets[j].prob[i], (b.x, b.y, b.w, b.h), i))
    res = sorted(res, key=lambda x: -x[1])
    free_detections(dets, num)
    return res


netMain = None
metaMain = None
altNames = None


def get_object_depth(depth, bounds):
    '''
    Calculates the median x, y, z position of top slice(area_div) of point cloud
    in camera frame.
    Arguments:
        depth: Point cloud data of whole frame.
        bounds: Bounding box for object in pixels.
            bounds[0]: x-center
            bounds[1]: y-center
            bounds[2]: width of bounding box.
            bounds[3]: height of bounding box.

    Return:
        x, y, z: Location of object in meters.
    '''
    area_div = 2

    x_vect = []
    y_vect = []
    z_vect = []

    for j in range(int(bounds[0] - area_div), int(bounds[0] + area_div)):
        for i in range(int(bounds[1] - area_div), int(bounds[1] + area_div)):
            z = depth[i, j, 2]
            if not np.isnan(z) and not np.isinf(z):
                x_vect.append(depth[i, j, 0])
                y_vect.append(depth[i, j, 1])
                z_vect.append(z)
    try:
        x_median = statistics.median(x_vect)
        y_median = statistics.median(y_vect)
        z_median = statistics.median(z_vect)
    except Exception:
        x_median = -1
        y_median = -1
        z_median = -1
        pass

    return x_median, y_median, z_median


def generate_color(meta_path):
    '''
    Generate random colors for the number of classes mentioned in data file.
    Arguments:
    meta_path: Path to .data file.

    Return:
    color_array: RGB color codes for each class.
    '''
    
    
    random.seed(42)
    with open(meta_path, 'r') as f:
        content = f.readlines()
    class_num = int(content[0].split("=")[1])
    color_array = []
    for x in range(0, class_num):
        color_array.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    return color_array


def main(argv):
    global frame_numberx
    global num_rectsx
    global Object1
    global Object2
    global Object3
    global Object4
    global Object5
    global Object6
    global Object7
    global Object8
    global Object9
    global Object10
    global Object11
    global Object12
    global Object13
    global Object14
    global Object15
    global Object16
    global Object17
    global Object18
    global Object19
    global Object20
    global relay1
    global relay2
    global relay3
    global relay4
    global relay5
    global relay6
    global results
    
    
    thresh = 0.25
    darknet_path="../libdarknet/"
    config_path = darknet_path + "cfg/yolov4.cfg"
    weight_path = "yolov4.weights"
    meta_path = "coco.data"
    svo_path = None
    zed_id = 0

    help_str = 'darknet_zed.py -c <config> -w <weight> -m <meta> -t <threshold> -s <svo_file> -z <zed_id>'
    try:
        opts, args = getopt.getopt(
            argv, "hc:w:m:t:s:z:", ["config=", "weight=", "meta=", "threshold=", "svo_file=", "zed_id="])
    except getopt.GetoptError:
        log.exception(help_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            log.info(help_str)
            sys.exit()
        elif opt in ("-c", "--config"):
            config_path = arg
        elif opt in ("-w", "--weight"):
            weight_path = arg
        elif opt in ("-m", "--meta"):
            meta_path = arg
        elif opt in ("-t", "--threshold"):
            thresh = float(arg)
        elif opt in ("-s", "--svo_file"):
            svo_path = arg
        elif opt in ("-z", "--zed_id"):
            zed_id = int(arg)

    input_type = sl.InputType()
    if svo_path is not None:
        log.info("SVO file : " + svo_path)
        input_type.set_from_svo_file(svo_path)
    else:
        # Launch camera by id
        input_type.set_from_camera_id(zed_id)

    init = sl.InitParameters(input_t=input_type)
    init.coordinate_units = sl.UNIT.METER

    cam = sl.Camera()
    if not cam.is_opened():
        log.info("Opening ZED Camera...")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        log.error(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    # Use STANDARD sensing mode
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD
    mat = sl.Mat()
    point_cloud_mat = sl.Mat()
    image_size = cam.get_camera_information().camera_resolution

    # Import the global variables. This lets us instance Darknet once,
    # then just call performDetect() again without instancing again
    global metaMain, netMain, altNames  # pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(config_path):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(config_path)+"`")
    if not os.path.exists(weight_path):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weight_path)+"`")
    if not os.path.exists(meta_path):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(meta_path)+"`")
    if netMain is None:
        netMain = load_net_custom(config_path.encode(
            "ascii"), weight_path.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(meta_path.encode("ascii"))
    if altNames is None:
        # In thon 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(meta_path) as meta_fh:
                meta_contents = meta_fh.read()
                import re
                match = re.search("names *= *(.*)$", meta_contents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as names_fh:
                            names_list = names_fh.read().strip().split("\n")
                            altNames = [x.strip() for x in names_list]
                except TypeError:
                    pass
        except Exception:
            pass

    color_array = generate_color(meta_path)

    log.info("Running...")
    
   
######################################################################
    os.system ("sudo /etc/init.d/qhmiserver start")     
    mqttc = mqtt.Client()
    # Start of main program - Set up the MQTT client connection
    
    client.on_message = on_message
    client.on_connect = on_connect
    client.on_publish = on_publish
    client.on_subscribe = on_subscribe
    client.connect("localhost", 1883,60)
   
    
    def foo():            
       
        # Sit and wait for inbound or outbound events
        for _ in range(1):
              time.sleep(1)
              client.loop()
        threading.Timer(WAIT_SECONDS, foo).start()

    foo()     
    
    key = ''
    while key != 113:  # for 'q' key
        #start_time = time.time() # start time of the loop
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            image = mat.get_data()

            cam.retrieve_measure(
                point_cloud_mat, sl.MEASURE.XYZRGBA)
            depth = point_cloud_mat.get_data()

            # Do the detection
            detections = detect(netMain, metaMain, image, thresh)
            results = []
            X_results = []
            Y_results = []
            Item_Count = len(detections)
            Obj0 = 0
            Obj1 = 1
            Obj2 = 2
            Obj3 = 3
            Obj4 = 4
            Obj5 = 5
            Obj6 = 6
            #log.info(chr(27) + "[2J"+"**** " + str(len(detections)) + " Results ****")
            for detection in detections:
                label = detection[0]              
                confidence = detection[1]
                pstring = label+": "+str(np.rint(100 * confidence))+"%"
                #log.info(pstring)
                bounds = detection[2]
                y_extent = int(bounds[3])
                x_extent = int(bounds[2])
                x_center = int(bounds[0])
                y_center = int(bounds[1])
                # Coordinates are around the center
                x_coord = int(bounds[0] - bounds[2]/2)
                y_coord = int(bounds[1] - bounds[3]/2)
                #boundingBox = [[x_coord, y_coord], [x_coord, y_coord + y_extent], [x_coord + x_extent, y_coord + y_extent], [x_coord + x_extent, y_coord]]
                thickness = 1
                x, y, z = get_object_depth(depth, bounds)
                distance = math.sqrt(x * x + y * y + z * z)
                distance = "{:.2f}".format(distance)
                result = distance
                results.append(result)
                X_result = x_center
                X_results.append(X_result)
                Y_result = y_center
                Y_results.append(Y_result)
                
                cv2.rectangle(image, (x_coord - thickness, y_coord - thickness),
                              (x_coord + x_extent + thickness, y_coord + (18 + thickness*4)),
                              color_array[detection[3]], -1)
                cv2.putText(image, label + " " +  (str(distance) + " m"),
                            (x_coord + (thickness * 4), y_coord + (10 + thickness * 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.rectangle(image, (x_coord - thickness, y_coord - thickness),
                              (x_coord + x_extent + thickness, y_coord + y_extent + thickness),
                              color_array[detection[3]], int(thickness*2))
                cv2.circle(image, (x_center,y_center),10,
                              color_array[detection[3]], -1)
                              
            if Obj1 <= Item_Count:
              Object1 = str(detections[0][0])
              Object2 = str(results[0])
              Object11 = str(X_results[0])
              Object12 = str(Y_results[0])
            else:
              Object1 = Obj0 
              Object2 = Obj0   
              Object11 = Obj0 
              Object12 = Obj0  
                   
            if Obj2 <= Item_Count:
              Object3 = str(detections[1][0])
              Object4 = str(results[1])
              Object13 = str(X_results[1])
              Object14 = str(Y_results[1]) 
            else:
              Object3 = Obj0 
              Object4 = Obj0 
              Object13 = Obj0 
              Object14 = Obj0   
                
            if Obj3 <= Item_Count:
              Object5 = str(detections[2][0])
              Object6 = str(results[2])
              Object15 = str(X_results[2])
              Object16 = str(Y_results[2]) 
            else:
              Object5 = Obj0  
              Object6 = Obj0   
              Object15 = Obj0 
              Object16 = Obj0 
              
            if Obj4 <= Item_Count:
              Object7 = str(detections[3][0])
              Object8 = str(results[3])
              Object17 = str(X_results[3])
              Object18 = str(Y_results[3]) 
            else:
              Object7 = Obj0 
              Object8 = Obj0   
              Object17 = Obj0 
              Object18 = Obj0 
                
            if Obj5 <= Item_Count:
              Object9 = str(detections[3][0])
              Object10 = str(results[4])
              Object19 = str(X_results[4])
              Object20 = str(Y_results[4]) 
            else: 
              Object9 = Obj0 
              Object10 = Obj0 
              Object19 = Obj0 
              Object20 = Obj0     
                               
            msgs = [{'topic':"Output/Device Output1", 'payload':Object1},{'topic':"Output/Device Output2", 'payload':Object2},{'topic':"Output/Device Output3", 'payload':Object3},{'topic':"Output/Device Output4", 'payload':Object4},{'topic':"Output/Device Output5", 'payload':Object5},{'topic':"Output/Device Output6", 'payload':Object6},{'topic':"Output/Device Output7", 'payload':Object7},{'topic':"Output/Device Output8", 'payload':Object8},{'topic':"Output/Device Output9", 'payload':Object9},{'topic':"Output/Device Output10", 'payload':Object10},{'topic':"Output/Device Output11", 'payload':Object11},{'topic':"Output/Device Output12", 'payload':Object12},{'topic':"Output/Device Output13", 'payload':Object13},{'topic':"Output/Device Output14", 'payload':Object14},{'topic':"Output/Device Output15", 'payload':Object15},{'topic':"Output/Device Output16", 'payload':Object16},{'topic':"Output/Device Output17", 'payload':Object17},{'topic':"Output/Device Output18", 'payload':Object18},{'topic':"Output/Device Output19", 'payload':Object19},{'topic':"Output/Device Output20", 'payload':Object20}]
            publish.multiple(msgs, hostname="ubuntu") 
                             
            cv2.imshow("ZED", image)
            key = cv2.waitKey(5)
            #log.info("FPS: {}".format(1.0 / (time.time() - start_time)))
        else:
            key = cv2.waitKey(5)
    cv2.destroyAllWindows()

    cam.close()
    #log.info("\nFINISH")


if __name__ == "__main__":
    main(sys.argv[1:])
