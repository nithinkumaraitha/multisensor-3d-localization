#!/usr/bin/env python3
import os, time, json
from typing import List, Dict
import numpy as np, cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge

# optional backends
_HAS_ORT = False
try:
    import onnxruntime as ort
    _HAS_ORT = True
except Exception:
    pass

_HAS_ULTRA = False
try:
    from ultralytics import YOLO as ULYOLO
    _HAS_ULTRA = True
except Exception:
    pass

COCO = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]
VEH = {"bicycle","motorcycle","car","bus","truck","train"}
ALLOW = {"person":"person","traffic light":"traffic_light","stop sign":"stop_sign", **{k:"vehicle" for k in VEH}}

def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float):
    if boxes.size == 0: return []
    x1,y1,x2,y2 = boxes.T
    areas = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    order = scores.argsort()[::-1]
    keep=[]
    while order.size:
        i=order[0]; keep.append(i)
        if order.size==1: break
        xx1=np.maximum(x1[i],x1[order[1:]])
        yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]])
        yy2=np.minimum(y2[i],y2[order[1:]])
        iw=np.maximum(0,xx2-xx1); ih=np.maximum(0,yy2-yy1)
        inter=iw*ih
        iou= inter / np.maximum(1e-6, areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thres]
    return keep

def letterbox(im, new_shape=640, color=(114,114,114)):
    h,w = im.shape[:2]
    if isinstance(new_shape,int): new_shape=(new_shape,new_shape)
    r = min(new_shape[0]/h, new_shape[1]/w)
    new_unpad=(int(w*r), int(h*r))
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
    dw/=2; dh/=2
    top,bottom,left,right = int(round(dh-0.1)), int(round(dh+0.1)), int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)

def to_xyxy(cxcywh: np.ndarray) -> np.ndarray:
    cx,cy,w,h = cxcywh.T
    x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
    return np.stack([x1,y1,x2,y2], axis=1)

def clip_xyxy(xyxy: np.ndarray, W: int, H: int) -> np.ndarray:
    xyxy[:,0::2] = np.clip(xyxy[:,0::2], 0, W)
    xyxy[:,1::2] = np.clip(xyxy[:,1::2], 0, H)
    return xyxy

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.bridge = CvBridge()

        # Params
        self.declare_parameter('image_topic', '/sensing/camera/camera0/image_rect_color')
        self.declare_parameter('image_compressed_topic', '/sensing/camera/camera0/image_rect_color/compressed')
        self.declare_parameter('det_topic', '/perception/yolo_dets')
      
        self.declare_parameter('model', '/home/nithin/models/yolov8n.pt')
        self.declare_parameter('img_size', 640)
        self.declare_parameter('min_conf', 0.25)
        self.declare_parameter('iou_thres', 0.50)
        self.declare_parameter('max_det', 200)
        self.declare_parameter('frame_skip', 1)
        self.declare_parameter('max_fps', 18.0)
        self.declare_parameter('min_box_area', 32)
        self.declare_parameter('classes_filter', ['person','car','bus','truck','train','bicycle','motorcycle','traffic light','stop sign'])

        gp=self.get_parameter
        self.image_topic = gp('image_topic').value
        self.image_compressed_topic = gp('image_compressed_topic').value
        self.det_topic = gp('det_topic').value
        self.model_path = gp('model').value
        self.img_size = int(gp('img_size').value)
        self.min_conf = float(gp('min_conf').value)
        self.iou_thres = float(gp('iou_thres').value)
        self.max_det = int(gp('max_det').value)
        self.frame_skip = max(1, int(gp('frame_skip').value))
        self.max_fps = float(gp('max_fps').value)
        self.min_box_area = int(gp('min_box_area').value)
        self.keep_classes = set([str(x) for x in gp('classes_filter').value])

        if not os.path.isfile(self.model_path):
            self.get_logger().fatal(f"Model not found: {self.model_path}")
        self.get_logger().info(f"loading model: {self.model_path}")

        self.backend = None  # 'ultra' | 'onnxrt' | 'cvdnn'
        suffix = os.path.splitext(self.model_path)[1].lower()
        if suffix == '.pt':
            if not _HAS_ULTRA:
                self.get_logger().fatal("ultralytics not installed. Install with: pip install ultralytics")
            self.ultra_model = ULYOLO(self.model_path)
            self.backend = 'ultra'
            self.get_logger().info("Ultralytics backend (.pt) ")
        elif suffix == '.onnx':
            # try ORT then OpenCV
            if _HAS_ORT:
                try:
                    sess_opts = ort.SessionOptions()
                    self.ort = ort.InferenceSession(self.model_path, sess_options=sess_opts, providers=['CPUExecutionProvider'])
                    self.ort_in = self.ort.get_inputs()[0].name
                    self.ort_out = [o.name for o in self.ort.get_outputs()]
                    self.backend='onnxrt'
                    self.get_logger().info("ONNXRuntime backend ")
                except Exception as e:
                    self.get_logger().warn(f"ONNXRuntime failed: {e}")
            if self.backend is None:
                try:
                    self.dnn = cv2.dnn.readNetFromONNX(self.model_path)
                    self.dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    self.backend='cvdnn'
                    self.get_logger().info("OpenCV DNN backend ")
                except Exception as e:
                    self.get_logger().fatal(f"Cannot read ONNX: {e}")
        else:
            self.get_logger().fatal("Unsupported model extension. Use .pt or .onnx")

        # subs/pubs
        self.sub_img  = self.create_subscription(Image,          self.image_topic,             self.on_image_raw, qos_profile_sensor_data)
        self.sub_imgc = self.create_subscription(CompressedImage,self.image_compressed_topic, self.on_image_compressed, qos_profile_sensor_data)
        reliable = QoSProfile(depth=10,history=HistoryPolicy.KEEP_LAST,
                              reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE)
        self.pub_det  = self.create_publisher(String, self.det_topic, reliable)

        self.frame_idx=0; self._last_infer_t=0.0

    # intake
    def on_image_raw(self, msg: Image):
        try:
            enc=(msg.encoding or '').lower()
            if enc=='bgr8':
                bgr=self.bridge.imgmsg_to_cv2(msg,'bgr8')
            else:
                im=self.bridge.imgmsg_to_cv2(msg,desired_encoding='passthrough')
                if im.ndim==2: bgr=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
                elif im.ndim==3 and im.shape[2]==3:
                    bgr=cv2.cvtColor(im,cv2.COLOR_RGB2BGR) if enc=='rgb8' else im
                else:
                    bgr=cv2.cvtColor(im,cv2.COLOR_BGRA2BGR)
        except Exception:
            return
        self._handle(bgr)

    def on_image_compressed(self, msg: CompressedImage):
        try:
            arr=np.frombuffer(msg.data,dtype=np.uint8)
            bgr=cv2.imdecode(arr,cv2.IMREAD_COLOR)
            if bgr is None: return
        except Exception:
            return
        self._handle(bgr)

    # backends
    def _detect_ultra(self, bgr: np.ndarray) -> List[Dict]:
        # Ultralytics handles resizing internally via model(img, imgsz=..)
        res = self.ultra_model.predict(bgr, imgsz=self.img_size, conf=self.min_conf, iou=self.iou_thres, classes=None, verbose=False, device='cpu', max_det=self.max_det)
        out=[]
        if not res: return out
        r = res[0]
        if r.boxes is None: return out
        boxes = r.boxes.xyxy.cpu().numpy()  # (N,4)
        confs = r.boxes.conf.cpu().numpy()  # (N,)
        clses = r.boxes.cls.cpu().numpy().astype(int)  # (N,)
        H,W=bgr.shape[:2]
        for i in range(boxes.shape[0]):
            x1,y1,x2,y2 = boxes[i].astype(int)
            w=max(0,x2-x1); h=max(0,y2-y1)
            if w*h < self.min_box_area: continue
            raw = COCO[clses[i]] if clses[i] < len(COCO) else str(clses[i])
            if raw not in self.keep_classes: continue
            name = ALLOW.get(raw, raw)
            out.append({"x":int(x1),"y":int(y1),"w":int(w),"h":int(h),
                        "name":name,"conf":float(confs[i])})
            if len(out)>=self.max_det: break
        return out

    def _forward_cvdnn(self, blob: np.ndarray):
        self.dnn.setInput(blob)
        names = self.dnn.getUnconnectedOutLayersNames()
        return [self.dnn.forward(n) for n in names] if names else [self.dnn.forward()]

    def _forward_onnxrt(self, blob: np.ndarray):
        return self.ort.run(self.ort_out, {self.ort_in: blob})

    def _parse_heads(self, heads: List[np.ndarray]) -> np.ndarray:
        rows=[]
        for h in heads:
            a=h
            while a.ndim>2: a=a.squeeze(0)
            if a.ndim==1: a=a.reshape(1,-1)
            if a.shape[0] in (84,85) and a.shape[1]>a.shape[0]: a=a.T
            rows.append(a)
        return np.concatenate(rows,0) if len(rows)>1 else rows[0]

    def _decode_preds(self, preds: np.ndarray, bgr_shape, ratio, pad, img_w, img_h) -> List[Dict]:
        if preds is None or preds.size==0 or preds.shape[1] < 6: return []
        cx,cy,w,h,obj = preds[:,0], preds[:,1], preds[:,2], preds[:,3], preds[:,4]
        cls_scores = preds[:,5:] if preds.shape[1] > 5 else np.zeros((preds.shape[0],1),dtype=np.float32)
        sig = lambda x: 1/(1+np.exp(-x))
        if obj.max(initial=0) > 1.0: obj = sig(obj)
        if cls_scores.max(initial=0) > 1.0: cls_scores = sig(cls_scores)
        cls_idx = np.argmax(cls_scores, axis=1)
        cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_idx]
        conf = obj * cls_conf
        m = conf >= self.min_conf
        if not np.any(m): return []
        cx,cy,w,h,conf,cls_idx = cx[m],cy[m],w[m],h[m],conf[m],cls_idx[m]
        normalized = (np.max([cx.max(initial=0),cy.max(initial=0),w.max(initial=0),h.max(initial=0)]) <= 1.5)
        if normalized:
            cx *= img_w; cy *= img_h; w *= img_w; h *= img_h
        xyxy = to_xyxy(np.stack([cx,cy,w,h],1))
        xyxy[:,[0,2]] -= pad[0]; xyxy[:,[1,3]] -= pad[1]
        xyxy[:,:4] /= ratio
        H,W=bgr_shape[:2]
        xyxy = clip_xyxy(xyxy, W, H)
        keep = nms_xyxy(xyxy, conf, self.iou_thres)
        xyxy, conf, cls_idx = xyxy[keep], conf[keep], cls_idx[keep]
        out=[]
        for i in range(xyxy.shape[0]):
            x1,y1,x2,y2 = xyxy[i].astype(int)
            ww=max(0,x2-x1); hh=max(0,y2-y1)
            if ww*hh < self.min_box_area: continue
            raw = COCO[int(cls_idx[i])] if int(cls_idx[i])<len(COCO) else str(int(cls_idx[i]))
            if raw not in self.keep_classes: continue
            name = ALLOW.get(raw, raw)
            out.append({"x":int(x1),"y":int(y1),"w":int(ww),"h":int(hh),"name":name,"conf":float(conf[i])})
            if len(out)>=self.max_det: break
        return out

    def _detect_cvdnn_or_onnxrt(self, bgr: np.ndarray) -> List[Dict]:
        im, ratio, pad = letterbox(bgr, self.img_size)
        blob = cv2.dnn.blobFromImage(im, 1/255.0, (self.img_size,self.img_size), swapRB=True, crop=False)
        heads = self._forward_onnxrt(blob) if self.backend=='onnxrt' else self._forward_cvdnn(blob)
        preds = self._parse_heads(heads)
        return self._decode_preds(preds, bgr.shape[:2], ratio, pad, self.img_size, self.img_size)

    def _handle(self, bgr: np.ndarray):
        self.frame_idx+=1
        now=time.time()
        if self.max_fps>0 and (now-self._last_infer_t) < (1.0/self.max_fps): return
        if (self.frame_idx % self.frame_skip) != 0: return

        if self.backend=='ultra':
            dets = self._detect_ultra(bgr)
        else:
            dets = self._detect_cvdnn_or_onnxrt(bgr)

        self._last_infer_t = now
        self.pub_det.publish(String(data=json.dumps(dets)))

def main():
    rclpy.init()
    node = YoloDetector()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()

