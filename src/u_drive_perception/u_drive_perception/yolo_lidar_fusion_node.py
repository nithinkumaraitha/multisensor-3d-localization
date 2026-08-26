#!/usr/bin/env python3
import json, math, time
from typing import Tuple, Optional
import numpy as np, cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from rclpy.duration import Duration
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, NavSatFix, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Header
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

def meters_per_deg(lat_deg: float)->Tuple[float,float]:
    lat = math.radians(lat_deg)
    m_lat = 111132.92 - 559.82*math.cos(2*lat) + 1.175*math.cos(4*lat) - 0.0023*math.cos(6*lat)
    m_lon = 111412.84*math.cos(lat) - 93.5*math.cos(3*lat) + 0.118*math.cos(5*lat)
    return m_lat, m_lon

def rotx(a): c,s=math.cos(a),math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]],np.float32)
def roty(a): c,s=math.cos(a),math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]],np.float32)
def rotz(a): c,s=math.cos(a),math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]],np.float32)

def pc2_to_xyz_fast(msg: PointCloud2, max_points: Optional[int]=None)->np.ndarray:
    if msg.width*msg.height==0 or msg.point_step<12 or len(msg.data)<msg.point_step: return np.zeros((0,3),np.float32)
    offs={f.name.lower():f.offset for f in msg.fields}
    if not all(k in offs for k in ('x','y','z')): return np.zeros((0,3),np.float32)
    endian='>' if msg.is_bigendian else '<'
    step_f=msg.point_step//4
    arr=np.frombuffer(msg.data,dtype=endian+'f4',count=(len(msg.data)//4))
    usable=(len(arr)//step_f)*step_f
    if usable==0: return np.zeros((0,3),np.float32)
    arr=arr[:usable].reshape(-1,step_f)
    xi,yi,zi=offs['x']//4,offs['y']//4,offs['z']//4
    xyz=np.stack((arr[:,xi],arr[:,yi],arr[:,zi]),1).astype(np.float32,copy=False)
    m=np.isfinite(xyz).all(1); xyz=xyz[m]
    if max_points and xyz.shape[0]>max_points:
        idx=np.random.choice(xyz.shape[0],size=max_points,replace=False); xyz=xyz[idx]
    return xyz

def cv_decode(bridge: CvBridge, msg: Image)->np.ndarray:
    enc=(msg.encoding or '').lower()
    if enc=='bgr8': return bridge.imgmsg_to_cv2(msg,'bgr8')
    im=bridge.imgmsg_to_cv2(msg,desired_encoding='passthrough')
    if im.ndim==2: return cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    if im.ndim==3 and im.shape[2]==3:
        return cv2.cvtColor(im,cv2.COLOR_RGB2BGR) if enc=='rgb8' else im
    return cv2.cvtColor(im,cv2.COLOR_BGRA2BGR)

def cv_decode_compressed(msg: CompressedImage)->np.ndarray:
    arr=np.frombuffer(msg.data,np.uint8); im=cv2.imdecode(arr,cv2.IMREAD_COLOR)
    return im if im is not None else np.zeros((0,0,3),np.uint8)

class YoloLidarFusion(Node):
    def __init__(self):
        super().__init__('yolo_lidar_fusion')
        self.bridge=CvBridge()

        # topics
        self.declare_parameter('image_topic','/sensing/camera/camera0/image_rect_color')
        self.declare_parameter('image_compressed_topic','/sensing/camera/camera0/image_rect_color/compressed')
        self.declare_parameter('camera_info_topic','/sensing/camera/camera0/camera_info')
        self.declare_parameter('cloud_topic','/sensing/lidar/top/outlier_filtered/pointcloud')
        self.declare_parameter('gnss_topic','/sensing/gnss/nav_sat_fix')
        self.declare_parameter('yolo_topic','/perception/yolo_dets')
        self.declare_parameter('overlay_topic','/perception/fusion_image')
        self.declare_parameter('objects_topic','/perception/objects')
        self.declare_parameter('markers_topic','/perception/markers')
        self.declare_parameter('status_topic','/perception/status')

        # frames
        self.declare_parameter('camera_frame_id','camera_front_optical_frame')
        self.declare_parameter('lidar_frame_id','velodyne_top_base_link')
        self.declare_parameter('fixed_frame','velodyne_top_base_link')

        # intrinsics / extrinsics
        self.declare_parameter('fx',700.0); self.declare_parameter('fy',700.0)
        self.declare_parameter('cx',400.0); self.declare_parameter('cy',300.0)
        self.declare_parameter('use_camera_info',True)

        self.declare_parameter('use_tf',False)
        self.declare_parameter('tf_timeout_sec',0.20)
        self.declare_parameter('t_lc_x',0.0); self.declare_parameter('t_lc_y',0.0); self.declare_parameter('t_lc_z',1.0)
        self.declare_parameter('yaw_deg',0.0); self.declare_parameter('pitch_deg',0.0); self.declare_parameter('roll_deg',0.0)

        # fusion
        self.declare_parameter('min_conf',0.25)
        self.declare_parameter('expand_px',14)
        self.declare_parameter('depth_band_beta',0.6)
        self.declare_parameter('reuse_last_ms',600)
        self.declare_parameter('lidar_downsample',1)
        self.declare_parameter('show_raw_yolo',True)
        self.declare_parameter('lidar_max_points',160000)
        self.declare_parameter('lidar_msg_skip',0)

        gp=self.get_parameter
        self.image_topic=gp('image_topic').value
        self.image_compressed_topic=gp('image_compressed_topic').value
        self.camera_info_topic=gp('camera_info_topic').value
        self.cloud_topic=gp('cloud_topic').value
        self.gnss_topic=gp('gnss_topic').value
        self.yolo_topic=gp('yolo_topic').value
        self.overlay_topic=gp('overlay_topic').value
        self.objects_topic=gp('objects_topic').value
        self.markers_topic=gp('markers_topic').value
        self.status_topic=gp('status_topic').value

        self.camera_frame_id=gp('camera_frame_id').value or 'camera_front_optical_frame'
        self.lidar_frame_id=gp('lidar_frame_id').value or 'velodyne_top_base_link'
        self.fixed_frame=gp('fixed_frame').value or self.lidar_frame_id

        self.fx=float(gp('fx').value); self.fy=float(gp('fy').value)
        self.cx=float(gp('cx').value); self.cy=float(gp('cy').value)
        self.use_camera_info=bool(gp('use_camera_info').value)

        self.use_tf=bool(gp('use_tf').value)
        self.tf_timeout=float(gp('tf_timeout_sec').value)

        # lidar(base x fwd,y left,z up) → camera(optical x right,y down,z fwd)
        R_base2opt = np.array([[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]],np.float32)
        yaw = math.radians(float(gp('yaw_deg').value))
        pit = math.radians(float(gp('pitch_deg').value))
        rol = math.radians(float(gp('roll_deg').value))
        self.R_lc = (R_base2opt @ (rotz(yaw) @ roty(pit) @ rotx(rol))).astype(np.float32)
        self.t_lc = np.array([float(gp('t_lc_x').value),float(gp('t_lc_y').value),float(gp('t_lc_z').value)],np.float32)

        self.min_conf=float(gp('min_conf').value)
        self.expand_px=int(gp('expand_px').value)
        self.depth_band_beta=float(gp('depth_band_beta').value)
        self.reuse_last_ms=int(gp('reuse_last_ms').value)
        self.lidar_downsample=max(1,int(gp('lidar_downsample').value))
        self.show_raw_yolo=bool(gp('show_raw_yolo').value)
        self.lidar_max_points=int(gp('lidar_max_points').value)
        self.lidar_msg_skip=max(0,int(gp('lidar_msg_skip').value))

        # state
        self.cloud_xyz=None; self._pc_count=0
        self.yolo_dets=[]; self.last_nonempty=[]
        self.lat0=None; self.lon0=None; self.alt0=0.0; self.have_origin=False
        self.m_per_deg_lat=111000.0; self.m_per_deg_lon=85000.0
        self._have_cam_info=False
        self._t_img=0.0; self._t_imgc=0.0; self._t_pc=0.0; self._t_gps=0.0; self._t_yolo=0.0

        self.tf_buf=Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener=TransformListener(self.tf_buf, self, spin_thread=True)

        img_qos=qos_profile_sensor_data
        lidar_qos=qos_profile_sensor_data
        reliable=QoSProfile(depth=10,history=HistoryPolicy.KEEP_LAST,
                            reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE)

        self.sub_img  = self.create_subscription(Image,          self.image_topic,             self.on_img,  img_qos)
        self.sub_imgc = self.create_subscription(CompressedImage,self.image_compressed_topic, self.on_imgc, img_qos)
        self.sub_pc   = self.create_subscription(PointCloud2,    self.cloud_topic,             self.on_pc,   lidar_qos)
        self.sub_gps  = self.create_subscription(NavSatFix,      self.gnss_topic,              self.on_gps,  reliable)
        self.sub_yolo = self.create_subscription(String,         self.yolo_topic,              self.on_yolo, reliable)
        self.sub_info = self.create_subscription(CameraInfo,     self.camera_info_topic,       self.on_info, img_qos) if self.use_camera_info else None

        self.pub_overlay = self.create_publisher(Image,       self.overlay_topic, reliable)
        self.pub_objects = self.create_publisher(String,      self.objects_topic, reliable)
        self.pub_markers = self.create_publisher(MarkerArray, self.markers_topic, reliable)
        self.pub_status  = self.create_publisher(String,      self.status_topic,  reliable)

        self.create_timer(1.0, self.on_timer)

    # callbacks
    def on_info(self,msg:CameraInfo):
        try:
            K=list(getattr(msg,'k',getattr(msg,'K',[])))
            if len(K)>=9:
                self.fx=float(K[0]); self.fy=float(K[4]); self.cx=float(K[2]); self.cy=float(K[5])
                self._have_cam_info=True
        except Exception: pass

    def on_gps(self,msg:NavSatFix):
        self._t_gps=self.get_clock().now().nanoseconds*1e-9
        if not self.have_origin and math.isfinite(msg.latitude) and math.isfinite(msg.longitude):
            self.lat0=float(msg.latitude); self.lon0=float(msg.longitude)
            self.alt0=float(msg.altitude) if math.isfinite(msg.altitude) else 0.0
            self.m_per_deg_lat, self.m_per_deg_lon = meters_per_deg(self.lat0)
            self.have_origin=True

    def on_pc(self,msg:PointCloud2):
        self._pc_count+=1
        if self.lidar_msg_skip>0 and (self._pc_count%(self.lidar_msg_skip+1))!=0: return
        try:
            self.cloud_xyz=pc2_to_xyz_fast(msg,max_points=self.lidar_max_points)
            self._t_pc=self.get_clock().now().nanoseconds*1e-9
        except Exception: pass

    def on_yolo(self,msg:String):
        self._t_yolo=self.get_clock().now().nanoseconds*1e-9
        try:
            arr=json.loads(msg.data); out=[]
            for d in arr:
                if not all(k in d for k in ('x','y','w','h')): continue
                nm=str(d.get('name','')).lower()
                cf=float(d.get('conf',1.0))
                out.append({'name':nm,'conf':cf,'bbox':{'x':int(d['x']),'y':int(d['y']),'w':int(d['w']),'h':int(d['h'])}})
            self.yolo_dets=out
        except Exception:
            self.yolo_dets=[]

    def on_img(self,msg:Image):
        self._t_img=self.get_clock().now().nanoseconds*1e-9
        try: bgr=cv_decode(self.bridge,msg)
        except Exception: return
        self.process(bgr, msg.header)

    def on_imgc(self,msg:CompressedImage):
        self._t_imgc=self.get_clock().now().nanoseconds*1e-9
        hdr=Header(); hdr.stamp=self.get_clock().now().to_msg(); hdr.frame_id=self.camera_frame_id
        try: bgr=cv_decode_compressed(msg)
        except Exception: return
        if bgr.size==0: return
        self.process(bgr, hdr)

    def on_timer(self):
        now=self.get_clock().now().nanoseconds*1e-9
        status={
            "image_raw_age": round(now-self._t_img,2) if self._t_img>0 else None,
            "image_comp_age": round(now-self._t_imgc,2) if self._t_imgc>0 else None,
            "cloud_age": round(now-self._t_pc,2) if self._t_pc>0 else None,
            "yolo_age": round(now-self._t_yolo,2) if self._t_yolo>0 else None,
            "gnss_age": round(now-self._t_gps,2) if self._t_gps>0 else None,
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "have_cam_info": self._have_cam_info, "fixed_frame": self.fixed_frame
        }
        self.pub_status.publish(String(data=json.dumps(status)))

    # transforms
    def lookup_T_lc(self):
        ts:TransformStamped=self.tf_buf.lookup_transform(
            target_frame=self.camera_frame_id, source_frame=self.lidar_frame_id,
            time=rclpy.time.Time(), timeout=Duration(seconds=self.tf_timeout))
        t=ts.transform.translation; q=ts.transform.rotation
        qw,qx,qy,qz = q.w,q.x,q.y,q.z
        R=np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qy), 2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)]
        ],dtype=np.float32)
        tvec=np.array([t.x,t.y,t.z],np.float32)
        return R,tvec

    def lidar_to_cam(self, xyz: np.ndarray)->np.ndarray:
        if xyz is None or xyz.size==0: return np.zeros((0,3),np.float32)
        R=self.R_lc; t=self.t_lc
        if self.use_tf:
            try: R,t=self.lookup_T_lc()
            except Exception: pass
        pts=(R@xyz.T).T + t
        return pts[pts[:,2]>0.1]

    def cam_to_lidar_point(self, p_cam: np.ndarray)->np.ndarray:
        R=self.R_lc; t=self.t_lc
        if self.use_tf:
            try: R,t=self.lookup_T_lc()
            except Exception: pass
        return (R.T @ (p_cam - t)).astype(np.float32)

    def project(self, pts_cam: np.ndarray, w:int, h:int):
        if pts_cam.size==0: return np.empty((0,2),np.int32), np.empty((0,3),np.float32)
        x,y,z=pts_cam[:,0],pts_cam[:,1],pts_cam[:,2]
        z=np.where(z==0,1e-6,z)
        u=(self.fx*x/z)+self.cx; v=(self.fy*y/z)+self.cy
        mask=(u>=0)&(u<w)&(v>=0)&(v<h)
        if not np.any(mask): return np.empty((0,2),np.int32), np.empty((0,3),np.float32)
        uv=np.rint(np.stack([u[mask],v[mask]],1)).astype(np.int32)
        uv[:,0]=np.clip(uv[:,0],0,w-1); uv[:,1]=np.clip(uv[:,1],0,h-1)
        return uv, pts_cam[mask]

    # ---- late fusion: take nearest/front-surface points in the lower part of the box ----
    def fuse_box(self, det, uv_all, pts_all, W, H):
        bb = det['bbox']; cls = det.get('name','')
        x = bb['x']; y = bb['y']; w = bb['w']; h = bb['h']
        # focus on lower region (reduce sky/background contamination)
        if cls in ('vehicle',):
            y0 = y + int(0.45*h)
        elif cls == 'person':
            y0 = y + int(0.25*h)
        else:
            y0 = y
        x0 = max(0, x - self.expand_px)
        y0 = max(0, y0 - self.expand_px)
        x1 = min(W, x + w + self.expand_px)
        y1 = min(H, y + h + self.expand_px)

        if uv_all.size == 0: return None
        inside = (uv_all[:,0] >= x0) & (uv_all[:,0] < x1) & (uv_all[:,1] >= y0) & (uv_all[:,1] < y1)
        if not np.any(inside): return None
        cluster = pts_all[inside]
        if cluster.shape[0] < 4: return None

        # valid depths & cap range
        z = cluster[:,2]
        m = np.isfinite(z) & (z > 0.3) & (z < 120.0)
        cluster = cluster[m]
        if cluster.shape[0] < 4: return None

        # nearest/front surface by low percentile + narrow band
        z = cluster[:,2]
        z_front = float(np.quantile(z, 0.12))
        band = max(self.depth_band_beta, 0.6)
        keep = (z >= (z_front - 0.2)) & (z <= (z_front + band))
        front = cluster[keep]
        if front.shape[0] < 4: return None

        # robust center
        p_cam = np.median(front, axis=0).astype(np.float32)

        # back to LiDAR/fixed
        p_lidar = self.cam_to_lidar_point(p_cam)
        dx_l, dy_l, dz_l = float(p_lidar[0]), float(p_lidar[1]), float(p_lidar[2])

        out = {
            'name': det['name'],
            'dx_cam': float(p_cam[0]), 'dy_cam': float(p_cam[1]), 'dz_cam': float(p_cam[2]),
            'dx_lidar': dx_l, 'dy_lidar': dy_l, 'dz_lidar': dz_l,
            'bbox': bb
        }
        if self.have_origin:
            east_m = p_cam[0]; north_m = p_cam[2]; up_m = -p_cam[1]
            out['lat'] = float(self.lat0 + north_m / self.m_per_deg_lat)
            out['lon'] = float(self.lon0 + east_m  / self.m_per_deg_lon)
            out['alt'] = float(self.alt0 + up_m)
        return out

    def process(self, img: np.ndarray, hdr: Header):
        H,W = img.shape[:2]
        overlay = img.copy()

        # project LiDAR
        uv_all=np.empty((0,2),np.int32); pts_all=np.empty((0,3),np.float32)
        if self.cloud_xyz is not None and self.cloud_xyz.size>0:
            cam_pts=self.lidar_to_cam(self.cloud_xyz)
            if cam_pts.size>0:
                uv_all, pts_all = self.project(cam_pts, W, H)
                if uv_all.size>0:
                    step=max(1,self.lidar_downsample)
                    overlay[uv_all[::step,1], uv_all[::step,0]] = (0,0,255)

        # draw YOLO
        yolo=[d for d in self.yolo_dets if d.get('conf',1.0)>=self.min_conf]
        if self.show_raw_yolo:
            for d in yolo:
                b=d['bbox']; x1=max(0,b['x']); y1=max(0,b['y']); x2=min(W,b['x']+b['w']); y2=min(H,b['y']+b['h'])
                cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,255,0),2)

        # fuse
        fused=[]
        if yolo and uv_all.size>0:
            for d in yolo:
                res = self.fuse_box(d, uv_all, pts_all, W, H)
                if res is None: continue
                fused.append(res)
                bb=d['bbox']
                cv2.rectangle(overlay,(bb['x'],bb['y']),(bb['x']+bb['w'],bb['y']+bb['h']),(255,0,255),2)
                cv2.putText(overlay,f"{d['name']} {res['dz_cam']:.1f}m",(bb['x'],max(0,bb['y']-6)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),1,cv2.LINE_AA)

        hud=f"YOLO:{len(yolo)} LiDARpx:{int(uv_all.shape[0])} fx={self.fx:.0f} fy={self.fy:.0f}"
        cv2.rectangle(overlay,(5,5),(5+340,5+40),(0,0,0),-1)
        cv2.putText(overlay,hud,(12,32),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)

        imsg=self.bridge.cv2_to_imgmsg(overlay,encoding='bgr8')
        imsg.header.stamp = hdr.stamp if (hdr.stamp.sec or hdr.stamp.nanosec) else self.get_clock().now().to_msg()
        imsg.header.frame_id = self.camera_frame_id
        self.pub_overlay.publish(imsg)

        self.pub_objects.publish(String(data=json.dumps(fused)))

        # RViz markers in fixed frame
        marray=MarkerArray()
        for i,o in enumerate(fused):
            px,py,pz=o['dx_lidar'],o['dy_lidar'],o['dz_lidar']
            dist=float(np.sqrt(px*px+py*py+pz*pz))
            m=Marker(); m.header.stamp=imsg.header.stamp; m.header.frame_id=self.fixed_frame
            m.ns='detections'; m.id=i*2; m.type=Marker.SPHERE; m.action=Marker.ADD
            m.pose.position.x=px; m.pose.position.y=py; m.pose.position.z=pz; m.pose.orientation.w=1.0
            m.scale.x=0.6; m.scale.y=0.6; m.scale.z=0.6; m.color.r=1.0; m.color.g=0.2; m.color.b=0.8; m.color.a=0.9
            marray.markers.append(m)
            t=Marker(); t.header.stamp=imsg.header.stamp; t.header.frame_id=self.fixed_frame
            t.ns='labels'; t.id=i*2+1; t.type=Marker.TEXT_VIEW_FACING; t.action=Marker.ADD
            t.pose.position.x=px; t.pose.position.y=py; t.pose.position.z=pz+1.2; t.pose.orientation.w=1.0
            t.scale.z=0.6; t.color.r=1.0; t.color.g=1.0; t.color.b=1.0; t.color.a=1.0
            t.text=f"{o['name']} {dist:.1f}m"
            marray.markers.append(t)
        self.pub_markers.publish(marray)

def main():
    rclpy.init()
    node=YoloLidarFusion()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__=='__main__':
    main()

