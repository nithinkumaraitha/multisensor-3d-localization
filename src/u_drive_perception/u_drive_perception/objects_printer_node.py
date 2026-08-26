#!/usr/bin/env python3
import json, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String

class ObjectsPrinter(Node):
    def __init__(self):
        super().__init__('objects_printer')
        self.declare_parameter('objects_topic', '/perception/objects')
        topic = self.get_parameter('objects_topic').value

        reliable = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST,
                              reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.VOLATILE)
        self.sub = self.create_subscription(String, topic, self.on_msg, reliable)
        

    def on_msg(self, msg: String):
        try:
            arr = json.loads(msg.data)
            if not arr:
                return
            for o in arr:
                name = o.get('name','obj')
                bb = o.get('bbox', {})
                dx = o.get('dx_lidar', 0.0)
                dy = o.get('dy_lidar', 0.0)
                dz = o.get('dz_lidar', 0.0)
                dist = (dx*dx + dy*dy + dz*dz) ** 0.5
                lat = o.get('lat', None); lon = o.get('lon', None); alt = o.get('alt', None)
                print(f"[{name:10s}] dist={dist:6.2f} m  lidar(x,y,z)=({dx:6.2f},{dy:6.2f},{dz:6.2f}) "
                      f"bbox=[{bb.get('x',0)},{bb.get('y',0)},{bb.get('w',0)}x{bb.get('h',0)}] "
                      f"geo=({lat:.7f},{lon:.7f},{alt:.2f})" if lat is not None else
                      f"[{name:10s}] dist={dist:6.2f} m  lidar(x,y,z)=({dx:6.2f},{dy:6.2f},{dz:6.2f}) "
                      f"bbox=[{bb.get('x',0)},{bb.get('y',0)},{bb.get('w',0)}x{bb.get('h',0)}]")
        except Exception as e:
            self.get_logger().warn(f"bad objects payload: {e}")

def main():
    rclpy.init()
    n = ObjectsPrinter()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    finally:
        n.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()

