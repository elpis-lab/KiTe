import socketio
import time
from datetime import datetime
import numpy as np


class Camera:
    def __init__(self, camera_ip="192.168.0.101", camera_port=5000):
        """Initialize with the camera IP address and port"""
        self.server_url = f"http://{camera_ip}:{camera_port}"
        self.sio = socketio.Client()
        self.latest_data = None
        self.setup_socketio_handlers()
        self.connect()

    def setup_socketio_handlers(self):
        @self.sio.on("connect")
        def on_connect():
            print(f"Connected to camera server at {self.server_url}")
            self.sio.emit("get_objects")

        @self.sio.on("disconnect")
        def on_disconnect():
            print("Disconnected from camera server")

        @self.sio.on("centroid_data")
        def on_centroid_data(data):
            objects = data.get("objects", [])
            if objects:
                # Store only the first detected object's data
                obj = objects[0]
                self.latest_data = {
                    "position": np.array(
                        [obj.get("robot_x", 0), obj.get("robot_y", 0), 0]
                    ),
                    "angle": obj.get("angle", 0),
                    "timestamp": data.get(
                        "timestamp", datetime.now().isoformat()
                    ),
                }
            else:
                self.latest_data = None

    def connect(self):
        """Connect to the camera server."""
        try:
            print(f"Connecting to camera server at {self.server_url}...")
            self.sio.connect(self.server_url)
            return True
        except socketio.exceptions.ConnectionError as e:
            print(f"Camera connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from the camera server."""
        if self.sio.connected:
            self.sio.disconnect()

    def get_object_pose(self):
        """Get the latest object pose from the camera.

        Returns:
            numpy.ndarray: [x, y, theta] if object is detected, None otherwise
        """
        # Request new data
        if self.sio.connected:
            self.sio.emit("get_objects")
            # Give some time for the server to respond
            time.sleep(0.5)

        if self.latest_data is not None:
            return np.array(
                [
                    self.latest_data["position"][0],
                    self.latest_data["position"][1],
                    np.deg2rad(
                        self.latest_data["angle"]
                    ),  # Convert angle to radians
                ]
            )
        return None

    def __del__(self):
        """Cleanup when the object is deleted."""
        self.disconnect()


if __name__ == "__main__":
    # Create a Camera instance. You can adjust host and port as needed.
    camera = Camera(camera_ip="192.168.0.101", camera_port="5000")
    pose = camera.get_object_pose()
    print("Received Pose:", pose)
