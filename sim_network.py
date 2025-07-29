import numpy as np
import socket
import pickle
import struct
import traceback

# from mjx_sim import Sim
from mujoco_sim import Sim

PORT = 8888


class SimServer:
    """A server class to receive commands"""

    def __init__(self, sim: Sim):
        """Initialize with a simulation instance"""
        self.sim = sim

        # Create a server socket.
        print(f"Starting TCP server for Simulation on port {PORT}.")
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.server.bind(("localhost", PORT))
        self.server.listen(5)

    def listen_and_process(self, conn):
        while True:
            # First, receive 4 bytes that
            # indicate the length of the incoming message
            raw_msglen = recvall(conn, 4)
            msglen = deserialize_len(raw_msglen)
            if msglen is None:
                break

            # Then, receive the full serialized message
            data = recvall(conn, msglen)
            try:
                # Deserialize to get the required command
                msg = deserialize_data(data)
                if msg is None:
                    break
            except Exception as e:
                print("Error processing received data:", e)
                break

            # Process command
            command = msg.get("command", [])
            params = msg.get("param", [])

            # Call the function in Sim
            if hasattr(self.sim, command) and callable(
                getattr(self.sim, command)
            ):
                try:
                    func = getattr(self.sim, command)
                    print(f"Trying to execute {func.__name__}")
                    if isinstance(params, list) or isinstance(params, tuple):
                        response = func(*params)
                    elif isinstance(params, dict):
                        response = func(**params)
                    else:
                        response = func(params)

                except Exception as e:
                    print(f"Error executing command function '{command}': {e}")
                    traceback.print_exc()
                    continue
            else:
                print(f"Unknown command function {command}")
                continue

            # Send the end_pose back to the client.
            response = serialize(response)
            conn.sendall(response)

    def run_server(self):
        """Run the simulation with TCP server"""
        try:
            while True:
                # Wait for a new connection.
                print("Waiting for a new connection...")
                conn, addr = self.server.accept()
                print(f"Connected by {addr}")

                # Process messages from this client until it disconnects.
                try:
                    self.listen_and_process(conn)

                except Exception as e:
                    print("Client Connection error:", e)
                finally:
                    conn.close()
                    print("Current Client Connection closed.")
        except KeyboardInterrupt as e:
            pass
        except Exception as e:
            print("Server error:", e)
        finally:
            self.close()

    def close(self):
        """Explicitly close the server socket."""
        if self.server:
            try:
                self.server.close()
                print("Server connection closed.")
            except Exception as e:
                print(f"Error closing server socket: {e}")
            finally:
                self.server = None


class SimClient:
    """A client class to send commands"""

    def __init__(self):
        """Initialize"""
        # Create and connect the client socket.
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(("localhost", PORT))
        print(f"Connected to server on port {PORT}.")

    def execute(self, function_name, params=[]):
        """Execute a function in the server given parameters"""
        msg = {
            "command": function_name,
            "param": params,
        }
        data = serialize(msg)
        response = self.send_and_wait_for_response(data)
        return response

    def send_and_wait_for_response(self, data):
        """Send data and Wait for server response"""
        try:
            # Send
            self.client.sendall(data)

            # First, receive 4 bytes that
            # indicate the length of the incoming message.
            raw_msglen = recvall(self.client, 4)
            msglen = deserialize_len(raw_msglen)
            if msglen is None:
                return None

            # Then, receive the full serialized message.
            data = recvall(self.client, msglen)
            if data is None:
                return None
            try:
                # Deserialize the response message
                msg = deserialize_data(data)
                return msg

            except Exception as e:
                print("Error processing received data:", e)
                return None
        except Exception as e:
            print("Connection error:", e)
            self.close()
            return None

    def close(self):
        """Close the socket connection."""
        if self.client:
            try:
                self.client.close()
                print("Client connection closed.")
            except Exception as e:
                print(f"Error closing socket: {e}")


# Helper functions for TCP communication
def recvall(sock, n):
    """
    Helper function to ensure we receive exactly n bytes from the socket.
    """
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def serialize(msg):
    """Serialize a message for tcp transmission"""
    # Serialize the message using pickle.
    data = pickle.dumps(msg)
    # Pack the length of the data into FOUR/4 bytes in network (big-endian) order.
    header = struct.pack(">I", len(data))
    return header + data


def deserialize_len(data):
    """Get the length of a message from TCP transmission."""
    if data is None:
        return None
    msglen = struct.unpack(">I", data)[0]
    return msglen


def deserialize_data(data):
    """Deserialize a message from TCP transmission."""
    if data is None:
        return None
    data = pickle.loads(data)
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("obj_name", nargs="?", default="cracker_box_flipped")
    args = parser.parse_args()
    seed = 42
    np.random.seed(seed)
    np.set_printoptions(precision=4, suppress=True)

    # Initialize simulation
    xml = open("mujoco_sim.xml").read()
    xml = xml.replace("object_name", args.obj_name)
    sim = Sim(
        xml,
        n_envs=20,
        robot_joint_dof=6,
        robot_ee_dof=0,
        dt=0.02,
        visualize=True,
    )
    server = SimServer(sim)
    server.run_server()
