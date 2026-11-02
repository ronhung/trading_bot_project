import zmq
import json
import msgpack
from datetime import datetime

class BinanceZmqClient:
    def __init__(self, host="localhost", sub_port=5555, push_port=5556, use_msgpack=True):
        self.host = host
        self.sub_port = sub_port
        self.push_port = push_port
        self.use_msgpack = use_msgpack
        self.context = zmq.Context()
        
        # 1. Market data listener (SUB)
        self.subscriber = self.context.socket(zmq.SUB)
        # 2. Command sender (PUSH)
        self.commander = self.context.socket(zmq.PUSH)
        
        self.kline_callback = None
        self.order_update_callback = None
        self.is_running = False

    def connect(self):
        sub_address = f"tcp://{self.host}:{self.sub_port}"
        self.subscriber.connect(sub_address)
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        self.subscriber.setsockopt(zmq.RCVTIMEO, 200)
        
        push_address = f"tcp://{self.host}:{self.push_port}"
        self.commander.connect(push_address)
        
        proto = "MessagePack" if self.use_msgpack else "JSON"
        print(f"[Radio] [ZMQ Client] Market feed connected: {sub_address} ({proto})")
        print(f"[Target] [ZMQ Client] Command channel connected: {push_address} ({proto})")

    def set_kline_callback(self, callback):
        self.kline_callback = callback

    def set_order_update_callback(self, callback):
        self.order_update_callback = callback

    def send_order_signal(self, action: str, symbol: str, price: float, stop_price: float = 0.0,
                          trailing_exit: dict | None = None, execution: dict | None = None):
        order_data = {
            "action": action,
            "symbol": symbol,
            "price": price,
            "stop_price": stop_price,
            "timestamp": datetime.now().timestamp()
        }
        if trailing_exit is not None:
            order_data["trailing_exit"] = trailing_exit
        if execution is not None:
            order_data["execution"] = execution

        if self.use_msgpack:
            self.commander.send(msgpack.packb(order_data))
        else:
            self.commander.send_string(json.dumps(order_data))
        print(f"[Out] [Comms] Sent {action} signal to the C++ execution engine")

    def _send_ack(self):
        """Tell C++ this bar has been processed (required for sync backtest)."""
        if self.use_msgpack:
            self.commander.send(msgpack.packb({"type": "ack"}))
        else:
            self.commander.send_string(json.dumps({"type": "ack"}))

    def _parse_message(self, raw_bytes: bytes) -> dict:
        if not raw_bytes:
            return {}
        if raw_bytes.startswith(b"{") or raw_bytes.startswith(b"["):
            return json.loads(raw_bytes.decode("utf-8"))
        try:
            return msgpack.unpackb(raw_bytes, raw=False)
        except Exception:
            return json.loads(raw_bytes.decode("utf-8"))

    def start_listening(self):
        self.is_running = True
        try:
            while self.is_running:
                try:
                    raw_msg = self.subscriber.recv()
                except zmq.Again:
                    continue
                data = self._parse_message(raw_msg)
                mtype = data.get("type")
                if mtype == "kline" and self.kline_callback:
                    self.kline_callback(data)
                    self._send_ack()
                elif mtype == "order_update" and self.order_update_callback:
                    self.order_update_callback(data)
        except KeyboardInterrupt:
            print("\n[Stop] [ZMQ Client] Listening stopped.")
        except Exception as e:
            import traceback
            print("[Fatal] [ZMQ Client] FATAL in start_listening:")
            traceback.print_exc()
        finally:
            self.close()

    def stop(self):
        self.is_running = False

    def close(self):
        self.subscriber.close()
        self.commander.close()
        self.context.term()
