import zmq
import json
from datetime import datetime

class BinanceZmqClient:
    def __init__(self, host="localhost", sub_port=5555, push_port=5556):
        self.host = host
        self.sub_port = sub_port
        self.push_port = push_port
        self.context = zmq.Context()
        
        # 1. Market data listener (SUB)
        self.subscriber = self.context.socket(zmq.SUB)
        # 2. Command sender (PUSH)
        self.commander = self.context.socket(zmq.PUSH)
        
        self.kline_callback = None
        self.is_running = False

    def connect(self):
        # connect the listener
        sub_address = f"tcp://{self.host}:{self.sub_port}"
        self.subscriber.connect(sub_address)
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # connect the sender
        push_address = f"tcp://{self.host}:{self.push_port}"
        self.commander.connect(push_address)
        
        print(f"📻 [ZMQ Client] Market feed connected: {sub_address}")
        print(f"🎯 [ZMQ Client] Command channel connected: {push_address}")

    def set_kline_callback(self, callback):
        self.kline_callback = callback

    # [added] dedicated function to send order signals
    def send_order_signal(self, action: str, symbol: str, price: float, stop_price: float = 0.0):
        """Package the strategy brain command into JSON and send it back to C++ at warp speed"""
        order_data = {
            "action": action,
            "symbol": symbol,
            "price": price,
            "stop_price": stop_price,
            "timestamp": datetime.now().timestamp()
        }
        self.commander.send_string(json.dumps(order_data))
        print(f"📤 [Comms] Sent {action} signal to the C++ execution engine")

    def _send_ack(self):
        """Tell C++ this bar has been processed (required for sync backtest)."""
        self.commander.send_string(json.dumps({"type": "ack"}))

    def start_listening(self):
        self.is_running = True
        try:
            while self.is_running:
                message = self.subscriber.recv_string()
                data = json.loads(message)
                if data.get("type") == "kline" and self.kline_callback:
                    self.kline_callback(data)
                    # ACK after strategy callback so any order is pushed first.
                    self._send_ack()
        except KeyboardInterrupt:
            print("\n🛑 [ZMQ Client] Listening stopped.")
        finally:
            self.close()

    def close(self):
        self.subscriber.close()
        self.commander.close()
        self.context.term()