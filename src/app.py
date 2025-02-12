from flask import Flask
import logging
from src.routes.route import route_blueprint  # Kiểm tra lại import

app = Flask(__name__)

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Đăng ký blueprint cho routes
app.register_blueprint(route_blueprint)

if __name__ == "__main__":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())  # Đảm bảo event loop policy được cài đặt
    app.run(debug=True)
