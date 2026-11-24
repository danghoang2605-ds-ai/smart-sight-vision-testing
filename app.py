"""Entry point: `python app.py`, then open http://localhost:5050."""
import logging

from smartsight.config import settings
from smartsight.server import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app(settings)

if __name__ == "__main__":
    # threaded=True lets the MJPEG stream and the status polling run side by side.
    app.run(host=settings.host, port=settings.port, debug=False, threaded=True)
