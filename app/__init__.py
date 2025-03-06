from flask import Flask

app = Flask(__name__)

# Import routes after initializing app to avoid circular imports
from app import routes
