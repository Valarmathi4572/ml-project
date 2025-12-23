from app import app
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

# Vercel expects the Flask app to be available as a WSGI application
application = app

# For local development
if __name__ == "__main__":
    app.run(debug=True)