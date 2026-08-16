import logging

import azure.functions as func

from app.tools.ai_search_tool import search_bp

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

app.register_blueprint(search_bp)
