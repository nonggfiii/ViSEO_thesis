from flask import Blueprint, request, render_template,jsonify
from src.controllers.import_data import upload_file
from src.controllers.seoController import generate_seo,search_topics_by_product_name

route_blueprint = Blueprint("route_blueprint", __name__)

@route_blueprint.route("/", methods=["GET", "POST"])
async def handle_upload():
    if request.method == "POST":
        return await upload_file(request)
    return render_template("index.html")
@route_blueprint.route('/canvas_view')
def view_page():
    return render_template("query_canvas.html")
@route_blueprint.route("/api/generate_seo", methods=["POST"])
def seo_generate():
    """Route for generating SEO content."""
    return generate_seo()
@route_blueprint.route("/api/search_topics", methods=["POST"])
def search_topics_route():
    """
    API route để tìm kiếm topic_title theo product_name.
    """
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Thiếu query trong request"}), 400

        query = data["query"]
        response, status_code = search_topics_by_product_name(query)
        return jsonify(response), status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500
