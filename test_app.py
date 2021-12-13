from flask import Flask, request, redirect, render_template, flash, url_for
from werkzeug.utils import secure_filename
import os

ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg"])
UPLOAD_FOLDER = "web/assets/uploads"

app = Flask(__name__, static_url_path="", static_folder="", template_folder="web/templates")
app.secret_key = "secret key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


@app.route("/")
def index():
    return render_template("responsive.html")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/display/<filename>")
def display_image(filename):
    return redirect(url_for("static", filename=UPLOAD_FOLDER + "/" + filename), code=301)


@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":

        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No image selected for uploading")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(
                # os.path.join(app.config["STATIC_FOLDER"], os.path.join(app.config["UPLOAD_FOLDER"], filename))
                os.path.join(app.config["UPLOAD_FOLDER"], filename)
            )
            # print('upload_image filename: ' + filename)
            flash("Image successfully uploaded and displayed below")
            return render_template("responsive.html", filename=filename)
        else:
            flash("Allowed image types are -> png, jpg, jpeg, gif")
            return redirect(request.url)


# @app.route("/generate", methods=["GET", "POST"])
# def generate():
#     return "생성한 시"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6006, debug=True)
