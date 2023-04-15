from my_imports import *
from src.exception import CustomException

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/indexform',methods=['GET','POST'])
def indexform():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        selected_preprocessing = request.form.getlist('preprocessing')
        data=CustomData(
                uploaded_file=request.files['uploaded_file'],
                sep=request.form["sep"],
                preprocessing=selected_preprocessing,
                word_embedding=request.form["word_embedding"],
                classifier=request.form["classifier"],
            )
        input_list=data.get_data_as_list()
        data.get_df()
        print(input_list)
        prediction=PredictPipeline()
        image_exists,output=prediction.run(input_list)
    return render_template("table.html",image_exists=image_exists ,d=output, f="")

@app.route('/download')
def download():
    df = pd.read_csv('artifacts/AI_ready_data.csv')
    response = Response(df.to_csv(index=False), content_type='text/csv')
    response.headers['Content-Disposition'] = 'attachment; filename=AI_ready_data.csv'
    return response

@app.route('/image')
def display_image():
    image_path = 'artifacts/plot.png'
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    else:
        return "Image not found", 404

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
