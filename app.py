from flask import Flask,request,render_template

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/indexform',methods=['GET','POST'])
def indexform():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    return render_template('table.html')
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
