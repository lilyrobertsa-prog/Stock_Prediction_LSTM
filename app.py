from flask import Flask, request, render_template
import os


app = Flask(__name__, template_folder = 'templates')


print("Looking for templates in:", os.path.abspath(app.template_folder))
print("Files there:", os.listdir(app.template_folder))


@app.route('/')
def index():
    mylist = [10, 20, 30, 40, 50]
    return render_template(template_name_or_list='index.html', mylist = mylist)



@app.route('/hello')
def hello():
    return render_template(
        'index.html',
        myvalue='test value',
        myresult='test result'
    )


@app.route('/greet/<name>')
def greet(name):
    return f"Hello {name}"


@app.route('/add/<int:number1>/<int:number2>')
def add(number1, number2):
    return f"{number1} + {number2} = {number1 + number2}"



@app.route('/handle_url_params')
def handle_params():
    if 'greeting' in request.args.keys() and "name" in request.args.keys():

        greeting = request.args.get('greeting')
        name = request.args['name']
        return f"{greeting}, {name}"
    
    else:
        return 'Some parameters are missing'





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=555, debug=True)

