from flask import Flask, request

debug = False
if debug:
    from servo_pulse_mock import set_servo_pulse
    from AdcTest_mock import close
else:
    from servo_pulse import set_servo_pulse
    from AdcTest import close


app = Flask('webservice')


@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/setarm', methods=['POST'])
def process_json():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        #data = json.loads(request.data)
        ar = json['array']
        for i in range(5):
            print("chanal= ",i, "pulse=", ar[i])
            set_servo_pulse(i, ar[i])
        return json
    else:
        return 'Content-Type not supported!'

@app.route('/close', methods=['GET'])
def gclose():
    close()
    return 'close'

app.run(host="0.0.0.0")