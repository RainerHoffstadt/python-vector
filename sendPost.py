import json
import urllib3
debug = 0

def send(array):

    pulses = []
    pulses.append(1.5)
    for x in array:
        pulses.append(x)

    pulses.append(1.5)

    encoded_body = json.dumps({
          "array": pulses
    })
    if debug:
        return
    http = urllib3.PoolManager()
    http.urlopen('POST', 'http://arm:5000/setarm', headers={'Content-Type': 'application/json'},
                     body=encoded_body
    )
