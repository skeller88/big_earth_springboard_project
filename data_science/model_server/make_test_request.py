import json

import numpy as np
import requests

if __name__ == "__main__":
    arr = np.load('./S2B_MSIL2A_20180421T100031_7_89.npy')
    req = json.dumps({'image': arr.tolist()})
    response = requests.post(url='http://localhost:8889/classify', json=req)
    print(response.json())
