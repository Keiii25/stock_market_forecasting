# Third party modules
import pytest

# # First party modules
from app import *

def test_dashboard():
    flask_app = main()
    test_client = flask_app.test_client
    response = test_client.get('/',methods ='GET')
    assert response.status_code == 200, "Code: " + str(response.status_code)
    assert b'Prediction Graph' in response.data
    # with flask_app.test_client as test_client:
    #     response = test_client.get('/',methods ='GET')
    #     # print(response.status_code)
    #     assert response.status_code == 200, "Code: " + str(response.status_code)
    #     assert b'Prediction Graph' in response.data