import unittest
import app

class MyAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app =  app.app.test_client()

    def test_initial(self):
        """
        Test the get request.form
        :return:
        """
        rv = self.app.get('/')
        assert rv.status_code ==200, "Should be" + str(rv.status_code)
        assert b'AAPL'in rv.data


    def test_valid_input_stock(self):
        """
        test the post request.form
        :return:
        """
        test_form = [
            {'companyname':'NFLX', 'ReferenceStartPeriod': '2022-07-01', 'PredictionDate':'2022-12-09'}
        ]
        rv = self.app.post('/', data = test_form[0])
        assert rv.status_code == 200, "Should be" + str(rv.status_code)
        assert b'NFLX' in rv.data

    def test_invalid_input(self):
        """
        Test the  invalid input
        :return:
        """
        test_form = [
            # invalid stock name
            {'companyname': 'MONASH', 'ReferenceStartPeriod': '2022-07-01', 'PredictionDate':'2022-12-09'},
            # reference period > prediction date
            {'companyname': 'AMZN', 'ReferenceStartPeriod': '2022-12-01', 'PredictionDate': '2022-10-09'},
            # reference period < prediction date
            {'companyname': 'AMZN', 'ReferenceStartPeriod': '2022-10-01', 'PredictionDate': '2022-09-09'},

        ]
        for form in test_form:
            rv = self.app.post('/', data = test_form[0])
            assert rv.status_code == 200, "Should be " + str(rv.status_code)


    def validation_test(self):
        test_form = [
            # empty
            {'companyname': '', 'ReferenceStartPeriod': '', 'PredictionDate': ''},
            # reference period > prediction date
            {'companyname': 'amzn', 'ReferenceStartPeriod': '2022-07-01', 'PredictionDate': '2022-12-09'},
            # reference period < prediction date
            {'companyname': 'AMZN', 'ReferenceStartPeriod': '2000-01-01', 'PredictionDate': '2023-12-31'}
        ]
        for form in test_form:
            rv = self.app.post('/', data=test_form[0])
            assert rv.status_code == 200, "Should be " + str(rv.status_code)