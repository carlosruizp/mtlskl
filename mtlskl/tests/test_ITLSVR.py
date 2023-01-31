import unittest
from sklearn.metrics import mean_absolute_error
from mtlskl.model.svm.itl.ITLSVM import ITLSVR
from mtlskl.data.DataLoader import DataLoader

from icecream import ic


class TestITLSVR(unittest.TestCase):
    def setUp(self):
        data_loader = DataLoader(dataset_name='computer')
        self.X, self.y, self.task_info = data_loader.data, data_loader.target, data_loader.task_info
        self.estim = ITLSVR()
        
    def test_fit(self):
        self.estim.fit(self.X, self.y)
        self.assertIsNotNone(self.estim.estim)
        
    def test_predict(self):
        self.estim.fit(self.X, self.y)
        y_pred = self.estim.predict(self.X)
        self.assertIsNotNone(y_pred)
        
    def test_score(self):
        self.estim.fit(self.X, self.y)
        y_pred = self.estim.predict(self.X)
        score = mean_absolute_error(self.y, y_pred)
        self.assertLessEqual(score, 2)

if __name__ == '__main__':
    unittest.main()