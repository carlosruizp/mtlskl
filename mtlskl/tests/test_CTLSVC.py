import unittest
from sklearn.metrics import accuracy_score
from mtlskl.model.svm.ctl.CTLSVM import CTLSVC
from mtlskl.data.DataLoader import DataLoader

from icecream import ic


class TestCTLSVC(unittest.TestCase):
    def setUp(self):
        data_loader = DataLoader(dataset_name='compas')
        self.X, self.y, self.task_info = data_loader.data, data_loader.target, data_loader.task_info
        self.estim = CTLSVC()
        
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
        score = accuracy_score(self.y, y_pred)
        self.assertGreaterEqual(score, 0.6)

if __name__ == '__main__':
    unittest.main()