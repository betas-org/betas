"""
Unittests for analysis_plot.py
"""

import unittest
import analysis_plot as plt
import seaborn as sns


DF = sns.load_dataset('titanic')
PREDS = [DF.columns[0]]
RESP = DF.columns[1]
MYPLOT = plt.analysis_plot(DF, PREDS, RESP)

class TestAnalysisPlot(unittest.TestCase):
    def test_get_dataframe(self):
        """
        Check if dataframe matched
        """
        self.assertEqual(MYPLOT.get_dataframe().all().all(), DF.all().all(), "Dataframe not matched")

    def test_get_predictors(self):
        """
        Check if predictors matched and in dataframe columns
        """
        self.assertEqual(MYPLOT.get_predictors(), PREDS, "Predictors not matched")
        for PRED in MYPLOT.get_predictors():
            self.assertIn(PRED, DF.columns, "Predictors not in dataframe columns")

    def test_get_response(self):
        """
        Check if response matched and in dataframe columns
        """
        self.assertIn(MYPLOT.get_response(), RESP, "Predictors no matched")
        self.assertIn(MYPLOT.get_response(), DF.columns, "Predictors not in dataframe columns")

if __name__ == "__main__":
    SUITE = unittest.TestLoader().loadTestsFromTestCase(TestAnalysisPlot)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
