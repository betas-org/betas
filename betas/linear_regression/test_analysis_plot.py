"""
Unit tests for analysis_plot.py
"""

import seaborn as sns
import unittest
from unittest.mock import patch
from linear_regression import analysis_plot as plt

DF = sns.load_dataset('iris')
PREDS = [DF.columns[0]]
RESP = DF.columns[1]
MYPLOT = plt.analysis_plot(DF, predictors=PREDS, response=RESP)

class TestAnalysisPlot(unittest.TestCase):
    def test_init_predictors(self):
        """
        Check constructor error if predictors not chosen from dataframe
        """
        try:
            TESTPREDS = PREDS + ['somename']
            TESTPLOT = plt.analysis_plot(DF, predictors=TESTPREDS)
        except Exception as err:
            self.assertEqual(err.args[0],
                             'Input predictor variable(s) not existed in the given dataframe')

    def test_init_response(self):
        """
        Check constructor error if response not chosen from dataframe
        """
        try:
            TESTRESP = 'somename'
            TESTPLOT = plt.analysis_plot(DF, response=TESTRESP)
        except Exception as err:
            self.assertEqual(err.args[0],
                             'Input response variable not existed in the given dataframe')

    def test_get_dataframe(self):
        """
        Test if dataframe matched
        """
        self.assertEqual(MYPLOT.get_dataframe().all().all(), DF.all().all(), "Dataframe not matched")

    def test_get_predictors(self):
        """
        Test if predictors matched and in dataframe columns
        """
        self.assertEqual(MYPLOT.get_predictors(), PREDS, "Predictors not matched")
        for PRED in MYPLOT.get_predictors():
            self.assertIn(PRED, DF.columns, "Predictors not in dataframe columns")

    def test_get_response(self):
        """
        Test if response matched and in dataframe columns
        """
        self.assertIn(MYPLOT.get_response(), RESP, "Predictors not matched")
        self.assertIn(MYPLOT.get_response(), DF.columns, "Predictors not in dataframe columns")
    
    def test_set_predictors(self):
        """
        Test predictors matched
        """
        TESTPREDS = [DF.columns[1]]
        MYPLOT.set_predictors(TESTPREDS)
        self.assertEqual(MYPLOT.get_predictors(), TESTPREDS, "Predictors not matched")
    
    def test_set_response(self):
        """
        Test response matched
        """
        TESTRESP = DF.columns[2]
        MYPLOT.set_response(TESTRESP)
        self.assertEqual(MYPLOT.get_response(), TESTRESP, "Response not matched")
    
    def test_set_predictors_err(self):
        """
        Check set error if predictors not chosen from dataframe
        """
        try:
            TESTPREDS = PREDS + ['somename']
            MYPLOT.set_predictors(TESTPREDS)
        except Exception as err:
            self.assertEqual(err.args[0],
                             'Input predictor variable(s) not existed in the given dataframe')

    def test_set_response_err(self):
        """
        Check set error if response not chosen from dataframe
        """
        try:
            TESTRESP = 'somename'
            MYPLOT.set_response(TESTRESP)
        except Exception as err:
            self.assertEqual(err.args[0],
                             'Input response variable not existed in the given dataframe')

    def test_matrix_plot_label(self):
        """
        Test matrix_plot with assigned label
        """
        MYPLOT.matrix_plot(label='species')

    def test_matrix_plot_no_label(self):
        """
        Test matrix_plot without assigned label
        """
        MYPLOT.matrix_plot()

    def test_corr_heatmap_figsize(self):
        """
        Test corr_heatmap with assigned figsize
        """
        MYPLOT.corr_heatmap(figsize=(10,15))

    def test_corr_heatmap_no_figsize(self):
        """
        Test corr_heatmap without assigned figsize
        """
        MYPLOT.corr_heatmap()

    def test_reg_plot(self):
        """
        Test reg_plot
        """
        MYPLOT.reg_plot(DF.columns[0], DF.columns[1])

    def test_box_plot(self):
        """
        Test box_plot
        """
        MYPLOT.box_plot(DF.columns[0], DF.columns[1])

    def test_dist_plot(self):
        """
        Test dist_plot
        """
        MYPLOT.dist_plot(DF.columns[0], 'species')

    def test_reg_print_report(self):
        """
        Test linear regression model, no input, print report summary
        """
        with patch('builtins.print') as mock_print:
            model = MYPLOT.reg(report=True)
            # string is the first line of report with no space
            string = str(model.summary())[:78].replace(' ', '')
            self.assertEqual(string, 'OLSRegressionResults')

    def test_reg_assign_arg(self):
        """
        Test linear regression model, with input
        """
        model = MYPLOT.reg(var_x=DF.columns[0], var_y=DF.columns[1])
        string = str(model.summary())
        # assigned arguments should take priority in model fitting
        self.assertIn(str(DF.columns[0]), string)
        self.assertIn(str(DF.columns[1]), string)

    def test_reg_arg_err(self):
        """
        Check error for linear regression model, with input not being casted
        """
        try:
            model = MYPLOT.reg(var_x=DF.columns[0], var_y='species')
        except Exception as err:
            self.assertEqual(err.args[0],
                             'Predictor/Response data type cannot be casted. Please select again')

    def test_resid_plot(self):
        """
        Test resid_plot
        """
        MYPLOT.resid_plot()

    def test_resid_plot_assign_arg(self):
        """
        Test resid_plot, with input
        """
        MYPLOT.resid_plot(var_x=DF.columns[0], var_y=DF.columns[1])

    def test_resid_plot_arg_err(self):
        """
        Check error for resid_plot, with no assigned variables
        """
        try:
            TESTPLOT = plt.analysis_plot(DF)
            TESTPLOT.resid_plot()
        except Exception as err:
            self.assertEqual(err.args[0],
                             'No predictors or response assigned')

    def test_qq_plot(self):
        """
        Test qq_plot
        """
        MYPLOT.qq_plot()

    def test_qq_plot_assign_arg(self):
        """
        Test qq_plot, with input
        """
        MYPLOT.qq_plot(var_x=DF.columns[0], var_y=DF.columns[1])

    def test_qq_plot_arg_err(self):
        """
        Check error for qq_plot, with no assigned variables
        """
        try:
            TESTPLOT = plt.analysis_plot(DF)
            TESTPLOT.qq_plot()
        except Exception as err:
            self.assertEqual(err.args[0],
                             'No predictors or response assigned')

    def test_scale_loc_plot(self):
        """
        Test scale_loc_plot
        """
        MYPLOT.scale_loc_plot()

    def test_scale_loc_plot_assign_arg(self):
        """
        Test scale_loc_plot, with input
        """
        MYPLOT.scale_loc_plot(var_x=DF.columns[0], var_y=DF.columns[1])

    def test_scale_loc_plot_arg_err(self):
        """
        Check error for scale_loc_plot, with no assigned variables
        """
        try:
            TESTPLOT = plt.analysis_plot(DF)
            TESTPLOT.scale_loc_plot()
        except Exception as err:
            self.assertEqual(err.args[0],
                             'No predictors or response assigned')

    def test_resid_lever_plot(self):
        """
        Test resid_lever_plot
        """
        MYPLOT.resid_lever_plot()

    def test_resid_lever_plot_assign_arg(self):
        """
        Test resid_lever_plot, with input
        """
        MYPLOT.resid_lever_plot(var_x=DF.columns[0], var_y=DF.columns[1])

    def test_resid_lever_plot_arg_err(self):
        """
        Check error for resid_lever_plot, with no assigned variables
        """
        try:
            TESTPLOT = plt.analysis_plot(DF)
            TESTPLOT.resid_lever_plot()
        except Exception as err:
            self.assertEqual(err.args[0],
                             'No predictors or response assigned')

if __name__ == "__main__":
    SUITE = unittest.TestLoader().loadTestsFromTestCase(TestAnalysisPlot)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
