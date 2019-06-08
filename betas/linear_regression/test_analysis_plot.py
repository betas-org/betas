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
        self.assertIn(MYPLOT.get_response(), RESP, "Predictors no matched")
        self.assertIn(MYPLOT.get_response(), DF.columns, "Predictors not in dataframe columns")

    def test_set_predictors(self):
        """
        Check set error if predictors not chosen from dataframe
        """
        try:
            TESTPREDS = PREDS + ['somename']
            MYPLOT.set_predictors(TESTPREDS)
        except Exception as err:
            self.assertEqual(err.args[0],
                             'Input predictor variable(s) not existed in the given dataframe')

    def test_set_response(self):
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
        try:
            MYPLOT.matrix_plot(label='species')
        except:
            self.fail('matrix_plot with assigned label does not work')

    def test_matrix_plot_no_label(self):
        """
        Test matrix_plot without assigned label
        """
        try:
            MYPLOT.matrix_plot()
        except:
            self.fail('matrix_plot without assigned label does not work')

    def test_corr_heatmap_figsize(self):
        """
        Test corr_heatmap with assigned figsize
        """
        try:
            MYPLOT.corr_heatmap(figsize=(10,15))
        except:
            self.fail('corr_heatmap with assigned figsize does not work')

    def test_corr_heatmap_no_figsize(self):
        """
        Test corr_heatmap without assigned figsize
        """
        try:
            MYPLOT.corr_heatmap()
        except:
            self.fail('corr_heatmap without assigned figsize does not work')

    def test_reg_plot(self):
        """
        Test reg_plot
        """
        try:
            MYPLOT.reg_plot(DF.columns[0], DF.columns[1])
        except:
            self.fail('reg_plot does not work')

    def test_box_plot(self):
        """
        Test box_plot
        """
        try:
            MYPLOT.box_plot(DF.columns[0], DF.columns[1])
        except:
            self.fail('box_plot does not work')

    def test_dist_plot(self):
        """
        Test dist_plot
        """
        try:
            MYPLOT.dist_plot(DF.columns[0], 'species')
        except:
            self.fail('dist_plot does not work')

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
        try:
            MYPLOT.resid_plot()
        except:
            self.fail('resid_plot does not work')

    def test_resid_plot_assign_arg(self):
        """
        Test resid_plot, with input
        """
        try:
            MYPLOT.resid_plot(var_x=DF.columns[0], var_y=DF.columns[1])
        except:
            self.fail('resid_plot does not work')

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
        try:
            MYPLOT.qq_plot()
        except:
            self.fail('qq_plot does not work')

    def test_qq_plot_assign_arg(self):
        """
        Test qq_plot, with input
        """
        try:
            MYPLOT.qq_plot(var_x=DF.columns[0], var_y=DF.columns[1])
        except:
            self.fail('qq_plot does not work')

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
        try:
            MYPLOT.scale_loc_plot()
        except:
            self.fail('scale_loc_plot does not work')

    def test_scale_loc_plot_assign_arg(self):
        """
        Test scale_loc_plot, with input
        """
        try:
            MYPLOT.scale_loc_plot(var_x=DF.columns[0], var_y=DF.columns[1])
        except:
            self.fail('scale_loc_plot does not work')

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
        try:
            MYPLOT.resid_lever_plot()
        except:
            self.fail('resid_lever_plot does not work')

    def test_resid_lever_plot_assign_arg(self):
        """
        Test resid_lever_plot, with input
        """
        try:
            MYPLOT.resid_lever_plot(var_x=DF.columns[0], var_y=DF.columns[1])
        except:
            self.fail('resid_lever_plot does not work')

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

