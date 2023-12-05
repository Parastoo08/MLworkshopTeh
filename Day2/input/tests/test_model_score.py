from nbresult import ChallengeResultTestCase
import pandas as pd
from sklearn.metrics import precision_score

class TestModel_score(ChallengeResultTestCase):
    def test_score(self):
        labels = pd.read_csv('seismic_events_target.csv')
        labels = labels.apply(lambda x: x!='earthquake')
        y_pred = pd.read_csv('../predictions.csv')
        score = precision_score(labels, y_pred)
        self.assertGreaterEqual(score, 0.92,msg=f"Your precision is {score} which is below 0.92.")

    def test_predicted_score(self):
        labels = pd.read_csv('seismic_events_target.csv')
        labels = labels.apply(lambda x: x!='earthquake')
        y_pred = pd.read_csv('../predictions.csv')
        score = precision_score(labels, y_pred)
        est_score = self.result.estimated_precision
        within_range = ((est_score <= (score+score*0.05)) and (est_score >= (score-score*0.05)))
        self.assertTrue(within_range, msg=f"Your estimated precision of {est_score} is not within ±20% of your precision on unseen data (precision of {score})")
    