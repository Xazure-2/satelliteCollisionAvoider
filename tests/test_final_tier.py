import unittest
import numpy as np
import pandas as pd
from src.final_tier import (tier_from, final_labels, event_features, candidates, decide, tune_thresholds,
                            make_classifier, predict_events, prefix_features, fit_spec, bundle_decide,
                            persistence, LATEST_ONLY)
from tests.test_model import fixture


def messages():
    x = fixture()  # 20 events x 4 messages
    x['time_to_tca'] = np.tile([5., 3., 1., 0.2], 20)
    return x


class FinalTierTests(unittest.TestCase):
    def test_tier_boundaries_and_placeholder(self):
        self.assertEqual(tier_from([-4, -4.01, -6, -6.01, -30, -30, -30], [9e3, 9e3, 9e3, 9e3, 999, 1000, 5001]).tolist(),
                         ['Critical', 'Elevated', 'Elevated', 'Safe', 'Critical', 'Elevated', 'Safe'])

    def test_final_label_is_smallest_time_to_tca_even_after_tca(self):
        x = pd.DataFrame({'event_id': [1, 1, 1, 2], 'time_to_tca': [3., 0.5, -0.1, 1.],
                          'risk': [-30., -30., -3., -30.], 'miss_distance': [9e3, 9e3, 9e3, 9e3]})
        labels = final_labels(x)
        self.assertEqual(labels.loc[1, 'final_tier'], 'Critical')
        self.assertTrue(labels.loc[1, 'usable']); self.assertFalse(labels.loc[2, 'usable'])

    def test_features_ignore_messages_inside_two_days_and_labels(self):
        x = messages()
        base = event_features(x)
        changed = x.copy()
        late = changed.time_to_tca < 2
        changed.loc[late, ['miss_distance', 'max_risk_estimate', 'c_sigma_r']] = [1., -1., 5.]
        changed['risk'] = 0.
        pd.testing.assert_frame_equal(base, event_features(changed))
        self.assertEqual(len(base), 20)
        self.assertTrue((base.latest_time_to_tca == 3.).all())
        self.assertTrue((base.message_count == 2).all())

    def test_candidates_reject_label_columns(self):
        f = event_features(messages())
        with self.assertRaises(ValueError):
            candidates(f.assign(final_tier='Safe'))

    def test_decision_rule_and_thresholds(self):
        proba = pd.DataFrame({'Safe': [.9, .5, .2], 'Elevated': [.05, .4, .3], 'Critical': [.05, .1, .5]})
        self.assertEqual(decide(proba, .4, .45).tolist(), ['Safe', 'Elevated', 'Critical'])
        tc, te = tune_thresholds(np.array(['Safe', 'Elevated', 'Critical']), proba)
        self.assertEqual(decide(proba, tc, te).tolist(), ['Safe', 'Elevated', 'Critical'])

    def test_predict_events_one_row_per_event(self):
        x = messages()
        f = event_features(x)
        y = np.tile(['Safe', 'Elevated', 'Critical', 'Safe'], 5)
        model = make_classifier(LATEST_ONLY).set_params(estimator__max_iter=3, estimator__min_samples_leaf=2).fit(f, y)
        bundle = {'pipeline': model, 'metadata': {'thresholds': [.5, .5], 'model_version': 't'}}
        out = predict_events(x, bundle)
        self.assertEqual(out.event_id.tolist(), sorted(x.event_id.unique()))
        np.testing.assert_allclose(out[['p_safe', 'p_elevated', 'p_critical']].sum(axis=1), 1)

    def test_hybrid_floor_only_raises(self):
        scores = pd.DataFrame({'critical': [0., 0., .9], 'elevated': [0., 0., .9]})
        self.assertEqual(decide(scores, .5, .5, floor=['Critical', 'Elevated', 'Safe']).tolist(),
                         ['Critical', 'Elevated', 'Critical'])

    def test_prefix_augmentation_covers_every_two_day_prefix(self):
        x = messages()
        features, source, weight = prefix_features(x, [0, 1])
        self.assertEqual(sorted(source.value_counts().tolist()), [2, 2])   # two >= 2-day messages per event
        np.testing.assert_allclose(weight.groupby(source).sum(), 1)
        full = event_features(x).loc[[0, 1]]
        longest = features.loc[features.message_count.eq(2)]
        np.testing.assert_allclose(longest.latest_miss_distance.to_numpy(), full.latest_miss_distance.to_numpy())

    def test_change_regressor_hybrid_bundle(self):
        x = messages()
        f = event_features(x)
        train = f.assign(final_tier=np.tile(['Safe', 'Elevated', 'Critical', 'Safe'], 5),
                         final_miss_distance=np.linspace(200, 9000, 20), final_time_to_tca=0.5)
        spec = dict(candidates(f)['HR1_distance_change_regressor'])
        spec['columns'] = LATEST_ONLY
        model = fit_spec(spec, train)
        bundle = {'pipeline': model, 'metadata': {'spec': spec, 'thresholds': [-3., -3.7], 'model_version': 't'}}
        decision = bundle_decide(bundle, f)
        floor = persistence(f, False)
        rank = {'Safe': 0, 'Elevated': 1, 'Critical': 2}
        self.assertTrue(all(rank[d] >= rank[p] for d, p in zip(decision, floor)))
        out = predict_events(x, bundle)
        self.assertIn('predicted_final_miss_distance_m', out)
        self.assertTrue((out.predicted_final_miss_distance_m > 0).all())


if __name__ == '__main__':
    unittest.main()
