import pandas as pd
import numpy as np

from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import accuracy_score, cohen_kappa_score

from hq.hdf.logging.logger import *


def classify_on_aspects(scalar_df):

    scalar_df = classify_on_heartrate(scalar_df)
    scalar_df = classify_on_hrv(scalar_df)
    scalar_df = classify_on_sleep(scalar_df)
    scalar_df = classify_on_activity(scalar_df)

    return scalar_df


def classify_on_heartrate(scalar_df):

    logging.info('Calculating Heart Rate UL Feature')

    feature_aspect = 'hr'

    classify_on = ['sedentary_mean_hr_mean',
                   'wake_no_motion_hr_amplitude_mean',
                   'wake_no_motion_hr_median_mean', 'wake_no_motion_hr_median_std',
                   'sleep_no_motion_hr_amplitude_mean',
                   'sleep_no_motion_hr_median_mean', 'sleep_no_motion_hr_median_std', 'sleep_no_motion_hr_5perc_mean',
                   'no_motion_hr_amplitude_mean', 'no_motion_hr_amplitude_std',
                   'no_motion_hr_dip_mean']

    scalar_df = physiological_health(scalar_df, classify_on, feature_aspect)

    return scalar_df


def classify_on_hrv(scalar_df):

    logging.info('Calculating Heart Rate Variability UL Feature')

    feature_aspect = 'hrv'

    classify_on = ['deep_hrv_sdnn_median_20_mean', 'deep_hrv_sdnn_median_20_std',
                   'deep_hrv_rmssd_median_20_mean', 'deep_hrv_rmssd_median_20_std',
                   'deep_hrv_pnn50_median_20_mean', 'deep_hrv_pnn50_median_20_std',
                   'RR_diff_std_mean', 'RR_diff_std_std',
                   'RR_diff_mean_mean', 'RR_diff_mean_std']

    scalar_df = physiological_health(scalar_df, classify_on, feature_aspect, n_components=2, switch=True)

    return scalar_df


def classify_on_sleep(scalar_df):

    logging.info('Calculating Sleep UL Feature')

    feature_aspect = 'sleep'

    classify_on = ['waso_mean', 'waso_std',
                   'total_sleep_mean', 'total_sleep_std',
                   'sleep_rem_perc_mean', 'sleep_rem_perc_std',
                   'sleep_light_perc_mean', 'sleep_light_perc_std',
                   'sleep_deep_perc_std',
                   'sleep_wake_perc_mean', 'sleep_wake_perc_std',
                   'sleep_continuity_mean', 'sleep_continuity_std',
                   'sleep_quality_mean', 'sleep_quality_std',
                   'breathing_rate_std_mean', 'breathing_rate_std_std',
                   'breathing_rate_amp_mean', 'breathing_rate_amp_std',
                   'apnea']

    scalar_df = physiological_health(scalar_df, classify_on, feature_aspect, n_components=2)

    return scalar_df


def classify_on_activity(scalar_df):

    logging.info('Calculating Activity UL Feature')

    feature_aspect = 'activity'

    classify_on = ['sedentary_mean_hr_mean',
                   'light_active_time_mean',
                   'moderate_active_time_mean',
                   'vigorous_active_time_mean',
                   'active_time_mean', 'active_time_std',
                   'steps_mean', 'energy_expenditure_30min_max_mean',
                   'energy_expenditure_30min_std_mean', 'energy_expenditure_30min_mets_mean']

    scalar_df = physiological_health(scalar_df, classify_on, feature_aspect, n_components=2)

    return scalar_df


def physiological_health(scalar_df, classify_on, feature_aspect, n_components=2, switch=False):

    if switch:
        cat_names = [feature_aspect + '_clf_unhealthy', feature_aspect + '_clf_healthy']
    else:
        cat_names = [feature_aspect + '_clf_healthy', feature_aspect + '_clf_unhealthy']

    df = scalar_df

    # only get what is needed from the dataframe
    features = np.concatenate([classify_on, ['lifeq_id', 'KfCV', 'HA_status']])

    df = df[features]

    if len(df) > len(df.dropna()):
        logging.warn('Removing %i individual(s) because of Nan Values' % (len(df) - len(df.dropna())))
        df = df.dropna()

    if len(df) > 10:
        if set(['meta_gender']).issubset(list(df.keys())):
            idx = df[df['meta_gender'] == 'male'].index
            df.loc[idx, 'meta_gender'] = float(0.0)
            idx = df[df['meta_gender'] == 'female'].index
            df.loc[idx, 'meta_gender'] = float(1.0)

        if set(['meta_bmi']).issubset(list(df.keys())):
            df = df[df['meta_bmi'] < 70]

        clf_normalised_features = []

        # normalise the features
        for key in list(df.keys()):
            if set([key]).issubset(set(['lifeq_id', 'KfCV', 'HA_status'])):
                pass
            else:
                # Normalise each of the features
                df['norm_' + key] = (df[key] - np.mean(df[key])) / np.std(df[key])
                clf_normalised_features.append('norm_' + key)

        means_cat0 = []
        means_cat1 = []

        # train and store results on KfCV
        for test_set in range(1, 11):

            # testing set and anon_ids
            testing_points = df[df.KfCV == test_set]
            testing_idxs = testing_points['lifeq_id']
            testing_refs = np.asarray(testing_points['HA_status'])
            testing_points = np.asarray(testing_points[clf_normalised_features])
            testing_points = testing_points.astype(float)

            # training set and anon_ids
            training_points = df[df.KfCV != test_set]
            training_points = np.asarray(training_points[clf_normalised_features])
            training_points = training_points.astype(float)

            # Initialise the model for training and fit test set
            model = BayesianGaussianMixture(n_components=n_components,
                                            covariance_type='full',
                                            tol=0.0001,
                                            max_iter=1000,
                                            n_init=15)

            model_fit = model.fit(X=training_points)
            model_res = model.predict(testing_points)

            if model_fit.means_[0][0] < model_fit.means_[1][0]:

                if cohen_kappa_score(testing_refs, model_res) < 0:
                    # catagory_1 is healthy
                    model_res[model_res == 1] = 2
                    model_res[model_res == 0] = 1
                    model_res[model_res == 2] = 0

                means_cat0.append(model_fit.means_[0])
                means_cat1.append(model_fit.means_[1])

                # calculate a score for each catagory
                distances = []
                for cat_nr, cat_name in zip([0, 1], cat_names):
                    mean_vals = np.matrix(model_fit.means_[cat_nr])
                    prec_vals = np.matrix(model_fit.precisions_[cat_nr])

                    distance = []
                    dist_df = df[clf_normalised_features]
                    for t_idx in dist_df.index:
                        obs_vals = np.matrix(dist_df.loc[t_idx].values)
                        distance.append(float((obs_vals - mean_vals) * prec_vals * (obs_vals - mean_vals).T))

                    distances.append(distance)

                for tst_idx, mdl_res, dist0, dist1 in zip(testing_idxs, model_res, distances[0], distances[1]):
                    scalar_df.loc[scalar_df[scalar_df['lifeq_id'] == tst_idx].index, feature_aspect + '_clf_HA_status'] = mdl_res
                    scalar_df.loc[scalar_df[scalar_df['lifeq_id'] == tst_idx].index, cat_names[0]] = dist0
                    scalar_df.loc[scalar_df[scalar_df['lifeq_id'] == tst_idx].index, cat_names[1]] = dist1
                    scalar_df.loc[scalar_df[scalar_df['lifeq_id'] == tst_idx].index, feature_aspect + '_clf_difference'] = dist1

                print 'Cat 0 Test Set %i \t\t %i \t %3.2f \t %3.2f' % (test_set,
                                                                       len(df[df.KfCV == test_set]),
                                                                       accuracy_score(testing_refs, model_res),
                                                                       cohen_kappa_score(testing_refs, model_res)
                                                                       )

            else:

                if cohen_kappa_score(testing_refs, model_res) < 0:
                    # catagory_1 is healthy
                    model_res[model_res == 1] = 2
                    model_res[model_res == 0] = 1
                    model_res[model_res == 2] = 0

                means_cat0.append(model_fit.means_[1])
                means_cat1.append(model_fit.means_[0])

                distances = []
                for cat_nr, cat_name in zip([1, 0], cat_names):
                    mean_vals = np.matrix(model_fit.means_[cat_nr])
                    prec_vals = np.matrix(model_fit.precisions_[cat_nr])

                    distance = []
                    dist_df = df[clf_normalised_features]
                    for t_idx in dist_df.index:
                        obs_vals = np.matrix(dist_df.loc[t_idx].values)
                        distance.append(float((obs_vals - mean_vals) * prec_vals * (obs_vals - mean_vals).T))

                    distances.append(distance)

                for tst_idx, mdl_res, dist0, dist1 in zip(testing_idxs, model_res, distances[0], distances[1]):
                    scalar_df.loc[scalar_df[scalar_df['lifeq_id'] == tst_idx].index, feature_aspect + '_clf_HA_status'] = mdl_res
                    scalar_df.loc[scalar_df[scalar_df['lifeq_id'] == tst_idx].index, cat_names[0]] = dist0
                    scalar_df.loc[scalar_df[scalar_df['lifeq_id'] == tst_idx].index, cat_names[1]] = dist1
                    scalar_df.loc[scalar_df[scalar_df['lifeq_id'] == tst_idx].index, feature_aspect + '_clf_difference'] = dist1 - dist0

                print 'Cat 1 Test Set %i \t\t %i \t %3.2f \t %3.2f' % (test_set,
                                                                       len(df[df.KfCV == test_set]),
                                                                       accuracy_score(testing_refs, model_res),
                                                                       cohen_kappa_score(testing_refs, model_res)
                                                                       )

    else:
        logging.warn('Not enough data to calculate classification features')

    return scalar_df