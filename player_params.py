import numpy as np


class Player_info:
    def __init__(self,bio_params = {}, propensity_params = {}):
        # default initialization
        if not bio_params:
            self.initial_weight = 250  # lbs
            self.gender = 'male'
            self.height = 72  # inches
            self.age = 35  # year
        else:
            self.initial_weight = bio_params['initial_weight']
            self.gender = bio_params['gender']
            self.height = bio_params['height']
            self.age = bio_params['age']

        # default initialization for propensity
        if not propensity_params:
            self.initial_activity_level = 'sedentary'
            self.initial_calorie_intake = 'indulge'
            self.initial_calorie_quality = 'poor'
            self.motivation_function = self.motivation_model()
            self.satiety_w = .4
            self.motivation_w = .6

        else:
            self.initial_activity_level = propensity_params['initial_activity_level']
            self.initial_calorie_intake = propensity_params['initial_calorie_intake']
            self.initial_calorie_quality = propensity_params['initial_calorie_quality']
            self.motivation_function = propensity_params['motivation_function']
            self.satiety_w = propensity_params['satiety_w']
            self.motivation_w = propensity_params['motivation_w']

    def motivation_model(self, motivation_peak_x=.5, motivation_increase_x=.8, motivation_taper_x=1.1,
                         motivation_disengage_x=1.3, motivation_init_y=.6, motivation_end_y=.8):
        def motivation_func(x):
            if x <= motivation_peak_x:
                # Close to target
                shift_x = -motivation_peak_x / 2
                scale_x = 10 / motivation_peak_x
                scale_y = (1 - motivation_end_y)
                shift_y = motivation_end_y
                y = 1. / (1 + np.exp(-(x + shift_x) * scale_x)) * scale_y + shift_y

            elif x <= motivation_increase_x:
                # showing results
                shift_x = (motivation_increase_x - 1) / 2 - motivation_peak_x
                scale_x = 10 / (1 - motivation_increase_x)
                scale_y = 1 - motivation_init_y
                shift_y = motivation_init_y
                y = 1. / (1 + np.exp((x + shift_x) * scale_x)) * scale_y + shift_y
            elif x <= motivation_taper_x:
                # Close to orginal weight
                y = motivation_init_y
            else:
                # showing poor results
                shift_x = (motivation_taper_x - motivation_disengage_x) / 2 - motivation_taper_x
                scale_x = 10 / (motivation_disengage_x - motivation_taper_x)
                scale_y = motivation_init_y
                shift_y = 0

                y = 1. / (1 + np.exp((x + shift_x) * scale_x)) * scale_y + shift_y
            return y
        return motivation_func


