import numpy as np
import itertools
"""
A model to simulate a user responding to weight loss recommendations
Karthik Kappaganthu
11/11/2017
"""

class WeighLossPlayer:
    def __init__(self, bio_params={}, propensity_params={}, target_weight=180, target_days=200):
        # Information about the player (# Abstract this to a different class later)
        if not bio_params: # default initialization
            self.initial_weight = 250  # lbs
            self.gender = 'male'
            self.height = 72  # inches
            self.age = 35  # years
        else:
            self.initial_weight = bio_params['initial_weight']
            self.gender = bio_params['gender']
            self.height = bio_params['height']
            self.age = bio_params['age']

        # Motivation and initial tendencies model of the player
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

        # Targets
        self.target_weight = target_weight
        self.target_days = target_days
        self.target_weight_loss_per_day = (self.initial_weight - self.target_weight)*1./self.target_days
        self.fat_calories_per_pound = 3500.
        self.target_calories_per_day = self.target_weight_loss_per_day * self.fat_calories_per_pound

        # Noise model for weight change
        self.delta_w_mu = 0
        self.delta_w_std = 1

        # Define available actions
        self.categories_to_values()
        self.activity_levels = list(self.activity_multipier_values.keys())
        self.calorie_intake_levels = list(self.calorie_intake_multiplier_values.keys())
        self.calorie_quality_levels = list(self.calorie_quality_multiplier_values.keys())

        self.reset()

    def weight_to_bmr(self, wt):
        """Compute basal metabolic rate
        (https://www.livestrong.com/article/382511-what-should-my-calorie-to-exercise-ratio-be-to-lose-weight/)"""
        if self.gender=='female':
            bmr = wt * 4.3 + self.height * 4.7 + 65 - self.age * 4.7
        else:
            bmr = wt * 6.3 + self.height * 12.9 + 66 - self.age * 6.8
        return bmr

    def categories_to_values(self):
        """ Parameters to  numeric values in equations"""
        self.activity_multipier_values = {'sedentary': 1.2, 'light': 1.4, 'moderate': 1.6, 'active': 1.7, 'extreme': 1.9}
        max_factor = self.weight_to_bmr(self.initial_weight)*1.2*1.1/self.target_calories_per_day # if indulging and sedentary,weight would grow

        # self.calorie_intake_multiplier_values = {'extreme': max_factor*.2, 'over_achieve': max_factor*.3, 'achieve': max_factor*.5, 'indulge': max_factor}
        self.calorie_intake_multiplier_values = {'extreme': .75, 'over_achieve': .8,
                                                 'achieve': 1, 'indulge': 2.5}
        self.calorie_intake_to_satiety_values = {'extreme': .25, 'over_achieve': .8, 'achieve': .9, 'indulge': 1}
        self.calorie_quality_multiplier_values = {'poor': 1.1, 'normal': 1, 'great': .9}
        self.calorie_quality_to_satiety_values = {'poor': .9, 'normal': .8, 'great': 1}

    def motivation_model(self, motivation_peak_x=.5, motivation_increase_x=.8, motivation_taper_x=1.1,
                         motivation_disengage_x=1.3, motivation_init_y=.6, motivation_end_y=.8):
        """ The motivation of the player as function of the distance fom target.
        4 phases: Near original weight, Beginning to show process, Close to target, Deterioration in results
        Each phase is a scaled and shifted logistic equation.
        TODO: A cool idea would be to create predifined functions and label them as warrior, knight, etc...
        and make progression from one level to another
        TODO: Include motivation as a function time/velocity/rate of change. Eg. how long a certain weight level etc.
        """
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

    def normalized_weight_to_target(self, weight):
        """ Weight normalized to target """
        return (weight - self.target_weight)/(self.initial_weight - self.target_weight)

    def get_space_description(self):
        """ Create the space as a combination of the actions available"""
        action_space = list(itertools.product(self.activity_levels,self.calorie_intake_levels,self.calorie_quality_levels))
        wt_range = np.around(np.arange(self.target_weight-3,self.initial_weight+7,.1), decimals=1)# Descritize to the nearest .1
        accept_range=[0, 1] # Can observe if the player accepts a recommendation or not
        observation_space = list(itertools.product(wt_range,accept_range))
        return action_space, observation_space

    def reset(self):
        """ Initialized states and observations"""
        self.weight_t = self.initial_weight
        self.weight_t = round(2.0 * self.weight_t) / 2.0  # Round to nearest .5 lbs

        self.satiety_t = self.calorie_intake_to_satiety_values[self.initial_calorie_intake] * \
                         self.calorie_quality_multiplier_values[self.initial_calorie_quality]
        weight_ratio = self.normalized_weight_to_target(self.weight_t)
        self.motivation_t = self.motivation_function(weight_ratio)
        self.accepted_flag = 1
        return (self.weight_t, self.accepted_flag)

    def compute_reward(self, weight, accept_flag):
        slope=1./(self.target_weight-self.initial_weight)
        if weight <= self.target_weight:
            reward = 10
            done_flag = True
        elif weight < self.initial_weight:
            # reward = slope*(weight -self.initial_weight) * (-.5) + 0 * accept_flag
            reward = -1 + .2 * accept_flag
            done_flag=False
        elif weight < self.initial_weight+2:
            reward = -2
            done_flag = False
        else:
            reward = -10
            done_flag = True
        return reward, done_flag


    def step(self,action,prob_of_accept_override=[]):
        recommended_activity = action[0]
        recommended_calorie_intake = action[1]
        recommended_calorie_quality = action[2]

        # Based on Sateiety and Motivation from previous instance, decide if player would pursue recommended action or choose another action
        if not prob_of_accept_override:
            prob_of_accept = self.satiety_t*self.satiety_w+self.motivation_t*self.motivation_w
        else:
            prob_of_accept=prob_of_accept_override # Override the probability of taking an action (for testing, etc.)

        p = np.random.random()
        if p < prob_of_accept:
            activity_level = recommended_activity
            calorie_intake = recommended_calorie_intake
            calorie_quality = recommended_calorie_quality
            self.accepted_flag =1 # This is one of the observations
            self.random_ch = 0 # for debugging
        else:
            self.random_ch=1
            activity_level = np.random.choice(list(self.activity_multipier_values)) # Make the choice based on person model
            calorie_intake = np.random.choice(list(self.calorie_intake_multiplier_values))
            calorie_quality =np.random.choice(list(self.calorie_quality_multiplier_values))
            self.accepted_flag = 0

        # Get the input  values based on the chosen action
        activity_multiplier = self.activity_multipier_values[activity_level]
        calorie_intake_multiplier = self.calorie_intake_multiplier_values[calorie_intake]
        calorie_quality_multiplier = self.calorie_quality_multiplier_values[calorie_quality]

        # Weight update
        active_metabolic_rate = self.weight_to_bmr(self.weight_t) * activity_multiplier
        calories_in = (calorie_intake_multiplier*calorie_quality_multiplier) * self.target_calories_per_day
        calories_deficit = active_metabolic_rate - calories_in
        delta_weight = - calories_deficit/self.fat_calories_per_pound
        nue_t = np.random.normal(self.delta_w_mu, self.delta_w_std)
        self.weight_t = self.weight_t + delta_weight #+ nue_t
        self.weight_t=round(10.0 * self.weight_t)/10.0# Round to nearest .1 lbs



        # Satiety Update
        self.satiety_t = self.calorie_intake_to_satiety_values[calorie_intake] * \
                         self.calorie_quality_to_satiety_values[calorie_quality]

        # Motivation update
        weight_ratio = self.normalized_weight_to_target(self.weight_t)
        self.motivation_t = self.motivation_function(weight_ratio)

        # Compute Reward and exit condition
        self.reward_t, self.done_flag_t=self.compute_reward(self.weight_t, self.accepted_flag)

        return (self.weight_t, self.accepted_flag), self.reward_t, self.done_flag_t
