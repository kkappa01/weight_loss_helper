## Weight Loss Helper 
[//]: # (Image References)

[image1]: ./motivation_func.png "Motivation Function"

---
**Introduction**
The objective of this project is to test and develop Reinforcment Learning based algorithms to help people achieve their weight loss goals. In order to accomplish this, the idea is to treat weight loss program as a game and at each step the system provides the user feedback/recommendations on activity/exercise, food intake (quantity and quality). The key is to:
a) Provide recommendations that helps achieve the goal
b) Ensure that feedback/recommendation provided matches the users motivation/state. i.e provide feedback that user will likely follow.

The second point is the key to designing an effective. non-trivial system. For example, asking the user to eat very little and exercise a lot will achieve a), but the user will likely not accept the recommendation.

In order to model b, the key insight is that a user is likely to accept a recommendation if the he is satisfied (had a large meal or lot of protiens), motivated (making progress towards the target weight).

The first step to achieve this is to create a simulation environment, developed in weight_loss_player.py. This class, models the weight and motivations. 

---
**Model**
(Disclaimer: This model is not based on real weight loss process, neither is it based on any research in the field. This is a rudimentry model, I have developed on my own to play with.)


States:
* weight
* satiety (How satisfied the user is after the last meal)
* motivation (How motivated the user is)

Observations:
* weight
* accepted_flag: 1 if the user accepted the provided recommendation, 0 is not

Actions:
* Activity Levels: ['active', 'light', 'sedentary', 'moderate', 'extreme']
* Calorie Intake Levels: ['over_achieve', 'indulge', 'achieve', 'extreme']
* Calorie Quality Levels: ['poor', 'great', 'normal']

Transitions:

1. pr(accept_recommendation | satiety, motivation) = w1 * satiety[t] +w2 * motivation[t]
2. With the above probabilty accept the reommendation or choose a random action (there can be a prior probability for this, but for now assuming uniform distribution)
3. delta_weight = basal_metabolic_rate * K_activity - (K_calorie_intake x K_calorie_quality)* K_goal
where: K_goal = (Initial_weight-Target_weight)/Target_duration
4) weight[t+1]=weight[t]+delta_weight
5) satiety[t+1] = K2_calorie_intake * K2_calorie_quality
6) Motivation[t+1] =  f((weight[t+1]- Target_weight)/Initial_weight)
![motivation_func.png][image1]
Desciption:
The user is likely to accept a recommendation if he had a good meal or if he is very motivated.

The change is weight is straight forward based on this a calorie deficit.

Satiety is modeled as the product of intake quantity and quality. Indulgence will have high satiety, extremely low intake will have low satiety. Similarly, protien rich food (great quality) will lead to high satiety and so does, high carb food.

Motivation is complex to model. Here it is assumed that the user would have some base motivtion (initial phase), it will start to increase as he/she makes progress (progress phase) and  will peak as he/she gets closer to goal, and towards the end settels down to a high value (target phase). Finally, if the weight increses he/she would have some patience and then get demotivated. Each of this phase is model using logistic function.

The python notebook Model_plots.ipynb shows the behavior of the model.

Note: The motivation model is implemented as a function so that it could be replaced by a neural netwrok later.

---
**Reinforcement Learning**
The RL system, only knows the weight of the user and weather the user has accepted or rejected prior recommendations. Using these it has to determine the next set of recommendations. Satitey and Motivation are unknown to the system. 
Currently only a rudimentry reward has been implemented.
* Reward is modeled as high (10) when target is achieved.
* If there is progress and a recommendation is accepted, -.8.
* If there is progress but the recommendation is not accepted -1
* If weight increases beyond initial weight -2
* If weight increases much beyond initial weight -10
 
The Q-Learning algorithm in player_Q_learning.ipynb has not been able perfrom well. This is most probably because of poorly chosen model parameters. There is very little sensityvity of the system to actions, this needs to be tuned. 

---
**Next Steps**
Determine better constants for the model.
Engineer better rewards
Implement DQN







