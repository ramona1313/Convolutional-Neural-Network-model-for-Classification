# Convolutional-Neural-Network-model-for-Classification

## Classify all the 43 key-poses of Taiji (art-form) based on MoCAP + Footpressure dataset. 
Below are the two main things I performed with dataset:

1. Adding in 1st and/or 2nd order kinematic pose information into classification while
testing a range of temporal windows for velocity and acceleration calculation. Velocity
and Acceleration of joints at different temporal windows may have impact or not on classification
accuracy of any ML technique. Basically can position, velocity, and acceleration
and difierent low pass filtering windows impact classification accuracy.

2. Evaluate the impact of foot pressure resolution (spatial scale) on classification accuracy. 
Holding all other aspects constant, and see 'Does reducing the resolution of the
foot pressure impact the classification accuracy?' Systematically reduce resolution from
current 60x21x2 (2000+) pixels to less than 20 pixels and see classification accuracy.
