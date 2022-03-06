# fashion_sgan
Colab link => https://colab.research.google.com/drive/12XE8s0flM1-LWTQ1Ljsxrxt75Mp8OZ4y
Accuracy 86% sample data 14% used

## Learnings
- With 4% data, the accuracy reached was 83%. However to increase accuracy, only suitable method found was increasing data size
- Increasing data size to 15% only helped accuracy increase by 3-4 % only
- The code adapted is not generalistic and hypertuning would be required for different usecases. I had not used Batch normalization only for generator but not for Discriminator.

## Running the code
- Prefrably run on Google colab with GPU runtime. 
- Without GPU time taken is 7 hours
- I havent added a python script for now. But can be done if needed.

## Libraries
- tensorflow>=2.1

## Resources
The code was adapted using following resources
- https://machinelearningmastery.com/semi-supervised-generative-adversarial-network/
- https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/
- https://arxiv.org/abs/1606.03498 - Improving GAN

## Other research considered
- Feature Matching and manifold regularization https://arxiv.org/abs/1805.08957
- Feature matching wasn't for the following reason: Feature matching is effective when the GAN model is unstable during training.
- https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b

## What more could have be done
- Current submitted as a notebook, would have preffered making a script.
- MLops Lifecycle using MLflow 
- Running on containers


