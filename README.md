The model is built by Tensorflow framework. For the text responses part, some preprocessing techniques 
have been used such as, remove stops, lower cases, stemming, remove punctuations and etc. This would help model 
to generalise the data better without too much inference of noise data. 

In order to improve the performance of the model, there are several ways to do it.
Fristly, to improve data quality and gain more data, this part is the most importent step of machine learning process.

Secondly, hyperparameters tunning, this is iterative process for neural network there are some hyperparameters we can
play with including, learning rate, number of hidden units, optimizer, activation function, kernel initializer
regularization (L1 or L2 or Elastic Net), batch size and etc.

Thirdly, feature engineering and data augmentation.

For docker deployment in AWS I would use Fargate, Fargate automatically allocates the appropriate amount of computation, 
removing the need to select instances and scale cluster capacity. It is serverless compute engine for containers that 
works with both Amazon Elastic Container Service (ECS) and Amazon Elastic Kubernetes Service (EKS).

If the service deployed as a web application, there are several AWS services could help to reduce response time and 
increase throughput such as AWS Elasticache, AWS Global Accelerator, CloudFront. For the application running in vitural 
machine like EC2, we can use the CloudWatch to monitor the some metrics such as CPU utilization to trigger
auto scalling group to create more EC2 instances.



For local machine:

    pip install -r requirements.text
    python main.py # to train the model
    python app.py # to start a flask server
    python request.py # to post data to api to do prediction
    
For docker:

    docker build . --tag<your image name>
    
    
