# SAM Machine Learning for FTA Question Answer
This project will infer a correct or incorrect label to a students answer when given that answer, the question, and the document text. The deployment creates a Lambda function which loads a Docker container with Python 3.8 and Pytorch as well as Transformers and other required libraries.  
  
Currently hard coded for the question id 7b7fd45f-4086-4cfd-adaa-409c1f8e65c1  
  
# Inference  
  
const thisResult = await fetch(  
      "https://b2oy73nnmh.execute-api.ap-southeast-2.amazonaws.com/Prod/score",  
      {  
        method: "POST",  
        headers,  
        body: JSON.stringify({  
          question: "How does budgeting help with financial goals?",  
          answer: "Budgeting does not allow for rethinking spending habits",  
          document_text: "A budget helps create financial stability and puts a person on stronger financial footing for both the day-to-day and the long term. By tracking expenses and following a plan, a budget makes it easier to pay bills on time, build an emergency fund, help you set long-term financial goals, keep you from overspending, and help shut down risky spending habits. Helps You Prepare for Emergencies: An emergency fund should consist of at least three to six months’ worth of living expenses. Whilst working out a budget you will be able to determine the size of the emergency fund, and also have a savings plan to build this up. Helps You Work Toward Long-Term Goals: A budget forces you to map out your goals, save your money, and keep track of your progress - making your dreams a reality. By seeing what money you earn and what money you have going out through a budget, you can create a map for where you need to go to get your goal. Can Keep You from Overspending: Having a budget means you’ll know exactly how much money you earn, how much you can afford to spend each month, and how much you need to save. Can Reveal Spending Habits: Building a budget forces you to take a close look at your spending habits. Budgeting allows you to rethink your spending habits and refocus your financial goals.",  
        }),  
      }  
    );  
  

# Training
Open the training.ipynb Jupyter notebook in Google Colab  
Set the instance for high RAM, and use GPU over CPU  
Training data is a csv file also included in this repo. The notebook will ask you to upload this.  
  
Once notebook is finished excuting, download the config.json file and the python_model.bin file and copy them into the app folder.  

Then deploy.  

# Deployment

Requires the following to be installed:  
AWS SAM CLI  
Docker Desktop  

Run:  
sam build  
sam deploy --guided  

# Notes
  
* * * NOT CURRENTLY DEPLOYED * * *  
  
The API will timeout on the first run and maybe the second as the docker container boots. There are various ways to fix this in Production deployment. 

There are four databases set up for this project, but deployed in the portal project.  
- portal-dev-ftaMlQuestionDataTable-S6V953BQWJOV 
- portal-dev-ftaMlQuestionTextTable-Y1FPS992EK6  
- portal-prod-ftaMlQuestionDataTable-17CRBBEYRMUVN  
- portal-prod-ftaMlQuestionTextTable-X8G1EBI6DU00  
  
These will save the predictions data to check the performance of the model.  
See the data at https://portal.financialtrainingaustralia/fch/mlTest




