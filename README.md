# Operation-prediction

![Word Probem](images/word_problem.png?raw=true "Dicision of a question")
Takes a question with two operands as input and predicts the operation between them. This system presents a novel approach to
solve simple arithmetic word problems.We predict the operation that is to be performed (’-’,’+’,’*’,’/’) through a deep neu-
ral network model and then use it to generate the answer. 

The question is divided into two parts - worldstate and query as shown in Figure .

The worldstate and the query are processed separately in two different networks and finally the networks
are merged to predict the final operation. 

Our system learns to predict operationswith 88.81 % in a corpus of primary school questions. 


### Prerequisites

pip install -r requirements.txt

## Getting Started

1) Testing on our data 

python3 model.py DATA/train_LSTM_26112016 DATA/test_LSTM_26112016

2) ### Web
run
python3 model1.py

The function main_function returns the answer

API code is in main.py

3) Testing on ARIS

python3 test_aris.py

## Results and Comparisions

![Result](images/DifferentModelsAccuracy.png?raw=true "Dicision of a question")
![Baselines](images/Comparison.png?raw=true "Dicision of a question")

## Testing

You can change the input question 
eg - 
1) Joan found 70 seashells on the beach . she gave Sam some of her seashells . She has 27 seashell . How many seashells did she give to Sam ? 

2) There were 28 bales of hay in the barn . Tim stacked bales in the barn today . There are now 54 bales of hay in the barn . How many bales did he store in the barn ? 
3) Mary is baking a cake . The recipe wants 8 cups of flour . She already put in 2 cups . How many cups does she need to add ? 
4) Sara 's high school played 12 basketball games this year . The team won most of their games . They were defeated during 4 games . How many games did they win ? 

