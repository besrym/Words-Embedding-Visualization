# Words Embedding & Visualization

# Installation
```
conda create -n we-venv -y python=3.7 && conda activate we-venv
pip install -r requirements.txt
```
# Usage

Insert project path e.g. : 
```
/Users/your_name/Desktop/WordsEmbedding
```
then you can choose between a tiny dataset (100 sentences) and a big dataset (15927 sentences).
Just uncomment the variable which you want to use in main.py. This script is computationally intensive 
and was written for educational purpose. I would recomment to use the tiny dataset.

then run the script:
```
python main.py
```

# Training

i used:
- optimizer: Stochastic gradient descent
- epochs: 10
- learning rate: 0.01
- momentum:   0.9

![](embedding.jpg)


for targetword in sentence:  
&emsp;&emsp;&emsp;&emsp;&emsp;for contextword arround targetword:  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;embedding = matmul(E, targetword)  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;tmp = matmul(W, embedding)  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;predicted_contextword = softmax(tmp)  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;minimize(predicted_contextword, contextword)

# Visualization

For the visualization I used the dimension reduction method T-SNE.



# Contact
If you have any Input for me to make the training more efficient or better feel free to contact me.




