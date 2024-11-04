# CS 441 Fall 2024 - homework 2 - Using Spark To train LLM
Author: Manh Phan

Email: dphan8@uic.edu

Youtube demo Deploying app in AWS: [https://youtu.be/aiObXevvZH4](https://youtu.be/aiObXevvZH4)

# Instruction on running this homework
Open this project at your favorite IDEA, my ideal IDEA is IntelliJ. But a terminal is also fine to run this homework.
The project is run using sbt command. Ensure you have Java 11 and Scala 2.13 installed before running this project.

Since this project uses Spark, you can run this homework in a Spark cluster in local environment. 
Download Spark 3.5.3 with prebuilt Hadoop and Scala 2.13 from the following website: [https://spark.apache.org/downloads.html](https://spark.apache.org/downloads.html).

To run this project: At the root of this project run this command:

Compile project: `sbt compile`

## There are two main class files for this project: Transformer & SparkLLMTraining
**Transformer** and **SparkLLMTraining** are basically the same: Each class has a main method that receives two arguments: <inputPath> <outputPath>

-  inputPath: is the text file containing your data you want to feed to LLM
-  outputPath: is the result of generated sentence after model has been trained. The sentence will start with "The man starts with" and model attempts to generate at most 20 more words

But **Transformer** is to train model in local sbt environment, while **SparkLLMTraining** is to train model with Spark. 
SparkLLMTraining is "basically Transformer that is wrapped with Spark context"

- **For Transformer**: run in sbt environment or terminal the command: `sbt run Transformer <inputPath> <outputPath>`
- **For SparkLLMTraining**: first you need to convert the project into jar file using the following command: `sbt assembly`. 
Then use the jar file, submit it to Spark using the command "spark-submit", with some configurations, for example:
`spark-submit --class SparkLLMTraining --driver-memory 8g --master local[*] <jarPath> <inputPath> <outputPath>`

From this spark command, we want to use all cores (local[*]), and use 8GB of memory (driver-memory) to execute the job. 
We specify SparkLLMTraining is the class that we want to execute, and we given the arguments to the class.

## Important Note: to deploy SparkLLMTraining in EMR AWS
The EMR AWS latest version has spark using scala 2.12 and unfortunately, if we run the jar file with scala 2.13 in EMR, we will get error for version conflict.
To fix this issue when we want to deploy SparkLLMTraining in EMR AWS, simply change scala version from 2.13 to 2.12 in build.sbt. 
Then clean and assembly to a new jar file.

## Training model follows the following pipeline:
1. Create sliding windows that will feed to model. Sliding windows is a list of pair window-size context and target word
2. Define model: model has DenseLayer, OutputLayer.
   According to "Build LLM from Scratch", ideally an LLM Model has the attention layer that is the core of the LLM.
   Deeplearning4j also has attention layer SelfAttentionLayer. So a model could have the following layers: SelfAttentionLayer, DenseLayer, OutputLayer.
3. The SparkLLMTraining does an additional step which is initializing a spark context, and wrap the model with that context
4. Training model with sliding windows
5. Test the model with the given string "The man starts with"

## Statistics File
- Training a model in Spark involves looking at statistics: Training Loss and Accuracy, Learning Rates, Memory Usage, Time per Epoch/Iteration, and Spark-specific metrics.
- File of training first two paragraphs of a Sherlock Holmes novel will be at: src/main/resources/input/data.txt
- Stat file after training are written in **stats.txt** in root folder

## Additional commands
`sbt clean`: clean project

`sbt test`: run test. There are 5 tests in the **src/test/** directory

## About the training data
The training data for this model are books from [https://www.gutenberg.org/](https://www.gutenberg.org/)


