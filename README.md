Automated Streamlit dashboard:  https://pyspark-and-mllib-project-yqgi3uylqbbvuhepygdbno.streamlit.app/

(*Didn't want to pay for a business level PowerBI online account)

Graph of k-means quantification elbow method.  Script automates selection of k value with logic.
https://hypercord114.github.io/PySpark-and-MLlib-project/

2026-06-08: Set of scripts utilizing PySpark so far for filtering and clustering transaction data.  Needs more work...  still a bit messy.  Will eventually utilize MLlib for predictions such as customer churn.  Will attempt to flesh out automated strings describing proposed actions for each cluster.

2026-06-10; wrestled with this for too long trying to get this to run as a seamless, unified pipeline.  was making a mess trying to set it up as a YAML file; i couldn't keep track of the permissions necessary between the docker environment and the GitActons environment how to save the generated files back to the repo.  i gave up and just setup a Linux shell script to run the pipeline.

i will write a better description of what is going on here soon, but i'm not quite finished with the entire process yet.

essentially, i am downloading a dataset of british transaction data, loading it into PySpark, cleaning the data, performing k-means clustering and feature generation.

the data cleaning step removes transactions with quantity of 0 or less, which is irrelevant for assessing customer loyalty, data where the customer id is null, which is likely a data error, and dropping duplicate transactions, which is another data error which would skew the analysis.

additionally, because this data is a bit old and because i'm attempting to replicate a current analysis, within the feature generation logic i calculated the "Recency" vector by iterating to the first day of the subsequent month from the most recent date in the entire dataset and then calculated days passed since that global anchor date.  so, given the age of the data the date of the analysis would be January 1st 2012 (I think...).

the clustering step, which is unsupervised machine learning, performs the elbow method, iterating through a range of k values from 2 to 10 and calculating the training cost for each step.  the process is depicted in a .HTML linegraph that is generated.  logic is used to identify the elbow of the line.  a model is then generated using the calculated k-value and saved to disk.

the data is then analyzed for a dashboard, saved to disk and displayed on a Streamlit dashboard with a simple descriptor about each cluster of customers.

as stated above, i will go on to predict churn and other metrics.

for now, i have achieved the goal of getting this pipeline streamlined into a unified pipeline.  i still need to wrestle with this, read a bit more about how the cluster process works so i can be more eloquent with description, and smooth out automation hiccups.  for now i'm going to leave this alone.
