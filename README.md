Automated Streamlit dashboard:  https://pyspark-and-mllib-project-dhurrncb7fbjafpcbyyfg5.streamlit.app/

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

2026-06-10/11:  ok wrestled with this more all night.  have predictor model generated and saved.  now i need to incorporate MLflow into the scripts, add another model, and display the results on the dashboard.  also need to write a better description of what i'm doing...  tomorrow.

as stated above, i will go on to predict churn and other metrics.

for now, i have achieved the goal of getting this set of scripts streamlined into a unified pipeline.  i still need to wrestle with this, read a bit more about how the cluster process works so i can be more eloquent with description, and smooth out automation hiccups.  for now i'm going to leave this alone.

2026-06-17:  made a mess of the dashboard last night trying to incorporate a histogram of the silhouette scores for all of the k values attempted during elbow method.  i was tired and just throwing random code into the streamlit dashboard script without thinking about it.  i'm not sure what happened...  i had to chase my tail today to get it to work again.

finally fixed and the histogram is incorporated.  i also included the elbow method line graph.

the silhouette score for the clusters is pretty high for k=2 and k=3, around 0.9, but it reduces to around 0.7 for k=4.  nonetheless the elbow method identifies k=4 as the correct number of clusters.  looking at the average monetary spend per cluster in the dashboard, it looks like there is a significant difference in spending between the two champion clusters.  i suppose an executive decision would need to be made about whether or not there is a need to develop tiers of VIP clusters.  if tiers is not an option, i suppose the silhouette scores would indicate that three clusters is better than four.

the revenue forecast seems like it would be helpful.  having spent 16 years at walmart in the meat department, i can see how this prediction would be valuable for ordering product in anticipation of customer demand, both for the weekend and for the first of the month when foodstamps are issued.  this kind of model would quantify how many boxes of family pack chicken breasts, for example, would need to be ordered for those periods.  simple estimation of revenue may be valuable for budgeting purposes.  it would depend on the objective of the organization how this data is used or spun.

the predictive models, the randomforest and decisiontree, as well as churn, would be able to class customers into a cluster on a running basis by assessing their spending habits against legacy data that has been clustered.  in other words, if certain perks are allowed for different clusters, these models could be used to class customers periodically, most likely in a batch assessment on a schedule.  i suppose it would be hard to do this instantaneously with data from a kafka server; it would most likely be done once a month or once a quarter or something like that.

likewise, the churn predictive model should be able to identify customers whose spending habits reveal them as sub-standard customers at risk of not generating consistent revenue.  similar to how the other clusters might be treated, this class may be offered certain benefits to draw them back and derive revenue from them.

ok, happy this is all working again and the metrics are published successfully to the dashboard.

will read about how to apply these predictive models and perhaps attempt to set up some kind of a system to class ongoing sales data...

will also start thinking about the next project.  have noticed in job postings some of the skills that are necessary for data scientist and data engineer roles.  not sure how much i can do on github.  kubernetes and Azure and AWS systems might be hard to develop familiarity with independently.  will keep researching.
