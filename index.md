January 22, 2022

<dl>
<dt>Authors</dt>
<dd>Mel Nguyen</dd>
<dd>Bojia Liu</dd>
<dd>Stanley Hung</dd>
<dt>Source Code</dt>
<dd><a href='https://github.com/mads-app-reviews-nlp/app-reviews-nlp'>https://github.com/mads-app-reviews-nlp/app-reviews-nlp</a></dd>
</dl>

# 1. Background
Food delivery apps have become an integral part of many people’s daily lives. From the perspective of food delivery companies, understanding and catering to the needs of millions are no easy feat. At the same time, the food delivery business is extremely competitive, making the task to understand and measure a customer’s “happiness meter” to the platform is even more imminent. 
In this project, we would like to leverage the power of natural language processing to achieve two goals. Firstly, how do customers feel about various food delivery platforms, and do these sentiments vary over time and locations? And secondly, what are the primary topics that customers write about on these reviews? Getting a low rating says very little about why that customer is deciding to give that rating, rather, we need to understand and mine the pain points that are driving the customers’ decisions. And by understanding them, we can have a more insightful view of the strengths of different food delivery platforms, while also potentially identifying the pain points of each. 

# 2. Data Sources
Our focus will be on food delivery platforms in Singapore & Australia. In each country, our team has curated a list of 3 to 4 food delivery apps, namely:
- Australia: DoorDash, UberEats, Grub, Deliveroo and Menulog
- Singapore: foodpanda, Grab, Deliveroo

For each country, the dataset is constructed from the iOS and Android app stores for each respective country, and for the purpose of the milestone project, the ground truth labels will be based on customers’ ratings. In other words, customers whose ratings are high (above 3 stars) will be considered positive reviews, and vice versa reviews that are 1 or 2 stars will be considered negative. 
Below is an example of a customer review:

<p align="center"><img src='images/sample_review.png' alt='images/sample_review.png'></p>

We use the `google_play_scraper` library to scrape the reviews from Google Store, and `app_store_scraper` to scrape the reviews from Apple Store. Both libraries allow specifying a `country` parameter to access the corresponding country's app store. However, in the raw data, we observe some duplicates for apps that are present in both countries (eg. Deliveroo operates in both Singapore and Australia). The impact is negligible, as the duplicates made up for less than 5% of the total reviews. Another known caveat is for Grab, which offers both transportation and food delivery services from the same app. The scope of this project does not cover filtering out non-food delivery specific reviews, but this is something that can be considered for subsequent iterations.

The scraped reviews are then saved directly into pandas dataframe, which we subsequently separate into 2 pickle files, one for each country. Each dataset is comprised of the following information:
- When a review is made
- The app/platform
- Rating
- Content of the review

In total, there are 659k reviews for Singapore, spanning from November 2013 to December 2021. Similarly, there are 626k reviews for Australia, spanning from September 2009 to December 2021. The same dataset is used for both **supervised** and **unsupervised** learning tasks. 

# 3. The Landscape Of Reviews




# 4. Sentiment Classification 

# 5. Topic Modelling

# 6. Discussion and Next Steps

# 7. Statement Of Work
This project is a collaborative effort with equal participation from everyone, specifically:
- Mel: Supervised Learning (Naive Bayes & Regression), Deep Learning, Final Report
- Bojia: Supervised Learning (SVM & Decision Trees), Final Report
- Stanley: Unsupervised Learning (LDA & NMF), Final Report


