# NLP task 

I have deployed this model on streamlit: https://iitksecyrecruitmentnlpproject-b4puxphxumdjmej6zh2ls6.streamlit.app/
## My Approach

#### Data cleaning and processing
1. Firstly , I cleaned the dataset involving duplicate rows and removed the rows that contain nill values and is of no use.

2. Then I removed the stopwords, other punctuations, extra spaces, trailing and leading spaces from the paragraphs dataset and then I have analysed the lenghts of paragraphs and the most common words, after that I have removed first 1000 most common words saved the dataset as test1.csv.

3. I have also created the other dataset named for_bot.csv which is just the original dataset with the indexing is according to the test1.csv.
   
#### So here is the whole plan !!!
4. Now i will convert the paragraphs inside the test1.csv into vectors using TF-IDF vectorizer then main behind TF-idf is that, it takes account of the uniqueness of the words inside the corpus, which will be very usefull to retrieve the best 5 paragraphs according to the query.

5. The query will also be processed in the same way as the test1.csv was gone through, then got converted into vector using tf-idf vector.

6. Then the indices of the best 5 paragraphs will be then retrieved using cosine-similarity.

7. Then paragraphs on those indices from the "for_bot.csv" will be sent to the llm to process.

## Instructions
1. pull this repo using the following command:
   "git clone https://github.com/omgupta-iitk/iitk_secy_recruitment_nlp_project.git"
   "cd iitk_secy_recruitment_nlp_project"

3. Install the requirements :
  "pip install requirements.txt"

4. just run this command to launch the browser:
   "streamlit run qa_bot_gemini.py"
   
