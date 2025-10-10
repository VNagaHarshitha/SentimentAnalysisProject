# SentimentAnalysisProject
Task 1 – Environment Setup & API Configuration
Set up the development environment and installed necessary dependencies.
Fetched API keys and integrated:
NewsAPI – for retrieving top headlines and news articles.
GNewsAPI – used as a secondary source for broader coverage.

Task 2 – Sentiment Analysis Using Gemini
The fetched news data was passed to Google Gemini API for LLM.
Gemini analyzed the textual content and produced sentiment scores for each article.
Stored sentiment results for use in later forecasting and alerting tasks.

Task 3 – Forecasting & Slack Alerts
Used Prophet to forecast future sentiment trends based on the analyzed data.
Implemented an alert system using Slack API:
If the predicted sentiment score is greater than +5 (strongly positive)
or less than -5 (strongly negative),
a notification is automatically sent to a Slack channel.

Task 4 – Streamlit Dashboard Deployment
Developed an interactive Streamlit dashboard to visualize:
Live news data
Sentiment scores
Forecast results and trends
sending alert
Deployed the Streamlit app for easy access and real-time monitoring.

# License
This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.

