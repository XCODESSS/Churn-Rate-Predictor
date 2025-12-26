## CUSTOMER CHURN RATE PREDICTOR

A machine learning system that predicts customer churn and recommends targeted retention strategies to maximize ROI.

## Overview

This project addresses customer retention challenges by predicting which customers are likely to churn. Using machine learning classification models, the system identifies at-risk customers and provides actionable insights to support data-driven retention strategies.

## BUISNESS STATEMENT

What is Churn?

Churn happens when customers stop using a company's product or service. For a streaming platform, it's when someone cancels their subscription. For a bank, it's when a customer closes their account. Companies track churn rate as the percentage of customers lost over a specific period—for instance, losing 50 out of 1,000 customers in a month means a 5% monthly churn rate. High churn rates signal trouble because they directly reduce revenue and require constant effort to replace lost customers with new ones.

Why predicting Churn matters?

Identifying customers likely to leave before they actually do gives companies a chance to intervene. Studies show that keeping an existing customer costs far less than acquiring a new one—sometimes up to 25 times less. When we can predict churn, businesses can offer personalized discounts, improve customer support, or fix product issues for at-risk customers. This proactive approach protects revenue and often reveals broader problems with the product or service that need attention. Essentially, churn prediction transforms customer retention from reactive damage control into strategic prevention.

What happens if we're wrong?

Prediction models make two types of mistakes. If we incorrectly predict someone will leave (false positive), we might waste money offering them discounts they didn't need to stay. This cuts into profits but isn't catastrophic—we might even build goodwill. The more costly error is missing customers who actually leave (false negative). These customers churn without any retention attempt, taking all their future revenue with them. The goal isn't perfect prediction but finding the right balance: catching enough real churners to protect revenue while not overspending on customers who would stay anyway

## PROBLEM FRAMING

### BUISNESS PROBLEM

Telecom companies face significant revenue loss when customers discontinue their services. In this dataset, churn is explicitly recorded as a binary label, indicating whether a customer has left the company or not. Since the data represents a static snapshot of customers at a given point in time, churn is treated as a final and irreversible outcome within the scope of this analysis.

### OBJECTIVE OF THE PROJECT

The objective of this project is to predict the likelihood of customer churn using customer demographic information, service usage patterns, and account-related features. By identifying customers who are at high risk of churn, the company can take proactive retention actions such as targeted offers, improved customer support, or contract incentives.

### EVALUATION MIDSET

In a real business setting, not all prediction errors are equally costly. Missing a customer who is likely to churn (false negative) can result in permanent revenue loss, whereas incorrectly flagging a loyal customer as a churn risk (false positive) may only incur a relatively small retention cost. Therefore, this project prioritizes an evaluation mindset focused on recall and risk identification, rather than maximizing accuracy alone.

For this reason, we will prioritize Recall (the percentage of actual churners we successfully identify) over metrics like accuracy or precision. A model that catches 80% of churners with some false alarms is more valuable than one that catches 50% of churners with perfect precision.

The evaluation strategy depends on business economics. In typical telecom scenarios where customer lifetime value ($500+) significantly exceeds retention intervention costs ($50-100), optimizing for recall takes priority. However, if retention programs become expensive, the model would need to balance recall with precision to maintain positive ROI.
