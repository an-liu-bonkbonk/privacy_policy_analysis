# privacy_policy_analysis
## by An Liu
Sentiment Analysis and correlation analysis between sentiment analysis metrics and readability metrics on an existing corpus of privacy policies from 1996 to 2021 using a published dataset in the paper "Privacy Policies Across the Ages: Content and Readability of Privacy Policies 1996--2021" (Wagner, 2022).  

Published paper can be accessed on https://arxiv.org/abs/2201.08739  

Original published dataset from the Wagner, 2022 paper is available on https://zenodo.org/records/7426577  

This project reproduces some of the readability metrics from the txt files (only the text files in policies_texts folder are used, not metadata), conducts a sentiment analysis to compute the polarity and subjectivity on the policy texts using Python library Textblob, and finally conducts a correlation analysis between the reability metrics and sentiment analysis results.  

Note: "policy-texts.zip" required for this analysis is not uploaded to this repository due to large file size (and that I don't want to pay for GitHub pro)

Reference:  
Wagner, I. (2022, January 21). Privacy Policies Across the Ages: Content and Readability of Privacy Policies 1996--2021. ArXiv.org. https://doi.org/10.48550/arXiv.2201.08739  
Loria, S. (2018). textblob Documentation. Release 0.15, 2.
