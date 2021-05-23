import requests

description = "We build products and services powered by payments data to find and stop financial crime. By combining data science technique with an intimate knowledge of payments data we develop solutions that will improve outcomes for people, businesses and economies. Headquartered in The City of London, we craft bespoke algorithms that help our clients gain an understanding of the underlying criminal behaviour that drives financial crime.As a Data Scientist, you will join one of the first teams in the world looking at payments data in the UK and across the world. In the research discipline you will help build systems that expose money laundering and detect fraud, managing other data scientists and working with clients to understand the underlying behaviours employed by criminals. You will be product focused, working in close collaboration with our engineering and operations data scientists as well as the wider sales, consulting, and product teams."
dict = {'desc': description}
url = 'http://127.0.0.1:8000/predict'
r = requests.post(url, params=dict)
print(r)
