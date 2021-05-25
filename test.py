import requests

description = "We build products and services powered by payments data to find and stop financial crime. By combining data science technique with an intimate knowledge of payments data we develop solutions that will improve outcomes for people, businesses and economies. Headquartered in The City of London, we craft bespoke algorithms that help our clients gain an understanding of the underlying criminal behaviour that drives financial crime.As a Data Scientist, you will join one of the first teams in the world looking at payments data in the UK and across the world. In the research discipline you will help build systems that expose money laundering and detect fraud, managing other data scientists and working with clients to understand the underlying behaviours employed by criminals. You will be product focused, working in close collaboration with our engineering and operations data scientists as well as the wider sales, consulting, and product teams."
description2 = "Tasks Development/configuration of new systems and applications on the basis of detailed specifications Design, coding and debugging of applications Understanding technical design specifications and translating them into implementation details Supporting the general delivery process on projects, helping the customer with on-site support activities Understanding and analysing a client's organisation/operation and requirements in order to produce recommendations, technical design documents and time/cost estimates"
description3 = "As the world’s leader in digital payments technology, Visa’s mission is to connect the world through the most creative, reliable and secure payment network - enabling individuals, businesses, and economies to thrive. Our advanced global processing network, VisaNet, provides secure and reliable payments around the world, and is capable of handling more than 65,000 transaction messages a second. The company’s relentless focus on innovation is a catalyst for the rapid growth of digital commerce on any device for everyone, everywhere. As the world moves from analog to digital, Visa is applying our brand, products, people, network and scale to reshape the future of commerce. At Visa, your individuality fits right in. Working here gives you an opportunity to impact the world, invest in your career growth, and be part of an inclusive and diverse workplace. We are a global team of disruptors, trailblazers, innovators and risk-takers who are helping drive economic growth in even the most remote parts of the world, creatively moving the industry forward, and doing meaningful work that brings financial literacy and digital commerce to millions of unbanked and underserved consumers. You’re an Individual. We’re the team for you. Together, let’s transform the way the world pays. Job Description What's it all about? The Data Scientist is a key member of the Data Science Lab in the Europe region. As a Data Science Manager, you will be accountable for leading the blueprint, development and delivery of analytics driven capabilities and solutions for Visa’s internal and external clients. In this role, you will work collaboratively with a multitude of internal functions and their senior leaders in promoting an evidence and insights-driven culture. You will be developing creative and cutting-edge data science solutions, develop best practices to drive business performance with a clear commercial ‘value’ and measurable ‘return’. What We Expect Of You, Day To Day. Drives automation of data analysis tools to enable scaling of Visa Consulting & Analytics services such as assessment of portfolio health to more clients and/or "
url = 'http://localhost:8080/predictresult'
r = requests.post(url, json={'desc': description3})
print(r.text)

# Reading Logged User Inputted Descriptions and Model's Predictions
f = open("descriptions.txt", 'r')
history_descriptions = f.read().replace("\n", " ")
f.close()
f = open("predictions.txt", 'r')
history_predictions = f.read().replace("\n", " ")
f.close()

# Comparing user inputted and training data descriptions distributions using Welsh's t tests
# which can handle samples of different sizes in comparison
url = 'http://localhost:8080/monitorinput'
input_decision = requests.post(url, json={'desc': history_descriptions})
print("Input distributions are: ", input_decision.text)

# Comparing distributions of tags predictions for user inputted job descriptions and training data
# target outcome using Welsh's t tests which can handle samples of different sizes in comparison
url = 'http://localhost:8080/monitoroutput'
output_decision = requests.post(url, json={'desc': history_predictions})
print("Output distributions are: ", output_decision.text)
