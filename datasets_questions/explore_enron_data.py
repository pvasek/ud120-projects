#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
print "loading..;"
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print "loaded"

total_count = len(enron_data)
print "number of records: ", total_count

# bigest payments
# for k, v in sorted(enron_data.iteritems(), key = lambda i: i[1]["total_payments"]):
#     print k, " total_payments: ", v["total_payments"]   

for k, v in enron_data["SKILLING JEFFREY K"].iteritems():
    print k, ": ", v

# email addresses
# for k, v in enron_data.iteritems():
#     print k, " email_address: ", v["email_address"]   

poi_items = filter(lambda i: i["poi"] == True, enron_data.values())
poi_count = len(poi_items)
print "pois: ", poi_count

import math
items = enron_data.values()
print "People with salary: ", len(filter(lambda v: isinstance(v["salary"], int), items))
print "People with email: ", len(filter(lambda v: v["email_address"] != "NaN", items))

without_total_payments = len(filter(lambda v: v["total_payments"] == "NaN", items))
print "People withhout total payments: ", without_total_payments, " percentage: ", without_total_payments / float(total_count)

pois_without_total_payments = len(filter(lambda v: v["total_payments"] == "NaN", poi_items))
print "POIs withhout total payments: ", pois_without_total_payments, " percentage: ", pois_without_total_payments / float(poi_count)

import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
