#!/usr/bin/python

import itertools

def unpack(array_of_arrays):
    return map(lambda i: i[0], array_of_arrays)

def rec((prediction, age, net_worth)):
    return (age, net_worth, abs(net_worth - prediction))

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    # transform it to simple lists
    remove_count = len(predictions) * 0.1
    predictions = unpack(predictions)
    ages = unpack(ages)
    net_worths = unpack(net_worths)
    list = map(rec, zip(predictions, ages, net_worths))
    list = sorted(list, key = lambda (age, net_worth, error): error, reverse=True)
    cleaned_data = list[int(remove_count):]
    
    return cleaned_data

