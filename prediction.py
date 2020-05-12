import joblib
import numpy as np


def pred(Beds, Baths, Cars, Area, Date, Lat, Long, Dis, NSW):

    """
    This is the function for predicting the housing price
    Beds: the number of bedrooms (int)
    Baths: the number of bathrooms (int)
    Cars : the number of cars (int)
    Area : the square area of the house (float)
    Date : the number days between the sales date and today (int)
    Lat : Latitude (float)
    Long : Longitude (float)
    Dis : the Distance to the nearest train station (km) (float)
    NSW : the postcode of the house (int)
    we use the NSW to select the model, and now the suburb we have trained are
    NSW 2112: Ryde, NSW 2114: West Ryde, NSW 2118: Carlingford, NSW 2119: Beecroft, NSW 2120: Thornleigh, Pennant Hills,
    NSW 2122: Eastwood, NSW 2126: Cherrybrook, NSW 2151: North Parramatta, NSW 2152: Northmead, NSW 2153: Baulkham Hills, Winston Hills
    NSW 2154: Castle Hill.
    """

    filename = './models/NSW' + str(NSW) + '.sav'
    # load the model from disk
    loaded_model = joblib.load(filename)
    data = np.array([Beds, Baths, Cars, Area, Date, Lat, Long, Dis]).reshape(1,-1)
    prediction = loaded_model.predict(data)
    return round(prediction.item(), 2)


# print(pred(Beds=2, Baths=1, Cars=1, Area=200, Date=20, Lat=-33, Long=151, Dis=1.5, NSW=2154))