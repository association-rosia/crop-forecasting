FOLDER = 'augment_100_5'

BANDS = ['red', 'green', 'blue', 'rededge1', 'rededge2', 'rededge3', 'nir', 'swir']
VI = ['ndvi', 'savi', 'evi', 'rep', 'osavi', 'rdvi', 'mtvi1', 'lswi']

M_COLUMNS = ['tempmax', 'tempmin', 'temp', 'dew', 'humidity', 'precip', 'precipprob', 'precipcover', 'windspeed',
             'winddir', 'sealevelpressure', 'cloudcover', 'solarradiation', 'solarenergy', 'uvindex', 'moonphase',
             'solarexposure']

S_COLUMNS = ['ndvi', 'savi', 'evi', 'rep', 'osavi', 'rdvi', 'mtvi1', 'lswi']

G_COLUMNS = ['Other Rice Yield (kg/ha)', 'Field size (ha)', 'Rice Crop Intensity(D=Double, T=Triple)']

TARGET = 'Rice Yield (kg/ha)'

TARGET_TEST = 'Predicted Rice Yield (kg/ha)'
