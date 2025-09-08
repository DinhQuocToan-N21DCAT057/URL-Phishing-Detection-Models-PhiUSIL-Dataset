import pandas as pd
import os

from url_features_extractor import URL_EXTRACTOR

url = "https://nacional1.webcindario.com/"

temp = []
extractor = URL_EXTRACTOR(url)
data = extractor.extract_to_dataset() 
temp.append(data)

test = pd.DataFrame(temp)
test.to_csv("test.csv", index=False)
