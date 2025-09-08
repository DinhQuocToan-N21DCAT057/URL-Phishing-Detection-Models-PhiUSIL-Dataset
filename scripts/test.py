import pandas as pd
import os

from url_features_extractor import URL_EXTRACTOR

url = "https://sexyyphotooo.weebly.com/"

temp = []
extractor = URL_EXTRACTOR(url)
data = extractor.extract_to_dataset()
print(extractor.exec_time)
temp.append(data)

test = pd.DataFrame(temp)
test.to_csv("test.csv", index=False)
