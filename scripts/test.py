import pandas as pd
import os

from url_features_extractor import URL_EXTRACTOR

url = "http://uaSH0kqRxr8xXctfQN.accompanysummary.my.id/VOHP0L3ZYAuMjE4NzktMzc3NjkxLTEwMTkzMTExMy1vLTUyNy0xMi01NjQzLTg1NjYyLTAtMC0wLTAtWW1mUnVUUFhELTM4YTQwNGNm"

temp = []
extractor = URL_EXTRACTOR(url)
data = extractor.extract_to_dataset() 
temp.append(data)

test = pd.DataFrame(temp)
test.to_csv("test.csv", index=False)
