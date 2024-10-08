import requests


data1 = {
        'ph': 0,
        'Hardness': 204.890455,
        'Solids': 20791.318981,
        'Chloramines': 7.300212,
        'Sulfate': 368.516441,
        'Conductivity': 564.308654,
        'Organic_carbon': 10.379783,
        'Trihalomethanes': 86.990970,
        'Turbidity': 2.963135,
    }

data2 = {
        'ph': 7.7984536762012135,
        'Hardness': 188.39494231709176,
        'Solids': 32704.569285770576,
        'Chloramines': 11.078872478914501,
        'Sulfate': 258.1911841475428,
        'Conductivity': 507.1786882733106,
        'Organic_carbon': 18.272439235274646,
        'Trihalomethanes': 85.17766213336226,
        'Turbidity': 4.107267203260775,
    }
response = requests.post("http://127.0.0.1:8000/prediction",json=data2)
print(response.json())