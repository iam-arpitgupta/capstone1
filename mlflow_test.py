# import mlflow
# with mlflow.start_run():
#   mlflow.log_param('parameter name', 'value')
#   mlflow.log_metric('metric name', 1)


# this command is req for mac os specific for the ssl verfification 
import ssl
import urllib.request

url = "https://www.google.com"
response = urllib.request.urlopen(url)
print(response.status)  # should print 200
