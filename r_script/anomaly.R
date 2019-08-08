print('hello world')
args = commandArgs(trailingOnly=TRUE)

# install and import packages
#install.packages("sparklyr",repos = "http://cran.us.r-project.org")
#install.packages("dplyr",repos = "http://cran.us.r-project.org")
#install.packages("tseries",repos = "http://cran.us.r-project.org")
#install.packages("anomalize",repos = "http://cran.us.r-project.org")
#install.packages("forecast",repos = "http://cran.us.r-project.org")
#install.packages("tsoutliers",repos = "http://cran.us.r-project.org")

library(dplyr)
#require(plyr)
library(sparklyr)
library(tseries)
library(anomalize)
library(forecast)
library(tsoutliers)


# set up spark
spark_install(version = "2.2.0")
Sys.setenv(JAVA_HOME = "/Library/Java/JavaVirtualMachines/jdk1.8.0_221.jdk/Contents/Home")
sc <- spark_connect(master = "local")

#spark_disconnect(sc)

# preprocessing data
df = spark_read_csv(sc, name=NULL, path=args[1], header=TRUE)
df = df %>% select(visitors=totalshoppertraffic_visitors, orders=digital_orders,
                   customer=cust_prospect_ind, device=visit_device_type, date=event_dt)

df = df %>% mutate(date=to_date(date, "MM/dd/yy")) 
df = df %>% arrange(date)     
df = df %>% mutate(rate=orders/visitors)

# segment data across customer and device
all = df %>% filter(customer == 'All Visitors', device == 'All Devices')

all_mobile = df %>% filter(customer == 'All Visitors', device == 'Mobile Phone')
all_desktop = df %>% filter(customer == 'All Visitors', device == 'Desktop')
all_tablet = df %>% filter(customer == 'All Visitors', device == 'Tablet')

cust_all = df %>% filter(customer == 'CUSTOMER', device == 'All Devices')
undetermined_all = df %>% filter(customer == 'UNDETERMINED', device == 'All Devices')
prospect_all = df %>% filter(customer == 'PROSPECT', device == 'All Devices')

cust_mobile = df %>% filter(customer == 'CUSTOMER', device == 'Mobile Phone')
undetermined_mobile = df %>% filter(customer == 'UNDETERMINED', device == 'Mobile Phone')
prospect_mobile = df %>% filter(customer == 'PROSPECT', device == 'Mobile Phone')

cust_desktop = df %>% filter(customer == 'CUSTOMER', device == 'Desktop')
undetermined_desktop = df %>% filter(customer == 'UNDETERMINED', device == 'Desktop')
prospect_desktop = df %>% filter(customer == 'PROSPECT', device == 'Desktop')

cust_tablet = df %>% filter(customer == 'CUSTOMER', device == 'Tablet')
undetermined_tablet = df %>% filter(customer == 'UNDETERMINED', device == 'Tablet')
prospect_tablet = df %>% filter(customer == 'PROSPECT', device == 'Tablet')


# models
arima <- function(data){
  fit <- auto.arima(data)
  res <- locate.outliers.oloop(data, fit, types = c("AO", "LS", "TC"))
  list = res$outliers$ind
  index = 1:length(data)
  anomaly = ifelse(index %in% list, 1, 0)
  return(anomaly)
}

require(plyr)
stl <- function(df, var){
  result = df %>%
    time_decompose(var, method = "stl", trend = "7 days") %>%
    anomalize(remainder, method = "gesd") %>%
    time_recompose()
  anomaly = mapvalues(result$anomaly, c("Yes", "No"), c(1, 0))
  
  return(anomaly)
}

twitter <- function(df, var){
  result = df %>%
    time_decompose(var, method = "twitter", trend = "7 days") %>%
    anomalize(remainder, method = "iqr") %>%
    time_recompose()
  anomaly = mapvalues(result$anomaly, c("Yes", "No"), c(1, 0))
  return(anomaly)
}


# detect anomalies
anomaly_detection <- function(data){ 
  df = collect(data)
  
  visitors_arima_anomaly = arima(df$visitors)
  orders_arima_anomaly = arima(df$orders)
  rate_arima_anomaly = arima(df$rate)
  
  visitors_stl_anomaly = stl(df, 'visitors')
  orders_stl_anomaly = stl(df, 'orders')
  rate_stl_anomaly = stl(df, 'rate')
  
  visitors_twitter_anomaly = twitter(df, 'visitors')
  orders_twitter_anomaly = twitter(df, 'orders')
  rate_twitter_anomaly = twitter(df, 'rate')
  
  result = data.frame(df, as.data.frame(visitors_arima_anomaly), 
                      as.data.frame(visitors_stl_anomaly),
                      as.data.frame(visitors_twitter_anomaly),
                      as.data.frame(orders_arima_anomaly), 
                      as.data.frame(orders_stl_anomaly),
                      as.data.frame(orders_twitter_anomaly),
                      as.data.frame(rate_arima_anomaly),
                      as.data.frame(rate_stl_anomaly),
                      as.data.frame(rate_twitter_anomaly))
  
  return(result)
}

a = anomaly_detection(all)

a_m = anomaly_detection(all_mobile)
a_d = anomaly_detection(all_desktop)
a_t = anomaly_detection(all_tablet)

c_a = anomaly_detection(cust_all)
p_a = anomaly_detection(prospect_all)
u_a = anomaly_detection(undetermined_all)

c_m = anomaly_detection(cust_mobile)
p_m = anomaly_detection(prospect_mobile)
u_m = anomaly_detection(undetermined_mobile)

c_d = anomaly_detection(cust_desktop)
p_d = anomaly_detection(prospect_desktop)
u_d = anomaly_detection(undetermined_desktop)

c_t = anomaly_detection(cust_tablet)
p_t = anomaly_detection(prospect_tablet)
u_t = anomaly_detection(undetermined_tablet)




result = rbind(a, a_m, a_d, a_t, c_a, p_a, u_a, c_m, p_m, u_m, c_d, p_d, u_d, c_t, p_t, u_t)
result = result %>% mutate(date=as.character(date))
spark_result = copy_to(sc, result, overwrite=TRUE)
spark_write_csv(spark_result, 'anomaly_result', mode='overwrite')
#sdf_bind_rows(spark_df, spark_df)