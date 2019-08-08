print('hello world')
args = commandArgs(trailingOnly=TRUE)
print(args[1])

# install and import packages
#install.packages("sparklyr",repos = "http://cran.us.r-project.org")
#install.packages("dplyr",repos = "http://cran.us.r-project.org")
#install.packages("tseries",repos = "http://cran.us.r-project.org")
#install.packages("anomalize",repos = "http://cran.us.r-project.org")
#install.packages("forecast",repos = "http://cran.us.r-project.org")
#install.packages("tsoutliers",repos = "http://cran.us.r-project.org")
#install.packages("zoo",repos = "http://cran.us.r-project.org")


library(dplyr)
library(sparklyr)
library(forecast)

# set up spark
#spark_install(version = "2.2.0")
Sys.setenv(JAVA_HOME = "/Library/Java/JavaVirtualMachines/jdk1.8.0_221.jdk/Contents/Home")
sc <- spark_connect(master = "local")
#spark_disconnect(sc)

# preprocessing data
#df = spark_read_csv(sc, name = NULL, path = "anomaly_det_dashboard_shopper_conv.csv", header=TRUE)
df = spark_read_csv(sc, name = NULL, path = args[1], header=TRUE)

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
f_arima <- function(data){
  #fit <- auto.arima(data)
  fit <- arima(data, c(1,0,7))
  f = predict(fit,7)
  pred = as.numeric(f$pred)
  return(pred)
}


f_ets <- function(data){
  data = ts(data, frequency=7)
  fit <- ets(data, model='ANA')
  f <- predict(fit, 7)
  
  pred = as.numeric(f$mean)
  
  return(pred)
}

hw <- function(data){
  data = ts(data, frequency=7)
  fit <- HoltWinters(data, seasonal='additive')
  f <- predict(fit, 7)
  pred = as.numeric(f)
  return(pred)
}


forecast <- function(data){
  df = collect(data)
  customer = rep(df$customer)[0:7]
  device = rep(df$device)[0:7]
    
  visitors_forecast = hw(df$visitors)
  orders_forecast = f_ets(df$orders)
  rate_forecast = f_arima(df$rate)
  date = as.character(seq(max(df$date)+1, max(df$date)+7, "days"))
  f = cbind(customer, device, date, visitors_forecast, orders_forecast, rate_forecast)
  return(f)
}



a = forecast(all)
a_m = forecast(all_mobile)
a_d = forecast(all_desktop)
a_t = forecast(all_tablet)

c_a = forecast(cust_all)
p_a = forecast(prospect_all)
u_a = forecast(undetermined_all)

c_m = forecast(cust_mobile)
p_m = forecast(prospect_mobile)
u_m = forecast(undetermined_mobile)

c_d = forecast(cust_desktop)
p_d = forecast(prospect_desktop)
u_d = forecast(undetermined_desktop)

c_t = forecast(cust_tablet)
p_t = forecast(prospect_tablet)
u_t = forecast(undetermined_tablet)


forecast_result = rbind(a, a_m, a_d, a_t, c_a, p_a, u_a, c_m, p_m, u_m, c_d, p_d, u_d, c_t, p_t, u_t)


library(zoo)
outlier <- function(data){
  mean = mean(data)
  std = sd(data)
  list = which((data > mean + 1.5*std) | (data < mean - 1.5*std))
  index=1:length(data)
  outliers = ifelse(index %in% list, 1, 0)
  return(outliers)
}


find_outlier <- function(df){
  v = as.numeric(df[,'visitors_forecast'])
  o = as.numeric(df[,'orders_forecast'])
  r = as.numeric(df[,'rate_forecast'])
  
  v_outlier = outlier(v)
  o_outlier = outlier(o)
  r_outlier = outlier(r)
  
  result = cbind(v_outlier, o_outlier, r_outlier)
  return(result)
}


a2 = find_outlier(a)
a_m2 = find_outlier(a_m)
a_d2 = find_outlier(a_d)
a_t2 = find_outlier(a_t)

c_a2 = find_outlier(c_a)
p_a2 = find_outlier(p_a)
u_a2 = find_outlier(u_a)

c_m2 = find_outlier(c_m)
p_m2 = find_outlier(p_m)
u_m2 = find_outlier(u_m)

c_d2 = find_outlier(c_d)
p_d2 = find_outlier(p_d)
u_d2 = find_outlier(u_d)

c_t2 = find_outlier(c_t)
p_t2 = find_outlier(p_t)
u_t2 = find_outlier(u_t)






anomaly_result = rbind(a2, a_m2, a_d2, a_t2, c_a2, p_a2, u_a2, c_m2, p_m2, u_m2, c_d2, p_d2, u_d2, c_t2, p_t2, u_t2)


result = cbind(forecast_result, anomaly_result)
spark_result = copy_to(sc, result, overwrite=TRUE)
spark_write_csv(spark_result, 'forecast_result', mode='overwrite')
#sdf_bind_rows(spark_df, spark_df)