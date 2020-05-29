#Set directory to import dataset as 'a' & to export dataset as 'b'

from="~/outlier/data/csvform_artificiallog1_0.05"
from="~/outlier/data/csvform_artificiallog2_0.01"
from="~/outlier/data/csvform_reallog_0.05"

to="~/outlier/data/encodedform_artificiallog1_0.05"
to="~/outlier/data/encodedform_artificiallog2_0.01"
to="~/outlier/data/encodedform_reallog_0.05"


setwd(from)
fn<-list.files(getwd())


for(j in 1:length(fn)){
  print(paste("Start preprocessing for log :",fn[j] ))
  
  anomolous<-data.frame(read.csv(fn[j], header=T))
  anomolous= anomolous[ with(anomolous, order(caseid, order)),]
  data<- anomolous[,c("caseid","name")]  

  names(data)[1:2] <- c("ID", "ActivityID") 
  
  data$ID <- as.factor(data$ID)
  data$ActivityID <- as.factor(data$ActivityID)
  
  
  ########################################
  print(paste(fn[j],": Start One-hot encoding & zero-padding"))
  preprocess_start <- Sys.time()
  
  a<- model.matrix(~ActivityID, data = data)
  A<- as.numeric(data[,2])
  A[which(A!=1)] <- 0
  a<- cbind(ActivityID1 = A, a[,-1])
  onehot<- as.data.frame(a)
  
  data1 <- onehot
  newdat <- cbind(data[,1], data1)
  newdat[,1] <- as.factor(newdat[,1])
  n<- length(levels((newdat[,1])))   # the number of cases
  m<- summary((newdat[,1]))[1]      # maximum trace length
  mean(summary((newdat[,1]), maxsum=n))
  sd(summary((newdat[,1]), maxsum=n))
  l<-levels((newdat[,1]))
  max<- m*(ncol(newdat)-1)
  c=unique(anomolous[,c("caseid","label")])
  label = as.character(c[,2])
  # label[which(label!='normal')] = 1    # change
  # label[which(label=='normal')] = 0
  
  newdat2<- matrix(NA, nrow=n , ncol=max)
  for(i in 1:n){
    save2 <- as.vector(t(newdat[which(newdat[,1]==l[i]),-1]))  
    newdat2[i,1:length(save2)] <- save2
  }
  newdat2[which(is.na(newdat2))] <- 0 # zero-padding
  newdat3 <-data.frame(cbind(caseid=l, label= label, newdat2))
  
  preprocess_end <- Sys.time()
  print(preprocess_end - preprocess_start )
  print(paste(fn[j],": Finished One-hot encoding & zero-padding"))
  

  setwd(to)
  write.csv(newdat3, paste("Preprocessed_",fn[j],sep=''), row.names = F)
  setwd(from)
  
}



