library(mice)
library(data.table)
setwd("")

# Load and generate per individual dataset
sepsis_data <- fread(file.choose())
sepsis_data <- as.data.frame(sepsis_data)
sepsis_ind <- unique(sepsis_data$filename)
for(ind in sepsis_ind) {
  sepsis_ind_data <- sepsis_data[sepsis_data$filename == ind,]
  write.table(sepsis_ind_data,
              paste("./sepsis_ind_data/sepsis_ind_data_",
                    ind, ".txt", sep=""),
              sep="\t", row.names=F, col.names=T, quote=F)
}

# Load per individual dataset
for(ind in sepsis_ind[39869:length(sepsis_ind)]) {
  print(ind)
  
  sepsis_data <- fread(paste("./sepsis_ind_data/sepsis_ind_data_",
                             ind, ".txt", sep=""))
  sepsis_data <- as.data.frame(sepsis_data)
  
  sepsis_data_format <- sepsis_data
  for(j in 1:ncol(sepsis_data_format)) {
    sepsis_data_format[,j] <- 
      ifelse(is.nan(sepsis_data_format[,j]), NA, sepsis_data_format[,j])
  }
  sepsis_data_format$filename <- as.factor(sepsis_data_format$filename)
  sepsis_data_format$Gender <- as.factor(sepsis_data_format$Gender)
  sepsis_data_format$Unit1 <- as.factor(sepsis_data_format$Unit1)
  sepsis_data_format$Unit2 <- as.factor(sepsis_data_format$Unit2)
  sepsis_data_format$SepsisLabel <- as.factor(sepsis_data_format$SepsisLabel)
  
  sepsis_miss_trim <- sepsis_data_format
  
  # sepsis_miss_trim <- sepsis_data_format[,1,drop=F]
  # for(j in 2:ncol(sepsis_data_format)) {
  #   miss_rate <- sum(is.na(sepsis_data_format[,j])) / nrow(sepsis_data_format)
  #   if(miss_rate <= 0.9) {
  #     sepsis_miss_trim <- cbind(sepsis_miss_trim, sepsis_data_format[,j,drop=F])
  #   }
  # }
  # dim(sepsis_miss_trim)
  
  #sepsis_miss_trim <- sepsis_miss_trim[1:100,c(1:5,11:16)]
  #dim(sepsis_miss_trim)
  
  # Initialize MICE required data
  ini <- mice(sepsis_miss_trim, maxit = 0)
  
  pred <- ini$pred
  #View(pred)
  
  meth <- ini$meth
  #View(meth)
  
  # Fixed effect model
  pred_fix <- pred
  pred_fix[,"filename"] <- 0
  pred_fix["filename",] <- 0
  #View(pred_fix)
  
  meth_fix <- meth
  meth_fix[] <- "pmm"
  meth_fix["filename"] <- ""
  meth_fix[c("Gender","Unit1","Unit2","SepsisLabel")] <- "logreg"
  #View(meth_fix)
  
  sepsis_impute_fix <- mice(sepsis_miss_trim, m=1,
                            meth = meth_fix, pred = pred_fix, 
                            print = FALSE)
  
  sepsis_ind_data_impute <- complete(sepsis_impute_fix)
  
  write.table(sepsis_ind_data_impute,
              paste("./sepsis_ind_data_impute/sepsis_ind_data_impute_",
                    ind, ".txt", sep=""),
              sep="\t", row.names=F, col.names=T, quote=F)
}

# Combine individual dataset
sepsis_ind_data_impute_all <- sepsis_ind_data_impute[0,]
for(ind in sepsis_ind) {
  
  if(!(ind %in% sepsis_ind_data_impute_all$filename)) {
    print(ind)
    
    sepsis_ind_data_impute <- 
      fread(paste("./sepsis_ind_data_impute/sepsis_ind_data_impute_",
                  ind, ".txt", sep=""))
    
    sepsis_ind_data_impute_all <- rbind(sepsis_ind_data_impute_all,
                                        sepsis_ind_data_impute)
  }
  
}

write.table(sepsis_ind_data_impute_all,
            paste("./sepsis_ind_data_impute_all",
                  ".txt", sep=""),
            sep="\t", row.names=F, col.names=T, quote=F)












































