library(ggplot2)
library(data.table)


val<-fread(file.choose())
train <-fread(file.choose())
imputed <- rbind(val,train)
imputed <- as.data.frame(imputed)
non_imputed <- fread(file.choose())

non_imputed <- as.data.frame(non_imputed)

non_imputed <- non_imputed[non_imputed$filename %in% imputed$filename,]
imputed <- imputed[,intersect(colnames(imputed), colnames(non_imputed))]
non_imputed <- non_imputed[,intersect(colnames(imputed), colnames(non_imputed))]

imputed$data <- "Imputed"
non_imputed$data <- "Original"

all <- rbind(imputed, non_imputed)

plot <- data.frame("Variable"=NA,
                   "Data"=NA,
                   "Dataset"=NA)

for(i in colnames(all)) {
  if(i != "filename" & i != "data") {
    ind <- data.frame("Variable"=rep(i, nrow(all)),
                      "Data"=all[,i],
                      "Dataset"=all$data)
    plot <- rbind(plot, ind)
  }
}

plot <- plot[-1,]

p1 <- colnames(all)[1:8]
p2 <- colnames(all)[9:16]
p3 <- colnames(all)[17:24]
p4 <- colnames(all)[25:32]
p5 <- colnames(all)[33:39]

k <- 1
for(j in list(p1,p2,p3,p4,p5)) {
  p <- plot[plot$Variable %in% j,]
  
  boxplot <- ggplot(p, aes(x=Variable, y=Data, fill=Dataset)) +
    geom_boxplot() + 
    ylab("Variable Data Point Values") + 
    xlab("")
  
  ggsave(filename=paste0("", k, ".png"),
         plot=boxplot,
         width = 11, height = 8.5, units = c("in"))
  
  k <- k + 1
}
